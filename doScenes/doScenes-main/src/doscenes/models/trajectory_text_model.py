from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer


class TrajectoryTextModel(nn.Module):
    def __init__(
        self,
        text_model_name: str,
        hidden_dim: int,
        pred_steps: int,
        freeze_text_encoder: bool,
        local_files_only: bool,
        attention_heads: int,
        text_unfreeze_last_n_layers: int = 0,
    ) -> None:
        super().__init__()
        self.pred_steps = pred_steps

        model_source = str(Path(text_model_name).expanduser())
        source_path = Path(model_source)
        resolved_local_only = local_files_only or source_path.is_dir()

        self.tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=resolved_local_only)
        self.text_encoder = AutoModel.from_pretrained(model_source, local_files_only=resolved_local_only)

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
            if text_unfreeze_last_n_layers > 0:
                # DistilBERT encoder layers are under transformer.layer.*
                layers = getattr(getattr(self.text_encoder, "transformer", None), "layer", None)
                if layers is not None:
                    n = min(text_unfreeze_last_n_layers, len(layers))
                    for layer in layers[-n:]:
                        for p in layer.parameters():
                            p.requires_grad = True

        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)
        self.token_proj = nn.Linear(self.text_encoder.config.hidden_size, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, attention_heads, batch_first=True)

        self.traj_encoder = nn.GRU(input_size=2, hidden_size=hidden_dim, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder_instruction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_steps * 2),
        )
        self.decoder_baseline = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_steps * 2),
        )

    def forward(self, history_xy: torch.Tensor, instructions: list[str], head: str = "instruction") -> torch.Tensor:
        if history_xy.dim() != 3 or history_xy.size(-1) != 2:
            raise ValueError("history_xy must have shape [B, T, 2]")
        if len(instructions) != history_xy.size(0):
            raise ValueError("instructions size mismatch")
        if head not in {"instruction", "baseline"}:
            raise ValueError("head must be 'instruction' or 'baseline'")

        device = history_xy.device
        text_inputs = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=64,
        ).to(device)

        text_outputs = self.text_encoder(**text_inputs)
        text_tokens = self.token_proj(text_outputs.last_hidden_state)
        _ = self.text_proj(text_outputs.last_hidden_state[:, 0, :])

        _, hidden_state = self.traj_encoder(history_xy)
        traj_feat = hidden_state[-1]
        query = self.query_proj(traj_feat).unsqueeze(1)
        attended_tokens, _ = self.cross_attn(query=query, key=text_tokens, value=text_tokens)
        attended_feat = attended_tokens.squeeze(1)

        fused_feat = self.fusion_mlp(torch.cat([attended_feat, traj_feat], dim=-1))
        fused_feat = self.fusion_norm(fused_feat + traj_feat)
        decoder = self.decoder_instruction if head == "instruction" else self.decoder_baseline
        output = decoder(fused_feat)
        return output.view(-1, self.pred_steps, 2)
