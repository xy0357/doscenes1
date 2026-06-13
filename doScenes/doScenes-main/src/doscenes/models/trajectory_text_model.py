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
        future_decoder_type: str = "mlp",
    ) -> None:
        super().__init__()
        self.pred_steps = pred_steps
        self.future_decoder_type = future_decoder_type.lower()

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
        self.cls_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, attention_heads, batch_first=True)

        self.traj_encoder = nn.GRU(input_size=2, hidden_size=hidden_dim, batch_first=True)
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if self.future_decoder_type == "mlp":
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
        elif self.future_decoder_type == "gru_delta":
            self.future_time_embed = nn.Embedding(pred_steps, hidden_dim)
            self.decoder_instruction_gru = nn.GRU(
                input_size=hidden_dim * 2,
                hidden_size=hidden_dim,
                batch_first=True,
            )
            self.decoder_baseline_gru = nn.GRU(
                input_size=hidden_dim * 2,
                hidden_size=hidden_dim,
                batch_first=True,
            )
            self.decoder_instruction_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
            self.decoder_baseline_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
        elif self.future_decoder_type == "query_residual":
            self.future_query_embed = nn.Embedding(pred_steps, hidden_dim)
            self.future_query_attn = nn.MultiheadAttention(hidden_dim, attention_heads, batch_first=True)
            self.future_query_norm = nn.LayerNorm(hidden_dim)
            self.future_query_ffn = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.decoder_instruction_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
            self.decoder_baseline_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
        else:
            raise ValueError(f"Unsupported future_decoder_type: {future_decoder_type}")

    def _build_motion_anchor(self, history_xy: torch.Tensor) -> torch.Tensor:
        if history_xy.size(1) >= 2:
            last_delta = history_xy[:, -1, :] - history_xy[:, -2, :]
        else:
            last_delta = torch.zeros(history_xy.size(0), 2, device=history_xy.device, dtype=history_xy.dtype)
        anchor_steps = last_delta.unsqueeze(1).expand(-1, self.pred_steps, -1)
        return torch.cumsum(anchor_steps, dim=1)

    def _decode_future(
        self,
        fused_feat: torch.Tensor,
        text_tokens: torch.Tensor,
        attention_mask: torch.Tensor,
        history_xy: torch.Tensor,
        head: str,
    ) -> torch.Tensor:
        if self.future_decoder_type == "mlp":
            decoder = self.decoder_instruction if head == "instruction" else self.decoder_baseline
            output = decoder(fused_feat)
            return output.view(-1, self.pred_steps, 2)

        batch_size = fused_feat.size(0)
        device = fused_feat.device

        if self.future_decoder_type == "gru_delta":
            step_ids = torch.arange(self.pred_steps, device=device)
            time_feat = self.future_time_embed(step_ids).unsqueeze(0).expand(batch_size, -1, -1)
            context_feat = fused_feat.unsqueeze(1).expand(-1, self.pred_steps, -1)
            decoder_inputs = torch.cat([context_feat, time_feat], dim=-1)

            if head == "instruction":
                decoder_out, _ = self.decoder_instruction_gru(decoder_inputs, fused_feat.unsqueeze(0))
                delta_xy = self.decoder_instruction_head(decoder_out)
            else:
                decoder_out, _ = self.decoder_baseline_gru(decoder_inputs, fused_feat.unsqueeze(0))
                delta_xy = self.decoder_baseline_head(decoder_out)

            return torch.cumsum(delta_xy, dim=1)

        step_ids = torch.arange(self.pred_steps, device=device)
        future_queries = self.future_query_embed(step_ids).unsqueeze(0).expand(batch_size, -1, -1)
        future_queries = future_queries + fused_feat.unsqueeze(1)
        key_padding_mask = attention_mask == 0
        attended_future, _ = self.future_query_attn(
            query=future_queries,
            key=text_tokens,
            value=text_tokens,
            key_padding_mask=key_padding_mask,
        )
        future_feat = self.future_query_norm(attended_future + future_queries)
        fused_steps = self.future_query_ffn(
            torch.cat([future_feat, fused_feat.unsqueeze(1).expand(-1, self.pred_steps, -1)], dim=-1)
        )
        residual_head = self.decoder_instruction_head if head == "instruction" else self.decoder_baseline_head
        residual_xy = residual_head(fused_steps)
        motion_anchor = self._build_motion_anchor(history_xy)
        return motion_anchor + residual_xy

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
        text_cls = self.text_proj(text_outputs.last_hidden_state[:, 0, :])

        _, hidden_state = self.traj_encoder(history_xy)
        traj_feat = hidden_state[-1]
        query = self.query_proj(traj_feat).unsqueeze(1)
        attended_tokens, _ = self.cross_attn(query=query, key=text_tokens, value=text_tokens)
        attended_feat = attended_tokens.squeeze(1)
        cls_gate = self.cls_gate(torch.cat([attended_feat, text_cls], dim=-1))
        text_feat = attended_feat + cls_gate * text_cls

        fused_feat = self.fusion_mlp(torch.cat([text_feat, traj_feat], dim=-1))
        fused_feat = self.fusion_norm(fused_feat + traj_feat)
        return self._decode_future(
            fused_feat=fused_feat,
            text_tokens=text_tokens,
            attention_mask=text_inputs["attention_mask"],
            history_xy=history_xy,
            head=head,
        )
