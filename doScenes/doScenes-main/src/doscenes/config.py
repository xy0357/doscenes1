from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    path_file: str = "paths.txt"
    hist_sec: float = 2.0
    fut_sec: float = 6.0
    sample_freq: float = 2.0
    relative_coords: bool = True
    align_heading: bool = True
    coord_scale: float = 10.0
    window_stride: int = 2
    val_ratio: float = 0.2
    split_seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = True
    keep_empty_instruction: bool = True
    rebalance_non_empty_instruction: bool = True
    non_empty_instruction_weight: float = 2.0


@dataclass(frozen=True)
class ModelConfig:
    text_model_name: str = "distilbert-base-uncased"
    hidden_dim: int = 256
    pred_steps: int = 12
    freeze_text_encoder: bool = True
    local_files_only: bool = False
    attention_heads: int = 4
    text_unfreeze_last_n_layers: int = 0
    future_decoder_type: str = "mlp"


@dataclass(frozen=True)
class TrainingConfig:
    run_name: str = "baseline"
    epochs: int = 5
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 0.0
    checkpoint_dir: str = "artifacts/checkpoints"
    checkpoint_name: str = "baseline_best.pth"
    report_dir: str = "artifacts/experiments"
    early_stop_patience: int = 3
    early_stop_min_delta: float = 0.0
    eval_every_n_epochs: int = 1
    baseline_head_loss_weight: float = 1.0
    rank_loss_weight: float = 0.5
    rank_margin: float = 0.01
    fde_loss_weight: float = 0.0
    init_checkpoint: str | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    device: str = "auto"
    gpu_max_util: float = 0.8
    gpu_poll_sec: float = 0.5
    amp: bool = True
    matmul_precision: str = "high"


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    runtime: RuntimeConfig = RuntimeConfig()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def load_config(config_path: str | None) -> AppConfig:
    if not config_path:
        return AppConfig()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    data = DataConfig(**payload.get("data", {}))
    model = ModelConfig(**payload.get("model", {}))
    training = TrainingConfig(**payload.get("training", {}))
    runtime = RuntimeConfig(**payload.get("runtime", {}))
    return AppConfig(data=data, model=model, training=training, runtime=runtime)
