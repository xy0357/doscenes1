from doscenes.config import load_config


def test_load_config_supports_init_checkpoint() -> None:
    cfg = load_config("configs/ade_finetune.json")
    assert cfg.training.init_checkpoint == "artifacts/checkpoints/baseline_best.pth"
    assert cfg.data.keep_empty_instruction is False
    assert cfg.training.baseline_head_loss_weight == 0.05


def test_load_config_supports_ade_push_profile() -> None:
    cfg = load_config("configs/ade_push.json")
    assert cfg.training.init_checkpoint == "artifacts/checkpoints/ade_finetune_best.pth"
    assert cfg.training.baseline_head_loss_weight == 0.0
    assert cfg.training.rank_loss_weight == 0.0


def test_load_config_supports_ade_recover_profile() -> None:
    cfg = load_config("configs/ade_recover.json")
    assert cfg.training.init_checkpoint == "artifacts/checkpoints/baseline_best.pth"
    assert cfg.training.baseline_head_loss_weight == 0.15
    assert cfg.training.rank_loss_weight == 0.15
    assert cfg.model.future_decoder_type == "gru_delta"


def test_load_config_supports_ade_research_v1_profile() -> None:
    cfg = load_config("configs/ade_research_v1.json")
    assert cfg.training.init_checkpoint == "artifacts/checkpoints/baseline_best.pth"
    assert cfg.model.hidden_dim == 256
    assert cfg.model.future_decoder_type == "gru_delta"


def test_load_config_supports_ade_query_residual_profile() -> None:
    cfg = load_config("configs/ade_query_residual.json")
    assert cfg.training.init_checkpoint == "artifacts/checkpoints/baseline_best.pth"
    assert cfg.model.hidden_dim == 192
    assert cfg.model.future_decoder_type == "query_residual"
