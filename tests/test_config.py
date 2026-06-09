from pathlib import Path

from vnnli_engram_moe.config import load_config


def test_default_config_loads() -> None:
    config = load_config("configs/default.yaml")

    assert config.project.name == "vnnli-engram-moe"
    assert config.training.max_length == 256
    assert config.training.num_train_epochs == 7
    assert config.model.architecture == "ffn"


def test_model_config_merges_into_default() -> None:
    config = load_config(
        "configs/default.yaml",
        model_config_path="configs/models/phobert_base.yaml",
    )

    assert config.model.model_key == "phobert_base"
    assert config.model.checkpoint == "vinai/phobert-base"
    assert config.model.requires_word_segmentation is True


def test_cli_style_overrides_work() -> None:
    output_dir = Path("outputs/tests")
    config = load_config(
        "configs/default.yaml",
        overrides=[
            "training.num_train_epochs=1",
            f"project.output_dir='{output_dir.as_posix()}'",
            "training.max_steps=3",
        ],
    )

    assert config.training.num_train_epochs == 1
    assert config.project.output_dir == output_dir.as_posix()
    assert config.training.max_steps == 3

