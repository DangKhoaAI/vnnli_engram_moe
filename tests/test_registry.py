import pytest

from vnnli_engram_moe.config import load_config
from vnnli_engram_moe.models.registry import build_model, get_builder, get_checkpoint


def test_known_model_keys_resolve() -> None:
    assert get_checkpoint("mbert_cased") == "bert-base-multilingual-cased"
    assert get_checkpoint("phobert_base") == "vinai/phobert-base"
    assert get_checkpoint("videberta_base") == "Fsoft-AIC/videberta-base"


def test_known_architectures_exist() -> None:
    assert callable(get_builder("ffn"))
    assert callable(get_builder("moe"))
    assert callable(get_builder("engram_moe"))


def test_future_engram_architecture_raises_explicitly() -> None:
    config = load_config("configs/default.yaml", overrides=["model.architecture='engram_moe'"])
    with pytest.raises(NotImplementedError):
        build_model(config, checkpoint="dummy-checkpoint", pretrained=False)


def test_unknown_entries_raise() -> None:
    with pytest.raises(KeyError):
        get_checkpoint("missing_model")
    with pytest.raises(KeyError):
        get_builder("missing_architecture")
