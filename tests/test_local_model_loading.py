import argparse
import sys
from types import SimpleNamespace

from vnnli_engram_moe.cli import _build_common_overrides
from vnnli_engram_moe.config import load_config
from vnnli_engram_moe.models.ffn import build_ffn_model
from vnnli_engram_moe.training.trainer import load_tokenizer


def test_train_cli_checkpoint_override_wins_and_enables_local_only() -> None:
    args = argparse.Namespace(
        override=[],
        model_key="mbert_cased",
        checkpoint="/kaggle/input/models/mbert",
        local_files_only=True,
    )

    overrides = _build_common_overrides(args)
    checkpoint_overrides = [item for item in overrides if item.startswith("model.checkpoint=")]

    assert checkpoint_overrides[-1] == 'model.checkpoint="/kaggle/input/models/mbert"'
    assert "model.local_files_only=true" in overrides


def test_load_tokenizer_respects_local_files_only(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class RecordingAutoTokenizer:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            calls.append((checkpoint, kwargs))
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=RecordingAutoTokenizer),
    )

    config = load_config(
        "configs/default.yaml",
        overrides=["model.local_files_only=true"],
    )

    load_tokenizer(config, checkpoint="/mounted/mbert")

    assert calls == [
        (
            "/mounted/mbert",
            {
                "use_fast": True,
                "local_files_only": True,
            },
        )
    ]


def test_build_ffn_model_passes_local_files_only_to_transformers(monkeypatch) -> None:
    config_calls: list[tuple[str, dict[str, object]]] = []
    model_calls: list[tuple[str, dict[str, object]]] = []

    class RecordingAutoConfig:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            config_calls.append((checkpoint, kwargs))
            return SimpleNamespace()

    class RecordingAutoModelForSequenceClassification:
        @classmethod
        def from_pretrained(cls, checkpoint, **kwargs):
            model_calls.append((checkpoint, kwargs))
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoConfig=RecordingAutoConfig,
            AutoModelForSequenceClassification=RecordingAutoModelForSequenceClassification,
        ),
    )

    config = load_config(
        "configs/default.yaml",
        overrides=["model.local_files_only=true"],
    )

    build_ffn_model(config, checkpoint="/mounted/mbert", pretrained=True)

    assert config_calls == [
        (
            "/mounted/mbert",
            {
                "local_files_only": True,
            },
        )
    ]
    assert model_calls
    checkpoint, kwargs = model_calls[0]
    assert checkpoint == "/mounted/mbert"
    assert kwargs["local_files_only"] is True
    assert kwargs["ignore_mismatched_sizes"] is True
    assert kwargs["config"].num_labels == 3
