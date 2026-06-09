from __future__ import annotations

from collections.abc import Callable

from vnnli_engram_moe.config import AppConfig
from vnnli_engram_moe.models.engram_moe import build_engram_moe_model
from vnnli_engram_moe.models.ffn import build_ffn_model
from vnnli_engram_moe.models.moe import build_moe_model

ModelBuilder = Callable[[AppConfig], object]

MODEL_CHECKPOINTS = {
    "mbert_cased": "bert-base-multilingual-cased",
    "mbert_uncased": "bert-base-multilingual-uncased",
    "xlmr_base": "xlm-roberta-base",
    "xlmr_large": "xlm-roberta-large",
    "cafebert": "uitnlp/CafeBERT",
    "phobert_base": "vinai/phobert-base",
    "phobert_large": "vinai/phobert-large",
    "phobert_base_v2": "vinai/phobert-base-v2",
}

ARCHITECTURES = {
    "ffn": build_ffn_model,
    "moe": build_moe_model,
    "engram_moe": build_engram_moe_model,
}


def get_checkpoint(model_key: str) -> str:
    try:
        return MODEL_CHECKPOINTS[model_key]
    except KeyError as error:
        raise KeyError(
            f"Unknown model key '{model_key}'. Available keys: {', '.join(sorted(MODEL_CHECKPOINTS))}."
        ) from error


def get_builder(architecture: str):
    try:
        return ARCHITECTURES[architecture]
    except KeyError as error:
        raise KeyError(
            f"Unknown architecture '{architecture}'. Available keys: {', '.join(sorted(ARCHITECTURES))}."
        ) from error


def resolve_checkpoint(config: AppConfig) -> str:
    return config.model.checkpoint or get_checkpoint(config.model.model_key)


def build_model(config: AppConfig, *, checkpoint: str | None = None, pretrained: bool = True):
    builder = get_builder(config.model.architecture)
    resolved_checkpoint = checkpoint or resolve_checkpoint(config)
    return builder(config, checkpoint=resolved_checkpoint, pretrained=pretrained)

