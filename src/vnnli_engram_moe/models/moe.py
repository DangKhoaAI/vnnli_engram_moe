from __future__ import annotations

from vnnli_engram_moe.config import AppConfig


def build_moe_model(config: AppConfig, *, checkpoint: str, pretrained: bool = True):
    raise NotImplementedError(
        "The 'moe' architecture is reserved for future work. "
        "Keep using architecture=ffn for the current baseline."
    )

