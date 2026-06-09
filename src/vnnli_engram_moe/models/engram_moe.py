from __future__ import annotations

from vnnli_engram_moe.config import AppConfig


def build_engram_moe_model(config: AppConfig, *, checkpoint: str, pretrained: bool = True):
    raise NotImplementedError(
        "The 'engram_moe' architecture is reserved for future research work. "
        "The registry path is ready so future implementations can plug in cleanly."
    )

