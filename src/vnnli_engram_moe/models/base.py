from __future__ import annotations

from dataclasses import dataclass

from vnnli_engram_moe.config import AppConfig


@dataclass(slots=True)
class BuildContext:
    config: AppConfig
    checkpoint: str
    pretrained: bool = True

