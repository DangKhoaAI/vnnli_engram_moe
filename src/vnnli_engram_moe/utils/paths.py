from __future__ import annotations

from datetime import datetime
from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def create_run_directory(base_dir: str | Path, model_key: str, run_name: str | None = None) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    suffix = run_name or model_key
    return ensure_directory(Path(base_dir) / f"{timestamp}_{suffix}")
