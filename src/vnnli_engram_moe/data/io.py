from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from vnnli_engram_moe.config import DataConfig


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at {path}:{line_number} must be an object.")
            records.append(record)
    return records


def _read_csv(path: Path, delimiter: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as error:
        raise RuntimeError("Reading parquet files requires pandas to be installed.") from error
    return pd.read_parquet(path).to_dict(orient="records")


def read_records(path: str | Path, data_config: DataConfig) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(source)
    if suffix == ".csv":
        return _read_csv(source, data_config.delimiter)
    if suffix in {".parquet", ".pq"}:
        return _read_parquet(source)
    raise ValueError(f"Unsupported dataset format for {source}. Use JSONL, CSV, or Parquet.")


def validate_records(records: list[dict[str, Any]], data_config: DataConfig) -> None:
    required_columns = {
        data_config.uid_column,
        data_config.text_columns.premise,
        data_config.text_columns.hypothesis,
        data_config.label_column,
    }
    if not records:
        raise ValueError("Dataset is empty.")
    missing = required_columns.difference(records[0])
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_text}")

