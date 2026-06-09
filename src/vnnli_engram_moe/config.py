from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from typing import get_type_hints

import yaml


@dataclass(slots=True)
class ProjectConfig:
    name: str = "vnnli-engram-moe"
    seed: int = 42
    output_dir: str = "outputs/runs"
    run_name: str | None = None


@dataclass(slots=True)
class DataTextColumns:
    premise: str = "premise"
    hypothesis: str = "hypothesis"


@dataclass(slots=True)
class DataConfig:
    train_file: str = "data/ViANLI/train.jsonl"
    validation_file: str = "data/ViANLI/dev.jsonl"
    test_file: str = "data/ViANLI/test.jsonl"
    text_columns: DataTextColumns = field(default_factory=DataTextColumns)
    label_column: str = "label"
    uid_column: str = "uid"
    delimiter: str = ","


@dataclass(slots=True)
class ModelConfig:
    model_key: str = "mbert_cased"
    checkpoint: str = "bert-base-multilingual-cased"
    architecture: str = "ffn"
    num_labels: int = 3
    use_fast_tokenizer: bool = True
    requires_word_segmentation: bool = False
    allow_noop_word_segmenter: bool = True


@dataclass(slots=True)
class TrainingConfig:
    max_length: int = 256
    learning_rate: float = 1.0e-5
    eval_frequency: int = 400
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    weight_decay: float = 0.0
    adam_epsilon: float = 1.0e-8
    dropout: float = 0.4
    num_train_epochs: int = 7
    warmup_ratio: float = 0.0
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    bf16: bool = False
    logging_steps: int = 50
    save_steps: int = 400
    max_steps: int | None = None
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "macro_f1"
    device: str = "auto"
    num_workers: int = 0


@dataclass(slots=True)
class MoeConfig:
    enabled: bool = False
    num_experts: int = 4
    top_k: int = 2
    expert_hidden_size: int | None = None
    router_temperature: float = 1.0
    load_balance_loss_weight: float = 0.01


@dataclass(slots=True)
class EngramConfig:
    enabled: bool = False
    memory_size: int = 0
    retrieval_top_k: int = 0
    update_policy: str = "none"
    fusion: str = "none"


@dataclass(slots=True)
class AppConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    moe: MoeConfig = field(default_factory=MoeConfig)
    engram: EngramConfig = field(default_factory=EngramConfig)


def _load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at the top level.")
    return data


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse_override_value(raw_value: str) -> Any:
    return yaml.safe_load(raw_value)


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return config
    result = dict(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Override '{override}' must have KEY=VALUE format.")
        dotted_key, raw_value = override.split("=", 1)
        keys = dotted_key.split(".")
        cursor = result
        for key in keys[:-1]:
            child = cursor.get(key)
            if child is None:
                child = {}
                cursor[key] = child
            if not isinstance(child, dict):
                raise ValueError(f"Cannot set nested override for non-mapping key '{key}'.")
            cursor = child
        cursor[keys[-1]] = _parse_override_value(raw_value)
    return result


def _build_dataclass(data_class: type[Any], values: dict[str, Any]) -> Any:
    type_hints = get_type_hints(data_class)
    kwargs: dict[str, Any] = {}
    for field_name, field_info in data_class.__dataclass_fields__.items():  # type: ignore[attr-defined]
        current_value = values.get(field_name)
        field_type = type_hints.get(field_name, field_info.type)
        if hasattr(field_type, "__dataclass_fields__") and isinstance(current_value, dict):
            kwargs[field_name] = _build_dataclass(field_type, current_value)
        elif current_value is not None:
            kwargs[field_name] = current_value
    return data_class(**kwargs)


def load_config(
    config_path: str | Path,
    *,
    model_config_path: str | Path | None = None,
    user_config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> AppConfig:
    merged = _load_yaml(config_path)
    for optional_path in (model_config_path, user_config_path):
        merged = _deep_merge(merged, _load_yaml(optional_path))
    merged = apply_overrides(merged, overrides)
    return _build_dataclass(AppConfig, merged)


def config_to_dict(config: AppConfig) -> dict[str, Any]:
    return asdict(config)


def save_config(config: AppConfig, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(config), handle, allow_unicode=True, sort_keys=False)


def load_run_config(run_dir: str | Path) -> AppConfig:
    return load_config(Path(run_dir) / "config.resolved.yaml")
