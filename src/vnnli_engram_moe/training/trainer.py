from __future__ import annotations

import copy
import json
import logging
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from vnnli_engram_moe.config import AppConfig, config_to_dict, load_run_config, save_config
from vnnli_engram_moe.data.dataset import EncodedNLIDataset, collate_batch, load_available_splits
from vnnli_engram_moe.data.preprocess import build_preprocessor
from vnnli_engram_moe.metrics import compute_classification_metrics
from vnnli_engram_moe.models.registry import build_model, resolve_checkpoint
from vnnli_engram_moe.training.seed import set_random_seed
from vnnli_engram_moe.utils.paths import create_run_directory, ensure_directory

LOGGER = logging.getLogger(__name__)


def load_tokenizer(config: AppConfig, *, checkpoint: str | None = None):
    from transformers import AutoTokenizer

    resolved_checkpoint = checkpoint or resolve_checkpoint(config)
    return AutoTokenizer.from_pretrained(
        resolved_checkpoint,
        use_fast=config.model.use_fast_tokenizer,
        local_files_only=config.model.local_files_only,
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def save_json(payload: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _extract_loss_and_logits(model_output) -> tuple[torch.Tensor | None, torch.Tensor]:
    if isinstance(model_output, dict):
        return model_output.get("loss"), model_output["logits"]
    loss = getattr(model_output, "loss", None)
    logits = getattr(model_output, "logits")
    return loss, logits


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _build_dataloader(
    examples: list[dict[str, object]],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = EncodedNLIDataset(examples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
    )


def evaluate_model(model, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    predictions: list[int] = []
    labels: list[int] = []
    total_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = _move_batch_to_device(batch, device)
            model_output = model(**batch)
            loss, logits = _extract_loss_and_logits(model_output)
            if loss is not None:
                total_loss += float(loss.item())
            batch_predictions = torch.argmax(logits, dim=-1)
            predictions.extend(batch_predictions.cpu().tolist())
            labels.extend(batch["labels"].cpu().tolist())
            batch_count += 1

    metrics = compute_classification_metrics(predictions, labels)
    metrics["loss"] = total_loss / batch_count if batch_count else 0.0
    return metrics


def _package_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
    }
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except ImportError:
        versions["transformers"] = "not-installed"
    return versions


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _save_training_metadata(config: AppConfig, run_dir: Path) -> None:
    metadata = {
        "project": config.project.name,
        "model_key": config.model.model_key,
        "checkpoint": resolve_checkpoint(config),
        "architecture": config.model.architecture,
        "versions": _package_versions(),
        "git_commit": _git_commit(),
    }
    save_json(metadata, run_dir / "run_metadata.json")
    save_json({"label2id": {"entailment": 0, "contradiction": 1, "neutral": 2}}, run_dir / "label_mapping.json")


def train(
    config: AppConfig,
    *,
    tokenizer_builder=load_tokenizer,
    model_builder=build_model,
    output_dir: str | Path | None = None,
    pretrained: bool = True,
) -> Path:
    set_random_seed(config.project.seed)
    run_dir = create_run_directory(
        output_dir or config.project.output_dir,
        config.model.model_key,
        config.project.run_name,
    )
    save_config(config, run_dir / "config.resolved.yaml")
    _save_training_metadata(config, run_dir)

    checkpoint = resolve_checkpoint(config)
    tokenizer = tokenizer_builder(config, checkpoint=checkpoint)
    preprocessor = build_preprocessor(config.model)
    splits = load_available_splits(config, tokenizer, preprocessor)
    if "train" not in splits:
        raise ValueError("Training requires data.train_file to be configured.")

    train_loader = _build_dataloader(
        splits["train"].examples,
        batch_size=config.training.per_device_train_batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    validation_loader = None
    if "validation" in splits:
        validation_loader = _build_dataloader(
            splits["validation"].examples,
            batch_size=config.training.per_device_eval_batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
        )
    test_loader = None
    if "test" in splits:
        test_loader = _build_dataloader(
            splits["test"].examples,
            batch_size=config.training.per_device_eval_batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
        )

    model = model_builder(config, checkpoint=checkpoint, pretrained=pretrained)
    device = resolve_device(config.training.device)
    LOGGER.info("Using device: %s", device)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        eps=config.training.adam_epsilon,
        weight_decay=config.training.weight_decay,
    )

    global_step = 0
    train_loss_total = 0.0
    optimizer.zero_grad(set_to_none=True)
    best_metric = float("-inf")
    best_state_dict = None
    best_validation_metrics: dict[str, float] | None = None
    metric_name = config.training.metric_for_best_model

    steps_per_epoch = max(1, math.ceil(len(train_loader) / config.training.gradient_accumulation_steps))
    total_target_steps = config.training.max_steps or (steps_per_epoch * config.training.num_train_epochs)

    for epoch in range(config.training.num_train_epochs):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}", leave=False)
        for batch_index, batch in enumerate(progress, start=1):
            batch = _move_batch_to_device(batch, device)
            model_output = model(**batch)
            loss, _ = _extract_loss_and_logits(model_output)
            if loss is None:
                raise RuntimeError("Training requires the model output to include a loss value.")
            normalized_loss = loss / config.training.gradient_accumulation_steps
            normalized_loss.backward()

            train_loss_total += float(loss.item())
            if batch_index % config.training.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % config.training.logging_steps == 0 or global_step == 1:
                    progress.set_postfix(step=global_step, loss=f"{loss.item():.4f}")

                if validation_loader and global_step % config.training.eval_frequency == 0:
                    validation_metrics = evaluate_model(model, validation_loader, device)
                    validation_metrics["step"] = float(global_step)
                    save_json(validation_metrics, run_dir / "validation_metrics.latest.json")
                    current_metric = validation_metrics.get(metric_name, float("-inf"))
                    if current_metric > best_metric:
                        best_metric = current_metric
                        best_state_dict = copy.deepcopy(model.state_dict())
                        best_validation_metrics = validation_metrics

                if config.training.max_steps and global_step >= config.training.max_steps:
                    break

            if global_step >= total_target_steps:
                break

        if config.training.max_steps and global_step >= config.training.max_steps:
            break

    if best_state_dict is not None and config.training.load_best_model_at_end:
        model.load_state_dict(best_state_dict)

    final_model_dir = ensure_directory(run_dir / "final_model")
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(final_model_dir)
    else:
        torch.save(model.state_dict(), final_model_dir / "pytorch_model.bin")
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(final_model_dir)

    average_train_loss = train_loss_total / max(1, global_step)
    train_metrics = {
        "global_step": float(global_step),
        "epochs_configured": float(config.training.num_train_epochs),
        "train_loss": average_train_loss,
    }
    save_json(train_metrics, run_dir / "train_metrics.json")

    if best_validation_metrics is not None:
        save_json(best_validation_metrics, run_dir / "validation_metrics.best.json")

    if validation_loader:
        validation_metrics = evaluate_model(model, validation_loader, device)
        save_json(validation_metrics, run_dir / "dev_metrics.json")

    if test_loader:
        test_metrics = evaluate_model(model, test_loader, device)
        save_json(test_metrics, run_dir / "test_metrics.json")

    return run_dir


def evaluate_saved_run(
    *,
    run_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    split: str = "test",
    config: AppConfig | None = None,
    data_file: str | None = None,
    tokenizer_builder=load_tokenizer,
    model_builder=build_model,
) -> dict[str, float]:
    if run_dir is None and config is None:
        raise ValueError("Pass either run_dir or an explicit config for evaluation.")

    if run_dir is not None:
        run_path = Path(run_dir)
        config = load_run_config(run_path)
        checkpoint = checkpoint or (run_path / "final_model")
    assert config is not None

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be one of: train, validation, test")

    if data_file is not None:
        setattr(config.data, f"{split}_file", data_file)

    resolved_checkpoint = str(checkpoint or resolve_checkpoint(config))
    tokenizer = tokenizer_builder(config, checkpoint=resolved_checkpoint)
    preprocessor = build_preprocessor(config.model)
    splits = load_available_splits(config, tokenizer, preprocessor)
    if split not in splits:
        raise ValueError(f"No file configured for split '{split}'.")

    dataloader = _build_dataloader(
        splits[split].examples,
        batch_size=config.training.per_device_eval_batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    model = model_builder(config, checkpoint=resolved_checkpoint, pretrained=True)
    device = resolve_device(config.training.device)
    model.to(device)
    metrics = evaluate_model(model, dataloader, device)

    if run_dir is not None:
        save_json(metrics, Path(run_dir) / f"{split}_metrics.json")
    return metrics


def predict_text_pair(
    *,
    premise: str,
    hypothesis: str,
    run_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    config: AppConfig | None = None,
    tokenizer_builder=load_tokenizer,
    model_builder=build_model,
) -> dict[str, Any]:
    if run_dir is None and config is None:
        raise ValueError("Pass either run_dir or an explicit config for prediction.")

    if run_dir is not None:
        run_path = Path(run_dir)
        config = load_run_config(run_path)
        checkpoint = checkpoint or (run_path / "final_model")
    assert config is not None

    resolved_checkpoint = str(checkpoint or resolve_checkpoint(config))
    tokenizer = tokenizer_builder(config, checkpoint=resolved_checkpoint)
    preprocessor = build_preprocessor(config.model)
    model = model_builder(config, checkpoint=resolved_checkpoint, pretrained=True)
    device = resolve_device(config.training.device)
    model.to(device)
    model.eval()

    encoded = tokenizer(
        preprocessor.prepare(premise),
        preprocessor.prepare(hypothesis),
        truncation=True,
        padding="max_length",
        max_length=config.training.max_length,
        return_tensors="pt",
    )
    encoded = _move_batch_to_device(encoded, device)

    with torch.no_grad():
        model_output = model(**encoded)
        _, logits = _extract_loss_and_logits(model_output)
        probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu()

    predicted_index = int(torch.argmax(probabilities).item())
    labels = ["entailment", "contradiction", "neutral"]
    return {
        "label": labels[predicted_index],
        "scores": {label: float(probabilities[index]) for index, label in enumerate(labels)},
        "checkpoint": resolved_checkpoint,
        "config": config_to_dict(config),
    }
