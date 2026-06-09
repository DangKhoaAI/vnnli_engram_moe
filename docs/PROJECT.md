# Project Overview

This repository fine-tunes transformer baselines for Vietnamese natural language inference. It is designed to be usable in local development and Kaggle, while keeping later architectural work for MoE and Engram+MoE isolated behind registry interfaces.

## Architecture And Tech Stack

- Python `3.12`
- PyTorch for training loops and model execution
- Transformers for pretrained checkpoints and tokenizers
- YAML + dataclasses for centralized configuration
- Offline-first pytest suite with dummy model/tokenizer smoke coverage

## Directory Structure

```text
configs/
  default.yaml
  kaggle.yaml
  models/
docs/
notebooks/
scripts/
src/vnnli_engram_moe/
tests/
plan/
```

## Module / Function List

- `vnnli_engram_moe.config`
  Loads YAML config layers, applies CLI overrides, and serializes resolved configs.
- `vnnli_engram_moe.data.io`
  Reads JSONL, CSV, and optional Parquet ViANLI files.
- `vnnli_engram_moe.data.labels`
  Normalizes and maps the three NLI labels.
- `vnnli_engram_moe.data.preprocess`
  Builds the text preprocessor and paired tokenization flow.
- `vnnli_engram_moe.data.dataset`
  Produces encoded splits and PyTorch dataloaders.
- `vnnli_engram_moe.models.registry`
  Resolves model checkpoints and architecture builders.
- `vnnli_engram_moe.models.ffn`
  Implements the current baseline using `AutoModelForSequenceClassification`.
- `vnnli_engram_moe.models.moe`
  Reserved extension point that raises `NotImplementedError` today.
- `vnnli_engram_moe.models.engram_moe`
  Reserved extension point for future memory-augmented MoE work.
- `vnnli_engram_moe.training.trainer`
  Owns train/eval/predict orchestration and output artifact writing.
- `vnnli_engram_moe.cli`
  Exposes `train`, `evaluate`, and `predict` subcommands.

## Training Workflow

1. Load and merge config layers.
2. Load tokenizer and optional PhoBERT preprocessor.
3. Read train/dev/test data and encode paired premise/hypothesis inputs.
4. Build the configured architecture from the registry.
5. Train with AdamW.
6. Evaluate on validation/test splits.
7. Save `config.resolved.yaml`, metrics, metadata, label mapping, and `final_model/`.

## Output Artifacts

Each run writes to `outputs/runs/<timestamp>_<model_or_run_name>/`:

- `config.resolved.yaml`
- `run_metadata.json`
- `label_mapping.json`
- `train_metrics.json`
- `dev_metrics.json`
- `test_metrics.json`
- `final_model/`

## Future MoE / Engram+MoE Notes

The code already reserves:

- model registry keys: `moe`, `engram_moe`
- config namespaces: `moe`, `engram`
- dedicated modules for future implementation

That means the current CLI, config, and data pipeline should not need major refactors when those architectures are implemented later.

