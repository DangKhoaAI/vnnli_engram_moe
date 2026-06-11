# PROJECT

This document is the main project entry point for understanding what this repository is, how it is organized, and how it is expected to evolve.

## 1. Overview

This repository provides a fine-tuning codebase for Vietnamese Natural Language Inference (NLI). The current implementation focuses on a practical baseline for sentence-pair classification using pretrained transformer encoders, while keeping the project structure ready for future Mixture-of-Experts research without forcing a large refactor later.

The task is to predict one of three labels for a `(premise, hypothesis)` pair:

- `entailment`
- `contradiction`
- `neutral`

The project is driven by two human-authored source-of-truth files:

- `TASK.md`
- `INFO.md`

If any derived documentation disagrees with those two files, `TASK.md` and `INFO.md` take priority.

## 2. Scope

### Implemented now

- Design 1 baseline: transformer encoder plus classification head for 3-way NLI
- Centralized YAML configuration
- CLI entry points for training, evaluation, and prediction
- Offline-first tests
- Kaggle notebook for mounted-source setup and training flow

### Reserved for future work

- Design 2: `moe`
- Design 3: `engram_moe`

These future architectures are represented in the codebase as extension points, but they are not yet implemented as full training-ready models.

## 3. Architecture And Tech Stack

- Python `3.12.13`
- PyTorch `2.10.0+cu128`
- Transformers `5.0.0`
- YAML configuration plus typed Python config schema
- CLI-based workflow through `scripts/train.py`, `scripts/evaluate.py`, and `scripts/predict.py`
- Pytest for automated testing
- Kaggle-oriented execution path via `notebooks/kaggle_wrapup.ipynb`, using mounted Kaggle inputs for source code and ViANLI data

The architecture is intentionally conservative:

- configuration is centralized
- model creation is registry-driven
- dataset loading and preprocessing are isolated
- training orchestration is separated from model definitions

That separation is what allows the current FFN baseline to exist without blocking future MoE-style work.

## 4. Supported Model Scope

The repository is designed around the model families named in the task context:

- `bert-base-multilingual-cased`
- `xlm-roberta-base`
- `uitnlp/CafeBERT`
- `vinai/phobert-base`

The configuration system also makes it straightforward to add heavier or alternative variants later, but the current documented baseline centers on the core set above.

## 5. Data Contract

The expected NLI input schema is:

- `uid`
- `premise`
- `hypothesis`
- `label`

Supported labels are:

- `entailment`
- `contradiction`
- `neutral`

Supported ingestion paths:

- local JSONL
- local CSV
- optional local Parquet
- Hugging Face dataset loading for the ViANLI path used by this repository

## 6. Directory Structure

```text
configs/        global config and model-specific overrides
notebooks/      Kaggle notebook entry point
scripts/        train / evaluate / predict CLI entry points
src/            main Python package
tests/          unit tests and smoke tests
plan/           long-lived agent working space
```

Inside `plan/`:

```text
plan/project/   deeper project guidance and implementation planning
plan/status/    progress log, decisions, blockers, and agent workflow notes
```

## 7. Module / Function List

### Configuration

- `src/vnnli_engram_moe/config.py`
  Loads YAML layers, merges overrides, validates typed config objects, and writes resolved configs to run outputs.

### Data pipeline

- `src/vnnli_engram_moe/data/io.py`
  Loads ViANLI-style datasets from JSONL, CSV, optional Parquet, and the Hugging Face dataset path used by the repo.
- `src/vnnli_engram_moe/data/labels.py`
  Defines the canonical 3-label mapping.
- `src/vnnli_engram_moe/data/preprocess.py`
  Handles pair preprocessing and optional PhoBERT segmentation behavior.
- `src/vnnli_engram_moe/data/dataset.py`
  Converts raw splits into encoded datasets and dataloaders for training and evaluation.

### Models

- `src/vnnli_engram_moe/models/registry.py`
  Resolves model keys, checkpoints, and architecture builders.
- `src/vnnli_engram_moe/models/ffn.py`
  Implements the current Design 1 baseline path.
- `src/vnnli_engram_moe/models/moe.py`
  Placeholder module for future MoE implementation.
- `src/vnnli_engram_moe/models/engram_moe.py`
  Placeholder module for future Engram + MoE implementation.

### Training and inference

- `src/vnnli_engram_moe/training/trainer.py`
  Orchestrates training, evaluation, metric computation, artifact writing, and prediction flow.
- `src/vnnli_engram_moe/metrics.py`
  Computes evaluation metrics such as accuracy and macro-F1.
- `src/vnnli_engram_moe/cli.py`
  Exposes the internal command surface used by the thin scripts.

### CLI scripts

- `scripts/train.py`
  Main training entry point.
- `scripts/evaluate.py`
  Evaluates a saved run or model output directory.
- `scripts/predict.py`
  Runs single-example inference from a saved run.

## 8. Setup Flow

Typical local setup:

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest
```

Typical training flow with Hugging Face dataset input:

```bash
uv run python scripts/train.py \
  --config configs/default.yaml \
  --model-config configs/models/mbert_cased.yaml \
  --hf-dataset uitnlp/ViANLI
```

Typical training flow with local files:

```bash
uv run python scripts/train.py \
  --config configs/default.yaml \
  --model-config configs/models/mbert_cased.yaml \
  --train-file path/to/train.jsonl \
  --validation-file path/to/dev.jsonl \
  --test-file path/to/test.jsonl
```

Typical evaluation flow:

```bash
uv run python scripts/evaluate.py --run-dir outputs/runs/<run_name> --split test
```

Typical prediction flow:

```bash
uv run python scripts/predict.py \
  --run-dir outputs/runs/<run_name> \
  --premise "..." \
  --hypothesis "..."
```

## 9. Output Artifacts

Each training run writes a structured output directory under `outputs/runs/`. The run output typically includes:

- resolved config
- run metadata
- label mapping
- train metrics
- validation metrics
- test metrics
- final saved model

This layout is important because it gives both humans and future agents a stable place to inspect past experiments.

## 10. Extension Strategy

The repository is organized so future architecture work can plug into the existing flow rather than replacing it:

- data format stays the same
- config layering stays the same
- CLI entry points stay the same
- model selection happens through the registry

That means future `moe` and `engram_moe` work should mostly add new model implementations and config fields, instead of rewriting the project structure.

## 11. Reference Points

| PURPOSE | WHERE |
| --- | --- |
| Read the normalized project requirements derived from `TASK.md` and `INFO.md` | [plan/project/PROJECT_REQUIREMENTS.md](plan/project/PROJECT_REQUIREMENTS.md) |
| Read the target architecture and package structure | [plan/project/ARCHITECTURE_PLAN.md](plan/project/ARCHITECTURE_PLAN.md) |
| Read the implementation phases and completion checklist | [plan/project/IMPLEMENTATION_BACKLOG.md](plan/project/IMPLEMENTATION_BACKLOG.md) |
| Read the configuration strategy and experiment layout | [plan/project/CONFIG_AND_EXPERIMENTS.md](plan/project/CONFIG_AND_EXPERIMENTS.md) |
| Read the Kaggle execution plan and notebook contract | [plan/project/KAGGLE_RUNBOOK.md](plan/project/KAGGLE_RUNBOOK.md) |
| Read the testing scope and acceptance criteria | [plan/project/TESTING_AND_ACCEPTANCE.md](plan/project/TESTING_AND_ACCEPTANCE.md) |
| Read the documentation strategy for the repository | [plan/project/DOCS_PLAN.md](plan/project/DOCS_PLAN.md) |
| Read the uploadable Kaggle notebook | [notebooks/kaggle_wrapup.ipynb](notebooks/kaggle_wrapup.ipynb) |
| Read the current public project status | [STATUS.md](STATUS.md) |
