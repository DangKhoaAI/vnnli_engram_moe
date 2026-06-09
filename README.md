# Vietnamese NLI Fine-tuning Toolkit

Source code project for fine-tuning transformer baselines on Vietnamese Natural Language Inference (ViANLI / adversarial NLI), with clean extension points for future `moe` and `engram_moe` research.

The current implemented baseline is Design 1: transformer encoder + classification head for the three labels `entailment`, `contradiction`, and `neutral`. The repo also includes configs, CLI scripts, offline-first tests, and a Kaggle notebook that can clone the project and run setup/training.

## Environment

- Python `3.12.13`
- Target production stack: PyTorch `2.10.0+cu128`, Transformers `5.0.0`
- Local workflow uses `uv`

## Installation

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Dataset Format

Expected columns:

- `uid`
- `premise`
- `hypothesis`
- `label`

Accepted files today:

- `.jsonl`
- `.csv`
- `.parquet` when parquet support is available in the environment

Example JSONL row:

```json
{"uid":"uit_Adver_365_3_11_02","premise":"...","hypothesis":"...","label":"entailment"}
```

## Quickstart

Train:

```bash
uv run python scripts/train.py \
  --config configs/default.yaml \
  --model-config configs/models/mbert_cased.yaml \
  --train-file data/ViANLI/train.jsonl \
  --validation-file data/ViANLI/dev.jsonl \
  --test-file data/ViANLI/test.jsonl
```

Evaluate a saved run:

```bash
uv run python scripts/evaluate.py --run-dir outputs/runs/<timestamp>_mbert_cased --split test
```

Predict a single pair:

```bash
uv run python scripts/predict.py \
  --run-dir outputs/runs/<timestamp>_mbert_cased \
  --premise "Công ty mở thêm văn phòng tại Đà Nẵng." \
  --hypothesis "Công ty mở rộng hiện diện tại miền Trung."
```

## Config Strategy

All tunable values live in YAML:

- `configs/default.yaml`: project-wide defaults
- `configs/models/*.yaml`: model-specific overrides
- optional experiment configs can layer on top later

CLI overrides are also supported via `--override KEY=VALUE`.

## Project Structure

```text
configs/      YAML configs and model overrides
docs/         project, data, experiment, and Kaggle documentation
notebooks/    Kaggle wrapup notebook
scripts/      thin train/evaluate/predict entrypoints
src/          package implementation
tests/        offline-first unit and smoke tests
plan/         mutable planning and handoff state
```

## PhoBERT Note

PhoBERT can optionally use VnCoreNLP-style word segmentation. The hook is present in the codebase and will try `py_vncorenlp` when installed. By default the project falls back to no-op segmentation unless strict mode is enabled later.

## Verification

Default validation path:

```bash
uv run pytest
uv run python scripts/train.py --help
uv run python scripts/evaluate.py --help
uv run python scripts/predict.py --help
```

## Current Status

- Implemented now: Design 1 FFN baseline path
- Reserved for future work: `moe`, `engram_moe`
- Handoff status lives in `plan/STATE.md`

More detailed docs are in [docs/PROJECT.md](/home/khoa/KHOA/FPTStudy/Semester7/DAT301m/PROJECT/train_model/docs/PROJECT.md).

