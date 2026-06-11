# Kaggle Runbook

## Goal

Create a notebook the user can upload to Kaggle. When executed, it should:

1. Find the mounted source repo under `/kaggle/input`.
2. Copy the repo into `/kaggle/working/{repo}`.
3. Create a local environment from the current Kaggle interpreter.
4. Install the project in editable mode without depending on a live network download.
5. Resolve the mounted ViANLI split files automatically.
6. Run training.
7. Save outputs to Kaggle working directory.

## Current Kaggle Mount Hints

The current human-provided Kaggle paths in `INFO.md` are:

```text
/kaggle/input/datasets/khoa05ai/fu-s7dat-tuningmodel/vnnli_engram_moe-main
/kaggle/input/datasets/khoa05ai/fu-s7dat-vieanli/vianli_kaggle
```

The notebook should still keep these as editable hints, not hard requirements.

## Notebook Location

Target file:

```text
notebooks/kaggle_wrapup.ipynb
```

The primary notebook target is `notebooks/kaggle_wrapup.ipynb`. If a future user explicitly asks for a different top-level notebook filename, treat that as a separate repo-surface decision.

## Notebook Structure

Recommended cells:

1. Markdown: title and variables explanation.
2. Code: define run variables and Kaggle mount hints.
3. Code: resolve mounted repo/data paths and copy the repo into `/kaggle/working`.
4. Code: create `.venv`, install the project, and verify versions/GPU.
5. Code: run optional tests/help checks.
6. Code: run training.
7. Code: show output files and metrics.

## Suggested Variables

```python
REPO_SOURCE_HINT = "/kaggle/input/datasets/khoa05ai/fu-s7dat-tuningmodel/vnnli_engram_moe-main"
DATASET_DIR_HINT = "/kaggle/input/datasets/khoa05ai/fu-s7dat-vieanli/vianli_kaggle"
REPO_DIR = "/kaggle/working/vnnli_engram_moe"
OUTPUT_ROOT = "/kaggle/working/outputs/runs"

MODEL_CONFIG = "configs/models/mbert_cased.yaml"
BASE_CONFIG = "configs/default.yaml"
USER_CONFIG = "configs/kaggle.yaml"
RUN_NAME = "kaggle_mbert_vianli"

RUN_PYTEST = False
USE_FP16 = True
MAKE_OUTPUT_ZIP = False
EXTRA_OVERRIDES = []
```

## Suggested Notebook Flow

```bash
cp -r /kaggle/input/.../vnnli_engram_moe-main /kaggle/working/vnnli_engram_moe
cd /kaggle/working/vnnli_engram_moe
uv venv --python "$(python -c 'import sys; print(sys.executable)')" --system-site-packages
uv pip install --python .venv/bin/python --no-deps -e ".[dev]"
.venv/bin/python scripts/train.py \
  --config configs/default.yaml \
  --user-config configs/kaggle.yaml \
  --model-config configs/models/mbert_cased.yaml \
  --train-file /kaggle/input/.../train.jsonl \
  --validation-file /kaggle/input/.../validation.csv \
  --test-file /kaggle/input/.../test.jsonl \
  --output-dir /kaggle/working/outputs/runs \
  --run-name kaggle_mbert_vianli
```

If `uv` is unavailable in the Kaggle image, the notebook should fall back to:

```bash
python -m venv .venv --system-site-packages
.venv/bin/python -m pip install --no-deps -e ".[dev]"
```

## Kaggle Config Differences

`configs/kaggle.yaml` should override:

```yaml
project:
  output_dir: /kaggle/working/outputs/runs

training:
  fp16: true
  num_workers: 2
```

If Kaggle GPU does not support fp16 well, notebook should expose a variable to turn it off.

## Dataset Mounting Assumption

The notebook should not assume the exact Kaggle dataset slug. It should make the mount hints easy to edit in one place and then auto-detect the rest.

Support these likely names:

- `train.jsonl`, `validation.jsonl`, `test.jsonl`
- `train.csv`, `validation.csv`, `test.csv`
- `dev.jsonl` / `dev.csv` as a validation fallback

If files are missing, notebook should print the contents of `/kaggle/input` to help the user adjust paths.
If only `validation.json` exists, the notebook should fail with a clear message because the repo expects JSONL, CSV, or Parquet for direct file loading.

## Output Contract

After training, notebook should display:

- output run directory
- `dev_metrics.json`
- `test_metrics.json`
- path to `final_model/`

Optional:

- zip the final output:

```bash
cd /kaggle/working
zip -r vnnli_outputs.zip outputs
```

## Kaggle Acceptance

Notebook is acceptable when:

- It is valid `.ipynb` JSON.
- It has no local machine paths.
- All user-adjustable values are in one top cell.
- It can run mounted-source setup without manual shell editing.
- It can run default `mBERT + FFN` training against the mounted ViANLI data described in `INFO.md`.
- It does not depend on a live GitHub clone to begin execution.
