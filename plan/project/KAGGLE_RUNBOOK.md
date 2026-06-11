# Kaggle Runbook

## Goal

Create a notebook the user can upload to Kaggle. When executed, it should:

1. Find the mounted source repo under `/kaggle/input`.
2. Copy the repo into `/kaggle/working/{repo}`.
3. Use the default Kaggle Python environment directly.
4. Resolve the mounted ViANLI split files automatically.
5. Run training.
6. Save outputs to Kaggle working directory.

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
2. Markdown: explain the config cell.
3. Code: define run variables and Kaggle mount hints.
4. Markdown: explain repo/data discovery.
5. Code: resolve mounted repo/data paths and copy the repo into `/kaggle/working`.
6. Markdown: explain default-environment verification.
7. Code: verify versions/GPU and optionally run tests with Kaggle's default Python.
8. Markdown: explain the training cell.
9. Code: run training.
10. Markdown: explain the artifact summary cell.
11. Code: show output files and metrics.

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
python scripts/train.py --help
python scripts/train.py \
  --config configs/default.yaml \
  --user-config configs/kaggle.yaml \
  --model-config configs/models/mbert_cased.yaml \
  --train-file /kaggle/input/.../train.jsonl \
  --validation-file /kaggle/input/.../validation.csv \
  --test-file /kaggle/input/.../test.jsonl \
  --output-dir /kaggle/working/outputs/runs \
  --run-name kaggle_mbert_vianli
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
The notebook should not create a separate virtual environment unless a future user explicitly asks for it.

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
- Every code cell has a markdown explanation directly above it.
- It can run mounted-source setup without manual shell editing.
- It can run default `mBERT + FFN` training against the mounted ViANLI data described in `INFO.md`.
- It uses Kaggle's default Python environment instead of creating a new virtual environment.
- It does not depend on a live GitHub clone to begin execution.
