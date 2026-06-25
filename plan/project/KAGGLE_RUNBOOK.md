# Kaggle Runbook

## Goal

Create a notebook the user can upload to Kaggle. When executed, it should:

1. Find the mounted source repo under `/kaggle/input`.
2. Copy the repo into `/kaggle/working/{repo}`.
3. Resolve the mounted ViANLI split files automatically.
4. Resolve the mounted local model directory automatically.
5. Use the default Kaggle Python environment directly.
6. Run training without depending on internet access.
7. Save outputs to Kaggle working directory.

## Current Kaggle Mount Hints

The current human-provided Kaggle paths in `INFO.md` are:

```text
/kaggle/input/datasets/khoa05ai/fu-s7dat-tuningmodel/vnnli_engram_moe-main
/kaggle/input/datasets/khoa05ai/fu-s7dat-vieanli/vianli_kaggle
```

The previously used mBERT model mount may still be used for FFN runs, but the current notebook default is ViDeBERTa + MoE. Keep `MODEL_SOURCE_HINT` editable because `INFO.md` does not currently provide a concrete ViDeBERTa Kaggle model mount path.

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
4. Markdown: explain repo/data/model discovery.
5. Code: resolve mounted repo/data/model paths and copy the repo into `/kaggle/working`.
6. Markdown: explain default-environment verification.
7. Code: verify versions/GPU and smoke-load the local model with Kaggle's default Python.
8. Markdown: explain the training cell.
9. Code: run training with the mounted model directory.
10. Markdown: explain the artifact summary cell.
11. Code: show output files and metrics.

## Suggested Variables

```python
REPO_SOURCE_HINT = "/kaggle/input/datasets/khoa05ai/fu-s7dat-tuningmodel/vnnli_engram_moe-main"
DATASET_DIR_HINT = "/kaggle/input/datasets/khoa05ai/fu-s7dat-vieanli/vianli_kaggle"
MODEL_SOURCE_HINT = ""
MODEL_KEY_HINTS = ["videberta", "deberta", "fsoft"]
REPO_DIR = "/kaggle/working/vnnli_engram_moe"
OUTPUT_ROOT = "/kaggle/working/outputs/runs"

MODEL_CONFIG = "configs/models/videberta_base_moe.yaml"
BASE_CONFIG = "configs/default.yaml"
USER_CONFIG = "configs/kaggle.yaml"
RUN_NAME = "kaggle_videberta_moe_vianli"

RUN_PYTEST = False
USE_FP16 = True
LOCAL_FILES_ONLY = True
MAKE_OUTPUT_ZIP = False
EXTRA_OVERRIDES = []
```

## Suggested Notebook Flow

```bash
cp -r /kaggle/input/.../vnnli_engram_moe-main /kaggle/working/vnnli_engram_moe
cd /kaggle/working/vnnli_engram_moe
python scripts/train.py --help
/usr/bin/python3 /kaggle/working/vnnli_engram_moe/scripts/train.py \
  --config /kaggle/working/vnnli_engram_moe/configs/default.yaml \
  --user-config /kaggle/working/vnnli_engram_moe/configs/kaggle.yaml \
  --model-config /kaggle/working/vnnli_engram_moe/configs/models/videberta_base_moe.yaml \
  --checkpoint /kaggle/input/.../videberta-base \
  --local-files-only \
  --train-file /kaggle/input/.../train.jsonl \
  --validation-file /kaggle/input/.../validation.csv \
  --test-file /kaggle/input/.../test.jsonl \
  --output-dir /kaggle/working/outputs/runs \
  --run-name kaggle_videberta_moe_vianli
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
The notebook should default to a mounted model directory plus `--local-files-only` so `transformers` does not try to reach the Hugging Face Hub.

## Dataset Mounting Assumption

The notebook should not assume the exact Kaggle dataset slug. It should make the mount hints easy to edit in one place and then auto-detect the rest.

Support these likely names:

- `train.jsonl`, `validation.jsonl`, `test.jsonl`
- `train.csv`, `validation.csv`, `test.csv`
- `dev.jsonl` / `dev.csv` as a validation fallback

If files are missing, notebook should print the contents of `/kaggle/input` to help the user adjust paths.
If only `validation.json` exists, the notebook should fail with a clear message because the repo expects JSONL, CSV, or Parquet for direct file loading.

## Model Mounting Assumption

The notebook should not assume internet access for model downloads.

Support this mounted-model pattern:

- a directory containing `config.json`
- at least one weight file such as `model.safetensors` or `pytorch_model.bin`
- tokenizer files such as `tokenizer.json`, `tokenizer_config.json`, `vocab.txt`, or `tokenizer.model`

If no such model directory is found, notebook should print `/kaggle/input/models` and `/kaggle/input` to help the user adjust `MODEL_SOURCE_HINT`.
The current notebook sorts discovered model directories by `MODEL_KEY_HINTS` so ViDeBERTa paths are preferred when several mounted model directories exist.

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
- It can run default `ViDeBERTa + MoE` training against the mounted ViANLI data described in `INFO.md`.
- It can run default `ViDeBERTa + MoE` training against a mounted local ViDeBERTa directory; if the mount name is unusual, editing `MODEL_SOURCE_HINT` in the first code cell should be enough.
- It uses Kaggle's default Python environment instead of creating a new virtual environment.
- It uses absolute paths for `scripts/train.py` and config files so manual notebook cwd differences do not break training.
- It passes the mounted model directory through the train CLI without relying on Hugging Face Hub downloads.
- It prints captured `stdout` and `stderr` when a subprocess fails.
- It does not depend on a live GitHub clone to begin execution.
