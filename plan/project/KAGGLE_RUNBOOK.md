# Kaggle Runbook

## Goal

Create a notebook the user can upload to Kaggle. When executed, it should:

1. Clone the GitHub repo into `/kaggle/working/{repo}`.
2. Create a `uv` environment.
3. Install dependencies.
4. Run a smoke check.
5. Run training.
6. Save outputs to Kaggle working directory.

## User-Provided GitHub Repo

```text
https://github.com/DangKhoaAI/vnnli_engram_moe
```

## Notebook Location

Target file:

```text
notebooks/kaggle_wrapup.ipynb
```

The primary notebook target is `notebooks/kaggle_wrapup.ipynb`. If a future user explicitly asks for a different top-level notebook filename, treat that as a separate repo-surface decision.

## Notebook Structure

Recommended cells:

1. Markdown: title and variables explanation.
2. Code: define run variables.
3. Code: install `uv` if missing.
4. Code: clone or update repo.
5. Code: create venv and install project.
6. Code: verify GPU and package versions.
7. Code: configure data paths.
8. Code: run unit tests or smoke command.
9. Code: run training.
10. Code: show output files and metrics.

## Suggested Variables

```python
REPO_URL = "https://github.com/DangKhoaAI/vnnli_engram_moe"
REPO_NAME = "vnnli_engram_moe"
REPO_DIR = f"/kaggle/working/{REPO_NAME}"

MODEL_KEY = "mbert_cased"
MODEL_CONFIG = "configs/models/mbert_cased.yaml"
BASE_CONFIG = "configs/kaggle.yaml"

DATA_DIR = "/kaggle/input/vianli"
TRAIN_FILE = f"{DATA_DIR}/train.jsonl"
DEV_FILE = f"{DATA_DIR}/dev.jsonl"
TEST_FILE = f"{DATA_DIR}/test.jsonl"

RUN_TESTS = True
RUN_FULL_TRAIN = True
```

## Suggested Shell Flow

```bash
cd /kaggle/working
git clone https://github.com/DangKhoaAI/vnnli_engram_moe
cd /kaggle/working/vnnli_engram_moe
python -m pip install uv
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest
uv run python scripts/train.py \
  --config configs/kaggle.yaml \
  --model-config configs/models/mbert_cased.yaml \
  --train-file /kaggle/input/vianli/train.jsonl \
  --validation-file /kaggle/input/vianli/dev.jsonl \
  --test-file /kaggle/input/vianli/test.jsonl \
  --output-dir /kaggle/working/outputs
```

## Kaggle Config Differences

`configs/kaggle.yaml` should override:

```yaml
project:
  output_dir: /kaggle/working/outputs

training:
  fp16: true
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
```

If Kaggle GPU does not support fp16 well, notebook should expose a variable to turn it off.

## Dataset Mounting Assumption

The notebook should not assume the exact Kaggle dataset slug. It should make `DATA_DIR` easy to edit in one place.

Support these likely names:

- `train.jsonl`, `dev.jsonl`, `test.jsonl`
- `train.csv`, `dev.csv`, `test.csv`

If files are missing, notebook should print the contents of `/kaggle/input` to help the user adjust paths.

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
- It can run clone/setup without manual shell editing.
- It can run at least smoke training without needing full ViANLI data.
