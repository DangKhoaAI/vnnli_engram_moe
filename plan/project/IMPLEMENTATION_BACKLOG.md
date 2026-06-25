# Implementation Backlog

Use this as the working checklist. Each phase should leave the repo runnable.

## Phase 0: Safety And Repo Orientation

Goal: understand current state and avoid clobbering user work.

Tasks:

- Run `git status --short`.
- Run `rg --files`.
- Confirm `TASK.md` and `INFO.md` remain untouched, and align top-level entry points with the current repo state.
- Create a branch if the user asks for Git workflow; otherwise continue locally.

Acceptance:

- Agent knows which files are pre-existing.
- No user file is overwritten accidentally.

## Phase 1: Project Scaffold

Goal: create the package skeleton and local dev setup.

Tasks:

- Add `pyproject.toml`.
- Add package under `src/vnnli_engram_moe/`.
- Add `configs/default.yaml` and model configs.
- Add `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py`.
- Add `tests/` with placeholder/sample fixtures.
- Add `.gitignore` if missing.

Dependency guidance:

- Required runtime:
  - `torch`
  - `transformers`
  - `datasets`
  - `evaluate` or `scikit-learn`
  - `pyyaml`
  - `numpy`
  - `pandas`
  - `tqdm`
- Dev/test:
  - `pytest`
  - `ruff` if desired
  - `ipykernel`
  - `nbformat`

Acceptance:

- `uv venv --python 3.12` can create the environment.
- `uv pip install -e ".[dev]"` can install the package.
- `python -m vnnli_engram_moe.cli --help` or equivalent command works.

## Phase 2: Config System

Goal: centralize all tunable params.

Tasks:

- Implement config dataclasses.
- Implement YAML loader.
- Implement model config merge.
- Implement CLI overrides.
- Save resolved config into run output directory.

Acceptance:

- `tests/test_config.py` validates default config load.
- Override behavior is tested.
- No training hyperparameter is hardcoded in training code except schema fallback.

## Phase 3: Data Pipeline

Goal: load ViANLI and transform it into model-ready inputs.

Tasks:

- Implement label mapping.
- Implement file readers for JSONL and CSV.
- Implement optional Parquet reader if dependency is available.
- Validate required columns.
- Implement tokenizer pair preprocessing.
- Implement PhoBERT word-segmentation hook.
- Add tiny fixture dataset.

Acceptance:

- `tests/test_labels.py` passes.
- `tests/test_dataset.py` passes with fixture data.
- Invalid label gives a clear error.
- Missing columns give a clear error.

## Phase 4: Model Registry And Design 1

Goal: build FFN baseline model reliably.

Tasks:

- Add model checkpoint registry.
- Add architecture registry.
- Implement `ffn` builder using `AutoConfig`, `AutoTokenizer`, and `AutoModelForSequenceClassification`.
- Ensure `num_labels=3`, `id2label`, `label2id`.
- Apply dropout config where model supports it.
- Add ViDeBERTa/DeBERTa-style `moe` implementation and keep `engram_moe` as an explicit future-work path.

Acceptance:

- `tests/test_registry.py` validates known model keys.
- `architecture=ffn` resolves to a buildable path.
- `architecture=moe` resolves to a trainable ViDeBERTa/DeBERTa-style MoE path.
- `architecture=engram_moe` fails explicitly with a future-work error.

## Phase 5: Training And Evaluation

Goal: train/evaluate/save a baseline run.

Tasks:

- Implement seed setting.
- Implement metrics: accuracy, macro-F1, optional per-class report.
- Implement training runner.
- Prefer HF Trainer if compatible with Transformers 5.0.0.
- Add checkpoint saving and final model saving.
- Add train/dev/test metrics JSON outputs.

Acceptance:

- CLI can run on tiny fixture data in a smoke mode.
- Metrics files are written.
- Output directory includes resolved config and final model path.

## Phase 6: Kaggle Wrapup Notebook

Goal: make a notebook the user can upload to Kaggle to clone/setup/train.

Tasks:

- Create `notebooks/kaggle_wrapup.ipynb`.
- Include variables:
  - `REPO_URL`
  - `REPO_DIR`
  - `MODEL_KEY`
  - `DATA_DIR`
  - `RUN_FULL_TRAIN`
- Shell cells:
  - install `uv` if unavailable
  - clone repo into `/kaggle/working/{repo}`
  - create `.venv`
  - install package
  - run tests/smoke train
  - run full train if enabled
- Save outputs under `/kaggle/working/outputs`.

Acceptance:

- Notebook is valid JSON.
- Running top-to-bottom on Kaggle should require only setting data path and model choice.
- Notebook avoids hardcoding local machine paths.

## Phase 7: Documentation

Goal: make the project understandable to a future reader.

Tasks:

- Add top-level `README.md`.
- Add `PROJECT.md`.
- Add `STATUS.md`.
- Keep `AGENT.md` aligned with the repository entry-point contract.

Acceptance:

- README includes quickstart, training command, data format.
- `PROJECT.md` includes overview, architecture, tech stack, directory structure, module/function list.
- `STATUS.md` clearly describes completed work, remaining work, and deeper status references.

## Phase 8: Verification

Goal: prove the project is ready to hand off.

Tasks:

- Run formatting/lint if configured.
- Run tests.
- Run a local smoke train.
- Validate notebook JSON.
- Inspect `git diff`.

Suggested commands:

```bash
uv run pytest
uv run python scripts/train.py --config configs/default.yaml --train-file tests/fixtures/sample_vianli.jsonl --validation-file tests/fixtures/sample_vianli.jsonl --training.num_train_epochs 1 --training.max_steps 1
uv run python -m json.tool notebooks/kaggle_wrapup.ipynb >/tmp/kaggle_wrapup.validated.json
git diff --stat
```

Acceptance:

- Tests pass.
- Smoke train produces output directory and metrics.
- Docs and notebook exist.
- Final response reports what was implemented and what remains future work.
