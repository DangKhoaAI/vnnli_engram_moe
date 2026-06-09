# Progress Log

Append a new entry at the top or bottom of this file at the end of every work session. Do not delete prior entries unless the user explicitly asks for cleanup.

## Entry Template

```markdown
## YYYY-MM-DD - Agent/Session Name

### Summary

- What changed:
- Why:

### Files Changed

- `path/to/file`: short reason

### Verification

- Command/check: result

### Next Recommended Action

- Next step for the next agent
```

## 2026-06-09 - Codex Planning Session

### Summary

- Created the first detailed project implementation plan from `TASK.md` and `INFO.md`.
- Added mutable handoff files so future agents can update project status, progress, decisions, and blockers.

### Files Changed

- `plan/HANDOFF.md`: entry point and update protocol references.
- `plan/guides/PROJECT_REQUIREMENTS.md`: normalized requirements.
- `plan/guides/ARCHITECTURE_PLAN.md`: target architecture and directory structure.
- `plan/guides/IMPLEMENTATION_BACKLOG.md`: phase-by-phase implementation checklist.
- `plan/guides/CONFIG_AND_EXPERIMENTS.md`: config and experiment strategy.
- `plan/guides/KAGGLE_RUNBOOK.md`: Kaggle notebook/run flow.
- `plan/guides/TESTING_AND_ACCEPTANCE.md`: test plan and acceptance gates.
- `plan/guides/DOCS_PLAN.md`: documentation deliverables.
- `plan/STATE.md`: mutable current status.
- `plan/status/PROGRESS_LOG.md`: append-only progress history.
- `plan/status/DECISIONS.md`: technical decision log.
- `plan/status/BLOCKERS.md`: blockers and risks.
- `plan/status/AGENT_UPDATE_PROTOCOL.md`: rules for updating handoff state.

### Verification

- `rg --files plan`: confirmed planning files are present.
- `sed -n '1,240p' plan/HANDOFF.md`: reviewed handoff content.
- `git status --short`: confirmed files remain untracked and user files are untouched.

### Next Recommended Action

- Start Phase 1 in `plan/guides/IMPLEMENTATION_BACKLOG.md`: scaffold the Python package, configs, scripts, tests, docs, and Kaggle notebook location.

## 2026-06-09 - Codex Plan Reorganization

### Summary

- Reorganized `plan/` so the root stays small.
- Kept `plan/HANDOFF.md` and `plan/STATE.md` at root.
- Moved project guidance files into `plan/guides/`.
- Moved mutable state support files into `plan/status/`.

### Files Changed

- `plan/HANDOFF.md`: updated read order and mutable state paths.
- `plan/STATE.md`: updated known file layout, next action, and verification log.
- `plan/status/PROGRESS_LOG.md`: updated paths and appended this reorganization entry.
- `plan/status/DECISIONS.md`: updated state-file decision paths.
- `plan/status/AGENT_UPDATE_PROTOCOL.md`: updated status/protocol paths.

### Verification

- `rg --files plan`: confirmed reorganized directory layout.
- `rg "plan/(PROJECT|ARCHITECTURE|IMPLEMENTATION|CONFIG|KAGGLE|TESTING|DOCS|PROGRESS|DECISIONS|BLOCKERS|AGENT)" plan`: found old references before patching; paths were updated.
- `ls -la`: confirmed current root contains `TASK.md`, `INFO.md`, and `plan/`.

### Next Recommended Action

- Start Phase 1 from `plan/guides/IMPLEMENTATION_BACKLOG.md`.

## 2026-06-09 - Codex Baseline Implementation

### Summary

- Implemented the full Design 1 baseline code path: config loader, data pipeline, model registry, training/evaluation/prediction CLI, docs, tests, and Kaggle notebook.
- Kept `moe` and `engram_moe` as explicit future extension points without forcing a later refactor.
- Completed syntax/help/notebook validation and started dependency installation for full runtime verification.

### Files Changed

- `.gitignore`: ignored virtualenv, caches, and output artifacts.
- `pyproject.toml`: added package metadata and dependencies.
- `configs/*`: centralized defaults, Kaggle overrides, and model-specific configs.
- `src/vnnli_engram_moe/*`: added config, data, metrics, model registry, training loop, and CLI code.
- `scripts/*`: added thin train/evaluate/predict entrypoints.
- `tests/*`: added fixtures, unit tests, and an offline smoke-train path using dummy components.
- `README.md`: added project quickstart and usage docs.
- `docs/*`: added project, data, Kaggle, and experiments documentation.
- `notebooks/kaggle_wrapup.ipynb`: added uploadable Kaggle clone/setup/train notebook.
- `plan/STATE.md`: updated implementation and verification status.
- `plan/status/DECISIONS.md`: recorded the custom-training-loop decision.
- `plan/status/BLOCKERS.md`: added the local dependency/runtime blocker.

### Verification

- `python3 scripts/train.py --help`: passed.
- `python3 scripts/evaluate.py --help`: passed.
- `python3 scripts/predict.py --help`: passed.
- `python3 -m json.tool notebooks/kaggle_wrapup.ipynb`: passed.
- `python3 -m compileall src scripts tests`: passed.
- `python3 -m py_compile src/vnnli_engram_moe/config.py src/vnnli_engram_moe/cli.py src/vnnli_engram_moe/models/ffn.py src/vnnli_engram_moe/models/moe.py src/vnnli_engram_moe/models/engram_moe.py src/vnnli_engram_moe/models/registry.py src/vnnli_engram_moe/training/trainer.py`: passed.
- `python3 -m pytest`: blocked because `pytest` is not installed in the base environment.
- `python3 -m pip install uv -t /tmp/uvpkg`: failed in the sandbox because DNS/network access was unavailable.
- `python3 -m venv .venv`: passed.
- `.venv/bin/pip install uv pytest numpy pandas pyyaml tqdm torch transformers datasets`: started after approval; full runtime verification still depends on completion.

### Next Recommended Action

- Finish `.venv` dependency installation, then run pytest and a 1-step smoke train against `tests/fixtures/sample_vianli.jsonl`.

## 2026-06-09 - Codex GPU Verification

### Summary

- Replaced the temporary CPU-only torch install with `torch==2.10.0+cu128` inside `.venv`.
- Installed the remaining project dependencies, ran the full pytest suite, and completed a real 1-step CUDA smoke train from the CLI.
- Prepared the repo for clean logical commits.

### Files Changed

- `plan/STATE.md`: marked verification complete and recorded GPU/runtime checks.
- `plan/status/BLOCKERS.md`: resolved the local dependency/runtime blocker and updated the risk note.
- `plan/status/PROGRESS_LOG.md`: appended this verification entry.

### Verification

- `.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128`: passed.
- `.venv/bin/python -c "import torch; ..."`: passed; CUDA available on RTX 5050 Laptop GPU.
- `.venv/bin/pip install pytest numpy pandas pyyaml tqdm transformers datasets ipykernel nbformat ruff -e .`: passed.
- `.venv/bin/python -m pytest`: passed; `13 passed`.
- `.venv/bin/python scripts/train.py --config configs/default.yaml --model-config configs/models/mbert_cased.yaml --train-file tests/fixtures/sample_vianli.jsonl --validation-file tests/fixtures/sample_vianli.jsonl --test-file tests/fixtures/sample_vianli.jsonl --override model.checkpoint='\"hf-internal-testing/tiny-random-bert\"' --epochs 1 --max-steps 1 --batch-size 2 --run-name smoke_cli`: passed on CUDA.

### Next Recommended Action

- Create and review the separated commits for scaffold, core baseline code, docs/tests/notebook, and mutable handoff state.

## 2026-06-09 - Codex Hugging Face Dataset Support

### Summary

- Added direct Hugging Face dataset loading so `scripts/train.py` can accept `--hf-dataset uitnlp/ViANLI`.
- Kept the existing file-based flow intact, while preferring HF splits when `data.hf_dataset` is set.
- Verified the new path with unit tests and a CUDA smoke train against `uitnlp/ViANLI`.

### Files Changed

- `src/vnnli_engram_moe/config.py`: added optional Hugging Face dataset fields to the data config.
- `src/vnnli_engram_moe/data/io.py`: added HF split loading with cached dataset reuse.
- `src/vnnli_engram_moe/data/dataset.py`: taught split loading to use HF data directly.
- `src/vnnli_engram_moe/cli.py`: added `--hf-dataset` and `--hf-dataset-config`.
- `tests/test_dataset.py`: added tests for HF split loading and split resolution.
- `README.md`: documented direct HF dataset training.
- `docs/KAGGLE.md`: documented HF dataset usage in Kaggle.

### Verification

- `.venv/bin/python -m pytest`: passed; `15 passed`.
- `.venv/bin/python scripts/train.py --config configs/default.yaml --model-config configs/models/mbert_cased.yaml --hf-dataset uitnlp/ViANLI --override model.checkpoint='\"hf-internal-testing/tiny-random-bert\"' --epochs 1 --max-steps 1 --batch-size 2 --run-name smoke_hf_dataset_cached`: passed on CUDA.

### Next Recommended Action

- If desired, mirror the same `--hf-dataset` option into the Kaggle notebook cells so the notebook no longer needs any local dataset export step.
