# Project State

This is the mutable source of truth for the current project status. Every agent should read this after `plan/HANDOFF.md` and update it before ending a work session.

## Snapshot

| Field | Value |
| --- | --- |
| Last updated | 2026-06-09 |
| Updated by | Codex implementation session |
| Overall status | Baseline implementation and runtime verification complete |
| Active phase | Phase 8: Verification |
| Current objective | Hand off a verified baseline repo with clean commit history |
| Current owner | Codex |
| Repo path | `/home/khoa/KHOA/FPTStudy/Semester7/DAT301m/PROJECT/train_model` |
| Git remote target | `https://github.com/DangKhoaAI/vnnli_engram_moe` |

## Current Repo State

Known files at the time of this update:

- Existing user files:
  - `TASK.md`
  - `INFO.md`
- Implementation files:
  - `.gitignore`
  - `pyproject.toml`
  - `README.md`
  - `configs/default.yaml`
  - `configs/kaggle.yaml`
  - `configs/models/mbert_cased.yaml`
  - `configs/models/xlmr_base.yaml`
  - `configs/models/cafebert.yaml`
  - `configs/models/phobert_base.yaml`
  - `scripts/train.py`
  - `scripts/evaluate.py`
  - `scripts/predict.py`
  - `src/vnnli_engram_moe/`
  - `tests/`
  - `docs/PROJECT.md`
  - `docs/DATA.md`
  - `docs/KAGGLE.md`
  - `docs/EXPERIMENTS.md`
  - `notebooks/kaggle_wrapup.ipynb`
- Planning files:
  - `plan/HANDOFF.md`
  - `plan/STATE.md`
  - `plan/guides/PROJECT_REQUIREMENTS.md`
  - `plan/guides/ARCHITECTURE_PLAN.md`
  - `plan/guides/IMPLEMENTATION_BACKLOG.md`
  - `plan/guides/CONFIG_AND_EXPERIMENTS.md`
  - `plan/guides/KAGGLE_RUNBOOK.md`
  - `plan/guides/TESTING_AND_ACCEPTANCE.md`
  - `plan/guides/DOCS_PLAN.md`
  - `plan/status/PROGRESS_LOG.md`
  - `plan/status/DECISIONS.md`
  - `plan/status/BLOCKERS.md`
  - `plan/status/AGENT_UPDATE_PROTOCOL.md`
- Plan subfolders:
  - `plan/guides/` for project implementation guidance.
  - `plan/status/` for mutable progress/state support files.

Latest observed `git status --short`:

```text
?? .gitignore
?? README.md
?? configs/
?? docs/
?? notebooks/
?? pyproject.toml
?? scripts/
?? src/
?? tests/
```

## Phase Status

| Phase | Status | Notes |
| --- | --- | --- |
| Phase 0: Safety and repo orientation | Done | Initial files inspected; user files left untouched |
| Phase 1: Project scaffold | Done | Added package layout, configs, scripts, tests, docs, notebook, and local `.venv` scaffold |
| Phase 2: Config system | Done | YAML + typed dataclass schema + CLI override support |
| Phase 3: Data pipeline | Done | JSONL/CSV/Parquet readers, label mapping, pair tokenization, PhoBERT segmentation hook |
| Phase 4: Model registry and Design 1 | Done | FFN baseline builder plus `moe`/`engram_moe` placeholders |
| Phase 5: Training and evaluation | Done | Custom training loop, metrics, run artifacts, and evaluate/predict flows |
| Phase 6: Kaggle wrapup notebook | Done | Added `notebooks/kaggle_wrapup.ipynb` for clone/setup/train flow |
| Phase 7: Documentation | Done | Added README and project/data/experiment/Kaggle docs |
| Phase 8: Verification | Done | `.venv` now has GPU PyTorch, pytest passed, and CLI smoke train ran successfully on CUDA |

Allowed status values:

- `Todo`
- `In Progress`
- `Blocked`
- `Done`
- `Deferred`

## Next Action

Create logical commits and hand off:

1. Commit scaffold/config files.
2. Commit baseline source and CLI implementation.
3. Commit tests, docs, and Kaggle notebook.
4. Commit mutable plan/status updates from this verified implementation session.

## Last Completed Work

- Switched the local `.venv` to GPU PyTorch `2.10.0+cu128` and verified CUDA is available on the RTX 5050 Laptop GPU.
- Ran the full pytest suite successfully inside `.venv`.
- Ran a 1-step CLI smoke train successfully on CUDA with `hf-internal-testing/tiny-random-bert`.

## Open Questions

| Question | Status | Notes |
| --- | --- | --- |
| Exact local/Kaggle dataset path | Open | Config remains user-editable via YAML or CLI |
| Whether to reuse root `Finetuning.ipynb` | Deferred | The current implementation uses `notebooks/kaggle_wrapup.ipynb` instead |
| Whether full MoE should be implemented now | Deferred | Current plan says placeholders only unless user asks |

## Verification Log

| Date | Command/Check | Result |
| --- | --- | --- |
| 2026-06-09 | `rg --files plan` | Planning files listed successfully |
| 2026-06-09 | `sed -n '1,240p' plan/HANDOFF.md` | Handoff reviewed |
| 2026-06-09 | `git status --short` | Existing user files and `plan/` are untracked |
| 2026-06-09 | `rg --files plan` | Confirmed reorganized `plan/guides/` and `plan/status/` layout |
| 2026-06-09 | `ls -la` | Current root contains `TASK.md`, `INFO.md`, and `plan/`; earlier empty `Finetuning.ipynb` and `STATE.` were no longer present |
| 2026-06-09 | `python3 scripts/train.py --help` | Passed after lazy-loading training imports |
| 2026-06-09 | `python3 scripts/evaluate.py --help` | Passed |
| 2026-06-09 | `python3 scripts/predict.py --help` | Passed |
| 2026-06-09 | `python3 -m json.tool notebooks/kaggle_wrapup.ipynb` | Notebook JSON is valid |
| 2026-06-09 | `python3 -m compileall src scripts tests` | Core package, scripts, and tests compiled successfully |
| 2026-06-09 | `python3 -m py_compile src/vnnli_engram_moe/config.py src/vnnli_engram_moe/cli.py src/vnnli_engram_moe/models/ffn.py src/vnnli_engram_moe/models/moe.py src/vnnli_engram_moe/models/engram_moe.py src/vnnli_engram_moe/models/registry.py src/vnnli_engram_moe/training/trainer.py` | Explicit syntax check passed for config, CLI, model registry, and trainer modules |
| 2026-06-09 | `python3 -m pytest` | Blocked: `pytest` not installed in base environment |
| 2026-06-09 | `python3 -m pip install uv -t /tmp/uvpkg` | Failed due network/DNS restriction in the sandboxed environment |
| 2026-06-09 | `python3 -m venv .venv` | Passed |
| 2026-06-09 | `.venv/bin/pip install uv pytest numpy pandas pyyaml tqdm torch transformers datasets` | Started with approval; full runtime verification depends on completion |
| 2026-06-09 | `.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0+cu128` | Passed; local env now uses GPU PyTorch matching the target CUDA line |
| 2026-06-09 | `.venv/bin/python -c "import torch; ..."` | Passed; `torch.cuda.is_available()` is `True` and detected `NVIDIA GeForce RTX 5050 Laptop GPU` |
| 2026-06-09 | `.venv/bin/pip install pytest numpy pandas pyyaml tqdm transformers datasets ipykernel nbformat ruff -e .` | Passed |
| 2026-06-09 | `.venv/bin/python -m pytest` | Passed; `13 passed` |
| 2026-06-09 | `.venv/bin/python scripts/train.py --config configs/default.yaml --model-config configs/models/mbert_cased.yaml --train-file tests/fixtures/sample_vianli.jsonl --validation-file tests/fixtures/sample_vianli.jsonl --test-file tests/fixtures/sample_vianli.jsonl --override model.checkpoint='\"hf-internal-testing/tiny-random-bert\"' --epochs 1 --max-steps 1 --batch-size 2 --run-name smoke_cli` | Passed on CUDA; run output created under `outputs/runs/2026-06-09_191430_smoke_cli` |
| 2026-06-09 | `.venv/bin/python -m pytest` after `--hf-dataset` support | Passed; `15 passed` |
| 2026-06-09 | `.venv/bin/python scripts/train.py --config configs/default.yaml --model-config configs/models/mbert_cased.yaml --hf-dataset uitnlp/ViANLI --override model.checkpoint='\"hf-internal-testing/tiny-random-bert\"' --epochs 1 --max-steps 1 --batch-size 2 --run-name smoke_hf_dataset_cached` | Passed on CUDA; trained directly from HF dataset and wrote `outputs/runs/2026-06-09_193736_smoke_hf_dataset_cached` |
