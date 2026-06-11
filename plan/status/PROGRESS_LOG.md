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

## 2026-06-11 - Codex Documentation Restructure Session

### Summary

- Re-read `TASK.md` and aligned the repository surface with the task addendum.
- Added top-level human and agent entry points.
- Renamed `plan/guides/` to `plan/project/` and updated the plan/state protocol accordingly.

### Files Changed

- `README.md`: rewritten as a short Vietnamese human entry point.
- `AGENT.md`: added as the short English agent entry point.
- `PROJECT.md`: added as the human-readable project overview and structure document.
- `STATUS.md`: added as the human-readable project status document.
- `plan/HANDOFF.md`: reframed as an internal plan-layer handoff.
- `plan/STATE.md`: rewritten to reflect the new entry-point structure and current workspace.
- `plan/status/AGENT_UPDATE_PROTOCOL.md`: updated agent read order and `plan/project/` path.
- `plan/status/DECISIONS.md`: recorded the new structure decisions.
- `plan/project/*`: updated root-doc references where needed.

### Verification

- `sed -n '1,260p' TASK.md`: confirmed the addendum requirements.
- `mv plan/guides plan/project`: moved deeper project guidance to the requested folder name.
- `ls -la plan plan/project plan/status`: confirmed the new layout.

### Next Recommended Action

- Sweep any remaining stale references so all plan docs consistently use `plan/project/` and the new top-level entry points.

## 2026-06-11 - Codex Entry Point Simplification

### Summary

- Replaced the old `HANDOFF` and `STATE` entry-point model with `PROJECT.md` and `STATUS.md`.
- Rewrote both documents in English and made them the primary public project and status surfaces.
- Removed `plan/HANDOFF.md` and `plan/STATE.md` from the active repository contract.

### Files Changed

- `PROJECT.md`: rewritten as the detailed English project entry point.
- `STATUS.md`: rewritten as the detailed English status entry point.
- `AGENT.md`: updated agent read order and session discipline.
- `plan/status/AGENT_UPDATE_PROTOCOL.md`: switched status ownership from `plan/STATE.md` to `STATUS.md`.
- `plan/status/DECISIONS.md`: recorded the entry-point change.
- `plan/HANDOFF.md`: removed.
- `plan/STATE.md`: removed.

### Verification

- `rg -n "HANDOFF.md|STATE.md|plan/HANDOFF|plan/STATE|Project State|Handoff" .`: remaining mentions are historical inside decision/progress logs rather than active entry-point docs.
- `ls -la README.md AGENT.md PROJECT.md STATUS.md plan plan/project plan/status`: confirmed the new top-level and plan layout.

### Next Recommended Action

- Keep `PROJECT.md` and `STATUS.md` updated whenever the implementation scope or verified status changes.

## 2026-06-11 - Codex Link Privacy Cleanup

### Summary

- Removed machine-specific absolute paths from active documentation links.
- Changed the reference sections in `PROJECT.md` and `STATUS.md` from simple link lists to `PURPOSE | WHERE` tables.

### Files Changed

- `README.md`: changed repo-surface links to relative paths.
- `PROJECT.md`: changed the reference section to a two-column table with relative links.
- `STATUS.md`: changed the reference section to a two-column table with relative links.
- `docs/KAGGLE.md`: changed the notebook link to a relative path.

### Verification

- `rg -n "<local-absolute-link-patterns>" .`: used to find privacy-sensitive absolute-link patterns before cleanup.

### Next Recommended Action

- Keep future documentation links relative to the repository root unless a file truly requires another link style.

## 2026-06-11 - Codex Docs Surface Cleanup

## 2026-06-11 - Codex Kaggle Notebook Default Runtime Refresh

### Summary

- Reworked the Kaggle notebook again to match the latest user instruction.
- Added a markdown explanation directly above every code cell.
- Removed the `.venv` / dependency-install flow so the notebook now uses Kaggle's default Python environment end to end.

### Files Changed

- `notebooks/kaggle_wrapup.ipynb`: added markdown-before-code structure and removed the venv/install cell in favor of Kaggle default runtime checks.
- `plan/project/KAGGLE_RUNBOOK.md`: updated the notebook contract to require markdown before each code cell and default-environment execution.
- `STATUS.md`: updated the public status notes and risks for the new Kaggle runtime behavior.
- `plan/status/DECISIONS.md`: recorded the new notebook/runtime decision and superseded the earlier venv-oriented notebook decision.

### Verification

- `python3 -m json.tool notebooks/kaggle_wrapup.ipynb`: passed.
- `python3 - <<'PY' ... compile(...) ... PY`: passed; all notebook code cells compile successfully after the rewrite.

### Next Recommended Action

- Upload the revised notebook to Kaggle and confirm the default runtime image already includes the required package versions for one full training run.

## 2026-06-11 - Codex Kaggle Path Debug Follow-Up

### Summary

- Investigated the Kaggle failure pattern reported by the user.
- Identified that the manual debug commands were running from `/kaggle/working`, while `scripts/train.py` and `configs/default.yaml` live under `/kaggle/working/vnnli_engram_moe`.
- Hardened the notebook so it calls `scripts/train.py` and all config files by absolute path and prints captured subprocess output on failure.

### Files Changed

- `notebooks/kaggle_wrapup.ipynb`: changed train/help commands to absolute repo paths and improved `run()` failure logging.
- `plan/project/KAGGLE_RUNBOOK.md`: documented the absolute-path command pattern and subprocess-output acceptance requirement.
- `STATUS.md`: recorded the Kaggle cwd/path debugging note.
- `plan/status/PROGRESS_LOG.md`: appended this follow-up entry.

### Verification

- `python3 -m json.tool notebooks/kaggle_wrapup.ipynb`: passed.
- `python3 - <<'PY' ... compile(...) ... PY`: passed; all notebook code cells compile successfully.

### Next Recommended Action

- Re-upload the notebook to Kaggle and rerun the training cell. If it still fails, the printed traceback should now show the real training/runtime issue instead of only `CalledProcessError`.

## 2026-06-11 - Codex Kaggle Notebook Refresh

### Summary

- Rewrote `notebooks/kaggle_wrapup.ipynb` around the Kaggle mount information now documented in `INFO.md`.
- Switched the notebook from a GitHub-clone-first flow to a mounted-source copy flow with automatic ViANLI split discovery.
- Updated the public/project status docs so future sessions can see why the Kaggle flow changed.

### Files Changed

- `notebooks/kaggle_wrapup.ipynb`: replaced the old clone-based Kaggle flow with mounted source/data discovery, offline-friendly environment setup, default mBERT training, and artifact summary cells.
- `AGENT.md`: noted that Kaggle flow now prefers mounted inputs over live clone.
- `README.md`: aligned the short human-facing summary with the current Kaggle notebook behavior.
- `PROJECT.md`: aligned the project overview with the mounted-input Kaggle execution path.
- `STATUS.md`: refreshed the public status/risk text for the new notebook behavior.
- `plan/project/KAGGLE_RUNBOOK.md`: updated the deeper Kaggle contract and acceptance criteria.
- `plan/status/BLOCKERS.md`: resolved the old dataset-path blocker and removed stale active blockers.
- `plan/status/DECISIONS.md`: recorded the mounted-input/offline-friendly notebook decision.

### Verification

- `python3 -m json.tool notebooks/kaggle_wrapup.ipynb`: passed.
- `python3 - <<'PY' ... compile(...) ... PY`: passed; all notebook code cells compile successfully.

### Next Recommended Action

- Upload the refreshed notebook to Kaggle and run one end-to-end training pass against the mounted ViANLI dataset to verify the real production image behavior.

### Summary

- Removed `docs/` from the active repository contract.
- Expanded `README.md` into a longer Vietnamese overview by pulling in material from `PROJECT.md` and `STATUS.md`.
- Updated planning documents so they no longer require `docs/` as a deliverable.

### Files Changed

- `README.md`: expanded into a longer Vietnamese project overview and entry point.
- `PROJECT.md`: removed `docs/` from the directory structure and added the notebook as a reference point.
- `plan/project/ARCHITECTURE_PLAN.md`: removed `docs/` from the proposed repo structure.
- `plan/project/DOCS_PLAN.md`: shifted documentation expectations toward top-level repo docs instead of `docs/`.
- `plan/project/IMPLEMENTATION_BACKLOG.md`: removed `docs/` deliverables from the documentation phase.
- `plan/project/TESTING_AND_ACCEPTANCE.md`: updated acceptance criteria to match the new doc surface.
- `plan/project/PROJECT_REQUIREMENTS.md`: replaced generic additional docs expectation with `STATUS.md` and `plan/project/`.
- `plan/status/DECISIONS.md`: recorded the `docs/` removal decision.
- `docs/DATA.md`, `docs/EXPERIMENTS.md`, `docs/KAGGLE.md`, `docs/PROJECT.md`: removed.

### Verification

- `ls -la docs`: used to inspect the old docs surface before cleanup.
- `rg -n "docs/|docs\\b|DATA.md|KAGGLE.md|EXPERIMENTS.md" plan/project AGENT.md PROJECT.md STATUS.md README.md`: used to find remaining active references before patching.

### Next Recommended Action

- Keep the public repo documentation concentrated in `README.md`, `PROJECT.md`, and `STATUS.md` unless a new explicit need appears.

## 2026-06-11 - Codex Kaggle Local-Model Notebook Update

### Summary

- Added a first-class train CLI path for mounted local model directories via `--checkpoint`.
- Added `--local-files-only` plus config/model propagation so `transformers` can be forced to stay offline when loading tokenizer/config/model assets.
- Refreshed the Kaggle notebook so it auto-detects mounted repo/data/model inputs, smoke-loads the mounted model before training, and runs the default `mBERT + FFN` flow without relying on Hugging Face Hub downloads.

### Files Changed

- `src/vnnli_engram_moe/cli.py`: added train-time `--checkpoint` and `--local-files-only` handling.
- `src/vnnli_engram_moe/config.py`: added `model.local_files_only`.
- `src/vnnli_engram_moe/training/trainer.py`: passed `local_files_only` into tokenizer loading.
- `src/vnnli_engram_moe/models/ffn.py`: passed `local_files_only` into config/model loading.
- `tests/test_config.py`: covered the new config default and override.
- `tests/test_local_model_loading.py`: added regression tests for CLI override precedence and offline local-loading propagation.
- `notebooks/kaggle_wrapup.ipynb`: rewrote the notebook flow around mounted source/data/model discovery and offline local-model training.
- `README.md`, `AGENT.md`, `PROJECT.md`, `STATUS.md`: aligned the public repo surface with the mounted local-model Kaggle flow.
- `plan/project/KAGGLE_RUNBOOK.md`: updated the deeper Kaggle contract and acceptance criteria.
- `plan/status/DECISIONS.md`: recorded the mounted-local-model decision.
- `plan/status/BLOCKERS.md`: added the mounted-model-layout risk.

### Verification

- `./.venv/bin/python -m pytest tests/test_config.py tests/test_local_model_loading.py tests/test_smoke_train.py`: passed.
- `./.venv/bin/python -m pytest`: passed; `19 passed`.
- `./.venv/bin/python scripts/train.py --help`: passed and showed `--checkpoint` plus `--local-files-only`.
- `./.venv/bin/python -m json.tool notebooks/kaggle_wrapup.ipynb`: passed.
- `./.venv/bin/python - <<'PY' ... compile(...) ... PY`: passed; all notebook code cells compile successfully.

### Next Recommended Action

- Upload the refreshed notebook to Kaggle and run one full offline training pass using the mounted ViANLI dataset plus the mounted local mBERT directory from `INFO.md`.
