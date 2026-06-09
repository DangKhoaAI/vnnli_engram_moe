# HANDOFF: Vietnamese NLI Fine-tuning Project

This is the entry point for any future agent/session. Read this file first, then follow the linked plan files when deeper detail is needed.

## Project Intent

Build a production-shaped source code project for fine-tuning transformer base models on Vietnamese Natural Language Inference (ViANLI/adversarial NLI).

Current source files from the user:

- `TASK.md`: implementation requirements.
- `INFO.md`: task/data/model background.
- `Finetuning.ipynb`: currently empty, should not be treated as completed work.

The target repository should eventually support:

- Design 1 MVP: transformer encoder + FFN/linear classification head for 3 NLI labels.
- Design 2 future: MoE architecture.
- Design 3 future: Engram + MoE architecture.
- Kaggle wrapup notebook that clones this GitHub repo and runs setup/training.
- Centralized hyperparameters/configuration.
- Tests proving the code path runs.
- README and `docs/PROJECT.md` style documentation.

## Current Workspace State

As of this handoff creation:

- Repo path: `/home/khoa/KHOA/FPTStudy/Semester7/DAT301m/PROJECT/train_model`
- Git remote mentioned by user: `https://github.com/DangKhoaAI/vnnli_engram_moe`
- Existing files currently observed: `TASK.md`, `INFO.md`
- These files were untracked when last inspected: `TASK.md`, `INFO.md`, `plan/`
- This plan adds files only under `plan/`.

Do not overwrite user source files unless explicitly implementing the project later. Work with the current dirty tree.

## Recommended Read Order

1. `plan/HANDOFF.md` - this file.
2. `plan/STATE.md` - current status, active phase, next action.
3. `plan/status/AGENT_UPDATE_PROTOCOL.md` - how every agent must update state/progress before stopping.
4. `plan/guides/PROJECT_REQUIREMENTS.md` - normalized requirements from `TASK.md` and `INFO.md`.
5. `plan/guides/ARCHITECTURE_PLAN.md` - target package/module architecture.
6. `plan/guides/IMPLEMENTATION_BACKLOG.md` - phase-by-phase implementation checklist.
7. `plan/guides/CONFIG_AND_EXPERIMENTS.md` - config strategy, model registry, hyperparameters.
8. `plan/guides/KAGGLE_RUNBOOK.md` - Kaggle clone/setup/train flow.
9. `plan/guides/TESTING_AND_ACCEPTANCE.md` - required tests and final acceptance gates.
10. `plan/guides/DOCS_PLAN.md` - README/docs deliverables.

## Plan Directory Layout

Keep the root of `plan/` small:

- `plan/HANDOFF.md`: stable entry point for future sessions.
- `plan/STATE.md`: current high-level state and next action.
- `plan/guides/`: project implementation guidance.
- `plan/status/`: mutable progress, decisions, blockers, and update protocol.

## Mutable Handoff State

This handoff is intentionally updatable. Future agents must keep these files current:

- `plan/STATE.md`: single source of truth for current phase, status, next action, and last known repo state.
- `plan/status/PROGRESS_LOG.md`: append-only timeline of completed work.
- `plan/status/DECISIONS.md`: important technical decisions and rationale.
- `plan/status/BLOCKERS.md`: active blockers, risks, and required user input.
- `plan/status/AGENT_UPDATE_PROTOCOL.md`: update rules and templates.

Before ending any implementation session, update `plan/STATE.md` and append a new entry to `plan/status/PROGRESS_LOG.md`. If the session made an architectural choice, also update `plan/status/DECISIONS.md`. If anything is blocked, update `plan/status/BLOCKERS.md`.

## Execution Summary For Next Agent

Implement in this order:

1. Scaffold Python package and environment files:
   - `pyproject.toml`
   - `src/vnnli_engram_moe/`
   - `configs/`
   - `scripts/`
   - `tests/`
   - `notebooks/`
   - `docs/`
2. Implement Design 1 only:
   - dataset loading for CSV/JSONL/Parquet where feasible
   - label mapping: `entailment`, `contradiction`, `neutral`
   - paired sentence tokenization
   - PhoBERT optional word segmentation hook
   - model factory using Hugging Face `AutoModelForSequenceClassification`
   - train/eval/predict CLI
3. Add placeholder interfaces for Design 2 and 3:
   - model architecture registry
   - `ffn`, `moe`, `engram_moe` keys
   - MoE classes may raise `NotImplementedError` initially, but should be isolated so no refactor is needed later.
4. Create Kaggle wrapup notebook:
   - clones repo into `/kaggle/working/{repo}`
   - installs with `uv`
   - writes/uses Kaggle config
   - runs a small smoke train or full train command depending on variable.
5. Add tests:
   - no network/model-download tests by default
   - unit tests for config, labels, dataset transform, model registry
   - smoke train path using tiny mocked/model fixture if possible.
6. Add docs:
   - top-level `README.md`
   - `docs/PROJECT.md`
   - optional docs for data format, Kaggle, experiments.

## Non-Negotiable Requirements

- Python target: `3.12.13`
- Production library targets from user:
  - PyTorch `2.10.0+cu128`
  - Transformers `5.0.0`
- Training baseline hyperparameters:
  - `max_length=256`
  - `learning_rate=1e-5`
  - `eval_frequency=400`
  - `batch_size=16`
  - `weight_decay=0.0`
  - `adam_epsilon=1e-8`
  - `dropout=0.4`
  - `epochs=7`
- Models:
  - `bert-base-multilingual-cased`
  - `xlm-roberta-base`
  - `uitnlp/CafeBERT`
  - `vinai/phobert-base`
  - optional variants listed in `INFO.md`
- Output labels: 3 classes exactly.
- All tunable params should live in config files, not scattered constants.

## Key Design Decisions To Preserve

- Use a simple YAML config plus typed Python dataclasses for centralization.
- Use a model registry so future MoE/Engram changes add implementations rather than rewriting CLI/training/data.
- Prefer Hugging Face `AutoModelForSequenceClassification` for Design 1 to reduce custom training risk.
- Use a tokenizer/preprocessor abstraction because PhoBERT may need VnCoreNLP word segmentation while mBERT/XLM-R/CafeBERT do not.
- Keep tests offline by default; integration tests that download models should be opt-in.

## First Concrete Task

Start with `plan/STATE.md`, then continue with `plan/guides/IMPLEMENTATION_BACKLOG.md` Phase 0 and Phase 1.

Before editing existing files, check:

```bash
git status --short
rg --files
```

Then scaffold project files with `apply_patch` or equivalent safe edits. Avoid reverting untracked user files.

Before sending the final response, update:

```text
plan/STATE.md
plan/status/PROGRESS_LOG.md
plan/status/BLOCKERS.md, if needed
plan/status/DECISIONS.md, if needed
```
