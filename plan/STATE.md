# Project State

This is the mutable source of truth for the current project status. Every agent should read this after `plan/HANDOFF.md` and update it before ending a work session.

## Snapshot

| Field | Value |
| --- | --- |
| Last updated | 2026-06-09 |
| Updated by | Codex planning session |
| Overall status | Planning complete, implementation not started |
| Active phase | Phase 1: Project Scaffold |
| Current objective | Implement the source code project according to the plan |
| Current owner | Next implementation agent |
| Repo path | `/home/khoa/KHOA/FPTStudy/Semester7/DAT301m/PROJECT/train_model` |
| Git remote target | `https://github.com/DangKhoaAI/vnnli_engram_moe` |

## Current Repo State

Known files at the time of this update:

- Existing user files:
  - `TASK.md`
  - `INFO.md`
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
?? INFO.md
?? TASK.md
?? plan/
```

## Phase Status

| Phase | Status | Notes |
| --- | --- | --- |
| Phase 0: Safety and repo orientation | Done | Initial files inspected; user files left untouched |
| Phase 1: Project scaffold | Todo | Create `pyproject.toml`, `src/`, `configs/`, `scripts/`, `tests/`, `docs/`, notebook |
| Phase 2: Config system | Todo | YAML + typed schema + overrides |
| Phase 3: Data pipeline | Todo | ViANLI loading, labels, tokenizer pair pipeline, PhoBERT hook |
| Phase 4: Model registry and Design 1 | Todo | FFN baseline plus MoE/Engram placeholders |
| Phase 5: Training and evaluation | Todo | Train/eval/predict CLI, metrics, output contract |
| Phase 6: Kaggle wrapup notebook | Todo | Clone/setup/train notebook |
| Phase 7: Documentation | Todo | README and `docs/PROJECT.md` etc. |
| Phase 8: Verification | Todo | Tests, smoke train, notebook validation |

Allowed status values:

- `Todo`
- `In Progress`
- `Blocked`
- `Done`
- `Deferred`

## Next Action

Begin implementation with:

1. Re-check `git status --short`.
2. Re-check `rg --files`.
3. Scaffold Phase 1 files from `plan/guides/IMPLEMENTATION_BACKLOG.md`.
4. Update this file with Phase 1 progress before stopping.

## Last Completed Work

- Created detailed planning documents under `plan/`.
- Added mutable handoff state system so future agents can update status, progress, decisions, and blockers.
- Reorganized plan files so `plan/HANDOFF.md` and `plan/STATE.md` stay at root, implementation guides live in `plan/guides/`, and mutable status files live in `plan/status/`.

## Open Questions

| Question | Status | Notes |
| --- | --- | --- |
| Exact local/Kaggle dataset path | Open | Config should make this user-editable |
| Whether to reuse root `Finetuning.ipynb` | Open | It is empty; plan currently targets `notebooks/kaggle_wrapup.ipynb` |
| Whether full MoE should be implemented now | Deferred | Current plan says placeholders only unless user asks |

## Verification Log

| Date | Command/Check | Result |
| --- | --- | --- |
| 2026-06-09 | `rg --files plan` | Planning files listed successfully |
| 2026-06-09 | `sed -n '1,240p' plan/HANDOFF.md` | Handoff reviewed |
| 2026-06-09 | `git status --short` | Existing user files and `plan/` are untracked |
| 2026-06-09 | `rg --files plan` | Confirmed reorganized `plan/guides/` and `plan/status/` layout |
| 2026-06-09 | `ls -la` | Current root contains `TASK.md`, `INFO.md`, and `plan/`; earlier empty `Finetuning.ipynb` and `STATE.` were no longer present |
