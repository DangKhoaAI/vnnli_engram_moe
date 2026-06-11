# Agent Entry Point

This is the short entry point for future agents working in this repository.

## Source Of Truth

- `TASK.md` and `INFO.md` are the highest-truth human-authored documents.
- Treat them as read-only unless the user explicitly asks to edit them.
- If any derived document disagrees with them, trust `TASK.md` and `INFO.md`.

## Human-Facing Entry Points

- `README.md`: short Vietnamese overview and quick start.
- `PROJECT.md`: human-readable project structure and implementation summary.
- `STATUS.md`: human-readable project status and open issues.

## Agent Read Order

1. `AGENT.md`
2. `TASK.md`
3. `INFO.md`
4. `STATUS.md`
5. `PROJECT.md`
6. `plan/status/AGENT_UPDATE_PROTOCOL.md`
7. relevant files under `plan/project/`
8. relevant files under `plan/status/`

## Writable Areas

- Agents may write and update:
  - `README.md`
  - `AGENT.md`
  - `PROJECT.md`
  - `STATUS.md`
  - `plan/`
- Agents should not edit `TASK.md` or `INFO.md` unless the user explicitly asks.

## Current Scope

- Implemented baseline: Design 1 FFN-style fine-tuning path.
- Reserved for future research: `moe`, `engram_moe`.
- Kaggle setup/train flow is part of the repository contract, and the notebook now prefers mounted Kaggle inputs over live Git clone.

## Session Discipline

- Keep `STATUS.md` aligned with the real project state.
- Keep `plan/status/PROGRESS_LOG.md` updated before ending a work session.
- Update `STATUS.md` when the public project status meaningfully changes.
- If you make structural or architectural decisions, update `plan/status/DECISIONS.md`.
