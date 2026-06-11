# Agent Update Protocol

This plan directory is not static. Every agent should update it as work progresses.

## Start Of Session

Before implementing anything:

1. Read `AGENT.md`.
2. Read `TASK.md` and `INFO.md`.
3. Read `STATUS.md` and `PROJECT.md`.
4. Read the relevant detailed plan file in `plan/project/` for the active phase.
5. Read relevant files in `plan/status/` if you need recent progress, blockers, or decisions.
6. Run:

```bash
git status --short
rg --files
```

7. If repo state differs materially from `STATUS.md`, update `STATUS.md` and any related status files.

## During Work

When a phase changes:

- Update `STATUS.md`.
- Mark only real progress as `Done`.
- Use `Blocked` if no meaningful progress can continue without user/external input.

When making a technical decision:

- Add an entry to `plan/status/DECISIONS.md`.
- Use the next `DEC-XXX` number.
- Include context, decision, and consequences.

When discovering a blocker or risk:

- Add or update `plan/status/BLOCKERS.md`.
- Keep active blockers separate from risks.

## End Of Session

Before final response to the user, always update:

1. `STATUS.md`
   - project summary
   - completed / pending work
   - open issues if changed
2. `plan/status/PROGRESS_LOG.md`
   - append a new dated entry
3. `plan/status/BLOCKERS.md`
   - if any blockers/risks changed
4. `plan/status/DECISIONS.md`
   - if any decisions were made

## Status Rules

Allowed status values:

- `Todo`: not started.
- `In Progress`: started and still active.
- `Blocked`: cannot continue without user input or external state.
- `Done`: completed and verified enough for that phase.
- `Deferred`: intentionally postponed.

Do not mark a phase `Done` only because some files were created. A phase is done when its acceptance criteria in `plan/project/IMPLEMENTATION_BACKLOG.md` are satisfied.

## Progress Entry Rules

Each progress entry should include:

- Summary.
- Files changed.
- Verification performed.
- Next recommended action.

Keep entries concise but concrete enough that a new session can resume without reading the entire diff.

## Report Format For Final Response

When finishing a work session, report to the user:

- What changed.
- Which state/progress files were updated.
- Verification commands run.
- Next action.

Do not claim implementation is complete unless `STATUS.md` and tests/verification support that claim.
