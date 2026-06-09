# Agent Update Protocol

This plan directory is not static. Every agent should update it as work progresses.

## Start Of Session

Before implementing anything:

1. Read `plan/HANDOFF.md`.
2. Read `plan/STATE.md`.
3. Read the relevant detailed plan file in `plan/guides/` for the active phase.
4. Run:

```bash
git status --short
rg --files
```

5. If repo state differs materially from `plan/STATE.md`, update `plan/STATE.md`.

## During Work

When a phase changes:

- Update `plan/STATE.md` phase table.
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

1. `plan/STATE.md`
   - `Last updated`
   - `Updated by`
   - `Overall status`
   - `Active phase`
   - `Current objective`
   - `Phase Status`
   - `Next Action`
   - `Last Completed Work`
   - `Verification Log`
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

Do not mark a phase `Done` only because some files were created. A phase is done when its acceptance criteria in `plan/guides/IMPLEMENTATION_BACKLOG.md` are satisfied.

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

Do not claim implementation is complete unless `plan/STATE.md` and tests/verification support that claim.
