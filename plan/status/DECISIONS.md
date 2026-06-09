# Decision Log

Use this file for technical or process decisions that future agents should not have to rediscover.

## Decision Template

```markdown
## DEC-XXX: Title

- Date:
- Status: Proposed | Accepted | Superseded
- Context:
- Decision:
- Consequences:
```

## DEC-001: Keep `plan/HANDOFF.md` As The Entry Point

- Date: 2026-06-09
- Status: Accepted
- Context: Future sessions may start with only a request to read `HANDOFF.md`.
- Decision: `plan/HANDOFF.md` remains the human/agent entry point. It should link to both stable plan files and mutable state files.
- Consequences: Any new plan document should be linked from `HANDOFF.md` if it is required for pickup.

## DEC-002: Use Dedicated Mutable State Files Under `plan/`

- Date: 2026-06-09
- Status: Accepted
- Context: User requested that handoff and plan directory be updatable with status, state, and progress reports.
- Decision: Use `plan/STATE.md`, `plan/status/PROGRESS_LOG.md`, `plan/status/DECISIONS.md`, and `plan/status/BLOCKERS.md` instead of relying only on static planning docs.
- Consequences: Every agent must update these files before ending a work session.

## DEC-005: Split Plan Guides And Mutable Status Files

- Date: 2026-06-09
- Status: Accepted
- Context: The user requested fewer files at `plan/` root and asked to separate project guidance from status/state/progress/decision/report files.
- Decision: Keep only `plan/HANDOFF.md` and `plan/STATE.md` at root. Move implementation guidance into `plan/guides/` and mutable status support files into `plan/status/`.
- Consequences: Future links should use `plan/guides/...` for project instructions and `plan/status/...` for progress, decisions, blockers, and agent update protocol.

## DEC-003: Implement Design 1 First, Keep MoE/Engram As Extension Points

- Date: 2026-06-09
- Status: Accepted
- Context: The user wants FFN now and MoE/Engram+MoE as future implementations, while avoiding major refactors later.
- Decision: Build the source architecture around a model registry with `ffn`, `moe`, and `engram_moe` architecture keys. Implement only `ffn` initially; future keys may raise explicit `NotImplementedError`.
- Consequences: Baseline work stays small, but future architecture work has stable integration points.

## DEC-004: Use YAML Config Plus Typed Python Schema

- Date: 2026-06-09
- Status: Accepted
- Context: User requires centralized hyperparameters for tuning.
- Decision: Store experiment values in YAML and load them into typed Python config objects.
- Consequences: Hyperparameter tuning should happen via config files/overrides, not source edits.
