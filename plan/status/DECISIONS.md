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
- Status: Superseded
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
- Status: Superseded
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

## DEC-006: Use A Lightweight Custom Training Loop For The Baseline

- Date: 2026-06-09
- Status: Accepted
- Context: The target runtime mentions Transformers 5.0.0, which may differ from the 4.x Trainer examples most snippets rely on.
- Decision: Implement Design 1 training/evaluation with a small PyTorch loop and keep Hugging Face use focused on tokenizer/model loading.
- Consequences: The baseline path is less sensitive to Trainer API drift, and offline smoke tests can monkeypatch the model/tokenizer more easily.

## DEC-007: Add Top-Level Human And Agent Entry Points

- Date: 2026-06-11
- Status: Accepted
- Context: The task addendum says `README.md` is the human entry point and `AGENT.md` is the agent entry point, while `TASK.md` and `INFO.md` remain the highest-truth human-authored documents.
- Decision: Introduce top-level `README.md`, `AGENT.md`, `PROJECT.md`, and `STATUS.md` as the stable surface of the repository. `AGENT.md` becomes the first file an agent should read.
- Consequences: `plan/HANDOFF.md` remains useful, but now as an internal detailed handoff inside the `plan/` layer rather than the primary entry point.

## DEC-008: Rename `plan/guides/` To `plan/project/`

- Date: 2026-06-11
- Status: Accepted
- Context: The task addendum explicitly says deeper project guidance should live under `plan/project/`.
- Decision: Rename `plan/guides/` to `plan/project/` and update references.
- Consequences: Future plan-level technical guidance should be created under `plan/project/`, while mutable work history continues to live under `plan/status/`.

## DEC-009: Remove `plan/HANDOFF.md` And `plan/STATE.md`

- Date: 2026-06-11
- Status: Accepted
- Context: The user explicitly requested that `PROJECT.md` become the project entry point and `STATUS.md` become the status entry point, with no need for `HANDOFF.md` or `STATE.md`.
- Decision: Remove `plan/HANDOFF.md` and `plan/STATE.md` from the repository contract and move active status ownership to `STATUS.md` plus the files under `plan/status/`.
- Consequences: Future agents should start from `AGENT.md`, `PROJECT.md`, and `STATUS.md`, then use `plan/project/` and `plan/status/` for deeper context.

## DEC-010: Remove `docs/` From The Repository Contract

- Date: 2026-06-11
- Status: Accepted
- Context: The user explicitly requested that `docs/` be cleaned up and removed from active references because it is no longer needed.
- Decision: Remove the `docs/` directory from the repository surface and stop using it in active documentation references. Consolidate public repository documentation into `README.md`, `PROJECT.md`, and `STATUS.md`.
- Consequences: Future documentation should either live in the top-level repo entry-point files or in `plan/project/` when it is implementation-planning context rather than user-facing repo documentation.

## DEC-011: Kaggle Notebook Should Prefer Mounted Inputs And Offline-Friendly Setup

- Date: 2026-06-11
- Status: Superseded
- Context: The updated human task and `INFO.md` now provide concrete Kaggle input mount paths for both the source repo and the ViANLI dataset, so the old GitHub-clone-first notebook no longer matches the real execution environment.
- Decision: Refresh `notebooks/kaggle_wrapup.ipynb` so it discovers the mounted repo/data under `/kaggle/input`, copies the repo into `/kaggle/working`, creates a local environment from the current Kaggle interpreter, and installs the project in editable mode without re-downloading heavy dependencies when possible.
- Consequences: The notebook is more robust for offline or restricted Kaggle sessions, but future changes to Kaggle mount names may still require editing the top configuration cell.

## DEC-012: Kaggle Notebook Should Use The Default Kaggle Environment

- Date: 2026-06-11
- Status: Accepted
- Context: The latest user instruction says Kaggle already has the required libraries, so the notebook should not create a new virtual environment or reinstall packages. The user also wants every code cell to have a markdown explanation directly above it.
- Decision: Keep the mounted-input notebook flow, but simplify execution so `notebooks/kaggle_wrapup.ipynb` runs entirely with Kaggle's default Python environment and inserts explanatory markdown cells before every code cell.
- Consequences: The notebook becomes shorter and closer to real Kaggle usage, but it now depends more directly on whatever package versions Kaggle ships in its default runtime image.

## DEC-013: Treat Mounted Local Model Directories As First-Class Training Checkpoints

- Date: 2026-06-11
- Status: Accepted
- Context: The latest human task says Kaggle internet access cannot be used to download models from the Hugging Face Hub, and `INFO.md` now provides a mounted local mBERT directory under `/kaggle/input/models/...`.
- Decision: Extend the train CLI so `--checkpoint` can explicitly point at a mounted model directory and add `--local-files-only` so tokenizer/config/model loading can be forced to stay local. Refresh the Kaggle notebook to auto-detect a local Transformers directory and pass that path through the train command.
- Consequences: Offline Kaggle runs become a first-class supported path instead of relying on raw `--override` strings, but the mounted model directory now has to preserve a standard local Transformers file layout.
