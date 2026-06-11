# STATUS

This document is the main status entry point for the repository. It is meant to answer three questions quickly:

1. What has already been completed?
2. What is still pending?
3. Where should the next session look for deeper operational context?

## 1. Current Summary

- Project stage: baseline implementation completed
- Active implemented design: Design 1 FFN-style transformer baseline
- Future designs: `moe`, `engram_moe`
- Verification level: baseline code path, tests, and smoke training were completed in a prior implementation session

## 2. Progress Snapshot

The current baseline scope is tracked as 8 major steps.

- Completed: `8/8`
- In progress: `0/8`
- Blocked: `0/8`
- Deferred future work: `2` architecture tracks (`moe`, `engram_moe`)

## 3. Step-by-Step Status

1. Repository scaffold and environment setup: complete
2. Centralized configuration system: complete
3. NLI data pipeline: complete
4. Model registry and Design 1 baseline: complete
5. Training, evaluation, and prediction flow: complete
6. Kaggle wrapup notebook: complete
7. Documentation and agent context preservation: complete
8. Baseline verification: complete

## 4. What Is Working

- YAML-driven configuration
- file-based dataset loading
- Hugging Face dataset loading for `uitnlp/ViANLI`
- sentence-pair tokenization
- PhoBERT preprocessing hook
- training CLI
- evaluation CLI
- prediction CLI
- offline-first pytest suite
- Kaggle notebook flow

## 5. What Is Not Finished Yet

- `moe` is still only an extension point
- `engram_moe` is still only an extension point
- exact production dataset path still depends on where the user mounts or stores the data

These are not regressions in the baseline; they are simply outside the currently implemented scope.

## 6. Open Issues And Risks

- Local workspace contains extra artifacts such as `data.py`, `src.zip`, and `vianli_kaggle/`. They exist in the working tree, but they are not yet part of the documented repository contract.
- `.gitignore` is currently modified in the workspace. That may be intentional local work, so it should be reviewed before cleanup or commit decisions.
- Future MoE work will need its own implementation and verification plan, even though the current structure already reserves the integration points.

## 7. Verification Snapshot

The baseline was previously verified through:

- pytest passing in the project virtual environment
- CLI help checks for train, evaluate, and predict
- notebook JSON validation
- CUDA smoke training with a tiny checkpoint
- Hugging Face dataset smoke training against the ViANLI dataset path used by the repo

This means the project is past the planning-only stage and already has a validated baseline path.

## 8. Recommended Next Actions

- Decide whether the extra local artifacts should be documented, ignored, or removed from the working contract.
- If future research continues, implement `moe` first because the project already reserves a clean registry path for it.
- Keep `PROJECT.md` and this `STATUS.md` aligned whenever the repository surface or implementation scope changes.

## 9. Reference Points

| PURPOSE | WHERE |
| --- | --- |
| Read the full technical project overview | [PROJECT.md](PROJECT.md) |
| Read the long-term progress history across sessions | [plan/status/PROGRESS_LOG.md](plan/status/PROGRESS_LOG.md) |
| Read architectural and process decisions that should persist across sessions | [plan/status/DECISIONS.md](plan/status/DECISIONS.md) |
| Read active blockers, risks, and unresolved issues | [plan/status/BLOCKERS.md](plan/status/BLOCKERS.md) |
| Read the agent workflow for updating status and context | [plan/status/AGENT_UPDATE_PROTOCOL.md](plan/status/AGENT_UPDATE_PROTOCOL.md) |
| Read deeper implementation-planning documents | [plan/project/](plan/project/) |
