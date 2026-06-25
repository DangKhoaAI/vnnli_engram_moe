# STATUS

This document is the main status entry point for the repository. It is meant to answer three questions quickly:

1. What has already been completed?
2. What is still pending?
3. Where should the next session look for deeper operational context?

## 1. Current Summary

- Project stage: baseline implementation completed; ViDeBERTa MoE implementation added
- Active implemented designs: Design 1 FFN-style transformer baseline; ViDeBERTa MoE FFN replacement path
- Future designs: `engram_moe`
- Verification level: baseline code path, MoE replacement unit tests, full pytest suite, and notebook JSON validation are passing locally; the Kaggle notebook now defaults to ViDeBERTa + MoE with mounted source/data/model discovery

## 2. Progress Snapshot

The current baseline scope is tracked as 8 major steps.

- Completed: `8/8`
- In progress: `0/8`
- Blocked: `0/8`
- Deferred future work: `1` architecture track (`engram_moe`)

## 3. Step-by-Step Status

1. Repository scaffold and environment setup: complete
2. Centralized configuration system: complete
3. NLI data pipeline: complete
4. Model registry and Design 1 baseline: complete
5. Training, evaluation, and prediction flow: complete
6. Kaggle wrapup notebook: complete
7. Documentation and agent context preservation: complete
8. Baseline verification: complete
9. ViDeBERTa model keys/configs and MoE path: implemented and locally unit-tested

## 4. What Is Working

- YAML-driven configuration
- file-based dataset loading
- Hugging Face dataset loading for `uitnlp/ViANLI`
- sentence-pair tokenization
- PhoBERT preprocessing hook
- training CLI
- evaluation CLI
- prediction CLI
- local-checkpoint training via `--checkpoint` and `--local-files-only`
- ViDeBERTa model keys for xsmall/base/large
- ViDeBERTa base MoE config at `configs/models/videberta_base_moe.yaml`
- DeBERTa-style MoE FFN replacement for the last configurable encoder blocks
- MoE router/expert warmup freeze before full-model training
- offline-first pytest suite
- Kaggle notebook flow for mounted source/data/model inputs and the default Kaggle Python environment

## 5. What Is Not Finished Yet

- `engram_moe` is still only an extension point
- alternate Kaggle mount names may still require editing the top configuration cell in the notebook, even though the current notebook now auto-detects the known mounted source/data/model layout

These are not regressions in the baseline; they are simply outside the currently implemented scope.

## 6. Open Issues And Risks

- The refreshed Kaggle notebook is aligned to the mounted paths documented in `INFO.md`, but unusual Kaggle mount names may still require editing the top configuration cell.
- The Kaggle runtime still needs the mounted Transformer checkpoint directory to contain a standard local `transformers` layout (`config.json`, weights, tokenizer files); otherwise the notebook will stop early with a clear path-resolution error.
- The refreshed Kaggle notebook now depends more directly on Kaggle's default runtime image, so a future package-version change in that image could require revisiting the notebook.
- Manual Kaggle shell debugging must either run from `/kaggle/working/vnnli_engram_moe` or use absolute paths for both `scripts/train.py` and config files; the notebook now uses absolute paths for those values.
- Full Kaggle verification for ViDeBERTa + MoE still depends on a mounted ViDeBERTa checkpoint directory being available in the Kaggle input.

## 7. Verification Snapshot

The baseline was previously verified through:

- pytest passing in the project virtual environment
- CLI help checks for train, evaluate, and predict
- local-checkpoint CLI help check for `scripts/train.py --checkpoint ... --local-files-only`
- notebook JSON validation
- notebook code-cell compilation after the mounted source/data/model refresh
- notebook structure refresh so every code cell has a markdown explanation above it
- notebook path hardening so training uses absolute script/config paths, prints subprocess failure output, and smoke-loads the mounted local model before training
- pytest coverage for local model loading and local-files-only config propagation
- pytest coverage for ViDeBERTa/DeBERTa-style MoE replacement, top-k routing shape, expert initialization, and warmup-freeze parameter filtering
- CUDA smoke training with a tiny checkpoint
- Hugging Face dataset smoke training against the ViANLI dataset path used by the repo

This means the project is past the planning-only stage and already has a validated baseline path.

## 8. Recommended Next Actions

- Upload the refreshed notebook to Kaggle and run one end-to-end ViDeBERTa + MoE training pass against the mounted ViANLI dataset plus a mounted local ViDeBERTa checkpoint in the default Kaggle runtime with internet disabled.
- If future research continues, implement `engram_moe` next on top of the current MoE path.
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
