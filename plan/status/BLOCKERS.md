# Blockers And Risks

Use this file to track anything that blocks implementation or requires user input.

## Active Blockers

| ID | Status | Blocker | Needed To Resolve | Owner |
| --- | --- | --- | --- | --- |
| - | - | No active blockers at the moment | - | - |

## Active Risks

| ID | Risk | Impact | Mitigation |
| --- | --- | --- | --- |
| R-001 | Transformers 5.0.0 APIs may differ from older examples | Training code may need adjustment | Verify against installed environment; isolate Trainer usage in `training/trainer.py` |
| R-002 | PhoBERT word segmentation may require VnCoreNLP/Java setup | PhoBERT training can fail if segmentation is mandatory | Make segmentation configurable and document setup |
| R-003 | Full target models are large for local dev | Local training may be slow/impossible | Keep offline tests small; use Kaggle for full runs |
| R-004 | CafeBERT checkpoint/tokenizer behavior may need runtime verification | Model-specific failures | Keep model config isolated and add opt-in integration test |
| R-005 | GPU PyTorch installation is large and time-consuming | Dependency installation can take a long time and consume substantial disk/bandwidth | Prefer reusing the prepared `.venv` or a prebuilt image once verification is complete |
| R-006 | Kaggle offline runs now depend on a correctly mounted local Transformers checkpoint directory | Notebook training will fail early if the mounted model path changes or is incomplete | Keep `MODEL_SOURCE_HINT` editable in the top notebook cell, auto-detect standard model layouts, and pass `--local-files-only` through the train CLI |
| R-007 | `INFO.md` does not yet name a concrete Kaggle mount path for ViDeBERTa | The new ViDeBERTa + MoE notebook default may pick the wrong model if several checkpoints are mounted or none include `videberta`/`deberta`/`fsoft` in the path | Keep `MODEL_KEY_HINTS` and `MODEL_SOURCE_HINT` editable and fail early with directory previews |

## Resolved Blockers

| ID | Resolved Date | Resolution |
| --- | --- | --- |
| B-001 | 2026-06-11 | Refreshed `notebooks/kaggle_wrapup.ipynb` so it searches `/kaggle/input` for the mounted source repo and ViANLI data, then picks supported train/validation/test files automatically. |
| B-002 | 2026-06-09 | Created `.venv`, installed `torch==2.10.0+cu128` plus the project dependencies, then verified pytest and a CUDA smoke train. |

## Blocker Entry Template

```markdown
| B-XXX | Open | Short blocker description | What is needed | Owner |
```
