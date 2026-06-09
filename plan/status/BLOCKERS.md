# Blockers And Risks

Use this file to track anything that blocks implementation or requires user input.

## Active Blockers

| ID | Status | Blocker | Needed To Resolve | Owner |
| --- | --- | --- | --- | --- |
| B-001 | Open | Exact ViANLI dataset path is unknown | Keep data paths configurable; ask user only when running full training locally | Future implementation agent |

## Active Risks

| ID | Risk | Impact | Mitigation |
| --- | --- | --- | --- |
| R-001 | Transformers 5.0.0 APIs may differ from older examples | Training code may need adjustment | Verify against installed environment; isolate Trainer usage in `training/trainer.py` |
| R-002 | PhoBERT word segmentation may require VnCoreNLP/Java setup | PhoBERT training can fail if segmentation is mandatory | Make segmentation configurable and document setup |
| R-003 | Full target models are large for local dev | Local training may be slow/impossible | Keep offline tests small; use Kaggle for full runs |
| R-004 | CafeBERT checkpoint/tokenizer behavior may need runtime verification | Model-specific failures | Keep model config isolated and add opt-in integration test |

## Resolved Blockers

| ID | Resolved Date | Resolution |
| --- | --- | --- |
| None | - | - |

## Blocker Entry Template

```markdown
| B-XXX | Open | Short blocker description | What is needed | Owner |
```

