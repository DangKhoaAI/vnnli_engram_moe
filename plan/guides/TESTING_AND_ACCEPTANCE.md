# Testing And Acceptance Plan

## Testing Strategy

Tests should be offline by default. Do not require downloading `bert-base-multilingual-cased` or other large models during normal `pytest`.

Use integration tests for real Hugging Face downloads only when explicitly enabled by environment variable, for example:

```bash
RUN_HF_INTEGRATION=1 uv run pytest tests/integration
```

## Unit Tests

### `tests/test_config.py`

Cover:

- default YAML loads
- model YAML merges with default
- CLI/simple overrides work
- resolved config contains required hyperparameters

### `tests/test_labels.py`

Cover:

- `entailment -> 0`
- `contradiction -> 1`
- `neutral -> 2`
- unknown labels raise clear errors
- id-to-label round trip

### `tests/test_dataset.py`

Cover:

- JSONL fixture loading
- CSV fixture loading if implemented
- required column validation
- premise/hypothesis pair is preserved
- tokenization wrapper can process a tiny batch using a fake tokenizer

### `tests/test_registry.py`

Cover:

- known model keys resolve to checkpoints
- `ffn` architecture exists
- `moe` and `engram_moe` placeholders exist
- unknown model/architecture raises clear error

### `tests/test_smoke_train.py`

Cover:

- train runner can execute a tiny no-download path
- metrics JSON is written
- output directory is created

Implementation options for no-download smoke:

- use a tiny local `torch.nn.Module` implementing the expected forward output shape
- monkeypatch model builder/tokenizer
- or create a tiny random model config without downloading pretrained weights

## Integration Tests

Optional tests when network/GPU is available:

- download a small public test model, not the full target models
- train for 1 step on fixture data
- evaluate and save metrics

Do not make this part of default CI unless runtime is acceptable.

## Manual Verification Checklist

Before final handoff, run:

```bash
uv run pytest
uv run python scripts/train.py --help
uv run python scripts/evaluate.py --help
uv run python scripts/predict.py --help
python -m json.tool notebooks/kaggle_wrapup.ipynb >/tmp/kaggle_wrapup.validated.json
git diff --stat
```

If dependencies are not installed, first run:

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Acceptance Criteria

The project is complete enough for the original request when:

- Design 1 FFN fine-tuning source code exists and runs.
- Config files centralize all hyperparameters.
- At least one smoke training path passes locally.
- Kaggle notebook exists and is valid.
- README and docs exist.
- Future MoE/Engram+MoE extension points are present and documented.
- Final model outputs include metrics and resolved config.

## Known Risks

- Transformers 5.0.0 may differ from commonly documented 4.x APIs.
- PhoBERT word segmentation can add Java/VnCoreNLP setup complexity.
- CafeBERT checkpoint availability and tokenizer behavior should be verified in the target runtime.
- Full training may exceed local hardware capacity; Kaggle flow should be the preferred full-run target.

## Definition Of Done For Agent

An implementing agent should finish with:

- summary of files changed
- test commands run and results
- any commands not run and why
- remaining future work limited to Design 2/3 research unless explicitly implemented

