# Architecture Plan

## Proposed Directory Structure

```text
.
├── configs/
│   ├── default.yaml
│   ├── kaggle.yaml
│   └── models/
│       ├── mbert_cased.yaml
│       ├── xlmr_base.yaml
│       ├── cafebert.yaml
│       └── phobert_base.yaml
├── notebooks/
│   └── kaggle_wrapup.ipynb
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── src/
│   └── vnnli_engram_moe/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── constants.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── dataset.py
│       │   ├── io.py
│       │   ├── labels.py
│       │   └── preprocess.py
│       ├── metrics.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── ffn.py
│       │   ├── moe.py
│       │   ├── engram_moe.py
│       │   └── registry.py
│       ├── training/
│       │   ├── __init__.py
│       │   ├── callbacks.py
│       │   ├── trainer.py
│       │   └── seed.py
│       └── utils/
│           ├── __init__.py
│           ├── logging.py
│           └── paths.py
├── tests/
│   ├── fixtures/
│   │   └── sample_vianli.jsonl
│   ├── test_config.py
│   ├── test_labels.py
│   ├── test_dataset.py
│   ├── test_registry.py
│   └── test_smoke_train.py
├── AGENT.md
├── PROJECT.md
├── README.md
├── STATUS.md
├── pyproject.toml
└── uv.lock
```

## Module Responsibilities

### `config.py`

Owns config schema and loading.

Recommended approach:

- YAML on disk.
- Python dataclasses for typed access.
- Merge order:
  1. `configs/default.yaml`
  2. optional model config
  3. optional user config
  4. CLI overrides

Avoid scattering defaults through the training code.

### `constants.py`

Stable labels and project constants:

```python
LABELS = ["entailment", "contradiction", "neutral"]
LABEL2ID = {"entailment": 0, "contradiction": 1, "neutral": 2}
ID2LABEL = {0: "entailment", 1: "contradiction", 2: "neutral"}
```

### `data/io.py`

Loads ViANLI files.

Support priority:

1. JSONL
2. CSV
3. Parquet if dependencies are installed

Return a standard in-memory structure or Hugging Face `DatasetDict`.

### `data/preprocess.py`

Own preprocessing/tokenization.

Requirements:

- Pair premise/hypothesis through tokenizer pair API when possible.
- Respect `max_length=256`.
- Support truncation and padding behavior from config.
- Include a `TextPreprocessor` interface.
- Include `NoOpPreprocessor`.
- Include `PhoBertWordSegmenter` hook that can call VnCoreNLP when available.

PhoBERT detail:

- PhoBERT usually expects Vietnamese word segmentation before BPE tokenization.
- Make this optional/configurable so tests and non-PhoBERT models do not require VnCoreNLP.

### `models/registry.py`

Maps architecture keys to builders:

- `ffn`: implemented now.
- `moe`: placeholder/future.
- `engram_moe`: placeholder/future.

Also maps model keys to Hugging Face checkpoints.

### `models/ffn.py`

Build Design 1.

Recommended MVP:

```python
AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    classifier_dropout=config.training.dropout,
)
```

If `classifier_dropout` is not accepted by a model config, set dropout on the loaded config object before instantiating.

### `models/moe.py`

Future placeholder.

Include:

- config dataclass or builder params
- explicit `NotImplementedError`
- docstring describing intended router/expert behavior

### `models/engram_moe.py`

Future placeholder.

Include:

- config dataclass or builder params
- explicit `NotImplementedError`
- docstring describing memory/retrieval/fusion responsibilities

### `training/trainer.py`

Owns training orchestration.

Preferred:

- Use Hugging Face Trainer if compatible with installed Transformers 5.0.0.
- Keep a fallback/custom loop small and isolated if Trainer API breaks.

Responsibilities:

- set seed
- load config
- load dataset
- tokenize
- build model
- train
- evaluate dev/test
- save model/checkpoints/metrics

### `metrics.py`

Compute:

- accuracy
- macro-F1
- per-class precision/recall/F1 if practical

### `cli.py` and `scripts/*.py`

Expose stable commands:

```bash
python scripts/train.py --config configs/default.yaml --model mbert_cased
python scripts/evaluate.py --checkpoint outputs/... --split test
python scripts/predict.py --checkpoint outputs/... --premise "..." --hypothesis "..."
```

The scripts should be thin wrappers around package functions.

## Output Directory Contract

Each run should create a timestamped or user-named directory:

```text
outputs/
└── runs/
    └── 2026-06-09_1654_mbert_cased/
        ├── config.resolved.yaml
        ├── train_metrics.json
        ├── dev_metrics.json
        ├── test_metrics.json
        ├── trainer_state.json
        ├── checkpoints/
        └── final_model/
```

## Future-Proofing Boundary

Prepare extension points for MoE and Engram+MoE, but do not let placeholders complicate Design 1.

Good boundary:

- shared config schema includes future namespaces
- registry has future keys
- future modules exist with clear errors

Bad boundary:

- custom encoder wrappers everywhere before baseline works
- hand-written training loop only to support hypothetical future architectures
- MoE-specific assumptions inside dataset/tokenization code
