# Experiments

## Default Hyperparameters

- `max_length=256`
- `learning_rate=1e-5`
- `eval_frequency=400`
- `batch_size=16`
- `weight_decay=0.0`
- `adam_epsilon=1e-8`
- `dropout=0.4`
- `epochs=7`

## Baseline Matrix

- `mbert_cased` with `configs/models/mbert_cased.yaml`
- `xlmr_base` with `configs/models/xlmr_base.yaml`
- `cafebert` with `configs/models/cafebert.yaml`
- `phobert_base` with `configs/models/phobert_base.yaml`

## Creating New Experiments

Recommended pattern:

1. Keep `configs/default.yaml` as the stable baseline.
2. Add a small override file under `configs/experiments/`.
3. Run with `--user-config`.
4. Keep one run directory per experiment for reproducibility.

## Comparing Runs

Every run saves:

- resolved config
- metadata
- train metrics
- dev metrics
- test metrics

That makes it easy to compare runs by reading JSON artifacts without diffing source code.

