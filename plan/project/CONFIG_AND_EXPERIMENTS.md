# Config And Experiment Plan

## Configuration Principle

All tunable parameters must be centralized in config files. Source code may define schema defaults, but experiment values should be visible in YAML.

## Proposed `configs/default.yaml`

```yaml
project:
  name: vnnli-engram-moe
  seed: 42
  output_dir: outputs/runs

data:
  train_file: data/ViANLI/train.jsonl
  validation_file: data/ViANLI/dev.jsonl
  test_file: data/ViANLI/test.jsonl
  text_columns:
    premise: premise
    hypothesis: hypothesis
  label_column: label
  uid_column: uid

model:
  model_key: mbert_cased
  checkpoint: bert-base-multilingual-cased
  architecture: ffn
  num_labels: 3
  use_fast_tokenizer: true
  requires_word_segmentation: false

training:
  max_length: 256
  learning_rate: 1.0e-5
  eval_frequency: 400
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  weight_decay: 0.0
  adam_epsilon: 1.0e-8
  dropout: 0.4
  num_train_epochs: 7
  warmup_ratio: 0.0
  gradient_accumulation_steps: 1
  fp16: false
  bf16: false
  save_strategy: steps
  save_steps: 400
  evaluation_strategy: steps
  logging_steps: 50
  load_best_model_at_end: true
  metric_for_best_model: macro_f1

moe:
  enabled: false
  num_experts: 4
  top_k: 2
  replace_last_n_layers: 2
  expert_hidden_size: null
  router_temperature: 1.0
  load_balance_loss_weight: 0.01
  router_expert_warmup_steps: 0
  router_expert_learning_rate: null
  backbone_learning_rate: null

engram:
  enabled: false
  memory_size: 0
  retrieval_top_k: 0
  update_policy: none
  fusion: none
```

## Model Config Files

Each model config overrides only model-specific fields.

Example `configs/models/phobert_base.yaml`:

```yaml
model:
  model_key: phobert_base
  checkpoint: vinai/phobert-base
  architecture: ffn
  use_fast_tokenizer: false
  requires_word_segmentation: true
```

Example `configs/models/cafebert.yaml`:

```yaml
model:
  model_key: cafebert
  checkpoint: uitnlp/CafeBERT
  architecture: ffn
  use_fast_tokenizer: true
  requires_word_segmentation: false
```

## CLI Overrides

Support practical overrides without editing YAML:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --model-config configs/models/xlmr_base.yaml \
  --data.train_file /path/train.jsonl \
  --training.num_train_epochs 1 \
  --project.output_dir outputs/debug
```

If dot-path overrides are too much for MVP, support a smaller set first:

- `--config`
- `--model`
- `--train-file`
- `--validation-file`
- `--test-file`
- `--output-dir`
- `--epochs`
- `--batch-size`
- `--learning-rate`

## Experiment Matrix

Minimum baseline experiments:

| Run | Model | Config | Notes |
| --- | --- | --- | --- |
| `mbert_cased_ffn` | `bert-base-multilingual-cased` | `default.yaml` + `mbert_cased.yaml` | primary multilingual baseline |
| `xlmr_base_ffn` | `xlm-roberta-base` | `default.yaml` + `xlmr_base.yaml` | multilingual RoBERTa baseline |
| `cafebert_ffn` | `uitnlp/CafeBERT` | `default.yaml` + `cafebert.yaml` | Vietnamese-focused multilingual model |
| `videberta_base_ffn` | `Fsoft-AIC/videberta-base` | `default.yaml` + `videberta_base.yaml` | Vietnamese DeBERTa baseline |
| `videberta_base_moe` | `Fsoft-AIC/videberta-base` | `default.yaml` + `videberta_base_moe.yaml` | ViDeBERTa with routed MoE FFN replacement |
| `phobert_base_ffn` | `vinai/phobert-base` | `default.yaml` + `phobert_base.yaml` | requires word segmentation |

Optional heavier experiments:

- `xlm-roberta-large`
- `vinai/phobert-large`
- `vinai/phobert-base-v2`

## Reproducibility Requirements

Every training run should save:

- resolved config
- git commit hash when available
- command used
- package versions
- train/dev/test metrics
- label mapping

## Data Format Examples

JSONL:

```json
{"uid":"uit_Adver_365_3_11_02","premise":"...","hypothesis":"...","label":"entailment"}
```

CSV:

```csv
uid,premise,hypothesis,label
uit_Adver_365_3_11_02,...,...,entailment
```

## Hyperparameter Tuning Workflow

Use config copies rather than editing source code:

```text
configs/experiments/
├── mbert_lr1e-5_bs16.yaml
├── mbert_lr2e-5_bs16.yaml
└── phobert_dropout0.4.yaml
```

Each experiment config should include only overrides and inherit from `default.yaml` in the loader or via command composition.
