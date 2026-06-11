# Project Requirements

This file normalizes the user-provided requirements in `TASK.md` and `INFO.md`.

## Problem

Vietnamese Natural Language Inference is a 3-class classification problem over sentence pairs:

- `entailment`: premise implies hypothesis.
- `contradiction`: premise contradicts hypothesis.
- `neutral`: premise does not imply or contradict hypothesis.

Input columns expected from ViANLI:

- `uid`
- `premise`
- `hypothesis`
- `label`

Dataset sizes from `INFO.md`:

- Train: 8012
- Dev: 1000
- Test: 1000
- Total: 10012

## Target Models

Baseline models to fine-tune:

| Key | Hugging Face checkpoint | Notes |
| --- | --- | --- |
| `mbert_cased` | `bert-base-multilingual-cased` | multilingual BERT |
| `mbert_uncased` | `bert-base-multilingual-uncased` | optional variant |
| `xlmr_base` | `xlm-roberta-base` | multilingual RoBERTa |
| `xlmr_large` | `xlm-roberta-large` | heavier optional variant |
| `cafebert` | `uitnlp/CafeBERT` | based on XLM-R style architecture |
| `phobert_base` | `vinai/phobert-base` | Vietnamese-specific |
| `phobert_large` | `vinai/phobert-large` | heavier optional variant |
| `phobert_base_v2` | `vinai/phobert-base-v2` | optional variant |

## Design 1: FFN Baseline

Implement now.

Expected flow:

1. Pair premise and hypothesis into one tokenizer input.
2. Use transformer hidden representation for paired input.
3. Add classification head with output dimension 3.
4. Train using cross entropy over labels.
5. Report accuracy and macro-F1 at minimum.

For Hugging Face models, `AutoModelForSequenceClassification(num_labels=3)` is acceptable for the MVP because it implements the same practical baseline: encoder plus classification head.

## Design 2: MoE

Future implementation. Do not build full MoE now unless requested.

The codebase should still include extension points:

- architecture key: `moe`
- expert module interface
- router interface
- config fields for `num_experts`, `top_k`, `expert_hidden_size`, `router_temperature`, and `load_balance_loss_weight`

Initial behavior may raise `NotImplementedError` when `architecture=moe`, as long as the error is explicit and tests cover registry behavior.

## Design 3: Engram + MoE

Future/research implementation. Do not build full Engram+MoE now unless requested.

Prepare extension points:

- architecture key: `engram_moe`
- config namespace: `engram`
- optional fields for memory size, retrieval top-k, memory update policy, and engram feature fusion mode.

## Environment

Target production environment from user:

- Python `3.12.13`
- PyTorch `2.10.0+cu128`
- Transformers `5.0.0`

Implementation should use `uv` in the current repo for local dev.

Important: Transformers 5.0.0 may have API differences from older 4.x examples. Future implementer should verify exact APIs in the installed environment instead of blindly copying older Trainer code.

## Training Hyperparameters

Central defaults:

- `max_length: 256`
- `learning_rate: 1e-5`
- `eval_frequency: 400`
- `batch_size: 16`
- `weight_decay: 0.0`
- `adam_epsilon: 1e-8`
- `dropout: 0.4`
- `epochs: 7`

## Deliverables

Required final deliverables:

- Python training source code.
- Central config files.
- Kaggle wrapup notebook.
- Tests.
- README.
- `PROJECT.md`.
- `STATUS.md`.
- Additional planning notes under `plan/project/` if useful.
