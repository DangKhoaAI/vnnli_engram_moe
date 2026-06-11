# Documentation Plan

## Top-Level README

Target file:

```text
README.md
```

Required sections:

- Project title and one-paragraph overview.
- NLI label definitions.
- Supported models.
- Environment and installation.
- Dataset format.
- Quickstart training command.
- Evaluation/prediction examples.
- Kaggle usage pointer.
- Project structure.
- Current status and future work.

Example quickstart:

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run python scripts/train.py \
  --config configs/default.yaml \
  --model-config configs/models/mbert_cased.yaml \
  --train-file data/ViANLI/train.jsonl \
  --validation-file data/ViANLI/dev.jsonl \
  --test-file data/ViANLI/test.jsonl
```

## `PROJECT.md`

This file is explicitly required by the user at the repo root. It should include:

- Project overview.
- Architecture and tech stack.
- Directory structure.
- Module/function list.
- Training workflow.
- Future MoE/Engram+MoE architecture notes.
- Output artifacts.

## `STATUS.md`

Include:

- current project summary
- completed vs pending work
- open issues and risks
- verification snapshot
- reference table for deeper agent-facing status files

## Optional Supporting Docs

If more repository-facing documentation becomes useful later, prefer either:

- expanding `PROJECT.md`
- expanding `STATUS.md`
- or placing deeper agent-oriented guidance under `plan/project/`

## Documentation Acceptance

Docs are acceptable when a new agent can answer:

- What problem is this solving?
- What command trains a model?
- Where are data paths configured?
- Which files should I edit to tune hyperparameters?
- How do I run on Kaggle?
- How will MoE/Engram+MoE fit later?
