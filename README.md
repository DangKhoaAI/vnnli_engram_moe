# VNNLI Engram MoE

Repository này cung cấp một codebase fine-tuning cho bài toán Vietnamese Natural Language Inference (NLI).
Scope hiện tại gồm Design 1: transformer encoder + classification head, và nhánh ViDeBERTa + MoE cho bài toán phân loại cặp câu.
Codebase vẫn giữ sẵn đường mở rộng để sau này thêm `engram_moe` mà không cần refactor lớn.

## Bài toán
Model nhận vào một cặp câu:
- `premise`
- `hypothesis`

Và dự đoán một trong 3 nhãn:
- `entailment`
- `contradiction`
- `neutral`

## Trạng thái hiện tại
- Baseline Design 1 đã được implement
- ViDeBERTa keys và config đã được thêm
- MoE head cho ViDeBERTa/DeBERTa-style encoder đã được implement
- CLI train / evaluate / predict đã có
- Config đã được centralize
- Offline-first test suite đã có
- Notebook Kaggle đã có và đã bám flow mounted source/data/model input hiện tại
- `engram_moe` vẫn đang ở mức extension point

Tóm tắt tiến độ:
- Hoàn thành `8/8` bước cho baseline scope
- Đã có nhánh ViDeBERTa + MoE để train thử nghiệm
- Chưa hoàn thành phần Engram + MoE

## Stack kỹ thuật
- Python `3.12.13`
- PyTorch `2.10.0+cu128`
- Transformers `5.0.0`
- YAML config + typed Python schema
- `uv` cho local environment
- Pytest cho test và smoke run
- Kaggle notebook cho flow setup / train tren mounted Kaggle inputs

## Model và data scope
Model chính hiện tại: `bert-base-multilingual-cased`, `xlm-roberta-base`, `uitnlp/CafeBERT`, `Fsoft-AIC/videberta-base`, `vinai/phobert-base`.
Schema đầu vào mong đợi:
- `uid`
- `premise`
- `hypothesis`
- `label`

Supported ingestion paths:
- local JSONL
- local CSV
- optional local Parquet
- Hugging Face dataset loading cho `uitnlp/ViANLI`

## Cấu trúc repo
```text
configs/        config tổng và model-specific overrides
notebooks/      notebook để upload lên Kaggle
scripts/        train / evaluate / predict entry points
src/            package chính
tests/          unit tests và smoke tests
plan/           bộ nhớ dài hạn cho agent
```

Trong `plan/`:
```text
plan/project/   tài liệu planning và implementation chi tiết
plan/status/    progress, decisions, blockers, agent workflow
```

## Bắt đầu nhanh
```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest
```

Train bằng Hugging Face dataset:
```bash
uv run python scripts/train.py \
  --config configs/default.yaml \
  --model-config configs/models/mbert_cased.yaml \
  --hf-dataset uitnlp/ViANLI
```

Train bằng local files:
```bash
uv run python scripts/train.py \
  --config configs/default.yaml \
  --model-config configs/models/mbert_cased.yaml \
  --train-file path/to/train.jsonl \
  --validation-file path/to/dev.jsonl \
  --test-file path/to/test.jsonl
```

Train ViDeBERTa + MoE bằng local files hoặc Kaggle-mounted checkpoint:
```bash
uv run python scripts/train.py \
  --config configs/default.yaml \
  --model-config configs/models/videberta_base_moe.yaml \
  --checkpoint /path/to/videberta-base \
  --local-files-only \
  --train-file path/to/train.jsonl \
  --validation-file path/to/validation.jsonl \
  --test-file path/to/test.jsonl
```

Evaluate:
```bash
uv run python scripts/evaluate.py --run-dir outputs/runs/<run_name> --split test
```

Predict:
```bash
uv run python scripts/predict.py \
  --run-dir outputs/runs/<run_name> \
  --premise "..." \
  --hypothesis "..."
```

## Bên trong code có gì
- `src/vnnli_engram_moe/config.py`: config loading và resolved config
- `src/vnnli_engram_moe/data/`: đọc data, map label, preprocess, tokenize pair
- `src/vnnli_engram_moe/models/registry.py`: chọn checkpoint và architecture builder
- `src/vnnli_engram_moe/models/ffn.py`: baseline hiện tại
- `src/vnnli_engram_moe/models/moe.py`: ViDeBERTa/DeBERTa-style MoE FFN replacement
- `src/vnnli_engram_moe/training/trainer.py`: orchestration cho train / eval / predict
- `scripts/train.py`, `scripts/evaluate.py`, `scripts/predict.py`: CLI surface của repo

## Output và phạm vi còn lại
Mỗi run được ghi vào `outputs/runs/`, thường gồm resolved config, metadata, label mapping, metrics, và final saved model.

Đã xong:
- baseline Design 1
- config system
- dataset pipeline
- CLI train / evaluate / predict
- Kaggle notebook
- ViDeBERTa + MoE config và training path
- test suite cơ bản
- project/status surface cho human và agent

Chưa xong:
- `engram_moe`
- quyết định cuối cùng về local artifact như `data.py`, `src.zip`, `vianli_kaggle/`, `vianli_kaggle.zip`

## Nên đọc gì tiếp
- Muốn hiểu dự án kỹ hơn: [PROJECT.md](PROJECT.md)
- Muốn xem tình trạng hiện tại: [STATUS.md](STATUS.md)
- Muốn xem implementation planning chi tiết: [plan/project/](plan/project/)
- Muốn xem progress, decision, blocker của agent: [plan/status/](plan/status/)
- Muốn xem notebook upload lên Kaggle: [notebooks/kaggle_wrapup.ipynb](notebooks/kaggle_wrapup.ipynb)
