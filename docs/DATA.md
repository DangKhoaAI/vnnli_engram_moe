# Data Guide

## Task

Vietnamese NLI classifies the logical relation between a `premise` and a `hypothesis` into:

- `entailment`
- `contradiction`
- `neutral`

## ViANLI Sizes

- Train: 8012
- Dev: 1000
- Test: 1000
- Total: 10012

## Required Columns

- `uid`
- `premise`
- `hypothesis`
- `label`

## Accepted Formats

- JSONL
- CSV
- Parquet when parquet support is present in the runtime

## Label Mapping

- `entailment -> 0`
- `contradiction -> 1`
- `neutral -> 2`

## Example Row

```json
{
  "uid": "uit_Adver_365_3_11_02",
  "premise": "Tọa đàm do Tổng cục Du lịch phối hợp với báo điện tử VnExpress tổ chức ngày 3/4 tại FLC Sầm Sơn, Thanh Hóa.",
  "hypothesis": "Đầu tháng 4 có một buổi gặp mặt trao đổi của Tổng cục Du lịch.",
  "label": "entailment"
}
```

## PhoBERT Segmentation

PhoBERT often works best when Vietnamese compound words are segmented before subword tokenization. The codebase includes an optional `py_vncorenlp` hook for this. If the package is absent and no-op fallback is allowed, training can still continue for development and testing.

