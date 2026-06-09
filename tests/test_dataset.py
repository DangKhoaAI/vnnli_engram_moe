from pathlib import Path

import pytest

from vnnli_engram_moe.config import load_config
from vnnli_engram_moe.data.dataset import load_available_splits
from vnnli_engram_moe.data.io import read_hf_records, read_records, validate_records
from vnnli_engram_moe.data.preprocess import NoOpPreprocessor, encode_records


class FakeTokenizer:
    def __call__(self, premises, hypotheses, truncation, padding, max_length):
        if isinstance(premises, str):
            premises = [premises]
            hypotheses = [hypotheses]

        input_ids = []
        attention_mask = []
        for premise, hypothesis in zip(premises, hypotheses, strict=True):
            base = [len(premise) % 7 + 1, len(hypothesis) % 7 + 2]
            padded = (base + [0] * max_length)[:max_length]
            input_ids.append(padded)
            attention_mask.append([1 if token else 0 for token in padded])
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_jsonl_loading_and_encoding() -> None:
    config = load_config("configs/default.yaml")
    records = read_records("tests/fixtures/sample_vianli.jsonl", config.data)
    validate_records(records, config.data)

    encoded = encode_records(
        records,
        FakeTokenizer(),
        data_config=config.data,
        training_config=config.training,
        preprocessor=NoOpPreprocessor(),
    )

    assert len(records) == 4
    assert encoded[0]["labels"] == 0
    assert len(encoded[0]["input_ids"]) == config.training.max_length


def test_csv_loading() -> None:
    config = load_config("configs/default.yaml")
    records = read_records("tests/fixtures/sample_vianli.csv", config.data)
    validate_records(records, config.data)

    assert len(records) == 3
    assert records[1]["label"] == "contradiction"


def test_missing_columns_raise(tmp_path: Path) -> None:
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"uid":"x","premise":"a","label":"neutral"}\n', encoding="utf-8")

    config = load_config("configs/default.yaml")
    records = read_records(broken, config.data)
    with pytest.raises(ValueError):
        validate_records(records, config.data)


def test_hf_dataset_split_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(
        "configs/default.yaml",
        overrides=["data.hf_dataset='uitnlp/ViANLI'"],
    )

    class FakeDatasetDict(dict):
        pass

    def fake_load_dataset(dataset_name, name=None):
        assert dataset_name == "uitnlp/ViANLI"
        assert name is None
        return FakeDatasetDict(
            train=[{"uid": "1", "premise": "a", "hypothesis": "b", "label": "entailment"}],
            validation=[{"uid": "2", "premise": "c", "hypothesis": "d", "label": "neutral"}],
            test=[{"uid": "3", "premise": "e", "hypothesis": "f", "label": "contradiction"}],
        )

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    records = read_hf_records("validation", config.data)

    assert len(records) == 1
    assert records[0]["label"] == "neutral"


def test_load_available_splits_prefers_hf_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(
        "configs/default.yaml",
        overrides=["data.hf_dataset='uitnlp/ViANLI'"],
    )

    def fake_load_dataset(dataset_name, name=None):
        return {
            "train": [{"uid": "1", "premise": "train", "hypothesis": "sample", "label": "entailment"}],
            "validation": [{"uid": "2", "premise": "dev", "hypothesis": "sample", "label": "neutral"}],
            "test": [{"uid": "3", "premise": "test", "hypothesis": "sample", "label": "contradiction"}],
        }

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    splits = load_available_splits(config, FakeTokenizer(), NoOpPreprocessor())

    assert set(splits) == {"train", "validation", "test"}
    assert splits["train"].examples[0]["labels"] == 0
