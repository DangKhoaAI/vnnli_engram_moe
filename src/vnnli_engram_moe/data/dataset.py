from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from vnnli_engram_moe.config import AppConfig
from vnnli_engram_moe.data.io import read_records, validate_records
from vnnli_engram_moe.data.preprocess import TextPreprocessor, encode_records


@dataclass(slots=True)
class EncodedSplit:
    name: str
    examples: list[dict[str, object]]


class EncodedNLIDataset(Dataset):
    def __init__(self, examples: list[dict[str, object]]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.examples[index]


def load_available_splits(config: AppConfig, tokenizer, preprocessor: TextPreprocessor) -> dict[str, EncodedSplit]:
    split_files = {
        "train": config.data.train_file,
        "validation": config.data.validation_file,
        "test": config.data.test_file,
    }
    output: dict[str, EncodedSplit] = {}
    for split_name, source in split_files.items():
        if not source:
            continue
        records = read_records(source, config.data)
        validate_records(records, config.data)
        encoded = encode_records(
            records,
            tokenizer,
            data_config=config.data,
            training_config=config.training,
            preprocessor=preprocessor,
        )
        output[split_name] = EncodedSplit(name=split_name, examples=encoded)
    return output


def collate_batch(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
    tensor_batch: dict[str, torch.Tensor] = {}
    for key in batch[0]:
        if key == "uid":
            continue
        tensor_batch[key] = torch.tensor([example[key] for example in batch], dtype=torch.long)
    return tensor_batch

