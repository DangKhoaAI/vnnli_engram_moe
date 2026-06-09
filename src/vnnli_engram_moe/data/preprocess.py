from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vnnli_engram_moe.config import DataConfig, ModelConfig, TrainingConfig
from vnnli_engram_moe.data.labels import label_to_id


class TextPreprocessor(Protocol):
    def prepare(self, text: str) -> str:
        """Convert raw text into the form expected by the tokenizer."""


@dataclass(slots=True)
class NoOpPreprocessor:
    def prepare(self, text: str) -> str:
        return text


@dataclass(slots=True)
class PhoBertWordSegmenter:
    strict: bool = False

    def __post_init__(self) -> None:
        self._segmenter = self._load_segmenter()

    def _load_segmenter(self):
        try:
            import py_vncorenlp  # type: ignore
        except ImportError:
            return None
        return py_vncorenlp.VnCoreNLP(annotators=["wseg"])

    def prepare(self, text: str) -> str:
        if self._segmenter is None:
            if self.strict:
                raise RuntimeError(
                    "PhoBERT word segmentation was requested but py_vncorenlp is not installed."
                )
            return text
        segmented = self._segmenter.word_segment(text)
        if isinstance(segmented, list):
            return " ".join(segmented)
        return str(segmented)


def build_preprocessor(model_config: ModelConfig) -> TextPreprocessor:
    if not model_config.requires_word_segmentation:
        return NoOpPreprocessor()
    return PhoBertWordSegmenter(strict=not model_config.allow_noop_word_segmenter)


def encode_records(
    records: list[dict[str, object]],
    tokenizer,
    *,
    data_config: DataConfig,
    training_config: TrainingConfig,
    preprocessor: TextPreprocessor,
) -> list[dict[str, object]]:
    premise_key = data_config.text_columns.premise
    hypothesis_key = data_config.text_columns.hypothesis

    premises = [preprocessor.prepare(str(record[premise_key])) for record in records]
    hypotheses = [preprocessor.prepare(str(record[hypothesis_key])) for record in records]
    encoded_batch = tokenizer(
        premises,
        hypotheses,
        truncation=True,
        padding="max_length",
        max_length=training_config.max_length,
    )

    examples: list[dict[str, object]] = []
    for index, record in enumerate(records):
        example = {key: value[index] for key, value in encoded_batch.items()}
        example["labels"] = label_to_id(str(record[data_config.label_column]))
        example["uid"] = str(record[data_config.uid_column])
        examples.append(example)
    return examples

