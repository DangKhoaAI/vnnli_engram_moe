from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from vnnli_engram_moe.config import load_config
from vnnli_engram_moe.training.trainer import train


class DummyTokenizer:
    def __call__(
        self,
        premises,
        hypotheses,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors=None,
    ):
        if isinstance(premises, str):
            premises = [premises]
            hypotheses = [hypotheses]

        input_ids = []
        attention_mask = []
        for premise, hypothesis in zip(premises, hypotheses, strict=True):
            seed = (len(premise) + len(hypothesis)) % 13 + 1
            tokens = [seed, seed + 1, seed + 2]
            padded = (tokens + [0] * max_length)[:max_length]
            input_ids.append(padded)
            attention_mask.append([1 if token else 0 for token in padded])

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, directory) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "tokenizer.json").write_text("{}", encoding="utf-8")


class DummyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 16)
        self.classifier = torch.nn.Linear(16, 3)

    def forward(self, input_ids, attention_mask=None, labels=None):
        embeddings = self.embedding(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = embeddings.mean(dim=1)
        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, directory) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target / "pytorch_model.bin")


def test_smoke_training_run(tmp_path: Path) -> None:
    config = load_config(
        "configs/default.yaml",
        overrides=[
            "training.num_train_epochs=1",
            "training.max_steps=1",
            "training.per_device_train_batch_size=2",
            "training.per_device_eval_batch_size=2",
            "training.eval_frequency=1",
            "training.logging_steps=1",
            "training.device='cpu'",
            "project.run_name='smoke'",
            f"project.output_dir='{tmp_path.as_posix()}'",
            "data.train_file='tests/fixtures/sample_vianli.jsonl'",
            "data.validation_file='tests/fixtures/sample_vianli.jsonl'",
            "data.test_file='tests/fixtures/sample_vianli.jsonl'",
        ],
    )

    run_dir = train(
        config,
        tokenizer_builder=lambda *_args, **_kwargs: DummyTokenizer(),
        model_builder=lambda *_args, **_kwargs: DummyModel(),
        pretrained=False,
    )

    assert run_dir.exists()
    assert (run_dir / "config.resolved.yaml").exists()
    assert (run_dir / "train_metrics.json").exists()
    assert (run_dir / "dev_metrics.json").exists()
    assert (run_dir / "test_metrics.json").exists()
    assert (run_dir / "final_model" / "pytorch_model.bin").exists()

