from __future__ import annotations

from vnnli_engram_moe.config import AppConfig
from vnnli_engram_moe.constants import ID2LABEL, LABEL2ID


def build_ffn_model(config: AppConfig, *, checkpoint: str, pretrained: bool = True):
    from transformers import AutoConfig, AutoModelForSequenceClassification

    model_config = AutoConfig.from_pretrained(
        checkpoint,
        local_files_only=config.model.local_files_only,
    )
    model_config.num_labels = config.model.num_labels
    model_config.id2label = ID2LABEL
    model_config.label2id = LABEL2ID

    dropout = config.training.dropout
    for attribute in ("classifier_dropout", "hidden_dropout_prob", "attention_probs_dropout_prob"):
        if hasattr(model_config, attribute):
            setattr(model_config, attribute, dropout)

    if pretrained:
        return AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            config=model_config,
            ignore_mismatched_sizes=True,
            local_files_only=config.model.local_files_only,
        )
    return AutoModelForSequenceClassification.from_config(model_config)
