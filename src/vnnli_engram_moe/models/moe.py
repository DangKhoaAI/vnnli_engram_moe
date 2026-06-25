from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from vnnli_engram_moe.config import AppConfig
from vnnli_engram_moe.constants import ID2LABEL, LABEL2ID

MOE_MARKER_FILE = "vnnli_moe_config.json"


class PassThroughIntermediate(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class DenseFFNExpert(nn.Module):
    def __init__(
        self,
        input_dense: nn.Linear,
        output_dense: nn.Linear,
        activation,
    ) -> None:
        super().__init__()
        self.input_dense = copy.deepcopy(input_dense)
        self.output_dense = copy.deepcopy(output_dense)
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.input_dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.output_dense(hidden_states)


class RoutedMoEOutput(nn.Module):
    def __init__(
        self,
        *,
        input_dense: nn.Linear,
        output_dense: nn.Linear,
        activation,
        layer_norm: nn.Module,
        dropout: nn.Module,
        num_experts: int,
        top_k: int,
        router_temperature: float,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("moe.num_experts must be >= 1.")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("moe.top_k must be between 1 and moe.num_experts.")

        self.num_experts = num_experts
        self.top_k = top_k
        self.router_temperature = router_temperature
        self.experts = nn.ModuleList(
            DenseFFNExpert(input_dense, output_dense, activation) for _ in range(num_experts)
        )
        self.router = nn.Linear(input_dense.in_features, num_experts)
        self.dropout = copy.deepcopy(dropout)
        self.LayerNorm = copy.deepcopy(layer_norm)
        self.auxiliary_loss = torch.tensor(0.0)

    def reset_auxiliary_loss(self) -> None:
        self.auxiliary_loss = torch.tensor(0.0)

    def _load_balance_loss(
        self,
        router_probs: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        expert_mask = torch.zeros_like(router_probs)
        expert_mask.scatter_add_(
            -1,
            topk_indices,
            torch.ones_like(topk_indices, dtype=router_probs.dtype),
        )
        expert_density = expert_mask.mean(dim=(0, 1)) / self.top_k
        router_density = router_probs.mean(dim=(0, 1))
        return self.num_experts * torch.sum(expert_density * router_density)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        temperature = max(float(self.router_temperature), 1.0e-6)
        router_logits = self.router(hidden_states) / temperature
        router_probs = torch.softmax(router_logits, dim=-1)
        topk_probs, topk_indices = torch.topk(router_probs, k=self.top_k, dim=-1)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)

        routing_weights = torch.zeros_like(router_probs)
        routing_weights.scatter_add_(-1, topk_indices, topk_probs)

        expert_outputs = torch.stack(
            [expert(hidden_states) for expert in self.experts],
            dim=-2,
        )
        mixed_output = torch.sum(expert_outputs * routing_weights.unsqueeze(-1), dim=-2)
        self.auxiliary_loss = self._load_balance_loss(router_probs, topk_indices)

        mixed_output = self.dropout(mixed_output)
        return self.LayerNorm(mixed_output + input_tensor)


class MoESequenceClassificationWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        moe_outputs: list[RoutedMoEOutput],
        config: AppConfig,
    ) -> None:
        super().__init__()
        self.model = model
        self.moe_outputs = nn.ModuleList(moe_outputs)
        self.app_config = config

    def forward(self, *args, **kwargs):
        for moe_output in self.moe_outputs:
            moe_output.reset_auxiliary_loss()

        outputs = self.model(*args, **kwargs)
        loss = getattr(outputs, "loss", None)
        if loss is not None and self.moe_outputs:
            auxiliary_loss = sum(
                moe_output.auxiliary_loss.to(loss.device) for moe_output in self.moe_outputs
            )
            outputs.loss = loss + self.app_config.moe.load_balance_loss_weight * auxiliary_loss
        return outputs

    def save_pretrained(self, directory: str | Path, **kwargs) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("safe_serialization", False)
        self.model.save_pretrained(target, **kwargs)
        marker = {
            "architecture": "moe",
            "num_experts": self.app_config.moe.num_experts,
            "top_k": self.app_config.moe.top_k,
            "replace_last_n_layers": self.app_config.moe.replace_last_n_layers,
            "router_temperature": self.app_config.moe.router_temperature,
        }
        (target / MOE_MARKER_FILE).write_text(json.dumps(marker, indent=2), encoding="utf-8")


def _configure_sequence_classification(model_config: Any, config: AppConfig) -> Any:
    model_config.num_labels = config.model.num_labels
    model_config.id2label = ID2LABEL
    model_config.label2id = LABEL2ID

    dropout = config.training.dropout
    for attribute in ("classifier_dropout", "hidden_dropout_prob", "attention_probs_dropout_prob"):
        if hasattr(model_config, attribute):
            setattr(model_config, attribute, dropout)
    return model_config


def _resolve_attr(root: Any, dotted_path: str) -> Any | None:
    cursor = root
    for part in dotted_path.split("."):
        cursor = getattr(cursor, part, None)
        if cursor is None:
            return None
    return cursor


def _find_encoder_layers(model: nn.Module):
    for candidate_path in (
        "deberta.encoder.layer",
        "base_model.encoder.layer",
        "encoder.layer",
    ):
        layers = _resolve_attr(model, candidate_path)
        if layers is not None:
            return layers
    raise ValueError(
        "Could not locate encoder layers for MoE replacement. "
        "Expected a DeBERTa-style path such as model.deberta.encoder.layer."
    )


def _get_ffn_parts(layer: nn.Module):
    intermediate = getattr(layer, "intermediate", None)
    output = getattr(layer, "output", None)
    input_dense = getattr(intermediate, "dense", None)
    output_dense = getattr(output, "dense", None)
    layer_norm = getattr(output, "LayerNorm", None)
    dropout = getattr(output, "dropout", None)
    activation = getattr(intermediate, "intermediate_act_fn", None)
    if not all((input_dense, output_dense, layer_norm, dropout, activation)):
        return None
    return input_dense, output_dense, activation, layer_norm, dropout


def replace_last_ffn_blocks_with_moe(model: nn.Module, config: AppConfig) -> list[RoutedMoEOutput]:
    layers = list(_find_encoder_layers(model))
    replace_count = min(config.moe.replace_last_n_layers, len(layers))
    if replace_count < 1:
        raise ValueError("moe.replace_last_n_layers must be >= 1.")

    moe_outputs: list[RoutedMoEOutput] = []
    for layer in layers[-replace_count:]:
        ffn_parts = _get_ffn_parts(layer)
        if ffn_parts is None:
            raise ValueError(
                "Selected encoder layer does not expose DeBERTa-style "
                "intermediate.dense/output.dense FFN modules."
            )
        input_dense, output_dense, activation, layer_norm, dropout = ffn_parts
        routed_output = RoutedMoEOutput(
            input_dense=input_dense,
            output_dense=output_dense,
            activation=activation,
            layer_norm=layer_norm,
            dropout=dropout,
            num_experts=config.moe.num_experts,
            top_k=config.moe.top_k,
            router_temperature=config.moe.router_temperature,
        )
        layer.intermediate = PassThroughIntermediate()
        layer.output = routed_output
        moe_outputs.append(routed_output)
    return moe_outputs


def _is_saved_moe_checkpoint(checkpoint: str | Path) -> bool:
    return (Path(checkpoint) / MOE_MARKER_FILE).exists()


def _load_saved_moe_weights(wrapper: MoESequenceClassificationWrapper, checkpoint: str | Path) -> None:
    state_path = Path(checkpoint) / "pytorch_model.bin"
    if not state_path.exists():
        raise FileNotFoundError(
            f"Saved MoE checkpoint is missing {state_path.name}. "
            "This project saves MoE checkpoints with safe_serialization=False."
        )
    state_dict = torch.load(state_path, map_location="cpu")
    wrapper.model.load_state_dict(state_dict)


def build_moe_model(config: AppConfig, *, checkpoint: str, pretrained: bool = True):
    from transformers import AutoConfig, AutoModelForSequenceClassification

    model_config = AutoConfig.from_pretrained(
        checkpoint,
        local_files_only=config.model.local_files_only,
    )
    model_config = _configure_sequence_classification(model_config, config)

    if pretrained and _is_saved_moe_checkpoint(checkpoint):
        model = AutoModelForSequenceClassification.from_config(model_config)
        moe_outputs = replace_last_ffn_blocks_with_moe(model, config)
        wrapper = MoESequenceClassificationWrapper(model, moe_outputs, config)
        _load_saved_moe_weights(wrapper, checkpoint)
        return wrapper

    if pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            config=model_config,
            ignore_mismatched_sizes=True,
            local_files_only=config.model.local_files_only,
        )
    else:
        model = AutoModelForSequenceClassification.from_config(model_config)

    moe_outputs = replace_last_ffn_blocks_with_moe(model, config)
    return MoESequenceClassificationWrapper(model, moe_outputs, config)
