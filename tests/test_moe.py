from types import SimpleNamespace

import torch
from torch import nn

from vnnli_engram_moe.config import load_config
from vnnli_engram_moe.models.moe import PassThroughIntermediate, RoutedMoEOutput
from vnnli_engram_moe.models.moe import replace_last_ffn_blocks_with_moe
from vnnli_engram_moe.training.trainer import _is_moe_warmup_trainable


class DummyIntermediate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(4, 6)
        self.intermediate_act_fn = torch.relu


class DummyOutput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(6, 4)
        self.dropout = nn.Dropout(0.0)
        self.LayerNorm = nn.LayerNorm(4)

    def forward(self, hidden_states, input_tensor):
        return self.LayerNorm(self.dropout(self.dense(hidden_states)) + input_tensor)


class DummyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.intermediate = DummyIntermediate()
        self.output = DummyOutput()


class DummyDebertaModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.deberta = SimpleNamespace(
            encoder=SimpleNamespace(layer=nn.ModuleList(DummyLayer() for _ in range(4)))
        )


def test_replace_last_ffn_blocks_with_moe_keeps_early_layers_dense() -> None:
    model = DummyDebertaModel()
    original_input_weight = model.deberta.encoder.layer[-1].intermediate.dense.weight.detach().clone()
    original_output_weight = model.deberta.encoder.layer[-1].output.dense.weight.detach().clone()
    config = load_config(
        "configs/default.yaml",
        overrides=[
            "model.architecture='moe'",
            "moe.replace_last_n_layers=2",
            "moe.num_experts=4",
            "moe.top_k=2",
        ],
    )

    moe_outputs = replace_last_ffn_blocks_with_moe(model, config)

    assert len(moe_outputs) == 2
    assert isinstance(model.deberta.encoder.layer[0].intermediate, DummyIntermediate)
    assert isinstance(model.deberta.encoder.layer[-1].intermediate, PassThroughIntermediate)
    assert isinstance(model.deberta.encoder.layer[-1].output, RoutedMoEOutput)
    assert torch.equal(moe_outputs[-1].experts[0].input_dense.weight, original_input_weight)
    assert torch.equal(moe_outputs[-1].experts[0].output_dense.weight, original_output_weight)


def test_routed_moe_output_forward_shape_and_auxiliary_loss() -> None:
    model = DummyDebertaModel()
    config = load_config(
        "configs/default.yaml",
        overrides=["model.architecture='moe'", "moe.replace_last_n_layers=1"],
    )
    moe_output = replace_last_ffn_blocks_with_moe(model, config)[0]
    hidden_states = torch.randn(2, 3, 4)

    output = model.deberta.encoder.layer[-1].output(hidden_states, hidden_states)

    assert output.shape == hidden_states.shape
    assert moe_output.auxiliary_loss.ndim == 0


def test_moe_warmup_trainable_name_filter() -> None:
    assert _is_moe_warmup_trainable("model.deberta.encoder.layer.10.output.router.weight")
    assert _is_moe_warmup_trainable("model.deberta.encoder.layer.10.output.experts.0.input_dense.weight")
    assert _is_moe_warmup_trainable("model.classifier.weight")
    assert not _is_moe_warmup_trainable("model.deberta.encoder.layer.0.attention.self.query_proj.weight")
