"""Tests for decoder and vision-encoder LoRA adapters."""

from typing import Any

import pytest
import torch
from torch import nn
from transformers import PretrainedConfig

from bitikocr.params import LoraArguments
from bitikocr.train.adapters import (
    VISION_LORA_SUFFIXES,
    configure_adapters,
    visual_digest,
)

DECODER = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


class _VisionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(8, 24)
        self.attn.proj = nn.Linear(8, 8)
        self.mlp = nn.Module()
        self.mlp.linear_fc1 = nn.Linear(8, 16)
        self.mlp.linear_fc2 = nn.Linear(16, 8)


class _DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self.self_attn, name, nn.Linear(8, 8))
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(8, 16)
        self.mlp.up_proj = nn.Linear(8, 16)
        self.mlp.down_proj = nn.Linear(16, 8)


class _TinyVisionLanguageModel(nn.Module):
    """Named like Qwen's vision-language models, a few units wide."""

    def __init__(self) -> None:
        super().__init__()
        self.config = PretrainedConfig()
        self.model = nn.Module()
        self.model.visual = nn.Module()
        self.model.visual.patch_embed = nn.Module()
        self.model.visual.patch_embed.proj = nn.Conv2d(3, 8, 2)
        self.model.visual.blocks = nn.ModuleList(
            [_VisionBlock(), _VisionBlock()]
        )
        self.model.visual.merger = nn.Module()
        self.model.visual.merger.linear_fc1 = nn.Linear(8, 8)
        self.model.visual.merger.linear_fc2 = nn.Linear(8, 8)
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([_DecoderLayer()])

    def prepare_inputs_for_generation(self, *args: Any, **kwargs: Any) -> Any:
        return kwargs


def _adapted(model: nn.Module, **options: Any) -> Any:
    args = LoraArguments(
        lora_r=8, lora_alpha=16, lora_target_modules=DECODER, **options
    )
    return configure_adapters(model, args)


def _vision_lora(model: Any) -> dict[str, torch.nn.Parameter]:
    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if ".visual." in name and ".lora_" in name
    }


def test_a_frozen_encoder_gets_no_lora() -> None:
    model = _adapted(_TinyVisionLanguageModel())
    assert not _vision_lora(model)
    assert visual_digest(model) == ""


def test_vision_lora_reaches_every_block_and_the_merger() -> None:
    model = _adapted(_TinyVisionLanguageModel(), vision_lora_rank=4)
    adapted = {name.split(".lora_")[0] for name in _vision_lora(model)}
    layers = {name.rsplit(".", 1)[-1] for name in adapted}
    assert layers == VISION_LORA_SUFFIXES
    assert any(".merger." in name for name in adapted)
    assert not any(".patch_embed." in name for name in adapted)
    assert len(adapted) == 2 * 4 + 2


def test_vision_lora_has_its_own_rank() -> None:
    model = _adapted(_TinyVisionLanguageModel(), vision_lora_rank=4)
    ranks = {
        (".visual." in name): parameter.shape[0]
        for name, parameter in model.named_parameters()
        if name.endswith("lora_A.default.weight")
    }
    assert ranks == {True: 4, False: 8}


def test_vision_lora_updates_survive_a_bfloat16_model() -> None:
    """A learning-rate-sized step must change the weights (issue #12)."""
    model = _adapted(
        _TinyVisionLanguageModel().to(torch.bfloat16), vision_lora_rank=4
    )
    weights = _vision_lora(model)
    assert {p.dtype for p in weights.values()} == {torch.float32}
    before = {name: p.detach().clone() for name, p in weights.items()}
    optimizer = torch.optim.AdamW(weights.values(), lr=1e-4)
    for parameter in weights.values():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert all(
        not torch.equal(before[name], p.detach()) for name, p in weights.items()
    )


def test_training_the_whole_encoder_in_bfloat16_is_refused() -> None:
    with pytest.raises(ValueError, match="issue #12"):
        _adapted(
            _TinyVisionLanguageModel().to(torch.bfloat16),
            freeze_vision_encoder=False,
        )


def test_a_vision_lora_digest_is_recorded() -> None:
    model = _adapted(_TinyVisionLanguageModel(), vision_lora_rank=4)
    assert len(visual_digest(model)) == 64


@pytest.mark.parametrize(
    "options",
    [
        {"vision_lora_rank": -1},
        {"vision_lora_rank": 4, "freeze_vision_encoder": False},
    ],
)
def test_invalid_vision_settings_are_rejected(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        LoraArguments(lora_target_modules=DECODER, **options)
