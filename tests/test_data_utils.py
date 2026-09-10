"""Check image preparation and Qwen metadata without remote model access."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from bitikocr.data import data_utils


@pytest.mark.parametrize("present", [False, True])
def test_modality_ids(present: bool) -> None:
    """Preserve processor IDs or derive only image positions as needed."""
    ids = torch.tensor([[10, 91, 91, 12]])
    supplied = torch.tensor([[0, 1, 1, 0]], dtype=torch.int32)
    inputs = {"mm_token_type_ids": supplied} if present else {}
    result = data_utils.get_mm_token_type_ids(inputs, ids, 91)
    assert result.dtype == torch.long
    assert result.tolist() == [[0, 1, 1, 0]]


@pytest.mark.parametrize(
    ("model_type", "patch_size", "system_message"),
    [
        ("qwen2_vl", 14, True),
        ("qwen2_5_vl", 14, True),
        ("qwen3_vl", 16, False),
        ("qwen3_vl_moe", 16, False),
        ("qwen3_5", 16, False),
        ("qwen3_5_moe", 16, False),
    ],
)
def test_qwen_settings(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
    patch_size: int,
    system_message: bool,
) -> None:
    """Select the expected patch alignment and system prompt by Qwen family."""
    monkeypatch.setattr(
        data_utils.AutoConfig,
        "from_pretrained",
        lambda _: SimpleNamespace(model_type=model_type),
    )
    data_utils.get_qwen_multimodal_settings.cache_clear()
    try:
        assert data_utils.get_qwen_multimodal_settings("test-model") == (
            model_type,
            patch_size,
        )
        assert (
            data_utils.use_default_system_message(model_type) is system_message
        )
    finally:
        data_utils.get_qwen_multimodal_settings.cache_clear()


@pytest.mark.parametrize("patch_size", [14, 16])
def test_prepare_local_image(tmp_path: Path, patch_size: int) -> None:
    """Resize real local images to the model's merged-patch alignment in RGB."""
    path = tmp_path / "image.png"
    Image.new("L", (80, 100), 100).save(path)
    result = data_utils.get_image_info(
        str(path), 3136, 12800, None, None, patch_size
    )
    assert result.mode == "RGB"
    assert result.width % (2 * patch_size) == 0
    assert result.height % (2 * patch_size) == 0
    assert 3136 <= result.width * result.height <= 12800


def test_explicit_image_dimensions(tmp_path: Path) -> None:
    """Pass requested dimensions through Qwen's image resize utility."""
    path = tmp_path / "image.png"
    Image.new("RGB", (80, 100)).save(path)
    result = data_utils.get_image_info(str(path), 3136, 12800, 112, 56, 14)
    assert result.size == (112, 56)
