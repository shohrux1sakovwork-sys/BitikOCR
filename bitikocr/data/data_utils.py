"""Prepare Qwen image inputs and preserve processor modality information."""

from functools import lru_cache

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig


def get_mm_token_type_ids(
    inputs: dict, input_ids: torch.Tensor, image_token_id: int
) -> torch.Tensor:
    """Return modality IDs, deriving image positions if they are absent.

    Args:
        inputs: Processor output for an image-only OCR prompt.
        input_ids: Prompt tokens, including expanded image placeholders.
        image_token_id: The processor's image placeholder token ID.

    Returns:
        Processor IDs, or zero for text and one for image placeholders.
    """
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if mm_token_type_ids is None:
        return (input_ids == image_token_id).to(dtype=torch.long)
    return mm_token_type_ids.to(dtype=torch.long)


@lru_cache(maxsize=32)
def get_qwen_multimodal_settings(model_id_or_path: str) -> tuple[str, int]:
    """Read the model type and select its image patch size."""
    model_type = AutoConfig.from_pretrained(model_id_or_path).model_type
    if model_type in {"qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"}:
        return model_type, 16
    return model_type, 14


def use_default_system_message(model_type: str) -> bool:
    """Identify Qwen models that use the default system message."""
    return model_type in {"qwen2_vl", "qwen2_5_vl"}


def get_image_info(
    image_path: str,
    min_pixel: int,
    max_pixel: int,
    width: int | None,
    height: int | None,
    image_patch_size: int,
) -> Image.Image:
    """Load and resize an image using Qwen's image preparation utility."""
    content = {
        "type": "image",
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel,
    }
    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height

    messages = [{"role": "user", "content": [content]}]
    image_input, _ = process_vision_info(
        messages, image_patch_size=image_patch_size
    )
    return image_input[0]
