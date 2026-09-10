"""Configure model selection and supervised OCR data preparation."""

from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    """Select the pretrained vision-language model."""

    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"


@dataclass
class DataArguments:
    """Configure OCR data files and image preparation."""

    data_path: str | None = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    image_folder: str | None = None
    image_min_pixels: int = 3136
    image_max_pixels: int = 12845056
    image_resized_width: int | None = None
    image_resized_height: int | None = None
