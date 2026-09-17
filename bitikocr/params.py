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
    eval_path: str | None = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    image_folder: str | None = None
    eval_image_folder: str | None = None
    image_min_pixels: int = 3136
    image_max_pixels: int = 12845056
    image_resized_width: int | None = None
    image_resized_height: int | None = None


@dataclass
class GenerationArguments:
    """Configure the maximum generated transcription length for evaluation."""

    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "Maximum generated tokens per evaluation image."},
    )
