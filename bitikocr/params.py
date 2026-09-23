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


@dataclass
class LoraArguments:
    """Configure decoder adapters and optional visual model training."""

    lora_r: int = 16
    lora_alpha: int = 32
    lora_target_modules: str = "q_proj,v_proj"
    lora_dropout: float = 0.0
    freeze_vision_encoder: bool = True

    def __post_init__(self) -> None:
        """Reject invalid adapter configurations before loading weights."""
        allowed = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        targets = [item.strip() for item in self.lora_target_modules.split(",")]
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1).")
        if not targets or not set(targets) <= allowed:
            raise ValueError(
                "Specify nonempty supported decoder projection names."
            )


@dataclass
class ExperimentArguments:
    """Control final evaluation, baseline execution, and study artifacts."""

    model_revision: str | None = None
    evaluation_only: bool = False
    final_evaluation: bool = False
    resolved_config_path: str | None = None
    adapter_path: str | None = None
    verify_vision_update: bool = False
