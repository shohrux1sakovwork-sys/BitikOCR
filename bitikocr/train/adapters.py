"""Build decoder adapters and verify optional visual weight updates."""

import hashlib
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback

from bitikocr.params import LoraArguments

#: Vision-encoder layers that take LoRA: each block's attention and MLP, and
#: the merger's. The patch embedding is a convolution and stays frozen.
VISION_LORA_SUFFIXES = frozenset({"qkv", "proj", "linear_fc1", "linear_fc2"})

#: Dtypes too coarse to train full weights in: a learning-rate-sized step is
#: rounded away (see issue #12).
_LOW_PRECISION = (torch.bfloat16, torch.float16)


def configure_adapters(model: Any, args: LoraArguments) -> Any:
    """Resolve decoder targets, and vision targets when the encoder trains.

    With ``vision_lora_rank`` set, the vision encoder's attention, MLP and
    merger layers get LoRA of that rank beside the decoder's. PEFT keeps
    adapter weights in float32, so their updates survive a bfloat16 model.

    Raises:
        ValueError: If a target cannot be resolved, or the whole vision
            encoder would be trained in bfloat16 or float16, where its
            updates are rounded away.
    """
    suffixes = {name.strip() for name in args.lora_target_modules.split(",")}
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.rsplit(".", 1)[-1] in suffixes
        and "visual" not in name.split(".")
        and "language_model" in name.split(".")
    ]
    if {name.rsplit(".", 1)[-1] for name in targets} != suffixes:
        raise ValueError(
            "Cannot resolve all requested language decoder projections."
        )
    visual = [
        name
        for name, _ in model.named_modules()
        if name.split(".")[-1] == "visual"
    ]
    if len(visual) != 1:
        raise ValueError(
            "Expected exactly one visual backbone including merger."
        )
    if not args.freeze_vision_encoder and _is_low_precision(model):
        raise ValueError(
            "Training the whole vision encoder in bfloat16 or float16 loses "
            "its updates to rounding (issue #12); set vision_lora_rank "
            "instead."
        )
    vision_targets = _vision_targets(model) if args.vision_lora_rank else []
    if args.vision_lora_rank and not vision_targets:
        raise ValueError("Cannot resolve the vision encoder's LoRA layers.")
    vision_rank = {name: args.vision_lora_rank for name in vision_targets}
    vision_alpha = {name: 2 * args.vision_lora_rank for name in vision_targets}
    result: Any = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[*targets, *vision_targets],
            rank_pattern=vision_rank,
            alpha_pattern=vision_alpha,
            modules_to_save=None if args.freeze_vision_encoder else visual,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    vision_trainable = [
        name
        for name, p in result.named_parameters()
        if ".visual." in name and p.requires_grad
    ]
    trains_vision = not args.freeze_vision_encoder or args.vision_lora_rank > 0
    if bool(vision_trainable) != trains_vision:
        raise ValueError(
            "Visual trainability does not match the requested setting."
        )
    result.ocr_target_modules = [*targets, *vision_targets]
    return result


def _vision_targets(model: Any) -> list[str]:
    """Return the vision encoder's linear layers that take LoRA."""
    return [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and "visual" in name.split(".")
        and name.rsplit(".", 1)[-1] in VISION_LORA_SUFFIXES
    ]


def _is_low_precision(model: Any) -> bool:
    """Whether the vision encoder's weights are bfloat16 or float16."""
    return any(
        parameter.dtype in _LOW_PRECISION
        for name, parameter in model.named_parameters()
        if "visual" in name.split(".")
    )


def visual_digest(model: Any) -> str:
    """Hash the active trained visual parameters for fresh-process reloads."""
    digest = hashlib.sha256()
    found = False
    for name, parameter in model.named_parameters():
        if ".visual." in name and (
            ".modules_to_save.default." in name or ".lora_" in name
        ):
            digest.update(name.encode())
            digest.update(
                parameter.detach()
                .cpu()
                .contiguous()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
            found = True
    return digest.hexdigest() if found else ""


class VisionUpdateCheck(TrainerCallback):
    """Require a nonzero visual gradient and an actual optimizer update."""

    def __init__(self) -> None:
        """Initialize observations for a short resource pilot."""
        self.observed_gradient = False
        self.observed_update = False
        self.parameter: torch.Tensor | None = None
        self.index = 0
        self.before = 0.0

    def on_pre_optimizer_step(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        """Inspect a visual gradient already included in the optimizer."""
        optimizer_ids = {
            id(p)
            for group in kwargs["optimizer"].param_groups
            for p in group["params"]
        }
        for name, parameter in kwargs["model"].named_parameters():
            if ".visual." not in name or parameter.grad is None:
                continue
            if id(parameter) not in optimizer_ids:
                raise ValueError("Visual parameter is absent from optimizer.")
            flat = parameter.grad.detach().flatten()
            index = int(flat.abs().argmax())
            if flat[index].isfinite() and flat[index].abs() > 0:
                self.observed_gradient = True
                self.parameter = parameter
                self.index = index
                self.before = float(parameter.detach().flatten()[index])
                break

    def on_step_end(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        """Confirm an optimizer step changed the observed visual weight."""
        if self.parameter is not None:
            self.observed_update |= self.before != float(
                self.parameter.detach().flatten()[self.index]
            )
