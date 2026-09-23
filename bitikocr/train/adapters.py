"""Build decoder adapters and verify optional visual weight updates."""

import hashlib
from typing import Any

import torch
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback

from bitikocr.params import LoraArguments


def configure_adapters(model: Any, args: LoraArguments) -> Any:
    """Resolve decoder-only targets and save fully trained visual weights."""
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
    result: Any = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=targets,
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
    if bool(vision_trainable) == args.freeze_vision_encoder:
        raise ValueError(
            "Visual trainability does not match the requested setting."
        )
    result.ocr_target_modules = targets
    return result


def visual_digest(model: Any) -> str:
    """Hash the active trained visual parameters for fresh-process reloads."""
    digest = hashlib.sha256()
    found = False
    for name, parameter in model.named_parameters():
        if ".visual." in name and ".modules_to_save.default." in name:
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
