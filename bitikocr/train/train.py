"""Train Qwen2.5-VL LoRA adapters and evaluate generated OCR text."""

import json
import os
import sys
import time
from collections.abc import Callable, Sized
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.hf_argparser import DataClassType

from bitikocr.data.sft_dataset import make_supervised_data_module
from bitikocr.params import (
    DataArguments,
    ExperimentArguments,
    GenerationArguments,
    LoraArguments,
    ModelArguments,
)
from bitikocr.train.adapters import (
    VisionUpdateCheck,
    configure_adapters,
    visual_digest,
)
from bitikocr.train.metrics import compute_ocr_metrics


def _validate_execution(args: TrainingArguments) -> None:
    if args.world_size != 1 or args.n_gpu > 1 or args.fsdp or args.deepspeed:
        raise ValueError(
            "OCR generation evaluation supports one process on one device; "
            "distributed, DataParallel, FSDP, and DeepSpeed runs are unsupported."
        )


class QwenOCRTrainer(Trainer):
    """Train adapters and evaluate OCR with a separate generation collator."""

    def __init__(
        self,
        *args: Any,
        eval_data_collator: Callable | None = None,
        max_new_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        """Configure single-device evaluation and its generation token budget."""
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        super().__init__(*args, **kwargs)
        _validate_execution(self.args)
        self.eval_data_collator = eval_data_collator
        self.max_new_tokens = max_new_tokens
        self.prediction_rows: list[dict[str, Any]] = []
        self.generation_limit_hits: list[bool] = []

    def get_eval_dataloader(
        self, eval_dataset: str | Dataset | None = None
    ) -> DataLoader:
        """Keep reference metadata and use left-padding for every eval batch."""
        _validate_execution(self.args)
        dataset = (
            eval_dataset if eval_dataset is not None else self.eval_dataset
        )
        if dataset is None or isinstance(dataset, (str, dict)):
            raise ValueError("Provide one nonempty OCR evaluation dataset.")
        if not isinstance(dataset, Sized):
            raise TypeError("OCR evaluation requires a dataset with a length.")
        if len(dataset) == 0:
            raise ValueError("The OCR evaluation dataset must not be empty.")
        if self.eval_data_collator is None:
            raise ValueError("Provide eval_data_collator for OCR evaluation.")
        # Trainer's standard loader uses the training collator and removes
        # reference_text because it is not a model.forward parameter.
        return DataLoader(
            cast(Dataset, dataset),
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.eval_data_collator,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _get_tokenizer(self) -> PreTrainedTokenizerBase:
        processor = self.processing_class
        if isinstance(processor, ProcessorMixin):
            processor = processor.tokenizer
        if not isinstance(processor, PreTrainedTokenizerBase):
            raise TypeError("OCR evaluation requires a tokenizer or processor.")
        if processor.pad_token_id is None:
            raise ValueError("OCR evaluation requires a tokenizer pad token.")
        return processor

    def _generate_predictions(
        self, batch: dict[str, Any], tokenizer: PreTrainedTokenizerBase
    ) -> list[str]:
        inputs = self._prepare_inputs(
            {
                key: batch[key]
                for key in (
                    "input_ids",
                    "attention_mask",
                    "mm_token_type_ids",
                    "pixel_values",
                    "image_grid_thw",
                )
            }
        )
        # Trainer also accepts ordinary torch modules; the OCR model must
        # additionally implement the Transformers generate interface.
        model = cast(Any, self.model)
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
        output_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
        for row in output_ids:
            eos = tokenizer.eos_token_id
            self.generation_limit_hits.append(
                len(row) >= self.max_new_tokens
                and (eos is None or not bool((row == eos).any()))
            )
        return tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def evaluate(
        self,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Generate transcriptions, report corpus metrics, and notify callbacks."""
        if isinstance(eval_dataset, dict):
            raise TypeError(
                "Provide one OCR evaluation dataset, not a mapping."
            )
        dataloader = self.get_eval_dataloader(eval_dataset)
        tokenizer = self._get_tokenizer()
        predictions: list[str] = []
        self.generation_limit_hits = []
        references: list[str] = []
        model = self.model
        if model is None:
            raise ValueError("OCR evaluation requires a model.")
        was_training = model.training
        model.eval()
        try:
            with torch.inference_mode():
                for batch in dataloader:
                    predictions.extend(
                        self._generate_predictions(batch, tokenizer)
                    )
                    references.extend(batch["reference_text"])
        finally:
            model.train(was_training)

        self.prediction_rows = [
            {
                "prediction": prediction,
                "reference": reference,
                "generation_limit_hit": hit,
            }
            for prediction, reference, hit in zip(
                predictions, references, self.generation_limit_hits
            )
        ]
        metrics = {
            f"{metric_key_prefix}_{key}": value
            for key, value in compute_ocr_metrics(
                predictions, references
            ).items()
        }
        self.log(metrics.copy())
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics


def train() -> None:
    """Parse arguments, train a LoRA adapter, and save it with its processor."""
    parser = HfArgumentParser(
        [
            DataClassType(ModelArguments),
            DataClassType(DataArguments),
            DataClassType(GenerationArguments),
            DataClassType(LoraArguments),
            DataClassType(ExperimentArguments),
            DataClassType(TrainingArguments),
        ]
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        parsed = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        parsed = parser.parse_args_into_dataclasses()
    (
        model_args,
        data_args,
        generation_args,
        lora_args,
        experiment_args,
        training_args,
    ) = parsed
    set_seed(training_args.seed)
    started = time.monotonic()
    _validate_execution(training_args)
    if generation_args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if (
        training_args.eval_strategy != "no"
        or experiment_args.final_evaluation
        or experiment_args.evaluation_only
    ) and not data_args.eval_path:
        raise ValueError("Set eval_path when enabling evaluation.")

    processor = AutoProcessor.from_pretrained(
        model_args.model_id, revision=experiment_args.model_revision
    )
    wandb_run = None
    if (
        experiment_args.resolved_config_path
        and "wandb" in training_args.report_to
    ):
        import yaml

        import wandb

        resolved = yaml.safe_load(
            Path(experiment_args.resolved_config_path).read_text()
        )
        wandb_run = wandb.init(
            project=os.environ["WANDB_PROJECT"],
            entity=os.environ.get("WANDB_ENTITY"),
            group=os.environ.get("WANDB_RUN_GROUP"),
            name=training_args.run_name,
            config=resolved,
            dir=training_args.output_dir,
            settings=wandb.Settings(init_timeout=120),
        )
        if wandb_run is not None:
            Path(training_args.output_dir, "wandb_url.txt").write_text(
                wandb_run.url or ""
            )
    data_module = make_supervised_data_module(
        model_args.model_id, processor, data_args
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_id,
        revision=experiment_args.model_revision,
        dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            **(training_args.gradient_checkpointing_kwargs or {}),
            "use_reentrant": False,
        }
    peft_model: Any
    if experiment_args.adapter_path:
        peft_model = PeftModel.from_pretrained(
            model, experiment_args.adapter_path
        )
    elif experiment_args.evaluation_only:
        peft_model = model
    else:
        peft_model = configure_adapters(model, lora_args)
    output = Path(training_args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    update_check = (
        VisionUpdateCheck() if experiment_args.verify_vision_update else None
    )
    trainer = QwenOCRTrainer(
        model=peft_model,
        processing_class=processor,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module.get("eval_dataset"),
        data_collator=data_module["data_collator"],
        eval_data_collator=data_module.get("eval_data_collator"),
        max_new_tokens=generation_args.max_new_tokens,
        callbacks=[update_check] if update_check is not None else [],
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_metrics = {}
    if not experiment_args.evaluation_only:
        result = trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint
        )
        train_metrics = result.metrics
        trainer.save_metrics("train", train_metrics)
        trainer.save_state()
        trainer.save_model(training_args.output_dir)
        processor.save_pretrained(training_args.output_dir)
    metrics = {}
    if experiment_args.final_evaluation or experiment_args.evaluation_only:
        metrics = trainer.evaluate()
        rows = trainer.prediction_rows
        dataset = data_module["eval_dataset"]
        for row, record in zip(rows, dataset.list_data_dict):
            row.update(
                {
                    key: record.get(key)
                    for key in ("id", "document_type", "script")
                }
            )
        metrics["eval_samples"] = len(rows)
        metrics["eval_generation_limit_hits"] = sum(
            row["generation_limit_hit"] for row in rows
        )
        for category in ("document_type", "script"):
            for value in sorted({str(row[category]) for row in rows}):
                subset = [row for row in rows if str(row[category]) == value]
                for key, score in compute_ocr_metrics(
                    [row["prediction"] for row in subset],
                    [row["reference"] for row in subset],
                ).items():
                    metrics[f"eval_{category}_{value}_{key}"] = score
        trainer.save_metrics("eval", metrics)
        (output / "predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
        if wandb_run is not None:
            wandb_run.log(metrics)
    resources = {
        "wall_seconds": time.monotonic() - started,
        "trainable_parameters": sum(
            p.numel() for p in peft_model.parameters() if p.requires_grad
        ),
        "total_parameters": sum(p.numel() for p in peft_model.parameters()),
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else 0
        ),
        "peak_reserved_bytes": (
            torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
        ),
        "target_modules": getattr(peft_model, "ocr_target_modules", []),
        "lora": asdict(lora_args),
        "vision_digest": (
            visual_digest(peft_model)
            if not lora_args.freeze_vision_encoder
            else ""
        ),
        "vision_gradient_verified": (
            update_check.observed_gradient if update_check else None
        ),
        "vision_update_verified": (
            update_check.observed_update if update_check else None
        ),
    }
    (output / "resources.json").write_text(json.dumps(resources, indent=2))
    if update_check and not (
        update_check.observed_gradient and update_check.observed_update
    ):
        raise ValueError(
            "Vision pilot did not verify a nonzero gradient and weight update."
        )
    if wandb_run is not None:
        wandb_run.summary.update(
            {k: v for k, v in resources.items() if k != "target_modules"}
        )
        wandb_run.finish()


if __name__ == "__main__":
    train()
