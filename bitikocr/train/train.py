"""Train Qwen2.5-VL LoRA adapters and evaluate generated OCR text."""

import os
import sys
from collections.abc import Callable, Sized
from typing import Any, cast

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from transformers.hf_argparser import DataClassType

from bitikocr.data.sft_dataset import make_supervised_data_module
from bitikocr.params import DataArguments, GenerationArguments, ModelArguments
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
            DataClassType(TrainingArguments),
        ]
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        parsed = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        parsed = parser.parse_args_into_dataclasses()
    model_args, data_args, generation_args, training_args = parsed
    _validate_execution(training_args)
    if generation_args.max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if training_args.eval_strategy != "no" and not data_args.eval_path:
        raise ValueError("Set eval_path when enabling evaluation.")

    processor = AutoProcessor.from_pretrained(model_args.model_id)
    data_module = make_supervised_data_module(
        model_args.model_id, processor, data_args
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_id, dtype=torch.bfloat16
    )
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            **(training_args.gradient_checkpointing_kwargs or {}),
            "use_reentrant": False,
        }
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    peft_model.print_trainable_parameters()
    trainer = QwenOCRTrainer(
        model=peft_model,
        processing_class=processor,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module.get("eval_dataset"),
        data_collator=data_module["data_collator"],
        eval_data_collator=data_module.get("eval_data_collator"),
        max_new_tokens=generation_args.max_new_tokens,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
