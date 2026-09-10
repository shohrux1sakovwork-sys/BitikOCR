"""Prepare image/transcription examples and padded Qwen OCR batches."""

import json
import os

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from bitikocr.data.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    SYSTEM_MESSAGE,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
)
from bitikocr.data.data_utils import (
    get_image_info,
    get_mm_token_type_ids,
    get_qwen_multimodal_settings,
    use_default_system_message,
)
from bitikocr.params import DataArguments


class SupervisedDataset(Dataset):
    """Prepare local OCR records with one ``image`` path and ``text`` string.

    Accept a JSON array, JSONL file, or Python list. Optional metadata stays
    in the original records and is not passed to the model.
    """

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id: str,
    ) -> None:
        """Load records and prepare the fixed OCR instruction."""
        super().__init__()
        if isinstance(data_path, str):
            with open(data_path, encoding="utf-8") as data_file:
                contents = data_file.read()
            if contents.lstrip().startswith("["):
                self.list_data_dict = json.loads(contents)
            else:
                self.list_data_dict = [
                    json.loads(line)
                    for line in contents.splitlines()
                    if line.strip()
                ]
        else:
            self.list_data_dict = data_path

        self.processor = processor
        self.data_args = data_args
        model_type, self.image_patch_size = get_qwen_multimodal_settings(
            model_id
        )
        self.prompt = ""
        if SYSTEM_MESSAGE and use_default_system_message(model_type):
            self.prompt += (
                f"{DEFAULT_IM_START_TOKEN}system\n"
                f"{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
            )
        self.prompt += (
            f"{DEFAULT_IM_START_TOKEN}user\n"
            f"{VISION_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{VISION_END_TOKEN}"
            f"Read the text in this image.{DEFAULT_IM_END_TOKEN}\n"
            f"{DEFAULT_IM_START_TOKEN}assistant\n"
        )
        # Qwen3.5 expects a closed thinking block for a direct answer.
        if model_type in {"qwen3_5", "qwen3_5_moe"}:
            self.prompt += "<think>\n\n</think>\n\n"

    def __len__(self) -> int:
        """Return the number of OCR records."""
        return len(self.list_data_dict)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """Load one image and supervise only its transcription and ending."""
        record = self.list_data_dict[i]
        if (
            not isinstance(record, dict)
            or not isinstance(record.get("image"), str)
            or not record["image"]
            or not isinstance(record.get("text"), str)
        ):
            raise ValueError(
                f"OCR record {i} requires an 'image' path and a 'text' string."
            )

        image_path = record["image"]
        if self.data_args.image_folder and not os.path.isabs(image_path):
            image_path = os.path.join(self.data_args.image_folder, image_path)
        image = get_image_info(
            image_path,
            self.data_args.image_min_pixels,
            self.data_args.image_max_pixels,
            self.data_args.image_resized_width,
            self.data_args.image_resized_height,
            self.image_patch_size,
        )
        inputs = self.processor(
            text=[self.prompt],
            images=[image],
            padding=False,
            do_resize=False,
            return_tensors="pt",
        )
        prompt_ids = inputs["input_ids"].squeeze(0)
        answer_ids = self.processor.tokenizer(
            f"{record['text']}{DEFAULT_IM_END_TOKEN}\n",
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].squeeze(0)
        input_ids = torch.cat([prompt_ids, answer_ids])
        labels = torch.cat(
            [torch.full_like(prompt_ids, IGNORE_INDEX), answer_ids]
        )
        prompt_types = get_mm_token_type_ids(
            inputs,
            inputs["input_ids"],
            self.processor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN),
        ).squeeze(0)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
            "mm_token_type_ids": torch.cat(
                [prompt_types, torch.zeros_like(answer_ids)]
            ),
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"],
        }


class DataCollatorForSupervisedDataset:
    """Pad OCR sequences and combine their image tensors into a batch."""

    def __init__(self, pad_token_id: int) -> None:
        """Store the tokenizer's padding token ID."""
        self.pad_token_id = pad_token_id

    def __call__(
        self, examples: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Pad text fields on the right and concatenate image patches."""
        input_ids = pad_sequence(
            [example["input_ids"] for example in examples],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        return {
            "input_ids": input_ids,
            "labels": pad_sequence(
                [example["labels"] for example in examples],
                batch_first=True,
                padding_value=IGNORE_INDEX,
            ),
            "attention_mask": pad_sequence(
                [example["attention_mask"] for example in examples],
                batch_first=True,
                padding_value=0,
            ),
            "mm_token_type_ids": pad_sequence(
                [example["mm_token_type_ids"] for example in examples],
                batch_first=True,
                padding_value=0,
            ),
            "pixel_values": torch.cat(
                [example["pixel_values"] for example in examples]
            ),
            "image_grid_thw": torch.cat(
                [example["image_grid_thw"] for example in examples]
            ),
        }


def make_supervised_data_module(
    model_id: str,
    processor: transformers.ProcessorMixin,
    data_args: DataArguments,
) -> dict:
    """Return ``dataset`` and ``data_collator`` for an OCR data loader."""
    if data_args.data_path is None:
        raise ValueError("Set data_path to the OCR training data file.")
    dataset = SupervisedDataset(
        data_args.data_path, processor, data_args, model_id
    )
    return {
        "dataset": dataset,
        "data_collator": DataCollatorForSupervisedDataset(
            processor.tokenizer.pad_token_id
        ),
    }
