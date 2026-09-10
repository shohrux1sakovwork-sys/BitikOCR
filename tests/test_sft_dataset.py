"""Exercise OCR loading, supervision, and batching without model downloads."""

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import transformers

from bitikocr.data import sft_dataset
from bitikocr.data.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from bitikocr.params import DataArguments


class FakeTokenizer:
    """Encode text deterministically while retaining special-token metadata."""

    pad_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> int:
        """Resolve the image placeholder used by the fake processor."""
        assert token == DEFAULT_IMAGE_TOKEN
        return 91

    def __call__(self, text: str, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Return reversible character IDs for a transcription."""
        assert kwargs == {
            "add_special_tokens": False,
            "return_tensors": "pt",
        }
        return {"input_ids": torch.tensor([[ord(char) + 200 for char in text]])}


class FakeProcessor:
    """Provide deterministic multimodal prompt tensors for dataset tests."""

    def __init__(self, include_modality: bool = True) -> None:
        self.tokenizer = FakeTokenizer()
        self.include_modality = include_modality

    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Return one prompt with two expanded image placeholders."""
        assert kwargs["padding"] is False
        assert kwargs["do_resize"] is False
        assert kwargs["return_tensors"] == "pt"
        result = {
            "input_ids": torch.tensor([[11, 91, 91, 12]]),
            "pixel_values": torch.ones(8, 3),
            "image_grid_thw": torch.tensor([[1, 2, 4]]),
        }
        if self.include_modality:
            result["mm_token_type_ids"] = torch.tensor([[0, 1, 1, 0]])
        return result


def as_processor(fake: FakeProcessor) -> transformers.ProcessorMixin:
    """Hand the fake to code that is typed for a real processor.

    FakeProcessor implements only the slice of the processor interface the
    dataset actually touches, which is the whole point of it: these tests
    run without downloading a model. The cast is where that deliberate
    narrowing is declared, so it is stated once rather than at every call.
    """
    return cast(transformers.ProcessorMixin, fake)


@pytest.fixture
def processor(monkeypatch: pytest.MonkeyPatch) -> FakeProcessor:
    """Replace external model metadata and image loading for unit tests."""
    monkeypatch.setattr(
        sft_dataset,
        "get_qwen_multimodal_settings",
        lambda _: ("qwen2_5_vl", 14),
    )
    monkeypatch.setattr(sft_dataset, "get_image_info", lambda *args: object())
    return FakeProcessor()


@pytest.mark.parametrize("file_format", ["json", "jsonl", "list"])
def test_load_records(
    tmp_path: Path, processor: FakeProcessor, file_format: str
) -> None:
    """Load Unicode records and keep additional metadata out of supervision."""
    records = [
        {
            "image": "page.png",
            "text": "O‘zbekcha matn\n第二行",
            "source": "scan",
        }
    ]
    source: Any = records
    if file_format != "list":
        path = tmp_path / f"records.{file_format}"
        contents = json.dumps(records if file_format == "json" else records[0])
        path.write_text(f"\n{contents}\n\n", encoding="utf-8")
        source = str(path)
    dataset = sft_dataset.SupervisedDataset(
        source, as_processor(processor), DataArguments(), "test-model"
    )
    assert len(dataset) == 1
    assert dataset.list_data_dict == records
    example = dataset[0]
    expected_answer = processor.tokenizer(
        records[0]["text"] + "<|im_end|>\n",
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    assert torch.equal(example["labels"][:4], torch.full((4,), IGNORE_INDEX))
    assert torch.equal(example["labels"][4:], expected_answer)
    assert torch.equal(example["input_ids"][4:], expected_answer)
    assert torch.all(example["attention_mask"] == 1)
    assert example["input_ids"].shape == example["mm_token_type_ids"].shape
    assert example["mm_token_type_ids"][:4].tolist() == [0, 1, 1, 0]
    assert torch.all(example["mm_token_type_ids"][4:] == 0)


@pytest.mark.parametrize(
    "record",
    [
        None,
        42,
        [],
        {},
        {"image": "x"},
        {"image": "", "text": "x"},
        {"image": "x", "text": 3},
    ],
)
def test_invalid_record(processor: FakeProcessor, record: Any) -> None:
    """Report malformed records as validation errors instead of crashes."""
    dataset = sft_dataset.SupervisedDataset(
        [record], as_processor(processor), DataArguments(), "test-model"
    )
    with pytest.raises(ValueError, match="OCR record 0 requires"):
        dataset[0]


def test_missing_modality_ids(processor: FakeProcessor) -> None:
    """Recover image positions when processor modality output is absent."""
    processor.include_modality = False
    dataset = sft_dataset.SupervisedDataset(
        [{"image": "x", "text": ""}],
        as_processor(processor),
        DataArguments(),
        "test-model",
    )
    example = dataset[0]
    assert example["mm_token_type_ids"][:4].tolist() == [0, 1, 1, 0]
    assert torch.all(example["mm_token_type_ids"][4:] == 0)
    assert torch.any(example["labels"] != IGNORE_INDEX)


@pytest.mark.parametrize("absolute", [False, True])
def test_image_folder_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    processor: FakeProcessor,
    absolute: bool,
) -> None:
    """Honor the image root even when a same-named working-directory file exists."""
    monkeypatch.chdir(tmp_path)
    cwd_image = tmp_path / "page.png"
    cwd_image.touch()
    image_root = tmp_path / "images"
    image_root.mkdir()
    selected_paths = []
    monkeypatch.setattr(
        sft_dataset,
        "get_image_info",
        lambda path, *args: selected_paths.append(path),
    )
    image_path = str(cwd_image) if absolute else "page.png"
    dataset = sft_dataset.SupervisedDataset(
        [{"image": image_path, "text": "hello"}],
        as_processor(processor),
        DataArguments(image_folder=str(image_root)),
        "test-model",
    )
    dataset[0]
    assert selected_paths == [
        str(cwd_image if absolute else image_root / "page.png")
    ]


def test_collate_preserves_real_padding_token_and_image_order() -> None:
    """Pad by sequence length without hiding an EOS token aliased to padding."""
    examples = [
        {
            "input_ids": torch.tensor([5, 7, 2, 8]),
            "labels": torch.tensor([-100, 7, 2, 8]),
            "attention_mask": torch.ones(4, dtype=torch.long),
            "mm_token_type_ids": torch.tensor([0, 1, 0, 0]),
            "pixel_values": torch.full((4, 3), 10.0),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
        {
            "input_ids": torch.tensor([5, 2]),
            "labels": torch.tensor([-100, 2]),
            "attention_mask": torch.ones(2, dtype=torch.long),
            "mm_token_type_ids": torch.tensor([1, 0]),
            "pixel_values": torch.full((8, 3), 20.0),
            "image_grid_thw": torch.tensor([[1, 2, 4]]),
        },
    ]
    batch = sft_dataset.DataCollatorForSupervisedDataset(2)(examples)
    assert batch["input_ids"].tolist() == [[5, 7, 2, 8], [5, 2, 2, 2]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 7, 2, 8], [-100, 2, -100, -100]]
    assert batch["mm_token_type_ids"].tolist() == [[0, 1, 0, 0], [1, 0, 0, 0]]
    assert batch["image_grid_thw"].tolist() == [[1, 2, 2], [1, 2, 4]]
    assert batch["pixel_values"].shape == (12, 3)
    assert torch.all(batch["pixel_values"][:4] == 10)
    assert torch.all(batch["pixel_values"][4:] == 20)
    assert examples[1]["input_ids"].tolist() == [5, 2]


def test_data_module(tmp_path: Path, processor: FakeProcessor) -> None:
    """Return the documented dataset/collator pair and reject an absent path."""
    with pytest.raises(ValueError, match="Set data_path"):
        sft_dataset.make_supervised_data_module(
            "test-model", as_processor(processor), DataArguments()
        )
    path = tmp_path / "data.jsonl"
    path.write_text('{"image": "page.png", "text": "hello"}\n')
    module = sft_dataset.make_supervised_data_module(
        "test-model",
        as_processor(processor),
        DataArguments(data_path=str(path)),
    )
    assert set(module) == {"dataset", "data_collator"}
    batch = module["data_collator"]([module["dataset"][0]])
    assert batch["input_ids"].shape[0] == 1
