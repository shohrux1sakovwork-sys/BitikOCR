# BitikOCR

Prepare image/transcription pairs for supervised OCR training with Qwen
vision-language models. The data module loads existing local records; it does
not collect or annotate source documents.

## Setup

Use Python 3.11 or newer and install the optional training dependencies:

```bash
uv sync --locked --extra train
```

## Training data

Supply a JSON array, a JSONL file (one record per line), or a Python list to
`SupervisedDataset`. Each record must contain an image path and its exact text:

```json
{"image": "page-001.png", "text": "The transcription of this page."}
```

Additional metadata is preserved in the source records and is not sent to the
model. Relative image paths are resolved under `image_folder` when configured,
otherwise under the working directory. Absolute paths are used directly.

```python
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from bitikocr.data.sft_dataset import make_supervised_data_module
from bitikocr.params import DataArguments, ModelArguments

model_id = ModelArguments().model_id
processor = AutoProcessor.from_pretrained(model_id)
data_args = DataArguments(
    data_path="sample_data/train.jsonl",
    image_folder="sample_data/images",
)
data_module = make_supervised_data_module(model_id, processor, data_args)
loader = DataLoader(
    dataset=data_module["dataset"],
    collate_fn=data_module["data_collator"],
    batch_size=2,
    shuffle=True,
)
batch = next(iter(loader))
```

The default model is `Qwen/Qwen2.5-VL-7B-Instruct`. Loading its processor and
configuration may download files from Hugging Face on first use. This example
prepares a batch without loading model weights or starting training.

## What is the data collator?

`SupervisedDataset` prepares one image and its transcription at a time.
`DataCollatorForSupervisedDataset` combines those examples into a batch:

- Pads token sequences on the right to the longest sequence in that batch.
- Pads attention masks with zero, preserving every real token's visibility.
- Uses `-100` labels for the prompt and padding so only the transcription and
  answer ending contribute to the training loss.
- Preserves modality IDs and concatenates image patches and image grids in
  the same order as the text examples.

For example, sequences with lengths 120 and 90 produce a `[2, 120]` token tensor.
The second sequence receives 30 padding positions with attention mask `0` and
label `-100`. Image patches are concatenated because different image sizes can
produce different patch counts. See the
[Transformers data collator documentation](https://huggingface.co/docs/transformers/main_classes/data_collator)
for the general batching concept.

`make_supervised_data_module` returns `dataset` and `data_collator`. When using
Transformers `Trainer`, pass these explicitly as `train_dataset` and
`data_collator`. No evaluation split or training loop is provided yet. Sequence
length is not truncated; choose image limits and batch sizes for the model's
context window and available memory. Sample data and notebooks stay local.

## Development checks

```bash
make checks
make test
```

`make checks` runs Black, isort, Ruff, and mypy. `make test` runs the offline
regression suite; it requires the training extra but no pretrained weights.
