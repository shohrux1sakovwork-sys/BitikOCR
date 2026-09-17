"""Exercise Trainer evaluation and training transitions without model weights."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    TrainerCallback,
    TrainingArguments,
)

from bitikocr.data.sft_dataset import DataCollatorForSupervisedDataset
from bitikocr.train.train import QwenOCRTrainer, _validate_execution


class GenerationModel(torch.nn.Module):
    """Return predetermined transcriptions while recording generation inputs."""

    def __init__(self) -> None:
        """Create one trainable scalar for the offline training smoke test."""
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.calls: list[dict[str, Any]] = []
        self.fail_generation = False
        self.training_modes: list[bool] = []

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        mm_token_type_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        labels=None,
    ) -> dict[str, torch.Tensor]:
        """Expose model fields to Trainer and produce a differentiable loss."""
        self.training_modes.append(self.training)
        return {"loss": self.weight.square().sum()}

    def generate(self, **kwargs: Any) -> torch.Tensor:
        """Check multimodal inputs and append sample-specific answer tokens."""
        assert not self.training
        assert not torch.is_grad_enabled()
        assert "reference_text" not in kwargs
        assert "labels" not in kwargs
        assert kwargs["do_sample"] is False
        assert kwargs["num_beams"] == 1
        assert kwargs["max_new_tokens"] == 8
        self.calls.append(kwargs)
        if self.fail_generation:
            raise RuntimeError("generation failed")
        inputs = kwargs["input_ids"]
        masks = kwargs["attention_mask"]
        modality = kwargs["mm_token_type_ids"]
        assert inputs.shape == masks.shape == modality.shape
        assert torch.equal(modality, (inputs == 7).long())
        assert kwargs["pixel_values"].shape[0] == int(
            kwargs["image_grid_thw"].prod(dim=1).sum()
        )
        endings = {10: [2, 1, 0], 11: [3, 5, 1], 12: [1, 0, 0]}
        answers = [
            endings[int(row[mask.bool()][0])]
            for row, mask in zip(inputs, masks)
        ]
        return torch.cat([inputs, inputs.new_tensor(answers)], dim=1)


class EvaluationCallback(TrainerCallback):
    """Record callback metrics and optimizer progress."""

    def __init__(self) -> None:
        """Initialize observations for both evaluation and training tests."""
        self.metrics: list[dict[str, float]] = []
        self.weights: list[float] = []

    def on_evaluate(self, args, state, control, **kwargs):
        """Verify the custom evaluator delivers metrics to callbacks."""
        self.metrics.append(kwargs["metrics"].copy())
        control.should_log = True
        return control

    def on_step_end(self, args, state, control, **kwargs):
        """Record weights after each optimizer update."""
        self.weights.append(kwargs["model"].weight.item())
        return control


@pytest.fixture
def examples() -> list[dict[str, Any]]:
    """Supply unequal prompts, distinct images, and a blank reference."""
    result = []
    for index, (ids, text, width) in enumerate(
        [([10, 7, 7, 0], "a", 2), ([11, 7], "b c", 4), ([12, 7, 0], "", 2)]
    ):
        result.append(
            {
                "input_ids": torch.tensor(ids),
                "attention_mask": torch.ones(len(ids), dtype=torch.long),
                "mm_token_type_ids": (torch.tensor(ids) == 7).long(),
                "pixel_values": torch.full((2 * width, 3), float(index + 1)),
                "image_grid_thw": torch.tensor([[1, 2, width]]),
                "reference_text": text,
            }
        )
    return result


@pytest.fixture
def trainer(tmp_path: Path, examples: list[dict[str, Any]]) -> QwenOCRTrainer:
    """Build an actual CPU Trainer with a local tokenizer and stub model."""
    backend = Tokenizer(
        WordLevel({"[PAD]": 0, "[EOS]": 1, "a": 2, "b": 3, "c": 4, "d": 5})
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]"
    )
    return QwenOCRTrainer(
        model=GenerationModel(),
        processing_class=tokenizer,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            use_cpu=True,
            report_to="none",
            disable_tqdm=True,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=2,
            dataloader_pin_memory=False,
            dataloader_drop_last=True,
        ),
        eval_dataset=examples,
        data_collator=DataCollatorForSupervisedDataset(0),
        eval_data_collator=DataCollatorForSupervisedDataset(0, is_eval=True),
        max_new_tokens=8,
    )


def test_evaluate_preserves_metadata_and_callbacks(trainer) -> None:
    """Exercise the real loader with column filtering enabled in arguments."""
    callback = EvaluationCallback()
    trainer.add_callback(callback)
    assert trainer.args.remove_unused_columns
    assert not trainer.data_collator.is_eval
    metrics = trainer.evaluate(metric_key_prefix="validation")
    assert metrics == pytest.approx(
        {"validation_cer": 1 / 4, "validation_wer": 1 / 3}
    )
    assert callback.metrics == [metrics]
    assert trainer.control.should_log
    assert (
        trainer.state.log_history[-1]["validation_cer"]
        == metrics["validation_cer"]
    )
    assert trainer.model.training
    assert len(trainer.model.calls) == 2
    batch = trainer.model.calls[0]
    assert batch["input_ids"].tolist() == [[10, 7, 7, 0], [0, 0, 11, 7]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [0, 0, 1, 1]]
    assert torch.all(batch["pixel_values"][:4] == 1)
    assert torch.all(batch["pixel_values"][4:] == 2)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_metrics_independent_of_batch_size(trainer, batch_size) -> None:
    """Compute corpus scores across all examples, including partial batches."""
    trainer.args.per_device_eval_batch_size = batch_size
    assert trainer.evaluate() == pytest.approx(
        {"eval_cer": 1 / 4, "eval_wer": 1 / 3}
    )


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_restore_model_mode(trainer, training, fail) -> None:
    """Restore the previous model mode even after a generation exception."""
    trainer.model.train(training)
    trainer.model.fail_generation = fail
    if fail:
        with pytest.raises(RuntimeError, match="generation failed"):
            trainer.evaluate()
    else:
        trainer.evaluate()
    assert trainer.model.training == training


def test_eval_override_and_validation(trainer, examples) -> None:
    """Use dataset overrides and reject absent data or evaluation collators."""
    assert trainer.evaluate(examples[:1]) == {"eval_cer": 0, "eval_wer": 0}
    with pytest.raises(ValueError, match="empty"):
        trainer.evaluate([])
    trainer.eval_dataset = None
    with pytest.raises(ValueError, match="nonempty"):
        trainer.evaluate()
    trainer.eval_data_collator = None
    with pytest.raises(ValueError, match="eval_data_collator"):
        trainer.evaluate(examples)


@pytest.mark.parametrize(
    "override",
    [
        {"world_size": 2},
        {"n_gpu": 2},
        {"fsdp": ["full_shard"]},
        {"deepspeed": "config.json"},
    ],
)
def test_reject_unsupported_execution(override) -> None:
    """Fail explicitly before silently computing metrics on one shard."""
    settings = {"world_size": 1, "n_gpu": 1, "fsdp": [], "deepspeed": None}
    settings.update(override)
    with pytest.raises(ValueError, match="one process on one device"):
        _validate_execution(SimpleNamespace(**settings))


def test_training_continues_after_evaluation(trainer, examples) -> None:
    """Run two optimizer steps with generation evaluation after each step."""
    trainer.train_dataset = [
        {
            key: value
            for key, value in example.items()
            if key != "reference_text"
        }
        | {"labels": example["input_ids"].clone()}
        for example in examples
    ]
    trainer.args.max_steps = 2
    trainer.args.per_device_train_batch_size = 1
    trainer.args.eval_strategy = "steps"
    trainer.args.eval_steps = 1
    trainer.args.save_strategy = "no"
    trainer.args.learning_rate = 0.1
    callback = EvaluationCallback()
    trainer.add_callback(callback)
    trainer.train()
    assert len(callback.metrics) == 2
    assert len(callback.weights) == 2
    assert callback.weights[0] < 1
    assert callback.weights[1] < callback.weights[0]
    assert all(trainer.model.training_modes)
