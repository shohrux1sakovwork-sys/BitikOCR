"""Verify split isolation, experiment ranking, and trainable visual adapters."""

import json
import random
from pathlib import Path

import pytest
import torch
from peft import PeftModel
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from bitikocr.params import LoraArguments
from bitikocr.train.adapters import configure_adapters, visual_digest
from scripts.prepare_ablation_data import (
    grouped_records,
    prepare,
    sample_groups,
)
from scripts.run_ablation import (
    candidate,
    candidate_key,
    load_config,
    read_metrics,
    select_best,
)


def test_transitive_duplicate_groups() -> None:
    """Keep transitively connected records together regardless of key type."""
    records = [
        {"id": "a", "image": "1"},
        {"id": "a", "image": "2"},
        {"id": "b", "image": "2"},
        {"id": "c", "image": "3"},
    ]
    groups = grouped_records(records, ["id", "image"])
    assert sorted(map(len, groups)) == [1, 3]
    selected, remaining = sample_groups(groups, 3, random.Random(42))
    assert len(selected) == 1 and len(selected[0]) == 3
    assert len(remaining[0]) == 1


def test_split_reserved_isolation_and_repeatability(tmp_path: Path) -> None:
    """Reject held-out overlap and reproduce exact selections and manifests."""
    config = load_config(Path("ablation.yaml"))
    records = []
    for index in range(40):
        image = tmp_path / f"{index}.png"
        image.touch()
        records.append(
            {
                "id": str(index),
                "image": str(image),
                "text": "text",
                "document_type": "a" if index < 20 else "b",
                "script": "latin",
            }
        )
    source = tmp_path / "source.jsonl"
    reserved = tmp_path / "reserved.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in records))
    reserved.write_text(
        json.dumps({"id": "held-out", "image": "reserved.png"}) + "\n"
    )
    config["data"].update(
        source_train=str(source),
        reserved_eval=str(reserved),
        train_path=str(tmp_path / "train.jsonl"),
        development_path=str(tmp_path / "dev.jsonl"),
        manifest_path=str(tmp_path / "manifest.json"),
        train_samples=16,
        development_samples=4,
    )
    first = prepare(config)
    assert first == prepare(config)
    assert set(first["splits"]["train"]["ids"]).isdisjoint(
        first["splits"]["development"]["ids"]
    )
    reserved.write_text(json.dumps(records[0]) + "\n")
    with pytest.raises(ValueError, match="overlaps"):
        prepare(config)


def test_ranking_excludes_failures_and_nonfinite() -> None:
    """Rank scores above one correctly while excluding invalid candidates."""
    good = {
        "status": "completed",
        "metrics": {"eval_cer": 1.2, "eval_wer": 1.4},
        "resources": {"trainable_parameters": 10},
        "name": "good",
    }
    bad = {**good, "metrics": {"eval_cer": float("nan"), "eval_wer": 0}}
    assert select_best([{"status": "failed"}, bad, good]) is good
    with pytest.raises(ValueError, match="No successful"):
        select_best([bad])
    assert candidate_key(candidate("A", 16, 0.00002)) != candidate_key(
        candidate("A", 16, 0.00002, seed=43)
    )


def test_metrics_require_complete_artifacts(tmp_path: Path) -> None:
    """Reject incomplete predictions even if scalar scores are valid."""
    (tmp_path / "eval_results.json").write_text(
        json.dumps({"eval_cer": 1.1, "eval_wer": 2.0, "eval_samples": 2})
    )
    (tmp_path / "predictions.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="incomplete"):
        read_metrics(tmp_path, 2)
    (tmp_path / "predictions.jsonl").write_text("{}\n{}\n")
    assert read_metrics(tmp_path, 2)["eval_cer"] == 1.1


def tiny_model() -> Qwen2_5_VLForConditionalGeneration:
    """Build the actual Qwen architecture with tiny random CPU weights."""
    config = Qwen2_5_VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "out_hidden_size": 32,
        },
    )
    return Qwen2_5_VLForConditionalGeneration(config)


@pytest.mark.parametrize("freeze", [True, False])
def test_visual_training_and_checkpoint_roundtrip(
    tmp_path: Path, freeze: bool
) -> None:
    """Verify real PEFT wrapping, visual updates, and restored visual weights."""
    model = configure_adapters(
        tiny_model(), LoraArguments(freeze_vision_encoder=freeze)
    )
    visual = [
        p
        for n, p in model.named_parameters()
        if ".visual." in n and p.requires_grad
    ]
    assert bool(visual) != freeze
    assert all("visual" not in name for name in model.ocr_target_modules)
    if visual:
        optimizer = torch.optim.AdamW(visual, lr=0.1)
        before = visual[0].detach().clone()
        visual[0].square().sum().backward()
        assert visual[0].grad.abs().sum() > 0
        optimizer.step()
        assert not torch.equal(before, visual[0])
    model.save_pretrained(tmp_path)
    restored = PeftModel.from_pretrained(tiny_model(), tmp_path)
    assert visual_digest(model) == visual_digest(restored)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lora_r": 0},
        {"lora_alpha": -1},
        {"lora_dropout": 1.0},
        {"lora_target_modules": ""},
        {"lora_target_modules": "lm_head"},
    ],
)
def test_invalid_lora_settings(kwargs) -> None:
    """Fail bad adapter settings before allocating a model."""
    with pytest.raises(ValueError):
        LoraArguments(**kwargs)


def test_generated_arguments_parse_with_installed_trainer(
    tmp_path: Path,
) -> None:
    """Catch version-specific argument changes before a costly GPU run."""
    from transformers import HfArgumentParser, TrainingArguments

    from bitikocr.params import (
        DataArguments,
        ExperimentArguments,
        GenerationArguments,
        ModelArguments,
    )
    from scripts.run_ablation import Study

    study = Study.__new__(Study)
    study.config = load_config(Path("ablation.yaml"))
    args = study.arguments(candidate("C", 64, 0.00002), tmp_path, "test")
    args.update(use_cpu=True, bf16=False, report_to="none")
    parsed = HfArgumentParser(
        [
            ModelArguments,
            DataArguments,
            GenerationArguments,
            LoraArguments,
            ExperimentArguments,
            TrainingArguments,
        ]
    ).parse_dict(args)
    assert parsed[-1].get_warmup_steps(200) == 0
    assert parsed[-2].final_evaluation
