"""Confirm the Stage 3 finalists over three independent seeds."""

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from scripts.prepare_ablation_data import read_records

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: Any) -> None:
    """Persist one JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def trainer_args(
    config: dict[str, Any], candidate: dict[str, Any], seed: int, output: Path
) -> dict[str, Any]:
    """Resolve the same screening controls with three epochs and a new seed."""
    fixed = config["fixed"]
    data = config["data"]
    return {
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "data_path": data["train_path"],
        "eval_path": data["eval_path"],
        "image_folder": data["image_folder"],
        "eval_image_folder": data["eval_image_folder"],
        "output_dir": str(output),
        "run_name": output.name,
        "seed": seed,
        "data_seed": seed,
        "num_train_epochs": config["study"]["epochs"],
        "per_device_train_batch_size": fixed["per_device_train_batch_size"],
        "per_device_eval_batch_size": fixed["per_device_eval_batch_size"],
        "gradient_accumulation_steps": fixed["gradient_accumulation_steps"],
        "bf16": fixed["bf16"],
        "gradient_checkpointing": fixed["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": fixed["optimizer"],
        "learning_rate": candidate["learning_rate"],
        "lr_scheduler_type": fixed["scheduler"],
        "warmup_steps": fixed["warmup_steps"],
        "weight_decay": fixed["weight_decay"],
        "max_grad_norm": fixed["max_grad_norm"],
        "eval_strategy": "no",
        "save_strategy": "no",
        "report_to": ["wandb"],
        "image_min_pixels": fixed["image_min_pixels"],
        "image_max_pixels": fixed["image_max_pixels"],
        "max_new_tokens": fixed["max_new_tokens"],
        "final_evaluation": True,
        "lora_r": candidate["rank"],
        "lora_alpha": candidate["alpha"],
        "lora_dropout": fixed["lora_dropout"],
        "lora_target_modules": ",".join(candidate["target_modules"]),
        "freeze_vision_encoder": fixed["freeze_vision_encoder"],
        "resolved_config_path": str(output / "resolved_config.yaml"),
        "logging_steps": 1,
        "logging_first_step": True,
    }


def complete(
    output: Path, args: dict[str, Any], expected: int
) -> dict[str, Any]:
    """Validate metrics, predictions, checkpoint, and exact run settings."""
    saved_args = json.loads((output / "trainer_args.json").read_text())
    if saved_args != args:
        raise ValueError(f"Configuration mismatch for {output}.")
    metrics = json.loads((output / "eval_results.json").read_text())
    if metrics.get("eval_samples") != expected:
        raise ValueError(f"Wrong evaluation count for {output}.")
    if len(read_records(output / "predictions.jsonl")) != expected:
        raise ValueError(f"Incomplete predictions for {output}.")
    for key in ("eval_cer", "eval_wer"):
        if not isinstance(metrics.get(key), (int, float)) or not math.isfinite(
            metrics[key]
        ):
            raise ValueError(f"Invalid {key} for {output}.")
    if not (output / "adapter_model.safetensors").is_file():
        raise ValueError(f"Missing adapter for {output}.")
    resources = json.loads((output / "resources.json").read_text())
    return {"metrics": metrics, "resources": resources}


def run_one(
    config: dict[str, Any], candidate: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Run one independent confirmation or reuse its complete artifacts."""
    if candidate["alpha"] != 2 * candidate["rank"]:
        raise ValueError("LoRA alpha must be twice the rank.")
    output = (
        ROOT
        / config["study"]["output_root"]
        / (f"p4-confirm-{candidate['id']}-s{seed}")
    )
    args = trainer_args(config, candidate, seed, output)
    expected = config["study"]["expected_eval_samples"]
    if (output / "eval_results.json").is_file():
        result = complete(output, args, expected)
        return {"name": output.name, "seed": seed, "status": "reused", **result}
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Incomplete output exists: {output}.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "trainer_args.json", args)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "phase4": config,
                "candidate": candidate,
                "seed": seed,
                "trainer": args,
            },
            sort_keys=False,
        )
    )
    command = [
        sys.executable,
        "-m",
        "bitikocr.train.train",
        str(output / "trainer_args.json"),
    ]
    write_json(output / "command.json", command)
    wandb = config["wandb"]
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=config["study"]["gpu"],
        WANDB_ENTITY=wandb["entity"],
        WANDB_PROJECT=wandb["project"],
        WANDB_RUN_GROUP=wandb["group"],
        WANDB_MODE="online",
        WANDB_TAGS=",".join(wandb["tags"] + [candidate["id"], f"seed-{seed}"]),
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
    )
    print(f"Starting {output.name}", flush=True)
    started = time.monotonic()
    with (output / "console.log").open("w") as log:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if process.returncode:
        raise RuntimeError(
            f"{output.name} exited {process.returncode}; see console.log."
        )
    result = complete(output, args, expected)
    return {
        "name": output.name,
        "seed": seed,
        "status": "completed",
        "wall_seconds": time.monotonic() - started,
        **result,
    }


def save_results(config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Save per-seed and aggregate results after all runs complete."""
    root = ROOT / config["study"]["output_root"]
    write_json(root / "run_results.json", rows)
    summary = []
    for candidate in config["candidates"]:
        selected = [row for row in rows if row["candidate"] == candidate["id"]]
        cer = [row["metrics"]["eval_cer"] for row in selected]
        wer = [row["metrics"]["eval_wer"] for row in selected]
        summary.append(
            {
                "candidate": candidate["id"],
                "source": candidate["source"],
                "rank": candidate["rank"],
                "learning_rate": candidate["learning_rate"],
                "seeds": [row["seed"] for row in selected],
                "mean_cer": statistics.mean(cer),
                "std_cer": statistics.stdev(cer),
                "mean_wer": statistics.mean(wer),
                "std_wer": statistics.stdev(wer),
                "trainable_parameters": selected[0]["resources"][
                    "trainable_parameters"
                ],
                "peak_allocated_bytes": max(
                    row["resources"]["peak_allocated_bytes"] for row in selected
                ),
            }
        )
    summary.sort(
        key=lambda row: (
            row["mean_cer"],
            row["mean_wer"],
            row["trainable_parameters"],
        )
    )
    write_json(root / "confirmation.json", summary)
    with (root / "confirmation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)
    import wandb

    wb = config["wandb"]
    run = wandb.init(
        entity=wb["entity"],
        project=wb["project"],
        group=wb["group"],
        name="p4-confirm-comparison",
        tags=wb["tags"] + ["comparison"],
        config=config,
    )
    columns: list[str | int] = list(summary[0])
    run.log(
        {
            "confirmation": wandb.Table(
                columns=columns,
                data=[
                    [
                        (
                            str(row[key])
                            if isinstance(row[key], list)
                            else row[key]
                        )
                        for key in summary[0]
                    ]
                    for row in summary
                ],
            )
        }
    )
    run.summary.update(
        {
            "best_candidate": summary[0]["candidate"],
            "best_mean_cer": summary[0]["mean_cer"],
        }
    )
    run.finish()


def main() -> None:
    """Run and compare all planned Stage 4 confirmations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/ablation/phase4/phase4-confirm.yaml"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text())
    if config["schema_version"] != 1:
        raise ValueError("Unsupported configuration schema.")
    if args.dry_run:
        print(
            json.dumps(
                [
                    {"candidate": c["id"], "seed": s}
                    for c in config["candidates"]
                    for s in config["study"]["seeds"]
                ],
                indent=2,
            )
        )
        return
    rows = []
    for candidate in config["candidates"]:
        for seed in config["study"]["seeds"]:
            result = run_one(config, candidate, seed)
            rows.append({"candidate": candidate["id"], **result})
            write_json(
                ROOT / config["study"]["output_root"] / "run_results.json", rows
            )
            print(
                f"Finished {result['name']}: CER {result['metrics']['eval_cer']:.5f}",
                flush=True,
            )
    save_results(config, rows)
    print("Completed Stage 4 confirmation.", flush=True)


if __name__ == "__main__":
    main()
