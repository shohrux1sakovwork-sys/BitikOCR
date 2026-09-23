"""Run the Stage 2 LoRA-rank sweep from its YAML specifications."""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from scripts.prepare_ablation_data import read_records

ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path: Path) -> dict[str, Any]:
    """Read one nonempty YAML mapping."""
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping in {path}.")
    return value


def write_json(path: Path, value: Any) -> None:
    """Write an indented JSON artifact, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def make_trainer_args(
    config: dict[str, Any], run: dict[str, Any], output: Path
) -> dict[str, Any]:
    """Translate a rank-run YAML entry into trainer arguments."""
    fixed = config["fixed"]
    data = config["data"]
    lora = config["lora"]
    return {
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "data_path": data["train_path"],
        "eval_path": data["eval_path"],
        "image_folder": data["image_folder"],
        "eval_image_folder": data["eval_image_folder"],
        "output_dir": str(output),
        "run_name": run["name"],
        "seed": run["seed"],
        "data_seed": fixed["data_seed"],
        "num_train_epochs": fixed["epochs"],
        "per_device_train_batch_size": fixed["per_device_train_batch_size"],
        "per_device_eval_batch_size": fixed["per_device_eval_batch_size"],
        "gradient_accumulation_steps": fixed["gradient_accumulation_steps"],
        "bf16": fixed["bf16"],
        "gradient_checkpointing": fixed["gradient_checkpointing"],
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": fixed["optimizer"],
        "learning_rate": run["learning_rate"],
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
        "lora_r": run["rank"],
        "lora_alpha": run["alpha"],
        "lora_dropout": lora["dropout"],
        "lora_target_modules": ",".join(lora["target_modules"]),
        "freeze_vision_encoder": lora["freeze_vision_encoder"],
        "resolved_config_path": str(output / "resolved_config.yaml"),
        "logging_steps": 1,
        "logging_first_step": True,
    }


def validate(config: dict[str, Any], run: dict[str, Any]) -> None:
    """Reject inconsistent rank-sweep inputs before allocating a GPU."""
    if run["alpha"] != config["lora"]["alpha_per_rank"] * run["rank"]:
        raise ValueError(f"{run['name']} does not preserve alpha/rank scaling.")
    if run["learning_rate"] != config["fixed"]["learning_rate"]:
        raise ValueError(f"{run['name']} changes the fixed learning rate.")
    if run["rank"] == config["reference"]["rank"]:
        raise ValueError("Rank 16 is a completed reference, not a new run.")


def validate_result(output: Path, expected_samples: int) -> dict[str, Any]:
    """Load complete finite evaluation metrics from one successful run."""
    metrics = json.loads((output / "eval_results.json").read_text())
    if metrics.get("eval_samples") != expected_samples:
        raise ValueError(f"{output} evaluated the wrong number of examples.")
    for key in ("eval_cer", "eval_wer"):
        if not isinstance(metrics.get(key), (int, float)) or not math.isfinite(
            metrics[key]
        ):
            raise ValueError(f"{output} has invalid {key}.")
    if len(read_records(output / "predictions.jsonl")) != expected_samples:
        raise ValueError(f"{output} has incomplete predictions.")
    return metrics


def run_one(
    config: dict[str, Any], run: dict[str, Any], dry_run: bool
) -> dict[str, Any]:
    """Run one rank candidate or reuse a complete matching output directory."""
    validate(config, run)
    output = ROOT / run["output_dir"]
    expected_samples = config["data"]["expected_eval_samples"]
    if (output / "eval_results.json").is_file():
        metrics = validate_result(output, expected_samples)
        resources = json.loads((output / "resources.json").read_text())
        return {
            "name": run["name"],
            "status": "reused",
            "run": run,
            "metrics": metrics,
            "resources": resources,
        }
    if dry_run:
        return {"name": run["name"], "status": "planned", "run": run}
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            f"Incomplete output exists: {output}. Use a new attempt directory."
        )
    output.mkdir(parents=True, exist_ok=True)
    arguments = make_trainer_args(config, run, output)
    resolved = {"phase2": config, "run": run, "trainer": arguments}
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False)
    )
    write_json(output / "trainer_args.json", arguments)
    wandb = config["wandb"]
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=config["study"]["gpu"],
        WANDB_ENTITY=wandb["entity"],
        WANDB_PROJECT=wandb["project"],
        WANDB_RUN_GROUP=wandb["group"],
        WANDB_MODE="online",
        WANDB_TAGS=",".join(wandb["tags"] + run["wandb_tags"]),
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
    )
    command = [
        sys.executable,
        "-m",
        "bitikocr.train.train",
        str(output / "trainer_args.json"),
    ]
    write_json(output / "command.json", command)
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
            f"{run['name']} failed; see {output / 'console.log'}."
        )
    metrics = validate_result(output, expected_samples)
    resources = json.loads((output / "resources.json").read_text())
    return {
        "name": run["name"],
        "status": "completed",
        "run": run,
        "metrics": metrics,
        "resources": resources,
        "wall_seconds": time.monotonic() - started,
    }


def reference(config: dict[str, Any]) -> dict[str, Any]:
    """Load the completed rank-16 result as the sweep comparison reference."""
    item = config["reference"]
    metrics = json.loads((ROOT / item["source_metrics"]).read_text())
    resources = json.loads((ROOT / item["source_resources"]).read_text())
    return {
        "name": item["name"],
        "status": "reference",
        "run": item,
        "metrics": metrics,
        "resources": resources,
    }


def save_comparison(config: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Save the rank table and publish it as one W&B comparison run."""
    comparison = ROOT / config["study"]["comparison_output"]
    comparison.parent.mkdir(parents=True, exist_ok=True)
    table = []
    for row in rows:
        run = row["run"]
        metrics = row["metrics"]
        resources = row["resources"]
        table.append(
            {
                "name": row["name"],
                "rank": run["rank"],
                "alpha": run["alpha"],
                "learning_rate": run.get(
                    "learning_rate", config["fixed"]["learning_rate"]
                ),
                "status": row["status"],
                "eval_cer": metrics["eval_cer"],
                "eval_wer": metrics["eval_wer"],
                "generation_limit_hits": metrics["eval_generation_limit_hits"],
                "trainable_parameters": resources["trainable_parameters"],
                "peak_allocated_bytes": resources["peak_allocated_bytes"],
            }
        )
    table.sort(key=lambda item: item["rank"])
    write_json(comparison, table)
    with comparison.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)
    import wandb

    wb = config["wandb"]
    run = wandb.init(
        entity=wb["entity"],
        project=wb["project"],
        group=wb["group"],
        name=wb["comparison_run_name"],
        tags=wb["tags"] + ["comparison"],
        config={
            "source_stage": config["study"]["source_stage"],
            "fixed": config["fixed"],
            "lora": config["lora"],
        },
    )
    columns: list[str | int] = [str(column) for column in table[0]]
    run.log(
        {
            "rank_comparison": wandb.Table(
                columns=columns,
                data=[
                    [row[str(column)] for column in columns] for row in table
                ],
            )
        }
    )
    best = min(
        table,
        key=lambda item: (
            item["eval_cer"],
            item["eval_wer"],
            item["trainable_parameters"],
        ),
    )
    run.summary.update(
        {
            "best_rank": best["rank"],
            "best_eval_cer": best["eval_cer"],
            "best_eval_wer": best["eval_wer"],
        }
    )
    run.finish()


def main() -> None:
    """Execute all Stage 2 rank candidates and publish their comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/ablation/phase2/phase2-rank.yaml"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    config = read_yaml(ROOT / args.config)
    rows = [reference(config)]
    for entry in config["runs"]:
        run = read_yaml(ROOT / entry["config"])["run"]
        print(f"Starting {run['name']}", flush=True)
        rows.append(run_one(config, run, args.dry_run))
    if args.dry_run:
        print(json.dumps(rows, indent=2))
        return
    save_comparison(config, rows)
    print(
        f"Completed Stage 2. Comparison: {config['study']['comparison_output']}"
    )


if __name__ == "__main__":
    main()
