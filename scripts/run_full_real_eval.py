"""Train the selected adapter on a synthetic set with real fact evaluation.

With no arguments it reruns the 18K training; the options point it at
another synthetic set, such as the 5K v2 corpus::

    uv run --extra train python scripts/run_full_real_eval.py \
        --data data/synthetic-v2/index.jsonl \
        --image-folder data/synthetic-v2 \
        --output outputs/full/20260924-5k-v2-real-facts \
        --eval-steps 125 --group full-5k-v2-real-facts

``--evaluation-only`` skips training and scores a model as it is on the
175 real pages, such as an untuned Qwen3-VL::

    uv run --extra train python scripts/run_full_real_eval.py \
        --model Qwen/Qwen3-VL-8B-Instruct --evaluation-only \
        --output outputs/eval/qwen3-vl-8b-untuned --group eval-untuned
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
REAL_BENCH = ROOT / "data/uzbek-htr-bench"

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
#: The Qwen2.5-VL snapshot every earlier run trained from. Other models load
#: their latest snapshot unless a revision is given.
DEFAULT_MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse which synthetic set to train on and where to write the run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data/train.jsonl")
    parser.add_argument("--image-folder", default="data")
    parser.add_argument(
        "--output", default="outputs/full/20260923-18k-literal-real-facts"
    )
    parser.add_argument("--eval-steps", type=int, default=450)
    parser.add_argument("--group", default="full-18k-real-facts")
    parser.add_argument("--gpu", default="1")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face id of the base model (default: %(default)s)",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="snapshot of the model to load; the default model is pinned",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="score the model on the real pages without training it",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="stop after this many steps; -1 trains the full epoch",
    )
    parser.add_argument(
        "--save-only-model",
        action="store_true",
        help="keep only the adapter in checkpoints, not the optimizer",
    )
    return parser.parse_args(argv)


def write_json(path: Path, value: Any) -> None:
    """Write a stable JSON artifact."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def model_revision(options: argparse.Namespace) -> str | None:
    """Return the snapshot to load: the one asked for, or the pinned one."""
    if options.model_revision is not None:
        return str(options.model_revision)
    if options.model == DEFAULT_MODEL:
        return DEFAULT_MODEL_REVISION
    return None


def benchmark_rows(bench: Path) -> list[dict[str, Any]]:
    """Read the benchmark's pages from its annotation files.

    The published benchmark carries one annotation per page but no index,
    so a fresh download works without the ``bench.jsonl`` built locally.
    """
    rows = []
    for path in sorted((bench / "annotations").glob("*.json")):
        annotation = json.loads(path.read_text())
        metadata = annotation["metadata"]
        rows.append(
            {
                "id": annotation["id"],
                "image": annotation["image"],
                "text": annotation["text"],
                "document_type": metadata["document_type"],
                "script": "+".join(sorted(metadata["scripts"])),
            }
        )
    return rows


def prepare_real_eval(path: Path, bench: Path = REAL_BENCH) -> int:
    """Join benchmark rows with their structured fact annotations."""
    rows = benchmark_rows(bench)
    for row in rows:
        facts = json.loads((bench / "facts" / f"{row['id']}.json").read_text())
        row["facts"] = facts["facts"]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    )
    return len(rows)


def main() -> None:
    """Materialize configuration and run resumable full training."""
    options = parse_args()
    output = ROOT / options.output
    output.mkdir(parents=True, exist_ok=True)
    eval_path = output / "real_eval_with_facts.jsonl"
    expected = prepare_real_eval(eval_path)
    if expected != 175:
        raise ValueError(f"Expected 175 real documents, found {expected}.")
    args = {
        "model_id": options.model,
        "model_revision": model_revision(options),
        "evaluation_only": options.evaluation_only,
        # Evaluation reads no training data; a fresh clone has none.
        "data_path": str(
            eval_path if options.evaluation_only else ROOT / options.data
        ),
        "eval_path": str(eval_path),
        "image_folder": str(ROOT / options.image_folder),
        "eval_image_folder": str(REAL_BENCH),
        "output_dir": str(output),
        "run_name": output.name,
        "seed": 42,
        "data_seed": 42,
        "num_train_epochs": 1,
        "max_steps": options.max_steps,
        "save_only_model": options.save_only_model,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "optim": "adamw_torch",
        "learning_rate": 0.0001,
        "lr_scheduler_type": "linear",
        "warmup_steps": 0,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "eval_strategy": "steps",
        "eval_steps": options.eval_steps,
        "save_strategy": "steps",
        "save_steps": options.eval_steps,
        "save_total_limit": 5,
        "load_best_model_at_end": True,
        "metric_for_best_model": "fact_all_accuracy",
        "greater_is_better": True,
        "report_to": ["wandb"],
        "logging_steps": 10,
        "logging_first_step": True,
        "image_min_pixels": 3136,
        "image_max_pixels": 1003520,
        "max_new_tokens": 1024,
        "final_evaluation": True,
        "lora_r": 64,
        "lora_alpha": 128,
        "lora_dropout": 0.0,
        "lora_target_modules": (
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
        ),
        "freeze_vision_encoder": True,
        "resolved_config_path": str(output / "resolved_config.yaml"),
    }
    write_json(output / "trainer_args.json", args)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(
            {
                "purpose": (
                    f"{options.model} on {options.data}: literal OCR with "
                    "175-page factual checkpoint evaluation"
                ),
                "checkpoint_selection": "eval_fact_all_accuracy",
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
    environment = os.environ.copy()
    environment.update(
        CUDA_VISIBLE_DEVICES=options.gpu,
        WANDB_ENTITY="isakovsh",
        WANDB_PROJECT="BitikOCR",
        WANDB_RUN_GROUP=options.group,
        WANDB_MODE=os.environ.get("WANDB_MODE", "online"),
        WANDB_TAGS=f"ocr,{options.group},literal-prompt,real-fact-eval",
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
    )
    with (output / "console.log").open("a") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
