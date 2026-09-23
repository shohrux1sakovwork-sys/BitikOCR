"""Run staged OCR experiments from one YAML configuration."""

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import HfApi

from scripts.prepare_ablation_data import (
    file_hash,
    prepare,
    read_records,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict[str, Any]:
    """Validate study settings before starting any subprocess."""
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported study schema_version.")
    if not config["project"]["wandb"]["project"]:
        raise ValueError("One shared W&B project is required.")
    for key in ("train_samples", "development_samples"):
        if not isinstance(config["data"][key], int) or config["data"][key] <= 0:
            raise ValueError(f"{key} must be a positive integer.")
    for lr in config["phases"]["learning_rate"]["candidates"]:
        if not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
            raise ValueError("Learning rates must be finite positive numbers.")
    if not config["study"]["fresh_pretrained_weights_per_run"]:
        raise ValueError("Independent runs must start from pretrained weights.")
    if not config["training"]["final_evaluation"]:
        raise ValueError("Ablation requires final evaluation.")
    if (
        config["processing"]["do_sample"]
        or config["processing"]["num_beams"] != 1
    ):
        raise ValueError("This evaluator supports greedy generation only.")
    return config


def candidate(
    target: str, rank: int, lr: float, seed: int = 42, epochs: int = 1
) -> dict[str, Any]:
    """Describe a complete independent training candidate."""
    return {
        "target": target,
        "rank": rank,
        "learning_rate": lr,
        "seed": seed,
        "epochs": epochs,
    }


def candidate_key(spec: dict[str, Any]) -> str:
    """Identify matching training settings independently of phase labels."""
    return hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()[:16]


def select_best(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Rank successful finite results with deterministic tie breaking."""
    valid = [
        r
        for r in results
        if r["status"] == "completed"
        and all(
            math.isfinite(r["metrics"][k]) for k in ("eval_cer", "eval_wer")
        )
    ]
    if not valid:
        raise ValueError(
            "No successful finite candidate remains in this phase."
        )
    return min(
        valid,
        key=lambda r: (
            r["metrics"]["eval_cer"],
            r["metrics"]["eval_wer"],
            r["resources"]["trainable_parameters"],
            r["name"],
        ),
    )


def read_metrics(directory: Path, expected_samples: int) -> dict[str, Any]:
    """Reject stale/incomplete/nonfinite metrics, allowing error rates above 1."""
    metrics = json.loads((directory / "eval_results.json").read_text())
    if metrics.get("eval_samples") != expected_samples:
        raise ValueError(
            "Evaluation sample count does not match the fixed split."
        )
    if not all(
        isinstance(metrics.get(k), (float, int)) and math.isfinite(metrics[k])
        for k in ("eval_cer", "eval_wer")
    ):
        raise ValueError("Missing or nonfinite OCR scores.")
    rows = read_records(directory / "predictions.jsonl")
    if len(rows) != expected_samples:
        raise ValueError("Prediction artifact is incomplete.")
    return metrics


class Study:
    """Persist study provenance and execute restartable sequential runs."""

    def __init__(self, config: dict[str, Any], study_id: str) -> None:
        """Freeze configuration and code provenance for safe result reuse."""
        self.config = copy.deepcopy(config)
        self.study_id = study_id
        self.directory = ROOT / config["study"]["output_root"] / study_id
        self.directory.mkdir(parents=True, exist_ok=True)
        self.results: list[dict[str, Any]] = []
        self.phases: dict[str, list[str]] = {}
        packages = (
            "torch",
            "torchvision",
            "transformers",
            "peft",
            "wandb",
            "PyYAML",
        )
        self.provenance = {
            "code": {
                str(p.relative_to(ROOT)): file_hash(p)
                for folder in ("bitikocr", "scripts")
                for p in sorted((ROOT / folder).rglob("*.py"))
            },
            "versions": {
                name: importlib.metadata.version(name) for name in packages
            },
            "manifest": file_hash(config["data"]["manifest_path"]),
        }
        frozen = self.directory / "study_config.yaml"
        if frozen.exists():
            saved = yaml.safe_load(frozen.read_text())
            if config["model"]["revision"] is None:
                self.config["model"]["revision"] = saved["model"]["revision"]
            if saved != self.config:
                raise ValueError("Study settings changed; use a new study ID.")
            if (
                json.loads((self.directory / "provenance.json").read_text())
                != self.provenance
            ):
                raise ValueError(
                    "Code, dependencies, or data changed; use a new study ID."
                )
            summary = self.directory / "summary.json"
            if summary.exists():
                self.results = json.loads(summary.read_text())
        else:
            self.config["model"]["revision"] = (
                HfApi()
                .model_info(
                    config["model"]["model_id"],
                    revision=config["model"]["revision"],
                )
                .sha
            )
            frozen.write_text(yaml.safe_dump(self.config, sort_keys=False))
            write_json(self.directory / "provenance.json", self.provenance)
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.free,memory.total,driver_version",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        (self.directory / "hardware.txt").write_text(gpu.stdout)

    def save_summary(self) -> None:
        """Save machine-readable and tabular progress after each subprocess."""
        write_json(self.directory / "summary.json", self.results)
        write_json(self.directory / "phase_runs.json", self.phases)
        fields = [
            "name",
            "phase",
            "status",
            "target",
            "rank",
            "learning_rate",
            "seed",
            "epochs",
            "eval_cer",
            "eval_wer",
            "baseline_cer_difference",
            "wall_seconds",
            "trainable_parameters",
            "peak_allocated_bytes",
        ]
        baseline = next(
            (
                r["metrics"]["eval_cer"]
                for r in self.results
                if r["phase"] == "baseline" and r["status"] == "completed"
            ),
            None,
        )
        with (self.directory / "summary.csv").open("w") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=fields, extrasaction="ignore"
            )
            writer.writeheader()
            for result in self.results:
                row = {
                    **result,
                    **result["spec"],
                    **result.get("metrics", {}),
                    **result.get("resources", {}),
                }
                if baseline is not None and "eval_cer" in row:
                    row["baseline_cer_difference"] = row["eval_cer"] - baseline
                writer.writerow(row)

    def arguments(
        self, spec: dict[str, Any], directory: Path, name: str
    ) -> dict[str, Any]:
        """Translate the declarative configuration to trainer arguments."""
        config = self.config
        data = config["data"]
        args = {
            k: v
            for k, v in config["training"].items()
            if k not in ("final_evaluation", "save_final_checkpoint")
        }
        args.update(
            {
                k: v
                for k, v in config["processing"].items()
                if k not in ("do_sample", "num_beams")
            }
        )
        # Transformers 5 represents fractional warmup through warmup_steps.
        args["warmup_steps"] = args.pop("warmup_ratio")
        target = config["lora"]["targets"][spec["target"]]
        args.update(
            model_id=config["model"]["model_id"],
            model_revision=config["model"]["revision"],
            data_path=data["train_path"],
            eval_path=data["development_path"],
            image_folder=data["image_folder"],
            eval_image_folder=data["eval_image_folder"],
            output_dir=str(directory),
            run_name=name,
            final_evaluation=True,
            seed=spec["seed"],
            data_seed=spec["seed"],
            num_train_epochs=spec["epochs"],
            learning_rate=spec["learning_rate"],
            lora_r=spec["rank"],
            lora_alpha=config["lora"]["alpha_per_rank"] * spec["rank"],
            lora_dropout=config["lora"]["dropout"],
            lora_target_modules=",".join(target["modules"]),
            freeze_vision_encoder=target["freeze_vision_encoder"],
            resolved_config_path=str(
                directory / config["study"]["resolved_config_filename"]
            ),
            logging_steps=1,
            logging_first_step=True,
        )
        for key in (
            "evaluation_only",
            "adapter_path",
            "verify_vision_update",
            "max_steps",
        ):
            if key in spec:
                args[key] = spec[key]
        if spec.get("pilot"):
            for split, setting in (
                ("train", "data_path"),
                ("development", "eval_path"),
            ):
                records = read_records(
                    data[
                        "train_path" if split == "train" else "development_path"
                    ]
                )
                # Include both large images and long reference sequences.
                longest = sorted(
                    records, key=lambda r: len(r["text"]), reverse=True
                )[:2]
                largest = sorted(
                    records,
                    key=lambda r: math.prod(r.get("size", [0])),
                    reverse=True,
                )[:2]
                selected = list(
                    {r["id"]: r for r in longest + largest}.values()
                )
                path = directory / f"pilot_{split}.jsonl"
                path.write_text("".join(json.dumps(r) + "\n" for r in selected))
                args[setting] = str(path)
            args["max_steps"] = len(read_records(args["data_path"]))
            args["gradient_accumulation_steps"] = 1
            args["warmup_steps"] = 0.0
        return args

    def run(self, phase: str, spec: dict[str, Any]) -> dict[str, Any]:
        """Run a candidate once or reuse a verified matching completed result."""
        key = candidate_key(spec)
        matches = [r for r in self.results if r["key"] == key]
        if self.config["study"]["reuse_matching_completed_runs"]:
            for result in reversed(matches):
                if result["status"] == "completed":
                    read_metrics(
                        Path(result["directory"]), result["expected_samples"]
                    )
                    self.phases.setdefault(phase, []).append(result["name"])
                    return result
        attempt = len(matches) + 1
        wb = self.config["project"]["wandb"]
        template = (
            wb["baseline_name_template"]
            if phase == "baseline"
            else wb["run_name_template"]
        )
        name = template.format(
            phase=phase,
            target=self.config["lora"]["targets"][spec["target"]]["name"],
            rank=spec["rank"],
            alpha=self.config["lora"]["alpha_per_rank"] * spec["rank"],
            lr_label=f"{spec['learning_rate']:.0e}",
            seed=spec["seed"],
            attempt=attempt,
        )
        directory = self.directory / name
        directory.mkdir(exist_ok=False)
        args = self.arguments(spec, directory, name)
        resolved = {
            "project": self.config["project"],
            "study_id": self.study_id,
            "phase": phase,
            "candidate": spec,
            "trainer": args,
            "provenance": self.provenance,
        }
        Path(args["resolved_config_path"]).write_text(
            yaml.safe_dump(resolved, sort_keys=False)
        )
        write_json(directory / "trainer_args.json", args)
        expected = len(read_records(args["eval_path"]))
        result = {
            "name": name,
            "phase": phase,
            "key": key,
            "spec": spec,
            "directory": str(directory),
            "expected_samples": expected,
            "status": "running",
        }
        self.results.append(result)
        self.phases.setdefault(phase, []).append(name)
        self.save_summary()
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES=str(self.config["study"]["gpu"]),
            WANDB_PROJECT=wb["project"],
            WANDB_ENTITY=wb["entity"],
            WANDB_RUN_GROUP=wb["group_template"].format(study_id=self.study_id),
            WANDB_MODE=wb["mode"],
            WANDB_RUN_ID=hashlib.sha256(
                f"{self.study_id}/{name}".encode()
            ).hexdigest()[:16],
            WANDB_TAGS=",".join(wb["tags"] + [phase]),
            TOKENIZERS_PARALLELISM="false",
            PYTHONUNBUFFERED="1",
        )
        command = [
            sys.executable,
            "-m",
            "bitikocr.train.train",
            str(directory / "trainer_args.json"),
        ]
        write_json(directory / "command.json", command)
        print(f"Starting {name}", flush=True)
        started = time.monotonic()
        with (directory / "console.log").open("w") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result.update(
            exit_code=process.returncode,
            wall_seconds=time.monotonic() - started,
        )
        try:
            if process.returncode:
                raise ValueError(
                    f"Training process exited {process.returncode}; see console.log."
                )
            result["metrics"] = read_metrics(directory, expected)
            result["resources"] = json.loads(
                (directory / "resources.json").read_text()
            )
            if (
                not spec.get("evaluation_only")
                and not (directory / "adapter_config.json").is_file()
            ):
                raise ValueError("Final adapter checkpoint is missing.")
            result["status"] = "completed"
        except (ValueError, OSError, KeyError) as error:
            result["status"] = "failed"
            result["error"] = str(error)
        self.save_summary()
        print(
            f"{name}: {result['status']} {result.get('metrics', result.get('error'))}",
            flush=True,
        )
        return result

    def pilot(self) -> bool:
        """Check large adapters and verify saved visual weights in a fresh process."""
        vision_ok = False
        for item in self.config["phases"]["pilot"]["candidates"]:
            spec = candidate(
                item["target"],
                item["rank"],
                self.config["training"]["learning_rate"],
            )
            spec.update(pilot=True, verify_vision_update=item["target"] == "D")
            trained = self.run("p0-pilot", spec)
            if trained["status"] != "completed":
                if item["target"] == "D":
                    write_json(
                        self.directory / "vision_skipped.json",
                        {
                            "reason": trained.get("error"),
                            "run": trained["name"],
                        },
                    )
                    continue
                raise ValueError(
                    "Decoder resource pilot failed; inspect its log before screening."
                )
            reload_spec = {
                **spec,
                "evaluation_only": True,
                "verify_vision_update": False,
                "adapter_path": trained["directory"],
            }
            restored = self.run("p0-reload", reload_spec)
            if restored["status"] != "completed":
                raise ValueError("Fresh-process checkpoint reload failed.")
            original_rows = read_records(
                Path(trained["directory"]) / "predictions.jsonl"
            )
            restored_rows = read_records(
                Path(restored["directory"]) / "predictions.jsonl"
            )
            if original_rows != restored_rows:
                raise ValueError(
                    "Pilot predictions changed after checkpoint reload."
                )
            if item["target"] == "D":
                if (
                    not trained["resources"]["vision_digest"]
                    or trained["resources"]["vision_digest"]
                    != restored["resources"]["vision_digest"]
                ):
                    raise ValueError(
                        "Reloaded visual weights differ from the trained checkpoint."
                    )
                vision_ok = True
        return vision_ok

    def execute(self, stop_after_pilot: bool = False) -> None:
        """Run the pilot, baseline, three screening phases, and confirmation."""
        vision_ok = self.pilot()
        if stop_after_pilot:
            return
        cfg = self.config
        seed = cfg["training"]["seed"]
        lr = cfg["training"]["learning_rate"]
        baseline = self.run(
            "baseline",
            {**candidate("A", 16, lr, seed), "evaluation_only": True},
        )
        if baseline["status"] != "completed":
            raise ValueError("Untuned baseline failed.")
        p1 = [
            self.run("p1-targets", candidate(target, 16, lr, seed))
            for target in cfg["phases"]["targets"]["candidates"]
            if target != "D" or vision_ok
        ]
        best_target = select_best(p1)["spec"]["target"]
        p2 = [
            self.run("p2-rank", candidate(best_target, rank, lr, seed))
            for rank in cfg["phases"]["rank"]["candidates"]
        ]
        best_rank = select_best(p2)["spec"]["rank"]
        p3 = [
            self.run("p3-lr", candidate(best_target, best_rank, rate, seed))
            for rate in cfg["phases"]["learning_rate"]["candidates"]
        ]
        best_lr = select_best(p3)["spec"]["learning_rate"]
        distinct = {
            r["key"]: r for r in p1 + p2 + p3 if r["status"] == "completed"
        }
        finalists = []
        for _ in range(
            min(
                cfg["phases"]["confirmation"]["top_distinct_screening_configs"],
                len(distinct),
            )
        ):
            chosen = select_best(list(distinct.values()))
            finalists.append(chosen["spec"])
            del distinct[chosen["key"]]
        if cfg["phases"]["confirmation"]["include_current_baseline_config"]:
            finalists.append(candidate("A", 16, lr, seed))
        runners_up = [r for r in p1 if r["spec"]["target"] != best_target]
        if (
            runners_up
            and cfg["phases"]["confirmation"][
                "include_runner_up_target_with_selected_rank_and_lr"
            ]
        ):
            finalists.append(
                candidate(
                    select_best(runners_up)["spec"]["target"],
                    best_rank,
                    best_lr,
                    seed,
                )
            )
        unique = {candidate_key(c): c for c in finalists}
        confirmation = []
        for spec in unique.values():
            runs = [
                self.run(
                    "p4-confirm",
                    {
                        **spec,
                        "seed": trial_seed,
                        "epochs": cfg["phases"]["confirmation"][
                            "num_train_epochs"
                        ],
                    },
                )
                for trial_seed in cfg["phases"]["confirmation"]["seeds"]
            ]
            if all(r["status"] == "completed" for r in runs):
                scores = [r["metrics"]["eval_cer"] for r in runs]
                mean = sum(scores) / len(scores)
                confirmation.append(
                    {
                        "configuration": spec,
                        "mean_cer": mean,
                        "std_cer": (
                            sum((v - mean) ** 2 for v in scores)
                            / max(1, len(scores) - 1)
                        )
                        ** 0.5,
                        "mean_wer": sum(r["metrics"]["eval_wer"] for r in runs)
                        / len(runs),
                        "trainable_parameters": runs[0]["resources"][
                            "trainable_parameters"
                        ],
                    }
                )
        if not confirmation:
            raise ValueError(
                "No configuration completed every confirmation seed."
            )
        confirmation.sort(
            key=lambda r: (
                r["mean_cer"],
                r["mean_wer"],
                r["trainable_parameters"],
            )
        )
        write_json(self.directory / "confirmation.json", confirmation)
        self.save_summary()
        print(f"Study complete: {self.directory}", flush=True)


def main() -> None:
    """Execute the requested study without launching final full-data training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ablation.yaml")
    parser.add_argument("--study-id")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    config = load_config(Path(args.config))
    prepare(config)
    if args.prepare_only:
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check driver compatibility and GPU access."
        )
    if config["project"]["wandb"]["mode"] == "online" and not os.environ.get(
        "WANDB_API_KEY"
    ):
        raise ValueError(
            "Set WANDB_API_KEY in the process environment; do not put credentials in YAML."
        )
    study_id = (
        args.study_id
        or config["study"]["id"]
        or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    study = Study(config, study_id)
    study.execute(stop_after_pilot=args.pilot_only)


if __name__ == "__main__":
    main()
