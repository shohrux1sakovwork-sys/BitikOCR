"""Prepare reproducible OCR development splits without evaluation leakage."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """Read JSONL records without changing their source metadata."""
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def file_hash(path: str | Path) -> str:
    """Hash a file for provenance and restart validation."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    """Atomically save a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def grouped_records(
    records: list[dict[str, Any]], keys: list[str]
) -> list[list[dict[str, Any]]]:
    """Group transitively linked duplicate identifiers before splitting."""
    parents = list(range(len(records)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for key in keys:
            value = record.get(key)
            if value:
                identifier = (key, str(value))
                if identifier in seen:
                    parents[root(index)] = root(seen[identifier])
                seen[identifier] = index
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[root(index)].append(record)
    return list(groups.values())


def sample_groups(
    groups: list[list[dict[str, Any]]], count: int, rng: random.Random
) -> tuple[list[list[dict[str, Any]]], list[list[dict[str, Any]]]]:
    """Select whole groups with deterministic proportional stratum allocation."""
    strata: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for group in groups:
        # Script is included to preserve coverage within document categories.
        label = min(
            f"{r.get('document_type', '')}/{r.get('script', '')}" for r in group
        )
        strata[label].append(group)
    total = sum(len(g) for g in groups)
    if count > total or count <= 0:
        raise ValueError(
            "Requested split size must be positive and fit available records."
        )
    sizes = {key: sum(map(len, value)) for key, value in strata.items()}
    quotas = {key: count * size // total for key, size in sizes.items()}
    remainder = count - sum(quotas.values())
    for key in sorted(sizes, key=lambda k: (-(count * sizes[k] % total), k))[
        :remainder
    ]:
        quotas[key] += 1
    selected: list[list[dict[str, Any]]] = []
    remaining: list[list[dict[str, Any]]] = []
    for key in sorted(strata):
        candidates = strata[key][:]
        rng.shuffle(candidates)
        # Subset-sum retains duplicate groups even when group sizes differ.
        paths: dict[int, tuple[int, int] | None] = {0: None}
        target = quotas[key]
        for index, group in enumerate(candidates):
            for size in list(paths):
                new_size = size + len(group)
                if new_size <= target and new_size not in paths:
                    paths[new_size] = (size, index)
            if target in paths:
                break
        if target not in paths:
            raise ValueError(
                f"Cannot allocate {target} records in {key} without splitting duplicate groups."
            )
        chosen = set()
        cursor = target
        while cursor:
            previous = paths[cursor]
            assert previous is not None
            cursor, index = previous
            chosen.add(index)
        selected.extend(g for i, g in enumerate(candidates) if i in chosen)
        remaining.extend(g for i, g in enumerate(candidates) if i not in chosen)
    return selected, remaining


def prepare(config: dict[str, Any]) -> dict[str, Any]:
    """Write fixed training/development records and a verifiable manifest."""
    settings = config["data"]
    train = read_records(settings["source_train"])
    reserved = read_records(settings["reserved_eval"])
    keys = settings["duplicate_group_keys"]
    for key in keys:
        held_out = {str(r[key]) for r in reserved if r.get(key)}
        if any(str(r[key]) in held_out for r in train if r.get(key)):
            raise ValueError(
                f"Training source overlaps reserved evaluation by {key}."
            )
    groups = grouped_records(train, keys)
    rng = random.Random(settings["sampling_seed"])
    development_groups, remaining = sample_groups(
        groups, settings["development_samples"], rng
    )
    train_groups, _ = sample_groups(remaining, settings["train_samples"], rng)
    splits = {
        "train": [r for g in train_groups for r in g],
        "development": [r for g in development_groups for r in g],
    }
    manifest: dict[str, Any] = {
        "seed": settings["sampling_seed"],
        "allocation": "document_type/script largest remainder; whole duplicate groups",
        "source_hashes": {
            name: file_hash(settings[name])
            for name in ("source_train", "reserved_eval")
        },
        "splits": {},
    }
    for name, records in splits.items():
        rng.shuffle(records)
        root = Path(
            settings["image_folder" if name == "train" else "eval_image_folder"]
        )
        for record in records:
            if not (root / record["image"]).is_file():
                raise FileNotFoundError(root / record["image"])
        path = Path(
            settings["train_path" if name == "train" else "development_path"]
        )
        content = "".join(
            json.dumps(r, ensure_ascii=False) + "\n" for r in records
        )
        if path.exists() and path.read_text() != content:
            raise ValueError(
                f"{path} already contains a different split; preserve it or choose new output paths."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        manifest["splits"][name] = {
            "count": len(records),
            "sha256": file_hash(path),
            "ids": [r["id"] for r in records],
            "document_types": dict(
                Counter(r.get("document_type") for r in records)
            ),
            "scripts": dict(Counter(r.get("script") for r in records)),
        }
    write_json(Path(settings["manifest_path"]), manifest)
    return manifest


def main() -> None:
    """Prepare splits using the project's YAML configuration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ablation.yaml")
    args = parser.parse_args()
    manifest = prepare(yaml.safe_load(Path(args.config).read_text()))
    print(
        json.dumps(
            {
                k: {x: v[x] for x in ("count", "document_types", "scripts")}
                for k, v in manifest["splits"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
