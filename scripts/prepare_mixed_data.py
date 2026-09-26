"""Mix the gold Adliya pages into synthetic sets for training.

The gold pages are transcribed with bracketed notes (``[имзо]``,
``[Штамп: ...]``) where the benchmark and the synthetic sets use tags
(``<signature>``, ``<stamp>...</stamp>``). A model trained on both would
learn both, so every note is rewritten in the benchmark's conventions first,
and a note no rule covers stops the run rather than slipping through::

    uv run python scripts/prepare_mixed_data.py \\
        --real data/adliya-real --synthetic data/synthetic-v3/index.jsonl \\
        --repeat 13 --output data/mix-v3-adliya

Several synthetic sets can be mixed at once. Their ids restart at
``<type>_000000`` in every set, so each id is prefixed with its set's folder.
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]

#: Longest side a gold page is kept at: an A4 sheet at 300 dpi. A few pages
#: were scanned at over 100 megapixels, far past what the model is shown.
MAX_SIDE = 3508

#: Stands in for ``[...]`` while notes are rewritten, so the benchmark's own
#: illegible mark is not read as a note.
_ILLEGIBLE = "\x00"

_SIGNATURE_NOTES = {"имзо", "imzo", "қолы", "подпись"}
_DELETED_NOTES = {"ўчирилган", "o'chirilgan", "o‘chirilgan"}
_STAMP_NAMES = r"штамп|муҳр|shtamp|muhr"
_STAMP = re.compile(
    rf"^(?:{_STAMP_NAMES})\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL
)
_BARE_STAMP = re.compile(rf"^(?:{_STAMP_NAMES})$", re.IGNORECASE)
# A resolution or a clerk's note is page text the benchmark transcribes as
# it stands, so only its label goes.
_WRITTEN = re.compile(
    r"^(?:резолюция|resolyutsiya|қўлда|qo'lda)\s*:\s*(.+)$",
    re.IGNORECASE | re.DOTALL,
)
_ILLEGIBLE_NOTE = re.compile(
    r"ўқилиши қийин|ўқилмайди|o'qilishi qiyin|o'qilmaydi", re.IGNORECASE
)
# Notes about the scan rather than the page: not text the model should write.
_DESCRIPTION = re.compile(
    r"^(?:фото санаси|орқа фонда|орқа томонидаги)", re.IGNORECASE
)
_INNERMOST_NOTE = re.compile(r"\[([^\[\]]*)\]")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse where the sets are and how often to repeat the gold pages."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--real", type=Path, default=ROOT / "data/adliya-real")
    parser.add_argument(
        "--synthetic",
        type=Path,
        nargs="+",
        default=[ROOT / "data/synthetic-v3/index.jsonl"],
        help="one or more synthetic index files",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=13,
        help="copies of each gold page in the mix (default: %(default)s)",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/mix-v3-adliya"
    )
    parser.add_argument(
        "--image-folder",
        type=Path,
        default=ROOT / "data",
        help="folder the index's image paths are relative to",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def convert_notes(text: str) -> str:
    """Rewrite a gold transcription's bracketed notes as benchmark tags.

    Args:
        text: The transcription as the reviewers wrote it.

    Returns:
        The same transcription in the benchmark's conventions.

    Raises:
        ValueError: If a note matches none of the known kinds.
    """
    text = text.replace("[...]", _ILLEGIBLE)
    # Notes nest (a resolution carries its signature), so the innermost
    # are rewritten first until none are left.
    while True:
        rewritten = _INNERMOST_NOTE.sub(lambda m: _convert_note(m[1]), text)
        if rewritten == text:
            break
        text = rewritten
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text.replace(_ILLEGIBLE, "[...]")


def _convert_note(note: str) -> str:
    """Return the benchmark's form of one note's contents."""
    content = note.strip()
    lowered = content.lower()
    if lowered in _SIGNATURE_NOTES:
        return "<signature>"
    if lowered in _DELETED_NOTES:
        return "<deleted></deleted>"
    if stamp := _STAMP.match(content):
        body = stamp[1].strip().replace("...", _ILLEGIBLE)
        return f"\n<stamp>\n{body}\n</stamp>\n"
    if _BARE_STAMP.match(content):
        return "<stamp/>"
    if written := _WRITTEN.match(content):
        return written[1].strip()
    if _ILLEGIBLE_NOTE.search(content):
        return _ILLEGIBLE
    if _DESCRIPTION.match(content):
        return ""
    raise ValueError(f"no rule for the note [{note}]")


def prepare_image(source: Path, target: Path) -> tuple[int, int]:
    """Save a gold page upright and no larger than an A4 scan at 300 dpi."""
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if max(image.size) > MAX_SIDE:
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, "JPEG", quality=95)
    return image.size


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records, skipping blank lines."""
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def prepare_real(real: Path, image_folder: Path) -> list[dict[str, Any]]:
    """Convert every gold page and return its training record."""
    prepared = real / "prepared/images"
    records = []
    for row in read_jsonl(real / "data/train/metadata.jsonl"):
        target = prepared / f"{row['id']}.jpg"
        size = prepare_image(real / "data/train" / row["file_name"], target)
        records.append(
            {
                "id": f"adliya_{row['id']}",
                "image": str(target.relative_to(image_folder)),
                "text": convert_notes(row["text"]),
                "source": "adliya-real",
                "size": list(size),
            }
        )
    return records


def main(argv: list[str] | None = None) -> None:
    """Write the mixed index and a manifest of what went into it."""
    args = parse_args(argv)
    image_folder = args.image_folder.resolve()
    real = prepare_real(args.real.resolve(), image_folder)

    synthetic = []
    for index in args.synthetic:
        root = index.resolve().parent
        for row in read_jsonl(index):
            image = (root / row["image"]).relative_to(image_folder)
            synthetic.append(
                {
                    **row,
                    "id": f"{root.name}/{row['id']}",
                    "image": str(image),
                    "source": root.name,
                }
            )

    mixed = synthetic + [
        {**record, "id": f"{record['id']}_r{copy}"}
        for record in real
        for copy in range(args.repeat)
    ]
    random.Random(args.seed).shuffle(mixed)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "index.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in mixed)
    )
    (args.output / "real.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in real)
    )
    manifest = {
        "synthetic": [str(index) for index in args.synthetic],
        "synthetic_pages": len(synthetic),
        "real": str(args.real),
        "real_pages": len(real),
        "repeat": args.repeat,
        "rows": len(mixed),
        "real_share": round(len(real) * args.repeat / len(mixed), 3),
        "seed": args.seed,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
