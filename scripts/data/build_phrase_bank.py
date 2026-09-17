"""Fill a phrase bank with wording written by local language models.

Asks models served by Ollama for interchangeable pieces of Uzbek wording,
one slot at a time, and keeps only what passes the checks in
:mod:`bitikocr.data.synthetic.phrases`. The models take turns, so no one
model's habits dominate the bank, and each request is steered towards a
different theme and shown a few phrases it must not repeat.

Every batch is then read by the next model in turn, which keeps only the
phrases it judges to be natural, grammatical Uzbek. A model never reviews
its own writing.

The bank is saved after every request, so a run can be stopped and resumed.

Usage::

    uv run --extra data python scripts/data/build_phrase_bank.py \
        -o work/phrases.json --models qwen3.5:9b gemma4:e4b muse-glimmer
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bitikocr.data.synthetic.phrases import PHRASE_SLOTS, PhraseBank

logger = logging.getLogger("build_phrase_bank")

DEFAULT_MODELS = ("qwen3.5:9b", "gemma4:e4b", "muse-glimmer:latest")
DEFAULT_HOST = "http://localhost:11434"

#: How many phrases each slot is filled to.
DEFAULT_TARGETS: dict[str, int] = {
    "ariza_request": 300,
    "ariza_detail": 250,
    "explanation_reason": 250,
    "explanation_task": 120,
    "explanation_closing": 120,
    "consent_closing": 150,
}

#: Themes a request is steered towards, so successive batches differ.
_THEMES = (
    "housing and land",
    "family circumstances",
    "health and hospitals",
    "weather and roads",
    "public transport",
    "schooling and children",
    "farming and irrigation",
    "utilities: gas, water, electricity",
    "pensions and social support",
    "work and the workplace",
    "neighbours and the mahalla",
    "documents and registration",
    "small business and trade",
    "elderly relatives",
    "construction and repairs",
)

_REVIEW_SYSTEM = (
    "You are a strict native Uzbek editor. You judge whether short pieces "
    "of formal Uzbek, in the Latin alphabet, are grammatical and natural. "
    'Reply with JSON only, as {"good": [numbers of the good phrases]}.'
)

_SYSTEM = (
    "You write short pieces of formal Uzbek for handwritten official "
    "letters. Write only in the Uzbek Latin alphabet, using a plain "
    "apostrophe in o', g' and for the tutuq belgisi. Never use digits, "
    "personal names, place names or organisation names. Reply with JSON "
    'only, as {"phrases": ["...", "..."]}.'
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="bank file to fill"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Ollama models to take turns (default: %(default)s)",
    )
    parser.add_argument(
        "--turn",
        type=int,
        default=1,
        help="requests each model makes before the next takes over "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=12,
        help="phrases asked for per request (default: %(default)s)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="multiply every slot's target (default: %(default)s)",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=600,
        help="give up after this many requests (default: %(default)s)",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama server")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
    )
    rng = random.Random(args.seed)
    bank = (
        PhraseBank.load(args.output) if args.output.exists() else PhraseBank()
    )
    targets = {
        slot: max(1, round(count * args.scale))
        for slot, count in DEFAULT_TARGETS.items()
    }
    kept: Counter[str] = Counter()
    asked: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    models = list(args.models)

    for request in range(args.max_requests):
        missing = {
            slot: target - bank.count(slot)
            for slot, target in targets.items()
            if bank.count(slot) < target
        }
        if not missing or not models:
            break
        slot = max(missing, key=lambda name: missing[name] / targets[name])
        model = models[(request // args.turn) % len(models)]
        reviewer = _reviewer(models, model)
        prompt = _prompt(slot, bank, rng, args.batch)
        started = time.perf_counter()
        try:
            phrases = _ask(args.host, model, prompt, rng)
            approved = _review(args.host, reviewer, slot, phrases, rng)
        except (OSError, ValueError) as error:
            failures[model] += 1
            logger.warning("%s failed: %s", model, error)
            if failures[model] >= 3:
                logger.warning("dropping %s after repeated failures", model)
                models.remove(model)
            continue

        added = 0
        for phrase in phrases:
            asked[model] += 1
            if phrase in approved and bank.add(slot, phrase, model) is None:
                added += 1
                kept[model] += 1
        bank.save(args.output)
        logger.info(
            "#%d %-20s %-20s +%2d/%2d approved %2d  (%d/%d) %.0fs",
            request + 1,
            model,
            slot,
            added,
            len(phrases),
            len(approved),
            bank.count(slot),
            targets[slot],
            time.perf_counter() - started,
        )

    for model in args.models:
        _unload(args.host, model)
    print(f"{len(bank)} phrases in {args.output}")
    for slot, target in targets.items():
        print(f"  {slot:<22} {bank.count(slot):>4} / {target}")
    for model in args.models:
        print(f"  {model:<28} kept {kept[model]:>4} of {asked[model]:>4} asked")
    return 0


def _reviewer(models: Sequence[str], writer: str) -> str:
    """Return the model that reviews a writer's batch: the next one along."""
    if len(models) == 1:
        return writer
    return models[(models.index(writer) + 1) % len(models)]


def _review(
    host: str,
    model: str,
    slot_name: str,
    phrases: Sequence[str],
    rng: random.Random,
) -> set[str]:
    """Ask a model which phrases are natural Uzbek, and return those."""
    if not phrases:
        return set()
    slot = PHRASE_SLOTS[slot_name]
    numbered = "\n".join(f"{i}. {text}" for i, text in enumerate(phrases, 1))
    prompt = (
        f"These phrases are meant to be: {slot.purpose}.\n"
        f"Example of a good one: {slot.example}\n\n{numbered}\n\n"
        "List the numbers of the phrases that are grammatical, natural "
        "Uzbek, correctly spelled, and fit that purpose. Leave out any "
        "with invented words, broken suffixes, or nonsense."
    )
    content = _chat(host, model, _REVIEW_SYSTEM, prompt, rng, 0.1)
    good = _json_value(content, "good")
    if not isinstance(good, list):
        good = re.findall(r"\d+", content)
    chosen = set()
    for item in good:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= index <= len(phrases):
            chosen.add(phrases[index - 1])
    return chosen


def _prompt(
    slot_name: str, bank: PhraseBank, rng: random.Random, batch: int
) -> str:
    """Write one request for a slot."""
    slot = PHRASE_SLOTS[slot_name]
    existing = list(bank.phrases(slot_name))
    avoid = rng.sample(existing, min(6, len(existing)))
    shape = (
        "Each phrase is one complete sentence, starting with a capital "
        "letter and ending with a single full stop."
        if slot.sentence
        else "Each phrase is a lower-case fragment with no punctuation at "
        f"the end, whose last word ends in one of: {', '.join(slot.endings)}."
    )
    lines = [
        f"Write {batch} different phrases: {slot.purpose}.",
        shape,
        f"Each phrase has {slot.words[0]} to {slot.words[1]} words.",
        f"Example: {slot.example}",
        f"Lean towards this theme where it fits: {rng.choice(_THEMES)}.",
        "Vary the vocabulary and structure between phrases.",
    ]
    if avoid:
        lines.append("Do not repeat or paraphrase these:")
        lines.extend(f"- {text}" for text in avoid)
    return "\n".join(lines)


def _ask(host: str, model: str, prompt: str, rng: random.Random) -> list[str]:
    """Send one request and return the phrases in the reply."""
    content = _chat(host, model, _SYSTEM, prompt, rng, rng.uniform(0.8, 1.1))
    parsed = _json_value(content, "phrases")
    if not isinstance(parsed, list):
        # A reply cut short or badly escaped still carries its phrases as
        # quoted strings; take those rather than lose the batch.
        parsed = [
            item
            for item in re.findall(r'"((?:[^"\\]|\\.){6,})"', content)
            if item != "phrases"
        ]
    if not parsed:
        raise ValueError(f"reply holds no phrases: {content[:80]!r}")
    return [item for item in parsed if isinstance(item, str)]


def _json_value(content: str, key: str) -> Any:
    """Return one key of a JSON reply, or None if it cannot be read."""
    try:
        parsed: Any = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed.get(key, next(iter(parsed.values()), None))
    return parsed


def _chat(
    host: str,
    model: str,
    system: str,
    prompt: str,
    rng: random.Random,
    temperature: float,
) -> str:
    """Send one chat request and return the reply's text."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "format": "json",
        "stream": False,
        "think": False,
        "keep_alive": "10m",
        "options": {
            "temperature": temperature,
            "top_p": 0.95,
            "seed": rng.randrange(2**31),
            "num_predict": 2000,
        },
    }
    reply = _post(host, "/api/chat", body)
    return str(reply.get("message", {}).get("content", ""))


def _post(host: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST JSON to the Ollama server and decode the reply."""
    request = urllib.request.Request(
        host.rstrip("/") + path,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            decoded: dict[str, Any] = json.loads(response.read())
            return decoded
    except urllib.error.HTTPError as error:
        raise ValueError(
            f"HTTP {error.code}: {error.read()[:200]!r}"
        ) from error


def _unload(host: str, model: str) -> None:
    """Free the GPU memory a model holds."""
    try:
        _post(host, "/api/generate", {"model": model, "keep_alive": 0})
    except (OSError, ValueError):
        pass


if __name__ == "__main__":
    sys.exit(main())
