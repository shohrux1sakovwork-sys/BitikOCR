"""Wording drawn from outside the samplers, to keep letters from repeating.

The samplers in :mod:`bitikocr.data.synthetic.records` write letters from a
handful of phrases each. That is enough to teach a recogniser the genre, but
at tens of thousands of pages the same sentence comes round again and
again. A phrase bank widens that vocabulary: it holds interchangeable
pieces of wording for named slots — what an applicant asks for, why an
employee was late — and a sampler draws from it alongside its own phrases.

The bank is written by language models (see
``scripts/data/build_phrase_bank.py``), so every entry is checked here
before it is accepted: it must be plain Uzbek in the Latin alphabet, carry
no names or numbers a facts record would miss, and fit the grammar of the
sentence it is dropped into. Everything is stored in Latin and
transliterated when a Cyrillic letter needs it, as the rest of the corpus
is.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "PHRASE_SLOTS",
    "PhraseBank",
    "PhraseSlot",
    "normalise_phrase",
    "phrase_problem",
]

_FORMAT_VERSION = 1


@dataclass(frozen=True)
class PhraseSlot:
    """A place in a letter a bank phrase can fill.

    Args:
        name: What the slot is called in the bank.
        purpose: What the phrase says, for whoever writes them.
        example: One phrase that fits.
        sentence: Whether the phrase is a whole sentence, capitalised and
            ending in a full stop, or a clause dropped into one.
        endings: Word endings a clause must finish on to fit its sentence.
        words: The fewest and most words a phrase may have.
    """

    name: str
    purpose: str
    example: str
    sentence: bool
    endings: tuple[str, ...] = ()
    words: tuple[int, int] = (3, 18)


#: Every slot a sampler draws from, by name.
PHRASE_SLOTS: Mapping[str, PhraseSlot] = {
    slot.name: slot
    for slot in (
        PhraseSlot(
            name="ariza_request",
            purpose=(
                "what a citizen asks a district mayor for in an application, "
                "as the object of the closing verb so'rayman, ending in a "
                "verb with -ishingizni or -ib berishingizni, as in 'uyimga "
                "gaz o'tkazib berishingizni' or 'nafaqa tayinlashingizni'"
            ),
            example="farzandimni maktabgacha ta'lim muassasasiga qabul "
            "qilishingizni",
            sentence=False,
            endings=("ingizni", "shingizni", "ni"),
            words=(3, 16),
        ),
        PhraseSlot(
            name="ariza_detail",
            purpose=(
                "one sentence of background an applicant gives about their "
                "family, home or circumstances"
            ),
            example="Oilamiz ko'p bolali bo'lib, hozirda ijarada yashaymiz.",
            sentence=True,
            words=(4, 18),
        ),
        PhraseSlot(
            name="explanation_reason",
            purpose=(
                "why an employee or a pupil was late or absent, as a noun "
                "clause that fits the sentence 'Bunga ... sabab bo'ldi', "
                "ending in a real verb with -gani or -ganim, as in "
                "'onamning kasal bo'lib qolgani' or 'kech uyg'onganim'"
            ),
            example="yo'lda avtobusning buzilib qolgani",
            sentence=False,
            endings=("gani", "ganim", "kani", "kanim", "qani", "qanim"),
            words=(2, 12),
        ),
        PhraseSlot(
            name="explanation_task",
            purpose=(
                "a piece of office work an employee failed to hand in on "
                "time, as a direct object ending in -ni"
            ),
            example="yillik moliyaviy hisobotni",
            sentence=False,
            endings=("ni",),
            words=(1, 6),
        ),
        PhraseSlot(
            name="explanation_closing",
            purpose=(
                "one sentence in which an employee or pupil undertakes not "
                "to repeat a lapse, without apologising"
            ),
            example="Kelgusida bunday holatga yo'l qo'ymayman.",
            sentence=True,
            words=(3, 14),
        ),
        PhraseSlot(
            name="consent_closing",
            purpose=(
                "one closing sentence of a consent letter, saying it is "
                "given freely and the writer has no claims"
            ),
            example="Ushbu rozilik xatini o'z ixtiyorim bilan yozdim.",
            sentence=True,
            words=(3, 16),
        ),
    )
}

#: Apostrophe look-alikes a model writes for the Uzbek tutuq and the
#: marks on o' and g'. The corpus spells all of them with a plain one.
_APOSTROPHES = str.maketrans({c: "'" for c in "ʻʼ‘’`´ʹ′"})

_ALLOWED = re.compile(r"^[A-Za-z' ,.\-]+$")

#: A Latin ``c`` appears in Uzbek only in ``ch``; anything else is a
#: borrowed spelling the transliteration would get wrong.
_STRAY_C = re.compile(r"c(?!h)", re.IGNORECASE)

#: Letters Uzbek Latin does not use.
_FOREIGN = re.compile(r"[w]", re.IGNORECASE)

_LEADING_NOISE = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")


def normalise_phrase(text: str) -> str:
    """Tidy a phrase the way a model tends to mangle it.

    Args:
        text: One phrase as a model wrote it.

    Returns:
        The phrase with list markers, quotes and odd apostrophes removed
        and its spacing collapsed.
    """
    text = _LEADING_NOISE.sub("", text.translate(_APOSTROPHES))
    text = text.strip().strip('"“”«»').strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"\s+([,.])", r"\1", text)


def phrase_problem(text: str, slot: PhraseSlot) -> str | None:
    """Say why a phrase cannot fill a slot, or None when it can.

    Args:
        text: A normalised phrase.
        slot: The slot it is meant for.

    Returns:
        The reason it was rejected, or None.
    """
    if not text:
        return "empty"
    if not _ALLOWED.match(text):
        return "characters outside plain Uzbek Latin"
    if _STRAY_C.search(text) or _FOREIGN.search(text):
        return "letters Uzbek Latin does not use"
    if "''" in text or text.startswith("'"):
        return "stray apostrophe"
    words = text.rstrip(".").split()
    low, high = slot.words
    if not low <= len(words) <= high:
        return f"{len(words)} words, expected {low} to {high}"

    if slot.sentence:
        if not text[0].isupper():
            return "a sentence must start with a capital"
        if not text.endswith(".") or text.count(".") != 1:
            return "a sentence must end in its only full stop"
        if any(word[:1].isupper() for word in words[1:]):
            return "proper names would escape the facts record"
        return None

    if text != text.lower():
        return "a clause must be lower case"
    if text.endswith((".", ",")):
        return "a clause must not end in punctuation"
    if slot.endings and not words[-1].endswith(slot.endings):
        return f"must end in one of {', '.join(slot.endings)}"
    if len(words[-1]) < max(len(e) for e in slot.endings) + 2:
        # "... bo'lmasligi kani": the suffix written as a word of its own.
        return "the ending stands alone instead of closing a word"
    return None


def _key(text: str) -> str:
    """Return what two phrases are compared by."""
    return re.sub(r"[^a-z' ]", "", text.lower()).strip()


@dataclass
class PhraseBank:
    """Checked wording for each slot, and which model wrote it.

    Args:
        entries: Slot name mapped to ``(phrase, source)`` pairs.
    """

    entries: dict[str, list[tuple[str, str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._seen = {
            _key(text) for pairs in self.entries.values() for text, _ in pairs
        }

    def __len__(self) -> int:
        return sum(len(pairs) for pairs in self.entries.values())

    def __iter__(self) -> Iterator[tuple[str, str, str]]:
        for slot, pairs in self.entries.items():
            for text, source in pairs:
                yield slot, text, source

    def add(self, slot: str, text: str, source: str) -> str | None:
        """Check a phrase and keep it if it fits and is new.

        Args:
            slot: The slot it is for.
            text: The phrase, as written.
            source: Which model wrote it.

        Returns:
            None when the phrase was kept, otherwise why it was not.

        Raises:
            KeyError: If ``slot`` is not a known slot.
        """
        cleaned = normalise_phrase(text)
        problem = phrase_problem(cleaned, PHRASE_SLOTS[slot])
        if problem is not None:
            return problem
        key = _key(cleaned)
        if key in self._seen:
            return "duplicate"
        self._seen.add(key)
        self.entries.setdefault(slot, []).append((cleaned, source))
        return None

    def count(self, slot: str) -> int:
        """Return how many phrases a slot holds."""
        return len(self.entries.get(slot, ()))

    def phrases(self, slot: str) -> tuple[str, ...]:
        """Return a slot's phrases, in the order they were added."""
        return tuple(text for text, _ in self.entries.get(slot, ()))

    def sample(self, rng: random.Random, slot: str) -> str | None:
        """Draw one phrase for a slot, or None when it holds none."""
        pairs = self.entries.get(slot)
        return rng.choice(pairs)[0] if pairs else None

    def save(self, path: Path) -> None:
        """Write the bank as JSON.

        Args:
            path: Where to write it.
        """
        payload = {
            "version": _FORMAT_VERSION,
            "slots": {
                slot: [{"text": t, "source": s} for t, s in pairs]
                for slot, pairs in sorted(self.entries.items())
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> PhraseBank:
        """Read a bank written by :meth:`save`, re-checking every phrase.

        Args:
            path: The file to read.

        Returns:
            The bank. Phrases that no longer pass the checks are dropped.

        Raises:
            ValueError: If the file is not a phrase bank.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != _FORMAT_VERSION:
            raise ValueError(f"{path} is not a phrase bank")
        bank = cls()
        for slot, items in payload.get("slots", {}).items():
            if slot not in PHRASE_SLOTS:
                continue
            for item in items:
                bank.add(slot, str(item["text"]), str(item["source"]))
        return bank
