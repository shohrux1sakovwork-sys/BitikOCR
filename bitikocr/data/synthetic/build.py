"""Building a whole corpus: plan every page once, then render in parallel.

A corpus of tens of thousands of pages has needs a single ``generate`` run
does not:

- **No page twice.** Every document's content is fingerprinted while it is
  planned, and a record whose fingerprint has been seen is drawn again.
- **Even coverage.** Fonts, form variants and pen weights are dealt out so
  each gets its share, rather than left to chance.
- **Parallel rendering.** Handwriting is drawn by a pool of CPU processes;
  the page-wide augmentation, which costs more, runs on a GPU when there is
  one.
- **Resumable.** A page is finished once its annotation is on disk, so an
  interrupted build picks up where it stopped.

The plan is a JSON-lines file with one :class:`PlannedDocument` per page. It
holds everything a page is made from, so rendering needs nothing else and
any page can be re-rendered exactly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import queue
import random
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from bitikocr.config import SyntheticConfig
from bitikocr.data.synthetic.annotation import DocumentAnnotation
from bitikocr.data.synthetic.augment import (
    AugmentationPlan,
    AugmentationProfile,
    Backend,
    apply_plan,
    plan_augmentation,
)
from bitikocr.data.synthetic.dataset import (
    INDEX_NAME,
    DatasetLayout,
    index_entry,
    iter_index,
    write_sample,
)
from bitikocr.data.synthetic.export import document_id
from bitikocr.data.synthetic.fonts import FontLibrary
from bitikocr.data.synthetic.generators import (
    GENERATOR_TYPES,
    DocumentGenerator,
    FormGenerator,
    create_generator,
)
from bitikocr.data.synthetic.layout import CLIPPED_KEY
from bitikocr.data.synthetic.phrases import PhraseBank
from bitikocr.data.synthetic.records import (
    DEFAULT_LATIN_SHARE,
    DocumentRecord,
    sample_record,
)
from bitikocr.data.synthetic.scripts import Script
from bitikocr.data.synthetic.templates import FormTemplate

__all__ = [
    "PLAN_NAME",
    "BuildProgress",
    "BuildSummary",
    "CorpusSpec",
    "PageRenderer",
    "PlannedDocument",
    "build_corpus",
    "iter_plan_counts",
    "pending_documents",
    "plan_corpus",
    "read_plan",
    "record_fingerprint",
    "templates_for",
    "write_plan",
]

logger = logging.getLogger(__name__)

#: What the plan is called inside a build's work directory.
PLAN_NAME = "plan.jsonl"

#: Pen weights a page is drawn with. A few fixed levels rather than a
#: continuous range, so a worker can keep one generator per level.
INK_LEVELS: tuple[float, ...] = (1.2, 1.4, 1.6, 1.8)

#: How many times a duplicate record is redrawn before the plan gives up on
#: finding a new one.
_MAX_REDRAWS = 200

#: Letters a font must draw to be dealt a page in an alphabet.
_ALPHABETS: Mapping[Script, str] = {
    "latin": "abcdefghijklmnopqrstuvxyz",
    "cyrillic": "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
}


@dataclass(frozen=True)
class CorpusSpec:
    """What a corpus should hold.

    Args:
        counts: Document type mapped to how many pages of it.
        seed: Makes the whole plan reproducible.
        latin_share: Share of pages written in Latin, where the form allows
            either alphabet.
        profile: How pages are spoiled.
        id_prefix: Prepended to every id; the document type follows it.
    """

    counts: Mapping[str, int]
    seed: int = 0
    latin_share: float = DEFAULT_LATIN_SHARE
    profile: AugmentationProfile = field(
        default_factory=AugmentationProfile.varied
    )
    id_prefix: str = ""

    @property
    def total(self) -> int:
        """How many pages the corpus holds."""
        return sum(self.counts.values())


@dataclass(frozen=True)
class PlannedDocument:
    """Everything one page is made from.

    Args:
        id: The id every file of the page shares.
        record: What the page says.
        template: The form variant it fills, for form documents.
        font: The handwriting font's name.
        ink: How heavily the pen writes.
        augmentation: How the page is spoiled.
        fingerprint: The record's content fingerprint.
    """

    id: str
    record: DocumentRecord
    template: str | None
    font: str
    ink: float
    augmentation: AugmentationPlan
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable form of the plan entry."""
        return {
            "id": self.id,
            "template": self.template,
            "font": self.font,
            "ink": self.ink,
            "fingerprint": self.fingerprint,
            "record": self.record.to_dict(),
            "augmentation": self.augmentation.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlannedDocument:
        """Rebuild a plan entry from its JSON form."""
        return cls(
            id=str(payload["id"]),
            record=DocumentRecord.from_dict(payload["record"]),
            template=payload.get("template"),
            font=str(payload["font"]),
            ink=float(payload["ink"]),
            augmentation=AugmentationPlan.from_dict(payload["augmentation"]),
            fingerprint=str(payload["fingerprint"]),
        )


# -- planning --------------------------------------------------------------


def record_fingerprint(record: DocumentRecord) -> str:
    """Return a digest of what a record says, ignoring how it is drawn.

    Args:
        record: The record.

    Returns:
        A hex digest; two records with the same fields share it.
    """
    content = json.dumps(
        {
            "type": record.document_type,
            "script": record.script,
            "fields": record.fields,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(content.encode("utf-8")).hexdigest()


def templates_for(
    document_type: str, config: SyntheticConfig
) -> tuple[str | None, ...]:
    """Return the form variants a document type can fill.

    Args:
        document_type: A registered document type.
        config: Where the layouts live.

    Returns:
        The variants' names, or ``(None,)`` for a type drawn on a blank
        sheet.
    """
    if not issubclass(GENERATOR_TYPES[document_type], FormGenerator):
        return (None,)
    names = tuple(
        name
        for name in config.available_layouts()
        if name == document_type or name.startswith(f"{document_type}_")
    )
    return names or (None,)


def _forced_script(
    template: str | None, config: SyntheticConfig
) -> Script | None:
    """Return the alphabet a form must be filled in, if it insists on one.

    A form printed only in Cyrillic is filled in Cyrillic; a bilingual one
    takes either.
    """
    if template is None:
        return None
    printed = FormTemplate.load(config.layout(template)).printed_languages
    if printed and all(language == "uz-cyrillic" for language in printed):
        return "cyrillic"
    return None


class _Dealer:
    """Deals out choices so each is used about equally."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._used: Counter[str] = Counter()

    def deal(self, options: Sequence[str]) -> str:
        """Return the least-used option, breaking ties at random."""
        fewest = min(self._used[option] for option in options)
        chosen = self._rng.choice(
            [option for option in options if self._used[option] == fewest]
        )
        self._used[chosen] += 1
        return chosen

    @property
    def counts(self) -> Counter[str]:
        """How often each option was dealt."""
        return Counter(self._used)


def plan_corpus(
    spec: CorpusSpec,
    config: SyntheticConfig,
    phrases: PhraseBank | None = None,
) -> list[PlannedDocument]:
    """Decide everything about every page before any is drawn.

    Args:
        spec: What the corpus should hold.
        config: Where fonts and layouts live.
        phrases: Extra wording for the samplers.

    Returns:
        One entry per page, in id order.

    Raises:
        ValueError: If a type is unknown, or no new record can be found.
    """
    rng = random.Random(spec.seed)
    library = FontLibrary.from_directory(config.fonts_dir)
    fonts = _Dealer(rng)
    templates = _Dealer(rng)
    seen: set[str] = set()
    plan: list[PlannedDocument] = []

    for document_type, count in sorted(spec.counts.items()):
        if document_type not in GENERATOR_TYPES:
            raise ValueError(f"Unknown document type {document_type!r}")
        variants = templates_for(document_type, config)
        forced = {v: _forced_script(v, config) for v in variants}
        generators = {
            v: create_generator(document_type, config, template=v)
            for v in variants
        }
        prefix = f"{spec.id_prefix}{document_type}"

        for position in range(count):
            template = _deal_template(templates, document_type, variants)
            generator = generators[template]
            record, fingerprint = _new_record(
                document_type,
                rng,
                forced[template],
                spec.latin_share,
                phrases,
                seen,
            )
            text = generator.coverage_text(record.fields)
            usable = [
                font.name
                for font in library
                if font.can_render(_ALPHABETS[record.script])
                and font.can_render(text)
            ]
            if not usable:
                raise ValueError(
                    f"No font can write {document_type} record {position}"
                )
            plan.append(
                PlannedDocument(
                    id=document_id(prefix, position),
                    record=record,
                    template=template,
                    font=fonts.deal(usable),
                    ink=rng.choice(INK_LEVELS),
                    augmentation=plan_augmentation(rng, spec.profile),
                    fingerprint=fingerprint,
                )
            )
    return plan


def _deal_template(
    dealer: _Dealer, document_type: str, variants: Sequence[str | None]
) -> str | None:
    """Pick a form variant, spreading a type's pages across its variants."""
    if len(variants) == 1:
        return variants[0]
    keys = [str(v) for v in variants]
    chosen = dealer.deal(keys)
    return variants[keys.index(chosen)]


def _new_record(
    document_type: str,
    rng: random.Random,
    script: Script | None,
    latin_share: float,
    phrases: PhraseBank | None,
    seen: set[str],
) -> tuple[DocumentRecord, str]:
    """Draw records until one says something no earlier one said."""
    for _ in range(_MAX_REDRAWS):
        record = sample_record(document_type, rng, script, latin_share, phrases)
        fingerprint = record_fingerprint(record)
        if fingerprint not in seen:
            seen.add(fingerprint)
            return record, fingerprint
    raise ValueError(
        f"Could not draw a new {document_type} record after "
        f"{_MAX_REDRAWS} tries; the samplers are too narrow"
    )


def write_plan(plan: Sequence[PlannedDocument], path: Path) -> None:
    """Write a plan as JSON lines.

    Args:
        plan: The entries.
        path: Where to write them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in plan:
            handle.write(json.dumps(entry.to_dict(), ensure_ascii=False))
            handle.write("\n")


def read_plan(path: Path) -> list[PlannedDocument]:
    """Read a plan written by :func:`write_plan`."""
    with path.open(encoding="utf-8") as handle:
        return [
            PlannedDocument.from_dict(json.loads(line))
            for line in handle
            if line.strip()
        ]


def pending_documents(
    plan: Sequence[PlannedDocument], output_dir: Path
) -> list[PlannedDocument]:
    """Return the entries whose page is not yet finished.

    Args:
        plan: The whole plan.
        output_dir: The dataset being built.

    Returns:
        The entries still to render, in plan order.
    """
    layout = DatasetLayout(output_dir)
    return [
        entry
        for entry in plan
        if not (layout.annotations / f"{entry.id}.json").exists()
    ]


# -- rendering -------------------------------------------------------------


class PageRenderer:
    """Draws and finishes planned pages, keeping one generator per setup.

    Args:
        config: Where fonts, backgrounds and layouts live.
        output_dir: The dataset pages are written into.
        collection: The batch name recorded on every page.
    """

    def __init__(
        self, config: SyntheticConfig, output_dir: Path, collection: str
    ) -> None:
        self.config = config
        self.layout = DatasetLayout(output_dir)
        self.collection = collection
        self._library = FontLibrary.from_directory(config.fonts_dir)
        self._generators: dict[
            tuple[str, str | None, str, float], DocumentGenerator
        ] = {}

    def generator(self, entry: PlannedDocument) -> DocumentGenerator:
        """Return the generator a planned page is drawn with."""
        key = (
            entry.record.document_type,
            entry.template,
            entry.font,
            entry.ink,
        )
        if key not in self._generators:
            font = self._library.by_name(entry.font)
            if font is None:
                raise ValueError(f"Font {entry.font!r} is not installed")
            self._generators[key] = create_generator(
                entry.record.document_type,
                self.config,
                font_path=font.path,
                template=entry.template,
                ink_strength=entry.ink,
            )
        return self._generators[key]

    def draw(
        self, entry: PlannedDocument
    ) -> tuple[Image.Image, DocumentAnnotation]:
        """Draw a planned page, clean.

        Raises:
            ValueError: If the page cannot be drawn.
        """
        document = self.generator(entry).generate(
            entry.record.fields, seed=entry.record.seed
        )
        clipped = document.annotation.metadata.get(CLIPPED_KEY)
        if clipped:
            # Refused rather than written: its transcription would claim
            # letters the image does not show.
            raise ValueError(f"text cut off by the page edge: {clipped}")
        return document.image, document.annotation

    def finish(
        self,
        entry: PlannedDocument,
        image: Image.Image,
        annotation: DocumentAnnotation,
        backend: Backend,
    ) -> dict[str, Any]:
        """Spoil a drawn page, write it, and return its index entry."""
        image, annotation, quality = apply_plan(
            image, annotation, entry.augmentation, backend
        )
        annotation.metadata.setdefault("script", entry.record.script)
        self.layout.create()
        files = write_sample(
            layout=self.layout,
            identifier=entry.id,
            record=entry.record,
            generator=self.generator(entry),
            image=image,
            annotation=annotation,
            quality=quality,
            collection=self.collection,
            draw_boxes=False,
        )
        line = index_entry(entry.record, files, annotation, self.layout)
        line.update(
            font=entry.font,
            template=entry.template,
            ink=entry.ink,
            capture=quality.capture,
            effects=list(entry.augmentation.effects),
            fingerprint=entry.fingerprint,
        )
        return line


@dataclass(frozen=True)
class BuildProgress:
    """How far a build has got.

    Args:
        done: Pages finished in this run.
        failed: Pages that could not be drawn.
        total: Pages this run set out to finish.
        elapsed: Seconds since the run started.
    """

    done: int
    failed: int
    total: int
    elapsed: float

    @property
    def rate(self) -> float:
        """Pages per second so far."""
        return self.done / self.elapsed if self.elapsed > 0 else 0.0

    @property
    def remaining(self) -> float:
        """Estimated seconds left."""
        left = self.total - self.done - self.failed
        return left / self.rate if self.rate > 0 else float("inf")


@dataclass(frozen=True)
class BuildSummary:
    """What a build run did.

    Args:
        done: Pages finished in this run.
        failed: ``(id, reason)`` for pages that could not be drawn.
        skipped: Pages already finished by an earlier run.
        elapsed: Seconds the run took.
        backend: Where the augmentation ran.
    """

    done: int
    failed: tuple[tuple[str, str], ...]
    skipped: int
    elapsed: float
    backend: Backend


def build_corpus(
    plan: Sequence[PlannedDocument],
    output_dir: Path,
    config: SyntheticConfig,
    workers: int,
    gpu_workers: int = 0,
    collection: str | None = None,
    progress: Callable[[BuildProgress], None] | None = None,
    progress_every: float = 30.0,
) -> BuildSummary:
    """Render every unfinished page of a plan.

    With ``gpu_workers`` above zero, CPU workers only draw the handwriting
    and hand each page to a GPU worker, which spoils and writes it.
    Otherwise every worker does the whole page on the CPU.

    Args:
        plan: The whole plan; finished pages are skipped.
        output_dir: The dataset to build.
        config: Where fonts, backgrounds and layouts live.
        workers: CPU processes drawing handwriting.
        gpu_workers: Processes spoiling pages on the GPU.
        collection: The batch name on every page; the directory's name when
            omitted.
        progress: Called with the build's progress now and then.
        progress_every: Seconds between progress calls.

    Returns:
        What the run did.
    """
    started = time.perf_counter()
    todo = pending_documents(plan, output_dir)
    skipped = len(plan) - len(todo)
    backend: Backend = "cuda" if gpu_workers > 0 else "cpu"
    collection = collection or output_dir.name
    DatasetLayout(output_dir).create()
    if not todo:
        return BuildSummary(0, (), skipped, 0.0, backend)

    context = mp.get_context("spawn")
    tasks: Any = context.Queue()
    results: Any = context.Queue()
    handoff: Any = context.Queue(maxsize=max(4, gpu_workers * 6))
    for entry in todo:
        tasks.put(entry.to_dict())
    for _ in range(workers):
        tasks.put(None)

    setup = (config, output_dir, collection)
    drawers = [
        context.Process(
            target=_draw_worker,
            args=(setup, tasks, handoff if gpu_workers else None, results),
            daemon=True,
        )
        for _ in range(workers)
    ]
    finishers = [
        context.Process(
            target=_gpu_worker, args=(setup, handoff, results), daemon=True
        )
        for _ in range(gpu_workers)
    ]
    for process in (*drawers, *finishers):
        process.start()

    index_part = output_dir / f"{INDEX_NAME}.part"
    done = 0
    failed: list[tuple[str, str]] = []
    last_report = started
    with index_part.open("a", encoding="utf-8") as index:
        while done + len(failed) < len(todo):
            try:
                kind, identifier, payload = results.get(timeout=60)
            except queue.Empty:
                if not any(p.is_alive() for p in (*drawers, *finishers)):
                    logger.error("Every worker has stopped; ending the build")
                    break
                continue
            if kind == "done":
                index.write(json.dumps(payload, ensure_ascii=False) + "\n")
                done += 1
            else:
                failed.append((identifier, payload))
                logger.warning("%s failed: %s", identifier, payload)
            now = time.perf_counter()
            if progress and now - last_report >= progress_every:
                index.flush()
                progress(
                    BuildProgress(done, len(failed), len(todo), now - started)
                )
                last_report = now

    for process in drawers:
        process.join()
    for _ in finishers:
        handoff.put(None)
    for process in finishers:
        process.join()

    _merge_index(output_dir)
    elapsed = time.perf_counter() - started
    if progress:
        progress(BuildProgress(done, len(failed), len(todo), elapsed))
    return BuildSummary(done, tuple(failed), skipped, elapsed, backend)


def _draw_worker(
    setup: tuple[SyntheticConfig, Path, str],
    tasks: Any,
    handoff: Any | None,
    results: Any,
) -> None:
    """Draw pages; finish them too when there is no GPU stage."""
    renderer = PageRenderer(*setup)
    while True:
        payload = tasks.get()
        if payload is None:
            return
        entry = PlannedDocument.from_dict(payload)
        try:
            image, annotation = renderer.draw(entry)
            if handoff is not None:
                handoff.put((payload, image, annotation))
                continue
            line = renderer.finish(entry, image, annotation, "cpu")
        except ValueError as error:
            results.put(("failed", entry.id, str(error)))
            continue
        except Exception:  # noqa: BLE001 - one page must not end the build
            results.put(("failed", entry.id, traceback.format_exc(limit=3)))
            continue
        results.put(("done", entry.id, line))


def _gpu_worker(
    setup: tuple[SyntheticConfig, Path, str], handoff: Any, results: Any
) -> None:
    """Spoil drawn pages on the GPU and write them."""
    renderer = PageRenderer(*setup)
    while True:
        item = handoff.get()
        if item is None:
            return
        payload, image, annotation = item
        entry = PlannedDocument.from_dict(payload)
        try:
            line = renderer.finish(entry, image, annotation, "cuda")
        except Exception:  # noqa: BLE001 - one page must not end the build
            results.put(("failed", entry.id, traceback.format_exc(limit=3)))
            continue
        results.put(("done", entry.id, line))


def _merge_index(output_dir: Path) -> None:
    """Fold the running index into ``index.jsonl``, one line per page."""
    layout = DatasetLayout(output_dir)
    part = output_dir / f"{INDEX_NAME}.part"
    entries: dict[str, dict[str, Any]] = {}
    for source in (layout.index, part):
        if source.exists():
            entries.update((e["id"], e) for e in iter_index(source))
    lines = [
        json.dumps(entries[key], ensure_ascii=False) for key in sorted(entries)
    ]
    layout.index.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )
    part.unlink(missing_ok=True)


def iter_plan_counts(
    plan: Sequence[PlannedDocument],
) -> Iterator[tuple[str, Counter[str]]]:
    """Yield how a plan spreads over each axis it balances.

    Args:
        plan: The plan.

    Yields:
        ``(axis, counts)`` for type, script, template, font, ink, capture
        and effect.
    """
    yield "type", Counter(e.record.document_type for e in plan)
    yield "script", Counter(e.record.script for e in plan)
    yield "template", Counter(str(e.template) for e in plan)
    yield "font", Counter(e.font for e in plan)
    yield "ink", Counter(str(e.ink) for e in plan)
    yield "capture", Counter(e.augmentation.capture for e in plan)
    yield "effect", Counter(
        effect for e in plan for effect in e.augmentation.effects
    )
