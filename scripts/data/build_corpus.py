"""Build a large synthetic corpus: plan it, render it in parallel, check it.

Steps, each its own command, and ``all`` to run them in order::

    plan     decide every page up front: content, font, form, spoiling
    render   draw the pages, handwriting on the CPU, augmentation on the GPU
    check    validate every page against the schema and the others
    clean    delete the working files once the corpus has passed

Usage::

    uv run --extra data --extra gpu python scripts/data/build_corpus.py \
        all -o corpus/v1 --per-type 4000 --phrases work/phrases.json

The working files — the plan and the phrase bank — live in ``<output>/_work``
and are removed by ``clean``. What remains is the dataset: ``images/``,
``annotations/``, ``facts/`` and ``index.jsonl``.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from bitikocr.config import SyntheticConfig
from bitikocr.data.synthetic.augment import AugmentationProfile
from bitikocr.data.synthetic.build import (
    PLAN_NAME,
    BuildProgress,
    CorpusSpec,
    build_corpus,
    iter_plan_counts,
    pending_documents,
    plan_corpus,
    read_plan,
    write_plan,
)
from bitikocr.data.synthetic.generators import available_document_types
from bitikocr.data.synthetic.phrases import PhraseBank
from bitikocr.data.synthetic.records import DEFAULT_LATIN_SHARE
from bitikocr.data.synthetic.validate import check_corpus

logger = logging.getLogger("build_corpus")

WORK_DIR = "_work"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan", help="decide every page up front")
    render = commands.add_parser("render", help="draw the planned pages")
    check = commands.add_parser("check", help="validate the finished corpus")
    clean = commands.add_parser("clean", help="delete the working files")
    everything = commands.add_parser("all", help="plan, render, check, clean")

    for command in (plan, render, check, clean, everything):
        command.add_argument(
            "-o", "--output", type=Path, required=True, help="corpus directory"
        )
    for command in (plan, everything):
        _add_plan_arguments(command)
    for command in (render, everything):
        _add_render_arguments(command)
    everything.add_argument(
        "--keep-work",
        action="store_true",
        help="keep the plan and phrase bank after a clean check",
    )
    clean.add_argument(
        "--phrases",
        type=Path,
        default=None,
        help="phrase bank to delete along with the plan",
    )
    return parser


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--per-type",
        type=int,
        default=4000,
        help="pages per document type (default: %(default)s)",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=list(available_document_types()),
        help="document types to include (default: all)",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--latin-share", type=float, default=DEFAULT_LATIN_SHARE
    )
    parser.add_argument(
        "--augment",
        type=float,
        default=1.0,
        help="augmentation strength (default: %(default)s)",
    )
    parser.add_argument(
        "--look",
        choices=("varied", "archive"),
        default="varied",
        help="varied wear and photographs, or the archive's own flat, "
        "levelled scans with punch holes (default: %(default)s)",
    )
    parser.add_argument(
        "--phrases", type=Path, default=None, help="phrase bank to draw from"
    )


def _add_render_arguments(parser: argparse.ArgumentParser) -> None:
    cores = os.cpu_count() or 4
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cores - 4),
        help="CPU processes drawing handwriting (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu-workers",
        type=int,
        default=None,
        help="processes spoiling pages on the GPU; 0 for CPU only "
        "(default: 3 when CUDA is available)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="render at most this many pages this run",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    config = SyntheticConfig.from_env()
    output: Path = args.output.resolve()
    steps = {
        "plan": [_plan],
        "render": [_render],
        "check": [_check],
        "clean": [_clean],
        "all": [_plan, _render, _check, _clean],
    }[args.command]
    for step in steps:
        status = step(args, config, output)
        if status:
            return status
    return 0


def _plan_path(output: Path) -> Path:
    return output / WORK_DIR / PLAN_NAME


def _plan(
    args: argparse.Namespace, config: SyntheticConfig, output: Path
) -> int:
    path = _plan_path(output)
    if path.exists():
        print(f"plan already exists: {path} (delete it to plan again)")
        return 0
    phrases = PhraseBank.load(args.phrases) if args.phrases else None
    if phrases is not None:
        print(f"phrase bank: {len(phrases)} phrases from {args.phrases}")
    spec = CorpusSpec(
        counts={name: args.per_type for name in args.types},
        seed=args.seed,
        latin_share=args.latin_share,
        profile=(
            AugmentationProfile.archive(args.augment)
            if args.look == "archive"
            else AugmentationProfile.varied(args.augment)
        ),
    )
    started = time.perf_counter()
    plan = plan_corpus(spec, config, phrases)
    write_plan(plan, path)
    print(
        f"planned {len(plan)} pages in {time.perf_counter() - started:.0f}s "
        f"-> {path}"
    )
    for axis, counts in iter_plan_counts(plan):
        shown = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  {axis:<9} {shown}")
    return 0


def _render(
    args: argparse.Namespace, config: SyntheticConfig, output: Path
) -> int:
    plan = read_plan(_plan_path(output))
    todo = pending_documents(plan, output)
    if args.limit is not None:
        # Keep the finished pages in the plan so they are skipped, and cut
        # only what this run takes on.
        keep = {entry.id for entry in todo[: args.limit]}
        plan = [e for e in plan if e.id in keep]
    gpu_workers = args.gpu_workers
    if gpu_workers is None:
        from bitikocr.data.synthetic.augment_gpu import cuda_available

        gpu_workers = 3 if cuda_available() else 0
    print(
        f"rendering {min(len(todo), len(plan))} of {len(todo)} pending pages "
        f"({len(read_plan(_plan_path(output)))} planned) with "
        f"{args.workers} CPU workers and {gpu_workers} GPU workers"
    )
    summary = build_corpus(
        plan,
        output,
        config,
        workers=args.workers,
        gpu_workers=gpu_workers,
        collection=output.name,
        progress=_report_progress,
    )
    print(
        f"rendered {summary.done} pages in {summary.elapsed / 60:.1f} min "
        f"({summary.done / max(summary.elapsed, 1e-9):.1f}/s, "
        f"augmentation on {summary.backend}); "
        f"{summary.skipped} already done, {len(summary.failed)} failed"
    )
    for identifier, reason in summary.failed[:20]:
        print(f"  FAILED {identifier}: {reason.strip().splitlines()[-1]}")
    return 1 if summary.failed else 0


def _report_progress(progress: BuildProgress) -> None:
    eta = progress.remaining
    eta_text = f"{eta / 60:.0f} min" if eta != float("inf") else "?"
    print(
        f"[{time.strftime('%H:%M:%S')}] {progress.done}/{progress.total} "
        f"pages, {progress.failed} failed, {progress.rate:.1f}/s, "
        f"about {eta_text} left",
        flush=True,
    )


def _check(
    args: argparse.Namespace, config: SyntheticConfig, output: Path
) -> int:
    del args, config
    started = time.perf_counter()
    report = check_corpus(output)
    print(
        f"checked {report.pages} pages in "
        f"{time.perf_counter() - started:.0f}s: "
        f"{len(report.problems)} problem(s)"
    )
    for axis, counts in sorted(report.counts.items()):
        shown = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  {axis:<9} {shown}")
    for identifier, problem in report.problems[:30]:
        print(f"  {identifier}: {problem}")
    plan_path = _plan_path(output)
    if plan_path.exists():
        missing = pending_documents(read_plan(plan_path), output)
        if missing:
            print(f"  {len(missing)} planned page(s) are not rendered yet")
            return 1
    return 0 if report.ok else 1


def _clean(
    args: argparse.Namespace, config: SyntheticConfig, output: Path
) -> int:
    del config
    if getattr(args, "keep_work", False):
        print("kept the working files")
        return 0
    work = output / WORK_DIR
    if work.exists():
        shutil.rmtree(work)
        print(f"deleted {work}")
    phrases = getattr(args, "phrases", None)
    if phrases is not None and phrases.exists():
        phrases.unlink()
        print(f"deleted {phrases}")
    for leftover in output.glob("annotations/*.json.part"):
        leftover.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
