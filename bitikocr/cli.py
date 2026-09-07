"""Command line entry point.

Configuration is loaded here, at the edge, and passed down explicitly. No
module below this one reads the environment on its own.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bitikocr.config import SyntheticConfig
from bitikocr.data.synthetic.augment import AugmentationProfile
from bitikocr.data.synthetic.dataset import (
    DEFAULT_ID_PREFIX,
    DatasetLayout,
    DatasetSummary,
    read_records,
    render_records,
    write_records,
)
from bitikocr.data.synthetic.fonts import FontInfo, FontLibrary
from bitikocr.data.synthetic.generators import (
    DEFAULT_INK_STRENGTH,
    available_document_types,
    create_generator,
)
from bitikocr.data.synthetic.records import (
    DEFAULT_LATIN_SHARE,
    DocumentRecord,
    sample_records,
)
from bitikocr.data.synthetic.scripts import SCRIPTS
from bitikocr.data.synthetic.templates import FormTemplate

__all__ = ["main"]

logger = logging.getLogger("bitikocr")

_LOG_FORMAT = "%(levelname)s %(name)s: %(message)s"


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser.

    Returns:
        The parser for the ``bitikocr`` executable.
    """
    parser = argparse.ArgumentParser(
        prog="bitikocr",
        description="Uzbek handwritten text recognition toolkit.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log every generated sample",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    synth = commands.add_parser(
        "synth", help="generate synthetic training data"
    ).add_subparsers(dest="synth_command", required=True)

    _add_facts_command(synth)
    _add_render_command(synth)
    _add_generate_command(synth)
    _add_listing_commands(synth)
    return parser


def _add_document_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments that name what to generate."""
    parser.add_argument(
        "document_type",
        choices=available_document_types(),
        help="which document to generate",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=30,
        help="how many documents (default: %(default)s)",
    )
    parser.add_argument(
        "--script",
        choices=SCRIPTS,
        default=None,
        help="write every document in one alphabet (default: both)",
    )
    parser.add_argument(
        "--latin-share",
        type=float,
        default=DEFAULT_LATIN_SHARE,
        metavar="SHARE",
        help=(
            "share of documents written in Latin when no script is forced "
            "(default: %(default)s, low because most fonts are Cyrillic)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="make the whole run reproducible",
    )


def _add_id_argument(parser: argparse.ArgumentParser) -> None:
    """Add the argument naming documents in a dataset."""
    parser.add_argument(
        "--id-prefix",
        default=DEFAULT_ID_PREFIX,
        help=(
            "what to call documents in this set; namespace it per type if "
            "several sets are merged (default: %(default)s)"
        ),
    )


def _add_output_argument(parser: argparse.ArgumentParser) -> None:
    """Add the dataset directory argument."""
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="dataset directory (default: <output dir>/<document type>)",
    )


def _add_render_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the arguments that control how pages are drawn."""
    parser.add_argument(
        "--template",
        default=None,
        help="form variant to fill, for document types that use one",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=None,
        help="force one handwriting font instead of sampling",
    )
    parser.add_argument(
        "--fonts-dir",
        type=Path,
        default=None,
        help="handwriting fonts to sample from",
    )
    parser.add_argument(
        "--augment",
        type=float,
        default=1.0,
        metavar="STRENGTH",
        help="how hard to spoil each page, 0 to disable (default: %(default)s)",
    )
    parser.add_argument(
        "--ink",
        type=float,
        default=DEFAULT_INK_STRENGTH,
        metavar="STRENGTH",
        help=(
            "how heavily the pen writes; raise it if a hand comes out too "
            "faint (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--boxes",
        action="store_true",
        help="also write a box overlay per page, under previews/",
    )


def _add_facts_command(synth: argparse._SubParsersAction[Any]) -> None:
    """Register ``synth facts``."""
    facts = synth.add_parser(
        "facts",
        help="sample what each document says, without rendering anything",
    )
    _add_document_arguments(facts)
    _add_output_argument(facts)
    _add_id_argument(facts)
    facts.set_defaults(handler=_run_facts)


def _add_render_command(synth: argparse._SubParsersAction[Any]) -> None:
    """Register ``synth render``."""
    render = synth.add_parser(
        "render", help="draw the pages for a dataset that has its facts"
    )
    render.add_argument(
        "output_dir",
        type=Path,
        help="dataset directory holding a facts/ directory",
    )
    _add_render_arguments(render)
    _add_id_argument(render)
    render.set_defaults(handler=_run_render)


def _add_generate_command(synth: argparse._SubParsersAction[Any]) -> None:
    """Register ``synth generate``, which does both stages at once."""
    generate = synth.add_parser(
        "generate", help="sample records and render them in one go"
    )
    _add_document_arguments(generate)
    _add_output_argument(generate)
    _add_render_arguments(generate)
    _add_id_argument(generate)
    generate.set_defaults(handler=_run_generate)


def _add_listing_commands(synth: argparse._SubParsersAction[Any]) -> None:
    """Register the commands that only report what is available."""
    list_fonts = synth.add_parser(
        "list-fonts", help="show the handwriting fonts that will be sampled"
    )
    list_fonts.add_argument(
        "--fonts-dir", type=Path, default=None, help="fonts to inspect"
    )
    list_fonts.set_defaults(handler=_run_list_fonts)

    list_types = synth.add_parser(
        "list-types", help="show the document types that can be generated"
    )
    list_types.set_defaults(handler=_run_list_types)

    list_templates = synth.add_parser(
        "list-templates", help="show the form variants that can be filled"
    )
    list_templates.set_defaults(handler=_run_list_templates)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface.

    Args:
        argv: Arguments to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code: 0 on success, 1 on a reported failure.
    """
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format=_LOG_FORMAT,
    )

    config = _config_from_args(args)
    try:
        return int(args.handler(args, config))
    except (FileNotFoundError, KeyError, ValueError) as error:
        logger.error("%s", error)
        return 1


def _config_from_args(args: argparse.Namespace) -> SyntheticConfig:
    """Load configuration, letting command line flags win over the environment."""
    config = SyntheticConfig.from_env()
    fonts_dir = getattr(args, "fonts_dir", None)
    if fonts_dir is not None:
        config = dataclasses.replace(config, fonts_dir=fonts_dir)
    return config


def _dataset_dir(args: argparse.Namespace, config: SyntheticConfig) -> Path:
    """Resolve where a dataset lives."""
    return args.output_dir or config.dataset_dir(args.document_type)


def _augmentation(args: argparse.Namespace) -> AugmentationProfile | None:
    """Build the augmentation profile the flags ask for."""
    if args.augment <= 0:
        return None
    return dataclasses.replace(
        AugmentationProfile(), strength=float(args.augment)
    )


# -- commands --------------------------------------------------------------


def _sample(args: argparse.Namespace) -> list[DocumentRecord]:
    """Sample the batch of records the arguments describe."""
    return sample_records(
        args.document_type,
        args.count,
        random.Random(args.seed),
        args.script,
        args.latin_share,
    )


def _run_facts(args: argparse.Namespace, config: SyntheticConfig) -> int:
    """Sample what each document says and write it, rendering nothing."""
    records = _sample(args)
    layout = DatasetLayout(_dataset_dir(args, config))
    write_records(records, layout, args.id_prefix)
    print(f"Wrote {len(records)} facts files to {layout.facts.resolve()}")
    return 0


def _run_render(args: argparse.Namespace, config: SyntheticConfig) -> int:
    """Draw the pages for a dataset that already has its facts."""
    layout = DatasetLayout(args.output_dir)
    records = read_records(layout)
    if not records:
        raise ValueError(f"{layout.facts} holds no facts files")

    document_type = records[0].document_type
    generator = create_generator(
        document_type, config, args.font, args.template, args.ink
    )
    summary = render_records(
        generator=generator,
        records=records,
        output_dir=layout.root,
        augmentation=_augmentation(args),
        draw_boxes=args.boxes,
        prefix=args.id_prefix,
    )
    _report(summary)
    return 0


def _run_generate(args: argparse.Namespace, config: SyntheticConfig) -> int:
    """Sample the facts and render them in one go."""
    records = _sample(args)
    layout = DatasetLayout(_dataset_dir(args, config))
    write_records(records, layout, args.id_prefix)

    generator = create_generator(
        args.document_type, config, args.font, args.template, args.ink
    )
    summary = render_records(
        generator=generator,
        records=records,
        output_dir=layout.root,
        augmentation=_augmentation(args),
        draw_boxes=args.boxes,
        prefix=args.id_prefix,
    )
    _report(summary)
    return 0


def _report(summary: DatasetSummary) -> None:
    """Print what a render produced."""
    layout = summary.layout
    print(f"Wrote {len(summary)} pages to {layout.root.resolve()}")
    print(f"  facts:       {layout.facts.name}/")
    print(f"  images:      {layout.images.name}/")
    print(f"  annotations: {layout.annotations.name}/")
    print(f"  index:       {layout.index.name}")
    if summary.skipped:
        print(f"  skipped:     {len(summary.skipped)} record(s)")


def _run_list_fonts(args: argparse.Namespace, config: SyntheticConfig) -> int:
    """Print each available handwriting font and what it can write."""
    del args  # The fonts directory already reached us through the config.
    library = FontLibrary.from_directory(config.fonts_dir)
    print(f"{len(library)} font(s) in {config.fonts_dir}")
    for font in library:
        scripts = ", ".join(_font_scripts(font)) or "none"
        print(
            f"  {font.name:<20} glyphs={len(font.codepoints):<5}"
            f" scripts={scripts:<17}"
            f" x-height={font.xheight_ratio:.2f}"
            f" width={font.width_ratio:.2f}"
        )
    return 0


def _font_scripts(font: FontInfo) -> list[str]:
    """Return which alphabets a font covers."""
    covers = []
    if font.can_render("abcdefghijklmnopqrstuvwxyz"):
        covers.append("latin")
    if font.can_render("абвгдеёжзийклмнопрстуфхцчшщъыьэюя"):
        covers.append("cyrillic")
    if font.can_render("0123456789"):
        covers.append("digits")
    return covers


def _run_list_types(args: argparse.Namespace, config: SyntheticConfig) -> int:
    """Print the registered document types."""
    del args, config  # Listing the registry needs neither.
    for document_type in available_document_types():
        print(document_type)
    return 0


def _run_list_templates(
    args: argparse.Namespace, config: SyntheticConfig
) -> int:
    """Print each form variant and the fields it defines."""
    del args  # The layouts directory already reached us through the config.
    names = config.available_layouts()
    if not names:
        print(f"No form layouts in {config.layouts_dir}")
        return 0

    for name in names:
        template = FormTemplate.load(config.layout(name))
        width, height = template.native_size
        print(f"{template.name}  ({width}x{height}, {template.background})")
        print(f"  fields:   {', '.join(template.field_names)}")
        if template.printed:
            print(f"  printed:  {', '.join(template.printed_names)}")
        marks = [
            label
            for label, area in (
                ("seal", template.seal),
                ("signature", template.signature),
            )
            if area is not None
        ]
        if template.keep_out:
            marks.append(
                "keep-out: " + ", ".join(a.name for a in template.keep_out)
            )
        print(f"  marks:    {', '.join(marks) or 'none'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
