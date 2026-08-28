"""Command line entry point.

Configuration is loaded here, at the edge, and passed down explicitly. No
module below this one reads the environment on its own.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from bitikocr.config import SyntheticConfig
from bitikocr.synthetic.dataset import generate_dataset
from bitikocr.synthetic.fonts import FontLibrary
from bitikocr.synthetic.generators import (
    available_document_types,
    create_generator,
)
from bitikocr.synthetic.sample_data import sample_fields_for
from bitikocr.synthetic.templates import FormTemplate

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

    generate = synth.add_parser(
        "generate", help="render a batch of synthetic documents"
    )
    generate.add_argument(
        "document_type",
        choices=available_document_types(),
        help="which document to render",
    )
    generate.add_argument(
        "-n",
        "--count",
        type=int,
        default=5,
        help="how many samples to generate (default: %(default)s)",
    )
    generate.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="where to write samples (default: <output dir>/<document type>)",
    )
    generate.add_argument(
        "--seed",
        type=int,
        default=None,
        help="make the whole run reproducible",
    )
    generate.add_argument(
        "--template",
        default=None,
        help="form variant to fill, for document types that use one",
    )
    generate.add_argument(
        "--font",
        type=Path,
        default=None,
        help="force one handwriting font instead of sampling",
    )
    generate.add_argument(
        "--fonts-dir",
        type=Path,
        default=None,
        help="handwriting fonts to sample from",
    )
    generate.add_argument(
        "--boxes",
        action="store_true",
        help="also write a _boxes.png overlay per sample",
    )
    generate.set_defaults(handler=_run_generate)

    list_fonts = synth.add_parser(
        "list-fonts", help="show the handwriting fonts that will be sampled"
    )
    list_fonts.add_argument(
        "--fonts-dir",
        type=Path,
        default=None,
        help="handwriting fonts to inspect",
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

    return parser


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


def _run_generate(args: argparse.Namespace, config: SyntheticConfig) -> int:
    """Render a batch of documents and report where they went."""
    generator = create_generator(
        args.document_type, config, args.font, args.template
    )
    output_dir = args.output_dir or config.output_dir / args.document_type

    summary = generate_dataset(
        generator=generator,
        field_sets=list(sample_fields_for(args.document_type)),
        count=args.count,
        output_dir=output_dir,
        seed=args.seed,
        draw_boxes=args.boxes,
    )
    print(f"Wrote {len(summary)} samples to {summary.output_dir.resolve()}")
    return 0


def _run_list_fonts(args: argparse.Namespace, config: SyntheticConfig) -> int:
    """Print each available handwriting font and its measured proportions."""
    del args  # The fonts directory already reached us through the config.
    library = FontLibrary.from_directory(config.fonts_dir)
    print(f"{len(library)} font(s) in {config.fonts_dir}")
    for font in library:
        print(
            f"  {font.name:<32} glyphs={len(font.codepoints):<6}"
            f" x-height={font.xheight_ratio:.2f}"
            f" width={font.width_ratio:.2f}"
            f" stroke={font.stroke_ratio:.2f}"
        )
    return 0


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
        print(f"  fields: {', '.join(template.field_names)}")
        marks = [
            label
            for label, area in (
                ("seal", template.seal),
                ("serial", template.serial),
                ("signature", template.signature),
            )
            if area is not None
        ]
        print(f"  marks:  {', '.join(marks) or 'none'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
