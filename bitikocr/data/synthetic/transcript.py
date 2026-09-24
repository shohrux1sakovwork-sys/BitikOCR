"""Spelling a finished page out as text, the way an annotator would.

A page is drawn piece by piece, but it is read by rows: a printed label and
the value a clerk wrote on its rule are one line to whoever transcribes the
page, whichever was put there first. This module turns the measured pieces
of a page into that text.

It follows the conventions the real archive is annotated in, so a synthetic
target and a real one spell the same things the same way:

- Pieces whose ink shares a row are joined left to right with a space.
- A signature is the mark ``<signature>``, at the place it was signed.
- A seal's lettering is wrapped in ``<stamp>`` and ``</stamp>`` on lines of
  their own, or is ``<stamp/>`` when nothing on it can be read.
- A page printed as two facing sheets is read one sheet after the other,
  with a blank line between them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from bitikocr.data.models.geometry import BoundingBox

__all__ = [
    "SIGNATURE_MARK",
    "Column",
    "MarkKind",
    "Piece",
    "PieceKind",
    "compose",
    "stamp_markup",
]

#: What a signature is transcribed as.
SIGNATURE_MARK = "<signature>"

#: Which kind of mark a textless block is.
MarkKind = Literal["signature", "stamp"]

#: A sheet's horizontal extent, ``(left, right)`` in page pixels.
Column = tuple[int, int]

#: What a piece of the page is: printed type, handwriting, or a mark.
PieceKind = Literal["print", "writing", "signature", "stamp"]

#: Share of the shorter of two pieces' bands they must share to sit on one
#: row. Consecutive lines of a letter share none; a label and the value
#: written on its rule share most.
_ROW_OVERLAP = 0.5

#: Share of a piece's height, around its middle, that counts when it is
#: matched to a row. Type sits squarely on its line. Handwriting's
#: ascenders and descenders reach into the rows either side — a value's
#: tail drops into the caption printed under its rule — and a scribble's
#: flourish reaches further still.
_CORE: dict[PieceKind, float] = {
    "print": 1.0,
    "writing": 0.4,
    "signature": 0.5,
    "stamp": 1.0,
}

#: What separates two facing sheets in the text.
_SHEET_BREAK = "\n\n"


@dataclass(frozen=True)
class Piece:
    """One thing on the page that the transcription mentions.

    Args:
        text: What it says: a line of print or writing, the mark
            ``<signature>``, or a seal's lettering.
        box: Where its ink is, in page pixels.
        kind: What the piece is.
        baseline: The y a line was written along, when that is known.
            Type is taken to stand on the foot of its box.
    """

    text: str
    box: BoundingBox
    kind: PieceKind = "writing"
    baseline: float | None = None

    @property
    def line_y(self) -> float | None:
        """The y the piece stands on, or None for a mark."""
        if self.baseline is not None:
            return self.baseline
        if self.kind == "print":
            return float(self.box.bottom)
        return None


def stamp_markup(lettering: str) -> str:
    """Wrap a seal's lettering in the archive's stamp markup.

    Args:
        lettering: The seal's lines, newline separated; may be empty.

    Returns:
        ``<stamp>`` and ``</stamp>`` around the lettering, or ``<stamp/>``
        when there is none.
    """
    lines = [line.strip() for line in lettering.splitlines() if line.strip()]
    if not lines:
        return "<stamp/>"
    return "\n".join(["<stamp>", *lines, "</stamp>"])


def compose(pieces: Sequence[Piece], columns: Sequence[Column] = ()) -> str:
    """Spell a page out in reading order.

    Args:
        pieces: Everything the transcription mentions, in any order.
        columns: The facing sheets a form is printed as, left to right.
            Empty for a single sheet.

    Returns:
        The page's text: rows top to bottom, each read left to right, the
        sheets one after another.
    """
    sheets = [_read_sheet(group) for group in _by_column(pieces, columns)]
    return _SHEET_BREAK.join(sheet for sheet in sheets if sheet)


def _by_column(
    pieces: Sequence[Piece], columns: Sequence[Column]
) -> list[list[Piece]]:
    """Deal the pieces out to the sheet their middle lies on.

    A piece between two sheets, or past either edge, goes to the nearest.
    """
    if not columns:
        return [list(pieces)]
    groups: list[list[Piece]] = [[] for _ in columns]
    for piece in pieces:
        middle = (piece.box.left + piece.box.right) / 2
        nearest = min(
            range(len(columns)),
            key=lambda index: _distance(middle, columns[index]),
        )
        groups[nearest].append(piece)
    return groups


def _distance(x: float, column: Column) -> float:
    """How far a point lies outside a column; zero inside it."""
    left, right = column
    return max(left - x, 0.0, x - right)


def _read_sheet(pieces: Sequence[Piece]) -> str:
    """Spell one sheet out, row by row, with its seals where they sit.

    Pieces are taken top to bottom. Each joins the row it shares most with,
    or starts a row of its own when it shares too little with any.
    """
    rows: list[list[Piece]] = []
    for piece in sorted(pieces, key=_middle):
        best = max(
            (row for row in rows if _shared(row[0], piece) >= _ROW_OVERLAP),
            key=lambda row: _shared(row[0], piece),
            default=None,
        )
        if best is None:
            rows.append([piece])
        else:
            best.append(piece)
    rows.sort(key=lambda row: _middle(row[0]))
    return "\n".join(_read_row(row) for row in rows)


def _read_row(row: Sequence[Piece]) -> str:
    """Spell one row out, left to right."""
    if row[0].kind == "stamp":
        return stamp_markup(row[0].text)
    ordered = sorted(row, key=lambda piece: piece.box.left)
    return " ".join(piece.text for piece in ordered if piece.text)


def _shared(anchor: Piece, piece: Piece) -> float:
    """How much of a row a piece shares, as a share of the shorter band.

    A piece is compared with the row's first piece only, so a tall piece
    can join a row but never chain two rows together. A seal shares its
    row with nothing.

    Two lines that both stand on a known line are on one row when they
    stand on the same one, give or take half a line: a value and the label
    printed on its rule. Anything else is matched by how much of the
    middle of its ink it shares.
    """
    if "stamp" in (anchor.kind, piece.kind):
        return 0.0
    top, bottom = _band(anchor)
    other_top, other_bottom = _band(piece)
    shorter = min(bottom - top, other_bottom - other_top)
    if shorter <= 0:
        return 0.0

    anchor_y, piece_y = anchor.line_y, piece.line_y
    if anchor_y is not None and piece_y is not None:
        printed = [p.box.height for p in (anchor, piece) if p.kind == "print"]
        scale = min(printed) if printed else shorter
        return 1.0 - abs(anchor_y - piece_y) / scale
    return (min(bottom, other_bottom) - max(top, other_top)) / shorter


def _band(piece: Piece) -> tuple[float, float]:
    """The vertical extent a piece is matched to a row by."""
    top, bottom = float(piece.box.top), float(piece.box.bottom)
    trim = (bottom - top) * (1 - _CORE[piece.kind]) / 2
    return top + trim, bottom - trim


def _middle(piece: Piece) -> float:
    """The vertical middle of a piece's ink."""
    return (piece.box.top + piece.box.bottom) / 2
