"""The neighbourhood committee's certificate — ``Маълумотнома``.

A mahalla committee certifies that a resident lives at an address and lists
the household, on a small form it has printed for itself: the title, a few
sentences with blanks for the name and address, numbered rows for the
family, and lines for the chairman and the secretary. It carries the
committee's rectangular stamp, with the outgoing number and date written
in, and its round seal over the chairman's signature.

There is no single blank to scan: every committee prints its own, worded a
little differently and cut to its own size. So this generator draws a
fresh blank for each page, measures it as it draws, and hands the result to
:class:`~bitikocr.data.synthetic.generators.form.FormGenerator`, which fills
and labels it exactly as it fills a scanned certificate.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from PIL import Image, ImageDraw, ImageFont

from bitikocr.config import SyntheticConfig
from bitikocr.data.models.geometry import BoundingBox
from bitikocr.data.synthetic.effects import (
    SEAL_COLORS,
    box_stamp_size,
    draw_box_stamp,
)
from bitikocr.data.synthetic.fonts import FontInfo
from bitikocr.data.synthetic.generators.ariza import (
    read_stamp,
    write_in_stamp,
)
from bitikocr.data.synthetic.generators.base import (
    DEFAULT_INK_STRENGTH,
    DocumentGenerator,
    FieldValues,
)
from bitikocr.data.synthetic.generators.form import FormGenerator, FormOptions
from bitikocr.data.synthetic.hand import Hand
from bitikocr.data.synthetic.layout import Page
from bitikocr.data.synthetic.style import HandwritingStyle
from bitikocr.data.synthetic.system_fonts import find_print_font
from bitikocr.data.synthetic.templates import (
    FieldGeometry,
    FormTemplate,
    LineSegment,
    MarkArea,
    PrintedLine,
)

__all__ = ["FIELD_NAMES", "MAX_FAMILY_ROWS", "MalumotnomaGenerator"]

#: Most rows a committee's blank leaves for the household.
MAX_FAMILY_ROWS = 14

#: Everything a certificate's record can carry. The blank drawn for a page
#: rules only some of them, so this is the whole vocabulary rather than one
#: page's fields.
FIELD_NAMES: tuple[str, ...] = (
    "holder",
    "holder_birth_year",
    "town",
    "mahalla",
    "street",
    "house",
    *(f"member_{index}" for index in range(1, 15)),
    "purpose",
    "form_number",
    "chairman_name",
    "secretary_name",
    "reg_number",
    "reg_date",
    "reg_stamp",
    "stamp_ring",
    "stamp_center",
    "blank",
)

#: The officials' names, which the committee has typeset on its blank.
_PRINTED_NAMES = ("chairman_name", "secretary_name")

#: Page width in pixels: A4 at 200 dpi, like every other generated page.
_PAGE_WIDTH = 1654
_A4_HEIGHT = 2339

#: Share of blanks printed at the top of a whole A4 sheet; the rest are cut
#: to the form's own height, as a committee's small forms are.
_FULL_SHEET_SHARE = 0.45

#: How the top of the form is taken up: the committee's rectangular stamp,
#: its name printed as a letterhead, or nothing but the title.
_HEADER_WEIGHTS = {"stamp": 0.6, "letterhead": 0.25, "plain": 0.15}

#: Print faces a committee's blank is set in, regular then bold, tried in
#: order; the stamp font stands in when the system has none of them.
_PRINT_FACES = (
    ("LiberationSerif-Regular.ttf", "LiberationSerif-Bold.ttf"),
    ("DejaVuSerif.ttf", "DejaVuSerif-Bold.ttf"),
    ("DejaVuSerifCondensed.ttf", "DejaVuSerifCondensed-Bold.ttf"),
    ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
    ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
)
_FONT_DIRS = (
    Path("/usr/share/fonts/truetype/liberation"),
    Path("/usr/share/fonts/truetype/dejavu"),
)

_INK = (28, 28, 32)
_PAPER = (252, 252, 250)

#: The clerk's hand inside the stamp, relative to the stamp's lettering.
_ENTRY_SCALE = 1.1

#: How much closer the rows are set on each redraw of a form too long for
#: A4.
_ROW_SQUEEZES = (1.0, 0.88, 0.78, 0.7)

#: The clerk's nominal hand size as a share of the blank's row height.
#: Real clerks write large, often up to the rule above.
_HAND_TO_ROW = (0.62, 0.85)

#: Letters the blank must be able to print: Uzbek Cyrillic's own.
_UZBEK_LETTERS = "ҲҳҚқҒғЎўЪъ"


@dataclass(frozen=True)
class _Printed:
    """Text the blank prints."""

    text: str


@dataclass(frozen=True)
class _Blank:
    """A ruled blank for a field; ``share`` of the text width, or the rest."""

    name: str
    share: float | None = None
    is_numeric: bool = False


_Token = _Printed | _Blank


@dataclass(frozen=True)
class _StampPlan:
    """Where the committee's rectangular stamp goes, decided with the blank."""

    left: int
    top: int
    type_size: int
    blank_width: int


class MalumotnomaGenerator(FormGenerator):
    """Draw a committee's certificate blank and fill it in by hand.

    Args:
        config: Paths to the fonts to use.
        font_path: Force a specific handwriting font instead of sampling.
        options: Filling options; a template name is not accepted.
        ink_strength: How heavily the pen writes.
    """

    name: ClassVar[str] = "malumotnoma"
    default_template: ClassVar[str] = "malumotnoma"
    layout: ClassVar[str] = "single_column"

    def __init__(
        self,
        config: SyntheticConfig,
        font_path: Path | str | None = None,
        options: FormOptions | None = None,
        ink_strength: float = DEFAULT_INK_STRENGTH,
    ) -> None:
        # The blank is drawn per page, so there is no layout file to load.
        DocumentGenerator.__init__(self, config, font_path, ink_strength)
        # The blank is drawn at the page's own resolution, where a scanned
        # certificate is doubled.
        self.options = options or FormOptions(scale=1.0)
        if self.options.template is not None:
            raise ValueError("A malumotnoma's blank is drawn, not chosen.")
        # Until a page draws its blank, the template says only what every
        # blank prints: the officials' names.
        self.template = FormTemplate(
            name=self.default_template,
            background="",
            native_size=(_PAGE_WIDTH, _A4_HEIGHT),
            fields=(),
            printed=tuple(
                MarkArea(name, BoundingBox(0, 0, 1, 1))
                for name in _PRINTED_NAMES
            ),
        )
        self._stamp_plan: _StampPlan | None = None
        self._row_height = 70

    @classmethod
    def with_template(
        cls,
        config: SyntheticConfig,
        font_path: Path | str | None,
        template: str,
        ink_strength: float = DEFAULT_INK_STRENGTH,
    ) -> FormGenerator:
        """Refuse a template: every page draws its own blank."""
        raise ValueError("A malumotnoma's blank is drawn, not chosen.")

    @property
    def field_names(self) -> tuple[str, ...]:
        """Every field a certificate's record can carry."""
        return FIELD_NAMES

    def _collect_text(self, fields: FieldValues) -> str:
        """Join what the clerk writes by hand, for font coverage checks."""
        return " ".join(
            value
            for name, value in fields.items()
            if isinstance(value, str)
            and name not in (*_PRINTED_NAMES, "stamp_ring")
        )

    def _as_form_style(  # type: ignore[override]
        self,
        style: HandwritingStyle,
        rng: random.Random,
        scale: float,
        style_overrides: Any,
    ) -> HandwritingStyle:
        """Size the clerk's hand to this blank's rows.

        A certificate's hand is sized for its doubled scan; a drawn blank is
        at page resolution, so the hand follows the rows instead.
        """
        style = FormGenerator._as_form_style(style, rng, 2.0, None)
        style = style.replace(
            font_size=int(self._row_height * rng.uniform(*_HAND_TO_ROW))
        )
        return style.replace(**style_overrides) if style_overrides else style

    # -- the blank ---------------------------------------------------------

    def _prepare_blank(
        self, fields: FieldValues, rng: random.Random
    ) -> tuple[FormTemplate, Image.Image, str]:
        """Draw this committee's blank and measure it."""
        blank = fields.get("blank") or {}
        regular, bold = _print_faces(self.library)
        body_size = int(_PAGE_WIDTH * rng.uniform(0.0150, 0.0185))
        row_pitch = int(body_size * rng.uniform(2.5, 3.1))
        left = int(_PAGE_WIDTH * rng.uniform(0.07, 0.11))
        right = int(_PAGE_WIDTH * rng.uniform(0.90, 0.95))

        # A long household can push the form past A4; it is drawn again with
        # the same choices and closer rows, as a print shop would set it.
        state = rng.getstate()
        for squeeze in _ROW_SQUEEZES:
            rng.setstate(state)
            pitch = int(row_pitch * squeeze)
            self._row_height = pitch
            canvas = Image.new(
                "RGB", (_PAGE_WIDTH, int(_A4_HEIGHT * 1.4)), _PAPER
            )
            drawer = _BlankDrawer(
                draw=ImageDraw.Draw(canvas),
                font=ImageFont.truetype(str(regular), body_size),
                left=left,
                right=right,
                pitch=pitch,
            )

            top = int(_PAGE_WIDTH * rng.uniform(0.03, 0.06))
            header = rng.choices(
                list(_HEADER_WEIGHTS), weights=list(_HEADER_WEIGHTS.values())
            )[0]
            title_left, title_right = left, right
            self._stamp_plan = None
            header_bottom = top
            if header == "stamp" and fields.get("reg_stamp"):
                type_size = int(body_size * rng.uniform(0.75, 0.95))
                blank_width = int(type_size * rng.uniform(4.0, 6.0))
                width, height = box_stamp_size(
                    list(fields["reg_stamp"]),
                    type_size,
                    blank_width,
                    find_print_font(self.library),
                )
                stamp_left = max(8, left - int(_PAGE_WIDTH * 0.03))
                self._stamp_plan = _StampPlan(
                    stamp_left, top, type_size, blank_width
                )
                header_bottom = top + height
            elif header == "letterhead":
                header_bottom = drawer.letterhead(
                    _letterhead_lines(blank, rng),
                    ImageFont.truetype(str(regular), int(body_size * 0.85)),
                    top,
                )

            title_font = ImageFont.truetype(
                str(bold), int(body_size * rng.uniform(1.35, 1.8))
            )
            # The title goes below a stamp or letterhead, never beside it. The
            # page is read row by row, so a title beside the stamp would be read
            # before it and one beside a letterhead would merge into its lines;
            # the archive's transcripts read the top-left block first.
            title_baseline = header_bottom + int(title_font.size * 1.6)
            drawer.title(
                title_font,
                title_left,
                title_right,
                title_baseline,
                spaced=rng.random() < 0.6,
                number_blank=bool(fields.get("form_number")),
            )

            y = max(header_bottom, title_baseline) + int(pitch * 1.3)
            for line in _body_lines(fields, blank, rng):
                y = drawer.line(line, y)
            y = drawer.signature_block(
                _chairman_label(blank, rng),
                rng.choice(("Котиба:", "Котиби:", "Котиба")),
                y + int(pitch * 0.4),
                rng,
            )

            used = y + int(pitch * rng.uniform(0.5, 1.5))
            if used <= _A4_HEIGHT:
                break

        if rng.random() < _FULL_SHEET_SHARE and used <= _A4_HEIGHT:
            height = _A4_HEIGHT
        else:
            height = max(used, int(_PAGE_WIDTH * 0.6))
        image = canvas.crop((0, 0, _PAGE_WIDTH, height))

        self.template = FormTemplate(
            name=self.default_template,
            background="",
            native_size=(_PAGE_WIDTH, height),
            fields=tuple(drawer.fields),
            seal=drawer.seal,
            printed=tuple(drawer.printed_areas),
            signature=drawer.signature,
            printed_languages=("uz-cyrillic",),
            printed_text=tuple(drawer.printed),
        )
        return self.template, image, f"drawn:{header}"

    # -- the committee's stamp ---------------------------------------------

    def _decorate(
        self,
        page: Page,
        fields: FieldValues,
        style: HandwritingStyle,
        rng: random.Random,
    ) -> dict[str, Any]:
        """Press the committee's rectangular stamp and write it in."""
        plan = self._stamp_plan
        rows = list(fields.get("reg_stamp") or [])
        if plan is None or not rows:
            return {}
        box, blanks = draw_box_stamp(
            page=page.image,
            top_left=(plan.left, plan.top),
            rows=rows,
            type_size=plan.type_size,
            blank_width=plan.blank_width,
            color=rng.choice(SEAL_COLORS),
            font_path=find_print_font(self.library),
            rng=rng,
        )
        entries = [
            str(fields[name])
            for name in ("reg_date", "reg_number")
            if fields.get(name)
        ]
        info = self.library.pick(" ".join(entries), rng)
        hand = Hand(info, int(plan.type_size * _ENTRY_SCALE), style, rng)
        ink = style.ink
        boxes = [box]
        for block, blank, value in zip(
            ("reg_entry_date", "reg_entry_number"), blanks, entries
        ):
            boxes.append(write_in_stamp(page, block, blank, value, hand, ink))
        page.add_block(
            "stamp",
            read_stamp(rows, entries[: len(blanks)]),
            BoundingBox.union(boxes),
        )
        written = dict(zip(("reg_date", "reg_number"), entries[: len(blanks)]))
        return {"reg_stamp": rows, **written}


@dataclass
class _BlankDrawer:
    """Print a blank line by line, recording what it measures as it goes."""

    draw: ImageDraw.ImageDraw
    font: Any
    left: int
    right: int
    pitch: int

    def __post_init__(self) -> None:
        self.fields: list[FieldGeometry] = []
        self.printed: list[PrintedLine] = []
        self.printed_areas: list[MarkArea] = []
        self.seal: MarkArea | None = None
        self.signature: MarkArea | None = None
        self.space = int(self.font.size * 0.4)

    def text(self, text: str, x: int, baseline: int, font: Any = None) -> int:
        """Print ``text`` on a baseline and record it; return its right."""
        font = font or self.font
        width = int(font.getlength(text))
        self.draw.text((x, baseline), text, font=font, fill=_INK, anchor="ls")
        self.printed.append(
            PrintedLine(
                text,
                BoundingBox(
                    left=x,
                    top=baseline - int(font.size * 0.95),
                    right=x + width,
                    bottom=baseline + int(font.size * 0.3),
                ),
            )
        )
        return x + width

    def rule(self, blank: _Blank, x: int, x_end: int, baseline: int) -> None:
        """Rule a blank and record it as a field's line."""
        self.draw.line(
            [(x, baseline + 3), (x_end, baseline + 3)], fill=_INK, width=2
        )
        self.fields.append(
            FieldGeometry(
                blank.name,
                (LineSegment(baseline, x, x_end),),
                is_numeric=blank.is_numeric,
            )
        )

    def line(self, tokens: Sequence[_Token], baseline: int) -> int:
        """Print one line of the body, wrapping where it runs out of room.

        Returns:
            The baseline of the next line.
        """
        width = self.right - self.left
        x = self.left
        for token in tokens:
            if isinstance(token, _Printed):
                needed = int(self.font.getlength(token.text))
            else:
                needed = int(width * (token.share or 0.18))
            if x > self.left and x + needed > self.right:
                baseline += self.pitch
                x = self.left
            if isinstance(token, _Printed):
                x = self.text(token.text, x, baseline) + self.space
                continue
            end = (
                self.right
                if token.share is None
                else min(self.right, x + needed)
            )
            self.rule(token, x, end, baseline)
            x = end + self.space
        return baseline + self.pitch

    def letterhead(self, lines: Sequence[str], font: Any, top: int) -> int:
        """Print the committee's name as a letterhead; return its bottom."""
        baseline = top + font.size
        for text in lines:
            self.text(text, self.left, baseline, font)
            baseline += int(font.size * 1.35)
        return baseline

    def title(
        self,
        font: Any,
        left: int,
        right: int,
        baseline: int,
        spaced: bool,
        number_blank: bool,
    ) -> None:
        """Print the title centred, often letter-spaced, and its number."""
        title = "МАЪЛУМОТНОМА"
        shown = " ".join(title) if spaced else title
        width = int(font.getlength(shown))
        x = left + max(0, (right - left - width) // 2)
        self.draw.text((x, baseline), shown, font=font, fill=_INK, anchor="ls")
        # The page is read as the word, however widely it is set.
        self.printed.append(
            PrintedLine(
                title,
                BoundingBox(x, baseline - font.size, x + width, baseline + 4),
            )
        )
        if number_blank:
            x = self.text("№", x + width + self.space * 2, baseline)
            self.rule(
                _Blank("form_number", is_numeric=True),
                x + self.space,
                min(self.right, x + self.space + int(font.size * 3.5)),
                baseline,
            )

    def signature_block(
        self,
        chairman_label: str,
        secretary_label: str,
        baseline: int,
        rng: random.Random,
    ) -> int:
        """Print the officials' lines and mark where they sign and stamp.

        Returns:
            The baseline below the block.
        """
        baseline += self.pitch
        label_end = self.text(chairman_label, self.left, baseline)
        name_left = int(self.left + (self.right - self.left) * 0.66)
        sign_left = label_end + self.space * 2
        self.signature = MarkArea(
            "signature",
            BoundingBox(
                sign_left,
                baseline - int(self.pitch * 0.9),
                max(sign_left + 60, name_left - self.space),
                baseline + int(self.pitch * 0.3),
            ),
        )
        radius = int(self.pitch * rng.uniform(1.3, 1.6))
        centre_x = int(sign_left + (name_left - sign_left) * 0.4)
        self.seal = MarkArea(
            "seal",
            BoundingBox(
                centre_x - radius,
                baseline - radius,
                centre_x + radius,
                baseline + radius,
            ),
        )
        self.printed_areas.append(
            MarkArea(
                "chairman_name",
                BoundingBox(
                    name_left,
                    baseline - self.font.size,
                    self.right,
                    baseline + int(self.font.size * 0.35),
                ),
            )
        )
        baseline += self.pitch
        self.text(secretary_label, self.left, baseline)
        self.printed_areas.append(
            MarkArea(
                "secretary_name",
                BoundingBox(
                    name_left,
                    baseline - self.font.size,
                    self.right,
                    baseline + int(self.font.size * 0.35),
                ),
            )
        )
        return baseline + self.pitch


def _body_lines(
    fields: FieldValues, blank: dict[str, Any], rng: random.Random
) -> list[list[_Token]]:
    """Word the blank's sentences, as one committee's print shop did."""
    settlement = blank.get("settlement", "шаҳар")
    lines: list[list[_Token]] = []
    opening = rng.randrange(3)
    if opening == 0:
        lines.append(
            [
                _Printed("Берилди ушбу маълумотнома фуқаро"),
                _Blank("holder"),
            ]
        )
        lines.append([_Printed("га шу ҳақдаки, у ҳақиқатдан ҳам")])
    elif opening == 1:
        lines.append(
            [
                _Printed(rng.choice(("Берилди", "Берилади"))),
                _Blank("holder", 0.6),
                _Printed(rng.choice(("га шу ҳақдаки,", "га шу ҳақдаким,"))),
            ]
        )
        lines.append([_Printed("ҳақиқатдан ҳам")])
    else:
        lines.append(
            [
                _Printed("Берилди ушбу маълумотнома"),
                _Blank("holder_birth_year", 0.12, is_numeric=True),
                _Printed("йилда туғилган фуқаро"),
            ]
        )
        lines.append([_Blank("holder")])
        lines.append([_Printed("га шу ҳақдаки, у ҳақиқатдан ҳам")])

    lines[-1].extend(
        [
            _Blank("town", 0.2),
            _Printed(settlement),
            _Blank("mahalla", 0.22),
            _Printed("маҳалласи"),
        ]
    )
    lines.append(
        [
            _Blank("street", 0.3),
            _Printed(rng.choice(("кўча", "кўчаси"))),
            _Blank("house", 0.08, is_numeric=True),
            _Printed(
                rng.choice(("уйда яшайди.", "уйда оила аъзолари билан яшайди."))
            ),
        ]
    )
    if opening != 2:
        lines.append(
            [
                _Blank("holder_birth_year", 0.12, is_numeric=True),
                _Printed("йилда туғилган."),
            ]
        )

    lines.append(
        [
            _Printed(
                rng.choice(
                    (
                        "Унинг оила аъзолари:",
                        "Оила аъзолари:",
                        "Унинг хонадонида яшовчи оила аъзолари:",
                    )
                )
            )
        ]
    )
    members = sum(
        1
        for index in range(1, MAX_FAMILY_ROWS + 1)
        if fields.get(f"member_{index}")
    )
    rows = max(members, rng.choice((3, 4, 5, 6, 7, 8, 10, 12, 14)))
    for index in range(1, min(rows, MAX_FAMILY_ROWS) + 1):
        lines.append([_Printed(f"{index}."), _Blank(f"member_{index}")])

    if rng.random() < 0.6:
        lines.append(
            [
                _Printed("Маълумотнома"),
                _Blank("purpose", 0.4),
                _Printed("учун берилди."),
            ]
        )
    else:
        lines.append(
            [
                _Printed(
                    rng.choice(
                        (
                            (
                                "Маълумотнома талаб қилинган жойга тақдим "
                                "этиш учун берилди."
                            ),
                            "Маълумотнома сўралган жойга берилди.",
                        )
                    )
                )
            ]
        )
    return lines


def _letterhead_lines(blank: dict[str, Any], rng: random.Random) -> list[str]:
    """Word the committee's printed letterhead."""
    lines = []
    if rng.random() < 0.7 and blank.get("region"):
        lines.append(f"{blank['region']} вилояти")
    if blank.get("town"):
        lines.append(f"{blank['town']} {blank.get('settlement', 'шаҳар')}")
    lines.append(f"{blank.get('number', 1)}-сон «{blank.get('mahalla', '')}»")
    lines.append("маҳалла фуқаролар йиғини")
    return lines


def _chairman_label(blank: dict[str, Any], rng: random.Random) -> str:
    """Word the chairman's printed title on the signature line."""
    number, mahalla = blank.get("number", 1), blank.get("mahalla", "")
    return rng.choice(
        (
            "Маҳалла раиси:",
            f"{number}-сон «{mahalla}» маҳалла раиси:",
            "Маҳалла фуқаролар йиғини раиси:",
            "МФЙ раиси:",
        )
    )


def _print_faces(library: Any) -> tuple[Path, Path]:
    """Return a regular and a bold print face that can set Uzbek Cyrillic."""
    for regular, bold in _PRINT_FACES:
        paths = [_find_font(regular), _find_font(bold)]
        if all(
            path is not None
            # Print has no fallback: a missing letter prints as a gap.
            and all(
                FontInfo.from_path(path).has_glyph(letter)
                for letter in _UZBEK_LETTERS
            )
            for path in paths
        ):
            return paths[0], paths[1]  # type: ignore[return-value]
    fallback = find_print_font(library)
    return fallback, fallback


def _find_font(name: str) -> Path | None:
    """Find a system font file by name."""
    for directory in _FONT_DIRS:
        path = directory / name
        if path.exists():
            return path
    return None
