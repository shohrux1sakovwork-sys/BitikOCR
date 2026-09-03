"""Tests for the document generators and the generator registry."""

from __future__ import annotations

import itertools
import json
import statistics
from typing import Any

import pytest

from bitikocr.config import SyntheticConfig
from bitikocr.models.annotation import DocumentAnnotation
from bitikocr.synthetic.generators import (
    ArizaGenerator,
    BirthCertificateGenerator,
    DeathCertificateGenerator,
    FormGenerator,
    available_document_types,
    create_generator,
)


def assert_boxes_are_inside_the_page(annotation: DocumentAnnotation) -> None:
    """Every recorded box must lie within the page it was drawn on."""
    width, height = annotation.size
    for line in annotation.lines:
        assert line.bbox is not None, f"{line.block} has no box"
        assert line.bbox.left >= 0 and line.bbox.top >= 0
        assert line.bbox.right <= width and line.bbox.bottom <= height


def assert_values_sit_on_their_rules(
    generator: FormGenerator, fields: dict[str, Any], seeds: range = range(6)
) -> None:
    """Every line's ink must sit on the rule its own segment measured.

    The layouts note that handwriting rises above the line and may spill a
    little past its right end, so the tolerances are one-sided.
    """
    template = generator.template
    for seed in seeds:
        annotation = generator.generate(fields, seed=seed).annotation

        by_block: dict[str, list[Any]] = {}
        for line in annotation.lines:
            by_block.setdefault(line.block, []).append(line)

        for block, lines in by_block.items():
            try:
                geometry = template.field(block)
            except KeyError:
                continue
            for line, segment in zip(lines, geometry.segments):
                assert line.bbox is not None
                where = f"{block}={line.text!r} (seed {seed})"
                # Sits on the rule: not far above it, never dropped below.
                assert -12 <= line.bbox.bottom - segment.baseline_y <= 40, where
                # Starts at the rule's left end, barely overruns its right.
                assert line.bbox.left >= segment.x_start - 5, where
                assert line.bbox.right <= segment.x_end + 25, where


# -- registry --------------------------------------------------------------


def test_every_document_type_is_registered() -> None:
    assert available_document_types() == (
        "ariza",
        "birth_certificate",
        "death_certificate",
    )


def test_a_generator_is_built_from_its_name(config: SyntheticConfig) -> None:
    assert isinstance(create_generator("ariza", config), ArizaGenerator)


def test_an_unknown_document_type_is_rejected(
    config: SyntheticConfig,
) -> None:
    with pytest.raises(KeyError, match="Unknown document type"):
        create_generator("passport", config)


# -- ariza -----------------------------------------------------------------


def test_an_ariza_renders_at_the_requested_size(
    ariza_generator: ArizaGenerator, ariza_fields: dict[str, Any]
) -> None:
    document = ariza_generator.generate(ariza_fields, seed=11)
    assert document.image.size == ariza_generator.page_size
    assert document.annotation.size == ariza_generator.page_size


def test_an_ariza_is_reproducible_from_its_seed(
    ariza_generator: ArizaGenerator, ariza_fields: dict[str, Any]
) -> None:
    first = ariza_generator.generate(ariza_fields, seed=11)
    second = ariza_generator.generate(ariza_fields, seed=11)
    assert first.image.tobytes() == second.image.tobytes()
    assert first.annotation.to_dict() == second.annotation.to_dict()


def test_an_ariza_records_the_seed_it_used(
    ariza_generator: ArizaGenerator, ariza_fields: dict[str, Any]
) -> None:
    assert ariza_generator.generate(ariza_fields).seed is not None


def test_an_ariza_transcribes_every_field_it_was_given(
    ariza_generator: ArizaGenerator, ariza_fields: dict[str, Any]
) -> None:
    annotation = ariza_generator.generate(ariza_fields, seed=11).annotation
    written = {block.kind for block in annotation.blocks if block.text}
    assert {"recipient", "applicant", "body", "title"} <= written


def test_an_ariza_keeps_every_box_on_the_page(
    ariza_generator: ArizaGenerator, ariza_fields: dict[str, Any]
) -> None:
    assert_boxes_are_inside_the_page(
        ariza_generator.generate(ariza_fields, seed=11).annotation
    )


def test_a_style_override_reaches_the_rendered_page(
    ariza_generator: ArizaGenerator, ariza_fields: dict[str, Any]
) -> None:
    document = ariza_generator.generate(
        ariza_fields, seed=11, style_overrides={"pen": "soft"}
    )
    assert document.annotation.metadata["style"]["pen"] == "soft"


def test_an_unknown_style_override_is_rejected(
    ariza_generator: ArizaGenerator, ariza_fields: dict[str, Any]
) -> None:
    with pytest.raises(ValueError, match="Unknown style fields"):
        ariza_generator.generate(ariza_fields, style_overrides={"nope": 1})


def test_a_long_body_is_shrunk_to_fit_the_page(
    ariza_generator: ArizaGenerator, ariza_fields: dict[str, Any]
) -> None:
    crowded = {**ariza_fields, "body": ariza_fields["body"] * 6}
    roomy = ariza_generator.generate(ariza_fields, seed=11)
    tight = ariza_generator.generate(crowded, seed=11)

    assert (
        tight.annotation.metadata["style"]["font_size"]
        < roomy.annotation.metadata["style"]["font_size"]
    )
    assert_boxes_are_inside_the_page(tight.annotation)


# -- death certificate -----------------------------------------------------


def test_a_certificate_fills_every_given_field(
    certificate_generator: DeathCertificateGenerator,
    certificate_fields: dict[str, Any],
) -> None:
    annotation = certificate_generator.generate(
        certificate_fields, seed=5
    ).annotation
    filled = {block.kind for block in annotation.blocks if block.text}
    assert "surname" in filled
    assert "registrar_name" in filled
    assert "serial_number" in filled


def test_a_certificate_records_one_block_per_field(
    certificate_generator: DeathCertificateGenerator,
    certificate_fields: dict[str, Any],
) -> None:
    annotation = certificate_generator.generate(
        certificate_fields, seed=5
    ).annotation
    kinds = [block.kind for block in annotation.blocks]
    assert len(kinds) == len(set(kinds)), "a field was recorded twice"


def test_a_certificate_stamps_and_signs(
    certificate_generator: DeathCertificateGenerator,
    certificate_fields: dict[str, Any],
) -> None:
    annotation = certificate_generator.generate(
        certificate_fields, seed=5
    ).annotation
    kinds = {block.kind for block in annotation.blocks}
    assert {"stamp", "signature"} <= kinds


def test_a_certificate_keeps_every_box_on_the_page(
    certificate_generator: DeathCertificateGenerator,
    certificate_fields: dict[str, Any],
) -> None:
    assert_boxes_are_inside_the_page(
        certificate_generator.generate(certificate_fields, seed=5).annotation
    )


def test_a_partly_filled_certificate_only_reports_what_was_written(
    certificate_generator: DeathCertificateGenerator,
) -> None:
    annotation = certificate_generator.generate(
        {"surname": "Раҳимова", "age_at_death": "71"}, seed=5
    ).annotation
    assert annotation.metadata["fields"] == {
        "surname": "Раҳимова",
        "age_at_death": "71",
    }
    assert annotation.text == "Раҳимова\n71"


def test_the_annotation_serialises_to_json(
    certificate_generator: DeathCertificateGenerator,
    certificate_fields: dict[str, Any],
) -> None:
    annotation = certificate_generator.generate(
        certificate_fields, seed=5
    ).annotation
    payload = json.loads(json.dumps(annotation.to_dict(), ensure_ascii=False))
    assert payload["document_type"] == "death_certificate"
    assert payload["size"] == list(annotation.size)
    assert payload["blocks"][0]["type"] == annotation.blocks[0].kind


def test_every_written_value_lands_on_its_printed_rule(
    certificate_generator: DeathCertificateGenerator,
    certificate_fields: dict[str, Any],
) -> None:
    assert_values_sit_on_their_rules(certificate_generator, certificate_fields)


def test_lines_of_a_double_ruled_field_do_not_collide(
    certificate_generator: DeathCertificateGenerator,
) -> None:
    """The two 'cause of death' rules are 20px apart; the hand must compress."""
    long_cause = (
        "Miya qon aylanishining o'tkir buzilishi va yurak ishemik "
        "kasalligi asoratlari"
    )
    annotation = certificate_generator.generate(
        {"cause_of_death": long_cause}, seed=3
    ).annotation
    boxes = [
        line.bbox for line in annotation.lines if line.block == "cause_of_death"
    ]
    for upper, lower in itertools.pairwise(boxes):
        assert upper is not None and lower is not None
        assert upper.bottom <= lower.bottom, "lines must run top-down"


def test_the_template_name_is_recorded(
    certificate_generator: DeathCertificateGenerator,
    certificate_fields: dict[str, Any],
) -> None:
    annotation = certificate_generator.generate(
        certificate_fields, seed=5
    ).annotation
    assert annotation.metadata["template"] == "death_certificate_bilingual"


def test_a_template_can_be_chosen_by_name(config: SyntheticConfig) -> None:
    generator = create_generator(
        "death_certificate", config, template="death_certificate_bilingual"
    )
    assert isinstance(generator, DeathCertificateGenerator)
    assert generator.template.name == "death_certificate_bilingual"


def test_an_unknown_template_is_reported(config: SyntheticConfig) -> None:
    with pytest.raises(FileNotFoundError, match="Layout 'nope' not found"):
        create_generator("death_certificate", config, template="nope")


def test_a_template_is_rejected_for_a_free_layout_document(
    config: SyntheticConfig,
) -> None:
    with pytest.raises(ValueError, match="does not use form templates"):
        create_generator("ariza", config, template="anything")


# -- birth certificate -----------------------------------------------------


def test_a_birth_certificate_fills_child_parents_and_office(
    birth_generator: BirthCertificateGenerator,
    birth_fields: dict[str, Any],
) -> None:
    annotation = birth_generator.generate(birth_fields, seed=5).annotation
    filled = {block.kind for block in annotation.blocks if block.text}
    assert {"child_surname", "child_given_name"} <= filled
    assert {"father_surname", "mother_surname"} <= filled
    assert {"registry_office", "registry_head_name"} <= filled


def test_a_birth_certificate_prints_both_series_and_number(
    birth_generator: BirthCertificateGenerator,
    birth_fields: dict[str, Any],
) -> None:
    """This form typesets a series beside the number, not one serial."""
    annotation = birth_generator.generate(birth_fields, seed=5).annotation
    printed = {
        block.kind: block.text
        for block in annotation.blocks
        if block.kind in ("form_series", "form_number")
    }
    assert printed == {
        "form_series": birth_fields["form_series"],
        "form_number": birth_fields["form_number"],
    }


def test_a_birth_certificate_stamps_and_signs(
    birth_generator: BirthCertificateGenerator,
    birth_fields: dict[str, Any],
) -> None:
    annotation = birth_generator.generate(birth_fields, seed=5).annotation
    kinds = {block.kind for block in annotation.blocks}
    assert {"stamp", "signature"} <= kinds


def test_the_head_of_office_is_a_field_not_a_signature_caption(
    birth_generator: BirthCertificateGenerator,
) -> None:
    """The birth form gives the name its own rule, unlike the death form."""
    assert not birth_generator.registrar_writes_on_signature
    assert "registrar_name" not in birth_generator.field_names
    assert "registry_head_name" in birth_generator.field_names


def test_a_birth_certificate_is_reproducible_from_its_seed(
    birth_generator: BirthCertificateGenerator,
    birth_fields: dict[str, Any],
) -> None:
    first = birth_generator.generate(birth_fields, seed=7)
    second = birth_generator.generate(birth_fields, seed=7)
    assert first.image.tobytes() == second.image.tobytes()
    assert first.annotation.to_dict() == second.annotation.to_dict()


def test_a_birth_certificate_keeps_every_box_on_the_page(
    birth_generator: BirthCertificateGenerator,
    birth_fields: dict[str, Any],
) -> None:
    assert_boxes_are_inside_the_page(
        birth_generator.generate(birth_fields, seed=5).annotation
    )


def test_every_birth_value_lands_on_its_printed_rule(
    birth_generator: BirthCertificateGenerator,
    birth_fields: dict[str, Any],
) -> None:
    assert_values_sit_on_their_rules(birth_generator, birth_fields)


def test_nothing_is_drawn_over_the_printed_qr_code(
    birth_generator: BirthCertificateGenerator,
    birth_fields: dict[str, Any],
) -> None:
    """The blank form already carries a QR; ink there would ruin both."""
    template = birth_generator.template
    assert template.keep_out, "expected the layout to reserve the QR region"

    for seed in range(8):
        annotation = birth_generator.generate(
            birth_fields, seed=seed
        ).annotation
        for area in template.keep_out:
            for block in annotation.blocks:
                if block.bbox is None:
                    continue
                overlaps = (
                    block.bbox.left < area.bbox.right
                    and area.bbox.left < block.bbox.right
                    and block.bbox.top < area.bbox.bottom
                    and area.bbox.top < block.bbox.bottom
                )
                assert (
                    not overlaps
                ), f"{block.kind} overlaps {area.name} at seed {seed}"


def test_both_certificates_share_one_generator(
    config: SyntheticConfig,
) -> None:
    """Certificates differ by template, not by rendering code."""
    birth = create_generator("birth_certificate", config)
    death = create_generator("death_certificate", config)
    assert isinstance(birth, FormGenerator)
    assert isinstance(death, FormGenerator)
    assert type(birth).generate is type(death).generate
    assert birth.template.name != death.template.name


# -- placement inside a form's cells ----------------------------------------


def measure_off_centre(
    generator: FormGenerator, fields: dict[str, Any], seeds: range = range(10)
) -> list[float]:
    """Return how far each value sits from the middle of its cell.

    The result is normalised against the slack around the value: 0.0 is
    dead centre and ±1.0 is flush against one end. Only single-line fields
    with real room to move are measured.
    """
    template = generator.template
    offsets: list[float] = []
    for seed in seeds:
        annotation = generator.generate(fields, seed=seed).annotation
        for line in annotation.lines:
            try:
                geometry = template.field(line.block)
            except KeyError:
                continue
            if len(geometry.segments) != 1 or line.bbox is None:
                continue
            segment = geometry.segments[0]
            slack = segment.width - line.bbox.width
            if slack < 60:
                continue
            cell_centre = (segment.x_start + segment.x_end) / 2
            ink_centre = (line.bbox.left + line.bbox.right) / 2
            offsets.append((ink_centre - cell_centre) / (slack / 2))
    return offsets


@pytest.mark.parametrize(
    "generator_fixture", ["birth_generator", "certificate_generator"]
)
def test_values_sit_around_the_middle_of_their_cell(
    generator_fixture: str,
    request: pytest.FixtureRequest,
    birth_fields: dict[str, Any],
    certificate_fields: dict[str, Any],
) -> None:
    """A clerk aims for the middle of the space, not its left edge."""
    generator = request.getfixturevalue(generator_fixture)
    fields = (
        birth_fields
        if generator_fixture == "birth_generator"
        else certificate_fields
    )
    offsets = measure_off_centre(generator, fields)

    assert offsets, "expected fields with room to move"
    assert abs(statistics.mean(offsets)) < 0.2, "values drift off centre"
    assert (
        max(abs(offset) for offset in offsets) < 0.8
    ), "a value was written flush against one end of its cell"


def test_values_are_not_centred_to_the_pixel(
    birth_generator: BirthCertificateGenerator,
    birth_fields: dict[str, Any],
) -> None:
    """Perfect centring reads as typeset; a hand never lands twice alike."""
    offsets = measure_off_centre(birth_generator, birth_fields)
    assert statistics.pstdev(offsets) > 0.05


# -- the office seal --------------------------------------------------------


def seal_centres(
    generator: FormGenerator, fields: dict[str, Any], seeds: range = range(20)
) -> list[tuple[float, float]]:
    """Return where the seal's ink landed on each generated page."""
    centres: list[tuple[float, float]] = []
    for seed in seeds:
        annotation = generator.generate(fields, seed=seed).annotation
        box = next(
            block.bbox for block in annotation.blocks if block.kind == "stamp"
        )
        assert box is not None
        centres.append(((box.left + box.right) / 2, (box.top + box.bottom) / 2))
    return centres


def test_the_seal_is_pressed_off_centre_every_time(
    birth_generator: BirthCertificateGenerator,
    birth_fields: dict[str, Any],
) -> None:
    """A hand-pressed seal never lands in the same place twice."""
    centres = seal_centres(birth_generator, birth_fields)
    assert len({round(x) for x, _ in centres}) > 5
    assert len({round(y) for _, y in centres}) > 5


def test_the_seal_leans_left_of_its_printed_circle(
    birth_generator: BirthCertificateGenerator,
    birth_fields: dict[str, Any],
) -> None:
    centres = seal_centres(birth_generator, birth_fields)
    area = birth_generator.template.seal
    assert area is not None
    assert statistics.mean(x for x, _ in centres) < area.centre[0]


def test_the_seal_stays_on_the_page_wherever_it_lands(
    birth_generator: BirthCertificateGenerator,
    birth_fields: dict[str, Any],
) -> None:
    width, height = birth_generator.template.native_size
    for seed in range(20):
        annotation = birth_generator.generate(
            birth_fields, seed=seed
        ).annotation
        box = next(
            block.bbox for block in annotation.blocks if block.kind == "stamp"
        )
        assert box is not None
        assert 0 <= box.left and box.right <= width, seed
        assert 0 <= box.top and box.bottom <= height, seed


def test_the_seal_box_measures_its_ink_not_its_canvas(
    birth_generator: BirthCertificateGenerator,
    birth_fields: dict[str, Any],
) -> None:
    """The stamp layer is padded for its curved lettering; the box is not."""
    annotation = birth_generator.generate(birth_fields, seed=5).annotation
    box = next(
        block.bbox for block in annotation.blocks if block.kind == "stamp"
    )
    area = birth_generator.template.seal
    assert box is not None and area is not None

    canvas = area.radius * 2 * 1.35 * 2.6  # widest radius, padded layer
    assert box.width < canvas
    assert box.width > area.radius  # but it did draw a seal
