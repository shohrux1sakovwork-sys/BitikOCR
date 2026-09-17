"""Shared fixtures for the whole test suite.

Every test module lives in this directory, so every fixture defined here
reaches all of them without being imported.
"""

from __future__ import annotations

import random

import pytest

from bitikocr.config import SyntheticConfig
from bitikocr.data.synthetic.fonts import FontLibrary
from bitikocr.data.synthetic.generators import (
    ArizaGenerator,
    BirthCertificateGenerator,
    ConsentLetterGenerator,
    DeathCertificateGenerator,
    ExplanatoryLetterGenerator,
    FormOptions,
)
from bitikocr.data.synthetic.generators.death_certificate import (
    SINGLE_TEMPLATE,
)
from bitikocr.data.synthetic.records import sample_record
from bitikocr.data.synthetic.style import HandwritingStyle, sample_style


@pytest.fixture(scope="session")
def config() -> SyntheticConfig:
    """The default configuration, pointing at the packaged assets."""
    return SyntheticConfig()


@pytest.fixture(scope="session")
def library(config: SyntheticConfig) -> FontLibrary:
    """The packaged handwriting fonts."""
    return FontLibrary.from_directory(config.fonts_dir)


@pytest.fixture()
def style(library: FontLibrary) -> HandwritingStyle:
    """A style sampled from a fixed seed."""
    return sample_style(random.Random(1234), library)


@pytest.fixture(scope="session")
def ariza_fields() -> dict[str, object]:
    """One sampled ariza record's fields."""
    return sample_record("ariza", random.Random(11), "cyrillic").fields


@pytest.fixture(scope="session")
def certificate_fields() -> dict[str, object]:
    """One sampled death certificate record's fields."""
    return sample_record("death_certificate", random.Random(12), "latin").fields


@pytest.fixture(scope="session")
def ariza_generator(config: SyntheticConfig) -> ArizaGenerator:
    """An ariza generator using the packaged assets."""
    return ArizaGenerator(config)


@pytest.fixture(scope="session")
def certificate_generator(
    config: SyntheticConfig,
) -> DeathCertificateGenerator:
    """A death certificate generator rendered small and unaugmented.

    Dropping the scale and the augmentation keeps the suite fast without
    changing any of the layout logic under test.
    """
    return DeathCertificateGenerator(config, options=FormOptions(scale=1.0))


@pytest.fixture(scope="session")
def single_certificate_fields() -> dict[str, object]:
    """One sampled death certificate record's fields, in Cyrillic."""
    return sample_record(
        "death_certificate", random.Random(14), "cyrillic"
    ).fields


@pytest.fixture(scope="session")
def single_generator(config: SyntheticConfig) -> DeathCertificateGenerator:
    """The single-page Cyrillic death certificate, rendered small."""
    return DeathCertificateGenerator(
        config, options=FormOptions(template=SINGLE_TEMPLATE, scale=1.0)
    )


def _consent_record(kind: str, certified: bool) -> dict[str, object]:
    """Sample the first consent letter of one shape.

    Who writes a consent letter decides what reaches the page, so the three
    shapes are drawn out by searching rather than by a lucky seed.

    Args:
        kind: ``individual`` or ``organisation``.
        certified: Whether an official attested to the signature. Only a
            citizen's letter is ever certified.

    Returns:
        That letter's field values.
    """
    rng = random.Random(15)
    for _ in range(200):
        record = sample_record("consent_letter", rng, "cyrillic")
        sealed = bool(record.fields.get("certifier_name"))
        if record.notes["author_kind"] == kind and sealed == certified:
            return record.fields
    raise AssertionError(f"no {kind} letter, certified={certified}, was drawn")


@pytest.fixture(scope="session")
def consent_fields() -> dict[str, object]:
    """A citizen's consent letter, certified by an official."""
    return _consent_record("individual", certified=True)


@pytest.fixture(scope="session")
def consent_plain_fields() -> dict[str, object]:
    """A citizen's consent letter that nobody certified."""
    return _consent_record("individual", certified=False)


@pytest.fixture(scope="session")
def consent_organisation_fields() -> dict[str, object]:
    """A consent letter written and sealed by a legal entity."""
    return _consent_record("organisation", certified=False)


@pytest.fixture(scope="session")
def consent_generator(config: SyntheticConfig) -> ConsentLetterGenerator:
    """A consent letter generator using the packaged assets."""
    return ConsentLetterGenerator(config)


def _explanation_record(kind: str) -> dict[str, object]:
    """Sample the first explanatory letter written by one sort of writer.

    Args:
        kind: ``citizen``, ``employee`` or ``student``.

    Returns:
        That letter's field values.
    """
    rng = random.Random(16)
    for _ in range(200):
        record = sample_record("explanatory_letter", rng, "cyrillic")
        if record.notes["author_kind"] == kind:
            return record.fields
    raise AssertionError(f"no explanatory letter from a {kind} was drawn")


@pytest.fixture(scope="session")
def citizen_explanation_fields() -> dict[str, object]:
    """A citizen's explanation to the district mayor."""
    return _explanation_record("citizen")


@pytest.fixture(scope="session")
def employee_explanation_fields() -> dict[str, object]:
    """An employee's explanation of a lapse at work."""
    return _explanation_record("employee")


@pytest.fixture(scope="session")
def explanatory_generator(
    config: SyntheticConfig,
) -> ExplanatoryLetterGenerator:
    """An explanatory letter generator using the packaged assets."""
    return ExplanatoryLetterGenerator(config)


@pytest.fixture(scope="session")
def birth_fields() -> dict[str, object]:
    """One sampled birth certificate record's fields."""
    return sample_record("birth_certificate", random.Random(13), "latin").fields


@pytest.fixture(scope="session")
def birth_generator(config: SyntheticConfig) -> BirthCertificateGenerator:
    """A birth certificate generator rendered small and unaugmented."""
    return BirthCertificateGenerator(config, options=FormOptions(scale=1.0))
