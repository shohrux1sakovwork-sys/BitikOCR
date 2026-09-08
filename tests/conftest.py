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
    DeathCertificateGenerator,
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


@pytest.fixture(scope="session")
def birth_fields() -> dict[str, object]:
    """One sampled birth certificate record's fields."""
    return sample_record("birth_certificate", random.Random(13), "latin").fields


@pytest.fixture(scope="session")
def birth_generator(config: SyntheticConfig) -> BirthCertificateGenerator:
    """A birth certificate generator rendered small and unaugmented."""
    return BirthCertificateGenerator(config, options=FormOptions(scale=1.0))
