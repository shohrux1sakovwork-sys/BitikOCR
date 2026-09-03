"""Shared fixtures for the test suite."""

from __future__ import annotations

import random

import pytest

from bitikocr.config import SyntheticConfig
from bitikocr.synthetic.fonts import FontLibrary
from bitikocr.synthetic.generators import (
    ArizaGenerator,
    BirthCertificateGenerator,
    DeathCertificateGenerator,
    FormOptions,
)
from bitikocr.synthetic.sample_data import sample_fields_for
from bitikocr.synthetic.style import HandwritingStyle, sample_style


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
    """One built-in ariza field set."""
    return dict(sample_fields_for("ariza")[0])


@pytest.fixture(scope="session")
def certificate_fields() -> dict[str, object]:
    """One built-in death certificate field set."""
    return dict(sample_fields_for("death_certificate")[0])


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
    return DeathCertificateGenerator(
        config, options=FormOptions(scale=1.0, augment=False)
    )


@pytest.fixture(scope="session")
def birth_fields() -> dict[str, object]:
    """One built-in birth certificate field set."""
    return dict(sample_fields_for("birth_certificate")[0])


@pytest.fixture(scope="session")
def birth_generator(config: SyntheticConfig) -> BirthCertificateGenerator:
    """A birth certificate generator rendered small and unaugmented."""
    return BirthCertificateGenerator(
        config, options=FormOptions(scale=1.0, augment=False)
    )
