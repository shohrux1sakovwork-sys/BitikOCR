"""Configuration loading for the whole application.

This is the only module allowed to read environment variables. Everything
else receives a :class:`SyntheticConfig` instance from the caller, so no
module reaches for global state on its own.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ENV_BACKGROUNDS_DIR",
    "ENV_DATA_DIR",
    "ENV_FONTS_DIR",
    "ENV_LAYOUTS_DIR",
    "PACKAGE_ROOT",
    "SyntheticConfig",
]

PACKAGE_ROOT = Path(__file__).resolve().parent
_ASSETS_DIR = PACKAGE_ROOT / "synthetic" / "assets"

DEFAULT_FONTS_DIR = _ASSETS_DIR / "fonts"
DEFAULT_BACKGROUNDS_DIR = _ASSETS_DIR / "backgrounds"
DEFAULT_LAYOUTS_DIR = _ASSETS_DIR / "layouts"
DEFAULT_DATA_DIR = Path("data")

#: Where a dataset sits under the data root. It matches the schema's
#: ``source.origin``, so generated pages, real scans and augmented
#: copies of them stay in separate trees.
DEFAULT_ORIGIN = "synthetic"

ENV_FONTS_DIR = "BITIKOCR_FONTS_DIR"
ENV_BACKGROUNDS_DIR = "BITIKOCR_BACKGROUNDS_DIR"
ENV_LAYOUTS_DIR = "BITIKOCR_LAYOUTS_DIR"
ENV_DATA_DIR = "BITIKOCR_DATA_DIR"


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class SyntheticConfig:
    """Paths the synthetic data pipeline needs.

    Args:
        fonts_dir: Directory scanned for ``.ttf`` / ``.otf`` handwriting fonts.
        backgrounds_dir: Directory holding blank form scans.
        layouts_dir: Directory holding the measured layout JSON that says
            where each field goes on those scans.
        data_dir: Root of the corpus. Datasets live under it as
            ``<data_dir>/<origin>/<document type>``.
    """

    fonts_dir: Path = DEFAULT_FONTS_DIR
    backgrounds_dir: Path = DEFAULT_BACKGROUNDS_DIR
    layouts_dir: Path = DEFAULT_LAYOUTS_DIR
    data_dir: Path = DEFAULT_DATA_DIR

    @classmethod
    def from_env(cls) -> SyntheticConfig:
        """Build a config from environment variables, falling back to defaults.

        Returns:
            A config whose paths come from ``BITIKOCR_FONTS_DIR``,
            ``BITIKOCR_BACKGROUNDS_DIR``, ``BITIKOCR_LAYOUTS_DIR`` and
            ``BITIKOCR_DATA_DIR`` when set.
        """
        return cls(
            fonts_dir=_path_from_env(ENV_FONTS_DIR, DEFAULT_FONTS_DIR),
            backgrounds_dir=_path_from_env(
                ENV_BACKGROUNDS_DIR, DEFAULT_BACKGROUNDS_DIR
            ),
            layouts_dir=_path_from_env(ENV_LAYOUTS_DIR, DEFAULT_LAYOUTS_DIR),
            data_dir=_path_from_env(ENV_DATA_DIR, DEFAULT_DATA_DIR),
        )

    def dataset_dir(
        self, document_type: str, origin: str = DEFAULT_ORIGIN
    ) -> Path:
        """Return where one document type's dataset lives.

        Args:
            document_type: Which documents the dataset holds.
            origin: How they were produced — ``synthetic``, ``real`` or
                ``augmented``, matching the schema's ``source.origin``.

        Returns:
            The dataset directory, which may not exist yet.
        """
        return self.data_dir / origin / document_type

    def background(self, name: str) -> Path:
        """Resolve a background template by file name.

        Args:
            name: File name inside :attr:`backgrounds_dir`.

        Returns:
            The absolute path to the template.

        Raises:
            FileNotFoundError: If no such template exists.
        """
        path = self.backgrounds_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Background scan not found: {path}")
        return path

    def layout(self, name: str) -> Path:
        """Resolve a form layout by template name.

        Args:
            name: Template name, matching a ``<name>.json`` in
                :attr:`layouts_dir`.

        Returns:
            The absolute path to the layout JSON.

        Raises:
            FileNotFoundError: If no such layout exists.
        """
        path = self.layouts_dir / f"{name}.json"
        if not path.is_file():
            known = ", ".join(self.available_layouts()) or "none"
            raise FileNotFoundError(
                f"Layout {name!r} not found in {self.layouts_dir}. "
                f"Available: {known}"
            )
        return path

    def available_layouts(self) -> tuple[str, ...]:
        """Return the name of every layout JSON that can be loaded, sorted."""
        if not self.layouts_dir.is_dir():
            return ()
        return tuple(sorted(p.stem for p in self.layouts_dir.glob("*.json")))
