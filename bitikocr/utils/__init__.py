"""Pure helper functions. No business logic, no project-specific state."""

from bitikocr.utils.image_ops import alpha_bounding_box, light_augment

__all__ = ["alpha_bounding_box", "light_augment"]
