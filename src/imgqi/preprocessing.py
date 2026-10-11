"""
preprocessing.py — Image loading, validation, and preparation.

All functions in this module operate on raw file paths or BGR numpy arrays
as produced by OpenCV. They do NOT apply any enhancement that would alter
sharpness measurements (no denoising, sharpening, or contrast stretching).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from *path* and return it as a BGR uint8 array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist on disk.
    ValueError
        If OpenCV cannot decode the file (unsupported format, corrupted, empty).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            f"Could not decode image: {path}. "
            "Ensure it is a supported format (JPEG, PNG, BMP, TIFF, …)."
        )
    return image


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to a single-channel uint8 grayscale array.

    If *image* is already single-channel it is returned unchanged.
    """
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def validate_image(image: np.ndarray) -> None:
    """Raise *ValueError* if *image* is None, empty, or has unexpected dtype.

    Parameters
    ----------
    image:
        Any numpy array produced by :func:`load_image`.
    """
    if image is None:
        raise ValueError("Image is None.")
    if image.size == 0:
        raise ValueError("Image array is empty (zero-size).")
    if image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 image, got dtype={image.dtype}. "
            "Load images with cv2.IMREAD_COLOR or cv2.IMREAD_GRAYSCALE."
        )


def image_info(image: np.ndarray) -> dict[str, int]:
    """Return a dict with ``width``, ``height``, and ``channels``."""
    h, w = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return {"width": int(w), "height": int(h), "channels": int(channels)}

