"""
conftest.py — Shared pytest fixtures for imgqi tests.

Fixtures generate synthetic images programmatically so no external files are
needed.  All images are uint8 BGR arrays compatible with OpenCV.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic image generators
# ---------------------------------------------------------------------------

def _make_textured_gray(h: int = 256, w: int = 256, seed: int = 42) -> np.ndarray:
    """Return a uint8 grayscale array with mid-frequency sinusoidal texture."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4 * np.pi, w)
    y = np.linspace(0, 4 * np.pi, h)
    xx, yy = np.meshgrid(x, y)
    base = (np.sin(xx) * np.cos(yy) + 1) / 2  # [0, 1]
    noise = rng.normal(0, 0.02, size=(h, w))
    img = np.clip((base + noise) * 200 + 27, 0, 255).astype(np.uint8)
    return img


def _gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sharp_gray() -> np.ndarray:
    """High-frequency textured grayscale image — should score high sharpness."""
    return _make_textured_gray()


@pytest.fixture(scope="session")
def sharp_bgr(sharp_gray) -> np.ndarray:
    return _gray_to_bgr(sharp_gray)


@pytest.fixture(scope="session")
def blurry_gray(sharp_gray) -> np.ndarray:
    """Same image after heavy Gaussian blur — should score low sharpness."""
    return cv2.GaussianBlur(sharp_gray, (31, 31), 10)


@pytest.fixture(scope="session")
def blurry_bgr(blurry_gray) -> np.ndarray:
    return _gray_to_bgr(blurry_gray)


@pytest.fixture(scope="session")
def smooth_gray() -> np.ndarray:
    """Flat grey image — smooth, not blurry; should yield INCONCLUSIVE tiles."""
    return np.full((256, 256), 128, dtype=np.uint8)


@pytest.fixture(scope="session")
def dark_gray() -> np.ndarray:
    """Nearly-black image — should trigger underexposure warning."""
    return np.full((256, 256), 4, dtype=np.uint8)


@pytest.fixture(scope="session")
def bright_gray() -> np.ndarray:
    """Nearly-white image — should trigger overexposure warning."""
    return np.full((256, 256), 252, dtype=np.uint8)


@pytest.fixture(scope="session")
def noisy_gray(sharp_gray) -> np.ndarray:
    """Sharp image with heavy salt-and-pepper noise."""
    rng = np.random.default_rng(7)
    noise = rng.integers(-40, 41, size=sharp_gray.shape, dtype=np.int16)
    return np.clip(sharp_gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def mixed_gray(sharp_gray, blurry_gray) -> np.ndarray:
    """Left half sharp, right half blurry — tests regional detection."""
    mixed = sharp_gray.copy()
    h, w = mixed.shape
    mixed[:, w // 2:] = blurry_gray[:, w // 2:]
    return mixed

