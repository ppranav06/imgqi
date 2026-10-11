"""
quality_metrics.py — Contrast, exposure, and high-frequency residual indicators.

These functions accept a uint8 grayscale array and return plain Python scalars
or dicts so they can be serialised to JSON/CSV without conversion.

Notes
-----
* ``contrast_score`` returns the pixel-intensity standard deviation — a simple
  but scene-dependent measure.  A blank white wall can legitimately score low.
* ``exposure_metrics`` reports the fraction of near-black and near-white pixels.
  The thresholds are configurable because a night photo or snow scene may show
  extreme values by design.
* ``high_frequency_residual`` is *not* a pure noise measurement.  Genuine fine
  texture, hair, and sharp edges also raise this value.  Treat it as an
  indicator, not a certified noise estimate.
"""

from __future__ import annotations

import cv2
import numpy as np


def contrast_score(gray: np.ndarray) -> float:
    """Return the standard deviation of pixel intensities.

    Higher values indicate a wider spread of intensities (more contrast).
    Lower values indicate a narrow distribution (flat or washed-out image).

    Parameters
    ----------
    gray:
        Single-channel uint8 grayscale array.
    """
    return float(gray.std())


def contrast_percentile_range(gray: np.ndarray, low: float = 5.0, high: float = 95.0) -> float:
    """Return the intensity range between the *low* and *high* percentiles.

    This is less sensitive to a few extreme pixels than the full range
    ``max(I) - min(I)`` while still capturing most of the intensity spread.

    Parameters
    ----------
    gray:
        Single-channel uint8 grayscale array.
    low, high:
        Percentile bounds (0–100).
    """
    lo, hi = np.percentile(gray, [low, high])
    return float(hi - lo)


def exposure_metrics(
    gray: np.ndarray,
    dark_threshold: int = 10,
    bright_threshold: int = 245,
) -> dict[str, float]:
    """Return dark- and bright-pixel fractions as percentages.

    Parameters
    ----------
    gray:
        Single-channel uint8 grayscale array.
    dark_threshold:
        Pixels with intensity < this value are counted as *dark*.
        Default 10 (out of 255).
    bright_threshold:
        Pixels with intensity > this value are counted as *bright*.
        Default 245 (out of 255).

    Returns
    -------
    dict with keys:
        ``dark_pixel_percent``  — fraction below *dark_threshold* × 100
        ``bright_pixel_percent`` — fraction above *bright_threshold* × 100
        ``mean_intensity``      — average pixel value (0–255)
    """
    n = gray.size
    dark_ratio = float(np.count_nonzero(gray < dark_threshold) / n * 100)
    bright_ratio = float(np.count_nonzero(gray > bright_threshold) / n * 100)
    mean_intensity = float(gray.mean())
    return {
        "dark_pixel_percent": dark_ratio,
        "bright_pixel_percent": bright_ratio,
        "mean_intensity": mean_intensity,
    }


def high_frequency_residual(gray: np.ndarray, ksize: int = 5, sigma: float = 0.0) -> float:
    """Return the standard deviation of the Gaussian-blur residual.

    ``residual = gray - GaussianBlur(gray)``

    This captures high-frequency energy (noise *and* fine texture).
    Use this as an indicator rather than a certified noise measurement.

    Parameters
    ----------
    gray:
        Single-channel uint8 grayscale array.
    ksize:
        Kernel size for the Gaussian blur.  Must be odd and ≥ 1.
    sigma:
        Standard deviation for the Gaussian kernel.  0 means OpenCV picks it
        automatically based on *ksize*.
    """
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError(f"ksize must be a positive odd integer, got {ksize}.")

    gray_f = gray.astype(np.float32)
    smooth = cv2.GaussianBlur(gray_f, (ksize, ksize), sigma)
    residual = gray_f - smooth
    return float(np.std(residual))


def summarise_histogram(gray: np.ndarray) -> dict[str, float]:
    """Return basic histogram statistics useful for diagnostics.

    Returns
    -------
    dict with keys:
        ``mean``, ``std``, ``p5``, ``p25``, ``p50``, ``p75``, ``p95``
    """
    p5, p25, p50, p75, p95 = np.percentile(gray, [5, 25, 50, 75, 95])
    return {
        "mean": float(gray.mean()),
        "std": float(gray.std()),
        "p5": float(p5),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
    }

