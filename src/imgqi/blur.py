"""
blur.py — Global and regional sharpness estimation.

Algorithm
---------
Global sharpness
    Variance of the Laplacian of the full grayscale image.
    ``S_global = Var(∇²I)``

Regional sharpness
    The full-image Laplacian is computed *once* and its values are sliced into
    tiles.  This avoids introducing artificial boundary artefacts that would
    result from computing the Laplacian independently on each tile.

    For each tile (i, j):
        ``S_{i,j} = Var(∇²I[y₁:y₂, x₁:x₂])``

Regional classification
    Three-state decision per tile:
    - SHARP       S_{i,j} ≥ sharpness_threshold
    - BLURRY      S_{i,j} <  sharpness_threshold  AND  gradient energy is
                  high enough to rule out a naturally smooth region
    - INCONCLUSIVE otherwise (too little texture for a reliable verdict)

    The gradient-energy gate uses the Sobel-based mean squared gradient (MSG).
    If MSG < texture_threshold the tile is classified as INCONCLUSIVE regardless
    of its Laplacian score — a blank wall should not be called blurry.

Notes
-----
* No universal threshold exists.  The defaults are calibrated for 8-bit images
  at typical camera resolutions and mid-texture subjects.  Always validate
  thresholds against representative images for your specific use case.
* Noise can *inflate* the Laplacian variance, making a noisy image look sharp.
* Large tiles are more statistically stable; small tiles give finer spatial
  resolution.  The recommended starting point is 64 px.
"""

from __future__ import annotations

import math
from enum import IntEnum

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Region-classification states
# ---------------------------------------------------------------------------

class TileState(IntEnum):
    """Classification outcome for a single image tile."""
    INCONCLUSIVE = 0   # Not enough texture/content for a reliable verdict
    SHARP = 1          # Above the sharpness threshold
    BLURRY = 2         # Below the sharpness threshold with sufficient texture


# ---------------------------------------------------------------------------
# Global sharpness
# ---------------------------------------------------------------------------

def global_sharpness(gray: np.ndarray) -> float:
    """Return the variance of the Laplacian as a global sharpness estimate.

    Parameters
    ----------
    gray:
        Single-channel uint8 grayscale image.

    Returns
    -------
    float
        Non-negative variance value.  Larger → more high-frequency content.
        A very low value *may* indicate blur, but also occurs with smooth
        content such as clear sky or blank documents.
    """
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    return float(laplacian.var())


# ---------------------------------------------------------------------------
# Regional sharpness
# ---------------------------------------------------------------------------

def regional_sharpness(
    gray: np.ndarray,
    tile_size: int = 64,
) -> np.ndarray:
    """Compute per-tile Laplacian-variance sharpness scores.

    The Laplacian is computed over the full image first, then sliced into
    tiles.  Tiles that are too small (fewer than ``tile_size // 4`` pixels on
    either dimension) are marked NaN.

    Parameters
    ----------
    gray:
        Single-channel uint8 grayscale image.
    tile_size:
        Side length of each square tile in pixels.  Must be ≥ 8.

    Returns
    -------
    scores : np.ndarray, shape (rows, cols), dtype float32
        Regional sharpness scores.  NaN indicates an unreliable tile.
    """
    if tile_size < 8:
        raise ValueError(f"tile_size must be ≥ 8, got {tile_size}.")

    h, w = gray.shape
    rows = math.ceil(h / tile_size)
    cols = math.ceil(w / tile_size)

    # Compute the Laplacian once over the full image.
    lap_full = cv2.Laplacian(gray, cv2.CV_32F)

    scores = np.full((rows, cols), np.nan, dtype=np.float32)
    min_side = max(tile_size // 4, 4)  # smallest acceptable tile dimension

    for r in range(rows):
        y1, y2 = r * tile_size, min((r + 1) * tile_size, h)
        if (y2 - y1) < min_side:
            continue
        for c in range(cols):
            x1, x2 = c * tile_size, min((c + 1) * tile_size, w)
            if (x2 - x1) < min_side:
                continue
            tile_lap = lap_full[y1:y2, x1:x2]
            scores[r, c] = float(tile_lap.var())

    return scores


# ---------------------------------------------------------------------------
# Tile gradient energy
# ---------------------------------------------------------------------------

def tile_gradient_energy(
    gray: np.ndarray,
    tile_size: int = 64,
) -> np.ndarray:
    """Compute mean squared Sobel-gradient energy per tile.

    This supplements the Laplacian score to distinguish smooth regions from
    blurry ones: a tile with low sharpness *and* low gradient energy is likely
    a naturally smooth surface (sky, blank paper), not a blurry one.

    Parameters
    ----------
    gray:
        Single-channel uint8 grayscale image.
    tile_size:
        Side length of each square tile in pixels.

    Returns
    -------
    energy : np.ndarray, shape (rows, cols), dtype float32
        Mean squared gradient energy per tile.
    """
    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    gradient_sq = gx ** 2 + gy ** 2

    h, w = gray.shape
    rows = math.ceil(h / tile_size)
    cols = math.ceil(w / tile_size)
    min_side = max(tile_size // 4, 4)

    energy = np.full((rows, cols), np.nan, dtype=np.float32)
    for r in range(rows):
        y1, y2 = r * tile_size, min((r + 1) * tile_size, h)
        if (y2 - y1) < min_side:
            continue
        for c in range(cols):
            x1, x2 = c * tile_size, min((c + 1) * tile_size, w)
            if (x2 - x1) < min_side:
                continue
            energy[r, c] = float(gradient_sq[y1:y2, x1:x2].mean())

    return energy


# ---------------------------------------------------------------------------
# Regional classification
# ---------------------------------------------------------------------------

def classify_tiles(
    scores: np.ndarray,
    energy: np.ndarray,
    sharpness_threshold: float = 100.0,
    texture_threshold: float = 20.0,
) -> np.ndarray:
    """Classify each tile as SHARP, BLURRY, or INCONCLUSIVE.

    Decision rules
    --------------
    1. If scores[r,c] is NaN → INCONCLUSIVE
    2. If energy[r,c] < texture_threshold → INCONCLUSIVE (smooth/featureless)
    3. If scores[r,c] >= sharpness_threshold → SHARP
    4. Otherwise → BLURRY

    Parameters
    ----------
    scores:
        Laplacian-variance sharpness scores, shape (rows, cols).
    energy:
        Mean squared gradient energy per tile, shape (rows, cols).
    sharpness_threshold:
        Laplacian variance below this value, combined with sufficient texture,
        marks a tile BLURRY.  Tune this against your image corpus.
    texture_threshold:
        Mean squared gradient energy below this value causes INCONCLUSIVE
        classification regardless of Laplacian variance.

    Returns
    -------
    states : np.ndarray, shape (rows, cols), dtype int32
        Values are ``TileState`` members: 0=INCONCLUSIVE, 1=SHARP, 2=BLURRY.
    """
    states = np.full(scores.shape, TileState.INCONCLUSIVE, dtype=np.int32)

    valid_mask = np.isfinite(scores) & np.isfinite(energy)
    textured_mask = valid_mask & (energy >= texture_threshold)
    sharp_mask = textured_mask & (scores >= sharpness_threshold)
    blurry_mask = textured_mask & (scores < sharpness_threshold)

    states[sharp_mask] = TileState.SHARP
    states[blurry_mask] = TileState.BLURRY

    return states


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def blur_summary(states: np.ndarray) -> dict[str, int | float]:
    """Return aggregate statistics from a tile-state array.

    Returns
    -------
    dict with keys:
        ``total_tiles``, ``sharp``, ``blurry``, ``inconclusive``,
        ``blurry_fraction`` (ratio of BLURRY to tiles with a verdict).
    """
    sharp = int(np.count_nonzero(states == TileState.SHARP))
    blurry = int(np.count_nonzero(states == TileState.BLURRY))
    inconclusive = int(np.count_nonzero(states == TileState.INCONCLUSIVE))
    total = int(states.size)
    decided = sharp + blurry
    blurry_fraction = float(blurry / decided) if decided > 0 else float("nan")
    return {
        "total_tiles": total,
        "sharp": sharp,
        "blurry": blurry,
        "inconclusive": inconclusive,
        "blurry_fraction": blurry_fraction,
    }

