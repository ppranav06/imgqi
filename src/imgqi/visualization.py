"""
visualization.py — Heatmaps and annotated overlays.

Design decisions
----------------
* Raw score matrices are the ground truth.  Heatmaps are *derived* views
  generated on demand from the stored matrices.
* Absolute thresholds are used for cross-image consistency.  The caller
  provides the same threshold as used in classification so the colour scale
  is meaningful across images.
* ``INTER_NEAREST`` is used when enlarging the score matrix so that tile
  boundaries remain crisp and the viewer is not misled into thinking
  pixel-level measurements were made.
* Tile-state overlays draw coloured rectangles rather than a smooth gradient,
  which better reflects the grid-based nature of the measurement.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from imgqi.blur import TileState


# Colour scheme (BGR): matches the tile-state enum
_STATE_COLOURS: dict[int, tuple[int, int, int]] = {
    TileState.SHARP:        (50,  205,  50),   # lime green
    TileState.BLURRY:       (0,   0,    220),   # red
    TileState.INCONCLUSIVE: (180, 180,  180),   # light grey
}


# ---------------------------------------------------------------------------
# Absolute-threshold heatmap
# ---------------------------------------------------------------------------

def create_sharpness_heatmap(
    scores: np.ndarray,
    image_shape: tuple[int, ...],
    sharpness_threshold: float = 100.0,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Return a BGR heatmap image showing per-tile sharpness.

    Colour encoding uses an *absolute* scale anchored at *sharpness_threshold*:
    - The midpoint of the colour scale corresponds to *sharpness_threshold*.
    - Tiles well above the threshold appear at the "hot" (yellow/white) end.
    - Tiles well below appear at the "cold" (blue/dark) end.

    The heatmap is scaled to match *image_shape* using INTER_NEAREST to
    preserve tile boundaries.

    Parameters
    ----------
    scores:
        Float32 array of shape (rows, cols) from :func:`~imgqi.blur.regional_sharpness`.
    image_shape:
        ``(height, width)`` or ``(height, width, channels)`` of the source image.
    sharpness_threshold:
        The calibrated threshold used for classification.
    colormap:
        OpenCV colormap constant.  Default ``COLORMAP_JET``.

    Returns
    -------
    heatmap : np.ndarray, shape (H, W, 3), dtype uint8
    """
    h, w = image_shape[:2]

    # Replace NaN with 0 for display only.
    display = np.where(np.isfinite(scores), scores, 0.0).astype(np.float32)

    # Map scores to [0, 255] where 128 ≈ threshold.
    # Use log scale so that both very low and very high values are visible.
    # Add 1 to avoid log(0).
    log_scores = np.log1p(display)
    log_thresh = math.log1p(sharpness_threshold)

    # Centre on threshold: values at threshold → 128
    scale = 128.0 / max(log_thresh, 1e-6)
    mapped = np.clip(log_scores * scale, 0, 255).astype(np.uint8)

    # Resize to image dimensions preserving tile grid structure.
    enlarged = cv2.resize(mapped, (w, h), interpolation=cv2.INTER_NEAREST)
    heatmap = cv2.applyColorMap(enlarged, colormap)
    return heatmap


# ---------------------------------------------------------------------------
# Tile-state overlay
# ---------------------------------------------------------------------------

def create_state_overlay(
    image: np.ndarray,
    states: np.ndarray,
    tile_size: int,
    alpha: float = 0.40,
    show_inconclusive: bool = False,
) -> np.ndarray:
    """Return *image* with semi-transparent coloured rectangles per tile state.

    Parameters
    ----------
    image:
        Original BGR image.
    states:
        Int32 array of shape (rows, cols) from :func:`~imgqi.blur.classify_tiles`.
    tile_size:
        Tile side length used during analysis (pixels).
    alpha:
        Opacity of the overlay layer.  0 = invisible, 1 = opaque.
    show_inconclusive:
        If True, INCONCLUSIVE tiles are tinted grey.  Defaults to False
        (inconclusive tiles are left uncoloured).

    Returns
    -------
    overlay : np.ndarray, same shape and dtype as *image*
    """
    h, w = image.shape[:2]
    overlay = image.copy()

    rows, cols = states.shape
    for r in range(rows):
        y1 = r * tile_size
        y2 = min(y1 + tile_size, h)
        for c in range(cols):
            state = int(states[r, c])
            if state == TileState.INCONCLUSIVE and not show_inconclusive:
                continue
            colour = _STATE_COLOURS.get(state)
            if colour is None:
                continue
            x1 = c * tile_size
            x2 = min(x1 + tile_size, w)
            cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), colour, thickness=-1)

    return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


# ---------------------------------------------------------------------------
# Grid-cell heatmap (no interpolation, exact tile boundaries)
# ---------------------------------------------------------------------------

def create_grid_heatmap(
    scores: np.ndarray,
    sharpness_threshold: float = 100.0,
    colormap: int = cv2.COLORMAP_JET,
    cell_px: int = 32,
) -> np.ndarray:
    """Render the score matrix as a small grid image (no spatial interpolation).

    Each tile becomes a solid square of *cell_px* × *cell_px* pixels.
    This is the most faithful visualisation of the grid measurements.

    Parameters
    ----------
    scores:
        Float32 array of shape (rows, cols).
    sharpness_threshold:
        Calibrated threshold; tiles at this value render at mid-colourmap.
    colormap:
        OpenCV colormap constant.
    cell_px:
        Pixel size of each cell in the output image.

    Returns
    -------
    grid_img : np.ndarray, shape (rows*cell_px, cols*cell_px, 3), dtype uint8
    """
    display = np.where(np.isfinite(scores), scores, 0.0).astype(np.float32)
    log_scores = np.log1p(display)
    log_thresh = math.log1p(sharpness_threshold)
    scale = 128.0 / max(log_thresh, 1e-6)
    mapped = np.clip(log_scores * scale, 0, 255).astype(np.uint8)

    # Enlarge using INTER_NEAREST for exact cell boundaries.
    rows, cols = mapped.shape
    grid = cv2.resize(mapped, (cols * cell_px, rows * cell_px), interpolation=cv2.INTER_NEAREST)
    return cv2.applyColorMap(grid, colormap)


# ---------------------------------------------------------------------------
# Composite output
# ---------------------------------------------------------------------------

def compose_report_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    state_overlay: np.ndarray,
) -> np.ndarray:
    """Stack the original, heatmap overlay, and state overlay side-by-side.

    All three images must have the same height.  The result is a wide BGR
    image suitable for saving as a single diagnostic JPEG.

    Parameters
    ----------
    image:
        Original BGR image.
    heatmap:
        Heatmap BGR image from :func:`create_sharpness_heatmap`.
    state_overlay:
        State overlay BGR image from :func:`create_state_overlay`.

    Returns
    -------
    composite : np.ndarray
    """
    # Blend heatmap 50/50 with original for context.
    blended = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
    return np.concatenate([image, blended, state_overlay], axis=1)


def add_legend(
    image: np.ndarray,
    sharpness_threshold: float,
    texture_threshold: float,
    global_sharpness: float,
    blurry_fraction: float,
) -> np.ndarray:
    """Burn summary text into the top-left corner of *image* in-place copy.

    Parameters
    ----------
    image:
        BGR image to annotate (a copy is made; original is not modified).
    sharpness_threshold, texture_threshold:
        Thresholds used during analysis.
    global_sharpness:
        Global Laplacian variance score.
    blurry_fraction:
        Fraction of decided tiles that are BLURRY.

    Returns
    -------
    annotated : np.ndarray
    """
    out = image.copy()
    lines = [
        f"Global sharpness: {global_sharpness:.1f}  (thr={sharpness_threshold:.0f})",
        f"Blurry tiles: {blurry_fraction * 100:.1f}%",
        "GREEN=Sharp  RED=Blurry  GREY=Inconcl.",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, lh = 0.55, 1, 22
    # Semi-transparent background bar.
    bar_h = len(lines) * lh + 10
    roi = out[:bar_h, :400]
    roi[:] = (roi * 0.4).astype(np.uint8)

    for i, line in enumerate(lines):
        y = 20 + i * lh
        cv2.putText(out, line, (8, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out

