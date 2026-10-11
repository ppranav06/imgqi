"""
pipeline.py — End-to-end single-image analysis orchestration.

This module is the primary public interface for library users.
It wires together preprocessing, blur analysis, quality metrics, and
visualization into a single ``analyze_image`` call and returns an
``AnalysisResult`` dataclass that is easy to inspect, serialise, and log.

Usage example
-------------
>>> from imgqi.preprocessing import load_image
>>> from imgqi.pipeline import analyze_image
>>> image = load_image("photo.jpg")
>>> result = analyze_image(image, tile_size=64, sharpness_threshold=100.0)
>>> print(result.global_sharpness)
>>> result.save_heatmap("heatmap.jpg")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from imgqi.preprocessing import to_grayscale, validate_image, image_info
from imgqi.blur import (
    global_sharpness,
    regional_sharpness,
    tile_gradient_energy,
    classify_tiles,
    blur_summary,
    TileState,
)
from imgqi.quality_metrics import (
    contrast_score,
    contrast_percentile_range,
    exposure_metrics,
    high_frequency_residual,
    summarise_histogram,
)
from imgqi.visualization import (
    create_sharpness_heatmap,
    create_state_overlay,
    create_grid_heatmap,
    compose_report_image,
    add_legend,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """All outputs from a single-image analysis run.

    Attributes
    ----------
    image_path:
        Source file path (empty string if image was passed as an array).
    width, height:
        Image dimensions in pixels.
    tile_size:
        Tile side length used for regional analysis.
    sharpness_threshold:
        Laplacian-variance threshold used for BLURRY classification.
    texture_threshold:
        Gradient-energy threshold used for INCONCLUSIVE classification.
    global_sharpness:
        Laplacian variance of the full grayscale image.
    regional_scores:
        Float32 array of shape (rows, cols) with per-tile Laplacian variances.
        NaN marks unreliable tiles.
    tile_states:
        Int32 array of shape (rows, cols) with :class:`~imgqi.blur.TileState` values.
    blur_counts:
        Aggregate tile counts from :func:`~imgqi.blur.blur_summary`.
    contrast_std:
        Standard deviation of pixel intensities.
    contrast_p90_range:
        Intensity range between the 5th and 95th percentiles.
    exposure:
        Dict with ``dark_pixel_percent``, ``bright_pixel_percent``,
        ``mean_intensity``.
    hf_residual:
        High-frequency residual standard deviation.
    histogram:
        Dict with basic histogram statistics.
    warnings:
        List of human-readable warning strings generated from the metrics.
    """

    image_path: str = ""
    width: int = 0
    height: int = 0
    tile_size: int = 64
    sharpness_threshold: float = 100.0
    texture_threshold: float = 20.0

    global_sharpness: float = 0.0
    regional_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    tile_states: np.ndarray = field(default_factory=lambda: np.array([]))
    blur_counts: dict[str, Any] = field(default_factory=dict)

    contrast_std: float = 0.0
    contrast_p90_range: float = 0.0
    exposure: dict[str, float] = field(default_factory=dict)
    hf_residual: float = 0.0
    histogram: dict[str, float] = field(default_factory=dict)

    warnings: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Lazy visualizations (generated on demand, not stored)
    # ------------------------------------------------------------------

    def get_heatmap(self, image: np.ndarray) -> np.ndarray:
        """Generate and return the sharpness heatmap for *image*.

        The heatmap is not cached; call this method each time you need it.
        """
        return create_sharpness_heatmap(
            self.regional_scores,
            image.shape,
            sharpness_threshold=self.sharpness_threshold,
        )

    def get_state_overlay(self, image: np.ndarray, alpha: float = 0.40) -> np.ndarray:
        """Generate and return the tile-state overlay for *image*."""
        return create_state_overlay(
            image,
            self.tile_states,
            self.tile_size,
            alpha=alpha,
        )

    def get_report_image(self, image: np.ndarray) -> np.ndarray:
        """Return a wide composite: original | heatmap blend | state overlay."""
        heatmap = self.get_heatmap(image)
        state_ov = self.get_state_overlay(image)
        composite = compose_report_image(image, heatmap, state_ov)
        annotated = add_legend(
            composite,
            sharpness_threshold=self.sharpness_threshold,
            texture_threshold=self.texture_threshold,
            global_sharpness=self.global_sharpness,
            blurry_fraction=self.blur_counts.get("blurry_fraction", float("nan")),
        )
        return annotated

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_heatmap(self, path: str | Path, image: np.ndarray) -> Path:
        """Write the sharpness heatmap to *path* and return the resolved path."""
        p = Path(path)
        heatmap = self.get_heatmap(image)
        cv2.imwrite(str(p), heatmap)
        return p

    def save_report_image(self, path: str | Path, image: np.ndarray) -> Path:
        """Write the composite report image to *path*."""
        p = Path(path)
        report_img = self.get_report_image(image)
        cv2.imwrite(str(p), report_img)
        return p

    def save_scores(self, path: str | Path) -> Path:
        """Save the regional score matrix as a compressed NumPy file (.npz)."""
        p = Path(path).with_suffix(".npz")
        np.savez_compressed(
            str(p),
            regional_scores=self.regional_scores,
            tile_states=self.tile_states,
        )
        return p

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict of scalar metrics (no arrays)."""
        return {
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
            "tile_size": self.tile_size,
            "sharpness_threshold": self.sharpness_threshold,
            "texture_threshold": self.texture_threshold,
            "global_sharpness": self.global_sharpness,
            "contrast_std": self.contrast_std,
            "contrast_p90_range": self.contrast_p90_range,
            **{f"exposure_{k}": v for k, v in self.exposure.items()},
            "hf_residual": self.hf_residual,
            **{f"hist_{k}": v for k, v in self.histogram.items()},
            **{f"tiles_{k}": v for k, v in self.blur_counts.items()},
            "warnings": "; ".join(self.warnings),
        }

    def print_report(self) -> None:
        """Print a formatted quality report to stdout."""
        sep = "─" * 60
        print(sep)
        print(f"  Image Quality Report")
        if self.image_path:
            print(f"  File  : {self.image_path}")
        print(f"  Size  : {self.width} × {self.height} px")
        print(sep)
        print(f"  Global sharpness (Laplacian var)  : {self.global_sharpness:>10.2f}")
        print(f"    Threshold                         : {self.sharpness_threshold:>10.2f}")
        print(f"  Contrast (intensity std dev)       : {self.contrast_std:>10.2f}")
        print(f"  Contrast (P5–P95 range)            : {self.contrast_p90_range:>10.2f}")
        print(f"  Exposure – dark pixels (<{self.exposure.get('dark_pixel_percent', 0):.0f} thr) : "
              f"{self.exposure.get('dark_pixel_percent', 0):>8.2f} %")
        print(f"  Exposure – bright pixels           : "
              f"{self.exposure.get('bright_pixel_percent', 0):>8.2f} %")
        print(f"  Mean intensity                     : "
              f"{self.exposure.get('mean_intensity', 0):>10.2f}")
        print(f"  High-freq residual (noise proxy)   : {self.hf_residual:>10.4f}")
        print(sep)
        bc = self.blur_counts
        print(f"  Tiles  total={bc.get('total_tiles', '?')}  "
              f"sharp={bc.get('sharp', '?')}  "
              f"blurry={bc.get('blurry', '?')}  "
              f"inconclusive={bc.get('inconclusive', '?')}")
        frac = bc.get("blurry_fraction", float("nan"))
        if frac == frac:  # not NaN
            print(f"  Blurry fraction (of decided tiles) : {frac * 100:.1f} %")
        print(sep)
        if self.warnings:
            print("  ⚠  Warnings:")
            for w in self.warnings:
                print(f"     • {w}")
        else:
            print("  ✓  No quality warnings triggered.")
        print(sep)


# ---------------------------------------------------------------------------
# Warning generation
# ---------------------------------------------------------------------------

def _generate_warnings(
    result: AnalysisResult,
    blur_fraction_threshold: float = 0.30,
    dark_pct_threshold: float = 20.0,
    bright_pct_threshold: float = 10.0,
    contrast_std_threshold: float = 25.0,
    hf_residual_threshold: float = 15.0,
) -> list[str]:
    """Produce human-readable warning strings based on measured metrics."""
    warnings: list[str] = []

    # Global blur
    if result.global_sharpness < result.sharpness_threshold:
        warnings.append(
            f"Global sharpness score ({result.global_sharpness:.1f}) is below the "
            f"threshold ({result.sharpness_threshold:.1f}). The image may be blurry overall."
        )

    # Regional blur
    frac = result.blur_counts.get("blurry_fraction", float("nan"))
    if frac == frac and frac >= blur_fraction_threshold:
        pct = frac * 100
        warnings.append(
            f"{pct:.1f}% of texture-rich tiles were classified as blurry "
            f"(threshold: {blur_fraction_threshold * 100:.0f}%)."
        )

    # Exposure
    dark = result.exposure.get("dark_pixel_percent", 0.0)
    bright = result.exposure.get("bright_pixel_percent", 0.0)
    if dark >= dark_pct_threshold:
        warnings.append(
            f"{dark:.1f}% of pixels are near-black (< intensity 10). "
            "The image may be underexposed."
        )
    if bright >= bright_pct_threshold:
        warnings.append(
            f"{bright:.1f}% of pixels are near-white (> intensity 245). "
            "The image may be overexposed or contain clipped highlights."
        )

    # Contrast
    if result.contrast_std < contrast_std_threshold:
        warnings.append(
            f"Low global contrast (std dev = {result.contrast_std:.1f}). "
            "The image may appear washed out or flat."
        )

    # High-frequency residual
    if result.hf_residual >= hf_residual_threshold:
        warnings.append(
            f"Elevated high-frequency residual ({result.hf_residual:.2f}). "
            "This may indicate noise or very fine texture — treat as an indicator."
        )

    return warnings


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_image(
    image: np.ndarray,
    image_path: str = "",
    tile_size: int = 64,
    sharpness_threshold: float = 100.0,
    texture_threshold: float = 20.0,
) -> AnalysisResult:
    """Run the full quality-analysis pipeline on a single BGR image.

    Parameters
    ----------
    image:
        BGR uint8 numpy array as returned by :func:`~imgqi.preprocessing.load_image`.
    image_path:
        Source file path, used for reporting only.
    tile_size:
        Side length (pixels) of each analysis tile.  Larger → more stable
        estimates; smaller → finer spatial resolution.
    sharpness_threshold:
        Laplacian variance below this value (with sufficient texture) marks a
        tile BLURRY.  Must be tuned for your image corpus.
    texture_threshold:
        Mean squared Sobel gradient below this value marks a tile
        INCONCLUSIVE (naturally smooth region).

    Returns
    -------
    AnalysisResult
        Dataclass containing all scalar metrics, tile arrays, and helpers for
        generating visualizations and saving results.
    """
    validate_image(image)
    info = image_info(image)
    gray = to_grayscale(image)

    # --- Blur ---
    g_sharp = global_sharpness(gray)
    reg_scores = regional_sharpness(gray, tile_size=tile_size)
    grad_energy = tile_gradient_energy(gray, tile_size=tile_size)
    states = classify_tiles(
        reg_scores,
        grad_energy,
        sharpness_threshold=sharpness_threshold,
        texture_threshold=texture_threshold,
    )
    b_summary = blur_summary(states)

    # --- Quality metrics ---
    c_std = contrast_score(gray)
    c_p90 = contrast_percentile_range(gray)
    exp = exposure_metrics(gray)
    hf = high_frequency_residual(gray)
    hist = summarise_histogram(gray)

    result = AnalysisResult(
        image_path=image_path,
        width=info["width"],
        height=info["height"],
        tile_size=tile_size,
        sharpness_threshold=sharpness_threshold,
        texture_threshold=texture_threshold,
        global_sharpness=g_sharp,
        regional_scores=reg_scores,
        tile_states=states,
        blur_counts=b_summary,
        contrast_std=c_std,
        contrast_p90_range=c_p90,
        exposure=exp,
        hf_residual=hf,
        histogram=hist,
    )
    result.warnings = _generate_warnings(result)
    return result

