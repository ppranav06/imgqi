"""
batch.py — Directory-level batch processing and CSV export.

Design
------
* Images are processed one at a time to keep memory usage bounded.
* Results are flushed to CSV after each image so a crash does not lose
  previously completed work.
* Full-resolution heatmaps are NOT persisted by default.  Pass
  ``save_heatmaps=True`` to write them; they will be regenerated on demand.
* Regional score matrices are optionally saved as compressed .npz files.

Supported image extensions: JPEG, PNG, BMP, TIFF, WebP.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from imgqi.preprocessing import load_image
from imgqi.pipeline import analyze_image, AnalysisResult


logger = logging.getLogger(__name__)

# Extensions recognized as images.
_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def iter_images(directory: str | Path, recursive: bool = False) -> Iterator[Path]:
    """Yield paths of supported image files inside *directory*.

    Parameters
    ----------
    directory:
        Root folder to search.
    recursive:
        If True, descend into sub-directories.
    """
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    glob = root.rglob if recursive else root.glob
    for p in sorted(glob("*")):
        if p.suffix.lower() in _IMAGE_SUFFIXES:
            yield p


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def process_directory(
    input_dir: str | Path,
    output_dir: str | Path | None = None,
    tile_size: int = 64,
    sharpness_threshold: float = 100.0,
    texture_threshold: float = 20.0,
    save_heatmaps: bool = False,
    save_scores: bool = False,
    recursive: bool = False,
) -> list[dict]:
    """Analyse every image in *input_dir* and return a list of metric dicts.

    Parameters
    ----------
    input_dir:
        Folder containing images.
    output_dir:
        Where to write CSV report, heatmaps, and score arrays.
        Defaults to *input_dir* / ``imgqi_output``.
    tile_size:
        Analysis tile size (pixels).
    sharpness_threshold:
        Laplacian variance threshold for BLURRY classification.
    texture_threshold:
        Gradient energy threshold for INCONCLUSIVE classification.
    save_heatmaps:
        If True, write a heatmap JPEG next to each result.
    save_scores:
        If True, write regional scores as .npz alongside results.
    recursive:
        If True, search *input_dir* sub-directories for images.

    Returns
    -------
    list[dict]
        One dict per successfully processed image, suitable for CSV export.
        Failed images are logged and included with an ``error`` key.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir / "imgqi_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "report.csv"
    image_paths = list(iter_images(input_dir, recursive=recursive))
    total = len(image_paths)
    logger.info("Found %d image(s) in %s", total, input_dir)

    all_rows: list[dict] = []
    csv_writer: csv.DictWriter | None = None
    csv_file = None

    try:
        for idx, img_path in enumerate(image_paths, start=1):
            logger.info("[%d/%d] Processing %s", idx, total, img_path.name)
            t0 = time.perf_counter()
            row: dict = {"image_path": str(img_path)}
            try:
                image = load_image(img_path)
                result = analyze_image(
                    image,
                    image_path=str(img_path),
                    tile_size=tile_size,
                    sharpness_threshold=sharpness_threshold,
                    texture_threshold=texture_threshold,
                )
                row = result.to_dict()
                stem = img_path.stem

                if save_heatmaps:
                    heatmap_path = output_dir / f"{stem}_heatmap.jpg"
                    result.save_heatmap(heatmap_path, image)
                    logger.debug("  Heatmap saved: %s", heatmap_path)

                if save_scores:
                    scores_path = output_dir / f"{stem}_scores"
                    result.save_scores(scores_path)
                    logger.debug("  Scores saved: %s.npz", scores_path)

            except Exception as exc:
                logger.warning("  Failed to process %s: %s", img_path.name, exc)
                row["error"] = str(exc)

            row["processing_time_s"] = round(time.perf_counter() - t0, 4)
            all_rows.append(row)

            # Write CSV incrementally so partial results are preserved.
            if csv_writer is None:
                csv_file = open(csv_path, "w", newline="", encoding="utf-8")
                fieldnames = list(row.keys())
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
                csv_writer.writeheader()
            csv_writer.writerow(row)
            if csv_file:
                csv_file.flush()

    finally:
        if csv_file:
            csv_file.close()

    logger.info("Report written to %s", csv_path)
    return all_rows


# ---------------------------------------------------------------------------
# Convenience: load a previously saved score matrix
# ---------------------------------------------------------------------------

def load_scores(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load regional scores and tile states from a .npz file.

    Returns
    -------
    (regional_scores, tile_states)
    """
    data = np.load(str(npz_path))
    return data["regional_scores"], data["tile_states"]

