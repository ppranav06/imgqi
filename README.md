# imgqi — Image Quality Inspector

Regional blur detection and explainable image quality assessment using classical image processing.

---

## Features

| Feature | Details |
|---|---|
| Global blur detection | Laplacian variance over the full image |
| Regional blur detection | Tile-based Laplacian variance with configurable tile size |
| 3-state classification | SHARP / BLURRY / INCONCLUSIVE (avoids mislabelling smooth regions) |
| Gradient-energy gate | Sobel MSG distinguishes blank walls from blurry tiles |
| Sharpness heatmap | Absolute-threshold colourmap, reproducible across images |
| Tile-state overlay | Coloured rectangles per tile (green=sharp, red=blurry, grey=inconclusive) |
| Contrast analysis | Intensity std dev + P5–P95 percentile range |
| Exposure analysis | Near-black and near-white pixel fractions |
| Noise proxy | Gaussian residual standard deviation |
| Quality report | Human-readable warnings with per-metric explanations |
| Batch processing | Directory scan, incremental CSV export, optional .npz score caching |
| CLI | Single-image and batch modes with configurable thresholds |

---

## Installation

```bash
# Clone and enter the project
git clone <repo-url>
cd ImageProcessing-BlurDetection

# Install with uv (recommended)
uv sync --extra dev

# Or with pip
pip install -e ".[dev]"
```

**Dependencies:** `opencv-python`, `numpy`, `matplotlib`  
**Dev dependencies:** `pytest`, `pytest-cov`

---

## Quick Start

### Library

```python
from imgqi.preprocessing import load_image
from imgqi.pipeline import analyze_image

image = load_image("photo.jpg")
result = analyze_image(image, tile_size=64, sharpness_threshold=100.0)

result.print_report()

# Save visualisations
result.save_heatmap("heatmap.jpg", image)
result.save_report_image("report.jpg", image)  # original | heatmap blend | tile overlay

# Persist the score matrix (tiny — no full-res heatmap stored)
result.save_scores("photo_scores")  # → photo_scores.npz

# Serialise metrics to JSON / CSV
import json
print(json.dumps(result.to_dict(), indent=2))
```

### CLI — single image

```bash
imgqi photo.jpg
imgqi photo.jpg --tile-size 128 --threshold 150 --heatmap heatmap.jpg --report-image report.jpg
```

### CLI — batch mode

```bash
imgqi --batch ./photos/
imgqi --batch ./photos/ --output ./results --save-heatmaps --save-scores --recursive -v
```

The batch run writes `results/report.csv` with one row per image and, optionally, heatmap JPEGs and `.npz` score files.

---

## CLI Reference

```
imgqi <path> [options]

positional:
  path                   Image file or directory

mode:
  --batch                Treat path as a directory

analysis:
  --tile-size N          Tile side length in pixels (default: 64)
  --threshold T          Laplacian variance threshold for BLURRY (default: 100.0)
  --texture T            Gradient energy threshold for INCONCLUSIVE (default: 20.0)

single-image:
  --heatmap PATH         Save sharpness heatmap to PATH
  --report-image PATH    Save composite report image to PATH

batch:
  --save-heatmaps        Write a heatmap JPEG for every image
  --save-scores          Write regional scores as .npz for every image
  --recursive            Search sub-directories

general:
  -v, --verbose          Enable DEBUG logging
```

---

## Module Layout

```
src/imgqi/
├── __init__.py          Public API re-exports
├── preprocessing.py     load_image, to_grayscale, validate_image
├── blur.py              global_sharpness, regional_sharpness, classify_tiles
├── quality_metrics.py   contrast_score, exposure_metrics, hf_residual
├── visualization.py     heatmaps, tile-state overlay, composite image
├── pipeline.py          analyze_image → AnalysisResult
├── batch.py             process_directory, iter_images, load_scores
└── cli.py               argparse CLI entry point

tests/
├── conftest.py          Synthetic image fixtures (no external files needed)
├── test_preprocessing.py
├── test_blur.py
├── test_metrics.py
├── test_pipeline.py
└── test_batch.py
```

---

## Algorithm Summary

### Global sharpness

$$S_{\text{global}} = \operatorname{Var}(\nabla^2 I)$$

Higher variance → more high-frequency detail. A low score may indicate blur but also occurs with legitimately smooth content.

### Regional sharpness

The Laplacian is computed **once** over the full image and then sliced into tiles. This avoids artificial boundary artefacts from filtering each tile independently.

$$S_{i,j} = \operatorname{Var}(\nabla^2 I[y_i : y_i+T,\; x_j : x_j+T])$$

### Three-state tile classification

| Score | Gradient energy | State |
|---|---|---|
| any | < texture_threshold | **INCONCLUSIVE** (smooth region — cannot distinguish blur from legitimate smoothness) |
| ≥ sharpness_threshold | ≥ texture_threshold | **SHARP** |
| < sharpness_threshold | ≥ texture_threshold | **BLURRY** |

The gradient-energy gate (mean squared Sobel gradient) prevents sky, blank paper, and plain walls from being mislabelled as blurry.

### Heatmap colour scale

Uses a **log-scale absolute threshold**: tiles at `sharpness_threshold` render at the midpoint of the colourmap. This makes scores comparable across images, unlike per-image percentile normalisation.

---

## Running Tests

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=src/imgqi --cov-report=term-missing
```

All 83 tests use synthetic images generated in code — no external test files needed.

---

## Documented Limitations

- **No universal threshold.** Scores depend on content, resolution, texture, and noise. Calibrate `--threshold` on representative images for your use case.
- **Blur is not always a defect.** Shallow depth of field is intentional; the system cannot determine intent.
- **Noise inflates sharpness.** Heavy noise raises Laplacian variance, making a noisy image appear sharper than it is.
- **High-frequency residual ≠ noise.** Fine texture and sharp edges also contribute to this metric.
- **Fixed tile boundaries** may split objects or edges. Overlapping windows could reduce boundary artefacts at higher computational cost.
- **Blur-type classification is not implemented.** Distinguishing motion blur from defocus using classical features alone is unreliable in general.

---

## Storage Strategy

| Data | Storage |
|---|---|
| Original image | Untouched (referenced by path) |
| Scalar metrics | JSON / CSV (via `to_dict()`) |
| Regional scores | Optional `.npz` (< 10 KB for a typical 1080p image at tile_size=64) |
| Full-resolution heatmap | Generated on demand — not persisted by default |

