"""
cli.py — Command-line interface for imgqi.

Usage
-----
Single image:
    imgqi photo.jpg
    imgqi photo.jpg --output ./results --tile-size 128 --threshold 150

Batch mode:
    imgqi --batch ./photos/
    imgqi --batch ./photos/ --output ./results --save-heatmaps --save-scores --recursive

Options
-------
  path              Image file or directory (required)
  --batch           Treat path as a directory and process all images
  --output DIR      Output directory for reports and visualisations
  --tile-size N     Analysis tile size in pixels (default 64)
  --threshold T     Laplacian variance threshold for BLURRY (default 100.0)
  --texture T       Gradient energy threshold for INCONCLUSIVE (default 20.0)
  --save-heatmaps   Write heatmap JPEG for each image (batch mode only)
  --save-scores     Write regional score .npz for each image (batch mode only)
  --recursive       Recurse into sub-directories (batch mode only)
  --report-image P  Save composite report image to path P (single mode only)
  -v, --verbose     Enable DEBUG logging
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="imgqi",
        description="Image Quality Inspector — regional blur detection and quality analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "path",
        help="Image file (single mode) or directory (batch mode).",
    )

    # Mode
    p.add_argument(
        "--batch",
        action="store_true",
        default=False,
        help="Treat PATH as a directory and process all supported images.",
    )

    # Output
    p.add_argument(
        "--output",
        metavar="DIR",
        default=None,
        help="Directory for output files (heatmaps, scores, CSV). "
             "Defaults to ./imgqi_output relative to the input.",
    )

    # Analysis parameters
    p.add_argument(
        "--tile-size",
        type=int,
        default=64,
        metavar="N",
        help="Tile side length in pixels (default: 64).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=100.0,
        metavar="T",
        help="Laplacian variance below this value flags a tile BLURRY (default: 100.0).",
    )
    p.add_argument(
        "--texture",
        type=float,
        default=20.0,
        metavar="T",
        help="Gradient energy below this value flags a tile INCONCLUSIVE (default: 20.0).",
    )

    # Batch-specific options
    batch_grp = p.add_argument_group("batch options")
    batch_grp.add_argument(
        "--save-heatmaps",
        action="store_true",
        default=False,
        help="Write sharpness heatmap JPEG for each image.",
    )
    batch_grp.add_argument(
        "--save-scores",
        action="store_true",
        default=False,
        help="Write regional scores as compressed .npz files.",
    )
    batch_grp.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Recurse into sub-directories.",
    )

    # Single-image-specific options
    single_grp = p.add_argument_group("single-image options")
    single_grp.add_argument(
        "--report-image",
        metavar="PATH",
        default=None,
        help="Save the composite report image (original | heatmap | state overlay) to PATH.",
    )
    single_grp.add_argument(
        "--heatmap",
        metavar="PATH",
        default=None,
        help="Save the sharpness heatmap to PATH.",
    )

    # Logging
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )

    return p


def _run_single(args: argparse.Namespace) -> int:
    """Analyse one image and print a report.  Returns exit code."""
    from imgqi.preprocessing import load_image
    from imgqi.pipeline import analyze_image

    img_path = Path(args.path)
    if not img_path.exists():
        print(f"Error: file not found: {img_path}", file=sys.stderr)
        return 1

    try:
        image = load_image(img_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading image: {exc}", file=sys.stderr)
        return 1

    result = analyze_image(
        image,
        image_path=str(img_path),
        tile_size=args.tile_size,
        sharpness_threshold=args.threshold,
        texture_threshold=args.texture,
    )

    result.print_report()

    if args.heatmap:
        saved = result.save_heatmap(args.heatmap, image)
        print(f"Heatmap saved → {saved}")

    if args.report_image:
        saved = result.save_report_image(args.report_image, image)
        print(f"Report image saved → {saved}")

    return 0


def _run_batch(args: argparse.Namespace) -> int:
    """Process a directory of images.  Returns exit code."""
    from imgqi.batch import process_directory

    input_dir = Path(args.path)
    if not input_dir.is_dir():
        print(f"Error: not a directory: {input_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output) if args.output else input_dir / "imgqi_output"

    rows = process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        tile_size=args.tile_size,
        sharpness_threshold=args.threshold,
        texture_threshold=args.texture,
        save_heatmaps=args.save_heatmaps,
        save_scores=args.save_scores,
        recursive=args.recursive,
    )

    ok = sum(1 for r in rows if "error" not in r)
    failed = len(rows) - ok
    print(f"\nProcessed {ok} image(s) successfully, {failed} failed.")
    print(f"Report → {output_dir / 'report.csv'}")

    return 0 if failed == 0 else 2


def main() -> None:
    """CLI entry point registered in pyproject.toml."""
    parser = _build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.batch:
        sys.exit(_run_batch(args))
    else:
        sys.exit(_run_single(args))


if __name__ == "__main__":
    main()

