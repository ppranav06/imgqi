"""
test_batch.py — Tests for imgqi.batch.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from imgqi.batch import iter_images, process_directory, load_scores


def _write_test_image(path: Path, value: int = 128, size: int = 64) -> None:
    img = np.full((size, size, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# iter_images
# ---------------------------------------------------------------------------

class TestIterImages:
    def test_finds_images(self, tmp_path):
        for name in ("a.jpg", "b.png", "c.bmp"):
            _write_test_image(tmp_path / name)
        found = list(iter_images(tmp_path))
        assert len(found) == 3

    def test_ignores_non_images(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("x,y")
        _write_test_image(tmp_path / "photo.jpg")
        found = list(iter_images(tmp_path))
        assert len(found) == 1

    def test_not_a_directory_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            list(iter_images(tmp_path / "nonexistent"))

    def test_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        _write_test_image(tmp_path / "top.jpg")
        _write_test_image(sub / "nested.jpg")
        flat = list(iter_images(tmp_path, recursive=False))
        deep = list(iter_images(tmp_path, recursive=True))
        assert len(flat) == 1
        assert len(deep) == 2


# ---------------------------------------------------------------------------
# process_directory
# ---------------------------------------------------------------------------

class TestProcessDirectory:
    def test_produces_csv(self, tmp_path):
        _write_test_image(tmp_path / "img.jpg")
        out = tmp_path / "out"
        process_directory(tmp_path, output_dir=out)
        assert (out / "report.csv").exists()

    def test_csv_has_row_per_image(self, tmp_path):
        for i in range(3):
            _write_test_image(tmp_path / f"img_{i}.jpg")
        out = tmp_path / "out"
        process_directory(tmp_path, output_dir=out)
        with open(out / "report.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_returns_list_of_dicts(self, tmp_path):
        _write_test_image(tmp_path / "img.jpg")
        rows = process_directory(tmp_path, output_dir=tmp_path / "out")
        assert isinstance(rows, list)
        assert isinstance(rows[0], dict)

    def test_save_heatmaps_creates_files(self, tmp_path):
        _write_test_image(tmp_path / "img.jpg")
        out = tmp_path / "out"
        process_directory(tmp_path, output_dir=out, save_heatmaps=True)
        heatmaps = list(out.glob("*_heatmap.jpg"))
        assert len(heatmaps) == 1

    def test_save_scores_creates_npz(self, tmp_path):
        _write_test_image(tmp_path / "img.jpg")
        out = tmp_path / "out"
        process_directory(tmp_path, output_dir=out, save_scores=True)
        npz_files = list(out.glob("*.npz"))
        assert len(npz_files) == 1

    def test_corrupt_image_logs_error_not_crash(self, tmp_path):
        (tmp_path / "bad.jpg").write_bytes(b"not an image")
        _write_test_image(tmp_path / "good.jpg")
        out = tmp_path / "out"
        rows = process_directory(tmp_path, output_dir=out)
        errors = [r for r in rows if "error" in r]
        ok = [r for r in rows if "error" not in r]
        assert len(errors) == 1
        assert len(ok) == 1


# ---------------------------------------------------------------------------
# load_scores
# ---------------------------------------------------------------------------

class TestLoadScores:
    def test_roundtrip(self, tmp_path, sharp_bgr):
        from imgqi.pipeline import analyze_image
        result = analyze_image(sharp_bgr)
        saved = result.save_scores(tmp_path / "scores")
        loaded_scores, loaded_states = load_scores(saved)
        np.testing.assert_array_equal(loaded_scores, result.regional_scores)
        np.testing.assert_array_equal(loaded_states, result.tile_states)

