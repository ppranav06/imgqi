"""
test_preprocessing.py — Tests for imgqi.preprocessing.
"""

from __future__ import annotations

import numpy as np
import pytest

from imgqi.preprocessing import (
    load_image,
    to_grayscale,
    validate_image,
    image_info,
)


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_image(tmp_path / "nonexistent.jpg")

    def test_invalid_file_raises_value_error(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not an image at all!!!")
        with pytest.raises(ValueError):
            load_image(bad)

    def test_valid_image_loads_correctly(self, tmp_path):
        import cv2
        img = np.full((50, 80, 3), 128, dtype=np.uint8)
        path = tmp_path / "test.png"
        cv2.imwrite(str(path), img)
        loaded = load_image(path)
        assert loaded.shape == (50, 80, 3)
        assert loaded.dtype == np.uint8


# ---------------------------------------------------------------------------
# to_grayscale
# ---------------------------------------------------------------------------

class TestToGrayscale:
    def test_bgr_converts_to_2d(self, sharp_bgr):
        gray = to_grayscale(sharp_bgr)
        assert gray.ndim == 2

    def test_already_grayscale_passthrough(self, sharp_gray):
        gray = to_grayscale(sharp_gray)
        assert gray.ndim == 2
        assert gray is sharp_gray  # must be the same object

    def test_shape_preserved(self, sharp_bgr):
        gray = to_grayscale(sharp_bgr)
        assert gray.shape == sharp_bgr.shape[:2]


# ---------------------------------------------------------------------------
# validate_image
# ---------------------------------------------------------------------------

class TestValidateImage:
    def test_valid_image_passes(self, sharp_bgr):
        validate_image(sharp_bgr)  # should not raise

    def test_none_raises(self):
        with pytest.raises(ValueError, match="None"):
            validate_image(None)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            validate_image(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_wrong_dtype_raises(self):
        img = np.zeros((10, 10, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="uint8"):
            validate_image(img)


# ---------------------------------------------------------------------------
# image_info
# ---------------------------------------------------------------------------

class TestImageInfo:
    def test_bgr_info(self, sharp_bgr):
        info = image_info(sharp_bgr)
        assert info["width"] == sharp_bgr.shape[1]
        assert info["height"] == sharp_bgr.shape[0]
        assert info["channels"] == 3

    def test_gray_info(self, sharp_gray):
        info = image_info(sharp_gray)
        assert info["channels"] == 1

