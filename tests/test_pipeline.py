"""
test_pipeline.py — Integration tests for imgqi.pipeline.analyze_image.
"""

from __future__ import annotations

import numpy as np
import pytest

from imgqi.pipeline import analyze_image, AnalysisResult


class TestAnalyzeImage:
    def test_returns_analysis_result(self, sharp_bgr):
        result = analyze_image(sharp_bgr)
        assert isinstance(result, AnalysisResult)

    def test_dimensions_correct(self, sharp_bgr):
        result = analyze_image(sharp_bgr)
        assert result.width == sharp_bgr.shape[1]
        assert result.height == sharp_bgr.shape[0]

    def test_global_sharpness_positive(self, sharp_bgr):
        result = analyze_image(sharp_bgr)
        assert result.global_sharpness > 0.0

    def test_sharp_higher_than_blurry(self, sharp_bgr, blurry_bgr):
        r_sharp = analyze_image(sharp_bgr)
        r_blurry = analyze_image(blurry_bgr)
        assert r_sharp.global_sharpness > r_blurry.global_sharpness

    def test_regional_scores_shape(self, sharp_bgr):
        import math
        result = analyze_image(sharp_bgr, tile_size=64)
        h, w = sharp_bgr.shape[:2]
        expected = (math.ceil(h / 64), math.ceil(w / 64))
        assert result.regional_scores.shape == expected

    def test_tile_states_same_shape_as_scores(self, sharp_bgr):
        result = analyze_image(sharp_bgr)
        assert result.tile_states.shape == result.regional_scores.shape

    def test_blur_counts_keys_present(self, sharp_bgr):
        result = analyze_image(sharp_bgr)
        for k in ("total_tiles", "sharp", "blurry", "inconclusive", "blurry_fraction"):
            assert k in result.blur_counts

    def test_exposure_keys_present(self, sharp_bgr):
        result = analyze_image(sharp_bgr)
        for k in ("dark_pixel_percent", "bright_pixel_percent", "mean_intensity"):
            assert k in result.exposure

    def test_empty_image_raises(self):
        with pytest.raises(ValueError):
            analyze_image(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_none_raises(self):
        with pytest.raises((ValueError, AttributeError)):
            analyze_image(None)

    def test_warnings_is_list(self, sharp_bgr):
        result = analyze_image(sharp_bgr)
        assert isinstance(result.warnings, list)

    def test_blurry_image_generates_global_blur_warning(self, blurry_bgr):
        # Use a high threshold so the blurry image definitely triggers it.
        result = analyze_image(blurry_bgr, sharpness_threshold=10_000.0)
        assert any("blurry" in w.lower() or "sharpness" in w.lower() for w in result.warnings)

    def test_dark_image_generates_exposure_warning(self):
        dark = np.full((128, 128, 3), 2, dtype=np.uint8)
        result = analyze_image(dark)
        assert any("underexposed" in w.lower() or "dark" in w.lower() for w in result.warnings)

    def test_to_dict_is_serialisable(self, sharp_bgr):
        import json
        result = analyze_image(sharp_bgr)
        d = result.to_dict()
        # Should not raise.
        json.dumps(d)

    def test_to_dict_contains_expected_keys(self, sharp_bgr):
        result = analyze_image(sharp_bgr)
        d = result.to_dict()
        for k in ("width", "height", "global_sharpness", "contrast_std"):
            assert k in d

    def test_save_heatmap(self, sharp_bgr, tmp_path):
        result = analyze_image(sharp_bgr)
        path = result.save_heatmap(tmp_path / "heatmap.jpg", sharp_bgr)
        assert path.exists()
        assert path.stat().st_size > 0

    def test_save_scores(self, sharp_bgr, tmp_path):
        result = analyze_image(sharp_bgr)
        path = result.save_scores(tmp_path / "scores")
        assert path.exists()
        assert path.suffix == ".npz"

    def test_get_heatmap_shape(self, sharp_bgr):
        result = analyze_image(sharp_bgr)
        heatmap = result.get_heatmap(sharp_bgr)
        assert heatmap.shape[:2] == sharp_bgr.shape[:2]

    def test_get_state_overlay_shape(self, sharp_bgr):
        result = analyze_image(sharp_bgr)
        overlay = result.get_state_overlay(sharp_bgr)
        assert overlay.shape == sharp_bgr.shape

    def test_image_path_stored(self, sharp_bgr):
        result = analyze_image(sharp_bgr, image_path="/some/path.jpg")
        assert result.image_path == "/some/path.jpg"

    def test_different_tile_sizes(self, sharp_bgr):
        for ts in [32, 64, 128]:
            result = analyze_image(sharp_bgr, tile_size=ts)
            assert result.tile_size == ts

