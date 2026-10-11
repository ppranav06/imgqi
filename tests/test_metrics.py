"""
test_metrics.py — Tests for imgqi.quality_metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from imgqi.quality_metrics import (
    contrast_score,
    contrast_percentile_range,
    exposure_metrics,
    high_frequency_residual,
    summarise_histogram,
)


# ---------------------------------------------------------------------------
# contrast_score
# ---------------------------------------------------------------------------

class TestContrastScore:
    def test_flat_image_zero_contrast(self, smooth_gray):
        assert contrast_score(smooth_gray) == pytest.approx(0.0, abs=1e-3)

    def test_varied_image_nonzero_contrast(self, sharp_gray):
        assert contrast_score(sharp_gray) > 0.0

    def test_dark_vs_bright_images(self, dark_gray, bright_gray):
        # Both flat — both should have ~0 contrast.
        assert contrast_score(dark_gray) < 1.0
        assert contrast_score(bright_gray) < 1.0


# ---------------------------------------------------------------------------
# contrast_percentile_range
# ---------------------------------------------------------------------------

class TestContrastPercentileRange:
    def test_flat_image_zero_range(self, smooth_gray):
        assert contrast_percentile_range(smooth_gray) == pytest.approx(0.0, abs=1.0)

    def test_sharp_image_positive_range(self, sharp_gray):
        assert contrast_percentile_range(sharp_gray) > 0.0

    def test_range_less_than_full_range(self, sharp_gray):
        full_range = float(sharp_gray.max()) - float(sharp_gray.min())
        p_range = contrast_percentile_range(sharp_gray)
        assert p_range <= full_range


# ---------------------------------------------------------------------------
# exposure_metrics
# ---------------------------------------------------------------------------

class TestExposureMetrics:
    def test_dark_image_has_high_dark_percent(self, dark_gray):
        m = exposure_metrics(dark_gray, dark_threshold=10)
        assert m["dark_pixel_percent"] > 90.0

    def test_bright_image_has_high_bright_percent(self, bright_gray):
        m = exposure_metrics(bright_gray, bright_threshold=245)
        assert m["bright_pixel_percent"] > 90.0

    def test_mid_gray_has_low_both(self, smooth_gray):
        m = exposure_metrics(smooth_gray)
        assert m["dark_pixel_percent"] < 1.0
        assert m["bright_pixel_percent"] < 1.0

    def test_mean_intensity_in_range(self, sharp_gray):
        m = exposure_metrics(sharp_gray)
        assert 0.0 <= m["mean_intensity"] <= 255.0

    def test_percentages_non_negative(self, sharp_gray):
        m = exposure_metrics(sharp_gray)
        assert m["dark_pixel_percent"] >= 0.0
        assert m["bright_pixel_percent"] >= 0.0

    def test_custom_thresholds(self, smooth_gray):
        """Smooth mid-grey image with extreme thresholds.

        With dark_threshold=200, all pixels at 128 are < 200 → dark ~100%.
        With bright_threshold=50, all pixels at 128 are > 50 → bright ~100%.
        Both overlap: the same pixels satisfy both conditions simultaneously.
        """
        m = exposure_metrics(smooth_gray, dark_threshold=200, bright_threshold=50)
        assert m["dark_pixel_percent"] > 90.0    # 128 < 200
        assert m["bright_pixel_percent"] > 90.0  # 128 > 50


# ---------------------------------------------------------------------------
# high_frequency_residual
# ---------------------------------------------------------------------------

class TestHighFrequencyResidual:
    def test_smooth_image_low_residual(self, smooth_gray):
        assert high_frequency_residual(smooth_gray) < 1.0

    def test_noisy_image_higher_than_sharp(self, noisy_gray, sharp_gray):
        assert high_frequency_residual(noisy_gray) > high_frequency_residual(sharp_gray)

    def test_blurry_image_lower_than_sharp(self, blurry_gray, sharp_gray):
        assert high_frequency_residual(blurry_gray) < high_frequency_residual(sharp_gray)

    def test_invalid_ksize_raises(self, sharp_gray):
        with pytest.raises(ValueError, match="ksize"):
            high_frequency_residual(sharp_gray, ksize=4)


# ---------------------------------------------------------------------------
# summarise_histogram
# ---------------------------------------------------------------------------

class TestSummariseHistogram:
    def test_keys_present(self, sharp_gray):
        hist = summarise_histogram(sharp_gray)
        for key in ("mean", "std", "p5", "p25", "p50", "p75", "p95"):
            assert key in hist

    def test_percentile_order(self, sharp_gray):
        h = summarise_histogram(sharp_gray)
        assert h["p5"] <= h["p25"] <= h["p50"] <= h["p75"] <= h["p95"]

    def test_mean_in_range(self, sharp_gray):
        h = summarise_histogram(sharp_gray)
        assert 0.0 <= h["mean"] <= 255.0
