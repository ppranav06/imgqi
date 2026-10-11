"""
test_blur.py — Tests for imgqi.blur.

Core invariants tested
-----------------------
1. Sharp images score HIGHER than their blurred counterparts.
2. Smooth (blank) images produce INCONCLUSIVE tiles, not BLURRY.
3. Mixed images have more BLURRY tiles in the blurry half.
4. The score matrix has the correct shape for given tile sizes.
5. Classify_tiles correctly separates the three states.
6. blur_summary fractions are in [0, 1].
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from imgqi.blur import (
    global_sharpness,
    regional_sharpness,
    tile_gradient_energy,
    classify_tiles,
    blur_summary,
    TileState,
)


# ---------------------------------------------------------------------------
# global_sharpness
# ---------------------------------------------------------------------------

class TestGlobalSharpness:
    def test_sharp_scores_higher_than_blurry(self, sharp_gray, blurry_gray):
        assert global_sharpness(sharp_gray) > global_sharpness(blurry_gray)

    def test_smooth_image_is_low(self, smooth_gray):
        assert global_sharpness(smooth_gray) < 5.0

    def test_result_is_non_negative(self, sharp_gray):
        assert global_sharpness(sharp_gray) >= 0.0

    def test_noisy_image_scores_high(self, noisy_gray, blurry_gray):
        """Noise inflates Laplacian variance — documented limitation."""
        assert global_sharpness(noisy_gray) > global_sharpness(blurry_gray)


# ---------------------------------------------------------------------------
# regional_sharpness
# ---------------------------------------------------------------------------

class TestRegionalSharpness:
    def test_shape_matches_ceil_division(self, sharp_gray):
        for tile_size in [32, 64, 128]:
            h, w = sharp_gray.shape
            scores = regional_sharpness(sharp_gray, tile_size=tile_size)
            expected_rows = math.ceil(h / tile_size)
            expected_cols = math.ceil(w / tile_size)
            assert scores.shape == (expected_rows, expected_cols), (
                f"Expected {(expected_rows, expected_cols)}, got {scores.shape} "
                f"for tile_size={tile_size}"
            )

    def test_dtype_is_float32(self, sharp_gray):
        scores = regional_sharpness(sharp_gray)
        assert scores.dtype == np.float32

    def test_sharp_scores_exceed_blurry_median(self, sharp_gray, blurry_gray):
        s_sharp = regional_sharpness(sharp_gray)
        s_blurry = regional_sharpness(blurry_gray)
        # Compare medians of valid (non-NaN) tiles.
        median_sharp = float(np.nanmedian(s_sharp))
        median_blurry = float(np.nanmedian(s_blurry))
        assert median_sharp > median_blurry, (
            f"Median sharp={median_sharp:.2f}, blurry={median_blurry:.2f}"
        )

    def test_invalid_tile_size_raises(self, sharp_gray):
        with pytest.raises(ValueError):
            regional_sharpness(sharp_gray, tile_size=3)

    def test_no_all_nan_for_large_image(self, sharp_gray):
        scores = regional_sharpness(sharp_gray, tile_size=32)
        assert np.any(np.isfinite(scores))


# ---------------------------------------------------------------------------
# tile_gradient_energy
# ---------------------------------------------------------------------------

class TestTileGradientEnergy:
    def test_smooth_image_has_very_low_energy(self, smooth_gray):
        energy = tile_gradient_energy(smooth_gray)
        assert float(np.nanmax(energy)) < 10.0, (
            "Smooth image should have near-zero gradient energy."
        )

    def test_textured_image_has_higher_energy_than_blurry(self, sharp_gray, blurry_gray):
        e_sharp = tile_gradient_energy(sharp_gray)
        e_blurry = tile_gradient_energy(blurry_gray)
        assert float(np.nanmedian(e_sharp)) > float(np.nanmedian(e_blurry))

    def test_shape_matches_scores(self, sharp_gray):
        scores = regional_sharpness(sharp_gray, tile_size=64)
        energy = tile_gradient_energy(sharp_gray, tile_size=64)
        assert scores.shape == energy.shape


# ---------------------------------------------------------------------------
# classify_tiles
# ---------------------------------------------------------------------------

class TestClassifyTiles:
    def _classify(self, gray, thr=100.0, tex=20.0):
        scores = regional_sharpness(gray, tile_size=64)
        energy = tile_gradient_energy(gray, tile_size=64)
        return classify_tiles(scores, energy, sharpness_threshold=thr, texture_threshold=tex)

    def test_smooth_tiles_are_inconclusive(self, smooth_gray):
        states = self._classify(smooth_gray)
        # Most tiles should be INCONCLUSIVE — the image has no texture.
        inconclusive_ratio = np.mean(states == TileState.INCONCLUSIVE)
        assert inconclusive_ratio > 0.80, (
            f"Expected mostly INCONCLUSIVE for blank image, got {inconclusive_ratio:.2f}"
        )

    def test_sharp_image_mostly_sharp(self, sharp_gray):
        states = self._classify(sharp_gray)
        sharp_ratio = np.mean(states == TileState.SHARP)
        assert sharp_ratio > 0.50, (
            f"Expected mostly SHARP for textured image, got {sharp_ratio:.2f}"
        )

    def test_blurry_image_mostly_blurry(self, blurry_gray):
        states = self._classify(blurry_gray)
        blurry_ratio = np.mean(states == TileState.BLURRY)
        assert blurry_ratio > 0.50, (
            f"Expected mostly BLURRY for blurred image, got {blurry_ratio:.2f}"
        )

    def test_mixed_image_regional_asymmetry(self, mixed_gray):
        """Left half (sharp) should have fewer BLURRY tiles than right half (blurry)."""
        scores = regional_sharpness(mixed_gray, tile_size=64)
        energy = tile_gradient_energy(mixed_gray, tile_size=64)
        states = classify_tiles(scores, energy)
        cols = states.shape[1]
        mid = cols // 2
        left_blurry = np.mean(states[:, :mid] == TileState.BLURRY)
        right_blurry = np.mean(states[:, mid:] == TileState.BLURRY)
        assert left_blurry < right_blurry, (
            f"Left blurry={left_blurry:.2f}, right blurry={right_blurry:.2f}"
        )

    def test_valid_states_are_enum_values(self, sharp_gray):
        states = self._classify(sharp_gray)
        valid_values = {int(s) for s in TileState}
        assert set(np.unique(states)).issubset(valid_values)


# ---------------------------------------------------------------------------
# blur_summary
# ---------------------------------------------------------------------------

class TestBlurSummary:
    def test_totals_add_up(self, sharp_gray):
        scores = regional_sharpness(sharp_gray)
        energy = tile_gradient_energy(sharp_gray)
        states = classify_tiles(scores, energy)
        s = blur_summary(states)
        assert s["sharp"] + s["blurry"] + s["inconclusive"] == s["total_tiles"]

    def test_blurry_fraction_in_range(self, sharp_gray):
        scores = regional_sharpness(sharp_gray)
        energy = tile_gradient_energy(sharp_gray)
        states = classify_tiles(scores, energy)
        s = blur_summary(states)
        frac = s["blurry_fraction"]
        if frac == frac:  # not NaN
            assert 0.0 <= frac <= 1.0

    def test_all_inconclusive_fraction_is_nan(self, smooth_gray):
        scores = regional_sharpness(smooth_gray)
        energy = tile_gradient_energy(smooth_gray)
        states = classify_tiles(scores, energy)
        s = blur_summary(states)
        # If no decided tiles exist, fraction should be NaN.
        if s["sharp"] == 0 and s["blurry"] == 0:
            assert s["blurry_fraction"] != s["blurry_fraction"]  # NaN check

