"""Focused tests for the bezier_and_hold schedule."""

import numpy as np
import pytest

from tipbox.bifurcation.schedule import bezier_and_hold


class TestBezierAndHoldSchedule:
    """Tests for bezier_and_hold() factory function."""

    def test_bezier_and_hold_basic(self):
        """Bezier-and-hold starts at start, ends at final_level, then holds."""
        sched = bezier_and_hold(
            hold_at=0.7,
            final_level=1.0,
            start=0.0,
            control_points=[0.2, 0.85, 0.55],
        )
        values = sched(1000)

        hold_idx = int(1000 * 0.7)
        assert np.isclose(values[0], 0.0, atol=1e-6)
        assert np.isclose(values[-1], 1.0, atol=1e-6)
        assert np.all(values >= -1e-6)
        assert np.all(values <= 1.0 + 1e-6)
        assert np.allclose(values[hold_idx:], 1.0, atol=1e-6)

    def test_bezier_and_hold_allows_down_ramps(self):
        """Non-monotonic control levels can create local downward ramps."""
        sched = bezier_and_hold(
            hold_at=0.8,
            final_level=1.0,
            start=0.0,
            control_points=[0.85, 0.76, 0.42, 0.26],
        )
        values = np.array(sched(1000))
        hold_idx = int(1000 * 0.8)
        diffs = np.diff(values[:hold_idx])
        assert np.any(diffs < -1e-4)

    def test_bezier_and_hold_respects_slope_cap(self):
        """Slope cap bounds the realized schedule slope."""
        max_abs_slope = 1.5
        sched = bezier_and_hold(
            hold_at=0.9,
            final_level=1.0,
            start=0.0,
            control_points=[1.0, 0.0, 1.0, 0.0, 1.0],
            max_abs_slope=max_abs_slope,
        )
        values = np.array(sched(4000))
        dt = 1.0 / (len(values) - 1)
        slope = np.abs(np.diff(values) / dt)
        assert slope.max() <= max_abs_slope * 1.05

    def test_bezier_and_hold_impossible_slope_raises(self):
        """Should raise when slope cap cannot reach final level in time."""
        with pytest.raises(ValueError, match="too small"):
            bezier_and_hold(hold_at=0.5, final_level=1.0, start=0.0, max_abs_slope=1.0)


class TestSmoothHold:
    """Tests for the smooth_hold parameter."""

    def test_smooth_hold_near_zero_terminal_slope(self):
        """With smooth_hold=True, the ramp slope near hold_at should be ~zero."""
        sched = bezier_and_hold(
            hold_at=0.7,
            final_level=1.0,
            start=0.0,
            control_points=[0.3, 0.8],
            smooth_hold=True,
        )
        values = np.array(sched(2000))
        hold_idx = int(2000 * 0.7)
        # Check slope over the last few ramp samples before hold
        ramp_end = values[max(hold_idx - 10, 0) : hold_idx]
        dt = 1.0 / (len(values) - 1)
        slopes = np.abs(np.diff(ramp_end) / dt)
        # Terminal slope should be much smaller than the mid-ramp slope
        mid_idx = hold_idx // 2
        mid_slopes = np.abs(np.diff(values[mid_idx : mid_idx + 10]) / dt)
        assert slopes.mean() < mid_slopes.mean() * 0.3

    def test_smooth_hold_no_overshoot(self):
        """smooth_hold=True must not overshoot final_level."""
        sched = bezier_and_hold(
            hold_at=0.8,
            final_level=1.0,
            start=0.0,
            control_points=[0.9, 0.95],
            smooth_hold=True,
        )
        values = np.array(sched(2000))
        assert np.all(values <= 1.0 + 1e-6)
        assert np.all(values >= -1e-6)

    def test_smooth_hold_false_preserves_old_behavior(self):
        """smooth_hold=False should give the same result as the original (no extra control point)."""
        levels = [0.2, 0.85, 0.55]
        sched_old_style = bezier_and_hold(
            hold_at=0.7,
            final_level=1.0,
            start=0.0,
            control_points=levels,
            smooth_hold=False,
        )
        values = np.array(sched_old_style(1000))
        hold_idx = int(1000 * 0.7)
        assert np.isclose(values[0], 0.0, atol=1e-6)
        assert np.isclose(values[-1], 1.0, atol=1e-6)
        assert np.allclose(values[hold_idx:], 1.0, atol=1e-6)

    def test_smooth_hold_default_is_true(self):
        """Default smooth_hold should be True."""
        sched = bezier_and_hold(
            hold_at=0.7,
            final_level=1.0,
            start=0.0,
            control_points=[0.3, 0.8],
        )
        values = np.array(sched(2000))
        hold_idx = int(2000 * 0.7)
        ramp_end = values[max(hold_idx - 10, 0) : hold_idx]
        dt = 1.0 / (len(values) - 1)
        slopes = np.abs(np.diff(ramp_end) / dt)
        # Should be smooth (low terminal slope)
        assert slopes.mean() < 0.5


class TestDictControlLevels:
    """Tests for dict-based sampled control levels."""

    def test_dict_absolute_range(self):
        """Dict with absolute value_range produces valid schedule."""
        sched = bezier_and_hold(
            hold_at=0.7,
            final_level=1.0,
            start=0.0,
            control_points={"num_points": 4, "value_range": [0.1, 0.9], "seed": 42},
        )
        values = np.array(sched(1000))
        hold_idx = int(1000 * 0.7)
        assert np.isclose(values[0], 0.0, atol=1e-6)
        assert np.isclose(values[-1], 1.0, atol=1e-6)
        assert np.all(values >= -1e-6)
        assert np.all(values <= 1.0 + 1e-6)
        assert np.allclose(values[hold_idx:], 1.0, atol=1e-6)

    def test_dict_relative_to_final(self):
        """Dict with relative_to_final=True scales value_range by final_level."""
        final = 2.0
        sched = bezier_and_hold(
            hold_at=0.7,
            final_level=final,
            start=0.0,
            control_points={
                "num_points": 3,
                "value_range": [0.1, 0.9],
                "relative_to_final": True,
                "seed": 123,
            },
        )
        values = np.array(sched(1000))
        hold_idx = int(1000 * 0.7)
        assert np.isclose(values[0], 0.0, atol=1e-6)
        assert np.isclose(values[-1], final, atol=1e-6)
        assert np.all(values >= -1e-6)
        assert np.all(values <= final + 1e-6)
        assert np.allclose(values[hold_idx:], final, atol=1e-6)

    def test_dict_seed_deterministic(self):
        """Same seed produces identical schedules."""
        kwargs = dict(
            hold_at=0.7,
            final_level=1.0,
            start=0.0,
            control_points={"num_points": 5, "value_range": [0.0, 1.0], "seed": 99},
        )
        v1 = np.array(bezier_and_hold(**kwargs)(1000))
        v2 = np.array(bezier_and_hold(**kwargs)(1000))
        assert np.allclose(v1, v2)

    def test_dict_no_seed_stochastic(self):
        """Without seed, repeated calls produce different schedules."""
        kwargs = dict(
            hold_at=0.7,
            final_level=1.0,
            start=0.0,
            control_points={"num_points": 5, "value_range": [0.0, 1.0]},
        )
        # Generate several and check they're not all identical
        results = [np.array(bezier_and_hold(**kwargs)(1000)) for _ in range(5)]
        all_same = all(np.allclose(results[0], r) for r in results[1:])
        assert not all_same, "Expected different schedules without seed"

    def test_dict_missing_num_points_raises(self):
        """Missing num_points in dict should raise ValueError."""
        with pytest.raises(ValueError, match="num_points"):
            bezier_and_hold(
                hold_at=0.7,
                control_points={"value_range": [0.0, 1.0]},
            )

    def test_dict_missing_value_range_raises(self):
        """Missing value_range in dict should raise ValueError."""
        with pytest.raises(ValueError, match="value_range"):
            bezier_and_hold(
                hold_at=0.7,
                control_points={"num_points": 3},
            )

    def test_dict_invalid_num_points_raises(self):
        """num_points < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="num_points"):
            bezier_and_hold(
                hold_at=0.7,
                control_points={"num_points": 0, "value_range": [0.0, 1.0]},
            )

    def test_dict_invalid_value_range_order_raises(self):
        """value_range[0] > value_range[1] should raise ValueError."""
        with pytest.raises(ValueError, match="value_range"):
            bezier_and_hold(
                hold_at=0.7,
                control_points={"num_points": 3, "value_range": [0.9, 0.1]},
            )

    def test_dict_invalid_type_raises(self):
        """Non-list/dict/None control_points should raise TypeError."""
        with pytest.raises(TypeError, match="control_points"):
            bezier_and_hold(hold_at=0.7, control_points="bad")
