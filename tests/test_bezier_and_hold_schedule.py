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
            control_levels=[0.2, 0.85, 0.55],
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
            control_levels=[0.85, 0.76, 0.42, 0.26],
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
            control_levels=[1.0, 0.0, 1.0, 0.0, 1.0],
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
