"""Tests for bifurcation parameter schedules."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from tipbox.bifurcation import schedule
from tipbox.bifurcation.schedule import (
    apply_to_params,
    constant,
    create_schedule,
    fixed_slope,
    from_array,
    legacy_multiplier_schedule,
    linear,
    piecewise_linear,
    ramp_and_hold,
    resolve_schedule,
)


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


class TestLinearSchedule:
    """Tests for linear() factory function."""

    def test_default_linear(self):
        """Default linear should ramp from 0 to 1."""
        sched = linear()
        values = sched(100)
        assert np.isclose(values[0], 0.0)
        assert np.isclose(values[-1], 1.0)
        assert len(values) == 100

    def test_linear_critical_at_half(self):
        """Linear with critical_at=0.5 should reach 1.0 at midpoint."""
        sched = linear(critical_at=0.5)
        values = sched(100)
        assert np.isclose(values[0], 0.0)
        assert np.isclose(values[50], 1.0, rtol=0.05)
        assert np.isclose(values[-1], 2.0)

    def test_linear_critical_at_various(self):
        """Linear schedule should reach 1.0 at exactly critical_at fraction."""
        for critical_at in [0.3, 0.5, 0.7, 0.9]:
            sched = linear(critical_at=critical_at)
            values = sched(1000)
            idx = int(1000 * critical_at)
            assert np.isclose(values[idx], 1.0, rtol=0.01)

    def test_linear_invalid_critical_at(self):
        """Should raise error for invalid critical_at."""
        with pytest.raises(ValueError):
            linear(critical_at=0)
        with pytest.raises(ValueError):
            linear(critical_at=-0.5)


class TestRampAndHoldSchedule:
    """Tests for ramp_and_hold() factory function."""

    def test_ramp_and_hold_basic(self):
        """Ramp should reach hold_level at hold_at, then stay constant."""
        sched = ramp_and_hold(hold_at=0.5, hold_level=1.0)
        values = sched(100)

        # Check ramp portion
        assert np.isclose(values[0], 0.0)
        assert np.isclose(values[50], 1.0, rtol=0.05)

        # Check hold portion
        assert np.allclose(values[50:], 1.0, rtol=0.02)

    def test_ramp_and_hold_partial_level(self):
        """Ramp should stop at hold_level < 1.0."""
        sched = ramp_and_hold(hold_at=0.5, hold_level=0.7)
        values = sched(100)

        assert np.isclose(values[50], 0.7, rtol=0.05)
        assert np.allclose(values[50:], 0.7, rtol=0.02)

    def test_ramp_and_hold_various_hold_at(self):
        """Hold should start at exactly hold_at fraction."""
        for hold_at in [0.3, 0.5, 0.7]:
            sched = ramp_and_hold(hold_at=hold_at, hold_level=1.0)
            values = sched(1000)
            hold_idx = int(1000 * hold_at)

            # After hold_idx, all values should be at hold_level
            assert np.allclose(values[hold_idx:], 1.0, rtol=0.02)

    def test_ramp_and_hold_invalid_hold_at(self):
        """Should raise error for invalid hold_at."""
        with pytest.raises(ValueError):
            ramp_and_hold(hold_at=0)
        with pytest.raises(ValueError):
            ramp_and_hold(hold_at=1.5)


class TestFixedSlopeSchedule:
    """Tests for fixed_slope() factory function."""

    def test_fixed_slope_basic(self):
        """Fixed slope should compute start value from slope."""
        # Slope of 2.0, hold at 0.5 -> start at 0.0, reach 1.0 at 0.5
        sched = fixed_slope(slope=2.0, hold_at=0.5, hold_level=1.0)
        values = sched(100)

        # start = hold_level - slope * hold_at = 1.0 - 2.0 * 0.5 = 0.0
        assert np.isclose(values[0], 0.0, atol=0.02)
        assert np.isclose(values[50], 1.0, rtol=0.05)
        assert np.allclose(values[50:], 1.0, rtol=0.02)

    def test_fixed_slope_nonzero_start(self):
        """Fixed slope can start above zero."""
        # Slope of 1.0, hold at 0.7 -> start at 0.3, reach 1.0 at 0.7
        sched = fixed_slope(slope=1.0, hold_at=0.7, hold_level=1.0)
        values = sched(1000)

        # start = hold_level - slope * hold_at = 1.0 - 1.0 * 0.7 = 0.3
        assert np.isclose(values[0], 0.3, atol=0.02)
        hold_idx = int(1000 * 0.7)
        assert np.isclose(values[hold_idx], 1.0, rtol=0.05)
        assert np.allclose(values[hold_idx:], 1.0, rtol=0.02)

    def test_fixed_slope_clips_at_zero_by_default(self):
        """Fixed slope clips values at 0 by default."""
        # Slope of 3.0, hold at 0.5 -> computed start at -0.5, but clipped to 0
        sched = fixed_slope(slope=3.0, hold_at=0.5, hold_level=1.0)
        values = sched(100)

        # Should be clipped at 0
        assert np.isclose(values[0], 0.0, atol=0.02)
        assert values.min() >= 0.0
        assert np.isclose(values[50], 1.0, rtol=0.05)

    def test_fixed_slope_negative_start_with_clip_none(self):
        """Fixed slope can go negative when clip_min=None."""
        # Slope of 3.0, hold at 0.5 -> start at -0.5
        sched = fixed_slope(slope=3.0, hold_at=0.5, hold_level=1.0, clip_min=None)
        values = sched(100)

        # start = hold_level - slope * hold_at = 1.0 - 3.0 * 0.5 = -0.5
        assert np.isclose(values[0], -0.5, atol=0.02)
        assert np.isclose(values[50], 1.0, rtol=0.05)

    def test_fixed_slope_partial_hold_level(self):
        """Fixed slope should work with hold_level < 1.0."""
        # Slope of 1.0, hold at 0.5, hold_level 0.7 -> start at 0.2
        sched = fixed_slope(slope=1.0, hold_at=0.5, hold_level=0.7)
        values = sched(100)

        # start = 0.7 - 1.0 * 0.5 = 0.2
        assert np.isclose(values[0], 0.2, atol=0.02)
        assert np.isclose(values[50], 0.7, rtol=0.05)
        assert np.allclose(values[50:], 0.7, rtol=0.02)

    def test_fixed_slope_vs_ramp_and_hold(self):
        """When slope = hold_level/hold_at, fixed_slope equals ramp_and_hold."""
        hold_at = 0.5
        hold_level = 1.0
        slope = hold_level / hold_at  # = 2.0

        sched_fixed = fixed_slope(slope=slope, hold_at=hold_at, hold_level=hold_level)
        sched_ramp = ramp_and_hold(hold_at=hold_at, hold_level=hold_level)

        values_fixed = sched_fixed(100)
        values_ramp = sched_ramp(100)

        assert np.allclose(values_fixed, values_ramp, rtol=0.01)

    def test_fixed_slope_invalid_hold_at(self):
        """Should raise error for invalid hold_at."""
        with pytest.raises(ValueError):
            fixed_slope(slope=1.0, hold_at=0)
        with pytest.raises(ValueError):
            fixed_slope(slope=1.0, hold_at=1.5)

    def test_fixed_slope_invalid_slope(self):
        """Should raise error for non-positive slope."""
        with pytest.raises(ValueError):
            fixed_slope(slope=0, hold_at=0.5)
        with pytest.raises(ValueError):
            fixed_slope(slope=-1.0, hold_at=0.5)


class TestConstantSchedule:
    """Tests for constant() factory function."""

    def test_constant_default(self):
        """Default constant should be all zeros."""
        sched = constant()
        values = sched(100)
        assert np.allclose(values, 0.0)

    def test_constant_value(self):
        """Constant should be all the specified value."""
        sched = constant(value=0.5)
        values = sched(100)
        assert np.allclose(values, 0.5)

    def test_constant_length(self):
        """Constant should have correct length."""
        sched = constant(value=0.3)
        for n in [10, 100, 500]:
            values = sched(n)
            assert len(values) == n


class TestPiecewiseLinearSchedule:
    """Tests for piecewise_linear() factory function."""

    def test_piecewise_linear_basic(self):
        """Basic piecewise linear with two segments."""
        segments = [
            {"start": 0.0, "end": 0.5, "fraction": 0.5},
            {"start": 0.5, "end": 1.0, "fraction": 0.5},
        ]
        sched = piecewise_linear(segments)
        values = sched(100)

        assert np.isclose(values[0], 0.0, atol=0.02)
        assert np.isclose(values[50], 0.5, rtol=0.1)
        assert np.isclose(values[-1], 1.0, rtol=0.02)

    def test_piecewise_linear_hold(self):
        """Piecewise linear with hold segment."""
        segments = [
            {"start": 0.0, "end": 1.0, "fraction": 0.5},
            {"start": 1.0, "end": 1.0, "fraction": 0.5},  # hold
        ]
        sched = piecewise_linear(segments)
        values = sched(100)

        # First half ramps
        assert values[25] < values[49]
        # Second half holds
        assert np.allclose(values[50:], 1.0, rtol=0.05)

    def test_piecewise_linear_three_segments(self):
        """Piecewise linear with three segments."""
        segments = [
            {"start": 0.0, "end": 0.3, "fraction": 0.5},
            {"start": 0.3, "end": 1.0, "fraction": 0.2},
            {"start": 1.0, "end": 1.0, "fraction": 0.3},
        ]
        sched = piecewise_linear(segments)
        values = sched(1000)

        # Should reach ~0.3 at 50%
        assert np.isclose(values[500], 0.3, rtol=0.1)
        # Should reach ~1.0 at 70%
        assert np.isclose(values[700], 1.0, rtol=0.1)
        # Should hold at 1.0 after 70%
        assert np.allclose(values[700:], 1.0, rtol=0.05)

    def test_piecewise_linear_empty_segments(self):
        """Should raise error for empty segments."""
        with pytest.raises(ValueError, match="cannot be empty"):
            piecewise_linear([])

    def test_piecewise_linear_fractions_must_sum_to_one(self):
        """Should raise error if fractions don't sum to 1."""
        segments = [
            {"start": 0.0, "end": 0.5, "fraction": 0.3},
            {"start": 0.5, "end": 1.0, "fraction": 0.3},
        ]
        with pytest.raises(ValueError, match=r"sum to 1\.0"):
            piecewise_linear(segments)


class TestFromArraySchedule:
    """Tests for from_array() factory function."""

    def test_from_array_same_length(self):
        """Array of same length should pass through unchanged."""
        arr = jnp.linspace(0, 1, 100)
        sched = from_array(arr)
        values = sched(100)
        assert np.allclose(values, arr)

    def test_from_array_interpolation(self):
        """Array of different length should be interpolated."""
        arr = jnp.array([0.0, 0.5, 1.0])
        sched = from_array(arr)
        values = sched(100)

        assert np.isclose(values[0], 0.0)
        assert np.isclose(values[50], 0.5, rtol=0.05)
        assert np.isclose(values[-1], 1.0)

    def test_from_array_custom_shape(self):
        """Custom array should interpolate correctly."""
        # Step function - interpolation creates ramp at transition
        arr = jnp.concatenate([jnp.zeros(50), jnp.ones(50)])
        sched = from_array(arr)
        values = sched(200)

        # First 80% of first half should be near zero
        assert np.allclose(values[:80], 0.0, atol=0.1)
        # Last 80% of second half should be near one
        assert np.allclose(values[120:], 1.0, atol=0.1)


class TestCreateSchedule:
    """Tests for create_schedule() dict-based creation."""

    def test_create_linear(self):
        """Create linear schedule from dict."""
        config = {"mode": "linear", "linear": {"critical_at": 0.5}}
        sched = create_schedule(config)
        values = sched(100)

        assert np.isclose(values[50], 1.0, rtol=0.05)

    def test_create_ramp_and_hold(self):
        """Create ramp_and_hold schedule from dict."""
        config = {
            "mode": "ramp_and_hold",
            "ramp_and_hold": {"hold_at": 0.5, "hold_level": 1.0},
        }
        sched = create_schedule(config)
        values = sched(100)

        assert np.allclose(values[50:], 1.0, rtol=0.02)

    def test_create_fixed_slope(self):
        """Create fixed_slope schedule from dict."""
        config = {
            "mode": "fixed_slope",
            "fixed_slope": {"slope": 1.0, "hold_at": 0.7, "hold_level": 1.0},
        }
        sched = create_schedule(config)
        values = sched(1000)

        # start = 1.0 - 1.0 * 0.7 = 0.3
        assert np.isclose(values[0], 0.3, atol=0.02)
        assert np.allclose(values[700:], 1.0, rtol=0.02)

    def test_create_constant(self):
        """Create constant schedule from dict."""
        config = {"mode": "constant", "constant": {"value": 0.5}}
        sched = create_schedule(config)
        values = sched(100)

        assert np.allclose(values, 0.5)

    def test_create_default_mode(self):
        """Default mode should be linear."""
        config = {}
        sched = create_schedule(config)
        values = sched(100)

        # Default linear goes 0 to 1
        assert np.isclose(values[0], 0.0)
        assert np.isclose(values[-1], 1.0)

    def test_create_unknown_mode(self):
        """Should raise error for unknown mode."""
        config = {"mode": "unknown"}
        with pytest.raises(ValueError, match="Unknown schedule mode"):
            create_schedule(config)

    def test_create_with_extra_configs(self):
        """Should work when config has multiple mode configs (only active used)."""
        config = {
            "mode": "ramp_and_hold",
            "linear": {"critical_at": 0.7},
            "ramp_and_hold": {"hold_at": 0.5, "hold_level": 1.0},
        }
        sched = create_schedule(config)
        values = sched(100)

        # Should use ramp_and_hold, not linear
        assert np.allclose(values[50:], 1.0, rtol=0.02)


class TestResolveSchedule:
    """Tests for resolve_schedule() unified handling."""

    def test_resolve_array(self):
        """Should handle array input."""
        arr = jnp.linspace(0, 1, 100)
        result = resolve_schedule(arr, 100)
        assert np.allclose(result, arr)

    def test_resolve_callable(self):
        """Should handle callable input."""
        sched = linear(critical_at=0.5)
        result = resolve_schedule(sched, 100)
        assert np.isclose(result[50], 1.0, rtol=0.05)

    def test_resolve_dict(self):
        """Should handle dict input."""
        config = {"mode": "constant", "constant": {"value": 0.5}}
        result = resolve_schedule(config, 100)
        assert np.allclose(result, 0.5)

    def test_resolve_interpolation(self):
        """Should interpolate arrays to correct length."""
        arr = jnp.array([0.0, 1.0])
        result = resolve_schedule(arr, 100)
        assert len(result) == 100
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[-1], 1.0)


class TestApplyToParams:
    """Tests for apply_to_params() utility."""

    def test_apply_linear(self):
        """Should map [0,1] to [param_start, bif_value]."""
        normalized = jnp.array([0.0, 0.5, 1.0])
        result = apply_to_params(normalized, param_start=1.0, bif_value=3.0)

        assert np.isclose(result[0], 1.0)
        assert np.isclose(result[1], 2.0)
        assert np.isclose(result[2], 3.0)

    def test_apply_beyond_one(self):
        """Should extrapolate beyond 1.0."""
        normalized = jnp.array([0.0, 1.0, 2.0])
        result = apply_to_params(normalized, param_start=0.0, bif_value=1.0)

        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[1], 1.0)
        assert np.isclose(result[2], 2.0)


class TestLegacyMultiplierSchedule:
    """Tests for legacy_multiplier_schedule() backward compatibility."""

    def test_legacy_multiplier_1_1(self):
        """Should ramp from 0 to 1.1."""
        sched = legacy_multiplier_schedule(1.1)
        values = sched(100)

        assert np.isclose(values[0], 0.0)
        assert np.isclose(values[-1], 1.1)

    def test_legacy_multiplier_1_0(self):
        """Should ramp from 0 to 1.0."""
        sched = legacy_multiplier_schedule(1.0)
        values = sched(100)

        assert np.isclose(values[0], 0.0)
        assert np.isclose(values[-1], 1.0)


@pytest.mark.skipif(not os.environ.get("AUTO_DIR"), reason="AUTO_DIR not set - requires AUTO-07P")
class TestIntegrationWithGenerate:
    """Integration tests with the generate API."""

    def test_schedule_with_generate(self, rng_key):
        """Schedule should work with generate() API."""
        from tipbox.generate import generate

        full, _, info = generate(
            rng_key,
            tipping_type="b_fold",
            schedule=schedule.ramp_and_hold(hold_at=0.5, hold_level=1.0),
            verbose=False,
        )

        bif_value = info["bifurcation"].bif_value
        params = np.array(full.param_values)
        midpoint = len(params) // 2

        # After midpoint, all values should be at bif_value
        assert np.allclose(params[midpoint:], bif_value, rtol=0.02)

    def test_dict_schedule_with_generate(self, rng_key):
        """Dict schedule config should work with generate() API."""
        from tipbox.generate import generate

        full, _, info = generate(
            rng_key,
            tipping_type="b_fold",
            schedule={
                "mode": "ramp_and_hold",
                "ramp_and_hold": {"hold_at": 0.5, "hold_level": 1.0},
            },
            verbose=False,
        )

        bif_value = info["bifurcation"].bif_value
        params = np.array(full.param_values)
        midpoint = len(params) // 2

        # After midpoint, all values should be at bif_value
        assert np.allclose(params[midpoint:], bif_value, rtol=0.02)

    def test_linear_schedule_with_generate(self, rng_key):
        """Linear schedule should work with generate() API."""
        from tipbox.generate import generate

        full, _, info = generate(
            rng_key,
            tipping_type="b_fold",
            schedule=schedule.linear(critical_at=0.5),
            verbose=False,
        )

        bif_value = info["bifurcation"].bif_value
        params = np.array(full.param_values)
        midpoint = len(params) // 2

        # At midpoint, param should equal bif_value
        assert np.isclose(params[midpoint], bif_value, rtol=0.05)
        # At end, param should be extrapolated past bif_value
        assert params[-1] > bif_value

    def test_backward_compatibility_default(self, rng_key):
        """Default behavior (no schedule) should use param_end_multiplier."""
        from tipbox.generate import generate

        full, _, info = generate(
            rng_key,
            tipping_type="b_fold",
            verbose=False,
        )

        bif_value = info["bifurcation"].bif_value
        # Should ramp past bif_value (multiplier=1.1)
        assert np.max(full.param_values) > bif_value * 1.05

    def test_piecewise_schedule_with_generate(self, rng_key):
        """Piecewise linear schedule should work with generate() API."""
        from tipbox.generate import generate

        full, _, _ = generate(
            rng_key,
            tipping_type="b_fold",
            schedule=schedule.piecewise_linear(
                segments=[
                    {"start": 0.0, "end": 0.5, "fraction": 0.5},
                    {"start": 0.5, "end": 1.0, "fraction": 0.2},
                    {"start": 1.0, "end": 1.0, "fraction": 0.3},
                ]
            ),
            verbose=False,
        )

        # Should have correct trajectory length
        assert full.trajectory.shape[0] == 500  # default num_timesteps

    def test_array_schedule_with_generate(self, rng_key):
        """Direct array schedule should work with generate() API."""
        from tipbox.generate import generate

        # Custom schedule: slow start, fast finish
        custom = jnp.concatenate([jnp.linspace(0, 0.3, 250), jnp.linspace(0.3, 1.0, 250)])
        full, _, _ = generate(
            rng_key,
            tipping_type="b_fold",
            schedule=custom,
            verbose=False,
        )

        # Should have correct trajectory length
        assert full.trajectory.shape[0] == 500
