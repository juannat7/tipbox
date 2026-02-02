"""Tests for ramp-and-hold bifurcation parameter schedule."""

import jax
import numpy as np
import pytest

from tipbox.generate import generate


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


class TestRampAndHold:
    """Tests for transition_fraction + hold_at_bifurcation behavior."""

    def test_hold_at_bifurcation(self, rng_key):
        """Parameter should hold constant at bif_value after transition_fraction."""
        full, _, info = generate(
            rng_key,
            tipping_type='b_fold',
            param_end_multiplier=None,
            transition_fraction=0.5,
            hold_at_bifurcation=True,
            verbose=False,
        )

        bif_value = info['bifurcation'].bif_value
        params = np.array(full.param_values)
        midpoint = len(params) // 2

        # After midpoint, all values should be at bif_value
        assert np.allclose(params[midpoint:], bif_value, rtol=0.01)
        # Before midpoint, should be ramping up
        assert params[0] < params[midpoint]

    def test_linear_extrapolation(self, rng_key):
        """Without hold, parameter should continue past bif_value."""
        full, _, info = generate(
            rng_key,
            tipping_type='b_fold',
            param_end_multiplier=None,
            transition_fraction=0.5,
            hold_at_bifurcation=False,
            verbose=False,
        )

        bif_value = info['bifurcation'].bif_value
        params = np.array(full.param_values)
        midpoint = len(params) // 2

        # At midpoint, param should equal bif_value
        assert np.isclose(params[midpoint], bif_value, rtol=0.05)
        # At end, param should be extrapolated past bif_value
        assert params[-1] > bif_value

    @pytest.mark.parametrize("transition_fraction", [0.3, 0.5, 0.7])
    def test_transition_at_exact_position(self, rng_key, transition_fraction):
        """Transition should occur at exactly transition_fraction of trajectory."""
        num_timesteps = 256
        full, _, info = generate(
            rng_key,
            tipping_type='b_fold',
            param_end_multiplier=None,
            transition_fraction=transition_fraction,
            hold_at_bifurcation=True,
            num_timesteps=num_timesteps,
            verbose=False,
        )

        bif_value = info['bifurcation'].bif_value
        params = np.array(full.param_values)
        expected_idx = int(num_timesteps * transition_fraction)

        # Verify param reaches bif_value at expected index
        assert np.isclose(params[expected_idx], bif_value, rtol=0.02)
        # Verify trajectory has exact requested length
        assert len(params) == num_timesteps

    def test_full_trajectory_length_preserved(self, rng_key):
        """Trajectory should have full requested length even if NaN occurs."""
        num_timesteps = 256
        full, _, _ = generate(
            rng_key,
            tipping_type='b_fold',
            param_end_multiplier=None,
            transition_fraction=0.3,  # more post-tip time
            hold_at_bifurcation=True,
            num_timesteps=num_timesteps,
            verbose=False,
        )

        assert full.trajectory.shape[0] == num_timesteps


class TestBackwardCompatibility:
    """Tests to ensure existing behavior is preserved."""

    def test_default_param_end_multiplier(self, rng_key):
        """Default behavior should use param_end_multiplier=1.1."""
        full, _, info = generate(
            rng_key,
            tipping_type='b_fold',
            verbose=False,
        )

        bif_value = info['bifurcation'].bif_value
        # Should ramp past bif_value (multiplier=1.1)
        assert np.max(full.param_values) > bif_value * 1.05


class TestXORConstraint:
    """Tests for parameter validation."""

    def test_error_when_both_specified(self, rng_key):
        """Should error if both transition_fraction and param_end_multiplier are set."""
        with pytest.raises(ValueError, match="Cannot use transition_fraction"):
            generate(
                rng_key,
                tipping_type='b_fold',
                param_end_multiplier=1.1,
                transition_fraction=0.5,
                verbose=False,
            )

    def test_error_when_neither_specified(self, rng_key):
        """Should error if both are None."""
        with pytest.raises(ValueError, match="Must specify"):
            generate(
                rng_key,
                tipping_type='b_fold',
                param_end_multiplier=None,
                transition_fraction=None,
                verbose=False,
            )
