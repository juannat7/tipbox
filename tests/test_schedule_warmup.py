"""Integration tests for schedule warmup (non-zero start support).

These tests verify that the full generate() → simulate → trim pipeline works
correctly with schedules that start at non-zero param_normalized values.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(123)


class TestSyntheticWithWarmup:
    """Integration tests for synthetic bifurcation systems with warmup."""

    def test_linear_ramp_fold(self, rng_key):
        """Fold bifurcation with linear_ramp starting at 0.5."""
        from tipbox.generate import generate

        cfg = {
            "mode": "linear_ramp",
            "linear_ramp": {"start": 0.5, "end": 0.8},
            "schedule_warmup_time": 100.0,
        }
        full, part, info = generate(
            rng_key,
            tipping_type="b_fold",
            schedule=cfg,
            num_timesteps=200,
            verbose=False,
        )
        assert full.trajectory.shape[0] == 200
        # param_normalized should start near 0.5, not 0
        pn = info["param_normalized"]
        assert np.isclose(pn[0], 0.5, atol=0.02)

    def test_linear_ramp_hopf(self, rng_key):
        """Hopf bifurcation with linear_ramp starting at 0.3."""
        from tipbox.generate import generate

        cfg = {
            "mode": "linear_ramp",
            "linear_ramp": {"start": 0.3, "end": 0.7},
            "schedule_warmup_time": 100.0,
        }
        full, part, info = generate(
            rng_key,
            tipping_type="b_hopf",
            schedule=cfg,
            num_timesteps=200,
            verbose=False,
        )
        assert full.trajectory.shape[0] == 200
        pn = info["param_normalized"]
        assert np.isclose(pn[0], 0.3, atol=0.02)

    def test_linear_ramp_receding(self, rng_key):
        """Receding trajectory: start > end."""
        from tipbox.generate import generate

        cfg = {
            "mode": "linear_ramp",
            "linear_ramp": {"start": 0.6, "end": 0.2},
            "schedule_warmup_time": 100.0,
        }
        full, part, info = generate(
            rng_key,
            tipping_type="b_fold",
            schedule=cfg,
            num_timesteps=200,
            verbose=False,
        )
        assert full.trajectory.shape[0] == 200
        pn = info["param_normalized"]
        assert np.isclose(pn[0], 0.6, atol=0.02)
        assert np.isclose(pn[-1], 0.2, atol=0.02)

    def test_constant_nonzero(self, rng_key):
        """Constant schedule at non-zero param_normalized."""
        from tipbox.generate import generate

        cfg = {
            "mode": "constant",
            "constant": {"value": 0.4},
            "schedule_warmup_time": 100.0,
        }
        full, part, info = generate(
            rng_key,
            tipping_type="b_fold",
            schedule=cfg,
            num_timesteps=200,
            verbose=False,
        )
        assert full.trajectory.shape[0] == 200
        pn = info["param_normalized"]
        # Should be constant at 0.4
        assert np.allclose(pn, 0.4, atol=0.01)

    def test_zero_start_no_warmup(self, rng_key):
        """Standard zero-start schedule should work unchanged."""
        from tipbox.generate import generate

        cfg = {"mode": "ramp_and_hold", "ramp_and_hold": {"hold_at": 0.5}}
        full, part, info = generate(
            rng_key,
            tipping_type="b_fold",
            schedule=cfg,
            num_timesteps=200,
            verbose=False,
        )
        assert full.trajectory.shape[0] == 200
        pn = info["param_normalized"]
        assert np.isclose(pn[0], 0.0, atol=0.01)

    def test_warmup_time_configurable(self, rng_key):
        """Different schedule_warmup_time values should all produce valid output."""
        from tipbox.generate import generate

        for warmup_time in [50.0, 100.0, 200.0]:
            cfg = {
                "mode": "linear_ramp",
                "linear_ramp": {"start": 0.4, "end": 0.7},
                "schedule_warmup_time": warmup_time,
            }
            full, part, info = generate(
                rng_key,
                tipping_type="b_fold",
                schedule=cfg,
                num_timesteps=100,
                verbose=False,
            )
            assert full.trajectory.shape[0] == 100
            pn = info["param_normalized"]
            assert np.isclose(pn[0], 0.4, atol=0.02), f"Failed for warmup_time={warmup_time}"


class TestSemiRealWithWarmup:
    """Integration tests for semi-real bifurcation systems with warmup."""

    def test_harvesting_linear_ramp(self, rng_key):
        """May harvesting model with linear_ramp starting at 0.3."""
        from tipbox.generate import generate

        cfg = {
            "mode": "linear_ramp",
            "linear_ramp": {"start": 0.3, "end": 0.7},
            "schedule_warmup_time": 100.0,
        }
        full, part, info = generate(
            rng_key,
            tipping_type="b_harvesting",
            schedule=cfg,
            num_timesteps=100,
        )
        assert full.trajectory.shape[0] == 100
        pn = info["param_normalized"]
        assert np.isclose(pn[0], 0.3, atol=0.02)

    def test_harvesting_zero_start_unchanged(self, rng_key):
        """Standard zero-start should still work for semi-real."""
        from tipbox.generate import generate

        cfg = {"mode": "ramp_and_hold", "ramp_and_hold": {"hold_at": 0.5}}
        full, part, info = generate(
            rng_key,
            tipping_type="b_harvesting",
            schedule=cfg,
            num_timesteps=100,
        )
        assert full.trajectory.shape[0] == 100
