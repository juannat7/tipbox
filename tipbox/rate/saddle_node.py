from dataclasses import dataclass, replace
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

from ..types import BoxModel
from ..utils import convert_scalar_trajectory, extract_segment, register_dataclass_pytree


@register_dataclass_pytree
@dataclass(frozen=True)
class SaddleNodeConfig:
    """Configuration for the stochastic saddle-node system."""

    epsilon: float = 1.25
    lambda_max: float = 3.0
    dt: float = 0.01
    t_start: float = -16.0
    t_end: float = 8.0
    initial_state_min: float = -10.0
    initial_state_max: float = -1.0
    tip_threshold: float = 0.0

    @classmethod
    def default(cls) -> "SaddleNodeConfig":
        """Return default configuration."""
        return cls()


class SaddleNodeModel(BoxModel):
    """BoxModel abstraction for the stochastic saddle-node system."""

    def __init__(self, config: SaddleNodeConfig):
        self._config = config

    def get_parameters(self):
        """Get the parameters of the model."""
        return {
            "epsilon": self._config.epsilon,
            "lambda_max": self._config.lambda_max,
            "dt": self._config.dt,
            "t_start": self._config.t_start,
            "t_end": self._config.t_end,
            "initial_state_min": self._config.initial_state_min,
            "initial_state_max": self._config.initial_state_max,
            "tip_threshold": self._config.tip_threshold,
        }

    def _lambda_schedule(
        self,
        t: jnp.ndarray,
        config: SaddleNodeConfig,
    ):
        """Compute the lambda schedule."""
        return 0.5 * config.lambda_max * (jnp.tanh(0.5 * config.lambda_max * config.epsilon * t) + 1.0)

    def _sample_initial_state(
        self,
        key: jax.random.PRNGKey,
        config: SaddleNodeConfig,
    ):
        """Sample an initial state from the uniform distribution."""
        return float(
            jax.random.uniform(
                key,
                (),
                minval=config.initial_state_min,
                maxval=config.initial_state_max,
            )
        )

    def _simulate_path(
        self,
        key: jax.random.PRNGKey,
        lambda_t: jnp.ndarray,
        dt: float,
        noise_scale: float,
        initial_state: float,
        noise: Optional[jnp.ndarray] = None,
    ):
        """Stochastic integration."""

        if noise is None:
            noise = jax.random.normal(key, shape=(lambda_t.shape[0] - 1,))
        else:
            noise = jnp.asarray(noise)

        def step(x_prev, inputs):
            lam_t, noise_t = inputs
            det_part = ((x_prev + lam_t) ** 2 - 1.0) * dt
            x_next = x_prev + det_part + noise_scale * noise_t
            return x_next, x_next

        _, x_tail = lax.scan(step, initial_state, (lambda_t[:-1], noise))
        x0 = jnp.full((1,), initial_state, dtype=lambda_t.dtype)
        return jnp.concatenate([x0, x_tail], axis=0)

    def simulate(
        self,
        key: jax.random.PRNGKey,
        total_len: int,
        sigma_tilde: float = 0.01,
        schedule=None,
        schedule_start: float = 0.0,
        initial_state: Optional[jnp.ndarray] = None,
        noise: Optional[jnp.ndarray] = None,
        null_simulation: bool = False,
    ):
        """Simulate the saddle-node system.

        For null_simulation=True (legacy mode), keep forcing constant at 0.
        """
        key_noise, key_ic = jax.random.split(key)

        if schedule is not None:
            from ..bifurcation.schedule import resolve_schedule_for_simulation

            lambda_t, _, _sched_meta = resolve_schedule_for_simulation(
                schedule,
                total_len,
                0.0,
                self._config.lambda_max,
                expected_start=schedule_start,
            )
            time = jnp.arange(0, total_len * self._config.dt, self._config.dt)
        else:
            # Legacy: internal tanh forcing
            self._config = replace(self._config, t_start=-(total_len * self._config.dt))
            time = jnp.arange(self._config.t_start, self._config.t_end, self._config.dt)
            if null_simulation:
                lambda_t = jnp.zeros_like(time)
            else:
                lambda_t = self._lambda_schedule(time, self._config)

        # Randomly sample initial state if not provided
        if initial_state is None:
            initial_state = self._sample_initial_state(key_ic, self._config)
        else:
            initial_state = jnp.asarray(initial_state)

        # Configure noise scale
        noise_scale = jnp.sqrt(2.0 * sigma_tilde * self._config.dt)

        # Simulate trajectory
        trajectory = self._simulate_path(
            key_noise,
            lambda_t,
            self._config.dt,
            noise_scale,
            initial_state,
            noise=noise,
        )

        # Post-process trajectory
        trajectory = convert_scalar_trajectory(trajectory)

        return time, trajectory, lambda_t, noise_scale

    def is_tipped(
        self,
        trajectory: jnp.ndarray,
        **kwargs,
    ):
        """Check if the trajectory is tipped."""
        scalar_view = trajectory[:, 0] if trajectory.ndim > 1 else trajectory
        return bool(jnp.any(scalar_view > self._config.tip_threshold))

    def locate_transition(
        self,
        trajectory: jnp.ndarray,
        null_simulation: bool,
        **kwargs,
    ):
        """Locate the transition index."""
        scalar_view = trajectory[:, 0] if trajectory.ndim > 1 else trajectory
        if null_simulation:
            return int(scalar_view.shape[0] // 2)

        crossed = jnp.where(scalar_view > self._config.tip_threshold, 1, 0)
        has_crossed = bool(jnp.any(crossed))
        if not has_crossed:
            return None
        return int(jnp.argmax(crossed))

    def extract_segment(
        self,
        time: jnp.ndarray,
        trajectory: jnp.ndarray,
        param_values: jnp.ndarray,
        transition_idx: int,
        steps_before_transition: int,
        series_len: int,
    ):
        """Extract a segment of the trajectory around the transition."""
        time_seg, traj_seg, param_seg, trans_idx_out, success = extract_segment(
            time,
            trajectory,
            param_values,
            transition_idx,
            steps_before_transition,
            series_len,
        )

        return time_seg, traj_seg, param_seg, trans_idx_out, success

    def normalize_param_values(self, param_values):
        """Normalize forcing values by max forcing amplitude."""
        return param_values / self._config.lambda_max

    # -- Schedule translation -------------------------------------------------

    # Rate bounds: epsilon controls native tanh steepness.
    # Null simulation uses constant-at-zero, so r_null = 0.
    _r_crit: float = 1.25  # SaddleNodeConfig.epsilon default
    _r_null: float = 0.0   # legacy null: constant zero forcing

    def translate_schedule(self, param_normalized, rdtc, mode="time_to_critical_rate", **kw):
        """Translate abstract schedule to native saddle-node tanh forcing."""
        if mode == "time_to_critical_rate":
            return self._translate_time_to_critical_rate(param_normalized, rdtc, **kw)
        if mode == "distance_from_critical_rate":
            raise NotImplementedError("distance_from_critical_rate not yet implemented")
        raise ValueError(f"Unknown translation mode: {mode}")

    # Native tanh inflection at t=0.
    _t_inflection: float = 0.0

    def _translate_time_to_critical_rate(self, param_normalized, rdtc, **_kw):
        """Interpolate epsilon and shift native forcing so critical rate aligns with rdtc=0."""
        import random

        n = len(param_normalized)
        peak = float(jnp.max(param_normalized))
        epsilon = self._r_null + peak * (self._r_crit - self._r_null)

        min_rdtc = float(jnp.min(rdtc))
        native_range = self._config.t_end - self._config.t_start
        if min_rdtc < 0.1:
            t_cross_idx = int(jnp.argmin(jnp.abs(rdtc)))
            t_cross_frac = t_cross_idx / n if n > 0 else 0.5
        else:
            nsf = _kw.get("null_shift_frac", 0.75)
            if isinstance(nsf, (list, tuple)) and len(nsf) == 2:
                t_cross_frac = random.uniform(nsf[0], nsf[1])
            else:
                t_cross_frac = float(nsf)
        t_target = self._config.t_start + t_cross_frac * native_range
        shift = t_target - self._t_inflection
        t = jnp.linspace(self._config.t_start - shift, self._config.t_end - shift, n)

        config = replace(self._config, epsilon=epsilon)
        forcing = self._lambda_schedule(t, config)
        return forcing / self._config.lambda_max
