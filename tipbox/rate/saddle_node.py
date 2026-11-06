from dataclasses import dataclass, replace
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

from ..types import BoxModel
from ..utils import (
    parameter,
    convert_scalar_trajectory,
    register_dataclass_pytree,
)


@register_dataclass_pytree
@dataclass(frozen=True)
class SaddleNodeConfig:
    """Configuration for the stochastic saddle-node system."""

    epsilon: float = parameter(
        1.25, "Rate of the tanh equilibrium-shift forcing.", "model rate"
    )
    lambda_max: float = parameter(3.0, "Total equilibrium displacement.", "state")
    null_rate_factor: float = parameter(
        0.5,
        "Multiplier applied to the forcing rate for null simulations.",
        "dimensionless",
    )
    dt: float = parameter(
        0.01,
        "Default integration step.",
        "model time",
    )
    t_start: float = parameter(
        -16.0,
        "Simulation start time.",
        "model time",
    )
    t_end: float = parameter(
        8.0,
        "End time when make_trajectory is called without num_samples; explicit num_samples sets the end time.",
        "model time",
    )
    initial_state_min: float = parameter(
        -10.0, "Lower bound for the default uniform initial-state sampler.", "state"
    )
    initial_state_max: float = parameter(
        -1.0, "Upper bound for the default uniform initial-state sampler.", "state"
    )
    tip_threshold: float = parameter(
        0.0, "State threshold used by the tipping classifier.", "state"
    )

    @classmethod
    def default(cls) -> "SaddleNodeConfig":
        """Return default configuration."""
        return cls()


class SaddleNodeModel(BoxModel):
    """BoxModel abstraction for the stochastic saddle-node system."""

    config_class = SaddleNodeConfig

    state_names = ("state",)
    observable_names = state_names
    noise_dimension = 1
    forcing_name = "equilibrium_shift"

    def _initial_state(self, key, null_simulation):
        return self._sample_initial_state(key, self._config)

    def _forcing(self, time, null_simulation, param_end_multiplier):
        config = (
            replace(
                self._config,
                epsilon=self._config.epsilon * self._config.null_rate_factor,
            )
            if null_simulation
            else self._config
        )
        return self._lambda_schedule(time, config)

    def _integrate(
        self, time, forcing, dt, noise, state, innovations, midpoint_forcing
    ):
        scale = jnp.sqrt(2 * noise * dt)
        path = self._simulate_path(
            None, forcing, dt, scale, state[0], noise=innovations[:, 0]
        )
        return path[:, None], scale

    def _lambda_schedule(
        self,
        t: jnp.ndarray,
        config: SaddleNodeConfig,
    ):
        """Compute the lambda schedule."""
        return (
            0.5
            * config.lambda_max
            * (jnp.tanh(0.5 * config.lambda_max * config.epsilon * t) + 1.0)
        )

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
        initial_state: Optional[jnp.ndarray] = None,
        noise: Optional[jnp.ndarray] = None,
        null_simulation: bool = False,
    ):
        """Simulate the saddle-node system.

        For null_simulation=True, same ramp formula but epsilon at 50% (slower rate).
        """
        key_noise, key_ic = jax.random.split(key)

        config = replace(self._config, t_start=-(total_len * self._config.dt))
        time = jnp.arange(config.t_start, config.t_end, config.dt)
        if null_simulation:
            config_null = replace(
                self._config,
                epsilon=self._config.null_rate_factor * self._config.epsilon,
            )
            lambda_t = self._lambda_schedule(time, config_null)
        else:
            lambda_t = self._lambda_schedule(time, self._config)

        if initial_state is None:
            initial_state = self._sample_initial_state(key_ic, self._config)
        else:
            initial_state = jnp.asarray(initial_state)

        noise_scale = jnp.sqrt(2.0 * sigma_tilde * self._config.dt)

        trajectory = self._simulate_path(
            key_noise,
            lambda_t,
            self._config.dt,
            noise_scale,
            initial_state,
            noise=noise,
        )

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
