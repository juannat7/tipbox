from dataclasses import dataclass, replace
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

from ..types import BoxModel
from ..utils import parameter, register_dataclass_pytree


@register_dataclass_pytree
@dataclass(frozen=True)
class BautinConfig:
    """Coefficients and forcing controls for the stochastic Bautin normal form."""

    r: float = parameter(
        0.1, "Rate of the tanh equilibrium-shift forcing.", "model rate"
    )
    lambda_max: float = parameter(8.0, "Total equilibrium displacement.", "state")
    a: float = parameter(0.1, "Linear real drift coefficient.", "1 / model time")
    omega: float = parameter(3.0, "Angular rotation frequency.", "radian / model time")
    b: float = parameter(1.0, "Cubic radial drift coefficient.", "model units")
    D2: float = parameter(
        20,
        "Diffusion prefactor; per-step scale is D2*noise*sqrt(dt).",
        "state / sqrt(model time)",
    )
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
        -15.0,
        "Simulation start time.",
        "model time",
    )
    t_end: float = parameter(
        15.0,
        "End time when make_trajectory is called without n; explicit n sets the end time.",
        "model time",
    )
    initial_state_min: float = parameter(
        -1.0, "Lower bound for the default uniform initial-state sampler.", "state"
    )
    initial_state_max: float = parameter(
        1.0, "Upper bound for the default uniform initial-state sampler.", "state"
    )
    tip_threshold: float = parameter(
        10.0, "Radius about the origin used by the legacy tipping classifier.", "state"
    )

    @classmethod
    def default(cls) -> "BautinConfig":
        """Return default configuration."""
        return cls()


class BautinModel(BoxModel):
    """Shifted Bautin normal form in Cartesian coordinates (Ritchie et al., 2023)."""

    config_class = BautinConfig

    state_names = ("x", "y")
    observable_names = state_names
    noise_dimension = 2
    forcing_name = "equilibrium_shift"

    def _initial_state(self, key, null_simulation):
        return self._sample_initial_state(key, self._config)

    def _forcing(self, time, null_simulation, param_end_multiplier):
        config = (
            replace(self._config, r=self._config.r * self._config.null_rate_factor)
            if null_simulation
            else self._config
        )
        return self._lambda_schedule(time, config)

    def _integrate(
        self, time, forcing, dt, noise, state, innovations, midpoint_forcing
    ):
        scale = self._config.D2 * noise * jnp.sqrt(dt)
        path = self._simulate_path(
            None, forcing, dt, scale, state[0], state[1], noise=innovations
        )
        return path, scale

    def _lambda_schedule(
        self,
        t: jnp.ndarray,
        config: BautinConfig,
    ):
        """Compute the Λ(t) schedule (time-varying forcing)."""
        # Λ(t) = 0.5Λ_max[tanh(0.5Λ_max r t) + 1]
        return (
            0.5
            * config.lambda_max
            * (jnp.tanh(0.5 * config.lambda_max * config.r * t) + 1.0)
        )

    def _sample_initial_state(
        self,
        key: jax.random.PRNGKey,
        config: BautinConfig,
    ):
        """Sample an initial state in Cartesian coordinates (x, y)."""
        key_x, key_y = jax.random.split(key)
        x = float(
            jax.random.uniform(
                key_x,
                (),
                minval=config.initial_state_min,
                maxval=config.initial_state_max,
            )
        )
        y = float(
            jax.random.uniform(
                key_y,
                (),
                minval=config.initial_state_min,
                maxval=config.initial_state_max,
            )
        )
        return x, y

    def _simulate_path(
        self,
        key: jax.random.PRNGKey,
        lambda_t: jnp.ndarray,
        dt: float,
        noise_scale: float,
        initial_x: float,
        initial_y: float,
        noise: Optional[jnp.ndarray] = None,
    ):
        """Integrate the physical state on the supplied forcing grid."""
        if noise is None:
            key, subkey = jax.random.split(key)
            noise_x = jax.random.normal(key, shape=(lambda_t.shape[0] - 1,))
            noise_y = jax.random.normal(subkey, shape=(lambda_t.shape[0] - 1,))
        else:
            noise = jnp.asarray(noise)
            noise_x = noise[:, 0]
            noise_y = noise[:, 1]

        def step(state, inputs):
            lambda_val, noise_x_t, noise_y_t = inputs
            x, y = state

            # Compute w = Z_t - Λ(t) = (x, y) - (Λ(t), 0) = (x - Λ(t), y)
            w_x = x - lambda_val
            w_y = y

            # Compute |w|² = w_x² + w_y²
            w_squared = w_x**2 + w_y**2
            w_fourth = w_squared**2

            # dx/dt = ax - ωy - b(x²+y²)x + (x²+y²)²x
            # where (x, y) here refers to w = (w_x, w_y)
            dx_dt = (
                self._config.a * w_x
                - self._config.omega * w_y
                - self._config.b * w_squared * w_x
                + w_fourth * w_x
            )
            x_next = x + dx_dt * dt + noise_scale * noise_x_t

            # dy/dt = ay + ωx - b(x²+y²)y + (x²+y²)²y
            dy_dt = (
                self._config.a * w_y
                + self._config.omega * w_x
                - self._config.b * w_squared * w_y
                + w_fourth * w_y
            )
            y_next = y + dy_dt * dt + noise_scale * noise_y_t

            return (x_next, y_next), (x_next, y_next)

        initial_state = (initial_x, initial_y)
        _, states = lax.scan(step, initial_state, (lambda_t[:-1], noise_x, noise_y))

        x_traj = jnp.concatenate([jnp.array([initial_x]), states[0]], axis=0)
        y_traj = jnp.concatenate([jnp.array([initial_y]), states[1]], axis=0)

        return jnp.stack([x_traj, y_traj], axis=-1)

    def simulate(
        self,
        key: jax.random.PRNGKey,
        total_len: int,
        sigma_tilde: float = 0.01,
        initial_state: Optional[jnp.ndarray] = None,
        noise: Optional[jnp.ndarray] = None,
        null_simulation: bool = False,
    ):
        """Legacy simulator on the configured absolute-time grid.

        Returns time, Cartesian state, forcing and step-noise scale; total_len is ignored."""
        key_noise, key_ic = jax.random.split(key)

        time = jnp.arange(self._config.t_start, self._config.t_end, self._config.dt)
        if null_simulation:
            config_null = replace(
                self._config, r=self._config.null_rate_factor * self._config.r
            )
            lambda_t = self._lambda_schedule(time, config_null)
        else:
            lambda_t = self._lambda_schedule(time, self._config)

        if initial_state is None:
            initial_x, initial_y = self._sample_initial_state(key_ic, self._config)
        else:
            initial_state = jnp.asarray(initial_state)
            initial_x, initial_y = initial_state[0], initial_state[1]

        noise_scale = self._config.D2 * sigma_tilde * jnp.sqrt(self._config.dt)

        trajectory = self._simulate_path(
            key_noise,
            lambda_t,
            self._config.dt,
            noise_scale,
            initial_x,
            initial_y,
            noise=noise,
        )

        return time, trajectory, lambda_t, noise_scale

    def is_tipped(
        self,
        trajectory: jnp.ndarray,
        **kwargs,
    ):
        """Check if the trajectory is tipped (radius exceeds threshold)."""
        radius = jnp.sqrt(trajectory[:, 0] ** 2 + trajectory[:, 1] ** 2)
        return bool(jnp.any(radius > self._config.tip_threshold))

    def locate_transition(
        self,
        trajectory: jnp.ndarray,
        null_simulation: bool,
        **kwargs,
    ):
        """Locate the transition index."""
        if null_simulation:
            return int(trajectory.shape[0] // 2)

        radius = jnp.sqrt(trajectory[:, 0] ** 2 + trajectory[:, 1] ** 2)
        crossed = jnp.where(radius > self._config.tip_threshold, 1, 0)
        has_crossed = bool(jnp.any(crossed))
        if not has_crossed:
            return None
        return int(jnp.argmax(crossed))
