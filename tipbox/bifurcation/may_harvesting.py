from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

from ..types import BoxModel
from ..utils import (
    parameter,
    register_dataclass_pytree,
    sampling_stride,
)


@register_dataclass_pytree
@dataclass(frozen=True)
class MayHarvestingConfig:
    """Coefficients and numerical controls for May's harvesting model (Bury et al., 2021)."""

    # Model parameters (from paper)
    r: float = parameter(1.0, "Intrinsic population growth rate.", "1 / model time")
    k: float = parameter(1.0, "Biomass carrying capacity.", "biomass")
    s: float = parameter(
        0.1,
        "Harvesting saturation scale; the upper fold requires s/k < 1/sqrt(27).",
        "biomass",
    )

    # Bifurcation parameter range
    h_min: float = parameter(0.15, "Starting harvesting rate.", "biomass / model time")
    h_max: float = parameter(
        0.27, "Forced ramp endpoint and cap.", "biomass / model time"
    )
    h_bifurcation: float | None = parameter(
        None,
        "Optional fold marker. None computes the upper fold from r, k and s.",
        "biomass / model time",
    )

    # Noise parameter
    null_initial_mean: float = parameter(
        0.75, "Mean of the clipped Gaussian null initial-state sampler.", "biomass"
    )
    null_initial_std: float = parameter(
        0.1, "Standard deviation of the null initial-state sampler.", "biomass"
    )
    sigma: float = parameter(
        0.01,
        "Diffusion amplitude multiplied by noise and sqrt(dt) per step.",
        "biomass / sqrt(model time)",
    )

    # Simulation parameters
    dt: float = parameter(
        0.01,
        "Default integration step.",
        "model time",
    )
    t_start: float = parameter(
        0.0,
        "Simulation start time.",
        "model time",
    )
    t_end: float = parameter(
        100.0,
        "End time when make_trajectory is called without n; explicit n sets the end time.",
        "model time",
    )

    # Initial state
    initial_state_min: float = parameter(
        0.5, "Lower bound for the default uniform initial-state sampler.", "state"
    )
    initial_state_max: float = parameter(
        1.0, "Upper bound for the default uniform initial-state sampler.", "state"
    )

    @classmethod
    def default(cls) -> "MayHarvestingConfig":
        """Return default configuration."""
        return cls()


class MayHarvestingModel(BoxModel):
    """Stochastic logistic growth with saturating harvesting and an upper fold."""

    config_class = MayHarvestingConfig

    @property
    def critical_parameter(self):
        """Upper positive fold from f=0 and df/dx=0, or an explicit override."""
        c = self._config
        if c.h_bifurcation is not None:
            return c.h_bifurcation
        if c.r <= 0 or c.k <= 0 or c.s <= 0 or c.s / c.k >= 1 / jnp.sqrt(27.0):
            raise ValueError(
                "The analytic upper harvesting fold requires r,k,s > 0 and s/k < 1/sqrt(27)"
            )
        x = c.k / 6 + c.k / 3 * jnp.cos(jnp.arccos(1 - 54 * (c.s / c.k) ** 2) / 3)
        return c.r * (1 - x / c.k) * (c.s**2 + x**2) / x

    state_names = ("biomass",)
    observable_names = state_names
    noise_dimension = 1
    default_sample_dt = 1.0
    forcing_name = "harvesting_rate"

    def _initial_state(self, key, null_simulation):
        return self._sample_initial_state(key, null_simulation)

    def _forcing(self, time, null_simulation, param_end_multiplier):
        return self._h_schedule(time, null_simulation, param_end_multiplier)

    def _integrate(
        self, time, forcing, dt, noise, state, innovations, midpoint_forcing
    ):
        scale = self._config.sigma * noise * jnp.sqrt(dt)
        path = self._simulate_path(
            None, forcing, dt, scale, state[0], noise=innovations[:, 0]
        )
        return path[:, None], scale

    def _h_schedule(
        self,
        t: jnp.ndarray,
        null_simulation: bool = False,
        param_end_multiplier: float = 1.1,
    ):
        """Ramp harvesting to its endpoint; null forcing is capped below the fold."""
        t_total = t[-1] - t[0]
        t_to_bifurcation = t_total / param_end_multiplier
        t_rel = t - t[0]
        # Same linear ramp as forced (from h_min to h_max over t_to_bifurcation)
        ramp = self._config.h_min + (t_rel / t_to_bifurcation) * (
            self._config.h_max - self._config.h_min
        )

        if null_simulation:
            # Cap so distance (null_cap → critical) = (critical → h_max); clamp to h_min
            range_past = self._config.h_max - self.critical_parameter
            cap = self.critical_parameter - range_past
            cap = jnp.maximum(cap, self._config.h_min)
            return jnp.minimum(ramp, cap)
        # For times before reaching bifurcation: linear increase; after: constant at h_max
        h_values = jnp.where(t_rel <= t_to_bifurcation, ramp, self._config.h_max)
        return h_values

    def _sample_initial_state(
        self,
        key: jax.random.PRNGKey,
        null_simulation: bool = False,
    ):
        """Sample an initial state (biomass x).

        For null simulations, start near the stable equilibrium for h_min.
        For forced simulations, start near carrying capacity (stable for low h).
        """
        if null_simulation:
            # For null simulations, start near stable equilibrium at h_min
            # Approximate equilibrium: solve rx(1-x/k) - h_min*(x²/(s²+x²)) = 0
            # For h_min=0.15, r=1, k=1, s=0.1, equilibrium is around 0.7-0.8
            # Sample around this value
            equilibrium_approx = self._config.null_initial_mean
            x = float(
                jax.random.normal(key, ()) * self._config.null_initial_std
                + equilibrium_approx  # Small noise around equilibrium
            )
            x = jnp.clip(
                x, self._config.initial_state_min, self._config.initial_state_max
            )  # Ensure reasonable bounds
        else:
            # For forced simulations, start near carrying capacity
            x = float(
                jax.random.uniform(
                    key,
                    (),
                    minval=self._config.initial_state_min,
                    maxval=self._config.initial_state_max,
                )
            )
        return float(x)

    def _drift(self, x: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        """Deterministic drift in the model's physical state coordinates."""
        logistic = self._config.r * x * (1.0 - x / self._config.k)

        harvesting = h * (x**2 / (self._config.s**2 + x**2))

        return logistic - harvesting

    def _simulate_path(
        self,
        key: jax.random.PRNGKey,
        h_t: jnp.ndarray,
        dt: float,
        noise_scale: float,
        initial_x: float,
        noise: Optional[jnp.ndarray] = None,
    ):
        """Stochastic integration of May's harvesting model using Euler-Maruyama.

        dx/dt = rx(1 - x/k) - h(x^2 / (s^2 + x^2)) + σξ(t)
        """
        if noise is None:
            noise = jax.random.normal(key, shape=(h_t.shape[0] - 1,))
        else:
            noise = jnp.asarray(noise)

        def step(state, inputs):
            h_val, noise_t = inputs
            x = state

            drift = self._drift(x, h_val)

            x_next = x + drift * dt + noise_scale * noise_t

            # Ensure biomass is non-negative
            x_next = jnp.maximum(x_next, 0.0)

            return x_next, x_next

        initial_state = initial_x
        _, states = lax.scan(step, initial_state, (h_t[:-1], noise))

        x_traj = jnp.concatenate([jnp.array([initial_x]), states], axis=0)

        return x_traj

    def simulate(
        self,
        key: jax.random.PRNGKey,
        total_len: int,
        sigma_tilde: float = 0.01,
        null_simulation: bool = False,
        dt_sample: float = 1.0,
        param_end_multiplier: float = 1.1,
        initial_state: Optional[jnp.ndarray] = None,
        noise: Optional[jnp.ndarray] = None,
    ):
        """simulator: returns time, [biomass, 0], harvesting and step-noise scale.

        total_len counts output intervals; the result includes the initial sample."""
        key_noise, key_ic = jax.random.split(key)

        sample_every = sampling_stride(dt_sample, self._config.dt)
        total_time = self._config.t_start + total_len * dt_sample
        n_steps = total_len * sample_every + 1
        time_fine = jnp.linspace(self._config.t_start, total_time, n_steps)

        h_t = self._h_schedule(
            time_fine,
            null_simulation=null_simulation,
            param_end_multiplier=param_end_multiplier,
        )

        if initial_state is None:
            initial_x = self._sample_initial_state(
                key_ic, null_simulation=null_simulation
            )
        else:
            initial_x = jnp.asarray(initial_state)

        noise_scale = sigma_tilde * self._config.sigma * jnp.sqrt(self._config.dt)

        x_traj_fine = self._simulate_path(
            key_noise,
            h_t,
            self._config.dt,
            noise_scale,
            initial_x,
            noise=noise,
        )

        indices = jnp.arange(0, n_steps, sample_every)

        time = time_fine[indices]
        trajectory_1d = x_traj_fine[indices]
        param_values = h_t[indices]

        trajectory = jnp.stack([trajectory_1d, jnp.zeros_like(trajectory_1d)], axis=-1)

        return time, trajectory, param_values, noise_scale

    def is_tipped(
        self,
        trajectory: jnp.ndarray,
        param_values: jnp.ndarray,
    ):
        """Check if the forcing reached the configured or analytic upper fold."""
        return float(jnp.max(param_values)) >= self.critical_parameter

    def locate_transition(
        self,
        trajectory: jnp.ndarray,
        null_simulation: bool,
        param_values: jnp.ndarray,
        time=None,
    ):
        """Midpoint for null; first fold crossing for forced, or None if absent."""
        if null_simulation:
            return len(param_values) // 2

        crossed = param_values >= self.critical_parameter

        if bool(jnp.any(crossed)):
            return int(jnp.argmax(crossed))

        return None
