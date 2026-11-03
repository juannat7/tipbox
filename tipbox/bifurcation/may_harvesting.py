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
    """Configuration for May's harvesting model.

    Based on Thomas Bury et al. (PNAS, 2021):
    dx/dt = rx(1 - x/k) - h(x^2 / (s^2 + x^2)) + sigma * noise(t)

    where:
    - x: biomass of some population
    - k: carrying capacity
    - h: harvesting rate (bifurcation parameter)
    - s: characterizes the nonlinear dependence of harvesting output on current biomass
    - r: intrinsic per capita growth rate of the population
    - sigma: noise amplitude
    - noise(t): Gaussian white noise process

    A fold bifurcation occurs at approximately h = 0.260436 (for r=1, k=1, s=0.1).
    """

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
        "Default integration step; AMOC uses this field for output spacing.",
        "model time",
    )
    t_start: float = parameter(
        0.0,
        "Start time; rate-induced forcing uses this absolute clock origin.",
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
    """BoxModel abstraction for May's harvesting model.

    Implements the stochastic harvesting model:
    dx/dt = rx(1 - x/k) - h(x^2 / (s^2 + x^2)) + σξ(t)

    where h is the bifurcation parameter that varies linearly from h_min to h_max.
    The default fold is computed analytically at approximately h = 0.260436.
    """

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
        """Compute the h(t) schedule (harvesting rate).

        For null simulations: same linear ramp as forced but clipped so the range from
        null_cap to critical equals the range from critical to h_max (symmetric about critical).
        For forced simulations:
          - h increases linearly from h_min to h_max (up to bifurcation point)
          - Beyond the bifurcation point, h stays constant at h_max
          - param_end_multiplier extends the time series, keeping h constant beyond critical point
        """
        # Total time range
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
        else:
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
        """Compute the drift term: rx(1 - x/k) - h(x^2 / (s^2 + x^2)).

        Parameters
        ----------
        x : biomass
        h : harvesting rate

        Returns
        -------
        drift : dx/dt (without noise)
        """
        # Logistic growth: rx(1 - x/k)
        logistic = self._config.r * x * (1.0 - x / self._config.k)

        # Harvesting: h(x^2 / (s^2 + x^2))
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

            # Drift: rx(1 - x/k) - h(x^2 / (s^2 + x^2))
            drift = self._drift(x, h_val)

            # Euler-Maruyama: dx = drift * dt + noise_scale * dW
            x_next = x + drift * dt + noise_scale * noise_t

            # Ensure biomass is non-negative
            x_next = jnp.maximum(x_next, 0.0)

            return x_next, x_next

        initial_state = initial_x
        _, states = lax.scan(step, initial_state, (h_t[:-1], noise))

        # Build full trajectory including initial state
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
        """Simulate May's harvesting model.

        Parameters
        ----------
        key : JAX random key
        total_len : int
            Total number of output time points
        sigma_tilde : float, default=0.01
            Base noise amplitude
        null_simulation : bool, default=False
            If True, same linear ramp as forced but h capped so range to critical equals (h_max - critical) (null trajectory)
        dt_sample : float, default=1.0
            Sampling interval for output
        param_end_multiplier : float, default=1.1
            Multiplier for extending simulation time beyond transition

        Returns
        -------
        time : (total_len,) array
            Time points
        trajectory : (total_len, 2) array
            State trajectory [x, 0] (second column is zero for 1D system)
        param_values : (total_len,) array
            Harvesting rate h(t) values
        noise_scale : float
            Effective noise scale used
        """
        key_noise, key_ic = jax.random.split(key)

        # Setup time grid
        # We need enough time steps for the desired output length
        # Total simulation time = total_len * dt_sample
        sample_every = sampling_stride(dt_sample, self._config.dt)
        total_time = self._config.t_start + total_len * dt_sample
        n_steps = total_len * sample_every + 1
        time_fine = jnp.linspace(self._config.t_start, total_time, n_steps)

        # Compute h schedule
        # For null simulations: same linear ramp as forced, cap = critical - (h_max - critical)
        # For forced simulations: h increases linearly from h_min to h_bifurcation * param_end_multiplier
        # param_end_multiplier extends beyond the bifurcation point to allow simulation past tipping
        h_t = self._h_schedule(
            time_fine,
            null_simulation=null_simulation,
            param_end_multiplier=param_end_multiplier,
        )

        # Randomly sample initial state if not provided
        if initial_state is None:
            initial_x = self._sample_initial_state(
                key_ic, null_simulation=null_simulation
            )
        else:
            initial_x = jnp.asarray(initial_state)

        # Configure noise scale: sigma_tilde * sqrt(dt) for Euler-Maruyama
        noise_scale = sigma_tilde * self._config.sigma * jnp.sqrt(self._config.dt)

        # Simulate trajectory at fine time steps
        x_traj_fine = self._simulate_path(
            key_noise,
            h_t,
            self._config.dt,
            noise_scale,
            initial_x,
            noise=noise,
        )

        # Downsample to output rate
        indices = jnp.arange(0, n_steps, sample_every)

        time = time_fine[indices]
        trajectory_1d = x_traj_fine[indices]
        param_values = h_t[indices]

        # Convert to 2D trajectory [x, 0] for consistency with other models
        trajectory = jnp.stack([trajectory_1d, jnp.zeros_like(trajectory_1d)], axis=-1)

        return time, trajectory, param_values, noise_scale

    def is_tipped(
        self,
        trajectory: jnp.ndarray,
        param_values: jnp.ndarray,
    ):
        """Check if the parameter crossed the bifurcation point (h = 0.26)."""
        return float(jnp.max(param_values)) >= self.critical_parameter

    def locate_transition(
        self,
        trajectory: jnp.ndarray,
        null_simulation: bool,
        param_values: jnp.ndarray,
    ):
        """Locate transition index: midpoint for null, bifurcation crossing for forced."""
        if null_simulation:
            return len(param_values) // 2

        crossed = param_values >= self.critical_parameter

        if bool(jnp.any(crossed)):
            return int(jnp.argmax(crossed))

        return len(param_values) - 1
