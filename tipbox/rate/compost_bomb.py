import math
from dataclasses import dataclass, replace
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

from ..types import BoxModel
from ..utils import parameter, register_dataclass_pytree


@register_dataclass_pytree
@dataclass(frozen=True)
class CompostBombConfig:
    """Configuration for the stochastic compost-bomb dynamic system.

    Based on Luke & Cox (2011) equations:
    μ * dT_s/dt = -λ(T_s - T_a) + A * C_s * r_0 * e^(alpha(T_s - T_ref)) + D_3 * dW
    dC_s/dt = Π - C_s * r_0 * e^(alpha(T_s - T_ref))

    where:
    - T_a(t) = v * t (atmospheric temperature forcing)
    - r(T_s) = r_0 * e^(alpha(T_s - T_ref)) (temperature-dependent respiration)
    - The positive feedback: T ↑ → r(T) ↑ → more heat → T ↑ (compost-bomb instability)
    """

    # Forcing parameters
    v: float = parameter(0.15, "Linear air-temperature forcing rate.", "degC / year")
    D3: float = parameter(
        50,
        "Temperature diffusion prefactor applied after the deterministic heat-capacity division.",
        "degC / sqrt(year)",
    )

    Pi: float = parameter(
        1.055, "Soil-carbon input from litterfall.", "kg / m^2 / year"
    )
    r0: float = parameter(
        0.01, "Respiration rate at the reference temperature.", "1 / year"
    )
    alpha: float = parameter(
        math.log(2.5) / 10.0,
        "Exponential temperature sensitivity of respiration.",
        "1 / degC",
    )
    mu: float = parameter(
        2500000.0, "Soil heat capacity per surface area.", "J / m^2 / degC"
    )
    A: float = parameter(
        39000000.0, "Heat released per unit carbon respired.", "J / kg"
    )
    lambda_param: float = parameter(
        5049000.0, "Soil-to-air heat-transfer coefficient.", "J / year / m^2 / degC"
    )

    # Simulation parameters
    null_rate_factor: float = parameter(
        0.95,
        "Multiplier applied to the forcing rate for null simulations.",
        "dimensionless",
    )
    dt: float = parameter(
        0.05,
        "Default integration step; AMOC uses this field for output spacing.",
        "year",
    )
    t_start: float = parameter(
        0.0, "Start time; rate-induced forcing uses this absolute clock origin.", "year"
    )
    t_end: float = parameter(
        100.0,
        "End time when make_trajectory is called without n; explicit n sets the end time.",
        "year",
    )

    initial_state_min: float = parameter(
        0, "Lower bound for the default uniform initial-state sampler.", "state"
    )
    initial_state_max: float = parameter(
        0, "Upper bound for the default uniform initial-state sampler.", "state"
    )
    T_ref: float = parameter(0.0, "Reference soil temperature for respiration.", "degC")
    tip_threshold: float = parameter(
        30.0,
        "Soil-temperature threshold used by the legacy tipping classifier.",
        "degC",
    )

    @classmethod
    def default(cls) -> "CompostBombConfig":
        """Return default configuration."""
        return cls()


class CompostBombModel(BoxModel):
    """BoxModel abstraction for the stochastic compost-bomb dynamic system.

    Implements the compost-bomb system from Luke & Cox (2011):
    μ * dT_s/dt = -λ(T_s - T_a) + A * C_s * r(T_s) + D_3 * dW
    dC_s/dt = Π - C_s * r(T_s)

    where:
    - r(T_s) = r_0 * e^(alpha(T_s - T_ref)) (temperature-dependent respiration)
    - T_a(t) = v * t (atmospheric temperature forcing)
    - The positive feedback creates the compost-bomb instability
    """

    config_class = CompostBombConfig

    state_names = ("soil_carbon", "soil_temperature")
    observable_names = state_names
    noise_dimension = 1
    time_unit = "year"
    forcing_name = "air_temperature"
    integration_method = "Euler-Maruyama with sequential carbon/temperature update"

    def _initial_state(self, key, null_simulation):
        return self._sample_initial_state(key, self._config)

    def _forcing(self, time, null_simulation, param_end_multiplier):
        config = (
            replace(self._config, v=self._config.v * self._config.null_rate_factor)
            if null_simulation
            else self._config
        )
        return self._T_a_schedule(time, config)

    def _integrate(
        self, time, forcing, dt, noise, state, innovations, midpoint_forcing
    ):
        scale = self._config.D3 * noise * jnp.sqrt(dt)
        path = self._simulate_path(
            None, forcing, dt, scale, state[0], state[1], noise=innovations[:, 0]
        )
        return path, scale

    def _T_a_schedule(
        self,
        t: jnp.ndarray,
        config: CompostBombConfig,
    ):
        """Compute the T_a(t) schedule (atmospheric temperature forcing)."""
        # T_a(t) = v * t
        return config.v * t

    def _sample_initial_state(
        self,
        key: jax.random.PRNGKey,
        config: CompostBombConfig,
    ):
        """Sample an initial state (C, T)."""
        key_C, key_T = jax.random.split(key)
        C = float(
            jax.random.uniform(
                key_C,
                (),
                minval=config.initial_state_min,
                maxval=config.initial_state_max,
            )
        )
        T = float(
            jax.random.uniform(
                key_T,
                (),
                minval=config.initial_state_min,
                maxval=config.initial_state_max,
            )
        )
        return C, T

    def _simulate_path(
        self,
        key: jax.random.PRNGKey,
        T_a_t: jnp.ndarray,
        dt: float,
        noise_scale: float,
        initial_C: float,
        initial_T: float,
        noise: Optional[jnp.ndarray] = None,
    ):
        """Stochastic integration of the compost-bomb system.

        Implements (Luke & Cox 2011):
        μ * dT_s/dt = -λ(T_s - T_a) + A * C_s * r(T_s) + D_3 * dW
        dC_s/dt = Π - C_s * r(T_s)

        where r(T_s) = r_0 * e^(alpha(T_s - T_ref)) (temperature-dependent respiration)
        """
        if noise is None:
            noise_T = jax.random.normal(key, shape=(T_a_t.shape[0] - 1,))
        else:
            noise_T = jnp.asarray(noise)

        def step(state, inputs):
            T_a_val, noise_T_t = inputs
            C, T = state

            # Respiration depends on temperature, not time
            r_T = self._config.r0 * jnp.exp(
                self._config.alpha * (T - self._config.T_ref)
            )

            # dC/dt = Π - C * r(T)
            dC_dt = self._config.Pi - C * r_T
            C_next = C + dC_dt * dt
            C_next = jnp.maximum(C_next, 0.0)

            # μ dT/dt = C * r(T) * A - λ (T - T_a(t))
            dT_dt = (
                C_next * r_T * self._config.A
                - self._config.lambda_param * (T - T_a_val)
            ) / self._config.mu

            T_next = T + dT_dt * dt + noise_scale * noise_T_t
            return (C_next, T_next), (C_next, T_next)

        initial_state = (initial_C, initial_T)
        _, states = lax.scan(step, initial_state, (T_a_t[:-1], noise_T))

        # Build full trajectory including initial state
        C_traj = jnp.concatenate([jnp.array([initial_C]), states[0]], axis=0)
        T_traj = jnp.concatenate([jnp.array([initial_T]), states[1]], axis=0)

        return jnp.stack([C_traj, T_traj], axis=-1)

    def simulate(
        self,
        key: jax.random.PRNGKey,
        total_len: int,
        sigma_tilde: float = 0.01,
        initial_state: Optional[jnp.ndarray] = None,
        noise: Optional[jnp.ndarray] = None,
        null_simulation: bool = False,
    ):
        """Simulate the compost-bomb system.

        For null_simulation=True, same T_a(t)=v*t formula but v at 95% (slightly slower rate).
        """
        key_noise, key_ic = jax.random.split(key)

        # Setup time and T_a schedule
        time = jnp.arange(self._config.t_start, self._config.t_end, self._config.dt)
        if null_simulation:
            config_null = replace(
                self._config, v=self._config.null_rate_factor * self._config.v
            )
            T_a_t = self._T_a_schedule(time, config_null)
        else:
            T_a_t = self._T_a_schedule(time, self._config)

        # Randomly sample initial state if not provided
        if initial_state is None:
            initial_C, initial_T = self._sample_initial_state(key_ic, self._config)
        else:
            initial_state = jnp.asarray(initial_state)
            initial_C, initial_T = initial_state[0], initial_state[1]

        # Configure noise scale: D3 * sigma_tilde * sqrt(dt) for Euler-Maruyama
        # For the dT equation, we use D3 directly (D3 already has appropriate units)
        noise_scale = self._config.D3 * sigma_tilde * jnp.sqrt(self._config.dt)

        # Simulate trajectory
        trajectory = self._simulate_path(
            key_noise,
            T_a_t,
            self._config.dt,
            noise_scale,
            initial_C,
            initial_T,
            noise=noise,
        )

        # Return T_a_t as param_values for consistency with other models
        # Return noise scale (like other rate systems) for consistency
        return time, trajectory, T_a_t, noise_scale

    def is_tipped(
        self,
        trajectory: jnp.ndarray,
        **kwargs,
    ):
        """Check if the trajectory is tipped (temperature exceeds threshold)."""
        # Check if temperature (second component) exceeds threshold
        T = trajectory[:, 1]
        return bool(jnp.any(T > self._config.tip_threshold))

    def locate_transition(
        self,
        trajectory: jnp.ndarray,
        null_simulation: bool,
        **kwargs,
    ):
        """Locate the transition index."""
        if null_simulation:
            return int(trajectory.shape[0] // 2)

        # Check if temperature exceeds threshold
        T = trajectory[:, 1]
        crossed = jnp.where(T > self._config.tip_threshold, 1, 0)
        has_crossed = bool(jnp.any(crossed))
        if not has_crossed:
            return None
        return int(jnp.argmax(crossed))
