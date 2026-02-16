import math
from dataclasses import dataclass, replace
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

from ..types import BoxModel
from ..utils import extract_segment, register_dataclass_pytree


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
    v: float = 0.15  # Rate of atmospheric temperature forcing
    D3: float = 50  # Noise magnitude (paper default: 0.5, so 50 * 0.01 following default sigma_tilde)

    Pi: float = 1.055  # Litter fall from plants (kg m^-2 yr^-1)
    r0: float = 0.01  # Microbial respiration parameter (yr^-1)
    alpha: float = math.log(2.5) / 10.0  # Microbial respiration parameter
    mu: float = 2.5e6  # Soil heat capacity (J m^-2 degC^-1)
    A: float = 3.9e7  # Specific heat from microbial respiration (J kg^-1)
    lambda_param: float = 5.049e6  # Soil-to-air heat transfer coefficient (J yr^-1 m^-2 degC^-1)

    # Simulation parameters
    dt: float = 0.05  # Time step
    t_start: float = 0.0
    t_end: float = 100.0  # Paper specifies time range 0 to 100

    initial_state_min: float = 0  # Initial state (carbon and temperature) (paper: starts at 0)
    initial_state_max: float = 0  # Initial state (carbon and temperature) (paper: starts at 0)
    T_ref: float = 0.0  # Reference temperature (0.0 is the default)
    tip_threshold: float = 30.0  # Threshold for detecting tipping (temperature)

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

    def __init__(self, config: CompostBombConfig):
        self._config = config

    def get_parameters(self):
        """Get the parameters of the model."""
        return {
            "v": self._config.v,
            "D3": self._config.D3,
            "Pi": self._config.Pi,
            "r0": self._config.r0,
            "alpha": self._config.alpha,
            "mu": self._config.mu,
            "A": self._config.A,
            "lambda_param": self._config.lambda_param,
            "dt": self._config.dt,
            "t_start": self._config.t_start,
            "t_end": self._config.t_end,
            "initial_state_min": self._config.initial_state_min,
            "initial_state_max": self._config.initial_state_max,
            "T_ref": self._config.T_ref,
            "tip_threshold": self._config.tip_threshold,
        }

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
            r_T = self._config.r0 * jnp.exp(self._config.alpha * (T - self._config.T_ref))

            # dC/dt = Π - C * r(T)
            dC_dt = self._config.Pi - C * r_T
            C_next = C + dC_dt * dt
            C_next = jnp.maximum(C_next, 0.0)

            # μ dT/dt = C * r(T) * A - λ (T - T_a(t))
            dT_dt = (C_next * r_T * self._config.A - self._config.lambda_param * (T - T_a_val)) / self._config.mu

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
        schedule=None,
        initial_state: Optional[jnp.ndarray] = None,
        noise: Optional[jnp.ndarray] = None,
        null_simulation: bool = False,
    ):
        """Simulate the compost-bomb system.

        For null_simulation=True, same T_a(t)=v*t formula but v at 95% (slightly slower rate).
        """
        key_noise, key_ic = jax.random.split(key)

        if schedule is not None:
            from ..bifurcation.schedule import resolve_schedule_for_simulation

            # Max forcing = v * t_end
            T_a_t, _ = resolve_schedule_for_simulation(
                schedule,
                total_len,
                0.0,
                self._config.v * self._config.t_end,
            )
            time = jnp.arange(self._config.t_start, self._config.t_start + total_len * self._config.dt, self._config.dt)
        else:
            # Legacy: internal T_a schedule with optional slower null ramp
            time = jnp.arange(self._config.t_start, self._config.t_end, self._config.dt)
            if null_simulation:
                config_null = replace(self._config, v=0.95 * self._config.v)
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
        """Normalize atmospheric forcing by max value (v * t_end)."""
        return param_values / (self._config.v * self._config.t_end)
