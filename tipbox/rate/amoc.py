"""
AMOC 2-box model, rate-induced tipping (Ritchie et al. 2023; Alkhayuon et al. 2019).

Port from dews/experiments/amoc_boxmodel.py. H(t) = ramp then plateau (forcing_ramp).
Tip threshold: transition during ramp phase (t < 0).
"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

from .._amoc import AMOCPhysics, observe, rk4_step, strength_sv
from ..types import BoxModel
from ..utils import parameter, register_dataclass_pytree

# 2x CO2 parameters from BoxModel_2DH_IVP.m (rtippingreview)


def _forcing_ramp(t: jnp.ndarray, r: float, dH: float, H0: float = 0.0) -> jnp.ndarray:
    """H(t) = H0 + dH/cosh(r*t) for t < 0, H0 + dH for t >= 0 (ramp then plateau)."""
    out = jnp.where(t < 0, H0 + dH / jnp.cosh(r * t), H0 + dH)
    return out


@register_dataclass_pytree
@dataclass(frozen=True)
class AMOCRateConfig:
    """Configuration for AMOC 2-box model (rate-induced)."""

    physics: AMOCPhysics = parameter(
        AMOCPhysics(), "Shared calibrated physical AMOC parameters, expanded below.", ""
    )
    initial_SN: float = parameter(
        0.0324, "Mean initial North Atlantic salinity anomaly.", "100 × mass fraction"
    )
    initial_ST: float = parameter(
        0.1434,
        "Mean initial tropical Atlantic salinity anomaly.",
        "100 × mass fraction",
    )
    H0: float = parameter(0.0, "Initial freshwater-hosing level.", "Sv")
    dH: float = parameter(0.365, "Freshwater-hosing increase above H0.", "Sv")
    r_slow: float = parameter(0.005, "Null forcing rate.", "1 / model time")
    r_fast: float = parameter(0.017, "Forced ramp rate.", "1 / model time")
    sigma_SN: float = parameter(
        0.0001,
        "North Atlantic diffusion coefficient before the noise multiplier.",
        "state / sqrt(year)",
    )
    sigma_ST: float = parameter(
        0.0001,
        "Tropical Atlantic diffusion coefficient before the noise multiplier.",
        "state / sqrt(year)",
    )
    sigma_ic: float = parameter(
        0.01,
        "Initial-state standard deviation in make_trajectory; legacy simulate uses the configured mean.",
        "state",
    )
    dt: float = parameter(
        5.0,
        "Default integration step; AMOC uses this field for output spacing.",
        "year",
    )
    dt_inner: float = parameter(0.2, "AMOC RK4 integration step.", "year")
    t_start: float = parameter(
        -2505.0,
        "Start time; rate-induced forcing uses this absolute clock origin.",
        "year",
    )
    t_end: float = parameter(
        3200.0,
        "End time when make_trajectory is called without n; explicit n sets the end time.",
        "year",
    )
    # Tipped when Q (AMOC strength in Sv) < 0 (collapsed / reversed).
    Q_collapse_threshold: float = parameter(
        0.0, "AMOC transport threshold for the legacy collapse classifier.", "Sv"
    )

    @classmethod
    def default(cls) -> "AMOCRateConfig":
        return cls()


class AMOCRateModel(BoxModel):
    """AMOC 2-box model, rate-induced. H(t) = ramp then plateau (peak at t=0). Transition = when t crosses 0 (H peaks/plateaus). is_tipped = Q collapsed (below physical threshold)."""

    config_class = AMOCRateConfig

    state_names = ("north_salinity_anomaly", "tropical_salinity_anomaly")
    observable_names = ("amoc_strength",) + state_names
    noise_dimension = 2
    integration_step_field = "dt_inner"
    integration_method = "RK4 drift with additive Euler noise"
    uses_midpoint_forcing = True
    forcing_name = "freshwater_hosing"
    time_unit = "year"

    def _initial_state(self, key, null_simulation):
        return self._sample_initial_state(key)

    def _forcing(self, time, null_simulation, param_end_multiplier):
        return self._H_schedule(time, null_simulation)

    def _integrate(
        self, time, forcing, dt, noise, state, innovations, midpoint_forcing
    ):
        scale = (
            jnp.array([self._config.sigma_SN, self._config.sigma_ST])
            * noise
            * jnp.sqrt(dt)
        )
        path = self._simulate_path(
            forcing[:-1], midpoint_forcing, forcing[1:], dt, state, scale, innovations
        )
        return path, scale

    def _observe(self, path):
        return observe(path, self._config.physics)

    def _H_schedule(self, t: jnp.ndarray, null_simulation: bool) -> jnp.ndarray:
        """H(t) = forcing_ramp(t, r, dH, H0). null_simulation=True -> r_slow (no tip); False -> r_fast (tip)."""
        r = self._config.r_slow if null_simulation else self._config.r_fast
        return _forcing_ramp(t, r, self._config.dH, self._config.H0)

    def _sample_initial_state(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        """Gaussian anomalies about the configured mean; negative anomalies are valid."""
        noise = jax.random.normal(key, (2,))
        x0 = (
            jnp.array([self._config.initial_SN, self._config.initial_ST])
            + self._config.sigma_ic * noise
        )
        return x0

    def _simulate_path(
        self,
        H_cur: jnp.ndarray,
        H_mid: jnp.ndarray,
        H_next: jnp.ndarray,
        dt: float,
        initial_state: jnp.ndarray,
        noise_scale: jnp.ndarray,
        noise: jnp.ndarray,
    ) -> jnp.ndarray:
        """Single RK4 integrator: state_next = rk4_step(...) + noise_scale * noise_t. Use noise_scale=0 for deterministic."""

        def step(state, inputs):
            h_cur, h_mid, h_next, noise_t = inputs
            state_next = (
                rk4_step(state, h_cur, h_mid, h_next, dt, self._config.physics)
                + noise_scale * noise_t
            )
            return state_next, state_next

        _, states = lax.scan(step, initial_state, (H_cur, H_mid, H_next, noise))
        return jnp.concatenate([initial_state[None, :], states], axis=0)

    def simulate(
        self,
        key: jax.random.PRNGKey,
        total_len: int,
        sigma_tilde: float = 0.01,
        initial_state: Optional[jnp.ndarray] = None,
        noise: Optional[jnp.ndarray] = None,
        null_simulation: bool = False,
    ):
        """Simulate rate-induced AMOC. Fixed time range [t_start, t_end]; total_len ignored. Both branches use RK4."""
        key_noise, _ = jax.random.split(key)

        dt_out = self._config.dt
        dt_inner = self._config.dt_inner
        time_inner = jnp.arange(self._config.t_start, self._config.t_end, dt_inner)
        H_inner = self._H_schedule(time_inner, null_simulation)
        t_mid = time_inner[:-1] + 0.5 * dt_inner
        r = self._config.r_slow if null_simulation else self._config.r_fast
        H_mid = _forcing_ramp(t_mid, r, self._config.dH, self._config.H0)
        H_cur, H_next = H_inner[:-1], H_inner[1:]

        if initial_state is None:
            initial_state = jnp.array(
                [self._config.initial_SN, self._config.initial_ST]
            )
        else:
            initial_state = jnp.asarray(initial_state)

        deterministic = sigma_tilde <= 1e-9
        n_steps = H_cur.shape[0]
        if deterministic:
            noise_scale = jnp.zeros(2)
            noise_arr = jnp.zeros((n_steps, 2))
        else:
            noise_scale = jnp.array(
                [
                    self._config.sigma_SN * jnp.sqrt(dt_inner) * sigma_tilde,
                    self._config.sigma_ST * jnp.sqrt(dt_inner) * sigma_tilde,
                ]
            )
            noise_arr = (
                jax.random.normal(key_noise, (n_steps, 2))
                if noise is None
                else jnp.asarray(noise)
            )
        path_inner = self._simulate_path(
            H_cur, H_mid, H_next, dt_inner, initial_state, noise_scale, noise_arr
        )

        step = max(1, int(dt_out / dt_inner))
        time = time_inner[::step]
        path = path_inner[::step]
        H_t = H_inner[::step]

        Q_t = strength_sv(path[:, 0], self._config.physics)
        trajectory = Q_t  # (T,)
        sigma_float = float(jnp.max(noise_scale))
        return time, trajectory, H_t, sigma_float, path

    def is_tipped(
        self,
        trajectory: jnp.ndarray,
        time=None,
        param_values=None,
        **kwargs,
    ):
        """True if a finite Q falls below Q_collapse_threshold."""
        strength = trajectory[:, 0] if trajectory.ndim == 2 else trajectory
        return bool(
            jnp.any(
                jnp.isfinite(strength) & (strength < self._config.Q_collapse_threshold)
            )
        )

    def locate_transition(
        self,
        trajectory: jnp.ndarray,
        null_simulation: bool,
        time=None,
        param_values=None,
        **kwargs,
    ):
        """Transition = when t crosses 0 (H peaks and plateaus). Returns first index where time >= 0. Caller must pass time."""
        if null_simulation:
            return int(trajectory.shape[0] // 2)
        if time is None or time[0] > 0 or time[-1] < 0:
            return None
        time = jnp.asarray(time)
        idx = jnp.searchsorted(time, 0.0, side="left")
        return int(idx)
