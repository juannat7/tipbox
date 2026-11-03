"""
AMOC 2-box model, bifurcation-induced tipping (Ritchie et al. 2023; Alkhayuon et al. 2019).

Port from dews/experiments/amoc_boxmodel.py. State (SN, ST); control H(t) (freshwater hosing).
Saddle-node (fold) at H_fold; ramp H(t) with cap for tipping vs null.
"""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

from .._amoc import AMOCPhysics, drift, observe, rk4_step, strength_sv
from ..types import BoxModel
from ..utils import (
    parameter,
    register_dataclass_pytree,
    sampling_stride,
)

# 2x CO2 parameters from BoxModel_2DH_IVP.m (rtippingreview)


def _forcing_bifurcation_ramp_cap(
    t: jnp.ndarray, H0: float, rate: float, H_cap: float
) -> jnp.ndarray:
    """H(t) = min(H0 + rate*t, H_cap) — ramp then constant at H_cap (same as amoc_boxmodel.forcing_bifurcation_ramp_cap)."""
    return jnp.minimum(H0 + rate * t, H_cap)


def _bifurcation_ramp_cap_params(
    t_end: float,
    H0: float,
    H_fold: float,
    *,
    delta_cap: float = 0.05,
) -> tuple:
    """Rate and caps so H(t) reaches H_fold at t_end (like other box models: crossing at t_total/param_end_multiplier)."""
    rate = (H_fold - H0) / t_end
    H_cap_nontip = H_fold - delta_cap
    H_cap_tip = H_fold + delta_cap
    return rate, H_cap_nontip, H_cap_tip


@register_dataclass_pytree
@dataclass(frozen=True)
class AMOCConfig:
    """Configuration for AMOC 2-box model (bifurcation-induced)."""

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
    H_fold: float = parameter(
        0.4236,
        "Explicit fold marker for the default physical calibration; recalibrate after changing physics.",
        "Sv",
    )
    delta_cap: float = parameter(
        0.05, "Distance from H_fold to each forced/null cap.", "Sv"
    )
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
    t_start: float = parameter(
        0.0,
        "Start time; rate-induced forcing uses this absolute clock origin.",
        "year",
    )
    t_end: float = parameter(
        5200.0,
        "End time when make_trajectory is called without n; explicit n sets the end time.",
        "year",
    )
    dt_inner: float = parameter(0.2, "AMOC RK4 integration step.", "year")

    @classmethod
    def default(cls) -> "AMOCConfig":
        return cls()


class AMOCModel(BoxModel):
    """AMOC 2-box model, bifurcation-induced (H(t) ramp with fold at H_fold)."""

    config_class = AMOCConfig

    @property
    def critical_parameter(self):
        """Configured fold marker; calibration depends on the physical parameters."""
        return self._config.H_fold

    state_names = ("north_salinity_anomaly", "tropical_salinity_anomaly")
    observable_names = ("amoc_strength",) + state_names
    noise_dimension = 2
    integration_step_field = "dt_inner"
    integration_method = "RK4 drift with additive Euler noise"
    uses_midpoint_forcing = True
    forcing_name = "freshwater_hosing"
    time_unit = "year"

    def _initial_state(self, key, null_simulation):
        return self._sample_initial_state(key, null_simulation)

    def _forcing(self, time, null_simulation, param_end_multiplier):
        return self._H_schedule(time, null_simulation, param_end_multiplier)

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

    def _H_schedule(
        self,
        t: jnp.ndarray,
        null_simulation: bool = False,
        param_end_multiplier: float = 1.1,
    ) -> jnp.ndarray:
        """H(t) = forcing_bifurcation_ramp_cap(t, H0, rate, H_cap) — same as dews amoc_boxmodel."""
        H0 = self._config.H0
        H_fold = self._config.H_fold
        delta_cap = self._config.delta_cap
        t_total = t[-1] - t[0]
        t_end_eff = t_total / param_end_multiplier
        rate, H_cap_nontip, H_cap_tip = _bifurcation_ramp_cap_params(
            t_end_eff, H0, H_fold, delta_cap=delta_cap
        )
        H_cap = H_cap_nontip if null_simulation else H_cap_tip
        t_rel = t - t[0]  # time from 0 for forcing_bifurcation_ramp_cap
        return _forcing_bifurcation_ramp_cap(t_rel, H0, rate, H_cap)

    def _sample_initial_state(
        self,
        key: jax.random.PRNGKey,
        null_simulation: bool = False,
    ) -> jnp.ndarray:
        """Gaussian anomalies about the configured mean; negative anomalies are valid."""
        key, _ = jax.random.split(key)
        noise = jax.random.normal(key, (2,))
        x0 = (
            jnp.array([self._config.initial_SN, self._config.initial_ST])
            + self._config.sigma_ic * noise
        )
        return x0

    def _drift(self, state: jnp.ndarray, H: jnp.ndarray) -> jnp.ndarray:
        return drift(state, H, self._config.physics)

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
        sigma_tilde: float = 1.0,
        null_simulation: bool = False,
        dt_sample: Optional[float] = None,
        param_end_multiplier: float = 1.1,
        initial_state: Optional[jnp.ndarray] = None,
        noise: Optional[jnp.ndarray] = None,
    ):
        key_noise, _ = jax.random.split(key)
        total_time = self._config.t_end
        dt_inner = self._config.dt_inner
        dt_out = self._config.dt if dt_sample is None else dt_sample
        sample_every = sampling_stride(dt_out, dt_inner)
        n_steps = sampling_stride(total_time - self._config.t_start, dt_inner)
        # Single inner grid: RK4 for both branches (stochastic adds noise per step)
        time_inner = jnp.linspace(self._config.t_start, total_time, n_steps + 1)
        H_inner = self._H_schedule(
            time_inner,
            null_simulation=null_simulation,
            param_end_multiplier=param_end_multiplier,
        )
        H_cur = H_inner[:-1]
        H_next = H_inner[1:]
        H_mid = (H_cur + H_next) / 2.0

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
        trajectory_fine = self._simulate_path(
            H_cur, H_mid, H_next, dt_inner, initial_state, noise_scale, noise_arr
        )

        n_inner = trajectory_fine.shape[0]
        indices = jnp.arange(0, n_inner, sample_every)
        time = time_inner[indices]
        path_sampled = trajectory_fine[indices]

        param_values = H_inner[indices]

        # Output trajectory with both physical state (S_N, S_T) and AMOC strength Q(t).
        # Column convention:
        #   trajectory[:, 0] = Q(t)  (AMOC strength in Sv, for 1D EWS)
        #   trajectory[:, 1] = S_N(t)  (North Atlantic salinity)
        #   trajectory[:, 2] = S_T(t)  (Tropical Atlantic salinity)
        SN_sampled = path_sampled[:, 0]
        ST_sampled = path_sampled[:, 1]
        Q_t = strength_sv(SN_sampled, self._config.physics)
        trajectory = jnp.stack([Q_t, SN_sampled, ST_sampled], axis=1)

        sigma_float = float(jnp.max(noise_scale))
        return time, trajectory, param_values, sigma_float

    def is_tipped(
        self,
        trajectory: jnp.ndarray,
        param_values: jnp.ndarray,
    ):
        """True if H crossed the bifurcation point (max H >= H_fold)."""
        return float(jnp.max(param_values)) >= self._config.H_fold

    def locate_transition(
        self,
        trajectory: jnp.ndarray,
        null_simulation: bool,
        param_values: jnp.ndarray,
        param_end_multiplier: float = 1.1,
    ):
        if null_simulation:
            return len(param_values) // 2
        crossed = param_values >= self._config.H_fold
        if bool(jnp.any(crossed)):
            return int(jnp.argmax(crossed))
        return len(param_values) - 1
