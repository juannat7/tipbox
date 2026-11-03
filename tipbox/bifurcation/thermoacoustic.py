"""Subcritical thermoacoustic oscillator of Bonciolini et al. (2018), Eq. 3.7."""

from dataclasses import dataclass
from math import isfinite

import jax.numpy as jnp
from jax import lax

from ..types import BoxModel
from ..utils import parameter, register_dataclass_pytree


@register_dataclass_pytree
@dataclass(frozen=True)
class ThermoacousticConfig:
    """Published numerical-example coefficients, with an explicit ramp and IC."""

    frequency: float = parameter(120.0, "Acoustic eigenfrequency, omega0/(2*pi).", "Hz")
    kappa: float = parameter(
        8.0,
        "Destabilizing quadratic nonlinear damping coefficient.",
        "1 / (s * pressure^2)",
    )
    mu: float = parameter(
        2.0,
        "Stabilizing quartic nonlinear damping coefficient.",
        "1 / (s * pressure^4)",
    )
    Gamma: float = parameter(
        1e6,
        "White acceleration-noise intensity before noise**2 scaling.",
        "pressure^2 / s^3",
    )
    nu_start: float = parameter(-4.5, "Initial linear growth rate.", "1 / s")
    nu_end: float = parameter(5.5, "Final forced linear growth rate.", "1 / s")
    nu_null: float = parameter(
        -4.5, "Constant linear growth rate for null simulations.", "1 / s"
    )
    initial_pressure: float = parameter(
        0.1,
        "Initial pressure; an illustrative fixed IC, not a stationary draw.",
        "model pressure",
    )
    initial_scaled_velocity: float = parameter(
        0.0, "Initial pressure velocity divided by omega0.", "model pressure"
    )
    dt: float = parameter(1e-4, "RK4 drift integration step.", "s")
    t_start: float = parameter(0.0, "First observation time.", "s")
    t_end: float = parameter(
        1.1, "End time when n is omitted; default ramp reaches its cap at 1 s.", "s"
    )

    def __post_init__(self):
        for name in ("frequency", "kappa", "mu"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isfinite(self.Gamma) or self.Gamma < 0:
            raise ValueError("Gamma must be finite and nonnegative")


class ThermoacousticModel(BoxModel):
    """Noisy nonlinear pressure oscillator with a subcritical Hopf at nu=0.

    The physical state is [p, y], with y=dp/dt/omega0. Observations are
    [p, y, sqrt(p**2+y**2)]. The final column is an instantaneous phase-plane
    amplitude, rather than a Hilbert-envelope estimate from a pressure record.
    RK4 drift uses true midpoint forcing; additive noise is applied to y at the
    end of each integration step. noise=1 recovers the published Gamma intensity.
    """

    config_class = ThermoacousticConfig
    state_names = ("pressure", "scaled_velocity")
    observable_names = (*state_names, "amplitude")
    noise_dimension = 1
    forcing_name = "linear_growth_rate"
    integration_method = "RK4 drift with additive velocity-noise increments"
    default_sample_dt = 0.0005
    uses_midpoint_forcing = True
    time_unit = "s"
    critical_parameter = 0.0

    @property
    def fold_parameter(self):
        """Limit-cycle fold of the deterministic averaged amplitude equation."""
        return -(self._config.kappa**2) / (16 * self._config.mu)

    def _initial_state(self, key, null_simulation):
        return jnp.array(
            [self._config.initial_pressure, self._config.initial_scaled_velocity]
        )

    def _forcing(self, time, null_simulation, param_end_multiplier):
        config = self._config
        if null_simulation:
            return jnp.full_like(time, config.nu_null)
        fraction = jnp.clip(
            (time - time[0]) * param_end_multiplier / (time[-1] - time[0]), 0, 1
        )
        return config.nu_start + (config.nu_end - config.nu_start) * fraction

    def _integrate(
        self, time, forcing, dt, noise, state, innovations, midpoint_forcing
    ):
        config = self._config
        omega = 2 * jnp.pi * config.frequency
        scale = noise * jnp.sqrt(config.Gamma * dt) / omega

        def drift(values, nu):
            pressure, velocity = values
            damping = 2 * nu + config.kappa * pressure**2 - config.mu * pressure**4
            return jnp.array([omega * velocity, -omega * pressure + damping * velocity])

        def step(values, inputs):
            nu0, numid, nu1, innovation = inputs
            k1 = drift(values, nu0)
            k2 = drift(values + dt * k1 / 2, numid)
            k3 = drift(values + dt * k2 / 2, numid)
            k4 = drift(values + dt * k3, nu1)
            values = values + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            values = values.at[1].add(scale * innovation)
            return values, values

        _, tail = lax.scan(
            step,
            state,
            (forcing[:-1], midpoint_forcing, forcing[1:], innovations[:, 0]),
        )
        return jnp.concatenate([state[None, :], tail]), scale

    def _observe(self, path):
        amplitude = jnp.sqrt((path**2).sum(axis=1))
        return jnp.column_stack([path, amplitude])

    def is_tipped(self, trajectory, param_values, time=None):
        """True if the growth-rate forcing reaches the Hopf point, nu=0."""
        return bool(jnp.any(param_values >= self.critical_parameter))

    def locate_transition(self, trajectory, null_simulation, param_values, time=None):
        """Midpoint for null; first nonnegative growth-rate sample for forced paths."""
        if null_simulation:
            return len(param_values) // 2
        crossed = param_values >= self.critical_parameter
        return int(jnp.argmax(crossed)) if bool(jnp.any(crossed)) else None
