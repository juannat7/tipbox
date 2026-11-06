"""Stochastic Ricker map, suitable for fitting population-decline surrogates."""

from dataclasses import dataclass
from math import isfinite

import jax.numpy as jnp
from jax import lax

from ..types import BoxModel
from ..utils import parameter, register_dataclass_pytree


@register_dataclass_pytree
@dataclass(frozen=True)
class RickerConfig:
    """Illustrative population-map parameters; no experimental fit is implied."""

    r_start: float = parameter(0.2, "Initial log growth per map update.", "per update")
    r_end: float = parameter(
        -0.1, "Final forced log growth per map update.", "per update"
    )
    r_null: float = parameter(
        0.2, "Constant log growth in null simulations.", "per update"
    )
    density_dependence: float = parameter(
        0.002, "Density-dependent reduction of log growth.", "1 / population"
    )
    sigma: float = parameter(
        0.2,
        "Standard deviation of log-growth noise before the noise multiplier.",
        "log population",
    )
    initial_population: float = parameter(
        100.0,
        "Positive population used to initialize the log-coordinate state.",
        "population",
    )
    dt: float = parameter(
        1.0, "Time assigned to one map update; not an ODE accuracy step.", "model time"
    )
    t_start: float = parameter(0.0, "First observation time.", "model time")
    t_end: float = parameter(500.0, "End time when n is omitted.", "model time")

    def __post_init__(self):
        if not isfinite(self.density_dependence) or self.density_dependence <= 0:
            raise ValueError("density_dependence must be finite and positive")
        if not isfinite(self.initial_population) or self.initial_population <= 0:
            raise ValueError("initial_population must be finite and positive")
        if not isfinite(self.sigma) or self.sigma < 0:
            raise ValueError("sigma must be finite and nonnegative")


class RickerModel(BoxModel):
    """Ricker population map integrated in log-population coordinates.

    One update is z_next = z + r - b*exp(z) + sigma*noise*innovation,
    with observed population exp(z). dt is the time represented by one update;
    it is not a substep accuracy control. There is no sqrt(dt) noise scaling.
    Initial-state overrides must be log population, matching state_names.
    The default coefficients are illustrative.
    """

    config_class = RickerConfig
    state_names = ("log_population",)
    observable_names = ("population",)
    noise_dimension = 1
    forcing_name = "log_growth"
    integration_method = "Ricker map with Gaussian log-growth noise"
    critical_parameter = 0.0

    def _initial_state(self, key, null_simulation):
        return jnp.array([jnp.log(self._config.initial_population)])

    def _forcing(self, time, null_simulation, param_end_multiplier):
        config = self._config
        if null_simulation:
            return jnp.full_like(time, config.r_null)
        fraction = jnp.clip(
            (time - time[0]) * param_end_multiplier / (time[-1] - time[0]), 0, 1
        )
        return config.r_start + (config.r_end - config.r_start) * fraction

    def _integrate(
        self, time, forcing, dt, noise, state, innovations, midpoint_forcing
    ):
        scale = self._config.sigma * noise
        density = self._config.density_dependence

        def step(z, inputs):
            growth, innovation = inputs
            z = z + growth - density * jnp.exp(z) + scale * innovation
            return z, z

        _, tail = lax.scan(step, state[0], (forcing[:-1], innovations[:, 0]))
        path = jnp.concatenate([state, tail])[:, None]
        return path, scale

    def _observe(self, path):
        return jnp.exp(path)

    def is_tipped(self, trajectory, param_values, time=None):
        """True if log growth reaches or falls below zero."""
        return bool(jnp.any(param_values <= self.critical_parameter))

    def locate_transition(self, trajectory, null_simulation, param_values, time=None):
        """Midpoint for null; first nonpositive log-growth sample for forced paths."""
        if null_simulation:
            return len(param_values) // 2
        crossed = param_values <= self.critical_parameter
        return int(jnp.argmax(crossed)) if bool(jnp.any(crossed)) else None
