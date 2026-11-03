"""Shared model interface and trajectory container."""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
from math import isfinite
from numbers import Integral
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from .utils import sampling_stride


@dataclass(frozen=True)
class TrajectoryResult:
    """Time, observed state, forcing, per-step noise scale, and provenance.

    ``make_trajectory`` always returns observations shaped ``(n_time, n_observables)``.
    The historical ``generate`` interface retains each model's original shape.
    ``sigma`` is the standard deviation of an integration-step noise increment,
    including the square root of the integration step.
    """

    time: jax.Array
    trajectory: jax.Array
    param_values: jax.Array
    sigma: float | jax.Array
    metadata: dict[str, Any] | None


class BoxModel(ABC):
    """A configured box model with an exact-grid trajectory interface.

    Parameters
    ----------
    config : dataclass or None
        Typed configuration for this model. Omit to use published defaults.
    **parameters
        Configuration field overrides, checked by the typed dataclass constructor.
        ``model.config`` and ``model.get_parameters()`` expose all resolved values.
    """

    config_class: type
    state_names: tuple[str, ...]
    observable_names: tuple[str, ...]
    noise_dimension: int
    default_sample_dt: float | None = None
    integration_step_field: str = "dt"
    integration_method: str = "Euler-Maruyama"
    time_unit: str = "model time"
    forcing_name: str
    critical_parameter = None
    uses_midpoint_forcing = False

    def __init__(self, config=None, **parameters):
        config = self.config_class() if config is None else config
        if not isinstance(config, self.config_class):
            raise TypeError(
                f"{type(self).__name__} requires {self.config_class.__name__}"
            )
        self._config = replace(config, **parameters)

    @property
    def config(self):
        """Immutable, typed model configuration."""
        return self._config

    def get_parameters(self) -> dict:
        """Resolved configuration, including nested physical parameters."""
        return asdict(self._config)

    def make_trajectory(
        self,
        n: int | None = None,
        *,
        dt: float | None = None,
        dt_sample: float | None = None,
        noise: float = 0.01,
        initial_state=None,
        seed: int = 0,
        innovations=None,
        null_simulation: bool = False,
        param_end_multiplier: float = 1.1,
        forcing: Callable | None = None,
    ) -> TrajectoryResult:
        """Integrate exactly ``n`` observations from the configured start time.

        Parameters
        ----------
        n : int or None, default=None
            Number of returned samples, including the initial state; at least 2.
            None uses the configured start/end interval and output spacing.
            An explicit n overrides the configured end time.
        dt : float or None, default=None
            Integration step. Defaults to config.dt, or config.dt_inner for AMOC.
        dt_sample : float or None, default=None
            Output spacing; a positive integer multiple of dt. Defaults to 1.0
            for ecological bifurcation models and config.dt for rate/AMOC models.
        noise : float, default=0.01
            Nonnegative noise-strength parameter, with the same meaning as
            sigma_tilde in simulate. The model reference gives its exact diffusion
            formula; saddle-node uses an intensity and other models a multiplier.
            Zero disables process noise, but does not disable random initial states.
        initial_state : array-like or None, default=None
            Physical initial state in the order of model.state_names. None uses
            the model's initial-state sampler. A scalar is accepted for 1D models.
        seed : int, default=0
            Local random seed; global random state is never changed.
        innovations : array-like or None, default=None
            Standard-normal increments shaped (n_integration_steps, noise_dimension).
            These replace random increments and are multiplied by the model's
            per-step scale. The physical and observed dimensions can differ.
        null_simulation : bool, default=False
            Select the model's null forcing schedule. This is a forcing choice,
            not a guarantee that the simulated state remains non-tipping.
        param_end_multiplier : float, default=1.1
            For duration-normalized bifurcation ramps, the ratio of total duration
            to ramp duration. Rate schedules retain their absolute time origin.
        forcing : callable or None, default=None
            Optional forcing(t) evaluated on the complete integration grid. It must
            return one finite scalar per time point, including RK4 midpoints,
            in the model's forcing units.

        Returns
        -------
        TrajectoryResult
            Observations, time, forcing and per-step noise scale. Metadata includes
            the resolved parameters and sampled physical state trajectory.

        Notes
        -----
        Increasing n extends the time interval at fixed output spacing. The
        ecological bifurcation schedules normalize their ramp to that duration,
        so trajectories with different n are not generally prefixes of one another.
        Analytic critical parameters follow the model coefficients. AMOC uses
        an explicitly calibrated marker that must be reconsidered after changing
        its physical parameters. This method validates concrete arrays eagerly;
        use the low-level JAX kernels or legacy simulate method for autodiff.
        """
        critical_parameter = self.critical_parameter
        dt = getattr(self._config, self.integration_step_field) if dt is None else dt
        if dt_sample is None:
            dt_sample = (
                self._config.dt
                if self.default_sample_dt is None
                else self.default_sample_dt
            )
        stride = sampling_stride(dt_sample, dt)
        if n is None:
            n = (
                sampling_stride(self._config.t_end - self._config.t_start, dt_sample)
                + 1
            )
        if not isinstance(n, Integral) or isinstance(n, bool) or n < 2:
            raise ValueError("n must be an integer of at least 2")
        if not isfinite(noise) or noise < 0:
            raise ValueError("noise must be finite and nonnegative")
        if not isfinite(param_end_multiplier) or param_end_multiplier <= 0:
            raise ValueError("param_end_multiplier must be finite and positive")

        steps = (n - 1) * stride
        time = self._config.t_start + jnp.arange(steps + 1) * dt
        key_noise, key_ic = jax.random.split(jax.random.PRNGKey(seed))
        state = (
            self._initial_state(key_ic, null_simulation)
            if initial_state is None
            else initial_state
        )
        state = jnp.atleast_1d(jnp.asarray(state, dtype=float))
        if state.shape != (len(self.state_names),) or not np.isfinite(state).all():
            raise ValueError(
                f"initial_state must be finite and shaped {(len(self.state_names),)}"
            )
        increments = (
            jax.random.normal(key_noise, (steps, self.noise_dimension))
            if innovations is None
            else jnp.asarray(innovations)
        )
        if (
            increments.shape != (steps, self.noise_dimension)
            or not np.isfinite(increments).all()
        ):
            raise ValueError(
                f"innovations must be finite and shaped {(steps, self.noise_dimension)}"
            )
        control_time = (
            self._config.t_start + jnp.arange(2 * steps + 1) * (dt / 2)
            if self.uses_midpoint_forcing
            else time
        )
        control = (
            self._forcing(control_time, null_simulation, param_end_multiplier)
            if forcing is None
            else jnp.asarray(forcing(control_time))
        )
        if control.shape != control_time.shape or not np.isfinite(control).all():
            raise ValueError(
                "forcing must return one finite scalar per integration time"
            )
        midpoint_control = None
        if self.uses_midpoint_forcing:
            midpoint_control = control[1::2]
            control = control[::2]

        path, scale = self._integrate(
            time, control, dt, noise, state, increments, midpoint_control
        )
        sampled_path = path[::stride]
        observed = self._observe(sampled_path)
        metadata = {
            "model": type(self).__name__,
            "system_config": self.get_parameters(),
            "seed": int(seed),
            "noise": noise,
            "dt": dt,
            "dt_sample": dt_sample,
            "initial_state": state,
            "null_simulation": null_simulation,
            "param_end_multiplier": param_end_multiplier,
            "state_names": self.state_names,
            "observable_names": self.observable_names,
            "forcing_name": self.forcing_name,
            "time_unit": self.time_unit,
            "integration_method": self.integration_method,
            "state_trajectory": sampled_path,
            "custom_forcing": forcing is not None,
            "custom_innovations": innovations is not None,
            "critical_parameter": critical_parameter,
        }
        return TrajectoryResult(
            time[::stride], observed, control[::stride], scale, metadata
        )

    def _observe(self, path):
        return path

    @abstractmethod
    def _initial_state(self, key, null_simulation):
        """Sample a physical initial state."""

    @abstractmethod
    def _forcing(self, time, null_simulation, param_end_multiplier):
        """Evaluate the model's forcing on the integration grid."""

    @abstractmethod
    def _integrate(
        self, time, forcing, dt, noise, state, innovations, midpoint_forcing
    ):
        """Use the model's integration kernel; return physical path and noise scale."""

    @abstractmethod
    def simulate(self, key, total_len, **kwargs):
        """Historical model-specific simulator used by generate."""

    @abstractmethod
    def is_tipped(self, trajectory, **kwargs):
        """Classify using the model's configured transition marker."""

    @abstractmethod
    def locate_transition(self, trajectory, **kwargs):
        """Locate the model's transition/reference marker."""

    def extract_segment(
        self,
        time,
        trajectory,
        param_values,
        transition_idx,
        steps_before_transition,
        series_len,
    ):
        """Extract context before a full-trajectory marker."""
        from .utils import extract_segment

        return extract_segment(
            time,
            trajectory,
            param_values,
            transition_idx,
            steps_before_transition,
            series_len,
        )
