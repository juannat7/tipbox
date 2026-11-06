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
class RosenzweigMacArthurConfig:
    """Consumer-resource coefficients and forcing controls (Bury et al., 2021)."""

    # Model parameters (from Bury et al. 2021)
    r: float = parameter(4.0, "Intrinsic resource growth rate.", "1 / model time")
    k: float = parameter(1.7, "Resource carrying capacity.", "resource")
    e: float = parameter(
        0.5,
        "Conversion efficiency from resource consumption to consumer growth.",
        "consumer / resource",
    )
    h: float = parameter(
        0.15, "Handling time in the Holling type-II response.", "model time"
    )
    m: float = parameter(2.0, "Consumer mortality rate.", "1 / model time")

    # Noise parameters
    sigma1: float = parameter(
        0.01,
        "Resource diffusion amplitude before the noise multiplier.",
        "resource / sqrt(model time)",
    )
    sigma2: float = parameter(
        0.01,
        "Consumer diffusion amplitude before the noise multiplier.",
        "consumer / sqrt(model time)",
    )

    # Bifurcation type: "transcritical" or "hopf"
    bifurcation_type: str = parameter(
        "transcritical", "Selected mechanism: transcritical or hopf.", ""
    )

    # Bifurcation parameter ranges (attack rate a)
    # Transcritical: forced a spans [2,6]; the null cap and marker are automatic.
    a_transcritical_null: float | None = parameter(
        None,
        "Optional null cap in transcritical mode; None uses max(a_min, 2*a_critical-a_max).",
        "attack rate",
    )
    a_transcritical_min: float = parameter(
        2.0, "Initial attack rate in transcritical mode.", "attack rate"
    )
    a_transcritical_max: float = parameter(
        6.0, "Forced ramp endpoint in transcritical mode.", "attack rate"
    )
    a_transcritical_bifurcation: float | None = parameter(
        None,
        "Optional marker in transcritical mode; None computes m/[k(e-mh)].",
        "attack rate",
    )

    # Hopf: forced a spans [12,17]; the null cap and marker are automatic.
    a_hopf_null: float | None = parameter(
        None,
        "Optional null cap in Hopf mode; None uses max(a_min, 2*a_critical-a_max).",
        "attack rate",
    )
    a_hopf_min: float = parameter(
        12.0, "Initial attack rate in Hopf mode.", "attack rate"
    )
    a_hopf_max: float = parameter(
        17.0, "Forced ramp endpoint in Hopf mode.", "attack rate"
    )
    a_hopf_bifurcation: float | None = parameter(
        None,
        "Optional marker in Hopf mode; None computes (e+mh)/[hk(e-mh)].",
        "attack rate",
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
        "End time when make_trajectory is called without num_samples; explicit num_samples sets the end time.",
        "model time",
    )

    def __post_init__(self):
        if self.bifurcation_type not in ("transcritical", "hopf"):
            raise ValueError("bifurcation_type must be transcritical or hopf")

    @classmethod
    def default(
        cls, bifurcation_type: str = "transcritical"
    ) -> "RosenzweigMacArthurConfig":
        """Return default configuration for specified bifurcation type."""
        return cls(bifurcation_type=bifurcation_type)

    @classmethod
    def transcritical(cls) -> "RosenzweigMacArthurConfig":
        """Return configuration for transcritical bifurcation."""
        return cls(bifurcation_type="transcritical")

    @classmethod
    def hopf(cls) -> "RosenzweigMacArthurConfig":
        """Return configuration for Hopf bifurcation."""
        return cls(bifurcation_type="hopf")


class RosenzweigMacArthurModel(BoxModel):
    """Holling type-II consumer-resource model with transcritical or Hopf forcing."""

    config_class = RosenzweigMacArthurConfig

    @property
    def critical_parameter(self):
        """Analytic invasion/Hopf threshold, or a configured marker override."""
        c = self._config
        override = getattr(c, f"a_{c.bifurcation_type}_bifurcation")
        if override is not None:
            return override
        if min(c.k, c.e, c.h, c.m) <= 0 or c.e <= c.m * c.h:
            raise ValueError(
                "Analytic consumer bifurcations require k,e,h,m > 0 and e > m*h"
            )
        if c.bifurcation_type == "transcritical":
            return c.m / (c.k * (c.e - c.m * c.h))
        return (c.e + c.m * c.h) / (c.h * c.k * (c.e - c.m * c.h))

    state_names = ("resource", "consumer")
    observable_names = state_names
    noise_dimension = 2
    default_sample_dt = 1.0
    forcing_name = "attack_rate"

    def _initial_state(self, key, null_simulation):
        return self._sample_initial_state(key, null_simulation)

    def _forcing(self, time, null_simulation, param_end_multiplier):
        return self._a_schedule(time, null_simulation, param_end_multiplier)

    def _integrate(
        self, time, forcing, dt, noise, state, innovations, midpoint_forcing
    ):
        scale = (
            jnp.array([self._config.sigma1, self._config.sigma2]) * noise * jnp.sqrt(dt)
        )
        return self._simulate_path(
            None, forcing, dt, scale, state, noise=innovations
        ), scale

    @property
    def bifurcation_type(self) -> str:
        """Get the bifurcation type being simulated."""
        return self._config.bifurcation_type

    def _get_param_range(self) -> tuple:
        """Null cap, ramp bounds and threshold for the selected mechanism."""
        prefix = f"a_{self.bifurcation_type}_"
        return (
            getattr(self._config, prefix + "null"),
            getattr(self._config, prefix + "min"),
            getattr(self._config, prefix + "max"),
            self.critical_parameter,
        )

    def _a_schedule(
        self,
        t: jnp.ndarray,
        null_simulation: bool = False,
        param_end_multiplier: float = 1.1,
    ):
        """Ramp attack rate, with an explicit or symmetric null cap."""
        a_null, a_min, a_max, a_bif = self._get_param_range()
        t_total = t[-1] - t[0]
        t_to_bifurcation = t_total / param_end_multiplier
        t_rel = t - t[0]
        ramp = a_min + (t_rel / t_to_bifurcation) * (a_max - a_min)

        if null_simulation:
            range_past = a_max - a_bif
            cap = a_bif - range_past
            cap = jnp.maximum(cap, a_min) if a_null is None else a_null
            return jnp.minimum(ramp, cap)
        a_values = jnp.where(t_rel <= t_to_bifurcation, ramp, a_max)
        return a_values

    def _compute_equilibrium(self, a: float) -> tuple:
        """Consumer-free equilibrium below invasion, coexistence equilibrium above it."""
        r, k, e, h, m = (
            self._config.r,
            self._config.k,
            self._config.e,
            self._config.h,
            self._config.m,
        )

        # Transcritical threshold
        a_tc = self._config.m / (
            self._config.k * (self._config.e - self._config.m * self._config.h)
        )

        if a < a_tc:
            # Consumer-free equilibrium: x = k, y = 0
            return k, 0.0
        # Coexistence equilibrium
        # From dy/dt = 0: eax/(1+ahx) = m
        # Solving for x: x* = m / (a(e - mh))
        x_star = m / (a * (e - m * h))

        # From dx/dt = 0: r(1 - x/k) = ay/(1+ahx)
        # Solving for y: y* = r(1 - x*/k)(1 + ahx*) / a
        functional_response_denom = 1.0 + a * h * x_star
        y_star = r * (1.0 - x_star / k) * functional_response_denom / a

        return max(x_star, 0.01), max(y_star, 0.01)

    def _sample_initial_state(
        self,
        key: jax.random.PRNGKey,
        null_simulation: bool = False,
    ):
        """Sample near the initial equilibrium; keep a positive invasion seed."""
        a_null, a_min, a_max, a_bif = self._get_param_range()

        key1, key2 = jax.random.split(key)

        if self._config.bifurcation_type == "transcritical":
            # For transcritical: start near carrying capacity for x
            # and with a small but significant consumer population
            x = float(
                jax.random.uniform(key1, (), minval=0.8, maxval=1.0) * self._config.k
            )

            # Consumer needs enough initial population to survive until a > bifurcation point
            # Start with ~5% of resource as consumer
            y = float(jax.random.uniform(key2, (), minval=0.02, maxval=0.08))
        else:
            # For Hopf: use coexistence equilibrium at ramp start (a_min; null now ramps from a_min too)
            a_init = a_min
            x_eq, y_eq = self._compute_equilibrium(a_init)

            # Add small noise around equilibrium
            x = float(jax.random.normal(key1, ()) * 0.05 + x_eq)
            y = float(jax.random.normal(key2, ()) * 0.05 + y_eq)

            x = max(x, 0.1)
            y = max(y, 0.01)

        return jnp.array([x, y])

    def _drift(self, state: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """Deterministic drift in the model's physical state coordinates."""
        x, y = state[0], state[1]
        r, k, e, h, m = (
            self._config.r,
            self._config.k,
            self._config.e,
            self._config.h,
            self._config.m,
        )

        # Functional response (Type II): axy / (1 + ahx)
        functional_response = a * x * y / (1.0 + a * h * x)

        # Resource dynamics: logistic growth - predation
        dx = r * x * (1.0 - x / k) - functional_response

        # Consumer dynamics: predation gain - mortality
        dy = e * functional_response - m * y

        return jnp.array([dx, dy])

    def _simulate_path(
        self,
        key: jax.random.PRNGKey,
        a_t: jnp.ndarray,
        dt: float,
        noise_scale: jnp.ndarray,
        initial_state: jnp.ndarray,
        noise: Optional[jnp.ndarray] = None,
    ):
        """Stochastic integration using Euler-Maruyama.

        dx = drift_x * dt + σ₁ * dW₁
        dy = drift_y * dt + σ₂ * dW₂
        """
        n_steps = a_t.shape[0] - 1

        if noise is None:
            noise = jax.random.normal(key, shape=(n_steps, 2))
        else:
            noise = jnp.asarray(noise)

        def step(state, inputs):
            a_val, noise_t = inputs

            drift = self._drift(state, a_val)

            state_next = state + drift * dt + noise_scale * noise_t

            # Ensure non-negative populations
            state_next = jnp.maximum(state_next, 0.0)

            return state_next, state_next

        _, states = lax.scan(step, initial_state, (a_t[:-1], noise))

        trajectory = jnp.concatenate([initial_state[None, :], states], axis=0)

        return trajectory

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
        """simulator: returns time, [resource, consumer], attack rate and step-noise scale.

        total_len counts output intervals; the result includes the initial sample."""
        key_noise, key_ic = jax.random.split(key)

        sample_every = sampling_stride(dt_sample, self._config.dt)
        total_time = self._config.t_start + total_len * dt_sample
        n_steps = total_len * sample_every + 1
        time_fine = jnp.linspace(self._config.t_start, total_time, n_steps)

        a_t = self._a_schedule(
            time_fine,
            null_simulation=null_simulation,
            param_end_multiplier=param_end_multiplier,
        )

        # Sample initial state if not provided
        if initial_state is None:
            initial_state = self._sample_initial_state(
                key_ic, null_simulation=null_simulation
            )
        else:
            initial_state = jnp.asarray(initial_state)

        noise_scale = jnp.array(
            [
                self._config.sigma1 * sigma_tilde * jnp.sqrt(self._config.dt),
                self._config.sigma2 * sigma_tilde * jnp.sqrt(self._config.dt),
            ]
        )

        trajectory_fine = self._simulate_path(
            key_noise,
            a_t,
            self._config.dt,
            noise_scale,
            initial_state,
            noise=noise,
        )

        indices = jnp.arange(0, n_steps, sample_every)

        time = time_fine[indices]
        trajectory = trajectory_fine[indices]
        param_values = a_t[indices]

        return time, trajectory, param_values, noise_scale

    def is_tipped(
        self,
        trajectory: jnp.ndarray,
        param_values: jnp.ndarray,
    ):
        """Check if the parameter crossed the bifurcation point."""
        _, _, _, a_bif = self._get_param_range()
        return float(jnp.max(param_values)) >= a_bif

    def locate_transition(
        self,
        trajectory: jnp.ndarray,
        null_simulation: bool,
        param_values: jnp.ndarray,
        param_end_multiplier: float = 1.1,
        time=None,
    ):
        """Midpoint for null; first bifurcation crossing for forced, or None."""
        if null_simulation:
            return len(param_values) // 2

        _, _, _, a_bif = self._get_param_range()

        crossed = param_values >= a_bif

        if bool(jnp.any(crossed)):
            return int(jnp.argmax(crossed))

        return None
