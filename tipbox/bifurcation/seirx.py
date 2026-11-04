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
class SEIRxConfig:
    """Configuration for SEIRx behaviour-disease model.

    Based on Bury et al. (PNAS, 2021):
    dS/dt = μN(1-x) - βSI/N - μS + σ₁ξ₁(t)
    dE/dt = βSI/N - (ε+μ)E + σ₂ξ₂(t)
    dI/dt = εE - (gamma+μ)I + σ₃ξ₃(t)
    dx/dt = κx(1-x)(-ω + I + δ(2x-1)) + σ₅ξ₅(t)

    where:
    - S: susceptible population
    - E: exposed population
    - I: infectious population
    - R: recovered population (R = N - S - E - I)
    - x: fraction with provaccine opinion
    - N: total population (constant)
    - μ: birth/death rate
    - β: transmission rate
    - ε: exposed to infectious rate
    - gamma: recovery rate
    - κ: social learning rate
    - δ: strength of injunctive social norms
    - ω: perceived risk of vaccination (bifurcation parameter)
    - σ₁₋₄, σ₅: noise amplitudes
    - ξ(t): independent Gaussian white noise processes

    Bifurcation point (for default parameters):
    - Transcritical bifurcation at ω = δ = 50

    Default time unit is weeks: daily rates from Bury et al. are multiplied by 7
    (e.g. β = 1.5/day → 10.5/week). δ=50, ω∈[0,100], σ₁₋₃=5, σ₅=5x10⁻⁴,
    initial (S,E,I,x)=(0,0,0,1). See paper repo test_models/seirx_*.
    """

    # Model parameters (Bury et al. 2021; time in weeks, daily rates × 7)
    N: float = parameter(
        100000.0,
        "Total population including the unobserved recovered compartment.",
        "individuals",
    )
    mu: float = parameter(0.02 / 52.0, "Birth and death rate.", "1 / week")
    beta: float = parameter(
        1.5 * 7,
        "Transmission rate in frequency-dependent incidence beta*S*I/N.",
        "1 / week",
    )
    epsilon: float = parameter(
        0.1 * 7, "Progression rate from exposed to infectious.", "1 / week"
    )
    gamma: float = parameter(0.1 * 7, "Recovery rate.", "1 / week")
    kappa: float = parameter(
        0.001 * 7, "Social-learning rate.", "model payoff scaling / week"
    )
    delta: float = parameter(
        50.0,
        "Injunctive social-norm strength; the disease-free behavioural threshold is omega=delta.",
        "payoff units",
    )

    # Noise (Bury: σ₁₋₃ = 5, σ₅ = 5×10⁻⁴)
    sigma_S: float = parameter(
        5.0,
        "Susceptible diffusion amplitude before the noise multiplier.",
        "individuals / sqrt(week)",
    )
    sigma_E: float = parameter(
        5.0,
        "Exposed diffusion amplitude before the noise multiplier.",
        "individuals / sqrt(week)",
    )
    sigma_I: float = parameter(
        5.0,
        "Infectious diffusion amplitude before the noise multiplier.",
        "individuals / sqrt(week)",
    )
    sigma_x: float = parameter(
        0.0005,
        "Vaccinator-fraction diffusion amplitude before the noise multiplier.",
        "1 / sqrt(week)",
    )

    # Bifurcation parameter ranges (ω: perceived vaccination risk)
    # Transcritical bifurcation at ω = δ; keep omega_bifurcation equal to delta.
    omega_null: float | None = parameter(
        None,
        "Optional null cap; None uses max(omega_min, 2*omega_critical-omega_max).",
        "payoff units",
    )
    omega_min: float = parameter(
        0.0, "Initial perceived vaccination risk.", "payoff units"
    )
    omega_max: float = parameter(100.0, "Forced ramp endpoint and cap.", "payoff units")
    omega_bifurcation: float | None = parameter(
        None, "Optional behavioural marker; None uses delta.", "payoff units"
    )

    # Simulation parameters (time in weeks)
    dt: float = parameter(
        0.01,
        "Default integration step; AMOC uses this field for output spacing.",
        "week",
    )
    t_start: float = parameter(
        0.0, "Start time; rate-induced forcing uses this absolute clock origin.", "week"
    )
    t_end: float = parameter(
        100.0,
        "End time when make_trajectory is called without n; explicit n sets the end time.",
        "week",
    )

    @classmethod
    def default(cls) -> "SEIRxConfig":
        """Return default configuration."""
        return cls()


class SEIRxModel(BoxModel):
    """BoxModel abstraction for SEIRx behaviour-disease model.

    Implements the stochastic SEIR model with vaccination behaviour:
    dS/dt = μN(1-x) - βSI/N - μS + σ₁ξ₁(t)
    dE/dt = βSI/N - (ε+μ)E + σ₂ξ₂(t)
    dI/dt = εE - (gamma+μ)I + σ₃ξ₃(t)
    dx/dt = κx(1-x)(-ω + I + δ(2x-1)) + σ₅ξ₅(t)

    where ω (perceived vaccination risk) is the bifurcation parameter.

    This model exhibits:
    - Transcritical bifurcation at ω = δ (vaccine coverage collapse).
    - I (infectious count) lags after ω crosses δ because x moves slowly (κ small)
      and I only rises once x drops and susceptibles accumulate.
    """

    config_class = SEIRxConfig

    @property
    def critical_parameter(self):
        """Disease-free behavioural threshold omega=delta, or an override."""
        value = self._config.omega_bifurcation
        return self._config.delta if value is None else value

    state_names = ("susceptible", "exposed", "infectious", "vaccinator_fraction")
    observable_names = ("infectious", "vaccinator_fraction")
    noise_dimension = 4
    default_sample_dt = 1.0
    time_unit = "week"
    forcing_name = "vaccination_risk"

    def _initial_state(self, key, null_simulation):
        return self._sample_initial_state(key, null_simulation)

    def _forcing(self, time, null_simulation, param_end_multiplier):
        return self._omega_schedule(time, null_simulation, param_end_multiplier)

    def _integrate(
        self, time, forcing, dt, noise, state, innovations, midpoint_forcing
    ):
        c = self._config
        scale = (
            jnp.array([c.sigma_S, c.sigma_E, c.sigma_I, c.sigma_x])
            * noise
            * jnp.sqrt(dt)
        )
        return self._simulate_path(
            None, forcing, dt, scale, state, noise=innovations
        ), scale

    def _observe(self, path):
        return path[:, 2:]

    def _get_param_range(self) -> tuple:
        """Get the bifurcation parameter range (ω)."""
        return (
            self._config.omega_null,
            self._config.omega_min,
            self._config.omega_max,
            self.critical_parameter,
        )

    def _omega_schedule(
        self,
        t: jnp.ndarray,
        null_simulation: bool = False,
        param_end_multiplier: float = 1.1,
    ) -> jnp.ndarray:
        """Compute the ω(t) schedule (perceived vaccination risk).

        For null simulations: same linear ramp as forced but clipped so the range from
        null_cap to critical equals the range from critical to omega_max (symmetric about critical).
        For forced simulations:
          - ω increases linearly from omega_min to omega_max
          - param_end_multiplier extends the time series beyond the bifurcation point
        """
        omega_null, omega_min, omega_max, omega_bif = self._get_param_range()
        t_total = t[-1] - t[0]
        t_to_bifurcation = t_total / param_end_multiplier
        t_rel = t - t[0]
        ramp = omega_min + (t_rel / t_to_bifurcation) * (omega_max - omega_min)

        if null_simulation:
            range_past = omega_max - omega_bif
            cap = omega_bif - range_past
            cap = jnp.maximum(cap, omega_min) if omega_null is None else omega_null
            return jnp.minimum(ramp, cap)
        else:
            omega_values = jnp.where(
                t_rel <= t_to_bifurcation,
                ramp,
                omega_max,
            )
            return omega_values

    def _sample_initial_state(
        self,
        key: jax.random.PRNGKey,
        null_simulation: bool = False,
    ) -> jnp.ndarray:
        """Sample an initial state (S, E, I, x).

        Start at provaccine, disease-free equilibrium: S = 0, E = 0, I = 0, x = 1 (R = N).
        """
        S = 0.0
        E = 0.0
        I = 0.0
        x = 1.0
        return jnp.array([S, E, I, x])

    def _drift(self, state: jnp.ndarray, omega: jnp.ndarray) -> jnp.ndarray:
        """Compute the drift terms for the SEIRx model.

        dS/dt = μN(1-x) - βSI/N - μS
        dE/dt = βSI/N - (ε+μ)E
        dI/dt = εE - (gamma+μ)I
        dx/dt = κx(1-x)(-ω + I + δ(2x-1))

        Parameters
        ----------
        state : [S, E, I, x] current state
        omega : perceived vaccination risk (bifurcation parameter)

        Returns
        -------
        drift : [dS/dt, dE/dt, dI/dt, dx/dt] (without noise)
        """
        S, E, I, x = state[0], state[1], state[2], state[3]
        N = self._config.N
        mu, beta, epsilon, gamma, kappa, delta = (
            self._config.mu,
            self._config.beta,
            self._config.epsilon,
            self._config.gamma,
            self._config.kappa,
            self._config.delta,
        )

        # Force of infection
        force_infection = beta * S * I / N

        # Disease dynamics
        dS = mu * N * (1.0 - x) - force_infection - mu * S
        dE = force_infection - (epsilon + mu) * E
        dI = epsilon * E - (gamma + mu) * I

        # Behavioural dynamics: payoff = -ω + I + δ(2x - 1)
        payoff = -omega + I + delta * (2.0 * x - 1.0)
        dx = kappa * x * (1.0 - x) * payoff

        return jnp.array([dS, dE, dI, dx])

    def _simulate_path(
        self,
        key: jax.random.PRNGKey,
        omega_t: jnp.ndarray,
        dt: float,
        noise_scale: jnp.ndarray,
        initial_state: jnp.ndarray,
        noise: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """Stochastic integration using Euler-Maruyama.

        d[S,E,I,x] = drift * dt + noise_scale * dW
        """
        n_steps = omega_t.shape[0] - 1

        if noise is None:
            noise = jax.random.normal(key, shape=(n_steps, 4))
        else:
            noise = jnp.asarray(noise)

        def step(state, inputs):
            omega_val, noise_t = inputs

            # Drift
            drift = self._drift(state, omega_val)

            # Euler-Maruyama: dx = drift * dt + noise_scale * dW
            state_next = state + drift * dt + noise_scale * noise_t

            # Ensure non-negative populations and x ∈ [0, 1]
            state_next = jnp.maximum(state_next, 1e-8)
            state_next = state_next.at[3].set(jnp.clip(state_next[3], 0.0, 1.0))

            # Ensure S + E + I <= N
            N = self._config.N
            sei = state_next[0] + state_next[1] + state_next[2]
            scale = jnp.where(sei > N, N / sei, 1.0)
            state_next = state_next.at[0].set(state_next[0] * scale)
            state_next = state_next.at[1].set(state_next[1] * scale)
            state_next = state_next.at[2].set(state_next[2] * scale)

            return state_next, state_next

        _, states = lax.scan(step, initial_state, (omega_t[:-1], noise))

        # Build full trajectory including initial state
        trajectory = jnp.concatenate([initial_state[None, :], states], axis=0)

        return trajectory

    def simulate(
        self,
        key: jax.random.PRNGKey,
        total_len: int,
        sigma_tilde: float = 1.0,
        null_simulation: bool = False,
        dt_sample: float = 1.0,
        param_end_multiplier: float = 1.1,
        initial_state: Optional[jnp.ndarray] = None,
        noise: Optional[jnp.ndarray] = None,
    ):
        """Simulate SEIRx model.

        Parameters
        ----------
        key : JAX random key
        total_len : int
            Total number of output time points
        sigma_tilde : float, default=1.0
            Noise scaling factor (multiplies config sigma values)
        null_simulation : bool, default=False
            If True, same linear ramp as forced but ω capped so range to critical equals (omega_max - critical) (null trajectory)
        dt_sample : float, default=1.0
            Sampling interval for output (weeks)
        param_end_multiplier : float, default=1.1
            Multiplier for extending simulation time beyond transition

        Returns
        -------
        time : (total_len,) array
            Time points
        trajectory : (total_len, 2) array
            State trajectory [I, x] (infectious count, provaccine fraction)
        param_values : (total_len,) array
            ω(t) values
        noise_scale : (4,) array
            Effective noise scales used [noise_S, noise_E, noise_I, noise_x]
        """
        key_noise, key_ic = jax.random.split(key)

        # Setup time grid
        sample_every = sampling_stride(dt_sample, self._config.dt)
        total_time = self._config.t_start + total_len * dt_sample
        n_steps = total_len * sample_every + 1
        time_fine = jnp.linspace(self._config.t_start, total_time, n_steps)

        # Compute ω schedule
        omega_t = self._omega_schedule(
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

        # Configure noise scales: σ * sqrt(dt) for Euler-Maruyama
        noise_scale = jnp.array(
            [
                sigma_tilde * self._config.sigma_S * jnp.sqrt(self._config.dt),
                sigma_tilde * self._config.sigma_E * jnp.sqrt(self._config.dt),
                sigma_tilde * self._config.sigma_I * jnp.sqrt(self._config.dt),
                sigma_tilde * self._config.sigma_x * jnp.sqrt(self._config.dt),
            ]
        )

        # Simulate trajectory at fine time steps
        trajectory_fine = self._simulate_path(
            key_noise,
            omega_t,
            self._config.dt,
            noise_scale,
            initial_state,
            noise=noise,
        )

        # Downsample to output rate
        indices = jnp.arange(0, n_steps, sample_every)

        time = time_fine[indices]
        path_sampled = trajectory_fine[indices]
        param_values = omega_t[indices]

        # Output [I, x] as in Bury et al. Suppl. Fig. 1 (infectious count, provaccine fraction)
        I_count = path_sampled[:, 2]
        x_frac = path_sampled[:, 3]
        trajectory = jnp.stack([I_count, x_frac], axis=-1)

        return time, trajectory, param_values, noise_scale

    def is_tipped(
        self,
        trajectory: jnp.ndarray,
        param_values: jnp.ndarray,
    ):
        """Check if the parameter crossed the bifurcation point."""
        _, _, _, omega_bif = self._get_param_range()
        return float(jnp.max(param_values)) >= omega_bif

    def locate_transition(
        self,
        trajectory: jnp.ndarray,
        null_simulation: bool,
        param_values: jnp.ndarray,
        param_end_multiplier: float = 1.1,
    ):
        """Midpoint for null; first omega crossing for forced, or None if absent."""
        if null_simulation:
            return len(param_values) // 2

        _, _, _, omega_bif = self._get_param_range()

        crossed = param_values >= omega_bif

        if bool(jnp.any(crossed)):
            return int(jnp.argmax(crossed))

        return None
