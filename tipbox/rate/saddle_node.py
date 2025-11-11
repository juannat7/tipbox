from dataclasses import replace, dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax


@dataclass(frozen=True)
class SaddleNodeConfig:
    """Configuration for the stochastic saddle-node system."""

    epsilon: float = 1.25
    lambda_max: float = 3.0
    dt: float = 0.01
    t_start: float = -16.0
    t_end: float = 8.0
    initial_state_min: float = -10.0
    initial_state_max: float = -1.0
    tip_threshold: float = 0.0


@jax.jit
def _simulate_path(
    key: jax.random.PRNGKey,
    lambda_t: jnp.ndarray,
    dt: float,
    noise_scale: float,
    initial_state: float,
) -> jnp.ndarray:
    """JIT-compiled stochastic integration using lax.scan."""

    noise = jax.random.normal(key, shape=(lambda_t.shape[0] - 1,))

    def step(x_prev, inputs):
        lam_t, noise_t = inputs
        det_part = ((x_prev + lam_t) ** 2 - 1.0) * dt
        x_next = x_prev + det_part + noise_scale * noise_t
        return x_next, x_next

    _, x_tail = lax.scan(step, initial_state, (lambda_t[:-1], noise))
    x0 = jnp.full((1,), initial_state, dtype=lambda_t.dtype)
    return jnp.concatenate([x0, x_tail], axis=0)


def simulate(
    key: jax.random.PRNGKey,
    config: SaddleNodeConfig,
    total_len: int,
    sigma_tilde: float = 0.01,
) -> Tuple[jnp.ndarray, jnp.ndarray, float, float, SaddleNodeConfig]:
    """Simulate a stochastic saddle-node system and detect tipping.

    Returns time grid, trajectory, tipping flag, effective noise scale (sigma),
    and the initial state sampled for this simulation.
    """

    key_noise, key_ic = jax.random.split(key)
    config = replace(config, t_start=-(total_len * config.dt))

    # Randomly sample IC
    initial_state = float(
        jax.random.uniform(
            key_ic, 
            (), 
            minval=config.initial_state_min, 
            maxval=config.initial_state_max
        )
    )

    # Get lambda_t (forcing term for tipping point)
    t = jnp.arange(config.t_start, config.t_end, config.dt)
    lambda_t = 0.5 * config.lambda_max * (
        jnp.tanh(0.5 * config.lambda_max * config.epsilon * t) + 1.0
    )

    # Get noise scale for stochastic integration
    noise = jnp.sqrt(2.0 * sigma_tilde * config.dt)

    # Simulate trajectory
    trajectory = _simulate_path(
        key_noise, lambda_t, config.dt, noise, initial_state
    )

    return t, trajectory, lambda_t, noise, config


def locate_transition_index(
    trajectory: jnp.ndarray,
    null_simulation: bool,
    tip_threshold: float,
    null_position: Optional[int] = None,
) -> Optional[int]:
    """Locate the tipping index in the trajectory."""

    data = jnp.asarray(trajectory)

    if not null_simulation:
        crossed = jnp.where(data > tip_threshold, 1, 0)
        has_crossed = bool(jnp.any(crossed))
        if not has_crossed:
            return None
        first_idx = int(jnp.argmax(crossed))
        return first_idx

    if null_position is not None:
        return int(jnp.minimum(null_position, data.shape[0] - 1))

    return int(data.shape[0] // 2)

