from typing import Tuple, Optional

import jax.numpy as jnp


def convert_scalar_trajectory(x: jnp.ndarray) -> jnp.ndarray:
    """Lift a scalar trajectory to a 2D array [x, 0] to reuse 2D utilities."""
    return jnp.stack([x, jnp.zeros_like(x)], axis=-1)


def extract_segment(
    time: jnp.ndarray,
    trajectory: jnp.ndarray,
    params: jnp.ndarray,
    transition_idx: int,
    steps_before_transition: int,
    series_len: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int, bool]:
    """Extract a segment mirroring bifurcation.extract_pre_transition_series."""

    gap = max(steps_before_transition, 0)
    end_idx = transition_idx - gap
    start_idx = end_idx - series_len
    success = (end_idx > 0) and (end_idx >= series_len) and (start_idx >= 0)

    if success:
        time_out = time[start_idx:end_idx] - time[start_idx]
        trajectory_out = trajectory[start_idx:end_idx]
        params_out = params[start_idx:end_idx]
    else:
        time_out = time[:series_len] - time[0]
        trajectory_out = trajectory[:series_len]
        params_out = params[:series_len]

    return time_out, trajectory_out, params_out, transition_idx, success

