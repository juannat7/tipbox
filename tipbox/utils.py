from dataclasses import field
from math import isclose, isfinite
from typing import Tuple, Type, TypeVar

import jax
import jax.numpy as jnp

T = TypeVar("T")


def parameter(default, description: str, units: str = ""):
    """Declare a dataclass parameter and its documentation in one place."""
    return field(default=default, metadata={"description": description, "units": units})


def sampling_stride(dt_sample: float, dt_integrate: float) -> int:
    """Return an exact positive sampling stride on a fixed integration grid."""
    if not isfinite(dt_sample) or not isfinite(dt_integrate) or dt_integrate <= 0:
        raise ValueError("Sampling and integration steps must be finite and positive")
    stride = round(dt_sample / dt_integrate)
    if stride < 1 or not isclose(stride * dt_integrate, dt_sample, rel_tol=1e-7):
        raise ValueError(
            "dt_sample must be a positive integer multiple of the integration step"
        )
    return stride


def register_dataclass_pytree(cls: Type[T]) -> Type[T]:
    """Register dataclass fields as a JAX pytree."""
    from dataclasses import fields

    def flatten(instance: T):
        children = tuple(getattr(instance, f.name) for f in fields(instance))
        return children, None

    def unflatten(aux_data, children):
        return cls(*children)

    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls


def convert_scalar_trajectory(x: jnp.ndarray) -> jnp.ndarray:
    """Lift a scalar trajectory to a 2D array [x, 0] to reuse 2D utilities."""
    return jnp.stack([x, jnp.zeros_like(x)], axis=-1)


def calculate_total_len(
    context_samples: int, buffer_samples: int, param_end_multiplier: float = 1.1
) -> int:
    """Calculate total simulation length with optional extension.

    For simulations with param_end_multiplier > 1.0, extends the simulation time
    to ensure enough points before transition (forced) or after transition (null).

    Parameters
    ----------
    context_samples : int
        Desired output trajectory length
    buffer_samples : int
        Extra samples simulated prior to the transition
    param_end_multiplier : float, default=1.1
        Multiplier for extending simulation time beyond transition

    Returns
    -------
    int
        Total simulation length (extended if param_end_multiplier > 1.0)
    """
    base_len = context_samples + buffer_samples
    if param_end_multiplier > 1.0:
        return int(base_len * param_end_multiplier)
    return base_len


def extract_segment(
    time: jnp.ndarray,
    trajectory: jnp.ndarray,
    params: jnp.ndarray,
    transition_idx: int,
    excluded_samples: int,
    context_samples: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, int, bool]:
    """Extract context ending excluded_samples + 1 samples before the reference."""

    gap = max(excluded_samples, 0)
    end_idx = transition_idx - gap
    start_idx = end_idx - context_samples
    success = (end_idx > 0) and (end_idx >= context_samples) and (start_idx >= 0)

    if success:
        time_out = time[start_idx:end_idx] - time[start_idx]
        trajectory_out = trajectory[start_idx:end_idx]
        params_out = params[start_idx:end_idx]
    else:
        time_out = time[:context_samples] - time[0]
        trajectory_out = trajectory[:context_samples]
        params_out = params[:context_samples]

    return time_out, trajectory_out, params_out, transition_idx, success
