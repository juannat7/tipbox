from dataclasses import asdict
from typing import Any, Dict

import jax
import jax.numpy as jnp

from ..types import TrajectoryResult
from ..registry import get_rate_system_spec, get_rate_systems
from ..utils import calculate_total_len


def available_rate_systems():
    """Get the available rate-induced tipping systems (internal names)."""
    return tuple(get_rate_systems())


def _select_system(key: jax.random.PRNGKey):
    """Randomly select a rate-induced tipping system."""
    from ..registry import SYSTEM_REGISTRY, SystemType

    rate_systems = [
        name
        for name, spec in SYSTEM_REGISTRY.items()
        if spec.system_type == SystemType.RATE and spec.rate_system
    ]
    idx = int(jax.random.randint(key, (), 0, len(rate_systems)))
    system_name = rate_systems[idx]
    return SYSTEM_REGISTRY[system_name]


def simulate_rate_trajectory(
    key: jax.random.PRNGKey,
    series_len: int = 500,
    steps_before_transition: int = 1,
    buffer_size: int = 1000,
    sigma_tilde: float = 0.01,
    null_simulation: bool = False,
    system_name: str = None,
    param_end_multiplier: float = 1.1,
    config=None,
    **generate_kwargs,
):
    """Simulate and extract legacy rate-model context. Return None on outcome/context rejection."""

    key_select, key_sim = jax.random.split(key)

    if steps_before_transition > buffer_size:
        raise ValueError(
            "steps_before_transition exceeds buffer_size. "
            "Increase buffer_size to maintain constant parameter rate."
        )

    # Calculate total simulation length (extended if param_end_multiplier > 1.0)
    total_len = calculate_total_len(series_len, buffer_size, param_end_multiplier)

    # Select, setup and initialize model using unified registry
    if system_name is not None:
        system_spec = get_rate_system_spec(system_name)
    else:
        # Random selection (backward compatibility)
        system_spec = _select_system(key_select)

    config = system_spec.default_config() if config is None else config
    model = system_spec.create_model(config)

    sim_ret = model.simulate(
        key_sim,
        total_len=total_len,
        sigma_tilde=sigma_tilde,
        null_simulation=null_simulation,
        **generate_kwargs,
    )
    time, trajectory, param_values, sigma = (
        sim_ret[0],
        sim_ret[1],
        sim_ret[2],
        sim_ret[3],
    )
    state_trajectory = sim_ret[4] if len(sim_ret) > 4 else None

    if null_simulation and (
        not bool(jnp.isfinite(trajectory).all())
        or (
            state_trajectory is not None
            and not bool(jnp.isfinite(state_trajectory).all())
        )
    ):
        return None

    # Check if the trajectory is tipped (pass time/param_values for models that need them, e.g. AMOC)
    is_tipped = model.is_tipped(
        trajectory,
        time=time,
        param_values=param_values,
    )
    if is_tipped == null_simulation:
        return None

    # Locate transition index (transition = when t crosses 0 for rate-induced AMOC)
    trans_idx = model.locate_transition(
        trajectory,
        null_simulation=null_simulation,
        time=time,
        param_values=param_values,
    )

    if trans_idx is None:
        return None

    # Extract segment
    time_seg, traj_seg, param_seg, trans_idx_out, success = model.extract_segment(
        time,
        trajectory,
        param_values,
        trans_idx,
        steps_before_transition=steps_before_transition,
        series_len=series_len,
    )

    if not success:
        return None

    # Return results
    metadata: Dict[str, Any] = {
        "label": "null" if null_simulation else "rate",
        "rate_system": system_spec.rate_system,
        "steps_before_transition": steps_before_transition,
        "system_config": asdict(config),
        "transition_index": int(trans_idx_out),
        "param_value_transition": float(
            param_values[min(trans_idx_out, param_values.shape[0] - 1)]
        ),
    }
    if state_trajectory is not None:
        metadata["state_trajectory"] = state_trajectory

    full_trajectory = TrajectoryResult(
        time=time,
        trajectory=trajectory,
        param_values=param_values,
        sigma=sigma,
        metadata=metadata,
    )

    part_trajectory = TrajectoryResult(
        time=time_seg,
        trajectory=traj_seg,
        param_values=param_seg,
        sigma=sigma,
        metadata=metadata,
    )

    return (full_trajectory, part_trajectory)
