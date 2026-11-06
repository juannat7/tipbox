from dataclasses import asdict
from typing import Any, Dict

import jax

from ..types import TrajectoryResult
from ..registry import get_bifurcation_real_system_spec
from ..utils import calculate_total_len


def available_bifurcation_real_systems():
    """Get the available real systems with bifurcation tipping (internal names)."""
    from ..registry import SYSTEM_REGISTRY

    return tuple(
        spec.bifurcation_system
        for spec in SYSTEM_REGISTRY.values()
        if spec.bifurcation_system is not None
    )


def _select_system(key: jax.random.PRNGKey, system_name: str = None):
    """Resolve a named bifurcation model and its typed configuration."""
    from ..registry import SYSTEM_REGISTRY, SystemType

    if system_name is not None:
        return get_bifurcation_real_system_spec(system_name)

    # Random selection
    real_systems = [
        spec
        for spec in SYSTEM_REGISTRY.values()
        if spec.system_type == SystemType.BIFURCATION
        and spec.bifurcation_system is not None
    ]
    if not real_systems:
        raise ValueError("No bifurcation real systems available")
    idx = int(jax.random.randint(key, (), 0, len(real_systems)))
    return real_systems[idx]


def simulate_real_bifurcation_trajectory(
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
    """Simulate and extract legacy bifurcation context. Return None on outcome/context rejection."""

    key_select, key_sim = jax.random.split(key)

    if steps_before_transition > buffer_size:
        raise ValueError(
            "steps_before_transition exceeds buffer_size. "
            "Increase buffer_size to maintain constant parameter rate."
        )

    # Calculate total simulation length (extended if param_end_multiplier > 1.0)
    total_len = calculate_total_len(series_len, buffer_size, param_end_multiplier)

    # Select, setup and initialize model using unified registry
    system_spec = _select_system(key_select, system_name=system_name)
    config = system_spec.default_config() if config is None else config
    model = system_spec.create_model(config)

    # Simulate
    time, trajectory, param_values, sigma = model.simulate(
        key_sim,
        total_len=total_len,
        sigma_tilde=sigma_tilde,
        null_simulation=null_simulation,
        param_end_multiplier=param_end_multiplier,
        **generate_kwargs,
    )

    # Check if the trajectory crossed the bifurcation point
    is_tipped = model.is_tipped(trajectory, param_values=param_values)
    if is_tipped == null_simulation:
        return None

    # Locate transition index (pass param_values for parameter-based detection)
    trans_idx = model.locate_transition(
        trajectory,
        null_simulation=null_simulation,
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
        "label": "null" if null_simulation else system_spec.bifurcation_system,
        "bifurcation_real_system": system_spec.bifurcation_system,
        "steps_before_transition": steps_before_transition,
        "system_config": asdict(config),
        "transition_index": int(trans_idx_out),
        "param_value_transition": float(
            param_values[min(trans_idx_out, param_values.shape[0] - 1)]
        ),
    }

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
