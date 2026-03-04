from dataclasses import asdict
from typing import Any, Dict

import jax

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
        name for name, spec in SYSTEM_REGISTRY.items() if spec.system_type == SystemType.RATE and spec.rate_system
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
    schedule=None,
    **generate_kwargs,
):
    """Simulate a single rate-induced tipping trajectory.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for reproducibility
    series_len : int, default=500
        Desired output trajectory length
    steps_before_transition : int, default=1
        Number of samples to stay before the detected transition (>=0).
    buffer_size : int, default=1000
        Extra samples simulated prior to the transition to ensure enough
        pre-transition context. Keep this constant to preserve the
        bifurcation parameter ramp rate.
    null_simulation : bool, default=False
        If True, simulate null (non-tipping) trajectory
    system_name : str, optional
        Specific rate system to use. If None, randomly selects from available systems.
        Options: 'bautin', 'compost_bomb', 'saddle_node'
    param_end_multiplier : float, default=1.1
        Multiplier for extending simulation time beyond transition
    schedule : optional
        Parameter schedule. When provided, uses direct mode (exact length,
        schedule controls parameter evolution). When None, uses legacy
        extraction mode.
    **generate_kwargs : additional arguments
        Passed to the model.simulate method (dt_sample, dt_integrate, etc.)

    Returns
    -------
    tuple[TrajectoryResult, TrajectoryResult] or None
        Tuple of (full_trajectory, part_trajectory), or None if the trajectory
        does not match the requested tipping/null behaviour or lacks sufficient
        pre-transition context.

        Each TrajectoryResult.metadata includes:
        - label: "rate" or "null"
        - rate_system: name of the rate system used
        - transition_index: detected transition position
        - param_value_transition: parameter value at transition
        - param_normalized: normalized forcing values (0=initial, 1=max forcing)

    Notes
    -----
    Rate system trajectories may contain NaN values after the tipping point
    due to exponential or polynomial divergence in the ODE integration
    (especially the compost bomb system with its exponential temperature
    feedback). This is expected physical behavior. Trajectories that diverge
    to NaN before threshold detection will fail ``is_tipped()`` and cause
    this function to return None.
    """
    import jax.numpy as jnp

    key_select, key_sim = jax.random.split(key)

    # Select, setup and initialize model using unified registry
    if system_name is not None:
        system_spec = get_rate_system_spec(system_name)
    else:
        # Random selection (backward compatibility)
        system_spec = _select_system(key_select)

    config = system_spec.default_config()
    model = system_spec.create_model(config)

    state_trajectory = None
    if schedule is not None:
        # Direct mode: schedule controls parameter evolution
        sim_ret = model.simulate(
            key_sim,
            total_len=series_len,
            sigma_tilde=sigma_tilde,
            schedule=schedule,
            **generate_kwargs,
        )
        time, trajectory, param_values, sigma = sim_ret[0], sim_ret[1], sim_ret[2], sim_ret[3]
        state_trajectory = sim_ret[4] if len(sim_ret) > 4 else None
        normalized = model.normalize_param_values(param_values)

        # Estimate transition from normalized values
        above = jnp.where(normalized >= 1.0, jnp.arange(len(normalized)), len(normalized) - 1)
        trans_idx = int(jnp.min(above))

        metadata = {
            "label": "null" if null_simulation else "rate",
            "rate_system": system_spec.rate_system,
            "transition_index": trans_idx,
            "param_value_transition": float(param_values[min(trans_idx, len(param_values) - 1)]),
            "param_normalized": normalized,
            "system_config": asdict(config),
        }
        if state_trajectory is not None:
            metadata["state_trajectory"] = state_trajectory
        result = TrajectoryResult(
            time=time,
            trajectory=trajectory,
            param_values=param_values,
            sigma=sigma,
            metadata=metadata,
        )
        return (result, result)  # direct mode: full = part

    # Legacy extraction mode
    if steps_before_transition > buffer_size:
        raise ValueError(
            "steps_before_transition exceeds buffer_size. Increase buffer_size to maintain constant parameter rate."
        )

    total_len = calculate_total_len(series_len, buffer_size, param_end_multiplier)

    # Simulate legacy mode (with null_simulation for supported systems)
    sim_kwargs = {"total_len": total_len, "sigma_tilde": sigma_tilde}
    if system_spec.rate_system in ("saddle_node", "bautin", "compost_bomb", "amoc"):
        sim_kwargs["null_simulation"] = null_simulation
    sim_ret = model.simulate(key_sim, **sim_kwargs)
    time, trajectory, param_values, sigma = sim_ret[0], sim_ret[1], sim_ret[2], sim_ret[3]
    state_trajectory = sim_ret[4] if len(sim_ret) > 4 else None

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

    # Compute normalized parameter values
    normalized = model.normalize_param_values(param_values)
    normalized_seg = model.normalize_param_values(param_seg)

    # Return results
    full_metadata: Dict[str, Any] = {
        "label": "null" if null_simulation else "rate",
        "rate_system": system_spec.rate_system,
        "steps_before_transition": steps_before_transition,
        "system_config": asdict(config),
        "transition_index": int(trans_idx_out),
        "param_value_transition": float(param_values[min(trans_idx_out, param_values.shape[0] - 1)]),
        "param_normalized": normalized,
    }

    part_metadata: Dict[str, Any] = {
        "label": "null" if null_simulation else "rate",
        "rate_system": system_spec.rate_system,
        "steps_before_transition": steps_before_transition,
        "system_config": asdict(config),
        "transition_index": int(trans_idx_out),
        "param_value_transition": float(param_values[min(trans_idx_out, param_values.shape[0] - 1)]),
        "param_normalized": normalized_seg,
    }
    if state_trajectory is not None:
        full_metadata["state_trajectory"] = state_trajectory
        part_metadata["state_trajectory"] = state_trajectory

    full_trajectory = TrajectoryResult(
        time=time,
        trajectory=trajectory,
        param_values=param_values,
        sigma=sigma,
        metadata=full_metadata,
    )

    part_trajectory = TrajectoryResult(
        time=time_seg,
        trajectory=traj_seg,
        param_values=param_seg,
        sigma=sigma,
        metadata=part_metadata,
    )

    return (full_trajectory, part_trajectory)
