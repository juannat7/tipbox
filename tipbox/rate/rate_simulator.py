from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple, Type

import jax
import jax.numpy as jnp

from .saddle_node import SaddleNodeConfig, locate_transition_index, simulate
from ..types import TrajectoryResult
from ..utils import convert_scalar_trajectory, extract_segment


class RateSystemSpec:
    """Specification for a rate-induced tipping system."""

    def __init__(
        self,
        name: str,
        config_cls: Type[Any],
        simulate_fn,
        locate_fn,
    ) -> None:
        self.name = name
        self.config_cls = config_cls
        self.simulate = simulate_fn
        self.locate_transition = locate_fn

    def default_config(self) -> Any:
        return self.config_cls()


RATE_SYSTEMS: Dict[str, RateSystemSpec] = {
    "saddle_node": RateSystemSpec(
        name="saddle_node",
        config_cls=SaddleNodeConfig,
        simulate_fn=simulate,
        locate_fn=locate_transition_index,
    )
}



def available_rate_systems() -> Tuple[str, ...]:
    """Return the tuple of available rate-induced tipping systems."""

    return tuple(RATE_SYSTEMS.keys())


def _select_system(key: jax.random.PRNGKey) -> RateSystemSpec:
    names = available_rate_systems()
    idx = int(jax.random.randint(key, (), 0, len(names)))
    return RATE_SYSTEMS[names[idx]]


def simulate_rate_trajectory(
    key: jax.random.PRNGKey,
    series_len: int = 500,
    steps_before_transition: int = 1,
    buffer_size: int = 1000,
    sigma_tilde: float = 0.01,
    null_simulation: bool = False,
    **generate_kwargs
) -> Optional[TrajectoryResult]:
    """Simulate a single rate-induced trajectory attempt.

    Returns ``None`` if the simulated trajectory does not match the requested
    tipping/null behaviour or lacks sufficient pre-transition context.
    """
    
    key_select, key_sim = jax.random.split(key)

    if steps_before_transition > buffer_size:
        raise ValueError(
            "steps_before_transition exceeds buffer_size. "
            "Increase buffer_size to maintain constant parameter rate."
        )

    # Simulate with constant buffer to preserve bifurcation ramp rate
    total_len = series_len + buffer_size

    # Randomly select rate system
    system = _select_system(key_select)
    config = system.default_config()

    # Simulate trajectory
    time, trajectory, param_values, sigma, config = system.simulate(
        key_sim, config, total_len=total_len, sigma_tilde=sigma_tilde
    )

    # Check if tipping occurred
    is_tipped = bool(jnp.any(trajectory > config.tip_threshold))
    if is_tipped == null_simulation:
        return None

    # Detect transition
    trans_idx = system.locate_transition(
        trajectory,
        null_simulation=null_simulation,
        tip_threshold=getattr(config, "tip_threshold", 0.0),
        null_position=trajectory.shape[0] // 2,
    )

    # Extract fixed-length series before transition
    trajectory = convert_scalar_trajectory(trajectory)
    time_seg, traj_seg, param_seg, trans_idx_out, success = extract_segment(
        time,
        trajectory,
        param_values,
        trans_idx,
        steps_before_transition,
        series_len,
    )

    if not success:
        return None

    # Return results
    metadata: Dict[str, Any] = {
        "label": "null" if null_simulation else "rate",
        "rate_system": system.name,
        "steps_before_transition": steps_before_transition,
        "system_config": asdict(config),
        "transition_index": int(trans_idx_out),
        "param_value_transition": float(param_values[min(trans_idx_out, param_values.shape[0] - 1)])
    }

    full_trajectory = TrajectoryResult(
        time=time,
        trajectory=trajectory,
        param_values=param_values,
        sigma=sigma,
        metadata=metadata
    )

    part_trajectory = TrajectoryResult(
        time=time_seg,
        trajectory=traj_seg,
        param_values=param_seg,
        sigma=sigma,
        metadata=metadata
    )

    return (full_trajectory, part_trajectory)
