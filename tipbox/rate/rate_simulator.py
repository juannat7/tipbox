from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple, Type

import jax
import jax.numpy as jnp

from .saddle_node import SaddleNodeConfig, SaddleNodeModel
from ..types import BoxModel, TrajectoryResult


class RateSystemSpec:
    """Specification for a rate-induced tipping system."""

    def __init__(
        self,
        name: str,
        config_cls: Type[Any],
        model_cls: Type[BoxModel],
    ) -> None:
        self.name = name
        self.config_cls = config_cls
        self.model_cls = model_cls

    def default_config(self) -> Any:
        return self.config_cls()

    def create_model(self, config: Any) -> BoxModel:
        return self.model_cls(config)


RATE_SYSTEMS: Dict[str, RateSystemSpec] = {
    "saddle_node": RateSystemSpec(
        name="saddle_node",
        config_cls=SaddleNodeConfig,
        model_cls=SaddleNodeModel,
    )
    # TODO: Add more rate-induced tipping systems
}


def available_rate_systems() -> Tuple[str, ...]:
    """Return the tuple of available rate-induced tipping systems."""

    return tuple(RATE_SYSTEMS.keys())


def _select_system(key: jax.random.PRNGKey) -> RateSystemSpec:
    """Select a random rate-induced tipping system."""
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
):
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

    # Global configuration
    total_len = series_len + buffer_size

    # Select, setup and initialize model
    system = _select_system(key_select)
    config = system.default_config()
    model = system.create_model(config)

    # Simulate
    time, trajectory, param_values, sigma = model.simulate(
        key_sim,
        total_len=total_len,
        sigma_tilde=sigma_tilde,
    )

    # Check if the trajectory is tipped
    is_tipped = model.is_tipped(trajectory)
    if is_tipped == null_simulation:
        return None

    # Locate transition index
    trans_idx = model.locate_transition(
        trajectory,
        null_simulation=null_simulation,
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
