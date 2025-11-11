from dataclasses import dataclass
from typing import Any, Dict

import jax.numpy as jnp


@dataclass(frozen=True)
class TrajectoryResult:
    """Common container for simulated tipping trajectories."""

    time: jnp.ndarray
    trajectory: jnp.ndarray
    label: str
    sigma: float
    metadata: Dict[str, Any]

