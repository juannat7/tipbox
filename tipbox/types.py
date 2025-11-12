from dataclasses import dataclass
from typing import Any, Dict, Optional

import jax.numpy as jnp 


@dataclass(frozen=True)
class TrajectoryResult:
    """Common container for simulated tipping trajectories."""

    time: jnp.ndarray
    trajectory: jnp.ndarray
    param_values: jnp.ndarray
    sigma: float
    metadata: Optional[Dict[str, Any]]

