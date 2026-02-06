from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class TrajectoryResult:
    """Common container for simulated tipping trajectories.

    Attributes:
        time: Time points array.
        trajectory: State trajectory array (n_time, n_dims).
        param_values: Absolute parameter values at each time point.
        sigma: Noise scale used during simulation.
        metadata: Dict with at least 'label', 'transition_index', and
            'param_normalized' (normalized parameter values, generally [0,1]).
    """

    time: jnp.ndarray
    trajectory: jnp.ndarray
    param_values: jnp.ndarray
    sigma: float
    metadata: Optional[Dict[str, Any]]


class BoxModel(ABC):
    """Abstract interface for box models used in trajectory simulation."""

    @abstractmethod
    def simulate(self):
        """Run a simulation and return the resulting trajectory."""

    @abstractmethod
    def get_parameters(self):
        """Expose model parameters (useful for logging/metadata)."""
    
    @abstractmethod
    def is_tipped(self):
        """Return True if the trajectory exhibits tipping behaviour."""

    @abstractmethod
    def locate_transition(self):
        """Locate transition index in the provided trajectory."""

    @abstractmethod
    def extract_segment(self):
        """Return updated full trajectory and extracted segment around the transition."""

    @abstractmethod
    def normalize_param_values(self, param_values):
        """Normalize parameter values to [0, 1] scale.

        Convention: 0 = equilibrium/initial, 1 = critical/max forcing.
        """
