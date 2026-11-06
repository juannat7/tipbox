"""Box models and reproducible tipping-point benchmarks."""

from . import generate
from .registry import get_system_info, list_systems
from .types import BoxModel, TrajectoryResult

__all__ = [
    "BoxModel",
    "TrajectoryResult",
    "generate",
    "get_system_info",
    "list_systems",
]
