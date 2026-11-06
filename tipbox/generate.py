"""Generate labeled trajectories from the named box models."""

from math import isfinite

import jax

from .bifurcation.bifurcation_simulator import simulate_real_bifurcation_trajectory
from .rate.rate_simulator import simulate_rate_trajectory
from .registry import SystemType, get_system_info
from .types import TrajectoryResult


def generate(
    key: jax.Array,
    tipping_type: str = "b_harvesting",
    null_simulation: bool = False,
    num_timesteps: int = 500,
    steps_before_transition: int = 1,
    buffer_size: int = 1000,
    dt_sample: float | None = None,
    sigma_tilde: float = 0.01,
    max_attempts: int = 100,
    **generate_kwargs,
) -> tuple[TrajectoryResult, TrajectoryResult, dict]:
    """Return a full simulation, a pre-transition segment, and its metadata.

    Parameters
    ----------
    key : jax.Array
        Independent JAX random key for the simulation and any retries.
    tipping_type : str, default="b_harvesting"
        Named system from :func:`tipbox.registry.list_systems`.
    null_simulation : bool, default=False
        Request a non-tipping trajectory using the model's null forcing.
    num_timesteps : int, default=500
        Number of samples in the extracted segment.
    steps_before_transition : int, default=1
        Number of samples excluded before the model's transition marker.
    buffer_size : int, default=1000
        Additional simulation context. Its effect on physical duration is
        model-specific; fixed-duration rate models retain their own time grid.
    dt_sample : float or None, default=None
        Bifurcation-model output interval. None keeps the model's default
        (1.0 for ecological models; 5.0 years for AMOC). Rate models sample
        on their configured ``dt`` grid and reject this argument.
    sigma_tilde : float, default=0.01
        Model-specific noise scaling, interpreted by the model's simulator.
    max_attempts : int, default=100
        Maximum number of independent simulations before reporting rejection.
    **generate_kwargs
        Arguments passed to the model simulator, including ``initial_state``,
        ``noise``, and a typed ``config``. Unsupported names raise ``TypeError``.

    Returns
    -------
    full, segment, info
        Two :class:`tipbox.types.TrajectoryResult` objects and metadata.
        ``transition_index`` refers to the full trajectory. Null trajectories
        use a model-defined reference marker rather than an observed event.

    Raises
    ------
    ValueError
        For an unknown system or invalid requested lengths.
    RuntimeError
        If every simulation is rejected or lacks the requested context.
    """
    if num_timesteps < 1 or max_attempts < 1:
        raise ValueError("num_timesteps and max_attempts must be positive")
    if not 0 <= steps_before_transition <= buffer_size:
        raise ValueError("Require 0 <= steps_before_transition <= buffer_size")
    if not isfinite(sigma_tilde) or sigma_tilde < 0:
        raise ValueError("sigma_tilde must be finite and nonnegative")
    if dt_sample is not None:
        generate_kwargs["dt_sample"] = dt_sample

    spec = get_system_info(tipping_type)
    simulators = {
        SystemType.BIFURCATION: simulate_real_bifurcation_trajectory,
        SystemType.RATE: simulate_rate_trajectory,
    }
    system_name = spec.bifurcation_system or spec.rate_system
    for attempt in range(1, max_attempts + 1):
        key, subkey = jax.random.split(key)
        result = simulators[spec.system_type](
            subkey,
            series_len=num_timesteps,
            steps_before_transition=steps_before_transition,
            buffer_size=buffer_size,
            null_simulation=null_simulation,
            sigma_tilde=sigma_tilde,
            system_name=system_name,
            **generate_kwargs,
        )
        if result is not None:
            full, segment = result
            info = dict(full.metadata)
            info.update(
                attempt=attempt, system_name=spec.name, type=spec.system_type.value
            )
            return full, segment, info
    raise RuntimeError(
        f"Could not generate {spec.name} trajectory after {max_attempts} attempts. "
        "Adjust the model configuration or requested segment length."
    )
