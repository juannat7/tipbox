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
    context_samples: int = 500,
    excluded_samples: int = 1,
    buffer_samples: int = 1000,
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
    context_samples : int, default=500
        Number of samples in the extracted segment.
    excluded_samples : int, default=1
        Additional samples skipped before the reference. Zero ends immediately
        before the reference; one leaves one further observation out.
        Equivalent batch distance: lead_samples = excluded_samples + 1.
    buffer_samples : int, default=1000
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
    if context_samples < 1 or max_attempts < 1:
        raise ValueError("context_samples and max_attempts must be positive")
    if not 0 <= excluded_samples <= buffer_samples:
        raise ValueError("Require 0 <= excluded_samples <= buffer_samples")
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
            context_samples=context_samples,
            excluded_samples=excluded_samples,
            buffer_samples=buffer_samples,
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
