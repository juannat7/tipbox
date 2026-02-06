"""
Main API for generating tipping point time series data.

Generates random polynomial dynamical systems with bifurcations,
then simulates stochastic trajectories approaching the tipping point.
"""

import jax
import jax.numpy as jnp

from .bifurcation.bifurcation_finder import find_all_bifurcations, select_bifurcation
from .bifurcation.bifurcation_simulator import simulate_real_bifurcation_trajectory
from .bifurcation.polynomial_model import generate_random_polynomial_model
from .bifurcation.stochastic_simulator import simulate_synthetic_bifurcation_trajectory
from .rate.rate_simulator import simulate_rate_trajectory
from .registry import (
    SystemType,
    get_system_info,
    normalize_system_name,
)


def generate(
    key: jax.random.PRNGKey,
    tipping_type: str = "b_fold",
    null_simulation: bool = False,
    num_timesteps: int = 500,
    steps_before_transition: int = 1,
    buffer_size: int = 1000,
    dt_sample: float = 1.0,
    sigma_tilde: float = 0.01,
    max_attempts: int = 100,
    verbose: bool = True,
    **generate_kwargs,
) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
    """
    Generate time series using PNAS paper approach.

    Pipeline:
    1. Generate random polynomial dynamical system (or rate-induced system)
    2. Detect bifurcations with AUTO-07P (bifurcation types only)
    3. Simulate stochastic trajectory (to bifurcation or null)
    4. Extract pre-transition time series

    Parameters
    ----------
    key : JAX random key
    tipping_type : str, default='b_fold'
        System name. Examples:
        - Synthetic bifurcation: 'b_fold', 'b_hopf', 'b_transcritical'
        - Rate-induced: 'r_bautin', 'r_compost_bomb', 'r_saddle_node'
        - Semi-real bifurcation: 'b_harvesting'
    null_simulation : bool, default=False
        If True, simulate null (non-tipping) trajectory for this tipping type.
        Finds bifurcation but keeps parameter fixed (no approach to tipping point).
    num_timesteps : int, default=500
        Length of output trajectory
    steps_before_transition : int, default=1
        Number of samples to stay before the detected transition (>=0).
    buffer_size : int, default=1000
        Extra samples simulated before the transition to guarantee enough
        pre-transition context while keeping the parameter ramp rate constant.
    dt_sample : float, default=1.0
        Time between samples
    sigma_tilde : float, default=0.01
        Base noise amplitude (PNAS default)
    max_attempts : int, default=100
        Max attempts to find suitable model
    verbose : bool, default=True
        If True (default), print AUTO-07P output during bifurcation detection.
        Set to False to suppress verbose compilation output.
    **generate_kwargs : additional arguments
        Passed to simulation functions. Key parameters include:
        - param_end_multiplier: Optional[float], default=1.1
            Legacy multiplier for extending simulation past bif_value.
            Ignored if schedule is provided.
        - schedule: Optional[Schedule], default=None
            Parameter schedule (array, callable, or dict config).
            When provided, uses direct mode (exact length, pre-determined position).
            See tipbox.bifurcation.schedule for available schedule types:
            - linear: Linear ramp reaching 1.0 at critical_at fraction
            - ramp_and_hold: Ramp to param_level at hold_at, then hold
            - constant: Fixed value (useful for null simulations)
            - piecewise_linear: Concatenated linear segments
            - from_array: Direct array passthrough

    Returns
    -------
    full_result : TrajectoryResult
        Full trajectory with metadata
    part_result : TrajectoryResult
        Extracted pre-transition segment
    info : dict
        Metadata describing the sampled trajectory:
        - system_name: str (normalized system name)
        - type: str ('bifurcation' or 'rate')
        - steps_before_transition: int gap applied before transition
        - sigma: float noise scale used during simulation
        - attempt: int number of the successful attempt (>=1)
        - transition_index: int detected transition position
        - param_normalized: array of normalized parameter values in [0,1]
        - Additional fields depend on system type

    Notes
    -----
    The `param_normalized` in metadata contains normalized parameter values
    (generally in [0,1]) useful for understanding the forcing profile. Access via:
    - `full_result.metadata["param_normalized"]` - values for full trajectory
    - `part_result.metadata["param_normalized"]` - values for extracted segment
    - `info["param_normalized"]` - same as full_result (copied to info dict)

    Normalization semantics differ by system type:
    - Synthetic bifurcation: 0 = param_start, 1 = bif_value (schedule system)
    - Rate-induced: normalized by max forcing (0 = initial, 1 = max)
    - Real bifurcation: normalized by bifurcation point (0 = param_min, 1 = bif_value)

    Raises
    ------
    RuntimeError
        If no suitable model found after max_attempts
    ValueError
        If system name is not recognized
    """

    # Normalize system name
    try:
        system_name = normalize_system_name(tipping_type)
    except ValueError as e:
        raise ValueError(
            f"Unknown system: {tipping_type}. "
            f"Use explicit names like 'b_fold', 'r_bautin', 'b_harvesting', etc. "
            f"See tipbox.registry for available systems."
        ) from e

    system_spec = get_system_info(system_name)
    system_type = system_spec.system_type

    # Route to appropriate simulator based on system type
    if system_type == SystemType.RATE:
        # Rate-induced tipping
        rate_system = system_spec.rate_system
        if rate_system is None:
            raise ValueError(
                f"Rate system '{system_name}' missing rate_system specification"
            )

        for attempt in range(max_attempts):
            key, subkey = jax.random.split(key)
            results = simulate_rate_trajectory(
                subkey,
                series_len=num_timesteps,
                steps_before_transition=steps_before_transition,
                buffer_size=buffer_size,
                null_simulation=null_simulation,
                sigma_tilde=sigma_tilde,
                system_name=rate_system,
                **generate_kwargs,
            )

            if results is not None:
                full_result, part_result = results
                info = {
                    "attempt": attempt + 1,
                    "system_name": system_name,
                    "type": system_spec.system_type.value,
                }
                info.update(full_result.metadata)
                return full_result, part_result, info

        raise RuntimeError(
            f"Could not generate {system_name} trajectory after {max_attempts} attempts. "
            "Try increasing max_attempts or adjust configuration."
        )

    elif system_spec.is_synthetic_bifurcation:
        # Synthetic bifurcation (polynomial models) - b_fold, b_hopf, b_transcritical
        bifurcation_type = system_spec.bifurcation_system

        for attempt in range(max_attempts):
            # Generate random polynomial model
            key, subkey = jax.random.split(key)
            model = generate_random_polynomial_model(subkey)

            if model is None:
                continue

            # Find bifurcations with AUTO-07P
            bifurcations = find_all_bifurcations(model, verbose=verbose)
            if not bifurcations:
                continue

            # Select requested bifurcation type
            bifurcation = select_bifurcation(bifurcations, bifurcation_type)
            if bifurcation is None:
                continue

            # Simulate stochastic trajectory
            key, subkey = jax.random.split(key)
            results = simulate_synthetic_bifurcation_trajectory(
                model,
                bifurcation,
                subkey,
                series_len=num_timesteps,
                steps_before_transition=steps_before_transition,
                buffer_size=buffer_size,
                null_simulation=null_simulation,
                dt_sample=dt_sample,
                sigma_tilde=sigma_tilde,
                **generate_kwargs,
            )

            if results is not None:
                full_result, part_result = results
                info = {
                    "model": model,
                    "bifurcation": bifurcation,
                    "attempt": attempt + 1,
                    "system_name": system_name,
                    "type": system_type.value,
                }
                info.update(full_result.metadata)
                return full_result, part_result, info

        raise RuntimeError(
            f"Could not generate {system_name} trajectory after {max_attempts} attempts. "
            "Try increasing max_attempts or choose different system."
        )

    else:
        # Real bifurcation system (e.g., May's harvesting)
        real_system_name = system_spec.bifurcation_system

        for attempt in range(max_attempts):
            key, subkey = jax.random.split(key)
            results = simulate_real_bifurcation_trajectory(
                subkey,
                series_len=num_timesteps,
                steps_before_transition=steps_before_transition,
                buffer_size=buffer_size,
                null_simulation=null_simulation,
                sigma_tilde=sigma_tilde,
                system_name=real_system_name,
                **generate_kwargs,
            )

            if results is not None:
                full_result, part_result = results
                info = {
                    "attempt": attempt + 1,
                    "system_name": system_name,
                    "type": system_spec.system_type.value,
                }
                info.update(full_result.metadata)
                return full_result, part_result, info

        raise RuntimeError(
            f"Could not generate {system_name} trajectory after {max_attempts} attempts. "
            "Try increasing max_attempts or adjust configuration."
        )
