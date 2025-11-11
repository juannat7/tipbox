"""
Main API for generating tipping point time series data.

Generates random polynomial dynamical systems with bifurcations,
then simulates stochastic trajectories approaching the tipping point.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict
from .bifurcation.polynomial_model import generate_random_polynomial_model
from .bifurcation.bifurcation_finder import find_all_bifurcations, select_bifurcation
from .bifurcation.stochastic_simulator import simulate_bifurcation_trajectory
from .rate.rate_simulator import simulate_rate_trajectory


def generate(
    key: jax.random.PRNGKey,
    tipping_type: str = 'fold',
    null_simulation: bool = False,
    num_timesteps: int = 500,
    steps_before_transition: int = 1,
    buffer_size: int = 1000,
    dt_sample: float = 1.0,
    sigma_tilde: float = 0.01,
    max_attempts: int = 100
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
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
    tipping_type : str, default='fold'
        Type: 'fold', 'hopf', 'transcritical', or 'rate'
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

    Returns
    -------
    time : (num_timesteps,) array
        Time points (starts at 0)
    trajectory : (num_timesteps, 2) array
        State trajectory [x, y]. For rate-induced tipping, the second column is zero.
    info : dict
        Metadata describing the sampled trajectory:
        - model: PolynomialModel object (bifurcation types only)
        - bifurcation: BifurcationPoint object (bifurcation types only)
        - tipping_type: str ('fold', 'hopf', 'transcritical', 'rate', or 'null')
        - steps_before_transition: int gap applied before transition
        - sigma: float noise scale used during simulation
        - attempt: int number of the successful attempt (>=1)
        - transition_index: int detected transition position (common to all types)
        - rate_system: str (rate only)
        - system_config: dict (rate only)

    Raises
    ------
    RuntimeError
        If no suitable model found after max_attempts
    """

    if tipping_type == 'rate':

        for attempt in range(max_attempts):
            key, subkey = jax.random.split(key)
            result = simulate_rate_trajectory(
                subkey,
                series_len=num_timesteps,
                steps_before_transition=steps_before_transition,
                buffer_size=buffer_size,
                null_simulation=null_simulation,
                sigma_tilde=sigma_tilde,
            )
            
            if result is None:
                continue

            info = {
                'tipping_type': 'null' if null_simulation else 'rate',
                'steps_before_transition': steps_before_transition,
                'sigma': result.sigma,
                'attempt': attempt + 1,
            }
            info.update(result.metadata)
            return result.time, result.trajectory, info

        raise RuntimeError(
            f"Could not generate rate trajectory after {max_attempts} attempts. "
            "Try increasing max_attempts or adjust configuration."
        )

    for attempt in range(max_attempts):
        # Generate random polynomial model
        key, subkey = jax.random.split(key)
        model = generate_random_polynomial_model(subkey)

        if model is None:
            continue

        # Find bifurcations with AUTO-07P
        bifurcations = find_all_bifurcations(model)
        if not bifurcations:
            continue

        # Select requested tipping type
        bifurcation = select_bifurcation(bifurcations, tipping_type)
        if bifurcation is None:
            continue

        # Simulate stochastic trajectory
        key, subkey = jax.random.split(key)
        result = simulate_bifurcation_trajectory(
            model, bifurcation, subkey,
            series_len=num_timesteps,
            steps_before_transition=steps_before_transition,
            buffer_size=buffer_size,
            null_simulation=null_simulation,
            dt_sample=dt_sample,
            sigma_tilde=sigma_tilde
        )

        if result is not None:
            info = {
                'model': model,
                'bifurcation': bifurcation,
                'tipping_type': result.label,
                'steps_before_transition': steps_before_transition,
                'sigma': result.sigma,
                'attempt': attempt + 1,
            }
            info.update(result.metadata)
            return result.time, result.trajectory, info

    raise RuntimeError(
        f"Could not generate {tipping_type} trajectory after {max_attempts} attempts. "
        "Try increasing max_attempts or choose different tipping_type."
    )
