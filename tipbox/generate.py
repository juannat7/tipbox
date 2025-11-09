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
from .bifurcation.stochastic_simulator import simulate_training_trajectory


def generate(
    key: jax.random.PRNGKey,
    bifurcation_type: str = 'fold',
    null_simulation: bool = False,
    num_timesteps: int = 500,
    steps_before_transition: int = 1,
    dt_sample: float = 1.0,
    sigma_tilde: float = 0.01,
    max_attempts: int = 100
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict]:
    """
    Generate time series using PNAS paper approach.
    
    Pipeline:
    1. Generate random polynomial dynamical system
    2. Detect bifurcations with AUTO-07P
    3. Simulate stochastic trajectory (to bifurcation or null)
    4. Extract pre-transition time series
    
    Parameters
    ----------
    key : JAX random key
    bifurcation_type : str, default='fold'
        Type: 'fold', 'hopf', or 'transcritical'
    null_simulation : bool, default=False
        If True, simulate null (non-tipping) trajectory for this bifurcation type.
        Finds bifurcation but keeps parameter fixed (no approach to tipping point).
    num_timesteps : int, default=500
        Length of output trajectory
    steps_before_transition : int, default=1
        Number of samples to stay before the detected transition (>=0).
    dt_sample : float, default=1.0
        Time between samples
    sigma_tilde : float, default=0.01
        Base noise amplitude (PNAS default)
    max_attempts : int, default=100
        Max attempts to find suitable model
    
    Returns
    -------
    time : (num_timesteps,) array
        Time points
    trajectory : (num_timesteps, 2) array
        State trajectory [x, y]
    info : dict
        Model and bifurcation details:
        - model: PolynomialModel object
        - bifurcation: BifurcationPoint object
        - bifurcation_type: str ('fold', 'hopf', 'transcritical', or 'null')
        - steps_before_transition: int gap applied before transition
        - parameters: array of 20 parameters
        - equilibrium: equilibrium point [x, y]
        - recovery_rate: dominant eigenvalue magnitude
    
    Raises
    ------
    RuntimeError
        If no suitable model found after max_attempts
    """
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
        
        # Select requested bifurcation type
        bifurcation = select_bifurcation(bifurcations, bifurcation_type)
        if bifurcation is None:
            continue
        
        # Simulate stochastic trajectory
        key, subkey = jax.random.split(key)
        result = simulate_training_trajectory(
            model, bifurcation, subkey,
            series_len=num_timesteps,
            steps_before_transition=steps_before_transition,
            null_sim=null_simulation,
            dt_sample=dt_sample,
            sigma_tilde=sigma_tilde
        )
        
        if result is not None:
            time, trajectory, label, sigma = result
            
            info = {
                'model': model,
                'bifurcation': bifurcation,
                'bifurcation_type': label,
                'steps_before_transition': steps_before_transition,
                'sigma': sigma
            }
            return time, trajectory, info
    
    raise RuntimeError(
        f"Could not generate {bifurcation_type} trajectory after {max_attempts} attempts. "
        "Try increasing max_attempts or choose different bifurcation_type."
    )
