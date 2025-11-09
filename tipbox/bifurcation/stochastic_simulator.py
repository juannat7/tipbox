"""
Stochastic simulation of polynomial dynamical systems approaching bifurcations.

Implements Euler-Maruyama method with:
- Linearly increasing bifurcation parameter
- Additive Gaussian white noise
- Noise amplitude scaling based on recovery rate
"""

import jax
import jax.numpy as jnp
import numpy as np
import ruptures as rpt
from statsmodels.nonparametric.smoothers_lowess import lowess
from typing import Tuple, Optional
from .polynomial_model import PolynomialModel, polynomial_basis
from .bifurcation_finder import BifurcationPoint


@jax.jit
def euler_maruyama_step(
    state: jnp.ndarray,
    pars_a: jnp.ndarray,
    pars_b: jnp.ndarray,
    dt: float,
    dW: jnp.ndarray
) -> jnp.ndarray:
    """
    Single Euler-Maruyama integration step.
    
    dx = f(x) * dt + dW
    
    Parameters
    ----------
    state : [x, y] current state
    pars_a, pars_b : polynomial parameters
    dt : time step
    dW : Brownian increment [dW_x, dW_y]
    
    Returns
    -------
    new_state : [x, y] next state
    """
    polys = polynomial_basis(state)
    drift = jnp.array([jnp.dot(pars_a, polys), jnp.dot(pars_b, polys)])
    return state + drift * dt + dW


@jax.jit
def _burn_in_scan(carry, dW):
    """Scan function for burn-in period (JIT-compiled)."""
    state, pars_a, pars_b, dt = carry
    state = euler_maruyama_step(state, pars_a, pars_b, dt, dW)
    return (state, pars_a, pars_b, dt), None


@jax.jit
def _simulation_scan(carry, inputs):
    """Scan function for main simulation (JIT-compiled)."""
    state, param_idx, dt = carry
    pars, dW = inputs
    
    pars_a = pars[:10]
    pars_b = pars[10:]
    state = euler_maruyama_step(state, pars_a, pars_b, dt, dW)
    
    return (state, param_idx, dt), state


def simulate_to_bifurcation(
    model: PolynomialModel,
    bifurcation: BifurcationPoint,
    key: jax.random.PRNGKey,
    series_len: int = 500,
    dt_sample: float = 1.0,
    dt_integrate: float = 0.01,
    sigma_tilde: float = 0.01,
    noise_scale_triangular: bool = True,
    burn_in_time: float = 100.0,
    null_sim: bool = False,
    null_location: float = 0.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Simulate system approaching bifurcation with stochastic noise.
    
    - Bifurcation parameter increases linearly from initial value to bifurcation value
    - Noise amplitude: sigma = sqrt(2 * recovery_rate) * sigma_tilde * rv_tri
    - Uses Euler-Maruyama integration
    
    Parameters
    ----------
    model : PolynomialModel
        Base model with equilibrium and parameters
    bifurcation : BifurcationPoint
        Bifurcation info (type, parameter, value)
    key : JAX random key
    series_len : int, default=500
        Number of output time points
    dt_sample : float, default=1.0
        Sampling interval for output
    dt_integrate : float, default=0.01
        Integration time step
    sigma_tilde : float, default=0.01
        Base noise amplitude (PNAS default)
    noise_scale_triangular : bool, default=True
        Scale noise by triangular random variable
    burn_in_time : float, default=100.0
        Burn-in period at initial parameter
    null_sim : bool, default=False
        If True, fix parameter (null trajectory)
    null_location : float, default=0.0
        For null: parameter fraction [0,1] from start to bifurcation
    
    Returns
    -------
    time : (series_len,) array of time points
    trajectory : (series_len, 2) array of [x, y] states
    param_values : (series_len,) array of bifurcation parameter values
    """
    # Noise amplitude: sigma = sqrt(2*lambda) * sigma_tilde * triangular_rv
    key, subkey = jax.random.split(key)
    rv_tri = jax.random.triangular(subkey, 0.75, 1.0, 1.25) if noise_scale_triangular else 1.0
    sigma = jnp.sqrt(2 * model.recovery_rate) * sigma_tilde * rv_tri
    
    # Parameter schedule
    pars = model.get_parameters()
    param_idx = bifurcation.bif_param_index
    param_start = pars[param_idx]
    param_end = bifurcation.bif_value
    
    if null_sim:
        param_fixed = param_start + null_location * (param_end - param_start)
        param_start = param_end = param_fixed
    
    # Time discretization
    total_time = series_len * dt_sample
    n_steps = int(total_time / dt_integrate)
    n_burn = int(burn_in_time / dt_integrate)
    
    param_schedule = jnp.linspace(param_start, param_end, n_steps)
    
    # Generate noise
    key, subkey1, subkey2 = jax.random.split(key, 3)
    noise_scale = sigma * jnp.sqrt(dt_integrate)
    dW_burn = jax.random.normal(subkey1, (n_burn, 2)) * noise_scale
    dW = jax.random.normal(subkey2, (n_steps, 2)) * noise_scale
    
    # Burn-in: run at initial parameter
    pars_a, pars_b = pars[:10], pars[10:]
    carry_burn = (model.equi, pars_a, pars_b, dt_integrate)
    (state_after_burn, _, _, _), _ = jax.lax.scan(_burn_in_scan, carry_burn, dW_burn)
    
    # Main simulation: vectorized with scan
    # Create parameter arrays for each step
    pars_varying = jax.vmap(lambda p: pars.at[param_idx].set(p))(param_schedule)
    
    carry_sim = (state_after_burn, param_idx, dt_integrate)
    _, states = jax.lax.scan(_simulation_scan, carry_sim, (pars_varying, dW))
    
    # Downsample to output rate
    sample_every = int(dt_sample / dt_integrate)
    indices = jnp.arange(0, n_steps, sample_every)[:series_len]
    
    trajectory = states[indices]
    time = indices * dt_integrate
    param_values = param_schedule[indices]
    
    return time, trajectory, param_values, sigma


def detect_transition(
    trajectory: jnp.ndarray,
    time: Optional[jnp.ndarray] = None
) -> int:
    """
    Detect transition point in trajectory.
    
    Finds transition via:
    1. NaN jump detection
    2. LOWESS detrending (span=0.2)
    3. Change-point detection on residuals (window=10, penalty=1)
    
    Parameters
    ----------
    trajectory : (n_time, 2) array
    time : (n_time,) array, optional
    
    Returns
    -------
    int : transition index (or len-1 if none detected)
    """
    x_series = np.array(trajectory[:, 0])
    time_arr = np.array(time) if time is not None else np.arange(len(x_series))
    
    # Find first NaN (if any)
    nan_mask = np.isnan(x_series)
    t_nan = time_arr[np.argmax(nan_mask)] if np.any(nan_mask) else 1e6
    
    # Change-point detection (only if no early NaN)
    if t_nan > 500:
        # Truncate at NaN if present
        cutoff = np.argmax(nan_mask) if t_nan < 1e6 else len(x_series)
        x_data = x_series[:cutoff]
        t_data = time_arr[:cutoff]
        
        # LOWESS smoothing
        smooth = lowess(x_data, t_data, frac=0.2)[:, 1]
        if len(smooth) != len(x_data):  # Rare failure
            return 0
        
        # Detect jump in residuals
        residuals = x_data - smooth
        algo = rpt.Window(width=10, model="l2", jump=1, min_size=2)
        bps = algo.fit(residuals).predict(pen=1)
        t_jump = bps[0]
    else:
        t_jump = 1e6
    
    # Take earlier transition
    t_trans = min(max(0, t_nan - 1), t_jump - 1)
    
    # Convert to array index
    if t_trans >= 1e6:
        return len(trajectory) - 1
    
    return int(np.argmin(np.abs(time_arr - t_trans)))


def extract_pre_transition_series(
    time: jnp.ndarray,
    trajectory: jnp.ndarray,
    param_values: jnp.ndarray,
    transition_idx: int,
    steps_before_transition: int,
    series_len: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, bool]:
    """
    Extract fixed-length series before transition with configurable offset.
    
    Extracts `series_len` points ending `steps_before_transition` samples
    before the detected transition index. Time is reset to start at 0.
    
    Parameters
    ----------
    time : (n,) array of time points
    trajectory : (n, 2) array of states
    param_values : (n,) array of parameter values
    transition_idx : int, transition detection index
    steps_before_transition : int
        Number of samples to stay before the transition index (>=0).
    series_len : int, desired output length
    
    Returns
    -------
    time_out : (series_len,) time array starting at 0
    trajectory_out : (series_len, 2) trajectory array
    param_out : (series_len,) parameter array
    success : bool, whether extraction succeeded
    """
    gap = max(steps_before_transition, 0)
    end_idx = transition_idx - gap
    start_idx = end_idx - series_len
    success = (end_idx > 0) and (end_idx >= series_len) and (start_idx >= 0)
    
    if success:
        time_out = time[start_idx:end_idx]
        trajectory_out = trajectory[start_idx:end_idx]
        param_out = param_values[start_idx:end_idx]
        time_out = time_out - time_out[0]
    else:
        time_out = time[:series_len]
        trajectory_out = trajectory[:series_len]
        param_out = param_values[:series_len]
        time_out = time_out - time_out[0]
    
    return time_out, trajectory_out, param_out, success


def simulate_training_trajectory(
    model: PolynomialModel,
    bifurcation: BifurcationPoint,
    key: jax.random.PRNGKey,
    series_len: int = 500,
    steps_before_transition: int = 1,
    null_sim: bool = False,
    **sim_kwargs
) -> Optional[Tuple[jnp.ndarray, jnp.ndarray, str]]:
    """
    Generate a single training trajectory (either tipping or null).
    
    This is the high-level function that wraps the PNAS approach:
    1. Simulate system to bifurcation
    2. Detect transition point
    3. Extract series_len points before transition
    
    Parameters
    ----------
    model : PolynomialModel
        Random polynomial model with equilibrium
    bifurcation : BifurcationPoint
        Bifurcation information (type, parameter, value)
    key : JAX random key
    series_len : int, default=500
        Desired output trajectory length
    steps_before_transition : int, default=1
        Number of samples to stay before the detected transition (>=0).
    null_sim : bool, default=False
        If True, simulate null (non-tipping) trajectory
    **sim_kwargs : additional arguments
        Passed to simulate_to_bifurcation (dt_sample, dt_integrate, etc.)
    
    Returns
    -------
    time : (series_len,) time array starting at 0
    trajectory : (series_len, 2) state array [x, y]
    label : str, bifurcation type ('fold', 'hopf', 'transcritical', 'null')
    
    Returns None if transition occurs too early (< series_len points before)
    """
    # Simulate with buffer to ensure enough pre-transition data
    buffer_len = series_len + steps_before_transition + 200
    
    time, trajectory, param_values, sigma = simulate_to_bifurcation(
        model, bifurcation, key,
        series_len=buffer_len,
        null_sim=null_sim,
        **sim_kwargs
    )
    
    # Detect transition
    trans_idx = detect_transition(trajectory, time)
    
    # Extract fixed-length series before transition
    time_out, traj_out, _, success = extract_pre_transition_series(
        time, trajectory, param_values, trans_idx, steps_before_transition, series_len
    )
    
    if not success:
        return None
    
    # Label based on bifurcation type
    label = 'null' if null_sim else bifurcation.bif_type
    
    return time_out, traj_out, label, sigma

