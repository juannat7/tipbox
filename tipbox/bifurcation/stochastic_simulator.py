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
from typing import Tuple, Optional, Dict, Any
from .polynomial_model import PolynomialModel, polynomial_basis
from .bifurcation_finder import BifurcationPoint
from ..types import TrajectoryResult
from ..utils import extract_segment


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
    null_simulation: bool = False,
    null_location: float = 0.0,
    param_end_multiplier: float = 1.1,
    initial_state: Optional[jnp.ndarray] = None,
    noise_burn: Optional[jnp.ndarray] = None,
    noise: Optional[jnp.ndarray] = None,
    rv_tri: Optional[jnp.ndarray] = None,
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
    null_simulation : bool, default=False
        If True, fix parameter (null trajectory)
    null_location : float, default=0.0
        For null: parameter fraction [0,1] from start to bifurcation
    param_end_multiplier : float, default=1.1
        Multiplier for extending simulation time beyond transition
    
    Returns
    -------
    time : (series_len,) array of time points
    trajectory : (series_len, 2) array of [x, y] states
    param_values : (series_len,) array of bifurcation parameter values
    """
    # Noise amplitude: sigma = sqrt(2*lambda) * sigma_tilde * triangular_rv
    key, subkey = jax.random.split(key)
    if rv_tri is None:
        rv_tri = jax.random.triangular(subkey, 0.75, 1.0, 1.25) if noise_scale_triangular else 1.0
    sigma = jnp.sqrt(2 * model.recovery_rate) * sigma_tilde * rv_tri
    
    # Parameter schedule
    pars = model.get_parameters()
    param_idx = bifurcation.bif_param_index
    param_start = pars[param_idx]
    param_end = bifurcation.bif_value * param_end_multiplier
    
    if null_simulation:
        # For null simulations, parameter is fixed at a point in the range
        # null_location is relative to the range from start to (bifurcation * multiplier)
        # This ensures param_end_multiplier affects the range used for null simulations
        param_fixed = param_start + null_location * (param_end - param_start)
        param_start = param_end = param_fixed
    
    # Time discretization
    # For simulations with param_end_multiplier > 1.0, extend simulation time
    # to ensure enough points before transition (forced) or after transition (null)
    if param_end_multiplier > 1.0:
        # Extend simulation time by multiplier
        total_time = series_len * dt_sample * param_end_multiplier
    else:
        total_time = series_len * dt_sample
    
    n_steps = int(total_time / dt_integrate)
    n_burn = int(burn_in_time / dt_integrate)
    
    # Parameter schedule: extend to cover the full simulation time
    param_schedule = jnp.linspace(param_start, param_end, n_steps)
    
    # Generate noise
    key, subkey1, subkey2 = jax.random.split(key, 3)
    noise_scale = sigma * jnp.sqrt(dt_integrate)
    if noise_burn is None:
        dW_burn = jax.random.normal(subkey1, (n_burn, 2)) * noise_scale
    else:
        dW_burn = jnp.asarray(noise_burn) * noise_scale
    if noise is None:
        dW = jax.random.normal(subkey2, (n_steps, 2)) * noise_scale
    else:
        dW = jnp.asarray(noise) * noise_scale
    
    # Burn-in: run at initial parameter
    pars_a, pars_b = pars[:10], pars[10:]
    burn_start = model.equi if initial_state is None else jnp.asarray(initial_state)
    carry_burn = (burn_start, pars_a, pars_b, dt_integrate)
    (state_after_burn, _, _, _), _ = jax.lax.scan(_burn_in_scan, carry_burn, dW_burn)
    
    # Main simulation: vectorized with scan
    # Create parameter arrays for each step
    pars_varying = jax.vmap(lambda p: pars.at[param_idx].set(p))(param_schedule)
    
    carry_sim = (state_after_burn, param_idx, dt_integrate)
    _, states = jax.lax.scan(_simulation_scan, carry_sim, (pars_varying, dW))
    
    # Downsample to output rate
    sample_every = int(dt_sample / dt_integrate)
    indices = jnp.arange(0, n_steps, sample_every)
    
    trajectory = states[indices]
    time = indices * dt_integrate
    param_values = param_schedule[indices]
    
    return time, trajectory, param_values, noise_scale


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


def simulate_synthetic_bifurcation_trajectory(
    model: PolynomialModel,
    bifurcation: BifurcationPoint,
    key: jax.random.PRNGKey,
    series_len: int = 500,
    steps_before_transition: int = 1,
    buffer_size: int = 1000,
    null_simulation: bool = False,
    **generate_kwargs
) -> Optional[TrajectoryResult]:
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
    buffer_size : int, default=1000
        Extra samples simulated prior to the transition to ensure enough
        pre-transition context. Keep this constant to preserve the
        bifurcation parameter ramp rate.
    null_simulation : bool, default=False
        If True, simulate null (non-tipping) trajectory
    **generate_kwargs : additional arguments
        Passed to simulate_to_bifurcation (dt_sample, dt_integrate, param_end_multiplier, etc.)
    
    Returns
    -------
    time : (series_len,) time array starting at 0
    trajectory : (series_len, 2) state array [x, y]
    label : str, bifurcation type ('fold', 'hopf', 'transcritical', 'null')
    sigma : float, effective noise amplitude used
    
    Returns None if transition occurs too early (< series_len points before)
    """
    if steps_before_transition > buffer_size:
        raise ValueError(
            "steps_before_transition exceeds buffer_size. "
            "Increase buffer_size to maintain constant parameter rate."
        )

    # Simulate with constant buffer to preserve bifurcation ramp rate
    total_len = series_len + buffer_size
    
    time, trajectory, param_values, sigma = simulate_to_bifurcation(
        model,
        bifurcation,
        key,
        series_len=total_len,
        null_simulation=null_simulation,
        param_end_multiplier=generate_kwargs.get('param_end_multiplier', 1.1)
    )
    
    # Detect transition (or use midpoint for null simulations)
    if null_simulation:
        # For null simulations, use midpoint as transition point
        trans_idx = len(trajectory) // 2
    else:
        trans_idx = detect_transition(trajectory, time)
    
    # Extract fixed-length series before transition
    time_seg, traj_seg, param_seg, trans_idx_out, success = extract_segment(
        time,
        trajectory,
        param_values,
        trans_idx,
        steps_before_transition,
        series_len
    )
    
    if not success:
        return None

    # Return results
    metadata: Dict[str, Any] = {
        'label': 'null' if null_simulation else bifurcation.bif_type,
        'steps_before_transition': steps_before_transition,
        'transition_index': int(trans_idx_out),
        'param_value_transition': float(param_values[min(trans_idx_out, param_values.shape[0] - 1)]),
    }

    full_trajectory = TrajectoryResult(
        time=time,
        trajectory=trajectory,
        param_values=param_values,
        sigma=sigma,
        metadata=metadata
    )

    part_trajectory = TrajectoryResult(
        time=time_seg,
        trajectory=traj_seg,
        param_values=param_seg,
        sigma=sigma,
        metadata=metadata
    )

    return (full_trajectory, part_trajectory)
