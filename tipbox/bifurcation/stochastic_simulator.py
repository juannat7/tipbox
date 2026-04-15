"""
Stochastic simulation of polynomial dynamical systems approaching bifurcations.

Implements Euler-Maruyama method with:
- Linearly increasing bifurcation parameter
- Additive Gaussian white noise
- Noise amplitude scaling based on recovery rate
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import ruptures as rpt
from statsmodels.nonparametric.smoothers_lowess import lowess

from ..types import TrajectoryResult
from ..utils import extract_segment
from .bifurcation_finder import BifurcationPoint
from .polynomial_model import PolynomialModel, polynomial_basis
from .schedule import (
    Schedule,
    apply_to_params,
    constant,
    legacy_multiplier_schedule,
    linear,
    resolve_schedule,
    validate_schedule_start,
)


@jax.jit
def euler_maruyama_step(
    state: jnp.ndarray, pars_a: jnp.ndarray, pars_b: jnp.ndarray, dt: float, dW: jnp.ndarray
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
    param_end_multiplier: float | None = 1.1,
    schedule: Schedule | None = None,
    initial_state: jnp.ndarray | None = None,
    noise_burn: jnp.ndarray | None = None,
    noise: jnp.ndarray | None = None,
    rv_tri: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray]:
    """
    Simulate system approaching bifurcation with stochastic noise.

    - Bifurcation parameter changes according to the schedule
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
    param_end_multiplier : Optional[float], default=1.1
        Legacy multiplier for extending simulation past bif_value.
        Ignored if schedule is provided.
    schedule : Optional[Schedule], default=None
        Parameter schedule (array, callable, or dict config).
        If provided, takes precedence over param_end_multiplier.
        See tipbox.bifurcation.schedule for available schedule types.

    Returns
    -------
    time : (series_len,) array of time points
    trajectory : (series_len, 2) array of [x, y] states
    param_values : (series_len,) array of bifurcation parameter values
    noise_scale : float, effective noise scale used
    normalized : (series_len,) array of normalized schedule values in [0,1]
    """
    # Noise amplitude: sigma = sqrt(2*lambda) * sigma_tilde * triangular_rv
    key, subkey = jax.random.split(key)
    if rv_tri is None:
        rv_tri = jax.random.triangular(subkey, 0.75, 1.0, 1.25) if noise_scale_triangular else 1.0
    sigma = jnp.sqrt(2 * model.recovery_rate) * sigma_tilde * rv_tri

    # Parameter schedule setup
    pars = model.get_parameters()
    param_idx = bifurcation.bif_param_index
    param_start = pars[param_idx]
    bif_value = bifurcation.bif_value

    # Time discretization
    sample_every = int(dt_sample / dt_integrate)

    # Determine effective schedule (all paths go through schedule system)
    if null_simulation:
        # Null: constant schedule at null_location
        effective_schedule = constant(null_location)
    elif schedule is not None:
        # Explicit schedule provided (array, callable, or dict)
        effective_schedule = schedule
    elif param_end_multiplier is not None:
        # Legacy API: translate multiplier to schedule
        effective_schedule = legacy_multiplier_schedule(param_end_multiplier)
    else:
        # Default: linear ramp to bif_value
        effective_schedule = linear()

    # For legacy mode with param_end_multiplier > 1.0, extend total time
    # to match original behavior (but this is usually not needed with schedule)
    if schedule is None and param_end_multiplier is not None and param_end_multiplier > 1.0:
        total_time = series_len * dt_sample * param_end_multiplier
    else:
        total_time = series_len * dt_sample

    n_steps = int(total_time / dt_integrate)
    n_burn = int(burn_in_time / dt_integrate)

    # Calculate actual number of output samples (may differ from series_len when extended)
    n_output_samples = len(range(0, n_steps, sample_every))

    # Resolve schedule to array at output resolution
    normalized, _sched_meta = resolve_schedule(effective_schedule, n_output_samples)

    # Validate that non-null schedules start at 0 (null simulations use constant schedules)
    if not null_simulation:
        validate_schedule_start(normalized)

    # Interpolate to fine integration resolution
    normalized_fine = jnp.interp(jnp.arange(n_steps), jnp.arange(0, n_steps, sample_every), normalized)

    # Apply to parameter range
    param_schedule = apply_to_params(normalized_fine, param_start, bif_value)

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

    return time, trajectory, param_values, noise_scale, normalized


def detect_transition(trajectory: jnp.ndarray, time: jnp.ndarray | None = None) -> int:
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
    schedule: Schedule | None = None,
    param_end_multiplier: float | None = 1.1,
    **simulate_kwargs,
) -> TrajectoryResult | None:
    """
    Generate a single training trajectory (either tipping or null).

    This is the high-level function that wraps the PNAS approach:
    1. Simulate system to bifurcation
    2. Detect transition point
    3. Extract series_len points before transition

    When a schedule is provided, uses direct mode:
    - Simulates exact series_len points
    - Transition occurs at pre-determined position based on schedule
    - No detection or extraction needed

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
        Only used in legacy mode (when schedule is None).
    buffer_size : int, default=1000
        Extra samples simulated prior to the transition to ensure enough
        pre-transition context. Keep this constant to preserve the
        bifurcation parameter ramp rate. Only used in legacy mode.
    null_simulation : bool, default=False
        If True, simulate null (non-tipping) trajectory
    schedule : Schedule, optional
        Parameter schedule (array, callable, or dict config).
        When provided, uses direct mode with exact length and pre-determined
        transition position. See tipbox.bifurcation.schedule for available types.
    param_end_multiplier : float, optional, default=1.1
        Multiplier for extending simulation past bifurcation value.
        Only used in legacy mode (when schedule is None).
    **simulate_kwargs : additional arguments
        Passed to simulate_to_bifurcation. Common parameters include:
        - dt_sample: float - sampling interval (default 1.0)
        - dt_integrate: float - integration time step (default 0.01)
        - sigma_tilde: float - base noise amplitude (default 0.01)
        - burn_in_time: float - burn-in period (default 100.0)

    Returns
    -------
    tuple[TrajectoryResult, TrajectoryResult]
        Tuple of (full_trajectory, part_trajectory). In schedule mode, both are
        identical. In legacy mode, part_trajectory is the extracted segment.
        Returns None if transition occurs too early in legacy mode.

        Each TrajectoryResult.metadata includes:
        - label: bifurcation type or "null"
        - transition_index: detected/estimated transition position
        - param_value_transition: parameter value at transition
        - param_normalized: normalized parameter values in [0,1] matching trajectory length
    """

    if schedule is not None:
        from .schedule import prepare_schedule_with_warmup

        # Prepare schedule: prepend warmup ramp if non-zero start
        dt_sample = simulate_kwargs.get("dt_sample", 1.0)
        prepared = prepare_schedule_with_warmup(schedule, series_len, dt_sample=dt_sample)

        # Direct mode: simulate (including warmup prefix)
        time, trajectory, param_values, sigma, normalized = simulate_to_bifurcation(
            model,
            bifurcation,
            key,
            series_len=prepared.total_output_steps,
            null_simulation=null_simulation,
            schedule=prepared.schedule,
            param_end_multiplier=None,  # schedule takes precedence
            **simulate_kwargs,
        )

        # Trim warmup prefix from outputs
        time, trajectory, param_values, normalized = prepared.trim(
            time, trajectory, param_values, normalized
        )

        # Estimate transition index from schedule
        # Find first index where normalized >= 1.0 (reaching bif_value)
        above_threshold = jnp.where(normalized >= 1.0, size=series_len, fill_value=series_len - 1)
        trans_idx = int(above_threshold[0][0]) if len(above_threshold[0]) > 0 else series_len - 1

        # Return full trajectory (no extraction needed)
        metadata: dict[str, Any] = {
            "label": "null" if null_simulation else bifurcation.bif_type,
            "transition_index": trans_idx,
            "param_value_transition": float(param_values[min(trans_idx, len(param_values) - 1)]),
            "param_normalized": normalized,
            "schedule_metadata": prepared.metadata,
        }

        result = TrajectoryResult(
            time=time, trajectory=trajectory, param_values=param_values, sigma=sigma, metadata=metadata
        )
        return (result, result)  # full and part are identical in direct mode

    # Legacy mode: use buffer + detection + extraction
    if steps_before_transition > buffer_size:
        raise ValueError(
            "steps_before_transition exceeds buffer_size. Increase buffer_size to maintain constant parameter rate."
        )

    # Simulate with constant buffer to preserve bifurcation ramp rate
    total_len = series_len + buffer_size

    time, trajectory, param_values, sigma, normalized = simulate_to_bifurcation(
        model,
        bifurcation,
        key,
        series_len=total_len,
        null_simulation=null_simulation,
        param_end_multiplier=param_end_multiplier,
        **simulate_kwargs,
    )

    # Detect transition (or use midpoint for null simulations)
    if null_simulation:
        # For null simulations, use midpoint as transition point
        trans_idx = len(trajectory) // 2
    else:
        trans_idx = detect_transition(trajectory, time)

    # Extract fixed-length series before transition
    time_seg, traj_seg, param_seg, trans_idx_out, success = extract_segment(
        time, trajectory, param_values, trans_idx, steps_before_transition, series_len
    )

    if not success:
        return None

    # Slice the normalized schedule to match the extracted segment
    # extract_segment uses: start = trans_idx - series_len + steps_before_transition + 1
    start_idx = trans_idx - series_len + steps_before_transition + 1
    end_idx = start_idx + series_len
    param_normalized_seg = normalized[start_idx:end_idx]

    # Return results
    full_metadata: dict[str, Any] = {
        "label": "null" if null_simulation else bifurcation.bif_type,
        "steps_before_transition": steps_before_transition,
        "transition_index": int(trans_idx_out),
        "param_value_transition": float(param_values[min(trans_idx_out, param_values.shape[0] - 1)]),
        "param_normalized": normalized,
    }

    part_metadata: dict[str, Any] = {
        "label": "null" if null_simulation else bifurcation.bif_type,
        "steps_before_transition": steps_before_transition,
        "transition_index": int(trans_idx_out),
        "param_value_transition": float(param_values[min(trans_idx_out, param_values.shape[0] - 1)]),
        "param_normalized": param_normalized_seg,
    }

    full_trajectory = TrajectoryResult(
        time=time, trajectory=trajectory, param_values=param_values, sigma=sigma, metadata=full_metadata
    )

    part_trajectory = TrajectoryResult(
        time=time_seg, trajectory=traj_seg, param_values=param_seg, sigma=sigma, metadata=part_metadata
    )

    return (full_trajectory, part_trajectory)
