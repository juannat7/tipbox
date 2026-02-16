"""Bifurcation parameter schedule generation.

This module provides flexible parameter schedules for bifurcation simulations.
Schedules define how the bifurcation parameter changes over time, normalized to [0, 1].

Resolved schedules are exposed as ``param_normalized`` in TrajectoryResult metadata.
Normalization semantics differ by system type:

- **Synthetic bifurcation** (b_fold, b_hopf, b_transcritical): the schedule system
  produces [0, 1] directly where 0 = param_start and 1 = bif_value.
- **Rate-induced** (r_bautin, r_compost_bomb, r_saddle_node): normalized by max
  forcing value (0 = initial forcing, 1 = max forcing).
- **Real bifurcation** (b_harvesting): normalized by bifurcation point
  (0 = param_min, 1 = bif_value). Values > 1 are possible past bifurcation.

IMPORTANT: Schedules must start at 0. This is required because the burn-in phase
equilibrates the system at param_start (normalized=0). A non-zero start would cause
a sudden parameter jump at t=0, creating transient artifacts. Future versions may
support non-zero starts via an extended burn-in that slowly ramps to the schedule start.

Supported schedule types:
- linear: Linear ramp reaching 1.0 at a specified fraction
- ramp_and_hold: Ramp from 0 to a level, then hold constant
- fixed_slope: Ramp with fixed slope to a level, then hold (must start at 0 after clipping)
- constant: Fixed value (useful for null simulations, but must be 0 for tipping trajectories)
- piecewise_linear: Concatenated linear segments (first segment must start at 0)
- from_array: Direct array passthrough with interpolation (must start at 0)

Usage:
    # Factory function (direct)
    sched = schedule.ramp_and_hold(hold_at=0.5, hold_level=1.0)
    values = sched(n_steps=500)

    # Dict config (YAML-compatible)
    sched = schedule.create_schedule({
        'mode': 'ramp_and_hold',
        'ramp_and_hold': {'hold_at': 0.5, 'hold_level': 1.0}
    })
"""

from collections.abc import Callable
from typing import Any, Protocol

import jax.numpy as jnp


class ParameterSchedule(Protocol):
    """Protocol for parameter schedules.

    A schedule is a callable that takes the number of steps and returns
    a normalized array of values in [0, 1].
    """

    def __call__(self, n_steps: int) -> jnp.ndarray:
        """Generate schedule array.

        Args:
            n_steps: Number of output steps.

        Returns:
            Array of shape (n_steps,) with values in [0, 1].
        """
        ...


Schedule = jnp.ndarray | ParameterSchedule | dict[str, Any]


# === Factory Functions ===


def linear(critical_at: float = 1.0) -> ParameterSchedule:
    """Create a linear ramp schedule.

    The schedule ramps linearly from 0 to 1, reaching 1.0 at the specified
    fraction of the timeline. If critical_at < 1.0, the schedule continues
    past 1.0 at the same rate.

    Args:
        critical_at: Fraction of timeline (0-1) when value reaches 1.0.
            Default 1.0 means the ramp ends exactly at the end.

    Returns:
        ParameterSchedule callable.

    Example:
        >>> sched = linear(critical_at=0.5)
        >>> values = sched(100)  # reaches 1.0 at step 50, 2.0 at step 100
    """
    if critical_at <= 0:
        raise ValueError("critical_at must be positive")

    def schedule(n_steps: int) -> jnp.ndarray:
        end_value = 1.0 / critical_at
        return jnp.linspace(0.0, end_value, n_steps)

    return schedule


def ramp_and_hold(hold_at: float, hold_level: float = 1.0) -> ParameterSchedule:
    """Create a ramp-and-hold schedule starting from zero.

    The schedule ramps linearly from 0 to hold_level, reaching it at hold_at
    fraction of the timeline, then holds constant for the remainder.

    Args:
        hold_at: Fraction of timeline (0-1) when ramp ends and hold begins.
        hold_level: Normalized value (0-1) to ramp to and hold.
            Default 1.0 means ramping to the critical/bifurcation value.

    Returns:
        ParameterSchedule callable.

    Example:
        >>> sched = ramp_and_hold(hold_at=0.5, hold_level=1.0)
        >>> values = sched(100)  # ramps from 0 to 1.0 by step 50, then holds
    """
    if not 0 < hold_at <= 1:
        raise ValueError("hold_at must be in (0, 1]")

    def schedule(n_steps: int) -> jnp.ndarray:
        ramp_steps = max(int(n_steps * hold_at), 1)
        hold_steps = n_steps - ramp_steps
        ramp_part = jnp.linspace(0.0, hold_level, ramp_steps)
        hold_part = jnp.full(hold_steps, hold_level)
        return jnp.concatenate([ramp_part, hold_part])

    return schedule


def fixed_slope(
    slope: float, hold_at: float, hold_level: float = 1.0, clip_min: float | None = 0.0
) -> ParameterSchedule:
    """Create a fixed-slope ramp-and-hold schedule.

    Unlike ramp_and_hold which always starts at 0, this schedule maintains a
    fixed slope during the ramp phase. The start value is computed from the
    slope: start = hold_level - slope * hold_at.

    By default, values are clipped at 0 to avoid negative parameter values.

    This is useful when you want consistent rate of parameter change regardless
    of where the hold point is.

    NOTE: The schedule must start at 0 after clipping. This means you need
    slope >= hold_level / hold_at, or clip_min=0.0 (default) to ensure
    non-zero computed starts are clipped to 0.

    Args:
        slope: Rate of change per unit time (in normalized coordinates).
            A slope of 2.0 means the value increases by 2.0 over the full timeline.
        hold_at: Fraction of timeline (0-1) when ramp ends and hold begins.
        hold_level: Normalized value (0-1) to ramp to and hold.
            Default 1.0 means ramping to the critical/bifurcation value.
        clip_min: Minimum value to clip the schedule at. Default 0.0.
            Set to None to allow negative values (will fail validation if start > 0).

    Returns:
        ParameterSchedule callable.

    Example:
        >>> # Slope of 2.0, hold at 50% -> starts at 0.0, reaches 1.0 at 50%
        >>> sched = fixed_slope(slope=2.0, hold_at=0.5, hold_level=1.0)
        >>> values = sched(100)
        >>>
        >>> # Slope of 1.0, hold at 70% -> computed start 0.3, clipped to 0
        >>> sched = fixed_slope(slope=1.0, hold_at=0.7, hold_level=1.0)  # OK, clipped
        >>>
        >>> # This would fail validation (start=0.3, no clipping):
        >>> # sched = fixed_slope(slope=1.0, hold_at=0.7, hold_level=1.0, clip_min=None)
    """
    if not 0 < hold_at <= 1:
        raise ValueError("hold_at must be in (0, 1]")
    if slope <= 0:
        raise ValueError("slope must be positive")

    # Calculate start value from slope
    start_value = hold_level - slope * hold_at

    def schedule(n_steps: int) -> jnp.ndarray:
        ramp_steps = max(int(n_steps * hold_at), 1)
        hold_steps = n_steps - ramp_steps
        ramp_part = jnp.linspace(start_value, hold_level, ramp_steps)
        hold_part = jnp.full(hold_steps, hold_level)
        result = jnp.concatenate([ramp_part, hold_part])
        if clip_min is not None:
            result = jnp.maximum(result, clip_min)
        return result

    return schedule


def constant(value: float = 0.0) -> ParameterSchedule:
    """Create a constant value schedule.

    Useful for null simulations where the parameter doesn't change.

    Args:
        value: Constant value in [0, 1]. Default 0.0.

    Returns:
        ParameterSchedule callable.

    Example:
        >>> sched = constant(value=0.5)
        >>> values = sched(100)  # all values are 0.5
    """

    def schedule(n_steps: int) -> jnp.ndarray:
        return jnp.full(n_steps, value)

    return schedule


def piecewise_linear(segments: list[dict[str, float]]) -> ParameterSchedule:
    """Create a piecewise linear schedule from segments.

    Each segment defines a linear ramp over a fraction of the total time.

    Args:
        segments: List of segment dicts, each with:
            - start: Starting value (0-1)
            - end: Ending value (0-1)
            - fraction: Proportion of total steps for this segment

    Returns:
        ParameterSchedule callable.

    Example:
        >>> # Slow ramp, fast ramp, hold
        >>> segments = [
        ...     {'start': 0.0, 'end': 0.5, 'fraction': 0.5},
        ...     {'start': 0.5, 'end': 1.0, 'fraction': 0.2},
        ...     {'start': 1.0, 'end': 1.0, 'fraction': 0.3},
        ... ]
        >>> sched = piecewise_linear(segments)
    """
    if not segments:
        raise ValueError("segments list cannot be empty")

    total_fraction = sum(seg.get("fraction", 0) for seg in segments)
    if not 0.99 <= total_fraction <= 1.01:
        raise ValueError(f"Segment fractions must sum to 1.0, got {total_fraction}")

    def schedule(n_steps: int) -> jnp.ndarray:
        parts = []
        remaining_steps = n_steps

        for i, seg in enumerate(segments):
            start = seg["start"]
            end = seg["end"]
            fraction = seg["fraction"]

            # Last segment gets remaining steps to avoid rounding issues
            if i == len(segments) - 1:
                seg_steps = remaining_steps
            else:
                seg_steps = max(int(n_steps * fraction), 1)
                remaining_steps -= seg_steps

            if seg_steps > 0:
                part = jnp.linspace(start, end, seg_steps + 1)[:-1]
                parts.append(part)

        # Add final value
        parts.append(jnp.array([segments[-1]["end"]]))
        result = jnp.concatenate(parts)

        # Trim or pad to exact length
        if len(result) > n_steps:
            result = result[:n_steps]
        elif len(result) < n_steps:
            result = jnp.concatenate([result, jnp.full(n_steps - len(result), segments[-1]["end"])])

        return result

    return schedule


def from_array(values: jnp.ndarray) -> ParameterSchedule:
    """Create a schedule from a pre-computed array.

    The array is interpolated to the requested length.

    Args:
        values: Array of values to use as the schedule.

    Returns:
        ParameterSchedule callable.

    Example:
        >>> custom = jnp.concatenate([jnp.linspace(0, 0.5, 250), jnp.full(250, 0.5)])
        >>> sched = from_array(custom)
    """
    arr = jnp.asarray(values)

    def schedule(n_steps: int) -> jnp.ndarray:
        if len(arr) == n_steps:
            return arr
        return jnp.interp(jnp.linspace(0, 1, n_steps), jnp.linspace(0, 1, len(arr)), arr)

    return schedule


# === Registry & Config-Driven Creation ===

SCHEDULES: dict[str, Callable[..., ParameterSchedule]] = {
    "linear": linear,
    "ramp_and_hold": ramp_and_hold,
    "fixed_slope": fixed_slope,
    "constant": constant,
    "piecewise_linear": piecewise_linear,
    "from_array": from_array,
}


def create_schedule(config: dict[str, Any]) -> ParameterSchedule:
    """Create a schedule from a dict config.

    The 'mode' key selects the factory; mode-specific args are in a nested dict.

    Args:
        config: Dict with 'mode' key and mode-specific parameters.

    Returns:
        ParameterSchedule callable.

    Examples:
        >>> create_schedule({'mode': 'linear', 'linear': {'critical_at': 0.5}})
        >>> create_schedule({'mode': 'ramp_and_hold', 'ramp_and_hold': {'hold_at': 0.5}})
        >>> create_schedule({'mode': 'constant', 'constant': {'value': 0.5}})
    """
    mode = config.get("mode", "linear")
    if mode not in SCHEDULES:
        raise ValueError(f"Unknown schedule mode: {mode}. Available: {list(SCHEDULES.keys())}")
    factory = SCHEDULES[mode]
    mode_args = config.get(mode, {})
    return factory(**mode_args)


def legacy_multiplier_schedule(param_end_multiplier: float) -> ParameterSchedule:
    """Create schedule equivalent to legacy param_end_multiplier behavior.

    The legacy behavior ramps linearly from param_start to bif_value * multiplier.
    In normalized [0, 1] terms with param_start as 0 and bif_value as 1,
    this means ramping from 0 to `multiplier`.

    Args:
        param_end_multiplier: Multiplier for the bifurcation value.

    Returns:
        ParameterSchedule callable.
    """

    def schedule(n_steps: int) -> jnp.ndarray:
        return jnp.linspace(0.0, param_end_multiplier, n_steps)

    return schedule


# === Resolution Handling ===


def resolve_schedule(schedule: Schedule, n_steps: int) -> jnp.ndarray:
    """Resolve any schedule type to a concrete array.

    Args:
        schedule: Array, callable, or dict config.
        n_steps: Output length.

    Returns:
        Array of shape (n_steps,) with values in [0, 1].
    """
    if isinstance(schedule, dict):
        schedule = create_schedule(schedule)

    if callable(schedule):
        result = schedule(n_steps)
    else:
        arr = jnp.asarray(schedule)
        if len(arr) == n_steps:
            result = arr
        else:
            # Interpolate to requested length
            result = jnp.interp(jnp.linspace(0, 1, n_steps), jnp.linspace(0, 1, len(arr)), arr)

    return result


def validate_schedule_start(normalized: jnp.ndarray) -> None:
    """Validate that a schedule starts at 0.

    This is required for tipping simulations because burn-in runs at param_start
    (normalized=0) and computes equilibrium there. A non-zero start would cause
    a sudden parameter jump at t=0, creating transient artifacts.

    Future versions may support non-zero starts by implementing a "slow ramp to
    schedule start" during an extended burn-in phase.

    Args:
        normalized: Resolved schedule array.

    Raises:
        ValueError: If schedule does not start at 0.
    """
    start_val = float(normalized[0])
    if abs(start_val) > 1e-6:
        raise ValueError(
            f"Schedule must start at 0, got {start_val:.4f}. "
            f"This is required because burn-in equilibrates at param_start (normalized=0). "
            f"Use ramp_and_hold or linear instead of fixed_slope with non-zero start."
        )


def resolve_schedule_for_simulation(
    schedule: Schedule,
    n_output: int,
    param_start: float,
    param_end: float,
    upsample_factor: int = 1,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Resolve schedule and map to model parameter range at integration resolution.

    Combines resolve_schedule() + validate_schedule_start() + apply_to_params()
    into a single call. Optionally upsamples for models that integrate at finer
    resolution than output (semi-real models with dt << dt_sample).

    Args:
        schedule: Schedule specification (dict config, callable, or array).
        n_output: Number of output time points.
        param_start: Parameter value at schedule=0 (equilibrium).
        param_end: Parameter value at schedule=1 (critical/max forcing).
        upsample_factor: Ratio of fine integration steps to output steps.
            1 = no upsampling (rate models integrate at output resolution).
            >1 = upsample via linear interpolation (semi-real models).

    Returns:
        (param_values, normalized) tuple:
        - param_values: Parameter values at integration resolution.
        - normalized: Normalized schedule [0,1] at output resolution (n_output).
    """
    normalized = resolve_schedule(schedule, n_output)
    validate_schedule_start(normalized)

    if upsample_factor <= 1:
        param_values = apply_to_params(normalized, param_start, param_end)
        return param_values, normalized

    # Upsample to fine integration resolution
    # n_fine = n_output * upsample_factor so that downsampling by
    # upsample_factor yields exactly n_output points.
    param_coarse = apply_to_params(normalized, param_start, param_end)
    n_fine = n_output * upsample_factor
    param_fine = jnp.interp(
        jnp.linspace(0, 1, n_fine),
        jnp.linspace(0, 1, n_output),
        param_coarse,
    )
    return param_fine, normalized


def apply_to_params(schedule: jnp.ndarray, param_start: float, bif_value: float) -> jnp.ndarray:
    """Convert normalized [0,1] schedule to absolute parameter values.

    Args:
        schedule: Normalized schedule array.
        param_start: Starting parameter value.
        bif_value: Bifurcation parameter value (corresponds to schedule=1.0).

    Returns:
        Array of absolute parameter values.
    """
    return param_start + schedule * (bif_value - param_start)
