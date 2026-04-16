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

Schedules may start at any value in [0, 1].  When a schedule starts at a non-zero
value, use ``prepare_schedule_with_warmup`` to prepend an exponential ramp from 0.
This lets the system equilibrate at the desired parameter level before the recorded
trajectory begins.  The ``schedule_warmup_time`` key in the schedule dict controls
the ramp duration (default 100.0 time-units).

Supported schedule types:
- linear: Linear ramp from 0, reaching 1.0 at a specified fraction
- linear_ramp: Linear interpolation between arbitrary start and end values
- ramp_and_hold: Ramp from 0 to a level, then hold constant
- fixed_slope: Ramp with fixed slope to a level, then hold
- constant: Fixed value (useful for null simulations or stationary forcing)
- piecewise_linear: Concatenated linear segments
- bezier_and_hold: Bezier ramp to a level, then hold constant
- from_array: Direct array passthrough with interpolation

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

import random
from collections.abc import Callable
from dataclasses import dataclass
from math import comb
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


def linear_ramp(start: float = 0.0, end: float = 1.0) -> ParameterSchedule:
    """Create a linear ramp schedule from start to end.

    Unlike ``linear()`` which always starts at 0, this schedule supports
    arbitrary start and end values.  Useful for validation datasets with
    non-zero ``param_normalized_start``.

    Args:
        start: Starting normalized value.
        end: Ending normalized value.

    Returns:
        ParameterSchedule callable.

    Example:
        >>> sched = linear_ramp(start=0.3, end=0.7)
        >>> values = sched(100)  # linearly from 0.3 to 0.7
    """

    def schedule(n_steps: int) -> jnp.ndarray:
        return jnp.linspace(start, end, n_steps)

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


def _bezier_curve(points: list[float], t_values: jnp.ndarray) -> jnp.ndarray:
    """Evaluate a 1D Bezier curve at t_values."""
    degree = len(points) - 1
    if degree < 1:
        raise ValueError("Bezier curve requires at least 2 control points")

    result = jnp.zeros_like(t_values)
    for i, pt in enumerate(points):
        basis = comb(degree, i) * (1.0 - t_values) ** (degree - i) * (t_values**i)
        result = result + pt * basis
    return result


def _resolve_control_points(
    control_points: "list[float] | dict[str, Any] | None",
    start: float,
    final_level: float,
) -> list[float]:
    """Resolve control_points to a concrete list of floats.

    Accepts:
    - ``None``: single midpoint default.
    - ``list[float]``: used as-is.
    - ``dict``: sample ``num_points`` values uniformly from ``value_range``.
      Keys:
        num_points (int, required): number of interior control points (>= 1).
        value_range ([lo, hi], required): sampling bounds.
        relative_to_final (bool, default False): if True, multiply
            value_range entries by ``final_level`` before sampling.
        seed (int | None, default None): RNG seed for reproducibility.
            When None, a fresh random sample is drawn each call.
    """
    if control_points is None:
        return [0.5 * (start + final_level)]

    if isinstance(control_points, list):
        return list(control_points)

    if not isinstance(control_points, dict):
        raise TypeError(f"control_points must be list, dict, or None, got {type(control_points).__name__}")

    # Dict-based sampling path
    if "num_points" not in control_points:
        raise ValueError("control_points dict must contain 'num_points'")
    if "value_range" not in control_points:
        raise ValueError("control_points dict must contain 'value_range'")

    num = control_points["num_points"]
    if not isinstance(num, int) or num < 1:
        raise ValueError(f"num_points must be an integer >= 1, got {num}")

    vr = control_points["value_range"]
    if not (isinstance(vr, (list, tuple)) and len(vr) == 2):
        raise ValueError(f"value_range must be [lo, hi], got {vr}")

    lo, hi = float(vr[0]), float(vr[1])
    if lo > hi:
        raise ValueError(f"value_range[0] must be <= value_range[1], got [{lo}, {hi}]")

    if control_points.get("relative_to_final", False):
        lo *= final_level
        hi *= final_level

    seed = control_points.get("seed", None)
    rng = random.Random(seed)
    return [rng.uniform(lo, hi) for _ in range(num)]


def bezier_and_hold(
    hold_at: float,
    final_level: float = 1.0,
    control_points: "list[float] | dict[str, Any] | None" = None,
    start: float = 0.0,
    max_abs_slope: float | None = None,
    smooth_hold: bool = True,
) -> ParameterSchedule:
    """Create a Bezier ramp to final_level at hold_at, then hold.

    Control points are equally spaced in time over [0, hold_at]. Values are
    constrained to [0, final_level] to avoid negative forcing and overshoot.

    Args:
        hold_at: Fraction of timeline (0, 1] when hold begins.
        final_level: Value to ramp to and hold.
        control_points: Interior Bezier control points. Accepts:
            - list[float]: explicit values.
            - dict: sampled controls with keys ``num_points``, ``value_range``,
              optional ``relative_to_final`` (bool, default False),
              optional ``seed`` (int | None, default None for stochastic).
            - None: single midpoint default.
        start: Starting value (must be <= final_level).
        max_abs_slope: Optional cap on realized slope magnitude.
        smooth_hold: If True (default), duplicate the final control point so
            the Bezier tangent is ~zero at the hold boundary (C1-smooth join).
            If False, the ramp may have non-zero slope entering the hold.
    """
    if not 0 < hold_at <= 1:
        raise ValueError("hold_at must be in (0, 1]")
    if final_level < 0:
        raise ValueError("final_level must be non-negative")
    if start < 0:
        raise ValueError("start must be non-negative")
    if start > final_level:
        raise ValueError("start must be <= final_level")
    if max_abs_slope is not None and max_abs_slope <= 0:
        raise ValueError("max_abs_slope must be positive")

    interior = _resolve_control_points(control_points, start, final_level)
    pts = [float(start), *[float(v) for v in interior], float(final_level)]

    # Prevent overshoot and negative values.
    pts = [float(min(max(v, 0.0), final_level)) for v in pts]
    pts[0] = float(start)
    pts[-1] = float(final_level)

    # Optionally add an extra terminal control point equal to final_level.
    # This forces the Bezier derivative to zero at t=1, creating a smooth
    # (C1) transition into the constant hold regime.
    if smooth_hold:
        pts = pts + [float(final_level)]

    n_pts = len(pts)
    if n_pts < 2:
        raise ValueError("Need at least 2 Bezier points")

    if max_abs_slope is not None:
        if abs(final_level - start) > max_abs_slope * hold_at:
            raise ValueError("max_abs_slope is too small to connect start to final_level within hold_at")

        dt_cp = hold_at / float(n_pts - 1)
        max_delta = max_abs_slope * dt_cp

        for i in range(1, n_pts - 1):
            remaining = n_pts - 1 - i
            lower = max(
                0.0,
                pts[i - 1] - max_delta,
                final_level - remaining * max_delta,
            )
            upper = min(
                final_level,
                pts[i - 1] + max_delta,
                final_level + remaining * max_delta,
            )
            if lower > upper:
                midpoint = 0.5 * (lower + upper)
                lower = midpoint
                upper = midpoint
            pts[i] = float(min(max(pts[i], lower), upper))

    def schedule(n_steps: int) -> jnp.ndarray:
        ramp_steps = max(int(n_steps * hold_at), 1)
        hold_steps = n_steps - ramp_steps

        t = jnp.linspace(0.0, 1.0, ramp_steps)
        ramp = _bezier_curve(pts, t)
        ramp = jnp.clip(ramp, 0.0, final_level)

        hold = jnp.full(hold_steps, final_level)
        return jnp.concatenate([ramp, hold])

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


# === Schedule Warmup ===


@dataclass
class ScheduleWithWarmup:
    """Preprocessed schedule with warmup metadata.

    When a schedule starts at a non-zero ``param_normalized`` value, the system
    needs time to equilibrate at that parameter level before the recorded
    trajectory begins.  ``prepare_schedule_with_warmup`` creates this object by
    prepending an exponential ramp from 0 to the schedule's start value.

    The ``schedule`` attribute always starts at 0 and can be passed directly to
    any simulator.  After simulation, call ``trim`` on the output arrays to
    remove the warmup prefix.
    """

    schedule: "ParameterSchedule"
    """Schedule callable that starts at 0 (warmup prepended if needed)."""

    total_output_steps: int
    """Total output steps including warmup."""

    warmup_output_steps: int
    """Number of output steps consumed by warmup (0 if no warmup)."""

    metadata: "dict[str, Any]"
    """Schedule metadata from the factory (e.g. ``linear_classified`` group info)."""

    def trim(self, *arrays: jnp.ndarray) -> "jnp.ndarray | tuple[jnp.ndarray, ...]":
        """Remove warmup prefix from downsampled output arrays.

        Args:
            *arrays: One or more arrays whose first axis corresponds to output
                time steps (including warmup).

        Returns:
            Single trimmed array if one argument is given, otherwise a tuple of
            trimmed arrays.
        """
        if self.warmup_output_steps == 0:
            return arrays[0] if len(arrays) == 1 else arrays
        result = tuple(a[self.warmup_output_steps :] for a in arrays)
        return result[0] if len(result) == 1 else result


def prepare_schedule_with_warmup(
    schedule: "Schedule",
    n_output: int,
    dt_sample: float = 1.0,
) -> ScheduleWithWarmup:
    """Preprocess a schedule for simulation, adding warmup if needed.

    If the resolved schedule starts at a non-zero value **and**
    ``schedule_warmup_time > 0``, an exponential ramp from 0 to the schedule's
    start value is prepended.  The returned schedule always starts at 0 and can
    be passed to any simulator without triggering the start-at-zero validation.

    ``schedule_warmup_time`` is read from the schedule dict config (key
    ``"schedule_warmup_time"``, default ``100.0`` time-units).  The warmup
    triggers iff ``schedule_warmup_time > 0`` AND the schedule start is not 0.

    Args:
        schedule: Schedule specification (dict config, callable, or array).
            If a dict, it may contain a ``"schedule_warmup_time"`` key (float,
            in time units matching ``dt_sample``).
        n_output: Number of output steps for the user's schedule (excluding
            any warmup).
        dt_sample: Sampling interval used to convert ``schedule_warmup_time``
            (time units) into output steps.

    Returns:
        ``ScheduleWithWarmup`` containing the preprocessed schedule, the total
        output step count (including warmup), and the number of warmup steps
        to trim from the output.
    """
    # Extract schedule_warmup_time from dict config (if present)
    warmup_time: float = 0.0
    if isinstance(schedule, dict):
        warmup_time = float(schedule.get("schedule_warmup_time", 100.0))

    # Resolve the user's schedule to a concrete array
    resolved, metadata = resolve_schedule(schedule, n_output)
    start_val = float(resolved[0])

    # No warmup needed: schedule starts at 0 or warmup disabled
    if abs(start_val) < 1e-6 or warmup_time <= 0:
        # Wrap the original schedule as-is
        if callable(schedule) and not isinstance(schedule, dict):
            sched_callable = schedule
        else:
            sched_callable = from_array(resolved)
        return ScheduleWithWarmup(
            schedule=sched_callable,
            total_output_steps=n_output,
            warmup_output_steps=0,
            metadata=metadata,
        )

    # Build exponential ramp from 0 → start_val
    ramp_steps = max(int(warmup_time / dt_sample), 1)
    tau = ramp_steps / 8.0  # reaches 99% at ~58% of ramp; ~42% spent settled at target
    t = jnp.arange(ramp_steps, dtype=jnp.float32)
    ramp = start_val * (1.0 - jnp.exp(-t / tau))

    # Concatenate ramp + original schedule
    combined = jnp.concatenate([ramp, resolved])
    wrapped = from_array(combined)

    return ScheduleWithWarmup(
        schedule=wrapped,
        total_output_steps=n_output + ramp_steps,
        warmup_output_steps=ramp_steps,
        metadata=metadata,
    )


# === Classified Linear Schedule ===

_LC_GROUPS = (
    "critical",
    "nc_flat",
    "nc_approaching",
    "nc_receding",
    "nc_equilibrium",
    "noncritical",
)


def linear_classified(
    group: str = "noncritical",
    start_min: float = 0.0,
    start_max: float = 0.8,
    critical_threshold: float = 1.0,
    noncritical_ceil: float = 0.9,
    constant_angle_deg: float = 5.0,
    crit_cross_not_before: float = 0.65,
    crit_cross_not_after: float = 0.90,
    zero_cross_not_before: float = 1.0,
    zero_cross_not_after: float | None = None,
    clip_min: float | None = 0.0,
    clip_max: float | None = 1.0,
    seed: int | None = None,
) -> tuple[ParameterSchedule, dict[str, Any]]:
    """Create a classified linear forcing schedule.

    Samples ``(start, angle)`` uniformly in angle space within the feasible
    region for the requested group, then returns a linear schedule together
    with classification metadata.

    By default, values are clipped to ``[0, 1]``.  Set ``clip_min`` /
    ``clip_max`` to ``None`` to disable the respective bound.

    Groups:
        ``critical``: schedule reaches ``critical_threshold`` between
        ``crit_cross_not_before`` and ``crit_cross_not_after``.

        ``nc_flat``: near-zero slope (``|angle| < constant_angle_deg``),
        stays below ``noncritical_ceil``.

        ``nc_approaching``: positive slope, stays below ``noncritical_ceil``.

        ``nc_receding``: negative slope, doesn't cross 0 before
        ``zero_cross_not_before``.  If ``zero_cross_not_after`` is set, must
        cross 0 by that fraction.

        ``nc_equilibrium``: constant at zero (``start=0, slope=0``).  The
        system stays at its initial equilibrium throughout.

        ``noncritical``: umbrella — picks one of the four ``nc_*`` sub-types
        with uniform probability.

    Args:
        group: One of ``critical``, ``nc_flat``, ``nc_approaching``,
            ``nc_receding``, ``nc_equilibrium``, or ``noncritical``.
        start_min: Minimum start value (param_normalized at t=0).
        start_max: Maximum start value.
        critical_threshold: Value that defines "reaching bifurcation".
        noncritical_ceil: Non-critical schedules stay below this.
        constant_angle_deg: Angle threshold (degrees) for the constant regime.
        crit_cross_not_before: Earliest fraction at which a critical schedule
            may cross ``critical_threshold``.
        crit_cross_not_after: Latest fraction by which a critical schedule must
            have crossed ``critical_threshold``.
        zero_cross_not_before: Earliest fraction at which a receding schedule
            may cross zero.  Default ``1.0`` means never crosses during the
            trajectory.
        zero_cross_not_after: Latest fraction by which a receding schedule must
            cross zero.  ``None`` (default) means no requirement to cross.
        clip_min: Lower clip bound.  ``None`` to disable.
        clip_max: Upper clip bound.  ``None`` to disable.
        seed: RNG seed.  ``None`` for stochastic sampling.

    Returns:
        ``(schedule, metadata)`` tuple.
    """
    import math

    if group not in _LC_GROUPS:
        raise ValueError(f"Unknown group: {group!r}. Must be one of {_LC_GROUPS}")

    rng = random.Random(seed)
    const_slope = math.tan(math.radians(constant_angle_deg))

    # Resolve umbrella group
    resolved_group = group
    if group == "noncritical":
        resolved_group = rng.choice(
            ["nc_flat", "nc_approaching", "nc_receding", "nc_equilibrium"]
        )

    # Equilibrium is deterministic — no sampling needed
    if resolved_group == "nc_equilibrium":
        start = 0.0
        slope = 0.0
        angle_deg = 0.0
    else:

        def _sample_slope_uniform_angle(slope_lo: float, slope_hi: float) -> tuple[float, float]:
            """Sample slope uniformly in angle space. Returns (slope, angle_deg)."""
            a_lo = math.atan(slope_lo)
            a_hi = math.atan(slope_hi)
            angle = rng.uniform(a_lo, a_hi)
            return math.tan(angle), math.degrees(angle)

        # Retry loop: sample start, compute feasible slopes, sample angle
        for _ in range(200):
            start = rng.uniform(start_min, start_max)

            if resolved_group == "critical":
                if start >= critical_threshold:
                    continue
                gap = critical_threshold - start
                slope_lo = gap / crit_cross_not_after
                slope_hi = gap / crit_cross_not_before
                if slope_lo > slope_hi:
                    continue

            elif resolved_group == "nc_flat":
                slope_lo = max(
                    -start / zero_cross_not_before if start > 1e-9 else 0.0,
                    -const_slope,
                )
                slope_hi = min(noncritical_ceil - start, const_slope)
                if slope_lo >= slope_hi:
                    continue

            elif resolved_group == "nc_approaching":
                slope_lo = const_slope
                slope_hi = noncritical_ceil - start
                if slope_lo >= slope_hi:
                    continue

            elif resolved_group == "nc_receding":
                slope_lo = -start / zero_cross_not_before if start > 1e-9 else 0.0
                slope_hi = -const_slope
                if zero_cross_not_after is not None and start > 1e-9:
                    slope_hi = min(slope_hi, -start / zero_cross_not_after)
                if slope_lo >= slope_hi:
                    continue

            else:
                raise RuntimeError(f"Unexpected resolved group: {resolved_group}")

            slope, angle_deg = _sample_slope_uniform_angle(slope_lo, slope_hi)
            break
        else:
            raise RuntimeError(
                f"Could not find feasible (start, slope) for group={resolved_group!r} "
                f"after 200 attempts. Check constraint parameters."
            )

    end = start + slope

    # Capture clip bounds for the closure
    _clip_lo = clip_min
    _clip_hi = clip_max

    def schedule(n_steps: int) -> jnp.ndarray:
        values = jnp.linspace(start, end, n_steps)
        if _clip_lo is not None:
            values = jnp.maximum(values, _clip_lo)
        if _clip_hi is not None:
            values = jnp.minimum(values, _clip_hi)
        return values

    metadata = {
        "group": resolved_group,
        "group_requested": group,
        "start": start,
        "slope": slope,
        "angle_deg": angle_deg,
        "end_unclipped": end,
        "clip_min": clip_min,
        "clip_max": clip_max,
    }

    return schedule, metadata


# === Registry & Config-Driven Creation ===

SCHEDULES: dict[str, Callable[..., ParameterSchedule]] = {
    "linear": linear,
    "linear_ramp": linear_ramp,
    "linear_classified": linear_classified,
    "ramp_and_hold": ramp_and_hold,
    "fixed_slope": fixed_slope,
    "constant": constant,
    "piecewise_linear": piecewise_linear,
    "bezier_and_hold": bezier_and_hold,
    "from_array": from_array,
}


def create_schedule(config: dict[str, Any]) -> tuple[ParameterSchedule, dict[str, Any]]:
    """Create a schedule from a dict config.

    The 'mode' key selects the factory; mode-specific args are in a nested dict.

    Factories may return either a bare ``ParameterSchedule`` or a
    ``(ParameterSchedule, metadata_dict)`` tuple.  This function normalizes
    the return to always be a ``(schedule, metadata)`` pair.

    Args:
        config: Dict with 'mode' key and mode-specific parameters.

    Returns:
        ``(schedule, metadata)`` tuple.  ``metadata`` is an empty dict for
        factories that don't produce metadata.

    Examples:
        >>> create_schedule({'mode': 'linear', 'linear': {'critical_at': 0.5}})
        >>> create_schedule({'mode': 'constant', 'constant': {'value': 0.5}})
    """
    mode = config.get("mode", "linear")
    if mode not in SCHEDULES:
        raise ValueError(f"Unknown schedule mode: {mode}. Available: {list(SCHEDULES.keys())}")
    factory = SCHEDULES[mode]
    mode_args = config.get(mode, {})
    result = factory(**mode_args)
    if isinstance(result, tuple):
        return result  # (schedule, metadata)
    return result, {}  # bare schedule → empty metadata


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


def resolve_schedule(schedule: Schedule, n_steps: int) -> tuple[jnp.ndarray, dict[str, Any]]:
    """Resolve any schedule type to a concrete array.

    Args:
        schedule: Array, callable, or dict config.
        n_steps: Output length.

    Returns:
        ``(values, metadata)`` tuple.  ``values`` has shape ``(n_steps,)``.
        ``metadata`` is populated by factories that produce it (e.g.
        ``linear_classified``), empty dict otherwise.
    """
    metadata: dict[str, Any] = {}

    if isinstance(schedule, dict):
        schedule, metadata = create_schedule(schedule)

    if callable(schedule):
        result = schedule(n_steps)
    else:
        arr = jnp.asarray(schedule)
        if len(arr) == n_steps:
            result = arr
        else:
            # Interpolate to requested length
            result = jnp.interp(jnp.linspace(0, 1, n_steps), jnp.linspace(0, 1, len(arr)), arr)

    return result, metadata


def validate_schedule_start(normalized: jnp.ndarray, expected_start: float = 0.0) -> None:
    """Validate that a schedule starts at the expected value.

    By default, schedules must start at 0 because the burn-in phase equilibrates
    the system at ``param_start`` (normalized=0).  When using
    ``prepare_schedule_with_warmup``, the preprocessed schedule always starts at
    0, so this validation passes automatically.

    The ``expected_start`` parameter allows callers to verify a specific start
    value when needed (e.g. after warmup trimming).

    Args:
        normalized: Resolved schedule array.
        expected_start: Expected value of ``normalized[0]``.  Default ``0.0``.

    Raises:
        ValueError: If ``normalized[0]`` differs from ``expected_start`` by more
            than ``1e-6``.
    """
    start_val = float(normalized[0])
    if abs(start_val - expected_start) > 1e-6:
        raise ValueError(
            f"Schedule must start at {expected_start:.4f}, got {start_val:.4f}. "
            f"Use prepare_schedule_with_warmup() to handle non-zero starts."
        )


def resolve_schedule_for_simulation(
    schedule: Schedule,
    n_output: int,
    param_start: float,
    param_end: float,
    upsample_factor: int = 1,
    expected_start: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, dict[str, Any]]:
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
        expected_start: Expected normalized value at t=0.  Default ``0.0``.
            Passed to ``validate_schedule_start``.

    Returns:
        ``(param_values, normalized, metadata)`` tuple:
        - param_values: Parameter values at integration resolution.
        - normalized: Normalized schedule [0,1] at output resolution (n_output).
        - metadata: Schedule metadata dict (empty for most schedule types).
    """
    normalized, metadata = resolve_schedule(schedule, n_output)
    validate_schedule_start(normalized, expected_start=expected_start)

    if upsample_factor <= 1:
        param_values = apply_to_params(normalized, param_start, param_end)
        return param_values, normalized, metadata

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
    return param_fine, normalized, metadata


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
