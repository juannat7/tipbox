"""Calibration script for translate_schedule on rate-induced tipping systems.

Tests end-to-end: abstract schedule → translate → simulate → check tipping.
Sweeps steepness (sigmoid models) and reports tipping rates per group.

Usage:
    cd priors/
    uv run --extra tipping python tipbox/tests/run_translate_schedule_calibration.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tipbox.bifurcation.schedule import linear_classified, resolve_schedule
from tipbox.rate.rate_simulator import simulate_rate_trajectory
from tipbox.registry import get_rate_system_spec


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYSTEMS = ["bautin", "compost_bomb", "saddle_node", "amoc"]
N_EPISODES = 30  # per group per system
SERIES_LEN = 500
SIGMA_TILDE = 0.10
STEEPNESS_VALUES = [3.0, 4.0, 6.0, 8.0, 10.0]

# linear_classified params matching val-systems-rate.yaml
LC_COMMON = dict(
    start_min=0.0,
    start_max=0.8,
    critical_threshold=1.0,
    noncritical_ceil=0.9,
    constant_angle_deg=5.0,
    crit_cross_not_before=0.65,
    crit_cross_not_after=0.90,
    zero_cross_not_before=1.0,
    clip_min=0.0,
    clip_max=1.0,
)


def make_abstract_schedule(group: str, seed: int):
    """Create an abstract schedule via linear_classified."""
    sched_fn, meta = linear_classified(group=group, seed=seed, **LC_COMMON)
    return sched_fn, meta


def run_single(
    system_name: str,
    group: str,
    seed: int,
    steepness: float = 6.0,
):
    """Run a single episode: abstract schedule → translate → simulate."""
    sched_fn, sched_meta = make_abstract_schedule(group, seed)

    # Build schedule config dict for rate_simulator
    # We pass the callable directly + translate params
    key = jax.random.PRNGKey(seed)

    translate_kwargs = {}
    if system_name in ("bautin", "saddle_node"):
        translate_kwargs["steepness"] = steepness

    result = simulate_rate_trajectory(
        key,
        series_len=SERIES_LEN,
        sigma_tilde=SIGMA_TILDE,
        system_name=system_name,
        schedule=sched_fn,
        translate_mode="time_to_critical_rate",
        translate_kwargs=translate_kwargs,
    )

    if result is None:
        return None

    full_res, _ = result
    meta = full_res.metadata

    # Check if actually tipped using the model's is_tipped
    spec = get_rate_system_spec(system_name)
    config = spec.default_config()
    model = spec.create_model(config)
    tipped = model.is_tipped(
        full_res.trajectory,
        time=full_res.time,
        param_values=full_res.param_values,
    )

    return {
        "system": system_name,
        "group": group,
        "resolved_group": sched_meta.get("group", group),
        "tipped": bool(tipped),
        "steepness": steepness,
        "param_normalized_mode": meta.get("param_normalized_mode"),
        "has_nan": bool(np.any(np.isnan(np.asarray(full_res.trajectory)))),
        "sched_meta": sched_meta,
    }


def main():
    print("=" * 70)
    print("translate_schedule calibration for rate-induced tipping systems")
    print("=" * 70)
    print(f"  episodes per group: {N_EPISODES}")
    print(f"  series_len: {SERIES_LEN}")
    print(f"  sigma_tilde: {SIGMA_TILDE}")
    print(f"  steepness sweep: {STEEPNESS_VALUES}")
    print()

    for system_name in SYSTEMS:
        print(f"\n{'─' * 60}")
        print(f"  System: {system_name}")
        print(f"{'─' * 60}")

        is_sigmoid = system_name in ("bautin", "saddle_node")
        sweep = STEEPNESS_VALUES if is_sigmoid else [6.0]  # only one pass for non-sigmoid

        for steepness in sweep:
            if is_sigmoid:
                print(f"\n  steepness = {steepness}")

            for group in ["critical", "noncritical"]:
                results = []
                failures = 0
                for i in range(N_EPISODES):
                    seed = hash((system_name, group, steepness, i)) % (2**31)
                    try:
                        r = run_single(system_name, group, seed, steepness)
                        if r is None:
                            failures += 1
                        else:
                            results.append(r)
                    except Exception as e:
                        print(f"    ERROR [{group} #{i}]: {e}")
                        failures += 1

                n_ok = len(results)
                n_tipped = sum(1 for r in results if r["tipped"])
                n_nan = sum(1 for r in results if r["has_nan"])
                tip_rate = n_tipped / n_ok if n_ok > 0 else float("nan")

                # Count resolved subgroups
                subgroups = {}
                for r in results:
                    sg = r["resolved_group"]
                    subgroups[sg] = subgroups.get(sg, 0) + 1

                print(
                    f"    {group:15s}  "
                    f"tip_rate={tip_rate:.0%}  "
                    f"ok={n_ok}  fail={failures}  nan={n_nan}  "
                    f"subgroups={subgroups}"
                )

    print("\n" + "=" * 70)
    print("Done. Adjust steepness so critical tip_rate is high and noncritical is low.")
    print("=" * 70)


if __name__ == "__main__":
    main()
