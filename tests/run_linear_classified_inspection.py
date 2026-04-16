"""End-to-end inspection of linear_classified schedules across all system types.

Runs tipbox.generate.generate() with each linear_classified group for every
supported system, plots trajectories + param_normalized, and prints metadata.

Usage:
    cd tipbox && uv run python tests/run_linear_classified_inspection.py

Saves plots to tests/linear_classified_plots/
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tipbox.generate import generate

PLOT_DIR = Path(__file__).parent / "linear_classified_plots"
PLOT_DIR.mkdir(exist_ok=True)

GROUPS = ["critical", "nc_flat", "nc_approaching", "nc_receding"]
GROUP_COLORS = {
    "critical": "tab:red",
    "nc_flat": "tab:green",
    "nc_approaching": "tab:blue",
    "nc_receding": "tab:purple",
}

SYSTEMS = [
    # Synthetic bifurcation
    "b_fold",
    "b_hopf",
    "b_transcritical",
    # Semi-real bifurcation
    "b_harvesting",
    "b_rosenzweig_macarthur_tc",
    "b_rosenzweig_macarthur_hopf",
    "b_seirx_tc",
    "b_amoc",
]

NUM_TIMESTEPS = 256
SCHEDULE_WARMUP_TIME = 100.0


def make_schedule_config(group, seed):
    return {
        "mode": "linear_classified",
        "linear_classified": {
            "group": group,
            "start_min": 0.0,
            "start_max": 0.8,
            "crit_cross_not_before": 0.65,
            "crit_cross_not_after": 0.90,
            "critical_threshold": 1.0,
            "noncritical_ceil": 0.9,
            "constant_angle_deg": 5.0,
            "seed": seed,
        },
        "schedule_warmup_time": SCHEDULE_WARMUP_TIME,
    }


def run_system(system_name, base_key):
    """Generate one trajectory per group for a system, return results."""
    results = {}
    for i, group in enumerate(GROUPS):
        key, subkey = jax.random.split(base_key)
        base_key = key
        cfg = make_schedule_config(group, seed=42 + i)

        try:
            full, part, info = generate(
                subkey,
                tipping_type=system_name,
                schedule=cfg,
                num_timesteps=NUM_TIMESTEPS,
                verbose=False,
            )
            results[group] = {"full": full, "info": info, "error": None}
        except Exception as e:
            results[group] = {"full": None, "info": None, "error": str(e)}

    return results


def plot_system(system_name, results):
    """Plot 4-column grid: one column per group."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 7), sharex=True)
    fig.suptitle(f"{system_name}", fontsize=14, fontweight="bold")

    t = np.arange(NUM_TIMESTEPS)

    for col, group in enumerate(GROUPS):
        res = results[group]
        color = GROUP_COLORS[group]

        if res["error"] is not None:
            axes[0, col].text(0.5, 0.5, f"ERROR:\n{res['error'][:80]}",
                              transform=axes[0, col].transAxes, ha="center", va="center",
                              fontsize=7, color="red", wrap=True)
            axes[0, col].set_title(group, color=color, fontweight="bold")
            continue

        traj = np.array(res["full"].trajectory)
        pn = np.array(res["info"]["param_normalized"])

        # Top: trajectory
        ax = axes[0, col]
        if traj.ndim == 1:
            ax.plot(t, traj, color=color, alpha=0.8, label="x")
        else:
            ax.plot(t, traj[:, 0], color=color, alpha=0.8, label="x")
            if traj.shape[1] > 1 and not np.allclose(traj[:, 1], 0):
                ax.plot(t, traj[:, 1], color=color, alpha=0.4, ls="--", label="y")
        ax.set_title(group, color=color, fontweight="bold")
        ax.set_ylabel("state")
        ax.legend(fontsize=7)

        # Bottom: param_normalized
        ax = axes[1, col]
        ax.plot(t, pn, color=color, lw=1.5)
        ax.axhline(1.0, ls=":", color="k", alpha=0.3)
        ax.axhline(0.0, ls=":", color="k", alpha=0.3)
        ax.axhline(0.9, ls="--", color="orange", alpha=0.3)
        ax.set_ylabel("param_normalized")
        ax.set_xlabel("time step")
        ax.set_ylim(-0.05, 1.15)

    plt.tight_layout()
    fname = f"{system_name.replace('/', '_')}.png"
    fig.savefig(PLOT_DIR / fname, bbox_inches="tight")
    plt.close(fig)
    return fname


def print_metadata(system_name, results):
    """Print schedule metadata for each group."""
    print(f"\n{'=' * 70}")
    print(f"  {system_name}")
    print(f"{'=' * 70}")

    all_ok = True
    for group in GROUPS:
        res = results[group]
        if res["error"] is not None:
            print(f"  {group:20s}: ERROR — {res['error'][:80]}")
            all_ok = False
            continue

        info = res["info"]
        pn = np.array(info["param_normalized"])
        meta = info.get("schedule_metadata", {})

        # Validate
        checks = []
        if group == "critical":
            if float(pn.max()) < 0.99:
                checks.append(f"FAIL: max pn={pn.max():.3f} < 1.0")
        elif group in ("nc_flat", "nc_approaching", "nc_receding"):
            if float(pn.max()) > 0.91:
                checks.append(f"WARN: max pn={pn.max():.3f} > 0.9")

        if not meta:
            checks.append("FAIL: no schedule_metadata")

        status = " ".join(checks) if checks else "OK"

        print(f"  {group:20s}: pn=[{pn[0]:.3f} → {pn[-1]:.3f}], "
              f"max={pn.max():.3f}, "
              f"grp={meta.get('group', '?'):20s} "
              f"angle={meta.get('angle_deg', 0):.1f}° "
              f"slope={meta.get('slope', 0):.4f} "
              f"  [{status}]")
        if checks:
            all_ok = False

    return all_ok


def main():
    key = jax.random.PRNGKey(123)
    all_ok = True

    for system in SYSTEMS:
        key, subkey = jax.random.split(key)
        print(f"\nGenerating {system}...", end=" ", flush=True)

        results = run_system(system, subkey)
        fname = plot_system(system, results)
        ok = print_metadata(system, results)
        all_ok = all_ok and ok

        print(f"  → {fname}")

    print(f"\n{'=' * 70}")
    print(f"  All plots saved to {PLOT_DIR}/")
    print(f"  Overall: {'ALL OK' if all_ok else 'SOME ISSUES — check above'}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
