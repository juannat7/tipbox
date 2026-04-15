"""Visual inspection script for schedule warmup.

Run from the tipbox directory:
    uv run python tests/run_warmup_inspection.py

Saves plots to tests/warmup_plots/
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
from tipbox.bifurcation.schedule import prepare_schedule_with_warmup

PLOT_DIR = Path(__file__).parent / "warmup_plots"
PLOT_DIR.mkdir(exist_ok=True)


def plot_trajectory(full, info, title="", filename=None):
    traj = np.array(full.trajectory)
    pn = np.array(info["param_normalized"])
    rdtc = 1.0 - pn
    t = np.arange(len(pn))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    ax = axes[0]
    ax.plot(t, traj[:, 0], label="x", alpha=0.8)
    if traj.shape[1] > 1:
        ax.plot(t, traj[:, 1], label="y", alpha=0.8)
    ax.set_xlabel("time step")
    ax.set_ylabel("state")
    ax.set_title("Trajectory")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(t, pn, color="tab:orange")
    ax.set_xlabel("time step")
    ax.set_ylabel("param_normalized")
    ax.set_title("Forcing (param_normalized)")
    ax.axhline(0, ls=":", color="gray", alpha=0.5)
    ax.axhline(1, ls=":", color="gray", alpha=0.5)

    ax = axes[2]
    ax.plot(t, rdtc, color="tab:green")
    ax.set_xlabel("time step")
    ax.set_ylabel("rdtc")
    ax.set_title("Relative Distance to Critical")
    ax.axhline(0, ls=":", color="gray", alpha=0.5)
    ax.axhline(1, ls=":", color="gray", alpha=0.5)

    plt.tight_layout()
    if filename:
        fig.savefig(PLOT_DIR / filename, bbox_inches="tight")
    plt.close(fig)

    print(
        f"  {title}: pn start={pn[0]:.4f} end={pn[-1]:.4f} "
        f"shape={traj.shape} nan={np.any(np.isnan(traj))}"
    )


def main():
    key = jax.random.PRNGKey(42)

    schedules = {
        "approaching": {
            "mode": "linear_ramp",
            "linear_ramp": {"start": 0.4, "end": 0.8},
            "schedule_warmup_time": 100.0,
        },
        "receding": {
            "mode": "linear_ramp",
            "linear_ramp": {"start": 0.6, "end": 0.2},
            "schedule_warmup_time": 100.0,
        },
        "constant": {
            "mode": "constant",
            "constant": {"value": 0.4},
            "schedule_warmup_time": 100.0,
        },
    }

    # --- Section 1: Synthetic bifurcations ---
    print("=== 1. Synthetic bifurcations ===")
    for bif_type in ["b_fold", "b_hopf", "b_transcritical"]:
        for sched_name, sched_cfg in schedules.items():
            key, subkey = jax.random.split(key)
            full, part, info = generate(
                subkey,
                tipping_type=bif_type,
                schedule=sched_cfg,
                num_timesteps=256,
                verbose=False,
            )
            fname = f"synthetic_{bif_type}_{sched_name}.png"
            plot_trajectory(full, info, title=f"{bif_type} — {sched_name}", filename=fname)

    # --- Section 2: Semi-real (harvesting) ---
    print("\n=== 2. Semi-real (harvesting) ===")
    for sched_name, sched_cfg in schedules.items():
        key, subkey = jax.random.split(key)
        full, part, info = generate(
            subkey,
            tipping_type="b_harvesting",
            schedule=sched_cfg,
            num_timesteps=256,
        )
        fname = f"semireal_harvesting_{sched_name}.png"
        plot_trajectory(full, info, title=f"b_harvesting — {sched_name}", filename=fname)

    # --- Section 3: Warmup time comparison ---
    print("\n=== 3. Warmup time comparison ===")
    base_key = jax.random.PRNGKey(99)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    fig.suptitle("Effect of schedule_warmup_time (fold, start=0.5 -> 0.8)", fontweight="bold")

    for ax, wt in zip(axes, [50.0, 100.0, 200.0]):
        cfg = {
            "mode": "linear_ramp",
            "linear_ramp": {"start": 0.5, "end": 0.8},
            "schedule_warmup_time": wt,
        }
        full, _, info = generate(
            base_key, tipping_type="b_fold", schedule=cfg,
            num_timesteps=256, verbose=False,
        )
        traj = np.array(full.trajectory)
        pn = np.array(info["param_normalized"])
        ax.plot(traj[:, 0], alpha=0.7, label="x")
        ax.set_title(f"warmup_time={wt:.0f}")
        ax.set_xlabel("time step")
        ax.axvline(0, ls="--", color="red", alpha=0.3, label="t=0")
        ax.legend(fontsize=8)
        print(f"  warmup_time={wt:.0f}: pn[0]={pn[0]:.4f}, pn[-1]={pn[-1]:.4f}")

    axes[0].set_ylabel("x state")
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "warmup_time_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # --- Section 4: Ramp shape ---
    print("\n=== 4. Warmup ramp shape ===")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Full schedule (incl. warmup prefix before trim)", fontweight="bold")

    test_cases = [
        ("approaching 0.4->0.8", {"mode": "linear_ramp", "linear_ramp": {"start": 0.4, "end": 0.8}, "schedule_warmup_time": 100.0}),
        ("receding 0.6->0.2", {"mode": "linear_ramp", "linear_ramp": {"start": 0.6, "end": 0.2}, "schedule_warmup_time": 100.0}),
        ("constant 0.5", {"mode": "constant", "constant": {"value": 0.5}, "schedule_warmup_time": 100.0}),
    ]

    for ax, (label, cfg) in zip(axes, test_cases):
        prepared = prepare_schedule_with_warmup(cfg, n_output=256, dt_sample=1.0)
        values = np.array(prepared.schedule(prepared.total_output_steps))
        ax.plot(values, color="tab:blue", alpha=0.8)
        ax.axvline(prepared.warmup_output_steps, ls="--", color="red", alpha=0.6, label=f"trim at {prepared.warmup_output_steps}")
        ax.set_title(label)
        ax.set_xlabel("step (incl. warmup)")
        ax.set_ylabel("param_normalized")
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(PLOT_DIR / "ramp_shape.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
