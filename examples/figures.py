"""Reproduce the README figures from simulations and the original microcosm source.

Run from the repository root with JAX_PLATFORMS=cpu for the reference backend.
The manifest records all model coefficients, simulation/analysis settings,
source checksums, software versions, and display transformations.
"""

import argparse
import hashlib
import json
import platform
from functools import partial
from pathlib import Path

import jax
import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

import tipbox
from tipbox.datasets import generate_dataset, load_microcosm_original
from tipbox.indicators import rolling_ar1, rolling_variance
from tipbox.metrics import ensemble_roc, plot_roc
from tipbox.models import MayHarvesting, SaddleNode
from tipbox.registry import get_system_info

INK = "#19394d"
BLUE = "#216a91"
TEAL = "#27998f"
ORANGE = "#ad752d"
RED = "#ba5c4b"
GRAY = "#aab8be"
PALE = "#e8eff2"
CMAP = LinearSegmentedColormap.from_list(
    "tipbox_time", ["#66949f", "#2f8393", "#19394d"]
)
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#aebfc7",
    "axes.titleweight": "semibold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE, TEAL]),
    "xtick.color": "#526b79",
    "ytick.color": "#526b79",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "svg.fonttype": "none",
    "svg.hashsalt": "tipbox-reference-figures",
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
}
EXPORT_METADATA = {
    "svg": {"Creator": "TipBox", "Date": None},
    "pdf": {"Creator": "TipBox", "CreationDate": None, "ModDate": None},
    "png": {"Software": "TipBox"},
}

# These row slices recover the published 250-point M1 recovery intervals while
# retaining original timestamps; applying rounded time bounds would lose a row.
RECOVERIES = (
    ("P1", 161, 411),
    ("P2", 1283, 1533),
    ("P3", 3001, 3251),
    ("P4", 4076, 4326),
    ("P5", 5273, 5523),
    ("P6", 6684, 6934),
)
GALLERY = (
    ("b_harvesting", "FOLD", {"n": 1400, "noise": 1.0}, "forcing", 0),
    ("b_rosenzweig_macarthur_tc", "TRANSCRITICAL", {"n": 1800, "noise": 1.0}, 0, 1),
    ("b_rosenzweig_macarthur_hopf", "HOPF", {"n": 1800, "noise": 1.0}, 0, 1),
    ("b_seirx_tc", "TRANSCRITICAL", {"n": 1800, "noise": 0.01}, 0, 1),
    ("b_amoc", "FOLD", {"noise": 0.01}, "forcing", 0),
    ("r_saddle_node", "RATE", {"noise": 0.01}, "forcing", 0),
    ("r_bautin", "RATE", {"noise": 0.01}, 0, 1),
    ("r_compost_bomb", "RATE", {"noise": 0.01}, 0, 1),
    ("r_amoc", "RATE", {"noise": 0.01}, "forcing", 0),
    ("b_ricker", "TRANSCRITICAL", {"n": 600, "noise": 1.0}, "forcing", 0),
    ("b_thermoacoustic", "SUBCRITICAL HOPF", {"noise": 1.0}, 0, 1),
)


def _simulation_metadata(model, result, arguments):
    return {
        "model": type(model).__name__,
        "parameters": model.get_parameters(),
        "arguments": arguments,
        "seed": result.metadata["seed"],
        "dt": result.metadata["dt"],
        "dt_sample": result.metadata["dt_sample"],
        "initial_state": np.asarray(result.metadata["initial_state"]).tolist(),
        "observables": model.observable_names,
        "time_unit": model.time_unit,
        "n_observations": len(result.time),
        "time_start": float(result.time[0]),
        "time_end": float(result.time[-1]),
    }


def _save(fig, name, output, formats):
    for extension in formats:
        fig.savefig(
            output / f"{name}.{extension}",
            dpi=180,
            bbox_inches="tight",
            pad_inches=0.12,
            metadata=EXPORT_METADATA[extension],
        )
    plt.close(fig)


def _clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=3, width=0.6)
    ax.grid(axis="y", color=PALE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _colored_path(ax, x, y):
    x = (x - x.min()) / max(float(np.ptp(x)), 1e-12)
    y = (y - y.min()) / max(float(np.ptp(y)), 1e-12)
    points = np.column_stack([x, y])
    segments = np.stack([points[:-1], points[1:]], axis=1)
    line = LineCollection(segments, cmap=CMAP, linewidth=0.8)
    line.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(line)
    ax.plot(x[0], y[0], ".", color=CMAP(0.0), markersize=4)
    ax.plot(x[-1], y[-1], ".", color=INK, markersize=4)
    ax.set(xlim=(-0.06, 1.06), ylim=(-0.08, 1.12))
    ax.set_axis_off()


def gallery(seed):
    fig = plt.figure(figsize=(15.2, 8.6))
    fig.text(
        0.055,
        0.945,
        "TipBox",
        fontsize=43,
        fontfamily="STIXGeneral",
        weight="bold",
        color=INK,
    )
    fig.text(
        0.058,
        0.885,
        "A collection of dynamical systems for early-warning benchmarks",
        fontsize=14,
        color="#56717f",
    )
    grid = fig.add_gridspec(
        3, 4, left=0.05, right=0.97, bottom=0.075, top=0.82, hspace=0.38, wspace=0.23
    )
    manifest = {}
    for index, (name, kind, arguments, horizontal, vertical) in enumerate(GALLERY):
        model = get_system_info(name).create_model()
        result = model.make_trajectory(seed=seed, **arguments)
        observations = np.asarray(result.trajectory)
        x = (
            np.asarray(result.param_values)
            if horizontal == "forcing"
            else observations[:, horizontal]
        )
        y = observations[:, vertical]
        valid = np.isfinite(x) & np.isfinite(y)
        if name == "r_saddle_node":
            valid &= np.abs(y) <= 5
        if name == "r_bautin":
            valid &= (
                np.linalg.norm(observations[:, :2], axis=1)
                <= model.config.tip_threshold
            )
        if name == "r_compost_bomb":
            valid &= observations[:, 1] <= 150
        ax = fig.add_subplot(grid[index // 4, index % 4])
        _colored_path(ax, x[valid], y[valid])
        ax.text(
            0.02,
            1.15,
            type(model).__name__.removesuffix("Model"),
            transform=ax.transAxes,
            fontsize=12,
            weight="semibold",
            color=INK,
        )
        ax.text(
            0.02,
            1.04,
            kind,
            transform=ax.transAxes,
            fontsize=9,
            color=TEAL if kind == "RATE" else BLUE,
            weight="bold",
        )
        manifest[name] = {
            **_simulation_metadata(model, result, arguments),
            "horizontal": model.forcing_name
            if horizontal == "forcing"
            else model.observable_names[horizontal],
            "vertical": model.observable_names[vertical],
            "display_scaling": "each coordinate independently maps its displayed min/max to [0,1]",
            "display_filter": "finite points; saddle-node |state|<=5; Bautin radius<=tip_threshold; compost temperature<=150",
            "n_displayed": int(valid.sum()),
        }
    ax = fig.add_subplot(grid[2, 3])
    ax.set_axis_off()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    for index, text in enumerate(
        (
            "Simulate or observe",
            "Compute classical signals",
            "Evaluate trajectory scores",
        )
    ):
        y = 0.88 - index * 0.29
        ax.text(0.04, y, f"0{index + 1}", color=TEAL, fontsize=10, weight="bold")
        ax.text(0.21, y, text, color=INK, fontsize=10)
        if index < 2:
            ax.plot([0.08, 0.08], [y - 0.07, y - 0.19], color="#b9d8dd", linewidth=1)
    fig.text(
        0.058,
        0.025,
        "11 named configurations  ·  explicit equations and parameters  ·  classical indicators and ROC/AUROC",
        fontsize=10,
        color="#56717f",
    )
    return fig, manifest


def indicator_examples(seed, record):
    fig, axes = plt.subplots(
        3, 3, figsize=(14.2, 8.9), gridspec_kw={"height_ratios": [1.45, 1, 1]}
    )
    fig.subplots_adjust(
        left=0.065, right=0.98, top=0.85, bottom=0.1, wspace=0.33, hspace=0.48
    )
    fig.text(
        0.065,
        0.955,
        "From trajectories to warning signals",
        fontsize=24,
        fontfamily="STIXGeneral",
        weight="bold",
    )
    fig.text(
        0.065,
        0.914,
        "Trailing windows use earlier observations; reference markers are defined by each case.",
        fontsize=11,
        color="#56717f",
    )
    manifest = {}
    model_cases = (
        (
            MayHarvesting(),
            {"n": 1200, "noise": 1.0},
            "Bifurcation-induced",
            "Harvesting population",
            "Population",
            200,
        ),
        (
            SaddleNode(),
            {"noise": 0.01},
            "Rate-induced",
            "Moving equilibrium",
            "State",
            200,
        ),
    )
    for column, (model, arguments, kind, title, ylabel, window) in enumerate(
        model_cases
    ):
        selection = generate_dataset(
            model,
            num_timesteps=300,
            n_tipping=1,
            n_null=0,
            seed=seed,
            **arguments,
        ).metadata[0]
        result = model.make_trajectory(seed=selection["seed"], **arguments)
        marker = model.locate_transition(
            result.trajectory, False, param_values=result.param_values, time=result.time
        )
        time = np.asarray(result.time)
        values = np.asarray(result.trajectory[:, 0])
        display = np.isfinite(values)
        if column == 1:
            display &= (time >= -8) & (values <= 5)
        ax = axes[0, column]
        ax.plot(time[display], values[display], color=GRAY, linewidth=0.9)
        before = display & (np.arange(len(time)) < marker)
        ax.plot(time[before], values[before], color=BLUE, linewidth=1.1)
        ax.set_title(f"{kind}\n{title}", loc="left", pad=12)
        ax.set_ylabel(ylabel)
        limits = (float(time[display].min()), float(time[display].max()))
        indicator_display = time[:marker] >= limits[0]
        for row, (function, label, color) in enumerate(
            ((rolling_ar1, "AR1 slope", BLUE), (rolling_variance, "Variance", ORANGE)),
            start=1,
        ):
            indicator = function(values[:marker], window=window, detrend="linear")
            axes[row, column].plot(
                time[:marker][indicator_display],
                indicator[indicator_display],
                color=color,
                linewidth=1.3,
            )
            axes[row, column].set_ylabel(label)
        for row in range(3):
            axes[row, column].set_xlim(limits)
            axes[row, column].axvline(
                time[marker], color=RED, linestyle=(0, (3, 3)), linewidth=1
            )
            _clean(axes[row, column])
        axes[0, column].text(
            0.98,
            1.02,
            "Fold reference" if column == 0 else "State threshold",
            transform=axes[0, column].transAxes,
            ha="right",
            va="bottom",
            color=RED,
            fontsize=8,
        )
        axes[2, column].set_xlabel(f"Time ({model.time_unit})")
        axes[0, column].text(
            0.03,
            0.06,
            f"window = {window} samples; noise = {arguments['noise']:g}",
            transform=ax.transAxes,
            color="#56717f",
            fontsize=8,
        )
        manifest[type(model).__name__] = {
            **_simulation_metadata(model, result, arguments),
            "selection": "first accepted tipping trajectory from generate_dataset",
            "root_seed": seed,
            "attempt": selection["attempt"],
            "marker_index": int(marker),
            "marker_time": float(time[marker]),
            "indicator_input_slice": [0, int(marker)],
            "window": window,
            "step": 1,
            "detrend": "linear",
            "display_time_limits": limits,
        }
    time, values = record.time, record.observations[:, 0]
    axes[0, 2].plot(time, values, color=GRAY, linewidth=0.8)
    colors = ("#0072b2", "#d55e00", "#009e73", "#a05d8b", "#a66b00", "#46566b")
    markers = ("o", "s", "^", "D", "v", "X")
    intervals = []
    for (label, start, stop), color, shape in zip(RECOVERIES, colors, markers):
        t, x = time[start:stop], values[start:stop]
        axes[0, 2].plot(t, x, color=color, linewidth=1.4)
        axes[0, 2].annotate(
            label,
            (float(t.mean()), float(x.max())),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            color=color,
            fontsize=7.5,
        )
        for row, function in enumerate((rolling_ar1, rolling_variance), start=1):
            indicator = function(x, window=200, step=5, detrend="linear")
            computed = np.isfinite(indicator)
            axes[row, 2].plot(
                (t - t[0])[computed],
                indicator[computed],
                color=color,
                linewidth=1.1,
                marker=shape,
                markersize=3,
                label=label,
            )
        intervals.append(
            {
                "label": label,
                "rows": [start, stop],
                "time_start": float(t[0]),
                "time_stop": float(t[-1]),
                "n": len(t),
            }
        )
    axes[0, 2].set_title(
        "Observed microcosm\nSeparate recovery experiments", loc="left", pad=12
    )
    axes[0, 2].set_ylabel(r"Light attenuation (m$^{-1}$)")
    axes[0, 2].set_xlabel("Experimental time (day)")
    axes[0, 2].axvline(27, color=RED, linestyle=(0, (3, 3)), linewidth=1)
    axes[0, 2].set_xlim(0, 29)
    axes[0, 2].text(
        0.98,
        1.02,
        "27-day reference",
        transform=axes[0, 2].transAxes,
        ha="right",
        va="bottom",
        color=RED,
        fontsize=8,
    )
    axes[1, 2].set_ylabel("AR1 slope")
    axes[2, 2].set_ylabel(r"Variance (m$^{-2}$)")
    axes[1, 2].legend(
        ncol=6,
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        handlelength=1.0,
        columnspacing=0.7,
        fontsize=7.5,
        borderaxespad=0,
    )
    for row in (1, 2):
        axes[row, 2].set_xlabel("Time within recovery (day)")
        axes[row, 2].set_xlim(0.69, 0.9)
    for ax in axes[:, 2]:
        _clean(ax)
    for ax in axes[2]:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    fig.text(
        0.065,
        0.025,
        "Microcosm: M1, Veraart et al. (2012), doi:10.17026/dans-ztg-93aw · CC BY-NC 4.0. Recovery windows follow Grziwotz et al. (2023).",
        fontsize=8,
        color="#56717f",
    )
    manifest["microcosm"] = {
        "source_doi": record.metadata["source_doi"],
        "source_sha256": record.metadata["source_sha256"],
        "license": record.metadata["license"],
        "observable": "M1",
        "reference_time": 27.0,
        "reference_source_doi": "10.5281/zenodo.7379358",
        "recovery_row_slices": intervals,
        "slice_note": "row correspondence verified against the pinned aggregated copy; original times are not rounded",
        "window": 200,
        "step": 5,
        "detrend": "linear",
        "plot_time": "raw experiment time above; elapsed time within each recovery below",
    }
    return fig, manifest


def roc_examples(seed, n_per_class):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.9))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.76, bottom=0.2, wspace=0.3)
    fig.text(
        0.075,
        0.94,
        "Evaluate an ensemble, one score per trajectory",
        fontsize=22,
        fontfamily="STIXGeneral",
        weight="bold",
    )
    fig.text(
        0.075,
        0.89,
        "Illustrative outcome-conditioned batches · Kendall trends of trailing, linearly detrended windows",
        fontsize=10.5,
        color="#56717f",
    )
    manifest = {}
    for ax, model, arguments, title in (
        (
            axes[0],
            MayHarvesting(),
            {"n": 1000, "noise": 1.0},
            "Bifurcation-induced · harvesting",
        ),
        (axes[1], SaddleNode(), {"noise": 0.01}, "Rate-induced · moving equilibrium"),
    ):
        data = generate_dataset(
            model,
            num_timesteps=300,
            n_tipping=n_per_class,
            n_null=n_per_class,
            lead_steps=1,
            seed=seed,
            **arguments,
        )
        indicators = {
            "AR1": [
                rolling_ar1(x, window=100, detrend="linear")
                for x in data.observations[:, :, 0]
            ],
            "Variance": [
                rolling_variance(x, window=100, detrend="linear")
                for x in data.observations[:, :, 0]
            ],
        }
        result = ensemble_roc(data.labels, indicators)
        plot_roc(result, ax=ax)
        ax.set_title(title, loc="left", pad=12)
        ax.legend(loc="lower right", fontsize=8)
        _clean(ax)
        leads = [metadata["lead_time"] for metadata in data.metadata]
        manifest[type(model).__name__] = {
            "parameters": model.get_parameters(),
            "arguments": arguments,
            "root_seed": seed,
            "n_per_class": n_per_class,
            "num_timesteps": 300,
            "lead_steps": 1,
            "physical_lead_min": float(min(leads)),
            "physical_lead_max": float(max(leads)),
            "window": 100,
            "detrend": "linear",
            "score": "Kendall tau-b, increasing direction declared for both indicators",
            "accepted_seeds": [metadata["seed"] for metadata in data.metadata],
            "attempts": [metadata["attempt"] for metadata in data.metadata],
            "curves": {
                name: {
                    key: getattr(curve, key)
                    for key in (
                        "auroc",
                        "n_valid",
                        "n_missing",
                        "n_positive",
                        "n_negative",
                    )
                }
                for name, curve in result.curves.items()
            },
        }
    fig.text(
        0.075,
        0.07,
        "Each batch has matched sample counts, not matched forcing histories. Different models have different time units and event rules.\nWindows, score direction, seed and noise settings are fixed before evaluation; these examples are not a general performance claim.",
        fontsize=9,
        color="#56717f",
        linespacing=1.6,
    )
    return fig, manifest


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/_static/figures"),
        help="Export directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="TipBox source-file cache; None uses its normal user cache",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        help="Local original microcosm ODS; otherwise download to cache",
    )
    parser.add_argument(
        "--n-per-class",
        type=int,
        default=50,
        help="Tipping and null counts per model in the ROC panels",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Declared root seed for simulations and batch selection",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "svg", "pdf"),
        default=["png", "svg", "pdf"],
        help="Figure export formats",
    )
    args = parser.parse_args()
    if args.n_per_class < 1:
        parser.error("--n-per-class must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = load_microcosm_original(
        cache_dir=args.cache_dir, source_path=args.source_path
    )
    manifest = {
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "jax": jax.__version__,
            "scipy": scipy.__version__,
            "matplotlib": mpl.__version__,
        },
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": jax.config.jax_enable_x64,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_sources_sha256": {
            str(path.relative_to(Path(tipbox.__file__).parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted(Path(tipbox.__file__).parent.rglob("*.py"))
        },
    }
    with mpl.rc_context(STYLE):
        for name, render in (
            ("overview", partial(gallery, args.seed)),
            ("indicators", partial(indicator_examples, args.seed, record)),
            ("roc", partial(roc_examples, args.seed, args.n_per_class)),
        ):
            print(f"Rendering {name}...", flush=True)
            fig, metadata = render()
            _save(fig, name, args.output_dir, args.formats)
            manifest[name] = metadata
            print(f"Rendered {name}", flush=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
