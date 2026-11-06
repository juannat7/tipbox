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

import tipbox
from tipbox.datasets import generate_dataset, load_microcosm_original
from tipbox.indicators import rolling_ar1, rolling_variance
from tipbox.metrics import ensemble_roc, plot_roc
from tipbox.models import (
    AMOC,
    AMOCRate,
    Bautin,
    CompostBomb,
    MayHarvesting,
    Ricker,
    RosenzweigMacArthur,
    SaddleNode,
    SEIRx,
    Thermoacoustic,
)

INK = "#222222"
BLUE = "#356f9c"
ORANGE = "#ac592b"
RED = "#9c443b"
GRAY = "#b3b3b3"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.prop_cycle": mpl.cycler(color=[BLUE, ORANGE]),
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.frameon": False,
    "legend.fontsize": 8,
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
    (
        "b_harvesting",
        MayHarvesting(),
        "May harvesting",
        {"n": 1400, "noise": 1.0},
        "forcing",
        0,
    ),
    (
        "b_rosenzweig_macarthur_tc",
        RosenzweigMacArthur(bifurcation_type="transcritical"),
        "Rosenzweig–MacArthur: TC",
        {"n": 1800, "noise": 1.0},
        0,
        1,
    ),
    (
        "b_rosenzweig_macarthur_hopf",
        RosenzweigMacArthur(bifurcation_type="hopf"),
        "Rosenzweig–MacArthur: Hopf",
        {"n": 1800, "noise": 1.0},
        0,
        1,
    ),
    ("b_seirx_tc", SEIRx(), "SEIRx", {"n": 1800, "noise": 0.01}, 0, 1),
    ("b_amoc", AMOC(), "AMOC: fold", {"noise": 0.01}, "forcing", 0),
    (
        "r_saddle_node",
        SaddleNode(epsilon=1.4),
        "Saddle-node: rate",
        {"noise": 0.01},
        "forcing",
        0,
    ),
    ("r_bautin", Bautin(), "Bautin: rate", {"noise": 0.01}, 0, 1),
    ("r_compost_bomb", CompostBomb(), "Compost bomb", {"noise": 0.01}, 0, 1),
    ("r_amoc", AMOCRate(), "AMOC: rate", {"noise": 0.01}, "forcing", 0),
    ("b_ricker", Ricker(), "Ricker", {"n": 600, "noise": 1.0}, "time", 0),
    ("b_thermoacoustic", Thermoacoustic(), "Thermoacoustic", {"noise": 1.0}, 0, 1),
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


def _tipping_example(model, arguments, seed):
    selection = generate_dataset(
        model, num_timesteps=300, n_tipping=1, n_null=0, seed=seed, **arguments
    ).metadata[0]
    result = model.make_trajectory(seed=selection["seed"], **arguments)
    return result, {
        "selection": "first accepted tipping trajectory from generate_dataset",
        "selection_context_samples": 300,
        "root_seed": seed,
        "attempt": selection["attempt"],
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
    ax.tick_params(length=3, width=0.6)
    ax.margins(x=0.02)


def gallery(seed):
    fig, axes = plt.subplots(3, 4, figsize=(12.8, 7.8), layout="constrained")
    manifest = {}
    for index, (name, model, title, arguments, horizontal, vertical) in enumerate(
        GALLERY
    ):
        result, selection = _tipping_example(model, arguments, seed)
        observations = np.asarray(result.trajectory)
        coordinates = {"forcing": result.param_values, "time": result.time}
        names = {"forcing": model.forcing_name, "time": f"time ({model.time_unit})"}
        x = (
            np.asarray(coordinates[horizontal])
            if isinstance(horizontal, str)
            else observations[:, horizontal]
        )
        xlabel = (
            names[horizontal]
            if isinstance(horizontal, str)
            else model.observable_names[horizontal]
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
        ax = axes.flat[index]
        ax.plot(x[valid], y[valid], color=BLUE, linewidth=0.75)
        ax.plot(
            x[valid][0],
            y[valid][0],
            "o",
            color=BLUE,
            markerfacecolor="white",
            markersize=3,
        )
        ax.plot(x[valid][-1], y[valid][-1], "o", color=BLUE, markersize=3)
        ax.set_title(f"({chr(97 + index)}) {title}", loc="left", y=1.12)
        ax.set(
            xlabel=xlabel.replace("_", " "),
            ylabel=model.observable_names[vertical].replace("_", " "),
        )
        ax.ticklabel_format(
            axis="both", style="sci", scilimits=(-2, 3), useMathText=True
        )
        ax.locator_params(nbins=3)
        _clean(ax)
        manifest[name] = {
            **_simulation_metadata(model, result, arguments),
            **selection,
            "horizontal": xlabel,
            "vertical": model.observable_names[vertical],
            "display_scaling": "physical coordinates; no min/max normalization",
            "display_filter": "finite points; saddle-node |state|<=5; Bautin radius<=tip_threshold; compost temperature<=150",
            "n_displayed": int(valid.sum()),
        }
    axes.flat[-1].remove()
    return fig, manifest


def indicator_examples(seed, record):
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(11.8, 7.3),
        layout="constrained",
        gridspec_kw={"height_ratios": [1.35, 1, 1]},
    )
    manifest = {}
    model_cases = (
        (
            MayHarvesting(),
            {"n": 1200, "noise": 1.0},
            "May harvesting",
            "Population",
            200,
        ),
        (
            SaddleNode(epsilon=1.4),
            {"noise": 0.01},
            "Saddle-node",
            "State",
            200,
        ),
    )
    for column, (model, arguments, title, ylabel, window) in enumerate(model_cases):
        result, selection = _tipping_example(model, arguments, seed)
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
        ax.set_title(f"({chr(97 + column)}) {title}", loc="left")
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
        manifest[type(model).__name__] = {
            **_simulation_metadata(model, result, arguments),
            **selection,
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
            color=INK,
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
    axes[0, 2].set_title("(c) Microcosm (M1)", loc="left")
    axes[0, 2].set_ylabel(r"Light attenuation (m$^{-1}$)")
    axes[0, 2].set_xlabel("Experimental time (day)")
    axes[0, 2].axvline(27, color=RED, linestyle=(0, (3, 3)), linewidth=1)
    axes[0, 2].set_xlim(0, 29)
    axes[0, 2].set_ylim(top=np.nanmax(values) + 8)
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
        0.01,
        -0.025,
        "Microcosm: M1, Veraart et al. (2012), doi:10.17026/dans-ztg-93aw · CC BY-NC 4.0. Recovery windows follow Grziwotz et al. (2023).",
        fontsize=8,
        color=INK,
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
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), layout="constrained")
    manifest = {}
    for ax, model, arguments, title in (
        (
            axes[0],
            MayHarvesting(),
            {"n": 1000, "noise": 1.0},
            "(a) May harvesting",
        ),
        (axes[1], SaddleNode(epsilon=1.4), {"noise": 0.01}, "(b) Saddle-node"),
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
