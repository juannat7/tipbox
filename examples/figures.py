"""Reproduce the README figures from simulations and the original microcosm source.

Run from the repository root. The manifest records the selected JAX backend.
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
from tipbox.datasets import generate_dataset, load_veraart2012
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
        ax.plot(x[valid], y[valid], color=BLUE, linewidth=1.15)
        ax.plot(
            x[valid][0],
            y[valid][0],
            "o",
            color=BLUE,
            markerfacecolor="white",
            markersize=3.5,
            label="Start",
        )
        ax.plot(
            x[valid][-1], y[valid][-1], "o", color=BLUE, markersize=3.5, label="End"
        )
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
    axes.flat[-1].set_axis_off()
    axes.flat[-1].legend(
        *axes.flat[0].get_legend_handles_labels(), loc="upper left", fontsize=10
    )
    return fig, manifest


def example_data(seed, record):
    simulations = []
    metadata = {}
    for column, (model, arguments) in enumerate(
        (
            (MayHarvesting(), {"n": 1200, "noise": 1.0}),
            (SaddleNode(epsilon=1.4), {"noise": 0.01}),
        )
    ):
        result, selection = _tipping_example(model, arguments, seed)
        marker = model.locate_transition(
            result.trajectory, False, param_values=result.param_values, time=result.time
        )
        time = np.asarray(result.time)
        values = np.asarray(result.trajectory[:, 0])
        display = np.isfinite(values)
        if column == 1:
            display &= (time >= -8) & (values <= 5)
        limits = (float(time[display].min()), float(time[display].max()))
        simulations.append((model, time, values, marker, display))
        metadata[type(model).__name__] = {
            **_simulation_metadata(model, result, arguments),
            **selection,
            "marker_index": int(marker),
            "marker_time": float(time[marker]),
            "indicator_input_slice": [0, int(marker)],
            "display_time_limits": limits,
            "window": 200,
            "step": 1,
            "detrend": "linear",
        }
    intervals = [
        {
            "label": label,
            "rows": [start, stop],
            "time_start": float(record.time[start]),
            "time_stop": float(record.time[stop - 1]),
            "n": stop - start,
        }
        for label, start, stop in RECOVERIES
    ]
    metadata["microcosm"] = {
        "source_doi": record.metadata["source_doi"],
        "source_sha256": record.metadata["source_sha256"],
        "license": record.metadata["license"],
        "observable": "M1",
        "reference_time": 27.0,
        "reference_source_doi": "10.5281/zenodo.7379358",
        "recovery_row_slices": intervals,
        "window": 200,
        "step": 5,
        "detrend": "linear",
        "plot_time": "experiment time for observations; elapsed time for separate recoveries",
    }
    return simulations, record, metadata


RECOVERY_COLORS = ("#0072b2", "#d55e00", "#009e73", "#a05d8b", "#a66b00", "#46566b")
RECOVERY_MARKERS = ("o", "s", "^", "D", "v", "X")


def _data_credit(fig):
    fig.text(
        0.01,
        -0.04,
        "Veraart et al. (2012) · doi:10.17026/dans-ztg-93aw · CC BY-NC 4.0",
        fontsize=7,
    )


def trajectory_examples(examples):
    simulations, record, metadata = examples
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.0), layout="constrained")
    for column, ((model, time, values, marker, display), title, ylabel) in enumerate(
        zip(simulations, ("May harvesting", "Saddle-node"), ("Population", "State"))
    ):
        ax = axes[column]
        before = display & (np.arange(len(time)) < marker)
        ax.plot(time[display], values[display], color=GRAY, linewidth=1.1)
        ax.plot(time[before], values[before], color=BLUE, linewidth=1.2)
        ax.axvline(time[marker], color=RED, linestyle="--", linewidth=0.8)
        ax.set(
            title=f"({chr(97 + column)}) {title}",
            ylabel=ylabel,
            xlabel=f"Time ({model.time_unit})",
        )
        ax.set_xlim(float(time[display].min()), float(time[display].max()))
    time, values = record.time, record.observations[:, 0]
    axes[2].plot(time, values, color=GRAY, linewidth=0.9)
    for (label, start, stop), color in zip(RECOVERIES, RECOVERY_COLORS):
        t, x = time[start:stop], values[start:stop]
        axes[2].plot(t, x, color=color, linewidth=1.4)
        axes[2].annotate(
            label,
            (float(t.mean()), float(x.max())),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    axes[2].axvline(27, color=RED, linestyle="--", linewidth=0.8)
    axes[2].set(
        title="(c) Veraart2012 · M1",
        xlabel="Time (day)",
        ylabel=r"Light attenuation (m$^{-1}$)",
        xlim=(0, 29),
    )
    axes[2].set_ylim(top=np.nanmax(values) + 8)
    for ax in axes:
        _clean(ax)
    _data_credit(fig)
    return fig, metadata


def indicator_examples(examples):
    simulations, record, metadata = examples
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 4.5), layout="constrained")
    for column, ((model, time, values, marker, display), title) in enumerate(
        zip(simulations, ("May harvesting", "Saddle-node"))
    ):
        for row, (function, label, color) in enumerate(
            ((rolling_ar1, "AR1 slope", BLUE), (rolling_variance, "Variance", ORANGE))
        ):
            indicator = function(values[:marker], window=200, detrend="linear")
            limits = metadata[type(model).__name__]["display_time_limits"]
            shown = time[:marker] >= limits[0]
            axes[row, column].plot(
                time[:marker][shown], indicator[shown], color=color, linewidth=1.3
            )
            axes[row, column].axvline(
                time[marker], color=RED, linestyle="--", linewidth=0.8
            )
            axes[row, column].set_ylabel(label)
            axes[row, column].set_xlim(
                metadata[type(model).__name__]["display_time_limits"]
            )
        axes[0, column].set_title(f"({chr(97 + column)}) {title}")
        axes[1, column].set_xlabel(f"Time ({model.time_unit})")
    for (label, start, stop), color, shape in zip(
        RECOVERIES, RECOVERY_COLORS, RECOVERY_MARKERS
    ):
        time, values = record.time[start:stop], record.observations[start:stop, 0]
        for row, function in enumerate((rolling_ar1, rolling_variance)):
            indicator = function(values, window=200, step=5, detrend="linear")
            valid = np.isfinite(indicator)
            axes[row, 2].plot(
                (time - time[0])[valid],
                indicator[valid],
                color=color,
                marker=shape,
                markersize=3,
                linewidth=1.1,
                label=label,
            )
    axes[0, 2].set(title="(c) Veraart2012 · M1", ylabel="AR1 slope")
    axes[1, 2].set(xlabel="Time within recovery (day)", ylabel=r"Variance (m$^{-2}$)")
    fig.legend(
        *axes[0, 2].get_legend_handles_labels(),
        loc="outside upper right",
        ncol=6,
        handlelength=1,
        columnspacing=0.6,
        fontsize=7,
    )
    for row in (0, 1):
        axes[row, 2].set_xlim(0.69, 0.9)
    for ax in axes.flat:
        _clean(ax)
    for ax in axes[1]:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    _data_credit(fig)
    return fig, metadata


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
    record = load_veraart2012(cache_dir=args.cache_dir, source_path=args.source_path)
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
    examples = example_data(args.seed, record)
    with mpl.rc_context(STYLE):
        for name, render in (
            ("overview", partial(gallery, args.seed)),
            ("trajectories", partial(trajectory_examples, examples)),
            ("indicators", partial(indicator_examples, examples)),
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
