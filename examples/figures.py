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
from matplotlib.lines import Line2D

import tipbox
from tipbox.datasets import Microcosm, generate_dataset
from tipbox.indicators import rolling_ar1, rolling_variance
from tipbox.metrics import ensemble_roc, plot_roc
from tipbox.models import (
    AMOCFold,
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
BLUE = "#176b78"
ORANGE = "#b7653b"
RED = "#a34a43"
GRAY = "#b8babd"
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.labelcolor": INK,
    "axes.edgecolor": "#666666",
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
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
        {"num_samples": 1400, "noise": 1.0},
        "forcing",
        0,
    ),
    (
        "b_rosenzweig_macarthur_tc",
        RosenzweigMacArthur(bifurcation_type="transcritical", a_transcritical_max=10.0),
        "Rosenzweig–\nMacArthur: TC",
        {"num_samples": 89951, "dt_sample": 0.02, "noise": 1.0},
        0,
        1,
    ),
    (
        "b_rosenzweig_macarthur_hopf",
        RosenzweigMacArthur(bifurcation_type="hopf"),
        "Rosenzweig–\nMacArthur: Hopf",
        {"num_samples": 89951, "dt_sample": 0.02, "noise": 1.0},
        0,
        1,
    ),
    ("b_seirx_tc", SEIRx(), "SEIRx", {"num_samples": 1800, "noise": 0.01}, 0, 1),
    ("b_amoc", AMOCFold(), "AMOC: fold", {"noise": 0.01}, "forcing", 0),
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
    ("b_ricker", Ricker(), "Ricker", {"num_samples": 600, "noise": 1.0}, "time", 0),
    (
        "b_thermoacoustic",
        Thermoacoustic(),
        "Thermoacoustic",
        {"num_samples": 11001, "dt_sample": 0.0001, "noise": 1.0},
        0,
        1,
    ),
)


# Physical-time offsets around each reference; resolve the dense oscillations.
VIEW_WINDOWS = {
    "b_rosenzweig_macarthur_tc": (-100.0, 400.0),
    "b_rosenzweig_macarthur_hopf": (-2.0, 4.0),
    "b_thermoacoustic": (-0.01, 0.02),
}


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


def _example(model, arguments, seed, *, null=False):
    selection = generate_dataset(
        model,
        context_samples=300,
        num_tipping=int(not null),
        num_controls=int(null),
        seed=seed,
        **arguments,
    ).metadata[0]
    result = model.make_trajectory(
        seed=selection["seed"], null_simulation=null, **arguments
    )
    return result, {
        "selection": "first accepted control"
        if null
        else "first accepted tipping trajectory",
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


def gallery(runs):
    fig, axes = plt.subplots(4, 3, figsize=(12.8, 10.2), layout="constrained")
    manifest = {}
    for index, (name, model, title, arguments, horizontal, vertical) in enumerate(
        GALLERY
    ):
        result, selection = runs[name]
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
        time = np.asarray(result.time)
        control = np.asarray(result.param_values)
        marker = model.locate_transition(
            observations, False, param_values=control, time=time
        )
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
        if name in VIEW_WINDOWS:
            start, stop = VIEW_WINDOWS[name]
            valid &= (time >= time[marker] + start) & (time <= time[marker] + stop)
        ax = axes.flat[index]
        ax.plot(x[valid], y[valid], color=BLUE, linewidth=1.35)
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
        critical = model.critical_parameter
        inset = ax.inset_axes([0.63, 1.04, 0.34, 0.27])
        inset.plot(time, control, color=INK, linewidth=0.8)
        inset.axvspan(time[valid][0], time[valid][-1], color=BLUE, alpha=0.15)
        if critical is not None:
            inset.axhline(float(critical), color=RED, linestyle="--", linewidth=0.7)
        inset.set_title("Parameter", fontsize=8, pad=2)
        inset.tick_params(labelsize=6, length=2, width=0.5)
        inset.locator_params(nbins=2)
        manifest[name] = {
            **_simulation_metadata(model, result, arguments),
            **selection,
            "horizontal": xlabel,
            "vertical": model.observable_names[vertical],
            "display_scaling": "physical coordinates; no min/max normalization",
            "display_filter": "finite points; saddle-node |state|<=5; Bautin radius<=tip_threshold; compost temperature<=150",
            "n_displayed": int(valid.sum()),
            "display_time_interval": [float(time[valid][0]), float(time[valid][-1])],
            "reference_index": int(marker),
            "reference_time": float(time[marker]),
            "critical_parameter": None if critical is None else float(critical),
            "parameter_inset": "full forcing schedule; shading marks displayed interval; dashed line is the critical parameter when defined",
        }
    axes.flat[-1].set_axis_off()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=RED, linestyle="--", linewidth=0.8))
    labels.append("Critical parameter")
    axes.flat[-1].legend(handles, labels, loc="upper left", fontsize=9)
    return fig, manifest


def model_preview(name, model, result, control, arguments, selection):
    """Compare an accepted tipping path with a control on the same time window."""
    time = np.asarray(result.time)
    values = np.asarray(result.trajectory)
    forcing = np.asarray(result.param_values)
    marker = model.locate_transition(values, False, param_values=forcing, time=time)
    column = {"b_rosenzweig_macarthur_tc": 1, "r_compost_bomb": 1}.get(name, 0)
    display = np.isfinite(values[:, column])
    if name == "r_saddle_node":
        display &= (time >= -8) & (values[:, 0] <= 5)
    if name == "r_bautin":
        display &= np.linalg.norm(values[:, :2], axis=1) <= model.config.tip_threshold
    if name == "r_compost_bomb":
        display &= values[:, 1] <= 150
    start = float(time[display][0])
    # Leave room for the reference line without plotting an escaped state.
    stop = max(
        float(time[display][-1]),
        float(time[marker]) + 0.02 * float(np.ptp(time[display])),
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.2, 3.5),
        height_ratios=(1, 3),
        sharex="col",
        sharey="row",
        layout="constrained",
    )
    for index, (run, mask, title, color) in enumerate(
        (
            (result, display, "Tipping", BLUE),
            (control, np.ones(len(control.time), dtype=bool), "Control", ORANGE),
        )
    ):
        t = np.asarray(run.time)
        observed = np.asarray(run.trajectory[:, column])
        shown = mask & (t >= start) & (t <= stop) & np.isfinite(observed)
        parameter, trajectory = axes[:, index]
        parameter.plot(t, run.param_values, color=INK, linewidth=1)
        parameter.set_title(title, loc="left")
        trajectory.plot(
            t, np.where(shown, observed, np.nan), color=color, linewidth=0.8
        )
        critical = model.critical_parameter
        if critical is not None:
            parameter.axhline(float(critical), color=RED, linestyle="--", linewidth=0.8)
        if index == 0:
            parameter.axvline(time[marker], color=RED, linestyle="--", linewidth=0.8)
            trajectory.axvline(time[marker], color=RED, linestyle="--", linewidth=0.8)
        parameter.set_xlim(start, stop)
        trajectory.set_xlabel(f"Time ({model.time_unit})")
        for ax in (parameter, trajectory):
            ax.locator_params(nbins=4)
            _clean(ax)
    axes[0, 0].set_ylabel("Parameter")
    axes[1, 0].set_ylabel(model.observable_names[column].replace("_", " "))
    return fig, {
        "observed_column": column,
        "display_time_interval": [start, stop],
        "reference_time": float(time[marker]),
        "simulation": f"overview.{name}",
        "control": _simulation_metadata(
            model, control, {**arguments, "null_simulation": True}
        ),
        "control_selection": selection,
    }


def example_data(seed, record):
    simulations = []
    metadata = {}
    for column, (model, arguments) in enumerate(
        (
            (MayHarvesting(), {"num_samples": 1200, "noise": 1.0}),
            (SaddleNode(epsilon=1.4), {"noise": 0.01}),
        )
    ):
        result, selection = _example(model, arguments, seed)
        marker = model.locate_transition(
            result.trajectory, False, param_values=result.param_values, time=result.time
        )
        time = np.asarray(result.time)
        values = np.asarray(result.trajectory[:, 0])
        display = np.isfinite(values)
        if column == 1:
            display &= (time >= -8) & (values <= 5)
        limits = (float(time[display].min()), float(time[display].max()))
        control, control_selection = _example(model, arguments, seed, null=True)
        simulations.append(
            (model, time, values, marker, display, np.asarray(control.trajectory[:, 0]))
        )
        metadata[type(model).__name__] = {
            **_simulation_metadata(model, result, arguments),
            **selection,
            "control": _simulation_metadata(
                model, control, {**arguments, "null_simulation": True}
            ),
            "control_selection": control_selection,
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
            "num_samples": stop - start,
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


RECOVERY_COLORS = (BLUE, ORANGE, "#647d49", "#8b6b8f", "#9d852d", "#596579")
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
    for column, (
        (model, time, values, marker, display, control),
        title,
        ylabel,
    ) in enumerate(
        zip(simulations, ("May harvesting", "Saddle-node"), ("Population", "State"))
    ):
        ax = axes[column]
        ax.plot(
            time[display], values[display], color=BLUE, linewidth=1.2, label="Tipping"
        )
        ax.plot(
            time[display], control[display], color=ORANGE, linewidth=1, label="Control"
        )
        ax.legend(loc="lower left")
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
        title="(c) Microcosm · M1",
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
    for column, ((model, time, values, marker, display, control), title) in enumerate(
        zip(simulations, ("May harvesting", "Saddle-node"))
    ):
        for row, (function, label) in enumerate(
            ((rolling_ar1, "AR1 slope"), (rolling_variance, "Variance"))
        ):
            limits = metadata[type(model).__name__]["display_time_limits"]
            shown = time[:marker] >= limits[0]
            for signal, color, name in (
                (values, BLUE, "Tipping"),
                (control, ORANGE, "Control"),
            ):
                indicator = function(signal[:marker], window=200, detrend="linear")
                axes[row, column].plot(
                    time[:marker][shown],
                    indicator[shown],
                    color=color,
                    linewidth=1.1,
                    label=name,
                )
            axes[row, column].axvline(
                time[marker], color=RED, linestyle="--", linewidth=0.8
            )
            axes[row, column].set_ylabel(label)
            axes[row, column].set_xlim(limits)
        axes[0, column].legend(loc="lower left")
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
    axes[0, 2].set(title="(c) Microcosm · M1", ylabel="AR1 slope")
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


def roc_examples(seed, num_per_class):
    fig, ax = plt.subplots(figsize=(4.2, 3.6), layout="constrained")
    manifest = {}
    model = MayHarvesting()
    arguments = {"num_samples": 1000, "noise": 1.0}
    data = generate_dataset(
        model,
        context_samples=300,
        num_tipping=num_per_class,
        num_controls=num_per_class,
        lead_samples=1,
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
    ax.set_title("May harvesting", loc="left", pad=12)
    ax.legend(loc="lower right", fontsize=8)
    _clean(ax)
    leads = [metadata["lead_time"] for metadata in data.metadata]
    manifest[type(model).__name__] = {
        "parameters": model.get_parameters(),
        "arguments": arguments,
        "root_seed": seed,
        "num_per_class": num_per_class,
        "context_samples": 300,
        "lead_samples": 1,
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
        "--num-per-class",
        type=int,
        default=50,
        help="Tipping and null counts per model in the ROC panels",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=43,
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
    if args.num_per_class < 1:
        parser.error("--num-per-class must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    record = Microcosm().load(cache_dir=args.cache_dir, source_path=args.source_path)
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
    runs = {
        name: _example(model, arguments, args.seed)
        for name, model, _, arguments, _, _ in GALLERY
    }
    controls = {
        name: _example(model, arguments, args.seed, null=True)
        for name, model, _, arguments, _, _ in GALLERY
    }
    with mpl.rc_context(STYLE):
        for name, render in (
            ("overview", partial(gallery, runs)),
            ("trajectories", partial(trajectory_examples, examples)),
            ("indicators", partial(indicator_examples, examples)),
            ("roc", partial(roc_examples, args.seed, args.num_per_class)),
        ):
            print(f"Rendering {name}...", flush=True)
            fig, metadata = render()
            _save(fig, name, args.output_dir, args.formats)
            manifest[name] = metadata
            print(f"Rendered {name}", flush=True)
        manifest["model_previews"] = {}
        for name, model, _, arguments, _, _ in GALLERY:
            fig, metadata = model_preview(
                name,
                model,
                runs[name][0],
                controls[name][0],
                arguments,
                controls[name][1],
            )
            _save(fig, name, args.output_dir, args.formats)
            manifest["model_previews"][name] = metadata
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )


if __name__ == "__main__":
    main()
