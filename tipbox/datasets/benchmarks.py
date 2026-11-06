"""Labeled, fixed-length context from reproducible box-model simulations."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..types import BoxModel


@dataclass(frozen=True)
class BenchmarkDataset:
    """One row per accepted trajectory, with auditable simulation metadata.

    ``observations`` is (n_trajectories, n_time, n_observables), ``time`` and
    ``param_values`` are (n_trajectories, n_time), and ``labels`` is a binary
    vector. Times retain the model's absolute clock. Metadata contains one
    mapping per row, including resolved parameters, the accepted seed, rejection
    counts, full-trajectory bounds, reference time, and physical lead duration.
    Physical state paths in metadata are cropped to the retained context.
    """

    observations: np.ndarray
    time: np.ndarray
    param_values: np.ndarray
    labels: np.ndarray
    metadata: tuple[dict, ...]


def generate_dataset(
    model: BoxModel,
    num_timesteps: int,
    *,
    n_tipping: int = 100,
    n_null: int = 100,
    lead_steps: int = 1,
    seed: int = 0,
    max_attempts: int = 100,
    **trajectory_kwargs,
) -> BenchmarkDataset:
    """Generate labeled pre-reference context using the shared model interface.

    Parameters
    ----------
    model : BoxModel
        Configured named model. No configuration or time origin is changed.
    num_timesteps : int
        Positive number of observations retained before each reference marker.
    n_tipping, n_null : int, default=100 each
        Nonnegative class counts; require at least one trajectory overall.
        Label 1 passes model.is_tipped under forced simulation; label 0 fails it
        under null simulation. These are the model's classification conventions,
        which can describe forcing crossings rather than observed state jumps.
        Null rows precede tipping rows; no shuffling is performed.
    lead_steps : int, default=1
        Positive sample distance from the final retained sample to the marker.
        One ends immediately before the marker. This differs from legacy
        generate's steps_before_transition, which counts additional omitted
        samples: lead_steps = steps_before_transition + 1.
    seed : int, default=0
        Root seed in [0, 2**32). Draw i, attempt a (both zero based), uses
        seed + i * max_attempts + a. The entire reserved range must fit in uint32,
        ensuring distinct seeds without wraparound. Metadata records actual seeds.
    max_attempts : int, default=100
        Maximum attempts per requested row. Rejections and invalid context are
        counted; exhausted attempts raise RuntimeError without returning a partial
        dataset. Exceptions from the model propagate immediately.
    **trajectory_kwargs
        Forwarded to model.make_trajectory: n, dt, dt_sample, noise, initial_state,
        innovations, param_end_multiplier, and forcing. Defaults remain the
        model's defaults. seed and null_simulation are controlled by this function.
        An explicit n sets the full simulation length, not the retained length.

    Notes
    -----
    This is outcome-conditioned rejection sampling, not an unbiased sample of
    tipping probabilities. Null markers are comparison references. Equal context
    lengths need not mean matched forcing histories. Failed or nonfinite context
    is never silently relabeled. Null simulations must remain finite throughout;
    positive simulations may diverge after their finite reference crossing.
    Supplied innovations are reused for each draw; seeds then affect only the
    default initial-state sampler. Retain custom inputs for simulation replay.
    """
    if not isinstance(model, BoxModel):
        raise TypeError("model must be a configured BoxModel instance")
    controls = {
        "num_timesteps": num_timesteps,
        "n_tipping": n_tipping,
        "n_null": n_null,
        "lead_steps": lead_steps,
        "seed": seed,
        "max_attempts": max_attempts,
    }
    for name, value in controls.items():
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
    if num_timesteps < 1 or lead_steps < 1 or max_attempts < 1:
        raise ValueError("num_timesteps, lead_steps, and max_attempts must be positive")
    n_total = int(n_tipping) + int(n_null)
    if n_tipping < 0 or n_null < 0 or n_total == 0:
        raise ValueError(
            "class counts must be nonnegative with at least one trajectory"
        )
    if seed < 0 or int(seed) + n_total * int(max_attempts) > 2**32:
        raise ValueError("seed and the reserved attempt seeds must fit in uint32")

    labels = np.repeat([0, 1], [n_null, n_tipping])
    observations, times, parameters, metadata = [], [], [], []
    for index, label in enumerate(labels):
        rejected = {"label": 0, "marker": 0, "context": 0, "nonfinite": 0}
        for attempt in range(max_attempts):
            sample_seed = int(seed) + index * int(max_attempts) + attempt
            full = model.make_trajectory(
                seed=sample_seed,
                null_simulation=not bool(label),
                **trajectory_kwargs,
            )
            if full.time.size < num_timesteps + lead_steps:
                raise ValueError(
                    "Simulation is too short for the requested context and lead. "
                    "Increase n or the configured duration."
                )
            tipped = model.is_tipped(full.trajectory, param_values=full.param_values)
            if tipped != bool(label):
                rejected["label"] += 1
                continue
            marker = model.locate_transition(
                full.trajectory,
                null_simulation=not bool(label),
                time=full.time,
                param_values=full.param_values,
            )
            if marker is None:
                rejected["marker"] += 1
                continue
            if (
                not isinstance(marker, Integral)
                or isinstance(marker, bool)
                or not 0 <= marker < full.time.size
            ):
                raise ValueError("model returned an invalid reference index")
            stop = marker - lead_steps + 1
            start = stop - num_timesteps
            if start < 0:
                rejected["context"] += 1
                continue
            finite_stop = marker + 1 if label else full.time.size
            state = full.metadata["state_trajectory"]
            if not (
                np.isfinite(full.trajectory[:finite_stop]).all()
                and np.isfinite(state[:finite_stop]).all()
            ):
                rejected["nonfinite"] += 1
                continue
            observations.append(np.asarray(full.trajectory[start:stop]))
            times.append(np.asarray(full.time[start:stop]))
            parameters.append(np.asarray(full.param_values[start:stop]))
            metadata.append(
                {
                    **full.metadata,
                    "state_trajectory": np.asarray(state[start:stop]),
                    "root_seed": int(seed),
                    "label": int(label),
                    "label_rule": model.is_tipped.__qualname__,
                    "marker_rule": model.locate_transition.__qualname__,
                    "marker_kind": "model_reference" if label else "null_reference",
                    "simulation_length": int(full.time.size),
                    "marker_index": int(marker),
                    "marker_time": float(full.time[marker]),
                    "segment_start_index": int(start),
                    "segment_stop_index": int(stop),
                    "lead_steps": int(lead_steps),
                    "lead_time": float(full.time[marker] - full.time[stop - 1]),
                    "integration_step_noise_scale": np.asarray(full.sigma).tolist(),
                    "attempt": attempt + 1,
                    "rejections": rejected,
                }
            )
            break
        else:
            raise RuntimeError(
                f"Could not generate row {index}, label {label}, after "
                f"{max_attempts} attempts; rejections={rejected}. Adjust the "
                "simulation duration, model parameters, context, or lead."
            )
    return BenchmarkDataset(
        observations=np.stack(observations),
        time=np.stack(times),
        param_values=np.stack(parameters),
        labels=labels,
        metadata=tuple(metadata),
    )
