"""Trajectory-level Kendall trends and binary ROC/AUROC for indicator ensembles."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import kendalltau


@dataclass(frozen=True)
class BinaryRocResult:
    """ROC for finite scores; a missing class gives empty curves and NaN AUROC.

    ``n_positive`` and ``n_negative`` count retained labels. ``n_missing`` counts
    excluded scores, so n_valid + n_missing is the original ensemble size.
    Thresholds descend from infinity, and predictions use score >= threshold.
    """

    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auroc: float
    n_valid: int
    n_missing: int
    n_positive: int
    n_negative: int


@dataclass(frozen=True)
class EnsembleRocResult:
    """One scalar score per trajectory and one ROC result per named indicator."""

    y_true: np.ndarray
    scores: dict[str, np.ndarray]
    curves: dict[str, BinaryRocResult]


def kendall_tau(series, *, time=None) -> float:
    """Kendall tau-b trend against time, dropping only nonfinite indicator values.

    ``series`` must be one dimensional. ``time=None`` uses sample indices;
    otherwise supply finite, strictly increasing times of the same shape.
    Fewer than two finite values or a constant indicator gives NaN. The result
    uses SciPy's tie-corrected tau-b; the p-value is not used. Filtering warm-up
    NaNs does not assert that the underlying windows were independent.
    """
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError("series must be one dimensional")
    time = np.arange(values.size) if time is None else np.asarray(time, dtype=float)
    if (
        time.shape != values.shape
        or not np.isfinite(time).all()
        or not (np.diff(time) > 0).all()
    ):
        raise ValueError("time must be finite, strictly increasing, and match series")
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return float("nan")
    return float(kendalltau(time[valid], values[valid], variant="b")[0])


def _labels(y_true):
    labels = np.asarray(y_true)
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("y_true must be a one-dimensional binary 0/1 array")
    if np.unique(labels).size != 2:
        raise ValueError("y_true must contain both classes 0 and 1")
    return labels.astype(int)


def binary_roc(y_true, scores) -> BinaryRocResult:
    """ROC and AUROC from one scalar score per independent trajectory.

    Parameters
    ----------
    y_true : array-like, shape (n_trajectories,)
        Binary labels: 1 for tipping, 0 for non-tipping. Both classes are required.
    scores : array-like, shape (n_trajectories,)
        Larger means stronger evidence for tipping. Choose any direction change
        before evaluating labels; this function never maximizes over signs.
        NaN and infinite scores are excluded, with coverage counts returned.

    Notes
    -----
    Tied scores share a threshold and contribute half credit to AUROC. If finite
    scores retain only one class, return NaN AUROC and empty curve arrays.
    Nonfinite scores never become negative predictions implicitly.
    """
    labels = _labels(y_true)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != labels.shape:
        raise ValueError("scores must be one dimensional and match y_true")
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    n_positive = int(labels.sum())
    n_negative = int(labels.size - n_positive)
    counts = {
        "n_valid": int(valid.sum()),
        "n_missing": int((~valid).sum()),
        "n_positive": n_positive,
        "n_negative": n_negative,
    }
    if n_positive == 0 or n_negative == 0:
        return BinaryRocResult(
            fpr=np.array([]),
            tpr=np.array([]),
            thresholds=np.array([]),
            auroc=float("nan"),
            **counts,
        )
    order = np.argsort(-scores, kind="stable")
    labels, scores = labels[order], scores[order]
    endpoints = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), scores.size - 1]
    positives = np.cumsum(labels)[endpoints]
    negatives = endpoints + 1 - positives
    tpr = np.r_[0.0, positives / n_positive]
    fpr = np.r_[0.0, negatives / n_negative]
    auroc = float(np.dot(np.diff(fpr), tpr[1:] + tpr[:-1]) / 2)
    return BinaryRocResult(
        fpr=fpr,
        tpr=tpr,
        thresholds=np.r_[np.inf, scores[endpoints]],
        auroc=auroc,
        **counts,
    )


def ensemble_roc(y_true, indicators: Mapping) -> EnsembleRocResult:
    """Evaluate named indicator series or precomputed trajectory scores.

    Each mapping value contains one item per trajectory. An item is either a
    scalar score or a one-dimensional indicator series reduced with
    :func:`kendall_tau`. Series can differ in length. Input trajectories and
    labels must be aligned; higher scores always mean more evidence for tipping.
    Even undefined indicators remain in the result, with NaN AUROC and counts.
    """
    labels = _labels(y_true)
    if not indicators:
        raise ValueError("indicators must contain at least one named indicator")
    scores, curves = {}, {}
    for name, values in indicators.items():
        if len(values) != labels.size:
            raise ValueError(f"{name}: number of trajectories must match y_true")
        score = np.asarray(
            [item if np.ndim(item) == 0 else kendall_tau(item) for item in values],
            dtype=float,
        )
        scores[name] = score
        curves[name] = binary_roc(labels, score)
    return EnsembleRocResult(y_true=labels, scores=scores, curves=curves)


def plot_roc(result: EnsembleRocResult, *, ax=None):
    """Plot defined ensemble ROC curves and coverage; return the Matplotlib axes.

    Pass an existing axes to control the figure and styling. Undefined curves
    remain available in result.curves but are omitted from the plot. Raise
    ValueError if every curve is undefined. Matplotlib is imported on demand.
    """
    if not any(np.isfinite(curve.auroc) for curve in result.curves.values()):
        raise ValueError("no ROC curves retain finite scores from both classes")
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
    for name, curve in result.curves.items():
        if np.isfinite(curve.auroc):
            ax.plot(
                curve.fpr,
                curve.tpr,
                label=f"{name} (AUC {curve.auroc:.2f}; n={curve.n_valid}/{len(result.y_true)})",
            )
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="False positive rate",
        ylabel="True positive rate",
        aspect="equal",
    )
    ax.legend()
    return ax
