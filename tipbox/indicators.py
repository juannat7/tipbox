"""Early-warning statistics on trailing windows of one scalar signal.

All functions return a NumPy array aligned with the input. Warm-up samples and
windows containing nonfinite observations are NaN. Each window uses observations
through its endpoint; linear detrending is fitted within that window. For streams,
use an integer window: fractional windows depend on the full input length.
These eager analysis functions accept NumPy and concrete JAX arrays.
"""

from numbers import Integral, Real

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def window_size(n: int, window: float) -> int:
    """Resolve a window to samples: integer count, or float fraction in (0, 1].

    Fractions are rounded down, allowing one floating-point step at integer
    boundaries (e.g. 0.29 * 100). Require at least two samples and at most n.
    In particular, ``1`` is a one-sample count (invalid), while ``1.0`` selects
    the full series. Noninteger absolute counts such as ``10.5`` are invalid.
    """
    if isinstance(window, bool) or not isinstance(window, Real):
        raise TypeError("window must be an integer count or a float fraction")
    if isinstance(window, Integral):
        size = int(window)
    elif np.isfinite(window) and 0 < window <= 1:
        size = int(np.nextafter(window * n, np.inf))
    else:
        raise ValueError("a fractional window must be finite and in (0, 1]")
    if not 2 <= size <= n:
        raise ValueError(f"window must resolve to between 2 and {n} samples")
    return size


def _rolling(data, window, step, detrend, statistic, **controls):
    values = np.asarray(data, dtype=float)
    if values.ndim != 1:
        raise ValueError("indicators require a one-dimensional scalar signal")
    size = window_size(values.size, window)
    if not isinstance(step, Integral) or isinstance(step, bool) or step < 1:
        raise ValueError("step must be a positive integer sample count")
    if detrend not in (None, "linear"):
        raise ValueError("detrend must be None or 'linear'")
    windows = sliding_window_view(values, size)[::step]
    result = np.full(values.size, np.nan)
    output = result[size - 1 :: step]
    if detrend == "linear":
        time = np.arange(size, dtype=float) - (size - 1) / 2
        time_norm = time @ time
    # The stride view is cheap; bound the arrays used to prepare and reduce it.
    batch_size = max(1, 1_000_000 // size)
    for start in range(0, len(windows), batch_size):
        batch = windows[start : start + batch_size]
        valid = np.isfinite(batch).all(axis=1)
        # Invalid windows use zeros during arithmetic, then become NaN below.
        batch = np.where(valid[:, None], batch, 0.0)
        # Translation avoids mean-roundoff artifacts for decimal-valued constants.
        batch = batch - batch[:, :1]
        if detrend == "linear":
            scale = np.max(np.abs(batch), axis=1, keepdims=True)
            centered = batch - batch.mean(axis=1, keepdims=True)
            slopes = (centered @ time) / time_norm
            batch = centered - slopes[:, None] * time
            tolerance = 8 * np.finfo(float).eps * scale
            batch = np.where(
                np.max(np.abs(batch), axis=1, keepdims=True) <= tolerance,
                0.0,
                batch,
            )
        output[start : start + len(batch)] = np.where(
            valid, statistic(batch, **controls), np.nan
        )
    return result


def _variance(windows, ddof):
    if (
        not isinstance(ddof, Integral)
        or isinstance(ddof, bool)
        or not 0 <= ddof < windows.shape[1]
    ):
        raise ValueError("ddof must be an integer in [0, window_size)")
    return windows.var(axis=1, ddof=ddof)


def rolling_variance(data, window=0.5, *, ddof=0, step=1, detrend=None):
    """Trailing variance; ``ddof=0`` gives the population variance.

    Parameters
    ----------
    data : array-like, shape (n,)
        One scalar signal, sampled at regular intervals. Analyze separate recovery
        segments separately; no interpolation or missing-value removal is applied.
    window : int or float, default=0.5
        Integer sample count or float fraction of the input length, rounded down.
    ddof : int, default=0
        Variance denominator is window_size - ddof. Require 0 <= ddof < window_size.
    step : int, default=1
        Evaluate every step-th window, starting with the first complete window.
        Unevaluated output positions remain NaN.
    detrend : None or "linear", default=None
        Subtract a least-squares line fitted within each window, or use raw values.

    Returns
    -------
    numpy.ndarray, shape (n,)
        Values at the right edge of each evaluated window; NaN elsewhere.
        Constant finite windows have variance zero.
    """
    return _rolling(data, window, step, detrend, _variance, ddof=ddof)


def _correlation(windows, lag, pearson):
    size = windows.shape[1]
    if (
        not isinstance(lag, Integral)
        or isinstance(lag, bool)
        or not 1 <= lag <= size - 2
    ):
        raise ValueError("lag must leave at least two pairs in each window")
    x = windows[:, :-lag] - windows[:, :1]
    y = windows[:, lag:] - windows[:, lag : lag + 1]
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    denominator = (x * x).sum(axis=1)
    if pearson:
        denominator = np.sqrt(denominator * (y * y).sum(axis=1))
    statistic = np.divide(
        (x * y).sum(axis=1),
        denominator,
        out=np.full(len(windows), np.nan),
        where=denominator > 0,
    )
    return statistic


def rolling_ar1(data, window=0.5, *, lag=1, step=1, detrend=None):
    """Trailing lagged OLS slope with an intercept, default lag=1 (AR1).

    Regress x[t] on x[t-lag] inside each window, centering each paired vector
    separately. This slope can exceed [-1, 1]. Require at least two pairs and
    nonzero predictor variance; undefined slopes are NaN. ``lag`` is a sample
    count, not a physical-time interval. The window, step and detrend arguments
    have the same meanings as in :func:`rolling_variance`.
    """
    return _rolling(data, window, step, detrend, _correlation, lag=lag, pearson=False)


def rolling_acf(data, window=0.5, *, lag=1, step=1, detrend=None):
    """Trailing Pearson correlation of the two lagged, separately centered vectors.

    This uses sqrt(Sxx * Syy) in the denominator; it is not the full-window
    autocovariance divided by the full-window variance. Require at least two
    pairs and nonzero variance in both vectors; undefined correlations are NaN.
    Arguments share the meanings in :func:`rolling_ar1`.
    """
    return _rolling(data, window, step, detrend, _correlation, lag=lag, pearson=True)


def _moment(windows, order):
    centered = windows - windows.mean(axis=1, keepdims=True)
    variance = (centered**2).mean(axis=1)
    statistic = np.divide(
        (centered**order).mean(axis=1),
        variance ** (order / 2),
        out=np.full(len(windows), np.nan),
        where=variance > 0,
    )
    return statistic


def rolling_skewness(data, window=0.5, *, step=1, detrend=None):
    """Trailing population skewness m3 / m2**1.5, without bias correction.

    Constant windows have undefined skewness (NaN). Window, step and detrend
    arguments have the same meanings as in :func:`rolling_variance`.
    """
    return _rolling(data, window, step, detrend, _moment, order=3)


def rolling_kurtosis(data, window=0.5, *, excess=True, step=1, detrend=None):
    """Trailing population kurtosis, without bias correction.

    ``excess=True`` (default) returns m4 / m2**2 - 3, whose Gaussian limit is zero.
    ``excess=False`` returns Pearson kurtosis, whose Gaussian limit is three.
    Constant windows are undefined (NaN). Other arguments share the meanings in
    :func:`rolling_variance`.
    """
    return _rolling(data, window, step, detrend, _moment, order=4) - (
        3 if excess else 0
    )
