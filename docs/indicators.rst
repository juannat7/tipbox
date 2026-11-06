Classical warning indicators
============================

Choose one scalar observable before analysis. For example, the first observation
column of :class:`~tipbox.models.MayHarvesting` is population density:

.. code-block:: python

   from tipbox.models import MayHarvesting
   from tipbox.indicators import rolling_ar1, rolling_variance

   result = MayHarvesting().make_trajectory(1000, seed=42)
   population = result.trajectory[:, 0]
   ar1 = rolling_ar1(population, window=200, detrend="linear")
   variance = rolling_variance(population, window=200, detrend="linear")

The functions accept concrete NumPy or JAX arrays and return NumPy arrays of the
same length. They are eager analysis routines, rather than JAX-transformable
integration kernels. Their definitions follow the classical comparisons in
`kews <https://github.com/juannat7/kews/blob/main/experiments/indicators.py>`_.
TipBox adds explicit missing-data handling and population moments.

Window preparation and statistics run in bounded batches. Long records do not
materialize the complete window matrix; computational work still grows with
the number and length of windows. Definitions, missingness rules, and trailing
alignment are preserved; floating-point roundoff can depend on batch shape.

Windows and defaults
--------------------

* ``window=0.5`` uses half the input length, rounded down. An integer such as
  ``200`` means samples; a float in ``(0, 1]`` means a fraction. ``1.0`` selects
  the full series, while the integer ``1`` is invalid. At least two samples
  are required, with additional lag requirements below. A one-ULP correction
  at integer boundaries makes ``window=0.29`` select 29 out of 100 samples.
* ``step=1`` evaluates every complete trailing window. ``step=5`` evaluates the
  first complete window and every fifth window thereafter. Every result is
  aligned to its window's last observation. Warm-up and skipped positions are
  ``NaN``.
* ``detrend=None`` uses raw observations. ``detrend="linear"`` fits an intercept
  and slope separately inside each window, then analyzes the residuals. No
  full-series smoothing is applied. Later observations cannot change earlier
  output values when using a fixed integer window.
* Any ``NaN`` or infinite observation invalidates its entire window. The
  functions never drop observations or join across a gap. Constant finite
  windows have zero variance and undefined normalized statistics (``NaN``).

Computations first translate each window by its first value to avoid spurious
moments for decimal-valued constants. After linear detrending, a residual whose
maximum magnitude is at most eight float64 machine epsilons times the translated
window's maximum magnitude is treated as numerical zero. This prevents fitting
roundoff from becoming a warning signal on an exactly linear input.

Windows and lags count samples. These statistics assume equally spaced
observations: a lag of one has a fixed physical duration only on a regular grid.
For irregular empirical records, state and justify a resampling policy before
using autocorrelation. Analyze separate experimental recovery segments
separately. Choose any resampling, detrending, window, and indicator direction
using training data or prior knowledge before evaluating held-out labels.

Definitions
-----------

For the :math:`w` observations :math:`z_i` in a window (or their detrended
residuals), define :math:`\bar z=w^{-1}\sum_i z_i` and
:math:`m_p=w^{-1}\sum_i(z_i-\bar z)^p`.

.. list-table:: Statistics and additional defaults
   :header-rows: 1
   :widths: 24 42 34

   * - Function
     - Definition
     - Additional controls
   * - ``rolling_variance``
     - :math:`\sum_i(z_i-\bar z)^2/(w-\mathrm{ddof})`
     - ``ddof=0``; integer in :math:`[0,w)`
   * - ``rolling_skewness``
     - :math:`m_3/m_2^{3/2}`
     - Population moment, no bias correction
   * - ``rolling_kurtosis``
     - :math:`m_4/m_2^2-3`
     - ``excess=True``; False omits the subtraction
   * - ``rolling_ar1``
     - Lagged OLS slope with an intercept
     - ``lag=1`` sample; at least two paired observations
   * - ``rolling_acf``
     - Lagged Pearson correlation
     - ``lag=1`` sample; at least two paired observations

For lag :math:`\ell`, construct :math:`u=z_{0:w-\ell}` and
:math:`v=z_{\ell:w}` and center each paired vector separately. Writing
:math:`S_{uv}=\sum_i(u_i-\bar u)(v_i-\bar v)`, the AR slope is
:math:`S_{uv}/S_{uu}`, while the Pearson correlation is
:math:`S_{uv}/\sqrt{S_{uu}S_{vv}}`. A nonzero denominator is required.
The AR slope can exceed :math:`[-1,1]`. The correlation differs from an ACF
estimated using a shared full-window mean and variance, so specify the
convention when comparing implementations.

These classical signals need not increase before every transition, especially
for rate-induced tipping. Skewness also depends on the sign of the chosen
observable. Inspect the physics and predeclare score directions rather than
flipping indicators after seeing their test AUROC.

The population skewness and kurtosis conventions match SciPy's
`skew(bias=True) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html>`_
and `kurtosis(bias=True) <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html>`_.

API
---

.. automodule:: tipbox.indicators
   :members: window_size, rolling_variance, rolling_ar1, rolling_acf, rolling_skewness, rolling_kurtosis
