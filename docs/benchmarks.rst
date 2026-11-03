Trends and ROC/AUROC
====================

Use a labeled ensemble of independent trajectories: ``1`` denotes tipping and
``0`` denotes a non-tipping control. One trajectory contributes one scalar score.
The overlapping indicator windows within a trajectory are not independent
classification examples. A single empirical transition can illustrate an
indicator, but cannot supply an ensemble AUROC by itself.

.. code-block:: python

   import jax
   from tipbox.generate import generate
   from tipbox.indicators import rolling_ar1, rolling_variance
   from tipbox.metrics import ensemble_roc, plot_roc

   labels, signals = [], []
   for label in (0, 1):
       for replicate in range(10):
           _, segment, info = generate(
               jax.random.PRNGKey(10 * label + replicate),
               tipping_type="b_harvesting",
               null_simulation=(label == 0),
               num_timesteps=300,
               steps_before_transition=1,
               sigma_tilde=0.01,
           )
           labels.append(label)
           signals.append(segment.trajectory[:, 0])

   indicators = {
       "AR1": [rolling_ar1(x, window=100, detrend="linear") for x in signals],
       "Variance": [rolling_variance(x, window=100, detrend="linear") for x in signals],
   }
   result = ensemble_roc(labels, indicators)
   print(result.curves["Variance"].auroc)
   ax = plot_roc(result)

``ensemble_roc`` reduces each supplied indicator series to its Kendall trend.
Alternatively, supply one precomputed scalar per trajectory, or call
``binary_roc(labels, scores)`` for a single score vector. Negate a score before
calling these functions if a decreasing trend is your predeclared warning
direction. Signs are never automatically selected to improve AUROC.

Kendall trend
-------------

``kendall_tau(series, time=None)`` uses sample indices by default, or finite,
strictly increasing physical times. It removes nonfinite indicator values,
including the warm-up prefix. With :math:`C` concordant pairs, :math:`D`
discordant pairs, and :math:`T_x,T_y` pairs tied only in time or indicator,
respectively, the tie-corrected score is

.. math::

   \tau_b=\frac{C-D}{\sqrt{(C+D+T_x)(C+D+T_y)}}.

TipBox uses `SciPy's kendalltau <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html>`_
with ``variant="b"``. Fewer than two
finite values or a constant series gives ``NaN``. A monotone physical time axis
has the same rank ordering as sample indices, even with unequal time intervals;
this does not make sample-based AR lags physically uniform.

ROC definition and coverage
---------------------------

At threshold :math:`c`, predict tipping when :math:`s\ge c`. The curve gives
:math:`\mathrm{TPR}=\mathrm{TP}/N_+` against
:math:`\mathrm{FPR}=\mathrm{FP}/N_-` over all distinct finite score thresholds,
starting at :math:`+\infty`. AUROC is the trapezoidal area under that curve.
Tied positive/negative scores receive half credit. Perfect ranking has AUROC
one, reversed ranking zero, and a constant score gives one half.

``binary_roc`` requires a one-dimensional binary label vector containing both
classes. Nonfinite scores are excluded and reported in ``n_missing``;
``n_valid``, ``n_positive``, and ``n_negative`` describe the retained ensemble.
If either class disappears after filtering, AUROC is ``NaN`` and curve arrays
are empty. ``ensemble_roc`` retains such indicators in its result rather than
silently dropping them. ``plot_roc`` displays only defined curves and includes
their retained counts; it raises if none are defined.

Report AUROC together with coverage and class counts. Different missing-score
patterns can leave different subsets for different indicators; explicitly select
a common finite subset before evaluation if you need a matched comparison.

Event definitions and interpretation
------------------------------------

The generation example uses the historical rejection-sampled ``generate`` API.
Its positive class contains trajectories satisfying the model's tipping
classifier; the null class passes that classifier under a different forcing
schedule. Rejection therefore conditions on the outcome. The ``info`` mapping
records attempts, configuration, and the full-trajectory reference index.
Retain this metadata for benchmark provenance.

``steps_before_transition=1`` omits one additional sample between the segment
and the marker: the final retained index is ``transition_index - 2``. A null
reference is a model-defined comparison marker, not an observed transition.
For some models the positive marker is a configured bifurcation or forcing
reference rather than the observed state jump. See :doc:`models/index` and
:doc:`compatibility` for each convention. Equal sample counts do not guarantee
matched forcing histories or physical lead times across classes or models.
Do not interpret this illustrative AUROC as a universal detection score.

Keep trajectories or entire experiments together when splitting training and
test data. Select observables, sampling, lead times, windows, detrending, and
directions before evaluating the test set. This API provides descriptive scores;
it does not estimate confidence intervals or calibrate a deployment threshold.

API
---

.. automodule:: tipbox.metrics
   :members: BinaryRocResult, EnsembleRocResult, kendall_tau, binary_roc, ensemble_roc, plot_roc
