Trends and ROC/AUROC
====================

Use a labeled ensemble of independent trajectories: ``1`` denotes tipping and
``0`` denotes a non-tipping control. One trajectory contributes one scalar score.
The overlapping indicator windows within a trajectory are not independent
classification examples. A single empirical transition can illustrate an
indicator, but cannot supply an ensemble AUROC by itself.

.. code-block:: python

   from tipbox.datasets import generate_dataset
   from tipbox.indicators import rolling_ar1, rolling_variance
   from tipbox.metrics import ensemble_roc, plot_roc
   from tipbox.models import MayHarvesting

   dataset = generate_dataset(
       MayHarvesting(),
       num_timesteps=300,
       n_tipping=10,
       n_null=10,
       n=1000,               # Full simulation length, including initial sample.
       noise=0.01,
       lead_steps=1,
       seed=42,
   )
   signals = dataset.observations[:, :, 0]

   indicators = {
       "AR1": [rolling_ar1(x, window=100, detrend="linear") for x in signals],
       "Variance": [rolling_variance(x, window=100, detrend="linear") for x in signals],
   }
   result = ensemble_roc(dataset.labels, indicators)
   print(result.curves["Variance"].auroc)
   ax = plot_roc(result)

``ensemble_roc`` reduces each supplied indicator series to its Kendall trend.
Alternatively, supply one precomputed scalar per trajectory, or call
``binary_roc(labels, scores)`` for a single score vector. Negate a score before
calling these functions if a decreasing trend is your predeclared warning
direction. Signs are never automatically selected to improve AUROC.

Labeled generation
------------------

``generate_dataset`` accepts a configured model and a required retained length.
It calls the same :meth:`~tipbox.types.BoxModel.make_trajectory` interface as
individual simulations. It does not change the model's configuration, forcing
duration, or time origin. In the example, ``n=1000`` provides enough full-path
context for the 300 retained observations and both reference markers. Omitting
``n`` uses the model's configured start/end interval, which may be too short for
your requested window. A full simulation shorter than the context plus lead
raises ``ValueError``; exhausted label/context rejections raise ``RuntimeError``
with counts. Adjust the simulation grid or context explicitly.

.. list-table:: Batch controls and defaults
   :header-rows: 1
   :widths: 24 20 56

   * - Argument
     - Default
     - Meaning
   * - ``model``
     - required
     - Configured ``BoxModel`` instance; coefficients remain inspectable.
   * - ``num_timesteps``
     - required
     - Positive retained observation count per trajectory.
   * - ``n_tipping``, ``n_null``
     - 100 each
     - Nonnegative requested counts; at least one overall. Null rows precede
       positive rows. No shuffling is applied.
   * - ``lead_steps``
     - 1
     - Positive sample distance from the last retained observation to the marker.
   * - ``seed``
     - 0
     - Root seed in [0, 2**32); the complete reserved attempt range must fit.
   * - ``max_attempts``
     - 100
     - Maximum attempts per requested trajectory, including rejected labels,
       absent markers, insufficient context, and nonfinite paths.
   * - ``**trajectory_kwargs``
     - model defaults
     - ``n``, ``dt``, ``dt_sample``, ``noise``, ``initial_state``, ``innovations``,
       ``param_end_multiplier``, and ``forcing`` pass directly to ``make_trajectory``.

For zero-based row :math:`i` and attempt :math:`a`, the simulation seed is
``seed + i * max_attempts + a``. Distinct slots have distinct seeds without
wraparound. Changing an earlier row's rejection count does not change later
rows' seed slots. This layout is specific to the batch interface; it does not
reproduce legacy ``generate`` key splitting. Supplied innovations are reused
for each draw, so distinct seeds then affect only the default initial-state
sampler. Fixed initial states remove initial-state variation while default
process noise remains independently seeded. A custom forcing replaces both
class schedules, and all custom input arrays/callables must be retained for replay.

The result has ``observations`` shaped (trajectory, time, observable),
``time`` and ``param_values`` shaped (trajectory, time), a binary ``labels``
vector, and one ``metadata`` mapping per trajectory. All model observables are
retained, including three AMOC columns [Q, SN, ST]; no legacy zero-padding is
introduced for scalar models. Times keep the model's absolute clock. Metadata
includes the full initial state, accepted seed and attempt, rejected-attempt
counts, resolved coefficients, integration and output spacing, step-noise scale,
reference index/time, retained full-path bounds, and actual physical lead
duration. The metadata's physical state path is cropped to the retained context;
``initial_state`` still refers to the beginning of the full simulation.

Retained context and its reference observation must be finite. Positive paths
may diverge after the reference, as happens for the escaping saddle-node normal
form. A nonfinite reference is rejected; it cannot itself establish an event.
Null paths must remain finite over the entire simulated interval. Model
exceptions propagate immediately rather than being hidden as ordinary rejections.
Rate-AMOC's collapse classifier requires a finite transport value below its
threshold, so a later numerical ``-inf`` cannot supply the only evidence of
collapse after its time-zero reference.

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

Generation uses rejection sampling. The positive class satisfies
``model.is_tipped`` under forced simulation; the null class fails that same
classifier under null forcing. Rejection therefore conditions on the outcome;
class frequencies in the resulting dataset do not estimate tipping probabilities.
The per-row ``label_rule`` and ``marker_rule`` identify the model methods used,
and metadata records their parameters and accepted seeds.

With ``lead_steps=1``, the final retained index is ``marker_index - 1``. The
slice is ``full[marker_index - lead_steps + 1 - num_timesteps:
marker_index - lead_steps + 1]``. The actual physical lead is the marker time
minus the last retained time. To match the legacy ``generate`` API's gap,
use ``lead_steps = steps_before_transition + 1``. A null
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

.. automodule:: tipbox.datasets.benchmarks
   :members: BenchmarkDataset, generate_dataset

.. automodule:: tipbox.metrics
   :members: BinaryRocResult, EnsembleRocResult, kendall_tau, binary_roc, ensemble_roc, plot_roc
