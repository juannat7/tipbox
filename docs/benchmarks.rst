Benchmarks and AUROC
====================

Use independent trajectories: ``1`` means tipping, ``0`` means control.
One trajectory contributes one score. Overlapping windows are not independent
classification examples, and one empirical transition cannot supply an AUROC.
The :doc:`quickstart` contains a complete generation-to-ROC example.

Generate a batch
----------------

.. code-block:: python

   from tipbox.datasets import generate_dataset
   from tipbox.models import MayHarvesting

   data = generate_dataset(
       MayHarvesting(), num_timesteps=300, n_tipping=5, n_null=5,
       n=1000, noise=1, lead_steps=1, seed=42,
   )

``num_timesteps`` is retained context; ``n`` is the full simulation length.
Defaults are ``n_tipping=100``, ``n_null=100``, ``lead_steps=1``, ``seed=0``,
and ``max_attempts=100`` per row. All other keywords pass to
:meth:`~tipbox.types.BoxModel.make_trajectory`.

The result contains observations ``(N,T,D)``, time and forcing ``(N,T)``,
labels ``(N,)``, and per-row metadata. Null rows precede positives. Times retain
the absolute model clock. Physical state paths are cropped to context;
``initial_state`` still refers to the full simulation's beginning.

For zero-based row ``i`` and attempt ``a``, the seed is
``seed + i*max_attempts + a``. The reserved range must fit in uint32. Metadata
records accepted seeds, attempts, rejection reasons, parameters, reference
indices/times, context bounds, and physical lead times. Keep custom forcing
and innovation arrays for replay; seeds do not encode them.

Labels and references
---------------------

A positive path satisfies ``model.is_tipped`` under forced simulation; a null
path fails it under null forcing. Rejection sampling conditions on these
outcomes. Class frequencies therefore do not estimate tipping probabilities.
Some classifiers identify a forcing crossing rather than an observed state jump.

With ``lead_steps=1``, the final retained index is ``marker_index-1``.
Generally the slice is ``full[marker-lead+1-num_timesteps:marker-lead+1]``.
The physical lead is the reference time minus the last retained time.
Legacy ``generate`` uses ``lead_steps=steps_before_transition+1``.

Null references are comparison points. Class schedules, physical durations,
and event rules can differ; equal sample counts do not match these quantities.
TipBox's capped ecological null ramps differ from Bury et al.'s fixed null
forcing. Read the :doc:`model reference <models/index>` before comparing papers.

Positive paths must remain finite through the reference; later normal-form
escape is permitted. Null paths must remain finite throughout. An absent
reference or insufficient context causes rejection. Invalid inputs and model
errors propagate; exhausted attempts raise ``RuntimeError`` with counts.

Score and evaluate
------------------

``kendall_tau(series, time=None)`` returns SciPy's tie-corrected Kendall tau-b,
using sample order unless finite, increasing times are supplied. Nonfinite
indicator values are omitted; fewer than two values or a constant gives ``NaN``.

``binary_roc(labels, scores)`` requires both binary classes. Higher scores mean
tipping. At threshold ``c``, predict tipping when ``score >= c``. AUROC is the
trapezoidal area; ties receive half credit. Constant scores give 0.5.

``ensemble_roc(labels, indicators)`` accepts a mapping of per-trajectory series
or scalar scores. Series are reduced to Kendall trends. Negate decreasing
indicators before evaluation; signs are never optimized against the labels.

Nonfinite scores are excluded and counted in ``n_missing``. Report AUROC with
``n_valid``, ``n_positive``, and ``n_negative``. If filtering loses a class,
the curve is empty and AUROC is ``NaN``. Different indicators may retain
different subsets; select a common finite subset for matched comparisons.
``plot_roc`` omits undefined curves and raises if none remain.

References: `SciPy Kendall tau-b
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html>`_.
