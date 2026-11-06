API reference
=============

Choose the operation first. All public imports below work from the installed
package; examples use ``T`` time points, ``D`` observable columns, and ``N``
independent trajectories.

.. list-table:: API at a glance
   :header-rows: 1
   :widths: 22 37 41
   :class: contract-table

   * - Task
     - Import or method
     - Contract
   * - Simulate
     - ``model.make_trajectory(n, noise=..., seed=...)``
     - ``TrajectoryResult``: time ``(T,)``, observations ``(T,D)``, forcing ``(T,)``, noise scale, and metadata.
   * - Build a labeled batch
     - ``tipbox.datasets.generate_dataset``
     - ``BenchmarkDataset``: observations ``(N,T,D)``, labels ``(N,)``, time and forcing ``(N,T)``, metadata per row.
   * - Compute a window statistic
     - ``tipbox.indicators.rolling_*``
     - One scalar signal ``(T,)`` becomes an aligned NumPy array ``(T,)``; warm-up, skipped, and invalid windows are ``NaN``.
   * - Score a trend
     - ``tipbox.metrics.kendall_tau``
     - One indicator series becomes one Kendall tau-b score; undefined trends are ``NaN``.
   * - Evaluate scores
     - ``tipbox.metrics.binary_roc``
     - Binary labels plus one score per trajectory become a ``BinaryRocResult`` with AUROC and coverage counts.
   * - Evaluate several indicators
     - ``tipbox.metrics.ensemble_roc``
     - A mapping of indicator series or scalar scores becomes an ``EnsembleRocResult``; larger scores mean tipping.
   * - Plot ROC curves
     - ``tipbox.metrics.plot_roc``
     - Returns a Matplotlib ``Axes``. Undefined curves remain in the result and are omitted from the plot.
   * - Load observations
     - ``tipbox.datasets.load_empirical`` or ``load_veraart2012``
     - ``EmpiricalRecord``: time ``(T,)``, observations ``(T,D)``, and source/sampling metadata. No class labels are inferred.

See :doc:`indicators` for statistic definitions, :doc:`benchmarks` for batch
controls and metric signatures, and :doc:`datasets` for source/cache controls.
For an end-to-end example, start with :doc:`quickstart`.

Discover and configure a model
------------------------------

.. code-block:: python

   from tipbox import get_system_info, list_systems

   print(list_systems())
   model = get_system_info("b_harvesting").create_model()
   print(model.state_names, model.observable_names)
   print(model.get_parameters())

``get_parameters()`` exposes resolved configuration values. ``model.config``
is the immutable typed configuration. A parameter override belongs in the
constructor; integration, sampling, noise, seed, and custom forcing belong in
``make_trajectory``. The :doc:`model collection <models/index>` links equations,
units, defaults, and constructor examples.

References, missingness, and failures
-------------------------------------

* ``n`` counts returned samples, including the initial state. Batch
  ``num_timesteps`` counts retained pre-reference samples.
* ``initial_state`` and ``innovations`` use the model's state/noise dimensions,
  which can differ from its observable dimension. Ricker uses a log-population
  state; its observed column is population.
* ``locate_transition`` returns a reference index, or ``None`` when the requested
  reference is absent. A null reference is a comparison point. State-jump and
  forcing-crossing references have distinct meanings.
* Use fixed integer windows for streaming. Nonfinite observations invalidate
  the whole window; they are not dropped or imputed. A skipped endpoint is
  distinct from a computed statistic equal to zero.
* ROC requires both binary classes in the supplied labels. Missing scores are
  excluded with counts. If retained scores lose a class, AUROC is ``NaN`` and
  the curve is empty. ``plot_roc`` raises ``ValueError`` if no curve is defined.
* Invalid dimensions or controls raise ``TypeError`` or ``ValueError``.
  Exhausted batch rejection sampling raises ``RuntimeError`` with rejection
  counts. Empirical network failures propagate; unavailable offline files raise
  ``FileNotFoundError`` and checksum mismatches raise ``ValueError``.

Array outputs can be converted with ``numpy.asarray``. For strict JSON, convert
nonfinite indicator values to ``null`` and select the metadata fields you need;
``NaN`` is meaningful analysis output, not a zero-valued measurement.

Shared interface
----------------

.. autoclass:: tipbox.types.BoxModel
   :members: config, get_parameters, make_trajectory, is_tipped, locate_transition

.. autoclass:: tipbox.types.TrajectoryResult
   :members:

Discovery
---------

.. autofunction:: tipbox.registry.list_systems

.. autofunction:: tipbox.registry.get_system_info

Historical generation interface
--------------------------------

New code can use the model and batch APIs above. The established entry point
remains available with its original context-buffer conventions; see
:doc:`compatibility` before comparing earlier experiments.

.. autofunction:: tipbox.generate.generate
