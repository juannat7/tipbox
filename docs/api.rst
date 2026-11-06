API reference
=============

.. _simulation-api:

Simulation
----------

Constructors take the parameters listed on each :doc:`model page <models/index>`.
``make_trajectory`` returns time ``(T,)``, observations ``(T,D)``, and forcing ``(T,)``.

* ``num_samples`` includes the initial sample; ``dt_sample`` must be an integer multiple of ``dt``.
* Duration-normalized ramps change when ``num_samples`` changes. Rate forcing uses absolute time.
* ``noise`` scales diffusion; SaddleNode instead takes a diffusion intensity.
* Initial states use ``model.state_names``; Ricker uses log population.
* ``noise=0`` removes process noise. Supply ``initial_state`` for fixed initialization.
* Retain custom ``forcing`` and ``innovations`` alongside the seed to replay a run.

.. autoclass:: tipbox.types.BoxModel
   :members: config, get_parameters, make_trajectory, is_tipped, locate_transition

.. autoclass:: tipbox.types.TrajectoryResult
   :members:

.. _indicator-api:

Indicators
----------

Functions return NumPy arrays aligned to the input signal. Windows are trailing;
warm-up, skipped, and nonfinite windows produce ``NaN``. Autocorrelation assumes
regular sampling. Use integer windows for streaming; fractional windows depend
on the series length. Analyze recovery intervals separately.

.. list-table:: Shared options
   :header-rows: 1
   :class: catalog-table

   * - Argument
     - Default
     - Meaning
   * - ``window``
     - ``0.5``
     - Integer sample count, or fraction of the signal length in ``(0,1]``
   * - ``step``
     - ``1``
     - Spacing between evaluated window endpoints
   * - ``detrend``
     - ``None``
     - ``"linear"`` fits and removes a line inside each window

For centered lagged pairs :math:`u,v`, AR1 is :math:`S_{uv}/S_{uu}`;
Pearson correlation is :math:`S_{uv}/\sqrt{S_{uu}S_{vv}}`. AR1 may exceed
:math:`[-1,1]`. Variance uses ``ddof=0``; skewness and kurtosis use population
moments, with excess kurtosis by default. Constant windows have zero variance
and undefined normalized statistics.

.. automodule:: tipbox.indicators
   :members: window_size, rolling_ar1, rolling_acf, rolling_variance, rolling_skewness, rolling_kurtosis

.. _benchmark-api:

Benchmarking
------------

``generate_dataset`` returns observations ``(N,T,D)``, labels ``(N,)``, and
per-row times, forcing, and metadata. ``lead_samples=1`` ends the context one
sample before the model reference. Null references are comparison points;
a forcing crossing need not coincide with an observed state jump. Balanced,
outcome-selected batches do not estimate natural tipping probabilities.

Each indicator series contributes one Kendall tau-b trend score. Declare score
directions before evaluation; negate decreasing indicators. Nonfinite scores
are excluded and reported in ``n_missing``. Input labels must include both
classes; AUROC is ``NaN`` if filtering removes a class. Report coverage with the score.

.. autofunction:: tipbox.datasets.generate_dataset

.. automodule:: tipbox.metrics
   :members: kendall_tau, binary_roc, ensemble_roc, plot_roc

Observations
------------

``load()`` returns time ``(T,)``, observations ``(T,D)``, and source metadata.
Original ordering, missing values, and time coordinates are retained.
``cache_dir=None`` uses ``$XDG_CACHE_HOME/tipbox`` or ``~/.cache/tipbox``;
``download=False`` requires a cached file. Local source paths bypass downloads.
All loads verify the pinned source checksum. Source terms are listed in the
:ref:`empirical-observations` catalog and record metadata.

.. autoclass:: tipbox.datasets.Microcosm
   :members: load

.. autoclass:: tipbox.datasets.VoiceOnset
   :inherited-members: load

.. autoclass:: tipbox.datasets.CellularATP
   :inherited-members: load

.. autoclass:: tipbox.datasets.Paleoclimate
   :inherited-members: load

.. autoclass:: tipbox.datasets.ElectricalBlackout
   :inherited-members: load

.. autoclass:: tipbox.datasets.EmpiricalRecord
   :members:

Discovery
---------

.. autofunction:: tipbox.registry.list_systems

.. autofunction:: tipbox.registry.get_system_info
