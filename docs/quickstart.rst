Quickstart
==========

Install with Python 3.10 or newer:

.. code-block:: bash

   git clone https://github.com/juannat7/tipbox.git
   cd tipbox
   pip install -e .

Simulate and measure
--------------------

May's harvesting model loses its high-population equilibrium at a fold.
Compute warning signals from observations before that reference:

.. code-block:: python

   import matplotlib.pyplot as plt
   from tipbox.models import MayHarvesting
   from tipbox.indicators import rolling_ar1, rolling_variance

   model = MayHarvesting()
   result = model.make_trajectory(1000, noise=1, seed=42)
   reference = model.locate_transition(
       result.trajectory, False,
       param_values=result.param_values, time=result.time,
   )
   if reference is None:
       raise ValueError("This forcing does not reach the fold.")
   time = result.time[:reference]
   signal = result.trajectory[:reference, 0]
   ar1 = rolling_ar1(signal, window=200, detrend="linear")
   variance = rolling_variance(signal, window=200, detrend="linear")
   plt.plot(time, ar1)
   plt.xlabel("Time (model units)")
   plt.ylabel("AR1 slope")
   plt.show()

``n`` counts returned samples, including the initial state. ``noise`` scales the
model's configured diffusion; ``noise=1`` uses May's configured amplitude 0.01.
The first 199 indicator values are ``NaN``. Each detrending fit uses its own
trailing window. See :doc:`usage` for grid, initial-state and forcing controls.

Evaluate independent trajectories
---------------------------------

.. code-block:: python

   from tipbox.models import MayHarvesting
   from tipbox.datasets import generate_dataset
   from tipbox.indicators import rolling_ar1, rolling_variance
   from tipbox.metrics import ensemble_roc, plot_roc

   data = generate_dataset(
       MayHarvesting(), num_timesteps=300, n_tipping=5, n_null=5,
       n=1000, noise=1, lead_steps=1, seed=42,
   )
   signals = data.observations[:, :, 0]  # (10, 300)
   indicators = {
       "AR1": [rolling_ar1(x, window=100, detrend="linear") for x in signals],
       "Variance": [rolling_variance(x, window=100, detrend="linear") for x in signals],
   }
   report = ensemble_roc(data.labels, indicators)
   print(report.curves["Variance"].auroc)
   ax = plot_roc(report)

Each trajectory supplies one Kendall trend score. Both directions are declared
increasing. Null rows come first; accepted seeds and reference times are in
``data.metadata``. ``lead_steps=1`` ends one observation before the reference.
These outcome-conditioned batches do not estimate tipping probabilities.
See :doc:`benchmarks` for labels, coverage, and sampling conventions.

Use measured data
-----------------

.. code-block:: python

   import numpy as np
   from tipbox.datasets import load_microcosm_original
   from tipbox.indicators import rolling_variance

   record = load_microcosm_original()
   recovery = record.observations[161:411, 0]  # M1, first recovery
   variance = rolling_variance(recovery, window=200, step=5, detrend="linear")
   print(record.observations.shape)             # (7914, 2): M1, M2
   print(int(np.isfinite(variance).sum()))      # 11

Downloads are cached. Analyze recoveries separately; joining them creates
artificial lagged pairs. This source is CC BY-NC 4.0: cite Veraart et al. (2012)
and `the DANS deposit <https://doi.org/10.17026/dans-ztg-93aw>`_.
:doc:`datasets` lists sources, intervals, missingness, and offline controls.
