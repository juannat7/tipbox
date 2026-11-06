Quickstart
==========

Install with Python 3.10 or newer:

.. code-block:: bash

   pip install tipbox

Simulate
--------

.. code-block:: python

   from tipbox.models import MayHarvesting

   model = MayHarvesting()
   # num_samples: sampled observations; noise: process-noise strength; seed: random seed.
   result = model.make_trajectory(num_samples=1200, noise=1.0, seed=43)

Choose other models from the :doc:`catalog <models/index>`.

Compute an indicator
--------------------

Use the observations before the fold:

.. code-block:: python

   import matplotlib.pyplot as plt
   from tipbox.indicators import rolling_ar1

   before_fold = result.param_values < model.critical_parameter
   signal = result.trajectory[before_fold, 0]
   ar1 = rolling_ar1(signal, window=200, detrend="linear")
   plt.plot(result.time[before_fold], ar1)
   plt.show()

The window contains 200 observations. The first 199 results are ``NaN``.
See :doc:`indicators` for the other statistics and :doc:`benchmarks` for AUROC.

Load observations
-----------------

.. code-block:: python

   from tipbox.datasets import Microcosm

   record = Microcosm().load()
   print(record.time.shape)  # (7914,)
   print(record.observations.shape)  # (7914, 2): M1, M2

Downloads are cached. The :ref:`empirical-observations` catalog lists the
available records and their sources.
