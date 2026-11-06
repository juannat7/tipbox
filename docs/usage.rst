Simulation
==========

.. code-block:: python

   from tipbox.models import MayHarvesting

   model = MayHarvesting(sigma=0.02)
   # num_samples: sampled observations; noise: process-noise strength; seed: random seed.
   result = model.make_trajectory(num_samples=1200, noise=1.0, seed=43)

``result.time`` contains sample times, ``result.trajectory`` the observations,
and ``result.param_values`` the forcing. Each trajectory column is named in
``model.observable_names``.

Change sampling and initialization
----------------------------------

.. code-block:: python

   result = model.make_trajectory(num_samples=1200, dt=0.01, dt_sample=1.0, initial_state=[0.8], noise=1.0, seed=43)

``num_samples`` counts observations; ``dt`` is the integration step and ``dt_sample``
the spacing between observations. Time starts at ``model.config.t_start``.

Set a forcing schedule
----------------------

.. code-block:: python

   result = model.make_trajectory(num_samples=1200, forcing=lambda t: 0.15 + 0.0001 * t, noise=1.0, seed=43)

See the :doc:`catalog <models/index>` for model equations and defaults,
:ref:`API reference <simulation-api>` for all simulation arguments, and :doc:`benchmarks`
for labeled batches.
