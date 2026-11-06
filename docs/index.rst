TipBox
======

Named dynamical systems for studying critical transitions. Configure a physical
model, generate reproducible trajectories, and compare warning indicators using
explicit observations and event definitions.

.. image:: _static/figures/overview.png
   :alt: Eleven named box-model configurations, drawn from actual TipBox simulations.

.. code-block:: python

   from tipbox.models import MayHarvesting

   model = MayHarvesting(r=1.0, k=1.0, s=0.1)
   result = model.make_trajectory(1000, noise=0.01, seed=42)

The interface follows the compact named-model approach of
`dysts <https://github.com/GilpinLab/dysts>`_. TipBox retains the stochastic
integration kernels and forcing conventions of its tipping-point models.

.. toctree::
   :maxdepth: 2

   usage
   models/index
   indicators
   benchmarks
   datasets
   sim_to_real
   figures
   api
   compatibility

Research using TipBox
---------------------

* Nathaniel et al., `Koopman early warning signals for bifurcation and rate-induced
  tipping <https://arxiv.org/abs/2608.14716>`_ (2026).
* Sevinchan et al., `In-context learning to predict critical transitions in
  dynamical systems <https://arxiv.org/abs/2605.12308>`_ (2026).

These studies used earlier versions of the code. Historical generation settings
remain accessible through Git history; see :doc:`compatibility`.
