TipBox
======

TipBox collects stochastic tipping models, empirical observations, and simulation
components for sim-to-real studies. It covers bifurcation- and rate-induced
tipping, classical warning indicators, and trajectory-level ROC/AUROC.

.. code-block:: python

   from tipbox.models import MayHarvesting

   result = MayHarvesting().make_trajectory(1000, noise=1, seed=42)
   population = result.trajectory[:, 0]

:doc:`quickstart` covers installation through evaluation. Each
:doc:`model page <models/index>` gives its equations, parameters, units, and
reference event. Defaults are generated from the live configurations.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   quickstart
   usage

.. toctree::
   :maxdepth: 1
   :caption: Reference

   models/index
   indicators
   benchmarks
   datasets
   api

.. toctree::
   :maxdepth: 1
   :caption: Reproduction

   figures
   sim_to_real
   compatibility

Research using earlier versions:
`KEWS <https://arxiv.org/abs/2608.14716>`_ and
`TipPFN <https://arxiv.org/abs/2605.12308>`_.
A `plain-text documentation index <llms.txt>`_ is available for coding assistants.
