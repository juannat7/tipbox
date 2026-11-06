TipBox
======

TipBox collects stochastic tipping models and empirical observations. It covers
bifurcation- and rate-induced tipping, early-warning indicators, and
trajectory-level ROC/AUROC benchmarking.

.. code-block:: python

   from tipbox.models import MayHarvesting

   # num_samples: sampled observations; noise: process-noise strength; seed: random seed.
   result = MayHarvesting().make_trajectory(num_samples=1000, noise=1, seed=43)
   population = result.trajectory[:, 0]

:doc:`quickstart` covers installation through evaluation. Each
:doc:`model page <models/index>` gives its equations, parameters, units, and
reference event.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting started

   quickstart
   usage

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reference

   models/index
   indicators
   benchmarks
   api

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Reproduction

   figures
   citations

Research using TipBox:
`KEWS <https://arxiv.org/abs/2608.14716>`_ and
`TipPFN <https://arxiv.org/abs/2605.12308>`_.
See :doc:`citations` for BibTeX.
A `plain-text documentation index <llms.txt>`_ is available for coding assistants.
