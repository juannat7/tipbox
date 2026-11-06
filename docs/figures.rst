Reproduce the figures
=====================

From the repository root:

.. code-block:: bash

   python examples/figures.py

Exports PNG, SVG, PDF, and ``manifest.json`` to ``docs/_static/figures``.
The manifest records parameters, accepted seeds, sampling, display limits,
source checksums, and software versions.

.. list-table:: Command options
   :header-rows: 1

   * - Option
     - Default
   * - ``--seed``
     - ``42``
   * - ``--n-per-class``
     - ``50`` for the ROC example
   * - ``--formats``
     - ``png svg pdf``
   * - ``--output-dir``
     - ``docs/_static/figures``
   * - ``--cache-dir``
     - User cache
   * - ``--source-path``
     - Download the original microcosm ODS if absent

Model previews
--------------

Each page compares an accepted tipping trajectory with an accepted control.
Labels follow the model's event definition. Vertical dashed lines mark the
positive reference; horizontal lines mark critical forcing levels where defined.
Both panels use the same displayed time interval and vertical scale.

The transcritical Rosenzweig–MacArthur example uses attack-rate endpoint 10 to
show consumer invasion. Its constructor default remains 6. Escaping normal
forms are cropped; exact limits and selection seeds are in the manifest.

Indicators and AUROC
--------------------

The README's May and saddle-node examples use the same tipping/control pairs
for trajectories and indicators. Both indicators use 200-sample, linearly
detrended windows ending before the tipping reference. Controls use the same
analysis interval. May uses ``n=1200, noise=1``; saddle-node uses
``epsilon=1.4, n=2401, noise=0.01``.

The ROC example uses 50 tipping and 50 control May trajectories, ``n=1000``,
context 300, lead one, window 100, linear detrending, and increasing Kendall
trend scores. The :doc:`benchmark guide <benchmarks>` reproduces it.

Microcosm
---------

M1 recoveries P1–P6 use these zero-based, stop-exclusive row slices:
``[161:411]``, ``[1283:1533]``, ``[3001:3251]``, ``[4076:4326]``,
``[5273:5523]``, and ``[6684:6934]``. Each has 250 finite readings.
Indicators use window 200 and step 5 within each recovery, with linear detrending.

Data: Veraart et al. (2012), `DANS <https://doi.org/10.17026/dans-ztg-93aw>`_,
CC BY-NC 4.0. Recovery annotations and the 27-day reference follow
`Grziwotz et al. (2023) <https://doi.org/10.5281/zenodo.7379358>`_.
See :download:`figure terms <_static/figures/LICENSE.md>` and the
:download:`manifest <_static/figures/manifest.json>`.
