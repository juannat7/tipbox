System catalog
==============

Bifurcation-induced
-------------------

Import classes from ``tipbox.models``. Each page gives a runnable example,
trajectory, equation, and parameter defaults.

.. list-table::
   :header-rows: 1
   :widths: 42 28 30
   :class: catalog-table

   * - Class
     - Transition
     - Source
   * - :doc:`MayHarvesting <harvesting>`
     - Fold
     - May (1977)
   * - :doc:`RosenzweigMacArthur <consumer_resource>`
     - Transcritical / Hopf
     - Bury et al. (2021)
   * - :doc:`SEIRx <seirx>`
     - Transcritical
     - Bury et al. (2021)
   * - :ref:`AMOCFold <amoc-fold>`
     - Fold
     - Alkhayuon et al. (2019)
   * - :doc:`Ricker <ricker>`
     - Transcritical
     - Ricker (1954)
   * - :doc:`Thermoacoustic <thermoacoustic>`
     - Subcritical Hopf
     - Bonciolini et al. (2018)

Rate-induced
------------

.. list-table::
   :header-rows: 1
   :widths: 42 58
   :class: catalog-table

   * - Class
     - Source
   * - :doc:`SaddleNode <saddle_node>`
     - Ashwin et al. (2012)
   * - :doc:`Bautin <bautin>`
     - Huang et al. (2024)
   * - :doc:`CompostBomb <compost>`
     - Wieczorek et al. (2011)
   * - :ref:`AMOCRate <amoc-rate>`
     - Alkhayuon et al. (2019)

.. _empirical-observations:

Empirical
---------

Import classes from ``tipbox.datasets`` and call ``load()``.

.. list-table::
   :header-rows: 1
   :widths: 35 35 30
   :class: catalog-table

   * - Class
     - Observation
     - Source
   * - :class:`~tipbox.datasets.Microcosm`
     - Light attenuation, M1 and M2
     - `Veraart et al. (2012) <https://doi.org/10.17026/dans-ztg-93aw>`_
   * - :class:`~tipbox.datasets.VoiceOnset`
     - Subglottal pressure
     - `Grziwotz et al. (2023) <https://doi.org/10.5281/zenodo.7379358>`_
   * - :class:`~tipbox.datasets.CellularATP`
     - Fluorescence ratio
     - Grziwotz et al. (2023)
   * - :class:`~tipbox.datasets.Paleoclimate`
     - CaCO3
     - Grziwotz et al. (2023)
   * - :class:`~tipbox.datasets.ElectricalBlackout`
     - Frequency
     - Grziwotz et al. (2023)

Microcosm is CC BY-NC 4.0. The Grziwotz archive requires attribution, nonprofit
use, and contact with the listed data administrators before publication or
presentation; complete terms are in ``record.metadata["terms"]``.

Missing values and source clocks are preserved. CellularATP has inconsistent
source time units, reported as ``"unresolved"``. Microcosm has interrupted
recordings: compute indicators within individual recovery intervals; the
:doc:`figure guide <../figures>` lists their row slices.

Simulation
----------

:doc:`../usage` covers sampling, initial states, and forcing.
:doc:`../benchmarks` shows labeled batches and AUROC.

.. toctree::
   :hidden:
   :maxdepth: 1

   harvesting
   consumer_resource
   seirx
   saddle_node
   bautin
   compost
   amoc
   ricker
   thermoacoustic
