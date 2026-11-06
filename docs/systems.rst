System catalog
==============

All model classes are imported from ``tipbox.models``; observation classes come from
``tipbox.datasets``. Links below open runnable examples, equations, and defaults.

Box models
----------

.. list-table::
   :header-rows: 1
   :widths: 34 25 41
   :class: catalog-table

   * - Class
     - Mechanism
     - Source
   * - :doc:`MayHarvesting <models/harvesting>`
     - Fold
     - May (1977)
   * - :doc:`RosenzweigMacArthur <models/consumer_resource>`
     - Transcritical / Hopf
     - Rosenzweig and MacArthur (1963); Bury et al. (2021)
   * - :doc:`SEIRx <models/seirx>`
     - Transcritical
     - Bury et al. (2021)
   * - :doc:`AMOC <models/amoc>`
     - Fold
     - Alkhayuon et al. (2019)
   * - :doc:`SaddleNode <models/saddle_node>`
     - Rate-induced
     - Ashwin et al. (2012)
   * - :doc:`Bautin <models/bautin>`
     - Rate-induced
     - Huang et al. (2024)
   * - :doc:`CompostBomb <models/compost>`
     - Rate-induced
     - Wieczorek et al. (2011)
   * - :doc:`AMOCRate <models/amoc>`
     - Rate-induced
     - Alkhayuon et al. (2019)

Sim-to-real models
------------------

These entries provide simulations. Experimental recordings must be obtained
separately; see :doc:`sim_to_real` for data sources and calibration details.

.. list-table::
   :header-rows: 1
   :widths: 34 36 30
   :class: catalog-table

   * - Class
     - Application
     - Mechanism
   * - :doc:`Ricker <models/ricker>`
     - Daphnia population surrogate
     - Transcritical population decline
   * - :doc:`Thermoacoustic <models/thermoacoustic>`
     - Combustion pressure oscillator
     - Subcritical Hopf

Empirical observations
----------------------

``Microcosm`` loads the original M1 and M2 chemostat observations. The other
cases come from the Grziwotz et al. (2023) archive used by KEWS.
See :doc:`datasets` for sampling, provenance, source terms, and cache controls.

.. list-table::
   :header-rows: 1
   :widths: 26 45 29
   :class: catalog-table

   * - Record
     - Class / call
     - Observed quantity
   * - Microcosm · M1 and M2
     - :meth:`Microcosm().load() <tipbox.datasets.Microcosm.load>`
     - Light attenuation
   * - Voice onset
     - :meth:`VoiceOnset().load() <tipbox.datasets.VoiceOnset.load>`
     - Subglottal pressure
   * - Cellular ATP
     - :meth:`CellularATP().load() <tipbox.datasets.CellularATP.load>`
     - Relative fluorescence ratio
   * - Paleoclimate
     - :meth:`Paleoclimate().load() <tipbox.datasets.Paleoclimate.load>`
     - CaCO3
   * - Electrical blackout
     - :meth:`ElectricalBlackout().load() <tipbox.datasets.ElectricalBlackout.load>`
     - Frequency

Discover from Python
--------------------

.. code-block:: python

   from tipbox import list_systems, get_system_info

   print(list_systems())                # Eleven configurations, ten model classes.
   model = get_system_info("b_harvesting").create_model()
   print(model.get_parameters())
