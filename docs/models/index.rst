Model collection
================

Choose a model by its physical observable and reference event. The eleven
registered configurations cover ten classes; Rosenzweig–MacArthur supports
transcritical and Hopf scenarios. Import classes from ``tipbox.models``.

.. list-table:: Models and observations
   :header-rows: 1
   :widths: 26 20 24 30
   :class: contract-table

   * - Model
     - Mechanism
     - Observed columns
     - Reference
   * - :doc:`MayHarvesting <harvesting>`
     - Bifurcation
     - Biomass
     - Upper fold crossing
   * - :doc:`RosenzweigMacArthur <consumer_resource>`
     - Transcritical / Hopf
     - Resource, consumer
     - Selected analytic threshold
   * - :doc:`SEIRx <seirx>`
     - Bifurcation
     - Infectious count, vaccinator fraction
     - Behavioural threshold
   * - :doc:`AMOC <amoc>`
     - Bifurcation
     - Transport, northern and tropical salinity anomalies
     - Calibrated hosing fold
   * - :doc:`SaddleNode <saddle_node>`
     - Rate-induced
     - State
     - Configured state threshold
   * - :doc:`Bautin <bautin>`
     - Rate-induced
     - Cartesian state X, Y
     - Radius threshold about the origin
   * - :doc:`CompostBomb <compost>`
     - Rate-induced
     - Soil carbon, soil temperature
     - Soil-temperature threshold
   * - :doc:`AMOCRate <amoc>`
     - Rate-induced
     - Transport, northern and tropical salinity anomalies
     - Time zero after a finite-collapse outcome check
   * - :doc:`Ricker <ricker>`
     - Discrete population map
     - Population
     - Nonpositive log growth
   * - :doc:`Thermoacoustic <thermoacoustic>`
     - Subcritical Hopf
     - Pressure, scaled velocity, phase-plane amplitude
     - Linear growth reaches zero

Each page states its equation, time unit, noise convention, initial-state rule,
and numerical constraints. Parameter tables come from the live typed
configurations. “Automatic” marks an override calculated from other coefficients;
parameters for another selected mode are identified in their descriptions.

Equal numbers of samples do not imply equal physical durations or noise
strengths across models. Inspect ``model.state_names`` and
``model.observable_names`` before selecting a column for an indicator.

.. toctree::
   :maxdepth: 1
   :hidden:

   harvesting
   consumer_resource
   seirx
   saddle_node
   bautin
   compost
   amoc
   ricker
   thermoacoustic
