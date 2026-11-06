API reference
=============

Shared interface
----------------

.. autoclass:: tipbox.types.BoxModel
   :members: config, get_parameters, make_trajectory

.. autoclass:: tipbox.types.TrajectoryResult
   :members:

Generation and discovery
------------------------

.. autofunction:: tipbox.generate.generate

.. autofunction:: tipbox.registry.list_systems

.. autofunction:: tipbox.registry.get_system_info

Named models
------------

Import from ``tipbox.models``: ``MayHarvesting``, ``RosenzweigMacArthur``,
``SEIRx``, ``AMOC``, ``SaddleNode``, ``Bautin``, ``CompostBomb``, and ``AMOCRate``.
The corresponding ``*Config`` classes and shared ``AMOCPhysics`` dataclass are
also exported. Each model inherits the shared interface above. Full parameter
tables are in :doc:`models/index`.
