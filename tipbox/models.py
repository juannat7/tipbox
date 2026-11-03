"""Named models with inspectable configurations and a shared trajectory API.

Construct a model with defaults, a typed config, or keyword parameter overrides::

    model = MayHarvesting(r=1.0, k=1.0, s=0.1)
    result = model.make_trajectory(1000, noise=0.01, seed=42)

The legacy ``*Model`` classes remain available from their original modules.
"""

from ._amoc import AMOCPhysics
from .bifurcation.amoc import AMOCConfig, AMOCModel as AMOC
from .bifurcation.may_harvesting import (
    MayHarvestingConfig,
    MayHarvestingModel as MayHarvesting,
)
from .bifurcation.rosenzweig_macarthur import (
    RosenzweigMacArthurConfig,
    RosenzweigMacArthurModel as RosenzweigMacArthur,
)
from .bifurcation.seirx import SEIRxConfig, SEIRxModel as SEIRx
from .rate.amoc import AMOCRateConfig, AMOCRateModel as AMOCRate
from .rate.bautin import BautinConfig, BautinModel as Bautin
from .rate.compost_bomb import CompostBombConfig, CompostBombModel as CompostBomb
from .rate.saddle_node import SaddleNodeConfig, SaddleNodeModel as SaddleNode

__all__ = [
    "AMOC",
    "AMOCConfig",
    "AMOCPhysics",
    "AMOCRate",
    "AMOCRateConfig",
    "MayHarvesting",
    "MayHarvestingConfig",
    "RosenzweigMacArthur",
    "RosenzweigMacArthurConfig",
    "SEIRx",
    "SEIRxConfig",
    "Bautin",
    "BautinConfig",
    "CompostBomb",
    "CompostBombConfig",
    "SaddleNode",
    "SaddleNodeConfig",
]
