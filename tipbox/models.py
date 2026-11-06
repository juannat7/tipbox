"""Named models with inspectable configurations and a shared trajectory API.

Construct a model with defaults, a typed config, or keyword parameter overrides::

    model = MayHarvesting(r=1.0, k=1.0, s=0.1)
    result = model.make_trajectory(1000, noise=0.01, seed=42)
"""

from ._amoc import AMOCPhysics
from .bifurcation.amoc import AMOCFoldConfig
from .bifurcation.amoc import AMOCFoldModel as AMOCFold
from .bifurcation.may_harvesting import (
    MayHarvestingConfig,
)
from .bifurcation.may_harvesting import (
    MayHarvestingModel as MayHarvesting,
)
from .bifurcation.ricker import RickerConfig
from .bifurcation.ricker import RickerModel as Ricker
from .bifurcation.rosenzweig_macarthur import (
    RosenzweigMacArthurConfig,
)
from .bifurcation.rosenzweig_macarthur import (
    RosenzweigMacArthurModel as RosenzweigMacArthur,
)
from .bifurcation.seirx import SEIRxConfig
from .bifurcation.seirx import SEIRxModel as SEIRx
from .bifurcation.thermoacoustic import (
    ThermoacousticConfig,
)
from .bifurcation.thermoacoustic import (
    ThermoacousticModel as Thermoacoustic,
)
from .rate.amoc import AMOCRateConfig
from .rate.amoc import AMOCRateModel as AMOCRate
from .rate.bautin import BautinConfig
from .rate.bautin import BautinModel as Bautin
from .rate.compost_bomb import CompostBombConfig
from .rate.compost_bomb import CompostBombModel as CompostBomb
from .rate.saddle_node import SaddleNodeConfig
from .rate.saddle_node import SaddleNodeModel as SaddleNode

__all__ = [
    "AMOCFold",
    "AMOCFoldConfig",
    "AMOCPhysics",
    "AMOCRate",
    "AMOCRateConfig",
    "Bautin",
    "BautinConfig",
    "CompostBomb",
    "CompostBombConfig",
    "MayHarvesting",
    "MayHarvestingConfig",
    "Ricker",
    "RickerConfig",
    "RosenzweigMacArthur",
    "RosenzweigMacArthurConfig",
    "SEIRx",
    "SEIRxConfig",
    "SaddleNode",
    "SaddleNodeConfig",
    "Thermoacoustic",
    "ThermoacousticConfig",
]
