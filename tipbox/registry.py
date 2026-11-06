"""Named box models: ``b_*`` for bifurcation tipping and ``r_*`` for rate tipping."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from .bifurcation.amoc import AMOCFoldConfig, AMOCFoldModel
from .bifurcation.may_harvesting import MayHarvestingConfig, MayHarvestingModel
from .bifurcation.ricker import RickerConfig, RickerModel
from .bifurcation.rosenzweig_macarthur import (
    RosenzweigMacArthurConfig,
    RosenzweigMacArthurModel,
)
from .bifurcation.seirx import SEIRxConfig, SEIRxModel
from .bifurcation.thermoacoustic import ThermoacousticConfig, ThermoacousticModel
from .rate.amoc import AMOCRateConfig, AMOCRateModel
from .rate.bautin import BautinConfig, BautinModel
from .rate.compost_bomb import CompostBombConfig, CompostBombModel
from .rate.saddle_node import SaddleNodeConfig, SaddleNodeModel
from .types import BoxModel


class SystemType(Enum):
    """Types of tipping mechanisms."""

    BIFURCATION = "bifurcation"
    RATE = "rate"


@dataclass(frozen=True)
class SystemSpec:
    """Unified specification for a tipping point system."""

    name: str
    system_type: SystemType
    description: str
    config_cls: Type[Any]
    model_cls: Type[BoxModel]
    bifurcation_system: Optional[str] = None
    rate_system: Optional[str] = None
    config_kwargs: Dict[str, Any] = field(default_factory=dict)

    def default_config(self) -> Any:
        """Create default configuration for this system."""
        return self.config_cls(**self.config_kwargs)

    def create_model(self, config: Optional[Any] = None) -> BoxModel:
        """Create model instance for this system."""
        model = self.model_cls(self.default_config() if config is None else config)
        for name, expected in self.config_kwargs.items():
            if getattr(model.config, name) != expected:
                raise ValueError(f"{self.name} requires {name}={expected!r}")
        return model


# System registry with full specifications
SYSTEM_REGISTRY: Dict[str, SystemSpec] = {
    "b_ricker": SystemSpec(
        name="b_ricker",
        system_type=SystemType.BIFURCATION,
        bifurcation_system="ricker",
        config_cls=RickerConfig,
        model_cls=RickerModel,
        description="Stochastic Ricker map with a declining log-growth rate",
    ),
    "b_thermoacoustic": SystemSpec(
        name="b_thermoacoustic",
        system_type=SystemType.BIFURCATION,
        bifurcation_system="thermoacoustic",
        config_cls=ThermoacousticConfig,
        model_cls=ThermoacousticModel,
        description="Thermoacoustic pressure oscillator with a subcritical Hopf bifurcation",
    ),
    # Rate-induced systems
    "r_bautin": SystemSpec(
        name="r_bautin",
        system_type=SystemType.RATE,
        rate_system="bautin",
        config_cls=BautinConfig,
        model_cls=BautinModel,
        description="Bautin rate-induced tipping",
    ),
    "r_compost_bomb": SystemSpec(
        name="r_compost_bomb",
        system_type=SystemType.RATE,
        rate_system="compost_bomb",
        config_cls=CompostBombConfig,
        model_cls=CompostBombModel,
        description="Compost bomb rate-induced tipping",
    ),
    "r_saddle_node": SystemSpec(
        name="r_saddle_node",
        system_type=SystemType.RATE,
        rate_system="saddle_node",
        config_cls=SaddleNodeConfig,
        model_cls=SaddleNodeModel,
        description="Saddle-node rate-induced tipping",
    ),
    "r_amoc": SystemSpec(
        name="r_amoc",
        system_type=SystemType.RATE,
        rate_system="amoc",
        config_cls=AMOCRateConfig,
        model_cls=AMOCRateModel,
        description="AMOC 2-box model (rate-induced)",
    ),
    # Semi-real bifurcation systems (for validation)
    "b_harvesting": SystemSpec(
        name="b_harvesting",
        system_type=SystemType.BIFURCATION,
        bifurcation_system="may_harvesting",
        config_cls=MayHarvestingConfig,
        model_cls=MayHarvestingModel,
        description="May's harvesting model (semi-real bifurcation)",
    ),
    # Rosenzweig-MacArthur consumer-resource model (Bury et al. 2021)
    # Supports both transcritical and Hopf bifurcations
    "b_rosenzweig_macarthur_tc": SystemSpec(
        name="b_rosenzweig_macarthur_tc",
        system_type=SystemType.BIFURCATION,
        bifurcation_system="rosenzweig_macarthur_transcritical",
        config_cls=RosenzweigMacArthurConfig,
        model_cls=RosenzweigMacArthurModel,
        description="Rosenzweig-MacArthur consumer-resource model (transcritical bifurcation)",
        config_kwargs={"bifurcation_type": "transcritical"},
    ),
    "b_rosenzweig_macarthur_hopf": SystemSpec(
        name="b_rosenzweig_macarthur_hopf",
        system_type=SystemType.BIFURCATION,
        bifurcation_system="rosenzweig_macarthur_hopf",
        config_cls=RosenzweigMacArthurConfig,
        model_cls=RosenzweigMacArthurModel,
        description="Rosenzweig-MacArthur consumer-resource model (Hopf bifurcation)",
        config_kwargs={"bifurcation_type": "hopf"},
    ),
    # SEIRx behaviour–disease model (Bury et al. 2021) — transcritical bifurcation
    "b_seirx_tc": SystemSpec(
        name="b_seirx_tc",
        system_type=SystemType.BIFURCATION,
        bifurcation_system="seirx_transcritical",
        config_cls=SEIRxConfig,
        model_cls=SEIRxModel,
        description="SEIRx disease dynamics with vaccinator behaviour (transcritical bifurcation)",
    ),
    # AMOC 2-box model (Ritchie et al. 2023; Alkhayuon et al. 2019) — bifurcation-induced
    "b_amoc": SystemSpec(
        name="b_amoc",
        system_type=SystemType.BIFURCATION,
        bifurcation_system="amoc",
        config_cls=AMOCFoldConfig,
        model_cls=AMOCFoldModel,
        description="AMOC 2-box model (bifurcation-induced)",
    ),
}


def get_system_info(system_name: str) -> SystemSpec:
    """Get a registered system's configuration and model class."""
    if system_name not in SYSTEM_REGISTRY:
        raise ValueError(
            f"Unknown system name: {system_name}. Available: {list_systems()}"
        )
    return SYSTEM_REGISTRY[system_name]


def list_systems(
    system_type: Optional[SystemType] = None,
) -> List[str]:
    """List available systems, optionally filtered by type."""
    return sorted(
        name
        for name, spec in SYSTEM_REGISTRY.items()
        if system_type is None or spec.system_type == system_type
    )


def get_bifurcation_systems() -> List[str]:
    """Get list of all bifurcation systems."""
    return list_systems(system_type=SystemType.BIFURCATION)


def get_rate_systems() -> List[str]:
    """Get list of all rate-induced systems."""
    return list_systems(system_type=SystemType.RATE)


def get_rate_system_spec(rate_system_name: str) -> SystemSpec:
    """Get rate system specification by internal name (e.g., 'bautin' -> 'r_bautin')."""
    for name, spec in SYSTEM_REGISTRY.items():
        if spec.rate_system == rate_system_name:
            return spec
    raise ValueError(
        f"Unknown rate system: {rate_system_name}. "
        f"Available rate systems: {[s.rate_system for s in SYSTEM_REGISTRY.values() if s.rate_system]}"
    )


def get_bifurcation_real_system_spec(real_system_name: str) -> SystemSpec:
    """Get bifurcation real system specification by internal name (e.g., 'may_harvesting' -> 'b_harvesting')."""
    for name, spec in SYSTEM_REGISTRY.items():
        if spec.bifurcation_system == real_system_name:
            return spec
    raise ValueError(
        f"Unknown bifurcation real system: {real_system_name}. "
        f"Available systems: {[s.bifurcation_system for s in SYSTEM_REGISTRY.values() if s.bifurcation_system]}"
    )
