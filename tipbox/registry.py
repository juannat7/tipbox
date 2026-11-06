"""Named box models: ``b_*`` for bifurcation tipping and ``r_*`` for rate tipping."""

from typing import Dict, List, Optional, Type, Any
from enum import Enum

from .types import BoxModel

# Import config and model classes
from .rate.bautin import BautinConfig, BautinModel
from .rate.compost_bomb import CompostBombConfig, CompostBombModel
from .rate.saddle_node import SaddleNodeConfig, SaddleNodeModel
from .bifurcation.may_harvesting import MayHarvestingConfig, MayHarvestingModel
from .bifurcation.rosenzweig_macarthur import (
    RosenzweigMacArthurConfig,
    RosenzweigMacArthurModel,
)
from .bifurcation.seirx import SEIRxConfig, SEIRxModel
from .bifurcation.amoc import AMOCConfig, AMOCModel
from .rate.amoc import AMOCRateConfig, AMOCRateModel


class SystemType(Enum):
    """Types of tipping mechanisms."""

    BIFURCATION = "bifurcation"
    RATE = "rate"


class SystemSpec:
    """Unified specification for a tipping point system."""

    def __init__(
        self,
        name: str,
        system_type: SystemType,
        description: str,
        config_cls: Optional[Type[Any]] = None,
        model_cls: Optional[Type[BoxModel]] = None,
        bifurcation_system: Optional[str] = None,
        rate_system: Optional[str] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.system_type = system_type
        self.description = description
        self.config_cls = config_cls
        self.model_cls = model_cls
        self.bifurcation_system = bifurcation_system
        self.rate_system = rate_system
        self.config_kwargs = config_kwargs or {}

    def default_config(self) -> Any:
        """Create default configuration for this system."""
        if self.config_cls is None:
            raise ValueError(f"System '{self.name}' does not have a config class")
        return self.config_cls(**self.config_kwargs)

    def create_model(self, config: Optional[Any] = None) -> BoxModel:
        """Create model instance for this system."""
        if self.model_cls is None:
            raise ValueError(f"System '{self.name}' does not have a model class")
        if config is None:
            config = self.default_config()
        if not isinstance(config, self.config_cls):
            raise TypeError(f"{self.name} requires {self.config_cls.__name__}")
        for name, expected in self.config_kwargs.items():
            if getattr(config, name) != expected:
                raise ValueError(f"{self.name} requires {name}={expected!r}")
        return self.model_cls(config)


# System registry with full specifications
SYSTEM_REGISTRY: Dict[str, SystemSpec] = {
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
        config_cls=AMOCConfig,
        model_cls=AMOCModel,
        description="AMOC 2-box model (bifurcation-induced)",
    ),
}


def normalize_system_name(name: str) -> str:
    """Resolve a registered name or the legacy ``rate`` alias."""
    if name in SYSTEM_REGISTRY:
        return name

    legacy_map = {"rate": "r_bautin"}

    if name in legacy_map:
        return legacy_map[name]

    raise ValueError(
        f"Unknown system name: {name}. "
        f"Available systems: {list(SYSTEM_REGISTRY.keys())}. "
        f"Legacy names supported: {list(legacy_map.keys())}"
    )


def get_system_info(system_name: str) -> SystemSpec:
    """Get system specification."""
    normalized = normalize_system_name(system_name)
    return SYSTEM_REGISTRY[normalized]


def get_system_dict(system_name: str) -> Dict:
    """Get system information as dictionary (for backward compatibility)."""
    spec = get_system_info(system_name)
    result = {
        "type": spec.system_type,
        "description": spec.description,
    }
    if spec.bifurcation_system is not None:
        result["bifurcation_system"] = spec.bifurcation_system
    if spec.rate_system is not None:
        result["rate_system"] = spec.rate_system
    return result


def list_systems(
    system_type: Optional[SystemType] = None,
) -> List[str]:
    """List available systems, optionally filtered by type."""
    systems = []
    for name, spec in SYSTEM_REGISTRY.items():
        if system_type is not None and spec.system_type != system_type:
            continue
        systems.append(name)
    return sorted(systems)


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
