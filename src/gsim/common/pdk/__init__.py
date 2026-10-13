"""Canonical PDK resolution for simulation backends."""

from gsim.common.pdk.models import (
    ComponentResolutionError,
    LayerResolutionError,
    PdkResolutionError,
    ResolvedLayer,
    ResolvedPassivePcell,
    ResolvedPort,
    UnsupportedPortError,
)
from gsim.common.pdk.resolve import resolve_passive_pcell
from gsim.common.pdk.stack import (
    ResolvedDielectric,
    ResolvedProcessLayer,
    ResolvedStack,
    classify_layer_role,
    resolve_layer_stack,
    resolve_material_properties,
)

__all__ = [
    "ComponentResolutionError",
    "LayerResolutionError",
    "PdkResolutionError",
    "ResolvedDielectric",
    "ResolvedLayer",
    "ResolvedPassivePcell",
    "ResolvedPort",
    "ResolvedProcessLayer",
    "ResolvedStack",
    "UnsupportedPortError",
    "classify_layer_role",
    "resolve_layer_stack",
    "resolve_material_properties",
    "resolve_passive_pcell",
]
