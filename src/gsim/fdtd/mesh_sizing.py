"""Geometry-aware Gmsh size fields for FDTD transfer meshes."""

from __future__ import annotations

from dataclasses import dataclass

import gmsh

from gsim.common.pdk import ResolvedPassivePcell
from gsim.fdtd.mesh_geometry import UM_TO_NM, iter_polygons

_BULK_TO_FEATURE_RATIO = 8.0
_FEATURE_HALF_WIDTH_RATIO = 2.0
_TRANSITION_WIDTH_RATIO = 10.0

FeatureRegion = tuple[float, float, float, float]


@dataclass(frozen=True)
class GeometryAwareSizing:
    """Resolved feature, bulk, and transition sizes in nanometers."""

    feature_size_nm: float
    bulk_size_nm: float
    feature_half_width_nm: float
    transition_width_nm: float


def geometry_aware_sizing(
    mesh_size_nm: float,
    nanometers_per_cell: float,
) -> GeometryAwareSizing:
    """Resolve the production sizing policy from mesh and Yee-grid targets."""
    feature_size_nm = min(mesh_size_nm, nanometers_per_cell)
    return GeometryAwareSizing(
        feature_size_nm=feature_size_nm,
        bulk_size_nm=max(mesh_size_nm, _BULK_TO_FEATURE_RATIO * feature_size_nm),
        feature_half_width_nm=_FEATURE_HALF_WIDTH_RATIO * feature_size_nm,
        transition_width_nm=_TRANSITION_WIDTH_RATIO * feature_size_nm,
    )


def geometry_feature_regions_nm(
    resolved: ResolvedPassivePcell,
) -> list[FeatureRegion]:
    """Return unique vertical edge regions from all material polygon rings."""
    regions: set[FeatureRegion] = set()
    for layer in resolved.layers.values():
        z_lower_nm = round(layer.z_bounds[0] * UM_TO_NM, 6)
        z_upper_nm = round(layer.z_bounds[1] * UM_TO_NM, 6)
        for polygon in iter_polygons(layer.geometry, layer_key=layer.key):
            for ring in (polygon.exterior, *polygon.interiors):
                for x_um, y_um, *_ in list(ring.coords)[:-1]:
                    regions.add(
                        (
                            round(float(x_um) * UM_TO_NM, 6),
                            round(float(y_um) * UM_TO_NM, 6),
                            z_lower_nm,
                            z_upper_nm,
                        )
                    )
    return sorted(regions)


def install_geometry_aware_mesh_field(
    resolved: ResolvedPassivePcell,
    sizing: GeometryAwareSizing,
) -> None:
    """Install fine boxes at material edges and coarse sizing elsewhere."""
    gmsh.option.setNumber("Mesh.MeshSizeMin", sizing.feature_size_nm)
    gmsh.option.setNumber("Mesh.MeshSizeMax", sizing.bulk_size_nm)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    coarse_field = gmsh.model.mesh.field.add("MathEval")
    gmsh.model.mesh.field.setString(coarse_field, "F", str(sizing.bulk_size_nm))
    fields = [coarse_field]
    for x_nm, y_nm, z_lower_nm, z_upper_nm in geometry_feature_regions_nm(resolved):
        feature_field = gmsh.model.mesh.field.add("Box")
        gmsh.model.mesh.field.setNumber(feature_field, "VIn", sizing.feature_size_nm)
        gmsh.model.mesh.field.setNumber(feature_field, "VOut", sizing.bulk_size_nm)
        gmsh.model.mesh.field.setNumber(
            feature_field, "XMin", x_nm - sizing.feature_half_width_nm
        )
        gmsh.model.mesh.field.setNumber(
            feature_field, "XMax", x_nm + sizing.feature_half_width_nm
        )
        gmsh.model.mesh.field.setNumber(
            feature_field, "YMin", y_nm - sizing.feature_half_width_nm
        )
        gmsh.model.mesh.field.setNumber(
            feature_field, "YMax", y_nm + sizing.feature_half_width_nm
        )
        gmsh.model.mesh.field.setNumber(
            feature_field, "ZMin", z_lower_nm - sizing.feature_half_width_nm
        )
        gmsh.model.mesh.field.setNumber(
            feature_field, "ZMax", z_upper_nm + sizing.feature_half_width_nm
        )
        gmsh.model.mesh.field.setNumber(
            feature_field, "Thickness", sizing.transition_width_nm
        )
        fields.append(feature_field)

    minimum_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(minimum_field, "FieldsList", fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(minimum_field)


__all__ = [
    "GeometryAwareSizing",
    "geometry_aware_sizing",
    "geometry_feature_regions_nm",
    "install_geometry_aware_mesh_field",
]
