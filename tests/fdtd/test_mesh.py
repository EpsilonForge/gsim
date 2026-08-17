"""Tests for solver-specific mesh semantics."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from shapely.geometry import MultiPolygon, Polygon

from gsim.common.pdk import ResolvedLayer
from gsim.fdtd.mesh import _priority_by_mesh_order
from gsim.fdtd.mesh_geometry import iter_polygons, sidewall_slice_count


def test_lower_pdk_mesh_order_becomes_higher_fdtd_priority() -> None:
    layers = {
        "core": SimpleNamespace(mesh_order=1),
        "same_order": SimpleNamespace(mesh_order=1),
        "slab": SimpleNamespace(mesh_order=4),
        "cladding": SimpleNamespace(mesh_order=7),
    }

    priorities = _priority_by_mesh_order(layers)

    assert priorities == {
        "core": 3,
        "same_order": 3,
        "slab": 2,
        "cladding": 1,
    }


def _resolved_layer(geometry: Polygon | MultiPolygon) -> ResolvedLayer:
    return ResolvedLayer(
        key="core",
        declared_name="core",
        layer=(1, 0),
        derived_layer=None,
        geometry=geometry,
        material="Si",
        zmin=0,
        thickness=0.22,
        zmax=0.22,
        sidewall_angle=10,
        width_to_z=0.5,
        bias=None,
        z_to_bias=None,
        mesh_order=1,
    )


def test_sidewall_slices_bound_error_to_quarter_cell() -> None:
    layer = _resolved_layer(Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]))

    slice_count = sidewall_slice_count(layer, nanometers_per_cell=31.25)

    assert slice_count == 3
    total_displacement_nm = 38.7919357558623
    assert total_displacement_nm / (2 * slice_count) < 31.25 / 4


def test_disconnected_polygons_retain_holes() -> None:
    ring = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
    )
    bus = Polygon([(5, 0), (7, 0), (7, 1), (5, 1)])

    polygons = iter_polygons(MultiPolygon([ring, bus]), layer_key="core")

    assert len(polygons) == 2
    assert sum(len(polygon.interiors) for polygon in polygons) == 1
    assert sum(polygon.area for polygon in polygons) == pytest.approx(14)
