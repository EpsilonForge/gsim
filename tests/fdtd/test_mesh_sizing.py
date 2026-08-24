"""Tests for geometry-aware FDTD transfer-mesh sizing."""

from __future__ import annotations

from types import SimpleNamespace

from shapely.geometry import MultiPolygon, Polygon

from gsim.fdtd.mesh_sizing import (
    geometry_aware_sizing,
    geometry_feature_regions_nm,
)


def test_approved_fifty_nanometer_policy() -> None:
    """The selected prototype settings remain the production policy."""
    sizing = geometry_aware_sizing(mesh_size_nm=50, nanometers_per_cell=60)

    assert sizing.feature_size_nm == 50
    assert sizing.bulk_size_nm == 400
    assert sizing.feature_half_width_nm == 100
    assert sizing.transition_width_nm == 500


def test_default_policy_tracks_the_yee_grid_at_features() -> None:
    """Slow bulk regions remain coarse while features follow solver resolution."""
    sizing = geometry_aware_sizing(mesh_size_nm=500, nanometers_per_cell=60)

    assert sizing.feature_size_nm == 60
    assert sizing.bulk_size_nm == 500
    assert sizing.feature_half_width_nm == 120
    assert sizing.transition_width_nm == 600


def test_feature_regions_include_exterior_hole_and_disconnected_edges() -> None:
    """Every material ring contributes unique vertical refinement regions."""
    ring = Polygon(
        [(0, 0), (4, 0), (4, 4), (0, 4)],
        holes=[[(1, 1), (3, 1), (3, 3), (1, 3)]],
    )
    bus = Polygon([(5, 0), (7, 0), (7, 1), (5, 1)])
    layer = SimpleNamespace(
        key="core",
        geometry=MultiPolygon([ring, bus]),
        z_bounds=(0.1, 0.3),
    )
    resolved = SimpleNamespace(layers={"core": layer})

    regions = geometry_feature_regions_nm(resolved)

    assert len(regions) == 12
    assert (0.0, 0.0, 100.0, 300.0) in regions
    assert (1000.0, 1000.0, 100.0, 300.0) in regions
    assert (7000.0, 1000.0, 100.0, 300.0) in regions
