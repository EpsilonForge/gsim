"""Tests for the generic doping-profile helper (gsim.common.stack.doping)."""

from __future__ import annotations

import gdsfactory as gf
import pytest

from gsim.common.stack.doping import make_doping_profile


@pytest.fixture
def sides():
    return {
        "upper": {"base_layer": (23, 0), "name_prefix": "pp_slab_", "sign": 1},
        "lower": {"base_layer": (24, 0), "name_prefix": "npp_slab_", "sign": -1},
    }


def _profile():
    return {
        "upper": [(2.0, 2e4), (2.0, 8e4)],
        "lower": [(2.0, 2e4), (2.0, 8e4)],
    }


def test_make_doping_profile_basic(sides):
    comp = gf.Component()
    result = make_doping_profile(
        comp,
        length=10.0,
        rib_center_y=-20.0,
        rib_width=0.4,
        profile=_profile(),
        sides=sides,
        zmin=0.0,
        zmax=0.09,
    )

    expected = {"pp_slab_0", "pp_slab_1", "npp_slab_0", "npp_slab_1"}
    assert set(result["layer_specs"]) == expected
    assert set(result["materials"]) == set(result["layer_specs"])

    # Centres: regions start at the rib edges and are contiguous.
    upper = result["centres"]["upper"]
    lower = result["centres"]["lower"]
    rib_upper_edge = -20.0 + 0.4 / 2
    rib_lower_edge = -20.0 - 0.4 / 2
    assert upper[0] == pytest.approx(rib_upper_edge + 2.0 / 2)
    assert upper[1] == pytest.approx(rib_upper_edge + 2.0 + 2.0 / 2)
    assert lower[0] == pytest.approx(rib_lower_edge - 2.0 / 2)
    assert lower[1] == pytest.approx(rib_lower_edge - 2.0 - 2.0 / 2)


def test_make_doping_profile_layer_spec(sides):
    comp = gf.Component()
    result = make_doping_profile(
        comp,
        length=10.0,
        rib_center_y=0.0,
        rib_width=1.0,
        profile=_profile(),
        sides=sides,
        zmin=0.0,
        zmax=0.09,
    )

    layer = result["layer_specs"]["pp_slab_0"]
    assert layer.gds_layer == (23, 0)
    assert layer.zmin == 0.0
    assert layer.zmax == 0.09
    assert layer.thickness == 0.09
    assert layer.material == "pp_slab_0"

    # Second region uses an offset datatype within the base layer.
    assert result["layer_specs"]["pp_slab_1"].gds_layer == (23, 1)
    assert result["layer_specs"]["npp_slab_0"].gds_layer == (24, 0)


def test_make_doping_profile_materials(sides):
    comp = gf.Component()
    result = make_doping_profile(
        comp,
        length=10.0,
        rib_center_y=0.0,
        rib_width=1.0,
        profile=_profile(),
        sides=sides,
        zmin=0.0,
        zmax=0.09,
    )
    assert result["materials"]["pp_slab_0"].conductivity == 2e4
    assert result["materials"]["pp_slab_1"].conductivity == 8e4
    assert result["materials"]["npp_slab_1"].conductivity == 8e4


def test_make_doping_profile_empty_side(sides):
    comp = gf.Component()
    result = make_doping_profile(
        comp,
        length=10.0,
        rib_center_y=0.0,
        rib_width=1.0,
        profile={"upper": [], "lower": [(1.0, 1e4)]},
        sides=sides,
        zmin=0.0,
        zmax=0.09,
    )
    assert "pp_slab_0" not in result["layer_specs"]
    assert "npp_slab_0" in result["layer_specs"]
    assert result["centres"]["upper"] == []


def test_make_doping_profile_geometry_added(sides):
    comp = gf.Component()
    make_doping_profile(
        comp,
        length=10.0,
        rib_center_y=0.0,
        rib_width=1.0,
        profile=_profile(),
        sides=sides,
        zmin=0.0,
        zmax=0.09,
    )
    # 4 rectangles drawn (2 upper + 2 lower).
    total = len(comp.get_polygons())
    assert total == 4
