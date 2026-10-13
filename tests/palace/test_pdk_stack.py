"""Palace PDK-authoritative stack tests (gsim#230), IHP first."""

from __future__ import annotations

import gdsfactory as gf
import pytest

from gsim.common.pdk import resolve_layer_stack
from gsim.palace import DrivenSim

ihp = pytest.importorskip("ihp")

# Explicit, reviewed IHP background + interconnect declarations.  The IHP
# LayerStack declares no bulk BEOL/cladding region and no interconnect role, so
# the authoritative path requires these to be supplied explicitly (gsim#230).
IHP_DIELECTRICS = [
    {"name": "ild", "material": "sio2", "zmin": -2.0, "zmax": 14.03},
    {"name": "passivation", "material": "sin", "zmin": 14.03, "zmax": 15.93},
]
IHP_VIA_LAYERS = ("cont", "via1", "via2", "via3", "via4", "vmim", "topvia1", "topvia2")
IHP_ROLE_OVERRIDES = dict.fromkeys(IHP_VIA_LAYERS, "via")


@pytest.fixture
def _ihp_active():
    """Activate IHP for the test and restore the previous PDK."""
    try:
        previous = gf.get_active_pdk()
    except ValueError:
        previous = None
    ihp.PDK.activate()
    try:
        yield
    finally:
        if previous is not None:
            previous.activate()


def _component_on_metal1() -> gf.Component:
    metal1_gds = resolve_layer_stack().layers["metal1"].gds_layer
    component = gf.Component()
    component.add_polygon([(0, 0), (1, 0), (1, 1), (0, 1)], layer=metal1_gds)
    return component


def _straight_topmetal2(length: float = 100.0) -> gf.Component:
    component = gf.Component()
    component.add_polygon(
        [(0, 0), (length, 0), (length, 10), (0, 10)], layer=ihp.LAYER.TopMetal2drawing
    )
    return component


@pytest.mark.usefixtures("_ihp_active")
def test_authoritative_stack_is_empty_without_explicit_dielectrics() -> None:
    sim = DrivenSim()
    sim.set_stack(pdk_authoritative=True)
    # No airbox and no explicit dielectrics -> refuse rather than silently
    # producing an air-only domain that fails later with "No dielectric volumes".
    with pytest.raises(ValueError, match="No dielectric volumes"):
        sim._resolve_stack()


@pytest.mark.usefixtures("_ihp_active")
def test_explicit_dielectrics_and_roles_reach_the_resolver() -> None:
    sim = DrivenSim()
    sim.set_airbox(margin_x=0, margin_y=0, z_above=100.0, z_below=100.0)
    sim.set_stack(
        pdk_authoritative=True,
        dielectrics=IHP_DIELECTRICS,
        role_overrides=IHP_ROLE_OVERRIDES,
    )
    stack = sim._resolve_stack()

    assert [d["name"] for d in stack.dielectrics] == ["ild", "passivation"]
    assert stack.layers["via1"].layer_type == "via"
    assert stack.layers["topvia2"].layer_type == "via"
    # mesh_order survives the adapter (review: it used to be dropped).
    assert stack.layers["metal1"].mesh_order == 14


@pytest.mark.usefixtures("_ihp_active")
def test_explicit_dielectrics_become_mesh_volumes(tmp_path) -> None:
    sim = DrivenSim()
    sim.set_output_dir(str(tmp_path / "palace-sim"))
    sim.set_geometry(_straight_topmetal2())
    sim.set_stack(
        pdk_authoritative=True,
        dielectrics=IHP_DIELECTRICS,
        role_overrides=IHP_ROLE_OVERRIDES,
    )
    sim.set_airbox(margin_x=0, margin_y=0, z_above=100.0, z_below=100.0)
    sim.mesh(preset="coarse", margin_x=0, margin_y=0)

    volumes = sim._last_mesh_result.groups["volumes"]
    assert "sio2" in volumes  # explicit BEOL/cladding
    assert "sin" in volumes  # explicit passivation
    assert "air" in volumes


@pytest.mark.usefixtures("_ihp_active")
def test_legacy_kwargs_are_rejected_with_the_flag() -> None:
    sim = DrivenSim()
    with pytest.raises(ValueError, match="ignores legacy stack options"):
        sim.set_stack(pdk_authoritative=True, substrate_thickness=5.0)
    with pytest.raises(ValueError, match="ignores legacy stack options"):
        sim.set_stack(pdk_authoritative=True, include_substrate=True)
    with pytest.raises(ValueError, match="cannot be combined with yaml_path"):
        sim.set_stack(pdk_authoritative=True, yaml_path="stack.yaml")


@pytest.mark.usefixtures("_ihp_active")
def test_legacy_default_is_unchanged() -> None:
    sim = DrivenSim()
    sim.set_stack()
    stack = sim._resolve_stack()

    # The legacy extractor still inserts its synthetic oxide by default; we are
    # not changing the default behaviour in this increment.
    assert "oxide" in [d["name"] for d in stack.dielectrics]
    assert stack.simulation.get("pdk_authoritative") is None


@pytest.mark.usefixtures("_ihp_active")
def test_drawn_extent_uses_component() -> None:
    sim = DrivenSim()
    sim.set_geometry(_component_on_metal1())
    sim.set_airbox(margin_x=0, margin_y=0, z_above=100.0, z_below=100.0)
    sim.set_stack(pdk_authoritative=True)
    stack = sim._resolve_stack()

    # Only Metal1 (0.84..1.26 um) is drawn; the extent metadata reflects that.
    assert stack.simulation["drawn_z_range"] == pytest.approx([0.84, 1.26])
