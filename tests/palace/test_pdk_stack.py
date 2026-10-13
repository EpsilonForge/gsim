"""Palace PDK-authoritative stack tests (gsim#230), IHP first."""

from __future__ import annotations

import gdsfactory as gf
import pytest

from gsim.common.pdk import resolve_layer_stack
from gsim.palace import DrivenSim

ihp = pytest.importorskip("ihp")


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


@pytest.mark.usefixtures("_ihp_active")
def test_pdk_authoritative_stack_has_no_synthetic_geometry() -> None:
    sim = DrivenSim()
    sim.set_stack(pdk_authoritative=True)
    stack = sim._resolve_stack()

    assert stack.pdk_name == "IHP"
    assert len(stack.layers) == 31
    # No synthetic oxide / substrate / collapsed background slab.
    assert stack.dielectrics == []
    assert stack.simulation["pdk_authoritative"] is True

    # PDK passivation region is preserved with its authoritative extent.
    passivation_layer = stack.layers["passiv"]  # codespell:ignore passiv
    assert passivation_layer.material == "sin"
    assert (passivation_layer.zmin, passivation_layer.zmax) == pytest.approx(
        (14.03, 15.93)
    )

    # Materials resolve to concrete properties (no silent unknown fallback).
    assert stack.materials["aluminum"]["conductivity"] == pytest.approx(3.77e7)
    assert "tungsten" in stack.materials


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
def test_authoritative_path_rejects_yaml_path() -> None:
    sim = DrivenSim()
    with pytest.raises(ValueError, match="cannot be combined with yaml_path"):
        sim.set_stack(pdk_authoritative=True, yaml_path="stack.yaml")


@pytest.mark.usefixtures("_ihp_active")
def test_drawn_extent_uses_component() -> None:
    sim = DrivenSim()
    sim.set_geometry(_component_on_metal1())
    sim.set_stack(pdk_authoritative=True)
    stack = sim._resolve_stack()

    # Only Metal1 (0.84..1.26 um) is drawn, so undrawn top/bottom layers must
    # not inflate the domain extent.
    assert stack.simulation["drawn_z_range"] == pytest.approx([0.84, 1.26])
