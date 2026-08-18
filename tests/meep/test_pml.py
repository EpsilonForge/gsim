"""Tests for material handling at Meep absorbing boundaries."""

from __future__ import annotations

from gsim.common.stack import LayerStack
from gsim.meep.models.config import DomainConfig
from gsim.meep.pml import extend_dielectrics_into_pml


def _domain_config(**updates) -> DomainConfig:
    """Build a minimal resolved domain for PML-extension tests."""
    domain = DomainConfig(
        z_bounds=(-1.0, 2.0),
        dpml=0.5,
        margin_x_low=0.0,
        margin_x_high=0.0,
        margin_y_low=0.0,
        margin_y_high=0.0,
        margin_z_low=0.0,
        margin_z_high=0.0,
        port_margin=0.0,
        extend_ports=0.0,
        source_port_offset=0.0,
        distance_source_to_monitors=0.0,
    )
    return domain.model_copy(update=updates)


def _background_stack() -> LayerStack:
    """Build background slabs touching lower, upper, both, and neither side."""
    return LayerStack(
        dielectrics=[
            {"name": "lower", "material": "lower", "zmin": -1.0, "zmax": 0.0},
            {"name": "upper", "material": "upper", "zmin": 1.0, "zmax": 2.0},
            {"name": "both", "material": "both", "zmin": -1.0, "zmax": 2.0},
            {
                "name": "interior",
                "material": "interior",
                "zmin": 0.25,
                "zmax": 0.75,
            },
        ]
    )


def test_extend_dielectrics_into_adjacent_pml_faces():
    """Extend only the faces that touch the PML-inner Z boundaries."""
    stack = _background_stack()

    extended = extend_dielectrics_into_pml(
        stack,
        _domain_config(),
    )

    by_name = {dielectric["name"]: dielectric for dielectric in extended.dielectrics}
    assert (by_name["lower"]["zmin"], by_name["lower"]["zmax"]) == (-1.5, 0.0)
    assert (by_name["upper"]["zmin"], by_name["upper"]["zmax"]) == (1.0, 2.5)
    assert (by_name["both"]["zmin"], by_name["both"]["zmax"]) == (-1.5, 2.5)
    assert (by_name["interior"]["zmin"], by_name["interior"]["zmax"]) == (
        0.25,
        0.75,
    )
    assert stack.dielectrics[0]["zmin"] == -1.0


def test_extend_dielectrics_can_be_disabled():
    """Preserve existing material extents when explicitly disabled."""
    stack = _background_stack()

    assert (
        extend_dielectrics_into_pml(
            stack,
            _domain_config(extend_into_pml=False),
        )
        is stack
    )
    assert (
        extend_dielectrics_into_pml(
            stack,
            _domain_config(extend_into_pml=True, dpml=0.0),
        )
        is stack
    )
