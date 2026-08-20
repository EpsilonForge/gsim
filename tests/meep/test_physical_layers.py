"""Regression tests for collision-free physical-layer materialization."""

from __future__ import annotations

import json

import gdsfactory as gf
from kfactory import kdb

from gsim.common.cross_section import extract_xz_rectangles
from gsim.common.stack import get_stack
from gsim.meep.physical_layers import (
    allocate_physical_layers,
    materialize_physical_layers,
)


def _region(component, layer: tuple[int, int]) -> kdb.Region:
    layer_index = component.kcl.layer(*layer)
    return kdb.Region(component.kdb_cell.begin_shapes_rec(layer_index))


def test_allocator_is_deterministic_and_skips_source_layers():
    used_layers = {(65_535, 0), (65_534, 0)}

    first = allocate_physical_layers(["core", "slab"], used_layers)
    second = allocate_physical_layers(["core", "slab"], used_layers)

    assert first == second
    assert first == {"core": (65_533, 0), "slab": (65_532, 0)}


def test_direct_slab_does_not_create_shallow_etch():
    component = gf.Component()
    component.add_polygon(
        [(0, 0), (10, 0), (10, 2), (0, 2)],
        layer=gf.gpdk.LAYER.WG,
    )
    component.add_polygon(
        [(0, 0), (10, 0), (10, 2), (0, 2)],
        layer=gf.gpdk.LAYER.SLAB150,
    )

    physical = materialize_physical_layers(component, get_stack())
    core = _region(physical.component, physical.layer_map["core"])
    shallow = _region(physical.component, physical.layer_map["shallow_etch"])
    slab = _region(physical.component, physical.layer_map["slab150"])

    assert not core.is_empty()
    assert shallow.is_empty()
    assert not slab.is_empty()
    assert physical.layer_map["shallow_etch"] != physical.layer_map["slab150"]


def test_real_shallow_etch_stays_separate_from_direct_slab():
    component = gf.Component()
    component.add_polygon(
        [(0, 0), (10, 0), (10, 2), (0, 2)],
        layer=gf.gpdk.LAYER.WG,
    )
    component.add_polygon(
        [(4, 0.5), (6, 0.5), (6, 1.5), (4, 1.5)],
        layer=gf.gpdk.LAYER.SHALLOW_ETCH,
    )
    component.add_polygon(
        [(8, 3), (10, 3), (10, 4), (8, 4)],
        layer=gf.gpdk.LAYER.SLAB150,
    )

    physical = materialize_physical_layers(component, get_stack())
    shallow = _region(physical.component, physical.layer_map["shallow_etch"])
    slab = _region(physical.component, physical.layer_map["slab150"])

    assert shallow.area() == 2_000_000
    assert slab.area() == 2_000_000
    assert physical.stack.layers["shallow_etch"].gds_layer != (
        physical.stack.layers["slab150"].gds_layer
    )


def test_build_config_and_xz_geometry_omit_false_shallow_etch(tmp_path):
    from gsim.meep import Simulation
    from gsim.meep.viz import build_geometry_model

    component = gf.Component()
    component.add_polygon(
        [(0, -0.25), (10, -0.25), (10, 0.25), (0, 0.25)],
        layer=gf.gpdk.LAYER.WG,
    )
    component.add_polygon(
        [(0, -0.25), (10, -0.25), (10, 0.25), (0, 0.25)],
        layer=gf.gpdk.LAYER.SLAB150,
    )
    component.add_port(
        name="o1",
        center=(0, 0),
        orientation=180,
        width=0.5,
        layer=gf.gpdk.LAYER.WG,
    )

    simulation = Simulation()
    simulation.geometry(component=component, stack=get_stack())
    simulation.materials = {"si": 12.0, "SiO2": 2.1}
    simulation.solver(mode="2d", y_cut=0.0)
    simulation.source_fiber(x=5.0, z=1.0, waist=2.0)

    result = simulation.build_config()
    entries = {entry.layer_name: entry for entry in result.config.layer_stack}
    rectangles = extract_xz_rectangles(
        result.component,
        result.stack,
        y_cut=0.0,
    )
    rectangle_names = {rectangle.layer_name for rectangle in rectangles}
    geometry_model = build_geometry_model(
        result.component,
        result.stack,
        result.config.domain,
        extend_ports_length=0,
        gdsfactory_stack=result.gdsfactory_stack,
    )

    assert tuple(entries["shallow_etch"].gds_layer) != tuple(
        entries["slab150"].gds_layer
    )
    assert "shallow_etch" not in rectangle_names
    assert {"core", "slab150"} <= rectangle_names
    assert "shallow_etch" not in geometry_model.prisms
    assert geometry_model.prisms["slab150"]

    output_directory = simulation.write_config(tmp_path / "simulation")
    serialized = json.loads(
        (output_directory / "sim_config.json").read_text(encoding="utf-8")
    )
    serialized_entries = {
        entry["layer_name"]: tuple(entry["gds_layer"])
        for entry in serialized["layer_stack"]
    }
    written_component = gf.import_gds(
        output_directory / "layout.gds",
        rename_duplicated_cells=True,
    )

    assert _region(
        written_component,
        serialized_entries["shallow_etch"],
    ).is_empty()
    assert not _region(
        written_component,
        serialized_entries["slab150"],
    ).is_empty()
