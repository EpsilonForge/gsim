"""Visualization tests for XZ 2D preview (``plot_2d(slices='y')``)."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt


def _xz_sim_for_viz():
    """Build a minimal XZ simulation suitable for plot_2d tests."""
    import gdsfactory as gf

    from gsim.common.stack import Layer, LayerStack
    from gsim.meep.simulation import Simulation

    c = gf.Component()
    c.add_polygon(
        [(-5, -0.25), (5, -0.25), (5, 0.25), (-5, 0.25)],
        layer=(1, 0),
    )
    c.add_port(
        name="o1",
        center=(5.0, 0.0),
        orientation=0.0,
        width=0.5,
        layer=(1, 0),
    )

    stack = LayerStack(
        pdk_name="test",
        units="um",
        layers={
            "core": Layer(
                name="core",
                gds_layer=(1, 0),
                zmin=0.0,
                zmax=0.22,
                thickness=0.22,
                material="si",
                layer_type="dielectric",
            ),
        },
        materials={},
        dielectrics=[
            {"name": "box", "zmin": -2.0, "zmax": 0.0, "material": "SiO2"},
            {"name": "clad", "zmin": 0.22, "zmax": 1.0, "material": "SiO2"},
        ],
        simulation={},
    )

    sim = Simulation()
    sim.geometry.component = c
    sim.geometry.stack = stack
    sim.materials = {"si": 12.0, "SiO2": 2.1}
    sim.solver(mode="2d", y_cut="auto")
    sim.source_fiber(x=0.0, z=1.22, waist=5.4)
    return sim


class TestPlot2DXZ:
    """Smoke tests that plot_2d(slices='y') works for XZ sims."""

    def test_slices_y_returns_axes(self):
        sim = _xz_sim_for_viz()
        fig, ax = plt.subplots()
        result = sim.plot_2d(slices="y", ax=ax)
        assert result is ax
        plt.close(fig)

    def test_default_slice_when_plane_xz(self):
        sim = _xz_sim_for_viz()
        fig, ax = plt.subplots()
        # Default: plane='xz' -> slices='y'.
        result = sim.plot_2d(ax=ax)
        assert result is ax
        plt.close(fig)

    def test_auto_aspect(self):
        sim = _xz_sim_for_viz()
        fig, ax = plt.subplots()

        sim.plot_2d(ax=ax, aspect="auto")

        assert ax.get_aspect() == "auto"
        plt.close(fig)

    def test_invalid_aspect_raises(self):
        import pytest

        sim = _xz_sim_for_viz()
        fig, ax = plt.subplots()

        with pytest.raises(ValueError, match="aspect must be"):
            sim.plot_2d(ax=ax, aspect="invalid")

        plt.close(fig)

    def test_cross_section_omits_undrawn_stack_layers(self):
        """Absent GDS layers must not appear as full-width slabs."""
        from gsim.common.stack import Layer
        from gsim.meep.viz import build_cross_section_rectangles

        sim = _xz_sim_for_viz()
        assert sim.geometry.component is not None
        assert sim.geometry.stack is not None
        sim.geometry.stack.layers["undrawn_metal"] = Layer(
            name="undrawn_metal",
            gds_layer=(99, 0),
            zmin=1.0,
            zmax=3.0,
            thickness=2.0,
            material="aluminum",
            layer_type="conductor",
        )

        rectangles = build_cross_section_rectangles(
            sim.geometry.component,
            sim.geometry.stack,
            slice_axis="y",
            slice_coord=0.0,
        )

        assert [rectangle["layer_name"] for rectangle in rectangles] == ["core"]
