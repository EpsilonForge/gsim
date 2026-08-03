"""Tests for gsim.palace.plane_section (2D cross-section physical-group viz)."""

from __future__ import annotations

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import Rectangle

from gsim.common.cross_section import PolygonXY2D, Rect2D, RectYZ2D
from gsim.palace.plane_section import plot_plane_section


def _rect_yz(y0, y1, z0, z1, name="rib"):
    return RectYZ2D(y0=y0, y1=y1, zmin=z0, zmax=z1, layer_name=name, material=name)


def _rect_xz(x0, x1, z0, z1, name="rib"):
    return Rect2D(x0=x0, x1=x1, zmin=z0, zmax=z1, layer_name=name, material=name)


class TestPlotPlaneSection:
    def test_rect_yz(self):
        fig, ax = plt.subplots()
        out = plot_plane_section(
            [_rect_yz(-20, -19.8, 0, 0.22, "p_rib")], ax=ax, legend=False
        )
        assert out is ax
        assert len(ax.patches) == 1
        patch = ax.patches[0]
        assert isinstance(patch, Rectangle)
        assert patch.get_x() == -20
        assert patch.get_y() == 0
        assert patch.get_width() == pytest.approx(0.2)
        assert patch.get_height() == 0.22
        assert ax.get_xlabel() == "y (um)"
        assert ax.get_ylabel() == "z (um)"
        plt.close(fig)

    def test_rect_xz(self):
        fig, ax = plt.subplots()
        plot_plane_section([_rect_xz(0, 10, 0, 1, "wg")], ax=ax, legend=False)
        assert ax.get_xlabel() == "x (um)"
        assert ax.get_ylabel() == "z (um)"
        plt.close(fig)

    def test_polygon_xy(self):
        fig, ax = plt.subplots()
        poly = PolygonXY2D(
            exterior=((0, 0), (1, 0), (1, 1), (0, 1)),
            holes=(),
            layer_name="core",
            material="si",
        )
        plot_plane_section([poly], ax=ax, legend=False)
        assert ax.get_xlabel() == "x (um)"
        assert ax.get_ylabel() == "y (um)"
        assert len(ax.patches) == 1
        plt.close(fig)

    def test_custom_colors(self):
        fig, ax = plt.subplots()
        plot_plane_section(
            [_rect_yz(-1, 1, 0, 0.2, "a"), _rect_yz(2, 3, 0, 0.1, "b")],
            colors={"a": "red", "b": "blue"},
            ax=ax,
            legend=False,
        )
        assert ax.patches[0].get_facecolor()[:3] == (1.0, 0.0, 0.0)
        assert ax.patches[1].get_facecolor()[:3] == (0.0, 0.0, 1.0)
        plt.close(fig)

    def test_auto_limits_and_ranges(self):
        fig, ax = plt.subplots()
        plot_plane_section(
            [_rect_yz(1, 2, 3, 4, "a")],
            h_range=(0, 10),
            v_range=(-1, 5),
            ax=ax,
            legend=False,
        )
        assert ax.get_xlim() == (0, 10)
        assert ax.get_ylim() == (-1, 5)
        plt.close(fig)

    def test_legend_deduplicated(self):
        fig, ax = plt.subplots()
        plot_plane_section(
            [_rect_yz(-1, 1, 0, 0.2, "a"), _rect_yz(-2, -1.5, 0, 0.1, "a")],
            ax=ax,
        )
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert labels.count("a") == 1
        plt.close(fig)

    def test_unsupported_element(self):
        with pytest.raises(TypeError):
            plot_plane_section([object()])
