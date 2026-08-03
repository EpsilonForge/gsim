"""2D cross-section ('plane section') visualisation.

Given a list of axis-aligned rectangles / polygons produced by
``gsim.common.cross_section.extract_plane_section`` (e.g. an ``x=0`` cut
through a component), this module draws a matplotlib 2D cross-section where
every region is coloured by its layer / physical-group name.

This is the reusable, post-processing version of the inline plot that used to
live in ``nbs/palace_2d_twmzm.ipynb``.  It is solver-agnostic with respect to
the *section* data (only the small dataclass types from
``gsim.common.cross_section`` are required), so it can be used to visualise
the physical groups of any Palace cross-section.

Supported inputs:
    - ``Rect2D`` (XZ cut at fixed Y; horizontal axis = x, vertical = z)
    - ``RectYZ2D`` (YZ cut at fixed X; horizontal axis = y, vertical = z)
    - ``PolygonXY2D`` (XY cut at fixed Z; polygons in the x-y plane)
"""

from __future__ import annotations

from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon, Rectangle

from gsim.common.cross_section import PolygonXY2D, Rect2D, RectYZ2D

__all__ = ["plot_plane_section"]

_LINE_COLOR = "k"
_LINE_WIDTH = 0.5


def _rect_props(
    rect: Rect2D | RectYZ2D,
) -> tuple[tuple[float, float], float, float, tuple[str, str]]:
    """Return ``((h0, h1), v0, v1, (h_label, v_label))`` for a rectangle."""
    if isinstance(rect, RectYZ2D):
        return (rect.y0, rect.y1), rect.zmin, rect.zmax, ("y (um)", "z (um)")
    return (rect.x0, rect.x1), rect.zmin, rect.zmax, ("x (um)", "z (um)")


def _color_map(names: list[str], colors: dict[str, Any] | None) -> dict[str, Any]:
    """Merge user colours with deterministic ``tab10`` fallbacks."""
    if colors is None:
        colors = {}
    cmap = plt.colormaps.get_cmap("tab10")
    for i, name in enumerate(dict.fromkeys(names)):
        colors.setdefault(name, cmap(i % 10))
    return colors


def plot_plane_section(
    section: list[Any],
    *,
    colors: dict[str, Any] | None = None,
    h_range: tuple[float, float] | None = None,
    v_range: tuple[float, float] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 4),
    aspect: Literal["auto", "equal"] | float = "equal",
    legend: bool = True,
    ax: Axes | None = None,
) -> Axes:
    """Plot a 2D cross-section coloured by layer / physical-group name.

    Args:
        section: Regions from ``extract_plane_section`` (``Rect2D``,
            ``RectYZ2D`` or ``PolygonXY2D``).  A single mixed type is
            expected per call.
        colors: Optional ``{layer_name: colour}`` override.  When omitted,
            colours are drawn deterministically from the ``tab10`` colormap.
        h_range: ``(h_min, h_max)`` limits for the horizontal axis.  Auto
            from the section when ``None``.
        v_range: ``(v_min, v_max)`` limits for the vertical axis.  Auto
            from the section when ``None``.
        title: Optional plot title.
        figsize: Figure size (ignored if *ax* is supplied).
        aspect: Matplotlib aspect ratio for the axes (e.g. ``"equal"``).
        legend: Whether to show the per-group legend.
        ax: Optional existing axes to draw on.  If ``None`` a new figure is
            created and shown.

    Returns:
        The ``plt.Axes`` the section was drawn on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    names = [getattr(r, "layer_name", str(r)) for r in section]
    color_map = _color_map(names, colors)

    h_lims: list[float] = []
    v_lims: list[float] = []
    h_label: str | None = None
    v_label: str | None = None

    for r in section:
        name = getattr(r, "layer_name", str(r))
        color = color_map.get(name, "#dddddd")

        if isinstance(r, (Rect2D, RectYZ2D)):
            (h0, h1), v0, v1, (hl, vl) = _rect_props(r)
            h_label = hl
            v_label = vl
            h_lims += [h0, h1]
            v_lims += [v0, v1]
            ax.add_patch(
                Rectangle(
                    (h0, v0),
                    h1 - h0,
                    v1 - v0,
                    facecolor=color,
                    edgecolor=_LINE_COLOR,
                    linewidth=_LINE_WIDTH,
                    alpha=0.8,
                    label=name,
                )
            )
        elif isinstance(r, PolygonXY2D):
            h_label = "x (um)"
            v_label = "y (um)"
            h_lims += [p[0] for p in r.exterior]
            v_lims += [p[1] for p in r.exterior]
            ax.add_patch(
                Polygon(
                    r.exterior,
                    facecolor=color,
                    edgecolor=_LINE_COLOR,
                    linewidth=_LINE_WIDTH,
                    alpha=0.8,
                    label=name,
                )
            )
        else:
            msg = f"Unsupported section element: {type(r)!r}"
            raise TypeError(msg)

    if h_range is not None:
        ax.set_xlim(*h_range)
    elif h_lims:
        ax.set_xlim(min(h_lims), max(h_lims))
    if v_range is not None:
        ax.set_ylim(*v_range)
    elif v_lims:
        ax.set_ylim(min(v_lims), max(v_lims))

    if h_label is not None:
        ax.set_xlabel(h_label)
    if v_label is not None:
        ax.set_ylabel(v_label)
    ax.set_aspect(aspect)
    ax.set_title(title or "Cross-section physical groups")

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles, strict=True))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

    return ax
