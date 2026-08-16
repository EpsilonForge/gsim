"""Plotly annotations for interactive refractive-index maps."""

from __future__ import annotations

import math
from typing import Any

_SOURCE_COLOR = "red"
_MONITOR_COLOR = "royalblue"
_FIBER_ARROW_LENGTH = 0.6


def draw_index_overlay_interactive(
    figure: Any,
    overlay: Any,
    slice_axis: str,
    coordinate: float,
) -> None:
    """Draw hatched PML, source, and monitor annotations."""
    view_min, view_max = _view_bounds(overlay, slice_axis)
    _add_pml_regions(figure, view_min, view_max, overlay.dpml)

    labeled: set[str] = set()
    for source in overlay.sources:
        if _add_port_plane(
            figure,
            source,
            slice_axis,
            coordinate,
            color=_SOURCE_COLOR,
            legend_name="Source",
            show_legend="Source" not in labeled,
        ):
            labeled.add("Source")
    for monitor in overlay.monitors:
        if _add_port_plane(
            figure,
            monitor,
            slice_axis,
            coordinate,
            color=_MONITOR_COLOR,
            legend_name="Monitor",
            show_legend="Monitor" not in labeled,
        ):
            labeled.add("Monitor")
    if slice_axis == "y" and overlay.fiber is not None:
        _add_fiber_source(
            figure,
            overlay.fiber,
            show_legend="Source" not in labeled,
        )


def _view_bounds(
    overlay: Any,
    slice_axis: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the displayed coordinate bounds for a slice axis."""
    axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[slice_axis]
    return (
        (overlay.cell_min[axes[0]], overlay.cell_min[axes[1]]),
        (overlay.cell_max[axes[0]], overlay.cell_max[axes[1]]),
    )


def _add_pml_regions(
    figure: Any,
    view_min: tuple[float, float],
    view_max: tuple[float, float],
    thickness: float,
) -> None:
    """Add hatched PML strips around the material view."""
    width = view_max[0] - view_min[0]
    height = view_max[1] - view_min[1]
    regions = [
        (view_min[0], view_min[1], thickness, height),
        (view_max[0] - thickness, view_min[1], thickness, height),
        (view_min[0] + thickness, view_min[1], width - 2 * thickness, thickness),
        (
            view_min[0] + thickness,
            view_max[1] - thickness,
            width - 2 * thickness,
            thickness,
        ),
    ]
    show_legend = True
    for x, y, region_width, region_height in regions:
        if region_width <= 0 or region_height <= 0:
            continue
        _add_pml_rectangle(
            figure,
            x,
            y,
            region_width,
            region_height,
            show_legend=show_legend,
        )
        show_legend = False


def _add_pml_rectangle(
    figure: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    show_legend: bool,
) -> None:
    """Add one transparent hatched PML rectangle."""
    import plotly.graph_objects as go

    figure.add_trace(
        go.Scatter(
            x=[x, x + width, x + width, x, x],
            y=[y, y, y + height, y + height, y],
            mode="lines",
            fill="toself",
            fillcolor="rgba(0,0,0,0)",
            fillpattern={
                "shape": "/",
                "fgcolor": "black",
                "bgcolor": "rgba(0,0,0,0)",
                "size": 10,
                "solidity": 0.05,
            },
            line={"color": "rgba(0,0,0,0)", "width": 0},
            name="PML",
            legendgroup="PML",
            showlegend=show_legend,
            hoverinfo="text",
            hovertext="PML",
        )
    )


def _add_port_plane(
    figure: Any,
    port: Any,
    slice_axis: str,
    coordinate: float,
    *,
    color: str,
    legend_name: str,
    show_legend: bool,
) -> bool:
    """Add a source or monitor plane when it intersects the slice."""
    cx, cy, _ = port.center
    half_width = port.width / 2
    if slice_axis == "z":
        if port.normal_axis == 0:
            points = ([cx, cx], [cy - half_width, cy + half_width])
        else:
            points = ([cx - half_width, cx + half_width], [cy, cy])
    elif slice_axis == "y":
        points = _xz_port_points(port, coordinate)
    else:
        points = _yz_port_points(port, coordinate)
    if points is None:
        return False

    _add_plane_trace(
        figure,
        points[0],
        points[1],
        port.name,
        color=color,
        legend_name=legend_name,
        show_legend=show_legend,
    )
    return True


def _xz_port_points(
    port: Any,
    y_slice: float,
) -> tuple[list[float], list[float]] | None:
    """Return endpoints for a port plane on an XZ slice."""
    cx, cy, cz = port.center
    half_width = port.width / 2
    half_z = port.z_span / 2
    if port.normal_axis == 0 and abs(cy - y_slice) <= half_width + 1e-9:
        return [cx, cx], [cz - half_z, cz + half_z]
    if port.normal_axis == 1 and abs(cy - y_slice) <= 1e-9:
        return (
            [
                cx - half_width,
                cx + half_width,
                cx + half_width,
                cx - half_width,
                cx - half_width,
            ],
            [cz - half_z, cz - half_z, cz + half_z, cz + half_z, cz - half_z],
        )
    return None


def _yz_port_points(
    port: Any,
    x_slice: float,
) -> tuple[list[float], list[float]] | None:
    """Return endpoints for a port plane on a YZ slice."""
    cx, cy, cz = port.center
    half_width = port.width / 2
    half_z = port.z_span / 2
    if port.normal_axis == 1 and abs(cx - x_slice) <= half_width + 1e-9:
        return [cy, cy], [cz - half_z, cz + half_z]
    if port.normal_axis == 0 and abs(cx - x_slice) <= 1e-9:
        return (
            [
                cy - half_width,
                cy + half_width,
                cy + half_width,
                cy - half_width,
                cy - half_width,
            ],
            [cz - half_z, cz - half_z, cz + half_z, cz + half_z, cz - half_z],
        )
    return None


def _add_plane_trace(
    figure: Any,
    horizontal: list[float],
    vertical: list[float],
    annotation: str,
    *,
    color: str,
    legend_name: str,
    show_legend: bool,
) -> None:
    """Add a Plotly line trace for a source or monitor plane."""
    import plotly.graph_objects as go

    text = [""] * len(horizontal)
    text[len(text) // 2] = annotation
    figure.add_trace(
        go.Scatter(
            x=horizontal,
            y=vertical,
            mode="lines+text",
            line={"color": color, "width": 2},
            name=legend_name,
            legendgroup=legend_name,
            showlegend=show_legend,
            text=text,
            textposition="top center",
            textfont={"size": 9, "color": color},
            hoverinfo="text",
            hovertext=f"{annotation} ({legend_name})",
        )
    )


def _add_fiber_source(
    figure: Any,
    fiber: Any,
    *,
    show_legend: bool,
) -> None:
    """Add the fiber source plane and propagation arrow."""
    import plotly.graph_objects as go

    theta = math.radians(fiber.angle_deg)
    perpendicular_x = math.cos(theta)
    perpendicular_z = math.sin(theta)
    half_span = fiber.waist
    figure.add_trace(
        go.Scatter(
            x=[
                fiber.x - perpendicular_x * half_span,
                fiber.x + perpendicular_x * half_span,
            ],
            y=[
                fiber.z - perpendicular_z * half_span,
                fiber.z + perpendicular_z * half_span,
            ],
            mode="lines+text",
            line={"color": _SOURCE_COLOR, "width": 2},
            name="Source",
            legendgroup="Source",
            showlegend=show_legend,
            text=["", "fiber"],
            textposition="top center",
            textfont={"size": 9, "color": _SOURCE_COLOR},
            hoverinfo="text",
            hovertext="fiber source",
        )
    )
    direction_x = math.sin(theta)
    direction_z = -math.cos(theta)
    head_x = fiber.x + direction_x * _FIBER_ARROW_LENGTH
    head_z = fiber.z + direction_z * _FIBER_ARROW_LENGTH
    figure.add_annotation(
        x=head_x,
        y=head_z,
        ax=fiber.x,
        ay=fiber.z,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowcolor=_SOURCE_COLOR,
    )
