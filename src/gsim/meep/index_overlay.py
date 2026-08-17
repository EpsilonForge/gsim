"""Simulation annotations drawn over refractive-index maps."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from matplotlib.patches import Rectangle

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_PML_EDGE = (0.0, 0.0, 0.0, 1.0)
_PML_HATCH = "////"
_SOURCE_COLOR = "red"
_MONITOR_COLOR = "royalblue"
_FIBER_ARROW_LENGTH = 0.6


def draw_index_overlay(
    ax: Axes,
    overlay: Any,
    slice_axis: str,
    coordinate: float,
) -> None:
    """Draw PML, source, and monitor annotations."""
    view_min, view_max = _view_bounds(overlay, slice_axis)
    _draw_pml(ax, view_min, view_max, overlay.dpml)

    labeled: set[str] = set()
    for source in getattr(overlay, "sources", []):
        drawn = _draw_port_plane(
            ax,
            source,
            slice_axis,
            coordinate,
            color=_SOURCE_COLOR,
            label="Source" if "Source" not in labeled else None,
        )
        if drawn:
            labeled.add("Source")
    for monitor in getattr(overlay, "monitors", []):
        drawn = _draw_port_plane(
            ax,
            monitor,
            slice_axis,
            coordinate,
            color=_MONITOR_COLOR,
            label="Monitor" if "Monitor" not in labeled else None,
        )
        if drawn:
            labeled.add("Monitor")
    if slice_axis == "y":
        _draw_fiber_planes(ax, overlay, labeled)


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


def _draw_pml(
    ax: Axes,
    view_min: tuple[float, float],
    view_max: tuple[float, float],
    thickness: float,
) -> None:
    """Draw hatched PML strips around the material view."""
    width = view_max[0] - view_min[0]
    height = view_max[1] - view_min[1]
    _add_pml(ax, view_min[0], view_min[1], thickness, height, label="PML")
    _add_pml(ax, view_max[0] - thickness, view_min[1], thickness, height)
    _add_pml(
        ax,
        view_min[0] + thickness,
        view_min[1],
        width - 2 * thickness,
        thickness,
    )
    _add_pml(
        ax,
        view_min[0] + thickness,
        view_max[1] - thickness,
        width - 2 * thickness,
        thickness,
    )


def _add_pml(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str | None = None,
) -> None:
    """Add one unfilled hatched PML rectangle."""
    if width <= 0 or height <= 0:
        return
    patch = Rectangle(
        (x, y),
        width,
        height,
        facecolor="none",
        edgecolor=_PML_EDGE,
        hatch=_PML_HATCH,
        linewidth=0.0,
        label=label,
        zorder=80,
    )
    patch.set_hatch_linewidth(0.6)
    ax.add_patch(patch)


def _draw_port_plane(
    ax: Axes,
    port: Any,
    slice_axis: str,
    coordinate: float,
    *,
    color: str,
    label: str | None,
) -> bool:
    """Draw a source or monitor plane when it intersects the slice."""
    cx, cy, _ = port.center
    half_width = port.width / 2
    drawn = False
    if slice_axis == "z":
        if port.normal_axis == 0:
            ax.plot(
                [cx, cx],
                [cy - half_width, cy + half_width],
                color=color,
                linewidth=2,
                zorder=95,
                label=label,
            )
        else:
            ax.plot(
                [cx - half_width, cx + half_width],
                [cy, cy],
                color=color,
                linewidth=2,
                zorder=95,
                label=label,
            )
        drawn = True
    elif slice_axis == "y":
        drawn = _draw_xz_port_plane(
            ax,
            port,
            coordinate,
            color=color,
            label=label,
        )
    else:
        drawn = _draw_yz_port_plane(
            ax,
            port,
            coordinate,
            color=color,
            label=label,
        )

    if drawn:
        ax.annotate(
            port.name,
            _port_label_position(port, slice_axis),
            fontsize=7,
            color=color,
            ha="center",
            va="bottom",
            zorder=96,
        )
    return drawn


def _draw_xz_port_plane(
    ax: Axes,
    port: Any,
    y_slice: float,
    *,
    color: str,
    label: str | None,
) -> bool:
    """Draw a port plane on an XZ slice."""
    cx, cy, cz = port.center
    half_width = port.width / 2
    half_z = port.z_span / 2
    if port.normal_axis == 0 and abs(cy - y_slice) <= half_width + 1e-9:
        ax.plot(
            [cx, cx],
            [cz - half_z, cz + half_z],
            color=color,
            linewidth=2,
            zorder=95,
            label=label,
        )
        return True
    if port.normal_axis == 1 and abs(cy - y_slice) <= 1e-9:
        ax.add_patch(
            Rectangle(
                (cx - half_width, cz - half_z),
                port.width,
                port.z_span,
                facecolor="none",
                edgecolor=color,
                linewidth=1.5,
                zorder=95,
                label=label,
            )
        )
        return True
    return False


def _draw_yz_port_plane(
    ax: Axes,
    port: Any,
    x_slice: float,
    *,
    color: str,
    label: str | None,
) -> bool:
    """Draw a port plane on a YZ slice."""
    cx, cy, cz = port.center
    half_width = port.width / 2
    half_z = port.z_span / 2
    if port.normal_axis == 1 and abs(cx - x_slice) <= half_width + 1e-9:
        ax.plot(
            [cy, cy],
            [cz - half_z, cz + half_z],
            color=color,
            linewidth=2,
            zorder=95,
            label=label,
        )
        return True
    if port.normal_axis == 0 and abs(cx - x_slice) <= 1e-9:
        ax.add_patch(
            Rectangle(
                (cy - half_width, cz - half_z),
                port.width,
                port.z_span,
                facecolor="none",
                edgecolor=color,
                linewidth=1.5,
                zorder=95,
                label=label,
            )
        )
        return True
    return False


def _port_label_position(port: Any, slice_axis: str) -> tuple[float, float]:
    """Return the label position for a rendered port plane."""
    cx, cy, cz = port.center
    if slice_axis == "z":
        return cx, cy
    horizontal = cy if slice_axis == "x" else cx
    return horizontal, cz + port.z_span / 2


def _draw_fiber_planes(ax: Axes, overlay: Any, labeled: set[str]) -> None:
    """Draw the fiber source plane and its propagation direction."""
    fiber = getattr(overlay, "fiber", None)
    if fiber is not None:
        theta = math.radians(fiber.angle_deg)
        perpendicular_x = math.cos(theta)
        perpendicular_z = math.sin(theta)
        half_span = fiber.waist
        ax.plot(
            [
                fiber.x - perpendicular_x * half_span,
                fiber.x + perpendicular_x * half_span,
            ],
            [
                fiber.z - perpendicular_z * half_span,
                fiber.z + perpendicular_z * half_span,
            ],
            color=_SOURCE_COLOR,
            linewidth=2,
            zorder=95,
            label="Source" if "Source" not in labeled else None,
        )
        direction_x = math.sin(theta)
        direction_z = -math.cos(theta)
        head_x = fiber.x + direction_x * _FIBER_ARROW_LENGTH
        head_z = fiber.z + direction_z * _FIBER_ARROW_LENGTH
        ax.annotate(
            "",
            xy=(head_x, head_z),
            xytext=(fiber.x, fiber.z),
            arrowprops={
                "arrowstyle": "->",
                "color": _SOURCE_COLOR,
                "lw": 1.5,
                "mutation_scale": 8,
            },
            zorder=96,
        )
        labeled.add("Source")
