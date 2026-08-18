"""Refractive-index maps for MEEP simulation previews."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Polygon, Rectangle

from gsim.common.geometry_model import GeometryModel, Prism
from gsim.common.viz._matplotlib import add_bottom_legend
from gsim.meep.index_overlay import draw_index_overlay
from gsim.meep.models.config import MaterialData

if TYPE_CHECKING:
    from matplotlib.axes import Axes

IndexComponent = Literal["mean", "x", "y", "z"]

_AIR_INDEX = 1.0


def material_refractive_indices(
    material_data: Mapping[str, MaterialData],
    component: IndexComponent = "mean",
) -> dict[str, float]:
    """Convert resolved material permittivity to scalar refractive indices.

    ``component="mean"`` uses ``sqrt(mean(epsilon_diag))``. Axis-specific
    values use the corresponding diagonal tensor component.
    """
    if component not in {"mean", "x", "y", "z"}:
        raise ValueError(
            f"index_component must be 'mean', 'x', 'y', or 'z'. Got: {component!r}"
        )

    component_index = {"x": 0, "y": 1, "z": 2}
    indices: dict[str, float] = {}
    for material_name, data in material_data.items():
        if data.epsilon_diag is None:
            continue
        epsilon_values = [float(value) for value in data.epsilon_diag]
        if not epsilon_values:
            continue
        if component == "mean":
            epsilon = float(np.mean(epsilon_values))
        else:
            axis_index = component_index[component]
            if axis_index >= len(epsilon_values):
                raise ValueError(
                    f"Material {material_name!r} has no epsilon {component}-component"
                )
            epsilon = epsilon_values[axis_index]
        if epsilon <= 0 or not math.isfinite(epsilon):
            raise ValueError(
                f"Material {material_name!r} has non-positive or non-finite "
                f"permittivity for index component {component!r}: {epsilon}"
            )
        indices[material_name] = math.sqrt(epsilon)
    return indices


def plot_refractive_index_slices(
    geometry_model: GeometryModel,
    material_data: Mapping[str, MaterialData],
    *,
    wavelength: float,
    overlay: Any | None,
    layer_order: Sequence[str] | None = None,
    is_3d: bool = True,
    plane: Literal["xy", "xz"] = "xy",
    background_material: str = "air",
    index_component: IndexComponent = "mean",
    cmap: str = "Blues",
    x: float | str | None = None,
    y: float | str | None = None,
    z: float | str = "core",
    ax: Axes | None = None,
    legend: bool = True,
    slices: str = "z",
    aspect: Literal["equal", "auto"] = "equal",
) -> Axes | None:
    """Plot ideal material-index cross-sections with simulation overlays."""
    if aspect not in {"equal", "auto"}:
        raise ValueError(f"aspect must be 'equal' or 'auto'. Got: {aspect!r}")

    slices_to_plot = sorted(set(slices.lower()))
    if not slices_to_plot or not all(axis in "xyz" for axis in slices_to_plot):
        raise ValueError(f"slices must only contain 'x', 'y', 'z'. Got: {slices}")
    if ax is not None and len(slices_to_plot) > 1:
        raise ValueError("Cannot plot multiple slices when ax is provided")

    indices = material_refractive_indices(material_data, index_component)
    norm = index_normalization(indices)
    colormap = air_white_colormap(cmap)
    scalar_mappable = ScalarMappable(norm=norm, cmap=colormap)
    ordered_layers = list(layer_order or geometry_model.layer_names)

    if ax is not None:
        _plot_single_index_slice(
            ax,
            geometry_model,
            slices_to_plot[0],
            x=x,
            y=y,
            z=z,
            indices=indices,
            norm=norm,
            cmap=colormap,
            overlay=overlay,
            layer_order=ordered_layers,
            include_dielectrics=is_3d or plane == "xz",
            background_material=background_material,
            aspect=aspect,
            legend=legend,
        )
        ax.figure.colorbar(
            scalar_mappable,
            ax=ax,
            label=index_colorbar_label(index_component, wavelength),
        )
        return ax

    for slice_axis in slices_to_plot:
        figure, plot_axis = plt.subplots(constrained_layout=True)
        _plot_single_index_slice(
            plot_axis,
            geometry_model,
            slice_axis,
            x=x,
            y=y,
            z=z,
            indices=indices,
            norm=norm,
            cmap=colormap,
            overlay=overlay,
            layer_order=ordered_layers,
            include_dielectrics=is_3d or plane == "xz",
            background_material=background_material,
            aspect=aspect,
            legend=False,
        )
        figure.colorbar(
            scalar_mappable,
            ax=plot_axis,
            label=index_colorbar_label(index_component, wavelength),
        )
        if legend:
            add_bottom_legend(figure, plot_axis)
    plt.show()
    return None


def index_normalization(indices: Mapping[str, float]) -> Normalize:
    """Create shared normalization including air at n=1."""
    values = [_AIR_INDEX, *indices.values()]
    minimum = min(values)
    maximum = max(values)
    if math.isclose(minimum, maximum):
        padding = max(0.05 * minimum, 0.05)
        minimum -= padding
        maximum += padding
    return Normalize(vmin=minimum, vmax=maximum)


def air_white_colormap(name: str) -> ListedColormap:
    """Reserve the minimum colormap value for pure-white air."""
    base_colormap = plt.colormaps.get_cmap(name).resampled(256)
    colors = base_colormap(np.linspace(0.0, 1.0, 256))
    colors[0] = (1.0, 1.0, 1.0, 1.0)
    return ListedColormap(colors, name=f"air_white_{base_colormap.name}")


def index_colorbar_label(component: IndexComponent, wavelength: float) -> str:
    """Build the refractive-index colorbar label."""
    suffix = "" if component == "mean" else f"_{component}"
    return f"Refractive index n{suffix} at \u03bb={wavelength:g} µm"


def _plot_single_index_slice(
    ax: Axes,
    geometry_model: GeometryModel,
    slice_axis: str,
    *,
    x: float | str | None,
    y: float | str | None,
    z: float | str,
    indices: Mapping[str, float],
    norm: Normalize,
    cmap: Any,
    overlay: Any | None,
    layer_order: Sequence[str],
    include_dielectrics: bool,
    background_material: str,
    aspect: Literal["equal", "auto"],
    legend: bool,
) -> None:
    """Render one refractive-index slice on an axes."""
    coordinate = resolve_slice_coordinate(geometry_model, slice_axis, x=x, y=y, z=z)
    view_min, view_max = view_bounds(geometry_model, overlay, slice_axis)
    resolved_background = (
        background_material if background_material in indices else "air"
    )
    background_index = indices.get(resolved_background, _AIR_INDEX)
    _add_material_rectangle(
        ax,
        view_min[0],
        view_min[1],
        view_max[0] - view_min[0],
        view_max[1] - view_min[1],
        material=resolved_background,
        refractive_index=background_index,
        norm=norm,
        cmap=cmap,
        zorder=-20,
    )
    if include_dielectrics and overlay is not None:
        _draw_dielectrics(
            ax,
            overlay,
            slice_axis,
            coordinate,
            indices,
            norm,
            cmap,
        )
    _draw_prisms(
        ax,
        geometry_model,
        slice_axis,
        coordinate,
        layer_order,
        indices,
        norm,
        cmap,
    )
    if overlay is not None:
        draw_index_overlay(ax, overlay, slice_axis, coordinate)

    labels = {
        "x": ("y (um)", "z (um)", "YZ", "x"),
        "y": ("x (um)", "z (um)", "XZ", "y"),
        "z": ("x (um)", "y (um)", "XY", "z"),
    }
    xlabel, ylabel, plane_name, coordinate_name = labels[slice_axis]
    ax.set(
        xlim=(view_min[0], view_max[0]),
        ylim=(view_min[1], view_max[1]),
        xlabel=xlabel,
        ylabel=ylabel,
        title=(
            f"Refractive index · {plane_name} cross section at "
            f"{coordinate_name}={coordinate:.2f}"
        ),
    )
    ax.set_aspect(aspect)
    if legend:
        handles, labels_found = ax.get_legend_handles_labels()
        unique = dict(zip(labels_found, handles, strict=False))
        if unique:
            ax.legend(unique.values(), unique.keys(), fancybox=True, framealpha=1.0)


def resolve_slice_coordinate(
    geometry_model: GeometryModel,
    slice_axis: str,
    *,
    x: float | str | None,
    y: float | str | None,
    z: float | str,
) -> float:
    """Resolve a numeric or named slice coordinate."""
    requested = {"x": x, "y": y, "z": z}[slice_axis]
    if requested is None:
        requested = "core"
    if isinstance(requested, str):
        axis_index = {"x": 0, "y": 1, "z": 2}[slice_axis]
        return float(geometry_model.get_layer_center(requested)[axis_index])
    return float(requested)


def view_bounds(
    geometry_model: GeometryModel,
    overlay: Any | None,
    slice_axis: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the two-dimensional plot bounds for a slice."""
    cell_min = overlay.cell_min if overlay is not None else geometry_model.bbox[0]
    cell_max = overlay.cell_max if overlay is not None else geometry_model.bbox[1]
    axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[slice_axis]
    return (
        (cell_min[axes[0]], cell_min[axes[1]]),
        (cell_max[axes[0]], cell_max[axes[1]]),
    )


def _material_index(material: str, indices: Mapping[str, float]) -> float:
    """Return a material index, falling back to air."""
    return indices.get(material, _AIR_INDEX)


def _add_material_rectangle(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    material: str,
    refractive_index: float,
    norm: Normalize,
    cmap: Any,
    zorder: float,
) -> None:
    """Add a material-colored rectangle to an axes."""
    if width <= 0 or height <= 0:
        return
    patch = Rectangle(
        (x, y),
        width,
        height,
        facecolor=cmap(norm(refractive_index)),
        edgecolor="none",
        zorder=zorder,
    )
    patch.set_gid(f"material:{material}")
    ax.add_patch(patch)


def _draw_dielectrics(
    ax: Axes,
    overlay: Any,
    slice_axis: str,
    coordinate: float,
    indices: Mapping[str, float],
    norm: Normalize,
    cmap: Any,
) -> None:
    """Draw background dielectric slabs that intersect the slice."""
    dielectrics = sorted(overlay.dielectrics, key=lambda dielectric: dielectric.zmin)
    for order, dielectric in enumerate(dielectrics):
        refractive_index = indices.get(dielectric.material)
        if refractive_index is None:
            continue
        if slice_axis == "z":
            if not dielectric.zmin <= coordinate <= dielectric.zmax:
                continue
            x0, y0 = overlay.cell_min[0], overlay.cell_min[1]
            width = overlay.cell_max[0] - x0
            height = overlay.cell_max[1] - y0
        else:
            horizontal_axis = 1 if slice_axis == "x" else 0
            x0, y0 = overlay.cell_min[horizontal_axis], dielectric.zmin
            width = overlay.cell_max[horizontal_axis] - x0
            height = dielectric.zmax - dielectric.zmin
        _add_material_rectangle(
            ax,
            x0,
            y0,
            width,
            height,
            material=dielectric.material,
            refractive_index=refractive_index,
            norm=norm,
            cmap=cmap,
            zorder=-10 + order * 0.01,
        )


def _draw_prisms(
    ax: Axes,
    geometry_model: GeometryModel,
    slice_axis: str,
    coordinate: float,
    layer_order: Sequence[str],
    indices: Mapping[str, float],
    norm: Normalize,
    cmap: Any,
) -> None:
    """Draw geometry prisms that intersect the slice."""
    for layer_index, layer_name in enumerate(layer_order):
        for prism in geometry_model.prisms.get(layer_name, []):
            refractive_index = _material_index(prism.material, indices)
            color = cmap(norm(refractive_index))
            zorder = 10 + layer_index
            if slice_axis == "z":
                if prism.z_base <= coordinate <= prism.z_top:
                    patch = Polygon(
                        prism.vertices,
                        closed=True,
                        facecolor=color,
                        edgecolor="none",
                        zorder=zorder,
                    )
                    patch.set_gid(f"material:{prism.material or 'air'}")
                    ax.add_patch(patch)
                continue
            for low, high in prism_intervals(prism, slice_axis, coordinate):
                _add_material_rectangle(
                    ax,
                    low,
                    prism.z_base,
                    high - low,
                    prism.z_top - prism.z_base,
                    material=prism.material or "air",
                    refractive_index=refractive_index,
                    norm=norm,
                    cmap=cmap,
                    zorder=zorder,
                )


def prism_intervals(
    prism: Prism,
    slice_axis: str,
    coordinate: float,
) -> list[tuple[float, float]]:
    """Intersect a prism with an x- or y-normal slice line."""
    from shapely.geometry import (  # type: ignore[import-untyped]
        LineString,
        MultiLineString,
    )
    from shapely.geometry import (
        Polygon as ShapelyPolygon,
    )

    polygon = ShapelyPolygon(prism.vertices)
    if polygon.is_empty or not polygon.is_valid:
        return []
    xmin, ymin, xmax, ymax = polygon.bounds
    if slice_axis == "x":
        line = LineString([(coordinate, ymin - 1.0), (coordinate, ymax + 1.0)])
        coordinate_index = 1
    else:
        line = LineString([(xmin - 1.0, coordinate), (xmax + 1.0, coordinate)])
        coordinate_index = 0
    intersection = polygon.intersection(line)
    if isinstance(intersection, LineString):
        segments = [intersection]
    elif isinstance(intersection, MultiLineString):
        segments = list(intersection.geoms)
    else:
        segments = [
            geometry
            for geometry in getattr(intersection, "geoms", [])
            if isinstance(geometry, LineString)
        ]
    intervals = []
    for segment in segments:
        values = [point[coordinate_index] for point in segment.coords]
        if values and max(values) - min(values) > 1e-9:
            intervals.append((min(values), max(values)))
    return intervals
