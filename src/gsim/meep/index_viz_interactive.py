"""Interactive Plotly refractive-index maps for MEEP previews."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from matplotlib.colors import to_hex

from gsim.common.geometry_model import GeometryModel
from gsim.meep.index_overlay_interactive import draw_index_overlay_interactive
from gsim.meep.index_viz import (
    IndexComponent,
    air_white_colormap,
    index_colorbar_label,
    index_normalization,
    material_refractive_indices,
    prism_intervals,
    resolve_slice_coordinate,
    view_bounds,
)
from gsim.meep.models.config import MaterialData

_AIR_INDEX = 1.0


def plot_refractive_index_interactive(
    geometry_model: GeometryModel,
    material_data: Mapping[str, MaterialData],
    *,
    wavelength: float,
    overlay: Any | None,
    slice_axis: Literal["x", "y", "z"],
    layer_order: Sequence[str] | None = None,
    is_3d: bool = True,
    plane: Literal["xy", "xz"] = "xy",
    background_material: str = "air",
    index_component: IndexComponent = "mean",
    cmap: str = "Blues",
    x: float | str | None = None,
    y: float | str | None = None,
    z: float | str = "core",
    aspect: Literal["equal", "auto"] = "equal",
) -> Any:
    """Create an interactive refractive-index cross-section."""
    import plotly.graph_objects as go

    if aspect not in {"equal", "auto"}:
        raise ValueError(f"aspect must be 'equal' or 'auto'. Got: {aspect!r}")

    coordinate = resolve_slice_coordinate(
        geometry_model,
        slice_axis,
        x=x,
        y=y,
        z=z,
    )
    indices = material_refractive_indices(material_data, index_component)
    normalization = index_normalization(indices)
    colormap = air_white_colormap(cmap)
    plot_min, plot_max = view_bounds(geometry_model, overlay, slice_axis)
    figure = go.Figure()

    resolved_background = (
        background_material if background_material in indices else "air"
    )
    background_index = indices.get(resolved_background, _AIR_INDEX)
    _add_material_rectangle(
        figure,
        plot_min[0],
        plot_min[1],
        plot_max[0] - plot_min[0],
        plot_max[1] - plot_min[1],
        material=resolved_background,
        refractive_index=background_index,
        normalization=normalization,
        colormap=colormap,
    )
    if (is_3d or plane == "xz") and overlay is not None:
        _add_dielectrics(
            figure,
            overlay,
            slice_axis,
            coordinate,
            indices,
            normalization,
            colormap,
        )
    _add_prisms(
        figure,
        geometry_model,
        slice_axis,
        coordinate,
        list(layer_order or geometry_model.layer_names),
        indices,
        normalization,
        colormap,
    )
    if overlay is not None:
        draw_index_overlay_interactive(figure, overlay, slice_axis, coordinate)

    _add_colorbar(
        figure,
        normalization,
        colormap,
        index_colorbar_label(index_component, wavelength),
    )
    _configure_layout(
        figure,
        slice_axis,
        coordinate,
        plot_min,
        plot_max,
        aspect,
    )
    return figure


def _color(refractive_index: float, normalization: Any, colormap: Any) -> str:
    """Map a refractive index to a hexadecimal color."""
    return to_hex(colormap(normalization(refractive_index)), keep_alpha=False)


def _add_material_rectangle(
    figure: Any,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    material: str,
    refractive_index: float,
    normalization: Any,
    colormap: Any,
) -> None:
    """Add a material-colored rectangle to a Plotly figure."""
    if width <= 0 or height <= 0:
        return
    _add_filled_trace(
        figure,
        [x, x + width, x + width, x, x],
        [y, y, y + height, y + height, y],
        material=material,
        refractive_index=refractive_index,
        fillcolor=_color(refractive_index, normalization, colormap),
    )


def _add_filled_trace(
    figure: Any,
    horizontal: list[float],
    vertical: list[float],
    *,
    material: str,
    refractive_index: float,
    fillcolor: str,
) -> None:
    """Add a filled material polygon with hover metadata."""
    import plotly.graph_objects as go

    figure.add_trace(
        go.Scatter(
            x=horizontal,
            y=vertical,
            mode="lines",
            fill="toself",
            fillcolor=fillcolor,
            line={"color": "rgba(0,0,0,0)", "width": 0},
            showlegend=False,
            name=material,
            legendgroup=f"material:{material}",
            hovertemplate=(f"{material}<br>n={refractive_index:.4g}<extra></extra>"),
        )
    )


def _add_dielectrics(
    figure: Any,
    overlay: Any,
    slice_axis: str,
    coordinate: float,
    indices: Mapping[str, float],
    normalization: Any,
    colormap: Any,
) -> None:
    """Add background dielectric slabs that intersect the slice."""
    dielectrics = sorted(overlay.dielectrics, key=lambda dielectric: dielectric.zmin)
    for dielectric in dielectrics:
        refractive_index = indices.get(dielectric.material)
        if refractive_index is None:
            continue
        if slice_axis == "z":
            if not dielectric.zmin <= coordinate <= dielectric.zmax:
                continue
            x, y = overlay.cell_min[0], overlay.cell_min[1]
            width = overlay.cell_max[0] - x
            height = overlay.cell_max[1] - y
        else:
            horizontal_axis = 1 if slice_axis == "x" else 0
            x, y = overlay.cell_min[horizontal_axis], dielectric.zmin
            width = overlay.cell_max[horizontal_axis] - x
            height = dielectric.zmax - dielectric.zmin
        _add_material_rectangle(
            figure,
            x,
            y,
            width,
            height,
            material=dielectric.material,
            refractive_index=refractive_index,
            normalization=normalization,
            colormap=colormap,
        )


def _add_prisms(
    figure: Any,
    geometry_model: GeometryModel,
    slice_axis: str,
    coordinate: float,
    layer_order: Sequence[str],
    indices: Mapping[str, float],
    normalization: Any,
    colormap: Any,
) -> None:
    """Add geometry prisms that intersect the slice."""
    for layer_name in layer_order:
        for prism in geometry_model.prisms.get(layer_name, []):
            refractive_index = indices.get(prism.material, _AIR_INDEX)
            fillcolor = _color(refractive_index, normalization, colormap)
            material = prism.material or "air"
            if slice_axis == "z":
                if not prism.z_base <= coordinate <= prism.z_top:
                    continue
                vertices = prism.vertices.tolist()
                _add_filled_trace(
                    figure,
                    [float(vertex[0]) for vertex in vertices] + [float(vertices[0][0])],
                    [float(vertex[1]) for vertex in vertices] + [float(vertices[0][1])],
                    material=material,
                    refractive_index=refractive_index,
                    fillcolor=fillcolor,
                )
                continue
            for low, high in prism_intervals(prism, slice_axis, coordinate):
                _add_material_rectangle(
                    figure,
                    low,
                    prism.z_base,
                    high - low,
                    prism.z_top - prism.z_base,
                    material=material,
                    refractive_index=refractive_index,
                    normalization=normalization,
                    colormap=colormap,
                )


def _plotly_colorscale(colormap: Any) -> list[list[Any]]:
    """Sample a Matplotlib colormap into a Plotly colorscale."""
    return [
        [fraction, to_hex(colormap(fraction), keep_alpha=False)]
        for fraction in np.linspace(0.0, 1.0, 33)
    ]


def _add_colorbar(
    figure: Any,
    normalization: Any,
    colormap: Any,
    title: str,
) -> None:
    """Add the continuous refractive-index colorbar."""
    import plotly.graph_objects as go

    figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "color": [normalization.vmin],
                "cmin": normalization.vmin,
                "cmax": normalization.vmax,
                "colorscale": _plotly_colorscale(colormap),
                "showscale": True,
                "colorbar": {
                    "title": {"text": title, "side": "right"},
                    "x": 1.02,
                    "xanchor": "left",
                },
            },
            showlegend=False,
            hoverinfo="skip",
        )
    )


def _configure_layout(
    figure: Any,
    slice_axis: str,
    coordinate: float,
    plot_min: tuple[float, float],
    plot_max: tuple[float, float],
    aspect: Literal["equal", "auto"],
) -> None:
    """Configure axes, title, legend, and aspect ratio."""
    labels = {
        "x": ("y (um)", "z (um)", "YZ", "x"),
        "y": ("x (um)", "z (um)", "XZ", "y"),
        "z": ("x (um)", "y (um)", "XY", "z"),
    }
    xlabel, ylabel, plane_name, coordinate_name = labels[slice_axis]
    xaxis: dict[str, Any] = {
        "title": xlabel,
        "range": [plot_min[0], plot_max[0]],
        "showgrid": False,
        "zeroline": False,
    }
    yaxis: dict[str, Any] = {
        "title": ylabel,
        "range": [plot_min[1], plot_max[1]],
        "showgrid": False,
        "zeroline": False,
    }
    if aspect == "equal":
        xaxis.update({"scaleanchor": "y", "scaleratio": 1})
        yaxis["constrain"] = "domain"

    figure.update_layout(
        title=(
            f"Refractive index · {plane_name} cross section at "
            f"{coordinate_name}={coordinate:.2f}"
        ),
        xaxis=xaxis,
        yaxis=yaxis,
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.2,
            "yanchor": "top",
            "itemclick": "toggle",
            "itemdoubleclick": "toggleothers",
        },
        margin={"b": 110, "r": 140},
        hovermode="closest",
    )
