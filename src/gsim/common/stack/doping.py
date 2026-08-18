"""Doping-profile construction for semiconductor cross-sections.

This module provides solver-agnostic helpers to build contiguous (gapless)
doping regions on both sides of a rib/waveguide and to generate the
corresponding ``Layer`` specs (``gsim.common.stack.extractor``) and
``MaterialProperties`` (``gsim.common.stack.materials``).

All geometry-specific values (layer tuples, naming prefixes, doping widths,
conductivities, z-extents) are caller-supplied — nothing is hardcoded here so
the helpers are reusable across PDKs and processes.

Example:
-------
    >>> import gdsfactory as gf
    >>> from gsim.common.stack.doping import make_doping_profile
    >>> comp = gf.Component()
    >>> result = make_doping_profile(
    ...     comp,
    ...     length=10.0,
    ...     rib_center_y=-20.0,
    ...     rib_width=0.4,
    ...     profile={
    ...         "upper": [(2.0, 2e4), (2.0, 8e4)],
    ...         "lower": [(2.0, 2e4), (2.0, 8e4)],
    ...     },
    ...     sides={
    ...         "upper": {"base_layer": (23, 0), "name_prefix": "pp_slab_", "sign": 1},
    ...         "lower": {
    ...             "base_layer": (24, 0),
    ...             "name_prefix": "npp_slab_",
    ...             "sign": -1,
    ...         },
    ...     },
    ...     zmin=0.0,
    ...     zmax=0.09,
    ... )
    >>> result["layer_specs"]
    >>> result["materials"]
    >>> result["centres"]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import gdsfactory as gf

from gsim.common.stack.materials import make_doped_materials

if TYPE_CHECKING:
    from gsim.common.stack.extractor import Layer

_SideConfig = dict[str, dict[str, Any]]


def make_doping_profile(
    comp: gf.Component,
    *,
    length: float,
    rib_center_y: float,
    rib_width: float,
    profile: dict[str, list[tuple[float, float]]],
    sides: _SideConfig,
    zmin: float,
    zmax: float,
    permittivity: float = 11.9,
    fmax: float = 200e9,
    mesh_resolution: str | float = "fine",
) -> dict[str, dict[str, Any]]:
    """Add contiguous doping regions beside a rib and build layer/material specs.

    For each side (e.g. ``"upper"`` / ``"lower"``) the regions listed in
    *profile* are placed as adjacent rectangles starting at the rib edge and
    extending outward, so the doping is contiguous with no gaps.  Each region
    ``i`` on a side gets:

    - a gdsfactory rectangle of size ``(length, width)`` on the GDS layer
      ``(base_layer[0], base_layer[1] + i)``,
    - a ``Layer`` spec named ``"{name_prefix}{i}"``,
    - a ``MaterialProperties`` entry with the region's Drude conductivity.

    Args:
        comp: gdsfactory component the rectangles are added to.
        length: Rectangle length along the propagation direction (um).
        rib_center_y: Y coordinate of the rib centre (um).
        rib_width: Rib width (um); regions start at the rib edges.
        profile: Per-side region list ``{side: [(width_um, sigma_S_per_m), ...]}``.
        sides: Per-side configuration: each value is a dict with keys
            ``base_layer`` (``(layer, datatype)`` tuple for the first region),
            ``name_prefix`` (region-name prefix) and ``sign`` (+1 extends in
            +y, -1 in -y).
        zmin: Bottom z of the doping regions (um).
        zmax: Top z of the doping regions (um).
        permittivity: Relative permittivity shared by all regions (e.g. 11.9).
        fmax: Upper frequency of the dispersion-model validity range (Hz).
        mesh_resolution: Mesh resolution assigned to the generated ``Layer``.

    Returns:
        Dict with keys ``layer_specs`` (``{name: Layer}``), ``materials``
        (``{name: MaterialProperties}``) and ``centres``
        (``{side: [y_centre, ...]}``).
    """
    from gsim.common.stack.extractor import Layer

    result: dict[str, dict[str, Any]] = {
        "layer_specs": {},
        "materials": {},
        "centres": {},
    }
    layer_specs = cast("dict[str, Layer]", result["layer_specs"])
    materials: dict[str, Any] = result["materials"]
    centres: dict[str, list[float]] = result["centres"]

    for side, cfg in sides.items():
        regions = profile.get(side, [])
        sign = cfg["sign"]
        base_layer = tuple(cfg["base_layer"])
        prefix = cfg["name_prefix"]

        pos = rib_center_y + sign * rib_width / 2  # start at rib edge
        side_centres: list[float] = []
        side_specs: dict[str, tuple[Any, float]] = {}

        for i, (width, sigma) in enumerate(regions):
            name = f"{prefix}{i}"
            gds_layer = (base_layer[0], base_layer[1] + i)
            centre = pos + sign * width / 2

            rect = comp << gf.c.rectangle((length, width), layer=gds_layer)
            rect.y = centre
            side_centres.append(centre)
            side_specs[name] = (gds_layer, sigma)
            pos += sign * width

        centres[side] = side_centres
        if not side_specs:
            continue

        layer_specs.update(
            {
                name: Layer(
                    name=name,
                    gds_layer=gds_layer,
                    zmin=zmin,
                    zmax=zmax,
                    thickness=zmax - zmin,
                    material=name,
                    layer_type="dielectric",
                    mesh_resolution=mesh_resolution,
                )
                for name, (gds_layer, _sigma) in side_specs.items()
            }
        )
        materials.update(
            make_doped_materials(
                [(name, sigma) for name, (_gds, sigma) in side_specs.items()],
                permittivity=permittivity,
                fmax=fmax,
                source_prefix="doped Si",
            )
        )

    return result


__all__ = ["make_doping_profile"]
