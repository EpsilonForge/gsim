"""Physical material handling at Meep absorbing boundaries."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gsim.common.stack import LayerStack
    from gsim.meep.models.config import DomainConfig

_BOUNDARY_TOLERANCE = 1e-9


def extend_dielectrics_into_pml(
    stack: LayerStack,
    domain: DomainConfig,
) -> LayerStack:
    """Extend boundary-adjacent background dielectrics through the Z PML.

    Dielectric slabs are already infinite along X and Y in the Meep runner, so
    only the active Z boundaries require explicit extension. Patterned layers
    remain unchanged.
    """
    if not domain.extend_into_pml or domain.dpml == 0 or domain.z_bounds is None:
        return stack

    z_low, z_high = domain.z_bounds
    extended_stack = stack.model_copy(deep=True)
    for dielectric in extended_stack.dielectrics:
        if math.isclose(
            dielectric["zmin"],
            z_low,
            rel_tol=0.0,
            abs_tol=_BOUNDARY_TOLERANCE,
        ):
            dielectric["zmin"] = z_low - domain.dpml
        if math.isclose(
            dielectric["zmax"],
            z_high,
            rel_tol=0.0,
            abs_tol=_BOUNDARY_TOLERANCE,
        ):
            dielectric["zmax"] = z_high + domain.dpml
    return extended_stack
