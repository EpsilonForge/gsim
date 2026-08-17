"""Luke et al. silicon-nitride model."""

from pdk_schema import Sellmeier, SellmeierTerm

from gsim.common.materials._helpers import material_card, wavelength_validity

SIN_LUKE = material_card(
    name="SiN-Luke",
    temperature_ref=None,
    permittivity=Sellmeier(
        validity=wavelength_validity(0.310, 5.504),
        variation=None,
        conductivity=None,
        terms=(
            SellmeierTerm(b=3.0249, c_um=0.1353406),
            SellmeierTerm(b=40314.0, c_um=1239.842),
        ),
        offset=0.0,
    ),
)

__all__ = ["SIN_LUKE"]
