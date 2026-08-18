"""Malitson fused-silica model."""

from pdk_schema import Sellmeier, SellmeierTerm

from gsim.common.materials._helpers import material_card, wavelength_validity

SIO2_MALITSON = material_card(
    name="SiO2-Malitson",
    temperature_ref=293.0,
    permittivity=Sellmeier(
        validity=wavelength_validity(0.21, 6.7),
        variation=None,
        conductivity=None,
        terms=(
            SellmeierTerm(b=0.6961663, c_um=0.0684043),
            SellmeierTerm(b=0.4079426, c_um=0.1162414),
            SellmeierTerm(b=0.8974794, c_um=9.896161),
        ),
        offset=0.0,
    ),
)

__all__ = ["SIO2_MALITSON"]
