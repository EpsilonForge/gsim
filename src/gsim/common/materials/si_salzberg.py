"""Salzberg and Villa crystalline silicon model."""

from pdk_schema import Sellmeier, SellmeierTerm

from gsim.common.materials._helpers import material_card, wavelength_validity

SI_SALZBERG = material_card(
    name="Si-Salzberg",
    temperature_ref=299.15,
    permittivity=Sellmeier(
        validity=wavelength_validity(1.357, 11.04),
        variation=None,
        conductivity=None,
        terms=(
            SellmeierTerm(b=10.6684293, c_um=0.301516485),
            SellmeierTerm(b=0.0030434748, c_um=1.13475115),
            SellmeierTerm(b=1.54133408, c_um=1104.0),
        ),
        offset=0.0,
    ),
)

__all__ = ["SI_SALZBERG"]
