"""Helpers shared by the built-in material cards."""

from pdk_schema import (
    Band,
    Index,
    MaterialCard,
    Provenance,
    Regime,
    Sellmeier,
    Validity,
)


def wavelength_validity(minimum_um: float, maximum_um: float) -> Validity:
    """Return a strict wavelength validity range in micrometers."""
    return Validity(
        at=None,
        over={
            "wavelength": Band(
                min=minimum_um,
                max=maximum_um,
                unit="um",
                label=None,
            )
        },
        on_out_of_range="raise",
    )


def material_card(
    name: str,
    permittivity: Index | Sellmeier,
    temperature_ref: float,
) -> MaterialCard:
    """Build a compact optical material card."""
    provenance = Provenance(
        source="literature",
        label=name,
        maturity="empirical",
        citations=[],
        comment=None,
        url=None,
        data_url=None,
        info={},
    )
    return MaterialCard(
        name=name,
        optical=Regime(
            temperature_ref=temperature_ref,
            provenance=provenance,
            permittivity=permittivity,
            conductivity=None,
            permeability=None,
            perturbations=[],
            info={},
        ),
        rf=None,
        info={},
    )
