"""Built-in optical material cards."""

from collections.abc import Mapping

import gdsfactory as gf
from pdk_schema import (
    Band,
    Coord,
    Index,
    MaterialCard,
    Provenance,
    Regime,
    Sellmeier,
    SellmeierTerm,
    TableData,
    TabulatedValue,
    Validity,
)

_LI_293K_DATA = (
    (1.20, 3.5167),
    (1.22, 3.5133),
    (1.24, 3.5102),
    (1.26, 3.5072),
    (1.28, 3.5043),
    (1.30, 3.5016),
    (1.32, 3.4990),
    (1.34, 3.4965),
    (1.36, 3.4941),
    (1.38, 3.4918),
    (1.40, 3.4896),
    (1.45, 3.4845),
    (1.50, 3.4799),
    (1.55, 3.4757),
    (1.60, 3.4719),
    (1.65, 3.4684),
    (1.70, 3.4653),
    (1.80, 3.4597),
    (1.90, 3.4550),
    (2.00, 3.4510),
    (2.25, 3.4431),
    (2.50, 3.4375),
    (2.75, 3.4334),
    (3.00, 3.4302),
    (4.00, 3.4229),
    (5.00, 3.4195),
    (6.00, 3.4177),
    (7.00, 3.4165),
    (8.00, 3.4158),
    (9.00, 3.4153),
    (10.0, 3.4150),
    (11.0, 3.4147),
    (12.0, 3.4145),
    (13.0, 3.4144),
    (14.0, 3.4142),
)


def _validity(minimum_um: float, maximum_um: float) -> Validity:
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


def _provenance(label: str) -> Provenance:
    """Return the minimum required literature provenance."""
    return Provenance(
        source="literature",
        label=label,
        maturity="empirical",
        citations=[],
        comment=None,
        url=None,
        data_url=None,
        info={},
    )


def _card(
    name: str,
    permittivity: Index | Sellmeier,
    temperature_ref: float,
) -> MaterialCard:
    """Build an optical material card."""
    return MaterialCard(
        name=name,
        optical=Regime(
            temperature_ref=temperature_ref,
            provenance=_provenance(name),
            permittivity=permittivity,
            conductivity=None,
            permeability=None,
            perturbations=[],
            info={},
        ),
        rf=None,
        info={},
    )


_SI_SALZBERG = _card(
    name="Si-Salzberg",
    temperature_ref=299.15,
    permittivity=Sellmeier(
        validity=_validity(1.357, 11.04),
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

_SI_LI_293K = _card(
    name="Si-Li-293K",
    temperature_ref=293.0,
    permittivity=Index(
        validity=_validity(1.2, 14.0),
        variation=None,
        conductivity=None,
        n=TabulatedValue(
            unit="",
            data=TableData(
                dims=("wavelength",),
                coords={
                    "wavelength": Coord(
                        values=[row[0] for row in _LI_293K_DATA],
                        unit="um",
                    )
                },
                values=[row[1] for row in _LI_293K_DATA],
                attrs={},
                interp="linear",
            ),
        ),
        k=None,
    ),
)

_SIO2_MALITSON = _card(
    name="SiO2-Malitson",
    temperature_ref=293.0,
    permittivity=Sellmeier(
        validity=_validity(0.21, 6.7),
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

GSIM_MATERIAL_CARDS: dict[str, MaterialCard] = {
    "Si": _SI_SALZBERG.model_copy(update={"name": "Si"}),
    "Si-Salzberg": _SI_SALZBERG,
    "Si-Li-293K": _SI_LI_293K,
    "SiO2": _SIO2_MALITSON.model_copy(update={"name": "SiO2"}),
    "SiO2-Malitson": _SIO2_MALITSON,
}


def get_material_card(
    material_name: str,
    project_material_cards: Mapping[str, MaterialCard] | None = None,
) -> MaterialCard:
    """Return an active-PDK card when present, otherwise a built-in card."""
    if project_material_cards is None:
        active_pdk = gf.get_active_pdk()
        project_material_cards = getattr(active_pdk, "material_cards", None)
    if project_material_cards and material_name in project_material_cards:
        return project_material_cards[material_name]
    return GSIM_MATERIAL_CARDS[material_name]


__all__ = ["GSIM_MATERIAL_CARDS", "get_material_card"]
