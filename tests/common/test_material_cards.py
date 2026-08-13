from math import sqrt

import pytest
from pdk_schema import Index, MaterialCard, Sellmeier, TabulatedValue

from gsim.common.materials import GSIM_MATERIAL_CARDS, get_material_card


def _sellmeier_index(model: Sellmeier, wavelength_um: float) -> float:
    wavelength_squared = wavelength_um**2
    index_squared = (
        1.0
        + model.offset
        + sum(
            term.b * wavelength_squared / (wavelength_squared - term.c_um**2)
            for term in model.terms
        )
    )
    return sqrt(index_squared)


def test_material_card_names() -> None:
    assert set(GSIM_MATERIAL_CARDS) == {
        "Si",
        "Si-Salzberg",
        "Si-Li-293K",
        "SiO2",
        "SiO2-Malitson",
    }
    assert all(name == card.name for name, card in GSIM_MATERIAL_CARDS.items())


def test_silicon_fallback_uses_salzberg() -> None:
    fallback = GSIM_MATERIAL_CARDS["Si"]
    named = GSIM_MATERIAL_CARDS["Si-Salzberg"]

    assert fallback.optical == named.optical
    assert fallback.optical is not None
    assert isinstance(fallback.optical.permittivity, Sellmeier)
    assert _sellmeier_index(fallback.optical.permittivity, 1.55) == pytest.approx(
        3.477723756,
        abs=1e-9,
    )


def test_li_293k_data() -> None:
    card = GSIM_MATERIAL_CARDS["Si-Li-293K"]

    assert card.optical is not None
    assert isinstance(card.optical.permittivity, Index)
    assert isinstance(card.optical.permittivity.n, TabulatedValue)
    table = card.optical.permittivity.n.data
    telecom_row = table.coords["wavelength"].values.index(1.55)
    assert len(table.values) == 35
    assert table.values[telecom_row] == 3.4757


def test_silicon_dioxide_fallback_uses_malitson() -> None:
    fallback = GSIM_MATERIAL_CARDS["SiO2"]
    named = GSIM_MATERIAL_CARDS["SiO2-Malitson"]

    assert fallback.optical == named.optical
    assert fallback.optical is not None
    assert isinstance(fallback.optical.permittivity, Sellmeier)
    assert _sellmeier_index(fallback.optical.permittivity, 1.55) == pytest.approx(
        1.444023622,
        abs=1e-9,
    )


def test_project_material_card_takes_precedence() -> None:
    project_card = MaterialCard(name="Si", optical=None, rf=None, info={})

    assert get_material_card("Si", {"Si": project_card}) is project_card
    assert (
        get_material_card("SiO2", {"Si": project_card}) is GSIM_MATERIAL_CARDS["SiO2"]
    )


def test_cards_have_no_external_data_references() -> None:
    for card in GSIM_MATERIAL_CARDS.values():
        assert card.optical is not None
        assert card.optical.provenance.url is None
        assert card.optical.provenance.data_url is None
