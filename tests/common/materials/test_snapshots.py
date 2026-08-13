from __future__ import annotations

import pytest
from pdk_schema import MaterialCard

from gsim.common.materials import (
    GSIM_MATERIAL_CARDS,
    MaterialModelError,
    MaterialNotFoundError,
    WavelengthOutOfRangeError,
    resolve_material_snapshot,
)


@pytest.mark.parametrize(
    ("material_name", "expected_index"),
    [
        ("Si-Salzberg", 3.477723756),
        ("Si-Li-293K", 3.4757),
        ("SiO2-Malitson", 1.444023622),
    ],
)
def test_material_snapshots_at_telecom_wavelength(
    material_name: str,
    expected_index: float,
) -> None:
    snapshot = resolve_material_snapshot(material_name, 1.55, {})

    assert snapshot.refractive_index == pytest.approx(expected_index, abs=1e-9)
    assert snapshot.extinction_coefficient == 0
    assert snapshot.source == "gsim"


def test_tabulated_material_interpolates() -> None:
    snapshot = resolve_material_snapshot("Si-Li-293K", 1.525, {})

    assert snapshot.refractive_index == pytest.approx(3.4778)


def test_project_card_overrides_fallback_and_missing_card_uses_fallback() -> None:
    project_si = GSIM_MATERIAL_CARDS["Si-Li-293K"].model_copy(update={"name": "Si"})

    silicon = resolve_material_snapshot("Si", 1.55, {"Si": project_si})
    silica = resolve_material_snapshot("SiO2", 1.55, {"Si": project_si})

    assert silicon.source == "project"
    assert silicon.refractive_index == pytest.approx(3.4757)
    assert silica.source == "gsim"
    assert silica.refractive_index == pytest.approx(1.444023622)


def test_invalid_project_card_does_not_silently_fallback() -> None:
    project_si = MaterialCard(name="Si", optical=None, rf=None, info={})

    with pytest.raises(MaterialModelError, match="no optical permittivity"):
        resolve_material_snapshot("Si", 1.55, {"Si": project_si})


def test_wavelength_validation_is_strict() -> None:
    with pytest.raises(WavelengthOutOfRangeError, match="valid from"):
        resolve_material_snapshot("Si-Salzberg", 1.0, {})
    with pytest.raises(WavelengthOutOfRangeError, match="finite positive"):
        resolve_material_snapshot("Si", 0, {})


def test_unknown_material_error_explains_how_to_supply_card() -> None:
    with pytest.raises(MaterialNotFoundError, match="Attach the card"):
        resolve_material_snapshot("GaAs", 1.55, {})
