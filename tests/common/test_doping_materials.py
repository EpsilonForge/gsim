"""Tests for the generic doped-material builders.

These helpers must be free of hardcoded / PDK-specific values so they can be
reused across projects.
"""

from __future__ import annotations

import pytest

from gsim.common.stack.materials import (
    DispersionModel,
    ValidityRange,
    make_doped_material,
    make_doped_materials,
)


class TestMakeDopedMaterial:
    def test_defaults(self):
        mat = make_doped_material("p_rib", 1.6e3)
        assert mat.conductivity == 1.6e3
        assert mat.permittivity == 11.9
        assert len(mat.dispersion_models) == 1
        model = mat.dispersion_models[0]
        assert isinstance(model, DispersionModel)
        assert model.type == "constant"
        assert model.permittivity == 11.9
        assert model.validity == ValidityRange(valid_frequency=(0, 200e9))
        assert model.source == "doped Si (p_rib) -- Drude sigma"

    def test_custom_values(self):
        mat = make_doped_material(
            "a", 5e4, permittivity=12.0, fmax=50e9, source="custom"
        )
        assert mat.permittivity == 12.0
        assert mat.dispersion_models[0].validity == ValidityRange(
            valid_frequency=(0, 50e9)
        )
        assert mat.dispersion_models[0].source == "custom"

    def test_resolves_to_conductive_at_rf(self):
        mat = make_doped_material("x", 2e4)
        resolved = mat.evaluate_at_wavelength(1000.0)  # RF-ish wavelength
        assert resolved.conductivity == 2e4
        assert resolved.behavior == "conductive"


class TestMakeDopedMaterials:
    def test_dict_input(self):
        mats = make_doped_materials({"pp_slab_0": 2e4, "pp_slab_1": 8e4})
        assert set(mats) == {"pp_slab_0", "pp_slab_1"}
        assert mats["pp_slab_0"].conductivity == 2e4
        assert mats["pp_slab_1"].conductivity == 8e4

    def test_two_tuple_input(self):
        mats = make_doped_materials([("n_rib", 1.6e3), ("p_rib", 1.6e3)])
        assert mats["n_rib"].permittivity == 11.9
        assert mats["n_rib"].conductivity == 1.6e3

    def test_four_tuple_input(self):
        mats = make_doped_materials([("r1", 13.0, 7e3, "doc")])
        assert mats["r1"].permittivity == 13.0
        assert mats["r1"].conductivity == 7e3
        assert mats["r1"].dispersion_models[0].source == "doc"

    def test_invalid_entry(self):
        with pytest.raises(ValueError):
            make_doped_materials([("only_name",)])

    def test_returned_materials_resolvable(self):
        mats = make_doped_materials({"a": 1e4})
        # The generated object behaves like any database material.
        for mat in mats.values():
            assert mat.evaluate_at_frequency(1e9).behavior == "conductive"
