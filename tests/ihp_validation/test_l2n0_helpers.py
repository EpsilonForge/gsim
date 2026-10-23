"""Tests for the IHP L2n0 benchmark helpers (no Palace execution)."""

from __future__ import annotations

import numpy as np
import pytest

from gsim.common.benchmarks import ihp_l2n0


def test_benchmark_source_files_are_pinned() -> None:
    assert ihp_l2n0.LAYOUT.is_file()
    assert set(ihp_l2n0.MEASURED) == {"die_x2y7", "die_x5y8"}
    for path in ihp_l2n0.MEASURED.values():
        assert path.is_file()


@pytest.mark.parametrize("label", ["die_x2y7", "die_x5y8"])
def test_measured_data_reaches_30_ghz(label: str) -> None:
    import skrf as rf

    network = rf.Network(str(ihp_l2n0.MEASURED[label]))
    assert network.f[0] <= 30e9 <= network.f[-1]
    point = network.interpolate(rf.Frequency(30, 30, 1, unit="GHz"))
    assert np.all(np.isfinite(point.s))  # ty: ignore[unresolved-attribute]


def test_component_and_stack_have_the_benchmark_ports_and_layers() -> None:
    pytest.importorskip("ihp")
    component = ihp_l2n0.load_benchmark_component()
    assert [port.name for port in component.ports] == ["P1", "P2"]
    assert (
        component.ports["P1"].layer_info.layer,
        component.ports["P1"].layer_info.datatype,
    ) == (126, 0)
    stack = ihp_l2n0.build_benchmark_stack(component)
    assert "subgnd" in stack.layers
    assert stack.layers["subgnd"].layer_type == "conductor"
    assert {"substrate", "epi", "ild", "passivation"} <= {
        dielectric["name"] for dielectric in stack.dielectrics
    }


def test_lq_extraction_is_finite_for_recorded_run() -> None:
    run = ihp_l2n0.load_recorded_run(ihp_l2n0.RUN_ROOT / "l2n0_30ghz")
    inductance, q, impedance = ihp_l2n0.lq_at(run.sparams)
    assert math_is_finite(inductance)
    assert math_is_finite(q)
    assert math_is_finite(impedance.real)
    assert math_is_finite(impedance.imag)


def math_is_finite(value: float) -> bool:
    return bool(np.isfinite(value))


@pytest.mark.parametrize("label", ["die_x2y7", "die_x5y8"])
def test_measured_low_frequency_inductance_is_nominal_2nh(label: str) -> None:
    import skrf as rf

    network = rf.Network(str(ihp_l2n0.MEASURED[label]))
    point = network.interpolate(rf.Frequency(1, 1, 1, unit="GHz"))
    s = point.s[0]  # ty: ignore[unresolved-attribute]
    z = 2.0 * ihp_l2n0.Z0_OHM * (1.0 - s[1, 0]) / s[1, 0]
    inductance_nh = z.imag / (2 * np.pi * 1e9) * 1e9
    assert 1.8 < inductance_nh < 2.1
