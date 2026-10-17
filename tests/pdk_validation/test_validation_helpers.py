"""Unit tests for the MEEP validation harness (:mod:`_validation`).

These exercise the pure post-processing maths (de-embedding, normalisation,
agreement metrics) and the simulation *configuration*; none of them run MEEP
and none require ``pymeep``.  Tests that need a PDK skip cleanly when it is
not installed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests.pdk_validation import _validation as validation


class _Port:
    def __init__(self, name: str) -> None:
        self.name = name


class _Component:
    def __init__(self, *names: str) -> None:
        self.ports = [_Port(name) for name in names]


class _Result:
    """Minimal stand-in for SParameterResult."""

    def __init__(self, wavelengths, s_params, debug_info=None):
        self.wavelengths = list(wavelengths)
        self.s_params = {k: list(v) for k, v in s_params.items()}
        self.debug_info = debug_info or {}


# ---------------------------------------------------------------------------
# Facts derived from device sizes in square wavelengths
# ---------------------------------------------------------------------------


def test_lambda_squared_reference_value():
    lam = validation.VALIDATION_WAVELENGTH_UM
    assert lam == 1.55
    assert math.isclose(lam**2, 2.4025, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Port monitor offsets
# ---------------------------------------------------------------------------


def test_monitor_distances_source_gets_extra_offset():
    component = _Component("o1", "o2", "o3")
    distances = validation.monitor_distances(
        component,
        source_port="o1",
        source_port_offset=0.1,
        distance_source_to_monitors=0.2,
    )
    assert distances == {
        "o1": pytest.approx(0.3),
        "o2": pytest.approx(0.1),
        "o3": pytest.approx(0.1),
    }


def test_monitor_distances_defaults_to_first_port():
    component = _Component("o1", "o2")
    distances = validation.monitor_distances(component)
    assert distances["o1"] > distances["o2"]


# ---------------------------------------------------------------------------
# De-embedding
# ---------------------------------------------------------------------------


def test_s_param_indices():
    assert validation.s_param_indices("S11") == (0, 0)
    assert validation.s_param_indices("S41") == (3, 0)
    with pytest.raises(ValueError):
        validation.s_param_indices("x11")


def test_deembed_is_identity_without_offsets():
    result = _Result(
        [1.53, 1.55, 1.57],
        {"S11": [0.1 + 0j, 0.2 + 0j, 0.3 + 0j], "S21": [1 + 0j, 1 + 0j, 1 + 0j]},
        {"eigenmode_info": {"o1": {"n_eff": [2.4, 2.4, 2.4]}}},
    )
    out = validation.deembed_s_params(
        result, ["o1", "o2"], {"o1": 0.0, "o2": 0.0}, source_port="o1"
    )
    np.testing.assert_allclose(out["S11"], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(out["S21"], [1.0, 1.0, 1.0])


def test_deembed_applies_expected_phase():
    wavelengths = np.array([1.55])
    result = _Result(
        wavelengths,
        {"S11": [1 + 0j]},
        {"eigenmode_info": {"o1": {"n_eff": [2.0]}}},
    )
    distances = {"o1": 0.1, "o2": 0.1}
    out = validation.deembed_s_params(result, ["o1", "o2"], distances, source_port="o1")
    k = 2 * math.pi / 1.55
    expected = np.exp(1j * k * (2.0 * 0.1 + 2.0 * 0.1))
    np.testing.assert_allclose(out["S11"], [expected], rtol=1e-12)


def test_deembed_sign_flips_phase():
    result = _Result(
        [1.55], {"S11": [1 + 0j]}, {"eigenmode_info": {"o1": {"n_eff": [2.0]}}}
    )
    distances = {"o1": 0.1, "o2": 0.1}
    plus = validation.deembed_s_params(result, ["o1", "o2"], distances, sign=1.0)[
        "S11"
    ][0]
    minus = validation.deembed_s_params(result, ["o1", "o2"], distances, sign=-1.0)[
        "S11"
    ][0]
    np.testing.assert_allclose(minus, np.conj(plus), rtol=1e-12)


def test_deembed_only_touches_source_row():
    result = _Result(
        [1.55],
        {"S12": [0.5 + 0j]},  # excited from o2 -> not the configured source
        {"eigenmode_info": {"o1": {"n_eff": [2.0]}, "o2": {"n_eff": [2.0]}}},
    )
    out = validation.deembed_s_params(
        result, ["o1", "o2"], {"o1": 0.1, "o2": 0.1}, source_port="o1"
    )
    np.testing.assert_allclose(out["S12"], [0.5 + 0j])


def test_deembed_falls_back_when_no_n_eff():
    result = _Result([1.55], {"S11": [1 + 0j]}, {})
    out = validation.deembed_s_params(
        result, ["o1", "o2"], {"o1": 0.1, "o2": 0.0}, source_port="o1"
    )
    k = 2 * math.pi / 1.55
    # S11 has both the receive (o1) and source (o1) monitors offset by 0.1 um.
    expected = np.exp(1j * k * validation.FALLBACK_N_EFF * (0.1 + 0.1))
    np.testing.assert_allclose(out["S11"], [expected], rtol=1e-12)


# ---------------------------------------------------------------------------
# Metrics and normalisation
# ---------------------------------------------------------------------------


def test_to_db():
    assert math.isclose(float(validation.to_db([1.0])[0]), 0.0, abs_tol=1e-9)
    assert math.isclose(float(validation.to_db([0.1])[0]), -20.0, rel_tol=1e-6)


def test_normalize_measured_spectrum_max():
    power = np.array([-30.0, -20.0, -10.0])
    out = validation.normalize_measured_spectrum(power, method="max")
    np.testing.assert_allclose(out, [-20.0, -10.0, 0.0])


def test_normalize_measured_spectrum_percentile_is_robust():
    rng = np.random.default_rng(0)
    power = -30.0 + rng.normal(0.0, 0.1, 200)
    power[0] = 5.0  # single launch spike
    robust = validation.normalize_measured_spectrum(
        power, method="percentile", percentile=99.0
    )
    naive = validation.normalize_measured_spectrum(power, method="max")
    # The percentile reference ignores the spike, so the baseline sits near 0;
    # the max reference is dragged down to the spike.
    assert abs(np.median(robust)) < 1.0
    assert np.median(naive) < -30.0


def test_agreement_identical_curves():
    wl = np.linspace(1.5, 1.6, 51)
    db = -3 * np.ones_like(wl)
    metrics = validation.agreement_db(wl, db, wl.copy(), db.copy())
    assert metrics is not None
    assert metrics["rms_db"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["max_abs_db"] == pytest.approx(0.0, abs=1e-9)


def test_agreement_constant_bias():
    wl = np.linspace(1.5, 1.6, 51)
    metrics = validation.agreement_db(wl, np.zeros_like(wl), wl, np.full_like(wl, 2.0))
    assert metrics is not None
    assert metrics["bias_db"] == pytest.approx(-2.0, abs=1e-9)
    assert metrics["rms_db"] == pytest.approx(2.0, abs=1e-9)


def test_agreement_no_overlap_returns_none():
    metrics = validation.agreement_db(
        np.array([1.5, 1.55, 1.6]), np.zeros(3), np.array([1.8, 1.9, 2.0]), np.zeros(3)
    )
    assert metrics is None


def test_agreement_normalize_removes_offset():
    wl = np.linspace(1.5, 1.6, 51)
    sim = -3 * np.ones_like(wl)  # flat
    ref = 7 * np.ones_like(wl)  # flat, offset by 10 dB
    metrics = validation.agreement_db(wl, sim, wl, ref, normalize=True)
    assert metrics is not None
    assert metrics["rms_db"] == pytest.approx(0.0, abs=1e-9)


def test_power_conservation():
    total = validation.power_conservation({"S21": [0.6], "S31": [0.8]})
    np.testing.assert_allclose(total, [1.0])


# ---------------------------------------------------------------------------
# Simulation configuration (no run)
# ---------------------------------------------------------------------------


def test_build_validation_simulation_defaults():
    pytest.importorskip("gsim")
    component = _Component("o1", "o2")
    sim = validation.build_validation_simulation(component, stack=object())
    assert sim.solver.resolution == validation.VALIDATION_RESOLUTION == 20
    assert sim.num_freqs == validation.VALIDATION_NUM_FREQS == 101
    assert sim.solver.stopping == "energy_decay"
    assert sim.solver.stopping_threshold == pytest.approx(
        validation.VALIDATION_DECAY_BY
    )
    assert sim.solver.stopping_dt == pytest.approx(validation.VALIDATION_DECAY_DT)
    assert sim.source.wavelength_span == pytest.approx(validation.VALIDATION_SPAN_UM)
    assert sim.monitors == ["o1", "o2"]
    assert sim.source.port == "o1"


def test_build_validation_simulation_2d_plane():
    pytest.importorskip("gsim")
    component = _Component("o1", "o2")
    sim = validation.build_validation_simulation(
        component, dimension="2d", stack=object()
    )
    assert sim.solver.mode == "2d"
    assert sim.solver.z_cut == "auto"


# ---------------------------------------------------------------------------
# Matched-geometry devices (need gdsfactory, no MEEP)
# ---------------------------------------------------------------------------


def test_dbr_two_port_has_both_ports():
    pytest.importorskip("gdsfactory")
    import gdsfactory as gf

    try:
        gf.gpdk.PDK.activate()
    except Exception:
        pytest.skip("no generic PDK available")

    dbr = validation.dbr_two_port(w1=0.5, w2=0.6, n=20)
    names = [port.name for port in dbr.ports]
    assert sorted(names) == ["o1", "o2"]
    o1, o2 = dbr.ports["o1"], dbr.ports["o2"]
    assert o1.center[0] < o2.center[0]
    assert dbr.xsize > 6.0


@pytest.mark.parametrize(
    "name", ["straight", "coupler", "ring_coupling", "mzi", "ring", "bragg"]
)
def test_case_registry(name):
    pytest.importorskip("ubcpdk")
    case = validation.get_case(name)
    assert case.name == name
    assert case.resolution >= 20
    assert case.num_freqs >= 101


def test_mzi_case_uses_2d_for_size():
    pytest.importorskip("ubcpdk")
    assert validation.get_case("mzi").dimension == "2d"
