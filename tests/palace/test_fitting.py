"""Tests for the reusable RLC fitting workflow (gsim.palace.fitting)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gsim.palace.circuit import load_circuit_synthesis
from gsim.palace.fitting import (
    RLCFit,
    VectorFit,
    differential_impedance,
    fit_rlc,
    initial_guess_rlc,
)

R_TRUE, L_TRUE, C_TRUE = 3.5, 110e-12, 8e-15


@pytest.fixture
def rlc_data():
    """Exact one-pole RLC impedance over a 10-200 GHz band."""
    f = np.linspace(10e9, 200e9, 200)
    f0 = 1.0 / (2.0 * np.pi * np.sqrt(L_TRUE * C_TRUE))
    model = RLCFit(
        R=R_TRUE,
        L=L_TRUE,
        C=C_TRUE,
        f0=f0,
        Q=2.0 * np.pi * f0 * L_TRUE / R_TRUE,
        rms_error=0.0,
    )
    return f, model.z(f), model


def test_differential_impedance_matches_manual_formula():
    rng = np.random.default_rng(42)
    z = rng.normal(size=(7, 3, 3)) + 1j * rng.normal(size=(7, 3, 3))
    expected = z[:, 0, 0] - z[:, 0, 1] - z[:, 1, 0] + z[:, 1, 1]
    np.testing.assert_allclose(differential_impedance(z), expected)


def test_differential_impedance_rejects_invalid_shape():
    with pytest.raises(ValueError, match="nf, N, N"):
        differential_impedance(np.zeros((5,)))


def test_differential_impedance_single_port_is_driving_point():
    """A one-port network is already differential: return Z11 directly."""
    rng = np.random.default_rng(7)
    z = rng.normal(size=(9, 1, 1)) + 1j * rng.normal(size=(9, 1, 1))
    np.testing.assert_allclose(differential_impedance(z), z[:, 0, 0])


def test_initial_guess_rlc_finds_peak_and_low_freq_resistance(rlc_data):
    f, z, _ = rlc_data
    f0, _, r = initial_guess_rlc(f, z)
    assert f0 == pytest.approx(f[int(np.argmax(np.abs(z)))])
    assert r == pytest.approx(R_TRUE, rel=0.5)


@pytest.mark.parametrize("solver", ["scipy", "jax"])
def test_fit_rlc_recovers_synthetic_rlc(rlc_data, solver):
    """Both solvers recover the true (R, L, C, f0, Q) from exact data."""
    if solver == "jax":
        pytest.importorskip("jax")
    f, z, model = rlc_data
    fit = fit_rlc(f, z, solver=solver)
    assert isinstance(fit, RLCFit)
    if solver == "jax":
        # Log-space optimization with a fixed iteration budget converges to
        # ~1e-2 relative accuracy; scipy is essentially exact.
        assert pytest.approx(R_TRUE, rel=2e-2) == fit.R
        assert pytest.approx(L_TRUE, rel=5e-3) == fit.L
        assert pytest.approx(C_TRUE, rel=5e-3) == fit.C
        assert fit.f0 == pytest.approx(model.f0, rel=2e-3)
        assert pytest.approx(model.Q, rel=2e-2) == fit.Q
        # Near the resonance |Z| ~ 3.7 kOhm: ~2.6 % relative residual there.
        np.testing.assert_allclose(fit.z(f), z, rtol=3e-2)
    else:
        assert pytest.approx(R_TRUE, rel=2e-3) == fit.R
        assert pytest.approx(L_TRUE, rel=1e-3) == fit.L
        assert pytest.approx(C_TRUE, rel=1e-3) == fit.C
        assert fit.f0 == pytest.approx(model.f0, rel=2e-3)
        assert pytest.approx(model.Q, rel=3e-3) == fit.Q
        assert fit.rms_error == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(fit.z(f), z, rtol=1e-4)


def test_fit_rlc_auto_solver_returns_valid_fit():
    """solver='auto' works regardless of whether the optional JAX deps exist."""
    f = np.linspace(10e9, 100e9, 50)
    model = RLCFit(
        R=R_TRUE,
        L=L_TRUE,
        C=C_TRUE,
        f0=1.0 / (2.0 * np.pi * np.sqrt(L_TRUE * C_TRUE)),
        Q=2.0 * np.pi * 1e9 * L_TRUE / R_TRUE,
        rms_error=0.0,
    )
    fit = fit_rlc(f, model.z(f), solver="auto")
    assert isinstance(fit, RLCFit)
    assert pytest.approx(L_TRUE, rel=1e-2) == fit.L


def test_rlc_fit_repr_and_dict(rlc_data):
    f, z, _ = rlc_data
    fit = fit_rlc(f, z, solver="scipy")
    assert isinstance(fit, RLCFit)
    assert "RLCFit" in repr(fit)
    d = fit.to_dict()
    assert set(d) == {"R", "L", "C", "f0", "Q", "rms_error"}


def test_circuit_fit_rlc_on_real_data():
    """End-to-end one-pole fit on the committed circuit-synthesis data for the
    notebook inductor (nbs/data/inductor/circuit_synthesis, Palace v0.18.0 run).

    Note: an exact *synthetic* two-node circuit cannot reproduce the one-pole
    model — in Palace's pencil Y(w) = L^-1/(iw) + R^-1 + iw*C the R^-1 and
    L^-1 matrices are parallel-admittance contributions, so a series R+L
    branch (whose admittance 1/(R + iwL) has a frequency-dependent mix) is not
    expressible as constant R^-1/L^-1 entries. Series behavior only emerges
    from the network, which is exactly what this real-data test exercises.
    """
    data_dir = (
        Path(__file__).resolve().parents[2] / "nbs/data/inductor/circuit_synthesis"
    )
    if not (data_dir / "rom-Linv-re.csv").exists():  # pragma: no cover - repo layout
        pytest.skip("circuit-synthesis data not present")

    circuit = load_circuit_synthesis(data_dir, port_map={1: "P1", 2: "P2"})
    # Evaluation grid: the sweep frequencies from port-S.csv.
    port_s = pd.read_csv(data_dir / "port-S.csv")
    f_hz = port_s.iloc[:, 0].to_numpy() * 1e9

    fit = circuit.fit_rlc(f_hz, model="rlc1p", solver="scipy")
    assert isinstance(fit, RLCFit)
    # Physically meaningful one-pole parameters within the validated ranges
    # (fit values are solver-dependent to ~5 %; the resonance is not).
    assert 2.0 < fit.R < 6.0
    assert 80e-12 < fit.L < 160e-12
    assert 4e-15 < fit.C < 12e-15
    assert 160e9 < fit.f0 < 180e9
    assert 20 < fit.Q < 60
    assert fit.rms_error < 150.0


# ---------------------------------------------------------------------------
# scikit-rf vector fitting (model="vector_fit")
# ---------------------------------------------------------------------------


def test_fit_rlc_vector_fit_on_real_circuit_data():
    """Multi-pole vector fit of the exported circuit's S-parameters.

    Skips when skrf is unavailable. Uses a modest pole count so the fit is
    fast and deterministic; the exported circuit reproduces the FEM response
    to ~5e-4 in S, so the rational model must be far below the 0.01 target.
    """
    pytest.importorskip("skrf")
    data_dir = (
        Path(__file__).resolve().parents[2] / "nbs/data/inductor/circuit_synthesis"
    )
    if not (data_dir / "rom-Linv-re.csv").exists():  # pragma: no cover
        pytest.skip("circuit-synthesis data not present")

    circuit = load_circuit_synthesis(data_dir, port_map={1: "P1", 2: "P2"})
    f_hz = pd.read_csv(data_dir / "port-S.csv").iloc[:, 0].to_numpy() * 1e9

    fit = circuit.fit_rlc(f_hz, model="vector_fit", n_poles_real=4)
    assert isinstance(fit, VectorFit)
    assert "VectorFit" in repr(fit)
    # skrf may prune non-contributing poles during fitting.
    assert 0 < fit.n_poles <= 2 * 4

    # Stability and passivity of the fitted rational model (passive lossy
    # device referenced to 50 Ohm).
    assert fit.is_stable
    assert fit.is_passive()

    # Rational response quality: model S ~ exported-circuit S ~ FEM S.
    s_model = fit.s()
    s_target = circuit.s_parameters(f_hz)
    max_rel = np.max(np.abs(s_model - s_target) / (np.abs(s_target) + 1e-6))
    assert max_rel < 2e-3, f"vector fit S error {max_rel:.2e}"

    # Impedance evaluation round-trip.
    z_model = fit.z()
    z_target = differential_impedance(circuit.port_impedance(f_hz))
    rel = np.max(np.abs(differential_impedance(z_model) - z_target) / np.abs(z_target))
    assert rel < 5e-3, f"vector fit Z error {rel:.2e}"


def test_fit_rlc_vector_fit_detects_nonpassive_model():
    """A deliberately nonpassive response is detected.

    The issue's validation asks to detect deliberately nonpassive models.
    A negative-resistance inductor (Z = -R0 + jwL) yields |S11| > 1, so the
    fitted rational model must fail the passivity test. Enforcement is also
    attempted and must NOT pretend to succeed: skrf correctly refuses to
    manufacture passivity from a fundamentally active model (the DC point
    is not passive and the violations are unbounded).
    """
    pytest.importorskip("skrf")
    f = np.linspace(10e9, 200e9, 120)
    w = 2 * np.pi * f
    z_active = -0.5 + 1j * w * 100e-12  # active: Re(Z) < 0 -> |S11| > 1

    fit = fit_rlc(f, z_active, model="vector_fit", n_poles_real=2)
    assert isinstance(fit, VectorFit)
    assert not fit.is_passive(), "active impedance must fail the passivity test"
    assert fit.passivity_test().size > 0, "passivity violations must be reported"

    with pytest.warns((UserWarning, RuntimeWarning)):
        fit.passivity_enforce()
    assert not fit.is_passive(), "enforcement must not claim success on active data"


def test_fit_rlc_vector_fit_requires_skrf(monkeypatch):
    """Missing skrf yields an informative ImportError."""

    import builtins

    real_import = builtins.__import__

    def _no_skrf(name, *args, **kwargs):
        if name == "skrf" or name.startswith("skrf."):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_skrf)
    f = np.linspace(10e9, 100e9, 20)
    with pytest.raises(ImportError, match="scikit-rf is required"):
        fit_rlc(f, np.ones(len(f)) * 50.0, model="vector_fit")


def test_fit_rlc_dispatch_errors():
    """Ambiguous or unknown model inputs raise clear errors."""
    f = np.linspace(10e9, 100e9, 10)
    z = np.ones(len(f)) * 50.0
    s = np.zeros((len(f), 2, 2))
    with pytest.raises(ValueError, match=r"either z or s, not both"):
        fit_rlc(f, z, s=s, model="vector_fit")
    with pytest.raises(ValueError, match=r"rlc1p.*impedance"):
        fit_rlc(f, s=s, model="rlc1p")
    with pytest.raises(ValueError, match=r"unknown fit model"):
        fit_rlc(f, z, model="nonexistent")
    with pytest.raises(ValueError, match=r"provide impedance data"):
        fit_rlc(f, model="rlc1p")
