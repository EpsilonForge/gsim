"""Tests for Palace AC circuit synthesis (AdaptiveCircuitSynthesis) support.

Covers the DrivenConfig flag wiring and the rom-*.csv parser / circuit
evaluation helpers in gsim.palace.circuit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gsim.palace import DrivenSim
from gsim.palace.circuit import CircuitSynthesis, load_circuit_synthesis
from gsim.palace.models import DrivenConfig

# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------


def test_driven_config_emits_circuit_synthesis():
    """circuit_synthesis=True emits AdaptiveCircuitSynthesis in the JSON."""
    config = DrivenConfig(
        fmin=1e9, fmax=10e9, num_points=11, circuit_synthesis=True
    ).to_palace_config()
    assert config["AdaptiveCircuitSynthesis"] is True
    assert config["AdaptiveTol"] > 0


def test_driven_config_circuit_synthesis_requires_adaptive_sweep():
    """Circuit synthesis without an adaptive sweep raises (Palace rejects it)."""
    with pytest.raises(ValueError, match="adaptive"):
        DrivenConfig(adaptive_tol=0, circuit_synthesis=True).to_palace_config()


def test_driven_config_default_has_no_circuit_synthesis():
    """The flag is opt-in: default config has no AdaptiveCircuitSynthesis key."""
    config = DrivenConfig().to_palace_config()
    assert "AdaptiveCircuitSynthesis" not in config


def test_set_driven_circuit_synthesis():
    """DrivenSim.set_driven passes circuit_synthesis through to the config."""
    sim = DrivenSim()
    sim.set_driven(fmin=1e9, fmax=10e9, num_points=11, circuit_synthesis=True)
    assert sim.driven.circuit_synthesis is True
    palace_config = sim.driven.to_palace_config()
    assert palace_config["AdaptiveCircuitSynthesis"] is True


# ---------------------------------------------------------------------------
# Synthetic rom-*.csv parsing
# ---------------------------------------------------------------------------


def _write_matrix(path: Path, labels: list[str], values) -> None:
    """Write a Palace-style rom matrix CSV (header row of labels)."""
    values = np.asarray(values, dtype=float)
    lines = [",".join(labels)]
    lines.extend(",".join(f"{v:.17e}" for v in row) for row in values)
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def rom_dir(tmp_path: Path) -> Path:
    """Two-port + one-interior-node synthesized circuit output.

    Inductive network (proper graph Laplacian in L^-1): 1 nH between the two
    ports, 2 nH from port 1 to the interior node. Capacitors: 10 fF shunt at
    each port, 1 nF from the interior node to ground (dominates the interior
    shunt so the Schur complement in the tests is free of cancellation).
    Ports are terminated with 50 Ohm (R^-1 diagonal), exported as portload
    blocks in a separate test.
    """
    labels = ["port_1_re", "port_2_re", "sample_e1_s0_re"]
    g1 = 1.0 / 1e-9  # 1 nH port-to-port
    g2 = 1.0 / 2e-9  # 2 nH port1-to-interior
    L = np.array(
        [
            [g1 + g2, -g1, -g2],
            [-g1, g1, 0.0],
            [-g2, 0.0, g2],
        ]
    )
    R = np.diag([1.0 / 50.0, 1.0 / 50.0, 0.0])
    C = np.diag([10e-15, 10e-15, 1e-9])

    d = tmp_path / "output" / "palace"
    d.mkdir(parents=True)
    _write_matrix(d / "rom-Linv-re.csv", labels, L.real)
    _write_matrix(d / "rom-Rinv-re.csv", labels, R.real)
    _write_matrix(d / "rom-C-re.csv", labels, C.real)
    # Imaginary parts: only a loss-tangent-like contribution on C.
    _write_matrix(d / "rom-C-im.csv", labels, C.real * 1e-3)

    port_info = {
        "ports": [
            {"portnumber": 1, "name": "P1"},
            {"portnumber": 2, "name": "P2"},
        ],
        "unit": 1e-6,
        "name": "palace",
    }
    (tmp_path / "port_information.json").write_text(json.dumps(port_info))
    return tmp_path


def test_load_circuit_synthesis_from_dir(rom_dir: Path):
    """Directory loading parses labels, complex matrices and node kinds."""
    circuit = load_circuit_synthesis(rom_dir)
    assert circuit.nodes == ["port_1_re", "port_2_re", "sample_e1_s0_re"]
    assert circuit.port_labels == ["port_1_re", "port_2_re"]
    assert circuit.internal_indices == [2]
    assert circuit.L_inv.shape == (3, 3)
    assert circuit.R_inv[0, 0] == pytest.approx(1 / 50.0)
    # Complex assembly: re + 1j*im
    assert circuit.C[0, 0] == pytest.approx(10e-15 * (1 + 1e-3j))


def test_load_circuit_synthesis_port_map(rom_dir: Path):
    """port_map attaches gsim port names to palace port nodes."""
    circuit = load_circuit_synthesis(rom_dir, port_map={1: "P1", 2: "P2"})
    assert circuit.port_names == {
        "port_1_re": "P1",
        "port_2_re": "P2",
    }


def test_load_circuit_synthesis_missing_files(tmp_path: Path):
    """A directory without rom files raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="rom-"):
        load_circuit_synthesis(tmp_path)


def test_schur_complement_matches_analytic_ladder(rom_dir: Path):
    """Interior-node elimination reproduces the analytic network admittance.

    Synthetic network: 1 nH directly between the ports, 2 nH from port 1 to
    an interior node that carries a 1 nF shunt to ground, 10 fF shunts at
    each port. The fixture also writes an imaginary-C file (loss factor
    1e-3), so the closed-form expectation uses the same complex capacitors.
    """
    circuit = load_circuit_synthesis(rom_dir)
    f = np.array([1e9, 5e9, 20e9])
    w = 2 * np.pi * f
    Yp = circuit.port_admittance(f)

    loss = 1e-3  # fixture writes rom-C-im.csv = rom-C-re.csv * loss
    c_p = 10e-15 * (1 + 1j * loss)  # port shunt capacitors
    c_int = 1e-9 * (1 + 1j * loss)  # interior node capacitor

    y_1n = 1.0 / (1j * w * 1e-9)  # 1 nH port-to-port path
    y_2n = 1.0 / (1j * w * 2e-9)  # 2 nH port1-to-interior path
    y_int = y_2n + 1j * w * c_int  # interior node shunt

    expected = np.empty_like(Yp)
    expected[:, 0, 0] = 1 / 50.0 + 1j * w * c_p + y_1n + y_2n - y_2n * y_2n / y_int
    expected[:, 0, 1] = expected[:, 1, 0] = -y_1n
    expected[:, 1, 1] = 1 / 50.0 + 1j * w * c_p + y_1n

    np.testing.assert_allclose(Yp, expected, rtol=1e-10)


def test_port_load_subtraction_recovers_bare_device(rom_dir: Path):
    """Subtracting rom-portload blocks removes the 50 Ohm port terminations."""
    labels = ["port_1_re", "port_2_re", "sample_e1_s0_re"]
    d = rom_dir / "output" / "palace"
    load_r1 = np.zeros((3, 3))
    load_r1[0, 0] = 1.0 / 50.0
    load_r2 = np.zeros((3, 3))
    load_r2[1, 1] = 1.0 / 50.0
    _write_matrix(d / "rom-portload-port_1_re-Rinv-re.csv", labels, load_r1)
    _write_matrix(d / "rom-portload-port_2_re-Rinv-re.csv", labels, load_r2)

    circuit = load_circuit_synthesis(rom_dir)
    f = np.array([1e9])
    Y_loaded = circuit.port_admittance(f)
    Y_bare = circuit.port_admittance(f, subtract_port_loads=True)

    # The 50 Ohm terminations are removed from the terminal diagonal exactly;
    # the small residual real part comes from the fixture's lossy C.
    assert Y_loaded[0, 0, 0].real - Y_bare[0, 0, 0].real == pytest.approx(1 / 50.0)
    assert Y_bare[0, 0, 0].real == pytest.approx(0.0, abs=2e-6)
    assert Y_bare[0, 1, 0] == Y_loaded[0, 1, 0]  # off-diagonals untouched
    # Reference impedances come from the portload resistance.
    np.testing.assert_allclose(circuit.port_reference_impedances(), [50.0, 50.0])


def test_s_parameters_series_inductor_analytic():
    """S21 of a pure series inductor matches 2/(2 + Z/Z0)."""
    Ls = 1e-9
    g = 1.0 / Ls
    circuit = CircuitSynthesis(
        nodes=["port_1_re", "port_2_re"],
        L_inv=np.array([[g, -g], [-g, g]]),
        R_inv=np.zeros((2, 2)),
        C=np.zeros((2, 2)),
    )

    f = np.array([2e9, 10e9])
    S = circuit.s_parameters(f)
    omega = 2 * np.pi * f
    for k in range(len(f)):
        z_series = 1j * omega[k] * Ls
        assert S[k, 1, 0] == pytest.approx(2.0 / (2.0 + z_series / 50.0), rel=1e-10)
        # Series impedance between equal references: S11 = Z / (2 Z0 + Z).
        assert S[k, 0, 0] == pytest.approx(z_series / (2 * 50.0 + z_series), rel=1e-10)


def test_circuit_synthesis_class_validation():
    """Matrix shape mismatches are rejected."""
    with pytest.raises(ValueError, match="shape"):
        CircuitSynthesis(
            nodes=["port_1_re"],
            L_inv=np.zeros((2, 2)),
            R_inv=np.zeros((2, 2)),
            C=np.zeros((2, 2)),
        )
