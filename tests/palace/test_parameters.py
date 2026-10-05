"""Tests for batched S/Z/Y parameter conversions (gsim.palace.parameters)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from gsim.palace.parameters import (
    is_complete,
    s_to_y,
    s_to_z,
    y_to_s,
    y_to_z,
    z_to_s,
    z_to_y,
)


@pytest.fixture
def causal_two_port() -> NDArray:
    """A passive 2-port: series branch + shunt reference cell.

    A small ground-referenced shunt makes the admittance invertible (a pure
    differential Laplacian is singular — no common-mode return path).
    """
    f = np.linspace(10e9, 200e9, 64)
    w = 2 * np.pi * f
    L, C, R = 110e-12, 8e-15, 3.5
    y_tot = 1.0 / (R + 1j * w * L) + 1j * w * C
    y = np.empty((len(w), 2, 2), dtype=complex)
    y[:, 0, 0] = y[:, 1, 1] = y_tot + 1j * w * 1e-15
    y[:, 0, 1] = y[:, 1, 0] = -y_tot
    return y


def test_round_trip_s_z_y_s(causal_two_port):
    """S -> Z -> S and S -> Y -> S round-trips to machine precision."""
    s = y_to_s(causal_two_port, z0=50.0)
    np.testing.assert_allclose(z_to_s(s_to_z(s, z0=50.0), z0=50.0), s, rtol=1e-9)
    np.testing.assert_allclose(y_to_s(s_to_y(s, z0=50.0), z0=50.0), s, rtol=1e-9)
    np.testing.assert_allclose(
        z_to_y(y_to_z(causal_two_port)), causal_two_port, rtol=1e-9
    )


def test_series_inductor_analytic():
    """1-port series-L behavior: Z = Z0 (1 + S)/(1 - S)."""
    f = np.array([2e9, 10e9])
    L = 1e-9
    z_true = 1j * 2 * np.pi * f * L
    s = ((z_true - 50.0) / (z_true + 50.0))[:, None, None]
    z = s_to_z(s, z0=50.0)[:, 0, 0]
    np.testing.assert_allclose(z, z_true, rtol=1e-12)
    # and the reverse
    s_back = z_to_s(z[:, None, None], z0=50.0)[:, 0, 0]
    np.testing.assert_allclose(s_back, s[:, 0, 0], rtol=1e-12)


def test_four_port_round_trip_with_per_port_z0():
    """4-port random passive-ish matrices survive round trips with per-port z0."""
    rng = np.random.default_rng(3)
    nf, n = 12, 4
    a = rng.normal(size=(nf, n, n)) + 1j * rng.normal(size=(nf, n, n))
    y = a + np.eye(n)[None] * (5.0 + 5j)
    z0 = np.array([50.0, 50.0, 25.0, 100.0])
    s = y_to_s(y, z0=z0)
    np.testing.assert_allclose(y_to_s(s_to_y(s, z0=z0), z0=z0), s, rtol=1e-8)
    np.testing.assert_allclose(z_to_s(s_to_z(s, z0=z0), z0=z0), s, rtol=1e-8)
    np.testing.assert_allclose(s_to_z(s, z0=z0), y_to_z(y), rtol=1e-8)


def test_incomplete_matrix_rejected():
    """The issue's 'reject incomplete S matrices' criterion."""
    s_ok = np.zeros((3, 2, 2), dtype=complex)
    with pytest.raises(ValueError, match="complete \\(nf, N, N\\)"):
        s_to_z(s_ok[:, :, :1], z0=50.0)
    with pytest.raises(ValueError, match="complete \\(nf, N, N\\)"):
        s_to_z(s_ok[0], z0=50.0)
    assert is_complete(s_ok)
    assert not is_complete(s_ok[:, :, :1])
    assert not is_complete(s_ok * np.nan)


def test_negative_resistance_units_are_preserved():
    """z0 scaling, frequency units and port order are preserved (issue #272)."""
    f = np.linspace(10e9, 200e9, 24)
    z_load = 3.0 + 1j * 2 * np.pi * f * 100e-12
    s = ((z_load - 25.0) / (z_load + 25.0))[:, None, None]
    z = s_to_z(s, z0=25.0)[:, 0, 0]
    np.testing.assert_allclose(z, z_load, rtol=1e-12)
