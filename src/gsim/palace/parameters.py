"""Batched microwave network parameter conversions (S <-> Z <-> Y).

First-class conversions between scattering, impedance and admittance
parameter matrices with explicit, preserved reference impedance (issue
#272: frequency units, port order and reference impedances are kept).

Conventions (real, diagonal reference impedance ``z0`` — the standard
Palace/50 Ohm case; power-wave definitions):

    Z = z0 * (I + S) (I - S)^-1
    S = (I - z0 Y) (I + z0 Y)^-1
    Y = (1/z0) * (I - S) (I + S)^-1

All functions are batched over frequencies, accept ``z0`` as a scalar or a
per-port vector of length N, and keep the port order of the input. Matrices
must be complete ``(nf, N, N)`` arrays; incomplete matrices are rejected
(``unknown_entries`` in an incomplete S-matrix would silently bias the
conversion).

Usage::

    from gsim.palace.parameters import s_to_z, z_to_s, y_to_s

    z = s_to_z(network.s, z0=50.0)
    s_back = z_to_s(z, z0=50.0)  # round-trips to machine precision
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _as_matrix(x: NDArray, name: str) -> NDArray:
    """Validate a parameter matrix is the complete (nf, N, N) stack."""
    arr = np.asarray(x, dtype=complex)
    if arr.ndim != 3 or arr.shape[-1] != arr.shape[-2] or arr.shape[-1] < 1:
        raise ValueError(
            f"{name} must be a complete (nf, N, N) matrix stack, got {arr.shape}"
        )
    return arr


def _eye(n_ports: int) -> NDArray:
    """Identity matrix alias with complex dtype."""
    return np.eye(n_ports, dtype=complex)


def _z0_matrix(z0: float | NDArray, n_freq: int, n_ports: int) -> NDArray:
    """Broadcast ``z0`` (scalar or per-port vector) to (nf, N, N) diagonals."""
    arr = np.asarray(z0, dtype=float)
    if arr.ndim == 0:
        return np.eye(n_ports, dtype=complex) * float(arr)
    if arr.shape != (n_ports,):
        raise ValueError(f"z0 must be scalar or a length-N vector, got {arr.shape}")
    return np.tile(np.diag(arr), (n_freq, 1, 1))


def s_to_z(s: NDArray, z0: float | NDArray = 50.0) -> NDArray:
    """Convert scattering to impedance parameters, ``(nf, N, N)`` [Ohm]."""
    s = _as_matrix(s, "s")
    n_freq, n_ports = s.shape[0], s.shape[-1]
    eye = _eye(n_ports)
    inv = np.linalg.inv(eye - s)
    return (eye + s) @ inv @ _z0_matrix(z0, n_freq, n_ports)


def z_to_s(z: NDArray, z0: float | NDArray = 50.0) -> NDArray:
    """Convert impedance to scattering parameters, ``(nf, N, N)``."""
    z = _as_matrix(z, "z")
    n_ports = z.shape[-1]
    eye = _eye(n_ports)
    y = np.linalg.inv(z)
    z0_mat = _z0_matrix(z0, z.shape[0], n_ports)
    return (eye - z0_mat @ y) @ np.linalg.inv(eye + z0_mat @ y)


def s_to_y(s: NDArray, z0: float | NDArray = 50.0) -> NDArray:
    """Convert scattering to admittance parameters, ``(nf, N, N)`` [S]."""
    s = _as_matrix(s, "s")
    n_freq, n_ports = s.shape[0], s.shape[-1]
    eye = _eye(n_ports)
    inv = np.linalg.inv(eye + s)
    return np.linalg.inv(_z0_matrix(z0, n_freq, n_ports)) @ (eye - s) @ inv


def y_to_s(y: NDArray, z0: float | NDArray = 50.0) -> NDArray:
    """Convert admittance to scattering parameters, ``(nf, N, N)``."""
    y = _as_matrix(y, "y")
    n_ports = y.shape[-1]
    eye = _eye(n_ports)
    z0_mat = _z0_matrix(z0, y.shape[0], n_ports)
    return (eye - z0_mat @ y) @ np.linalg.inv(eye + z0_mat @ y)


def y_to_z(y: NDArray) -> NDArray:
    """Convert admittance to impedance parameters, ``(nf, N, N)`` [Ohm]."""
    return np.linalg.inv(_as_matrix(y, "y"))


def z_to_y(z: NDArray) -> NDArray:
    """Convert impedance to admittance parameters, ``(nf, N, N)`` [S]."""
    return np.linalg.inv(_as_matrix(z, "z"))


def is_complete(x: NDArray) -> bool:
    """Whether *x* is a full, finite ``(nf, N, N)`` parameter stack."""
    try:
        arr = _as_matrix(x, "x")
    except ValueError:
        return False
    return bool(np.all(np.isfinite(arr.view(float))))


__all__ = [
    "is_complete",
    "s_to_y",
    "s_to_z",
    "y_to_s",
    "y_to_z",
    "z_to_s",
    "z_to_y",
]
