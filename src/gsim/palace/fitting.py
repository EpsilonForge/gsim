"""Reusable EM-to-circuit fitting workflow.

Fits lumped RLC equivalent-circuit models to simulated impedance data —
Palace S-parameters, the synthesized circuit from
:mod:`gsim.palace.circuit`, or any complex Z(f) source — turning the
notebook fitting workflow into package API.

Model: a series R-L branch in parallel with a capacitance C (the one-pole
model used by the inductor tutorial):

    Y(f) = 1 / (R + i 2 pi f L) + i 2 pi f C
    Z(f) = 1 / Y(f)

For optimization the model is parameterized by the resonance frequency
``f0``, quality factor ``Q`` and low-frequency resistance ``R``:

    z(w~, Q) = (1 + i w~ Q) / (1 - w~^2 + i w~ / Q)
    Z(f)     = R * z(f / f0, Q)

which keeps the optimizer well conditioned over wide frequency bands.

Usage::

    from gsim.palace.fitting import differential_impedance, fit_rlc

    z = differential_impedance(network.z)  # (nf, N, N) impedance matrix
    fit = fit_rlc(f_hz, z)
    fit.L, fit.C, fit.f0, fit.Q

Two solvers are available: ``"jax"`` (Adam, autodiff — requires the optional
``jax``/``optax`` dependencies) and ``"scipy"`` (least-squares). With
``solver="auto"`` the JAX solver is used when available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

Solver = Literal["auto", "jax", "scipy"]

# Resistances below this are treated as zero when seeding the optimizer.
_MIN_R = 1e-9


@dataclass(frozen=True)
class RLCFit:
    """Fitted one-pole RLC equivalent circuit.

    Attributes:
        R: Low-frequency series resistance [Ohm].
        L: Series inductance [H].
        C: Parallel capacitance [F].
        f0: Resonance frequency (1 / (2 pi sqrt(L C))) [Hz].
        Q: Quality factor at resonance (2 pi f0 L / R).
        rms_error: RMS of |Z_model - Z_data| over the fit band [Ohm].
    """

    R: float
    L: float
    C: float
    f0: float
    Q: float
    rms_error: float

    def z(self, f: NDArray) -> NDArray:
        """Model impedance [Ohm] at frequencies *f* in Hz."""
        return 1.0 / self.y(f)

    def y(self, f: NDArray) -> NDArray:
        """Model admittance [S] at frequencies *f* in Hz."""
        f = np.asarray(f, dtype=float)
        w = 2.0 * np.pi * f
        return 1.0 / (self.R + 1j * w * self.L) + 1j * w * self.C

    def to_dict(self) -> dict[str, float]:
        """Return the fitted parameters as a plain dict (SI units)."""
        return {
            "R": self.R,
            "L": self.L,
            "C": self.C,
            "f0": self.f0,
            "Q": self.Q,
            "rms_error": self.rms_error,
        }

    def __repr__(self) -> str:
        """Return concise object representation."""
        return (
            f"RLCFit(R={self.R:.4g} Ohm, L={self.L * 1e12:.3f} pH, "
            f"C={self.C * 1e15:.3f} fF, f0={self.f0 / 1e9:.3f} GHz, "
            f"Q={self.Q:.2f}, rms={self.rms_error:.2e})"
        )


def z_rlc(w_norm, Q):
    """Normalized RLC impedance ``z(w~, Q) = (1 + i w~ Q) / (1 - w~^2 + i w~/Q)``.

    ``Z(f) = R * z_rlc(f / f0, Q)``. Untyped parameters on purpose: the
    formula is used with both numpy and jax arrays.
    """
    return (1 + 1j * w_norm * Q) / (1 - w_norm**2 + 1j * w_norm / Q)


class VectorFit:
    """Multi-pole rational fit (scikit-rf ``VectorFitting``) with passivity.

    Wraps :class:`skrf.vectorFitting.VectorFitting` and adds S/Z/Y
    evaluation, stability/passivity reporting, spurious-pole detection and
    SPICE export. Stability means every pole lies in the open left half
    plane. Passivity is scikit-rf's ``is_passive`` on the fitted rational
    model (run :meth:`passivity_enforce` when it fails).
    """

    def __init__(self, vf, *, z0: float | NDArray) -> None:
        """Wrap a fitted :class:`skrf.vectorFitting.VectorFitting`."""
        self._vf = vf
        self.z0: float | NDArray = z0

    # -- underlying model --------------------------------------------------

    @property
    def raw(self):
        """The wrapped :class:`skrf.vectorFitting.VectorFitting`."""
        return self._vf

    @property
    def network(self):
        """The fitted skrf ``Network`` (measured data used for training)."""
        return self._vf.network

    @property
    def poles(self) -> NDArray:
        """Model poles [rad/s] (complex)."""
        return np.asarray(self._vf.poles)

    @property
    def residues(self) -> NDArray:
        """Model residues (one row/port pair)."""
        return np.asarray(self._vf.residues)

    @property
    def zeros(self) -> NDArray:
        """Model zeros [rad/s]."""
        return np.asarray(self._vf.zeros)

    @property
    def n_poles(self) -> int:
        """Number of poles in the fitted model."""
        return len(self._vf.poles)

    # -- quality / passivity -----------------------------------------------

    @property
    def is_stable(self) -> bool:
        """Whether all poles are in the open left half plane."""
        poles = self.poles
        return bool(np.all(poles.real < 0.0))

    def rms_error(self, **kwargs) -> float:
        """RMS error of the rational model (delegates to skrf)."""
        return float(self._vf.get_rms_error(**kwargs))

    def is_passive(self, **kwargs) -> bool:
        """Whether the fitted model passes scikit-rf's passivity test."""
        return bool(self._vf.is_passive(**kwargs))

    def passivity_test(self, **kwargs) -> NDArray:
        """Passivity metric array (delegates to skrf ``passivity_test``)."""
        return np.asarray(self._vf.passivity_test(**kwargs))

    def passivity_enforce(self, **kwargs) -> VectorFit:
        """Enforce passivity (delegates to skrf ``passivity_enforce``)."""
        self._vf.passivity_enforce(**kwargs)
        return self

    def get_spurious(self, **kwargs) -> NDArray:
        """Boolean mask of spurious poles (delegates to skrf)."""
        return np.asarray(self._vf.get_spurious(self.poles, self.residues, **kwargs))

    # -- response evaluation ------------------------------------------------

    def s(self, f: NDArray | None = None) -> NDArray:
        """Model S-parameters ``(nf, N, N)`` (or the training data)."""
        if f is None:
            return np.asarray(self._vf.network.s)
        freqs = np.atleast_1d(np.asarray(f, dtype=float))
        n_ports = self._vf.network.s.shape[-1]
        out = np.empty((len(freqs), n_ports, n_ports), dtype=complex)
        for i in range(n_ports):
            for j in range(n_ports):
                out[:, i, j] = self._vf.get_model_response(i, j, freqs)
        return out

    def _z0_matrix(self, n_freq: int, n_ports: int) -> NDArray:
        """Reference-impedance diagonal matrices ``(nf, N, N)``."""
        arr = np.asarray(self.z0, dtype=float)
        if arr.ndim == 0:
            return np.eye(n_ports, dtype=complex) * float(arr)
        if arr.shape == (n_ports,):
            return np.tile(np.diag(arr), (n_freq, 1, 1))
        if arr.shape == (n_freq, n_ports):
            return np.stack([np.diag(row) for row in arr])
        raise ValueError(f"z0 must be scalar, (N,) or (nf, N), got {arr.shape}")

    def z(self, f: NDArray | None = None) -> NDArray:
        """Model impedance matrices ``(nf, N, N)`` [Ohm]."""
        s_model = self.s(f)
        n_freq, n_ports = s_model.shape[0], s_model.shape[-1]
        eye = np.eye(n_ports, dtype=complex)
        inv = np.linalg.inv(eye - s_model)
        return (eye + s_model) @ inv @ self._z0_matrix(n_freq, n_ports)

    def y(self, f: NDArray | None = None) -> NDArray:
        """Model admittance matrices ``(nf, N, N)`` [S]."""
        return np.linalg.inv(self.z(f))

    # -- export --------------------------------------------------------------

    def write_spice(self, path: str, **kwargs) -> None:
        """Export the rational model as a SPICE subcircuit (skrf)."""
        self._vf.write_spice_subcircuit_s(str(path), **kwargs)

    def __repr__(self) -> str:
        """Return concise object representation."""
        return (
            f"VectorFit(n_poles={self.n_poles}, ports={self._vf.network.s.shape[-1]}, "
            f"stable={self.is_stable}, passive={self.is_passive()}, "
            f"rms={self.rms_error():.2e})"
        )


def differential_impedance(z: NDArray) -> NDArray:
    """Differential impedance ``Z11 - Z12 - Z21 + Z22`` of a Z-matrix.

    Args:
        z: Impedance matrices with shape ``(nf, N, N)`` (e.g. from
            ``skrf.Network.z`` or :meth:`CircuitSynthesis.port_impedance`).
            For a single-port network (``N == 1``) the driving-point
            impedance ``Z11`` is returned directly — it is already the
            differential quantity.

    Returns:
        Array of shape ``(nf,)`` — the impedance seen between the two
        ports of a differential pair under differential excitation.
    """
    z = np.asarray(z)
    if z.ndim != 3 or z.shape[-1] != z.shape[-2] or z.shape[-1] < 1:
        raise ValueError(
            f"expected (nf, N, N) impedance matrix with N >= 1, got {z.shape}"
        )
    if z.shape[-1] == 1:
        return z[:, 0, 0]
    return z[:, 0, 0] - z[:, 0, 1] - z[:, 1, 0] + z[:, 1, 1]


def initial_guess_rlc(f: NDArray, z: NDArray) -> tuple[float, float, float]:
    """Estimate ``(f0, Q, R)`` directly from impedance data.

    - ``f0`` is the frequency where |Z| peaks,
    - ``R`` is the low-frequency resistance Re(Z),
    - ``Q`` comes from the -3 dB bandwidth of the |Z| peak.
    """
    f = np.asarray(f, dtype=float)
    z = np.asarray(z)
    abs_z = np.abs(z)
    f0 = float(f[int(np.argmax(abs_z))])
    r = max(float(np.real(z[0])), _MIN_R)
    mask = abs_z > abs_z.max() / np.sqrt(2)
    q = f0 / float(np.ptp(f[mask])) if mask.sum() > 1 else 5.0
    return f0, q, r


def fit_rlc(
    f: NDArray,
    z: NDArray | None = None,
    *,
    s: NDArray | None = None,
    model: Literal["rlc1p", "vector_fit"] = "rlc1p",
    z0: float | NDArray = 50.0,
    solver: Solver = "auto",
    steps: int = 1000,
    learning_rate: float = 0.05,
    n_poles_real: int | None = 3,
    n_poles_cmplx: int = 3,
    enforce_passivity: bool = False,
    target_error: float = 1e-2,
) -> RLCFit | VectorFit:
    """Fit a circuit model to (S- or Z-parameter) data.

    Two models are available:

    - ``model="rlc1p"`` — the one-pole equivalent circuit
      (series R-L branch in parallel with C), returned as
      :class:`RLCFit`. Fits impedance data ``z`` with the
      normalized ``(f0, Q, R)`` parameterization (JAX/Adam in log
      space when the optional JAX dependencies are installed, else a
      scipy least-squares fit).
    - ``model="vector_fit"`` — scikit-rf's rational (vector-fitting)
      model with multiple poles, returned as :class:`VectorFit`.
      Fits S-parameter data ``s`` (or ``z``, converted first, and for
      one-port data ``z`` may be a scalar series). Supports a built-in
      passivity test and enforcement (``passivity_test`` /
      ``passivity_enforce``), and SPICE export
      (``write_spice_subcircuit_s``). Requires the optional ``skrf``
      dependency.

    Args:
        f: Frequencies in Hz.
        z: Complex impedance data [Ohm]: either a scalar series
            ``(nf,)`` or a full impedance matrix ``(nf, N, N)`` (the
            one-pole model uses the differential impedance
            ``Z11 - Z12 - Z21 + Z22`` of the matrix). One of ``z``/``s``
            must be given.
        s: Complex S-parameter data ``(nf, N, N)``; used by
            ``model="vector_fit"`` directly.
        model: ``"rlc1p"`` or ``"vector_fit"``.
        z0: Reference impedance [Ohm] used by ``model="vector_fit"``
            when converting impedance data to S-parameters (scalar,
            or per-port array of length N).
        steps: Optimizer iterations (one-pole, JAX solver only).
        learning_rate: Adam learning rate (one-pole, JAX solver only).
        solver: One-pole solver: ``"auto"`` (JAX if installed, else
            scipy), ``"jax"`` or ``"scipy"``.
        n_poles_real: Number of real poles for the vector fit. Pass
            ``None`` to run scikit-rf's ``auto_fit`` pole-adding loop
            with ``target_error``.
        n_poles_cmplx: Number of complex pole pairs for the vector fit.
        enforce_passivity: For ``model="vector_fit"``, run
            ``passivity_enforce`` after fitting when the model fails the
            passivity test.
        target_error: Target RMS error for the ``auto_fit`` loop
            (vector fit with ``n_poles_real=None``).

    Returns:
        :class:`RLCFit` (one-pole) or :class:`VectorFit` (vector fit).
    """
    if model == "vector_fit":
        return _fit_vector_skrf(
            f,
            z=z,
            s=s,
            z0=z0,
            n_poles_real=n_poles_real,
            n_poles_cmplx=n_poles_cmplx,
            enforce_passivity=enforce_passivity,
            target_error=target_error,
        )
    if model != "rlc1p":
        raise ValueError(
            f"unknown fit model {model!r}; expected 'rlc1p' or 'vector_fit'"
        )
    if s is not None:
        if z is not None:
            raise ValueError("provide either z or s, not both")
        raise ValueError(
            "model='rlc1p' fits impedance data: provide z "
            "(s is only used by model='vector_fit')"
        )
    if z is None:
        raise ValueError("provide impedance data z (or use model='vector_fit' with s)")
    f = np.asarray(f, dtype=float)
    z = np.asarray(z, dtype=complex)
    if z.ndim == 3:
        z = differential_impedance(z)
    if solver == "auto":
        solver = "jax" if _jax_available() else "scipy"

    if solver == "jax":
        f0, q, r = _fit_rlc_jax(f, z, steps=steps, learning_rate=learning_rate)
    else:
        f0, q, r = _fit_rlc_scipy(f, z)
    return _finalize_fit(f, z, f0, q, r)


_SKRF_HINT = (
    "scikit-rf is required for model='vector_fit': install it with "
    "`pip install scikit-rf` (or `uv add scikit-rf`)."
)


def _fit_vector_skrf(
    f: NDArray,
    *,
    z: NDArray | None,
    s: NDArray | None,
    z0: float | NDArray,
    n_poles_real: int | None,
    n_poles_cmplx: int,
    enforce_passivity: bool,
    target_error: float,
) -> VectorFit:
    """Fit a scikit-rf rational model to S- (or converted Z-) data."""
    try:
        import skrf as rf
        from skrf.vectorFitting import VectorFitting as _SkrfVectorFitting
    except ImportError as exc:
        raise ImportError(_SKRF_HINT) from exc

    f = np.asarray(f, dtype=float)
    s_data = _to_s_data(f, z, s, z0)
    z0_net = _network_z0(z0, s_data.shape[1], len(f))
    network = rf.Network(
        frequency=rf.Frequency.from_f(f, unit="Hz"),
        s=s_data,
        z0=z0_net,
        name="fit_rlc",
    )
    vf = _SkrfVectorFitting(network)
    if n_poles_real is None:
        vf.auto_fit(target_error=target_error)
    else:
        vf.vector_fit(n_poles_real=n_poles_real, n_poles_cmplx=n_poles_cmplx)
    result = VectorFit(vf, z0=z0_net)
    if enforce_passivity and not result.is_passive():
        result.passivity_enforce(f_max=float(np.max(f)))
    return result


def _to_s_data(
    f: NDArray, z: NDArray | None, s: NDArray | None, z0: float | NDArray
) -> NDArray:
    """Normalize the ``z``/``s`` inputs into an ``(nf, N, N)`` S array."""
    del f
    if z is not None and s is not None:
        raise ValueError("provide either z or s, not both")
    if s is not None:
        s = np.asarray(s, dtype=complex)
        if s.ndim != 3:
            raise ValueError(f"s must have shape (nf, N, N), got {np.asarray(s).shape}")
        return s
    if z is None:
        raise ValueError("provide impedance data z (or S-parameter data s)")
    z = np.asarray(z, dtype=complex)
    if z.ndim == 1:
        if not np.isscalar(z0):
            raise ValueError("1-port z data requires a scalar z0")
        z0_value = complex(float(np.asarray(z0, dtype=float)))
        s_data = (z - z0_value) / (z + z0_value)
        return s_data[:, None, None]
    if z.ndim != 3 or z.shape[-1] != z.shape[-2]:
        raise ValueError(f"z must have shape (nf,) or (nf, N, N), got {z.shape}")
    zero = np.eye(z.shape[-1], dtype=complex)
    z0_arr = np.asarray(z0, dtype=float)
    if z0_arr.ndim == 0:
        Z0 = zero * complex(z0_arr)
    elif z0_arr.shape == (z.shape[-1],):
        Z0 = np.diag(z0_arr)
    else:
        raise ValueError(f"z0 must be scalar or length-N array, got {z0_arr.shape}")
    y = np.linalg.inv(z)
    eye = np.eye(z.shape[-1], dtype=complex)
    s_data = (eye - Z0[None, :, :] @ y) @ np.linalg.inv(eye + Z0[None, :, :] @ y)
    return s_data


def _network_z0(z0: float | NDArray, n_ports: int, n_freq: int) -> float | NDArray:
    """Broadcast a per-port z0 to skrf's expected ``(nf, N)`` layout."""
    arr = np.asarray(z0, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.shape == (n_ports,):
        return np.tile(arr, (n_freq, 1))
    if arr.shape == (n_freq, n_ports):
        return arr
    raise ValueError(f"z0 must be scalar, (N,) or (nf, N), got {arr.shape}")


def _finalize_fit(f: NDArray, z: NDArray, f0: float, q: float, r: float) -> RLCFit:
    """Build an :class:`RLCFit` from (f0, Q, R) and score it against the data."""
    f0 = max(float(f0), 1.0)
    q = max(float(q), 1e-6)
    r = max(float(r), _MIN_R)
    w0 = 2.0 * np.pi * f0
    inductance = q * r / w0
    capacitance = 1.0 / (inductance * w0**2)
    model = r * z_rlc(f / f0, q)
    rms = float(np.sqrt(np.mean(np.abs(model - z) ** 2)))
    return RLCFit(R=r, L=inductance, C=capacitance, f0=f0, Q=q, rms_error=rms)


def _jax_available() -> bool:
    """Return whether the optional JAX dependencies can be imported."""
    try:
        import jax  # noqa: F401
        import optax  # noqa: F401
    except ImportError:
        return False
    return True


def _fit_rlc_jax(
    f: NDArray,
    z: NDArray,
    *,
    steps: int,
    learning_rate: float,
) -> tuple[float, float, float]:
    """Adam + autodiff fit on the normalized (f0, Q, R) parameterization.

    Optimization runs in log space: the three parameters span decades of
    magnitude (f0 ~ 1e11 Hz vs R ~ Ohm), so a shared learning rate is only
    scale-free after this reparameterization.
    """
    import jax
    import jax.numpy as jnp
    import optax

    jax.config.update("jax_enable_x64", True)

    f_j = jnp.asarray(f, dtype=jnp.float64)
    z_t = jnp.asarray(z, dtype=jnp.complex128)

    @jax.jit
    def loss_fn(log_param):
        f0, q, r = jnp.exp(log_param[0]), jnp.exp(log_param[1]), jnp.exp(log_param[2])
        z_fit = r * z_rlc(f_j / f0, q)
        z_err = z_t - z_fit
        return jnp.real(jnp.sum(z_err * jnp.conj(z_err)))

    f0_ini, q_ini, r_ini = initial_guess_rlc(f, z)
    par = jnp.log(jnp.array([f0_ini, q_ini, r_ini]))
    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(par)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    for _ in range(steps):
        _, grads = value_and_grad(par)
        updates, opt_state = optimizer.update(grads, opt_state)
        par = optax.apply_updates(par, updates)
    f0, q, r = (float(x) for x in np.exp(np.asarray(par)))
    return f0, q, r


def _fit_rlc_scipy(f: NDArray, z: NDArray) -> tuple[float, float, float]:
    """Least-squares fit without the optional JAX dependencies."""
    from scipy.optimize import least_squares

    def residuals(param):
        model = param[2] * z_rlc(f / param[0], param[1])
        return np.r_[model.real - z.real, model.imag - z.imag]

    f0_ini, q_ini, r_ini = initial_guess_rlc(f, z)
    result = least_squares(
        residuals,
        x0=[f0_ini, q_ini, r_ini],
        bounds=([np.min(f), 1e-6, 0.0], [np.max(f) * 10, 1e4, np.inf]),
        xtol=1e-15,
        ftol=1e-15,
    )
    return float(result.x[0]), float(result.x[1]), float(result.x[2])


__all__ = [
    "RLCFit",
    "VectorFit",
    "differential_impedance",
    "fit_rlc",
    "initial_guess_rlc",
    "z_rlc",
]
