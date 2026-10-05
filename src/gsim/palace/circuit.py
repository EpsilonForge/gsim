"""Palace AC circuit synthesis (AdaptiveCircuitSynthesis) results parser.

Palace >= 0.17 can synthesize a lumped L/R/C circuit from the reduced-order
model of an adaptive driven simulation (``AdaptiveCircuitSynthesis: true`` in
the ``Solver/Driven`` section). The synthesized matrices are written next to
the S-parameters as ``rom-*.csv`` files:

- ``rom-Linv-re.csv`` / ``rom-Linv-im.csv`` — inverse inductance L^-1 [1/H]
- ``rom-Rinv-re.csv`` / ``rom-Rinv-im.csv`` — inverse resistance R^-1 [S]
- ``rom-C-re.csv`` / ``rom-C-im.csv`` — capacitance C [F]
- ``rom-orthogonalization-matrix-R.csv`` — Gram-Schmidt R factor
- ``rom-port-reference.csv`` — per-port reference Y_ref / Z_ref
- ``rom-portload-<label>-{Linv,Rinv,C}-{re,im}.csv`` — per-port load blocks
- ``rom-coupled-S.csv`` — S-parameters reconstructed from the circuit
- ``rom-eigenvalues.csv`` — synthesized-circuit eigenfrequency estimates

Each matrix file is square with node labels as the header row. Nodes are
ordered: lumped ports (``port_<idx>_re``), wave ports
(``waveport_<idx>_re/im``), synthesized interior nodes (``sample_*``), and —
for frequency-dependent boundary conditions — auxiliary states
(``<prefix>_p<k>d<j>``).

The circuit admittance pencil in SI units is::

    Y(omega) = L^-1 / (iomega) + R^-1 + iomega · C

with omega in rad/s. Port rows/columns carry the physical port terminal
admittance (voltage in V, current in A), so the external port admittance is
obtained by eliminating interior nodes with a Schur complement.

Usage::

    from gsim.palace.circuit import load_circuit_synthesis

    circuit = load_circuit_synthesis(results)  # sim.run() output or dir
    Y = circuit.Y(results.freq * 1e9)  # (nf, N, N) admittance
    Y_dev = circuit.port_admittance(f, subtract_port_loads=True)  # bare device
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from gsim.palace.fitting import RLCFit, VectorFit, differential_impedance, fit_rlc
from gsim.palace.parameters import y_to_s

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

_PORT_NODE_RE = re.compile(r"^port_(\d+)_re$")
_WAVEPORT_NODE_RE = re.compile(r"^waveport_(\d+)_(re|im)$")
_AUX_NODE_RE = re.compile(
    r"^(?:waveport_\d+|farfield|surfsigma_\d+|rationalz_\d+)_p\d+d\d+$"
)

# Minimum Palace version with lumped-port circuit synthesis (rom-*.csv output).
MIN_PALACE_VERSION = "0.17.0"


def _read_matrix_csv(path: Path) -> tuple[list[str], NDArray]:
    """Read a Palace rom matrix CSV: header row of node labels, then values."""
    import pandas as pd

    df = pd.read_csv(path)
    labels = [str(c).strip() for c in df.columns]
    return labels, df.to_numpy(dtype=float)


def _node_kind(label: str) -> Literal["port", "waveport", "sample", "aux"]:
    """Classify a synthesized circuit node label."""
    if _PORT_NODE_RE.match(label):
        return "port"
    if _WAVEPORT_NODE_RE.match(label):
        return "waveport"
    if _AUX_NODE_RE.match(label):
        return "aux"
    return "sample"


class CircuitSynthesis:
    """Synthesized L/R/C circuit from a Palace adaptive driven simulation.

    Attributes:
        nodes: Ordered node labels (ports first, then interior/aux nodes).
        L_inv: Inverse inductance matrix [1/H] (complex, N x N).
        R_inv: Inverse resistance matrix [S] (complex, N x N; zeros if lossless).
        C: Capacitance matrix [F] (complex, N x N).
        port_labels: Labels of the port terminal nodes.
        port_indices: Row/column indices of the port nodes in the matrices.
        port_loads: Per-port load matrices, ``{label: {"L_inv": ..., "R_inv": ...,
            "C": ...}}`` — the termination each port adds to the synthesized
            circuit. Subtract these to recover the bare device.
        orth_R: Gram-Schmidt R factor of the circuit modes (or ``None``).
        port_reference: ``rom-port-reference.csv`` rows (or ``None``).
        coupled_s: ``rom-coupled-S.csv`` rows (or ``None``).
        eigenvalues: ``rom-eigenvalues.csv`` rows (or ``None``).
        files: Mapping of parsed file name -> path.
    """

    def __init__(
        self,
        *,
        nodes: list[str],
        L_inv: NDArray,
        R_inv: NDArray,
        C: NDArray,
        port_loads: dict[str, dict[str, NDArray]] | None = None,
        orth_R: NDArray | None = None,
        port_reference: list[dict[str, str]] | None = None,
        coupled_s: list[dict[str, str]] | None = None,
        eigenvalues: list[dict[str, str]] | None = None,
        files: dict[str, Path] | None = None,
        port_names: dict[str, str] | None = None,
    ) -> None:
        """Create from parsed matrices (prefer :func:`load_circuit_synthesis`)."""
        n = len(nodes)
        for name, mat in (("L_inv", L_inv), ("R_inv", R_inv), ("C", C)):
            if mat.shape != (n, n):
                msg = f"{name} has shape {mat.shape}, expected ({n}, {n})"
                raise ValueError(msg)
        self.nodes = nodes
        self.L_inv = L_inv
        self.R_inv = R_inv
        self.C = C
        self.port_loads = port_loads or {}
        self.orth_R = orth_R
        self.port_reference = port_reference
        self.coupled_s = coupled_s
        self.eigenvalues = eigenvalues
        self.files = files or {}
        self.port_names: dict[str, str] = port_names or {}

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------

    @property
    def port_labels(self) -> list[str]:
        """Labels of the lumped/wave port terminal nodes."""
        return [n for n in self.nodes if _node_kind(n) in ("port", "waveport")]

    @property
    def port_indices(self) -> list[int]:
        """Row/column indices of the port terminal nodes."""
        return [
            i for i, n in enumerate(self.nodes) if _node_kind(n) in ("port", "waveport")
        ]

    @property
    def internal_indices(self) -> list[int]:
        """Row/column indices of interior (sample + aux) nodes."""
        return [
            i for i, n in enumerate(self.nodes) if _node_kind(n) in ("sample", "aux")
        ]

    def index(self, label: str) -> int:
        """Return the matrix row/column index of *label*."""
        try:
            return self.nodes.index(label)
        except ValueError:
            msg = f"Node {label!r} not in synthesized circuit nodes: {self.nodes}"
            raise KeyError(msg) from None

    # ------------------------------------------------------------------
    # Circuit evaluation
    # ------------------------------------------------------------------

    def _pencil(self, f: NDArray, *, subtract_port_loads: bool = False) -> NDArray:
        """Assemble Y(omega) = L^-1/(iomega) + R^-1 + iomega*C (frequencies in Hz).

        When *subtract_port_loads* is set, the per-port termination blocks
        (``rom-portload-*``) are removed, leaving the bare device pencil —
        the convention Palace documents for cascading and re-termination.
        """
        f = np.atleast_1d(np.asarray(f, dtype=float))
        omega = 2.0 * np.pi * f[:, None, None]
        Y = self.L_inv[None, :, :] / (1j * omega) + self.R_inv[None, :, :]
        Y = Y + 1j * omega * self.C[None, :, :]
        if subtract_port_loads and self.port_loads:
            zero = np.zeros_like(self.L_inv)
            for loads in self.port_loads.values():
                Y = Y - (
                    loads.get("L_inv", zero)[None, :, :] / (1j * omega)
                    + loads.get("R_inv", zero)[None, :, :]
                    + 1j * omega * loads.get("C", zero)[None, :, :]
                )
        return Y

    def Y(self, f: NDArray, *, subtract_port_loads: bool = False) -> NDArray:  # noqa: N802
        """Full synthesized admittance pencil, shape ``(nf, N, N)`` [S].

        Named ``Y`` to match the admittance convention
        ``Y(omega) = L^-1/(iomega) + R^-1 + iomega·C``.
        """
        return self._pencil(f, subtract_port_loads=subtract_port_loads)

    def port_admittance(
        self,
        f: NDArray,
        *,
        subtract_port_loads: bool = False,
    ) -> NDArray:
        """External port admittance seen at the synthesized circuit ports.

        Eliminates interior (sample + auxiliary) nodes via a Schur
        complement, leaving the ``(nf, Np, Np)`` admittance at the port
        terminals [S].

        Args:
            f: Frequencies in Hz.
            subtract_port_loads: Also subtract the per-port termination
                blocks (``rom-portload-*``) so the result is the bare device
                admittance (e.g. without the 50 Ohm port resistor).
        """
        Y = self._pencil(f, subtract_port_loads=subtract_port_loads)
        p_idx = self.port_indices
        i_idx = self.internal_indices
        Yp = Y[:, p_idx, :][:, :, p_idx]
        if i_idx:
            Yii = Y[:, i_idx, :][:, :, i_idx]
            Ypi = Y[:, p_idx, :][:, :, i_idx]
            Yip = Y[:, i_idx, :][:, :, p_idx]
            Yp = Yp - Ypi @ np.linalg.solve(Yii, Yip)
        return Yp

    def port_impedance(
        self, f: NDArray, *, subtract_port_loads: bool = True
    ) -> NDArray:
        """Port impedance matrix ``(nf, Np, Np)`` [Ohm] of the bare device."""
        Y = self.port_admittance(f, subtract_port_loads=subtract_port_loads)
        return np.linalg.inv(Y)

    def s_parameters(
        self,
        f: NDArray,
        z0: float | NDArray | None = None,
        *,
        subtract_port_loads: bool = True,
    ) -> NDArray:
        """S-parameters of the synthesized circuit at its port terminals.

        The bare-device port impedance (port loads subtracted by default) is
        referenced to *z0* via the power-wave relation
        ``S = (Z - Z0)(Z + Z0)^-1``.

        Args:
            f: Frequencies in Hz.
            z0: Reference impedance(s) [Ohm]. Defaults to the per-port
                termination read from the ``rom-portload`` resistance blocks
                (e.g. 50 Ohm), falling back to 50 Ohm when unavailable.
            subtract_port_loads: Subtract port loads before computing S
                (default, gives the device S-parameters).

        Returns:
            Complex S-parameter array of shape ``(nf, Np, Np)``.
        """
        Y = self.port_admittance(f, subtract_port_loads=subtract_port_loads)
        n_p = Y.shape[-1]
        if z0 is None:
            z0 = self.port_reference_impedances(default=50.0)
        z0 = np.asarray(z0, dtype=float) * np.ones(n_p)
        # Power-wave S from the admittance, valid for real diagonal Z0 and
        # equal to (Z - Z0)(Z + Z0)^-1 without inverting (possibly singular) Y.
        return y_to_s(Y, z0=z0)

    def port_reference_impedances(self, default: float = 50.0) -> NDArray:
        """Per-port real reference impedance from the ``rom-portload`` data.

        Falls back to *default* for ports without a resistive load.
        """
        z0 = np.full(len(self.port_labels), float(default))
        for k, label in enumerate(self.port_labels):
            load = self.port_loads.get(label, {})
            r_inv = load.get("R_inv")
            if isinstance(r_inv, np.ndarray):
                value = r_inv.diagonal()[self.index(label)].real
                if value > 0:
                    z0[k] = 1.0 / value
        return z0

    def fit_rlc(
        self,
        f: NDArray,
        *,
        model: Literal["rlc1p", "vector_fit"] = "rlc1p",
        subtract_port_loads: bool = True,
        z0: float | NDArray | None = None,
        steps: int = 1000,
        learning_rate: float = 0.05,
        solver: Literal["auto", "jax", "scipy"] = "auto",
        n_poles_real: int | None = 3,
        n_poles_cmplx: int = 3,
        enforce_passivity: bool = False,
        target_error: float = 1e-2,
    ) -> RLCFit | VectorFit:
        """Fit a circuit model to the exported circuit's terminal response.

        The exported circuit is an optional *model source*: evaluating the
        bare-device port admittance (interior nodes eliminated, port loads
        subtracted) and fitting yields compact parameters without touching
        the EM solver output. Two models are available:

        - ``model="rlc1p"``: one-pole (R, L, C, f0, Q) fit of the
          differential impedance (default). For a one-port circuit the
          driving-point impedance Z11 is used directly.
        - ``model="vector_fit"``: multi-pole rational model (scikit-rf
          VectorFitting) of the port S-parameters, with passivity test /
          enforcement.

        Args:
            f: Frequencies in Hz (e.g. the simulation sweep grid).
            model: ``"rlc1p"`` or ``"vector_fit"``.
            subtract_port_loads: De-embed the port terminations first
                (default) so the fit describes the bare device.
            z0: Reference impedance for the vector fit; defaults to the
                per-port values from the ``rom-portload`` data.
            steps: Optimizer iterations (one-pole, JAX solver only).
            learning_rate: Adam learning rate (one-pole, JAX solver only).
            solver: One-pole solver: ``"auto"`` (JAX if installed, else
                scipy), ``"jax"`` or ``"scipy"``.
            n_poles_real: Vector-fit real poles; ``None`` runs skrf's
                ``auto_fit`` loop with ``target_error``.
            n_poles_cmplx: Vector-fit complex pole pairs.
            enforce_passivity: Run ``passivity_enforce`` after the vector
                fit if the model fails the passivity test.
            target_error: Target RMS error for the ``auto_fit`` loop.

        Returns:
            :class:`~gsim.palace.fitting.RLCFit` (one-pole) or
            :class:`~gsim.palace.fitting.VectorFit` (vector fit).
        """
        if model == "rlc1p":
            z = differential_impedance(
                self.port_impedance(f, subtract_port_loads=subtract_port_loads)
            )
            return fit_rlc(
                f,
                z,
                solver=solver,
                steps=steps,
                learning_rate=learning_rate,
            )
        if model == "vector_fit":
            ref_z0 = z0 if z0 is not None else self.port_reference_impedances()
            s = self.s_parameters(
                f,
                z0=ref_z0,
                subtract_port_loads=subtract_port_loads,
            )
            return fit_rlc(
                f,
                s=s,
                model="vector_fit",
                z0=ref_z0,
                n_poles_real=n_poles_real,
                n_poles_cmplx=n_poles_cmplx,
                enforce_passivity=enforce_passivity,
                target_error=target_error,
            )
        raise ValueError(
            f"unknown fit model {model!r}; expected 'rlc1p' or 'vector_fit'"
        )

    # ------------------------------------------------------------------
    # Convenience readouts
    # ------------------------------------------------------------------

    @property
    def eigenfrequencies(self) -> NDArray:
        """Real parts of the estimated eigenfrequencies [Hz] of the circuit."""
        if not self.eigenvalues:
            return np.empty(0)
        try:
            import pandas as pd

            df = pd.DataFrame(self.eigenvalues)
            col = next((c for c in df.columns if c.strip().startswith("Re{f}")), None)
            if col is None:
                return np.empty(0)
            return df[col].to_numpy(dtype=float) * 1e9
        except Exception:  # pragma: no cover - defensive
            return np.empty(0)

    def __repr__(self) -> str:
        """Return concise object representation."""
        n_int = len(self.internal_indices)
        return (
            f"CircuitSynthesis(nodes={len(self.nodes)} "
            f"[{len(self.port_labels)} ports, {n_int} interior], "
            f"files={len(self.files)})"
        )


def load_circuit_synthesis(
    source: str | Path | dict,
    *,
    port_map: dict[int, str] | None = None,
) -> CircuitSynthesis:
    """Load Palace circuit-synthesis (``rom-*.csv``) results.

    Args:
        source: Results dict from ``sim.run()`` / ``run_local()``, an
            :class:`~gsim.palace.results.SParams` object, or a directory
            containing the Palace output files.
        port_map: Optional ``{palace_port_index: name}`` mapping used to
            attach port names to the parsed port nodes.

    Returns:
        Parsed :class:`CircuitSynthesis`.

    Raises:
        FileNotFoundError: If no ``rom-*.csv`` files are found.
    """
    files = _resolve_rom_files(source)
    if not files:
        msg = (
            "No Palace circuit-synthesis files (rom-*.csv) found. Run the "
            "simulation with adaptive sweep and circuit_synthesis=True."
        )
        raise FileNotFoundError(msg)

    first = files.get("rom-Linv-re.csv") or files.get("rom-C-re.csv")
    if first is None:  # pragma: no cover - rom-Linv is always written
        msg = "rom-Linv-re.csv missing from circuit-synthesis output"
        raise FileNotFoundError(msg)
    nodes, _ = _read_matrix_csv(first)
    n = len(nodes)

    def _load_complex(prefix: str) -> NDArray:
        mat = np.zeros((n, n), dtype=complex)
        re_path = files.get(f"{prefix}-re.csv")
        im_path = files.get(f"{prefix}-im.csv")
        if re_path is not None:
            labels, values = _read_matrix_csv(re_path)
            if [str(c).strip() for c in labels] != nodes:
                logger.warning("%s node labels differ from rom-Linv-re", prefix)
            mat += values
        if im_path is not None:
            mat += 1j * _read_matrix_csv(im_path)[1]
        return mat

    L_inv = _load_complex("rom-Linv")
    R_inv = _load_complex("rom-Rinv")
    C = _load_complex("rom-C")

    # Per-port load blocks: rom-portload-<label>-{Linv,Rinv,C}-{re,im}.csv
    port_loads: dict[str, dict[str, NDArray]] = {}
    for name in files:
        m = re.match(r"^rom-portload-(.+)-(Linv|Rinv|C)-(re|im)\.csv$", name)
        if m is None:
            continue
        label, part, sign = m.group(1), m.group(2), m.group(3)
        _, values = _read_matrix_csv(files[name])
        key = {"Linv": "L_inv", "Rinv": "R_inv", "C": "C"}[part]
        slot = port_loads.setdefault(label, {})
        base = slot.get(key)
        if base is None:
            base = np.zeros((n, n), dtype=complex)
            slot[key] = base
        if sign == "re":
            slot[key] = base + values
        else:
            slot[key] = base + 1j * values

    orth_R = None
    if "rom-orthogonalization-matrix-R.csv" in files:
        orth_R = _read_matrix_csv(files["rom-orthogonalization-matrix-R.csv"])[1]

    port_reference = _read_table(files.get("rom-port-reference.csv"))
    coupled_s = _read_table(files.get("rom-coupled-S.csv"))
    eigenvalues = _read_table(files.get("rom-eigenvalues.csv"))

    port_names: dict[str, str] = {}
    if port_map:
        for label in nodes:
            m = _PORT_NODE_RE.match(label)
            if m is not None and int(m.group(1)) in port_map:
                port_names[label] = port_map[int(m.group(1))]

    return CircuitSynthesis(
        nodes=nodes,
        L_inv=L_inv,
        R_inv=R_inv,
        C=C,
        port_loads=port_loads,
        orth_R=orth_R,
        port_reference=port_reference,
        coupled_s=coupled_s,
        eigenvalues=eigenvalues,
        files=files,
        port_names=port_names,
    )


def _read_table(path: Path | None) -> list[dict[str, str]] | None:
    """Read a tabular (non-matrix) rom CSV into row dicts."""
    if path is None:
        return None
    import csv

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return [{k or "": v or "" for k, v in row.items()} for row in reader]


def _resolve_rom_files(source: str | Path | dict) -> dict[str, Path]:
    """Collect ``rom-*.csv`` files from a results dict or directory."""
    files: dict[str, Path] = {}
    if isinstance(source, dict):
        candidates = list(source.values())
    elif isinstance(source, (str, Path)):
        base = Path(source)
        candidates = sorted(base.rglob("rom-*.csv")) if base.is_dir() else [base]
    else:
        # SParams-like object carrying a files mapping
        candidates = list(getattr(source, "files", {}).values())

    for value in candidates:
        if value is None:
            continue
        path = Path(value)
        if path.is_file() and path.name.startswith("rom-") and path.suffix == ".csv":
            files.setdefault(path.name, path)
    return files


__all__ = ["MIN_PALACE_VERSION", "CircuitSynthesis", "load_circuit_synthesis"]
