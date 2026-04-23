"""
Inductor Compact Model: Simulation Data → Polynomial Fit → SAX Circuit
=======================================================================
Workflow (as in the diagram):
  1. Load S-parameter simulation data (from gsim/Palace results)
  2. Fit polynomial models to each S-parameter
  3. Register compact model in SAX and simulate a circuit
"""

import numpy as np
import jax.numpy as jnp
import skrf as rf
import sax
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0  –  Build the Network from raw results (unchanged from your setup)
# ─────────────────────────────────────────────────────────────────────────────

def build_network(results) -> rf.Network:
    """Convert gsim Palace results → scikit-rf Network."""
    f = results.freq * 1e9          # GHz → Hz
    ports = results.port_names
    n = len(ports)
    S = np.zeros((len(f), n, n), dtype=complex)
    for i, pi in enumerate(ports):
        for j, pj in enumerate(ports):
            S[:, i, j] = results[(pi, pj)].complex
    return rf.Network(f=f, s=S, f_unit="hz")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1  –  Polynomial Fit  (generalised for any N-port)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SParamPolyModel:
    """
    Stores polynomial coefficients for every (i, j) entry of an S-matrix.
    All coefficients are JAX arrays so the model is differentiable end-to-end.
    """
    port_names: list[str]
    degree: int
    # keys: (i, j) → (jnp.array re_coeffs, jnp.array im_coeffs)
    coeffs: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def fit(self, f: np.ndarray, S: np.ndarray) -> "SParamPolyModel":
        """
        Fit degree-`degree` polynomials to Re and Im of every S[i,j].

        Parameters
        ----------
        f : (N,) float  –  frequency array in Hz
        S : (N, n, n) complex  –  S-matrix vs frequency
        """
        n = S.shape[1]
        for i in range(n):
            for j in range(n):
                re = np.polyfit(f, np.real(S[:, i, j]), self.degree)
                im = np.polyfit(f, np.imag(S[:, i, j]), self.degree)
                self.coeffs[(i, j)] = (jnp.array(re), jnp.array(im))
        return self

    # ------------------------------------------------------------------
    def eval(self, f_scalar) -> dict[tuple[str, str], complex]:
        """
        Evaluate the model at a single frequency (Hz).
        Returns a SAX-compatible dict keyed by (port_name_i, port_name_j).
        """
        result = {}
        for (i, j), (re_c, im_c) in self.coeffs.items():
            s_val = jnp.polyval(re_c, f_scalar) + 1j * jnp.polyval(im_c, f_scalar)
            pi = self.port_names[i]
            pj = self.port_names[j]
            result[(pi, pj)] = s_val
        return result

    # ------------------------------------------------------------------
    def plot_fit(self, f: np.ndarray, S: np.ndarray, pairs=None):
        """Quick sanity-check: overlay raw data and polynomial fit."""
        if pairs is None:
            pairs = list(self.coeffs.keys())
        f_plot = np.linspace(f.min(), f.max(), 500)
        n_pairs = len(pairs)
        fig, axes = plt.subplots(n_pairs, 2, figsize=(10, 3 * n_pairs), squeeze=False)
        fig.suptitle("Polynomial Fit vs. Simulation Data", fontsize=13)
        for row, (i, j) in enumerate(pairs):
            pi, pj = self.port_names[i], self.port_names[j]
            re_c, im_c = self.coeffs[(i, j)]
            re_fit = np.polyval(np.array(re_c), f_plot)
            im_fit = np.polyval(np.array(im_c), f_plot)
            for col, (label, raw, fit) in enumerate([
                ("Real", np.real(S[:, i, j]), re_fit),
                ("Imag", np.imag(S[:, i, j]), im_fit),
            ]):
                ax = axes[row][col]
                ax.scatter(f / 1e9, raw, s=6, alpha=0.5, label="sim data")
                ax.plot(f_plot / 1e9, fit, lw=1.8, label=f"poly deg={self.degree}")
                ax.set_title(f"S({pi},{pj}) – {label}")
                ax.set_xlabel("Frequency (GHz)")
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2  –  Build a SAX-compatible callable from the poly model
# ─────────────────────────────────────────────────────────────────────────────

def make_sax_model(poly_model: SParamPolyModel) -> Callable:
    """
    Wrap a SParamPolyModel so SAX can call it with f= as a keyword argument.

    The returned function accepts either a scalar or a JAX array of frequencies
    and is jit-compilable.
    """
    def _model(f=10e9):
        # vmap over frequency array transparently
        def _eval_single(f_val):
            sdict = poly_model.eval(f_val)
            # flatten to a dict of scalars (SAX requirement)
            return sdict

        # If f is an array, evaluate element-wise via Python-level loop
        # (SAX handles the vectorisation internally via its own vmap)
        return poly_model.eval(f)

    _model.__name__ = "inductor_poly_model"
    return _model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3  –  Circuit definition + simulation
# ─────────────────────────────────────────────────────────────────────────────

def build_series_circuit(model_name: str, n_stages: int = 2) -> dict:
    """
    Wire `n_stages` identical two-port components in series.

    port_names inside the model must be ["port1", "port2"].
    """
    instances = {f"L{k+1}": model_name for k in range(n_stages)}
    connections = {}
    for k in range(n_stages - 1):
        connections[f"L{k+1},port2"] = f"L{k+2},port1"
    ports = {"in": "L1,port1", "out": f"L{n_stages},port2"}
    return {"instances": instances, "connections": connections, "ports": ports}


def simulate_circuit(
    circuit_fn,
    f_start: float = 10e9,
    f_stop: float = 100e9,
    n_points: int = 500,
) -> tuple[np.ndarray, dict]:
    """Run a frequency sweep and return (f_array, S_dict)."""
    f_sweep = jnp.linspace(f_start, f_stop, n_points)
    S_total = circuit_fn(f=f_sweep)
    return np.array(f_sweep), S_total


def plot_circuit_results(f_sweep, S_total, port_pair=("out", "in"), title="SAX Circuit Simulation"):
    S_mag = jnp.abs(S_total[port_pair])
    # guard against log(0)
    S_db  = 20 * jnp.log10(jnp.where(S_mag > 1e-30, S_mag, 1e-30))
    plt.figure(figsize=(8, 4))
    plt.plot(f_sweep / 1e9, np.array(S_db), color="#5C6BC0", lw=2,
             label=f"|S({port_pair[0]},{port_pair[1]})| – series inductors")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("|S| (dB)")
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WORKFLOW  –  plug your `results` object here
# ─────────────────────────────────────────────────────────────────────────────

def run_inductor_workflow(results, degree: int = 5, n_stages: int = 2):
    """
    End-to-end workflow:
      results  –  gsim/Palace SimulationResults object
      degree   –  polynomial degree for the S-param fit
      n_stages –  how many inductors to wire in series
    """

    # ── 0. Build rf.Network ──────────────────────────────────────────────────
    ntwk = build_network(results)
    f = ntwk.f                          # Hz
    S = ntwk.s                          # (N, n, n) complex
    port_names = ["port1", "port2"]     # adjust if your results use different names

    # ── 1. Fit polynomial model ──────────────────────────────────────────────
    print(f"Fitting degree-{degree} polynomials to {S.shape[1]}×{S.shape[1]} S-matrix …")
    model = SParamPolyModel(port_names=port_names, degree=degree)
    model.fit(f, S)

    # Sanity-check: show fit quality for S11 and S21
    model.plot_fit(f, S, pairs=[(0, 0), (1, 0), (1, 1)])

    # ── 2. Wrap for SAX ─────────────────────────────────────────────────────
    sax_fn = make_sax_model(model)
    models = {"my_inductor": sax_fn}

    # Quick scalar test
    test_f   = 60e9
    test_out = sax_fn(f=test_f)
    print(f"\nModel self-test at {test_f/1e9:.0f} GHz:")
    for k, v in test_out.items():
        print(f"  S{k} = {v:.4f}")

    # ── 3. Build circuit & simulate ──────────────────────────────────────────
    netlist = build_series_circuit("my_inductor", n_stages=n_stages)
    circuit_fn, info = sax.circuit(netlist=netlist, models=models)

    f_sweep, S_total = simulate_circuit(circuit_fn, f_start=10e9, f_stop=100e9)

    # ── 4. Plot ──────────────────────────────────────────────────────────────
    plot_circuit_results(
        f_sweep, S_total,
        port_pair=("out", "in"),
        title=f"SAX: {n_stages}× Inductor Series Chain",
    )

    return model, circuit_fn, S_total


# ─────────────────────────────────────────────────────────────────────────────
# Example usage (comment out when importing as a module)
# ─────────────────────────────────────────────────────────────────────────────
# model, circuit_fn, S_total = run_inductor_workflow(results, degree=5, n_stages=2)