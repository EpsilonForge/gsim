"""
inductor_sweep.py
-----------------
Sweep and optimization utilities for IHP SG13G2 spiral inductor EM simulation.

Entry points:
    sweep_inductor(...)    — exhaustive parameter sweep
    optimize_inductor(...) — Powell optimizer, maximizes Q at a target frequency

PDK note:
    gf.components.inductor currently ignores `turns` in geometry (stores it as
    metadata only). All formulas are forward-compatible and will work once fixed.

Usage in notebook:
    from inductor_sweep import sweep_inductor, optimize_inductor
"""

import itertools
import shutil

import numpy as np
import skrf as rf
import gdsfactory as gf
from scipy.optimize import minimize
from gsim.palace import DrivenSim


# ── IHP SG13G2 physical constants ────────────────────────────────────────────

IHP_SUBSTRATE_THICKNESS = 180.0   # µm
IHP_AIR_ABOVE           = 200.0    # µm


# ── Guard ring margin ────────────────────────────────────────────────────────

def compute_margin_inner(width: float, space: float, diameter: float, turns: int = 1) -> float:
    """
    Auto-compute guard ring inner margin from spiral geometry.
 
    Rule:
        outer_extent = diameter/2 + turns * (width + space)
        margin_inner = -max(5 * width, 0.5 * outer_extent)
 
    Returns
    -------
    margin_inner : float (µm, negative) — distance inward from bbox edge
    """
    outer_extent = diameter / 2.0 + turns * (width + space)
    return -max(5.0 * width, 0.5 * outer_extent)


def _min_diameter(width: float, space: float, turns: int) -> float:
    """PDK minimum inner diameter: 2*turns*(width+space) + 4*width."""
    return 2 * turns * (width + space) + 4 * width


# ── Component builder ────────────────────────────────────────────────────────

def build_component(width: float = 2.0, space: float = 2.1, diameter: float = 50.0, turns: int = 1):
    """
    Build spiral inductor + Metal1drawing guard ring.
    diameter is clamped to PDK minimum automatically.

    Returns
    -------
    comp         : gdsfactory Component
    """
    diameter = max(diameter, _min_diameter(width, space, turns))

    c = gf.components.inductor(
        width=width,
        space=space,
        diameter=diameter,
        turns=turns,
        layer_metal='TopMetal2drawing',
        layer_inductor='INDdrawing',
        layer_metal_pin='TopMetal2drawing',
        layers_no_fill=('NoMetFillerdrawing',)
    ).copy()

    bbox         = c.bbox()
    xmin, ymin   = bbox.left,  bbox.bottom
    xmax, ymax   = bbox.right, bbox.top

    margin_inner = compute_margin_inner(width, space, diameter, turns)
    margin_outer = 0.0

    ol,   oright = xmin - margin_outer, xmax + margin_outer
    ob,   ot     = ymin - margin_outer, ymax + margin_outer
    il,   _      = xmin - margin_inner, xmax + margin_inner

    w_v  = il - ol
    h_h  = ot - (ymax + margin_inner)
    over = 0.5

    c.add_ref(gf.components.rectangle(
        size=(w_v + over, ot - ob), layer='Metal1drawing', centered=True)
    ).move((ol + w_v / 2 + over / 2, (ot + ob) / 2))

    c.add_ref(gf.components.rectangle(
        size=(w_v + over, ot - ob), layer='Metal1drawing', centered=True)
    ).move((oright - w_v / 2 - over / 2, (ot + ob) / 2))

    c.add_ref(gf.components.rectangle(
        size=(oright - ol, h_h + over), layer='Metal1drawing', centered=True)
    ).move(((oright + ol) / 2, ot - h_h / 2 - over / 2))

    c.add_ref(gf.components.rectangle(
        size=(oright - ol, h_h + over), layer='Metal1drawing', centered=True)
    ).move(((oright + ol) / 2, ob + h_h / 2 + over / 2))

    return c


# ── Simulation runner ────────────────────────────────────────────────────────

def run_sim(component, run_dir: str, fmin: float = 10e9, fmax: float = 200e9, num_points: int = 50):
    """
    Run Palace DrivenSim and return post-processed Z-parameter results.
 
    Returns
    -------
    f      : np.ndarray        — frequency array (Hz)
    Z_diff : np.ndarray complex — differential impedance
    L_diff : np.ndarray        — differential inductance (H)
    Q_diff : np.ndarray        — differential quality factor
    """
    sim = DrivenSim()
    sim.set_output_dir(run_dir)
    sim.set_geometry(component)
    sim.set_stack(
        substrate_thickness=IHP_SUBSTRATE_THICKNESS,
        air_above=IHP_AIR_ABOVE,
        include_substrate=True,
    )
    sim.add_port("P1", from_layer="metal1", to_layer="topmetal2",geometry="via", excited=True)
    sim.add_port("P2", from_layer="metal1", to_layer="topmetal2",geometry="via", excited=True)
    sim.set_driven(fmin=fmin, fmax=fmax, num_points=num_points)
    sim.mesh(preset="default", margin=50, refined_mesh_size=1.5)

    try:
        # parent_dir=run_dir ensures sim-data-palace-* lands inside run_dir, so a single rmtree cleans up everything in one shot.
        results = sim.run(parent_dir=run_dir)
 
        f     = results.freq * 1e9
        w     = 2 * np.pi * f
        ports = results.port_names
        n     = len(ports)
        S     = np.zeros((len(f), n, n), dtype=complex)
        for i, pi in enumerate(ports):
            for j, pj in enumerate(ports):
                S[:, i, j] = results[(pi, pj)].complex
 
        ntwk   = rf.Network(f=f, s=S, f_unit='hz')
        Z      = ntwk.z
        Z_diff = Z[:, 0, 0] + Z[:, 1, 1] - Z[:, 0, 1] - Z[:, 1, 0]
        L_diff = np.imag(Z_diff) / w
        Q_diff = np.imag(Z_diff) / np.real(Z_diff)
 
        return f, Z_diff, L_diff, Q_diff
 
    finally:
        # Always runs — even if simulation crashed.
        # Deletes run_dir and everything inside it (including sim-data-palace-*).
        shutil.rmtree(run_dir, ignore_errors=True)


# ── Objective function ───────────────────────────────────────────────────────

def compute_objective(f, L_diff, Q_diff, f_target=None):
    """
    Scalar objective: Q at f_target (or peak Q if f_target is None).

    Parameters
    ----------
    f        : np.ndarray — frequency (Hz)
    L_diff   : np.ndarray — inductance (H)
    Q_diff   : np.ndarray — quality factor
    f_target : float | None — evaluation frequency (Hz); None = peak Q

    Returns
    -------
    score       : float — Q value (higher is better; -inf if invalid)
    Q_at_target : float — Q at f_target or peak Q
    f_Q_peak    : float — frequency of absolute peak Q (Hz)
    L_low_f     : float — inductance at lowest valid frequency (H)
    """
    mask = L_diff > 0
    if not mask.any():
        return -np.inf, np.nan, np.nan, np.nan

    Q_masked = Q_diff[mask]
    f_masked = f[mask]
    L_masked = L_diff[mask]

    idx_peak = np.argmax(Q_masked)
    f_Q_peak = f_masked[idx_peak]
    L_low_f  = L_masked[0]

    if f_target is not None:
        if f_target < f_masked[0] or f_target > f_masked[-1]:
            return -np.inf, np.nan, f_Q_peak, L_low_f
        Q_at_target = float(np.interp(f_target, f_masked, Q_masked))
    else:
        Q_at_target = Q_masked[idx_peak]

    return Q_at_target, Q_at_target, f_Q_peak, L_low_f


# ── Sweep ────────────────────────────────────────────────────────────────────
 
def sweep_inductor(
    sweep_params: dict,
    fixed_params: dict | None = None,
    fmin: float = 10e9,
    fmax: float = 200e9,
    num_points: int = 50,
    base_run_dir: str = "runs/sweep",
    f_target: float | None = None,
):
    """
    Exhaustive sweep over all combinations in sweep_params (cartesian product).
    Prints a ranked summary table.
 
    Parameters
    ----------
    sweep_params : dict  — {param: [values]}, e.g. {'diameter': [40, 60, 80]}
                   Multiple keys → cartesian product of all combinations.
    fixed_params : dict  — fixed kwargs for build_component
    fmin / fmax  : float — simulation frequency range (Hz)
    num_points   : int   — frequency points per simulation
    base_run_dir : str   — base directory for Palace outputs
    f_target     : float | None — frequency for Q evaluation (Hz); None = peak Q
 
    Returns
    -------
    results : dict  — combo_tuple → result dict
    best    : dict  — highest-scoring result
    """
    if fixed_params is None:
        fixed_params = {}
 
    param_names = list(sweep_params.keys())
    param_grid  = list(itertools.product(*sweep_params.values()))
    results     = {}
    f_label     = f"{f_target/1e9:.1f} GHz" if f_target else "peak"
 
    print(f"Sweep: {len(param_grid)} combinations | Q @ {f_label}")
 
    for combo in param_grid:
        kwargs = {**fixed_params, **dict(zip(param_names, combo))}
        label  = ", ".join(f"{k}={v}" for k, v in zip(param_names, combo))
        print(f"\n{'='*60}\n  {label}\n{'='*60}")
 
        comp = build_component(**kwargs)
        run_dir = (base_run_dir + "/"
                   + "_".join(f"{k}{v}" for k, v in zip(param_names, combo)))
 
        f, Z_diff, L_diff, Q_diff = run_sim(
            comp, run_dir, fmin=fmin, fmax=fmax, num_points=num_points)
        score, Q_at_target, f_Q_peak, L_low_f = compute_objective(
            f, L_diff, Q_diff, f_target=f_target)
 
        results[combo] = dict(
            f=f, Z_diff=Z_diff, L_diff=L_diff, Q_diff=Q_diff,
            score=score, Q_at_target=Q_at_target, f_Q_peak=f_Q_peak,
            L_low_f=L_low_f, label=label,
        )
        print(f"  L = {L_low_f*1e12:.1f} pH | "
              f"Q @ {f_label} = {Q_at_target:.2f} | "
              f"peak Q @ {f_Q_peak/1e9:.1f} GHz")
 
    sorted_results = sorted(results.items(),
                            key=lambda x: x[1]['score'], reverse=True)
    print(f"\n{'='*60}\n  SUMMARY  (Q @ {f_label})\n{'='*60}")
    print(f"  {'Parameters':<30} {'L (pH)':>8} {'Q@target':>9} "
          f"{'f_peak (GHz)':>13} {'Score':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*9} {'-'*13} {'-'*8}")
    for _, r in sorted_results:
        print(f"  {r['label']:<30} {r['L_low_f']*1e12:>8.1f} "
              f"{r['Q_at_target']:>9.2f} {r['f_Q_peak']/1e9:>13.1f} "
              f"{r['score']:>8.3f}")
 
    _, best = sorted_results[0]
 
    return results, best
 
 
# ── Powell optimizer ─────────────────────────────────────────────────────────
 
def optimize_inductor(
    f_target: float,
    x0: dict | None = None,
    bounds: dict | None = None,
    fixed_params: dict | None = None,
    fmin: float = 10e9,
    fmax: float = 200e9,
    num_points: int = 50,
    base_run_dir: str = "runs/optimize",
    maxiter: int = 50,
    xtol: float = 0.5,
    ftol: float = 0.1,
):
    """
    Maximize Q at f_target using Powell's method (scipy.optimize.minimize).
 
    Powell iterates over parameter directions one at a time without gradients,
    making it suitable for expensive EM simulation objectives.
    Each function evaluation = one full Palace simulation.
 
    Bounds are enforced by clamping x_vec before each simulation — parameters
    that stray outside are silently pulled back to the boundary.
 
    Parameters
    ----------
    f_target     : float — operating frequency (Hz), e.g. 60e9 for 60 GHz
    x0           : dict  — starting geometry, e.g. {'width': 2.0, 'space': 2.1,
                   'diameter': 50.0}. Default: those values.
    bounds       : dict  — {param: (min, max)} in µm.
                   Default: width (1–10), space (1–10), diameter (20–150).
    fixed_params : dict  — parameters held constant, e.g. {'turns': 1}
    fmin / fmax  : float — simulation frequency range (Hz)
    num_points   : int   — frequency points per simulation
    base_run_dir : str   — each evaluation saved to {base_run_dir}/eval_NNN/
    maxiter      : int   — max Powell iterations
    xtol         : float — parameter convergence tolerance (µm)
    ftol         : float — objective convergence tolerance (Q units)
 
    Returns
    -------
    opt_result : scipy OptimizeResult
    history    : list of dicts, one per evaluation
    best       : dict — evaluation with highest Q score
    """
    if x0 is None:
        x0 = {'width': 2.0, 'space': 2.1, 'diameter': 50.0}
    if bounds is None:
        bounds = {
            'width':    (2.0, 10.0),  # TM2_a = 2.0 µm min width (sg13g2_tech_default.json)
            'space':    (2.0, 10.0),  # TM2_b = 2.0 µm min spacing (TM2_bR = 5.0 µm for wide lines)
            'diameter': (20.0, 150.0),
        }
    if fixed_params is None:
        fixed_params = {}
 
    param_names = list(x0.keys())
    x0_vec      = np.array([x0[k] for k in param_names], dtype=float)
    bounds_lo   = np.array([bounds[k][0] for k in param_names])
    bounds_hi   = np.array([bounds[k][1] for k in param_names])
 
    history    = []
    eval_count = [0]
 
    def _objective(x_vec):
        x_clamped = np.clip(x_vec, bounds_lo, bounds_hi)
        params    = dict(zip(param_names, x_clamped))
        kwargs    = {**fixed_params, **params}
 
        eval_count[0] += 1
        n       = eval_count[0]
        run_dir = f"{base_run_dir}/eval_{n:03d}"
 
        print(f"\n── Eval {n} " + "─" * 40)
        for k, v in params.items():
            print(f"   {k:12s} = {v:.3f} µm")
 
        try:
            comp = build_component(**kwargs)
 
            f, Z_diff, L_diff, Q_diff = run_sim(
                comp, run_dir, fmin=fmin, fmax=fmax, num_points=num_points)
 
            score, Q_at_target, f_Q_peak, L_low_f = compute_objective(
                f, L_diff, Q_diff, f_target=f_target)
 
            print(f"   L = {L_low_f*1e12:.1f} pH | "
                  f"Q @ {f_target/1e9:.1f} GHz = {Q_at_target:.3f} | "
                  f"peak Q @ {f_Q_peak/1e9:.1f} GHz")
 
            history.append(dict(
                eval=n, params=params.copy(),
                score=score, Q_at_target=Q_at_target,
                f_Q_peak=f_Q_peak, L_low_f=L_low_f,
                f=f, Z_diff=Z_diff, L_diff=L_diff, Q_diff=Q_diff,
            ))
 
        except Exception as e:
            print(f"   FAILED: {e}")
            score = -1e6
            history.append(dict(eval=n, params=params.copy(),
                                 score=score, failed=True))
 
        return -score   # Powell minimizes → negate Q to maximize it
 
    # ── Launch Powell ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Powell optimizer")
    print(f"  Objective  : maximize Q @ {f_target/1e9:.1f} GHz")
    print(f"  Parameters : {param_names}")
    print(f"  x0         : {x0}")
    print(f"  bounds     : {bounds}")
    print(f"{'='*60}")
 
    opt_result = minimize(
        _objective,
        x0_vec,
        method='Powell',
        options={
            'maxiter': maxiter,
            'xtol':    xtol,
            'ftol':    ftol,
            'disp':    True,
        },
    )
 
    # ── Best result ──────────────────────────────────────────────────────────
    valid = [h for h in history if not h.get('failed')]
    best  = max(valid, key=lambda h: h['score'])
 
    # ── Convergence summary ───────────────────────────────────────────────────
    print(f"\n  {'Eval':>5} {'Q @ target':>12} {'best so far':>12}")
    print(f"  {'-'*5} {'-'*12} {'-'*12}")
    best_so_far = -np.inf
    for h in valid:
        best_so_far = max(best_so_far, h['score'])
        print(f"  {h['eval']:>5} {h['score']:>12.3f} {best_so_far:>12.3f}")
 
    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULT")
    print(f"{'='*60}")
    print(f"  Converged   : {opt_result.success}")
    print(f"  Message     : {opt_result.message}")
    print(f"  Evaluations : {eval_count[0]}")
    print(f"\n  Best geometry:")
    for k, v in best['params'].items():
        print(f"    {k:12s} = {v:.3f} µm")
    print(f"\n  Q @ {f_target/1e9:.1f} GHz = {best['Q_at_target']:.3f}")
    print(f"  L @ low f      = {best['L_low_f']*1e12:.1f} pH")
 
    return opt_result, history, best