#!/usr/bin/env python
"""Build the live IHP-validation documentation variant."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
import skrf as rf

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gsim.common.benchmarks import ihp_l2n0


def _measured(label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (frequency_ghz, differential L_nH, differential Q)."""
    network = rf.Network(str(ihp_l2n0.MEASURED[label]))
    matrix = network.z
    z = matrix[:, 0, 0] - matrix[:, 0, 1] - matrix[:, 1, 0] + matrix[:, 1, 1]
    l_nh = z.imag / (2 * np.pi * network.f) * 1e9
    q = np.divide(z.imag, z.real, out=np.full_like(z.real, np.nan), where=z.real != 0)
    return network.f / 1e9, l_nh, q


def render_comparison(out: Path) -> dict[str, float]:
    """Render differential L/Q and return 1 GHz metrics."""
    run = ihp_l2n0.load_recorded_run(ihp_l2n0.RUN_ROOT / "l2n0_best")
    f_palace = np.asarray(run.sparams.freq, dtype=float)
    l_palace, q_palace = [], []
    for i in range(len(f_palace)):
        z = ihp_l2n0.differential_impedance_from_s(ihp_l2n0.s_matrix(run.sparams, i))
        l, q = ihp_l2n0.lq_from_impedance(z, f_palace[i] * 1e9)
        l_palace.append(l)
        q_palace.append(q)
    l_palace = np.asarray(l_palace)
    q_palace = np.asarray(q_palace)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    measured = {}
    for label, color in (("die_x2y7", "C0"), ("die_x5y8", "C1")):
        f, l, q = _measured(label)
        measured[label] = (f, l, q)
        axes[0].plot(f, l, color=color, lw=1.0, label=f"measured {label}")
        axes[1].plot(f, q, color=color, lw=1.0, label=f"measured {label}")
    axes[0].plot(f_palace, l_palace, "o-", color="C3", label="Palace")
    axes[1].plot(f_palace, q_palace, "o-", color="C3", label="Palace")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.set(xlabel="frequency (GHz)", xlim=(0, 20))
        ax.legend(fontsize=8)
    axes[0].set(title="Differential inductance", ylabel="L (nH)", ylim=(0, 5))
    axes[1].set(title="Differential quality factor", ylabel="Q", ylim=(0, 25))
    fig.tight_layout()
    fig.savefig(
        out / "l2n0_comparison.png", dpi=160, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)

    def at(f: np.ndarray, values: np.ndarray, target: float) -> float:
        return float(values[int(np.argmin(np.abs(f - target)))])

    rows = {
        "palace_diff_l_1ghz_nh": at(f_palace, l_palace, 1.0),
        "palace_diff_q_1ghz": at(f_palace, q_palace, 1.0),
    }
    for label, (f, l, q) in measured.items():
        rows[f"{label}_diff_l_1ghz_nh"] = at(f, l, 1.0)
        rows[f"{label}_diff_q_1ghz"] = at(f, q, 1.0)
    return rows


def build(
    out: Path, *, notebook_html: Path | None = None, notebook_ipynb: Path | None = None
) -> None:
    """Assemble the variant directory."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rows = render_comparison(out)
    (out / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    experiments = REPO / "nbs" / "data" / "ihp_validation" / "experiments"
    matrix = json.loads((experiments / "results.json").read_text())
    campaign = "".join(
        f"<tr><td>{name}</td><td>{matrix[name]['L_diff_nH']:.3f}</td>"
        f"<td>{matrix[name]['Q_diff']:.3f}</td></tr>"
        for name in (
            "baseline",
            "shell_conformal",
            "ground0_pec",
            "openems_ground",
        )
    )
    for name in ("L_2n0_simplified.gds", "README.md", "provenance.json"):
        source = ihp_l2n0.SOURCE / name
        if source.is_file():
            shutil.copy2(source, out / name)
    notebook_section = ""
    if notebook_html is not None and notebook_html.is_file():
        shutil.copy2(notebook_html, out / "notebook.html")
        links = '<a href="notebook.html">rendered notebook</a>'
        if notebook_ipynb is not None and notebook_ipynb.is_file():
            shutil.copy2(notebook_ipynb, out / "notebook.ipynb")
            links += ' · <a href="notebook.ipynb">executed .ipynb</a>'
        notebook_section = f"<p><strong>Notebook:</strong> {links}</p>"
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>IHP SG13G2 L2n0 — two-port validation audit</title>
<style>
:root {{ color-scheme:light dark; --fg:#1f2328; --muted:#656d76; --bg:#fff; --card:#f6f8fa; --line:#d0d7de; --accent:#0969da; }}
@media (prefers-color-scheme:dark) {{ :root {{ --fg:#e6edf3; --muted:#8b949e; --bg:#0d1117; --card:#161b22; --line:#30363d; --accent:#4493f8; }} }}
body {{ margin:0; font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; color:var(--fg); background:var(--bg); }}
main {{ max-width:1050px; margin:0 auto; padding:36px 24px 56px; }}
h1 {{ margin:0 0 6px; font-size:1.8rem; }} .muted {{ color:var(--muted); }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px; margin:20px 0; }}
img {{ width:100%; border-radius:10px; background:#fff; }}
table {{ border-collapse:collapse; width:100%; font-size:.92rem; }}
th,td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }}
code {{ font-size:.9em; }}
</style></head><body><main>
<p><a href="../">&larr; all variants</a></p>
<h1>IHP SG13G2 Octagon L2n0 — two-port validation audit</h1>
<p class="muted">The executed notebook compares all four complex S-parameters with two pinned
IHP measured dies. The L/Q figure uses the differential impedance extraction
from the pinned IHP script. Layout and measured data are pinned from IHP-Open-PDK
commit <code>9439837</code> (Apache-2.0).</p>
<div class="card"><img src="l2n0_comparison.png" alt="Measured and Palace extracted inductance and Q"></div>
<div class="card"><h2>Differential impedance at 1 GHz</h2>
<table><thead><tr><th>case</th><th>L (nH)</th><th>Q</th></tr></thead><tbody>
<tr><td>Palace</td><td>{rows["palace_diff_l_1ghz_nh"]:.3f}</td><td>{rows["palace_diff_q_1ghz"]:.2f}</td></tr>
<tr><td>die_x2y7</td><td>{rows["die_x2y7_diff_l_1ghz_nh"]:.3f}</td><td>{rows["die_x2y7_diff_q_1ghz"]:.2f}</td></tr>
<tr><td>die_x5y8</td><td>{rows["die_x5y8_diff_l_1ghz_nh"]:.3f}</td><td>{rows["die_x5y8_diff_q_1ghz"]:.2f}</td></tr>
</tbody></table>
<p class="muted">The simulated inductance is about half the measured value.
The older S21-only series formula happened to give 1.921 nH; it assumes an
isolated series element and is misleading for this two-port. At 30 GHz the
measured device is near/past resonance, so a single inductance is not an
agreement metric.</p></div>
{notebook_section}
<div class="card"><p><strong>Complete two-port result:</strong> L2n0 Palace S21 has
8.83-8.86 dB RMS magnitude error and 44-45° RMS phase error across the
recorded 1-50 GHz sweep. S11, S12 and S22 also disagree. The model is not
validated. The notebook also compares public measured L6n2 data with three
GDS2Palace simulations of that separate device at 5, 2 and 1 µm meshes.</p></div>
<div class="card"><h2>Campaign matrix (1 GHz)</h2>
<table><thead><tr><th>variant</th><th>Differential L (nH)</th><th>Differential Q</th></tr></thead><tbody>{campaign}</tbody></table>
<p class="muted">The variants were screened at one frequency. The `gap` case
has one port and is excluded from this two-port table. The `planar_pec` case
produced invalid numerical output. The 1/5/9/30 GHz files used an earlier
passivation model and do not establish frequency convergence. The recorded
mesh error indicator is about 0.18 against a 0.02 tolerance.</p></div>
<div class="card"><h2>Model and source</h2>
<p>Palace sweep: 1-50 GHz, 11 points; an IHP SG13G2 stack approximation with a
280 µm substrate, TopMetal2 winding, TopMetal1 feeds/underpass, and a minimal
ground pad and an unconditioned (Palace-default PMC) outer wall for the
`ground0_pec` variant. The historical name does not indicate a true PEC wall. The 5 µm refined / 70 µm
maximum mesh and Palace Conductivity+Thickness surface-impedance BCs
(sigma=3.77e7 S/m; delta ~2.6 um at 1 GHz) model finite conductors. This
conductivity and the dielectric/passivation geometry differ from the pinned
OpenEMS script. OpenEMS was unavailable for this campaign.</p>
<p>The IHP OpenEMS reference uses SG13_L2n0_2port_groundpolygon_mesh1.5um.py
with TopMetal1 edge ports, a central Metal1->TopMetal1 PEC pillar, Metal1 ground
box, 1.5 µm refined cells, lambda_air/(sqrt(eps_max)*20) maximum cells, PEC
boundaries and 0-30 GHz/401 points. The `openems_ground` variant approximates
this with a Metal1 rectangle and vertical via ports; it omits the central pillar.
The new `openems_edge` setup adds the pillar and gap-port positions. A one-thread
coarse mesh generated two distinct 50-ohm port sheets (21,977 nodes), but a
2.998 µm conductor feature is under-resolved at 5 µm. No Palace solve has been run.</p>
<p><a href="L_2n0_simplified.gds">benchmark GDS</a> ·
<a href="metrics.json">metrics</a> · <a href="provenance.json">provenance</a> ·
<a href="https://github.com/EpsilonForge/gsim/blob/docs/ihp-l2n0-validation/nbs/data/ihp_validation/CONVERGENCE_AUDIT.md">geometry audit</a></p></div>
</main></body></html>
"""
    (out / "index.html").write_text(html, encoding="utf8")
    (out / "README.md").write_text(
        "# IHP SG13G2 L2n0 two-port validation audit\n\nSee notebook.html for full complex S comparisons of L2n0 and an independent measured L6n2 reference.\n"
    )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--notebook-html", type=Path)
    parser.add_argument("--notebook-ipynb", type=Path)
    args = parser.parse_args()
    build(
        args.out, notebook_html=args.notebook_html, notebook_ipynb=args.notebook_ipynb
    )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
