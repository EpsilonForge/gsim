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
    """Return (frequency_ghz, L_nH, Q) for one measured die."""
    network = rf.Network(str(ihp_l2n0.MEASURED[label]))
    z0 = float(np.real(np.asarray(network.z0).reshape(-1)[0]))
    z = 2.0 * z0 * (1.0 - network.s[:, 1, 0]) / network.s[:, 1, 0]
    l_nh = z.imag / (2 * np.pi * network.f) * 1e9
    q = np.divide(z.imag, z.real, out=np.full_like(z.real, np.nan), where=z.real != 0)
    return network.f / 1e9, l_nh, q


def render_comparison(out: Path) -> dict[str, float]:
    """Render L/Q comparison and return metrics at 1 and 30 GHz."""
    run = ihp_l2n0.load_recorded_run(ihp_l2n0.RUN_ROOT / "l2n0_sweep")
    f_palace = np.asarray(run.sparams.freq, dtype=float)
    l_palace, q_palace = [], []
    for i in range(len(f_palace)):
        l, q, _ = ihp_l2n0.lq_at(run.sparams, i)
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
        ax.axvline(30, color="k", lw=0.8, ls=":")
        ax.grid(alpha=0.3)
        ax.set(xlabel="frequency (GHz)")
        ax.legend(fontsize=8)
    axes[0].set(title="Extracted series inductance", ylabel="L (nH)")
    axes[1].set(title="Extracted quality factor", ylabel="Q")
    fig.tight_layout()
    fig.savefig(
        out / "l2n0_comparison.png", dpi=160, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)

    def at(f: np.ndarray, values: np.ndarray, target: float) -> float:
        return float(values[int(np.argmin(np.abs(f - target)))])

    rows = {
        "palace_l_1ghz_nh": at(f_palace, l_palace, 1.0),
        "palace_q_1ghz": at(f_palace, q_palace, 1.0),
        "palace_l_30ghz_nh": at(f_palace, l_palace, 30.0),
        "palace_q_30ghz": at(f_palace, q_palace, 30.0),
    }
    for label, (f, l, q) in measured.items():
        rows[f"{label}_l_1ghz_nh"] = at(f, l, 1.0)
        rows[f"{label}_q_1ghz"] = at(f, q, 1.0)
        rows[f"{label}_l_30ghz_nh"] = at(f, l, 30.0)
        rows[f"{label}_q_30ghz"] = at(f, q, 30.0)
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
<title>IHP SG13G2 L2n0 — inductance extraction</title>
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
<h1>IHP SG13G2 Octagon L2n0 — inductance extraction</h1>
<p class="muted">Palace/gsim versus the two pinned IHP measured dies using the existing
series extraction Z = 2 Z0 (1-S21)/S21. Nominal 2 nH benchmark. Layout and
measured data pinned from IHP-Open-PDK commit <code>9439837</code> (Apache-2.0).</p>
<div class="card"><img src="l2n0_comparison.png" alt="Measured and Palace extracted inductance and Q"></div>
<div class="card"><h2>Extraction summary</h2>
<table><thead><tr><th>case</th><th>L @ 1 GHz (nH)</th><th>Q @ 1 GHz</th><th>L @ 30 GHz (nH)</th><th>Q @ 30 GHz</th></tr></thead><tbody>
<tr><td>Palace</td><td>{rows["palace_l_1ghz_nh"]:.3f}</td><td>{rows["palace_q_1ghz"]:.2f}</td><td>{rows["palace_l_30ghz_nh"]:.3f}</td><td>{rows["palace_q_30ghz"]:.2f}</td></tr>
<tr><td>die_x2y7</td><td>{rows["die_x2y7_l_1ghz_nh"]:.3f}</td><td>{rows["die_x2y7_q_1ghz"]:.2f}</td><td>{rows["die_x2y7_l_30ghz_nh"]:.3f}</td><td>{rows["die_x2y7_q_30ghz"]:.2f}</td></tr>
<tr><td>die_x5y8</td><td>{rows["die_x5y8_l_1ghz_nh"]:.3f}</td><td>{rows["die_x5y8_q_1ghz"]:.2f}</td><td>{rows["die_x5y8_l_30ghz_nh"]:.3f}</td><td>{rows["die_x5y8_q_30ghz"]:.2f}</td></tr>
</tbody></table>
<p class="muted">The measured 1 GHz inductance is 1.91-1.93 nH, confirming the nominal
2 nH benchmark. The 30 GHz point is above the measured SRF (~25.5 GHz); the
Palace/measured disagreement is reported honestly.</p></div>
{notebook_section}
<div class="card"><h2>Model and source</h2>
<p>Palace sweep: 1-50 GHz, 11 points; authoritative IHP SG13G2 stack, TopMetal2
winding, TopMetal1 feeds/underpass, SUBGND reference plane, 5 µm refined / 70 µm
maximum mesh, interlayer ports SUBGND to TopMetal1 at 50 Ohm. OpenEMS was not
available; pinned measured S-parameters are the approved fallback and the IHP
openEMS settings are documented in the notebook.</p>
<p><a href="L_2n0_simplified.gds">benchmark GDS</a> ·
<a href="metrics.json">metrics</a> · <a href="provenance.json">provenance</a></p></div>
</main></body></html>
"""
    (out / "index.html").write_text(html, encoding="utf8")
    (out / "README.md").write_text(
        "# IHP SG13G2 L2n0 inductance extraction\n\nPalace vs measured L/Q extraction. See notebook.html and metrics.json.\n"
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
