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


def render_comparison(out: Path) -> dict[str, float]:
    """Render measured sweeps plus the recorded Palace point at 30 GHz."""
    run = ihp_l2n0.load_recorded_run(ihp_l2n0.RUN_ROOT / "l2n0_30ghz")
    sim_s = ihp_l2n0.s_matrix(run.sparams, 0)
    sim_l, sim_q, _ = ihp_l2n0.lq_at(run.sparams)

    figure, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for label, color in (("die_x2y7", "C0"), ("die_x5y8", "C1")):
        network = rf.Network(str(ihp_l2n0.MEASURED[label]))
        axes[0].plot(
            network.f / 1e9,
            20 * np.log10(np.abs(network.s[:, 0, 0])),
            color=color,
            lw=1.0,
            label=f"{label} S11",
        )
        axes[0].plot(
            network.f / 1e9,
            20 * np.log10(np.abs(network.s[:, 1, 0])),
            color=color,
            lw=1.0,
            ls="--",
            label=f"{label} S21",
        )
        z0 = float(np.real(np.asarray(network.z0).reshape(-1)[0]))
        z = 2.0 * z0 * (1.0 - network.s[:, 1, 0]) / network.s[:, 1, 0]
        axes[1].plot(
            network.f / 1e9,
            z.imag / (2 * np.pi * network.f) * 1e9,
            color=color,
            lw=1.0,
            label=f"{label} L",
        )
    axes[0].axvline(30, color="k", lw=0.8, ls=":")
    axes[1].axvline(30, color="k", lw=0.8, ls=":")
    axes[0].plot(
        [30],
        [20 * np.log10(abs(sim_s[0, 0]))],
        "o",
        ms=7,
        color="C3",
        label="Palace S11",
    )
    axes[0].plot(
        [30],
        [20 * np.log10(abs(sim_s[1, 0]))],
        "s",
        ms=7,
        color="C3",
        label="Palace S21",
    )
    axes[1].plot([30], [sim_l], "o", ms=7, color="C3", label="Palace L")
    axes[0].set(
        title="Measured and Palace S-parameters", xlabel="frequency (GHz)", ylabel="dB"
    )
    axes[0].set_ylim(-45, 5)
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)
    axes[1].set(
        title="Effective series inductance", xlabel="frequency (GHz)", ylabel="L (nH)"
    )
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(
        out / "l2n0_comparison.png", dpi=160, bbox_inches="tight", facecolor="white"
    )
    plt.close(figure)

    rows = {
        "palace_s11_db": float(20 * np.log10(abs(sim_s[0, 0]))),
        "palace_s21_db": float(20 * np.log10(abs(sim_s[1, 0]))),
        "palace_l_nh": float(sim_l),
        "palace_q": float(sim_q),
    }
    for label in ("die_x2y7", "die_x5y8"):
        network = rf.Network(str(ihp_l2n0.MEASURED[label]))
        point = network.interpolate(rf.Frequency(30, 30, 1, unit="GHz"))
        s = point.s[0]  # ty: ignore[unresolved-attribute]
        l_nh, q, _ = _lq(s, 30e9)
        rows[f"{label}_s11_db"] = float(20 * np.log10(abs(s[0, 0])))
        rows[f"{label}_s21_db"] = float(20 * np.log10(abs(s[1, 0])))
        rows[f"{label}_l_nh"] = float(l_nh)
        rows[f"{label}_q"] = float(q)
    return rows


def _lq(s: np.ndarray, f_hz: float) -> tuple[float, float, complex]:
    """Return effective series (L_nH, Q, Z) from a 2x2 S matrix."""
    z = 2.0 * ihp_l2n0.Z0_OHM * (1.0 - s[1, 0]) / s[1, 0]
    l_nh = z.imag / (2 * np.pi * f_hz) * 1e9
    q = z.imag / z.real if z.real != 0 else float("nan")
    return l_nh, q, z


def build(
    out: Path,
    *,
    notebook_html: Path | None = None,
    notebook_ipynb: Path | None = None,
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
    table = "\n".join(
        f"<tr><td>{label}</td><td>{rows[f'{label}_s11_db']:.2f}</td>"
        f"<td>{rows[f'{label}_s21_db']:.2f}</td><td>{rows[f'{label}_l_nh']:.3f}</td>"
        f"<td>{rows[f'{label}_q']:.2f}</td></tr>"
        for label in ("palace", "die_x2y7", "die_x5y8")
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IHP SG13G2 L2n0 — 30 GHz validation</title>
<style>
:root {{ color-scheme:light dark; --fg:#1f2328; --muted:#656d76; --bg:#fff; --card:#f6f8fa; --line:#d0d7de; --accent:#0969da; }}
@media (prefers-color-scheme:dark) {{ :root {{ --fg:#e6edf3; --muted:#8b949e; --bg:#0d1117; --card:#161b22; --line:#30363d; --accent:#4493f8; }} }}
body {{ margin:0; font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; color:var(--fg); background:var(--bg); }}
main {{ max-width:1050px; margin:0 auto; padding:36px 24px 56px; }}
h1 {{ margin:0 0 6px; font-size:1.8rem; }}
.muted {{ color:var(--muted); }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:14px; padding:18px; margin:20px 0; }}
img {{ width:100%; border-radius:10px; background:#fff; }}
table {{ border-collapse:collapse; width:100%; font-size:.92rem; }}
th,td {{ border-bottom:1px solid var(--line); padding:7px 8px; text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }}
code {{ font-size:.9em; }}
</style></head><body><main>
<p><a href="../">&larr; all variants</a></p>
<h1>IHP SG13G2 Octagon L2n0 — 30 GHz validation</h1>
<p class="muted">Palace/gsim vs IHP THRU-de-embedded measured S-parameters for the
2-turn, nominal 2 nH benchmark. Layout and measured data are pinned from
IHP-Open-PDK commit <code>9439837</code> (Apache-2.0).</p>
<div class="card"><img src="l2n0_comparison.png" alt="Measured and Palace S-parameters and effective inductance"></div>
<div class="card"><h2>30 GHz comparison</h2>
<table><thead><tr><th>case</th><th>|S11| dB</th><th>|S21| dB</th><th>L nH</th><th>Q</th></tr></thead>
<tbody>{table}</tbody></table>
<p class="muted">The effective L/Q uses the documented series 2-port extraction
Z = 2 Z0 (1-S21)/S21. At 30 GHz the benchmark is above its measured SRF
(~25.5 GHz), so the S-parameters are the primary comparison.</p></div>
{notebook_section}
<div class="card"><h2>Model and source</h2>
<p>Palace model: authoritative IHP SG13G2 stack, TopMetal2 winding,
TopMetal1 feeds/underpass, SUBGND reference plane, 5 µm refined / 70 µm maximum
mesh, two interlayer ports SUBGND to TopMetal1 at 50 Ohm. The notebook records the
IHP openEMS settings and the measured-data fallback required by the validation
plan.</p>
<p><a href="L_2n0_simplified.gds">benchmark GDS</a> ·
<a href="metrics.json">metrics</a> ·
<a href="provenance.json">provenance</a></p></div>
</main></body></html>
"""
    (out / "index.html").write_text(html, encoding="utf8")
    (out / "README.md").write_text(
        "# IHP SG13G2 L2n0 validation\n\n"
        "Palace/gsim vs measured S-parameters at 30 GHz for the IHP "
        "SG13_Octagon_L2n0 benchmark. See `notebook.html` and `metrics.json`.\n",
        encoding="utf8",
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
