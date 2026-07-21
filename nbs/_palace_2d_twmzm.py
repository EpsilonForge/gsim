# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: gsim (3.12.3)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Palace 2D Mode Analysis: Travelling-Wave Mach-Zehnder Modulator
#
# This notebook builds a simplified cross-section of a Travelling-Wave Mach-Zehnder Modulator (TW-MZM) with a PN-junction embedded in a rib waveguide, using CPW electrodes for RF modulation.
#
# **Cross-section geometry (from literature):**
# - **SOI substrate**: 220 nm Si on 2 um buried oxide (BOX)
# - **Rib waveguide**: 400 nm width, 90 nm slab height
# - **CPW electrodes**: Aluminium, 1 um thick, signal width $w=20$ um, gap $g=20$ um
# - **PN junction**: Centred in the rib with P+/N+ contact regions in the slab
#
# We use `BoundaryModeSim` (Palace 2D eigenmode solver) to compute both RF and optical modes.
#
# **Requirements:**
# - gdsfactory + generic PDK (`gf.gpdk`)
# - [GDSFactory+](https://gdsfactory.com) account (for cloud Palace runs)

# %% [markdown]
# ### Build TW-MZM cross-section geometry
#
# We define the 3D layout component that will be sliced at $x=0$ for 2D mode analysis.
#
# **Layer assignments (gpdk):**
# - `WG` (1,0): Rib waveguide core (220 nm Si, 400 nm wide)
# - `SLAB90` (3,0): 90 nm slab regions
# - `N` (20,0) / `P` (21,0): PN junction doping
# - `NPP` (24,0) / `PP` (25,0): P+/N+ contact doping
# - `M3` (49,0): CPW electrodes (Al, 1 um thick)

# %%
import gdsfactory as gf

gf.gpdk.PDK.activate()

RIB_WIDTH = 0.4
SLAB_HALF = 20.0
SIG_WIDTH = 20.0
GAP_WIDTH = 20.0
GND_WIDTH = 40.0
TOTAL_HALF = SIG_WIDTH / 2 + GAP_WIDTH + GND_WIDTH
LENGTH = 10.0

LAYER = gf.gpdk.LAYER


def centered_rect(wx: float, wy: float, layer) -> gf.Component:
    r = gf.Component()
    r << gf.c.rectangle((wx, wy), centered=True, layer=layer)
    return r


# =============================================================================
#  Unified helper: contiguous doping profile + materials + layer specs
# =============================================================================
def add_doping_profile(
    comp: gf.Component,
    length: float,
    rib_center_y: float,
    rib_width: float,
    profile: dict[str, list[tuple[float, float]]],
    permittivity: float = 11.9,
    slab_zmax: float = 0.09,
) -> dict:
    """
    Create contiguous doping regions on both sides of the rib and generate
    the corresponding MaterialProperties and Layer definitions.

    profile = {
        "upper": [(width_um, conductivity_S_per_m), ...],  # P side (toward S)
        "lower": [(width_um, conductivity_S_per_m), ...],  # N side (toward G)
    }

    Regions are placed contiguously (no gaps) starting from the rib edge.

    Returns: {
        "layer_specs": {name: Layer, ...},
        "materials": {name: MaterialProperties, ...},
        "centres": {side: [y_centre, ...]},
    }
    """
    from gsim.common.stack.extractor import Layer
    from gsim.common.stack.materials import (
        DispersionModel,
        MaterialProperties,
        ValidityRange,
    )

    side_config = {
        "upper": {"base_layer": (23, 0), "prefix": "pp_slab_", "sign": 1},
        "lower": {"base_layer": (24, 0), "prefix": "npp_slab_", "sign": -1},
    }

    result = {"layer_specs": {}, "materials": {}, "centres": {}}

    for side in ("upper", "lower"):
        cfg = side_config[side]
        regions = profile.get(side, [])
        sign = cfg["sign"]
        pos = rib_center_y + sign * rib_width / 2  # start at rib edge
        centres = []

        for i, (width, sigma) in enumerate(regions):
            name = f"{cfg['prefix']}{i}"
            gds_layer = (cfg["base_layer"][0], cfg["base_layer"][1] + i)
            centre = pos + sign * width / 2

            rect = comp << gf.c.rectangle((length, width), layer=gds_layer)
            rect.y = centre
            centres.append(centre)
            pos += sign * width

            result["layer_specs"][name] = Layer(
                name=name,
                gds_layer=gds_layer,
                zmin=0.0,
                zmax=slab_zmax,
                thickness=slab_zmax,
                material=name,
                layer_type="dielectric",
                mesh_resolution="fine",
            )

            result["materials"][name] = MaterialProperties(
                permittivity=permittivity,
                conductivity=sigma,
                dispersion_models=[
                    DispersionModel(
                        type="constant",
                        permittivity=permittivity,
                        validity=ValidityRange(valid_frequency=(0, 200e9)),
                        source=f"doped Si ({name}) -- Drude sigma",
                    ),
                ],
            )

        result["centres"][side] = centres

    return result


# =============================================================================
#  Component: TW-MZM cross-section
# =============================================================================
comp = gf.Component()

# Rib centre: centred in the gap between left G (y approx -50) and centre S (y=0)
RIB_CENTER_Y = -(SIG_WIDTH / 2 + GAP_WIDTH / 2)  # -20.0

# 1. Rib waveguide core
wg = comp << centered_rect(LENGTH, RIB_WIDTH, LAYER.WG)
wg.y = RIB_CENTER_Y

# 2. Slab (90 nm)
slab = comp << centered_rect(LENGTH, 2 * SLAB_HALF + RIB_WIDTH, LAYER.SLAB90)
slab.y = RIB_CENTER_Y

# 3. PN junction
p_half = comp << gf.c.rectangle((LENGTH, RIB_WIDTH / 2), layer=LAYER.P)
p_half.y = RIB_CENTER_Y + RIB_WIDTH / 4
n_half = comp << gf.c.rectangle((LENGTH, RIB_WIDTH / 2), layer=LAYER.N)
n_half.y = RIB_CENTER_Y - RIB_WIDTH / 4

# 4+5. Doping gradient (contiguous, no gaps)
doping_profile = {
    "upper": [  # P side (toward signal)
        (2.0, 2.0e4),  # width=2um, sigma=2e4 S/m
        (2.0, 8.0e4),  # width=2um, sigma=8e4 S/m
    ],
    "lower": [  # N side (toward ground)
        (2.0, 2.0e4),
        (2.0, 8.0e4),
    ],
}
doping_result = add_doping_profile(
    comp, LENGTH, RIB_CENTER_Y, RIB_WIDTH, doping_profile
)

# 6. CPW electrodes (M1)
sig = comp << centered_rect(LENGTH, SIG_WIDTH, LAYER.M1)
gnd_top = comp << centered_rect(LENGTH, GND_WIDTH, LAYER.M1)
gnd_top.y = SIG_WIDTH / 2 + GAP_WIDTH + GND_WIDTH / 2
gnd_bot = comp << centered_rect(LENGTH, GND_WIDTH, LAYER.M1)
gnd_bot.y = -(SIG_WIDTH / 2 + GAP_WIDTH + GND_WIDTH / 2)

# 7. Vias at x=0 (one per electrode, overlapping electrode)
via_s_to_p = comp << gf.c.via_stack(
    layers=("SLAB90", "M1"), vias=("viac", None), size=(2.7, 2.7)
)
via_s_to_p.x = 0.0
via_s_to_p.y = -9.0  # overlaps S electrode at y=-10
via_g_to_n = comp << gf.c.via_stack(
    layers=("SLAB90", "M1"), vias=("viac", None), size=(2.7, 2.7)
)
via_g_to_n.x = 0.0
via_g_to_n.y = -29.0  # overlaps G electrode at y=-30

# -- Plot ----------------------------------------------------------------------
_cc = comp.copy()
_cc.draw_ports()
_cc.plot()

# %% [markdown]
# ### Inspect 2D cross-section
#
# Before meshing, we extract the geometric cross-section at $x=0$ to verify that all layers are correctly defined.

# %%
import gsim.common.cross_section as cross_section
from gsim.common.stack import get_stack
from gsim.common.stack.extractor import Layer
from gsim.common.stack.materials import (
    DispersionModel,
    MaterialProperties,
    ValidityRange,
)
from gsim.palace import BoundaryModeSim

# -- Stack --------------------------------------------------------------------
stack = get_stack(
    substrate_thickness=2.0,
    include_substrate=False,
)

# Override metal1 thickness to 1 um (electrodes on M1)
m1 = stack.layers.get("metal1")
if m1:
    m1.zmax = 1.1 + 1.0
    m1.thickness = 1.0
    print(f"M1 updated: zmin={m1.zmin}, zmax={m1.zmax}, thickness={m1.thickness}")

# -- Register doping materials and layers from the profile result -----
SI_RIB_ZMIN, SI_RIB_ZMAX = 0.0, 0.22
SI_SLAB_ZMAX = 0.09

# doping_result comes from the geometry cell's add_doping_profile()
doped_materials = doping_result["materials"]
doping_layers = doping_result["layer_specs"]

# Add the PN junction rib layers (not part of the gradient profile)
# PN junction (~1e19 cm^-3) -> sigma ~ 1600 S/m
for mat_name, gds_layer, zmax, sigma in [
    ("p_rib", LAYER.P, SI_RIB_ZMAX, 1.6e3),
    ("n_rib", LAYER.N, SI_RIB_ZMAX, 1.6e3),
]:
    doping_layers[mat_name] = Layer(
        name=mat_name,
        gds_layer=gds_layer,
        zmin=SI_RIB_ZMIN,
        zmax=zmax,
        thickness=zmax - SI_RIB_ZMIN,
        material=mat_name,
        layer_type="dielectric",
        mesh_resolution="fine",
    )
    doped_materials[mat_name] = MaterialProperties(
        permittivity=11.9,
        conductivity=sigma,
        dispersion_models=[
            DispersionModel(
                type="constant",
                permittivity=11.9,
                validity=ValidityRange(valid_frequency=(0, 200e9)),
                source=f"doped Si ({mat_name}) -- Drude sigma",
            ),
        ],
    )
for name, layer in doping_layers.items():
    stack.layers[name] = layer

# -- Via layer definitions ----------------------------------------------------
# The gpdk already defines via_contact, via1, via2 in the stack.  We
# ensure they are present and have the right z-extents for our cross-section.
# (They are already set correctly by the gpdk extractor, listed here for
#  reference.)
print(f"Stack: {stack.pdk_name}")
print("Layers:", sorted(stack.layers.keys()))
print("Custom materials:", sorted(doped_materials.keys()))
print("Dielectrics:", stack.dielectrics)
print()

# -- Cross-section extraction ------------------------------------------------
section = cross_section.extract_plane_section(comp.copy(), stack, axis="x", value=0.0)
print(f"Cross-section x=0 intersects {len(section)} layer regions:")
for r in section:
    print(
        f"  {r.layer_name:12s}  material={r.material:10s}  "
        f"y=[{r.y0:8.3f}, {r.y1:8.3f}]  z=[{r.zmin:6.3f}, {r.zmax:6.3f}]"
    )

# %%
# -- Zoomed cross-section plot (rib region) ----------------------------------
# The full layout spans 140 um, so the 400 nm rib is invisible at full scale.
# Here we plot only the central ±15 um to show the rib, PN junction, and
# the P+/N+ ohmic contacts clearly.
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig, ax = plt.subplots(figsize=(10, 4))

# Colour map per layer name
colors = {
    "core": "#c0392b",  # rib Si -- red
    "slab90": "#e67e22",  # slab Si -- orange
    "p_rib": "#2980b9",  # P doping -- blue
    "pp_slab_0": "#8e44ad",  # P+ graded inner -- purple
    "pp_slab_1": "#a569bd",  # P+ graded outer -- light purple
    "n_rib": "#27ae60",  # N doping -- green
    "npp_slab_0": "#4460ad",  # N+ graded inner -- indigo
    "npp_slab_1": "#5b7dcf",  # N+ graded outer -- light indigo
    "metal3": "#7f8c8d",  # Al -- grey
    "via_contact": "#f1c40f",  # via -- gold
}

for r in section:
    c = colors.get(r.layer_name, "#dddddd")
    # RectYZ2D: y0/y1 are lateral, zmin/zmax are vertical
    rect = Rectangle(
        (r.y0, r.zmin),
        r.y1 - r.y0,
        r.zmax - r.zmin,
        facecolor=c,
        edgecolor="k",
        linewidth=0.5,
        alpha=0.8,
        label=r.layer_name,
    )
    ax.add_patch(rect)

ax.set_xlim(-15, 15)
ax.set_ylim(-0.5, 5.0)
ax.set_aspect("equal")
ax.set_xlabel("y (um)")
ax.set_ylabel("z (um)")
ax.set_title("TW-MZM cross-section (zoomed on rib + doping + signal electrode)")

# Deduplicate legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)

plt.tight_layout()
plt.show()

# %%
# -- BoundaryMode 2D simulation setup ----------------------------------------
sim = BoundaryModeSim()
sim.set_output_dir("./palace-sim-mzm-pn")

# Custom stack with modified M3 + doping layers
sim.set_stack(stack)
sim.set_airbox(margin_x=50.0, margin_y=50.0, z_above=100.0, z_below=100.0)
sim.set_geometry(comp)

sim.set_cross_section("x=0")
sim.set_boundary_mode(freq=50e9, num_modes=2, save=2)

# -- Mesh ---------------------------------------------------------------------
# refined_mesh_size must be smaller than the rib (0.4 um wide, 0.22 um tall)
# to resolve the waveguide geometry.  Use 0.05 um near conductors.
sim.mesh(
    preset="default",
    refined_mesh_size=0.05,  # resolve the 400 nm rib
    max_mesh_size=40.0,
    fmax=150e9,
    margin_x=0.0,
    margin_y=50.0,
)

# Show the 2D domain groups created by the solver
domain_groups = list(sim._last_mesh_result.groups["volumes"].keys())
print("2D domain groups:", domain_groups)

# %%
# Interactive 3D mesh visualisation
# (Groups depend on what the mesh generator creates -- port surfaces
#  like P1_E0 are absent in BoundaryMode 2D native mode)
sim.plot_mesh(
    transparent_groups=["air__None", "air__passive", "oxide__passive"],
    style="solid",
    interactive=True,
)

# %%
# Generate Palace config file (mesh must be present)
sim.write_config()
print("Config written to:", sim.output_dir)

# %% [markdown]
# ### RF mode analysis (50 GHz)
#
# The BoundaryMode solver computes propagation constants and mode profiles at the RF frequency.
# We demonstrate the setup below -- uncomment the `run_local` call when ready to execute.

# %%
# -- RF simulation (50 GHz) ------------------------------------------------
# Uncomment to run.  The resolver tries, in order:
#   1. PALACE_BIN env var
#   2. PALACE_EXECUTABLE env var / "palace" in PATH
#   3. palace-toolkit-cpu (optional pip package)

import subprocess
import sys

from gsim.palace.runtime import (
    _palace_cpu_available,
    resolve_palace_binary,
)

bin_path = resolve_palace_binary()
if bin_path is None:
    print(
        "No Palace binary found. Install palace-toolkit-cpu: pip install gsim[palace-toolkit-cpu]"
    )
    sys.exit(1)

source = "palace-toolkit-cpu" if _palace_cpu_available() else "PATH / env var"
print(f"Palace binary: {bin_path}  (source: {source})")

try:
    ver = subprocess.run(  # noqa: S603
        [str(bin_path), "--version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    print(ver.stdout.strip())
except Exception:
    pass

results = sim.run_local(verbose=True)
results.print()

# %%
# -- Post-processing: RF mode fields ------------------------------------

import importlib

import gsim.palace.field_viz as field_viz
import gsim.palace.results as palace_results
from gsim.palace import plot_fields_2d

importlib.reload(field_viz)
importlib.reload(palace_results)

if not hasattr(results, "modes"):
    results = palace_results.load_text_results("./palace-sim-mzm-pn")

results.print()

fig, ax, stream_inputs = plot_fields_2d(
    "./palace-sim-mzm-pn",
    field="E_real",
    normal="x",
    origin=0.0,
    title="RF Mode |E_t| at x=0 (50 GHz)",
)

# %% [markdown]
# ### Optical mode analysis (1550 nm)
#
# The same cross-section can be analysed at optical frequencies ($\lambda = 1.55$ um) to compute the optical mode confined in the rib waveguide.  Here we set up a second `BoundaryModeSim` targeting the optical regime.

# %%
# Optical mode at 1550 nm (~193.4 THz)
f_opt = 193.4e12  # Hz

sim_opt = BoundaryModeSim()
sim_opt.set_output_dir("./palace-sim-mzm-pn-opt")
sim_opt.set_stack(stack)
sim_opt.set_airbox(margin_x=50.0, margin_y=50.0, z_above=100.0, z_below=100.0)
sim_opt.set_geometry(comp)

sim_opt.set_cross_section("x=0")
sim_opt.set_boundary_mode(
    freq=f_opt,
    num_modes=4,
    save=2,
    target=2.5,
    tolerance=1e-8,
)

sim_opt.mesh(
    preset="default",
    refined_mesh_size=0.02,
    max_mesh_size=0.5,
    margin_x=0.0,
    margin_y=50.0,
)

# -- Optical simulation (1550 nm) -------------------------------------------
# Uncomment to run:
# from gsim.palace.runtime import resolve_palace_binary, _palace_cpu_available
# import subprocess
# bin_path = resolve_palace_binary()
# src = "palace-toolkit-cpu" if _palace_cpu_available() else "PATH / env var"
# print(f"Palace binary: {bin_path}  (source: {src})")
# print(subprocess.run([str(bin_path), "--version"], capture_output=True, text=True).stdout.strip())
# opt_results = sim_opt.run_local(verbose=True)
# opt_results.print()

# %% [markdown]
# ### Summary
#
# The geometry has been built and meshed for both RF (50 GHz) and optical (193 THz / 1550 nm) analysis.
#
# **Cross-section elements:**
# | Component | Layer | y-range (um) | z-range (um) | Material | sigma (S/m) |
# |---|---|---|---|---|---|
# | Rib core | WG (1,0) | [-20.2, -19.8] | [0, 0.22] | Si (intrinsic) | 2 |
# | Slab (90 nm) | SLAB90 (3,0) | [-40.2, +0.2] | [0, 0.09] | Si (intrinsic) | 2 |
# | PN junction (P) | P (21,0) | [-20.0, -19.8] | [0, 0.22] | doped Si (p_rib) | 1.6x10^3 |
# | PN junction (N) | N (20,0) | [-20.2, -20.0] | [0, 0.22] | doped Si (n_rib) | 1.6x10^3 |
# | P+ graded inner | PP (23,0) | [-18.3, -16.3] | [0, 0.09] | doped Si (pp_slab_0) | 2x10^4 |
# | P+ graded outer | PP (23,0) | [-15.3, -13.3] | [0, 0.09] | doped Si (pp_slab_1) | 8x10^4 |
# | N+ graded inner | NPP (24,0) | [-21.7, -23.7] | [0, 0.09] | doped Si (npp_slab_0) | 2x10^4 |
# | N+ graded outer | NPP (24,0) | [-24.7, -26.7] | [0, 0.09] | doped Si (npp_slab_1) | 8x10^4 |
# | Vias (S to P+) | VIAC/VIA1/VIA2 | [-15.0, -9.0] | [0.09, 3.2] | W/Al | 3.5x10^7 |
# | Vias (G to N+) | VIAC/VIA1/VIA2 | [-31.0, -25.0] | [0.09, 3.2] | W/Al | 3.5x10^7 |
# | CPW signal | M1 (41,0) | [-10, +10] | [3.2, 4.2] | Al (1 um) | 3.5x10^7 |
# | CPW ground (top) | M1 (41,0) | [+30, +70] | [3.2, 4.2] | Al (1 um) | 3.5x10^7 |
# | CPW ground (bot) | M1 (41,0) | [-70, -30] | [3.2, 4.2] | Al (1 um) | 3.5x10^7 |
#
# **Material modelling notes:**
# - Doping regions are modelled as **semiconductors** (finite sigma from Drude free-carrier model), not metals. This avoids short-circuiting the PN junction.
# - Conductivities are derived from $\sigma = q\mu N$ with typical dopant concentrations ($N \sim 10^{19}$ cm-^3 for the junction, $\sim 10^{20}$ cm-^3 for the contacts).
# - Doping on each side of the rib uses a **configurable piecewise gradient**; the helper `make_doped_materials_dict()` generates the `MaterialProperties` dict from a list of `(name, permittivity, conductivity, source)` tuples.
# - The **depletion region** and voltage-dependent capacitance are NOT modelled here -- this is a linear small-signal analysis at a fixed bias point.
# - The **plasma-dispersion effect** (free-carrier $\Delta n$, $\Delta k$) is not applied to the optical simulation; the rib is treated as intrinsic Si at 1550 nm. A full electro-optic analysis would require a separate carrier-dependent optical material model (Soref–Bennett).
#
# **Next steps (user action):**
# 1. Verify the zoomed cross-section plot above shows the rib (centred at y=-20), PN junction, graded doping regions, and vias.
# 2. Run `sim.run_local(num_processes=4, verbose=True)` with palace-toolkit-cpu installed (`pip install gsim[palace-toolkit-cpu]`).
# 3. Run `sim_opt.run_local(num_processes=4, verbose=True)` for the optical mode.
# 4. Use `gsim.palace.plot_fields_2d()` to visualise mode profiles.
#

# %%
# Quick verification: dump the 2D cross-section layer regions
# The rib waveguide is now centred at y=-20 (between left G and centre S)
# Piecewise doping gradient: pp_slab_0 (inner) -> pp_slab_1 (outer) on P+ side,
#                               npp_slab_0 (inner) -> npp_slab_1 (outer) on N+ side
# Vias at x=0 visible as via_contact/via1/via2 regions connecting S/G to slab
print("=== RF cross-section regions (x=0) ===")
for r in section:
    print(
        f"  {r.layer_name:12s}  mat={r.material:10s}  "
        f"y=[{r.y0:6.2f},{r.y1:6.2f}]  z=[{r.zmin:5.3f},{r.zmax:5.3f}]"
    )
