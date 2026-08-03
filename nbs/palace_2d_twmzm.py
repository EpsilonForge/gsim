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
# - A Palace binary resolved internally by `run_local()` (e.g. `palace-toolkit-cpu`, `PALACE_BIN`, or `palace` on PATH)
# - [GDSFactory+](https://gdsfactory.com) account only for cloud runs
#
# **Workflow:** parameters are grouped next to the stage they configure — geometry &amp; materials, then RF, then optical.
#

# %% [markdown]
# ## Geometry &amp; materials parameters
#
# Define the layout dimensions, the substrate/metal stack, and the doping
# (geometry + material) for the rib and graded slab regions. These are consumed
# by the geometry build and by
# `gsim.common.cross_section.build_doped_cross_section()`.
#
# The generic helpers used here are PDK-agnostic — no hardcoded values live in
# the notebook.
#
# **Layer assignments (gpdk):**
# - `WG` (1,0): Waveguide core (220 nm Si, 400 nm wide)
# - `SLAB90` (3,0): 90 nm slab regions
# - `N` (20,0) / `P` (21,0): PN junction doping
# - `NPP` (24,0) / `PP` (23,0): N+/P+ graded contact doping (via `make_doping_profile`)
# - `M1` (41,0): CPW electrodes (Al, 1 um thick)
#

# %%
# =============================================================================
#  Geometry & materials parameters
# =============================================================================

# --- Device / rib geometry (um) ---------------------------------------------
RIB_WIDTH = 0.4  # rib waveguide width
RIB_HEIGHT = 0.22  # rib waveguide height (z)
SLAB_THICKNESS = 0.09  # slab height (z) on each side of the rib
SLAB_HALF = 50.0  # slab half-width (y, on each side of the rib)
SIG_WIDTH = 20.0  # CPW signal electrode width
GAP_WIDTH = 20.0  # CPW signal-to-ground gap
GND_WIDTH = 40.0  # CPW ground electrode width
LENGTH = 10.0  # layout length along the propagation axis (x)
TOTAL_HALF = SIG_WIDTH / 2 + GAP_WIDTH + GND_WIDTH  # lateral half-extent
RIB_CENTER_Y = -(SIG_WIDTH / 2 + GAP_WIDTH / 2)  # rib centre y (=-20)

VIA_SIZE = 2.7  # via_stack footprint (square)
VIA_S_TO_P_Y = -9.0  # via from signal to P+ contact (y)
VIA_G_TO_N_Y = -29.0  # via from ground to N+ contact (y)

# --- Substrate stack (um) -----------------------------------------------------
BOX_THICKNESS = 2.0  # buried-oxide thickness (below z=0)
METAL1_ZMIN = 1.1  # metal1 bottom (top of the oxide stack)
METAL1_THICKNESS = 1.0  # CPW electrode thickness on metal1

# --- PN junction / doping material model -------------------------------------
SI_PERMITTIVITY = 11.9
FMAX_RF_MATERIAL = 200e9  # validity range of the constant-eps doping models (Hz)
RIB_DOPING_SIGMA = 1.6e3  # p_rib / n_rib junction conductivity (S/m)

# Graded slab doping {side: [(width_um, sigma_S_per_m), ...]}, from the rib edge.
DOPING_PROFILE = {
    "upper": [(2.0, 2.0e4), (2.0, 8.0e4)],  # P+ graded (toward signal)
    "lower": [(2.0, 2.0e4), (2.0, 8.0e4)],  # N+ graded (toward ground)
}
DOPING_SIDES = {  # config passed to make_doping_profile()
    "upper": {"base_layer": (23, 0), "name_prefix": "pp_slab_", "sign": 1},
    "lower": {"base_layer": (24, 0), "name_prefix": "npp_slab_", "sign": -1},
}

# --- Cross-section plane ------------------------------------------------------
CROSS_SECTION_AXIS = "x"
CROSS_SECTION_VALUE = 0.0

# %% [markdown]
# ## Build TW-MZM cross-section geometry
#
# The 3D layout component is sliced at $x=0$ for 2D mode analysis. The graded
# doping regions are created with `make_doping_profile()` (no hardcoded geometry).
#

# %%
import gdsfactory as gf

from gsim.common.stack.doping import make_doping_profile

gf.gpdk.PDK.activate()

LAYER = gf.gpdk.LAYER


def centered_rect(wx: float, wy: float, layer) -> gf.Component:
    r = gf.Component()
    r << gf.c.rectangle((wx, wy), centered=True, layer=layer)
    return r


# --- Component: TW-MZM cross-section ------------------------------------
comp = gf.Component()

# 1. Rib waveguide core (PN junction sits inside it)
wg = comp << centered_rect(LENGTH, RIB_WIDTH, LAYER.WG)
wg.y = RIB_CENTER_Y

# 2. Slab (90 nm)
slab = comp << centered_rect(LENGTH, 2 * SLAB_HALF + RIB_WIDTH, LAYER.SLAB90)
slab.y = 0.0

# 3. PN junction (P above / N below the rib centre)
p_half = comp << gf.c.rectangle((LENGTH, RIB_WIDTH / 2), layer=LAYER.P)
p_half.y = RIB_CENTER_Y + RIB_WIDTH / 4
n_half = comp << gf.c.rectangle((LENGTH, RIB_WIDTH / 2), layer=LAYER.N)
n_half.y = RIB_CENTER_Y - RIB_WIDTH / 4

# 4. Graded N+/P+ slab doping (contiguous, no gaps)
doping_result = make_doping_profile(
    comp,
    length=LENGTH,
    rib_center_y=RIB_CENTER_Y,
    rib_width=RIB_WIDTH,
    profile=DOPING_PROFILE,
    sides=DOPING_SIDES,
    zmin=0.0,
    zmax=SLAB_THICKNESS,
    permittivity=SI_PERMITTIVITY,
    fmax=FMAX_RF_MATERIAL,
)

# 5. CPW electrodes (M1)
sig = comp << centered_rect(LENGTH, SIG_WIDTH, LAYER.M1)
gnd_top = comp << centered_rect(LENGTH, GND_WIDTH, LAYER.M1)
gnd_top.y = SIG_WIDTH / 2 + GAP_WIDTH + GND_WIDTH / 2
gnd_bot = comp << centered_rect(LENGTH, GND_WIDTH, LAYER.M1)
gnd_bot.y = -(SIG_WIDTH / 2 + GAP_WIDTH + GND_WIDTH / 2)

# 6. Vias at x=0 (signal-to-P+, ground-to-N+)
via_s_to_p = comp << gf.c.via_stack(
    layers=("SLAB90", "M1"), vias=("viac", None), size=(VIA_SIZE, VIA_SIZE)
)
via_s_to_p.x = 0.0
via_s_to_p.y = VIA_S_TO_P_Y

via_g_to_n = comp << gf.c.via_stack(
    layers=("SLAB90", "M1"), vias=("viac", None), size=(VIA_SIZE, VIA_SIZE)
)
via_g_to_n.x = 0.0
via_g_to_n.y = VIA_G_TO_N_Y

# -- Plot ----------------------------------------------------------------------
_cc = comp.copy()
_cc.draw_ports()
_cc.plot()

# %% [markdown]
# ## Inspect 2D cross-section
#
# The reusable `gsim.common.cross_section.build_doped_cross_section()` helper
# assembles the base PDK stack, overrides `metal1`, registers the gradient doping
# and PN-junction rib layers, and extracts the 2D cross-section at $x=0$.
#

# %%
# Make the helper's diagnostics visible
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

from gsim.common.cross_section import build_doped_cross_section

stack, section = build_doped_cross_section(
    comp,
    axis=CROSS_SECTION_AXIS,
    value=CROSS_SECTION_VALUE,
    substrate_thickness=BOX_THICKNESS,
    metal1=(METAL1_ZMIN, METAL1_THICKNESS),
    doping=doping_result,
    rib_layers=[
        ("p_rib", LAYER.P, RIB_DOPING_SIGMA),
        ("n_rib", LAYER.N, RIB_DOPING_SIGMA),
    ],
    rib_height=RIB_HEIGHT,
    permittivity=SI_PERMITTIVITY,
    fmax=FMAX_RF_MATERIAL,
)

# %% [markdown]
# ## Plot the 2D cross-section
#
# The physical-group regions are rendered with the reusable
# `gsim.palace.plot_plane_section()`. The full layout spans ~140 um, so we zoom
# on the central region to show the rib, PN junction, graded doping and vias.
#

# %%
# --- Cross-section plot parameters (physical-group regions) ---------------
PLOT_ZOOM = {"h_range": (-15.0, 15.0), "v_range": (-0.5, 5.0)}
PLOT_COLORS = {
    "core": "#c0392b",  # rib Si -- red
    "slab90": "#e67e22",  # slab Si -- orange
    "p_rib": "#2980b9",  # P doping -- blue
    "pp_slab_0": "#8e44ad",  # P+ graded inner -- purple
    "pp_slab_1": "#a569bd",  # P+ graded outer -- light purple
    "n_rib": "#27ae60",  # N doping -- green
    "npp_slab_0": "#4460ad",  # N+ graded inner -- indigo
    "npp_slab_1": "#5b7dcf",  # N+ graded outer -- light indigo
    "metal1": "#7f8c8d",  # CPW electrodes (Al) -- grey
    "metal2": "#7f8c8d",
    "metal3": "#7f8c8d",
    "via_contact": "#f1c40f",  # via -- gold
    "via1": "#f1c40f",
    "via2": "#f1c40f",
}
PLOT_TITLE = "TW-MZM cross-section (zoomed on rib + doping + signal electrode)"

# %%
import matplotlib.pyplot as plt

from gsim.palace import plot_plane_section

plot_plane_section(
    section,
    colors=PLOT_COLORS,
    h_range=PLOT_ZOOM["h_range"],
    v_range=PLOT_ZOOM["v_range"],
    title=PLOT_TITLE,
)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## RF simulation (50 GHz)
#
# Configure the 50 GHz BoundaryMode run: simulation airbox, mesh, number of
# modes, output directory, and the field quantities to plot.
#

# %%
# =============================================================================
#  RF parameters (50 GHz BoundaryMode run)
# =============================================================================

# --- Simulation box (airbox + mesh margins, um) -----------------------------
AIRBOX = {"margin_x": 50.0, "margin_y": 50.0, "z_above": 100.0, "z_below": 100.0}
MESH_MARGIN_X = 0.0
MESH_MARGIN_Y = 50.0

# --- RF solver settings -----------------------------------------------------
F_RF = 50e9
NUM_RF_MODES = 2
RF_OUTPUT_DIR = "./palace-sim-mzm-pn"
RF_MESH = {
    "preset": "default",
    "refined_mesh_size": 0.05,
    "max_mesh_size": 40.0,
    "fmax": 150e9,
}

# --- RF post-processing -----------------------------------------------------
RF_FIELD = "E_real"
RF_FIELD_TITLE = "RF Mode |E| at x=0 (50 GHz)"
RIB_PHYSICAL_GROUPS = [
    "n_rib",
    "npp_slab_0",
    "npp_slab_1",
    "p_rib",
    "pp_slab_0",
    "pp_slab_1",
]

# --- PN junction lumped model -------------------------------------------------
PN_JUNCTION_CAPACITANCE = 1e-15  # F, on the p_rib / n_rib interface

# %%
from gsim.palace import BoundaryModeSim

# -- Boundary 2D simulation setup (RF) ----------------------------------
sim = BoundaryModeSim()
sim.set_output_dir(RF_OUTPUT_DIR)
sim.set_stack(stack)
sim.set_airbox(**AIRBOX)
sim.set_geometry(comp)

sim.set_cross_section(f"{CROSS_SECTION_AXIS}={CROSS_SECTION_VALUE}")
sim.set_boundary_mode(freq=F_RF, num_modes=NUM_RF_MODES, save=2)

# -- Mesh ---------------------------------------------------------------------
sim.mesh(
    preset=RF_MESH["preset"],
    refined_mesh_size=RF_MESH["refined_mesh_size"],
    max_mesh_size=RF_MESH["max_mesh_size"],
    fmax=RF_MESH["fmax"],
    margin_x=MESH_MARGIN_X,
    margin_y=MESH_MARGIN_Y,
)

# Show the 2D domain groups created by the solver
domain_groups = list(sim._last_mesh_result.groups["volumes"].keys())
print("2D domain groups:", domain_groups)
sim.print_mesh_stats()

# %%
# Interactive 3D mesh visualisation
sim.plot_mesh(
    transparent_groups=["air__None", "air__passive", "oxide__passive"],
    style="solid",
    interactive=True,
)

# %%
# Generate Palace config file (mesh must be present)
sim.add_impedance_boundary("p_rib", "n_rib", capacitance=PN_JUNCTION_CAPACITANCE)
sim.write_config()
print("Config written to:", sim.output_dir)

# %%
# -- RF simulation (50 GHz) ------------------------------------------------
results = sim.run_local(verbose=True)
results.print()

# %%
# --- Post-processing: RF mode fields ---
import importlib

import gsim.palace.field_viz as field_viz
import gsim.palace.results as palace_results
from gsim.palace import plot_fields_2d

importlib.reload(field_viz)
importlib.reload(palace_results)

if not hasattr(results, "modes"):
    results = palace_results.load_text_results(RF_OUTPUT_DIR)

results.print()

pl = plot_fields_2d(
    RF_OUTPUT_DIR,
    field=RF_FIELD,
    title=RF_FIELD_TITLE,
)

# %%
# --- Rib waveguide zoom: select by physical group names ---
from gsim.palace import plot_fields_2d

pl = plot_fields_2d(
    RF_OUTPUT_DIR,
    field=RF_FIELD,
    physical_groups=RIB_PHYSICAL_GROUPS,
    title="RF Mode |E| in the Rib Waveguide (50 GHz, zoomed)",
)

# %% [markdown]
# ## Optical simulation (1550 nm)
#
# The same cross-section is analysed at optical frequencies to compute the mode
# confined in the rib waveguide. The airbox/mesh margins come from the RF
# section (cells run in order). `run_local()` locates the Palace binary
# internally — there is no manual `subprocess` / binary-search here.
#

# %%
# =============================================================================
#  Optical parameters (1550 nm BoundaryMode run)
# =============================================================================

F_OPT = 193.4e12  # ~1550 nm
LAMBDA_OPT = 1.55  # reference wavelength (um)
NUM_OPT_MODES = 4
TARGET_INDEX = 2.5
TOLERANCE = 1e-8
OPT_OUTPUT_DIR = "./palace-sim-mzm-pn-opt"
OPT_MESH = {
    "preset": "default",
    "refined_mesh_size": 0.02,
    "max_mesh_size": 0.5,
}

# %%
# -- Optical setup + run (1550 nm) ---------------------------------------
sim_opt = BoundaryModeSim()
sim_opt.set_output_dir(OPT_OUTPUT_DIR)
sim_opt.set_stack(stack)
sim_opt.set_airbox(**AIRBOX)
sim_opt.set_geometry(comp)

sim_opt.set_cross_section(f"{CROSS_SECTION_AXIS}={CROSS_SECTION_VALUE}")
sim_opt.set_boundary_mode(
    freq=F_OPT,
    num_modes=NUM_OPT_MODES,
    save=2,
    target=TARGET_INDEX,
    tolerance=TOLERANCE,
)

sim_opt.mesh(
    preset=OPT_MESH["preset"],
    refined_mesh_size=OPT_MESH["refined_mesh_size"],
    max_mesh_size=OPT_MESH["max_mesh_size"],
    margin_x=MESH_MARGIN_X,
    margin_y=MESH_MARGIN_Y,
)

opt_results = sim_opt.run_local(verbose=True)
opt_results.print()

# %% [markdown]
# ## Summary
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
# | P+ graded outer | PP (23,1) | [-15.3, -13.3] | [0, 0.09] | doped Si (pp_slab_1) | 8x10^4 |
# | N+ graded inner | NPP (24,0) | [-21.7, -23.7] | [0, 0.09] | doped Si (npp_slab_0) | 2x10^4 |
# | N+ graded outer | NPP (24,1) | [-24.7, -26.7] | [0, 0.09] | doped Si (npp_slab_1) | 8x10^4 |
# | Vias (S to P+) | VIAC/VIA1/VIA2 | [-15.0, -9.0] | [0.09, 3.2] | W/Al | 3.5x10^7 |
# | Vias (G to N+) | VIAC/VIA1/VIA2 | [-31.0, -25.0] | [0.09, 3.2] | W/Al | 3.5x10^7 |
# | CPW signal | M1 (41,0) | [-10, +10] | [3.2, 4.2] | Al (1 um) | 3.5x10^7 |
# | CPW ground (top) | M1 (41,0) | [+30, +70] | [3.2, 4.2] | Al (1 um) | 3.5x10^7 |
# | CPW ground (bot) | M1 (41,0) | [-70, -30] | [3.2, 4.2] | Al (1 um) | 3.5x10^7 |
#
# **Material modelling notes:**
# - Doping regions are modelled as **semiconductors** (finite sigma from the Drude free-carrier model), not metals. This avoids short-circuiting the PN junction.
# - Conductivities are derived from $\sigma = q\mu N$ with typical dopant concentrations ($N \sim 10^{19}\ \text{cm}^{-3}$ for the junction, $\sim 10^{20}\ \text{cm}^{-3}$ for the contacts).
# - Doping on each side of the rib uses a **configurable piecewise gradient** via `make_doping_profile()`, and the whole cross-section assembly is wrapped by `build_doped_cross_section()`.
# - The **depletion region** and voltage-dependent capacitance are NOT modelled here — this is a linear small-signal analysis at a fixed bias point.
# - The **plasma-dispersion effect** is not applied to the optical simulation; the rib is treated as intrinsic Si at 1550 nm.
#
# **Next steps (user action):**
# 1. Verify the zoomed cross-section plot shows the rib (centred at y=-20), PN junction, graded doping, and vias.
# 2. Run `sim.run_local(num_processes=4, verbose=True)` with a Palace CPU runner installed (`pip install gsim[palace-toolkit-cpu]`).
# 3. Run `sim_opt.run_local(num_processes=4, verbose=True)` for the optical mode.
# 4. Use `gsim.palace.plot_fields_2d()` to visualise mode profiles, and `gsim.palace.plot_plane_section()` for cross-section physical groups.
#

# %%
# Quick verification: dump the 2D cross-section layer regions
print("=== RF cross-section regions (x=0) ===")
for r in section:
    print(
        f"  {r.layer_name:12s}  mat={r.material:10s}  "
        f"y=[{r.y0:6.2f},{r.y1:6.2f}]  z=[{r.zmin:5.3f},{r.zmax:5.3f}]"
    )
