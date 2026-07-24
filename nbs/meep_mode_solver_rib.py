# ---
# jupyter:
#   jupytext:
#     jupytext_version: 1.19.2
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] papermill={"duration": 0.002045, "end_time": "2026-07-24T11:58:07.011380", "exception": false, "start_time": "2026-07-24T11:58:07.009335", "status": "completed"}
# # Mode Solver — Rib Waveguide (meep)
#
# Fundamental TE mode of a silicon rib waveguide: ``n_eff`` + 2D (Y, Z) field profile.
#
# **Cross-section:** SiO2 box / Si slab 70 nm / Si rib 150 nm (total 220 nm).
# Lateral confinement via effective-index contrast.

# %% [markdown] papermill={"duration": 0.000842, "end_time": "2026-07-24T11:58:07.013311", "exception": false, "start_time": "2026-07-24T11:58:07.012469", "status": "completed"}
# ### Imports

# %% papermill={"duration": 1.906519, "end_time": "2026-07-24T11:58:08.920596", "exception": false, "start_time": "2026-07-24T11:58:07.014077", "status": "completed"}
import gdsfactory as gf
import matplotlib.pyplot as plt
import numpy as np

import gsim.meep as gm
from gsim.common.stack.extractor import Layer, LayerStack

plt.close()

gf.gpdk.PDK.activate()

# %% [markdown] papermill={"duration": 0.000755, "end_time": "2026-07-24T11:58:08.922376", "exception": false, "start_time": "2026-07-24T11:58:08.921621", "status": "completed"}
# ### Build the GDS component

# %% papermill={"duration": 0.006301, "end_time": "2026-07-24T11:58:08.929378", "exception": false, "start_time": "2026-07-24T11:58:08.923077", "status": "completed"}
SLAB_WIDTH = 3.0  # um
RIB_WIDTH = 0.5  # um
LENGTH = 10.0  # um

c = gf.Component()

# Si rib layer (layer 2) - narrow ridge on top
c.add_polygon(
    [
        (-LENGTH / 2, -RIB_WIDTH / 2),
        (LENGTH / 2, -RIB_WIDTH / 2),
        (LENGTH / 2, RIB_WIDTH / 2),
        (-LENGTH / 2, RIB_WIDTH / 2),
    ],
    layer=(2, 0),
)

# Ports at both ends
c.add_port(
    name="o1", center=(-LENGTH / 2, 0), width=RIB_WIDTH, orientation=180, layer=(1, 0)
)
c.add_port(
    name="o2", center=(LENGTH / 2, 0), width=RIB_WIDTH, orientation=0, layer=(1, 0)
)

print(f"Component: {c.name}")
print(f"  Ports:  {[p.name for p in c.ports]}")
print(f"  Layers: {list(c.layers)}")

# %% [markdown] papermill={"duration": 0.00074, "end_time": "2026-07-24T11:58:08.931019", "exception": false, "start_time": "2026-07-24T11:58:08.930279", "status": "completed"}
# ### Layer stack

# %% papermill={"duration": 0.003585, "end_time": "2026-07-24T11:58:08.935300", "exception": false, "start_time": "2026-07-24T11:58:08.931715", "status": "completed"}
layers = {
    "ox": Layer(
        name="box",
        gds_layer=(0, 0),
        zmin=-1,
        zmax=0.0,
        thickness=1.0,
        material="sio2",
        layer_type="dielectric",
    ),
    "slab": Layer(
        name="slab",
        gds_layer=(1, 0),
        zmin=0.0,
        zmax=0.07,
        thickness=0.07,
        material="si",
        layer_type="dielectric",
    ),
    "rib": Layer(
        name="rib",
        gds_layer=(2, 0),
        zmin=0.07,
        zmax=0.22,
        thickness=0.15,
        material="si",
        layer_type="dielectric",
    ),
}
stack = LayerStack(layers=layers)

print("Layer stack:")
for name, l in stack.layers.items():
    print(
        f"  {name:6s}  z=[{l.zmin:+.3f}, {l.zmax:+.3f}]  t={l.thickness:.3f}  material={l.material}"
    )

# %% [markdown] papermill={"duration": 0.000739, "end_time": "2026-07-24T11:58:08.937474", "exception": false, "start_time": "2026-07-24T11:58:08.936735", "status": "completed"}
# ### Solve

# %% papermill={"duration": 104.636625, "end_time": "2026-07-24T11:59:53.574787", "exception": false, "start_time": "2026-07-24T11:58:08.938162", "status": "completed"}
WAVELENGTH = 1.55  # um
RESOLUTION = 64
PML_THICKNESS = 1 * WAVELENGTH


sim = gm.Simulation(
    geometry=gm.Geometry(component=c, stack=stack),
    domain=gm.Domain(
        pml=PML_THICKNESS,
        margin_z=(0.0, 0.5),
    ),
)
N_BANDS = 4

sim.mode_solver.wavelengths = [WAVELENGTH]
sim.mode_solver.first(N_BANDS).at_port("o1")
sim.mode_solver.y_span = SLAB_WIDTH
sim.mode_solver.n_field_y = 300
sim.mode_solver.n_field_z = 300

sweep = sim.solve_modes()

# n_eff for every band found at this wavelength
modes = sorted(sweep.at(WAVELENGTH).results, key=lambda r: r.band_num)
print(f"Solved {len(modes)} mode(s) at lambda={WAVELENGTH:.2f} µm:")
for m in modes:
    print(f"  band {m.band_num}: n_eff = {m.n_eff:.6f}  (parity={m.parity})")

# fundamental mode used for the plots below
mode = sweep.at(WAVELENGTH).band(1)

print(f"\nn_eff    = {mode.n_eff:.6f}")
print(f"n_group  = {mode.n_group}")
print(f"kdom     = {[f'{k:.6f}' for k in mode.kdom]}")
print(f"band     = {mode.band_num}, parity = {mode.parity}")
print(f"fields   = {list(mode.fields.keys())}")
for comp, arr in mode.fields.items():
    print(f"  {comp}: shape={arr.shape}  |max|={np.abs(arr).max():.6f}")

# %% [markdown] papermill={"duration": 0.001324, "end_time": "2026-07-24T11:59:53.577025", "exception": false, "start_time": "2026-07-24T11:59:53.575701", "status": "completed"}
# ### Index profile

# %% papermill={"duration": 0.093477, "end_time": "2026-07-24T11:59:53.671339", "exception": false, "start_time": "2026-07-24T11:59:53.577862", "status": "completed"}
mode.plot_index(show=True)


# %% [markdown] papermill={"duration": 0.000954, "end_time": "2026-07-24T11:59:53.673436", "exception": false, "start_time": "2026-07-24T11:59:53.672482", "status": "completed"}
# ### Mode profile (interactive)
#
# ``|field|`` maps for every component with zoom / pan / hover. Axes use equal
# aspect so the cross-section is physically proportioned.

# %% papermill={"duration": 0.203272, "end_time": "2026-07-24T11:59:53.877703", "exception": false, "start_time": "2026-07-24T11:59:53.674431", "status": "completed"}
import plotly.graph_objects as go
from plotly.subplots import make_subplots

comps = [c for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz") if c in mode.fields]

# Crop to the (small, centered) mode region and downsample so the embedded
# heatmaps stay light-weight for the docs while remaining smooth on zoom.
energy = sum(np.abs(mode.fields[c]) ** 2 for c in comps)
thr = 0.02 * energy.max()
yi = np.where(energy.max(axis=0) > thr)[0]
zi = np.where(energy.max(axis=1) > thr)[0]
dy, dz = mode.y_grid[1] - mode.y_grid[0], mode.z_grid[1] - mode.z_grid[0]
pad = 0.4  # µm margin around the mode
y0, y1 = (
    max(yi.min() - int(pad / dy), 0),
    min(yi.max() + int(pad / dy) + 1, mode.y_grid.size),
)
z0, z1 = (
    max(zi.min() - int(pad / dz), 0),
    min(zi.max() + int(pad / dz) + 1, mode.z_grid.size),
)
max_pts = 140  # per axis
sy, sz = max(1, (y1 - y0) // max_pts), max(1, (z1 - z0) // max_pts)
y_sub, z_sub = mode.y_grid[y0:y1:sy], mode.z_grid[z0:z1:sz]

fig = make_subplots(
    rows=2,
    cols=3,
    subplot_titles=[f"|{c}|" for c in comps],
    horizontal_spacing=0.08,
    vertical_spacing=0.12,
)

for i, comp in enumerate(comps):
    row, col = i // 3 + 1, i % 3 + 1
    # Mode fields are in arbitrary units; normalize each panel to its own max
    # and quantize to 8-bit (0..255). Plotly embeds arrays as binary, so this
    # keeps the notebook small while staying visually identical.
    panel = np.abs(mode.fields[comp][z0:z1:sz, y0:y1:sy])
    panel = np.round(panel / panel.max() * 255).astype(np.uint8)
    fig.add_trace(
        go.Heatmap(
            x=y_sub,
            y=z_sub,
            z=panel,
            colorscale="Inferno",
            showscale=False,
            zmin=0,
            zmax=255,
            hovertemplate=(
                "y=%{x:.3f} µm<br>z=%{y:.3f} µm<br>"
                "|" + comp + "| (rel.)=%{z}<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )
    x_anchor = "x" if i == 0 else f"x{i + 1}"
    fig.update_xaxes(title_text="y (µm)", row=row, col=col)
    fig.update_yaxes(
        title_text="z (µm)", scaleanchor=x_anchor, scaleratio=1, row=row, col=col
    )

fig.update_layout(
    height=550,
    autosize=True,
    title_text=(
        f"Rib waveguide fundamental TE mode  "
        f"(lambda={WAVELENGTH:.2f} µm, n_eff={mode.n_eff:.4f})"
    ),
)
fig.show(config={"responsive": True})
