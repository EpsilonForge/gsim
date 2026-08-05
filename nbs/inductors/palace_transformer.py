# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
#   kernelspec:
#     display_name: .venv (3.12.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Running Palace Simulations
#
# [Palace](https://awslabs.github.io/palace/) is an open-source 3D electromagnetic simulator supporting eigenmode, driven (S-parameter), and electrostatic simulations. This notebook demonstrates using the `gsim.palace` API to run a driven simulation on an integrated RF transformer, extracting the broadband S-parameters to evaluate differential performance metrics such as insertion loss and impedance matching.
#
# **Requirements:**
# - IHP PDK: `uv pip install ihp-gdsfactory`
# - gsim with Palace backend

# %%
import gdsfactory as gf
from gdsfactory import Component
from gdsfactory.typings import LayerSpec, LayerSpecs
from ihp import PDK

PDK.activate()


# %% [markdown]
# ### Build transformer + guardring


# %%
@gf.cell
def transformer_stacked(
    width_primary: float = 5.0,
    width_secondary: float = 4.5,
    space: float = 3.1,
    diameter: float = 60.0,
    turns: int = 1,
    layer_primary: LayerSpec = "TopMetal2drawing",
    layer_secondary: LayerSpec = "TopMetal1drawing",
    layer_inductor: LayerSpec = "INDdrawing",
    layers_no_fill: LayerSpecs = ("NoMetFillerdrawing",),
) -> Component:
    """Stacked 1:1 transformer optimized for millimeter-wave frequencies.

    Two spiral inductors stacked vertically. The secondary coil is placed
    on a lower metal layer.

    The secondary is rotated 180° — standard convention for stacked transformers:
    places secondary ports on the opposite side from primary ports, which
    simplifies differential routing and avoids crossing metal layers.

    Args:
        width_primary: Track width of the top coil in micrometers.
        width_secondary: Track width of the bottom coil in micrometers.
        space: Spacing between turns in micrometers.
        diameter: Outer spiral diameter in micrometers.
        turns: Number of turns per coil.
        layer_primary: Metal layer for the primary coil.
        layer_secondary: Metal layer for the secondary coil.
        layer_inductor: IND marker layer (used by the PDK).
        layers_no_fill: Layers excluded from metal fill.

    Returns:
        Component with 4 coil ports:
          P1, P2  -> primary   (layer_primary)
          S1, S2  -> secondary (layer_secondary)
    """
    c = gf.Component()

    # -- Primary coil --
    primary = gf.components.inductor(
        width=width_primary,
        space=space,
        diameter=diameter,
        turns=turns,
        layer_metal=layer_primary,
        layer_inductor=layer_inductor,
        layer_metal_pin=layer_primary,
        layers_no_fill=layers_no_fill,
    )
    prim_ref = c.add_ref(primary)

    # Center primary at origin
    cx, cy = prim_ref.center
    prim_ref.move((-cx, -cy))

    # -- Secondary coil --
    secondary = gf.components.inductor(
        width=width_secondary,
        space=space,
        diameter=diameter,
        turns=turns,
        layer_metal=layer_secondary,
        layer_inductor=layer_inductor,
        layer_metal_pin=layer_secondary,
        layers_no_fill=layers_no_fill,
    )
    sec_ref = c.add_ref(secondary)
    sec_ref.rotate(180)

    # Center secondary at origin (same center as primary)
    cx, cy = sec_ref.center
    sec_ref.move((-cx, -cy))

    # -- Expose coil ports --
    c.add_port(name="P1", port=prim_ref.ports["P1"])
    c.add_port(name="P2", port=prim_ref.ports["P2"])
    c.add_port(name="S1", port=sec_ref.ports["P1"])
    c.add_port(name="S2", port=sec_ref.ports["P2"])

    return c


# %%
c = transformer_stacked().copy()

layer_guard_ring = "Metal5drawing"
guard_ring_margin = 15.0  # µm  distance from coil bbox to ring inner edge
guard_ring_overlap = 0.5  # µm  corner overlap so Gmsh fuses the pieces

bbox = c.bbox()
xmin, ymin = bbox.left, bbox.bottom
xmax, ymax = bbox.right, bbox.top

margin_outer = 0.0  # outer edge of ring flush with the coil bbox
margin_inner = -guard_ring_margin  # (negative) inner edge is INSIDE the bbox, so the ring walls sit tightly around the coil outline.

xlo, xro = xmin - margin_outer, xmax + margin_outer
ybo, yto = ymin - margin_outer, ymax + margin_outer
xli, xri = xmin - margin_inner, xmax + margin_inner
ybi, yti = ymin - margin_inner, ymax + margin_inner

w_v = xli - xlo  # width of vertical walls
h_h = yto - yti  # height of horizontal walls
ov = guard_ring_overlap  # corner overlap so Gmsh fuses the pieces

# Left wall
c.add_ref(
    gf.components.rectangle(
        size=(w_v + ov, yto - ybo),
        layer=layer_guard_ring,
        centered=True,
    )
).move((xlo + w_v / 2 + ov / 2, (yto + ybo) / 2))

# Right wall
c.add_ref(
    gf.components.rectangle(
        size=(w_v + ov, yto - ybo),
        layer=layer_guard_ring,
        centered=True,
    )
).move((xro - w_v / 2 - ov / 2, (yto + ybo) / 2))

# Top wall (spans full width including corners)
c.add_ref(
    gf.components.rectangle(
        size=(xro - xlo, h_h + ov),
        layer=layer_guard_ring,
        centered=True,
    )
).move(((xro + xlo) / 2, yto - h_h / 2 - ov / 2))

# Bottom wall
c.add_ref(
    gf.components.rectangle(
        size=(xro - xlo, h_h + ov),
        layer=layer_guard_ring,
        centered=True,
    )
).move(((xro + xlo) / 2, ybo + h_h / 2 + ov / 2))

cc = c.copy()

c.draw_ports()
c.plot()

# %% [markdown]
# ### Configure and run simulation with DrivenSim

# %%
from gsim.palace import DrivenSim

# Create simulation object
sim = DrivenSim()

# Set output directory
sim.set_output_dir("./palace-sim-transformer")

# Set the component geometry
sim.set_geometry(cc)

# Configure layer stack from active PDK
sim.set_stack(substrate_thickness=180.0, include_substrate=True)

# Configure ports
sim.add_port(
    "P1", from_layer="metal5", to_layer="topmetal2", geometry="via", excited=True
)
sim.add_port(
    "P2", from_layer="metal5", to_layer="topmetal2", geometry="via", excited=True
)
sim.add_port(
    "S1", from_layer="metal5", to_layer="topmetal1", geometry="via", excited=True
)
sim.add_port(
    "S2", from_layer="metal5", to_layer="topmetal1", geometry="via", excited=True
)

# Configure driven simulation (frequency sweep for S-parameters)
sim.set_driven(fmin=10e9, fmax=200e9, num_points=50)

# Validate configuration
print(sim.validate_config())

# %%
# Generate mesh (presets: "coarse", "default", "fine")
sim.set_airbox(margin_x=50, margin_y=50, z_above=50, z_below=5)
sim.mesh(preset="default", refined_mesh_size=1.5)
sim.write_config()

# %%
sim.plot_mesh(show_groups=["metal", "P"])

# %%
sim.plot_mesh(
    style="solid",
    transparent_groups=["air__None", "sio2__None", "air__sio2"],
)

# %% [markdown]
# ### Run simulation on cloud

# %%
# Run simulation on GDSFactory+ cloud
results = sim.run(parent_dir="./palace-sim-transformer")
# results = sim.run_local(use_apptainer=False,palace_executable="~/palace/build/bin/palace",num_processes=14)

# %%
results.plot_interactive()

# %%
results.plot_interactive(phase=True)

# %%
results.plot()

# %% [markdown]
# ### Mixed-Mode S-Parameter Conversion
#
# This section converts the raw 4-port single-ended simulation data into 2-port differential metrics to evaluate transformer performance.
#
# * **Data Restructuring**: Compiles the complex single-ended data into a 3D NumPy array of shape `(frequencies, ports, ports)`.
# * **Network Transformation**: Instantiates a `scikit-rf.Network` and applies `.se2gmm(p=2)` to mathematically convert single-ended ports into differential pairs.
# * **Key Metrics Plotted**:
#   * **`S11_diff` (Primary Return Loss)**: Impedance match at the input.
#   * **`S22_diff` (Secondary Return Loss)**: Impedance match at the output.
#   * **`S21_diff` (Insertion Loss)**: Magnetic power transfer efficiency across the transformer gap.

# %%
import matplotlib.pyplot as plt
import numpy as np
import skrf as rf

f = results.freq

ports = results.port_names
n = len(ports)

S = np.zeros((len(f), n, n), dtype=complex)
for i, pi in enumerate(ports):
    for j, pj in enumerate(ports):
        S[:, i, j] = results[(pi, pj)].complex

ntwk = rf.Network(f=f, s=S, f_unit="ghz")
ntwk_mixed = ntwk.copy()
ntwk_mixed.se2gmm(p=2)

plt.figure(figsize=(10, 6))
ntwk_mixed.s11.plot_s_db(label="S11_diff (primary return loss)")
ntwk_mixed.s22.plot_s_db(label="S22_diff (secondary return loss)")
ntwk_mixed.s21.plot_s_db(label="S21_diff (insertion loss)")
plt.grid()
plt.title("Differential 2-port S-parameters")
plt.show()
