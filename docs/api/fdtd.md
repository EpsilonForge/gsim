# GDSFactory FDTD API

`gsim.fdtd` separates simulation setup into geometry, materials, source,
monitors, domain, and solver concerns. It generates the transfer mesh and
validated `config.json`, submits one source to one cloud job, and returns typed
results.

## Configure and run

```python
from gsim import fdtd

sim = fdtd.Simulation(pdk=gpdk)
sim.materials(background="SiO2")
sim.geometry(component=mmi, mesh_size_nm=400)
sim.source(
    port="o1",
    wavelength_um=1.55,
    wavelength_span_um=0.1,
    num_wavelengths=101,
)
sim.domain(padding_um=0.75, pml_cells=16)
sim.solver(
    cell_size_nm=40,
    energy_decay_fraction=1e-6,
    max_wall_seconds=3600,
)

result = sim.run(check_cache=True)
result.plot()
result.plot_plotly()
```

`mesh_size_nm` controls the coarse Gmsh transfer mesh. `cell_size_nm` requests
the actual Yee grid used by the FDTD solver and defaults to 60 nm. Source sweeps
default to 101 wavelengths. Guided-port monitors are implicit; one port source
returns one S-matrix column.

Use `write(path)` to generate `mesh.msh` and `config.json` without submitting.
The original flat constructor keywords remain accepted for compatibility.

## Setup visualization

FDTD setup views use the ported ZapFDTD Three.js viewer and do not depend on
PyVista. All physical groups are shown by default. Smaller material groups are
listed first, followed by the largest volumetric group and then the ports.

```python
sim.plot_3d()                                   # solid interactive geometry
sim.plot_3d(show_mesh=True)                     # add Gmsh surface edges
sim.plot_2d(axis="z", position_um=0.11)         # filled cross-section
sim.plot_2d(axis="x", position_um=0, show_mesh=True)  # add cell edges
```

Omit `position_um` to start at the geometry midpoint. The 2D viewer includes a
slider that moves the plane through the mesh. Both methods return a standalone
`MeshViewer`. Set `show_mesh=True` when element edges are useful, or use
`viewer.save("mesh.html")` to share the visualization outside a notebook.

## Sources

`PortSource` requires an explicit `sim.source(port="...")` selection before
artifacts can be written or a job can run. A guided PDK port maps to an
eigenmode source. Selecting a `vertical_te` or `vertical_tm` PDK port instead
derives a Gaussian beam and fiber monitor from the port metadata.

Replace `sim.source` to use another physical source:

```python
sim.source = fdtd.DipoleSource(
    position_um=(0, 0, 0.11),
    current_axis="z",
    wavelength_um=1.55,
)

sim.source = fdtd.LineCurrentSource(
    position_um=(0, -0.5, 0.11),
    line_axis="y",
    current_axis="z",
    length_um=1,
)
```

`GaussianBeamSource` exposes the aperture center and size, propagation and
polarization vectors, focal point, waist, and optional background index. Only
`PortSource` on a guided port produces normalized S-parameters. Other source
types return outgoing port amplitudes and powers.

## Plane monitors

Additional plane monitors can independently record flux, scalar heatmaps, and
Gaussian fiber overlap:

```python
sim.monitors.add_plane(
    name="top",
    center_um=(0, 0, 1),
    size_um=(10, 6, 0),
    normal="+z",
    flux=True,
    heatmap=fdtd.Heatmap(
        quantity="intensity",
        wavelengths_um=[1.55],
    ),
)
```

Monitor names are unique. Use `add()`, `remove()`, and `clear()` to manage the
collection. A plane's size must be zero along its normal.

## Results

```python
result.s_parameters.plot()
result.s_parameters.plot_plotly()
s_parameter_table = result.s_parameters.to_dataframe()

result.port_outputs.plot(quantity="modal_power")
port_table = result.port_outputs.to_dataframe()

top = result.monitors["top"]
top.plot_flux()
top.plot_heatmap(wavelength_um=1.55)
```

Heatmap arrays are loaded lazily from their `.npy` sidecars. S-parameter plots
gap samples flagged below the source noise floor by default. Convergence, grid,
timing, and resolved runtime settings remain available on the result object.

Simulation setup views remain separate from these result plots: `sim.plot_*()`
inspects the submitted geometry and mesh, while `result.plot*()` displays solver
output.

## Reference

::: gsim.fdtd.Simulation
    options:
      show_source: false
      inherited_members: false
      members:
        - write
        - upload
        - start
        - get_status
        - wait_for_results
        - run
        - plot_2d
        - plot_3d

::: gsim.fdtd.PortSource
    options:
      show_source: false
      members: false

::: gsim.fdtd.DipoleSource
    options:
      show_source: false
      members: false

::: gsim.fdtd.LineCurrentSource
    options:
      show_source: false
      members: false

::: gsim.fdtd.GaussianBeamSource
    options:
      show_source: false
      members: false

::: gsim.fdtd.PlaneMonitor
    options:
      show_source: false
      members: false

::: gsim.fdtd.FDTDResult
    options:
      show_source: false
      inherited_members: false
      members:
        - from_file
        - from_run_result
        - plot
        - plot_plotly
