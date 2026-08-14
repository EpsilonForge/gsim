# GDSFactory FDTD API

`gsim.fdtd` generates the coarse tetrahedral mesh and validated `config.json`
consumed by GDSFactory FDTD. The backend voxelizes this mesh onto its own Yee
grid, so the Gmsh mesh does not need to resolve the electromagnetic fields.

## PDK-native workflow

Pass the PDK module when it exposes project-level `MATERIAL_CARDS`; otherwise,
pass a PDK object or use the active PDK. Material names are resolved exactly,
using the project's cards first and gsim's built-in cards as fallbacks.

```python
import gpdk

from gsim import fdtd

simulation = fdtd.Simulation(pdk=gpdk)
simulation.geometry("mmi1x2")
artifacts = simulation.write("fdtd_output")

print(artifacts.mesh_path)  # fdtd_output/mesh.msh
print(artifacts.config_path)  # fdtd_output/config.json
```

The generated mesh is ASCII Gmsh MSH 2.2 with linear tetrahedra for material
regions and linear triangles for `port_<name>` groups. Geometry and wavelength
values in the artifacts are in nanometers. PML extrusion is left to GDSFactory
FDTD.

## Initial geometry limits

The backend supports disconnected polygons, polygon holes, axis-aligned guided
ports, and vertical or constant-angle sidewalls. Tapered layers use a small
number of midpoint-sampled prisms selected from the Yee-cell size, keeping the
lateral approximation error below one quarter cell while leaving field
resolution to the backend voxelizer.

Vertical `vertical_te` and `vertical_tm` ports are free-space apertures rather
than material-owned eigenmode ports. By default a vertical port becomes a
plane/fiber monitor while the first guided port is excited. Select the vertical
port explicitly to generate a Gaussian-beam source:

```python
simulation = fdtd.Simulation(pdk=gpdk, default_port="o2")
simulation.geometry("grating_coupler_elliptical")
simulation.write("fdtd_output/grating")
```

The aperture defaults to a square using the port width, top-facing `+z`, with a
beam waist equal to half the aperture width. Override these policies with
`vertical_port_axis`, `vertical_port_aperture_width_um`, and
`vertical_port_waist_radius_um`.

The initial implementation rejects unsupported `bias`/`z_to_bias` profiles and
lossy material snapshots because config schema version 1 accepts only real
scalar refractive indices.

## Reference

::: gsim.fdtd.Simulation
    options:
      show_source: false

::: gsim.fdtd.SimulationArtifacts
    options:
      show_source: false

::: gsim.fdtd.MeshManifest
    options:
      show_source: false

::: gsim.fdtd.FDTDConfig
    options:
      show_source: false
