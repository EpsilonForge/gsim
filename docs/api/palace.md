# Palace API

## Simulation Classes

::: gsim.palace.DrivenSim
    options:
      show_source: false
      inherited_members: false
      members:
        - set_output_dir
        - set_geometry
        - set_stack
        - set_driven
        - set_material
        - set_numerical
        - add_port
        - add_cpw_port
        - add_pec
        - mesh
        - plot_mesh
        - plot_stack
        - show_stack
        - preview
        - validate_config
        - validate_mesh
        - write_config
        - run
        - start
        - upload
        - get_status
        - wait_for_results

::: gsim.palace.EigenmodeSim
    options:
      show_source: false
      inherited_members: false
      members:
        - set_output_dir
        - set_geometry
        - set_stack
        - set_eigenmode
        - set_material
        - set_numerical
        - add_port
        - add_cpw_port
        - add_pec
        - mesh
        - plot_mesh
        - plot_stack
        - show_stack
        - preview
        - validate_config
        - validate_mesh
        - run

::: gsim.palace.ElectrostaticSim
    options:
      show_source: false
      inherited_members: false
      members:
        - set_output_dir
        - set_geometry
        - set_stack
        - set_electrostatic
        - set_material
        - set_numerical
        - add_terminal
        - nets
        - add_pec
        - mesh
        - plot_mesh
        - plot_stack
        - show_stack
        - preview
        - validate_config
        - validate_mesh
        - run
        - load_capacitance

## Capacitance

::: gsim.palace.CapacitanceMatrices
    options:
      show_source: false
      inherited_members: false
      members:
        - between
        - to_ground
        - maxwell_frame
        - mutual_frame
        - problems

::: gsim.palace.load_capacitance
    options:
      show_source: false

## Mesh

`sim.mesh()` also reports **Estimated Field DOFs** for tetrahedral 3D driven and
eigenmode problems, using the configured field order. For order 2, the estimate
is `2 * unique_edges + 2 * unique_triangular_faces`, counting interior and shared
entities once. Geometry order and field order are separate. This is the input
mesh count before Palace splits interior boundaries, applies periodic constraints
or refines the mesh; it is not a bound on the final solve size. Other problem
types and mixed/non-tetrahedral meshes have a `null` estimate.

Mesh results expose a JSON-compatible `result.metadata` dictionary, also written
to `metadata.json` beside the mesh and included in cloud input uploads:

```python
result = sim.mesh()
metadata = result.metadata
metadata["schema_version"]  # 1
metadata["mesh"]["topology"]  # edges and triangular_faces
metadata["mesh"]["field_dofs"]["estimated_field_dofs"]
metadata["mesh"]["kappa"]["max"]  # numerical distortion, not formatted text
```

`sim.write_config()` refreshes the file using the effective config's field order,
problem type, solver settings and refinement controls. The result object is a
snapshot from meshing. Consumers should accept additional metadata keys so new
measurements can be added. Allocation rules remain in DataLab; these are sizing
hints. The current cloud API does not accept a separate metadata parameter, so
the metadata travels as an input artifact pending that API integration.
The root-level `metadata.json` name is reserved for these diagnostics. It is
excluded from the Palace result-cache key so measurements can evolve without
rerunning identical mesh/config inputs. The full `compute_dir_digest()` includes
the metadata by default; the cache key is not a complete bundle-integrity hash.

`sim.mesh()` reports **Worst element distortion, κ** in its summary. The value is
also available in `result.mesh_stats["kappa"]["max"]` and `sim.print_mesh_stats()`.
It uses Palace/MFEM's normalized Jacobian condition number: 1 is an ideal
equilateral tetrahedron; larger values indicate greater distortion. Curved
elements are sampled at their centers, so this is not a bound over their interiors.
Use it alongside SICN's signed validity check and solution convergence; κ alone
does not measure simulation accuracy. A numerically singular center is displayed
as infinite and stored as `max=None` with a nonzero `singular_elements` count.

::: gsim.palace.MeshConfig
    options:
      show_source: false
      inherited_members: false
      members:
        - coarse
        - default
        - fine

::: gsim.palace.generate_mesh
    options:
      show_source: false

## Nets

::: gsim.palace.Nets
    options:
      show_source: false
      inherited_members: false
      members:
        - net_at

::: gsim.palace.Net
    options:
      show_source: false
      inherited_members: false
      members:
        - layers

::: gsim.palace.extract_nets
    options:
      show_source: false

## Stack

::: gsim.palace.LayerStack
    options:
      show_source: false
      inherited_members: false
      members: false

::: gsim.palace.Layer
    options:
      show_source: false
      inherited_members: false
      members: false

## Circuit Synthesis

::: gsim.palace.CircuitSynthesis
    options:
      show_source: false
      inherited_members: false
      members:
        - nodes
        - L_inv
        - R_inv
        - C
        - port_labels
        - port_indices
        - internal_indices
        - port_loads
        - port_names
        - Y
        - port_admittance
        - port_impedance
        - s_parameters
        - port_reference_impedances
        - eigenfrequencies
        - fit_rlc

::: gsim.palace.load_circuit_synthesis
    options:
      show_source: false

## Fitting

::: gsim.palace.RLCFit
    options:
      show_source: false
      inherited_members: false
      members:
        - R
        - L
        - C
        - f0
        - Q
        - rms_error
        - z
        - y
        - to_dict

::: gsim.palace.VectorFit
    options:
      show_source: false
      inherited_members: false
      members:
        - raw
        - network
        - poles
        - residues
        - zeros
        - n_poles
        - is_stable
        - rms_error
        - is_passive
        - passivity_test
        - passivity_enforce
        - get_spurious
        - s
        - z
        - y
        - write_spice

## Fitting

::: gsim.palace.RLCFit
    options:
      show_source: false
      inherited_members: false
      members:
        - R
        - L
        - C
        - f0
        - Q
        - rms_error
        - z
        - y
        - to_dict

::: gsim.palace.fit_rlc
    options:
      show_source: false

::: gsim.palace.differential_impedance
    options:
      show_source: false

::: gsim.palace.initial_guess_rlc
    options:
      show_source: false

::: gsim.palace.z_rlc
    options:
      show_source: false

## Parameter Conversions

::: gsim.palace.s_to_z
    options:
      show_source: false

::: gsim.palace.z_to_s
    options:
      show_source: false

::: gsim.palace.s_to_y
    options:
      show_source: false

::: gsim.palace.y_to_s
    options:
      show_source: false

::: gsim.palace.z_to_y
    options:
      show_source: false

::: gsim.palace.y_to_z
    options:
      show_source: false

::: gsim.palace.is_complete
    options:
      show_source: false
