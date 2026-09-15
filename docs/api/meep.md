# Meep API

## Simulation

::: gsim.meep.Simulation
    options:
      show_source: false
      inherited_members: false
      members:
        - geometry
        - source
        - domain
        - solver
        - validate_config
        - write_config
        - plot_2d
        - plot_3d
        - run
        - start
        - upload
        - get_status
        - wait_for_results

## Configuration

::: gsim.meep.Geometry
    options:
      show_source: false
      inherited_members: false
      members: false

::: gsim.meep.Domain
    options:
      show_source: false
      inherited_members: false
      members: false

`x_bounds`, `y_bounds`, and `z_bounds` set exact PML-inner intervals in
absolute micrometers; PML is added outside them. Leave an axis as `"auto"`
to size it from the component bounding box and that axis's margin:

```python
sim.domain(
    pml=1.0,
    margin_x=1.0,
    y_bounds=(-4.0, 4.0),
    z_bounds=(0.0, 3.0),
)
```

Do not combine an explicit bound with `margin_x` or `margin_y` on the same
axis. In XZ 2D simulations Y is collapsed, so `y_bounds` is not active.

::: gsim.meep.ModeSource
    options:
      show_source: false
      inherited_members: false
      members: false

## Results

::: gsim.meep.SParameterResult
    options:
      show_source: false
      inherited_members: false
      members:
        - from_csv
        - from_directory
        - plot
        - show_animation
        - show_diagnostics
