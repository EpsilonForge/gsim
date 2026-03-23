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
