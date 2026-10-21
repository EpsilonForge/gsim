# Recorded PDK validation run data

Each completed result folder contains the raw Meep `s_parameters.csv` and its `sim_config.json`. The notebook plots and
summary metrics are in the parent directory.

The completed and exploratory runs are:

- `2d_straight/`: effective-index run, res=30, 201 frequencies.
- `2d_coupler/`: effective-index run, res=30, 201 frequencies.
- `2d_ring_coupling/`: effective-index run, res=25, 201 frequencies.
- `2d_bragg/`: effective-index run, res=30, 201 frequencies.
- `2d_ring_exploratory/`: partial run, res=15, 51 frequencies; stopped at the simulation-time cap and not converged.
- `3d_straight/`: full-PDK run, res=20, 101 frequencies.
- `3d_bragg/`: previously completed full-PDK run, res=30, 201 frequencies.

Incomplete MZI, 3-D coupler, and 3-D ring-coupling attempts have no spectra and are documented in the notebook's results
table.
