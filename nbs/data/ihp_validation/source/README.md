# IHP SG13G2 L2n0 benchmark source data

Pinned from the IHP Open PDK at commit `943983786b04351feb20e33e8d3d76c0e0d87cce`
(`ihp-sg13g2/libs.tech/openems/testcase/SG13_Octagon_L2n0`), Apache-2.0.

- `L_2n0_simplified.gds` — octagonal 2-turn, nominal 2 nH inductor layout.
- `die_x2y7.s2p` — measured, THRU-de-embedded GSGSG data on die x2y7.
- `die_x5y8.s2p` — measured, THRU-de-embedded GSGSG data on die x5y8.
- `diff_die_x2y7.s1p`, `diff_die_x5y8.s1p` — measured differential one-port
  files for the same two dies. These are separate measurement products, not
  individual entries of the two-port S-matrices.
- `openems_twoport_1p5um.s2p` — published two-port Python OpenEMS result with
  Metal1 ground/pillar and TopMetal1 edge ports.
- `openems_oneport_1p5um.s1p`, `openems_oneport_1um.s1p` — published Python
  OpenEMS one-port results.
- `openems_matlab_oneport_1p5um.s1p`, `openems_matlab_oneport_1um.s1p`,
  `openems_matlab_oneport_0p5um.s1p`, and
  `openems_matlab_oneport_0p5um_finer_z.s1p` — published Matlab OpenEMS
  one-port results at the named mesh settings.

These are all four measured Touchstone files and all seven published OpenEMS
Touchstone result files in the pinned L2n0 testcase. Their original source
paths and hashes are recoverable from the pinned commit and `provenance.json`.

The separate [GDS2Palace L6n2 v3 study](https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2/blob/c8abeadc6ef8e4126d82f02a6adfd58c5b893c50/more_examples/measured_vs_simulated/more_accurate_models_L6n2_v3/README.md)
contains one measured two-port file, `meas_L5_6n2_THRU_deemb.S2P`. The same
Git blob is copied into its v1, v2, and v3 folders; the notebook fetches the
v3 file by pinned commit and SHA-256 rather than redistributing it.

Reference planes are the TopMetal1 feed ends at y = 0. The IHP openEMS model uses TopMetal2 windings, TopMetal1
feeds/underpass, a 280 um substrate, no passivation, IHP OpenEMS refined_cellsize=1.5 um, max_cellsize=lambda_air/(sqrt(eps_max)*20), and edge ports at the TopMetal1 feed ends referencing the Metal1/common ground pillar.
