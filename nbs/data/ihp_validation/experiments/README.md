# IHP L2n0 experiment matrix

All runs at **1 GHz** on the authoritative stack (280 µm substrate, no passivation). `L` is the mandated series extraction; `L diff` is the differential extraction used by the port-comparison notebook.

| variant | L 1GHz nH | Q | L diff nH | pass | notes |
| --- | ---: | ---: | ---: | --- | --- |
| baseline | 1.599 | 0.28 | 0.749 | True | {} |
| ground0 | 1.882 | 0.20 | 0.754 | True | {'ground_pad_um': 0.0} |
| ground0_pec | 1.921 | 0.21 | 0.802 | True | {'ground_pad_um': 0.0, 'pec_boundary': True} |
| openems_ground | 1.654 | 0.24 | 0.732 | True | {'openems_ground': True} |
| pec | 1.639 | 0.29 | 0.796 | True | {'pec_boundary': True} |
| mesh1.5 | 1.623 | 0.26 | 0.754 | True | {'refined_mesh_size': 1.5} |
| planar_pec | -148188.292 | 10.34 | -2294.982 | False | {'planar_conductors': True} |
| substrate180 | 1.599 | 0.28 | 0.749 | True | {'substrate_thickness_um': 180.0} |
| gap | 0.780 | 3.06 | 0.780 | True | {'port_variant': 'gap'} |

Best series result: `ground0_pec` 1.921 nH vs measured 1.911/1.932 nH. Differential comparison: ground0_pec 0.802 nH and gap 0.780 nH vs measured differential 1.674/1.672 nH — both ~2.1x low. The series match is therefore not a full validation.

Finite conductors use Palace `Conductivity` + `Thickness` surface-impedance BCs. For a good conductor, skin depth δ=sqrt(2/(ωμσ)) and surface resistance Rs=1/(σδ); the committed configs carry σ=3.77e7 S/m and the PDK layer thicknesses (2/3/3.75 µm). `planar_pec` is numerically invalid and marked pass=false.

OpenEMS-like ground: Metal1 rectangle (-240..240 × -140..10 µm) with Metal1→TopMetal1 ports; the exact central PEC pillar is still approximated.
