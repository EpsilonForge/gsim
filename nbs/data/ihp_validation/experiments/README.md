# IHP L2n0 experiment matrix

All runs at **1 GHz** on the authoritative stack (280 µm substrate, no passivation). `L` is the mandated series extraction; `L diff` is the port-comparison differential extraction.

| variant | L 1GHz nH | Q | L diff nH | pass | notes |
| --- | ---: | ---: | ---: | --- | --- |
| baseline | 1.599 | 0.28 | 0.749 | True | {} |
| ground0 | 1.882 | 0.20 | 0.754 | True | {'ground_pad_um': 0.0} |
| ground0_pec | 1.921 | 0.21 | 0.802 | True | {'ground_pad_um': 0.0, 'pec_boundary': True} |
| openems_ground | 1.654 | 0.24 | 0.732 | True | {'openems_ground': True} |
| pec | 1.639 | 0.29 | 0.796 | True | {'pec_boundary': True} |
| passiv_sensitivity | 3.361 | 0.57 | 0.198 | True | {'shell_variant': True} |
| mesh1.5 | 1.623 | 0.26 | 0.754 | True | {'refined_mesh_size': 1.5} |
| planar_pec | -148188.292 | 10.34 | -2294.982 | False | {'planar_conductors': True} |
| substrate180 | 1.599 | 0.28 | 0.749 | True | {'substrate_thickness_um': 180.0} |
| gap | 0.780 | 3.06 | 0.780 | True | {'port_variant': 'gap'} |

Target result: `ground0_pec` series L = 1.921 nH vs measured 1.911/1.932 nH. Its differential L is ~0.8 nH, and the gap port also gives ~0.78 nH: the differential comparison is negative. The 1.921 nH headline therefore depends on the series extraction convention and is not a full validation of the model.

`passiv_sensitivity` shows that adding passivation is inconsistent with the pinned IHP script, which omits passivation. `planar_pec` is numerically invalid and marked pass=false.
