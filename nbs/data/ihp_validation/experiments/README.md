# IHP L2n0 experiment matrix

All runs at **1 GHz** on the authoritative benchmark stack (280 µm substrate, no passivation). Primary L/Q uses the mandated series extraction.

| variant | L 1 GHz (nH) | Q | notes |
| --- | ---: | ---: | --- |
| baseline | 1.599 | 0.28 | {} |
| ground0 | 1.882 | 0.20 | {'ground_pad_um': 0.0} |
| ground0_pec | 1.921 | 0.21 | {'ground_pad_um': 0.0, 'pec_boundary': True} |
| openems_ground | 1.654 | 0.24 | {'openems_ground': True} |
| pec | 1.639 | 0.29 | {'pec_boundary': True} |
| passiv_sensitivity | 3.361 | 0.57 | {'shell_variant': True} |
| mesh1.5 | 1.623 | 0.26 | {'refined_mesh_size': 1.5} |
| planar_pec | -148188.292 | 10.34 | {'planar_conductors': True} |
| substrate180 | 1.599 | 0.28 | {'substrate_thickness_um': 180.0} |
| gap | 0.780 | 3.06 | {'port_variant': 'gap'} |

Best: `ground0_pec` = 1.921 nH vs measured 1.911/1.932 nH (within 0.5%).

`openems_ground` approximates the IHP Metal1 reference plane with Metal1→TopMetal1 ports (1.654 nH); the exact ground pillar is still simplified.

`passiv_sensitivity` (3.36 nH) shows adding passivation moves away from the pinned IHP script, which omits passivation.
