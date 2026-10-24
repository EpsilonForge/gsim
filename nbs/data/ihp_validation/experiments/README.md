# IHP L2n0 experiment matrix

All screening runs are at **1 GHz** on the corrected benchmark stack. The primary L/Q uses the mandated series extraction; `L diff` uses Zdiff from the Z matrix for cross-check.

| variant | L 1 GHz (nH) | Q | L diff 1 GHz (nH) | notes |
| --- | ---: | ---: | ---: | --- |
| baseline | 1.394 | 0.24 | 0.752 | {} |
| shell | 2.543 | 0.44 | 0.198 | {'shell_variant': True} |
| ground0 | 1.596 | 0.18 | 0.756 | {'ground_pad_um': 0.0} |
| pec | 1.427 | 0.25 | 0.795 | {'pec_boundary': True} |
| shell_ground0 | 2.085 | 0.23 | 0.203 | {'shell_variant': True, 'ground_pad_um': 0.0} |
| mesh1.5 | 1.403 | 0.23 | 0.757 | {'refined_mesh_size': 1.5} |
| planar_pec | 8377.376 | 0.01 | -6098.689 | {'planar_conductors': True} |
| gap | 0.781 | 3.06 | nan | {'port_variant': 'gap'} |

Best: `shell_ground0` at 2.085 nH vs measured 1.911/1.932 nH (within 10%). Full 1-50 GHz sweep is in `runs/l2n0_best/`.
