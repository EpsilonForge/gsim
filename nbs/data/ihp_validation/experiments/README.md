# IHP L2n0 experiment matrix

Screening runs at 1 GHz (same extraction for all variants). The best variant is `shell_ground0`: conformal-top
passivation shell (SiO2) plus a minimal SUBGND pad. It gives 2.099 nH at 1 GHz versus measured 1.911/1.932 nH (within
10%).

| variant       | L 1 GHz (nH) | Q 1 GHz | notes                                               |
| ------------- | -----------: | ------: | --------------------------------------------------- |
| baseline      |        1.346 |  -0.861 | coarse 5/70 mesh, sin passivation, 50 um ground pad |
| shell         |        2.547 |    5.08 | passivation shell material changed to SiO2          |
| ground0       |        0.850 |  -0.876 | minimal SUBGND pad                                  |
| pec           |        0.804 |  -0.817 | PEC outer boundary                                  |
| shell_ground0 |        2.099 |   26.71 | best low-frequency result                           |
| mesh1.5       |        0.803 |  -0.909 | finer 1.5 um mesh                                   |
| gap           |        2.610 |    4.45 | exploratory single gap port across terminals        |

Full 1-50 GHz sweep of the best variant is in `runs/l2n0_best/`.

Palace's `Impedance` boundary config is for dielectric-dielectric interfaces. For metals, finite thickness is represented by the conductor shell/volume with bulk conductivity; this is the explicit volume fallback allowed by the experiment plan.
