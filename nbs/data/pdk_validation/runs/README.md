# Recorded PDK validation runs

Each `2d_<case>/` directory is one fresh validation run recorded with the canonical PDK-authoritative stack.
`sim_config.json` records the PDK name, overlay, stack hash, resolved layer/material table, resolution and provenance.

Fresh cases: UBC straight, coupler, ring coupling, MZI, ring and Bragg; Cornerstone coupler and ring coupling;
generic-photonic-PDK SiEPIC coupler. All runs use the 2-D effective-index plane and a short validation window so the
full set is reproducible on one workstation.

The legacy `_stale_inferred_stack/` runs predate the PDK-authoritative stack and are retained only for history.
