# PDK validation results variant

Static dashboard for the Meep-vs-PDK comparisons from branch `pdk-validation`
(PR #406).

- `data/<case>.csv` — raw `s_parameters.csv` produced by the MEEP runner
  (`wavelength,S11_mag,S11_phase,…`).
- `data/manifest.json` — case list, mesh/frequency settings, simulated cell size
  in λ², and wall time when recorded.
- `index.html` — Plotly dashboard; reads the CSVs from `data/`.

Publish or update with:

```sh
_tools/publish_variant.sh results path/to/results \
  "PDK validation results" "Meep FDTD vs PDK reference data"
```

`size_lambda2` is the **simulated cell** (component + 0.5 µm margins + 1 µm PML
per side) at λ = 1.55 µm, not the bare component bounding box.
