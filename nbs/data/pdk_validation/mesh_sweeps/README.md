# UBC directional-coupler mesh sweep

`../runs/2d_coupler/` is the archived 30 pixels/µm baseline. `coupler_r50/` contains the 50 pixels/µm S-parameters,
generated Meep configuration, diagnostics and comparison summary. Both use the same GDS and the same asymmetric-slab
core permittivity (8.1081139); see `../GEOMETRY_AUDIT.md` for the physical comparison and its limits.

The 50 pixels/µm run used one Meep process and 101 frequencies:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python scripts/run_pdk_validation.py \
  --cases coupler --dimension 2d --resolution 50 --num-freqs 101 \
  --wall-time 90 --max-time 300 --processes 1 \
  --python /path/to/python-with-meep --output /tmp/gsim-coupler-r50-slab
```

The code now pins the 2-D core index to the local slab instead of the full-stack substrate mode. The archived files are
simulation output, not a measurement or a convergence certificate.
