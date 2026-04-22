# MEEP XZ-Plane 2D Simulations (Grating Couplers)

**Status:** design approved, ready for implementation plan **Target module:** `gsim.meep` **Motivation:** enable 2D FDTD
simulations of grating couplers (fiber-to-chip) in the XZ cross-section plane, analogous to the Tidy3D 2D grating
coupler example.

## Background

`gsim.meep` currently supports:

- Full 3D FDTD (`is_3d=True`, default).
- Effective-index 2D FDTD in the **XY plane** (`is_3d=False`). This collapses Z, ignores the stack, and enforces TE
  parity — good for top-down device studies, useless for grating couplers where the physics is vertical.

Grating couplers need the **XZ plane**: light propagates in X, the stack (substrate / BOX / Si core / partial etch /
cladding) varies in Z, geometry is assumed invariant in Y. The user's reference workflow is the Tidy3D 2D grating
coupler notebook — fiber above the chip at 14.5°, waveguide mode monitor at the end of a feed straight.

## Scope

**In scope**

- New `plane="xz"` mode in the FDTD solver config (when `is_3d=False`).
- GDS cross-section cutter: slice a gdsfactory component along Y=`y_cut`, producing XZ rectangles per layer.
- Background slabs from the layer stack (substrate, BOX, cladding) — unlike XY 2D, these ARE required.
- New `sim.source.fiber(...)` method for a tilted Gaussian beam above the chip.
- Existing waveguide port-based mode monitors reused (e.g. `o2` at the end of the feed straight).
- Cloud runner (`script.py`) XZ branch.
- XZ visualization (`sim.plot_2d(slices="y")`).
- Example notebook `nbs/_meep_2d_xz_gc.ipynb`.
- Unit tests for the cross-section cutter; an integration test driving a tiny GC sim end-to-end.

**Out of scope (deferred)**

- `sim.monitors.fiber(...)` for the reciprocal direction (waveguide → fiber emission). Spec notes the hook;
  implementation in a follow-up.
- YZ plane cross-section.
- Oblique cuts (cut lines not aligned with X).
- Multiple fiber sources per sim.

## Non-goals

- Backward-compatibility shims for the existing XY 2D mode. It remains unchanged; `plane` defaults to `"xy"`.
- A parametric (stack-only, no-GDS) geometry path. Users always provide a gdsfactory component.
- YZ / oblique simulations in this spec.

## Architecture

### User-facing API

```python
import gdsfactory as gf
from gsim import meep

# Component: grating coupler + feed straight (as in existing XY notebook)
c = gf.Component()
gc_r = c.add_ref(gf.components.grating_coupler_elliptical())
s_r = c.add_ref(gf.components.straight(length=3))
s_r.connect("o1", gc_r.ports["o1"])

sim = meep.Simulation()

# Geometry: same as before, but now with an optional y_cut
sim.geometry(component=c, y_cut=0.0)   # y_cut defaults to component bbox Y-center

sim.materials = {"si": 3.47, "SiO2": 1.44}

# NEW: fiber source (Gaussian beam above the chip at an angle)
sim.source.fiber(
    x=0.0,                  # beam-center X on the chip plane
    z_offset=1.0,            # distance above top of cladding (um)
    angle_deg=14.5,          # tilt from +Z normal; + tilts toward +X
    waist=5.4,               # beam waist / 2 (um), ≈ fiber MFD / 2
    wavelength=1.55,
    wavelength_span=0.05,
    num_freqs=21,
    polarization="TE",       # "TE" → Ey (E along waveguide width, out of XZ plane); "TM" → Ex (in-plane)
)

# Monitor the feed-waveguide port — standard port monitor
sim.monitors = ["o2"]

sim.domain(pml=1.0, margin=0.5)

# NEW: plane flag (only meaningful when is_3d=False)
sim.solver(resolution=25, is_3d=False, plane="xz")
sim.solver.stop_when_energy_decayed()

result = sim.run()
result.plot_interactive()       # shows XZ field + S-parameters
```

### API changes (Pydantic models)

**`gsim/meep/models/api.py`**

- `Geometry` gains `y_cut: float | None = None` (None → resolve at `build_config` time to bbox Y-center).
- `FDTD` gains `plane: Literal["xy", "xz"] = "xy"`. Validator: if `is_3d=True` and `plane != "xy"`, raise with a clear
  message.
- `ModeSource` stays unchanged.
- New `FiberSource` model:
  ```python
  class FiberSource(BaseModel):
      x: float                          # beam-center X on chip plane (um)
      z_offset: float                   # above cladding top (um)
      angle_deg: float = 0.0            # tilt from +Z normal
      waist: float                      # beam waist (um)
      wavelength: float = 1.55
      wavelength_span: float = 0.05
      num_freqs: int = 21
      polarization: Literal["TE", "TM"] = "TE"
  ```
- The `Simulation` holder exposes the fiber builder as `sim.source.fiber(**kwargs)`. Internally, `sim.source` becomes a
  thin dispatcher that holds either a `ModeSource` OR a `FiberSource` (only one active at a time). Calling `.fiber(...)`
  replaces any existing `ModeSource`.

**`gsim/meep/models/config.py`**

- `SimConfig` gains `plane: Literal["xy", "xz"] = "xy"`.
- `SimConfig` gains `fiber_source: FiberSourceConfig | None = None`. When set, takes precedence over `source`
  (ModeSource-based).
- New `FiberSourceConfig` mirroring the API model but with pre-computed k-direction and placement resolved against the
  stack (e.g. `cladding_top_z` already baked in).

### New module: `gsim/common/cross_section.py`

A solver-agnostic XZ cross-section extractor. Lives in `common` (not `meep`) so palace/fdtd could reuse later.

```python
@dataclass(frozen=True)
class Rect2D:
    x0: float
    x1: float
    zmin: float
    zmax: float
    layer_name: str
    material: str

def extract_xz_rectangles(
    component: "gf.Component",
    layer_stack: LayerStack,
    y_cut: float,
) -> list[Rect2D]:
    """Slice the component at Y=y_cut, return one Rect2D per layer×interval.

    For each layer in ``layer_stack``:
      1. Get polygons for the layer's gds_layer from the component.
      2. Intersect each polygon with the horizontal line Y=y_cut using shapely.
      3. Union the resulting 1D X-intervals.
      4. Emit one Rect2D per interval at the layer's (zmin, zmax).

    Holes and multi-polygon results are handled via shapely.Polygon/MultiPolygon.
    Empty layers (no intersection) are skipped.
    """
```

Contract:

- Input polygons come from `component.get_polygons(merge=True)` (the existing runner path).
- Output list is **not** merged across layers — each layer keeps its own rectangles even if they overlap in (x, z) with
  another layer. Ordering of rectangles preserves layer z-order.
- No background slabs are added here; those come from the stack in the runner.

### Runner changes: `gsim/meep/script.py`

The runner already branches on `is_3d`. Add a second branch keyed on `plane`.

Key changes in the generated runner template:

1. **Cell size.** When `plane=="xz"`:

   ```python
   cell_size = mp.Vector3(cell_x, 0.0, cell_z)
   ```

   meep treats any axis with `size=0` as the invariant axis, automatically running 2D.

1. **Geometry.** When `plane=="xz"`:

   - **Do** build background slabs from the stack (substrate/BOX/cladding). This is the opposite of XY 2D, where slabs
     are skipped. Each background slab becomes a `mp.Block` spanning the full X extent at the layer's `(zmin, zmax)`.
   - Extract foreground rectangles from the cutter (`extract_xz_rectangles`). Each `Rect2D` becomes an
     `mp.Block(size=Vector3(x1-x0, mp.inf, zmax-zmin), center=..., material=...)`.
   - Do **not** use prism extrusion in XZ — rectangles are sufficient since the geometry is invariant in Y. Holes in the
     original GDS become separate intervals via shapely intersection; no Delaunay triangulation needed.
   - Sidewall angles: ignored in XZ (same as XY 2D).

1. **Fiber source.** When `config["fiber_source"]` is present, construct a `mp.GaussianBeamSource`:

   ```python
   theta = math.radians(fs["angle_deg"])
   k_dir = mp.Vector3(math.sin(theta), 0.0, -math.cos(theta))

   # Polarization mapping (PIC convention, not meep's historical 2D TE/TM naming):
   #   "TE" — E along the waveguide width (Y axis), i.e. out of the XZ plane.
   #          This matches TE waveguide modes (Ey dominant) and is the standard
   #          fiber-to-chip GC polarization.
   #   "TM" — E in the XZ plane (Ex dominant), perpendicular to the waveguide.
   e_dir = mp.Vector3(0, 1, 0) if fs["polarization"] == "TE" else mp.Vector3(1, 0, 0)

   src = mp.GaussianBeamSource(
       src=mp.GaussianSource(frequency=fcen, fwidth=fwidth, is_integrated=True),
       center=mp.Vector3(fs["x"], 0, cladding_top + fs["z_offset"]),
       size=mp.Vector3(source_plane_x_size, 0, 0),   # a line source in X at the beam plane
       beam_x0=mp.Vector3(fs["x"], 0, cladding_top + fs["z_offset"]),
       beam_kdir=k_dir,
       beam_w0=fs["waist"],
       beam_E0=e_dir,
   )
   ```

   Size of the source line: equal to the full cell X extent minus PML (conservative).

1. **Ports.** The existing port-extraction logic (`extract_port_info`) still runs; ports come from the gdsfactory
   component. For XZ mode:

   - Each port's Z-center is computed from the highest-n layer midpoint (already done by `_get_z_center`).
   - A port is **valid** in XZ mode if `|port.center.y - y_cut| <= port.width / 2` AND its normal axis is X. Ports that
     fail this are dropped with `logger.warning("Dropping port %s (does not intersect y_cut=%.3f)")`.
   - The drop is done client-side in `simulation.py` before serializing. The runner sees only valid ports.

1. **Mode decomposition at waveguide port.** In an XZ 2D cell (`cell.y == 0`), meep treats the cell as 2D. An X-facing
   port becomes a 1D monitor line extending in Z at fixed X. `mpb.ModeSolver` + `get_eigenmode_coefficients` already
   support this — no code change.

1. **Diagnostic plots.** `script.py`'s geometry-diagnostic plotting path currently saves XY/XZ/YZ slices for 3D and
   XY-only for 2D. Add an "XZ-only" branch for `plane=="xz"`.

### Visualization: `gsim/meep/viz.py`

`plot_2d(slices="z")` currently plots the XY plane at z=z_center.

- Accept `slices="y"` → plot XZ plane at y=`y_cut`.
- Internally: reuse the existing slice infrastructure in `gsim/common/viz/render2d.py`, but drive the cutter with
  `y_cut` instead of `z_center`.
- When the simulation has `plane="xz"`, `plot_2d()` defaults to `slices="y"`.

### Simulation wiring: `gsim/meep/simulation.py`

- `Simulation.build_config()`:
  - If `geometry.y_cut` is None, resolve it to `component.dbbox().center().y`.
  - If `solver.plane == "xz"`:
    - Drop non-intersecting ports (with warning) before passing to `extract_port_info`.
    - Require at least one port **or** a fiber source; error if neither.
    - Pass through `plane` and `fiber_source` into `SimConfig`.
  - If `fiber_source` set and `is_3d=True`: error (fiber source is XZ-2D-only).
  - If `fiber_source` set and `plane=="xy"`: error (same reason).

## Data flow

```
User code
   │
   ▼
Simulation (Pydantic)
   │ build_config()
   ▼
SimConfig JSON  +  layout.gds
   │ upload to cloud
   ▼
run_meep.py (generated runner)
   │ reads JSON, imports GDS via gdsfactory
   ▼
   ├─ if plane=="xz":
   │     background slabs from layer_stack (full X extent, per-layer Z)
   │     foreground rectangles from extract_xz_rectangles(component, stack, y_cut)
   │     fiber source constructed if fiber_source in config
   │     cell = (cell_x, 0, cell_z)
   │
   └─ else (xy or 3d): existing pipeline
   │
   ▼
mp.Simulation → run → mode decomposition → s_params.csv + field snapshots
   │ download
   ▼
Result object, plot_interactive() etc.
```

## Error handling

- `solver(plane="xz", is_3d=True)` → pydantic validator raises `ValueError("plane='xz' requires is_3d=False")`.
- `sim.source.fiber(...)` in 3D mode → runtime error in `build_config` with a message pointing to
  `is_3d=False, plane="xz"`.
- `sim.monitors = ["o1"]` where `o1` does not intersect `y_cut` → `logger.warning` and drop. If the drop leaves **zero**
  valid monitors AND no fiber source is configured, raise
  `ValueError("no valid monitors and no fiber source — nothing to observe")`.
- Empty foreground (no layer polygons intersect `y_cut`) → warning; simulation still runs on the background stack alone
  (may be intentional for pure slab studies).

## Testing

**Unit — `tests/common/test_cross_section.py`** (new)

- Simple straight waveguide (single rectangle) on a single layer → one `Rect2D`.
- Strip + slab (partial-etch): core layer + slab layer with overlapping X coverage → two `Rect2D`s with correct Z
  ranges.
- Donut polygon (one hole): cut through the hole → two intervals → two `Rect2D`s.
- Off-center cut that misses a polygon → empty list for that layer, non-empty for others.
- Y_cut exactly on a polygon edge → shapely tolerance; should not crash, produces a zero-length interval that is
  filtered out.

**Integration — `tests/meep/test_xz_2d.py`** (new)

- Build a minimal GC-like stub component (a straight + a few teeth as rectangles, no PDK dependency) on a trivial stack
  (substrate, BOX, Si, clad).
- Run `sim.run()` with `is_3d=False, plane="xz"` at low resolution (e.g. 15 px/um) and a short fiber pulse.
- Assert: job completes; returned S-params array has correct shape; coupling efficiency is non-NaN and in [0, 1].
- Use the same cloud-mock / dry-run harness the existing 2D tests use if one exists; otherwise gate on
  `GSIM_RUN_CLOUD_TESTS` env var.

**Regression**

- The existing `tests/meep/` and `tests/common/` suites must continue to pass unchanged.
- Verify the existing `nbs/_meep_2d_gc.ipynb` XY flow still runs end-to-end.
- Run `uv run pytest tests/` — full suite green before merge.

**Notebook smoke**

- `nbs/_meep_2d_xz_gc.ipynb`: end-to-end GC example. Executes in CI via existing notebook-test harness if applicable.

## File-level change inventory

| File                                 | Change                                                                                                   |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| `src/gsim/meep/models/api.py`        | Add `plane` to `FDTD`; add `FiberSource`; `y_cut` on `Geometry`; dispatcher for `sim.source.fiber(...)`. |
| `src/gsim/meep/models/config.py`     | Add `plane` + `fiber_source: FiberSourceConfig \| None` to `SimConfig`; new `FiberSourceConfig`.         |
| `src/gsim/meep/simulation.py`        | `build_config` wires through `plane`, `y_cut`, `fiber_source`; port filtering.                           |
| `src/gsim/meep/ports.py`             | Add `filter_ports_for_xz(ports, y_cut)`; no change to `extract_port_info` signature.                     |
| `src/gsim/meep/script.py`            | XZ branch: cell sizing, background slabs, foreground rectangles, fiber source, geometry diagnostics.     |
| `src/gsim/meep/viz.py`               | `plot_2d` supports `slices="y"`; defaults when `plane=="xz"`.                                            |
| `src/gsim/common/cross_section.py`   | **New.** `Rect2D` + `extract_xz_rectangles`.                                                             |
| `tests/common/test_cross_section.py` | **New.** Unit tests for the cutter.                                                                      |
| `tests/meep/test_xz_2d.py`           | **New.** Integration test for XZ GC sim.                                                                 |
| `nbs/_meep_2d_xz_gc.ipynb`           | **New.** Example notebook.                                                                               |
| `MEMORY.md` / module docstrings      | Mention XZ mode in `gsim.meep` status.                                                                   |

Each of these is < ~300 lines of incremental change except `script.py` (runner template gains ~150 lines for the XZ
branch) and the notebook.

## Open questions (noted, not blocking)

- **Fiber monitor** for reciprocal (waveguide → fiber) simulations — deferred to a follow-up. Hook:
  `sim.monitors.fiber(...)` would compute Gaussian-mode overlap at the beam plane and report power coupled to the fiber
  mode.
- **Angle sign convention** confirmed: `angle_deg` is tilt from +Z normal, positive tilts toward +X. (Matches Tidy3D.)
- **TE vs TM polarization in a tilted beam**: documented mapping is `E0=Ey` for TE (PIC convention — E along waveguide
  width, out of the XZ plane) and `E0=Ex` for TM (E in the XZ plane). For small tilt angles (10–20°) the Gaussian-beam
  approximation treats `Ex` as perpendicular to `k`; the residual misalignment is negligible.
