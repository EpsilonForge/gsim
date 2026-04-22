# MEEP XZ-Plane 2D Grating Coupler — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add XZ-plane 2D FDTD simulations to `gsim.meep` so users can simulate grating couplers (fiber-to-chip)
analogously to the Tidy3D 2D grating-coupler workflow.

**Architecture:** New `plane="xz"` flag on the FDTD solver, a solver-agnostic GDS cross-section cutter in
`gsim/common/cross_section.py`, a new `sim.source.fiber(...)` Gaussian-beam source, and an XZ branch in the cloud runner
template. Background slabs (substrate/BOX/cladding) are included in XZ mode (unlike XY 2D).

**Tech Stack:** Python 3.12+, Pydantic v2, gdsfactory, shapely, meep (runner-side), numpy, pytest, `uv` for execution.

**Spec:** `docs/superpowers/specs/2026-04-21-meep-xz-2d-grating-coupler-design.md`

**Ground rules:**

- Use `uv run <cmd>` for all Python invocations (or activate `.venv`).
- One TDD cycle per task: write failing test → run it (see failure) → implement → run it (see pass) → commit.
- Keep modules < ~500 lines; split if a file grows.
- Conventional commit messages (`feat:`, `fix:`, `test:`, `refactor:`, `docs:`).
- Never add `Co-Authored-By: Claude` or similar attribution.
- Run `uv run pytest tests/` before every commit for the surrounding test directory; the full suite must stay green.

______________________________________________________________________

## File Structure

| File                                 | Status | Responsibility                                                                                          |
| ------------------------------------ | ------ | ------------------------------------------------------------------------------------------------------- |
| `src/gsim/common/cross_section.py`   | create | `Rect2D` dataclass + `extract_xz_rectangles()`. Pure, solver-agnostic.                                  |
| `tests/common/__init__.py`           | create | Empty file so pytest discovers `tests/common/`.                                                         |
| `tests/common/test_cross_section.py` | create | Unit tests for the cutter.                                                                              |
| `src/gsim/meep/models/api.py`        | modify | Add `y_cut` to `Geometry`, `plane` to `FDTD`, new `FiberSource`; `ModeSource.fiber()` dispatcher.       |
| `src/gsim/meep/models/config.py`     | modify | Add `plane` + `fiber_source: FiberSourceConfig \| None` to `SimConfig`.                                 |
| `src/gsim/meep/ports.py`             | modify | Add `filter_ports_for_xz()` helper.                                                                     |
| `src/gsim/meep/simulation.py`        | modify | Wire `plane`, `y_cut`, fiber source through `build_config`; port filtering.                             |
| `src/gsim/meep/script.py`            | modify | XZ branch in the runner template: cell sizing, slabs, foreground rectangles, fiber source, diagnostics. |
| `src/gsim/meep/viz.py`               | modify | Support `plot_2d(slices="y")` for XZ preview.                                                           |
| `tests/meep/test_xz_2d.py`           | create | Integration test (cloud-gated).                                                                         |
| `nbs/_meep_2d_xz_gc.ipynb`           | create | Example notebook.                                                                                       |

______________________________________________________________________

## Task 1: Cross-section cutter — `Rect2D` + `extract_xz_rectangles`

**Files:**

- Create: `src/gsim/common/cross_section.py`
- Create: `tests/common/__init__.py` (empty)
- Create: `tests/common/test_cross_section.py`

This is the foundation. Pure function, solver-agnostic, trivially testable. No meep dependency.

### Step 1: Scaffold empty module and test file

- [ ] **Step 1.1: Create `tests/common/__init__.py`**

Write an empty file:

```python
```

- [ ] **Step 1.2: Create skeletal `src/gsim/common/cross_section.py`**

```python
"""Solver-agnostic XZ cross-section extractor.

Given a gdsfactory component and a LayerStack, produce a list of
axis-aligned rectangles in the XZ plane sliced at Y=y_cut. These
rectangles are what an XZ 2D FDTD simulation actually extrudes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import gdsfactory as gf

    from gsim.common.stack import LayerStack


@dataclass(frozen=True)
class Rect2D:
    """Axis-aligned rectangle in the XZ plane.

    Attributes:
        x0: Low X extent (um).
        x1: High X extent (um).
        zmin: Low Z extent (um).
        zmax: High Z extent (um).
        layer_name: Source layer name from the LayerStack.
        material: Material name from the LayerStack layer.
    """

    x0: float
    x1: float
    zmin: float
    zmax: float
    layer_name: str
    material: str


def extract_xz_rectangles(
    component: "gf.Component",
    layer_stack: "LayerStack",
    y_cut: float,
    *,
    eps: float = 1e-9,
) -> list[Rect2D]:
    """Slice ``component`` at ``Y=y_cut``; return one Rect2D per layer-interval.

    For each layer in ``layer_stack`` that has a GDS layer tuple:

    1. Pull polygons for that GDS layer from the component.
    2. Intersect each polygon with the horizontal line Y=y_cut using shapely.
    3. Union the resulting 1D X-intervals within that layer.
    4. Emit one Rect2D per interval at the layer's (zmin, zmax).

    Args:
        component: gdsfactory Component (may contain references).
        layer_stack: LayerStack describing which layers to extract.
        y_cut: Y coordinate of the cross-section (um).
        eps: Drop intervals shorter than this (um) — filters out zero-length
            cuts that hit a polygon edge exactly.

    Returns:
        List of Rect2D in layer-stack order, unmerged across layers.
        Layers with no intersection are skipped.
    """
    raise NotImplementedError
```

- [ ] **Step 1.3: Verify the module imports**

Run: `uv run python -c "from gsim.common.cross_section import Rect2D, extract_xz_rectangles"` Expected: no output, exit
code 0.

### Step 2: TDD the `Rect2D` dataclass

- [ ] **Step 2.1: Write failing test for `Rect2D`**

Create `tests/common/test_cross_section.py`:

```python
"""Tests for gsim.common.cross_section."""

from __future__ import annotations

import pytest

from gsim.common.cross_section import Rect2D, extract_xz_rectangles


class TestRect2D:
    def test_frozen_dataclass_equal_by_value(self):
        a = Rect2D(x0=0.0, x1=1.0, zmin=-0.1, zmax=0.1, layer_name="core", material="si")
        b = Rect2D(x0=0.0, x1=1.0, zmin=-0.1, zmax=0.1, layer_name="core", material="si")
        assert a == b
        assert hash(a) == hash(b)

    def test_frozen_cannot_mutate(self):
        r = Rect2D(x0=0.0, x1=1.0, zmin=0.0, zmax=0.1, layer_name="core", material="si")
        with pytest.raises(Exception):
            r.x0 = 5.0  # frozen dataclass
```

- [ ] **Step 2.2: Run test — should PASS already** (Rect2D is defined)

Run: `uv run pytest tests/common/test_cross_section.py::TestRect2D -v` Expected: 2 passed.

### Step 3: TDD the `extract_xz_rectangles` function — simple waveguide

- [ ] **Step 3.1: Write failing test for a single-layer straight waveguide**

Append to `tests/common/test_cross_section.py`:

```python
class TestSimpleWaveguide:
    """Single-layer strip waveguide on the core layer."""

    def _build_stack(self):
        from gsim.common.stack import Layer, LayerStack

        return LayerStack(
            pdk_name="test",
            units="um",
            layers={
                "core": Layer(
                    name="core",
                    layer=(1, 0),
                    zmin=0.0,
                    zmax=0.22,
                    thickness=0.22,
                    material="si",
                ),
            },
            materials={},
            dielectrics=[],
            simulation={},
        )

    def _build_straight(self):
        import gdsfactory as gf

        c = gf.Component()
        c.add_polygon(
            [(-5, -0.25), (5, -0.25), (5, 0.25), (-5, 0.25)],
            layer=(1, 0),
        )
        return c

    def test_cut_through_center(self):
        c = self._build_straight()
        stack = self._build_stack()

        rects = extract_xz_rectangles(c, stack, y_cut=0.0)

        assert len(rects) == 1
        r = rects[0]
        assert r.layer_name == "core"
        assert r.material == "si"
        assert r.zmin == pytest.approx(0.0)
        assert r.zmax == pytest.approx(0.22)
        assert r.x0 == pytest.approx(-5.0)
        assert r.x1 == pytest.approx(5.0)

    def test_cut_misses_waveguide(self):
        c = self._build_straight()
        stack = self._build_stack()
        rects = extract_xz_rectangles(c, stack, y_cut=10.0)
        assert rects == []
```

- [ ] **Step 3.2: Run test — should FAIL with `NotImplementedError`**

Run: `uv run pytest tests/common/test_cross_section.py::TestSimpleWaveguide -v` Expected: 2 FAILED with
`NotImplementedError`.

- [ ] **Step 3.3: Implement the function**

Replace the body of `extract_xz_rectangles` in `src/gsim/common/cross_section.py`:

```python
def extract_xz_rectangles(
    component: "gf.Component",
    layer_stack: "LayerStack",
    y_cut: float,
    *,
    eps: float = 1e-9,
) -> list[Rect2D]:
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union

    polygons_by_layer = component.get_polygons(merge=True)

    rects: list[Rect2D] = []

    for layer_name, layer in layer_stack.layers.items():
        gds_layer = tuple(layer.layer) if not isinstance(layer.layer, tuple) else layer.layer
        layer_polys = polygons_by_layer.get(gds_layer, [])
        if not layer_polys:
            continue

        # Convert gdsfactory polygons (Nx2 arrays) to shapely Polygons.
        shapely_polys = [Polygon(poly) for poly in layer_polys if len(poly) >= 3]
        if not shapely_polys:
            continue

        merged = unary_union(shapely_polys)

        # Bound the cut line by the polygon bbox to keep intersection finite.
        minx, miny, maxx, maxy = merged.bounds
        if y_cut < miny - eps or y_cut > maxy + eps:
            continue

        cut_line = LineString([(minx - 1.0, y_cut), (maxx + 1.0, y_cut)])
        intersection = merged.intersection(cut_line)

        intervals = _line_intervals(intersection)

        for x0, x1 in intervals:
            if x1 - x0 <= eps:
                continue
            rects.append(
                Rect2D(
                    x0=x0,
                    x1=x1,
                    zmin=layer.zmin,
                    zmax=layer.zmax,
                    layer_name=layer_name,
                    material=layer.material,
                )
            )

    return rects


def _line_intervals(intersection) -> list[tuple[float, float]]:
    """Extract sorted, merged (x0, x1) intervals from a shapely intersection.

    Handles LineString, MultiLineString, empty, and degenerate Point results.
    """
    from shapely.geometry import LineString, MultiLineString

    if intersection.is_empty:
        return []

    lines: list[LineString] = []
    if isinstance(intersection, LineString):
        lines = [intersection]
    elif isinstance(intersection, MultiLineString):
        lines = list(intersection.geoms)
    else:
        # GeometryCollection / Point / MultiPoint: pull out any LineStrings.
        for geom in getattr(intersection, "geoms", []):
            if isinstance(geom, LineString):
                lines.append(geom)

    intervals: list[tuple[float, float]] = []
    for line in lines:
        xs = [coord[0] for coord in line.coords]
        intervals.append((min(xs), max(xs)))

    intervals.sort()

    # Merge overlapping intervals (shapely usually does this, but be defensive).
    merged: list[list[float]] = []
    for x0, x1 in intervals:
        if merged and x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    return [(a, b) for a, b in merged]
```

- [ ] **Step 3.4: Run test — should PASS**

Run: `uv run pytest tests/common/test_cross_section.py::TestSimpleWaveguide -v` Expected: 2 passed.

- [ ] **Step 3.5: Commit**

```bash
git add src/gsim/common/cross_section.py tests/common/__init__.py tests/common/test_cross_section.py
git commit -m "feat(common): add XZ cross-section cutter for GDS components"
```

### Step 4: TDD — partial etch (two-layer strip + slab)

- [ ] **Step 4.1: Write failing test**

Append to `tests/common/test_cross_section.py`:

```python
class TestPartialEtch:
    """Two-layer strip + slab: core rectangle on top of a wider slab."""

    def _build_stack(self):
        from gsim.common.stack import Layer, LayerStack

        return LayerStack(
            pdk_name="test",
            units="um",
            layers={
                "slab": Layer(
                    name="slab",
                    layer=(2, 0),
                    zmin=0.0,
                    zmax=0.09,
                    thickness=0.09,
                    material="si",
                ),
                "core": Layer(
                    name="core",
                    layer=(1, 0),
                    zmin=0.0,
                    zmax=0.22,
                    thickness=0.22,
                    material="si",
                ),
            },
            materials={},
            dielectrics=[],
            simulation={},
        )

    def _build_component(self):
        import gdsfactory as gf

        c = gf.Component()
        # Core strip: narrow, centered on y=0
        c.add_polygon(
            [(-3, -0.25), (3, -0.25), (3, 0.25), (-3, 0.25)],
            layer=(1, 0),
        )
        # Slab layer: wider, full extent
        c.add_polygon(
            [(-3, -1.5), (3, -1.5), (3, 1.5), (-3, 1.5)],
            layer=(2, 0),
        )
        return c

    def test_cut_through_both_layers(self):
        c = self._build_component()
        stack = self._build_stack()
        rects = extract_xz_rectangles(c, stack, y_cut=0.0)

        layers = {r.layer_name for r in rects}
        assert layers == {"slab", "core"}

        core = next(r for r in rects if r.layer_name == "core")
        slab = next(r for r in rects if r.layer_name == "slab")

        assert core.zmin == pytest.approx(0.0) and core.zmax == pytest.approx(0.22)
        assert slab.zmin == pytest.approx(0.0) and slab.zmax == pytest.approx(0.09)
        # Core extent narrower than slab extent at y=0:
        assert (core.x1 - core.x0) <= (slab.x1 - slab.x0) + 1e-6

    def test_cut_through_slab_only(self):
        c = self._build_component()
        stack = self._build_stack()
        rects = extract_xz_rectangles(c, stack, y_cut=1.0)

        layers = {r.layer_name for r in rects}
        assert layers == {"slab"}  # core polygon does not extend to y=1.0
```

- [ ] **Step 4.2: Run test — should PASS**

Run: `uv run pytest tests/common/test_cross_section.py::TestPartialEtch -v` Expected: 2 passed (implementation from Step
3 already handles multiple layers).

### Step 5: TDD — polygon with a hole

- [ ] **Step 5.1: Write failing test**

Append:

```python
class TestPolygonWithHole:
    """Donut polygon: outer ring with interior hole."""

    def _build_stack(self):
        from gsim.common.stack import Layer, LayerStack

        return LayerStack(
            pdk_name="test",
            units="um",
            layers={
                "core": Layer(
                    name="core",
                    layer=(1, 0),
                    zmin=0.0,
                    zmax=0.22,
                    thickness=0.22,
                    material="si",
                ),
            },
            materials={},
            dielectrics=[],
            simulation={},
        )

    def _build_donut(self):
        import gdsfactory as gf

        # gdsfactory represents holes via separate XOR polygons. Simulate a
        # donut as a 10x2 outer rectangle with a 4x1 inner rectangle cut out
        # using the boolean path.
        outer = gf.Component()
        outer.add_polygon(
            [(-5, -1), (5, -1), (5, 1), (-5, 1)],
            layer=(1, 0),
        )
        inner = gf.Component()
        inner.add_polygon(
            [(-2, -0.5), (2, -0.5), (2, 0.5), (-2, 0.5)],
            layer=(1, 0),
        )
        donut = gf.boolean(outer, inner, operation="not", layer=(1, 0))
        return donut

    def test_cut_through_hole_splits_into_two_intervals(self):
        c = self._build_donut()
        stack = self._build_stack()
        rects = extract_xz_rectangles(c, stack, y_cut=0.0)

        core_rects = sorted(
            (r for r in rects if r.layer_name == "core"),
            key=lambda r: r.x0,
        )
        assert len(core_rects) == 2
        # Left piece: x in [-5, -2], right piece: x in [2, 5]
        assert core_rects[0].x0 == pytest.approx(-5.0)
        assert core_rects[0].x1 == pytest.approx(-2.0)
        assert core_rects[1].x0 == pytest.approx(2.0)
        assert core_rects[1].x1 == pytest.approx(5.0)
```

- [ ] **Step 5.2: Run test**

Run: `uv run pytest tests/common/test_cross_section.py::TestPolygonWithHole -v` Expected: 1 passed.

- [ ] **Step 5.3: Commit**

```bash
git add tests/common/test_cross_section.py
git commit -m "test(common): add partial-etch and hole cases for cross-section cutter"
```

### Step 6: TDD — edge-case cut on polygon boundary

- [ ] **Step 6.1: Write failing test**

Append:

```python
class TestEdgeCaseCut:
    """Cut line exactly on a polygon edge should not crash."""

    def _build_stack(self):
        from gsim.common.stack import Layer, LayerStack

        return LayerStack(
            pdk_name="test",
            units="um",
            layers={
                "core": Layer(
                    name="core",
                    layer=(1, 0),
                    zmin=0.0,
                    zmax=0.22,
                    thickness=0.22,
                    material="si",
                ),
            },
            materials={},
            dielectrics=[],
            simulation={},
        )

    def test_cut_on_edge(self):
        import gdsfactory as gf

        c = gf.Component()
        c.add_polygon(
            [(-5, 0.0), (5, 0.0), (5, 1.0), (-5, 1.0)],
            layer=(1, 0),
        )
        stack = self._build_stack()

        # Cut exactly on the bottom edge: shapely intersection is a
        # LineString from (-5, 0) to (5, 0). The full interval should be
        # returned (not filtered out).
        rects = extract_xz_rectangles(c, stack, y_cut=0.0)

        core_rects = [r for r in rects if r.layer_name == "core"]
        assert len(core_rects) == 1
        assert core_rects[0].x0 == pytest.approx(-5.0)
        assert core_rects[0].x1 == pytest.approx(5.0)
```

- [ ] **Step 6.2: Run test**

Run: `uv run pytest tests/common/test_cross_section.py::TestEdgeCaseCut -v` Expected: 1 passed.

- [ ] **Step 6.3: Run the full cross_section suite**

Run: `uv run pytest tests/common/test_cross_section.py -v` Expected: 7+ passed total.

- [ ] **Step 6.4: Commit**

```bash
git add tests/common/test_cross_section.py
git commit -m "test(common): cover edge-on-cut case for cross-section cutter"
```

______________________________________________________________________

## Task 2: Port filter for XZ mode

**Files:**

- Modify: `src/gsim/meep/ports.py`
- Modify: `tests/meep/` (new test file: `tests/meep/test_ports.py` if absent, else append)

### Step 1: Check existing test file

- [ ] **Step 1.1: Look for existing port tests**

Run: `uv run python -c "import pathlib; p = pathlib.Path('tests/meep/test_ports.py'); print(p.exists())"`

If `False`, create `tests/meep/test_ports.py`. If `True`, append.

### Step 2: TDD `filter_ports_for_xz`

- [ ] **Step 2.1: Write failing test**

Create or append `tests/meep/test_ports.py`:

```python
"""Tests for gsim.meep.ports."""

from __future__ import annotations

import pytest

from gsim.meep.models.config import PortData
from gsim.meep.ports import filter_ports_for_xz


def _port(name: str, x: float, y: float, orientation: float, width: float = 0.5):
    normal_axis = 0 if orientation in (0, 180) else 1
    direction = "-" if orientation in (0, 90) else "+"
    return PortData(
        name=name,
        center=[x, y, 0.0],
        orientation=orientation,
        width=width,
        normal_axis=normal_axis,
        direction=direction,
    )


class TestFilterPortsForXZ:
    def test_keeps_port_intersecting_cut(self):
        # x-facing port at y=0 with width 0.5; y_cut=0 → included.
        ports = [_port("o1", x=0.0, y=0.0, orientation=180, width=0.5)]
        kept = filter_ports_for_xz(ports, y_cut=0.0)
        assert [p.name for p in kept] == ["o1"]

    def test_drops_port_off_cut(self):
        # Port at y=3.0, width 0.5: cut at y=0 is 3.0 away — too far.
        ports = [_port("o1", x=0.0, y=3.0, orientation=180, width=0.5)]
        kept = filter_ports_for_xz(ports, y_cut=0.0)
        assert kept == []

    def test_drops_y_facing_port(self):
        # y-facing port (normal_axis=1): not meaningful in an XZ 2D cell.
        ports = [_port("o1", x=0.0, y=0.0, orientation=90, width=0.5)]
        kept = filter_ports_for_xz(ports, y_cut=0.0)
        assert kept == []

    def test_partial_overlap_included(self):
        # Port at y=0.2 with width 0.5 → extends from y=-0.05 to y=0.45.
        # Cut at y=0 falls inside → included.
        ports = [_port("o1", x=0.0, y=0.2, orientation=180, width=0.5)]
        kept = filter_ports_for_xz(ports, y_cut=0.0)
        assert [p.name for p in kept] == ["o1"]

    def test_mixed_ports(self):
        ports = [
            _port("wg_in", x=-5.0, y=0.0, orientation=180, width=0.5),
            _port("wg_out", x=5.0, y=0.0, orientation=0, width=0.5),
            _port("y_oriented", x=0.0, y=0.0, orientation=90, width=0.5),
            _port("far_port", x=0.0, y=10.0, orientation=180, width=0.5),
        ]
        kept = filter_ports_for_xz(ports, y_cut=0.0)
        assert {p.name for p in kept} == {"wg_in", "wg_out"}
```

- [ ] **Step 2.2: Run test — should FAIL with `ImportError`**

Run: `uv run pytest tests/meep/test_ports.py -v` Expected: FAILED with
`ImportError: cannot import name 'filter_ports_for_xz' from 'gsim.meep.ports'`.

- [ ] **Step 2.3: Implement `filter_ports_for_xz`**

Append to `src/gsim/meep/ports.py`:

```python
def filter_ports_for_xz(
    ports: list[PortData],
    y_cut: float,
) -> list[PortData]:
    """Return ports compatible with an XZ 2D cross-section at Y=y_cut.

    Keeps an X-facing port (normal_axis=0) only when its Y span
    ``[center.y - width/2, center.y + width/2]`` straddles ``y_cut``.

    Drops:
      - Y-facing ports (not meaningful in an XZ 2D cell).
      - X-facing ports whose mode slice misses the cut entirely.

    Emits a ``logger.warning`` for each dropped port.
    """
    import logging

    log = logging.getLogger(__name__)

    kept: list[PortData] = []
    for p in ports:
        if p.normal_axis != 0:
            log.warning(
                "Dropping port %r for XZ 2D sim (normal_axis=%d != 0)",
                p.name,
                p.normal_axis,
            )
            continue

        y_center = p.center[1]
        if abs(y_center - y_cut) > p.width / 2:
            log.warning(
                "Dropping port %r for XZ 2D sim "
                "(center.y=%.4f, width=%.4f does not intersect y_cut=%.4f)",
                p.name,
                y_center,
                p.width,
                y_cut,
            )
            continue

        kept.append(p)

    return kept
```

- [ ] **Step 2.4: Run test — should PASS**

Run: `uv run pytest tests/meep/test_ports.py -v` Expected: 5 passed.

- [ ] **Step 2.5: Commit**

```bash
git add src/gsim/meep/ports.py tests/meep/test_ports.py
git commit -m "feat(meep): filter_ports_for_xz helper for XZ 2D mode"
```

______________________________________________________________________

## Task 3: API models — `FiberSource` + `plane` + `y_cut`

**Files:**

- Modify: `src/gsim/meep/models/api.py`
- Modify: `tests/meep/test_meep_models.py` (append)

### Step 1: TDD the `y_cut` field on `Geometry`

- [ ] **Step 1.1: Write failing test**

Append to `tests/meep/test_meep_models.py`:

```python
class TestGeometryYCut:
    def test_default_is_none(self):
        from gsim.meep.models.api import Geometry

        g = Geometry()
        assert g.y_cut is None

    def test_accepts_float(self):
        from gsim.meep.models.api import Geometry

        g = Geometry(y_cut=1.5)
        assert g.y_cut == 1.5
```

- [ ] **Step 1.2: Run — should FAIL**

Run: `uv run pytest tests/meep/test_meep_models.py::TestGeometryYCut -v` Expected: FAILED with `ValidationError` (extra
field).

- [ ] **Step 1.3: Add `y_cut` field to `Geometry` in `src/gsim/meep/models/api.py`**

Find the `Geometry` class and add the field after `z_crop`:

```python
    y_cut: float | None = Field(
        default=None,
        description=(
            "Y coordinate of the XZ cross-section cut (um). "
            "Only meaningful when solver.is_3d=False and solver.plane='xz'. "
            "None → resolved to the component bbox Y-center at build time."
        ),
    )
```

- [ ] **Step 1.4: Run test — should PASS**

Run: `uv run pytest tests/meep/test_meep_models.py::TestGeometryYCut -v` Expected: 2 passed.

### Step 2: TDD the `plane` field on `FDTD` with validator

- [ ] **Step 2.1: Write failing test**

Append to `tests/meep/test_meep_models.py`:

```python
class TestFDTDPlane:
    def test_default_is_xy(self):
        from gsim.meep.models.api import FDTD

        s = FDTD()
        assert s.plane == "xy"

    def test_xz_with_is_3d_false_ok(self):
        from gsim.meep.models.api import FDTD

        s = FDTD(is_3d=False, plane="xz")
        assert s.plane == "xz"

    def test_xz_with_is_3d_true_errors(self):
        import pydantic

        from gsim.meep.models.api import FDTD

        with pytest.raises(pydantic.ValidationError, match="plane='xz' requires is_3d=False"):
            FDTD(is_3d=True, plane="xz")

    def test_setting_plane_xz_when_is_3d_true_errors(self):
        import pydantic

        from gsim.meep.models.api import FDTD

        s = FDTD()  # defaults: is_3d=True, plane="xy"
        with pytest.raises(pydantic.ValidationError, match="plane='xz' requires is_3d=False"):
            s.plane = "xz"
```

- [ ] **Step 2.2: Run — should FAIL**

Run: `uv run pytest tests/meep/test_meep_models.py::TestFDTDPlane -v` Expected: 4 FAILED.

- [ ] **Step 2.3: Add `plane` field + validator to `FDTD`**

In `src/gsim/meep/models/api.py`, at the top of the file add the import:

```python
from pydantic import BaseModel, ConfigDict, Field, model_validator
```

(Keep existing imports — just add `model_validator` to the `pydantic` import list.)

In the `FDTD` class, add the field right after `is_3d`:

```python
    plane: Literal["xy", "xz"] = Field(
        default="xy",
        description=(
            "2D simulation plane. 'xy' is the effective-index top-down "
            "sim; 'xz' is a vertical cross-section (for grating couplers, "
            "edge couplers). Only meaningful when is_3d=False."
        ),
    )
```

And add a model validator at the bottom of the class body (before `model_config` if you prefer, but
Python-order-independent):

```python
    @model_validator(mode="after")
    def _validate_plane_vs_3d(self) -> FDTD:
        if self.is_3d and self.plane == "xz":
            raise ValueError("plane='xz' requires is_3d=False")
        return self
```

- [ ] **Step 2.4: Run — should PASS**

Run: `uv run pytest tests/meep/test_meep_models.py::TestFDTDPlane -v` Expected: 4 passed.

### Step 3: TDD the `FiberSource` model

- [ ] **Step 3.1: Write failing test**

Append to `tests/meep/test_meep_models.py`:

```python
class TestFiberSource:
    def test_construct_minimal(self):
        from gsim.meep.models.api import FiberSource

        fs = FiberSource(x=0.0, z_offset=1.0, waist=5.4)
        assert fs.x == 0.0
        assert fs.z_offset == 1.0
        assert fs.waist == 5.4
        assert fs.angle_deg == 0.0
        assert fs.wavelength == 1.55
        assert fs.polarization == "TE"

    def test_construct_grating_coupler_defaults(self):
        from gsim.meep.models.api import FiberSource

        fs = FiberSource(
            x=0.0,
            z_offset=1.0,
            angle_deg=14.5,
            waist=5.4,
            wavelength=1.55,
            wavelength_span=0.05,
            num_freqs=21,
            polarization="TE",
        )
        assert fs.angle_deg == 14.5
        assert fs.num_freqs == 21

    def test_waist_must_be_positive(self):
        import pydantic

        from gsim.meep.models.api import FiberSource

        with pytest.raises(pydantic.ValidationError):
            FiberSource(x=0.0, z_offset=1.0, waist=0.0)

    def test_z_offset_must_be_non_negative(self):
        import pydantic

        from gsim.meep.models.api import FiberSource

        with pytest.raises(pydantic.ValidationError):
            FiberSource(x=0.0, z_offset=-0.1, waist=5.4)
```

- [ ] **Step 3.2: Run — should FAIL** (`FiberSource` not defined)

Run: `uv run pytest tests/meep/test_meep_models.py::TestFiberSource -v` Expected: 4 FAILED.

- [ ] **Step 3.3: Implement `FiberSource`**

Append to `src/gsim/meep/models/api.py` (after `ModeSource`):

```python
class FiberSource(BaseModel):
    """Tilted Gaussian-beam source above the chip (fiber-to-chip coupling).

    The beam center sits ``z_offset`` above the top of the cladding at
    X=``x``. The beam tilts from the +Z normal by ``angle_deg`` toward +X.
    Only valid in XZ 2D mode (``solver.is_3d=False, solver.plane='xz'``).
    """

    model_config = ConfigDict(validate_assignment=True)

    x: float = Field(description="Beam-center X on the chip plane (um)")
    z_offset: float = Field(ge=0, description="Distance above cladding top (um)")
    angle_deg: float = Field(
        default=0.0,
        description="Tilt from +Z normal; positive tilts toward +X (degrees)",
    )
    waist: float = Field(gt=0, description="Gaussian beam waist w0 (um)")
    wavelength: float = Field(default=1.55, gt=0, description="Center wavelength (um)")
    wavelength_span: float = Field(default=0.05, ge=0, description="Wavelength span (um)")
    num_freqs: int = Field(default=21, ge=1, description="Number of frequency points")
    polarization: Literal["TE", "TM"] = Field(
        default="TE",
        description=(
            "PIC convention. 'TE' → E along waveguide width (Ey, out of XZ plane); "
            "'TM' → E in the XZ plane (Ex)."
        ),
    )

    def __call__(self, **kwargs: Any) -> FiberSource:
        """Update fields in place. Returns self for chaining."""
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self
```

- [ ] **Step 3.4: Run — should PASS**

Run: `uv run pytest tests/meep/test_meep_models.py::TestFiberSource -v` Expected: 4 passed.

### Step 4: TDD the `sim.source.fiber(...)` dispatcher

This method turns `ModeSource` into a dispatcher that can swap itself out for a `FiberSource`. We implement it as a
method on `ModeSource` that mutates a parent reference. But that's hard: `ModeSource` is a child field of `Simulation`.

Simpler approach: add a method on `Simulation` itself.

- [ ] **Step 4.1: Write failing test**

Append to `tests/meep/test_meep_models.py`:

```python
class TestSimulationFiberSource:
    def test_sim_fiber_helper_replaces_source(self):
        from gsim.meep.models.api import FiberSource, ModeSource
        from gsim.meep.simulation import Simulation

        sim = Simulation()
        sim.solver.is_3d = False
        sim.solver.plane = "xz"

        sim.source_fiber(
            x=0.0,
            z_offset=1.0,
            angle_deg=14.5,
            waist=5.4,
            wavelength=1.55,
            wavelength_span=0.05,
            num_freqs=21,
            polarization="TE",
        )

        assert isinstance(sim.fiber_source, FiberSource)
        assert sim.fiber_source.angle_deg == 14.5
        # ModeSource is still present but build_config must prefer fiber_source.
        assert isinstance(sim.source, ModeSource)

    def test_sim_fiber_helper_rejects_when_is_3d_true(self):
        from gsim.meep.simulation import Simulation

        sim = Simulation()  # defaults: is_3d=True
        with pytest.raises(ValueError, match="fiber source requires is_3d=False"):
            sim.source_fiber(x=0.0, z_offset=1.0, waist=5.4)
```

- [ ] **Step 4.2: Run — should FAIL** (both `Simulation.source_fiber` and `Simulation.fiber_source` missing)

Run: `uv run pytest tests/meep/test_meep_models.py::TestSimulationFiberSource -v` Expected: 2 FAILED.

- [ ] **Step 4.3: Implement on `Simulation`**

In `src/gsim/meep/simulation.py`, add to the imports:

```python
from gsim.meep.models.api import (
    FDTD,
    Domain,
    FiberSource,
    Geometry,
    Material,
    ModeSource,
)
```

Add the field to the `Simulation` class (after `solver`):

```python
    fiber_source: FiberSource | None = Field(
        default=None,
        description=(
            "Gaussian-beam fiber source for XZ 2D grating-coupler sims. "
            "When set, takes precedence over mode-source `source`."
        ),
    )
```

Add a method:

```python
    def source_fiber(self, **kwargs: Any) -> FiberSource:
        """Configure a tilted Gaussian-beam fiber source (XZ 2D only).

        Replaces any previous fiber source. Requires ``solver.is_3d=False``
        and ``solver.plane='xz'`` at run time; this method only checks
        ``is_3d`` because ``plane`` may not yet be set.
        """
        if self.solver.is_3d:
            raise ValueError(
                "fiber source requires is_3d=False (and plane='xz') — "
                "currently is_3d=True"
            )
        self.fiber_source = FiberSource(**kwargs)
        return self.fiber_source
```

- [ ] **Step 4.4: Run — should PASS**

Run: `uv run pytest tests/meep/test_meep_models.py::TestSimulationFiberSource -v` Expected: 2 passed.

- [ ] **Step 4.5: Commit**

```bash
git add src/gsim/meep/models/api.py src/gsim/meep/simulation.py tests/meep/test_meep_models.py
git commit -m "feat(meep): add plane, y_cut, FiberSource for XZ 2D mode"
```

______________________________________________________________________

## Task 4: Config model additions

**Files:**

- Modify: `src/gsim/meep/models/config.py`
- Modify: `tests/meep/test_meep_models.py` (append)

### Step 1: TDD `plane` and `FiberSourceConfig` on `SimConfig`

- [ ] **Step 1.1: Write failing test**

Append to `tests/meep/test_meep_models.py`:

```python
class TestSimConfigXZ:
    def _minimal_config_kwargs(self):
        """Minimum kwargs needed to construct SimConfig for tests."""
        return dict(
            gds_filename="layout.gds",
            layer_stack=[],
            dielectrics=[],
            ports=[],
            materials={},
            wavelength=dict(wavelength=1.55, bandwidth=0.05, num_freqs=21),
            source=dict(port=None, bandwidth=None, fwidth=0.0),
            stopping=dict(
                mode="energy_decay",
                max_time=2000.0,
                decay_dt=20.0,
                decay_component="Ey",
                threshold=0.01,
                dft_min_run_time=100.0,
            ),
            resolution=dict(pixels_per_um=25),
            domain=dict(
                dpml=1.0,
                margin_xy=0.5,
                margin_z_above=0.5,
                margin_z_below=0.5,
                port_margin=0.5,
                extend_ports=0.0,
                source_port_offset=0.1,
                distance_source_to_monitors=0.2,
            ),
            accuracy=dict(
                eps_averaging=False,
                subpixel_maxeval=0,
                subpixel_tol=1e-4,
                simplify_tol=0.0,
            ),
            verbose_interval=0.0,
            diagnostics=dict(
                save_geometry=True,
                save_fields=True,
                save_epsilon_raw=False,
                save_animation=False,
                animation_interval=0.5,
                preview_only=False,
                verbose_interval=0.0,
            ),
            symmetries=[],
        )

    def test_plane_default_xy(self):
        from gsim.meep.models.config import SimConfig

        cfg = SimConfig(**self._minimal_config_kwargs())
        assert cfg.plane == "xy"

    def test_plane_xz(self):
        from gsim.meep.models.config import SimConfig

        cfg = SimConfig(**self._minimal_config_kwargs(), plane="xz", is_3d=False)
        assert cfg.plane == "xz"

    def test_fiber_source_default_none(self):
        from gsim.meep.models.config import SimConfig

        cfg = SimConfig(**self._minimal_config_kwargs())
        assert cfg.fiber_source is None

    def test_fiber_source_set(self):
        from gsim.meep.models.config import FiberSourceConfig, SimConfig

        fs = dict(
            x=0.0,
            z_offset=1.0,
            angle_deg=14.5,
            waist=5.4,
            wavelength=1.55,
            wavelength_span=0.05,
            num_freqs=21,
            polarization="TE",
            k_direction=[0.25, 0.0, -0.97],  # pre-computed on the client
            center_z=2.68,                    # cladding_top + z_offset
        )
        cfg = SimConfig(
            **self._minimal_config_kwargs(),
            plane="xz",
            is_3d=False,
            fiber_source=fs,
        )
        assert isinstance(cfg.fiber_source, FiberSourceConfig)
        assert cfg.fiber_source.angle_deg == 14.5
        assert cfg.fiber_source.k_direction == [0.25, 0.0, -0.97]
```

- [ ] **Step 1.2: Run — should FAIL**

Run: `uv run pytest tests/meep/test_meep_models.py::TestSimConfigXZ -v` Expected: 4 FAILED (`plane`, `fiber_source`,
`FiberSourceConfig` missing).

- [ ] **Step 1.3: Add `FiberSourceConfig` + fields on `SimConfig`**

In `src/gsim/meep/models/config.py`, add `FiberSourceConfig` above `SimConfig`:

```python
class FiberSourceConfig(BaseModel):
    """Serialized Gaussian-beam fiber source for XZ 2D sims.

    Mirrors ``gsim.meep.models.api.FiberSource`` but with k-direction and
    center-z pre-computed on the client so the runner doesn't need the
    stack resolution logic.
    """

    model_config = ConfigDict(validate_assignment=True)

    x: float
    z_offset: float = Field(ge=0)
    angle_deg: float
    waist: float = Field(gt=0)
    wavelength: float = Field(gt=0)
    wavelength_span: float = Field(ge=0)
    num_freqs: int = Field(ge=1)
    polarization: Literal["TE", "TM"]
    k_direction: list[float] = Field(
        description="Unit k-vector in XZ plane: [sin(theta), 0, -cos(theta)]"
    )
    center_z: float = Field(
        description="Absolute Z of the beam-plane line source (um)"
    )
```

In `SimConfig`, add two fields (anywhere in the body; place near `source`):

```python
    plane: Literal["xy", "xz"] = Field(
        default="xy",
        description=(
            "2D simulation plane when is_3d=False. 'xy' → effective-index "
            "top-down; 'xz' → vertical cross-section at y=y_cut."
        ),
    )
    fiber_source: FiberSourceConfig | None = Field(
        default=None,
        description=(
            "Gaussian-beam fiber source (XZ 2D only). When set, the runner "
            "uses this instead of the EigenModeSource from `source`."
        ),
    )
    y_cut: float | None = Field(
        default=None,
        description=(
            "Y coordinate of XZ cross-section (um). Required when plane='xz'."
        ),
    )
```

- [ ] **Step 1.4: Run — should PASS**

Run: `uv run pytest tests/meep/test_meep_models.py::TestSimConfigXZ -v` Expected: 4 passed.

- [ ] **Step 1.5: Commit**

```bash
git add src/gsim/meep/models/config.py tests/meep/test_meep_models.py
git commit -m "feat(meep): add plane, y_cut, FiberSourceConfig to SimConfig"
```

______________________________________________________________________

## Task 5: Wire through `build_config` in `simulation.py`

**Files:**

- Modify: `src/gsim/meep/simulation.py`
- Modify: `tests/meep/test_simulation.py` (append)

This task resolves `y_cut` defaults, validates plane/is_3d/fiber coherence, filters ports, and serializes into
`SimConfig`.

### Step 1: Locate `build_config`

- [ ] **Step 1.1: Identify the method**

Run:
`uv run python -c "from gsim.meep.simulation import Simulation; import inspect; print(inspect.getsource(Simulation.build_config))" | head -50`

Read the output. You'll edit this method to:

1. Resolve `geometry.y_cut` default.
1. Validate plane/fiber coherence.
1. Filter ports when `plane == "xz"`.
1. Pre-compute `FiberSourceConfig` fields (`k_direction`, `center_z`).
1. Pass `plane`, `y_cut`, `fiber_source` into `SimConfig(...)`.

### Step 2: TDD y_cut default resolution

- [ ] **Step 2.1: Write failing test**

Append to `tests/meep/test_simulation.py`:

```python
class TestXZBuildConfig:
    def _build_straight_component(self):
        import gdsfactory as gf

        c = gf.Component()
        c.add_polygon(
            [(-5, -0.25), (5, -0.25), (5, 0.25), (-5, 0.25)],
            layer=(1, 0),
        )
        # Add a port at x=5, facing +X.
        c.add_port(
            name="o1",
            center=(5.0, 0.0),
            orientation=0.0,
            width=0.5,
            layer=(1, 0),
        )
        return c

    def _trivial_stack(self):
        from gsim.common.stack import Layer, LayerStack

        return LayerStack(
            pdk_name="test",
            units="um",
            layers={
                "substrate": Layer(
                    name="substrate", layer=None, zmin=-2.0, zmax=0.0,
                    thickness=2.0, material="SiO2",
                ),
                "core": Layer(
                    name="core", layer=(1, 0), zmin=0.0, zmax=0.22,
                    thickness=0.22, material="si",
                ),
                "clad": Layer(
                    name="clad", layer=None, zmin=0.22, zmax=1.0,
                    thickness=0.78, material="SiO2",
                ),
            },
            materials={},
            dielectrics=[
                {"zmin": -2.0, "zmax": 0.0, "material": "SiO2"},
                {"zmin": 0.22, "zmax": 1.0, "material": "SiO2"},
            ],
            simulation={},
        )

    def test_y_cut_defaults_to_bbox_center(self):
        from gsim.meep.simulation import Simulation

        sim = Simulation()
        sim.geometry.component = self._build_straight_component()
        sim.geometry.stack = self._trivial_stack()
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.solver.is_3d = False
        sim.solver.plane = "xz"
        sim.source_fiber(x=0.0, z_offset=1.0, waist=5.4)

        result = sim.build_config()

        # Straight is centered on y=0 → bbox center is 0.
        assert result.config.y_cut == pytest.approx(0.0, abs=1e-6)

    def test_y_cut_explicit_override(self):
        from gsim.meep.simulation import Simulation

        sim = Simulation()
        sim.geometry.component = self._build_straight_component()
        sim.geometry.y_cut = 0.1
        sim.geometry.stack = self._trivial_stack()
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.solver.is_3d = False
        sim.solver.plane = "xz"
        sim.source_fiber(x=0.0, z_offset=1.0, waist=5.4)

        result = sim.build_config()
        assert result.config.y_cut == pytest.approx(0.1)

    def test_xz_plane_serializes(self):
        from gsim.meep.simulation import Simulation

        sim = Simulation()
        sim.geometry.component = self._build_straight_component()
        sim.geometry.stack = self._trivial_stack()
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.solver.is_3d = False
        sim.solver.plane = "xz"
        sim.source_fiber(x=0.0, z_offset=1.0, waist=5.4)

        result = sim.build_config()
        assert result.config.plane == "xz"
        assert result.config.is_3d is False

    def test_fiber_source_serialized_with_k_direction(self):
        import math

        from gsim.meep.simulation import Simulation

        sim = Simulation()
        sim.geometry.component = self._build_straight_component()
        sim.geometry.stack = self._trivial_stack()
        sim.materials = {"si": 3.47, "SiO2": 1.44}
        sim.solver.is_3d = False
        sim.solver.plane = "xz"
        sim.source_fiber(x=0.0, z_offset=1.0, angle_deg=14.5, waist=5.4)

        result = sim.build_config()
        fs = result.config.fiber_source
        assert fs is not None
        theta = math.radians(14.5)
        assert fs.k_direction[0] == pytest.approx(math.sin(theta))
        assert fs.k_direction[1] == pytest.approx(0.0)
        assert fs.k_direction[2] == pytest.approx(-math.cos(theta))
        # clad top is at z=1.0 → center_z = 1.0 + 1.0 = 2.0
        assert fs.center_z == pytest.approx(2.0)
```

- [ ] **Step 2.2: Run — should FAIL**

Run: `uv run pytest tests/meep/test_simulation.py::TestXZBuildConfig -v` Expected: 4 FAILED (y_cut not plumbed through,
fiber_source not in config).

### Step 3: Implement wiring in `build_config`

- [ ] **Step 3.1: Edit `build_config` in `src/gsim/meep/simulation.py`**

Inside `build_config()`, immediately after the stack is resolved and before ports are extracted, resolve y_cut:

```python
        # Resolve y_cut default for XZ 2D simulations.
        if self.solver.plane == "xz" and self.geometry.y_cut is None:
            bbox = self.geometry.component.dbbox()
            y_cut = (bbox.bottom + bbox.top) / 2.0
        else:
            y_cut = self.geometry.y_cut
```

After `ports = extract_port_info(...)`, add:

```python
        # Drop ports that don't intersect the XZ cut.
        if self.solver.plane == "xz":
            from gsim.meep.ports import filter_ports_for_xz

            ports = filter_ports_for_xz(ports, y_cut=y_cut if y_cut is not None else 0.0)
```

Pre-compute `FiberSourceConfig` if a fiber source is set. After ports are filtered:

```python
        fiber_source_cfg = None
        if self.fiber_source is not None:
            if self.solver.is_3d:
                raise ValueError(
                    "fiber source requires is_3d=False (and plane='xz')"
                )
            if self.solver.plane != "xz":
                raise ValueError(
                    "fiber source requires plane='xz'"
                )

            import math

            from gsim.meep.models.config import FiberSourceConfig

            theta = math.radians(self.fiber_source.angle_deg)
            k_direction = [math.sin(theta), 0.0, -math.cos(theta)]

            # Cladding top = max zmax across all layers in the stack.
            cladding_top = max(
                layer.zmax for layer in self.geometry.stack.layers.values()
            )
            center_z = cladding_top + self.fiber_source.z_offset

            fiber_source_cfg = FiberSourceConfig(
                x=self.fiber_source.x,
                z_offset=self.fiber_source.z_offset,
                angle_deg=self.fiber_source.angle_deg,
                waist=self.fiber_source.waist,
                wavelength=self.fiber_source.wavelength,
                wavelength_span=self.fiber_source.wavelength_span,
                num_freqs=self.fiber_source.num_freqs,
                polarization=self.fiber_source.polarization,
                k_direction=k_direction,
                center_z=center_z,
            )
```

Finally, when constructing `SimConfig(...)`, pass:

```python
            plane=self.solver.plane,
            y_cut=y_cut,
            fiber_source=fiber_source_cfg,
```

- [ ] **Step 3.2: Run — should PASS**

Run: `uv run pytest tests/meep/test_simulation.py::TestXZBuildConfig -v` Expected: 4 passed.

### Step 4: Error when no monitors AND no fiber source in XZ mode

- [ ] **Step 4.1: Write failing test**

Append:

```python
class TestXZValidation:
    def test_xz_without_monitors_or_fiber_errors(self):
        from gsim.meep.simulation import Simulation

        sim = Simulation()
        # Component with no ports, no fiber source.
        import gdsfactory as gf

        c = gf.Component()
        c.add_polygon(
            [(-5, -0.25), (5, -0.25), (5, 0.25), (-5, 0.25)],
            layer=(1, 0),
        )
        sim.geometry.component = c

        from gsim.common.stack import Layer, LayerStack

        sim.geometry.stack = LayerStack(
            pdk_name="test",
            units="um",
            layers={
                "core": Layer(
                    name="core", layer=(1, 0), zmin=0.0, zmax=0.22,
                    thickness=0.22, material="si",
                ),
            },
            materials={},
            dielectrics=[],
            simulation={},
        )
        sim.materials = {"si": 3.47}
        sim.solver.is_3d = False
        sim.solver.plane = "xz"

        # No monitors, no fiber source → should error.
        with pytest.raises(ValueError, match="no valid monitors and no fiber source"):
            sim.build_config()
```

- [ ] **Step 4.2: Run — should FAIL**

Run: `uv run pytest tests/meep/test_simulation.py::TestXZValidation -v` Expected: FAILED (either different error or no
error).

- [ ] **Step 4.3: Add the check in `build_config`**

After port filtering, add:

```python
        if self.solver.plane == "xz" and not ports and self.fiber_source is None:
            raise ValueError(
                "XZ 2D sim has no valid monitors and no fiber source — "
                "nothing to observe. Either add a port intersecting y_cut, "
                "or call sim.source_fiber(...)."
            )
```

- [ ] **Step 4.4: Run — should PASS**

Run: `uv run pytest tests/meep/test_simulation.py::TestXZValidation -v` Expected: 1 passed.

### Step 5: Full suite check + commit

- [ ] **Step 5.1: Run the full meep test suite to confirm no regression**

Run: `uv run pytest tests/meep/ -v` Expected: all previously-passing tests still pass; new tests pass.

- [ ] **Step 5.2: Commit**

```bash
git add src/gsim/meep/simulation.py tests/meep/test_simulation.py
git commit -m "feat(meep): wire plane/y_cut/fiber_source through build_config"
```

______________________________________________________________________

## Task 6: Runner — XZ cell sizing + background slabs

**Files:**

- Modify: `src/gsim/meep/script.py` (the `_MEEP_RUNNER_TEMPLATE` string)

The runner is a triple-quoted Python source template. You edit the source inside the string and it flows to the cloud.
We can't unit-test the runner directly (it imports meep), so validation is: (a) syntax-check the rendered template, (b)
the integration test in Task 11.

### Step 1: Add syntax-validity guard test

- [ ] **Step 1.1: Write the test**

Create `tests/meep/test_script_template.py`:

```python
"""Smoke tests that the generated runner script is valid Python."""

from __future__ import annotations

import ast

from gsim.meep.script import _MEEP_RUNNER_TEMPLATE


def test_runner_template_parses():
    """Generated runner must be valid Python at import time."""
    ast.parse(_MEEP_RUNNER_TEMPLATE)


def test_runner_template_contains_xz_branch():
    """XZ 2D branch should be present in the runner."""
    assert 'plane = config.get("plane", "xy")' in _MEEP_RUNNER_TEMPLATE
    assert "is_xz = plane == \"xz\"" in _MEEP_RUNNER_TEMPLATE
```

- [ ] **Step 1.2: Run — first test passes, second FAILS**

Run: `uv run pytest tests/meep/test_script_template.py -v` Expected: `test_runner_template_parses` passes,
`test_runner_template_contains_xz_branch` FAILS.

### Step 2: Add the XZ branch — cell sizing + slabs

- [ ] **Step 2.1: Edit `_MEEP_RUNNER_TEMPLATE` in `src/gsim/meep/script.py`**

Find `build_background_slabs`:

```python
def build_background_slabs(config, materials):
    """..."""
    if not config.get("is_3d", True):
        return []
    ...
```

Replace the `if not config.get("is_3d", True):` guard with a plane-aware check:

```python
def build_background_slabs(config, materials):
    """Build background mp.Block slabs from dielectric entries.

    XY 2D (plane='xy') skips slabs entirely (z-dimension collapsed).
    XZ 2D (plane='xz') DOES include slabs — they form the vertical stack.
    3D always includes slabs.
    """
    is_3d = config.get("is_3d", True)
    plane = config.get("plane", "xy")
    if not is_3d and plane == "xy":
        return []

    slabs = []
    for diel in sorted(config["dielectrics"], key=lambda d: d["zmin"]):
        mat = materials.get(diel["material"])
        if mat is None:
            continue
        zmin = diel["zmin"]
        zmax = diel["zmax"]
        thickness = zmax - zmin
        if thickness <= 0:
            continue
        # In XZ 2D, slabs extend infinitely in X (cell_y=0 is meep's invariant axis)
        # but in 3D they're infinite in XY. In both cases the size expression here
        # stays (inf, inf, thickness) because meep's cell_size determines the bounds.
        block = mp.Block(
            size=mp.Vector3(mp.inf, mp.inf, thickness),
            center=mp.Vector3(0, 0, (zmin + zmax) / 2),
            material=mat,
        )
        slabs.append(block)
    return slabs
```

### Step 3: Cell-size computation

Find the block that computes `cell_size`. It currently branches on `is_3d` for `cell_z = 0`. Update it to:

- [ ] **Step 3.1: Update cell sizing**

Look for code like `cell_size = mp.Vector3(cell_x, cell_y, cell_z)` and the preceding `if not is_3d: cell_z = 0`.
Replace with:

```python
    plane = config.get("plane", "xy")
    is_xz = plane == "xz"

    if not is_3d:
        if is_xz:
            cell_y = 0.0
            # cell_z stays as computed from z-extent of stack.
        else:
            cell_z = 0.0
```

(You may need to hoist `plane` / `is_xz` earlier in the `build_simulation` function if other branches need them; place
it near the top of that function.)

### Step 4: PML sides

- [ ] **Step 4.1: Update PML boundary layers**

Find the PML setup (likely `pml_layers = [mp.PML(thickness=dpml)]` or similar). For XZ mode, we want PML on X and Z, NOT
on Y (since cell_y=0). meep handles this automatically when a cell dimension is 0, but be defensive:

```python
    if is_xz:
        pml_layers = [
            mp.PML(thickness=dpml, direction=mp.X),
            mp.PML(thickness=dpml, direction=mp.Z),
        ]
    elif not is_3d:
        pml_layers = [
            mp.PML(thickness=dpml, direction=mp.X),
            mp.PML(thickness=dpml, direction=mp.Y),
        ]
    else:
        pml_layers = [mp.PML(thickness=dpml)]
```

### Step 5: Re-run the template tests

- [ ] **Step 5.1: Run**

Run: `uv run pytest tests/meep/test_script_template.py -v` Expected: both pass.

- [ ] **Step 5.2: Full meep suite unchanged**

Run: `uv run pytest tests/meep/ -v` Expected: all green.

- [ ] **Step 5.3: Commit**

```bash
git add src/gsim/meep/script.py tests/meep/test_script_template.py
git commit -m "feat(meep-runner): add XZ cell sizing, PML, and background slabs"
```

______________________________________________________________________

## Task 7: Runner — foreground rectangles from cross-section cutter

**Files:**

- Modify: `src/gsim/meep/script.py`

The runner CANNOT import `gsim.common.cross_section` (cloud env is frozen). We inline the function body into the
template, copy-paste style — mirror of existing `triangulate_polygon_with_holes` pattern.

### Step 1: Inline the cutter into the template

- [ ] **Step 1.1: Edit `_MEEP_RUNNER_TEMPLATE` — add the cutter functions**

In the runner template, somewhere after `load_gds_component` and before `build_geometry`, add:

```python
def extract_xz_rectangles_runner(component, layer_stack, y_cut, eps=1e-9):
    """Inlined XZ cross-section cutter (mirrors gsim.common.cross_section)."""
    from shapely.geometry import LineString, Polygon
    from shapely.ops import unary_union

    polygons_by_layer = component.get_polygons(merge=True)
    rects = []

    for layer_entry in layer_stack:
        gds_layer = tuple(layer_entry["gds_layer"])
        layer_polys = polygons_by_layer.get(gds_layer, [])
        if not layer_polys:
            continue

        shapely_polys = [Polygon(p) for p in layer_polys if len(p) >= 3]
        if not shapely_polys:
            continue

        merged = unary_union(shapely_polys)
        minx, miny, maxx, maxy = merged.bounds
        if y_cut < miny - eps or y_cut > maxy + eps:
            continue

        cut_line = LineString([(minx - 1.0, y_cut), (maxx + 1.0, y_cut)])
        intersection = merged.intersection(cut_line)
        intervals = _xz_runner_line_intervals(intersection)

        for x0, x1 in intervals:
            if x1 - x0 <= eps:
                continue
            rects.append({
                "x0": x0,
                "x1": x1,
                "zmin": layer_entry["zmin"],
                "zmax": layer_entry["zmax"],
                "layer_name": layer_entry["layer_name"],
                "material": layer_entry["material"],
            })
    return rects


def _xz_runner_line_intervals(intersection):
    from shapely.geometry import LineString, MultiLineString

    if intersection.is_empty:
        return []
    lines = []
    if isinstance(intersection, LineString):
        lines = [intersection]
    elif isinstance(intersection, MultiLineString):
        lines = list(intersection.geoms)
    else:
        for geom in getattr(intersection, "geoms", []):
            if isinstance(geom, LineString):
                lines.append(geom)

    intervals = []
    for line in lines:
        xs = [c[0] for c in line.coords]
        intervals.append((min(xs), max(xs)))
    intervals.sort()
    merged = []
    for x0, x1 in intervals:
        if merged and x0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], x1)
        else:
            merged.append([x0, x1])
    return [(a, b) for a, b in merged]
```

### Step 2: Branch `build_geometry` on plane

- [ ] **Step 2.1: Edit `build_geometry` in the template**

At the top of `build_geometry`, add XZ handling. The function currently runs the prism-extrusion loop for both 3D and
XY-2D. We branch:

```python
def build_geometry(config, materials):
    """..."""
    gds_filename = config["gds_filename"]
    component = load_gds_component(gds_filename)

    is_3d = config.get("is_3d", True)
    plane = config.get("plane", "xy")
    is_xz = plane == "xz"

    if is_xz:
        return _build_geometry_xz(config, materials, component), component

    # Existing prism-extrusion path (XY 2D or 3D) follows unchanged.
    accuracy = config["accuracy"]
    simplify_tol = accuracy["simplify_tol"]
    ...
```

Add a new function:

```python
def _build_geometry_xz(config, materials, component):
    """Build meep geometry from XZ cross-section rectangles."""
    y_cut = config.get("y_cut", 0.0) or 0.0
    rects = extract_xz_rectangles_runner(
        component, config["layer_stack"], y_cut
    )

    geometry = []
    for r in rects:
        mat = materials.get(r["material"], mp.Medium())
        width_x = r["x1"] - r["x0"]
        thickness_z = r["zmax"] - r["zmin"]
        if width_x <= 0 or thickness_z <= 0:
            continue
        center_x = (r["x0"] + r["x1"]) / 2.0
        center_z = (r["zmin"] + r["zmax"]) / 2.0
        block = mp.Block(
            size=mp.Vector3(width_x, mp.inf, thickness_z),
            center=mp.Vector3(center_x, 0.0, center_z),
            material=mat,
        )
        geometry.append(block)

    logger.info("XZ: %d rectangles extracted at y=%.4f", len(geometry), y_cut)
    return geometry
```

### Step 3: Template parses + integration hook

- [ ] **Step 3.1: Template still parses**

Run: `uv run pytest tests/meep/test_script_template.py -v` Expected: passing.

- [ ] **Step 3.2: Add a grep test**

Append to `tests/meep/test_script_template.py`:

```python
def test_runner_has_xz_geometry_path():
    assert "_build_geometry_xz" in _MEEP_RUNNER_TEMPLATE
    assert "extract_xz_rectangles_runner" in _MEEP_RUNNER_TEMPLATE
```

Run it: `uv run pytest tests/meep/test_script_template.py -v` Expected: all pass.

- [ ] **Step 3.3: Commit**

```bash
git add src/gsim/meep/script.py tests/meep/test_script_template.py
git commit -m "feat(meep-runner): inline XZ cross-section cutter and geometry path"
```

______________________________________________________________________

## Task 8: Runner — fiber Gaussian-beam source

**Files:**

- Modify: `src/gsim/meep/script.py`

### Step 1: Add fiber source builder

- [ ] **Step 1.1: Edit `build_sources` (or add a companion)**

In the runner template, find `build_sources`. Add an early-return for the fiber case:

```python
def build_sources(config):
    """..."""
    fiber = config.get("fiber_source")
    if fiber is not None:
        return _build_fiber_source(config, fiber)

    # Existing eigenmode-source path below unchanged.
    fdtd = config["fdtd"]
    ...
```

Add the builder:

```python
def _build_fiber_source(config, fiber):
    """Construct a mp.GaussianBeamSource for XZ 2D fiber coupling."""
    fdtd = config["fdtd"]
    fcen = fdtd["fcen"]
    fwidth = config["source"]["fwidth"]

    k_dir = mp.Vector3(*fiber["k_direction"])
    # Polarization (PIC convention):
    #   TE → E along waveguide width (Ey, out of XZ plane)
    #   TM → E in the XZ plane (Ex)
    if fiber["polarization"] == "TE":
        e_dir = mp.Vector3(0, 1, 0)
    else:
        e_dir = mp.Vector3(1, 0, 0)

    center = mp.Vector3(fiber["x"], 0.0, fiber["center_z"])

    # Source extends across the cell in X. Size derived from cell_x minus PML.
    domain = config["domain"]
    dpml = domain["dpml"]
    # We don't have cell_x directly here; use a generous span that the
    # simulation cell will clip: entire stack X-extent + margin.
    src_x_size = _estimate_source_x_size(config)

    src = mp.GaussianBeamSource(
        src=mp.GaussianSource(frequency=fcen, fwidth=fwidth, is_integrated=True),
        center=center,
        size=mp.Vector3(src_x_size, 0, 0),
        beam_x0=center,
        beam_kdir=k_dir,
        beam_w0=fiber["waist"],
        beam_E0=e_dir,
    )
    return [src]


def _estimate_source_x_size(config):
    """Estimate a source-line X length that spans the interior of the cell."""
    bbox = config.get("component_bbox")
    domain = config["domain"]
    dpml = domain["dpml"]
    margin_xy = domain["margin_xy"]
    if bbox is not None:
        width = bbox[2] - bbox[0]
    else:
        width = 20.0  # fallback
    # Shrink by PML on both sides; keep a little margin so we don't bleed in.
    return max(width + 2 * margin_xy - 2 * dpml, 2.0)
```

### Step 2: Verify template parses + commit

- [ ] **Step 2.1: Run template tests**

Run: `uv run pytest tests/meep/test_script_template.py -v` Expected: passing.

- [ ] **Step 2.2: Add a coverage test**

Append to `tests/meep/test_script_template.py`:

```python
def test_runner_has_fiber_source_path():
    assert "_build_fiber_source" in _MEEP_RUNNER_TEMPLATE
    assert "GaussianBeamSource" in _MEEP_RUNNER_TEMPLATE
```

Run: `uv run pytest tests/meep/test_script_template.py -v` Expected: all pass.

- [ ] **Step 2.3: Commit**

```bash
git add src/gsim/meep/script.py tests/meep/test_script_template.py
git commit -m "feat(meep-runner): add Gaussian-beam fiber source path"
```

______________________________________________________________________

## Task 9: Runner — geometry diagnostics for XZ

**Files:**

- Modify: `src/gsim/meep/script.py`

The runner's diagnostic plotting path (look for `save_geometry_diagnostics` or similar) currently plots XY for 2D and
all three planes for 3D. Add XZ-only branch.

### Step 1: Locate the diagnostics function

- [ ] **Step 1.1: Skim the runner**

Run: `uv run python -c "from gsim.meep import script; print('save_geometry' in script._MEEP_RUNNER_TEMPLATE)"` Expected:
`True`.

Search for the substring `output_plane` in the template. This is where meep's `plot2D` is called with a specific slice.

### Step 2: Add XZ diagnostics branch

- [ ] **Step 2.1: Edit the geometry-diagnostics section**

When `plane == "xz"`, only produce an XZ (y=0) slice. Replace any current 2D-only XY diagnostic branch with:

```python
        if is_xz:
            # XZ cross-section at y=0 (cell is invariant in Y).
            fig, ax = plt.subplots(figsize=(10, 4))
            sim.plot2D(
                ax=ax,
                output_plane=mp.Volume(
                    center=mp.Vector3(0, 0, (zmin + zmax) / 2),
                    size=mp.Vector3(cell_x, 0, cell_z),
                ),
            )
            ax.set_title(f"XZ cross-section at y={config.get('y_cut', 0.0):.3f}")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "geometry_xz.png"), dpi=120)
            plt.close(fig)
        elif not is_3d:
            # Existing XY-only 2D path.
            ...
```

(Adapt variable names to what's in scope: `zmin`/`zmax` likely are computed elsewhere; reuse existing locals.)

### Step 3: Template still parses; commit

- [ ] **Step 3.1: Run**

Run: `uv run pytest tests/meep/test_script_template.py -v` Expected: passing.

- [ ] **Step 3.2: Commit**

```bash
git add src/gsim/meep/script.py
git commit -m "feat(meep-runner): add XZ geometry diagnostic plot"
```

______________________________________________________________________

## Task 10: Client-side visualization — `plot_2d(slices="y")`

**Files:**

- Modify: `src/gsim/meep/viz.py`
- Modify: `tests/meep/` (add `tests/meep/test_viz_xz.py`)

### Step 1: Locate the slice dispatch

- [ ] **Step 1.1: Understand current viz**

Run: `uv run python -c "from gsim.meep.viz import plot_2d; import inspect; print(inspect.signature(plot_2d))"`

Inspect how `slices="z"` routes through `gsim/common/viz/render2d.py`. You'll mirror that code path for a `"y"` slice
(XZ plane).

### Step 2: TDD the XZ preview

- [ ] **Step 2.1: Write test**

Create `tests/meep/test_viz_xz.py`:

```python
"""Visualization tests for XZ 2D preview."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pytest


def _xz_simulation():
    import gdsfactory as gf

    from gsim.common.stack import Layer, LayerStack
    from gsim.meep.simulation import Simulation

    c = gf.Component()
    c.add_polygon(
        [(-5, -0.25), (5, -0.25), (5, 0.25), (-5, 0.25)],
        layer=(1, 0),
    )
    c.add_port(name="o1", center=(5.0, 0.0), orientation=0.0, width=0.5, layer=(1, 0))

    stack = LayerStack(
        pdk_name="test",
        units="um",
        layers={
            "substrate": Layer(
                name="substrate", layer=None, zmin=-2.0, zmax=0.0,
                thickness=2.0, material="SiO2",
            ),
            "core": Layer(
                name="core", layer=(1, 0), zmin=0.0, zmax=0.22,
                thickness=0.22, material="si",
            ),
            "clad": Layer(
                name="clad", layer=None, zmin=0.22, zmax=1.0,
                thickness=0.78, material="SiO2",
            ),
        },
        materials={},
        dielectrics=[
            {"zmin": -2.0, "zmax": 0.0, "material": "SiO2"},
            {"zmin": 0.22, "zmax": 1.0, "material": "SiO2"},
        ],
        simulation={},
    )

    sim = Simulation()
    sim.geometry.component = c
    sim.geometry.stack = stack
    sim.materials = {"si": 3.47, "SiO2": 1.44}
    sim.solver.is_3d = False
    sim.solver.plane = "xz"
    sim.source_fiber(x=0.0, z_offset=1.0, waist=5.4)
    return sim


class TestPlot2DXZ:
    def test_slices_y_returns_figure(self):
        sim = _xz_simulation()
        fig = sim.plot_2d(slices="y")
        assert fig is not None
        # Smoke check: figure has at least one axis.
        assert len(fig.axes) >= 1

    def test_default_slice_when_plane_xz(self):
        sim = _xz_simulation()
        fig = sim.plot_2d()  # default should be "y" when plane=="xz"
        assert fig is not None
```

- [ ] **Step 2.2: Run — should FAIL**

Run: `uv run pytest tests/meep/test_viz_xz.py -v` Expected: FAILED (slices="y" unsupported).

### Step 3: Implement `slices="y"` path

- [ ] **Step 3.1: Extend `plot_2d` in `src/gsim/meep/viz.py`**

Find the existing `plot_2d` function. It currently accepts `slices: Literal["z", ...]`. Extend:

```python
    # At the top of plot_2d:
    if slices is None:
        # Auto-pick slice for the configured sim mode.
        slices = "y" if getattr(self.solver, "plane", "xy") == "xz" else "z"
```

Inside the dispatcher, add:

```python
    elif slices == "y":
        from gsim.common.cross_section import extract_xz_rectangles

        y_cut = self.geometry.y_cut
        if y_cut is None:
            bbox = self.geometry.component.dbbox()
            y_cut = (bbox.bottom + bbox.top) / 2.0

        rects = extract_xz_rectangles(
            self.geometry.component, self.geometry.stack, y_cut
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        for r in rects:
            ax.add_patch(
                plt.Rectangle(
                    (r.x0, r.zmin),
                    r.x1 - r.x0,
                    r.zmax - r.zmin,
                    facecolor="#6ab04c",
                    edgecolor="black",
                    linewidth=0.3,
                    alpha=0.8,
                    label=r.layer_name,
                )
            )

        # Background slabs (full-width rectangles from dielectrics).
        if self.geometry.stack is not None:
            for diel in self.geometry.stack.dielectrics:
                ax.add_patch(
                    plt.Rectangle(
                        (-20, diel["zmin"]),
                        40,
                        diel["zmax"] - diel["zmin"],
                        facecolor="#dfe4ea",
                        edgecolor="none",
                        alpha=0.4,
                    )
                )

        ax.relim()
        ax.autoscale_view()
        ax.set_xlabel("x (um)")
        ax.set_ylabel("z (um)")
        ax.set_title(f"XZ cross-section at y={y_cut:.3f}")
        ax.set_aspect("equal")
        fig.tight_layout()
        return fig
```

(Match the existing function signature and return type. If the existing implementation uses a shared helper for the
z-slice, factor it to symmetric helpers `_plot_slice_z` / `_plot_slice_y` and call them from the dispatcher.)

- [ ] **Step 3.2: Run — should PASS**

Run: `uv run pytest tests/meep/test_viz_xz.py -v` Expected: 2 passed.

- [ ] **Step 3.3: Commit**

```bash
git add src/gsim/meep/viz.py tests/meep/test_viz_xz.py
git commit -m "feat(meep-viz): support plot_2d(slices='y') for XZ preview"
```

______________________________________________________________________

## Task 11: Integration test (cloud-gated)

**Files:**

- Create: `tests/meep/test_xz_2d.py`

This test submits a real job if `GSIM_RUN_CLOUD_TESTS=1` is set; otherwise it only builds the config and checks
shape/serialization.

### Step 1: Author the test

- [ ] **Step 1.1: Create `tests/meep/test_xz_2d.py`**

```python
"""Integration test for XZ 2D grating coupler simulation."""

from __future__ import annotations

import os

import pytest


def _build_minimal_gc_sim():
    """Grating-coupler-ish stub: straight waveguide + 5 teeth on core layer."""
    import gdsfactory as gf

    from gsim.common.stack import Layer, LayerStack
    from gsim.meep.simulation import Simulation

    c = gf.Component()
    # Feed waveguide from x=-8..0 at y=0, width 0.5
    c.add_polygon(
        [(-8.0, -0.25), (0.0, -0.25), (0.0, 0.25), (-8.0, 0.25)],
        layer=(1, 0),
    )
    # Grating teeth: 5 rectangles at x=0.5..5, width 0.3, pitch 0.62, height 0.5 (y-span)
    pitch = 0.62
    tooth_w = 0.3
    for i in range(5):
        x0 = i * pitch
        c.add_polygon(
            [(x0, -0.25), (x0 + tooth_w, -0.25), (x0 + tooth_w, 0.25), (x0, 0.25)],
            layer=(1, 0),
        )
    # Waveguide port at the back of the straight.
    c.add_port(
        name="o1",
        center=(-8.0, 0.0),
        orientation=180.0,
        width=0.5,
        layer=(1, 0),
    )

    stack = LayerStack(
        pdk_name="test",
        units="um",
        layers={
            "substrate": Layer(
                name="substrate", layer=None, zmin=-2.0, zmax=0.0,
                thickness=2.0, material="SiO2",
            ),
            "core": Layer(
                name="core", layer=(1, 0), zmin=0.0, zmax=0.22,
                thickness=0.22, material="si",
            ),
            "clad": Layer(
                name="clad", layer=None, zmin=0.22, zmax=1.0,
                thickness=0.78, material="SiO2",
            ),
        },
        materials={},
        dielectrics=[
            {"zmin": -2.0, "zmax": 0.0, "material": "SiO2"},
            {"zmin": 0.22, "zmax": 1.0, "material": "SiO2"},
        ],
        simulation={},
    )

    sim = Simulation()
    sim.geometry.component = c
    sim.geometry.stack = stack
    sim.materials = {"si": 3.47, "SiO2": 1.44}
    sim.solver.is_3d = False
    sim.solver.plane = "xz"
    sim.solver.resolution = 15
    sim.solver.stop_when_energy_decayed()
    sim.source_fiber(
        x=1.2,
        z_offset=1.0,
        angle_deg=14.5,
        waist=5.4,
        wavelength=1.55,
        wavelength_span=0.04,
        num_freqs=5,
    )
    sim.monitors = ["o1"]
    sim.domain.pml = 1.0
    sim.domain.margin = 0.5
    return sim


def test_xz_build_config_produces_expected_shape():
    sim = _build_minimal_gc_sim()
    result = sim.build_config()

    cfg = result.config
    assert cfg.plane == "xz"
    assert cfg.is_3d is False
    assert cfg.fiber_source is not None
    assert cfg.y_cut == pytest.approx(0.0, abs=1e-6)

    # At least one port survives the filter.
    assert any(p.name == "o1" for p in cfg.ports)


@pytest.mark.skipif(
    os.environ.get("GSIM_RUN_CLOUD_TESTS") != "1",
    reason="Cloud integration test (set GSIM_RUN_CLOUD_TESTS=1 to enable)",
)
def test_xz_end_to_end_cloud():
    sim = _build_minimal_gc_sim()
    result = sim.run()

    # S-param array exists and has shape (num_freqs,).
    assert result.s_params is not None
    import numpy as np

    s = np.asarray(result.s_params.get("o1@fiber") or result.s_params.get("o1"))
    assert s.size == 5
    assert np.all(np.isfinite(s))
    # Coupling efficiency |S|^2 is in [0, 1] (allowing numerical slack).
    assert (np.abs(s) ** 2 <= 1.01).all()
```

- [ ] **Step 1.2: Run — the non-cloud test should PASS, the cloud test should SKIP**

Run: `uv run pytest tests/meep/test_xz_2d.py -v` Expected: 1 passed, 1 skipped.

- [ ] **Step 1.3: Commit**

```bash
git add tests/meep/test_xz_2d.py
git commit -m "test(meep): add XZ 2D build + cloud integration test"
```

______________________________________________________________________

## Task 12: Example notebook

**Files:**

- Create: `nbs/_meep_2d_xz_gc.ipynb`

Modeled on `nbs/_meep_2d_gc.ipynb` but with `plane="xz"` + fiber source.

### Step 1: Create the notebook

- [ ] **Step 1.1: Generate via jupyter**

Create a new notebook file with cells:

**Cell 1 (markdown):**

```markdown
# 2D FDTD — XZ Cross-section (Grating Coupler)

This notebook demonstrates **2D FDTD in the XZ cross-section plane** using `gsim.meep`. Unlike the top-down XY effective-index sim, this one models the vertical stack (substrate/BOX/core/cladding) and a Gaussian-beam fiber source above the chip — the standard grating-coupler workflow.

**Requirements:** GDSFactory+ account for cloud simulation.
```

**Cell 2 (code):**

```python
import gdsfactory as gf

gf.gpdk.PDK.activate()

c = gf.Component()
gc_r = c.add_ref(gf.components.grating_coupler_elliptical())
s_r = c.add_ref(gf.components.straight(length=3))
s_r.connect("o1", gc_r.ports["o1"])
c
```

**Cell 3 (markdown):**

```markdown
## Configure the XZ 2D simulation

Key differences from the XY notebook:
- `sim.solver.plane = "xz"` in addition to `is_3d = False`.
- `sim.source_fiber(...)` replaces the port-based mode source.
- `sim.monitors = ["o2"]` — monitor the waveguide end (feed straight).
```

**Cell 4 (code):**

```python
from gsim import meep

sim = meep.Simulation()

sim.geometry(component=c)
sim.materials = {"si": 3.47, "SiO2": 1.44}

sim.solver(resolution=25, is_3d=False, plane="xz")
sim.solver.stop_when_energy_decayed()

sim.source_fiber(
    x=0.0,
    z_offset=1.0,
    angle_deg=14.5,
    waist=5.4,
    wavelength=1.55,
    wavelength_span=0.04,
    num_freqs=21,
    polarization="TE",
)
sim.monitors = ["o2"]
sim.domain(pml=1.0, margin=0.5)

print(sim.validate_config())
```

**Cell 5 (markdown):**

```markdown
### Preview the XZ cross-section
```

**Cell 6 (code):**

```python
sim.plot_2d(slices="y")
```

**Cell 7 (markdown):**

```markdown
### Run the simulation
```

**Cell 8 (code):**

```python
result = sim.run()
```

**Cell 9 (code):**

```python
result.plot_interactive()
```

- [ ] **Step 1.2: Execute the notebook locally to build_config only (sanity check)**

Run the first few cells up to and including the `validate_config()` call. Do NOT run `sim.run()` unless you have a cloud
account provisioned.

- [ ] **Step 1.3: Commit**

```bash
git add nbs/_meep_2d_xz_gc.ipynb
git commit -m "docs(meep): add XZ 2D grating-coupler example notebook"
```

______________________________________________________________________

## Task 13: Final regression pass

### Step 1: Run the full test suite

- [ ] **Step 1.1: All tests pass**

Run: `uv run pytest tests/ -v` Expected: all tests green. Previously-passing tests remain passing; new tests pass.

- [ ] **Step 1.2: Lint**

Run: `uv tool run ruff check src/gsim/` Expected: no errors.

- [ ] **Step 1.3: Existing XY notebook still works**

Open `nbs/_meep_2d_gc.ipynb` and verify its first few cells execute unchanged. Do not re-run the cloud simulation unless
needed.

### Step 2: Update the memory note (optional)

- [ ] **Step 2.1: Note the new capability**

If `MEMORY.md` in the user's memory directory lists `gsim.meep` status, append a line noting XZ 2D mode is available.
This is optional but helpful for future sessions.

______________________________________________________________________

## Self-review checklist (run after implementation)

- All spec items from `docs/superpowers/specs/2026-04-21-meep-xz-2d-grating-coupler-design.md` have a corresponding
  task?
  - ✅ plane flag — Task 3, 4
  - ✅ cross-section cutter — Task 1
  - ✅ background slabs in XZ — Task 6
  - ✅ fiber source — Task 3, 4, 5, 8
  - ✅ waveguide ports reused + filtered — Task 2, 5
  - ✅ runner XZ branch — Task 6, 7, 8, 9
  - ✅ visualization — Task 10
  - ✅ notebook — Task 12
  - ✅ unit tests — Task 1
  - ✅ integration test — Task 11
  - ✅ regression — Task 13
- Every task has exact file paths, complete code, and exact commands?
- Type names used consistently (`FiberSource` vs `FiberSourceConfig`, `Rect2D`, `filter_ports_for_xz`)?
- No TBD / TODO / "add appropriate error handling" placeholders?
