"""IHP SG13G2 octagonal L2n0 benchmark helpers.

The benchmark source files are vendored under ``nbs/data/ihp_validation/source``
from IHP-Open-PDK commit ``943983786b04351feb20e33e8d3d76c0e0d87cce``
(Apache-2.0).  The helpers build the same 2-port inductor model used by the
notebook: IHP's TopMetal2 winding/feeder layout, a SUBGND reference plane, the
authoritative IHP stack, and 30 GHz interlayer ports from SUBGND to TopMetal1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[4]
SOURCE = REPO / "nbs" / "data" / "ihp_validation" / "source"
RUN_ROOT = REPO / "nbs" / "data" / "ihp_validation" / "runs"
LAYOUT = SOURCE / "L_2n0_simplified.gds"
MEASURED = {
    "die_x2y7": SOURCE / "die_x2y7.s2p",
    "die_x5y8": SOURCE / "die_x5y8.s2p",
}
TARGET_FREQ_HZ = 30e9
Z0_OHM = 50.0

#: IHP process coordinate shift used by the published reference stack.  The
#: benchmark's absolute stack is offset so TopMetal1 lands at 190.18-192.18 um.
STACK_SHIFT_UM = 183.95
SUBGND_ZMIN_UM = 180.0 - STACK_SHIFT_UM
SUBGND_ZMAX_UM = 183.75 - STACK_SHIFT_UM


@dataclass(frozen=True)
class BenchmarkRun:
    """Recorded benchmark artefacts."""

    directory: Path
    sparams: Any
    frequencies_ghz: np.ndarray


def load_benchmark_component() -> Any:
    """Import the IHP benchmark GDS and add the two reference-plane ports."""
    import gdsfactory as gf
    import ihp

    ihp.PDK.activate()
    component = gf.import_gds(LAYOUT)
    # SUBGND is an explicit reference plane, not a process layer in the GDS.
    bbox = component.dbbox()
    pad = 50.0
    component.add_polygon(
        [
            (bbox.left - pad, bbox.bottom - pad),
            (bbox.right + pad, bbox.bottom - pad),
            (bbox.right + pad, bbox.top + pad),
            (bbox.left - pad, bbox.top + pad),
        ],
        layer=(250, 0),
    )
    component.add_port(
        "P1", center=(-34.2, 0.0), width=12.0, orientation=180.0, layer=(126, 0)
    )
    component.add_port(
        "P2", center=(34.2, 0.0), width=12.0, orientation=0.0, layer=(126, 0)
    )
    return component


def build_benchmark_stack(component: Any) -> Any:
    """Build the authoritative IHP stack plus the benchmark SUBGND/EPI levels."""
    import ihp

    from gsim.common.pdk import build_layer_stack, resolve_layer_stack
    from gsim.common.stack.extractor import Layer
    from gsim.common.stack.materials import get_material_properties

    resolved = resolve_layer_stack(component, pdk=ihp.PDK, wavelength_um=1.0)
    stack = build_layer_stack(resolved, backend="palace")
    stack.layers["subgnd"] = Layer(
        name="subgnd",
        gds_layer=(250, 0),
        zmin=SUBGND_ZMIN_UM,
        zmax=SUBGND_ZMAX_UM,
        thickness=SUBGND_ZMAX_UM - SUBGND_ZMIN_UM,
        material="aluminum",
        layer_type="conductor",
        mesh_order=0,
    )
    for name in ("si", "sio2", "sin", "aluminum", "tungsten"):
        properties = get_material_properties(name)
        if properties is not None:
            stack.materials.setdefault(name, properties.to_dict())

    stack.dielectrics = [
        {
            "name": "substrate",
            "material": "si",
            "zmin": 0.0 - STACK_SHIFT_UM,
            "zmax": 180.0 - STACK_SHIFT_UM,
        },
        {
            "name": "epi",
            "material": "si",
            "zmin": 180.0 - STACK_SHIFT_UM,
            "zmax": 183.75 - STACK_SHIFT_UM,
        },
        {
            "name": "ild",
            "material": "sio2",
            "zmin": 183.75 - STACK_SHIFT_UM,
            "zmax": 196.48 - STACK_SHIFT_UM,
        },
        {
            "name": "passivation",
            "material": "sin",
            "zmin": 196.48 - STACK_SHIFT_UM,
            "zmax": 196.88 - STACK_SHIFT_UM,
        },
    ]
    return stack


def make_simulation(output_dir: str | Path, *, sweep: bool = False) -> Any:
    """Build the 2-port Palace simulation for the benchmark."""
    from gsim.palace import DrivenSim

    component = load_benchmark_component()
    stack = build_benchmark_stack(component)
    sim = DrivenSim()
    sim.set_output_dir(output_dir)
    sim.set_geometry(component)
    sim.set_stack(stack)
    sim.set_airbox(margin_x=150.0, margin_y=150.0, z_above=50.0, z_below=10.0)
    for name in ("P1", "P2"):
        sim.add_port(
            name,
            from_layer="subgnd",
            to_layer="topmetal1",
            geometry="interlayer",
            impedance=Z0_OHM,
        )
    if sweep:
        sim.set_driven(fmin=1e9, fmax=50e9, num_points=11, reference_impedance=Z0_OHM)
    else:
        sim.set_driven(f=TARGET_FREQ_HZ, num_points=1, reference_impedance=Z0_OHM)
    return sim


def run_benchmark(
    output_dir: str | Path | None = None,
    *,
    sweep: bool = False,
    palace_executable: str | Path | None = None,
    num_processes: int = 8,
) -> BenchmarkRun:
    """Mesh, run Palace, and load the benchmark S-parameters."""
    from gsim.palace.results import load_sparams

    output = Path(output_dir or (RUN_ROOT / "l2n0_30ghz"))
    output.mkdir(parents=True, exist_ok=True)
    sim = make_simulation(output, sweep=sweep)
    sim.mesh(
        preset="coarse",
        refined_mesh_size=5.0,
        max_mesh_size=70.0,
        margin=150.0,
        z_above=50.0,
        z_below=10.0,
        planar_conductors=False,
        verbose=False,
    )
    sim.write_config()
    sim.run_local(
        palace_executable=palace_executable,
        num_processes=num_processes,
        verbose=False,
    )
    sparams = load_sparams(output / "output" / "palace")
    return BenchmarkRun(
        directory=output,
        sparams=sparams,
        frequencies_ghz=np.asarray(sparams.freq, dtype=float),
    )


def load_recorded_run(directory: str | Path) -> BenchmarkRun:
    """Load a previously recorded benchmark run."""
    from gsim.palace.results import load_sparams

    output = Path(directory)
    sparams = load_sparams(output / "output" / "palace")
    return BenchmarkRun(
        directory=output,
        sparams=sparams,
        frequencies_ghz=np.asarray(sparams.freq, dtype=float),
    )


def s_matrix(sparams: Any, index: int = 0) -> np.ndarray:
    """Return the 2x2 complex S matrix at one recorded frequency."""
    names = sparams.port_names
    if len(names) != 2:
        raise ValueError(f"expected 2 ports, got {names}")
    return np.array(
        [
            [sparams[names[i], names[j]].complex[index] for j in range(2)]
            for i in range(2)
        ]
    )


def lq_from_s_matrix(
    s: np.ndarray, f_hz: float, *, z0: float = Z0_OHM
) -> tuple[float, float, complex]:
    """Return ``(L_nH, Q, Z_series)`` from a 2x2 S matrix."""
    z = 2.0 * z0 * (1.0 - s[1, 0]) / s[1, 0]
    inductance_nh = z.imag / (2.0 * np.pi * f_hz) * 1e9
    q = z.imag / z.real if z.real != 0 else float("nan")
    return inductance_nh, q, z


def series_impedance(sparams: Any, index: int = 0, *, z0: float = Z0_OHM) -> complex:
    """Return the series impedance from S21 for a 2-port series element."""
    s = s_matrix(sparams, index)
    return 2.0 * z0 * (1.0 - s[1, 0]) / s[1, 0]


def lq_at(sparams: Any, index: int = 0) -> tuple[float, float, complex]:
    """Return ``(L_nH, Q, Z_series)`` at the selected recorded frequency."""
    return lq_from_s_matrix(s_matrix(sparams, index), float(sparams.freq[index]) * 1e9)


__all__ = [
    "LAYOUT",
    "MEASURED",
    "RUN_ROOT",
    "SOURCE",
    "STACK_SHIFT_UM",
    "SUBGND_ZMAX_UM",
    "SUBGND_ZMIN_UM",
    "TARGET_FREQ_HZ",
    "Z0_OHM",
    "BenchmarkRun",
    "build_benchmark_stack",
    "load_benchmark_component",
    "load_recorded_run",
    "lq_at",
    "make_simulation",
    "run_benchmark",
    "s_matrix",
    "series_impedance",
]
