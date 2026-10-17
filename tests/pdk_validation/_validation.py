"""Quantitative MEEP-vs-reference harness for the open-PDK validation cases.

The notebook cells in ``nbs/photonic_pdk_validation.ipynb`` show the geometry
and the reference data for six devices.  This module turns those cells into a
*quantitative* comparison by:

* building each device at a **matched geometry** to its reference (compact
  model at the same ``gap``/``length``/``radius``, or the measured device),
* running MEEP at a resolution and frequency grid fine enough to resolve the
  device (``resolution >= 20`` px/um, ``num_freqs = 101`` over the band) with
  **field-energy decay stopping** instead of a fixed short run,
* **de-embedding** the port monitors to the port reference planes using the
  mode effective index reported by MEEP,
* **normalising** fiber-referenced measured spectra before comparing, and
* reporting agreement metrics (RMS / max dB error, correlation) per port.

Nothing here runs MEEP at import time.  Building a
:class:`~gsim.meep.Simulation` (via :func:`build_validation_simulation`) is
pure configuration; :func:`run_case` is the only function that executes a
simulation.

Design notes / limitations
--------------------------
* The de-embedding phase convention is the MEEP ``e^{+ikx}`` propagation
  convention with ``sign=+1``.  If a run shows the opposite sense, flip
  ``sign``; the magnitude comparisons are unaffected either way.
* MEEP reports S-parameters at the monitor planes, which sit
  ``source_port_offset`` (and, for the source port, an extra
  ``distance_source_to_monitors``) inside the device from the port centres
  (see ``gsim.meep.script.build_monitors``).  :func:`monitor_distances`
  reproduces that rule so the harness can undo it.
* Single-digit S-parameter names (``S11``..``S44``) are assumed; the six
  validation devices have at most four ports.
* The large MZI (``~2475 lambda^2``) is configured for the 2-D effective-index
  plane because a 3-D run at ``resolution=20`` is ~1.7e8 cells.  The other
  devices run in 3-D.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

# ---------------------------------------------------------------------------
# Validation defaults
# ---------------------------------------------------------------------------

VALIDATION_RESOLUTION = 20
VALIDATION_NUM_FREQS = 101
VALIDATION_WAVELENGTH_UM = 1.55
VALIDATION_SPAN_UM = 0.10  # 1.50-1.60 um: covers measured spectra and LUTs
VALIDATION_DECAY_DT = 50.0
VALIDATION_DECAY_BY = 1e-6
VALIDATION_MAX_TIME = 20000.0
FALLBACK_N_EFF = 2.4
DB_FLOOR = 1e-9

Direction = Literal["2d", "3d"]


# ---------------------------------------------------------------------------
# Simulation construction
# ---------------------------------------------------------------------------


def monitor_distances(
    component: Any,
    *,
    source_port: str | None = None,
    source_port_offset: float = 0.1,
    distance_source_to_monitors: float = 0.2,
) -> dict[str, float]:
    """Return each port's monitor offset from its reference plane, in um.

    Mirrors ``gsim.meep.script.build_monitors``: every monitor is displaced
    ``source_port_offset`` into the device; the source port's monitor is
    displaced an additional ``distance_source_to_monitors``.
    """
    names = [port.name for port in component.ports]
    source_port = source_port or names[0]
    return {
        port.name: float(
            source_port_offset
            + (distance_source_to_monitors if port.name == source_port else 0.0)
        )
        for port in component.ports
    }


def build_validation_simulation(
    component: Any,
    *,
    source_port: str | None = None,
    resolution: int = VALIDATION_RESOLUTION,
    num_freqs: int = VALIDATION_NUM_FREQS,
    wavelength: float = VALIDATION_WAVELENGTH_UM,
    span: float = VALIDATION_SPAN_UM,
    dimension: Direction = "3d",
    materials: Mapping[str, Any] | None = None,
    stack: Any = None,
    decay_dt: float = VALIDATION_DECAY_DT,
    decay_by: float = VALIDATION_DECAY_BY,
    max_time: float = VALIDATION_MAX_TIME,
    wall_time: float | None = None,
    subpixel: bool = True,
) -> Any:
    """Build a :class:`gsim.meep.Simulation` configured for a real comparison.

    Args:
        component: gdsfactory component to simulate.
        source_port: Port to excite (defaults to the first port).
        resolution: Pixels per micrometre (>= 20 for quantitative work).
        num_freqs: Frequency samples over the band (101 by default).
        wavelength: Centre wavelength in um.
        span: Wavelength span in um.
        dimension: ``"3d"`` (default) or ``"2d"`` effective-index plane.
        materials: Optional material override; ``None`` uses the PDK stack.
        stack: Optional pre-built layer stack; ``None`` uses ``get_stack()``.
        decay_dt: Energy-decay measurement window (MEEP time units).
        decay_by: Energy-decay stop threshold (fraction).
        max_time: Hard cap on simulation time after sources.
        wall_time: Optional wall-clock cap in seconds.
        subpixel: Enable subpixel smoothing (recommended for accuracy).

    Returns:
        A configured :class:`gsim.meep.Simulation` (not yet run).
    """
    from gsim import meep
    from gsim.common.stack import get_stack

    if stack is None:
        stack = get_stack()

    sim = meep.Simulation()
    sim.geometry(component=component, stack=stack)
    if materials is not None:
        sim.materials = dict(materials)

    names = [port.name for port in component.ports]
    source = source_port or names[0]
    sim.source.port = source
    sim.monitors = list(names)
    sim.num_freqs = num_freqs
    sim.source.wavelength = wavelength
    sim.source.wavelength_span = span

    sim.solver.resolution = resolution
    sim.solver.subpixel = subpixel
    if dimension == "2d":
        # ``solver(...)`` validates mode/cut together; assigning fields
        # individually would trip the mode/cut invariant.
        sim.solver(mode="2d", z_cut="auto")

    sim.solver.stop_when_energy_decayed(dt=decay_dt, decay_by=decay_by)
    sim.solver.max_time = max_time
    if wall_time:
        sim.solver.stop_after_walltime(wall_time)
    return sim


# ---------------------------------------------------------------------------
# De-embedding
# ---------------------------------------------------------------------------


def s_param_indices(name: str) -> tuple[int, int]:
    """Return 0-based ``(receive, source)`` indices from an ``Sij`` name."""
    if len(name) < 3 or name[0] != "S":
        raise ValueError(f"not an S-parameter name: {name!r}")
    return int(name[1]) - 1, int(name[2]) - 1


def n_eff_by_port(result: Any) -> dict[str, np.ndarray]:
    """Extract per-port effective indices from a MEEP result's debug info."""
    info = (getattr(result, "debug_info", None) or {}).get("eigenmode_info", {})
    out: dict[str, np.ndarray] = {}
    for name, entry in info.items():
        values = entry.get("n_eff") if isinstance(entry, dict) else None
        if values:
            out[name] = np.asarray(values, dtype=float)
    return out


def deembed_s_params(
    result: Any,
    port_names: list[str],
    distances: Mapping[str, float],
    *,
    source_port: str | None = None,
    sign: float = 1.0,
    fallback_n_eff: float = FALLBACK_N_EFF,
) -> dict[str, np.ndarray]:
    """Shift S-parameters from the monitor planes to the port planes.

    Each monitor sits ``distances[port]`` inside the device from its port
    reference plane.  Applying ``exp(i * sign * k (n_i d_i + n_j d_j))`` moves
    the reference planes back to the port centres (``k = 2*pi/lambda``), using
    the per-port, per-frequency mode effective index that MEEP reports.

    Args:
        result: An :class:`~gsim.meep.models.results.SParameterResult`.
        port_names: Port names in the order used to build ``Sij`` indices.
        distances: ``{port_name: distance_um}`` from :func:`monitor_distances`.
        source_port: Excited port (defaults to ``port_names[0]``).
        sign: ``+1`` for the MEEP ``e^{+ikx}`` convention; flip if needed.
        fallback_n_eff: Used when MEEP reports no effective index for a port.

    Returns:
        ``{S_name: complex ndarray}`` with the reference planes de-embedded.
    """
    wavelengths = np.asarray(result.wavelengths, dtype=float)
    source = source_port or port_names[0]
    source_idx = port_names.index(source)
    n_eff = n_eff_by_port(result)
    k = 2.0 * math.pi / wavelengths

    def effective_index(port: str) -> np.ndarray:
        values = n_eff.get(port)
        if values is None or values.size == 0:
            return np.full_like(wavelengths, fallback_n_eff)
        if values.size != wavelengths.size:
            values = np.interp(
                wavelengths,
                np.linspace(wavelengths.min(), wavelengths.max(), values.size),
                values,
            )
        return values

    out: dict[str, np.ndarray] = {}
    for name, values in result.s_params.items():
        receive_idx, excitation_idx = s_param_indices(name)
        vals = np.asarray(values, dtype=complex)
        if excitation_idx != source_idx:
            out[name] = vals
            continue
        d_receive = float(distances.get(port_names[receive_idx], 0.0))
        d_source = float(distances.get(source, 0.0))
        phase = k * (
            effective_index(port_names[receive_idx]) * d_receive
            + effective_index(source) * d_source
        )
        out[name] = vals * np.exp(1j * sign * phase)
    return out


# ---------------------------------------------------------------------------
# Reference normalisation and agreement metrics
# ---------------------------------------------------------------------------


def to_db(values: Any) -> np.ndarray:
    """Return ``20*log10(|values|)`` clipped at a -180 dB floor."""
    magnitude = np.abs(np.asarray(values, dtype=complex))
    return 20.0 * np.log10(np.clip(magnitude, DB_FLOOR, None))


def normalize_measured_spectrum(
    power_db: Any,
    *,
    method: Literal["max", "percentile"] = "percentile",
    percentile: float = 99.0,
) -> np.ndarray:
    """Return a measured spectrum as insertion-loss-normalised transmission.

    Fiber-to-fiber measurements (dB/dBm) carry an unknown launch/collection
    loss, so only the spectral *shape* is meaningful.  Subtracting a robust
    on-chip reference (the 99th percentile by default) turns the spectrum into
    a relative transmission in dB that can be compared with ``|S21|``.
    """
    power = np.asarray(power_db, dtype=float)
    if method == "max":
        reference = np.nanmax(power)
    else:
        reference = np.nanpercentile(power, percentile)
    return power - reference


def agreement_db(
    sim_wavelengths: Any,
    sim_db: Any,
    ref_wavelengths: Any,
    ref_db: Any,
    *,
    normalize: bool = False,
) -> dict[str, Any] | None:
    """Compare two dB curves on their shared wavelength range.

    Args:
        sim_wavelengths: Simulation wavelengths in um (increasing).
        sim_db: Simulation dB values.
        ref_wavelengths: Reference wavelengths in um.
        ref_db: Reference dB values.
        normalize: Subtract each curve's maximum before comparing (for
            insertion-loss-normalised transmission).

    Returns:
        ``{n, rms_db, max_abs_db, bias_db, corr, wavelengths, sim_db, ref_db}``
        or ``None`` when fewer than three points overlap.
    """
    sim_wl = np.asarray(sim_wavelengths, dtype=float)
    sim = np.asarray(sim_db, dtype=float)
    ref_wl = np.asarray(ref_wavelengths, dtype=float)
    ref = np.asarray(ref_db, dtype=float)

    order = np.argsort(ref_wl)
    ref_wl, ref = ref_wl[order], ref[order]
    low = max(sim_wl.min(), ref_wl.min())
    high = min(sim_wl.max(), ref_wl.max())
    mask = (sim_wl >= low) & (sim_wl <= high)
    if int(mask.sum()) < 3:
        return None

    sim_overlap = sim_wl[mask]
    sim_curve = sim[mask]
    ref_curve = np.interp(sim_overlap, ref_wl, ref)
    if normalize:
        sim_curve = sim_curve - np.nanmax(sim_curve)
        ref_curve = ref_curve - np.nanmax(ref_curve)

    diff = sim_curve - ref_curve
    corr = None
    if sim_overlap.size > 1 and np.std(sim_curve) > 0 and np.std(ref_curve) > 0:
        corr = float(np.corrcoef(sim_curve, ref_curve)[0, 1])
    return {
        "n": int(mask.sum()),
        "rms_db": float(np.sqrt(np.nanmean(diff**2))),
        "max_abs_db": float(np.nanmax(np.abs(diff))),
        "bias_db": float(np.nanmean(diff)),
        "corr": corr,
        "wavelengths": sim_overlap,
        "sim_db": sim_curve,
        "ref_db": ref_curve,
    }


def power_conservation(s_params: Mapping[str, Any]) -> np.ndarray:
    """Return ``sum |Sij|^2`` per frequency for the outgoing ports."""
    total: np.ndarray | None = None
    for values in s_params.values():
        term = np.abs(np.asarray(values, dtype=complex)) ** 2
        total = term if total is None else total + term
    if total is None:
        return np.zeros(0)
    return total


# ---------------------------------------------------------------------------
# Matched-geometry devices
# ---------------------------------------------------------------------------


def dbr_two_port(
    w1: float = 0.5,
    w2: float = 0.6,
    n: int = 20,
    *,
    cross_section: str = "strip",
    straight_length: float = 10e-3,
    period: float = 318e-3,
) -> Any:
    """Build a two-port DBR.

    ``gdsfactory.components.dbr`` exposes only ``o1`` (it never adds the output
    port), so the Bragg validation case cannot measure transmission with it.
    This rebuilds the same grating but terminates both ends with a short
    straight and exposes ``o1`` and ``o2``.
    """
    import gdsfactory as gf
    from gdsfactory.component import Component
    from gdsfactory.components.filters.dbr import dbr_cell

    l1 = l2 = period / 2
    c = Component()
    xs = gf.get_cross_section(cross_section)
    s1 = c << gf.c.straight(cross_section=xs, length=straight_length)
    s2 = c << gf.c.straight(cross_section=xs, length=straight_length)
    cell = dbr_cell(w1=w1, w2=w2, l1=l1, l2=l2, cross_section=cross_section)
    ref = c.add_ref(cell, columns=n, rows=1, column_pitch=l1 + l2)
    s1.connect(port="o1", other=cell.ports["o1"], allow_width_mismatch=True)
    s2.connect(port="o1", other=cell.ports["o2"], allow_width_mismatch=True)
    s2.xmin = ref.xmax
    c.add_port("o1", port=s1.ports["o2"])
    c.add_port("o2", port=s2.ports["o2"])
    c.flatten()
    return c


# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------

STRAIGHT_LENGTH_UM = 20.0
COUPLER_GAP_UM = 0.2
COUPLER_LENGTH_UM = 10.0
RING_GAP_UM = 0.1
RING_RADIUS_UM = 10.0
RING_LENGTH_X_UM = 4.0
MZI_DELTA_LENGTH_UM = 10.0
DBR_W1 = 0.5
DBR_W2 = 0.6
DBR_PERIODS = 20


@dataclass
class Reference:
    """Reference target for one validation case.

    Exactly one of ``s_params`` or ``transmission_db`` is populated.

    Attributes:
        wavelengths: Reference wavelengths in um.
        s_params: ``{S_name: complex array}`` absolute model S-parameters.
        transmission_db: ``(wavelengths_um, power_db)`` measured spectrum, to be
            insertion-loss normalised before comparison.
        label: Human-readable source description.
    """

    wavelengths: np.ndarray
    s_params: dict[str, np.ndarray] = field(default_factory=dict)
    transmission_db: tuple[np.ndarray, np.ndarray] | None = None
    label: str = ""


@dataclass
class ValidationCase:
    """One device validation case."""

    name: str
    make_component: Callable[[], Any]
    reference: Callable[[np.ndarray], Reference]
    source_port: str | None = None
    dimension: Direction = "3d"
    resolution: int = VALIDATION_RESOLUTION
    num_freqs: int = VALIDATION_NUM_FREQS
    span: float = VALIDATION_SPAN_UM
    notes: str = ""


def _ref_straight(wavelengths: np.ndarray) -> Reference:
    from ubcpdk.models import waveguides as wg

    s21 = np.asarray(
        wg.straight(wl=wavelengths, length=STRAIGHT_LENGTH_UM)[("o1", "o2")]
    )
    return Reference(wavelengths, {"S21": s21}, label="SAX straight")


def _ref_coupler(wavelengths: np.ndarray) -> Reference:
    from ubcpdk.models import couplers as dc

    model = dc.directional_coupler(
        wl=wavelengths, length=COUPLER_LENGTH_UM, gap=COUPLER_GAP_UM
    )
    return Reference(
        wavelengths,
        {
            "S41": np.asarray(model[("o1", "o4")]),  # bar
            "S31": np.asarray(model[("o1", "o3")]),  # cross
        },
        label="SAX directional-coupler LUT (matched gap/length)",
    )


def _ref_ring_coupling(wavelengths: np.ndarray) -> Reference:
    from ubcpdk.models import couplers as dc

    model = dc.coupler_ring_coupling_area(
        wl=wavelengths,
        gap=RING_GAP_UM,
        radius=RING_RADIUS_UM,
        length_x=RING_LENGTH_X_UM,
    )
    return Reference(
        wavelengths,
        {
            "S41": np.asarray(model[("o1", "o4")]),  # bar
            "S31": np.asarray(model[("o1", "o3")]),  # cross
        },
        label="SAX racetrack coupling LUT (matched gap/radius/length_x)",
    )


def _measured(path: str, label: str) -> Callable[[np.ndarray], Reference]:
    from tests.pdk_validation import _fetch

    def _reference(wavelengths: np.ndarray) -> Reference:
        local = _fetch.fetch_ubc(path)
        wavelength_nm, power_db = _fetch.load_ubc_spectrum(local)
        return Reference(
            wavelengths,
            transmission_db=(wavelength_nm / 1000.0, power_db),
            label=label,
        )

    return _reference


def build_cases() -> list[ValidationCase]:
    """Return the six open-PDK validation cases."""
    import ubcpdk

    from tests.pdk_validation import _fetch

    return [
        ValidationCase(
            name="straight",
            make_component=lambda: ubcpdk.cells.straight(length=STRAIGHT_LENGTH_UM),
            reference=_ref_straight,
            notes="Absolute |S21| vs SAX (loss included); group index from phase.",
        ),
        ValidationCase(
            name="coupler",
            make_component=lambda: ubcpdk.cells.coupler(
                gap=COUPLER_GAP_UM, length=COUPLER_LENGTH_UM
            ),
            reference=_ref_coupler,
            notes="Bar/cross vs the SAX LUT at the same gap and length.",
        ),
        ValidationCase(
            name="ring_coupling",
            make_component=lambda: ubcpdk.cells.coupler_ring(
                length_x=RING_LENGTH_X_UM, gap=RING_GAP_UM, radius=RING_RADIUS_UM
            ),
            reference=_ref_ring_coupling,
            notes="Racetrack coupling region vs the LUT at the same geometry.",
        ),
        ValidationCase(
            name="mzi",
            make_component=lambda: ubcpdk.cells.mzi(delta_length=MZI_DELTA_LENGTH_UM),
            reference=_measured(_fetch.UBC_MZI_MAT, "measured UBC MZI spectrum"),
            dimension="2d",
            notes=(
                "~2475 lambda^2 device; 2-D effective-index plane (3-D res=20 "
                "is ~1.7e8 cells). Compare normalised transmission."
            ),
        ),
        ValidationCase(
            name="ring",
            make_component=lambda: ubcpdk.cells.ring_single(
                gap=RING_GAP_UM, radius=RING_RADIUS_UM
            ),
            reference=_measured(_fetch.UBC_RING_MAT, "measured UBC ring spectrum"),
            notes="Measured TE r10 g100 spectrum; compare notches (normalised).",
        ),
        ValidationCase(
            name="bragg",
            make_component=lambda: dbr_two_port(w1=DBR_W1, w2=DBR_W2, n=DBR_PERIODS),
            reference=_measured(_fetch.UBC_BRAGG_MAT, "measured UBC Bragg spectrum"),
            notes=(
                "Two-port DBR (gdsfactory dbr exposes only o1); compare stopband "
                "(normalised)."
            ),
        ),
    ]


def get_case(name: str) -> ValidationCase:
    """Return a case by name, raising ``KeyError`` if unknown."""
    for case in build_cases():
        if case.name == name:
            return case
    raise KeyError(f"unknown validation case: {name!r}")


# ---------------------------------------------------------------------------
# Running and comparison
# ---------------------------------------------------------------------------


@dataclass
class ValidationOutcome:
    """Result bundle for one validation case."""

    case: str
    wavelengths: np.ndarray
    s_params: dict[str, np.ndarray]  # de-embedded
    reference: Reference
    metrics: dict[str, Any]
    best_port: str | None = None


def compare(
    wavelengths: np.ndarray,
    s_params: Mapping[str, Any],
    reference: Reference,
) -> tuple[dict[str, Any], str | None]:
    """Compare de-embedded S-parameters against a reference.

    Returns ``(metrics, best_port)`` where ``metrics`` maps each compared
    port to :func:`agreement_db` output.
    """
    metrics: dict[str, Any] = {}
    best_port: str | None = None

    if reference.s_params:
        for name, ref_values in reference.s_params.items():
            if name not in s_params:
                continue
            metrics[name] = agreement_db(
                wavelengths,
                to_db(s_params[name]),
                reference.wavelengths,
                to_db(ref_values),
            )
    elif reference.transmission_db is not None:
        ref_wl, ref_power_db = reference.transmission_db
        ref_db = normalize_measured_spectrum(ref_power_db)
        # Pick the port with the strongest average transmission as the through
        # port (large devices do not always label it S21).
        candidates = {
            name: float(np.nanmean(np.abs(np.asarray(values, dtype=complex))))
            for name, values in s_params.items()
        }
        if candidates:
            best_port = max(candidates, key=lambda name: candidates[name])
            metrics[best_port] = agreement_db(
                wavelengths,
                to_db(s_params[best_port]),
                ref_wl,
                ref_db,
                normalize=True,
            )
        for name, values in s_params.items():
            if name == best_port:
                continue
            metrics[name] = agreement_db(
                wavelengths, to_db(values), ref_wl, ref_db, normalize=True
            )
    return metrics, best_port


def run_case(
    case: ValidationCase | str,
    *,
    output_dir: str | None = None,
    python_executable: str | None = None,
    num_processes: int | None = None,
    verbose: bool = False,
    stack: Any = None,
) -> ValidationOutcome:
    """Build, run, de-embed and compare one validation case.

    This is the only function here that executes MEEP.
    """
    if isinstance(case, str):
        case = get_case(case)

    component = case.make_component()
    sim = build_validation_simulation(
        component,
        source_port=case.source_port,
        resolution=case.resolution,
        num_freqs=case.num_freqs,
        span=case.span,
        dimension=case.dimension,
        stack=stack,
    )
    distances = monitor_distances(
        component,
        source_port=case.source_port,
        source_port_offset=sim.domain.source_port_offset,
        distance_source_to_monitors=sim.domain.distance_source_to_monitors,
    )

    result = sim.run_local(
        output_dir,
        python_executable=python_executable,
        num_processes=num_processes,
        verbose=verbose,
    )
    port_names = [port.name for port in component.ports]
    deembedded = deembed_s_params(
        result, port_names, distances, source_port=case.source_port
    )
    wavelengths = np.asarray(result.wavelengths, dtype=float)
    reference = case.reference(wavelengths)
    metrics, best_port = compare(wavelengths, deembedded, reference)
    return ValidationOutcome(
        case=case.name,
        wavelengths=np.asarray(result.wavelengths),
        s_params=deembedded,
        reference=reference,
        metrics=metrics,
        best_port=best_port,
    )


def run_all(
    names: list[str] | None = None,
    *,
    output_root: str | None = None,
    python_executable: str | None = None,
    verbose: bool = False,
) -> dict[str, ValidationOutcome | Exception]:
    """Run several cases, returning outcomes (or the exception) per case."""
    selected = [get_case(name) for name in names] if names else build_cases()
    results: dict[str, ValidationOutcome | Exception] = {}
    for case in selected:
        output_dir = None
        if output_root:
            output_dir = str(Path(output_root) / case.name)
        try:
            results[case.name] = run_case(
                case,
                output_dir=output_dir,
                python_executable=python_executable,
                verbose=verbose,
            )
        except Exception as error:
            results[case.name] = error
    return results


__all__ = [
    "COUPLER_GAP_UM",
    "COUPLER_LENGTH_UM",
    "DBR_PERIODS",
    "DBR_W1",
    "DBR_W2",
    "FALLBACK_N_EFF",
    "MZI_DELTA_LENGTH_UM",
    "RING_GAP_UM",
    "RING_LENGTH_X_UM",
    "RING_RADIUS_UM",
    "STRAIGHT_LENGTH_UM",
    "VALIDATION_DECAY_BY",
    "VALIDATION_DECAY_DT",
    "VALIDATION_MAX_TIME",
    "VALIDATION_NUM_FREQS",
    "VALIDATION_RESOLUTION",
    "VALIDATION_SPAN_UM",
    "VALIDATION_WAVELENGTH_UM",
    "Reference",
    "ValidationCase",
    "ValidationOutcome",
    "agreement_db",
    "build_cases",
    "build_validation_simulation",
    "compare",
    "dbr_two_port",
    "deembed_s_params",
    "get_case",
    "monitor_distances",
    "n_eff_by_port",
    "normalize_measured_spectrum",
    "power_conservation",
    "run_all",
    "run_case",
    "s_param_indices",
    "to_db",
]
