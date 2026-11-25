"""P2.4 ingestion: session manifest plus one CSV per segment, rejected at load.

A recording that cannot support a spectrum is refused here, not annotated in
the report: a remark in a table does not stop a bad segment from being
averaged into a result. The checks follow the P2.4 plan
(``step_record/20-后续待办清单.md`` §四):

* every field the plan says "must be recorded" is present and non-null;
* ``data_origin`` is declared. The command-line entry point only accepts
  ``measured``; synthetic signals exist for unit tests and cannot reach a
  report through it;
* per-sample timestamps are present, and no inter-sample interval exceeds
  ``max_interval_factor`` nominal periods. Uneven sample loss makes false
  spectral peaks that cannot be told apart from texture, so a single detected
  gap rejects the segment;
* the timestamp-derived rate agrees with the declared ODR within
  ``odr_tolerance_rel``;
* sample count agrees with ODR x duration (the plan's "样本数 ≈ ODR × 时长"),
  which is the same check stated over the whole segment;
* the declared operating mode supports the declared ODR.

The interface (SPI or I2C, and its clock) is a declared field, not an
assumption: either bus is accepted. Its estimated throughput is reported next
to the stream's required throughput, but the decision is made by the gap
check on the data, because a bus estimate is not evidence of what arrived.

Two numbers here are akio's proposals, not user-confirmed thresholds:
``max_interval_factor = 1.5`` and ``odr_tolerance_rel = 0.01``. They are
parameters of :class:`IngestPolicy`, reported in every output, and listed as
"待用户确认" in the P2.4 acceptance document.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SESSION_SCHEMA = "p2.4-imu-session/v1"

DATA_ORIGINS = ("measured", "synthetic_test")
SEGMENT_KINDS = ("blank_lifted_slide", "blank_static_contact", "surface_slide")
BUSES = ("spi", "i2c")
STROKES = ("unidirectional", "reciprocating")
SPEED_METHODS = ("measured", "estimated", "commanded")
GRIT_SCALES = ("FEPA", "CAMI")
OPERATING_MODES = ("high_performance", "normal", "low_power")
UNITS_TO_MPS2 = {"m/s^2": 1.0, "g": 9.80665, "mg": 9.80665e-3}

#: Highest ODR each LSM6DSV16X accelerometer mode allows, as stated in the
#: P2.4 plan (high-performance 7.68 kHz, normal capped at 1.92 kHz). Taken
#: from the plan, not re-read from the datasheet by akio; a mode missing here
#: is rejected rather than assumed unlimited.
#: The nominal value is compared with ``odr_tolerance_rel`` of slack, because
#: the plan records the device running at 7690 Hz against a 7680 Hz setting.
MODE_MAX_ODR_HZ = {"high_performance": 7680.0, "normal": 1920.0}

#: Rough sustained-throughput estimates (bytes/s) the plan tabulates: I2C
#: about clock/10 including protocol overhead, SPI about clock/8. Reported
#: only; never used to accept or reject a segment.
BUS_BYTES_PER_S = {"i2c": 1.0 / 10.0, "spi": 1.0 / 8.0}


class RecordingRejected(ValueError):
    """A segment or session that cannot support a spectrum."""


@dataclass(frozen=True)
class IngestPolicy:
    """Load-time tolerances. Both defaults are proposals awaiting user confirmation."""

    max_interval_factor: float = 1.5
    odr_tolerance_rel: float = 0.01
    status: str = "proposed_unconfirmed"

    def to_dict(self) -> dict:
        return {
            "max_interval_factor": self.max_interval_factor,
            "odr_tolerance_rel": self.odr_tolerance_rel,
            "status": self.status,
        }


@dataclass(frozen=True)
class Segment:
    segment_id: str
    kind: str
    surface: dict
    trial: int
    speed: dict
    preload: dict
    stroke: str
    axes: dict
    sample_rate_hz: float
    time_s: np.ndarray
    accel_mps2: dict[str, np.ndarray]
    integrity: dict = field(default_factory=dict)

    @property
    def speed_mps(self) -> float:
        return float(self.speed["value_mps"])

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "kind": self.kind,
            "surface": self.surface,
            "trial": self.trial,
            "speed": self.speed,
            "preload": self.preload,
            "stroke": self.stroke,
            "axes": self.axes,
            "sample_rate_hz": self.sample_rate_hz,
            "integrity": self.integrity,
        }


@dataclass(frozen=True)
class Session:
    session_id: str
    data_origin: str
    device: dict
    interface: dict
    segments: tuple[Segment, ...]
    policy: IngestPolicy
    throughput: dict

    def by_kind(self, kind: str) -> list[Segment]:
        return [segment for segment in self.segments if segment.kind == kind]


def _require(mapping: dict, keys, where: str) -> None:
    missing = [key for key in keys if mapping.get(key) is None]
    if missing:
        raise RecordingRejected(f"{where}: required fields missing or null: {missing}")


def _choice(value, allowed, where: str) -> None:
    if value not in allowed:
        raise RecordingRejected(f"{where}: {value!r} is not one of {list(allowed)}")


def check_device(device: dict, interface: dict, policy: IngestPolicy | None = None) -> dict:
    """Validate the declared device, mode and bus. Returns the throughput record."""
    policy = policy or IngestPolicy()
    _require(device, ("model", "odr_hz", "operating_mode", "units", "bytes_per_sample"), "device")
    _require(interface, ("bus", "clock_hz"), "interface")
    _choice(device["operating_mode"], OPERATING_MODES, "device.operating_mode")
    _choice(device["units"], UNITS_TO_MPS2, "device.units")
    _choice(interface["bus"], BUSES, "interface.bus")
    odr = float(device["odr_hz"])
    if odr <= 0:
        raise RecordingRejected("device.odr_hz must be positive")
    ceiling = MODE_MAX_ODR_HZ.get(device["operating_mode"])
    if ceiling is None or odr > ceiling * (1.0 + policy.odr_tolerance_rel):
        raise RecordingRejected(
            f"device: ODR {odr} Hz is not available in {device['operating_mode']!r} mode "
            f"(plan: high_performance up to 7.68 kHz, normal up to 1.92 kHz)"
        )
    required = odr * float(device["bytes_per_sample"])
    estimate = float(interface["clock_hz"]) * BUS_BYTES_PER_S[interface["bus"]]
    return {
        "required_bytes_per_s": required,
        "bus_estimate_bytes_per_s": estimate,
        "estimate_margin": estimate / required,
        "status": "estimate_only",
        "note": "the gap check on the timestamps decides; this estimate does not",
    }


def check_record(raw: dict, where: str) -> None:
    """The fields the P2.4 plan says every segment must carry."""
    _require(raw, ("segment_id", "kind", "file", "trial", "speed", "preload", "stroke", "axes"), where)
    _choice(raw["kind"], SEGMENT_KINDS, f"{where}.kind")
    _choice(raw["stroke"], STROKES, f"{where}.stroke")
    _require(raw["speed"], ("value_mps", "method"), f"{where}.speed")
    _choice(raw["speed"]["method"], SPEED_METHODS, f"{where}.speed.method")
    _require(raw["preload"], ("description",), f"{where}.preload")
    _require(raw["axes"], ("slide", "normal"), f"{where}.axes")
    if raw["kind"] == "surface_slide":
        surface = raw.get("surface") or {}
        _require(surface, ("name",), f"{where}.surface")
        if surface.get("grit") is not None:
            # FEPA P600 and CAMI 600 differ by a factor of two to five in
            # particle size; a grit number without its scale is not comparable.
            _require(surface, ("grit_scale",), f"{where}.surface")
            _choice(surface["grit_scale"], GRIT_SCALES, f"{where}.surface.grit_scale")
    if raw["kind"] == "blank_static_contact" and float(raw["speed"]["value_mps"]) != 0.0:
        raise RecordingRejected(f"{where}: a static-contact blank must declare speed 0")


def read_columns(path: Path, columns: dict) -> dict[str, np.ndarray]:
    """Read the mapped CSV columns as float arrays."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        wanted = {name: columns[name] for name in ("time", "x", "y", "z")}
        missing = [column for column in wanted.values() if column not in header]
        if missing:
            raise RecordingRejected(f"{path.name}: columns {missing} not in header {header}")
        rows = {name: [] for name in wanted}
        for line, row in enumerate(reader, start=2):
            for name, column in wanted.items():
                text = row[column]
                try:
                    value = float(text)
                except (TypeError, ValueError):
                    raise RecordingRejected(f"{path.name}:{line}: {column}={text!r} is not a number") from None
                if not math.isfinite(value):
                    raise RecordingRejected(f"{path.name}:{line}: {column} is not finite")
                rows[name].append(value)
    return {name: np.asarray(values, dtype=float) for name, values in rows.items()}


def check_sampling(time_s: np.ndarray, odr_hz: float, policy: IngestPolicy, where: str) -> dict:
    """Reject dropped samples and a rate that disagrees with the declared ODR."""
    if time_s.size < 2:
        raise RecordingRejected(f"{where}: fewer than two samples")
    intervals = np.diff(time_s)
    if np.any(intervals <= 0):
        raise RecordingRejected(f"{where}: timestamps are not strictly increasing")
    period = 1.0 / odr_hz
    limit = policy.max_interval_factor * period
    gaps = np.flatnonzero(intervals > limit)
    duration = float(time_s[-1] - time_s[0])
    measured_rate = (time_s.size - 1) / duration
    expected_samples = duration * odr_hz + 1
    integrity = {
        "samples": int(time_s.size),
        "duration_s": duration,
        "declared_odr_hz": odr_hz,
        "timestamp_rate_hz": measured_rate,
        "rate_relative_error": abs(measured_rate - odr_hz) / odr_hz,
        "expected_samples_from_odr_and_duration": expected_samples,
        "max_interval_s": float(intervals.max()),
        "intervals_over_limit": int(gaps.size),
        "interval_limit_s": limit,
        "policy": policy.to_dict(),
    }
    if gaps.size:
        first = int(gaps[0])
        raise RecordingRejected(
            f"{where}: {gaps.size} inter-sample interval(s) exceed "
            f"{policy.max_interval_factor} x 1/ODR ({limit:.3e} s); first at sample "
            f"{first + 1}, interval {intervals[first]:.3e} s. Uneven sample loss creates "
            f"false spectral peaks, so the segment is refused"
        )
    if integrity["rate_relative_error"] > policy.odr_tolerance_rel:
        raise RecordingRejected(
            f"{where}: timestamp rate {measured_rate:.2f} Hz differs from declared ODR "
            f"{odr_hz} Hz by {100 * integrity['rate_relative_error']:.2f}% "
            f"(> {100 * policy.odr_tolerance_rel:.2f}%). Either the ODR is mis-declared "
            f"or samples are missing evenly; neither supports a frequency axis"
        )
    return integrity


def build_segment(raw: dict, data: dict[str, np.ndarray], device: dict, policy: IngestPolicy,
                  where: str) -> Segment:
    """Validate one segment's metadata and samples and return it in SI units."""
    check_record(raw, where)
    odr = float(device["odr_hz"])
    integrity = check_sampling(data["time"], odr, policy, where)
    scale = UNITS_TO_MPS2[device["units"]]
    return Segment(
        segment_id=raw["segment_id"],
        kind=raw["kind"],
        surface=raw.get("surface") or {},
        trial=int(raw["trial"]),
        speed=dict(raw["speed"]),
        preload=dict(raw["preload"]),
        stroke=raw["stroke"],
        axes=dict(raw["axes"]),
        sample_rate_hz=odr,
        time_s=data["time"],
        accel_mps2={axis: data[axis] * scale for axis in ("x", "y", "z")},
        integrity=integrity,
    )


def load_session(path: str | Path, *, policy: IngestPolicy | None = None,
                 require_measured: bool = True) -> Session:
    """Load and validate a whole session. Any rejected segment rejects the session."""
    path = Path(path)
    policy = policy or IngestPolicy()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SESSION_SCHEMA:
        raise RecordingRejected(f"session schema must be {SESSION_SCHEMA!r}")
    _require(manifest, ("session_id", "data_origin", "device", "interface", "columns", "segments"),
             "session")
    _choice(manifest["data_origin"], DATA_ORIGINS, "session.data_origin")
    if require_measured and manifest["data_origin"] != "measured":
        raise RecordingRejected(
            f"session.data_origin is {manifest['data_origin']!r}; only 'measured' "
            f"sessions produce an analysis report"
        )
    throughput = check_device(manifest["device"], manifest["interface"], policy)
    _require(manifest["columns"], ("time", "x", "y", "z"), "session.columns")
    segments = []
    seen = set()
    for index, raw in enumerate(manifest["segments"]):
        where = f"segments[{index}]"
        check_record(raw, where)
        if raw["segment_id"] in seen:
            raise RecordingRejected(f"{where}: duplicate segment_id {raw['segment_id']!r}")
        seen.add(raw["segment_id"])
        data = read_columns(path.parent / raw["file"], manifest["columns"])
        segments.append(build_segment(raw, data, manifest["device"], policy, f"{where} ({raw['file']})"))
    if not any(segment.kind == "blank_lifted_slide" for segment in segments):
        raise RecordingRejected(
            "session has no blank_lifted_slide segment; without it device resonance "
            "cannot be separated from surface texture"
        )
    return Session(
        session_id=manifest["session_id"],
        data_origin=manifest["data_origin"],
        device=dict(manifest["device"]),
        interface=dict(manifest["interface"]),
        segments=tuple(segments),
        policy=policy,
        throughput=throughput,
    )
