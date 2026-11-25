"""P2.4 analysis: blank subtraction, per-surface tables, the f = v / lambda check.

Everything here is computed from a validated :class:`~.ingest.Session`. It
reports numbers and the comparisons the P2.4 plan asks for; whether the plan's
four acceptance criteria hold is for the user to read from the output.

What is computed, per the plan's acceptance list:

1. **Device peaks.** The strongest local maxima of the lifted-slide blank are
   listed, and every surface peak within two bins of one is flagged
   ``coincides_with_blank_peak``.
2. **Grit order.** At each speed, peak and centroid frequency are listed in
   grit order, with a plain boolean for "strictly increasing". No threshold.
3. **Speed doubling.** For every surface recorded at two speeds, the ratio of
   mean frequencies is compared with the ratio of speeds. The comparison needs
   a tolerance; ``SPEED_SCALING_TOLERANCE_REL = 0.15`` is akio's proposal,
   marked ``proposed_unconfirmed`` wherever it is used.
4. **Wet surface.** Reported like any other surface. If its ratio does *not*
   follow the speed ratio, that is the plan's expected stick-slip signature;
   the script does not decide which surfaces "should" scale.

Blank subtraction is a difference of band powers, **never clipped**. A band
where the surface is not above the blank has a non-positive difference and is
flagged ``above_blank = false``; writing zero there would claim a measurement
of "no texture energy" that the data does not make.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import bands as band_module
from .bands import BANDS
from .ingest import Session
from .spectrum import (
    DEFAULT_SEGMENT_S,
    Psd,
    analysis_range,
    axis_psds,
    band_powers,
    peak_frequency,
    spectral_centroid,
)

REPORT_SCHEMA = "p2.4-imu-spectrum-report/v1"
SPEED_SCALING_TOLERANCE_REL = 0.15
BLANK_PEAKS_LISTED = 5
PEAK_COINCIDENCE_BINS = 2

#: Nominal FEPA particle sizes as written in the P2.4 plan. Literature values,
#: used only for the predicted-frequency column. Never treated as measured.
FEPA_NOMINAL_PARTICLE_M = {120: 125e-6, 600: 26e-6, 1200: 15e-6}


def _mean_density(psds_list, key):
    return np.mean([item[key].density for item in psds_list], axis=0)


def _local_maxima(frequency, density, low, high, count):
    mask = (frequency >= low) & (frequency < high)
    idx = [i for i in range(1, len(density) - 1)
           if mask[i] and density[i] > density[i - 1] and density[i] >= density[i + 1]]
    idx.sort(key=lambda i: density[i], reverse=True)
    return [{"frequency_hz": float(frequency[i]), "density": float(density[i])} for i in idx[:count]]


def _surface_key(segment) -> str:
    surface = segment.surface
    grit = surface.get("grit")
    if grit is None:
        return surface["name"]
    return f"{surface['name']}|{surface.get('grit_scale')}{grit}"


def analyse(session: Session, *, segment_s: float = DEFAULT_SEGMENT_S) -> dict:
    """The full report dictionary for one validated session."""
    rate = session.segments[0].sample_rate_hz
    low, high = analysis_range(rate)
    spectra = {segment.segment_id: axis_psds(segment.accel_mps2, rate, segment_s=segment_s)
               for segment in session.segments}
    frequency = next(iter(spectra.values()))["total"].frequency_hz
    resolution = next(iter(spectra.values()))["total"].resolution_hz

    lifted = [spectra[s.segment_id] for s in session.by_kind("blank_lifted_slide")]
    static = [spectra[s.segment_id] for s in session.by_kind("blank_static_contact")]
    blank_total = _mean_density(lifted, "total")
    blank_psd = Psd(frequency, blank_total, rate, lifted[0]["total"].segment_samples, len(lifted))
    blank_bands = band_powers(blank_psd)
    blank_peaks = _local_maxima(frequency, blank_total, low, high, BLANK_PEAKS_LISTED)

    def coincides(value):
        return value is not None and any(
            abs(value - peak["frequency_hz"]) <= PEAK_COINCIDENCE_BINS * resolution for peak in blank_peaks
        )

    rows = []
    for segment in session.segments:
        psds = spectra[segment.segment_id]
        total = band_powers(psds["total"])
        row = {
            "segment": segment.to_dict(),
            "surface_key": _surface_key(segment) if segment.kind == "surface_slide" else None,
            "welch_averages": psds["total"].averages,
            "band_power_total": total,
            "band_power_axes": {axis: band_powers(psds[axis]) for axis in ("x", "y", "z")},
        }
        if segment.kind == "surface_slide":
            difference = psds["total"].density - blank_total
            row["band_power_minus_blank"] = [
                None if (a is None or b is None) else a - b for a, b in zip(total, blank_bands)
            ]
            row["above_blank"] = [
                None if value is None else value > 0 for value in row["band_power_minus_blank"]
            ]
            peak = peak_frequency(frequency, difference, low, high)
            row["peak_hz_minus_blank"] = peak
            row["peak_coincides_with_blank_peak"] = coincides(peak)
            row["centroid_hz_minus_blank"] = spectral_centroid(frequency, difference, low, high)
            grit, scale = segment.surface.get("grit"), segment.surface.get("grit_scale")
            size = FEPA_NOMINAL_PARTICLE_M.get(grit) if scale == "FEPA" else None
            row["predicted_hz_v_over_lambda"] = (
                {"value": segment.speed_mps / size, "source": "literature_nominal_FEPA_particle_size"}
                if size else None
            )
        rows.append(row)

    surfaces = _surface_summary(rows)
    return {
        "schema": REPORT_SCHEMA,
        "session_id": session.session_id,
        "data_origin": session.data_origin,
        "device": session.device,
        "interface": session.interface,
        "throughput": session.throughput,
        "ingest_policy": session.policy.to_dict(),
        "bands": band_module.definition(),
        "spectrum": {
            "method": "welch_hann_one_sided_density",
            "segment_s": segment_s,
            "resolution_hz": resolution,
            "units": "(m/s^2)^2/Hz density; band power in (m/s^2)^2",
            "analysis_range_hz": [low, high],
        },
        "blank": {
            "lifted_slide_segments": len(lifted),
            "static_contact_segments": len(static),
            "band_power_lifted_total": blank_bands,
            "band_power_static_total": band_powers(Psd(
                frequency, _mean_density(static, "total"), rate, blank_psd.segment_samples, len(static)
            )) if static else None,
            "lifted_peaks": blank_peaks,
        },
        "segments": rows,
        "surfaces": surfaces,
        "grit_order": _grit_order(surfaces),
        "speed_scaling_tolerance_rel": {
            "value": SPEED_SCALING_TOLERANCE_REL, "status": "proposed_unconfirmed",
        },
        "scope": (
            "computed comparisons only; the P2.4 acceptance criteria are judged by "
            "the user from these tables"
        ),
    }


def _mean_or_none(values):
    values = [value for value in values if value is not None]
    return (float(np.mean(values)), float(np.std(values)), len(values)) if values else (None, None, 0)


def _surface_summary(rows) -> dict:
    grouped = defaultdict(lambda: defaultdict(list))
    meta = {}
    for row in rows:
        if row["surface_key"] is None:
            continue
        speed = row["segment"]["speed"]["value_mps"]
        grouped[row["surface_key"]][speed].append(row)
        meta[row["surface_key"]] = row["segment"]["surface"]
    summary = {}
    for key, by_speed in grouped.items():
        speeds = {}
        for speed, items in sorted(by_speed.items()):
            peak = _mean_or_none([item["peak_hz_minus_blank"] for item in items])
            centroid = _mean_or_none([item["centroid_hz_minus_blank"] for item in items])
            speeds[str(speed)] = {
                "speed_mps": speed,
                "trials": len(items),
                "speed_methods": sorted({item["segment"]["speed"]["method"] for item in items}),
                "peak_hz_mean": peak[0], "peak_hz_std": peak[1],
                "centroid_hz_mean": centroid[0], "centroid_hz_std": centroid[1],
                "peaks_coinciding_with_blank": sum(bool(item["peak_coincides_with_blank_peak"]) for item in items),
            }
        summary[key] = {"surface": meta[key], "speeds": speeds, "speed_scaling": _speed_scaling(speeds)}
    return summary


def _speed_scaling(speeds: dict) -> dict | None:
    if len(speeds) != 2:
        return None
    slow, fast = sorted(speeds.values(), key=lambda item: item["speed_mps"])
    if slow["speed_mps"] <= 0:
        return None
    expected = fast["speed_mps"] / slow["speed_mps"]
    result = {"speed_ratio": expected, "tolerance_rel": SPEED_SCALING_TOLERANCE_REL,
              "tolerance_status": "proposed_unconfirmed"}
    for name in ("peak", "centroid"):
        a, b = slow[f"{name}_hz_mean"], fast[f"{name}_hz_mean"]
        if a is None or b is None or a <= 0:
            result[name] = None
            continue
        ratio = b / a
        deviation = abs(ratio - expected) / expected
        result[name] = {
            "frequency_ratio": ratio,
            "relative_deviation": deviation,
            "consistent_with_v_over_lambda": deviation <= SPEED_SCALING_TOLERANCE_REL,
        }
    return result


def _grit_order(surfaces: dict) -> dict:
    """Per speed and scale: frequencies in increasing grit order, and whether they increase."""
    table = defaultdict(list)
    for key, entry in surfaces.items():
        grit, scale = entry["surface"].get("grit"), entry["surface"].get("grit_scale")
        if grit is None:
            continue
        for speed, stats in entry["speeds"].items():
            table[(scale, speed)].append((grit, key, stats["peak_hz_mean"], stats["centroid_hz_mean"]))
    result = {}
    for (scale, speed), items in sorted(table.items()):
        items.sort()
        order = {}
        for name, position in (("peak", 2), ("centroid", 3)):
            values = [item[position] for item in items]
            order[f"{name}_strictly_increasing"] = (
                None if any(value is None for value in values) or len(values) < 2
                else all(a < b for a, b in zip(values, values[1:]))
            )
        result[f"{scale}@{speed}"] = {
            "grits": [item[0] for item in items],
            "surfaces": [item[1] for item in items],
            "peak_hz": [item[2] for item in items],
            "centroid_hz": [item[3] for item in items],
            **order,
        }
    return result


def write_report(report: dict, out_dir: str | Path) -> dict[str, str]:
    """``report.json`` plus a flat ``band_table.csv`` (one row per segment)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8", newline="\n")
    labels = [band.label for band in BANDS]
    with (out / "band_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["data_origin", "segment_id", "kind", "surface_key", "speed_mps", "speed_method",
                         "quantity", *labels])
        for row in report["segments"]:
            segment = row["segment"]
            base = [report["data_origin"], segment["segment_id"], segment["kind"], row["surface_key"] or "",
                    segment["speed"]["value_mps"], segment["speed"]["method"]]
            writer.writerow([*base, "band_power_total", *["" if v is None else v for v in row["band_power_total"]]])
            if "band_power_minus_blank" in row:
                writer.writerow([*base, "band_power_minus_blank",
                                 *["" if v is None else v for v in row["band_power_minus_blank"]]])
    return {"report": str(out / "report.json"), "band_table": str(out / "band_table.csv")}
