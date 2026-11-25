"""P2.4 analysis-side tests. Every signal here is synthetic and exists only to
test the script; nothing below is, or may be used as, a measurement.

Sessions built here declare ``data_origin = "synthetic_test"``, which the
command-line entry point refuses, so a synthetic table cannot become a report.
"""

from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from validation.imu_spectrum import BANDS, IngestPolicy, RecordingRejected, analyse, load_session
from validation.imu_spectrum.__main__ import main as cli_main
from validation.imu_spectrum.bands import available, definition
from validation.imu_spectrum.ingest import check_device, check_sampling
from validation.imu_spectrum.spectrum import band_powers, welch_psd

ODR = 7690.0
DURATION = 3.0
RESONANCE_HZ = 700.0        # a fixed "device" line, present lifted or sliding
LAMBDA_M = 26e-6            # synthetic texture wavelength, not a property of any real paper


def _band_of(frequency_hz):
    return next(band for band in BANDS if band.lower_hz <= frequency_hz < band.upper_hz)


def _signal(rate, duration, components, noise=0.0, seed=0):
    t = np.arange(int(round(rate * duration))) / rate
    rng = np.random.default_rng(seed)
    x = sum(amp * np.sin(2 * np.pi * f * t + phase) for f, amp, phase in components)
    return t, x + noise * rng.standard_normal(t.size)


def _write_segment(directory, name, t, z, *, x=None, y=None):
    path = directory / f"{name}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_s", "ax", "ay", "az"])
        zeros = np.zeros_like(t)
        for row in zip(t, x if x is not None else zeros, y if y is not None else zeros, z):
            writer.writerow([f"{value:.9g}" for value in row])
    return path.name


def _segment(segment_id, kind, file, speed, surface=None, trial=1):
    return {
        "segment_id": segment_id, "kind": kind, "file": file, "trial": trial,
        "speed": {"value_mps": speed, "method": "commanded"},
        "preload": {"description": "synthetic"}, "stroke": "unidirectional",
        "axes": {"slide": "x", "normal": "z"}, "surface": surface,
    }


def _session(tmp_path, *, rate=ODR, bus="spi", clock=8e6, mode="high_performance",
             texture=True, origin="synthetic_test", drop=None, grits=(120, 600, 1200)):
    """Blank lifted + blank static + three grits x two speeds, all synthetic."""
    noise = 0.01
    segments = []
    t, z = _signal(rate, DURATION, [(RESONANCE_HZ, 0.05, 0.0)], noise, seed=1)
    segments.append(_segment("blank_lifted", "blank_lifted_slide",
                             _write_segment(tmp_path, "blank_lifted", t, z), 0.025))
    t, z = _signal(rate, DURATION, [], noise, seed=2)
    segments.append(_segment("blank_static", "blank_static_contact",
                             _write_segment(tmp_path, "blank_static", t, z), 0.0, {"name": "desk"}))
    seed = 10
    for grit in grits:
        # Synthetic wavelength shrinks with grit so that the order test has
        # something to order. These are test inputs, not paper properties.
        wavelength = {120: 125e-6, 600: 26e-6, 1200: 15e-6}.get(grit, LAMBDA_M)
        for speed in (0.0125, 0.025):
            texture_hz = speed / wavelength
            # Without texture, contact excites the device line harder than the
            # lifted blank does, so the residual peak is the resonance itself.
            components = [(RESONANCE_HZ, 0.05 if texture else 0.2, 0.3)]
            if texture:
                components.append((texture_hz, 0.4, 0.0))
            t, z = _signal(rate, DURATION, components, noise, seed=seed)
            seed += 1
            if drop is not None:
                t, z = drop(t, z)
            name = f"P{grit}_{speed}"
            segments.append(_segment(name, "surface_slide", _write_segment(tmp_path, name, t, z), speed,
                                     {"name": "sandpaper", "grit": grit, "grit_scale": "FEPA"}))
    manifest = {
        "schema": "p2.4-imu-session/v1", "session_id": "synthetic", "data_origin": origin,
        "device": {"model": "LSM6DSV16X", "odr_hz": rate, "operating_mode": mode,
                   "units": "m/s^2", "bytes_per_sample": 6},
        "interface": {"bus": bus, "clock_hz": clock},
        "columns": {"time": "t_s", "x": "ax", "y": "ay", "z": "az"},
        "segments": segments,
    }
    path = tmp_path / "session.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


# -- bands -----------------------------------------------------------------

def test_bands_are_contiguous_third_octaves():
    assert len(BANDS) == 22
    for a, b in zip(BANDS, BANDS[1:]):
        assert math.isclose(a.upper_hz, b.lower_hz, rel_tol=1e-12)
        assert math.isclose(b.centre_hz / a.centre_hz, 10 ** 0.1, rel_tol=1e-12)
    assert BANDS[0].nominal_hz == 25 and BANDS[-1].nominal_hz == 3150


def test_every_band_is_available_at_the_planned_odr_and_not_above_nyquist_at_3840():
    assert all(available(band, ODR) for band in BANDS)
    at_3840 = [band for band in BANDS if not available(band, 3840.0)]
    assert at_3840 and all(band.upper_hz > 1920.0 for band in at_3840)
    assert definition()["version"] == "p2.4-bands/v1"


# -- spectrum --------------------------------------------------------------

def test_sine_lands_in_its_band_and_parseval_holds():
    _, x = _signal(ODR, DURATION, [(320.0, 1.0, 0.0)])
    psd = welch_psd(x, ODR)
    powers = band_powers(psd)
    target = _band_of(320.0).index
    assert powers[target] == max(p for p in powers if p is not None)
    # A unit sine has variance 0.5; nearly all of it is in one band.
    assert powers[target] == pytest.approx(0.5, rel=0.02)
    assert psd.density.sum() * psd.resolution_hz == pytest.approx(np.var(x), rel=0.02)


def test_resolution_is_set_in_seconds_so_two_odrs_agree():
    for rate in (ODR, 3840.0):
        _, x = _signal(rate, DURATION, [(320.0, 1.0, 0.0)])
        psd = welch_psd(x, rate)
        assert psd.resolution_hz == pytest.approx(2.0, rel=0.01)
        assert band_powers(psd)[_band_of(320.0).index] == pytest.approx(0.5, rel=0.02)


def test_bands_above_nyquist_are_none_not_zero():
    _, x = _signal(3840.0, DURATION, [(320.0, 1.0, 0.0)])
    powers = band_powers(welch_psd(x, 3840.0))
    unavailable = [band.index for band in BANDS if not available(band, 3840.0)]
    assert unavailable and all(powers[i] is None for i in unavailable)


# -- ingestion: both buses, load-time rejection ----------------------------

@pytest.mark.parametrize("bus,clock", [("spi", 8e6), ("i2c", 1e6), ("i2c", 4e5)])
def test_either_bus_is_accepted_and_its_estimate_only_reported(tmp_path, bus, clock):
    session = load_session(_session(tmp_path, bus=bus, clock=clock), require_measured=False)
    assert session.interface["bus"] == bus
    assert session.throughput["status"] == "estimate_only"


def test_i2c_400k_estimate_falls_short_but_does_not_reject_by_itself():
    record = check_device({"model": "m", "odr_hz": ODR, "operating_mode": "high_performance",
                           "units": "g", "bytes_per_sample": 6}, {"bus": "i2c", "clock_hz": 4e5})
    assert record["estimate_margin"] < 1.0


def test_counterexample_uneven_sample_loss_is_rejected(tmp_path):
    def drop(t, z):
        keep = np.ones(t.size, bool)
        keep[[500, 501, 9000]] = False          # uneven loss
        return t[keep], z[keep]
    with pytest.raises(RecordingRejected, match="inter-sample interval"):
        load_session(_session(tmp_path, drop=drop), require_measured=False)


def test_counterexample_even_loss_shows_as_wrong_rate(tmp_path):
    """Dropping every other sample keeps intervals even; the rate check must catch it."""
    t = np.arange(0, 3.0, 2.0 / ODR)
    with pytest.raises(RecordingRejected):
        check_sampling(t, ODR, IngestPolicy(max_interval_factor=2.5), "even")


def test_counterexample_normal_mode_cannot_run_at_7690(tmp_path):
    with pytest.raises(RecordingRejected, match="normal"):
        load_session(_session(tmp_path, mode="normal"), require_measured=False)


def test_counterexample_missing_grit_scale_is_rejected(tmp_path):
    path = _session(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["segments"][2]["surface"].pop("grit_scale")
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RecordingRejected, match="grit_scale"):
        load_session(path, require_measured=False)


def test_counterexample_missing_speed_method_is_rejected(tmp_path):
    path = _session(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["segments"][3]["speed"]["method"] = None
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RecordingRejected, match="method"):
        load_session(path, require_measured=False)


def test_counterexample_session_without_lifted_blank_is_rejected(tmp_path):
    path = _session(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["segments"] = [s for s in data["segments"] if s["kind"] != "blank_lifted_slide"]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RecordingRejected, match="blank_lifted_slide"):
        load_session(path, require_measured=False)


def test_template_is_rejected_until_filled():
    from pathlib import Path
    template = Path(__file__).resolve().parents[1] / "imu_spectrum/session_template.json"
    with pytest.raises(RecordingRejected, match="null"):
        load_session(template)


# -- synthetic data cannot become a report --------------------------------

def test_synthetic_session_is_refused_by_default_and_by_the_cli(tmp_path):
    path = _session(tmp_path)
    with pytest.raises(RecordingRejected, match="measured"):
        load_session(path)
    assert cli_main(["--session", str(path), "--out", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out" / "report.json").exists()


# -- analysis: blank subtraction, grit order, speed doubling ---------------

@pytest.fixture
def report(tmp_path):
    return analyse(load_session(_session(tmp_path), require_measured=False))


def test_report_carries_origin_policy_and_band_definition(report):
    assert report["data_origin"] == "synthetic_test"
    assert report["ingest_policy"]["status"] == "proposed_unconfirmed"
    assert report["speed_scaling_tolerance_rel"]["status"] == "proposed_unconfirmed"
    assert report["bands"]["version"] == "p2.4-bands/v1"


def test_blank_peak_is_found_and_not_taken_for_texture(report):
    assert any(abs(p["frequency_hz"] - RESONANCE_HZ) <= 2 for p in report["blank"]["lifted_peaks"])
    for row in report["segments"]:
        if row["segment"]["kind"] == "surface_slide":
            assert abs(row["peak_hz_minus_blank"] - RESONANCE_HZ) > 10


def test_blank_subtraction_is_not_clipped(report):
    row = next(r for r in report["segments"] if r["segment"]["kind"] == "surface_slide")
    values = [v for v in row["band_power_minus_blank"] if v is not None]
    assert any(v < 0 for v in values), "a band at or below the blank must stay negative, not become 0"
    assert all(flag is False for v, flag in zip(row["band_power_minus_blank"], row["above_blank"])
               if v is not None and v <= 0)


def test_texture_peak_follows_v_over_lambda_and_doubles_with_speed(report):
    for key, entry in report["surfaces"].items():
        scaling = entry["speed_scaling"]
        assert scaling["speed_ratio"] == pytest.approx(2.0)
        assert scaling["peak"]["frequency_ratio"] == pytest.approx(2.0, rel=0.02), key
        assert scaling["peak"]["consistent_with_v_over_lambda"]


def test_grit_order_is_increasing_at_each_speed(report):
    for entry in report["grit_order"].values():
        assert entry["grits"] == [120, 600, 1200]
        assert entry["peak_strictly_increasing"] is True


def test_counterexample_resonance_only_does_not_double(tmp_path):
    """No texture, only the fixed device line: the speed check must fail."""
    result = analyse(load_session(_session(tmp_path, texture=False), require_measured=False))
    scalings = [entry["speed_scaling"] for entry in result["surfaces"].values()]
    assert scalings
    for scaling in scalings:
        peak = scaling["peak"]
        assert peak is not None and not peak["consistent_with_v_over_lambda"]
        assert peak["frequency_ratio"] == pytest.approx(1.0, abs=0.01)
    assert all(row["peak_coincides_with_blank_peak"]
               for row in result["segments"] if row["segment"]["kind"] == "surface_slide")


def test_counterexample_shuffled_grits_are_not_increasing(tmp_path):
    """Swap the synthetic wavelengths of P600 and P1200: the order check must say false."""
    path = _session(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    for segment in data["segments"]:
        surface = segment.get("surface") or {}
        if surface.get("grit") in (600, 1200):
            surface["grit"] = 1200 if surface["grit"] == 600 else 600
    path.write_text(json.dumps(data), encoding="utf-8")
    result = analyse(load_session(path, require_measured=False))
    assert any(entry["peak_strictly_increasing"] is False for entry in result["grit_order"].values())


def test_same_analysis_runs_at_the_i2c_fallback_rate(tmp_path):
    """3840 Hz: still analysable, with the top bands unavailable rather than zero."""
    result = analyse(load_session(_session(tmp_path, rate=3840.0, bus="i2c", clock=4e5, grits=(120,)),
                                  require_measured=False))
    row = next(r for r in result["segments"] if r["segment"]["kind"] == "surface_slide")
    assert None in row["band_power_total"]
    assert result["surfaces"]["sandpaper|FEPA120"]["speed_scaling"]["peak"]["consistent_with_v_over_lambda"]
