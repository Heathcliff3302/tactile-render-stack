"""The curve report plots the run, and only the run.

A report that silently redraws an approximation of the data, or that quietly
drops a series it could not render, is worse than no report: it looks like
evidence. So the checks here are that every declared chart actually produced
a line, that the values on it came from the run's own arrays, and that the
regulation band drawn is the one the gate scored against.
"""

import re

import pytest

from mechanics.curve_report import build_report, write_curve_report
from validation.k1_acceptance import acceptance_report


@pytest.fixture(scope="module")
def k1_html(k1_run, k1_experiment):
    report = acceptance_report(k1_run)
    return build_report(k1_run, report, k1_experiment.effective_runtime())


@pytest.fixture(scope="module")
def k2_html(k2_run, k2_experiment):
    report = acceptance_report(k2_run)
    return build_report(k2_run, report, k2_experiment.effective_runtime())


def polylines(html):
    return re.findall(r'<polyline[^>]*points="([^"]*)"', html)


def test_the_document_is_self_contained(k1_html):
    # It must open offline, with no server and no file beside it.
    assert k1_html.startswith("<!doctype html>")
    assert "<svg" in k1_html
    assert "http://" not in k1_html and "https://" not in k1_html
    assert "<script" not in k1_html


def test_every_chart_drew_a_line(k1_html):
    assert k1_html.count("<svg") == 4
    lines = polylines(k1_html)
    assert len(lines) >= 4
    # A silently empty series would render as an empty polyline.
    assert all(points.strip() for points in lines)


def test_no_undefined_value_reaches_the_drawing(k1_html, k2_html):
    # None and NaN must be skipped, never stringified into a coordinate.
    for html in (k1_html, k2_html):
        for points in polylines(html):
            assert "nan" not in points.lower()
            assert "None" not in points
            for pair in points.split():
                x, y = pair.split(",")
                float(x), float(y)


def test_a_gap_in_the_data_breaks_the_line_rather_than_bridging_it(k1_run, k1_experiment):
    """Joining across a no-contact frame would assert continuity the run denies."""
    report = acceptance_report(k1_run)
    html = build_report(k1_run, report, k1_experiment.effective_runtime())
    # The force series is undefined before contact and after release, so the
    # chart must contain more than one line for it rather than one that spans
    # the whole axis.
    force_svg = re.findall(r"<svg.*?</svg>", html, re.S)[1]
    # The sampled-force series specifically. The layer 5 measured series on the
    # same chart is defined on every frame, so it is legitimately continuous
    # and is not what this check is about.
    sampled = re.findall(
        r'<polyline[^>]*stroke="#1c7ed6"[^>]*points="([^"]*)"', force_svg
    )
    contact_frames = sum(
        1 for frame in k1_run.frames if frame.normal_force_n is not None
    )
    assert 0 < contact_frames < len(k1_run.frames)
    assert sampled, "the sampled force series was not drawn"
    assert sum(len(points.split()) for points in sampled) == contact_frames
    assert max(len(points.split()) for points in sampled) <= contact_frames


def test_the_plotted_force_matches_the_run(k2_run, k2_html):
    # Spot-check that the drawing is of this run: the peak force in the
    # document's own summary must be the run's peak.
    peak = max(frame.normal_force_n or 0.0 for frame in k2_run.frames)
    assert peak > 0.9
    assert f"{len(k2_run.frames)} 帧" in k2_html


def test_the_declared_band_is_drawn_for_a_force_run(k2_run, k2_html):
    force = k2_run.spec.to_dict()["controller"]["force"]
    assert force["target_force_n"] == 1.0
    # The band rectangle and its centre line are present.
    assert "#2f9e4420" in k2_html
    assert "target_force_n ± force_tolerance_n" in k2_html


def test_a_trajectory_run_has_no_band(k1_run, k1_html):
    assert k1_run.spec.to_dict()["controller"]["force"] is None
    assert "#2f9e4420" not in k1_html


def test_the_segments_and_gates_are_reported(k1_run, k1_html):
    for phase in ("pre_contact", "approach", "hold", "lateral_slide", "retract", "released"):
        assert phase in k1_html
    assert "run_completes_declared_program" in k1_html
    assert "force_matches_impulse_and_is_conserved" in k1_html


def test_force_segments_appear_for_a_force_run(k2_html):
    for phase in ("approach_force", "regulate_force", "lateral_slide_force"):
        assert phase in k2_html


def test_the_boundaries_are_restated_not_left_to_the_reader(k1_html):
    # A chart of a proxy area that does not say it is a proxy invites exactly
    # the reading this repository refuses elsewhere.
    assert "contact_area_proxy_m2" in k1_html
    assert "unavailable" in k1_html
    assert "modelled_not_measured" in k1_html
    assert "Isaac Sim" in k1_html


def test_the_milestone_is_not_overstated(k2_run, k2_experiment):
    report = acceptance_report(k2_run)
    html = build_report(k2_run, report, k2_experiment.effective_runtime())
    assert "K2-A" in html
    assert "isaac_sim_parity" in html


def test_writing_returns_a_record_of_what_was_drawn(tmp_path, k1_run, k1_experiment):
    report = acceptance_report(k1_run)
    record = write_curve_report(
        tmp_path / "contact_report.html", k1_run, report,
        k1_experiment.effective_runtime(),
    )
    assert (tmp_path / "contact_report.html").is_file()
    assert record["path"] == "contact_report.html"
    assert record["frames"] == len(k1_run.loop_frames)
    assert len(record["charts"]) == 4
    assert "not recomputed from a model" in record["source"]
