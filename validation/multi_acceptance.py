"""P2.2.2 acceptance: several probes, one stream per probe/surface pair.

Two layers of gates.

**Per stream.** Every stream is a complete single-pair ``InteractionState``
stream, so it must pass the full K1 trajectory gate order — the same ten
gates, the same thresholds, read by the same code in
``validation/k1_acceptance.py``. Nothing is re-implemented for the multi case;
each stream is handed to the single-probe evaluator as a per-stream view.

**Across streams.** Three gates that only exist once there is more than one
probe:

``streams_are_independent``
    Each stream, recorded next to its neighbour, is identical to the stream
    that probe produces when it is the only body in the kernel. "Do not
    contaminate each other" is checked against a counterfactual, not by
    inspecting the stream for foreign IDs: a contaminated value need not carry
    the other probe's name. Identity fields that legitimately differ between
    the two runs (``run_id``, ``scenario_id``, ``conditions_sha256``) are
    excluded and listed. If the neighbour's program runs longer, the stream's
    extra frames must all be ``released`` with no contact.

``patch_ids_are_stable``
    Step 6.5's Patch-ID stability, per stream. A patch is *primary* in a
    segment if it is present on at least ``primary_patch_min_occupancy`` of
    that segment's frames; every other patch seen there is an *edge* patch,
    reported separately. Stable patches are primary in both hold and slide.
    Each stream must keep at least one stable patch and
    ``|stable| / |primary in slide| >= patch_coverage_min``. Both thresholds
    are Step 6.5's declared values. This is the gate that consumes
    ``patch_coverage_min`` in ``multi_trajectory`` mode.

``box_streams_activate_multiple_cells``
    Every box stream's hold segment has a patch spanning at least two cells.
    It fails when the specification has no box at all, rather than passing
    vacuously.

Verdicts follow the K1 rule: ``run_acceptance_passed`` needs every gate
emitted and passed, and it is not milestone completion.
"""

from __future__ import annotations

from virtual_reality import PHASE_ROLES

from .k1_acceptance import _gate, acceptance_key_ledger, gate_order

MULTI_GATE_ORDER = (
    "streams_are_independent",
    "patch_ids_are_stable",
    "box_streams_activate_multiple_cells",
)

#: Frame fields that identify the run rather than describe the contact.
IDENTITY_FIELDS = ("run_id", "scenario_id", "conditions_sha256")

#: P2.2 deliverables. Code existing is not user acceptance; the ledger only
#: says what is still missing, so milestone_complete stays false until
#: P2.2.3 exists, whatever a run reports.
P2_2_DELIVERABLES: dict[str, tuple[bool, str]] = {
    "box_probe": (True, "P2.2.1: flat-bottom box probe, multi-cell footprint"),
    "multi_probe_streams": (True, "P2.2.2: one independent stream per probe/surface pair"),
    "resolution_regression": (
        False, "P2.2.3: 32/64/128 regression consuming resolution_force_spread_max"
    ),
}


def _role(frame) -> str | None:
    return PHASE_ROLES.get(frame.metadata["motion_phase"])


def _comparable(frame) -> dict:
    data = frame.to_dict()
    for name in IDENTITY_FIELDS:
        data.pop(name, None)
    return data


def evaluate_independence(stream_runs: dict, solo_runs: dict) -> dict:
    """Each stream versus the same probe running alone."""
    streams = []
    for stream_id, run in stream_runs.items():
        solo = solo_runs.get(stream_id)
        if solo is None:
            streams.append({"stream_id": stream_id, "passed": False, "reason": "no solo run"})
            continue
        n_solo, n_multi = len(solo.frames), len(run.frames)
        first_difference = None
        for index in range(min(n_solo, n_multi)):
            if _comparable(run.frames[index]) != _comparable(solo.frames[index]):
                first_difference = index
                break
        tail = run.frames[n_solo:] if n_multi >= n_solo else ()
        tail_quiet = all(
            frame.metadata["motion_phase"] == "released" and not frame.contact_present
            for frame in tail
        )
        passed = n_multi >= n_solo and first_difference is None and tail_quiet
        streams.append({
            "stream_id": stream_id,
            "passed": passed,
            "solo_frames": n_solo,
            "multi_frames": n_multi,
            "first_differing_frame": first_difference,
            "extra_frames_after_solo_end": max(0, n_multi - n_solo),
            "extra_frames_released_without_contact": tail_quiet,
        })
    return _gate("streams_are_independent", bool(streams) and all(s["passed"] for s in streams), {
        "streams": streams,
        "excluded_identity_fields": list(IDENTITY_FIELDS),
        "method": "frame-by-frame equality with the probe's solo run",
    })


def _segment_patch_counts(frames) -> dict[int, dict]:
    counts: dict[int, dict] = {}
    for frame in frames:
        for patch in frame.metadata["patches"]:
            entry = counts.setdefault(patch["patch_id"], {"frames": 0, "force_n": 0.0, "cells": 0})
            entry["frames"] += 1
            entry["force_n"] += patch["normal_force_n"]
            entry["cells"] += patch["cell_count"]
    return counts


def _split(counts: dict, segment_frames: int, min_occupancy: float):
    primary, edge = {}, {}
    for patch_id, entry in sorted(counts.items()):
        occupancy = entry["frames"] / segment_frames if segment_frames else 0.0
        summary = {
            "frames": entry["frames"],
            "occupancy": occupancy,
            "mean_normal_force_n": entry["force_n"] / entry["frames"],
            "mean_cell_count": entry["cells"] / entry["frames"],
        }
        (primary if segment_frames and occupancy >= min_occupancy else edge)[patch_id] = summary
    return primary, edge


def stream_patch_stability(run, limits: dict) -> dict:
    """Step 6.5 Patch-ID statistics for one stream, primary and edge apart."""
    min_occupancy = limits["primary_patch_min_occupancy"]
    hold = [frame for frame in run.frames if _role(frame) == "hold"]
    slide = [frame for frame in run.frames if _role(frame) == "slide"]
    hold_primary, hold_edge = _split(_segment_patch_counts(hold), len(hold), min_occupancy)
    slide_primary, slide_edge = _split(_segment_patch_counts(slide), len(slide), min_occupancy)
    stable = sorted(set(hold_primary) & set(slide_primary))
    coverage = len(stable) / len(slide_primary) if slide_primary else 0.0
    return {
        "stream_id": run.stream_id,
        "passed": bool(stable) and coverage >= limits["patch_coverage_min"],
        "hold_frames": len(hold),
        "slide_frames": len(slide),
        "primary_hold_patches": hold_primary,
        "primary_slide_patches": slide_primary,
        "edge_hold_patches": hold_edge,
        "edge_slide_patches": slide_edge,
        "stable_patch_ids": stable,
        "stable_patch_coverage": coverage,
    }


def evaluate_patch_stability(stream_runs: dict, spec) -> dict:
    limits = spec.to_dict()["acceptance"]
    streams = [stream_patch_stability(run, limits) for run in stream_runs.values()]
    return _gate("patch_ids_are_stable", bool(streams) and all(s["passed"] for s in streams), {
        "streams": streams,
        "patch_coverage_min": limits["patch_coverage_min"],
        "primary_patch_min_occupancy": limits["primary_patch_min_occupancy"],
        "patch_id_namespace": "per stream; each pair has its own patch tracker",
    })


def evaluate_box_multicell(stream_runs: dict) -> dict:
    streams = []
    for stream_id, run in stream_runs.items():
        shape = run.kernel_config.probes[0].shape
        hold = [frame for frame in run.frames if _role(frame) == "hold"]
        max_cells = max(
            (patch["cell_count"] for frame in hold for patch in frame.metadata["patches"]),
            default=0,
        )
        streams.append({
            "stream_id": stream_id,
            "shape": shape,
            "max_cells_per_patch_in_hold": max_cells,
            "required": shape == "box",
            "passed": max_cells >= 2 if shape == "box" else True,
        })
    boxes = [s for s in streams if s["required"]]
    return _gate("box_streams_activate_multiple_cells",
                 bool(boxes) and all(s["passed"] for s in boxes), {
        "streams": streams,
        "box_streams": len(boxes),
        "note": "fails when no box stream exists, so it cannot pass vacuously",
    })


def milestone_status() -> dict:
    outstanding = sorted(name for name, (done, _) in P2_2_DELIVERABLES.items() if not done)
    return {
        "schema": "milestone-status/v1",
        "controller_mode": "multi_trajectory",
        "milestone": "P2.2",
        "milestone_complete": not outstanding,
        "sub_milestone": "P2.2.2" if outstanding else "P2.2",
        "sub_milestone_scope": "CPU reference backend, single scenario",
        "deliverables": {
            name: {"delivered": done, "note": note}
            for name, (done, note) in sorted(P2_2_DELIVERABLES.items())
        },
        "outstanding": outstanding,
        "note": (
            "delivered means the code and its gates exist; user acceptance is "
            "recorded outside this ledger"
        ),
    }


def finalize_multi_report(spec, stream_reports: dict, multi_gates: list[dict]) -> dict:
    """One verdict over every per-stream gate and every cross-stream gate."""
    required_stream = gate_order("trajectory")
    missing = []
    all_passed = True
    for stream_id, report in stream_reports.items():
        names = {gate["gate"]: gate["passed"] for gate in report["gates"]}
        for name in required_stream:
            if name not in names:
                missing.append(f"{stream_id}.{name}")
        all_passed = all_passed and all(names.values())
    multi_names = {gate["gate"]: gate["passed"] for gate in multi_gates}
    missing.extend(name for name in MULTI_GATE_ORDER if name not in multi_names)
    all_passed = all_passed and all(multi_names.values())
    return {
        "schema": "p2-multi-acceptance/v1",
        "spec_id": spec.to_dict()["spec_id"],
        "acceptance_key_ledger": acceptance_key_ledger(spec),
        "streams": {
            stream_id: {gate["gate"]: gate["passed"] for gate in report["gates"]}
            for stream_id, report in stream_reports.items()
        },
        "multi_gates": multi_gates,
        "required_gates_per_stream": list(required_stream),
        "required_multi_gates": list(MULTI_GATE_ORDER),
        "missing_gates": missing,
        "run_acceptance_complete": not missing,
        "run_acceptance_passed": not missing and all_passed,
        "milestone": milestone_status(),
        "verdict_scope": (
            "run_acceptance_passed covers every stream's ten K1 gates plus the "
            "three cross-stream gates on this one scenario; it is not milestone "
            "completion"
        ),
    }
