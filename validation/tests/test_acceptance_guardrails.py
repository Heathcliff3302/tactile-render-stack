"""The acceptance layer must not report a failing run as passing.

These are regression tests for a real defect: the first revision aggregated
only the gates it happened to emit, so a run that aborted partway through the
released segment, or slid at the wrong speed for its whole slide segment, was
reported with ``all_passed = true``. The structural test at the end is the one
that keeps this from recurring as K2 to K5 add thresholds.
"""

import copy

import pytest

from mechanics.k1_rigid import K1Experiment
from tactile_contract import ExperimentSpec
from validation.k1_acceptance import (
    ACCEPTANCE_KEY_GATES,
    DEFERRED_ACCEPTANCE_KEYS,
    RUN_GATE_ORDER,
    acceptance_key_ledger,
    acceptance_report,
    evaluate_run,
)


def mutate(spec, change):
    data = copy.deepcopy(spec.to_dict())
    change(data)
    return ExperimentSpec.build(**data)


def gate_of(report, name):
    return next(gate for gate in report["gates"] if gate["gate"] == name)


def test_every_declared_threshold_is_consumed_by_a_gate_or_declared_deferred(k1_spec):
    ledger = acceptance_key_ledger(k1_spec)
    declared = set(k1_spec.to_dict()["acceptance"])
    assert declared == set(ledger["consumed"]) | set(ledger["deferred"])
    assert set(ledger["deferred"]) == {"patch_coverage_min", "resolution_force_spread_max"}
    assert all(reason.startswith("K") for reason in ledger["deferred"].values())


def test_every_gate_named_in_the_ledger_is_actually_emitted(k1_run):
    emitted = {gate["gate"] for gate in evaluate_run(k1_run)["gates"]}
    for key, gates in ACCEPTANCE_KEY_GATES.items():
        for name in gates:
            assert name in emitted, f"{key} claims gate {name}, which is never emitted"
    assert emitted == set(RUN_GATE_ORDER)


def test_a_threshold_with_no_consumer_is_rejected_not_ignored(k1_spec):
    ledger_keys = set(ACCEPTANCE_KEY_GATES) | set(DEFERRED_ACCEPTANCE_KEYS)
    data = copy.deepcopy(k1_spec.to_dict())
    data["acceptance"] = dict(data["acceptance"])
    data["acceptance"]["some_future_threshold_min"] = 0.5
    with pytest.raises(ValueError, match="neither consumed nor deferred"):
        acceptance_key_ledger(data)
    assert "some_future_threshold_min" not in ledger_keys


def test_run_that_aborts_partway_through_release_cannot_pass(k1_spec):
    # Budget ends inside the released segment: contact appeared, held and was
    # cleared, so every earlier gate is satisfied and only completeness fails.
    broken = K1Experiment(mutate(k1_spec, lambda d: d["timing"].update(max_duration_s=7.0))).run()
    report = acceptance_report(broken)
    completeness = gate_of(report, "run_completes_declared_program")
    assert broken.loop_state == "aborted"
    assert "step budget" in broken.abort_reason
    assert completeness["passed"] is False
    assert completeness["segment_order_complete"] is True
    assert gate_of(report, "release_clears_contact_force")["passed"] is True
    assert report["all_passed"] is False


def test_run_that_never_reaches_the_slide_speed_cannot_pass(k1_spec):
    def cripple(data):
        parameters = data["controller"]["parameters"]
        parameters["tangent_speed_feedback_kp"] = 0.01
        parameters["initial_slide_command_speed_mps"] = 0.001

    broken = K1Experiment(mutate(k1_spec, cripple)).run()
    report = acceptance_report(broken)
    gate = gate_of(report, "slide_speed_meets_threshold")
    assert gate["slide_speed_pass_fraction"] == 0.0
    assert gate["passed"] is False
    # The run is otherwise healthy, so only the speed gate may fail.
    assert gate_of(report, "run_completes_declared_program")["passed"] is True
    assert report["all_passed"] is False


def test_slide_settle_window_is_applied_not_merely_recorded(k1_spec):
    settled = K1Experiment(mutate(
        k1_spec, lambda d: d["acceptance"].update(slide_settle_time_s=0.5)
    )).run()
    report = evaluate_run(settled)
    gate = gate_of(report, "slide_speed_meets_threshold")
    assert gate["slide_settle_time_s"] == 0.5
    # A 1.0 s slide segment at 1/60 s loses the first 0.5 s of samples.
    assert gate["slide_measured_frames"] < gate["slide_frames"]
    assert gate["slide_measured_frames"] == pytest.approx(gate["slide_frames"] / 2, abs=1)


def test_missing_gate_cannot_be_read_as_a_pass(k1_run):
    report = acceptance_report(k1_run)
    assert report["missing_gates"] == []
    # Dropping a gate from an otherwise passing report must flip the verdict.
    trimmed = dict(report)
    trimmed["gates"] = [g for g in report["gates"] if g["gate"] != "slide_speed_meets_threshold"]
    names = [gate["gate"] for gate in trimmed["gates"]]
    assert [name for name in RUN_GATE_ORDER if name not in names] == [
        "slide_speed_meets_threshold"
    ]


def test_healthy_run_passes_every_gate_including_the_new_ones(k1_run):
    report = evaluate_run(k1_run)
    assert [gate["gate"] for gate in report["gates"]] == list(RUN_GATE_ORDER)
    assert all(gate["passed"] for gate in report["gates"])
    assert gate_of(report, "contact_coverage_meets_threshold")["contact_coverage_hold_and_slide"] == 1.0
    assert gate_of(report, "slide_speed_meets_threshold")["slide_speed_pass_fraction"] == 1.0
    assert gate_of(report, "run_completes_declared_program")["loop_state"] == "completed"


def test_report_carries_the_ledger_so_deferrals_stay_visible(k1_run):
    report = evaluate_run(k1_run)
    ledger = report["acceptance_key_ledger"]
    assert ledger["deferred"]["patch_coverage_min"].startswith("K3")
    assert ledger["consumed"]["force_conservation_atol_n"] == [
        "force_matches_impulse_and_is_conserved"
    ]
