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
    CROSS_RUN_GATE_ORDER,
    DEFERRED_ACCEPTANCE_KEYS,
    RUN_GATE_ORDER,
    acceptance_key_ledger,
    acceptance_report,
    evaluate_run,
    evaluate_time_refinement,
    finalize_report,
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
    # 7.5 s is 450 control steps, which lands inside the released segment: the
    # probe made contact, held, slid and released, so every earlier gate is
    # satisfied and only the completeness gate fails.
    broken = K1Experiment(mutate(k1_spec, lambda d: d["timing"].update(max_duration_s=7.5))).run()
    report = acceptance_report(broken)
    completeness = gate_of(report, "run_completes_declared_program")
    assert broken.loop_state == "aborted"
    assert "timeout" in broken.abort_reason
    assert completeness["passed"] is False
    assert completeness["segment_order_complete"] is True
    assert gate_of(report, "release_clears_contact_force")["passed"] is True
    assert report["run_checks_passed"] is False
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


def test_single_run_checks_do_not_amount_to_k1_acceptance(k1_run):
    # A caller that supplies neither refinement runs nor a written directory
    # can establish the single-run checks and nothing more.
    report = acceptance_report(k1_run)
    assert report["missing_run_gates"] == []
    assert report["run_checks_passed"] is True
    assert report["missing_gates"] == list(CROSS_RUN_GATE_ORDER)
    assert report["k1_acceptance_complete"] is False
    assert report["k1_acceptance_passed"] is False
    assert report["all_passed"] is False


def test_full_acceptance_needs_refinement_and_a_written_directory(
    k1_run, k1_refinement_runs, k1_run_directory
):
    partial = acceptance_report(k1_run, refinement_runs=k1_refinement_runs)
    assert partial["missing_gates"] == ["interaction_state_replays_completely"]
    assert partial["k1_acceptance_passed"] is False

    complete = acceptance_report(
        k1_run, refinement_runs=k1_refinement_runs, run_directory=k1_run_directory
    )
    assert complete["missing_gates"] == []
    assert complete["k1_acceptance_complete"] is True
    assert complete["k1_acceptance_passed"] is True
    assert complete["all_passed"] is True


def test_time_refinement_requires_two_distinct_physics_step_counts(k1_run):
    # Passing a run as its own refinement changes no time step and proves
    # nothing, so it is refused rather than counted as a pass.
    with pytest.raises(ValueError, match="distinct physics step counts"):
        evaluate_time_refinement([k1_run, k1_run])


def test_dropping_a_gate_flips_the_verdict_through_the_shared_finalizer(
    k1_run, k1_refinement_runs, k1_run_directory
):
    report = acceptance_report(
        k1_run, refinement_runs=k1_refinement_runs, run_directory=k1_run_directory
    )
    assert report["all_passed"] is True
    trimmed = finalize_report({
        **report,
        "gates": [g for g in report["gates"] if g["gate"] != "slide_speed_meets_threshold"],
    })
    assert trimmed["missing_run_gates"] == ["slide_speed_meets_threshold"]
    assert trimmed["run_checks_passed"] is False
    assert trimmed["all_passed"] is False


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
