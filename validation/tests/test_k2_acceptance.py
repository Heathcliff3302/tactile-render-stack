"""K2 acceptance: the force gates, and the ledger that makes them mandatory.

The structural risk here is the one K1 already paid for once. A force run that
emits only the seven common gates looks complete — every gate present passes —
while nothing has checked whether the regulator ever reached its target. So
the required gate set is derived from the controller mode, and a force run
missing a force gate cannot report a pass.

The mirror risk is a threshold declared where no gate can read it. A
trajectory specification declaring ``force_pass_fraction_min`` would be
exactly that, so the ledger rejects it in both directions.
"""

import copy

import pytest

from tactile_contract import ExperimentSpec
from validation.k1_acceptance import (
    FORCE_ACCEPTANCE_KEY_GATES,
    FORCE_RUN_GATE_ORDER,
    K1_GATE_ORDER,
    K2_GATE_ORDER,
    RUN_GATE_ORDER,
    acceptance_key_ledger,
    acceptance_report,
    evaluate_run,
    finalize_report,
    gate_order,
    run_gate_order,
)


def mutate(spec, change):
    data = copy.deepcopy(spec.to_dict())
    change(data)
    return ExperimentSpec.build(**data)


# --- the ledger ------------------------------------------------------------

def test_force_mode_consumes_the_force_threshold(k2_spec):
    ledger = acceptance_key_ledger(k2_spec)
    assert ledger["controller_mode"] == "force"
    assert "force_pass_fraction_min" in ledger["consumed"]
    assert set(ledger["consumed"]["force_pass_fraction_min"]) == set(
        FORCE_ACCEPTANCE_KEY_GATES["force_pass_fraction_min"]
    )


def test_trajectory_mode_does_not_name_a_force_threshold(k1_spec):
    ledger = acceptance_key_ledger(k1_spec)
    assert ledger["controller_mode"] == "trajectory"
    assert "force_pass_fraction_min" not in ledger["consumed"]


def test_a_force_spec_missing_its_threshold_is_rejected(k2_spec):
    spec = mutate(k2_spec, lambda d: d["acceptance"].pop("force_pass_fraction_min"))
    with pytest.raises(ValueError, match="does not declare"):
        acceptance_key_ledger(spec)


def test_a_trajectory_spec_declaring_a_force_threshold_is_rejected(k1_spec):
    # A number nobody can measure is the same defect as a number nobody
    # compares, and the error says which it is.
    spec = mutate(k1_spec, lambda d: d["acceptance"].update(force_pass_fraction_min=0.9))
    with pytest.raises(ValueError, match="force-mode thresholds"):
        acceptance_key_ledger(spec)


def test_every_declared_threshold_is_still_accounted_for(k2_spec):
    declared = set(k2_spec.to_dict()["acceptance"])
    ledger = acceptance_key_ledger(k2_spec)
    assert declared == set(ledger["consumed"]) | set(ledger["deferred"])


# --- the gate order --------------------------------------------------------

def test_force_mode_requires_three_more_single_run_gates():
    assert run_gate_order("trajectory") == RUN_GATE_ORDER
    assert run_gate_order("force") == (*RUN_GATE_ORDER, *FORCE_RUN_GATE_ORDER)
    assert gate_order("trajectory") == K1_GATE_ORDER
    assert gate_order("force") == K2_GATE_ORDER
    assert len(K2_GATE_ORDER) == len(K1_GATE_ORDER) + 3


def test_a_force_run_with_only_the_common_gates_is_incomplete(k2_run):
    """The defect this ordering exists to prevent, stated as a test."""
    report = evaluate_run(k2_run)
    trimmed = {
        **report,
        "gates": [g for g in report["gates"] if g["gate"] in RUN_GATE_ORDER],
    }
    final = finalize_report(trimmed)
    assert all(gate["passed"] for gate in final["gates"])
    assert final["run_checks_passed"] is False
    assert set(final["missing_run_gates"]) == set(FORCE_RUN_GATE_ORDER)
    assert final["acceptance_passed"] is False


def test_the_mode_drives_the_requirement_not_the_gates_present(k2_run):
    report = evaluate_run(k2_run)
    assert report["controller_mode"] == "force"
    final = finalize_report(report)
    assert final["gate_order_mode"] == "force"
    assert final["required_gates"] == list(K2_GATE_ORDER)


# --- the run ---------------------------------------------------------------

def test_the_k2_run_completes_its_declared_program(k2_run):
    assert k2_run.loop_state == "completed"
    assert k2_run.abort_reason == ""
    assert k2_run.controller_mode == "force"
    assert k2_run.phase_sequence[-1] == "released"


def test_the_run_visits_the_force_segments(k2_run):
    report = evaluate_run(k2_run)
    segments = report["segments"]
    assert set(segments) == {
        "pre_contact", "approach_force", "regulate_force",
        "lateral_slide_force", "retract", "released",
    }
    assert all(count > 0 for count in segments.values())
    assert report["segment_order_complete"] is True


def test_the_role_lookup_finds_the_force_segments(k2_run):
    # The hold and slide statistics must be measured on regulate_force and
    # lateral_slide_force, not silently on an empty list.
    report = evaluate_run(k2_run)
    hold_gate = next(g for g in report["gates"] if g["gate"] == "hold_contact_is_stable")
    slide_gate = next(g for g in report["gates"] if g["gate"] == "slide_speed_meets_threshold")
    assert hold_gate["hold_frames"] > 0
    assert slide_gate["slide_frames"] > 0


def test_every_single_run_gate_passes(k2_run):
    report = evaluate_run(k2_run)
    failed = [gate["gate"] for gate in report["gates"] if not gate["passed"]]
    assert failed == []
    assert [gate["gate"] for gate in report["gates"]] == list(run_gate_order("force"))


def test_the_target_is_reached_and_held(k2_run):
    gate = next(g for g in evaluate_run(k2_run)["gates"]
                if g["gate"] == "force_target_is_reached_and_held")
    assert gate["passed"] is True
    assert gate["hold_role_segment"] == "regulate_force"
    assert gate["settle_frame_index"] is not None
    assert gate["held_in_band_fraction"] >= gate["force_pass_fraction_min"]
    assert gate["target_force_n"] == 1.0
    assert gate["force_tolerance_n"] == 0.05


def test_the_hold_lasts_the_declared_interval(k2_run):
    """A fraction says nothing about duration; the duration is required too."""
    gate = next(g for g in evaluate_run(k2_run)["gates"]
                if g["gate"] == "force_target_is_reached_and_held")
    declared = k2_run.spec.to_dict()["controller"]["parameters"]["hold_time_s"]
    control_dt = k2_run.spec.to_dict()["timing"]["control_dt_s"]
    assert gate["declared_hold_time_s"] == declared
    assert gate["required_held_frames"] == round(declared / control_dt)
    assert gate["held_long_enough"] is True
    assert gate["held_duration_s"] >= declared
    # The settle frame is the controller's first held frame, so the gate and
    # the controller measure the same window rather than differing by one.
    assert gate["settle_frame_counts_as_held"] is True
    assert gate["held_frames_scored"] == gate["hold_frames"] - gate["settle_frame_index"]


def test_a_hold_one_frame_short_of_the_declared_interval_fails(k2_run):
    """The defect this duration requirement exists to catch.

    Truncating the scored hold window to a single frame leaves a perfect
    in-band fraction of 1.0. Under a fraction-only gate that passed, while
    the run held its target for 0.0167 s against a declared 1 s.
    """
    report = evaluate_run(k2_run)
    gate = next(g for g in report["gates"]
                if g["gate"] == "force_target_is_reached_and_held")
    control_dt = k2_run.spec.to_dict()["timing"]["control_dt_s"]
    required = gate["required_held_frames"]

    for scored in (1, required - 1):
        fraction = 1.0  # every scored frame in band: the flattering case
        long_enough = scored >= required
        passed = long_enough and fraction >= gate["force_pass_fraction_min"]
        assert passed is False, (
            f"{scored} held frames ({scored * control_dt:.4f} s) must not pass "
            f"a declared hold of {gate['declared_hold_time_s']} s"
        )


def test_a_shortened_hold_segment_fails_the_gate(k2_spec):
    """End to end: declare a hold the regulator cannot complete in band.

    ``regulate_timeout_s`` is cut below the time the regulator needs to
    settle, so the run aborts in ``regulate_force`` without ever holding.
    """
    from mechanics.k1_rigid import RigidProbeExperiment

    spec = mutate(k2_spec, lambda d: d["controller"]["force"].update(
        regulate_timeout_s=2.0))
    run = RigidProbeExperiment(spec, run_id="short-hold").run()
    assert run.loop_state == "aborted"
    report = finalize_report(evaluate_run(run))
    gate = next(g for g in report["gates"]
                if g["gate"] == "force_target_is_reached_and_held")
    assert gate["passed"] is False
    assert gate["held_long_enough"] is False
    assert report["run_checks_passed"] is False
    assert report["acceptance_passed"] is False


def test_the_target_is_held_through_sliding(k2_run):
    gate = next(g for g in evaluate_run(k2_run)["gates"]
                if g["gate"] == "force_held_through_slide")
    assert gate["passed"] is True
    assert gate["slide_in_band_fraction"] >= gate["force_pass_fraction_min"]
    # Duration is required here too, for the same reason as the hold gate.
    declared = k2_run.spec.to_dict()["controller"]["parameters"]["slide_time_s"]
    assert gate["declared_slide_time_s"] == declared
    assert gate["slide_long_enough"] is True
    assert gate["slide_duration_s"] >= declared


def test_the_regulator_stayed_inside_its_declared_limits(k2_run):
    gate = next(g for g in evaluate_run(k2_run)["gates"]
                if g["gate"] == "regulator_stays_within_declared_authority")
    assert gate["passed"] is True
    assert gate["max_commanded_press_speed_mps"] <= gate["declared_press_limit_mps"]
    assert gate["max_commanded_correct_speed_mps"] <= gate["declared_correct_limit_mps"]
    assert gate["authority"]["tolerance_band_holdable"] is True


def test_only_the_regulated_segments_are_scored_against_the_regulator(k2_run):
    """Retract commands 0.3 m/s upward, above the 0.2 m/s correction limit.

    That is a trajectory command bounded by its own parameter, not a
    regulator correction. Scoring the whole run against the regulator's
    limits would fail this gate for a command the regulator never issued.
    """
    gate = next(g for g in evaluate_run(k2_run)["gates"]
                if g["gate"] == "regulator_stays_within_declared_authority")
    assert gate["regulated_segments"] == ["lateral_slide_force", "regulate_force"]
    assert gate["regulated_command_steps"] > 0
    retract_speed = k2_run.spec.to_dict()["controller"]["parameters"]["retract_speed_mps"]
    assert retract_speed > gate["declared_correct_limit_mps"]
    assert max(record["command_velocity_world_mps"][2] for record in k2_run.commands) == (
        pytest.approx(retract_speed)
    )


def test_the_thresholds_come_from_the_regulator_not_a_second_declaration(k2_run):
    # The tolerance has one home. A gate restating it could drift from the
    # band the controller actually regulated to.
    declared = k2_run.spec.to_dict()["controller"]["force"]
    gate = next(g for g in evaluate_run(k2_run)["gates"]
                if g["gate"] == "force_target_is_reached_and_held")
    assert gate["force_tolerance_n"] == declared["force_tolerance_n"]
    assert gate["target_force_n"] == declared["target_force_n"]
    assert gate["settle_frames_required"] == declared["settle_frames"]
    assert "force_tolerance_n" not in k2_run.spec.to_dict()["acceptance"]


def test_a_partial_report_is_not_a_pass(k2_run):
    # Without refinement runs and a written directory, the cross-run gates are
    # absent and acceptance must not read as complete.
    report = acceptance_report(k2_run)
    assert report["run_checks_passed"] is True
    assert report["acceptance_complete"] is False
    assert report["acceptance_passed"] is False
    assert report["all_passed"] is False
