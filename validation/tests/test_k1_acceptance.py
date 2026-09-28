"""The K1 acceptance gates, plus the run-directory round trip."""

import json

import pytest

from mechanics.k1_report import environment_record, missing_required_artifacts, write_run
from mechanics.k1_rigid import K1Experiment
from scenarios.k1_variants import with_substeps
from tactile_contract.io import load_run
from validation.k1_acceptance import (
    RUN_GATE_ORDER,
    acceptance_report,
    evaluate_replay,
    evaluate_run,
    evaluate_time_refinement,
)

#: Full gate order: the single-run gates, then the two that need extra inputs.
GATE_ORDER = (
    *RUN_GATE_ORDER,
    "time_step_does_not_change_data_semantics",
    "interaction_state_replays_completely",
)


def test_run_level_gates_pass_and_are_reported_in_order(k1_run):
    report = evaluate_run(k1_run)
    assert [gate["gate"] for gate in report["gates"]] == list(RUN_GATE_ORDER)
    assert all(gate["passed"] for gate in report["gates"])
    assert report["segment_order_complete"] is True


def test_gate_thresholds_come_from_the_declared_spec_not_from_the_code(k1_run, k1_spec):
    limits = k1_spec.to_dict()["acceptance"]
    gates = {gate["gate"]: gate for gate in evaluate_run(k1_run)["gates"]}
    hold = gates["hold_contact_is_stable"]
    conservation = gates["force_matches_impulse_and_is_conserved"]
    assert hold["contact_coverage_min"] == limits["contact_coverage_min"]
    assert hold["hold_tangent_speed_limit_mps"] == limits["hold_tangent_speed_max_mps"]
    assert conservation["force_conservation_atol_n"] == limits["force_conservation_atol_n"]


def test_contact_and_hold_gates_measure_real_coverage(k1_run):
    gates = {gate["gate"]: gate for gate in evaluate_run(k1_run)["gates"]}
    assert gates["single_probe_contact_appears"]["contact_frames"] > 0
    assert gates["single_probe_contact_appears"]["patch_ids"] == [1]
    assert gates["hold_contact_is_stable"]["hold_contact_coverage"] == 1.0
    assert gates["release_clears_contact_force"]["max_released_total_force_n"] == 0.0


def test_slide_segment_meets_the_step_six_speed_threshold(k1_run, k1_spec):
    report = evaluate_run(k1_run)
    limits = k1_spec.to_dict()["acceptance"]
    assert report["slide_speed_pass_fraction"] >= limits["slide_speed_pass_fraction_min"]
    assert report["contact_coverage_hold_and_slide"] >= limits["contact_coverage_min"]


def test_time_refinement_keeps_data_semantics_identical(k1_run, k1_refinement_runs):
    gate = evaluate_time_refinement([k1_run, *k1_refinement_runs])
    assert gate["passed"] is True
    assert [item["substeps_per_control"] for item in gate["comparisons"]] == [2, 4]
    for item in gate["comparisons"]:
        assert item["frame_count_equal"]
        assert item["segment_sequence_equal"]
        assert item["contact_sequence_equal"]
        assert item["cell_sequence_equal"]
        assert item["patch_sequence_equal"]
    assert all(entry["eligible"] for entry in gate["eligibility"])


def test_refinement_variant_is_a_distinct_comparable_specification(k1_spec):
    variant = with_substeps(k1_spec, 4)
    original, derived = k1_spec.to_dict(), variant.to_dict()
    assert derived["timing"]["substeps_per_control"] == 4
    assert derived["timing"]["physics_dt_s"] == pytest.approx(original["timing"]["control_dt_s"] / 4)
    assert derived["spec_sha256"] != original["spec_sha256"]
    for key in ("surface", "probes", "controller", "grid", "sampling", "runtime", "acceptance"):
        assert derived[key] == original[key]


def test_time_refinement_needs_more_than_one_run(k1_run):
    with pytest.raises(ValueError, match="at least two runs"):
        evaluate_time_refinement([k1_run])


def test_run_directory_replays_through_the_manifest_and_hashes(k1_run_directory, k1_run):
    gate = evaluate_replay(k1_run_directory)
    assert gate["passed"] is True
    assert gate["frames"] == len(k1_run.frames)
    assert gate["run_id"] == k1_run.run_id
    assert gate["artifacts_verified"] == 14


def test_reloaded_frames_are_identical_to_the_frames_in_memory(k1_run_directory, k1_run):
    _, frames = load_run(k1_run_directory)
    assert [frame.to_dict() for frame in frames] == [frame.to_dict() for frame in k1_run.frames]


def test_effective_runtime_classifies_every_declared_field(k1_experiment):
    policy = k1_experiment.effective_runtime()["field_policy"]
    assert sum(policy["counts"].values()) == len(policy["fields"])
    assert set(policy["counts"]) == {"enforced", "recorded_inert", "single_implementation"}
    assert policy["fields"]["contact_offset_m"]["kernel_verdict"] == "single_implementation"
    assert policy["fields"]["gravity_mps2"]["kernel_verdict"] == "enforced"
    assert policy["fields"]["sampling.kinematics_source"]["kernel_verdict"] == "enforced"


def test_manifest_binds_conditions_observables_and_every_artifact(k1_run_directory, k1_run):
    manifest, _ = load_run(k1_run_directory)
    data = manifest.to_dict()
    assert data["conditions_sha256"] == k1_run.conditions_sha256
    assert data["acceptance"]["expected_frames"] == len(k1_run.frames)
    assert sum(1 for item in data["artifacts"] if item["role"] == "interaction_frames") == 1
    assert data["backend"]["id"] == "rigid_cpu_v1"
    for name, definition in data["observables"].items():
        assert definition["definition_id"] == f"k1_rigid_cpu.{name}"


def test_a_tampered_artifact_breaks_the_replay_gate(k1_run_directory, tmp_path):
    import shutil

    copy = tmp_path / "tampered"
    shutil.copytree(k1_run_directory, copy)
    target = copy / "points.csv"
    target.write_text(target.read_text(encoding="utf-8") + "0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
        load_run(copy)


def test_every_required_run_artifact_is_produced(k1_run, k1_run_directory):
    assert missing_required_artifacts(k1_run, k1_run_directory) == []


def test_written_run_reports_the_five_layer_streams(k1_run_directory, k1_run):
    for name in ("material_response.jsonl", "tactile_target.jsonl", "actuator_io.jsonl"):
        lines = (k1_run_directory / name).read_text(encoding="utf-8").splitlines()
        assert len(lines) == len(k1_run.frames)
        json.loads(lines[0])


def test_effective_runtime_records_the_drive_relocation_and_open_questions(k1_experiment):
    runtime = k1_experiment.effective_runtime()
    assert runtime["drive"]["gravity_compensation_mode"] == "per_substep_feedforward"
    assert "relocation_note" in runtime["drive"]
    assert runtime["unresolved_settings"] == []
    assert "no claim of numeric" in runtime["isaac_sim_parity"]


def test_environment_record_is_declared_as_host_specific(k1_run):
    record = environment_record(k1_run)
    assert record["compute_backend"] == "cpu_double_precision"
    assert record["wall_clock_s"] > 0.0
    assert "not part of the physical conditions" in record["note"]


def test_full_report_collects_every_gate(k1_run, k1_refinement_runs, k1_run_directory):
    report = acceptance_report(k1_run, refinement_runs=k1_refinement_runs,
                              run_directory=k1_run_directory)
    assert report["gate_names"] == list(GATE_ORDER)
    assert report["missing_gates"] == []
    assert report["all_passed"] is True


def test_a_broken_run_fails_its_gate_instead_of_being_reported_as_passing(k1_spec):
    import copy

    from tactile_contract import ExperimentSpec

    data = copy.deepcopy(k1_spec.to_dict())
    # An approach that cannot reach the surface must fail the contact gate.
    data["controller"]["parameters"]["approach_timeout_s"] = 0.1
    broken = K1Experiment(ExperimentSpec.build(**data)).run()
    report = acceptance_report(broken)
    gates = {gate["gate"]: gate["passed"] for gate in report["gates"]}
    assert broken.loop_state == "aborted"
    assert "approach timeout" in broken.abort_reason
    assert gates["single_probe_contact_appears"] is False
    assert gates["run_completes_declared_program"] is False
    assert report["segment_order_complete"] is False
    assert report["all_passed"] is False
