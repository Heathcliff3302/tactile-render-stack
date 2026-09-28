"""K1 run assembly: segments, contract frames, five layers and determinism."""

import pytest

from actuators import IDEAL_MODEL_ID, MODEL_STATUS
from interaction import ALWAYS_AVAILABLE, OBSERVABLE_METHODS
from mechanics.k1_rigid import PLACEHOLDER_DIGEST, K1Experiment
from mechanics.material import RIGID_MODEL_ID
from tactile_contract import InteractionState
from tactile_contract._common import digest
from virtual_reality import MOTION_PHASES


def test_run_walks_every_declared_segment_and_ends_released(k1_run):
    phases = k1_run.phase_sequence
    for phase in MOTION_PHASES:
        assert phase in phases, phase
    assert phases[0] == "pre_contact"
    assert phases[-1] == "released"
    assert k1_run.loop_state == "completed"
    assert k1_run.abort_reason == ""


def test_segment_durations_follow_the_declared_controller_parameters(k1_run, k1_spec):
    parameters = k1_spec.to_dict()["controller"]["parameters"]
    dt = k1_spec.to_dict()["timing"]["control_dt_s"]
    counts = {phase: k1_run.phase_sequence.count(phase) for phase in (*MOTION_PHASES, "released")}
    assert counts["pre_contact"] == round(parameters["pre_settle_time_s"] / dt)
    assert counts["hold"] == round(parameters["hold_time_s"] / dt)
    assert counts["lateral_slide"] == round(parameters["slide_time_s"] / dt)
    assert counts["released"] == round(parameters["release_time_s"] / dt)


def test_contact_appears_during_approach_and_is_held_through_the_hold_segment(k1_run):
    hold = [frame for frame in k1_run.frames if frame.metadata["motion_phase"] == "hold"]
    slide = [frame for frame in k1_run.frames if frame.metadata["motion_phase"] == "lateral_slide"]
    assert hold and slide
    assert all(frame.contact_present for frame in hold)
    assert all(frame.contact_present for frame in slide)
    assert all(frame.contact_mode == "hold" for frame in hold)
    assert all(frame.contact_mode == "sliding" for frame in slide)


def test_released_frames_carry_no_contact_force_at_all(k1_run):
    released = [frame for frame in k1_run.frames if frame.metadata["motion_phase"] == "released"]
    assert released
    for frame in released:
        assert frame.contact_present is False
        assert frame.contact_mode == "no_contact"
        assert frame.normal_force_n is None
        assert frame.normal_impulse_ns is None
        assert frame.contact_area_proxy_m2 is None
        assert frame.total_force_world_n == [0.0, 0.0, 0.0]


def test_normal_force_is_the_interval_impulse_over_the_output_step(k1_run):
    contact = [frame for frame in k1_run.frames if frame.contact_present]
    assert contact
    for frame in contact:
        assert frame.normal_force_n == pytest.approx(frame.normal_impulse_ns / frame.dt_s, abs=1e-12)
        assert frame.total_force_world_n[2] == pytest.approx(frame.normal_force_n, abs=1e-12)
    peak = max(frame.normal_force_n for frame in contact)
    assert peak == pytest.approx(0.05 * 0.2 * 60, rel=1e-9)


def test_point_cell_patch_chain_is_populated_and_conserves_force(k1_run):
    contact = [frame for frame in k1_run.frames if frame.contact_present]
    for frame in contact:
        conservation = frame.metadata["conservation"]
        assert frame.contact_point_count == 1
        assert frame.active_patch_count == 1
        assert len(frame.metadata["cells"]) == 1
        assert frame.metadata["patches"][0]["cell_count"] == 1
        assert conservation["point_to_cell_error_n"] == pytest.approx(0.0, abs=1e-12)
        assert conservation["cell_to_patch_error_n"] == pytest.approx(0.0, abs=1e-12)
    assert len(k1_run.episodes) == 1
    assert k1_run.episodes[0].sliding_detected is True


def test_all_three_area_layers_are_reported_together(k1_run):
    contact = [frame for frame in k1_run.frames if frame.contact_present]
    layers = contact[0].metadata["area_layers"]
    assert layers["methods"] == ["point_occupancy", "swept_path", "geometric_footprint"]
    assert layers["point_occupancy_m2"] > 0.0
    assert layers["swept_path_m2"] > 0.0
    # A rigid non-penetrating contact has no geometric footprint, and no model
    # area exists before the K4 material models.
    assert layers["geometric_footprint_m2"] == 0.0
    assert layers["model_footprint_m2"] is None
    assert all(frame.physical_contact_area_m2 is None for frame in k1_run.frames)


def test_frames_are_valid_interaction_states_with_one_method_per_observable(k1_run):
    assert all(isinstance(frame, InteractionState) for frame in k1_run.frames)
    seen: dict[str, set[str]] = {}
    for index, frame in enumerate(k1_run.frames):
        data = frame.to_dict()
        assert data["sequence_id"] == index
        assert data["conditions_sha256"] == k1_run.conditions_sha256
        for name, quality in data["quality"].items():
            if name == "contact" or quality["status"] == "unavailable":
                continue
            seen.setdefault(name, set()).add(quality["method"])
        for name in ALWAYS_AVAILABLE:
            assert data[name] is not None, name
    assert all(len(methods) == 1 for methods in seen.values())
    for name, methods in seen.items():
        assert methods == {OBSERVABLE_METHODS[name]}


def test_conditions_digest_matches_the_recorded_conditions(k1_run):
    assert k1_run.conditions_sha256 == digest(k1_run.conditions)
    assert k1_run.conditions_sha256 != PLACEHOLDER_DIGEST
    timing = k1_run.conditions["timing"]
    assert timing["start_time_s"] == 0.0
    assert timing["end_time_s"] == pytest.approx(k1_run.frames[-1].time_s)


def test_recorded_trajectory_spans_the_run_and_is_strictly_increasing(k1_run):
    samples = k1_run.conditions["trajectory"]["samples"]
    times = [sample[0] for sample in samples]
    assert k1_run.conditions["trajectory"]["kind"].startswith("realised")
    assert times[0] == k1_run.conditions["timing"]["start_time_s"]
    assert times[-1] == pytest.approx(k1_run.conditions["timing"]["end_time_s"])
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    assert len(samples) == len(k1_run.frames) + 1


def test_repeating_the_run_reproduces_every_frame_byte_for_byte(k1_spec):
    first = K1Experiment(k1_spec).run()
    second = K1Experiment(k1_spec).run()
    assert first.conditions_sha256 == second.conditions_sha256
    assert [frame.to_dict() for frame in first.frames] == [frame.to_dict() for frame in second.frames]
    assert [record for record in first.commands] == [record for record in second.commands]


def test_five_layers_all_report_a_record_for_every_step(k1_run):
    assert [entry["layer"] for entry in k1_run.stage_table] == [1, 2, 3, 4, 5]
    assert len(k1_run.loop_frames) == len(k1_run.frames)
    for loop_frame in k1_run.loop_frames:
        assert set(loop_frame.stage_status) == {f"layer{index}" for index in range(1, 6)}
        assert loop_frame.material.model_id == RIGID_MODEL_ID
        assert loop_frame.material.stiffness_n_per_m is None
        assert loop_frame.target.channels == ("normal_force",)
        assert loop_frame.target.vibration_amplitude_n is None
        assert loop_frame.measurement.model_id == IDEAL_MODEL_ID
        assert loop_frame.measurement.status == MODEL_STATUS


def test_material_and_target_layers_carry_the_solved_force_without_recomputing_it(k1_run):
    for loop_frame in k1_run.loop_frames:
        expected = loop_frame.interaction.normal_force_n or 0.0
        assert loop_frame.material.normal_force_n == pytest.approx(expected)
        assert loop_frame.target.normal_force_n == pytest.approx(expected)
        assert loop_frame.material.indentation_m == 0.0


def test_actuator_measurement_lags_the_target_by_the_declared_latency(k1_run):
    targets = [loop_frame.target.normal_force_n for loop_frame in k1_run.loop_frames]
    measured = [loop_frame.measurement.measured_force_n for loop_frame in k1_run.loop_frames]
    latency = k1_run.loop_frames[0].measurement.latency_s
    dt = k1_run.frames[0].dt_s
    assert latency == pytest.approx(dt)
    assert measured[0] == 0.0
    assert measured[1:] == pytest.approx(targets[:-1])
    assert not any(loop_frame.measurement.watchdog_tripped for loop_frame in k1_run.loop_frames)


def test_force_mode_and_multi_probe_specs_are_refused_until_their_milestones(k1_spec):
    import copy

    from tactile_contract import ExperimentSpec

    data = copy.deepcopy(k1_spec.to_dict())
    data["kind"] = "source_inventory"
    with pytest.raises(ValueError, match="cpu_reference"):
        K1Experiment(ExperimentSpec.build(**data))
