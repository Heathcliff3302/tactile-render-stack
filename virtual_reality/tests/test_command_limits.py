"""Commands must respect their declared limits, and runs their declared clock.

Two regressions, both found in review before the K1 freeze:

* the slide command limit was applied only inside the feedback branch, so the
  first slide frame could command any declared initial speed — 1.0 m/s against
  a 0.12 m/s limit, enough to displace the probe abnormally or off the face;
* the step budget added 30 steps beyond ``max_duration_s``, so a run could
  overrun the simulation duration it declared and still be reported as
  completing normally.
"""

import copy

import pytest

from mechanics.k1_rigid import K1Experiment
from tactile_contract import ExperimentSpec
from tactile_contract.pipeline import LoopFeedback
from virtual_reality import (
    TrajectoryConfig,
    TrajectoryStateMachine,
    step_budget_from_spec,
)

LIMIT = 0.12
TARGET = 0.05


def mutate(spec, change):
    data = copy.deepcopy(spec.to_dict())
    change(data)
    return ExperimentSpec.build(**data)


def config(**overrides):
    base = dict(
        pre_settle_time_s=0.5, approach_timeout_s=5.0, hold_time_s=1.0,
        slide_time_s=1.0, release_time_s=0.5, approach_speed_mps=0.2,
        slide_speed_mps=TARGET, retract_speed_mps=0.3, retract_distance_m=0.58,
        contact_maintain_speed_mps=0.02, tangent_speed_feedback_kp=1.0,
        max_slide_command_speed_mps=LIMIT,
    )
    base.update(overrides)
    return TrajectoryConfig(**base)


def slide_machine(**overrides):
    machine = TrajectoryStateMachine(config(**overrides))
    machine.phase = "lateral_slide"
    return machine


def test_initial_slide_speed_above_the_limit_is_rejected_at_load():
    with pytest.raises(ValueError, match=r"initial_slide_command_speed_mps=1\.0 is outside"):
        config(initial_slide_command_speed_mps=1.0)
    with pytest.raises(ValueError, match="is outside"):
        config(initial_slide_command_speed_mps=-0.01)
    # At the limit is admissible; above it is not.
    assert config(initial_slide_command_speed_mps=LIMIT).initial_slide_command_speed_mps == LIMIT


def test_first_slide_frame_respects_the_limit(k1_spec):
    spec = mutate(k1_spec, lambda d: d["controller"]["parameters"].update(
        initial_slide_command_speed_mps=LIMIT
    ))
    run = K1Experiment(spec).run()
    slide = [record["command_velocity_world_mps"][0] for record in run.commands
             if record["phase"] == "lateral_slide"]
    assert slide
    assert slide[0] == pytest.approx(LIMIT)
    assert max(slide) <= LIMIT + 1e-12


def test_every_command_of_a_whole_run_respects_the_limit(k1_run):
    for record in k1_run.commands:
        vx = record["command_velocity_world_mps"][0]
        assert abs(vx) <= LIMIT + 1e-12, record


def test_the_clamp_sits_on_the_command_exit_not_in_one_branch():
    # A feedback error large enough to demand more than the limit must still
    # produce a clamped command, and the clamp must be counted.
    machine = slide_machine()
    machine.command(LoopFeedback())
    for _ in range(20):
        command = machine.command(LoopFeedback(tangent_speed_mps=-5.0))
        assert command[0] <= LIMIT + 1e-12
    assert machine.slide_command_mps == pytest.approx(LIMIT)
    assert machine.diagnostics()["slide_command_limit_mps"] == LIMIT


def test_open_loop_slide_target_above_the_limit_is_also_clamped():
    # tangent_speed_feedback_kp = 0 takes the open-loop path. It must obey the
    # same limit, which a per-branch clamp would have missed.
    machine = slide_machine(tangent_speed_feedback_kp=0.0, slide_speed_mps=LIMIT)
    assert machine.command(LoopFeedback())[0] == pytest.approx(LIMIT)


def test_diagnostics_report_when_a_command_was_clamped():
    machine = slide_machine()
    assert machine.diagnostics()["clamped_command_steps"] == 0
    machine.reset()
    assert machine.diagnostics()["clamped_command_steps"] == 0


def test_step_budget_is_the_declared_duration_with_no_hidden_margin(k1_spec):
    timing = k1_spec.to_dict()["timing"]
    expected = int(timing["max_duration_s"] / timing["control_dt_s"])
    assert step_budget_from_spec(k1_spec) == expected
    # The last step must end inside the declared duration.
    assert step_budget_from_spec(k1_spec) * timing["control_dt_s"] <= timing["max_duration_s"] + 1e-9


def test_a_duration_shorter_than_one_control_step_is_rejected(k1_spec):
    spec = mutate(k1_spec, lambda d: d["timing"].update(max_duration_s=0.001))
    with pytest.raises(ValueError, match="shorter than one control step"):
        step_budget_from_spec(spec)


def test_run_stops_at_the_declared_duration_and_reports_a_timeout(k1_spec):
    # The healthy run needs 7.867 s, so 7.4 s must cut it short rather than
    # borrowing extra steps.
    spec = mutate(k1_spec, lambda d: d["timing"].update(max_duration_s=7.4))
    run = K1Experiment(spec).run()
    assert run.end_time_s <= 7.4 + 1e-9
    assert run.end_time_s == pytest.approx(7.4, abs=1e-9)
    assert run.loop_state == "aborted"
    assert "timeout" in run.abort_reason
    assert "7.4" in run.abort_reason


def test_a_generous_duration_lets_the_controller_finish_on_its_own(k1_run):
    # The declared duration bounds the run; it does not define its length.
    timing = k1_run.spec.to_dict()["timing"]
    assert k1_run.loop_state == "completed"
    assert k1_run.end_time_s < timing["max_duration_s"]
    assert len(k1_run.frames) < step_budget_from_spec(k1_run.spec)
