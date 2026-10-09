"""The force-mode segment machine and its PI regulator.

Same six segments as Step 6 and the same lock-and-hold transition rule. What
differs is the drive: here the frame force is the momentum one commanded
velocity injects, so the regulator's authority is bounded by
``ki * force_integral_limit`` rather than by how hard it can push into a
penetrating contact. See ``test_regulator_authority.py`` for that derivation.
"""

import pytest

from tactile_contract.pipeline import LoopFeedback
from virtual_reality import (
    FORCE_PHASE_ORDER,
    PHASE_ROLES,
    ForceConfig,
    ForceFeedbackStateMachine,
    controller_from_spec,
    force_config_from_spec,
    phase_with_role,
    phases_for_mode,
)

DT = 1.0 / 60.0
MASS = 0.05


def config(**overrides) -> ForceConfig:
    base = dict(
        pre_settle_time_s=0.5,
        approach_timeout_s=5.0,
        regulate_timeout_s=20.0,
        hold_time_s=1.0,
        slide_time_s=1.0,
        release_time_s=0.5,
        approach_speed_mps=0.2,
        slide_speed_mps=0.05,
        retract_speed_mps=0.3,
        retract_distance_m=0.58,
        target_force_n=1.0,
        force_tolerance_n=0.05,
        force_kp_m_per_ns=0.2,
        force_ki_m_per_ns2=0.25,
        force_integral_limit_ns=2.0,
        max_press_speed_mps=0.5,
        max_correct_speed_mps=0.2,
        settle_frames=5,
        control_dt_s=DT,
        max_slide_command_speed_mps=0.12,
    )
    base.update(overrides)
    return ForceConfig(**base)


def feedback(force_n=0.0, contact=True) -> LoopFeedback:
    return LoopFeedback(contact_present=contact, normal_force_n=force_n)


def machine(**overrides) -> ForceFeedbackStateMachine:
    return ForceFeedbackStateMachine(config(**overrides))


def drive_to(controller, phase, *, force_n=1.0):
    """Advance the controller to ``phase`` with a cooperative measurement."""
    time_s = 0.0
    for _ in range(4000):
        if controller.phase == phase:
            return time_s
        controller.command(feedback(force_n))
        time_s += DT
        controller.observe(time_s, True, force_n)
    raise AssertionError(f"never reached {phase}")


# --- segment structure -----------------------------------------------------

def test_force_mode_declares_step_6_segments():
    assert FORCE_PHASE_ORDER == (
        "pre_contact", "approach_force", "regulate_force",
        "lateral_slide_force", "retract", "released",
    )
    assert phases_for_mode("force") == FORCE_PHASE_ORDER
    assert phase_with_role("force", "hold") == "regulate_force"
    assert phase_with_role("force", "slide") == "lateral_slide_force"
    # The two modes name the same roles differently, which is exactly why a
    # gate must look a role up instead of hardcoding a segment name.
    assert phase_with_role("trajectory", "hold") == "hold"
    assert PHASE_ROLES["regulate_force"] == PHASE_ROLES["hold"]
    assert PHASE_ROLES["lateral_slide_force"] == PHASE_ROLES["lateral_slide"]


def test_every_declared_segment_has_a_role():
    for mode in ("trajectory", "force"):
        for phase in phases_for_mode(mode):
            assert phase in PHASE_ROLES


def test_an_unimplemented_mode_is_refused():
    with pytest.raises(ValueError, match="multi_trajectory"):
        phases_for_mode("multi_trajectory")


def test_the_spec_selects_the_controller(k1_spec, k2_spec):
    from virtual_reality import TrajectoryStateMachine

    assert isinstance(controller_from_spec(k1_spec), TrajectoryStateMachine)
    forced = controller_from_spec(k2_spec)
    assert isinstance(forced, ForceFeedbackStateMachine)
    assert forced.mode == "force"


def test_force_config_reads_only_declared_values(k2_spec):
    built = force_config_from_spec(k2_spec)
    declared = k2_spec.to_dict()["controller"]["force"]
    assert built.target_force_n == declared["target_force_n"]
    assert built.force_ki_m_per_ns2 == declared["force_ki_m_per_ns2"]
    assert built.force_integral_limit_ns == declared["force_integral_limit_ns"]
    assert built.regulate_timeout_s == declared["regulate_timeout_s"]
    assert built.control_dt_s == k2_spec.to_dict()["timing"]["control_dt_s"]


def test_a_trajectory_spec_is_refused_by_the_force_adapter(k1_spec):
    with pytest.raises(ValueError, match="mode == 'force'"):
        force_config_from_spec(k1_spec)


# --- the PI law ------------------------------------------------------------

def test_no_contact_falls_back_to_approaching_and_drops_the_integral():
    controller = machine()
    drive_to(controller, "regulate_force")
    controller.command(feedback(0.5))
    assert controller.diagnostics()["force_integral_ns"] != 0.0
    command = controller.command(feedback(contact=False))
    assert command[2] == pytest.approx(-0.2)
    assert controller.diagnostics()["force_integral_ns"] == 0.0


def test_the_integral_clamps_at_its_declared_limit():
    controller = machine(force_integral_limit_ns=0.5)
    drive_to(controller, "regulate_force", force_n=0.0)
    for _ in range(400):
        controller.command(feedback(0.0))
    diagnostics = controller.diagnostics()
    assert diagnostics["force_integral_ns"] == pytest.approx(0.5)
    assert diagnostics["saturated_integral_steps"] > 0


def test_the_command_is_clamped_to_the_declared_drive_limits():
    controller = machine(max_press_speed_mps=0.3, max_correct_speed_mps=0.1)
    drive_to(controller, "regulate_force", force_n=0.0)
    for _ in range(400):
        pressing = controller.command(feedback(0.0))
    assert pressing[2] == pytest.approx(-0.3)
    # Far above target, the regulator backs off and is bounded the other way.
    for _ in range(400):
        correcting = controller.command(feedback(50.0))
    assert correcting[2] == pytest.approx(0.1)


def test_the_regulator_reaches_the_target_against_the_kernel_force_law():
    """Closed loop against F = m * v / dt, which is what the kernel delivers."""
    controller = machine()
    drive_to(controller, "regulate_force", force_n=0.0)
    force = 0.0
    history = []
    for _ in range(1200):
        command = controller.command(feedback(force))
        force = max(0.0, -command[2]) * MASS / DT
        history.append(force)
    # Inside the declared band is the requirement. The approach is asymptotic,
    # so a tighter bound than the tolerance would be asserting something the
    # specification never asked for.
    assert abs(1.0 - history[-1]) <= 0.05
    assert history[-1] == pytest.approx(1.0, abs=1e-3)
    settled = next(i for i in range(len(history) - 5)
                   if all(abs(1.0 - f) <= 0.05 for f in history[i:i + 5]))
    assert settled * DT < 10.0
    # Monotone approach from below, with no overshoot past the target.
    assert max(history) <= 1.0 + 1e-12


def test_the_inherited_gains_never_reach_the_band():
    # The same loop with Phase 1's ki. This is the run the authority guard
    # prevents from ever starting.
    controller = machine(force_ki_m_per_ns2=0.1)
    drive_to(controller, "regulate_force", force_n=0.0)
    force = 0.0
    for _ in range(2000):
        command = controller.command(feedback(force))
        force = max(0.0, -command[2]) * MASS / DT
    assert force == pytest.approx(0.75, abs=1e-6)
    assert abs(1.0 - force) > 0.05


# --- lock and hold ---------------------------------------------------------

def test_the_hold_clock_starts_only_after_settle_frames_in_band():
    controller = machine(settle_frames=5)
    time_s = drive_to(controller, "regulate_force")
    for index in range(4):
        time_s += DT
        controller.observe(time_s, True, 1.0)
        assert controller.diagnostics()["within_tolerance_frames"] == index + 1
        assert controller.diagnostics()["target_locked"] is False
    time_s += DT
    controller.observe(time_s, True, 1.0)
    assert controller.diagnostics()["target_locked"] is True
    assert controller.force_target_reached is True
    assert controller.settled_at_time_s == pytest.approx(time_s)


def test_leaving_the_band_restarts_the_settle_and_hold_clocks():
    controller = machine(settle_frames=3)
    time_s = drive_to(controller, "regulate_force")
    for _ in range(3):
        time_s += DT
        controller.observe(time_s, True, 1.0)
    assert controller.diagnostics()["target_locked"] is True
    time_s += DT
    controller.observe(time_s, True, 0.2)
    diagnostics = controller.diagnostics()
    assert diagnostics["target_locked"] is False
    assert diagnostics["within_tolerance_frames"] == 0
    assert diagnostics["hold_elapsed_s"] == 0.0
    # force_target_reached records that it happened once; it is history, not
    # current state, and the two must not be confused.
    assert controller.force_target_reached is True


def test_losing_contact_clears_the_lock_and_the_integral():
    controller = machine(settle_frames=3)
    time_s = drive_to(controller, "regulate_force")
    for _ in range(3):
        time_s += DT
        controller.observe(time_s, True, 1.0)
    assert controller.diagnostics()["target_locked"] is True
    time_s += DT
    controller.observe(time_s, False, 0.0)
    diagnostics = controller.diagnostics()
    assert diagnostics["target_locked"] is False
    assert diagnostics["force_integral_ns"] == 0.0


def test_regulation_times_out_when_the_target_is_never_held():
    controller = machine(regulate_timeout_s=1.0)
    time_s = drive_to(controller, "regulate_force")
    for _ in range(200):
        controller.command(feedback(0.1))
        time_s += DT
        controller.observe(time_s, True, 0.1)
        if controller.done:
            break
    assert controller.done is True
    assert "force regulation timeout" in controller.abort_reason
    assert controller.phase == "regulate_force"


def test_the_approach_still_times_out_without_contact():
    controller = machine(approach_timeout_s=0.5)
    time_s = 0.0
    for _ in range(200):
        controller.command(feedback(contact=False))
        time_s += DT
        controller.observe(time_s, False, 0.0)
        if controller.done:
            break
    assert controller.done is True
    assert "approach timeout" in controller.abort_reason


# --- sliding ---------------------------------------------------------------

def test_sliding_commands_the_declared_tangential_speed_while_regulating():
    controller = machine()
    drive_to(controller, "lateral_slide_force")
    # On target with no accumulated integral the regulator has nothing to
    # correct, so the vertical command is zero — it is still the PI output,
    # not a fixed press speed, which the error response below shows.
    on_target = controller.command(feedback(1.0))
    assert on_target[0] == pytest.approx(0.05)
    assert on_target[2] == pytest.approx(0.0)
    below = controller.command(feedback(0.5))
    assert below[0] == pytest.approx(0.05)
    assert below[2] < 0.0


def test_the_tangential_command_never_exceeds_the_declared_limit():
    # In force mode the tangential command is the declared slide speed, and
    # the constructor already refuses a target above its own limit, so the
    # exit clamp cannot fire here. The invariant is what matters, and it is
    # asserted directly rather than through a diagnostic that cannot move.
    controller = machine(slide_speed_mps=0.12, max_slide_command_speed_mps=0.12)
    drive_to(controller, "lateral_slide_force")
    for force_n in (0.0, 0.5, 1.0, 5.0):
        command = controller.command(feedback(force_n))
        assert abs(command[0]) <= 0.12
    assert controller.diagnostics()["clamped_command_steps"] == 0


def test_a_slide_target_above_its_own_limit_is_refused_at_construction():
    with pytest.raises(ValueError, match="Slide command limit"):
        config(slide_speed_mps=0.2, max_slide_command_speed_mps=0.1)


def test_a_non_positive_integral_limit_is_refused():
    with pytest.raises(ValueError, match="no authority"):
        config(force_integral_limit_ns=0.0)


def test_reset_restores_the_initial_state():
    controller = machine()
    drive_to(controller, "lateral_slide_force")
    controller.reset()
    diagnostics = controller.diagnostics()
    assert controller.phase == "pre_contact"
    assert controller.force_target_reached is False
    assert diagnostics["force_integral_ns"] == 0.0
    assert diagnostics["clamped_command_steps"] == 0
    assert controller.transitions == ()
