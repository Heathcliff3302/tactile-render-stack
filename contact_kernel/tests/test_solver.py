"""Rigid impulse solver: contact boundaries, conservation and step refinement."""

import pytest

from contact_kernel import (
    GRAVITY_COMPENSATION_NONE,
    DriveSpec,
    GridSpec,
    KernelConfig,
    ProbeSpec,
    RigidContactKernel,
    SamplingSpec,
    SolverSpec,
    SurfaceSpec,
    TimingSpec,
)

PROBE_ID = "/World/Fingertip"
SURFACE_ID = "/World/Cube"
MASS = 0.05
RADIUS = 0.12
DT = 1.0 / 60.0
PAIR = RigidContactKernel.pair_key(PROBE_ID, SURFACE_ID)


def make_kernel(*, substeps=1, start_z=1.2, start_x=0.0, compensation=None):
    drive = DriveSpec() if compensation is None else DriveSpec(gravity_compensation_mode=compensation)
    config = KernelConfig(
        surface=SurfaceSpec(id=SURFACE_ID, center_world_m=(0, 0, 0.25), size_m=(0.5, 0.5, 0.5)),
        probes=(ProbeSpec(id=PROBE_ID, shape="sphere", mass_kg=MASS,
                          initial_position_world_m=(start_x, 0, start_z), radius_m=RADIUS),),
        timing=TimingSpec(physics_dt_s=DT / substeps, control_dt_s=DT, output_dt_s=DT,
                          substeps_per_control=substeps),
        grid=GridSpec(rows=64, cols=64, cell_size_m=0.5 / 64, origin_local_m=(-0.25, -0.25, 0.25)),
        solver=SolverSpec(),
        sampling=SamplingSpec(),
        drive=drive,
    )
    return RigidContactKernel(config)


def step(kernel, velocity, phase="hold"):
    return kernel.step({PROBE_ID: velocity}, phase=phase).pairs[PAIR]


def test_free_probe_holds_height_when_gravity_is_compensated():
    kernel = make_kernel()
    pair = step(kernel, (0.0, 0.0, 0.0), phase="pre_contact")
    assert pair.probe_position_world_m[2] == pytest.approx(1.2)
    assert pair.contact_present is False
    assert pair.normal_impulse_ns == 0.0


def test_uncompensated_probe_falls_under_the_declared_gravity():
    kernel = make_kernel(compensation=GRAVITY_COMPENSATION_NONE)
    pair = step(kernel, (0.0, 0.0, 0.0), phase="pre_contact")
    assert pair.probe_position_world_m[2] == pytest.approx(1.2 - 9.81 * DT * DT)


def test_first_contact_impulse_stops_the_commanded_normal_velocity():
    kernel = make_kernel(start_z=0.62)
    pair = step(kernel, (0.0, 0.0, -0.2), phase="approach")
    assert pair.contact_present is True
    assert pair.normal_impulse_ns == pytest.approx(MASS * 0.2)
    assert pair.normal_force_n == pytest.approx(MASS * 0.2 / DT)
    assert pair.probe_velocity_world_mps[2] == pytest.approx(0.0)
    assert pair.probe_position_world_m[2] == pytest.approx(0.62)


def test_approach_stops_exactly_at_the_contact_height_without_overshoot():
    kernel = make_kernel(start_z=0.6205)
    pair = step(kernel, (0.0, 0.0, -0.2), phase="approach")
    # Only the part of the command that would penetrate is cancelled.
    assert pair.probe_position_world_m[2] == pytest.approx(0.62)
    assert pair.normal_impulse_ns == pytest.approx(MASS * (0.2 - 0.0005 / DT))
    assert pair.penetration_m == 0.0
    assert pair.position_correction_m == 0.0


def test_hold_force_follows_the_declared_drive_semantics():
    kernel = make_kernel(start_z=0.62)
    step(kernel, (0.0, 0.0, -0.2), phase="approach")
    for _ in range(5):
        pair = step(kernel, (0.0, 0.0, -0.02), phase="hold")
    # Overwriting the velocity once per control step injects m*v of momentum,
    # so the frame force is m * v / output_dt, not the probe weight.
    assert pair.normal_force_n == pytest.approx(MASS * 0.02 / DT)
    assert pair.gap_m == pytest.approx(0.0, abs=1e-15)


def test_sliding_keeps_the_tangential_command_with_zero_friction():
    kernel = make_kernel(start_z=0.62)
    step(kernel, (0.0, 0.0, -0.2), phase="approach")
    pair = step(kernel, (0.05, 0.0, -0.02), phase="lateral_slide")
    assert pair.tangent_speed_mps == pytest.approx(0.05)
    assert pair.tangent_force_world_n == (0.0, 0.0, 0.0)
    assert pair.total_force_world_n[2] == pytest.approx(pair.normal_force_n)


def test_retract_clears_contact_force_and_impulse():
    kernel = make_kernel(start_z=0.62)
    step(kernel, (0.0, 0.0, -0.2), phase="approach")
    pair = step(kernel, (0.0, 0.0, 0.3), phase="retract")
    assert pair.contact_present is False
    assert pair.normal_impulse_ns == 0.0
    assert pair.normal_force_n == 0.0
    assert pair.total_force_world_n == (0.0, 0.0, 0.0)
    assert pair.points == ()
    assert pair.cells == {}


def test_contact_below_the_sampling_threshold_is_recorded_as_filtered():
    kernel = make_kernel(start_z=0.62)
    # m*v/dt = 0.03 N, under the 0.05 N declared sampling threshold.
    pair = step(kernel, (0.0, 0.0, -0.01), phase="approach")
    assert pair.solver_contact is True
    assert pair.contact_present is False
    assert pair.filtered_reason == "normal_force_below_sampling_threshold"
    assert pair.conservation.solver_impulse_ns > 0.0


def test_support_point_beyond_the_face_never_reports_contact():
    kernel = make_kernel(start_z=0.62, start_x=0.30)
    pair = step(kernel, (0.0, 0.0, -0.2), phase="approach")
    assert pair.inside_face is False
    assert pair.contact_present is False
    assert pair.normal_impulse_ns == 0.0
    assert pair.filtered_reason == "support_point_outside_top_face"


def test_probe_that_slides_off_the_face_mid_interval_is_flagged():
    kernel = make_kernel(start_z=0.62, start_x=0.24)
    pair = step(kernel, (2.0, 0.0, -0.2), phase="lateral_slide")
    assert pair.solution.contact_substeps == 1
    assert pair.inside_face is False
    assert pair.contact_present is False
    assert pair.filtered_reason == "support_point_left_top_face_during_interval"


def test_penetration_correction_is_recorded_separately_from_impulse():
    kernel = make_kernel(start_z=0.61)
    pair = step(kernel, (0.0, 0.0, 0.0), phase="hold")
    # A probe that starts sunk is lifted geometrically; the depenetration limit
    # is zero, so no extra contact impulse is invented to do it.
    assert pair.position_correction_m == pytest.approx(0.01)
    assert pair.normal_impulse_ns == 0.0
    assert pair.probe_position_world_m[2] == pytest.approx(0.62)


@pytest.mark.parametrize("substeps", (2, 4, 8))
def test_substep_refinement_preserves_trajectory_and_impulse(substeps):
    coarse, fine = make_kernel(), make_kernel(substeps=substeps)
    commands = [(0.0, 0.0, -0.2)] * 180 + [(0.0, 0.0, -0.02)] * 10 + \
               [(0.05, 0.0, -0.02)] * 10 + [(0.0, 0.0, 0.3)] * 5
    for command in commands:
        a, b = step(coarse, command), step(fine, command)
        assert a.contact_present == b.contact_present
        assert a.normal_force_n == pytest.approx(b.normal_force_n, abs=1e-12)
        assert a.probe_position_world_m[2] == pytest.approx(b.probe_position_world_m[2], abs=1e-12)
        assert sorted(a.cells) == sorted(b.cells)


def test_substep_traces_cover_the_whole_control_interval():
    kernel = make_kernel(substeps=4)
    pair = step(kernel, (0.0, 0.0, -0.2), phase="approach")
    traces = pair.solution.substeps
    assert len(traces) == 4
    assert [trace.index for trace in traces] == [0, 1, 2, 3]
    assert traces[-1].time_s == pytest.approx(DT)
    assert sum(trace.physics_dt_s for trace in traces) == pytest.approx(DT)


def test_unsupported_probe_shapes_fail_loudly():
    with pytest.raises(NotImplementedError, match="K3"):
        ProbeSpec(id="/World/Box", shape="box", mass_kg=0.08,
                  initial_position_world_m=(0, 0, 1.2), size_m=(0.1, 0.1, 0.05))
    with pytest.raises(ValueError):
        ProbeSpec(id="/World/Odd", shape="capsule", mass_kg=0.08,
                  initial_position_world_m=(0, 0, 1.2), radius_m=0.1)


def test_substeps_must_cover_the_control_interval_exactly():
    with pytest.raises(ValueError, match="cover one control interval"):
        TimingSpec(physics_dt_s=DT, control_dt_s=DT, output_dt_s=DT, substeps_per_control=2)


def test_missing_command_is_an_error_not_a_silent_zero():
    kernel = make_kernel()
    with pytest.raises(KeyError):
        kernel.step({}, phase="hold")
