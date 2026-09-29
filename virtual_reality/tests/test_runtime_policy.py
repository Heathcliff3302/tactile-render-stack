"""A declared setting must never look as though it took effect.

Regression tests for the second defect found before the K1 freeze: several
``runtime`` fields were recorded in the specification and carried into the
report while the solver never read them, and ``kinematics_source`` accepted
``finite_difference`` while the kernel hardcoded rigid-body twist.
"""

import copy

import pytest

from contact_kernel import KINEMATICS_METHOD
from mechanics.k1_rigid import K1Experiment
from tactile_contract import ExperimentSpec
from virtual_reality import (
    ENFORCED,
    RECORDED_INERT,
    RUNTIME_POLICY,
    SINGLE_IMPLEMENTATION,
    VERDICTS,
    UnsupportedRuntimeSetting,
    classify_runtime,
    enforce_runtime,
    kernel_config_from_spec,
)


def mutate(spec, change):
    data = copy.deepcopy(spec.to_dict())
    change(data)
    return ExperimentSpec.build(**data)


def with_runtime(spec, name, value):
    return mutate(spec, lambda d: d["runtime"][name].update(value=value))


def test_every_declared_runtime_field_has_a_verdict(k1_spec):
    declared = set(k1_spec.to_dict()["runtime"])
    assert declared == set(RUNTIME_POLICY)
    assert all(policy.verdict in VERDICTS for policy in RUNTIME_POLICY.values())
    assert all(policy.note for policy in RUNTIME_POLICY.values())


def test_an_unclassified_runtime_field_is_rejected(k1_spec):
    data = copy.deepcopy(k1_spec.to_dict())
    data["runtime"]["invented_setting"] = {
        "value": 1, "status": "design_choice", "evidence": "test",
    }
    with pytest.raises(UnsupportedRuntimeSetting, match="no declared kernel policy"):
        classify_runtime(data)


def test_the_frozen_spec_is_accepted_and_fully_classified(k1_spec):
    report = enforce_runtime(k1_spec)
    counts = report["counts"]
    assert sum(counts.values()) == len(RUNTIME_POLICY) + 1  # plus sampling.kinematics_source
    assert counts[ENFORCED] and counts[RECORDED_INERT] and counts[SINGLE_IMPLEMENTATION]
    for name, entry in report["fields"].items():
        assert entry["kernel_verdict"] in VERDICTS, name
        assert entry["note"], name


def test_verdict_is_orthogonal_to_the_declared_status(k1_spec):
    report = classify_runtime(k1_spec)
    # Every K0 value is a design_choice, yet the verdicts differ, so the two
    # fields cannot be collapsed into one.
    statuses = {entry["declared_status"] for entry in report["fields"].values()}
    verdicts = {entry["kernel_verdict"] for entry in report["fields"].values()}
    assert statuses == {"design_choice", "source_explicit"}
    assert len(verdicts) == 3
    assert "orthogonal" in report["note"]


@pytest.mark.parametrize("name,value", [
    ("contact_offset_m", 0.002),
    ("solver_type", "projected_gauss_seidel"),
    ("ccd_mode", "none"),
    ("static_friction", 0.5),
    ("dynamic_friction", 0.4),
    ("restitution", 0.3),
    ("max_depenetration_velocity_mps", 0.5),
])
def test_unimplemented_physics_or_solver_choices_are_rejected(k1_spec, name, value):
    with pytest.raises(UnsupportedRuntimeSetting):
        kernel_config_from_spec(with_runtime(k1_spec, name, value))


def test_depenetration_limit_is_not_claimed_to_be_enforced(k1_spec):
    """It was classified ``enforced`` while never binding.

    With a penetrated contact the target normal velocity is zero, and the
    minimum of zero and any non-negative limit is zero, so no value of the
    field ever changed the impulse, the velocity or the geometric correction.
    Only zero is accepted now, and the kernel guards it directly too.
    """
    from contact_kernel import SolverSpec

    policy = RUNTIME_POLICY["max_depenetration_velocity_mps"]
    assert policy.verdict == SINGLE_IMPLEMENTATION
    assert policy.accepted == 0
    assert "not implemented" in policy.note
    SolverSpec(max_depenetration_velocity_mps=0.0)
    with pytest.raises(NotImplementedError, match="Velocity-based depenetration"):
        SolverSpec(max_depenetration_velocity_mps=0.5)


def test_penetrated_contact_is_corrected_geometrically_not_by_impulse():
    from contact_kernel import (
        GridSpec, KernelConfig, ProbeSpec, RigidContactKernel,
        SolverSpec, SurfaceSpec, TimingSpec,
    )

    probe_id, surface_id, dt = "/World/Fingertip", "/World/Cube", 1.0 / 60.0
    kernel = RigidContactKernel(KernelConfig(
        surface=SurfaceSpec(id=surface_id, center_world_m=(0, 0, 0.25), size_m=(0.5, 0.5, 0.5)),
        probes=(ProbeSpec(id=probe_id, shape="sphere", mass_kg=0.05,
                          initial_position_world_m=(0, 0, 0.61), radius_m=0.12),),
        timing=TimingSpec(physics_dt_s=dt, control_dt_s=dt, output_dt_s=dt, substeps_per_control=1),
        grid=GridSpec(rows=64, cols=64, cell_size_m=0.5 / 64, origin_local_m=(-0.25, -0.25, 0.25)),
        solver=SolverSpec(),
    ))
    pair = kernel.step({probe_id: (0.0, 0.0, 0.0)}, phase="hold").pairs[
        RigidContactKernel.pair_key(probe_id, surface_id)
    ]
    assert pair.position_correction_m == pytest.approx(0.01)
    assert pair.normal_impulse_ns == 0.0
    assert pair.probe_velocity_world_mps[2] == 0.0


@pytest.mark.parametrize("name,value", [
    ("position_iterations", 4),
    ("velocity_iterations", 8),
    ("sleep_enabled", True),
    ("angular_damping_per_s", 0.1),
    ("initial_orientation_wxyz", [0.0, 1.0, 0.0, 0.0]),
])
def test_inert_fields_only_accept_the_value_that_is_genuinely_inert(k1_spec, name, value):
    # "Recorded but not read" is only honest if the recorded value is verified
    # to be a no-op. A non-inert value is rejected rather than ignored.
    with pytest.raises(UnsupportedRuntimeSetting):
        kernel_config_from_spec(with_runtime(k1_spec, name, value))


def test_enforced_fields_really_change_behaviour(k1_spec):
    baseline = K1Experiment(k1_spec).run()
    lifted = K1Experiment(with_runtime(k1_spec, "rest_offset_m", 0.01)).run()
    damped = K1Experiment(with_runtime(k1_spec, "linear_damping_per_s", 2.0)).run()
    first = next(f for f in baseline.frames if f.contact_present)
    first_lifted = next(f for f in lifted.frames if f.contact_present)
    # A lifted contact plane moves the contact height and therefore the frame
    # at which contact begins.
    assert first_lifted.contact_centroid_world_m[2] == pytest.approx(
        first.contact_centroid_world_m[2] + 0.01
    )
    assert first_lifted.sequence_id != first.sequence_id
    assert damped.conditions_sha256 != baseline.conditions_sha256


def test_combine_modes_are_inert_only_because_friction_is_zero(k1_spec):
    # They accept any value today. The justification is that both frictions and
    # restitution are pinned to zero, which the guards above enforce.
    for mode in ("average", "min", "multiply", "max"):
        config = kernel_config_from_spec(with_runtime(k1_spec, "friction_combine_mode", mode))
        assert config.solver.static_friction == 0
        assert config.solver.dynamic_friction == 0
    assert RUNTIME_POLICY["friction_combine_mode"].verdict == RECORDED_INERT
    assert "zero" in RUNTIME_POLICY["friction_combine_mode"].note


def test_unknown_kinematics_source_is_rejected(k1_spec):
    spec = mutate(k1_spec, lambda d: d["sampling"].update(kinematics_source="optical_flow"))
    with pytest.raises(UnsupportedRuntimeSetting, match="kinematics_source"):
        kernel_config_from_spec(spec)


def test_both_declared_kinematics_sources_are_implemented(k1_spec):
    assert set(KINEMATICS_METHOD) == {"rigid_body", "finite_difference"}
    for source in KINEMATICS_METHOD:
        spec = mutate(k1_spec, lambda d, s=source: d["sampling"].update(kinematics_source=s))
        config = kernel_config_from_spec(spec)
        assert config.sampling.kinematics_source == source
