"""The declared kinematics source must govern the data, not just the spec.

Phase 1's ``step6_3.py`` defaults to ``finite_difference`` and the regression
script overrides it to ``rigid_body``, so a parity comparison will use both.
Before the freeze the kernel accepted ``finite_difference`` and then recorded
``rigid_body_twist`` anyway; these tests pin the fix.
"""

import copy

import pytest

from contact_kernel import KINEMATICS_METHOD
from interaction import (
    KINEMATICS_DEPENDENT,
    always_available,
    manifest_observables,
    observable_methods,
)
from mechanics.k1_rigid import K1Experiment
from tactile_contract import ExperimentSpec
from tactile_contract.io import load_run
from validation.k1_acceptance import acceptance_report, evaluate_replay


@pytest.fixture(scope="module")
def fd_run(k1_spec):
    data = copy.deepcopy(k1_spec.to_dict())
    data["sampling"]["kinematics_source"] = "finite_difference"
    return K1Experiment(ExperimentSpec.build(**data)).run()


def test_method_table_follows_the_declared_source():
    rigid = observable_methods("rigid_body")
    differenced = observable_methods("finite_difference")
    for name in KINEMATICS_DEPENDENT:
        assert rigid[name] == "rigid_body_twist"
        assert differenced[name] == "contact_point_finite_difference"
    # Nothing else moves with the source.
    assert {k: v for k, v in rigid.items() if k not in KINEMATICS_DEPENDENT} == \
           {k: v for k, v in differenced.items() if k not in KINEMATICS_DEPENDENT}
    with pytest.raises(ValueError, match="Unknown kinematics source"):
        observable_methods("optical_flow")


def test_manifest_and_frames_read_the_same_table():
    for source in KINEMATICS_METHOD:
        definitions = manifest_observables(source)
        methods = observable_methods(source)
        for name, definition in definitions.items():
            assert definition["method"] == methods[name]


def test_differenced_run_records_its_own_source_everywhere(fd_run):
    contact = [frame for frame in fd_run.frames if frame.contact_present]
    assert contact
    for frame in contact:
        data = frame.to_dict()
        assert data["metadata"]["kinematics_source"] == "finite_difference"
        assert data["metadata"]["kinematics_method"] == "contact_point_finite_difference"
        for name in KINEMATICS_DEPENDENT:
            if data[name] is not None:
                assert data["quality"][name]["method"] == "contact_point_finite_difference"
    points = [
        point
        for loop_frame in fd_run.loop_frames
        for pair in loop_frame.world.kernel_frame.pairs.values()
        for point in pair.points
    ]
    assert {point.kinematics_source for point in points} == {"contact_point_finite_difference"}


def test_differenced_slide_speed_matches_the_command_and_keeps_the_cross_check(fd_run):
    slide = [
        frame.to_dict() for frame in fd_run.frames
        if frame.metadata["motion_phase"] == "lateral_slide"
        and frame.tangent_velocity_world_mps is not None
    ]
    assert slide
    for data in slide:
        assert data["tangent_velocity_world_mps"][0] == pytest.approx(0.05, abs=1e-12)
        # The other source stays available as a cross-check rather than being
        # discarded, which is what a parity comparison needs.
        assert data["metadata"]["rigid_body_tangent_velocity_mps"][0] == pytest.approx(0.05)


def test_a_frame_without_a_predecessor_reports_no_differenced_velocity(fd_run):
    contact = [frame.to_dict() for frame in fd_run.frames if frame.contact_present]
    first = contact[0]
    assert first["metadata"]["kinematics_available"] is False
    for name in KINEMATICS_DEPENDENT:
        assert first[name] is None
        assert first["quality"][name]["status"] == "unavailable"


def test_no_contact_frames_report_differenced_kinematics_as_unavailable(fd_run):
    absent = [frame.to_dict() for frame in fd_run.frames if not frame.contact_present]
    assert absent
    for data in absent:
        for name in KINEMATICS_DEPENDENT:
            assert data[name] is None
            assert data["quality"][name]["status"] == "unavailable"


def test_required_observables_shrink_for_the_differenced_source():
    rigid = always_available("rigid_body")
    differenced = always_available("finite_difference")
    for name in KINEMATICS_DEPENDENT:
        assert name in rigid
        assert name not in differenced
    assert set(differenced) < set(rigid)


def test_differenced_run_passes_every_gate_and_replays(fd_run, k1_spec, tmp_path):
    from mechanics.k1_report import write_run

    data = copy.deepcopy(k1_spec.to_dict())
    data["sampling"]["kinematics_source"] = "finite_difference"
    experiment = K1Experiment(ExperimentSpec.build(**data))
    report = acceptance_report(fd_run)
    assert report["all_passed"] is True
    assert report["kinematics_source"] == "finite_difference"

    write_run(fd_run, tmp_path, report, experiment.effective_runtime())
    assert evaluate_replay(tmp_path)["passed"] is True
    manifest, frames = load_run(tmp_path)
    definitions = manifest.to_dict()["observables"]
    for name in KINEMATICS_DEPENDENT:
        assert definitions[name]["method"] == "contact_point_finite_difference"
    assert len(frames) == len(fd_run.frames)


def test_adapter_refuses_a_kernel_whose_source_disagrees(k1_spec):
    from interaction import KernelInteractionStage

    run = K1Experiment(k1_spec).run()
    stage = KernelInteractionStage(
        run_id="x", scenario_id="y", conditions_sha256="0" * 64,
        backend_id="rigid_cpu_v1",
        pair_key=list(run.loop_frames[0].world.kernel_frame.pairs)[0],
        kinematics_source="finite_difference",
    )
    with pytest.raises(ValueError, match="adapter declares"):
        stage.step(run.loop_frames[0].world)
