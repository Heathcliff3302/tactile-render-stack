"""A declared force target must be reachable within its own declared limits.

Found while assessing whether K2 could start: the Phase 1 force parameters
declare a 1 N target with a 0.25 m/s press limit, which reaches only 0.75 N on
this kernel. Phase 1 got to 1 N because PhysX reports contact impulse that
also carries penetration recovery; this kernel removes penetration by
geometric projection and records it apart from the impulse, so that
contribution does not exist here.

Without this guard the run would simply fail its force gate, with nothing to
say the specification was inconsistent rather than the solver broken.
"""

import copy
import json
from pathlib import Path

import pytest

from tactile_contract import ExperimentSpec
from virtual_reality import (
    DERIVED_HEADROOM_FACTOR,
    UnreachableForceTarget,
    assess_force_reachability,
    command_speed_for_force_mps,
    derive_press_limit_mps,
    enforce_force_reachability,
    frame_force_n,
    kernel_config_from_spec,
)

K0 = Path(__file__).resolve().parents[2] / "scenarios" / "k0"
MASS = 0.05
DT = 1.0 / 60.0


def load(name):
    return ExperimentSpec.from_dict(json.loads((K0 / f"{name}.json").read_text(encoding="utf-8")))


@pytest.fixture(scope="module")
def k2_spec():
    return load("k2_force_cpu")


def mutate(spec, change):
    data = copy.deepcopy(spec.to_dict())
    change(data)
    return ExperimentSpec.build(**data)


def test_force_law_round_trips():
    for force in (0.06, 0.75, 1.0, 1.5):
        speed = command_speed_for_force_mps(mass_kg=MASS, force_n=force, output_dt_s=DT)
        assert frame_force_n(mass_kg=MASS, speed_mps=speed, output_dt_s=DT) == pytest.approx(force)
    # The numbers this whole finding turns on.
    assert command_speed_for_force_mps(mass_kg=MASS, force_n=1.0, output_dt_s=DT) == pytest.approx(1 / 3)
    assert frame_force_n(mass_kg=MASS, speed_mps=0.25, output_dt_s=DT) == pytest.approx(0.75)


def test_derived_press_limit_admits_the_declared_headroom():
    limit = derive_press_limit_mps(mass_kg=MASS, target_force_n=1.0, output_dt_s=DT)
    assert limit == pytest.approx(0.5)
    assert frame_force_n(mass_kg=MASS, speed_mps=limit, output_dt_s=DT) == pytest.approx(
        DERIVED_HEADROOM_FACTOR * 1.0
    )
    with pytest.raises(ValueError, match="cannot reach the target"):
        derive_press_limit_mps(mass_kg=MASS, target_force_n=1.0, output_dt_s=DT, headroom=0.9)


def test_phase1_force_parameters_are_unreachable_on_this_kernel():
    # The source inventory is a faithful record of Phase 1 and is not edited;
    # the assessment simply reports that those numbers do not carry over.
    report = assess_force_reachability(load("step6_force_64"))
    assert report["declared_press_limit_mps"] == 0.25
    assert report["max_reachable_force_n"] == pytest.approx(0.75)
    assert report["target_reachable"] is False
    assert report["tolerance_band_reachable"] is False


def test_k2_reference_reaches_its_target_with_headroom(k2_spec):
    report = enforce_force_reachability(k2_spec)
    assert report["target_reachable"] is True
    assert report["tolerance_band_reachable"] is True
    assert report["headroom_factor"] == pytest.approx(DERIVED_HEADROOM_FACTOR)
    assert report["max_reachable_force_n"] == pytest.approx(1.5)


def test_k2_reference_is_an_explicit_cpu_reference_in_force_mode(k2_spec):
    data = k2_spec.to_dict()
    assert data["kind"] == "cpu_reference"
    assert data["backend_id"] == "rigid_cpu_v1"
    assert data["controller"]["mode"] == "force"
    assert data["controller"]["force"]["target_force_n"] == 1.0
    assert [n for n, s in data["runtime"].items() if s["status"] == "unresolved"] == []
    assert data["grid"]["edge_policy"] == "reject_outside_top_face"


def test_k2_reference_records_why_its_press_limit_differs_from_phase_1(k2_spec):
    evidence = k2_spec.to_dict()["parameter_evidence"]
    assert "Derived" in evidence["max_press_speed"]
    assert "0.75" in evidence["max_press_speed"]
    # The inherited gains are flagged as untuned rather than presented as ready.
    assert "not yet tuned" in evidence["force_kp"]
    assert "not yet tuned" in evidence["force_ki"]


def test_an_unreachable_target_is_rejected_at_load(k2_spec):
    # Restoring Phase 1's press limit must be refused, not silently accepted.
    spec = mutate(k2_spec, lambda d: d["controller"]["force"].update(max_press_speed_mps=0.25))
    with pytest.raises(UnreachableForceTarget, match="reaches at most"):
        enforce_force_reachability(spec)
    with pytest.raises(UnreachableForceTarget):
        kernel_config_from_spec(spec)


def test_a_limit_admitting_only_the_bare_target_is_refused(k2_spec):
    # Exactly the target leaves no room at the top of the tolerance band.
    bare = command_speed_for_force_mps(mass_kg=MASS, force_n=1.0, output_dt_s=DT)
    spec = mutate(k2_spec, lambda d: d["controller"]["force"].update(max_press_speed_mps=bare))
    report = assess_force_reachability(spec)
    assert report["target_reachable"] is True
    assert report["tolerance_band_reachable"] is False
    with pytest.raises(UnreachableForceTarget):
        enforce_force_reachability(spec)


def test_reachability_depends_on_the_control_rate_not_only_the_force(k2_spec):
    # The force is injected momentum, so halving the step halves the speed a
    # given force needs. A spec that is fine at 60 Hz can fail at another rate.
    report = assess_force_reachability(k2_spec)
    slower = command_speed_for_force_mps(mass_kg=MASS, force_n=1.0, output_dt_s=DT * 2)
    assert slower == pytest.approx(2 * report["required_press_speed_mps"])
    assert "control rate" in report["note"]


def test_trajectory_specs_are_not_subjected_to_a_force_check(k1_spec):
    assert k1_spec.to_dict()["controller"]["force"] is None
    assert assess_force_reachability(k1_spec) is None
    assert enforce_force_reachability(k1_spec) is None


def test_the_kernel_actually_delivers_the_target_at_the_required_speed():
    from contact_kernel import (
        GridSpec, KernelConfig, ProbeSpec, RigidContactKernel,
        SolverSpec, SurfaceSpec, TimingSpec,
    )

    probe_id, surface_id = "/World/Fingertip", "/World/Cube"
    kernel = RigidContactKernel(KernelConfig(
        surface=SurfaceSpec(id=surface_id, center_world_m=(0, 0, 0.25), size_m=(0.5, 0.5, 0.5)),
        probes=(ProbeSpec(id=probe_id, shape="sphere", mass_kg=MASS,
                          initial_position_world_m=(0, 0, 0.62), radius_m=0.12),),
        timing=TimingSpec(physics_dt_s=DT, control_dt_s=DT, output_dt_s=DT, substeps_per_control=1),
        grid=GridSpec(rows=64, cols=64, cell_size_m=0.5 / 64, origin_local_m=(-0.25, -0.25, 0.25)),
        solver=SolverSpec(),
    ))
    key = RigidContactKernel.pair_key(probe_id, surface_id)
    speed = command_speed_for_force_mps(mass_kg=MASS, force_n=1.0, output_dt_s=DT)
    for _ in range(5):
        pair = kernel.step({probe_id: (0.0, 0.0, -speed)}, phase="hold").pairs[key]
        assert pair.normal_force_n == pytest.approx(1.0)
    # And Phase 1's limit really does top out where the assessment says.
    kernel.reset()
    pair = kernel.step({probe_id: (0.0, 0.0, -0.25)}, phase="hold").pairs[key]
    assert pair.normal_force_n == pytest.approx(0.75)
