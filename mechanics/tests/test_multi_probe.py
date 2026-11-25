"""P2.2.2: several probes, mixed shapes, one independent stream per pair.

Every positive claim below has a counter-example next to it that must fail,
because a gate that cannot fail is not evidence.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from contact_kernel import (
    DriveSpec, GridSpec, KernelConfig, ProbeSpec, RigidContactKernel, SamplingSpec,
    SolverSpec, SurfaceSpec, TimingSpec,
)
from mechanics.k1_rigid import RigidProbeExperiment
from mechanics.multi_probe import MultiProbeExperiment, solo_spec
from tactile_contract import ExperimentSpec
from validation.k1_acceptance import DEFERRED_ACCEPTANCE_KEYS, acceptance_key_ledger, evaluate_run
from validation.multi_acceptance import (
    evaluate_box_multicell,
    evaluate_independence,
    evaluate_patch_stability,
    stream_patch_stability,
)

ROOT = Path(__file__).resolve().parents[2]


def load(spec_id: str) -> ExperimentSpec:
    return ExperimentSpec.from_dict(json.loads(
        (ROOT / f"scenarios/k0/{spec_id}.json").read_text(encoding="utf-8")
    ))


@pytest.fixture(scope="module")
def box_spec():
    return load("p2_multi_box_cpu")


@pytest.fixture(scope="module")
def box_runs(box_spec):
    return MultiProbeExperiment(box_spec).run()


@pytest.fixture(scope="module")
def box_solo(box_spec):
    solo = {}
    for stream in box_spec.to_dict()["streams"]:
        solo[stream["id"]] = MultiProbeExperiment(solo_spec(box_spec, stream["body0"])).run()[stream["id"]]
    return solo


@pytest.fixture(scope="module")
def mixed_spec():
    return load("p2_multi_mixed_cpu")


@pytest.fixture(scope="module")
def mixed_runs(mixed_spec):
    return MultiProbeExperiment(mixed_spec).run()


@pytest.fixture(scope="module")
def mixed_solo(mixed_spec):
    solo = {}
    for stream in mixed_spec.to_dict()["streams"]:
        solo[stream["id"]] = MultiProbeExperiment(solo_spec(mixed_spec, stream["body0"])).run()[stream["id"]]
    return solo


# -- one stream per pair --------------------------------------------------

def test_one_stream_per_declared_pair(box_spec, box_runs):
    declared = {stream["id"]: stream["body0"] for stream in box_spec.to_dict()["streams"]}
    assert set(box_runs) == set(declared)
    for stream_id, run in box_runs.items():
        assert {frame.body0 for frame in run.frames} == {declared[stream_id]}
        assert [probe.id for probe in run.kernel_config.probes] == [declared[stream_id]]


@pytest.mark.parametrize("runs_name", ["box_runs", "mixed_runs"])
def test_every_stream_passes_the_k1_single_run_gates(request, runs_name):
    for run in request.getfixturevalue(runs_name).values():
        report = evaluate_run(run)
        failed = [gate["gate"] for gate in report["gates"] if not gate["passed"]]
        assert failed == [], (run.stream_id, failed)


def test_mixed_run_really_mixes_shapes(mixed_runs):
    shapes = {run.kernel_config.probes[0].shape for run in mixed_runs.values()}
    assert shapes == {"box", "sphere"}


# -- independence ---------------------------------------------------------

@pytest.mark.parametrize("pair", [("box_runs", "box_solo"), ("mixed_runs", "mixed_solo")])
def test_streams_equal_their_solo_runs(request, pair):
    runs, solo = (request.getfixturevalue(name) for name in pair)
    gate = evaluate_independence(runs, solo)
    assert gate["passed"], gate["streams"]


def test_mixed_sphere_stream_waits_released_for_the_box(mixed_runs, mixed_solo):
    gate = evaluate_independence(mixed_runs, mixed_solo)
    sphere = next(s for s in gate["streams"] if s["stream_id"] == "probe1_surface")
    assert sphere["extra_frames_after_solo_end"] > 0
    assert sphere["extra_frames_released_without_contact"]


def test_counterexample_swapped_streams_fail_independence(box_runs, box_solo):
    """A stream fed from the other pair must not pass as independent."""
    swapped = {"probe0_surface": box_solo["probe1_surface"],
               "probe1_surface": box_solo["probe0_surface"]}
    gate = evaluate_independence(box_runs, swapped)
    assert not gate["passed"]
    assert all(s["first_differing_frame"] is not None for s in gate["streams"])


def test_counterexample_missing_solo_fails(box_runs, box_solo):
    gate = evaluate_independence(box_runs, {"probe0_surface": box_solo["probe0_surface"]})
    assert not gate["passed"]


# -- patch stability, primary vs edge ------------------------------------

def test_patch_stability_passes_on_both_specs(box_runs, box_spec, mixed_runs, mixed_spec):
    assert evaluate_patch_stability(box_runs, box_spec)["passed"]
    assert evaluate_patch_stability(mixed_runs, mixed_spec)["passed"]


def _fake_run(stream_id, frames, shape="box"):
    return SimpleNamespace(
        stream_id=stream_id, frames=frames,
        kernel_config=SimpleNamespace(probes=[SimpleNamespace(shape=shape)]),
    )


def _fake_frame(phase, patch_ids, cells=4):
    return SimpleNamespace(metadata={
        "motion_phase": phase,
        "patches": [{"patch_id": pid, "normal_force_n": 0.1, "cell_count": cells} for pid in patch_ids],
    })


LIMITS = {"patch_coverage_min": 0.9, "primary_patch_min_occupancy": 0.5}


def test_counterexample_relabelled_patch_ids_fail_stability():
    """Every slide frame gets a fresh patch ID: nothing is primary in both segments."""
    frames = [_fake_frame("hold", [1]) for _ in range(60)]
    frames += [_fake_frame("lateral_slide", [100 + index]) for index in range(60)]
    result = stream_patch_stability(_fake_run("s", frames), LIMITS)
    assert not result["passed"]
    assert result["stable_patch_ids"] == []


def test_edge_patches_are_counted_apart_from_primary():
    frames = [_fake_frame("hold", [1] + ([7] if index < 10 else [])) for index in range(60)]
    frames += [_fake_frame("lateral_slide", [1]) for _ in range(60)]
    result = stream_patch_stability(_fake_run("s", frames), LIMITS)
    assert set(result["primary_hold_patches"]) == {1}
    assert set(result["edge_hold_patches"]) == {7}
    assert result["edge_hold_patches"][7]["frames"] == 10
    assert result["passed"]


def test_counterexample_low_coverage_fails_stability():
    """Two primary slide patches, only one of them stable: coverage 0.5 < 0.9."""
    frames = [_fake_frame("hold", [1]) for _ in range(60)]
    frames += [_fake_frame("lateral_slide", [1, 2]) for _ in range(60)]
    result = stream_patch_stability(_fake_run("s", frames), LIMITS)
    assert result["stable_patch_coverage"] == 0.5
    assert not result["passed"]


# -- multi-cell ----------------------------------------------------------

def test_box_streams_activate_multiple_cells(box_runs, mixed_runs):
    assert evaluate_box_multicell(box_runs)["passed"]
    assert evaluate_box_multicell(mixed_runs)["passed"]


def test_counterexample_no_box_stream_does_not_pass_vacuously():
    runs = {"s": _fake_run("s", [_fake_frame("hold", [1], cells=1)], shape="sphere")}
    assert not evaluate_box_multicell(runs)["passed"]


def test_counterexample_single_cell_box_fails():
    runs = {"s": _fake_run("s", [_fake_frame("hold", [1], cells=1)], shape="box")}
    assert not evaluate_box_multicell(runs)["passed"]


# -- threshold ledger ----------------------------------------------------

def test_multi_ledger_consumes_patch_coverage_and_defers_resolution(box_spec):
    ledger = acceptance_key_ledger(box_spec)
    assert ledger["consumed"]["patch_coverage_min"] == ["patch_ids_are_stable"]
    assert ledger["consumed"]["primary_patch_min_occupancy"] == ["patch_ids_are_stable"]
    assert set(ledger["deferred"]) == {"resolution_force_spread_max"}


def test_single_probe_ledger_still_defers_patch_coverage():
    ledger = acceptance_key_ledger(load("k1_single_probe_cpu"))
    assert "patch_coverage_min" in ledger["deferred"]
    assert set(ledger["deferred"]) == set(DEFERRED_ACCEPTANCE_KEYS)


def test_counterexample_multi_spec_without_occupancy_threshold_is_rejected(box_spec):
    data = copy.deepcopy(box_spec.to_dict())
    data["acceptance"].pop("primary_patch_min_occupancy")
    with pytest.raises(ValueError, match="primary_patch_min_occupancy"):
        acceptance_key_ledger(ExperimentSpec.build(**data))


# -- single box through the five-layer K1 assembly (a3) -----------------

def _k1_with_box() -> ExperimentSpec:
    data = copy.deepcopy(load("k1_single_probe_cpu").to_dict())
    data["probes"][0].update(shape="box", radius_m=None, size_m=[0.012, 0.012, 0.06],
                             mass_kg=0.08, z_contact_m=0.53, retract_distance_m=0.67)
    data["spec_id"] = "k1_single_box_cpu_test"
    return ExperimentSpec.build(**data)


def test_single_box_runs_through_the_five_layer_loop():
    run = RigidProbeExperiment(_k1_with_box(), run_id="k1-box-test").run()
    report = evaluate_run(run)
    failed = [gate["gate"] for gate in report["gates"] if not gate["passed"]]
    assert failed == []
    assert max(frame.contact_point_count for frame in run.frames) > 1
    assert [stage["layer"] for stage in run.stage_table] == [1, 2, 3, 4, 5]


def test_rigid_probe_experiment_still_rejects_several_probes(box_spec):
    with pytest.raises(ValueError, match="exactly one probe"):
        RigidProbeExperiment(box_spec)


def test_multi_experiment_rejects_single_mode_specs():
    with pytest.raises(ValueError, match="multi_trajectory"):
        MultiProbeExperiment(load("k1_single_probe_cpu"))


# -- kernel per-probe phase labels ---------------------------------------

def _two_probe_kernel():
    return RigidContactKernel(KernelConfig(
        surface=SurfaceSpec(id="/S", center_world_m=(0, 0, 0.25), size_m=(0.5, 0.5, 0.5)),
        probes=(ProbeSpec(id="/A", shape="sphere", mass_kg=0.05, radius_m=0.12,
                          initial_position_world_m=(-0.1, 0, 1.0)),
                ProbeSpec(id="/B", shape="box", mass_kg=0.08, size_m=(0.02, 0.02, 0.02),
                          initial_position_world_m=(0.1, 0, 1.0))),
        timing=TimingSpec(physics_dt_s=1 / 60, control_dt_s=1 / 60, output_dt_s=1 / 60,
                          substeps_per_control=1),
        grid=GridSpec(rows=32, cols=32, cell_size_m=0.5 / 32, origin_local_m=(-0.25, -0.25, 0.25)),
        solver=SolverSpec(), sampling=SamplingSpec(), drive=DriveSpec(),
    ))


def test_kernel_labels_each_pair_with_its_own_phase():
    frame = _two_probe_kernel().step({"/A": (0, 0, 0), "/B": (0, 0, 0)},
                                     phase="per_probe", probe_phases={"/A": "hold", "/B": "approach"})
    assert frame.pairs["/A|/S"].phase == "hold"
    assert frame.pairs["/B|/S"].phase == "approach"


def test_kernel_without_mapping_keeps_the_single_label():
    frame = _two_probe_kernel().step({"/A": (0, 0, 0), "/B": (0, 0, 0)}, phase="approach")
    assert {pair.phase for pair in frame.pairs.values()} == {"approach"}


# -- run_k3 claim (P2.2.1 entry point) -----------------------------------

def test_run_k3_multi_cell_claim_is_false_without_contact():
    from mechanics.run_k3 import multi_cell_claim
    assert multi_cell_claim([{"contact_present": True, "active_cells": 64}])
    assert not multi_cell_claim([{"contact_present": False, "active_cells": 0}] * 3)
    assert not multi_cell_claim([{"contact_present": True, "active_cells": 1}])
