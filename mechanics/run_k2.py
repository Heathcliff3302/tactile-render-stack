"""Run K2 force regulation and write the complete run directory.

    cd Phase_2
    .venv/Scripts/python.exe -m mechanics.run_k2

Same assembly as K1, same five layers, same artifacts. The only declared
difference is the controller: a PI regulator holding a normal-force target
instead of an open-loop trajectory. Keeping the pipeline identical is what
makes the two runs comparable; a force run with its own reporting path would
prove nothing about the trajectory baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechanics.k1_report import missing_required_artifacts, write_index, write_run
from mechanics.k1_rigid import (
    K2_SCENARIO_DESCRIPTION,
    K2_SCENARIO_ID,
    K2_SCENARIO_VERSION,
    RigidProbeExperiment,
)
from scenarios.k1_variants import time_refinement_specs
from tactile_contract import ExperimentSpec
from validation.k1_acceptance import (
    acceptance_report,
    evaluate_dense_field,
    evaluate_replay,
    finalize_report,
)


def main(argv=None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the K2 force-control CPU reference")
    parser.add_argument("--spec", type=Path, default=root / "scenarios/k0/k2_force_cpu.json")
    parser.add_argument("--output", type=Path, default=root / "experiments/runs/k2-force-cpu")
    parser.add_argument("--index", type=Path, default=root / "experiments/index/k2_force_cpu.json")
    parser.add_argument("--run-id", default="k2-force-cpu")
    parser.add_argument(
        "--refinement-substeps",
        type=int,
        nargs="*",
        default=[2, 4],
        help="physics substeps per control interval for the time-step semantics gate",
    )
    args = parser.parse_args(argv)

    spec = ExperimentSpec.from_dict(json.loads(args.spec.read_text(encoding="utf-8")))
    experiment = RigidProbeExperiment(
        spec,
        run_id=args.run_id,
        scenario_id=K2_SCENARIO_ID,
        scenario_version=K2_SCENARIO_VERSION,
        scenario_description=K2_SCENARIO_DESCRIPTION,
    )
    run = experiment.run()

    refinement_runs = [
        RigidProbeExperiment(
            variant,
            run_id=f"{args.run_id}-substeps{variant.to_dict()['timing']['substeps_per_control']}",
            scenario_id=K2_SCENARIO_ID,
            scenario_version=K2_SCENARIO_VERSION,
            scenario_description=K2_SCENARIO_DESCRIPTION,
        ).run()
        for variant in time_refinement_specs(spec, tuple(args.refinement_substeps))
    ] if args.refinement_substeps else []

    # Same ordering constraint as K1: the replay gate is a property of the
    # finished directory, so it cannot live inside the hash-bound manifest.
    report = acceptance_report(run, refinement_runs=refinement_runs)
    report["replay_gate_location"] = (
        "interaction_state_replays_completely and dense_field_matches_sparse_cells "
        "are verified after this file is hashed; their results are in "
        "replay_check.json"
    )
    manifest = write_run(run, args.output, report, experiment.effective_runtime())

    dense = evaluate_dense_field(run, args.output)
    replay = evaluate_replay(args.output)
    final = finalize_report({**report, "gates": [*report["gates"], dense, replay]})
    (args.output / "replay_check.json").write_text(
        json.dumps({
            "schema": "k2-replay-check/v1",
            "note": "verified after manifest.json was written; not a manifest artifact",
            "run_acceptance_complete": final["run_acceptance_complete"],
            "run_acceptance_passed": final["run_acceptance_passed"],
            "milestone": final["milestone"],
            "required_gates": final["required_gates"],
            "gate_names": final["gate_names"],
            "dense_field_gate": dense,
            "replay_gate": replay,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report = final

    index = write_index(run, manifest, report, args.output, args.index)
    missing = missing_required_artifacts(run, args.output)
    force_gates = {
        gate["gate"]: gate
        for gate in report["gates"]
        if gate["gate"].startswith("force_") or gate["gate"].startswith("regulator_")
    }
    summary = {
        "index": str(args.index),
        "manifest_sha256": index["manifest_sha256"],
        "output": str(args.output),
        "run_id": run.run_id,
        "controller_mode": run.controller_mode,
        "loop_state": run.loop_state,
        "abort_reason": run.abort_reason,
        "frames": len(run.frames),
        "conditions_sha256": run.conditions_sha256,
        "manifest_artifacts": len(manifest["artifacts"]),
        "segments": report["segments"],
        "gates": {gate["gate"]: gate["passed"] for gate in report["gates"]},
        "force_regulation": {
            "settle_time_s": force_gates.get(
                "force_target_is_reached_and_held", {}).get("settle_time_s"),
            "held_in_band_fraction": force_gates.get(
                "force_target_is_reached_and_held", {}).get("held_in_band_fraction"),
            "slide_in_band_fraction": force_gates.get(
                "force_held_through_slide", {}).get("slide_in_band_fraction"),
            "max_commanded_press_speed_mps": force_gates.get(
                "regulator_stays_within_declared_authority", {}
            ).get("max_commanded_press_speed_mps"),
        },
        "run_checks_passed": report["run_checks_passed"],
        "run_acceptance_complete": report["run_acceptance_complete"],
        "run_acceptance_passed": report["run_acceptance_passed"],
        # Deliberately separate from the run verdict above, and deliberately
        # false while the dense field, the Isaac Sim adapter and the curve
        # report are outstanding.
        "milestone": report["milestone"]["milestone"],
        "milestone_complete": report["milestone"]["milestone_complete"],
        "sub_milestone": report["milestone"]["sub_milestone"],
        "milestone_outstanding": report["milestone"]["outstanding"],
        "missing_gates": report["missing_gates"],
        "missing_required_artifacts": missing,
        "stage_table": list(run.stage_table),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # The exit status reports this run, which is what this command executed.
    # Milestone completion is reported in the summary, never conflated with it.
    return 0 if report["run_acceptance_passed"] and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
