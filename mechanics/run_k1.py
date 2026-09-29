"""Run K1 and write the complete run directory.

    cd "Phase 2"
    .venv/Scripts/python.exe -m mechanics.run_k1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechanics.k1_report import missing_required_artifacts, write_index, write_run
from mechanics.k1_rigid import K1Experiment
from scenarios.k1_variants import time_refinement_specs
from tactile_contract import ExperimentSpec
from validation.k1_acceptance import acceptance_report, evaluate_replay, finalize_report


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Run the K1 single-probe rigid CPU reference")
    parser.add_argument("--spec", type=Path, default=root / "scenarios/k0/k1_single_probe_cpu.json")
    parser.add_argument("--output", type=Path, default=root / "experiments/runs/k1-single-probe-cpu")
    parser.add_argument("--index", type=Path, default=root / "experiments/index/k1_single_probe_cpu.json")
    parser.add_argument("--run-id", default="k1-single-probe-cpu")
    parser.add_argument(
        "--refinement-substeps",
        type=int,
        nargs="*",
        default=[2, 4],
        help="physics substeps per control interval for the time-step semantics gate",
    )
    args = parser.parse_args()

    spec = ExperimentSpec.from_dict(json.loads(args.spec.read_text(encoding="utf-8")))
    experiment = K1Experiment(spec, run_id=args.run_id)
    run = experiment.run()

    refinement_runs = [
        K1Experiment(variant, run_id=f"{args.run_id}-substeps{variant.to_dict()['timing']['substeps_per_control']}").run()
        for variant in time_refinement_specs(spec, tuple(args.refinement_substeps))
    ] if args.refinement_substeps else []

    # The replay gate is a property of the finished directory, so it cannot be
    # inside the hash-bound manifest: a file cannot carry a verified statement
    # about its own hash. The manifest therefore binds an acceptance record
    # that is deliberately one gate short, and says so.
    report = acceptance_report(run, refinement_runs=refinement_runs)
    report["replay_gate_location"] = (
        "interaction_state_replays_completely is verified after this file is "
        "hashed; its result is in replay_check.json"
    )
    manifest = write_run(run, args.output, report, experiment.effective_runtime())

    replay = evaluate_replay(args.output)
    final = finalize_report({**report, "gates": [*report["gates"], replay]})
    (args.output / "replay_check.json").write_text(
        json.dumps({
            "schema": "k1-replay-check/v1",
            "note": "verified after manifest.json was written; not a manifest artifact",
            "k1_acceptance_complete": final["k1_acceptance_complete"],
            "k1_acceptance_passed": final["k1_acceptance_passed"],
            "gate_names": final["gate_names"],
            "replay_gate": replay,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    report = final

    index = write_index(run, manifest, report, args.output, args.index)
    missing = missing_required_artifacts(run, args.output)
    summary = {
        "index": str(args.index),
        "manifest_sha256": index["manifest_sha256"],
        "output": str(args.output),
        "run_id": run.run_id,
        "loop_state": run.loop_state,
        "abort_reason": run.abort_reason,
        "frames": len(run.frames),
        "conditions_sha256": run.conditions_sha256,
        "manifest_artifacts": len(manifest["artifacts"]),
        "segments": report["segments"],
        "gates": {gate["gate"]: gate["passed"] for gate in report["gates"]},
        "run_checks_passed": report["run_checks_passed"],
        "k1_acceptance_complete": report["k1_acceptance_complete"],
        "k1_acceptance_passed": report["k1_acceptance_passed"],
        "missing_gates": report["missing_gates"],
        "missing_required_artifacts": missing,
        "stage_table": list(run.stage_table),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["k1_acceptance_passed"] and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
