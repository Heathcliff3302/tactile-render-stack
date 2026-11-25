"""P2.2.2: run the multi-probe CPU reference and write one directory per stream.

    cd Phase_2
    ./.venv/Scripts/python.exe -m mechanics.run_k3_multi              # two boxes
    ./.venv/Scripts/python.exe -m mechanics.run_k3_multi --spec mixed # box + sphere

Each stream is written as a complete K1-style run directory under
``<output>/<stream_id>/`` (manifest, stream, dense field, curve report), and
each one goes through the full K1 trajectory gate order. The three
cross-stream gates are then evaluated on top. ``acceptance.json`` at the
output root holds the combined verdict.

The run is not a milestone verdict: P2.2.3 is still outstanding, and the
report says so in ``milestone``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechanics.k1_report import missing_required_artifacts, write_run
from mechanics.multi_probe import MultiProbeExperiment, solo_spec
from scenarios.k1_variants import time_refinement_specs
from tactile_contract import ExperimentSpec
from validation.k1_acceptance import (
    acceptance_report,
    evaluate_dense_field,
    evaluate_replay,
    finalize_report,
)
from validation.multi_acceptance import (
    evaluate_box_multicell,
    evaluate_independence,
    evaluate_patch_stability,
    finalize_multi_report,
    milestone_status,
)

SPECS = {"box": "p2_multi_box_cpu", "mixed": "p2_multi_mixed_cpu"}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="P2.2.2 multi-probe CPU reference")
    parser.add_argument("--spec", choices=sorted(SPECS), default="box")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--refinement-substeps", type=int, nargs="*", default=[2, 4])
    args = parser.parse_args()

    spec_id = SPECS[args.spec]
    run_id = spec_id.replace("_", "-")
    output = args.output or root / "experiments/runs" / run_id
    spec = ExperimentSpec.from_dict(json.loads(
        (root / f"scenarios/k0/{spec_id}.json").read_text(encoding="utf-8")
    ))
    experiment = MultiProbeExperiment(spec, run_id=run_id)
    runs = experiment.run()
    refinements = [
        MultiProbeExperiment(variant, run_id=f"{run_id}-substeps"
                             f"{variant.to_dict()['timing']['substeps_per_control']}").run()
        for variant in time_refinement_specs(spec, tuple(args.refinement_substeps))
    ] if args.refinement_substeps else []

    solo_runs = {}
    for stream in spec.to_dict()["streams"]:
        solo = MultiProbeExperiment(solo_spec(spec, stream["body0"]), run_id=f"{run_id}-solo").run()
        solo_runs[stream["id"]] = solo[stream["id"]]

    effective_runtime = experiment.effective_runtime()
    stream_reports = {}
    missing_artifacts = {}
    for stream_id, run in runs.items():
        report = acceptance_report(run, refinement_runs=[item[stream_id] for item in refinements])
        report["milestone"] = milestone_status()
        directory = output / stream_id
        write_run(run, directory, report, effective_runtime)
        final = finalize_report({**report, "gates": [
            *report["gates"], evaluate_dense_field(run, directory), evaluate_replay(directory),
        ]})
        final["milestone"] = milestone_status()
        stream_reports[stream_id] = final
        missing_artifacts[stream_id] = missing_required_artifacts(run, directory)

    multi_gates = [
        evaluate_independence(runs, solo_runs),
        evaluate_patch_stability(runs, spec),
        evaluate_box_multicell(runs),
    ]
    combined = finalize_multi_report(spec, stream_reports, multi_gates)
    combined["missing_required_artifacts"] = missing_artifacts
    (output / "acceptance.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n",
    )

    patch = next(gate for gate in multi_gates if gate["gate"] == "patch_ids_are_stable")
    independence = next(gate for gate in multi_gates if gate["gate"] == "streams_are_independent")
    summary = {
        "spec_id": spec_id,
        "output": str(output),
        "streams": {
            stream_id: {
                "shape": run.kernel_config.probes[0].shape,
                "frames": len(run.frames),
                "segments": stream_reports[stream_id]["segments"],
                "gates": combined["streams"][stream_id],
            }
            for stream_id, run in runs.items()
        },
        "multi_gates": {gate["gate"]: gate["passed"] for gate in multi_gates},
        "independence": [
            {k: s[k] for k in ("stream_id", "solo_frames", "multi_frames",
                               "first_differing_frame", "extra_frames_after_solo_end")}
            for s in independence["streams"]
        ],
        "patch_stability": [
            {
                "stream_id": s["stream_id"],
                "primary_hold": sorted(s["primary_hold_patches"]),
                "primary_slide": sorted(s["primary_slide_patches"]),
                "edge_hold": sorted(s["edge_hold_patches"]),
                "edge_slide": sorted(s["edge_slide_patches"]),
                "stable_patch_coverage": s["stable_patch_coverage"],
            }
            for s in patch["streams"]
        ],
        "acceptance_key_ledger": combined["acceptance_key_ledger"],
        "missing_gates": combined["missing_gates"],
        "missing_required_artifacts": missing_artifacts,
        "run_acceptance_passed": combined["run_acceptance_passed"],
        "milestone_complete": combined["milestone"]["milestone_complete"],
        "milestone_outstanding": combined["milestone"]["outstanding"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    ok = combined["run_acceptance_passed"] and not any(missing_artifacts.values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
