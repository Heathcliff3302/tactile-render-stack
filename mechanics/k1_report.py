"""Write one K1 run directory: evidence files, then the manifest that binds them.

Order matters. Every artifact is written first, then hashed, then the manifest
records the hashes. ``tactile_contract.io.load_run`` verifies those hashes and
revalidates the frames, which is what the K1 replay gate reads.

The artifact set covers ``required_run_artifacts`` from the specification. The
layer 3, 4 and 5 streams are additional evidence of the closed five-layer loop.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

from contact_kernel import RigidContactKernel
from interaction import manifest_observables
from tactile_contract._common import canonical_json
from tactile_contract.io import write_frames
from validation.k1_acceptance import manifest_from_run

ARTIFACT_ROLES = {
    "stream_index.json": "summary",
    "effective_runtime.json": "summary",
    "acceptance.json": "summary",
    "environment.json": "summary",
    "actual_commands.jsonl": "trajectory",
    "points.csv": "raw_backend",
    "cells.csv": "raw_backend",
    "patches.csv": "raw_backend",
    "episodes.csv": "raw_backend",
    "kinematics.jsonl": "log",
    "material_response.jsonl": "log",
    "tactile_target.jsonl": "log",
    "actuator_io.jsonl": "log",
}

POINT_HEADER = [
    "time_s", "control_step", "phase", "patch_id", "contact_id", "body0", "body1",
    "point_world_x_m", "point_world_y_m", "point_world_z_m",
    "point_local_x_m", "point_local_y_m", "point_local_z_m",
    "normal_world_x", "normal_world_y", "normal_world_z",
    "gap_m", "penetration_m", "normal_impulse_ns", "normal_force_n",
    "tangent_force_x_n", "tangent_force_y_n", "tangent_force_z_n",
    "grid_row", "grid_col", "curvature_1_per_m", "curvature_2_per_m", "curvature_valid",
    "normal_relative_velocity_mps",
    "tangent_velocity_x_mps", "tangent_velocity_y_mps", "tangent_velocity_z_mps",
    "tangent_speed_mps", "tangent_u_speed_mps", "tangent_v_speed_mps",
    "kinematics_source", "finite_difference_tangent_speed_mps",
    "motion_direction_u", "motion_direction_v", "direction_valid",
]
CELL_HEADER = [
    "time_s", "control_step", "phase", "patch_id", "interaction_state", "row", "col",
    "center_local_x_m", "center_local_y_m", "center_local_z_m",
    "cell_area_m2", "coverage_alpha", "area_method", "point_count", "contact_ids",
    "normal_force_n", "normal_impulse_ns",
    "tangent_force_x_n", "tangent_force_y_n", "tangent_force_z_n",
    "mean_normal_relative_velocity_mps", "mean_tangent_speed_mps",
    "motion_direction_u", "motion_direction_v", "direction_valid",
]
PATCH_HEADER = [
    "time_s", "control_step", "phase", "patch_id", "interaction_state", "lifecycle",
    "cell_count", "active_cells", "mapped_area_m2", "area_method",
    "contact_point_count", "contact_ids",
    "centroid_local_x_m", "centroid_local_y_m", "centroid_local_z_m",
    "geometric_centroid_local_x_m", "geometric_centroid_local_y_m", "geometric_centroid_local_z_m",
    "normal_force_n", "normal_impulse_ns", "mean_tangent_speed_mps",
    "merged_from_patch_ids", "split_from_patch_id",
]
EPISODE_HEADER = [
    "patch_id", "start_time_s", "end_time_s", "duration_s", "first_step", "last_step",
    "samples", "trajectory_length_m", "max_normal_force_n",
    "cumulative_normal_impulse_ns", "max_mapped_area_m2", "swept_area_m2",
    "sliding_detected", "end_reason",
]


def _finite_or_none(value):
    """Strict JSON has no NaN. An undefined diagnostic is recorded as null."""
    return value if isinstance(value, (int, bool)) or math.isfinite(value) else None


def _finite_or_blank(value):
    """CSV counterpart: an undefined value is an empty cell, never the text nan."""
    return value if math.isfinite(value) else ""


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(canonical_json(row) + "\n" for row in rows),
                    encoding="utf-8", newline="\n")


def _write_csv(path: Path, header, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _pair_frames(run):
    key = RigidContactKernel.pair_key(
        run.kernel_config.probes[0].id, run.kernel_config.surface.id
    )
    for loop_frame in run.loop_frames:
        yield loop_frame, loop_frame.world.kernel_frame.pairs[key]


def environment_record(run) -> dict:
    """Host and package record. Machine dependent by nature, so it is declared."""
    versions = {}
    for name in ("jsonschema", "numpy"):
        try:
            from importlib.metadata import version

            versions[name] = version(name)
        except Exception:  # pragma: no cover - absent optional dependency
            versions[name] = "unavailable"
    return {
        "schema": "k1-environment/v1",
        "python_version": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": versions,
        "compute_backend": "cpu_double_precision",
        "wall_clock_s": run.wall_clock_s,
        "simulated_time_s": run.end_time_s,
        "note": (
            "host record for this run only; wall-clock time and host fields "
            "differ between machines and are not part of the physical conditions"
        ),
    }


def point_rows(run):
    for _, pair in _pair_frames(run):
        for point in pair.points:
            yield [
                point.time_s, point.control_step, point.phase, point.patch_id, point.contact_id,
                point.body0, point.body1,
                *point.point_world_m, *point.point_local_m, *point.normal_world,
                point.gap_m, point.penetration_m, point.normal_impulse_ns, point.normal_force_n,
                *point.tangent_force_world_n,
                point.grid_row, point.grid_col,
                _finite_or_blank(point.curvature_1_per_m),
                _finite_or_blank(point.curvature_2_per_m),
                int(point.curvature_valid),
                point.normal_relative_velocity_mps,
                *point.tangent_velocity_world_mps,
                point.tangent_speed_mps, point.tangent_u_speed_mps, point.tangent_v_speed_mps,
                point.kinematics_source,
                _finite_or_blank(point.finite_difference_tangent_speed_mps),
                *(point.motion_direction_uv if point.direction_valid else ("", "")),
                int(point.direction_valid),
            ]


def cell_rows(run):
    for _, pair in _pair_frames(run):
        for key in sorted(pair.cells):
            cell = pair.cells[key]
            yield [
                cell.time_s, cell.control_step, cell.phase, cell.patch_id, cell.interaction_state,
                cell.row, cell.col, *cell.center_local_m,
                cell.cell_area_m2, cell.coverage_alpha, cell.area_method,
                cell.point_count, json.dumps(cell.contact_ids),
                cell.normal_force_n, cell.normal_impulse_ns, *cell.tangent_force_world_n,
                cell.mean_normal_relative_velocity_mps, cell.mean_tangent_speed_mps,
                *(cell.motion_direction_uv if cell.direction_valid else ("", "")),
                int(cell.direction_valid),
            ]


def patch_rows(run):
    for _, pair in _pair_frames(run):
        for patch in pair.patches:
            yield [
                patch.time_s, patch.control_step, patch.phase, patch.patch_id,
                patch.interaction_state, patch.lifecycle, len(patch.active_cells),
                json.dumps([list(cell) for cell in patch.active_cells]),
                patch.mapped_area_m2, patch.area_method,
                patch.contact_point_count, json.dumps(patch.contact_ids),
                *patch.centroid_local_m, *patch.geometric_centroid_local_m,
                patch.normal_force_n, patch.normal_impulse_ns, patch.mean_tangent_speed_mps,
                json.dumps(patch.merged_from_patch_ids), patch.split_from_patch_id,
            ]


def episode_rows(run):
    for episode in run.episodes:
        yield [
            episode.patch_id, episode.start_time_s, episode.end_time_s, episode.duration_s,
            episode.first_step, episode.last_step, episode.samples,
            episode.trajectory_length_m, episode.max_normal_force_n,
            episode.cumulative_normal_impulse_ns, episode.max_mapped_area_m2,
            episode.swept_area_m2, int(episode.sliding_detected), episode.end_reason,
        ]


def kinematics_rows(run):
    source = run.kernel_config.sampling.kinematics_source
    for loop_frame, pair in _pair_frames(run):
        yield {
            "time_s": loop_frame.time_s,
            "control_step": loop_frame.control_step,
            "phase": pair.phase,
            "kinematics_source": source,
            "probe_position_world_m": list(pair.probe_position_world_m),
            "probe_velocity_world_mps": list(pair.probe_velocity_world_mps),
            "command_velocity_world_mps": list(pair.command_velocity_world_mps),
            "normal_relative_velocity_mps": pair.normal_relative_velocity_mps,
            "tangent_velocity_world_mps": list(pair.tangent_velocity_world_mps),
            "tangent_speed_mps": pair.tangent_speed_mps,
            "finite_difference_tangent_speed_mps": (
                _finite_or_none(pair.points[0].finite_difference_tangent_speed_mps)
                if pair.points else None
            ),
            "gap_m": pair.gap_m,
            "penetration_m": pair.penetration_m,
            "position_correction_m": pair.position_correction_m,
            "substeps": [trace.to_dict() for trace in pair.solution.substeps],
        }


def write_run(run, directory, acceptance, effective_runtime) -> dict:
    """Write the complete run directory and return the manifest dictionary."""
    root = Path(directory)
    (root / "streams").mkdir(parents=True, exist_ok=True)
    stream_path = f"streams/{run.stream_id}.jsonl"
    write_frames(root / stream_path, run.frames)

    _write_json(root / "stream_index.json", {
        "schema": "k1-stream-index/v1",
        "run_id": run.run_id,
        "scenario_id": run.scenario_id,
        "conditions_sha256": run.conditions_sha256,
        "frame_schema": "interaction-state/v2",
        "streams": [{
            "id": run.stream_id,
            "body0": run.kernel_config.probes[0].id,
            "body1": run.kernel_config.surface.id,
            "path": stream_path,
            "frames": len(run.frames),
            "start_time_s": run.start_time_s,
            "end_time_s": run.end_time_s,
        }],
        "note": "one ordered stream per contact pair; multi-probe streams arrive at K3",
    })
    _write_json(root / "effective_runtime.json", effective_runtime)
    _write_json(root / "acceptance.json", acceptance)
    _write_json(root / "environment.json", environment_record(run))
    _write_jsonl(root / "actual_commands.jsonl", run.commands)
    _write_csv(root / "points.csv", POINT_HEADER, point_rows(run))
    _write_csv(root / "cells.csv", CELL_HEADER, cell_rows(run))
    _write_csv(root / "patches.csv", PATCH_HEADER, patch_rows(run))
    _write_csv(root / "episodes.csv", EPISODE_HEADER, episode_rows(run))
    _write_jsonl(root / "kinematics.jsonl", kinematics_rows(run))
    _write_jsonl(root / "material_response.jsonl",
                 [frame.material.to_dict() for frame in run.loop_frames])
    _write_jsonl(root / "tactile_target.jsonl",
                 [frame.target.to_dict() for frame in run.loop_frames])
    _write_jsonl(root / "actuator_io.jsonl", [
        {"command": frame.command.to_dict(), "measurement": frame.measurement.to_dict()}
        for frame in run.loop_frames
    ])

    artifacts = [{
        "path": stream_path,
        "role": "interaction_frames",
        "sha256": _sha256(root / stream_path),
    }]
    for name, role in ARTIFACT_ROLES.items():
        artifacts.append({"path": name, "role": role, "sha256": _sha256(root / name)})
    manifest = manifest_from_run(
        run, artifacts, manifest_observables(run.kernel_config.sampling.kinematics_source)
    )
    _write_json(root / "manifest.json", manifest.to_dict())
    return manifest.to_dict()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_index(run, manifest, report, directory, index_path) -> dict:
    """Small committed record of one run: identity, hashes and gate results.

    Bulk streams stay in the ignored run directory. This index is what a later
    phase reads to find a run, confirm which specification produced it, and see
    whether its gates passed, without loading megabytes of frames.
    """
    root = Path(directory)
    index = {
        "schema": "k1-run-index/v1",
        "run_id": run.run_id,
        "scenario": manifest["scenario"],
        "backend": manifest["backend"],
        "spec_id": run.spec.to_dict()["spec_id"],
        "spec_sha256": run.spec.to_dict()["spec_sha256"],
        "conditions_sha256": run.conditions_sha256,
        "manifest_sha256": _sha256(root / "manifest.json"),
        "run_directory": root.name,
        "frames": len(run.frames),
        "simulated_time_s": run.end_time_s,
        "substeps_per_control": run.substeps_per_control,
        "loop_state": run.loop_state,
        "abort_reason": run.abort_reason,
        "stage_table": list(run.stage_table),
        "segments": report["segments"],
        "gates": {gate["gate"]: gate["passed"] for gate in report["gates"]},
        "run_checks_passed": report["run_checks_passed"],
        "k1_acceptance_complete": report["k1_acceptance_complete"],
        "k1_acceptance_passed": report["k1_acceptance_passed"],
        "missing_gates": report["missing_gates"],
        "artifacts": manifest["artifacts"],
    }
    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(index_path, index)
    return index


def missing_required_artifacts(run, directory) -> list[str]:
    """Check the produced directory against ``required_run_artifacts``."""
    root = Path(directory)
    missing = []
    for pattern in run.spec.to_dict()["required_run_artifacts"]:
        if not list(root.glob(pattern)):
            missing.append(pattern)
    return missing
