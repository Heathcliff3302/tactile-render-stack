"""Rebuild K0 source inventories and reference specs without importing Isaac Sim.

Run from the Phase 2 root: python -m scenarios.build_k0
"""

import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / "Phase 1" / "isaacsim-rigid-contact-pipeline"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def obj(properties, required=None):
    return {"type": "object", "properties": properties, "required": list(properties) if required is None else required, "additionalProperties": False}


def array(items, n=None):
    result = {"type": "array", "items": items}
    if n is not None:
        result.update(minItems=n, maxItems=n)
    return result


def make_schema():
    text = {"type": "string", "minLength": 1}
    number = {"type": "number"}
    positive = {"type": "number", "exclusiveMinimum": 0}
    nonneg = {"type": "number", "minimum": 0}
    fraction = {"type": "number", "minimum": 0, "maximum": 1}
    integer = {"type": "integer", "minimum": 1}
    vec = array(number, 3)
    sha = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    def setting(value):
        return obj({"value": {"anyOf": [value, {"type": "null"}]}, "status": {"enum": ["source_explicit", "unresolved", "design_choice"]}, "evidence": text})
    runtime = {k: setting(v) for k, v in {
        "gravity_mps2": vec, "static_friction": nonneg, "dynamic_friction": nonneg,
        "restitution": fraction, "linear_damping_per_s": nonneg,
        "angular_damping_per_s": nonneg, "initial_linear_velocity_mps": vec,
        "initial_orientation_wxyz": array(number, 4), "contact_offset_m": nonneg,
        "rest_offset_m": number, "solver_type": text, "position_iterations": integer,
        "velocity_iterations": integer, "sleep_enabled": {"type": "boolean"},
        "ccd_mode": text, "friction_combine_mode": text, "restitution_combine_mode": text,
        "max_depenetration_velocity_mps": nonneg,
    }.items()}
    parameters = obj({
        "pre_settle_time_s": nonneg, "approach_timeout_s": positive, "hold_time_s": positive,
        "slide_time_s": positive, "release_time_s": nonneg, "approach_speed_mps": positive,
        "slide_speed_mps": positive, "retract_speed_mps": positive,
        "contact_maintain_speed_mps": positive, "gravity_compensation_mps2": positive,
        "tangent_speed_feedback_kp": nonneg, "max_slide_command_speed_mps": positive,
        "initial_slide_command_speed_mps": {"anyOf": [positive, {"type": "null"}]},
    })
    force = obj({"target_force_n": positive, "force_tolerance_n": positive,
                 "force_kp_m_per_ns": nonneg, "force_ki_m_per_ns2": nonneg,
                 "force_integral_limit_ns": positive, "max_press_speed_mps": positive,
                 "max_correct_speed_mps": positive, "settle_frames": integer, "regulate_timeout_s": positive})
    probe = obj({"id": text, "shape": {"enum": ["sphere", "box"]},
                 "mass_kg": positive, "radius_m": {"anyOf": [positive, {"type": "null"}]},
                 "size_m": {"anyOf": [array(positive, 3), {"type": "null"}]},
                 "initial_position_world_m": vec, "angular_mode": {"const": "locked"},
                 "z_contact_m": positive, "retract_distance_m": positive})
    probe["allOf"] = [{"if": {"properties": {"shape": {"const": "sphere"}}}, "then": {"properties": {"radius_m": positive, "size_m": {"type": "null"}}}, "else": {"properties": {"radius_m": {"type": "null"}, "size_m": array(positive, 3)}}}]
    schema = obj({
        "schema_version": {"const": "experiment-spec/v1"}, "spec_id": text, "spec_sha256": sha,
        "kind": {"enum": ["source_inventory", "cpu_reference"]}, "phase": {"const": "phase2"},
        "description": text, "backend_id": text,
        "coordinates": obj({"units": {"const": "SI"}, "handedness": {"const": "right"}, "up_axis": {"const": "Z"},
            "quaternion_order": {"const": "wxyz"}, "basis_world": array(vec, 3), "local_origin_world_m": vec,
            "surface_z_world_m": number, "normal_world": {"const": [0, 0, 1]}}),
        "surface": obj({"id": text, "shape": {"const": "box"}, "motion": {"const": "fixed"},
            "center_world_m": vec, "size_m": array(positive, 3), "mass_kg": {"type": "null"}}),
        "probes": {"type": "array", "items": probe, "minItems": 1},
        "streams": {"type": "array", "items": obj({"id": text, "body0": text, "body1": text,
            "frame_schema": {"const": "interaction-state/v2"}, "path_template": text}), "minItems": 1},
        "timing": obj({"start_time_s": {"const": 0}, "physics_dt_s": positive, "control_dt_s": positive,
            "output_dt_s": positive, "substeps_per_control": integer,
            "stop_policy": {"const": "controller_done_or_timeout"}, "max_duration_s": positive,
            "timestamp": {"const": "interval_end"}, "sequence_start": {"const": 0}}),
        "controller": obj({"mode": {"enum": ["trajectory", "force", "multi_trajectory"]},
            "drive_semantics": {"const": "overwrite_linear_velocity_once_per_control_step"},
            "feedback_delay_steps": {"const": 1}, "parameters": parameters,
            "force": {"anyOf": [force, {"type": "null"}]}, "phase_order": array(text)}),
        "runtime": obj(runtime),
        "grid": obj({"rows": integer, "cols": integer, "cell_size_m": positive, "cell_area_m2": positive,
            "origin_local_m": vec, "row_axis": {"const": "y"}, "col_axis": {"const": "x"},
            "area_method": {"const": "point_occupancy"}, "connectivity": {"const": 4},
            "edge_policy": {"enum": ["phase1_clamp", "reject_outside_top_face"]}}),
        "sampling": obj({"force_threshold_n": nonneg, "point_match_distance_m": positive,
            "max_gap_frames": {"type": "integer", "minimum": 0}, "slide_threshold_mps": positive,
            "direction_threshold_m": positive, "filter_alpha": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
            "kinematics_source": text, "force_definition": {"const": "sum_contact_impulses_over_output_interval"}}),
        "acceptance": obj({"contact_coverage_min": fraction, "slide_speed_tolerance_mps": positive,
            "slide_speed_pass_fraction_min": fraction, "hold_tangent_speed_max_mps": positive,
            "slide_settle_time_s": nonneg, "patch_coverage_min": fraction, "force_conservation_atol_n": positive,
            "resolution_force_spread_max": fraction, "release_contact_frames_max": {"const": 0}}),
        "source_files": {"type": "array", "items": obj({"path": text, "sha256": sha}), "minItems": 1},
        "source_arguments": {"type": "object"}, "parameter_evidence": {"type": "object", "additionalProperties": text},
        "required_run_artifacts": array(text),
    })
    schema.update({"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "K0 Experiment Specification v1"})
    return schema


def source(path):
    return ast.parse((BASE / path).read_text(encoding="utf-8"))


def defaults(path):
    args, evidence = {}, {}
    for node in ast.walk(source(path)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            key = node.args[0].value.removeprefix("--").replace("-", "_")
            for kw in node.keywords:
                if kw.arg == "default":
                    args[key] = ast.literal_eval(kw.value)
                    evidence[key] = f"{path}:{node.lineno} argparse default"
    return args, evidence


def regression_overrides(function, grid):
    tree = source("scripts/run_step6_regression.py")
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    result = {}
    values = ret.value.elts
    for index, node in enumerate(values):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str) or not node.value.startswith("--"):
            continue
        key = node.value.removeprefix("--").replace("-", "_")
        value = values[index + 1]
        if isinstance(value, ast.Constant):
            result[key] = value.value
        elif isinstance(value, ast.Call) and value.args and isinstance(value.args[0], ast.Name):
            result[key] = {"grid": grid, "print_every": 10, "output_dir": "<run_directory>"}[value.args[0].id]
        else:
            raise ValueError(f"Unsupported override at {function}:{getattr(value, 'lineno', '?')}")
    return result, fn.lineno


def build_spec(mode, grid):
    multi = mode == "multi"
    path = f"scripts/step6_{'5' if multi else '3'}.py"
    args, evidence = defaults(path)
    overrides, line = regression_overrides({"multi": "multi_command", "force": "force_command", "trajectory": "trajectory_command"}[mode], grid)
    for key, value in overrides.items():
        old = args[key]
        args[key] = type(old)(value) if old is not None else value
        evidence[key] = f"scripts/run_step6_regression.py:{line} {mode}_command override"
    args["grid"] = grid
    surface = {"id": "/World/Cube", "shape": "box", "motion": "fixed", "center_world_m": [0, 0, .25], "size_m": [.5, .5, .5], "mass_kg": None}
    probes = []
    for index in range(2 if multi else 1):
        zcontact = .5 + (args["probe_height"] / 2 if multi else .12)
        probes.append({"id": ("/World/ProbeLeft" if index == 0 else "/World/ProbeRight") if multi else "/World/Fingertip",
            "shape": "box" if multi else "sphere", "mass_kg": .08 if multi else .05,
            "radius_m": None if multi else .12,
            "size_m": [args["probe_width"], args["probe_depth"], args["probe_height"]] if multi else None,
            "initial_position_world_m": [(-1 if index == 0 else 1) * args["probe_half_separation"] if multi else 0, 0, 1.2],
            "angular_mode": "locked", "z_contact_m": zcontact, "retract_distance_m": 1.2-zcontact})
    runtime = {key: {"value": None, "status": "unresolved", "evidence": "Not explicitly set in scene. Capture effective runtime value before numeric parity claims."} for key in make_schema()["properties"]["runtime"]["properties"]}
    if multi:
        for key in ("static_friction", "dynamic_friction", "restitution"):
            runtime[key]["evidence"] = f"{path}:250 PhysicsMaterial requested {args.get(key, 0.0)} but exceptions are caught. Verify applied material."
    params = {name + "_s": args[name] for name in ("pre_settle_time", "approach_timeout", "hold_time", "slide_time", "release_time")}
    params.update({name + "_mps": args[name] for name in ("approach_speed", "slide_speed", "retract_speed", "contact_maintain_speed", "max_slide_command_speed")})
    params.update(gravity_compensation_mps2=9.81, tangent_speed_feedback_kp=args["tangent_speed_feedback_kp"], initial_slide_command_speed_mps=args.get("initial_slide_command_speed"))
    force = None
    if mode == "force":
        force = dict(target_force_n=args["target_force"], force_tolerance_n=args["force_tolerance"], force_kp_m_per_ns=args["force_kp"], force_ki_m_per_ns2=args["force_ki"], force_integral_limit_ns=args["force_integral_limit"], max_press_speed_mps=args["max_press_speed"], max_correct_speed_mps=args["max_correct_speed"], settle_frames=args["settle_frames"], regulate_timeout_s=args["regulate_timeout"])
    files = [path, "scripts/run_step6_regression.py", "src/nerv_tactile/controllers.py", "src/nerv_tactile/interfaces.py", "src/nerv_tactile/geometry.py", "src/nerv_tactile/contact_reader.py", "src/nerv_tactile/aggregation.py", "src/nerv_tactile/tracking.py", "environment/isaacsim_4090.json"]
    data = {
        "schema_version": "experiment-spec/v1", "spec_id": f"step6_{mode}_{grid}", "kind": "source_inventory", "phase": "phase2",
        "description": "Source-derived intended regression configuration, not measured simulation evidence.", "backend_id": "isaac_sim_step6",
        "coordinates": {"units": "SI", "handedness": "right", "up_axis": "Z", "quaternion_order": "wxyz", "basis_world": [[1,0,0],[0,1,0],[0,0,1]], "local_origin_world_m": [0,0,.25], "surface_z_world_m": .5, "normal_world": [0,0,1]},
        "surface": surface, "probes": probes,
        "streams": [{"id": f"probe{index}_surface", "body0": p["id"], "body1": surface["id"], "frame_schema": "interaction-state/v2", "path_template": f"streams/probe{index}_surface.jsonl"} for index,p in enumerate(probes)],
        "timing": {"start_time_s": 0, "physics_dt_s": 1/60, "control_dt_s": 1/60, "output_dt_s": 1/60, "substeps_per_control": 1, "stop_policy": "controller_done_or_timeout", "max_duration_s": 40 if mode == "force" else 20, "timestamp": "interval_end", "sequence_start": 0},
        "controller": {"mode": "multi_trajectory" if multi else mode, "drive_semantics": "overwrite_linear_velocity_once_per_control_step", "feedback_delay_steps": 1, "parameters": params, "force": force, "phase_order": (["pre_contact", "approach_force", "regulate_force", "lateral_slide_force", "retract", "released"] if mode == "force" else ["pre_contact", "approach", "hold", "lateral_slide", "retract", "released"])},
        "runtime": runtime,
        "grid": {"rows": grid, "cols": grid, "cell_size_m": .5/grid, "cell_area_m2": (.5/grid)**2, "origin_local_m": [-.25,-.25,.25], "row_axis": "y", "col_axis": "x", "area_method": "point_occupancy", "connectivity": 4, "edge_policy": "phase1_clamp"},
        "sampling": {"force_threshold_n": args["force_threshold"], "point_match_distance_m": args["match_distance"], "max_gap_frames": args["max_gap_frames"], "slide_threshold_mps": args["slide_threshold"], "direction_threshold_m": args.get("direction_threshold", 1e-4), "filter_alpha": args.get("filter_alpha", .5), "kinematics_source": args.get("kinematics_source", "contact_point_finite_difference"), "force_definition": "sum_contact_impulses_over_output_interval"},
        "acceptance": {"contact_coverage_min": args["min_contact_coverage"], "slide_speed_tolerance_mps": args["slide_speed_tolerance"], "slide_speed_pass_fraction_min": args.get("min_slide_coverage", args.get("min_slide_speed_in_tolerance")), "hold_tangent_speed_max_mps": args.get("max_hold_tangent_speed", .005), "slide_settle_time_s": args.get("slide_settle_time", 0), "patch_coverage_min": .90, "force_conservation_atol_n": 1e-5, "resolution_force_spread_max": .10, "release_contact_frames_max": 0},
        "source_files": [{"path": file, "sha256": hashlib.sha256((BASE/file).read_bytes()).hexdigest()} for file in files],
        "source_arguments": args, "parameter_evidence": evidence | {"geometry": f"{path} module constants/create_scene", "runtime": "unresolved means engine defaults or constructor defaults not captured", "max_duration_s": "Phase 2 watchdog budget, not a historical run duration"},
        "required_run_artifacts": ["effective_runtime.json", "actual_commands.jsonl", "stream_index.json", "streams/*.jsonl", "points.csv", "cells.csv", "patches.csv", "episodes.csv", "kinematics.jsonl", "acceptance.json", "environment.json"],
    }
    return data


#: Explicit CPU-reference runtime. These are design choices, not claims about
#: unresolved PhysX defaults. The evidence string is kept verbatim because the
#: K1 specification hash is frozen and feeds every recorded conditions digest.
CPU_REFERENCE_RUNTIME = {
    "gravity_mps2": [0, 0, -9.81], "static_friction": 0, "dynamic_friction": 0, "restitution": 0,
    "linear_damping_per_s": 0, "angular_damping_per_s": 0, "initial_linear_velocity_mps": [0, 0, 0],
    "initial_orientation_wxyz": [1, 0, 0, 0], "contact_offset_m": 0, "rest_offset_m": 0,
    "solver_type": "single_sphere_plane_normal_impulse", "position_iterations": 1, "velocity_iterations": 1,
    "sleep_enabled": False, "ccd_mode": "linear_sweep_in_drift", "friction_combine_mode": "explicit_pair",
    "restitution_combine_mode": "explicit_pair", "max_depenetration_velocity_mps": 0,
}
CPU_REFERENCE_EVIDENCE = "mechanics/designs/k1_single_probe_rigid.md"
CPU_REFERENCE_DESCRIPTION = (
    "Explicit CPU reference choices. No claim that unknown PhysX defaults have these values."
)


def to_cpu_reference(source, spec_id, description=CPU_REFERENCE_DESCRIPTION):
    """Turn a source inventory into an explicit CPU reference specification."""
    data = copy.deepcopy(source.to_dict())
    data.update(spec_id=spec_id, kind="cpu_reference", backend_id="rigid_cpu_v1",
                description=description)
    data["runtime"] = {
        key: {"value": value, "status": "design_choice", "evidence": CPU_REFERENCE_EVIDENCE}
        for key, value in CPU_REFERENCE_RUNTIME.items()
    }
    data["grid"]["edge_policy"] = "reject_outside_top_face"
    data["parameter_evidence"]["source_arguments"] = (
        "Inherited source args for traceability. Structured fields define this CPU reference."
    )
    return data


def force_cpu_reference(source):
    """K2 force-mode CPU reference with a press limit that can reach its target.

    Phase 1 regulated 1 N with a 0.25 m/s press limit because PhysX reports
    contact impulse that also carries penetration recovery. This kernel removes
    penetration by geometric projection and records it apart from the contact
    impulse, so under ``F = mass * commanded_speed / output_dt`` a 0.25 m/s
    limit reaches only 0.75 N and the declared 1 N target is unreachable.

    The press limit is therefore derived from the target rather than copied
    from Phase 1. The difference is a real difference between the two
    backends, not a parameter to transplant.
    """
    from virtual_reality.drive_limits import DERIVED_HEADROOM_FACTOR, derive_press_limit_mps

    data = to_cpu_reference(
        source, "k2_force_cpu",
        "Explicit CPU reference for force control. The press limit is derived "
        "from the force target under this kernel's drive semantics, not copied "
        "from Phase 1.",
    )
    force = data["controller"]["force"]
    probe = data["probes"][0]
    force["max_press_speed_mps"] = derive_press_limit_mps(
        mass_kg=probe["mass_kg"],
        target_force_n=force["target_force_n"],
        output_dt_s=data["timing"]["output_dt_s"],
    )
    data["parameter_evidence"]["max_press_speed"] = (
        f"Derived: {DERIVED_HEADROOM_FACTOR} x target_force_n under "
        "F = mass * commanded_speed / output_dt. Phase 1's 0.25 m/s reaches "
        "only 0.75 N on this kernel because penetration recovery is excluded "
        "from the contact impulse here."
    )
    data["parameter_evidence"]["max_correct_speed"] = (
        "Kept from Phase 1. An upward command separates the probe and clears "
        "the contact force, so this limit does not bound the reachable force."
    )
    data["parameter_evidence"]["force_kp"] = (
        "Inherited from Phase 1 and not yet tuned for this kernel. Under these "
        "drive semantics the force responds to the command within one step "
        "rather than accumulating, so K2 must verify the gains."
    )
    data["parameter_evidence"]["force_ki"] = data["parameter_evidence"]["force_kp"]
    return data


def main():
    write(ROOT / "tactile_contract/schemas/experiment-spec-v1.schema.json", make_schema())
    from tactile_contract.experiment_spec import ExperimentSpec
    specs = {}
    for mode, grid in [("trajectory",32),("trajectory",64),("trajectory",128),("force",64),("multi",64)]:
        spec = ExperimentSpec.build(**build_spec(mode,grid))
        write(ROOT / f"scenarios/k0/{spec.spec_id}.json", spec.to_dict())
        specs[(mode, grid)] = spec
    for spec_id, data in (
        ("k1_single_probe_cpu", to_cpu_reference(specs[("trajectory", 64)], "k1_single_probe_cpu")),
        ("k2_force_cpu", force_cpu_reference(specs[("force", 64)])),
    ):
        reference = ExperimentSpec.build(**data)
        write(ROOT / f"scenarios/k0/{spec_id}.json", reference.to_dict())
    print("Wrote 5 source inventories and 2 explicit CPU references.")


if __name__ == "__main__":
    main()
