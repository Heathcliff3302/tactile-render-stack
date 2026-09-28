"""Versioned pre-run specification. Kept separate from completed run evidence."""

import math

from ._common import JsonRecord, digest
from .interaction import check_basis

EXPERIMENT_SPEC_SCHEMA = "experiment-spec/v1"


class ExperimentSpec(JsonRecord):
    __slots__ = ()
    schema_file = "experiment-spec-v1.schema.json"

    @classmethod
    def build(cls, **data):
        data.pop("spec_sha256", None)
        data["spec_sha256"] = digest(data)
        return cls(**data)

    @staticmethod
    def validate_semantics(data):
        body = {k: v for k, v in data.items() if k != "spec_sha256"}
        if data["spec_sha256"] != digest(body):
            raise ValueError("spec_sha256 mismatch")
        frame = data["coordinates"]
        check_basis(frame["basis_world"])
        if data["surface"]["motion"] != "fixed":
            raise ValueError("K0 supports a fixed reference surface")
        if data["surface"]["center_world_m"][2] + data["surface"]["size_m"][2]/2 != frame["surface_z_world_m"]:
            raise ValueError("Surface top and reference coordinate disagree")
        if frame["local_origin_world_m"] != data["surface"]["center_world_m"]:
            raise ValueError("Phase 1 local origin is the cube center")
        ids = [p["id"] for p in data["probes"]]
        if len(set(ids)) != len(ids) or data["surface"]["id"] in ids:
            raise ValueError("Body IDs must be unique")
        if [s["body0"] for s in data["streams"]] != ids or any(s["body1"] != data["surface"]["id"] for s in data["streams"]):
            raise ValueError("Exactly one ordered stream per probe is required")
        if len({s["id"] for s in data["streams"]}) != len(ids):
            raise ValueError("Stream IDs must be unique")
        t = data["timing"]
        if not math.isclose(t["control_dt_s"], t["output_dt_s"], abs_tol=1e-12):
            raise ValueError("K0 records every control step")
        if not math.isclose(t["physics_dt_s"] * t["substeps_per_control"], t["control_dt_s"], abs_tol=1e-12):
            raise ValueError("Substeps must cover one control interval")
        for name, setting in data["runtime"].items():
            if (setting["status"] == "unresolved") != (setting["value"] is None):
                raise ValueError(f"{name}: unresolved/value mismatch")
        gravity = data["runtime"]["gravity_mps2"]["value"]
        if gravity is not None and gravity[:2] != [0, 0]:
            raise ValueError("K1 only supports gravity along world Z")
        if data["kind"] == "cpu_reference" and unresolved_settings(data):
            raise ValueError("CPU reference must resolve every runtime setting")
        grid = data["grid"]
        if grid["cols"] != grid["rows"]:
            raise ValueError("Step 6 grid is square")
        expected = data["surface"]["size_m"][0] / grid["cols"]
        if not math.isclose(grid["cell_size_m"], expected, abs_tol=1e-12):
            raise ValueError("Grid size does not match surface extent")
        if not math.isclose(grid["cell_area_m2"], expected**2, abs_tol=1e-12):
            raise ValueError("Grid area does not match cell size")
        for probe in data["probes"]:
            offset = probe["radius_m"] if probe["shape"] == "sphere" else probe["size_m"][2]/2
            if not math.isclose(probe["z_contact_m"], frame["surface_z_world_m"] + offset, abs_tol=1e-12):
                raise ValueError("Contact height disagrees with shape")
            if not math.isclose(probe["retract_distance_m"], probe["initial_position_world_m"][2] - probe["z_contact_m"], abs_tol=1e-12):
                raise ValueError("Retract distance mismatch")
        control = data["controller"]["parameters"]
        if control["max_slide_command_speed_mps"] < control["slide_speed_mps"]:
            raise ValueError("Slide command limit is below target")
        force = data["controller"]["force"]
        if (data["controller"]["mode"] == "force") != (force is not None):
            raise ValueError("Force controller configuration mismatch")


def unresolved_settings(spec):
    data = spec.to_dict() if isinstance(spec, ExperimentSpec) else spec
    return sorted(name for name, value in data["runtime"].items() if value["status"] == "unresolved")


def spec_comparison_gate(left, right, profile="same_controller"):
    """Conservative eligibility gate. Does not score simulation results."""
    if profile not in ("same_controller", "grid_refinement", "time_refinement"):
        raise ValueError("Unknown comparison profile")
    a, b = left.to_dict(), right.to_dict()
    reasons = []
    for label, spec in (("left", a), ("right", b)):
        missing = unresolved_settings(spec)
        if missing:
            reasons.append(f"{label}: unresolved runtime settings: {', '.join(missing)}")
    # Source paths, backend IDs, and numerical choices are not physical inputs.
    for key in ("coordinates", "surface", "probes", "controller"):
        if a[key] != b[key]:
            reasons.append(f"{key} differs")
    physical = ("gravity_mps2", "static_friction", "dynamic_friction", "restitution",
                "linear_damping_per_s", "initial_linear_velocity_mps", "initial_orientation_wxyz")
    for key in physical:
        if a["runtime"][key]["value"] != b["runtime"][key]["value"]:
            reasons.append(f"physical setting {key} differs")
    if a["sampling"] != b["sampling"]:
        reasons.append("sampling/threshold definitions differ")
    if profile != "grid_refinement" and a["grid"] != b["grid"]:
        reasons.append("grid differs outside a grid_refinement comparison")
    if profile != "time_refinement" and a["timing"] != b["timing"]:
        reasons.append("timing differs outside a time_refinement comparison")
    return {"eligible": not reasons, "profile": profile, "reasons": reasons,
            "scope": "pre-run configuration eligibility, not physical equivalence"}
