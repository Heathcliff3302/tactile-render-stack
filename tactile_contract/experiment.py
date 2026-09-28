"""Run manifest, frame binding, and comparison eligibility."""

import math
from pathlib import PurePosixPath

from ._common import JsonRecord, digest
from .interaction import OBSERVABLE_GROUP, OBSERVABLE_UNITS, check_basis

EXPERIMENT_MANIFEST_SCHEMA = "experiment-manifest/v1"


class ExperimentManifest(JsonRecord):
    __slots__ = ()
    schema_file = "experiment-manifest-v1.schema.json"

    @classmethod
    def build(cls, **data):
        data["conditions_sha256"] = digest(data["conditions"])
        return cls(**data)

    @staticmethod
    def validate_semantics(data):
        conditions = data["conditions"]
        if data["conditions_sha256"] != digest(conditions):
            raise ValueError("conditions_sha256 mismatch")
        check_basis(conditions["reference_frame"]["basis_world"])
        bodies = conditions["bodies"]
        if bodies[0]["id"] == bodies[1]["id"]:
            raise ValueError("Experiment bodies must differ")
        timing = conditions["timing"]
        if timing["end_time_s"] <= timing["start_time_s"]:
            raise ValueError("Invalid experiment duration")
        samples = conditions["trajectory"]["samples"]
        times = [sample[0] for sample in samples]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Trajectory times must increase")
        if times[0] != timing["start_time_s"] or times[-1] != timing["end_time_s"]:
            raise ValueError("Trajectory must span the experiment interval")
        for name, definition in data["observables"].items():
            if definition["unit"] != OBSERVABLE_UNITS[name]:
                raise ValueError(f"Wrong unit for {name}")
            if definition["group"] != OBSERVABLE_GROUP[name]:
                raise ValueError(f"Wrong quality group for {name}")
        for field in data["acceptance"]["required_observables"]:
            if field not in data["observables"]:
                raise ValueError(f"Required observable lacks a definition: {field}")
        paths = set()
        for artifact in data["artifacts"]:
            path = PurePosixPath(artifact["path"])
            if path.is_absolute() or ".." in path.parts or "\\" in artifact["path"] or ":" in artifact["path"]:
                raise ValueError("Artifact path must stay relative to its run directory")
            if artifact["path"] in paths:
                raise ValueError("Duplicate artifact path")
            paths.add(artifact["path"])
        if sum(a["role"] == "interaction_frames" for a in data["artifacts"]) != 1:
            raise ValueError("Exactly one interaction_frames artifact is required")


def validate_run(manifest, frames):
    """Validate a complete single-pair stream, including no-contact frames."""
    m = manifest.to_dict()
    frames = list(frames)
    timing = m["conditions"]["timing"]
    if len(frames) != m["acceptance"]["expected_frames"]:
        raise ValueError("Frame count differs from acceptance requirement")
    previous = None
    for index, frame in enumerate(frames):
        d = frame.to_dict()
        for key in ("run_id", "scenario_id", "conditions_sha256"):
            expected = m["scenario"]["id"] if key == "scenario_id" else m[key]
            if d[key] != expected:
                raise ValueError(f"Frame {key} differs from manifest")
        if d["source_backend"] != m["backend"]["id"]:
            raise ValueError("Frame backend differs from manifest")
        if [d["body0"], d["body1"]] != [b["id"] for b in m["conditions"]["bodies"]]:
            raise ValueError("Frame body order differs from manifest")
        if d["sequence_id"] != index:
            raise ValueError("Sequence must start at zero and be contiguous")
        if not math.isclose(d["dt_s"], timing["output_dt_s"], rel_tol=0, abs_tol=1e-10):
            raise ValueError("Frame output interval differs from manifest")
        expected_time = timing["start_time_s"] + (index + 1) * timing["output_dt_s"]
        if not math.isclose(d["time_s"], expected_time, rel_tol=0, abs_tol=1e-9):
            raise ValueError("Frame timestamp is not the end of the expected interval")
        if previous is not None and d["time_s"] <= previous:
            raise ValueError("Non-monotonic frame time")
        previous = d["time_s"]
        for name, quality in d["quality"].items():
            if name == "contact" or quality["status"] == "unavailable":
                continue
            if quality["status"] != "unavailable":
                if name not in m["observables"]:
                    raise ValueError(f"Observable {name} has no experiment definition")
                definitions = [m["observables"][name]]
                if not definitions or quality["method"] not in {definition["method"] for definition in definitions}:
                    raise ValueError(f"Quality method for {name} differs from manifest")
        for name in m["acceptance"]["required_observables"]:
            if d[name] is None:
                raise ValueError(f"Required observable unavailable: {name}")
    if not math.isclose(previous, timing["end_time_s"], rel_tol=0, abs_tol=1e-9):
        raise ValueError("Stream does not cover experiment end")
    return {"frames": len(frames), "run_id": m["run_id"], "contract_passed": True}


def comparison_gate(left, right, metrics):
    """Eligibility only. This does not claim physical agreement."""
    a, b = left.to_dict(), right.to_dict()
    reasons = []
    if not metrics or len(set(metrics)) != len(metrics):
        reasons.append("metrics must be nonempty and unique")
    if a["scenario"] != b["scenario"]:
        reasons.append("scenario identity/version differs")
    if a["conditions_sha256"] != b["conditions_sha256"]:
        reasons.append("physical inputs or time/frame conventions differ")
    for name in metrics:
        x, y = a["observables"].get(name), b["observables"].get(name)
        if x is None or y is None:
            reasons.append(f"{name}: unavailable in a backend")
        elif any(x[key] != y[key] for key in ("unit", "definition_id", "support_id")):
            reasons.append(f"{name}: unit, physical definition, or support differs")
    return {"eligible": not reasons, "reasons": reasons, "metrics": list(metrics),
            "scope": "declared conditions only; validate both streams before comparison"}
