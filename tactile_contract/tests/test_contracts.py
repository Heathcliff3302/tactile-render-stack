import hashlib
import json

import pytest

from tactile_contract import ExperimentManifest, InteractionState, comparison_gate, validate_run
from tactile_contract._common import canonical_json, digest
from tactile_contract.interaction import OBSERVABLE_UNITS


def quality(status, method, unit="mixed", source="fixture"):
    return {"status": status, "method": method, "unit": unit, "source": source}


def frame(index, *, backend="reduced_contact", force=1.0, method="analytic_sphere_plane"):
    values = {
        "schema_version": "interaction-state/v2",
        "run_id": "run-001",
        "scenario_id": "sphere-plane-v1",
        "conditions_sha256": "0" * 64,
        "source_backend": backend,
        "body0": "probe",
        "body1": "surface",
        "time_s": (index + 1) * 0.001,
        "sequence_id": index,
        "dt_s": 0.001,
        "contact_present": True,
        "contact_mode": "hold",
        "coordinate_frame": "world",
        "contact_centroid_world_m": [0.0, 0.0, 0.0],
        "contact_centroid_local_m": [0.0, 0.0, 0.0],
        "normal_world": [0.0, 0.0, 1.0],
        "tangent_basis_world": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "contact_point_count": 1,
        "active_patch_count": 1,
        "contact_area_proxy_m2": 0.0001,
        "physical_contact_area_m2": None,
        "normal_force_n": force,
        "tangent_force_world_n": [0.0, 0.0, 0.0],
        "total_force_world_n": [0.0, 0.0, force],
        "normal_impulse_ns": force * 0.001,
        "normal_relative_velocity_mps": 0.0,
        "tangent_velocity_world_mps": [0.0, 0.0, 0.0],
        "indentation_m": 0.0001,
        "penetration_m": 0.0,
        "deformation_max_m": None,
        "deformation_mean_m": None,
        "deformation_rms_m": None,
        "deformation_field": None,
        "pressure_mean_pa": force / 0.0001,
        "pressure_peak_pa": force / 0.0001,
        "pressure_field": None,
        "quality": {
            "contact": quality("measured", "contact_threshold", "1"),
            **{name: quality("unavailable", "not_available", unit) for name, unit in OBSERVABLE_UNITS.items()},
        },
        "confidence": 1.0,
        "metadata": {"fixture": True},
    }
    values["quality"].update({
        "contact_centroid_world_m": quality("measured", "contact_geometry", "m"),
        "normal_world": quality("measured", "contact_geometry", "1"),
        "tangent_force_world_n": quality("measured", "impulse_over_dt", "N"),
        "total_force_world_n": quality("measured", "impulse_over_dt", "N"),
        "tangent_velocity_world_mps": quality("measured", "finite_difference", "m/s"),
        "contact_area_proxy_m2": quality("measured", "grid_support_area", "m2"),
        "normal_force_n": quality("measured", "impulse_over_dt", "N"),
        "normal_impulse_ns": quality("measured", "impulse_integral", "N*s"),
        "normal_relative_velocity_mps": quality("measured", "finite_difference", "m/s"),
        "indentation_m": quality("measured", "finite_difference", "m"),
        "penetration_m": quality("measured", "geometry_overlap", "m"),
        "pressure_mean_pa": quality("estimated", "force_over_support_area", "Pa"),
        "pressure_peak_pa": quality("estimated", "force_over_support_area", "Pa"),
    })
    return InteractionState.from_dict(values)


def conditions():
    return {
        "reference_frame": {"id": "surface_frame", "basis_world": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        "bodies": [{"id": "probe", "role": "probe"}, {"id": "surface", "role": "surface"}],
        "timing": {"start_time_s": 0.0, "end_time_s": 0.004, "physics_dt_s": 0.0005, "output_dt_s": 0.001},
        "trajectory": {"id": "approach-hold-release-v1", "samples": [[0.0, [0, 0, 0.01], [0, 0, -0.01]], [0.004, [0, 0, 0.009], [0, 0, 0]]]},
        "material": {"id": "rigid-v1", "model": "rigid_reference", "parameters": {}},
        "probe": {"id": "sphere-10mm", "geometry": "sphere", "parameters": {"radius_m": 0.005}},
    }


def manifest(backend="reduced_contact", methods=None):
    methods = methods or {}
    observable_groups = {
        "contact_centroid_world_m": ("geometry", "m", "contact_geometry"),
        "normal_force_n": ("force", "N", "impulse_over_dt"),
        "indentation_m": ("kinematics", "m", "finite_difference"),
        "contact_area_proxy_m2": ("geometry", "m2", "grid_support_area"),
        "normal_world": ("geometry", "1", "contact_geometry"),
        "tangent_force_world_n": ("force", "N", "impulse_over_dt"),
        "total_force_world_n": ("force", "N", "impulse_over_dt"),
        "normal_impulse_ns": ("force", "N*s", "impulse_integral"),
        "normal_relative_velocity_mps": ("kinematics", "m/s", "finite_difference"),
        "tangent_velocity_world_mps": ("kinematics", "m/s", "finite_difference"),
        "penetration_m": ("kinematics", "m", "geometry_overlap"),
        "deformation_max_m": ("deformation", "m", "not_available"),
        "pressure_mean_pa": ("pressure", "Pa", "force_over_support_area"),
        "pressure_peak_pa": ("pressure", "Pa", "force_over_support_area"),
    }
    observables = {
        name: {"definition_id": name + "-v1", "unit": unit, "method": methods.get(name, method), "support_id": "surface_grid-v1", "group": group}
        for name, (group, unit, method) in observable_groups.items()
    }
    c = conditions()
    return ExperimentManifest.build(
        schema_version="experiment-manifest/v1",
        run_id="run-001",
        scenario={"id": "sphere-plane-v1", "version": "1.0.0", "description": "Fixture"},
        conditions_sha256="0" * 64,
        backend={"id": backend, "version": "0.1.0"},
        conditions=c,
        observables=observables,
        acceptance={"expected_frames": 4, "required_observables": ["normal_force_n", "pressure_mean_pa"], "max_missing_frame_fraction": 0.0},
        artifacts=[{"path": "interaction.jsonl", "role": "interaction_frames", "sha256": "0" * 64}],
    )


def test_manifest_binds_conditions_and_frames():
    m = manifest()
    sha = digest(conditions())
    assert m.conditions_sha256 == sha
    frames = [frame(i) for i in range(4)]
    for item in frames:
        item_dict = item.to_dict()
        item_dict["conditions_sha256"] = sha
        item = InteractionState.from_dict(item_dict)
    # Rebuild after replacing the frame hash to exercise the actual validator.
    frames = []
    for i in range(4):
        item = frame(i).to_dict()
        item["conditions_sha256"] = sha
        frames.append(InteractionState.from_dict(item))
    assert validate_run(m, frames)["contract_passed"]


def test_manifest_rejects_wrong_frame_method():
    m = manifest()
    sha = digest(conditions())
    items = []
    for index in range(4):
        item = frame(index).to_dict()
        item["conditions_sha256"] = sha
        items.append(item)
    items[0]["quality"]["normal_force_n"]["method"] = "wrong_method"
    with pytest.raises(ValueError, match="Quality method"):
        validate_run(m, [InteractionState.from_dict(item) for item in items])


def test_comparison_gate_requires_same_conditions_and_support():
    left = manifest("reduced_contact")
    right = manifest("isaac_sim")
    result = comparison_gate(left, right, ["normal_force_n"])
    assert result["eligible"]
    assert comparison_gate(left, left, ["normal_force_n"])["eligible"]


def test_no_contact_frame_cannot_claim_force_or_pressure():
    item = frame(0).to_dict()
    item["contact_present"] = False
    item["contact_mode"] = "no_contact"
    item["normal_force_n"] = None
    item["total_force_world_n"] = None
    item["normal_impulse_ns"] = None
    item["pressure_mean_pa"] = None
    item["pressure_peak_pa"] = None
    item["contact_area_proxy_m2"] = None
    item["quality"]["normal_force_n"] = quality("unavailable", "not_available", "N")
    item["quality"]["total_force_world_n"] = quality("unavailable", "not_available", "N")
    item["quality"]["normal_impulse_ns"] = quality("unavailable", "not_available", "N*s")
    item["quality"]["pressure_mean_pa"] = quality("unavailable", "not_available", "Pa")
    item["quality"]["pressure_peak_pa"] = quality("unavailable", "not_available", "Pa")
    item["quality"]["contact_area_proxy_m2"] = quality("unavailable", "not_available", "m2")
    assert InteractionState.from_dict(item).contact_present is False

    item["normal_force_n"] = 0.5
    item["quality"]["normal_force_n"] = quality("measured", "bad", "N")
    with pytest.raises(ValueError, match="No-contact frame"):
        InteractionState.from_dict(item)


def test_invalid_nan_and_duplicate_json_keys_are_rejected():
    with pytest.raises(ValueError):
        from tactile_contract._common import loads
        loads('{"schema_version": NaN}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        from tactile_contract._common import loads
        loads('{"x": 1, "x": 2}')
