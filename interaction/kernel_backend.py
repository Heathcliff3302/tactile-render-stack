"""Layer 2 adapter: contact kernel frames to ``InteractionState v2``.

The adapter does units, frames, nullability, aggregation hand-off and quality
marking. It performs no physics. Two rules keep the records honest:

* A field is ``unavailable`` whenever the kernel has no value for it. No
  fabricated zero ever enters a comparison.
* Each observable keeps one constant computation method for a whole run, which
  is what ``validate_run`` binds against the experiment manifest.
"""

from __future__ import annotations

from contact_kernel import KINEMATICS_FINITE_DIFFERENCE, KINEMATICS_METHOD
from tactile_contract.interaction import OBSERVABLE_GROUP, OBSERVABLE_UNITS, InteractionState

BACKEND_METHOD_SOURCE = "rigid_cpu_v1"
NOT_AVAILABLE = "not_available"

#: Observables whose computation method is set by the declared kinematics
#: source rather than fixed by the solver.
KINEMATICS_DEPENDENT = ("tangent_velocity_world_mps", "normal_relative_velocity_mps")

#: One computation method per observable for the whole run. ``None`` means the
#: K1 rigid mode cannot produce the quantity at all. Entries named in
#: ``KINEMATICS_DEPENDENT`` are placeholders resolved by
#: :func:`observable_methods`; read that rather than this table directly.
OBSERVABLE_METHODS: dict[str, str | None] = {
    "contact_centroid_world_m": "surface_projection_of_probe_axis",
    "normal_world": "analytic_top_face_normal",
    "tangent_force_world_n": "declared_zero_friction",
    "total_force_world_n": "normal_force_plus_tangent_force",
    "tangent_velocity_world_mps": "rigid_body_twist",
    "contact_area_proxy_m2": "grid_point_occupancy_proxy",
    "physical_contact_area_m2": None,
    "normal_force_n": "interval_impulse_over_output_dt",
    "normal_impulse_ns": "rigid_nonpenetration_impulse",
    "normal_relative_velocity_mps": "rigid_body_twist",
    "indentation_m": "rigid_constraint_zero_indentation",
    "penetration_m": "numerical_penetration_correction",
    "deformation_max_m": None,
    "deformation_mean_m": None,
    "deformation_rms_m": None,
    "deformation_field": None,
    "pressure_mean_pa": "grid_occupancy_proxy_pressure",
    "pressure_peak_pa": "grid_occupancy_proxy_pressure",
    "pressure_field": None,
}

#: ``measured`` marks a value the authoritative solver produces directly.
#: ``estimated`` marks a value that needs an extra assumption on top of it.
OBSERVABLE_STATUS: dict[str, str] = {
    "contact_centroid_world_m": "measured",
    "normal_world": "measured",
    "tangent_force_world_n": "measured",
    "total_force_world_n": "measured",
    "tangent_velocity_world_mps": "measured",
    "normal_force_n": "measured",
    "normal_impulse_ns": "measured",
    "normal_relative_velocity_mps": "measured",
    "penetration_m": "measured",
    "contact_area_proxy_m2": "estimated",
    "indentation_m": "estimated",
    "pressure_mean_pa": "estimated",
    "pressure_peak_pa": "estimated",
}

#: Support region each observable is defined on.
OBSERVABLE_SUPPORT: dict[str, str] = {
    "contact_centroid_world_m": "single_contact_point",
    "normal_world": "single_contact_point",
    "tangent_force_world_n": "single_contact_point",
    "total_force_world_n": "single_contact_point",
    "tangent_velocity_world_mps": "single_contact_point",
    "normal_force_n": "single_contact_point",
    "normal_impulse_ns": "single_contact_point",
    "normal_relative_velocity_mps": "single_contact_point",
    "penetration_m": "single_contact_point",
    "contact_area_proxy_m2": "occupied_output_grid_cells",
    "indentation_m": "single_contact_point",
    "pressure_mean_pa": "occupied_output_grid_cells",
    "pressure_peak_pa": "occupied_output_grid_cells",
}

PRESSURE_SEMANTICS = (
    "normal force divided by the area of occupied output cells; a resolution "
    "dependent proxy, not a physical contact pressure"
)

#: Observables present on every frame, contact or not, under rigid-body
#: kinematics. Differencing a contact point needs that point in two
#: consecutive frames, so both kinematics observables leave this set for the
#: finite-difference source.
ALWAYS_AVAILABLE = (
    "contact_centroid_world_m",
    "tangent_force_world_n",
    "total_force_world_n",
    "tangent_velocity_world_mps",
    "normal_relative_velocity_mps",
    "indentation_m",
    "penetration_m",
)


def observable_methods(kinematics_source: str) -> dict[str, str | None]:
    """Resolve the per-observable method table for one declared source."""
    if kinematics_source not in KINEMATICS_METHOD:
        raise ValueError(f"Unknown kinematics source {kinematics_source!r}")
    methods = dict(OBSERVABLE_METHODS)
    for name in KINEMATICS_DEPENDENT:
        methods[name] = KINEMATICS_METHOD[kinematics_source]
    return methods


def always_available(kinematics_source: str) -> tuple[str, ...]:
    """Observables required on every frame for one declared source."""
    if kinematics_source == KINEMATICS_FINITE_DIFFERENCE:
        return tuple(name for name in ALWAYS_AVAILABLE if name not in KINEMATICS_DEPENDENT)
    return ALWAYS_AVAILABLE


def manifest_observables(kinematics_source: str) -> dict[str, dict]:
    """Observable definitions for an ``ExperimentManifest``, from the same table.

    The adapter and the manifest read one table, so a method string cannot
    drift between the frames and the experiment definition that binds them.
    """
    definitions = {}
    for name, method in observable_methods(kinematics_source).items():
        if method is None:
            continue
        definitions[name] = {
            "definition_id": f"k1_rigid_cpu.{name}",
            "unit": OBSERVABLE_UNITS[name],
            "method": method,
            "support_id": OBSERVABLE_SUPPORT[name],
            "group": OBSERVABLE_GROUP[name],
        }
    return definitions


def _contact_mode(phase: str, contact_present: bool) -> str:
    if not contact_present:
        return "no_contact" if phase in ("pre_contact", "released") else "approach"
    if phase == "hold":
        return "hold"
    if phase == "lateral_slide":
        return "sliding"
    if phase == "retract":
        return "unloading"
    return "loading"


class KernelInteractionStage:
    """Layer 2: one ``InteractionState`` per output step for one contact pair."""

    layer = 2
    stage_id = "rigid_cpu_kernel_adapter"

    def __init__(self, *, run_id: str, scenario_id: str, conditions_sha256: str,
                 backend_id: str, pair_key: str, kinematics_source: str):
        self.run_id = run_id
        self.scenario_id = scenario_id
        self.conditions_sha256 = conditions_sha256
        self.backend_id = backend_id
        self.pair_key = pair_key
        self.kinematics_source = kinematics_source
        self.methods = observable_methods(kinematics_source)
        self.always_available = always_available(kinematics_source)
        self.last_tangent_speed_mps = 0.0
        self._degraded_reason = ""

    def reset(self) -> None:
        self.last_tangent_speed_mps = 0.0
        self._degraded_reason = ""

    def status(self) -> str:
        return "degraded" if self._degraded_reason else "ok"

    def step(self, world_frame) -> InteractionState:
        kernel_frame = world_frame.kernel_frame
        if kernel_frame is None:
            raise ValueError("Layer 2 requires a kernel frame from layer 1")
        pair = kernel_frame.pairs[self.pair_key]
        if pair.kinematics_source != self.kinematics_source:
            raise ValueError(
                f"Kernel reports {pair.kinematics_source!r} kinematics but this "
                f"adapter declares {self.kinematics_source!r}"
            )
        self.last_tangent_speed_mps = pair.tangent_speed_mps
        self._degraded_reason = pair.filtered_reason if pair.solver_contact and not pair.contact_present else ""
        contact = pair.contact_present
        area = pair.area_layers.point_occupancy_m2 if contact else None
        force = pair.normal_force_n if contact else None
        pressure = (force / area) if (contact and area) else None
        values = {
            "schema_version": "interaction-state/v2",
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "conditions_sha256": self.conditions_sha256,
            "source_backend": self.backend_id,
            "body0": pair.body0,
            "body1": pair.body1,
            "time_s": kernel_frame.time_s,
            "sequence_id": kernel_frame.control_step - 1,
            "dt_s": kernel_frame.dt_s,
            "contact_present": contact,
            "contact_mode": _contact_mode(pair.phase, contact),
            "coordinate_frame": "world",
            "contact_centroid_world_m": list(pair.contact_point_world_m),
            "contact_centroid_local_m": list(pair.contact_point_local_m),
            "normal_world": list(pair.normal_world) if contact else None,
            "tangent_basis_world": [list(pair.tangent_basis_world[0]), list(pair.tangent_basis_world[1])],
            "contact_point_count": len(pair.points),
            "active_patch_count": len(pair.patches),
            "contact_area_proxy_m2": area,
            "physical_contact_area_m2": None,
            "normal_force_n": force,
            "tangent_force_world_n": list(pair.tangent_force_world_n),
            "total_force_world_n": list(pair.total_force_world_n),
            "normal_impulse_ns": pair.normal_impulse_ns if contact else None,
            "normal_relative_velocity_mps": (
                pair.normal_relative_velocity_mps if pair.kinematics_available else None
            ),
            "tangent_velocity_world_mps": (
                list(pair.tangent_velocity_world_mps) if pair.kinematics_available else None
            ),
            "indentation_m": 0.0,
            "penetration_m": pair.penetration_m,
            "deformation_max_m": None,
            "deformation_mean_m": None,
            "deformation_rms_m": None,
            "deformation_field": None,
            "pressure_mean_pa": pressure,
            "pressure_peak_pa": pressure,
            "pressure_field": None,
            "quality": self._quality(contact, pair.kinematics_available),
            "confidence": 1.0,
            "metadata": self._metadata(kernel_frame, pair),
        }
        return InteractionState.from_dict(values)

    def _quality(self, contact: bool, kinematics_available: bool) -> dict[str, dict[str, str]]:
        quality = {
            "contact": {
                "status": "measured",
                "method": "rigid_nonpenetration_constraint",
                "unit": "1",
                "source": BACKEND_METHOD_SOURCE,
            }
        }
        for name, unit in OBSERVABLE_UNITS.items():
            method = self.methods[name]
            available = method is not None and (contact or name in self.always_available)
            if name in KINEMATICS_DEPENDENT and not kinematics_available:
                available = False
            quality[name] = {
                "status": OBSERVABLE_STATUS[name] if available else "unavailable",
                "method": method if available else NOT_AVAILABLE,
                "unit": unit,
                "source": BACKEND_METHOD_SOURCE,
            }
        return quality

    def _metadata(self, kernel_frame, pair) -> dict:
        return {
            "kernel_schema_version": kernel_frame.schema_version,
            "backend_version": kernel_frame.backend_version,
            "motion_phase": pair.phase,
            "control_step": kernel_frame.control_step,
            "physics_dt_s": kernel_frame.physics_dt_s,
            "substeps_per_control": kernel_frame.substeps_per_control,
            "kinematics_source": pair.kinematics_source,
            "kinematics_method": pair.kinematics_method,
            "kinematics_available": pair.kinematics_available,
            "rigid_body_tangent_velocity_mps": list(pair.rigid_body_tangent_velocity_mps),
            "solver_contact": pair.solver_contact,
            "reported_contact": pair.contact_present,
            "sampling_filter_reason": pair.filtered_reason,
            "inside_top_face": pair.inside_face,
            "gap_m": pair.gap_m,
            "position_correction_m": pair.position_correction_m,
            "solver_normal_impulse_ns": pair.solution.normal_impulse_ns,
            "contact_substeps": pair.solution.contact_substeps,
            "probe_position_world_m": list(pair.probe_position_world_m),
            "probe_velocity_world_mps": list(pair.probe_velocity_world_mps),
            "command_velocity_world_mps": list(pair.command_velocity_world_mps),
            "tangent_speed_mps": pair.tangent_speed_mps,
            "area_layers": pair.area_layers.to_dict(),
            "conservation": pair.conservation.to_dict(),
            "pressure_semantics": PRESSURE_SEMANTICS,
            "cells": [
                {
                    "row": cell.row,
                    "col": cell.col,
                    "patch_id": cell.patch_id,
                    "interaction_state": cell.interaction_state,
                    "point_count": cell.point_count,
                    "coverage_alpha": cell.coverage_alpha,
                    "area_method": cell.area_method,
                    "cell_area_m2": cell.cell_area_m2,
                    "normal_force_n": cell.normal_force_n,
                    "normal_impulse_ns": cell.normal_impulse_ns,
                    "mean_tangent_speed_mps": cell.mean_tangent_speed_mps,
                }
                for cell in (pair.cells[key] for key in sorted(pair.cells))
            ],
            "patches": [
                {
                    "patch_id": patch.patch_id,
                    "lifecycle": patch.lifecycle,
                    "interaction_state": patch.interaction_state,
                    "cell_count": len(patch.active_cells),
                    "active_cells": [list(cell) for cell in patch.active_cells],
                    "mapped_area_m2": patch.mapped_area_m2,
                    "area_method": patch.area_method,
                    "normal_force_n": patch.normal_force_n,
                    "centroid_local_m": list(patch.centroid_local_m),
                    "contact_ids": patch.contact_ids,
                    "merged_from_patch_ids": patch.merged_from_patch_ids,
                    "split_from_patch_id": patch.split_from_patch_id,
                }
                for patch in pair.patches
            ],
            "started_patch_ids": list(pair.started_patch_ids),
            "ended_episode_ids": [episode.patch_id for episode in pair.ended_episodes],
            "contact_ids": [point.contact_id for point in pair.points],
        }
