"""Layer 3 material response models.

K1 ships the rigid baseline only. It is a pass-through on purpose: in rigid
impulse mode the kernel constraint is the single authoritative dynamics solver
for the contact pair in that step, so recomputing a force here would apply the
same interaction twice. The stage therefore restates the solved force, asserts
zero material indentation, and leaves every elastic field unavailable.

Kelvin-Voigt, Standard Linear Solid, Hertz and elastic-foundation models arrive
at K4 with their own contact area and pressure field. They are the first
models allowed to mark pressure as derived from a physical area.
"""

from __future__ import annotations

from tactile_contract.pipeline import STATUS_PASSTHROUGH, MaterialResponse

RIGID_MODEL_ID = "rigid_constraint_baseline_v1"
RIGID_NOTES = (
    "rigid impulse mode: force comes from the kernel non-penetration "
    "constraint, material indentation is zero by assumption, and stiffness is "
    "not represented as a finite value"
)


class RigidBaselineMaterial:
    """Layer 3: restate the rigid contact solution as a material response."""

    layer = 3
    stage_id = RIGID_MODEL_ID

    def __init__(self, *, area_method: str = "grid_point_occupancy_proxy"):
        self.area_method = area_method
        self._previous_indentation_m = 0.0

    def reset(self) -> None:
        self._previous_indentation_m = 0.0

    def status(self) -> str:
        return "ok"

    def step(self, world_frame, interaction) -> MaterialResponse:
        data = interaction.to_dict()
        contact = bool(data["contact_present"])
        indentation = float(data["indentation_m"])
        rate = (indentation - self._previous_indentation_m) / data["dt_s"]
        self._previous_indentation_m = indentation
        return MaterialResponse(
            time_s=data["time_s"],
            control_step=data["metadata"]["control_step"],
            body0=data["body0"],
            body1=data["body1"],
            model_id=RIGID_MODEL_ID,
            status=STATUS_PASSTHROUGH,
            contact_present=contact,
            normal_force_n=data["normal_force_n"] or 0.0,
            tangent_force_world_n=tuple(data["tangent_force_world_n"]),
            indentation_m=indentation,
            indentation_rate_mps=rate,
            stiffness_n_per_m=None,
            damping_ns_per_m=None,
            contact_area_m2=data["contact_area_proxy_m2"],
            contact_area_method=self.area_method,
            pressure_mean_pa=data["pressure_mean_pa"],
            pressure_peak_pa=data["pressure_peak_pa"],
            internal_state={
                "normal_relative_velocity_mps": data["normal_relative_velocity_mps"],
                "numerical_penetration_m": data["penetration_m"],
                "motion_phase": data["metadata"]["motion_phase"],
                "material_cell_states": 0,
            },
            notes=RIGID_NOTES,
        )
