"""InteractionState v2: nullable observables with per-field provenance."""

import math

from ._common import JsonRecord

INTERACTION_STATE_SCHEMA = "interaction-state/v2"
SCALAR_UNITS = {
    "contact_area_proxy_m2": "m2", "physical_contact_area_m2": "m2",
    "normal_force_n": "N", "normal_impulse_ns": "N*s",
    "normal_relative_velocity_mps": "m/s", "indentation_m": "m",
    "penetration_m": "m", "deformation_max_m": "m",
    "deformation_mean_m": "m", "deformation_rms_m": "m",
    "pressure_mean_pa": "Pa", "pressure_peak_pa": "Pa",
}
VECTOR_UNITS = {
    "contact_centroid_world_m": "m", "normal_world": "1",
    "tangent_force_world_n": "N", "total_force_world_n": "N",
    "tangent_velocity_world_mps": "m/s",
}
FIELD_UNITS = {"pressure_field": "Pa", "deformation_field": "m"}
OBSERVABLE_UNITS = SCALAR_UNITS | VECTOR_UNITS | FIELD_UNITS
OBSERVABLE_GROUP = {
    "contact_centroid_world_m": "geometry",
    "normal_world": "geometry",
    "tangent_force_world_n": "force",
    "total_force_world_n": "force",
    "tangent_velocity_world_mps": "kinematics",
    "pressure_field": "pressure_field",
    "deformation_field": "deformation_field",
}
OBSERVABLE_GROUP.update({name: "geometry" for name in ("contact_area_proxy_m2", "physical_contact_area_m2")})
OBSERVABLE_GROUP.update({name: "force" for name in ("normal_force_n", "normal_impulse_ns")})
OBSERVABLE_GROUP.update({name: "kinematics" for name in ("normal_relative_velocity_mps", "indentation_m", "penetration_m")})
OBSERVABLE_GROUP.update({name: "deformation" for name in ("deformation_max_m", "deformation_mean_m", "deformation_rms_m")})
OBSERVABLE_GROUP.update({name: "pressure" for name in ("pressure_mean_pa", "pressure_peak_pa")})


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def check_basis(basis):
    for i, row in enumerate(basis):
        if not math.isclose(dot(row, row), 1.0, abs_tol=1e-8):
            raise ValueError("Basis vectors must have unit length")
        for other in basis[:i]:
            if abs(dot(row, other)) > 1e-8:
                raise ValueError("Basis vectors must be orthogonal")
    if dot(cross(basis[0], basis[1]), basis[2]) < 1 - 1e-8:
        raise ValueError("Basis must be right-handed")


def check_tangent_basis(basis, normal):
    if any(not math.isclose(dot(row, row), 1.0, abs_tol=1e-8) for row in basis):
        raise ValueError("Tangent basis vectors must have unit length")
    if abs(dot(basis[0], basis[1])) > 1e-8:
        raise ValueError("Tangent basis vectors must be orthogonal")
    if normal is not None and abs(dot(cross(basis[0], basis[1]), normal)) < 1 - 1e-8:
        raise ValueError("Tangent basis must align with normal_world")


def validate_field(field, unit):
    if field["units"] != unit:
        raise ValueError("Spatial field unit mismatch")
    check_basis(field["basis_world"])
    rows, cols = field["shape"]
    values, mask = field["values"], field["active_mask"]
    if len(values) != rows or len(mask) != rows:
        raise ValueError("Spatial field row count mismatch")
    for row, flags in zip(values, mask):
        if len(row) != cols or len(flags) != cols:
            raise ValueError("Spatial field column count mismatch")
        for value, active in zip(row, flags):
            if active != (value is not None):
                raise ValueError("Active cells need finite values, inactive cells need null")


class InteractionState(JsonRecord):
    __slots__ = ()
    schema_file = "interaction-state-v2.schema.json"

    @staticmethod
    def validate_semantics(data):
        if data["body0"] == data["body1"]:
            raise ValueError("body0 and body1 must differ")
        quality = data["quality"]
        check_tangent_basis(data["tangent_basis_world"], data["normal_world"])
        for name, unit in OBSERVABLE_UNITS.items():
            value = data[name]
            field_quality = quality[name]
            if field_quality["unit"] != unit:
                raise ValueError(f"{name}: quality unit mismatch")
            if (field_quality["status"] == "unavailable") != (value is None):
                raise ValueError(f"{name}: availability and value disagree")
            if name in FIELD_UNITS and value is not None:
                validate_field(value, unit)
        normal = data["normal_world"]
        if normal is not None:
            if not math.isclose(dot(normal, normal), 1.0, abs_tol=1e-8):
                raise ValueError("normal_world must be a unit vector")
            for name in ("tangent_force_world_n", "tangent_velocity_world_mps"):
                value = data[name]
                if value is not None and abs(dot(value, normal)) > 1e-8 * max(1, math.sqrt(dot(value, value))):
                    raise ValueError(f"{name} must be tangential")
        if all(data[k] is not None for k in ("normal_world", "normal_force_n", "tangent_force_world_n", "total_force_world_n")):
            expected = [data["normal_force_n"] * n + t for n, t in zip(normal, data["tangent_force_world_n"])]
            if any(not math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8) for a, b in zip(expected, data["total_force_world_n"])):
                raise ValueError("Total force must be compression normal plus tangential force on body0")
        for avg, peak in (("pressure_mean_pa", "pressure_peak_pa"), ("deformation_mean_m", "deformation_max_m"), ("deformation_rms_m", "deformation_max_m")):
            if data[avg] is not None and data[peak] is not None and data[avg] > data[peak] + 1e-12:
                raise ValueError(f"{avg} exceeds {peak}")
        if data["contact_present"] is False:
            if data["contact_mode"] not in ("no_contact", "approach", "unknown"):
                raise ValueError("Contact mode contradicts no contact")
            for key in ("normal_force_n", "normal_impulse_ns", "pressure_mean_pa", "pressure_peak_pa", "physical_contact_area_m2", "contact_area_proxy_m2"):
                if data[key] not in (None, 0):
                    raise ValueError(f"No-contact frame has nonzero {key}")
            for key in ("tangent_force_world_n", "total_force_world_n"):
                if data[key] is not None and any(data[key]):
                    raise ValueError("No-contact frame has nonzero contact force")
        if data["contact_present"] is True and data["contact_mode"] in ("no_contact", "approach"):
            raise ValueError("Contact mode contradicts contact")

    def scalar_observables(self):
        # Unavailable values remain null. No fabricated zeros enter comparisons.
        data = self.to_dict()
        return {name: data[name] for name in SCALAR_UNITS}
