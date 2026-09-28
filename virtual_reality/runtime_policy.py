"""Which declared runtime settings the kernel actually acts on.

A specification can declare a setting the solver never reads. Recording such a
setting is useful for provenance and for a later parity comparison, but it must
never look as though it took effect. Every field in an ``ExperimentSpec``
``runtime`` block therefore carries one of three verdicts:

``enforced``
    The kernel reads the value and different values change behaviour.

``recorded_inert``
    The kernel does not read the value, and for this solver's structure the
    setting is a genuine no-op — there is no iteration loop to count, and
    rotation is locked. Only the inert value is accepted, and that is verified
    at load rather than assumed.

``single_implementation``
    The field names a real physical or numerical choice, and exactly one value
    is implemented. Any other value is rejected.

The three verdicts are orthogonal to the specification's own
``status`` field (``source_explicit`` / ``unresolved`` / ``design_choice``).
That one says where a value came from; this one says whether the kernel acts on
it. Merging them would lose information.

Rejection happens at load, not in a note. A remark in a report cannot stop a
run from producing results that look valid.
"""

from __future__ import annotations

from dataclasses import dataclass

from contact_kernel import KINEMATICS_METHOD

ENFORCED = "enforced"
RECORDED_INERT = "recorded_inert"
SINGLE_IMPLEMENTATION = "single_implementation"
VERDICTS = (ENFORCED, RECORDED_INERT, SINGLE_IMPLEMENTATION)

#: Sentinel for a field whose value is unconstrained.
ANY = object()


@dataclass(frozen=True)
class FieldPolicy:
    verdict: str
    accepted: object
    note: str

    def accepts(self, value) -> bool:
        if self.accepted is ANY:
            return True
        if isinstance(self.accepted, bool) or isinstance(value, bool):
            return value is self.accepted
        if isinstance(self.accepted, (int, float)) and isinstance(value, (int, float)):
            return abs(value - self.accepted) <= 1e-12
        return value == self.accepted


def _enforced(note: str) -> FieldPolicy:
    return FieldPolicy(ENFORCED, ANY, note)


def _inert(accepted, note: str) -> FieldPolicy:
    return FieldPolicy(RECORDED_INERT, accepted, note)


def _single(accepted, note: str) -> FieldPolicy:
    return FieldPolicy(SINGLE_IMPLEMENTATION, accepted, note)


#: Verdict for every field of the K0 ``runtime`` block. A field missing from
#: this table is itself an error: an unclassified setting is one nobody has
#: checked, which is the situation the table exists to prevent.
RUNTIME_POLICY: dict[str, FieldPolicy] = {
    "gravity_mps2": _enforced("integrated every substep; world Z component only"),
    "linear_damping_per_s": _enforced("scales velocity each substep"),
    "rest_offset_m": _enforced("shifts the contact plane in every gap query"),
    "max_depenetration_velocity_mps": _enforced("caps the target normal velocity of the impulse"),
    "initial_linear_velocity_mps": _enforced("seeds the probe body state"),

    "position_iterations": _inert(1, "single analytic contact; the solver has no iteration loop"),
    "velocity_iterations": _inert(1, "single analytic contact; the solver has no iteration loop"),
    "sleep_enabled": _inert(False, "the kernel never deactivates a body"),
    "angular_damping_per_s": _inert(0, "probe rotation is locked, so angular damping cannot act"),
    "initial_orientation_wxyz": _inert(
        [1, 0, 0, 0], "the kernel carries no orientation state for a locked sphere probe"
    ),
    "friction_combine_mode": _inert(ANY, "inert while static and dynamic friction are both zero"),
    "restitution_combine_mode": _inert(ANY, "inert while restitution is zero"),

    "static_friction": _single(0, "Coulomb friction arrives after the K1 baseline"),
    "dynamic_friction": _single(0, "Coulomb friction arrives after the K1 baseline"),
    "restitution": _single(0, "restitution arrives with the K4 material modes"),
    "contact_offset_m": _single(
        0,
        "the kernel uses predictive gap checking, not a speculative contact "
        "margin; a nonzero offset would change contact onset and is not implemented",
    ),
    "solver_type": _single(
        "single_sphere_plane_normal_impulse", "the only implemented dynamics mode"
    ),
    "ccd_mode": _single(
        "linear_sweep_in_drift",
        "each substep tests the swept gap once; no other sweep mode is implemented",
    ),
}

#: Fields outside the ``runtime`` block whose enforcement is worth stating.
SAMPLING_POLICY: dict[str, FieldPolicy] = {
    "kinematics_source": _enforced(
        "selects the authoritative velocity source; the other source is kept "
        "on each point as a cross-check"
    ),
}


class UnsupportedRuntimeSetting(ValueError):
    """A declared setting the kernel cannot honour."""


def classify_runtime(spec) -> dict:
    """Report the verdict and declared value of every runtime setting."""
    data = spec.to_dict() if hasattr(spec, "to_dict") else spec
    unclassified = sorted(set(data["runtime"]) - set(RUNTIME_POLICY))
    if unclassified:
        raise UnsupportedRuntimeSetting(
            f"Runtime settings have no declared kernel policy: {unclassified}. "
            "Classify them in virtual_reality/runtime_policy.py before running."
        )
    fields = {}
    for name, setting in sorted(data["runtime"].items()):
        policy = RUNTIME_POLICY[name]
        fields[name] = {
            "declared_value": setting["value"],
            "declared_status": setting["status"],
            "kernel_verdict": policy.verdict,
            "accepted_value": None if policy.accepted is ANY else policy.accepted,
            "accepts_any_value": policy.accepted is ANY,
            "note": policy.note,
        }
    source = data["sampling"]["kinematics_source"]
    fields["sampling.kinematics_source"] = {
        "declared_value": source,
        "declared_status": "source_explicit",
        "kernel_verdict": SAMPLING_POLICY["kinematics_source"].verdict,
        "accepted_value": sorted(KINEMATICS_METHOD),
        "accepts_any_value": False,
        "note": SAMPLING_POLICY["kinematics_source"].note,
    }
    counts = {verdict: 0 for verdict in VERDICTS}
    for entry in fields.values():
        counts[entry["kernel_verdict"]] += 1
    return {
        "schema": "k1-field-policy/v1",
        "counts": counts,
        "fields": fields,
        "note": (
            "kernel_verdict says whether the solver acts on the value; it is "
            "orthogonal to declared_status, which says where the value came from"
        ),
    }


def enforce_runtime(spec) -> dict:
    """Reject any declared setting the kernel cannot honour. Returns the report."""
    report = classify_runtime(spec)
    violations = []
    for name, setting in sorted((spec.to_dict() if hasattr(spec, "to_dict") else spec)["runtime"].items()):
        policy = RUNTIME_POLICY[name]
        if not policy.accepts(setting["value"]):
            violations.append(
                f"{name}={setting['value']!r}: only {policy.accepted!r} is implemented "
                f"({policy.verdict}: {policy.note})"
            )
    source = report["fields"]["sampling.kinematics_source"]["declared_value"]
    if source not in KINEMATICS_METHOD:
        violations.append(
            f"sampling.kinematics_source={source!r}: implemented sources are "
            f"{sorted(KINEMATICS_METHOD)}"
        )
    if violations:
        raise UnsupportedRuntimeSetting(
            "Declared settings the kernel cannot honour:\n  " + "\n  ".join(violations)
        )
    return report
