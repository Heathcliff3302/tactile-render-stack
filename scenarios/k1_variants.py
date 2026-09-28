"""Derived K1 specifications for refinement comparisons.

The K0 CPU reference stays the single declared input. A refinement variant is
derived from it in memory, keeps every physical and sampling field identical,
changes only the numerical discretisation, and recomputes ``spec_sha256`` so
the comparison records a real, distinct specification instead of reusing one
spec while claiming a different time step.
"""

from __future__ import annotations

import copy

from tactile_contract import ExperimentSpec, spec_comparison_gate

TIME_REFINEMENT_PROFILE = "time_refinement"
GRID_REFINEMENT_PROFILE = "grid_refinement"


def with_substeps(spec: ExperimentSpec, substeps: int) -> ExperimentSpec:
    """Same experiment, ``substeps`` physics substeps per control interval."""
    if substeps < 1:
        raise ValueError("substeps must be at least one")
    data = copy.deepcopy(spec.to_dict())
    data.pop("spec_sha256", None)
    timing = data["timing"]
    timing["substeps_per_control"] = substeps
    timing["physics_dt_s"] = timing["control_dt_s"] / substeps
    data["spec_id"] = f"{data['spec_id']}_substeps{substeps}"
    data["description"] = (
        f"{data['description']} Physics substeps per control interval: {substeps}."
    )
    data["parameter_evidence"]["timing"] = (
        "Derived time-refinement variant of the K0 CPU reference; only "
        "physics_dt_s and substeps_per_control differ."
    )
    return ExperimentSpec.build(**data)


def time_refinement_specs(spec: ExperimentSpec, substeps=(2, 4)) -> tuple[ExperimentSpec, ...]:
    """Variants for the K1 time-step gate, each checked for comparability."""
    variants = []
    for count in substeps:
        variant = with_substeps(spec, count)
        gate = spec_comparison_gate(spec, variant, profile=TIME_REFINEMENT_PROFILE)
        if not gate["eligible"]:
            raise ValueError(f"substeps={count} variant is not comparable: {gate['reasons']}")
        variants.append(variant)
    return tuple(variants)
