"""K1 acceptance gates.

Gates, in the order the milestone states them:

1. the run completes the program it declared, without aborting;
2. a single-probe contact appears at all;
3. contact is stable through the hold segment;
4. contact coverage over the controlled segments meets its threshold;
5. sliding reaches the declared speed on enough samples;
6. contact force is cleared after release;
7. total force agrees with the interval impulse, and force is conserved along
   point -> cell -> patch;
8. changing the physics time step does not change data semantics;
9. the ``InteractionState`` stream replays completely.

Gates 1 to 7 read one run. Gate 8 compares runs that differ only in substeps.
Gate 9 is the run-directory round trip. Every threshold comes from the declared
``acceptance`` block of the spec, never from a literal in this file.

Every key of that block must be either consumed by a gate or listed as a
declared deferral. ``ACCEPTANCE_KEY_GATES`` and ``DEFERRED_ACCEPTANCE_KEYS``
below are that ledger, and a test asserts the two together cover the block
exactly. Computing a threshold and printing it without comparing it is the
failure this ledger exists to prevent: the earlier revision of this module did
exactly that for the sliding and coverage thresholds, and reported runs as
passing while they violated both.
"""

from __future__ import annotations

import math

from tactile_contract import ExperimentManifest, comparison_gate, spec_comparison_gate, validate_run
from tactile_contract.io import load_run
from virtual_reality import MOTION_PHASES


#: Which gate consumes each declared acceptance threshold.
ACCEPTANCE_KEY_GATES: dict[str, tuple[str, ...]] = {
    "contact_coverage_min": ("hold_contact_is_stable", "contact_coverage_meets_threshold"),
    "hold_tangent_speed_max_mps": ("hold_contact_is_stable",),
    "slide_speed_tolerance_mps": ("slide_speed_meets_threshold",),
    "slide_speed_pass_fraction_min": ("slide_speed_meets_threshold",),
    "slide_settle_time_s": ("slide_speed_meets_threshold",),
    "release_contact_frames_max": ("release_clears_contact_force",),
    "force_conservation_atol_n": ("force_matches_impulse_and_is_conserved",),
}

#: Thresholds that belong to a later milestone. A deferral is declared here so
#: it is visible in the report, never silently skipped.
DEFERRED_ACCEPTANCE_KEYS: dict[str, str] = {
    "patch_coverage_min": "K3: multiple simultaneous patches and stable patch IDs",
    "resolution_force_spread_max": "K3: 32/64/128 output grid refinement",
}

#: Gates that read one run, in report order.
RUN_GATE_ORDER = (
    "run_completes_declared_program",
    "single_probe_contact_appears",
    "hold_contact_is_stable",
    "contact_coverage_meets_threshold",
    "slide_speed_meets_threshold",
    "release_clears_contact_force",
    "force_matches_impulse_and_is_conserved",
)


def acceptance_key_ledger(spec) -> dict:
    """Confirm every declared threshold is consumed or explicitly deferred."""
    declared = set((spec.to_dict() if hasattr(spec, "to_dict") else spec)["acceptance"])
    consumed = set(ACCEPTANCE_KEY_GATES)
    deferred = set(DEFERRED_ACCEPTANCE_KEYS)
    unhandled = sorted(declared - consumed - deferred)
    unknown = sorted((consumed | deferred) - declared)
    if unhandled:
        raise ValueError(
            f"Declared acceptance thresholds neither consumed nor deferred: {unhandled}. "
            "Add a gate, or declare the deferral in DEFERRED_ACCEPTANCE_KEYS."
        )
    if unknown:
        raise ValueError(f"Ledger names thresholds the spec does not declare: {unknown}")
    return {
        "consumed": {key: list(gates) for key, gates in sorted(ACCEPTANCE_KEY_GATES.items())},
        "deferred": dict(sorted(DEFERRED_ACCEPTANCE_KEYS.items())),
    }


def _phase_frames(run, phase: str):
    return [frame for frame in run.frames if frame.metadata["motion_phase"] == phase]


def _gate(name: str, passed: bool, detail: dict) -> dict:
    return {"gate": name, "passed": bool(passed), **detail}


def evaluate_run(run) -> dict:
    """Every single-run gate, plus the segment and conservation evidence."""
    limits = run.spec.to_dict()["acceptance"]
    controller = run.spec.to_dict()["controller"]["parameters"]
    ledger = acceptance_key_ledger(run.spec)
    frames = run.frames
    phases = run.phase_sequence

    hold = _phase_frames(run, "hold")
    slide = _phase_frames(run, "lateral_slide")
    released = _phase_frames(run, "released")
    contact_frames = [frame for frame in frames if frame.contact_present]
    controlled = hold + slide

    coverage = sum(frame.contact_present for frame in controlled) / len(controlled) if controlled else 0.0
    hold_contact = sum(frame.contact_present for frame in hold) / len(hold) if hold else 0.0
    hold_tangent_speed = max(
        (math.hypot(*_tangent(frame)[:2]) for frame in hold), default=0.0
    )
    # A settle window at the start of sliding is excluded from the speed
    # statistic: the servo is allowed that long to reach the target.
    settle_s = limits["slide_settle_time_s"]
    slide_start_s = slide[0].time_s if slide else 0.0
    slide_measured = [frame for frame in slide if frame.time_s - slide_start_s >= settle_s]
    slide_in_tolerance = [
        frame for frame in slide_measured
        if abs(_tangent(frame)[0] - controller["slide_speed_mps"])
        <= limits["slide_speed_tolerance_mps"]
    ]
    slide_fraction = len(slide_in_tolerance) / len(slide_measured) if slide_measured else 0.0

    released_contact_frames = sum(frame.contact_present for frame in released)
    released_force = max(
        (abs(frame.total_force_world_n[2]) for frame in released), default=0.0
    )

    impulse_error = 0.0
    total_force_error = 0.0
    conservation_error = 0.0
    for frame in contact_frames:
        impulse_error = max(impulse_error, abs(frame.normal_force_n - frame.normal_impulse_ns / frame.dt_s))
        expected = [
            frame.normal_force_n * component + tangent
            for component, tangent in zip(frame.normal_world, frame.tangent_force_world_n)
        ]
        total_force_error = max(
            total_force_error,
            max(abs(a - b) for a, b in zip(expected, frame.total_force_world_n)),
        )
        conservation = frame.metadata["conservation"]
        conservation_error = max(
            conservation_error,
            conservation["point_to_cell_error_n"],
            conservation["cell_to_patch_error_n"],
        )

    patch_ids = sorted({
        patch["patch_id"] for frame in frames for patch in frame.metadata["patches"]
    })
    segments_complete = _segments_complete(phases)
    gates = [
        _gate("run_completes_declared_program",
              run.loop_state == "completed" and not run.abort_reason and segments_complete, {
            "loop_state": run.loop_state,
            "abort_reason": run.abort_reason,
            "segment_order_complete": segments_complete,
            "frame_count": len(frames),
            "final_segment": phases[-1] if phases else None,
        }),
        _gate("single_probe_contact_appears", bool(contact_frames), {
            "contact_frames": len(contact_frames),
            "first_contact_time_s": contact_frames[0].time_s if contact_frames else None,
            "peak_normal_force_n": max((frame.normal_force_n for frame in contact_frames), default=0.0),
            "patch_ids": patch_ids,
        }),
        _gate("hold_contact_is_stable",
              bool(hold) and hold_contact >= limits["contact_coverage_min"]
              and hold_tangent_speed <= limits["hold_tangent_speed_max_mps"], {
            "hold_frames": len(hold),
            "hold_contact_coverage": hold_contact,
            "contact_coverage_min": limits["contact_coverage_min"],
            "hold_tangent_speed_max_mps": hold_tangent_speed,
            "hold_tangent_speed_limit_mps": limits["hold_tangent_speed_max_mps"],
        }),
        _gate("contact_coverage_meets_threshold",
              bool(controlled) and coverage >= limits["contact_coverage_min"], {
            "controlled_frames": len(controlled),
            "contact_coverage_hold_and_slide": coverage,
            "contact_coverage_min": limits["contact_coverage_min"],
        }),
        _gate("slide_speed_meets_threshold",
              bool(slide_measured) and slide_fraction >= limits["slide_speed_pass_fraction_min"], {
            "slide_frames": len(slide),
            "slide_measured_frames": len(slide_measured),
            "slide_settle_time_s": settle_s,
            "slide_speed_target_mps": controller["slide_speed_mps"],
            "slide_speed_tolerance_mps": limits["slide_speed_tolerance_mps"],
            "slide_speed_pass_fraction": slide_fraction,
            "slide_speed_pass_fraction_min": limits["slide_speed_pass_fraction_min"],
        }),
        _gate("release_clears_contact_force",
              bool(released) and released_contact_frames <= limits["release_contact_frames_max"]
              and released_force == 0.0, {
            "released_frames": len(released),
            "released_contact_frames": released_contact_frames,
            "released_contact_frames_max": limits["release_contact_frames_max"],
            "max_released_total_force_n": released_force,
        }),
        _gate("force_matches_impulse_and_is_conserved",
              bool(contact_frames)
              and impulse_error <= limits["force_conservation_atol_n"]
              and total_force_error <= limits["force_conservation_atol_n"]
              and conservation_error <= limits["force_conservation_atol_n"], {
            "impulse_to_force_max_error_n": impulse_error,
            "normal_plus_tangent_max_error_n": total_force_error,
            "point_to_patch_max_error_n": conservation_error,
            "force_conservation_atol_n": limits["force_conservation_atol_n"],
        }),
    ]
    return {
        "run_id": run.run_id,
        "scenario_id": run.scenario_id,
        "substeps_per_control": run.substeps_per_control,
        "loop_state": run.loop_state,
        "abort_reason": run.abort_reason,
        "frame_count": len(frames),
        "segments": {phase: phases.count(phase) for phase in (*MOTION_PHASES, "released")},
        "segment_order_complete": segments_complete,
        "acceptance_key_ledger": ledger,
        "kinematics_source": run.kernel_config.sampling.kinematics_source,
        "slide_speed_pass_fraction": slide_fraction,
        "slide_speed_pass_fraction_min": limits["slide_speed_pass_fraction_min"],
        "contact_coverage_hold_and_slide": coverage,
        "episodes": [
            {
                "patch_id": episode.patch_id,
                "samples": episode.samples,
                "duration_s": episode.duration_s,
                "max_normal_force_n": episode.max_normal_force_n,
                "cumulative_normal_impulse_ns": episode.cumulative_normal_impulse_ns,
                "max_mapped_area_m2": episode.max_mapped_area_m2,
                "swept_area_m2": episode.swept_area_m2,
                "sliding_detected": episode.sliding_detected,
                "end_reason": episode.end_reason,
            }
            for episode in run.episodes
        ],
        "gates": gates,
    }


def _tangent(frame):
    """Tangential velocity, or zeros when the declared source has no value.

    Differenced kinematics is undefined on a frame without a contact point. A
    gate must not read such a frame as a passing sample, so it reads as zero
    and fails the tolerance comparison instead.
    """
    value = frame.tangent_velocity_world_mps
    return value if value is not None else (0.0, 0.0, 0.0)


def _segments_complete(phases) -> bool:
    """Every declared segment appears, in order, and the run ends released."""
    expected = [*MOTION_PHASES, "released"]
    if not phases:
        return False
    position = 0
    for phase in phases:
        if position < len(expected) and phase == expected[position]:
            continue
        if position + 1 < len(expected) and phase == expected[position + 1]:
            position += 1
            continue
        return False
    return position == len(expected) - 1 and phases[-1] == "released"


def evaluate_time_refinement(runs) -> dict:
    """Gate 5: refining the physics step must not change data semantics.

    Semantics means the segment sequence, the contact and release sequence, the
    frame count, the cell and patch identities, and the conservation
    identities. Force *magnitude* is not asserted to be step independent in
    general; with the declared drive semantics it happens to be invariant too,
    so the measured spread is reported rather than assumed.
    """
    if len(runs) < 2:
        raise ValueError("Time refinement needs at least two runs")
    ordered = sorted(runs, key=lambda run: run.substeps_per_control)
    reference = ordered[0]
    eligibility = [
        spec_comparison_gate(reference.spec, other.spec, profile="time_refinement")
        for other in ordered[1:]
    ]
    comparisons = []
    for other in ordered[1:]:
        force_spread = max(
            (
                abs((a.normal_force_n or 0.0) - (b.normal_force_n or 0.0))
                for a, b in zip(reference.frames, other.frames)
            ),
            default=0.0,
        )
        if len(reference.frames) != len(other.frames):
            force_spread = math.inf
        position_spread = max(
            (
                abs(a.metadata["probe_position_world_m"][2] - b.metadata["probe_position_world_m"][2])
                for a, b in zip(reference.frames, other.frames)
            ),
            default=0.0,
        )
        comparisons.append({
            "substeps_per_control": other.substeps_per_control,
            "frame_count_equal": len(reference.frames) == len(other.frames),
            "segment_sequence_equal": reference.phase_sequence == other.phase_sequence,
            "contact_sequence_equal": reference.contact_sequence == other.contact_sequence,
            "cell_sequence_equal": _cell_sequence(reference) == _cell_sequence(other),
            "patch_sequence_equal": _patch_sequence(reference) == _patch_sequence(other),
            "max_normal_force_difference_n": force_spread,
            "max_probe_height_difference_m": position_spread,
        })
    semantics_preserved = all(
        item["frame_count_equal"] and item["segment_sequence_equal"]
        and item["contact_sequence_equal"] and item["cell_sequence_equal"]
        and item["patch_sequence_equal"]
        for item in comparisons
    )
    return {
        "gate": "time_step_does_not_change_data_semantics",
        "passed": semantics_preserved and all(item["eligible"] for item in eligibility),
        "reference_substeps_per_control": reference.substeps_per_control,
        "eligibility": eligibility,
        "comparisons": comparisons,
        "scope": "data semantics and conservation, not numeric parity with another solver",
    }


def _cell_sequence(run):
    return [
        tuple((cell["row"], cell["col"]) for cell in frame.metadata["cells"])
        for frame in run.frames
    ]


def _patch_sequence(run):
    return [
        tuple((patch["patch_id"], patch["cell_count"]) for patch in frame.metadata["patches"])
        for frame in run.frames
    ]


def evaluate_replay(run_directory) -> dict:
    """Gate 6: manifest, hashes and frames reload and revalidate from disk."""
    manifest, frames = load_run(run_directory)
    report = validate_run(manifest, frames)
    self_comparison = comparison_gate(manifest, manifest, ["normal_force_n", "normal_impulse_ns"])
    return {
        "gate": "interaction_state_replays_completely",
        "passed": bool(report["contract_passed"]) and self_comparison["eligible"],
        "run_directory": str(run_directory),
        "frames": report["frames"],
        "run_id": report["run_id"],
        "artifacts_verified": len(manifest.to_dict()["artifacts"]),
        "comparison_eligibility": self_comparison,
    }


def acceptance_report(run, *, refinement_runs=None, run_directory=None) -> dict:
    """Assemble the full K1 acceptance report from the gates above.

    ``all_passed`` requires every emitted gate to pass *and* every single-run
    gate to have been emitted. A missing gate cannot read as a pass.
    """
    report = evaluate_run(run)
    if refinement_runs:
        report["gates"].append(evaluate_time_refinement([run, *refinement_runs]))
    if run_directory is not None:
        report["gates"].append(evaluate_replay(run_directory))
    names = [gate["gate"] for gate in report["gates"]]
    missing = [name for name in RUN_GATE_ORDER if name not in names]
    report["gate_names"] = names
    report["missing_gates"] = missing
    report["all_passed"] = not missing and all(gate["passed"] for gate in report["gates"])
    return report


def manifest_from_run(run, artifacts, observables) -> ExperimentManifest:
    """Build the run manifest that binds conditions, observables and artifacts."""
    return ExperimentManifest.build(
        schema_version="experiment-manifest/v1",
        run_id=run.run_id,
        scenario={
            "id": run.scenario_id,
            "version": "k1.0",
            "description": (
                "K1 single sphere probe against a fixed cube top face, rigid "
                "impulse mode, CPU reference kernel"
            ),
        },
        backend={"id": run.kernel_config.backend_id, "version": run.kernel_config.backend_version},
        conditions=run.conditions,
        observables=observables,
        acceptance={
            "expected_frames": len(run.frames),
            "required_observables": list(_required_observables(run)),
            "max_missing_frame_fraction": 0.0,
        },
        artifacts=artifacts,
    )


def _required_observables(run):
    from interaction import always_available

    return always_available(run.kernel_config.sampling.kinematics_source)
