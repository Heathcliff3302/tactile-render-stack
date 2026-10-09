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
from virtual_reality import (
    PHASE_ROLES,
    assess_regulator_authority,
    phase_with_role,
    phases_for_mode,
)


#: Which gate consumes each declared acceptance threshold, in every mode.
ACCEPTANCE_KEY_GATES: dict[str, tuple[str, ...]] = {
    "contact_coverage_min": ("hold_contact_is_stable", "contact_coverage_meets_threshold"),
    "hold_tangent_speed_max_mps": ("hold_contact_is_stable",),
    "slide_speed_tolerance_mps": ("slide_speed_meets_threshold",),
    "slide_speed_pass_fraction_min": ("slide_speed_meets_threshold",),
    "slide_settle_time_s": ("slide_speed_meets_threshold",),
    "release_contact_frames_max": ("release_clears_contact_force",),
    "force_conservation_atol_n": ("force_matches_impulse_and_is_conserved",),
}

#: Thresholds a force-mode specification must declare, and the gates that read
#: them. A trajectory specification must *not* declare these: a threshold
#: nothing can measure is the same defect as a threshold nothing compares.
FORCE_ACCEPTANCE_KEY_GATES: dict[str, tuple[str, ...]] = {
    "force_pass_fraction_min": (
        "force_target_is_reached_and_held", "force_held_through_slide",
    ),
}

#: Thresholds that belong to a later milestone. A deferral is declared here so
#: it is visible in the report, never silently skipped.
DEFERRED_ACCEPTANCE_KEYS: dict[str, str] = {
    "patch_coverage_min": "K3: multiple simultaneous patches and stable patch IDs",
    "resolution_force_spread_max": "K3: 32/64/128 output grid refinement",
}

#: Gates that read one run, in report order, in every mode.
RUN_GATE_ORDER = (
    "run_completes_declared_program",
    "single_probe_contact_appears",
    "hold_contact_is_stable",
    "contact_coverage_meets_threshold",
    "slide_speed_meets_threshold",
    "release_clears_contact_force",
    "force_matches_impulse_and_is_conserved",
)

#: Additional single-run gates a force-mode run must emit and pass.
FORCE_RUN_GATE_ORDER = (
    "force_target_is_reached_and_held",
    "force_held_through_slide",
    "regulator_stays_within_declared_authority",
)

#: Gates that need inputs beyond one run: refinement runs, and a written
#: directory. Full acceptance requires these too.
CROSS_RUN_GATE_ORDER = (
    "time_step_does_not_change_data_semantics",
    "interaction_state_replays_completely",
)

#: Every gate full K1 acceptance requires.
K1_GATE_ORDER = (*RUN_GATE_ORDER, *CROSS_RUN_GATE_ORDER)

#: Every gate full K2 acceptance requires.
K2_GATE_ORDER = (*RUN_GATE_ORDER, *FORCE_RUN_GATE_ORDER, *CROSS_RUN_GATE_ORDER)


#: Deliverables a milestone requires beyond a passing run, and whether they
#: exist yet. A passing CPU-reference run is one scenario on one backend; it
#: cannot establish a cross-backend milestone, and must not be read as one.
#: Each entry is ``(delivered, what it needs)``.
MILESTONE_DELIVERABLES: dict[str, dict[str, tuple[bool, str]]] = {
    "trajectory": {
        "cpu_reference_run": (True, "K1 single-probe trajectory run passes every gate"),
        "dense_contact_field": (
            False, "Step 6.4 dense field is a K2 deliverable; K1 ships sparse products only"
        ),
        "isaac_sim_parity": (
            False, "needs the Isaac Sim adapter and captured effective runtime settings"
        ),
    },
    "force": {
        "cpu_reference_run": (True, "K2 force-control run passes every gate"),
        "dense_contact_field": (
            False, "dense_contact_field.npz and its replay check are not written yet"
        ),
        "isaac_sim_parity": (
            False, "needs the Isaac Sim adapter and captured effective runtime settings"
        ),
        "dynamic_authority_probe": (
            True, "measured on the kernel, covering damping and the realised drive"
        ),
        "curve_report": (False, "trajectory and force curve report is not written yet"),
    },
}

#: Sub-milestone label for a mode whose CPU reference is complete but whose
#: cross-backend and data-product work is not.
MILESTONE_LABELS: dict[str, tuple[str, str]] = {
    "trajectory": ("K1", "K1-A"),
    "force": ("K2", "K2-A"),
}


def milestone_status(mode: str) -> dict:
    """What a passing run does and does not establish about the milestone."""
    deliverables = MILESTONE_DELIVERABLES.get(mode, {})
    outstanding = sorted(name for name, (done, _) in deliverables.items() if not done)
    full, partial = MILESTONE_LABELS.get(mode, (mode, f"{mode}-A"))
    return {
        "schema": "milestone-status/v1",
        "controller_mode": mode,
        "milestone": full,
        "milestone_complete": not outstanding,
        "sub_milestone": partial if outstanding else full,
        "sub_milestone_scope": "CPU reference backend, single scenario",
        "deliverables": {
            name: {"delivered": done, "note": note}
            for name, (done, note) in sorted(deliverables.items())
        },
        "outstanding": outstanding,
        "note": (
            "a passing run establishes run_acceptance_passed on one backend "
            "and one scenario; milestone_complete additionally requires every "
            "deliverable above"
        ),
    }


def run_gate_order(mode: str) -> tuple[str, ...]:
    """Single-run gates a run in ``mode`` must emit."""
    if mode == "force":
        return (*RUN_GATE_ORDER, *FORCE_RUN_GATE_ORDER)
    return RUN_GATE_ORDER


def gate_order(mode: str) -> tuple[str, ...]:
    """Every gate full acceptance requires for ``mode``."""
    return (*run_gate_order(mode), *CROSS_RUN_GATE_ORDER)


def acceptance_key_ledger(spec) -> dict:
    """Confirm every declared threshold is consumed or explicitly deferred.

    The consumed set depends on the controller mode, because a force
    threshold has no gate that could read it on a trajectory run. Declaring
    one there would be a number nobody compares, which is the whole failure
    this ledger exists to catch; so it is an error in both directions —
    missing in force mode, and present outside it.
    """
    data = spec.to_dict() if hasattr(spec, "to_dict") else spec
    mode = data["controller"]["mode"]
    declared = set(data["acceptance"])
    consumed = dict(ACCEPTANCE_KEY_GATES)
    if mode == "force":
        consumed.update(FORCE_ACCEPTANCE_KEY_GATES)
    deferred = set(DEFERRED_ACCEPTANCE_KEYS)
    unhandled = sorted(declared - set(consumed) - deferred)
    unknown = sorted((set(consumed) | deferred) - declared)
    if unhandled:
        misplaced = sorted(set(unhandled) & set(FORCE_ACCEPTANCE_KEY_GATES))
        hint = (
            f" {misplaced} are force-mode thresholds and this specification "
            f"declares controller.mode = {mode!r}, where no gate can measure them."
            if misplaced else
            " Add a gate, or declare the deferral in DEFERRED_ACCEPTANCE_KEYS."
        )
        raise ValueError(
            f"Declared acceptance thresholds neither consumed nor deferred: {unhandled}.{hint}"
        )
    if unknown:
        raise ValueError(
            f"Ledger names thresholds the spec does not declare: {unknown}. "
            f"controller.mode is {mode!r}."
        )
    return {
        "controller_mode": mode,
        "consumed": {key: list(gates) for key, gates in sorted(consumed.items())},
        "deferred": dict(sorted(DEFERRED_ACCEPTANCE_KEYS.items())),
    }


def _phase_frames(run, phase: str):
    return [frame for frame in run.frames if frame.metadata["motion_phase"] == phase]


def _role_frames(run, role: str):
    """Frames of the segment playing ``role``, whichever controller ran.

    Asking for a literal ``"hold"`` would return nothing on a force run,
    whose hold segment is named ``regulate_force`` — and an empty list reads
    as a passing statistic unless something notices.
    """
    return [
        frame for frame in run.frames
        if PHASE_ROLES.get(frame.metadata["motion_phase"]) == role
    ]


def _gate(name: str, passed: bool, detail: dict) -> dict:
    return {"gate": name, "passed": bool(passed), **detail}


def evaluate_run(run) -> dict:
    """Every single-run gate, plus the segment and conservation evidence."""
    limits = run.spec.to_dict()["acceptance"]
    controller = run.spec.to_dict()["controller"]["parameters"]
    ledger = acceptance_key_ledger(run.spec)
    frames = run.frames
    phases = run.phase_sequence

    hold = _role_frames(run, "hold")
    slide = _role_frames(run, "slide")
    released = _role_frames(run, "released")
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
    declared_phases = phases_for_mode(run.controller_mode)
    segments_complete = _segments_complete(phases, declared_phases)
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
    if run.controller_mode == "force":
        gates.extend(_force_gates(run, hold, slide, limits))
    return {
        "run_id": run.run_id,
        "scenario_id": run.scenario_id,
        "substeps_per_control": run.substeps_per_control,
        "loop_state": run.loop_state,
        "abort_reason": run.abort_reason,
        "frame_count": len(frames),
        "controller_mode": run.controller_mode,
        "segments": {phase: phases.count(phase) for phase in declared_phases},
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


def _in_band(frame, target_n: float, tolerance_n: float) -> bool:
    """Is this frame's reported normal force inside the regulator's band?

    A frame with no reported contact has no force to be inside anything, so it
    counts as out of band rather than as a zero that happens to miss. Reading
    ``normal_force_n or 0.0`` and comparing would say the same thing here, but
    only because the target is positive; stated explicitly it stays true if a
    later milestone regulates toward zero.
    """
    if not frame.contact_present or frame.normal_force_n is None:
        return False
    return abs(frame.normal_force_n - target_n) <= tolerance_n


def _settle_index(frames, target_n: float, tolerance_n: float, settle_frames: int):
    """Index of the first frame completing ``settle_frames`` in band.

    This is the *lock* frame, the one on which the controller sets
    ``target_locked``. The controller starts counting the hold interval on
    this same frame (``_hold_elapsed_s += control_dt_s`` runs on the frame
    that locks), so the lock frame is the first held frame, not the last
    settling one. The gate below scores from here inclusive for that reason;
    excluding it would measure a window one frame shorter than the one the
    controller actually held, and then compare it against the declared hold
    time it was one frame short of.
    """
    run_length = 0
    for index, frame in enumerate(frames):
        run_length = run_length + 1 if _in_band(frame, target_n, tolerance_n) else 0
        if run_length >= settle_frames:
            return index
    return None


def _force_gates(run, hold, slide, limits) -> list[dict]:
    """The three gates that only a force-mode run can answer.

    The tolerance, the settle count and the target come from
    ``controller.force`` — the regulator's own declared band — rather than
    being restated in the acceptance block. One number, one home: a tolerance
    declared twice is a tolerance that can disagree with itself.

    ``force_pass_fraction_min`` is the one genuinely new threshold, and it is
    Phase 1's ``min_slide_force_in_tolerance``, which Phase 1 declared but
    left in ``source_arguments`` where no gate could read it.
    """
    data = run.spec.to_dict()
    force = data["controller"]["force"]
    target = force["target_force_n"]
    tolerance = force["force_tolerance_n"]
    settle_frames = force["settle_frames"]
    fraction_min = limits["force_pass_fraction_min"]
    # The declared hold interval, from the controller parameters that produced
    # it. Not restated in the acceptance block: one number, one home.
    hold_time_s = data["controller"]["parameters"]["hold_time_s"]
    control_dt_s = data["timing"]["control_dt_s"]
    required_held_frames = math.ceil(hold_time_s / control_dt_s - 1e-9)

    settle_index = _settle_index(hold, target, tolerance, settle_frames)
    # From the lock frame inclusive, because that is where the controller
    # starts its own hold clock. See _settle_index.
    held = hold[settle_index:] if settle_index is not None else []
    held_in_band = [frame for frame in held if _in_band(frame, target, tolerance)]
    held_fraction = len(held_in_band) / len(held) if held else 0.0
    held_duration_s = len(held) * control_dt_s
    # A fraction alone says nothing about duration: one frame in band is a
    # fraction of 1.0. The run must hold the target for the interval it
    # declared, so the duration is required outright.
    held_long_enough = len(held) >= required_held_frames

    # Same duration requirement on the slide segment, for the same reason: a
    # fraction over a one-frame segment is not evidence the target survived
    # sliding.
    slide_time_s = data["controller"]["parameters"]["slide_time_s"]
    required_slide_frames = math.ceil(slide_time_s / control_dt_s - 1e-9)
    slide_in_band = [frame for frame in slide if _in_band(frame, target, tolerance)]
    slide_fraction = len(slide_in_band) / len(slide) if slide else 0.0
    slide_duration_s = len(slide) * control_dt_s
    slide_long_enough = len(slide) >= required_slide_frames

    authority = assess_regulator_authority(run.spec)
    press_limit = force["max_press_speed_mps"]
    correct_limit = force["max_correct_speed_mps"]
    # Only the segments the regulator drives. ``retract`` commands the
    # declared retract speed, which is a trajectory command and is bounded by
    # its own parameter; scoring it against the regulator's correction limit
    # would fail the gate for a command the regulator never issued.
    regulated_roles = {"hold", "slide"}
    regulated = [
        record["command_velocity_world_mps"] for record in run.commands
        if PHASE_ROLES.get(record["phase"]) in regulated_roles
    ]
    max_press = max((-command[2] for command in regulated), default=0.0)
    max_correct = max((command[2] for command in regulated), default=0.0)
    # Floating point only; a command is clamped at the limit, never beyond it.
    atol = 1e-12
    within_limits = bool(regulated) and (
        max_press <= press_limit + atol and max_correct <= correct_limit + atol
    )
    peak_force = max((frame.normal_force_n or 0.0 for frame in run.frames), default=0.0)

    return [
        _gate("force_target_is_reached_and_held",
              settle_index is not None and held_long_enough
              and held_fraction >= fraction_min, {
            "hold_role_segment": phase_with_role(run.controller_mode, "hold"),
            "hold_frames": len(hold),
            "target_force_n": target,
            "force_tolerance_n": tolerance,
            "settle_frames_required": settle_frames,
            "settle_frame_index": settle_index,
            "settle_time_s": hold[settle_index].time_s if settle_index is not None else None,
            "settle_frame_counts_as_held": True,
            "held_frames_scored": len(held),
            "held_duration_s": held_duration_s,
            "declared_hold_time_s": hold_time_s,
            "required_held_frames": required_held_frames,
            "held_long_enough": held_long_enough,
            "held_in_band_fraction": held_fraction,
            "force_pass_fraction_min": fraction_min,
        }),
        _gate("force_held_through_slide",
              slide_long_enough and slide_fraction >= fraction_min, {
            "slide_frames": len(slide),
            "slide_duration_s": slide_duration_s,
            "declared_slide_time_s": slide_time_s,
            "required_slide_frames": required_slide_frames,
            "slide_long_enough": slide_long_enough,
            "slide_in_band_frames": len(slide_in_band),
            "slide_in_band_fraction": slide_fraction,
            "force_pass_fraction_min": fraction_min,
            "target_force_n": target,
            "force_tolerance_n": tolerance,
        }),
        _gate("regulator_stays_within_declared_authority",
              within_limits and bool(authority) and authority["tolerance_band_holdable"], {
            "regulated_command_steps": len(regulated),
            "regulated_segments": sorted({
                record["phase"] for record in run.commands
                if PHASE_ROLES.get(record["phase"]) in regulated_roles
            }),
            "max_commanded_press_speed_mps": max_press,
            "declared_press_limit_mps": press_limit,
            "max_commanded_correct_speed_mps": max_correct,
            "declared_correct_limit_mps": correct_limit,
            "commands_within_declared_limits": within_limits,
            "peak_reported_normal_force_n": peak_force,
            "authority": authority,
            "scope": (
                "the realised commands stayed inside the limits the "
                "specification declared, and those limits can hold the target; "
                "this is not a claim about any other backend"
            ),
        }),
    ]


def _tangent(frame):
    """Tangential velocity, or zeros when the declared source has no value.

    Differenced kinematics is undefined on a frame without a contact point. A
    gate must not read such a frame as a passing sample, so it reads as zero
    and fails the tolerance comparison instead.
    """
    value = frame.tangent_velocity_world_mps
    return value if value is not None else (0.0, 0.0, 0.0)


def _segments_complete(phases, declared) -> bool:
    """Every declared segment appears, in order, and the run ends released."""
    expected = list(declared)
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
    steps = {run.substeps_per_control for run in runs}
    if len(steps) < 2:
        # Comparing a run against itself proves nothing about the time step.
        raise ValueError(
            f"Time refinement needs at least two distinct physics step counts; "
            f"got substeps_per_control={sorted(steps)}"
        )
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


def finalize_report(report: dict) -> dict:
    """Recompute every verdict from the gate list. The only place that does.

    Callers append gates that need inputs the report builder does not have,
    notably the replay gate, which can only be checked after the manifest is
    on disk. They must finalize through here rather than recomputing a verdict
    by hand: a hand-rolled ``all(gate["passed"] ...)`` silently drops the
    completeness requirement, which is the defect this function prevents.
    """
    gates = report["gates"]
    names = [gate["gate"] for gate in gates]
    passed = {gate["gate"]: gate["passed"] for gate in gates}
    # The required set depends on the controller mode: a force run that
    # emitted only the seven common gates has not been fully checked, and must
    # not be able to report a pass.
    mode = report.get("controller_mode", "trajectory")
    required_run = run_gate_order(mode)
    required_all = gate_order(mode)
    missing_run = [name for name in required_run if name not in names]
    missing_all = [name for name in required_all if name not in names]
    report["gate_order_mode"] = mode
    report["required_gates"] = list(required_all)
    report["gate_names"] = names
    report["missing_run_gates"] = missing_run
    report["missing_gates"] = missing_all
    report["run_checks_passed"] = not missing_run and all(
        passed[name] for name in required_run if name in passed
    )
    report["run_acceptance_complete"] = not missing_all
    report["run_acceptance_passed"] = not missing_all and all(passed.values())
    # Compatibility aliases. The K1 run index and its tests read these names.
    report["acceptance_complete"] = report["run_acceptance_complete"]
    report["acceptance_passed"] = report["run_acceptance_passed"]
    report["k1_acceptance_complete"] = report["run_acceptance_complete"]
    report["k1_acceptance_passed"] = report["run_acceptance_passed"]
    report["all_passed"] = report["run_acceptance_passed"]
    # What this verdict is *not*. Every gate here reads one scenario on one
    # backend; a milestone also requires the other backends, data products and
    # reports its plan declares, and nothing in a single run can establish
    # those. The milestone ledger below is the only thing that may answer it.
    report["milestone"] = milestone_status(mode)
    report["verdict_scope"] = (
        f"run_acceptance_passed requires every gate in {mode} mode's gate "
        f"order ({len(required_all)} gates) on this one run; it is not "
        f"milestone completion, which also requires the deliverables listed "
        f"in report['milestone']"
    )
    return report


def acceptance_report(run, *, refinement_runs=None, run_directory=None) -> dict:
    """Assemble an acceptance report, distinguishing partial from complete.

    Two verdicts, because they answer different questions:

    ``run_checks_passed``
        Every single-run gate was emitted and passed. This is what a partial
        call can establish, and it is *not* K1 acceptance.

    ``k1_acceptance_passed``
        Every gate in ``K1_GATE_ORDER`` was emitted and passed, which requires
        refinement runs at two distinct physics step counts and a written run
        directory. ``all_passed`` is this verdict, so a caller that omits
        those inputs cannot read the result as a full pass.
    """
    report = evaluate_run(run)
    if refinement_runs:
        report["gates"].append(evaluate_time_refinement([run, *refinement_runs]))
    if run_directory is not None:
        report["gates"].append(evaluate_replay(run_directory))
    return finalize_report(report)


def manifest_from_run(run, artifacts, observables) -> ExperimentManifest:
    """Build the run manifest that binds conditions, observables and artifacts."""
    return ExperimentManifest.build(
        schema_version="experiment-manifest/v1",
        run_id=run.run_id,
        scenario={
            "id": run.scenario_id,
            "version": run.scenario_version,
            "description": run.scenario_description,
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
