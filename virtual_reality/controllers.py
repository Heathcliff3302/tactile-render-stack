"""Layer 1 trajectory state machine.

Six segments, identical in order and in transition rules to the frozen Step 6
controller: ``pre_contact``, ``approach``, ``hold``, ``lateral_slide``,
``retract``, ``released``. ``released`` is a terminal segment that still
advances the declared release interval, which is what produces the recorded
evidence that contact force returns to zero after release.

One difference from Phase 1 is deliberate. Phase 1 folded gravity
compensation into the velocity command as ``gravity * physics_dt``. Here the
controller commands a target velocity and the drive layer compensates gravity
per substep, so the same command stays meaningful when the physics step is
refined. The numeric value still comes from the declared
``gravity_compensation_mps2`` parameter.
"""

from __future__ import annotations

from dataclasses import dataclass

PRE_CONTACT = "pre_contact"
APPROACH = "approach"
HOLD = "hold"
LATERAL_SLIDE = "lateral_slide"
RETRACT = "retract"
RELEASED = "released"
PHASE_ORDER = (PRE_CONTACT, APPROACH, HOLD, LATERAL_SLIDE, RETRACT, RELEASED)
MOTION_PHASES = (PRE_CONTACT, APPROACH, HOLD, LATERAL_SLIDE, RETRACT)

# Force mode runs the same five motion roles under its own segment names,
# exactly as Step 6 did. Only the approach, hold and slide segments differ;
# retract and released are shared.
APPROACH_FORCE = "approach_force"
REGULATE_FORCE = "regulate_force"
LATERAL_SLIDE_FORCE = "lateral_slide_force"
FORCE_PHASE_ORDER = (
    PRE_CONTACT, APPROACH_FORCE, REGULATE_FORCE, LATERAL_SLIDE_FORCE, RETRACT, RELEASED,
)
FORCE_MOTION_PHASES = FORCE_PHASE_ORDER[:-1]

#: Role each segment plays, so a gate or a tracker can ask "is this the hold
#: segment" without knowing which controller produced the run. Hardcoding
#: ``"hold"`` in a gate is exactly how the acceptance module would silently
#: measure nothing on a force-mode run: the lookup would simply return no
#: frames, and an empty statistic reads as a pass unless something checks.
PHASE_ROLES: dict[str, str] = {
    PRE_CONTACT: "pre_contact",
    APPROACH: "approach", APPROACH_FORCE: "approach",
    HOLD: "hold", REGULATE_FORCE: "hold",
    LATERAL_SLIDE: "slide", LATERAL_SLIDE_FORCE: "slide",
    RETRACT: "retract",
    RELEASED: "released",
}

#: The segment each role takes in a given controller mode.
MODE_PHASES: dict[str, tuple[str, ...]] = {
    "trajectory": PHASE_ORDER,
    "force": FORCE_PHASE_ORDER,
}


def phases_for_mode(mode: str) -> tuple[str, ...]:
    """Declared segment order for a controller mode."""
    try:
        return MODE_PHASES[mode]
    except KeyError:
        raise ValueError(
            f"No segment order is implemented for controller mode {mode!r}; "
            f"implemented modes are {sorted(MODE_PHASES)}"
        ) from None


def phase_with_role(mode: str, role: str) -> str:
    """The segment playing ``role`` in ``mode``."""
    for phase in phases_for_mode(mode):
        if PHASE_ROLES[phase] == role:
            return phase
    raise ValueError(f"Controller mode {mode!r} has no segment with role {role!r}")


@dataclass(frozen=True)
class TrajectoryConfig:
    pre_settle_time_s: float
    approach_timeout_s: float
    hold_time_s: float
    slide_time_s: float
    release_time_s: float
    approach_speed_mps: float
    slide_speed_mps: float
    retract_speed_mps: float
    retract_distance_m: float
    contact_maintain_speed_mps: float
    tangent_speed_feedback_kp: float = 0.0
    max_slide_command_speed_mps: float = 0.12
    initial_slide_command_speed_mps: float | None = None

    def __post_init__(self):
        if self.retract_speed_mps <= 0 or self.retract_distance_m <= 0:
            raise ValueError("Retract needs a positive distance and speed")
        if self.max_slide_command_speed_mps < self.slide_speed_mps:
            raise ValueError("Slide command limit is below the slide target")
        initial = self.initial_slide_command_speed_mps
        if initial is not None and not 0.0 <= initial <= self.max_slide_command_speed_mps:
            raise ValueError(
                f"initial_slide_command_speed_mps={initial} is outside "
                f"[0, {self.max_slide_command_speed_mps}]"
            )

    @property
    def retract_time_s(self) -> float:
        return self.retract_distance_m / self.retract_speed_mps


class TrajectoryStateMachine:
    """Open-loop segment machine with a tangential speed servo during sliding."""

    phases = PHASE_ORDER
    mode = "trajectory"

    def __init__(self, config: TrajectoryConfig):
        self.config = config
        self.phase = PRE_CONTACT
        self.phase_start_time_s = 0.0
        self.done = False
        self.abort_reason = ""
        self._slide_command_mps = 0.0
        self._slide_servo_started = False
        self._last_command = (0.0, 0.0, 0.0)
        self._clamped_steps = 0
        self._transitions: list[dict] = []

    def reset(self) -> None:
        self.phase = PRE_CONTACT
        self.phase_start_time_s = 0.0
        self.done = False
        self.abort_reason = ""
        self._slide_command_mps = 0.0
        self._slide_servo_started = False
        self._last_command = (0.0, 0.0, 0.0)
        self._clamped_steps = 0
        self._transitions = []

    @property
    def transitions(self) -> tuple[dict, ...]:
        return tuple(self._transitions)

    @property
    def last_command_world_mps(self) -> tuple[float, float, float]:
        return self._last_command

    @property
    def slide_command_mps(self) -> float:
        return self._slide_command_mps

    def command(self, feedback) -> tuple[float, float, float]:
        """Target velocity for the next control interval, from delayed feedback."""
        config = self.config
        if self.phase == APPROACH:
            command = (0.0, 0.0, -config.approach_speed_mps)
        elif self.phase == HOLD:
            command = (0.0, 0.0, -config.contact_maintain_speed_mps)
        elif self.phase == LATERAL_SLIDE:
            command = (self._slide_velocity(feedback), 0.0, -config.contact_maintain_speed_mps)
        elif self.phase == RETRACT:
            command = (0.0, 0.0, config.retract_speed_mps)
        else:
            command = (0.0, 0.0, 0.0)
        if self.phase != LATERAL_SLIDE:
            self._slide_command_mps = 0.0
            self._slide_servo_started = False
        # One clamp on the single exit, so no branch can emit an unbounded
        # tangential command. Clamping only inside the feedback branch left
        # the first slide frame free to command any declared initial speed.
        command = self._clamp(command)
        self._last_command = command
        return command

    def _clamp(self, command) -> tuple[float, float, float]:
        limit = self.config.max_slide_command_speed_mps
        tangential = min(max(command[0], -limit), limit)
        if tangential != command[0]:
            self._clamped_steps += 1
        return (tangential, command[1], command[2])

    def _slide_velocity(self, feedback) -> float:
        config = self.config
        limit = config.max_slide_command_speed_mps
        if config.tangent_speed_feedback_kp <= 0.0:
            self._slide_command_mps = min(config.slide_speed_mps, limit)
            return self._slide_command_mps
        if not self._slide_servo_started:
            initial = config.initial_slide_command_speed_mps
            raw = config.slide_speed_mps if initial is None else initial
            self._slide_command_mps = min(max(raw, 0.0), limit)
            self._slide_servo_started = True
            return self._slide_command_mps
        error = config.slide_speed_mps - float(feedback.tangent_speed_mps)
        updated = self._slide_command_mps + config.tangent_speed_feedback_kp * error
        self._slide_command_mps = min(max(updated, 0.0), limit)
        return self._slide_command_mps

    def observe(self, time_s: float, contact_present: bool, normal_force_n: float = 0.0) -> bool:
        """Advance the segment machine using the contact measured this step.

        ``normal_force_n`` completes the controller protocol shared with
        ``ForceFeedbackStateMachine``; the trajectory segments key off contact
        presence and elapsed time only, so it is accepted and not read.
        """
        if self.done:
            return False
        config = self.config
        elapsed = time_s - self.phase_start_time_s
        next_phase = None
        if self.phase == PRE_CONTACT:
            if elapsed >= config.pre_settle_time_s:
                next_phase = APPROACH
        elif self.phase == APPROACH:
            if contact_present:
                next_phase = HOLD
            elif elapsed >= config.approach_timeout_s:
                self.abort_reason = "approach timeout: no mechanical contact"
                self.done = True
        elif self.phase == HOLD:
            if elapsed >= config.hold_time_s:
                next_phase = LATERAL_SLIDE
        elif self.phase == LATERAL_SLIDE:
            if elapsed >= config.slide_time_s:
                next_phase = RETRACT
        elif self.phase == RETRACT:
            if elapsed >= config.retract_time_s:
                next_phase = RELEASED
        elif self.phase == RELEASED:
            if elapsed >= config.release_time_s:
                self.done = True
        if next_phase is None:
            return False
        self._transitions.append({
            "time_s": time_s,
            "from": self.phase,
            "to": next_phase,
            "contact_present": contact_present,
        })
        self.phase = next_phase
        self.phase_start_time_s = time_s
        return True

    def diagnostics(self) -> dict:
        return {
            "phase": self.phase,
            "phase_start_time_s": self.phase_start_time_s,
            "slide_command_mps": self._slide_command_mps,
            "slide_command_limit_mps": self.config.max_slide_command_speed_mps,
            "slide_servo_active": self._slide_servo_started,
            "clamped_command_steps": self._clamped_steps,
            "done": self.done,
            "abort_reason": self.abort_reason,
        }


@dataclass(frozen=True)
class ForceConfig:
    """Force-mode controller parameters, all declared by the specification."""

    pre_settle_time_s: float
    approach_timeout_s: float
    regulate_timeout_s: float
    hold_time_s: float
    slide_time_s: float
    release_time_s: float
    approach_speed_mps: float
    slide_speed_mps: float
    retract_speed_mps: float
    retract_distance_m: float
    target_force_n: float
    force_tolerance_n: float
    force_kp_m_per_ns: float
    force_ki_m_per_ns2: float
    force_integral_limit_ns: float
    max_press_speed_mps: float
    max_correct_speed_mps: float
    settle_frames: int
    control_dt_s: float
    max_slide_command_speed_mps: float = 0.12

    def __post_init__(self):
        if self.retract_speed_mps <= 0 or self.retract_distance_m <= 0:
            raise ValueError("Retract needs a positive distance and speed")
        if self.max_slide_command_speed_mps < self.slide_speed_mps:
            raise ValueError("Slide command limit is below the slide target")
        if self.settle_frames < 1:
            raise ValueError("settle_frames must be at least one")
        if self.force_integral_limit_ns <= 0:
            raise ValueError("A non-positive integral limit gives the regulator no authority")

    @property
    def retract_time_s(self) -> float:
        return self.retract_distance_m / self.retract_speed_mps


class ForceFeedbackStateMachine:
    """Step 6's PI normal-force regulator, on this kernel's drive semantics.

    Same six segments and the same transition rules as
    ``ForceFeedbackController`` in Phase 1: the target must be held within
    tolerance for ``settle_frames`` consecutive frames before the hold clock
    starts, and losing contact or leaving the band restarts it.

    Two differences from Phase 1, both forced by the drive semantics and both
    recorded rather than absorbed:

    *Gravity compensation* is applied per substep by the drive layer, not
    folded into the command, for the same reason as in
    ``TrajectoryStateMachine``: a command written once per control step must
    stay meaningful when the physics step is refined.

    *The integral gain is derived, not inherited.* Here the frame force is
    ``m * v_command / output_dt``, so at steady state the whole command speed
    must come from the integrator, bounded by ``ki * force_integral_limit``.
    Phase 1's gains supply 0.2 m/s where a 1 N target needs 0.3333 m/s. That
    is rejected at load by ``drive_limits.enforce_regulator_authority`` rather
    than discovered as a failing gate; see that module for the derivation.

    The regulator reads the *sampled* normal force, the same observable the
    acceptance gates score, so the controller and its gates cannot disagree
    about what the force was. The sampling threshold is therefore inside the
    loop: a frame filtered out as below threshold reads as no contact, which
    is what makes the regulator push harder rather than coast.
    """

    phases = FORCE_PHASE_ORDER
    mode = "force"

    def __init__(self, config: ForceConfig):
        self.config = config
        self.phase = PRE_CONTACT
        self.phase_start_time_s = 0.0
        self.done = False
        self.abort_reason = ""
        self._reset_regulator()
        self._last_command = (0.0, 0.0, 0.0)
        self._clamped_steps = 0
        self._saturated_integral_steps = 0
        self._transitions: list[dict] = []
        self._settled_at_time_s: float | None = None

    def _reset_regulator(self) -> None:
        self._integral_ns = 0.0
        self._within_tolerance_frames = 0
        self._target_locked = False
        self._hold_elapsed_s = 0.0
        self._command_vz_mps = 0.0

    def reset(self) -> None:
        self.phase = PRE_CONTACT
        self.phase_start_time_s = 0.0
        self.done = False
        self.abort_reason = ""
        self._reset_regulator()
        self._last_command = (0.0, 0.0, 0.0)
        self._clamped_steps = 0
        self._saturated_integral_steps = 0
        self._transitions = []
        self._settled_at_time_s = None

    @property
    def transitions(self) -> tuple[dict, ...]:
        return tuple(self._transitions)

    @property
    def last_command_world_mps(self) -> tuple[float, float, float]:
        return self._last_command

    @property
    def force_target_reached(self) -> bool:
        return self._settled_at_time_s is not None

    @property
    def settled_at_time_s(self) -> float | None:
        return self._settled_at_time_s

    def command(self, feedback) -> tuple[float, float, float]:
        """Target velocity for the next control interval, from delayed feedback."""
        config = self.config
        tangential = 0.0
        if self.phase == APPROACH_FORCE:
            vertical = -config.approach_speed_mps
        elif self.phase in (REGULATE_FORCE, LATERAL_SLIDE_FORCE):
            vertical = self._regulate(feedback)
            if self.phase == LATERAL_SLIDE_FORCE:
                tangential = config.slide_speed_mps
        elif self.phase == RETRACT:
            vertical = config.retract_speed_mps
        else:
            vertical = 0.0
        if self.phase not in (REGULATE_FORCE, LATERAL_SLIDE_FORCE):
            self._command_vz_mps = vertical
        command = self._clamp((tangential, 0.0, vertical))
        self._last_command = command
        return command

    def _regulate(self, feedback) -> float:
        """PI step on the sampled normal force. Returns a vertical command."""
        config = self.config
        if not feedback.contact_present:
            # No contact means no force to regulate. Phase 1 drops the
            # integral and falls back to approaching, so a regulator that has
            # separated does not carry stale windup into re-contact.
            self._integral_ns = 0.0
            self._command_vz_mps = -min(config.approach_speed_mps, config.max_press_speed_mps)
            return self._command_vz_mps
        error_n = config.target_force_n - feedback.normal_force_n
        limit = config.force_integral_limit_ns
        raw_integral = self._integral_ns + error_n * config.control_dt_s
        self._integral_ns = min(max(raw_integral, -limit), limit)
        if self._integral_ns != raw_integral:
            self._saturated_integral_steps += 1
        desired = -(config.force_kp_m_per_ns * error_n
                    + config.force_ki_m_per_ns2 * self._integral_ns)
        # Downward is negative, so the press limit bounds the negative side and
        # the correction limit the positive one.
        self._command_vz_mps = min(max(desired, -config.max_press_speed_mps),
                                   config.max_correct_speed_mps)
        return self._command_vz_mps

    def _clamp(self, command) -> tuple[float, float, float]:
        limit = self.config.max_slide_command_speed_mps
        tangential = min(max(command[0], -limit), limit)
        if tangential != command[0]:
            self._clamped_steps += 1
        return (tangential, command[1], command[2])

    def observe(self, time_s: float, contact_present: bool, normal_force_n: float = 0.0) -> bool:
        """Advance the segment machine using this step's sampled measurement."""
        if self.done:
            return False
        config = self.config
        elapsed = time_s - self.phase_start_time_s
        next_phase = None
        if self.phase == PRE_CONTACT:
            if elapsed >= config.pre_settle_time_s:
                next_phase = APPROACH_FORCE
        elif self.phase == APPROACH_FORCE:
            if contact_present:
                next_phase = REGULATE_FORCE
                self._reset_regulator()
            elif elapsed >= config.approach_timeout_s:
                self.abort_reason = "approach timeout: no mechanical contact"
                self.done = True
        elif self.phase == REGULATE_FORCE:
            if contact_present:
                if abs(config.target_force_n - normal_force_n) <= config.force_tolerance_n:
                    self._within_tolerance_frames += 1
                else:
                    self._within_tolerance_frames = 0
                    self._target_locked = False
                    self._hold_elapsed_s = 0.0
                if not self._target_locked and self._within_tolerance_frames >= config.settle_frames:
                    self._target_locked = True
                    self._hold_elapsed_s = 0.0
                    if self._settled_at_time_s is None:
                        self._settled_at_time_s = time_s
                if self._target_locked:
                    self._hold_elapsed_s += config.control_dt_s
                    if self._hold_elapsed_s >= config.hold_time_s:
                        next_phase = LATERAL_SLIDE_FORCE
            else:
                self._within_tolerance_frames = 0
                self._target_locked = False
                self._hold_elapsed_s = 0.0
                self._integral_ns = 0.0
            if next_phase is None and elapsed >= config.regulate_timeout_s:
                self.abort_reason = (
                    "force regulation timeout: the target was not reached and held"
                )
                self.done = True
        elif self.phase == LATERAL_SLIDE_FORCE:
            if elapsed >= config.slide_time_s:
                next_phase = RETRACT
        elif self.phase == RETRACT:
            if elapsed >= config.retract_time_s:
                next_phase = RELEASED
        elif self.phase == RELEASED:
            if elapsed >= config.release_time_s:
                self.done = True
        if next_phase is None:
            return False
        self._transitions.append({
            "time_s": time_s,
            "from": self.phase,
            "to": next_phase,
            "contact_present": contact_present,
            "normal_force_n": normal_force_n,
        })
        self.phase = next_phase
        self.phase_start_time_s = time_s
        return True

    def diagnostics(self) -> dict:
        return {
            "phase": self.phase,
            "phase_start_time_s": self.phase_start_time_s,
            "command_vz_mps": self._command_vz_mps,
            "force_integral_ns": self._integral_ns,
            "force_integral_limit_ns": self.config.force_integral_limit_ns,
            "integral_authority_mps": (
                self.config.force_ki_m_per_ns2 * self.config.force_integral_limit_ns
            ),
            "saturated_integral_steps": self._saturated_integral_steps,
            "within_tolerance_frames": self._within_tolerance_frames,
            "settle_frames": self.config.settle_frames,
            "target_locked": self._target_locked,
            "force_target_reached": self.force_target_reached,
            "settled_at_time_s": self._settled_at_time_s,
            "hold_elapsed_s": self._hold_elapsed_s,
            "slide_command_limit_mps": self.config.max_slide_command_speed_mps,
            "clamped_command_steps": self._clamped_steps,
            "done": self.done,
            "abort_reason": self.abort_reason,
        }
