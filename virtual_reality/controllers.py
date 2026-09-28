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
        self._transitions: list[dict] = []

    def reset(self) -> None:
        self.phase = PRE_CONTACT
        self.phase_start_time_s = 0.0
        self.done = False
        self.abort_reason = ""
        self._slide_command_mps = 0.0
        self._slide_servo_started = False
        self._last_command = (0.0, 0.0, 0.0)
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
        self._last_command = command
        return command

    def _slide_velocity(self, feedback) -> float:
        config = self.config
        if config.tangent_speed_feedback_kp <= 0.0:
            self._slide_command_mps = config.slide_speed_mps
            return self._slide_command_mps
        if not self._slide_servo_started:
            initial = config.initial_slide_command_speed_mps
            self._slide_command_mps = config.slide_speed_mps if initial is None else initial
            self._slide_servo_started = True
            return self._slide_command_mps
        error = config.slide_speed_mps - float(feedback.tangent_speed_mps)
        updated = self._slide_command_mps + config.tangent_speed_feedback_kp * error
        self._slide_command_mps = min(max(updated, 0.0), config.max_slide_command_speed_mps)
        return self._slide_command_mps

    def observe(self, time_s: float, contact_present: bool) -> bool:
        """Advance the segment machine using the contact measured this step."""
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
            "slide_servo_active": self._slide_servo_started,
            "done": self.done,
            "abort_reason": self.abort_reason,
        }
