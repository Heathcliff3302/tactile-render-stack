"""Semi-implicit rigid integrator with an analytic non-penetration impulse.

Scope is the K1 case stated in ``docs/contact_kernel_design.md``: one rotation
locked probe per contact pair, one fixed surface, one contact point, no
friction and no restitution. There is no generic LCP here, and a multi-point
or stacked configuration must not be routed through this solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import TopFaceGeometry, dot
from .spec import KernelConfig, Vec3


@dataclass
class BodyState:
    """Mutable rigid state of one probe. Angular motion is locked in K1."""

    id: str
    position_world_m: Vec3
    linear_velocity_mps: Vec3
    angular_velocity_rps: Vec3 = (0.0, 0.0, 0.0)

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "position_world_m": list(self.position_world_m),
            "linear_velocity_mps": list(self.linear_velocity_mps),
            "angular_velocity_rps": list(self.angular_velocity_rps),
        }


@dataclass(frozen=True)
class SubstepTrace:
    """One physics substep of one probe, including the no-contact case."""

    index: int
    time_s: float
    physics_dt_s: float
    gap_before_m: float
    gap_after_m: float
    normal_velocity_free_mps: float
    normal_velocity_after_mps: float
    normal_impulse_ns: float
    position_correction_m: float
    contact: bool
    inside_face: bool
    rejected_reason: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "time_s": self.time_s,
            "physics_dt_s": self.physics_dt_s,
            "gap_before_m": self.gap_before_m,
            "gap_after_m": self.gap_after_m,
            "normal_velocity_free_mps": self.normal_velocity_free_mps,
            "normal_velocity_after_mps": self.normal_velocity_after_mps,
            "normal_impulse_ns": self.normal_impulse_ns,
            "position_correction_m": self.position_correction_m,
            "contact": self.contact,
            "inside_face": self.inside_face,
            "rejected_reason": self.rejected_reason,
        }


@dataclass(frozen=True)
class ProbeSolution:
    """Solver result for one probe over one complete control interval."""

    probe_id: str
    surface_id: str
    command_velocity_mps: Vec3
    applied_velocity_mps: Vec3
    position_world_m: Vec3
    linear_velocity_mps: Vec3
    normal_world: Vec3
    normal_impulse_ns: float
    normal_force_n: float
    contact_substeps: int
    contact_point_world_m: Vec3
    gap_m: float
    penetration_m: float
    position_correction_m: float
    inside_face: bool
    rejected_reason: str
    substeps: tuple[SubstepTrace, ...] = field(default_factory=tuple)

    @property
    def contact(self) -> bool:
        return self.contact_substeps > 0

    def to_dict(self) -> dict:
        return {
            "probe_id": self.probe_id,
            "surface_id": self.surface_id,
            "command_velocity_mps": list(self.command_velocity_mps),
            "applied_velocity_mps": list(self.applied_velocity_mps),
            "position_world_m": list(self.position_world_m),
            "linear_velocity_mps": list(self.linear_velocity_mps),
            "normal_impulse_ns": self.normal_impulse_ns,
            "normal_force_n": self.normal_force_n,
            "contact_substeps": self.contact_substeps,
            "gap_m": self.gap_m,
            "penetration_m": self.penetration_m,
            "position_correction_m": self.position_correction_m,
            "inside_face": self.inside_face,
            "rejected_reason": self.rejected_reason,
            "substeps": [trace.to_dict() for trace in self.substeps],
        }


class RigidImpulseSolver:
    """Advance every probe by one control interval of ``substeps`` substeps."""

    def __init__(self, config: KernelConfig):
        self.config = config
        self.geometry = TopFaceGeometry(config.surface)
        self.time_s = 0.0
        self.control_step = 0
        self.physics_step = 0
        self.states: dict[str, BodyState] = {}
        self.reset()

    def reset(self) -> None:
        self.time_s = 0.0
        self.control_step = 0
        self.physics_step = 0
        self.states = {
            probe.id: BodyState(
                id=probe.id,
                position_world_m=probe.initial_position_world_m,
                linear_velocity_mps=probe.initial_linear_velocity_mps,
            )
            for probe in self.config.probes
        }

    def advance_control_step(self, commands: dict[str, Vec3]) -> tuple[ProbeSolution, ...]:
        missing = {probe.id for probe in self.config.probes} - set(commands)
        if missing:
            raise KeyError(f"Missing velocity command for {sorted(missing)}")
        substeps = self.config.timing.substeps_per_control
        step_h = self.config.timing.physics_dt_s
        gravity = self.config.solver.gravity_mps2
        compensation = self.config.gravity_compensation_world
        acceleration = (
            gravity[0] + compensation[0],
            gravity[1] + compensation[1],
            gravity[2] + compensation[2],
        )
        damping_factor = max(0.0, 1.0 - self.config.solver.linear_damping_per_s * step_h)

        # Drive semantics: the command overwrites the linear velocity once per
        # control interval, exactly as ``set_linear_velocity`` does in Step 6.
        for probe in self.config.probes:
            state = self.states[probe.id]
            state.linear_velocity_mps = tuple(float(value) for value in commands[probe.id])

        accumulator = {
            probe.id: {
                "impulse": 0.0,
                "contact_substeps": 0,
                "penetration": 0.0,
                "correction": 0.0,
                "traces": [],
            }
            for probe in self.config.probes
        }
        applied = {probe.id: self.states[probe.id].linear_velocity_mps for probe in self.config.probes}

        for index in range(substeps):
            self.physics_step += 1
            substep_time = self.time_s + (index + 1) * step_h
            for probe in self.config.probes:
                trace = self._advance_substep(probe, step_h, acceleration, damping_factor, index, substep_time)
                record = accumulator[probe.id]
                record["impulse"] += trace.normal_impulse_ns
                record["contact_substeps"] += int(trace.contact)
                record["penetration"] = max(
                    record["penetration"], -trace.gap_before_m, trace.position_correction_m, 0.0
                )
                record["correction"] += trace.position_correction_m
                record["traces"].append(trace)

        # Derived from the step count, not accumulated, so a long run cannot
        # drift away from the interval-end timestamps the contract expects.
        self.control_step += 1
        self.time_s = self.control_step * self.config.timing.control_dt_s

        solutions = []
        for probe in self.config.probes:
            state = self.states[probe.id]
            record = accumulator[probe.id]
            query = self.geometry.query_sphere(state.position_world_m, probe.radius_m,
                                               surface_offset_m=self.config.solver.rest_offset_m)
            impulse = record["impulse"]
            solutions.append(ProbeSolution(
                probe_id=probe.id,
                surface_id=self.config.surface.id,
                command_velocity_mps=tuple(float(value) for value in commands[probe.id]),
                applied_velocity_mps=applied[probe.id],
                position_world_m=state.position_world_m,
                linear_velocity_mps=state.linear_velocity_mps,
                normal_world=self.geometry.normal_world,
                normal_impulse_ns=impulse,
                normal_force_n=impulse / self.config.timing.output_dt_s,
                contact_substeps=record["contact_substeps"],
                contact_point_world_m=query.point_world,
                gap_m=query.gap_m,
                penetration_m=record["penetration"],
                position_correction_m=record["correction"],
                inside_face=query.inside_face,
                rejected_reason=query.rejected_reason,
                substeps=tuple(record["traces"]),
            ))
        return tuple(solutions)

    def _advance_substep(self, probe, step_h, acceleration, damping_factor, index, substep_time) -> SubstepTrace:
        state = self.states[probe.id]
        velocity = (
            (state.linear_velocity_mps[0] + acceleration[0] * step_h) * damping_factor,
            (state.linear_velocity_mps[1] + acceleration[1] * step_h) * damping_factor,
            (state.linear_velocity_mps[2] + acceleration[2] * step_h) * damping_factor,
        )
        query = self.geometry.query_sphere(state.position_world_m, probe.radius_m,
                                          surface_offset_m=self.config.solver.rest_offset_m)
        normal = query.normal_world
        free_normal_speed = dot(velocity, normal)
        impulse = 0.0
        contact = query.inside_face and (query.gap_m <= 0.0 or query.gap_m + free_normal_speed * step_h <= 0.0)
        if contact:
            # Keep the next gap non-negative. A separated probe may still close
            # the remaining gap within this substep; a penetrating one is only
            # stopped, never pushed out, because impulse-driven depenetration
            # is not implemented and ``max_depenetration_velocity_mps`` is
            # pinned to zero by SolverSpec.
            target_normal_speed = -query.gap_m / step_h if query.gap_m >= 0.0 else 0.0
            if free_normal_speed < target_normal_speed:
                impulse = probe.mass_kg * (target_normal_speed - free_normal_speed)
                velocity = (
                    velocity[0] + impulse * normal[0] / probe.mass_kg,
                    velocity[1] + impulse * normal[1] / probe.mass_kg,
                    velocity[2] + impulse * normal[2] / probe.mass_kg,
                )
        position = (
            state.position_world_m[0] + velocity[0] * step_h,
            state.position_world_m[1] + velocity[1] * step_h,
            state.position_world_m[2] + velocity[2] * step_h,
        )
        after = self.geometry.query_sphere(position, probe.radius_m,
                                          surface_offset_m=self.config.solver.rest_offset_m)
        correction = 0.0
        if after.inside_face and after.gap_m < 0.0:
            # Numerical penetration is removed geometrically and reported on its
            # own, so it never appears as extra contact impulse or force.
            correction = -after.gap_m
            position = (position[0], position[1], position[2] + correction)
            after = self.geometry.query_sphere(position, probe.radius_m,
                                              surface_offset_m=self.config.solver.rest_offset_m)
        state.position_world_m = position
        state.linear_velocity_mps = velocity
        return SubstepTrace(
            index=index,
            time_s=substep_time,
            physics_dt_s=step_h,
            gap_before_m=query.gap_m,
            gap_after_m=after.gap_m,
            normal_velocity_free_mps=free_normal_speed,
            normal_velocity_after_mps=dot(velocity, normal),
            normal_impulse_ns=impulse,
            position_correction_m=correction,
            contact=contact,
            inside_face=query.inside_face,
            rejected_reason=query.rejected_reason,
        )
