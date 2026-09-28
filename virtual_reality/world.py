"""Layer 1 stage and the ``ExperimentSpec`` to kernel-config adapter.

Layer 1 declares the world, owns the fixed simulation clock and issues the
velocity command. It also owns the kernel instance, because one contact pair
must have exactly one authoritative dynamics owner per step: layer 2 reads the
kernel result and formats it, it never advances physics of its own.
"""

from __future__ import annotations

import math

from contact_kernel import (
    DRIVE_OVERWRITE_PER_CONTROL_STEP,
    GRAVITY_COMPENSATION_PER_SUBSTEP,
    DriveSpec,
    GridSpec,
    KernelConfig,
    ProbeSpec,
    RigidContactKernel,
    SamplingSpec,
    SolverSpec,
    SurfaceSpec,
    TimingSpec,
)
from tactile_contract.pipeline import WorldFrame

from .controllers import TrajectoryConfig, TrajectoryStateMachine
from .runtime_policy import enforce_runtime


def kernel_config_from_spec(spec, *, substeps_per_control: int | None = None) -> KernelConfig:
    """Map a validated ``ExperimentSpec`` onto kernel dataclasses.

    Only declared fields are read. ``controller.parameters
    .gravity_compensation_mps2`` is routed to the drive layer rather than into
    the velocity command, which is the one documented relocation relative to
    Step 6; see ``virtual_reality/controllers.py``.
    """
    data = spec.to_dict()
    timing = data["timing"]
    runtime = {name: setting["value"] for name, setting in data["runtime"].items()}
    unresolved = sorted(name for name, setting in data["runtime"].items() if setting["status"] == "unresolved")
    if unresolved:
        raise ValueError(f"Kernel needs resolved runtime settings; unresolved: {unresolved}")
    # Reject anything the kernel cannot honour before a run can produce
    # results that look valid. See virtual_reality/runtime_policy.py.
    enforce_runtime(spec)
    substeps = int(timing["substeps_per_control"] if substeps_per_control is None else substeps_per_control)
    physics_dt = timing["control_dt_s"] / substeps
    surface = SurfaceSpec(
        id=data["surface"]["id"],
        center_world_m=data["surface"]["center_world_m"],
        size_m=data["surface"]["size_m"],
        motion=data["surface"]["motion"],
    )
    probes = tuple(
        ProbeSpec(
            id=probe["id"],
            shape=probe["shape"],
            mass_kg=probe["mass_kg"],
            initial_position_world_m=probe["initial_position_world_m"],
            radius_m=probe["radius_m"],
            size_m=probe["size_m"],
            angular_mode=probe["angular_mode"],
            initial_linear_velocity_mps=runtime["initial_linear_velocity_mps"],
        )
        for probe in data["probes"]
    )
    return KernelConfig(
        surface=surface,
        probes=probes,
        timing=TimingSpec(
            physics_dt_s=physics_dt,
            control_dt_s=timing["control_dt_s"],
            output_dt_s=timing["output_dt_s"],
            substeps_per_control=substeps,
        ),
        grid=GridSpec(
            rows=data["grid"]["rows"],
            cols=data["grid"]["cols"],
            cell_size_m=data["grid"]["cell_size_m"],
            origin_local_m=data["grid"]["origin_local_m"],
            edge_policy=data["grid"]["edge_policy"],
            connectivity=data["grid"]["connectivity"],
        ),
        solver=SolverSpec(
            gravity_mps2=runtime["gravity_mps2"],
            restitution=runtime["restitution"],
            static_friction=runtime["static_friction"],
            dynamic_friction=runtime["dynamic_friction"],
            linear_damping_per_s=runtime["linear_damping_per_s"],
            contact_offset_m=runtime["contact_offset_m"],
            rest_offset_m=runtime["rest_offset_m"],
            max_depenetration_velocity_mps=runtime["max_depenetration_velocity_mps"],
            solver_type=runtime["solver_type"],
        ),
        sampling=SamplingSpec(
            force_threshold_n=data["sampling"]["force_threshold_n"],
            point_match_distance_m=data["sampling"]["point_match_distance_m"],
            max_gap_frames=data["sampling"]["max_gap_frames"],
            slide_threshold_mps=data["sampling"]["slide_threshold_mps"],
            direction_threshold_m=data["sampling"]["direction_threshold_m"],
            filter_alpha=data["sampling"]["filter_alpha"],
            kinematics_source=data["sampling"]["kinematics_source"],
        ),
        drive=DriveSpec(
            semantics=data["controller"]["drive_semantics"],
            gravity_compensation_mps2=data["controller"]["parameters"]["gravity_compensation_mps2"],
            gravity_compensation_mode=GRAVITY_COMPENSATION_PER_SUBSTEP,
        ),
        backend_id=data["backend_id"],
    )


def trajectory_config_from_spec(spec) -> TrajectoryConfig:
    data = spec.to_dict()
    parameters = data["controller"]["parameters"]
    probe = data["probes"][0]
    return TrajectoryConfig(
        pre_settle_time_s=parameters["pre_settle_time_s"],
        approach_timeout_s=parameters["approach_timeout_s"],
        hold_time_s=parameters["hold_time_s"],
        slide_time_s=parameters["slide_time_s"],
        release_time_s=parameters["release_time_s"],
        approach_speed_mps=parameters["approach_speed_mps"],
        slide_speed_mps=parameters["slide_speed_mps"],
        retract_speed_mps=parameters["retract_speed_mps"],
        retract_distance_m=probe["retract_distance_m"],
        contact_maintain_speed_mps=parameters["contact_maintain_speed_mps"],
        tangent_speed_feedback_kp=parameters["tangent_speed_feedback_kp"],
        max_slide_command_speed_mps=parameters["max_slide_command_speed_mps"],
        initial_slide_command_speed_mps=parameters["initial_slide_command_speed_mps"],
    )


def step_budget_from_spec(spec, *, margin_steps: int = 30) -> int:
    timing = spec.to_dict()["timing"]
    return int(math.ceil(timing["max_duration_s"] / timing["control_dt_s"])) + margin_steps


class KernelWorldStage:
    """Layer 1: declared world, fixed clock, trajectory command, kernel owner."""

    layer = 1
    stage_id = "declared_trajectory_world"

    def __init__(self, kernel: RigidContactKernel, controller: TrajectoryStateMachine, *, probe_id: str):
        self.kernel = kernel
        self.controller = controller
        self.probe_id = probe_id
        self.pair_key = RigidContactKernel.pair_key(probe_id, kernel.config.surface.id)
        if kernel.config.drive.semantics != DRIVE_OVERWRITE_PER_CONTROL_STEP:
            raise ValueError("Layer 1 only issues once-per-control-step velocity commands")
        self._commands: list[dict] = []

    def reset(self) -> None:
        self.kernel.reset()
        self.controller.reset()
        self._commands = []

    def status(self) -> str:
        return "aborted" if self.controller.abort_reason else "ok"

    @property
    def done(self) -> bool:
        return self.controller.done

    @property
    def abort_reason(self) -> str:
        return self.controller.abort_reason

    @property
    def commands(self) -> tuple[dict, ...]:
        """Realised command stream. This is evidence, not a predeclared plan."""
        return tuple(self._commands)

    def step(self, feedback) -> WorldFrame:
        phase = self.controller.phase
        command = self.controller.command(feedback)
        kernel_frame = self.kernel.step({self.probe_id: command}, phase=phase)
        pair = kernel_frame.pairs[self.pair_key]
        self.controller.observe(kernel_frame.time_s, pair.contact_present)
        self._commands.append({
            "control_step": kernel_frame.control_step,
            "time_s": kernel_frame.time_s,
            "phase": phase,
            "phase_after_observation": self.controller.phase,
            "body": self.probe_id,
            "command_velocity_world_mps": list(command),
            "drive_semantics": self.kernel.config.drive.semantics,
            "gravity_compensation_mode": self.kernel.config.drive.gravity_compensation_mode,
            "gravity_compensation_mps2": self.kernel.config.drive.gravity_compensation_mps2,
            "feedback_used": feedback.to_dict(),
        })
        return WorldFrame(
            time_s=kernel_frame.time_s,
            control_step=kernel_frame.control_step,
            dt_s=kernel_frame.dt_s,
            phase=phase,
            command_velocity_world_mps={self.probe_id: command},
            body_states=kernel_frame.body_states,
            controller_mode=self.controller.mode,
            controller_diagnostics=self.controller.diagnostics(),
            kernel_frame=kernel_frame,
        )
