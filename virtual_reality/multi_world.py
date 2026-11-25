"""Layer 1 for several independent probes sharing one kernel (P2.2.2).

``multi_trajectory`` means one :class:`TrajectoryStateMachine` per probe, not
one machine driving every probe. Each machine reads only its own probe's
delayed feedback and keeps its own segment clock, so a sphere and a box that
reach the surface at different times each run their own approach, hold and
slide. Phase 1 Step 6.5 drove both boxes from one merged phase machine; with
identical probes the two descriptions realise the same timeline, and with
mixed probes only the per-probe one is meaningful.

The kernel is still the single dynamics owner: one ``kernel.step`` advances
every probe in the same control interval. What this stage adds is the
bookkeeping that keeps the per-probe streams separate: one command record and
one :class:`WorldFrame` per probe, each carrying that probe's segment label.
"""

from __future__ import annotations

from contact_kernel import DRIVE_OVERWRITE_PER_CONTROL_STEP, RigidContactKernel
from tactile_contract.pipeline import WorldFrame

from .controllers import TrajectoryStateMachine
from .world import trajectory_config_from_spec

MULTI_TRAJECTORY = "multi_trajectory"


def controllers_from_spec(spec, probe_ids=None) -> dict[str, TrajectoryStateMachine]:
    """One trajectory segment machine per declared probe, in declaration order."""
    data = spec.to_dict()
    mode = data["controller"]["mode"]
    if mode != MULTI_TRAJECTORY:
        raise ValueError(
            f"controllers_from_spec needs controller.mode == {MULTI_TRAJECTORY!r}, got {mode!r}"
        )
    machines = {}
    for index, probe in enumerate(data["probes"]):
        if probe_ids is not None and probe["id"] not in probe_ids:
            continue
        machines[probe["id"]] = TrajectoryStateMachine(trajectory_config_from_spec(spec, index))
    if not machines:
        raise ValueError("No probe selected")
    return machines


class MultiProbeWorldStage:
    """Layer 1: one kernel, one segment machine per probe, per-probe frames."""

    layer = 1
    stage_id = "declared_multi_trajectory_world"

    def __init__(self, kernel: RigidContactKernel, controllers: dict[str, TrajectoryStateMachine]):
        self.kernel = kernel
        self.controllers = dict(controllers)
        declared = [probe.id for probe in kernel.config.probes]
        if list(self.controllers) != declared:
            raise ValueError(
                f"Controllers {list(self.controllers)} do not match the kernel probes {declared}"
            )
        if kernel.config.drive.semantics != DRIVE_OVERWRITE_PER_CONTROL_STEP:
            raise ValueError("Layer 1 only issues once-per-control-step velocity commands")
        self.pair_keys = {
            probe_id: RigidContactKernel.pair_key(probe_id, kernel.config.surface.id)
            for probe_id in self.controllers
        }
        self._commands: dict[str, list[dict]] = {probe_id: [] for probe_id in self.controllers}

    def reset(self) -> None:
        self.kernel.reset()
        for controller in self.controllers.values():
            controller.reset()
        self._commands = {probe_id: [] for probe_id in self.controllers}

    def status(self) -> str:
        return "aborted" if self.abort_reason else "ok"

    @property
    def done(self) -> bool:
        """The run ends when every probe's program has ended, or one aborted."""
        if self.abort_reason:
            return True
        return all(controller.done for controller in self.controllers.values())

    @property
    def abort_reason(self) -> str:
        reasons = [
            f"{probe_id}: {controller.abort_reason}"
            for probe_id, controller in self.controllers.items() if controller.abort_reason
        ]
        return "; ".join(reasons)

    def commands(self, probe_id: str) -> tuple[dict, ...]:
        return tuple(self._commands[probe_id])

    def step(self, feedback: dict) -> dict[str, WorldFrame]:
        """Advance one control interval. ``feedback`` maps probe ID to LoopFeedback."""
        phases = {probe_id: controller.phase for probe_id, controller in self.controllers.items()}
        commands = {
            probe_id: controller.command(feedback[probe_id])
            for probe_id, controller in self.controllers.items()
        }
        kernel_frame = self.kernel.step(commands, phase="per_probe", probe_phases=phases)
        frames = {}
        for probe_id, controller in self.controllers.items():
            pair = kernel_frame.pairs[self.pair_keys[probe_id]]
            controller.observe(kernel_frame.time_s, pair.contact_present, pair.normal_force_n)
            self._commands[probe_id].append({
                "control_step": kernel_frame.control_step,
                "time_s": kernel_frame.time_s,
                "phase": phases[probe_id],
                "phase_after_observation": controller.phase,
                "body": probe_id,
                "command_velocity_world_mps": list(commands[probe_id]),
                "drive_semantics": self.kernel.config.drive.semantics,
                "gravity_compensation_mode": self.kernel.config.drive.gravity_compensation_mode,
                "gravity_compensation_mps2": self.kernel.config.drive.gravity_compensation_mps2,
                "feedback_used": feedback[probe_id].to_dict(),
            })
            frames[probe_id] = WorldFrame(
                time_s=kernel_frame.time_s,
                control_step=kernel_frame.control_step,
                dt_s=kernel_frame.dt_s,
                phase=phases[probe_id],
                command_velocity_world_mps={probe_id: commands[probe_id]},
                body_states=kernel_frame.body_states,
                controller_mode=controller.mode,
                controller_diagnostics=controller.diagnostics(),
                kernel_frame=kernel_frame,
            )
        return frames
