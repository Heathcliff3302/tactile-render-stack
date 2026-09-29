"""Five-layer loop: stage boundaries, boundary records and the driver.

The layers follow the system architecture in the root roadmap:

    layer 1 world state -> layer 2 contact -> layer 3 material response
    -> layer 4 standard tactile target -> layer 5 actuator inverse model

``InteractionState v2`` and ``ExperimentManifest v1`` are versioned JSON
contracts. The layer 3, 4 and 5 records below are K1 in-process interfaces:
they carry units, a model identifier and a status marker, but they are not
frozen contracts yet. They become versioned schemas at K4 and K5, when a real
material model and a real actuator define what has to stay stable. Nothing
here computes physics; each stage implementation lives in its own layer
directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

LOOP_SCHEMA_VERSION = "five-layer-loop/k1"

STATUS_MEASURED = "measured"
STATUS_DERIVED = "derived"
STATUS_PASSTHROUGH = "passthrough"
STATUS_UNAVAILABLE = "unavailable"

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class LoopFeedback:
    """What layer 1 and layer 5 may read from the previous closed step.

    The one-step delay is explicit: a controller never sees the force produced
    by the command it is about to emit.
    """

    control_step: int = 0
    contact_present: bool = False
    normal_force_n: float = 0.0
    tangent_speed_mps: float = 0.0
    actuator_measured_force_n: float = 0.0

    def to_dict(self) -> dict:
        return {
            "control_step": self.control_step,
            "contact_present": self.contact_present,
            "normal_force_n": self.normal_force_n,
            "tangent_speed_mps": self.tangent_speed_mps,
            "actuator_measured_force_n": self.actuator_measured_force_n,
        }


@dataclass(frozen=True)
class WorldFrame:
    """Layer 1 output: declared world state and the command it issued."""

    time_s: float
    control_step: int
    dt_s: float
    phase: str
    command_velocity_world_mps: dict[str, Vec3]
    body_states: dict[str, dict]
    controller_mode: str
    controller_diagnostics: dict = field(default_factory=dict)
    kernel_frame: object | None = None

    def to_dict(self) -> dict:
        return {
            "time_s": self.time_s,
            "control_step": self.control_step,
            "dt_s": self.dt_s,
            "phase": self.phase,
            "controller_mode": self.controller_mode,
            "command_velocity_world_mps": {
                body: list(command) for body, command in self.command_velocity_world_mps.items()
            },
            "body_states": self.body_states,
            "controller_diagnostics": self.controller_diagnostics,
        }


@dataclass(frozen=True)
class MaterialResponse:
    """Layer 3 output: what the material does under this contact condition.

    ``model_id`` names the model that produced the numbers. The K1 rigid
    baseline does not recompute force: the kernel constraint is the single
    authoritative dynamics solver for the step, so the response is marked
    ``passthrough`` and the elastic fields stay unavailable.
    """

    time_s: float
    control_step: int
    body0: str
    body1: str
    model_id: str
    status: str
    contact_present: bool
    normal_force_n: float
    tangent_force_world_n: Vec3
    indentation_m: float
    indentation_rate_mps: float
    stiffness_n_per_m: float | None
    damping_ns_per_m: float | None
    contact_area_m2: float | None
    contact_area_method: str
    pressure_mean_pa: float | None
    pressure_peak_pa: float | None
    internal_state: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "time_s": self.time_s,
            "control_step": self.control_step,
            "body0": self.body0,
            "body1": self.body1,
            "model_id": self.model_id,
            "status": self.status,
            "contact_present": self.contact_present,
            "normal_force_n": self.normal_force_n,
            "tangent_force_world_n": list(self.tangent_force_world_n),
            "indentation_m": self.indentation_m,
            "indentation_rate_mps": self.indentation_rate_mps,
            "stiffness_n_per_m": self.stiffness_n_per_m,
            "damping_ns_per_m": self.damping_ns_per_m,
            "contact_area_m2": self.contact_area_m2,
            "contact_area_method": self.contact_area_method,
            "pressure_mean_pa": self.pressure_mean_pa,
            "pressure_peak_pa": self.pressure_peak_pa,
            "internal_state": self.internal_state,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class TactileTarget:
    """Layer 4 output: one device-independent tactile target sample."""

    time_s: float
    control_step: int
    target_id: str
    contact_state: str
    normal_force_n: float
    tangent_force_world_n: Vec3
    contact_area_m2: float | None
    indentation_m: float
    vibration_amplitude_n: float | None
    vibration_frequency_hz: float | None
    temperature_k: float | None
    channels: tuple[str, ...] = ("normal_force",)
    source_model_id: str = ""
    status: str = STATUS_DERIVED

    def to_dict(self) -> dict:
        return {
            "time_s": self.time_s,
            "control_step": self.control_step,
            "target_id": self.target_id,
            "contact_state": self.contact_state,
            "normal_force_n": self.normal_force_n,
            "tangent_force_world_n": list(self.tangent_force_world_n),
            "contact_area_m2": self.contact_area_m2,
            "indentation_m": self.indentation_m,
            "vibration_amplitude_n": self.vibration_amplitude_n,
            "vibration_frequency_hz": self.vibration_frequency_hz,
            "temperature_k": self.temperature_k,
            "channels": list(self.channels),
            "source_model_id": self.source_model_id,
            "status": self.status,
        }


@dataclass(frozen=True)
class ActuatorCommand:
    """Layer 5 input: one command for one physical channel."""

    time_s: float
    control_step: int
    actuator_id: str
    channel: str
    commanded_force_n: float
    saturated: bool
    clipped_by_n: float
    inverse_model_id: str

    def to_dict(self) -> dict:
        return {
            "time_s": self.time_s,
            "control_step": self.control_step,
            "actuator_id": self.actuator_id,
            "channel": self.channel,
            "commanded_force_n": self.commanded_force_n,
            "saturated": self.saturated,
            "clipped_by_n": self.clipped_by_n,
            "inverse_model_id": self.inverse_model_id,
        }


@dataclass(frozen=True)
class ActuatorMeasurement:
    """Layer 5 output: what the channel actually delivered, and the error."""

    time_s: float
    control_step: int
    actuator_id: str
    channel: str
    target_force_n: float
    measured_force_n: float
    error_n: float
    latency_s: float
    model_id: str
    status: str
    watchdog_tripped: bool = False

    def to_dict(self) -> dict:
        return {
            "time_s": self.time_s,
            "control_step": self.control_step,
            "actuator_id": self.actuator_id,
            "channel": self.channel,
            "target_force_n": self.target_force_n,
            "measured_force_n": self.measured_force_n,
            "error_n": self.error_n,
            "latency_s": self.latency_s,
            "model_id": self.model_id,
            "status": self.status,
            "watchdog_tripped": self.watchdog_tripped,
        }


@dataclass(frozen=True)
class LoopFrame:
    """One closed pass through all five layers."""

    control_step: int
    time_s: float
    phase: str
    loop_state: str
    world: WorldFrame
    interaction: object
    material: MaterialResponse
    target: TactileTarget
    command: ActuatorCommand
    measurement: ActuatorMeasurement
    stage_status: dict[str, str]

    def to_dict(self) -> dict:
        return {
            "control_step": self.control_step,
            "time_s": self.time_s,
            "phase": self.phase,
            "loop_state": self.loop_state,
            "stage_status": self.stage_status,
            "world": self.world.to_dict(),
            "interaction": self.interaction.to_dict(),
            "material": self.material.to_dict(),
            "target": self.target.to_dict(),
            "command": self.command.to_dict(),
            "measurement": self.measurement.to_dict(),
        }


class LoopStage(Protocol):
    """Common identity every stage exposes to the driver."""

    layer: int
    stage_id: str

    def reset(self) -> None: ...

    def status(self) -> str: ...


class FiveLayerLoop:
    """Fixed-order driver with an explicit lifecycle state machine.

    States: ``initialized`` -> ``running`` -> ``completed`` or ``aborted``.
    Only layer 1 may end the run, and it must say why. The driver never
    invents a stage result: if a stage cannot produce one it raises, because a
    silently skipped layer would make the recorded loop unreadable.
    """

    STATES = ("initialized", "running", "completed", "aborted")

    def __init__(self, world, interaction, material, target, actuator, *, max_steps: int):
        self.world = world
        self.interaction = interaction
        self.material = material
        self.target = target
        self.actuator = actuator
        self.max_steps = int(max_steps)
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least one")
        self.state = "initialized"
        self.abort_reason = ""
        self.feedback = LoopFeedback()
        self.frames: list[LoopFrame] = []

    @property
    def stages(self) -> tuple:
        return (self.world, self.interaction, self.material, self.target, self.actuator)

    def stage_table(self) -> list[dict]:
        return [
            {"layer": stage.layer, "stage_id": stage.stage_id, "status": stage.status()}
            for stage in self.stages
        ]

    def reset(self) -> None:
        for stage in self.stages:
            stage.reset()
        self.state = "initialized"
        self.abort_reason = ""
        self.feedback = LoopFeedback()
        self.frames = []

    def step(self) -> LoopFrame:
        if self.state in ("completed", "aborted"):
            raise RuntimeError(f"Loop already {self.state}; reset before stepping again")
        self.state = "running"
        world_frame = self.world.step(self.feedback)
        interaction = self.interaction.step(world_frame)
        material = self.material.step(world_frame, interaction)
        target = self.target.step(material)
        command, measurement = self.actuator.step(target)
        frame = LoopFrame(
            control_step=world_frame.control_step,
            time_s=world_frame.time_s,
            phase=world_frame.phase,
            loop_state=self.state,
            world=world_frame,
            interaction=interaction,
            material=material,
            target=target,
            command=command,
            measurement=measurement,
            stage_status={f"layer{stage.layer}": stage.status() for stage in self.stages},
        )
        self.feedback = LoopFeedback(
            control_step=world_frame.control_step,
            contact_present=interaction.contact_present,
            normal_force_n=interaction.normal_force_n or 0.0,
            tangent_speed_mps=self.interaction.last_tangent_speed_mps,
            actuator_measured_force_n=measurement.measured_force_n,
        )
        self.frames.append(frame)
        return frame

    def run(self) -> list[LoopFrame]:
        """Drive the loop until layer 1 finishes, aborts, or the budget ends."""
        while True:
            frame = self.step()
            if self.world.done:
                self.state = "aborted" if self.world.abort_reason else "completed"
                self.abort_reason = self.world.abort_reason
                break
            if len(self.frames) >= self.max_steps:
                self.state = "aborted"
                self.abort_reason = (
                    f"timeout: reached the declared simulation limit of "
                    f"{self.max_steps} control steps "
                    f"({frame.time_s:.6f} s) before the controller finished"
                )
                break
        return self.frames
