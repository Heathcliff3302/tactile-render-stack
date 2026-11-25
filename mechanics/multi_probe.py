"""P2.2.2 experiment assembly: several probes, one kernel, one stream per pair.

Each probe/surface pair gets its own complete layer 2 to layer 5 chain and
its own ``InteractionState`` stream. The streams share exactly one thing, the
kernel step that advances every body in the same control interval; nothing a
stream records is read from another stream's pair.

Every stream is returned as a :class:`~mechanics.k1_rigid.K1Run` whose
``kernel_config`` names only that stream's probe. That is deliberate: the K1
writers, the dense-field gate and the replay gate all key on
``kernel_config.probes[0]``, so a per-stream view lets them run unchanged on
each stream — the single-probe code path is reused, not copied.

Like K1, the run is executed twice. Pass 1 realises the command timelines;
pass 2 re-runs from a clean state with each stream's conditions digest
stamped in, and must reproduce pass 1 exactly.

Mixed shapes are supported (sphere and box in one run). Each probe keeps its
own ``retract_distance_m`` and its own segment clock, so they need not reach
the surface together.
"""

from __future__ import annotations

import copy
import dataclasses
import time

from actuators import SingleAxisActuatorConfig, SingleAxisForceActuator
from contact_kernel import RigidContactKernel
from interaction import KernelInteractionStage
from mechanics.k1_rigid import CONDITIONS_SCHEMA, PLACEHOLDER_DIGEST, TRAJECTORY_COLUMNS, TRAJECTORY_KIND, K1Run
from mechanics.material import RigidBaselineMaterial
from std_tactile import NormalForceTargetStage
from tactile_contract import ExperimentSpec
from tactile_contract._common import digest
from tactile_contract.pipeline import LOOP_SCHEMA_VERSION, LoopFeedback, LoopFrame
from virtual_reality import classify_runtime, kernel_config_from_spec, step_budget_from_spec
from virtual_reality.multi_world import MULTI_TRAJECTORY, MultiProbeWorldStage, controllers_from_spec

SCENARIO_ID = "p2_2_2_multi_probe_rigid"
SCENARIO_VERSION = "p2.2.2"
SCENARIO_DESCRIPTION = (
    "P2.2.2 several rotation-locked probes against a fixed cube top face, "
    "one independent segment machine and one InteractionState stream per "
    "probe/surface pair, rigid impulse mode, CPU reference kernel"
)


def solo_spec(spec: ExperimentSpec, probe_id: str) -> ExperimentSpec:
    """The same experiment with only ``probe_id`` present.

    Used by the independence gate: a stream that is not contaminated by its
    neighbour must be identical to the stream that probe produces alone.
    """
    data = copy.deepcopy(spec.to_dict())
    keep = [index for index, probe in enumerate(data["probes"]) if probe["id"] == probe_id]
    if len(keep) != 1:
        raise ValueError(f"{probe_id!r} is not a declared probe")
    index = keep[0]
    data["probes"] = [data["probes"][index]]
    data["streams"] = [data["streams"][index]]
    data["spec_id"] = f"{data['spec_id']}_solo_{index}"
    data["description"] = f"{data['description']} Solo variant: only {probe_id}."
    data["parameter_evidence"]["probes"] = (
        "Derived solo variant for the P2.2.2 independence gate; every other field is unchanged."
    )
    return ExperimentSpec.build(**data)


class _Chain:
    """Layers 2 to 5 for one stream."""

    def __init__(self, *, stream, probe_id, surface_id, run_id, scenario_id, conditions_sha256,
                 kernel_config):
        self.stream_id = stream["id"]
        self.probe_id = probe_id
        self.pair_key = RigidContactKernel.pair_key(probe_id, surface_id)
        self.interaction = KernelInteractionStage(
            run_id=run_id,
            scenario_id=scenario_id,
            conditions_sha256=conditions_sha256,
            backend_id=kernel_config.backend_id,
            pair_key=self.pair_key,
            kinematics_source=kernel_config.sampling.kinematics_source,
        )
        self.material = RigidBaselineMaterial()
        self.target = NormalForceTargetStage(target_id=f"{scenario_id}/{self.stream_id}")
        self.actuator = SingleAxisForceActuator(SingleAxisActuatorConfig(
            actuator_id=f"{probe_id}/normal_force",
            control_dt_s=kernel_config.timing.control_dt_s,
        ))
        self.frames: list[LoopFrame] = []
        self.feedback = LoopFeedback()

    def stages(self, world):
        return (world, self.interaction, self.material, self.target, self.actuator)


class MultiProbeExperiment:
    """Build and run the multi-stream five-layer loop from a validated spec."""

    def __init__(self, spec: ExperimentSpec, *, run_id: str = "p2-2-2-multi-probe-cpu",
                 scenario_id: str = SCENARIO_ID, scenario_version: str = SCENARIO_VERSION,
                 scenario_description: str = SCENARIO_DESCRIPTION,
                 substeps_per_control: int | None = None):
        data = spec.to_dict()
        if data["kind"] != "cpu_reference":
            raise ValueError("This experiment requires an explicit cpu_reference ExperimentSpec")
        if data["controller"]["mode"] != MULTI_TRAJECTORY:
            raise ValueError(
                f"MultiProbeExperiment needs controller.mode == {MULTI_TRAJECTORY!r}, "
                f"got {data['controller']['mode']!r}"
            )
        self.spec = spec
        self.data = data
        self.run_id = run_id
        self.scenario_id = scenario_id
        self.scenario_version = scenario_version
        self.scenario_description = scenario_description
        self.kernel_config = kernel_config_from_spec(spec, substeps_per_control=substeps_per_control)
        self.surface_id = data["surface"]["id"]
        self.streams = list(data["streams"])
        self.probe_ids = [probe["id"] for probe in data["probes"]]
        self.step_budget = step_budget_from_spec(spec)

    def stream_kernel_config(self, probe_id: str):
        """The kernel config as one stream sees it: its own probe only."""
        probe = next(item for item in self.kernel_config.probes if item.id == probe_id)
        return dataclasses.replace(self.kernel_config, probes=(probe,))

    # -- execution ---------------------------------------------------------

    def _execute(self, digests: dict[str, str]):
        kernel = RigidContactKernel(self.kernel_config)
        world = MultiProbeWorldStage(kernel, controllers_from_spec(self.spec))
        chains = [
            _Chain(stream=stream, probe_id=stream["body0"], surface_id=self.surface_id,
                   run_id=self.run_id, scenario_id=self.scenario_id,
                   conditions_sha256=digests[stream["id"]], kernel_config=self.kernel_config)
            for stream in self.streams
        ]
        started = time.perf_counter()
        state, abort_reason = "running", ""
        while True:
            world_frames = world.step({chain.probe_id: chain.feedback for chain in chains})
            for chain in chains:
                world_frame = world_frames[chain.probe_id]
                interaction = chain.interaction.step(world_frame)
                material = chain.material.step(world_frame, interaction)
                target = chain.target.step(material)
                command, measurement = chain.actuator.step(target)
                chain.frames.append(LoopFrame(
                    control_step=world_frame.control_step,
                    time_s=world_frame.time_s,
                    phase=world_frame.phase,
                    loop_state="running",
                    world=world_frame,
                    interaction=interaction,
                    material=material,
                    target=target,
                    command=command,
                    measurement=measurement,
                    stage_status={f"layer{stage.layer}": stage.status() for stage in chain.stages(world)},
                ))
                chain.feedback = LoopFeedback(
                    control_step=world_frame.control_step,
                    contact_present=interaction.contact_present,
                    normal_force_n=interaction.normal_force_n or 0.0,
                    tangent_speed_mps=chain.interaction.last_tangent_speed_mps,
                    actuator_measured_force_n=measurement.measured_force_n,
                )
            if world.done:
                state = "aborted" if world.abort_reason else "completed"
                abort_reason = world.abort_reason
                break
            if len(chains[0].frames) >= self.step_budget:
                state = "aborted"
                abort_reason = (
                    f"timeout: reached the declared simulation limit of "
                    f"{self.step_budget} control steps before every probe finished"
                )
                break
        wall_clock = time.perf_counter() - started
        final = kernel.finalize("run_end")
        episodes = {}
        for chain in chains:
            items = [
                episode
                for frame in chain.frames
                for episode in frame.world.kernel_frame.pairs[chain.pair_key].ended_episodes
            ]
            items.extend(final[chain.pair_key])
            items.sort(key=lambda episode: (episode.first_step, episode.patch_id))
            episodes[chain.stream_id] = tuple(items)
        return world, chains, episodes, state, abort_reason, wall_clock

    def _conditions(self, world, chain) -> dict:
        """K1-shaped conditions for one stream, plus the bodies it shared a kernel with."""
        data = self.data
        timing = self.kernel_config.timing
        runtime = {name: setting["value"] for name, setting in data["runtime"].items()}
        probe = next(item for item in data["probes"] if item["id"] == chain.probe_id)
        samples = [[0.0, *probe["initial_position_world_m"], *runtime["initial_linear_velocity_mps"]]]
        for record, frame in zip(world.commands(chain.probe_id), chain.frames, strict=True):
            if record["control_step"] != frame.control_step:
                raise RuntimeError("Command stream and loop frames are out of step")
            position = frame.world.body_states[chain.probe_id]["position_world_m"]
            samples.append([record["time_s"], *position, *record["command_velocity_world_mps"]])
        return {
            "schema": CONDITIONS_SCHEMA,
            "spec_id": data["spec_id"],
            "spec_sha256": data["spec_sha256"],
            "stream_id": chain.stream_id,
            "reference_frame": {
                "basis_world": data["coordinates"]["basis_world"],
                "origin_world_m": data["coordinates"]["local_origin_world_m"],
                "handedness": data["coordinates"]["handedness"],
                "up_axis": data["coordinates"]["up_axis"],
                "units": data["coordinates"]["units"],
                "quaternion_order": data["coordinates"]["quaternion_order"],
                "surface_z_world_m": data["coordinates"]["surface_z_world_m"],
                "contact_normal_world": data["coordinates"]["normal_world"],
            },
            "bodies": [
                {
                    "id": probe["id"],
                    "role": "probe",
                    "shape": probe["shape"],
                    "radius_m": probe["radius_m"],
                    "size_m": probe["size_m"],
                    "mass_kg": probe["mass_kg"],
                    "motion": "velocity_driven",
                    "angular_mode": probe["angular_mode"],
                    "initial_position_world_m": probe["initial_position_world_m"],
                    "initial_linear_velocity_mps": runtime["initial_linear_velocity_mps"],
                    "initial_orientation_wxyz": runtime["initial_orientation_wxyz"],
                    "initial_angular_velocity_rps": [0, 0, 0],
                },
                {
                    "id": data["surface"]["id"],
                    "role": "reference_surface",
                    "shape": data["surface"]["shape"],
                    "size_m": data["surface"]["size_m"],
                    "mass_kg": data["surface"]["mass_kg"],
                    "motion": data["surface"]["motion"],
                    "center_world_m": data["surface"]["center_world_m"],
                    "initial_linear_velocity_mps": [0, 0, 0],
                    "initial_angular_velocity_rps": [0, 0, 0],
                },
            ],
            # The other probes in the same kernel. Recorded because they were
            # there, not because this stream reads them: the independence gate
            # is what shows it does not.
            "co_simulated_probes": [
                {"id": item["id"], "shape": item["shape"]}
                for item in data["probes"] if item["id"] != chain.probe_id
            ],
            "environment": {
                "gravity_mps2": runtime["gravity_mps2"],
                "static_friction": runtime["static_friction"],
                "dynamic_friction": runtime["dynamic_friction"],
                "restitution": runtime["restitution"],
                "friction_combine_mode": runtime["friction_combine_mode"],
                "restitution_combine_mode": runtime["restitution_combine_mode"],
                "linear_damping_per_s": runtime["linear_damping_per_s"],
                "angular_damping_per_s": runtime["angular_damping_per_s"],
                "contact_offset_m": runtime["contact_offset_m"],
                "rest_offset_m": runtime["rest_offset_m"],
            },
            "timing": {
                "start_time_s": 0.0,
                "end_time_s": chain.frames[-1].time_s,
                "physics_dt_s": timing.physics_dt_s,
                "control_dt_s": timing.control_dt_s,
                "output_dt_s": timing.output_dt_s,
                "substeps_per_control": timing.substeps_per_control,
                "timestamp": "interval_end",
                "sequence_start": 0,
            },
            "controller": data["controller"],
            "drive": {
                "semantics": self.kernel_config.drive.semantics,
                "gravity_compensation_mps2": self.kernel_config.drive.gravity_compensation_mps2,
                "gravity_compensation_mode": self.kernel_config.drive.gravity_compensation_mode,
                "feedback_delay_steps": data["controller"]["feedback_delay_steps"],
            },
            "discretisation": {
                "solver_type": runtime["solver_type"],
                "position_iterations": runtime["position_iterations"],
                "velocity_iterations": runtime["velocity_iterations"],
                "max_depenetration_velocity_mps": runtime["max_depenetration_velocity_mps"],
                "ccd_mode": runtime["ccd_mode"],
                "sleep_enabled": runtime["sleep_enabled"],
                "grid": data["grid"],
                "sampling": data["sampling"],
                "contact_area_methods": ["point_occupancy", "swept_path", "geometric_footprint"],
                "model_contact_area": None,
            },
            "trajectory": {
                "kind": TRAJECTORY_KIND,
                "columns": list(TRAJECTORY_COLUMNS),
                "body": probe["id"],
                "samples": samples,
            },
            "unresolved_settings": [
                name for name, setting in data["runtime"].items() if setting["status"] == "unresolved"
            ],
        }

    def run(self) -> dict[str, K1Run]:
        """Run both passes. Returns one per-stream ``K1Run`` view per declared stream."""
        placeholder = {stream["id"]: PLACEHOLDER_DIGEST for stream in self.streams}
        world, chains, _, _, _, _ = self._execute(placeholder)
        conditions = {chain.stream_id: self._conditions(world, chain) for chain in chains}
        digests = {stream_id: digest(item) for stream_id, item in conditions.items()}
        realised = {chain.probe_id: [r["command_velocity_world_mps"] for r in world.commands(chain.probe_id)]
                    for chain in chains}
        realised_frames = {chain.stream_id: len(chain.frames) for chain in chains}

        world, chains, episodes, state, abort_reason, wall_clock = self._execute(digests)
        for chain in chains:
            emitted = [r["command_velocity_world_mps"] for r in world.commands(chain.probe_id)]
            if emitted != realised[chain.probe_id]:
                raise RuntimeError(f"{chain.stream_id}: emission pass diverged from the realisation pass")
            if len(chain.frames) != realised_frames[chain.stream_id]:
                raise RuntimeError(f"{chain.stream_id}: emission pass produced a different frame count")

        runs = {}
        for chain in chains:
            stage_table = tuple(
                {"layer": stage.layer, "stage_id": stage.stage_id, "status": stage.status()}
                for stage in chain.stages(world)
            )
            runs[chain.stream_id] = K1Run(
                spec=self.spec,
                run_id=self.run_id,
                scenario_id=self.scenario_id,
                scenario_version=self.scenario_version,
                scenario_description=self.scenario_description,
                controller_mode="trajectory",
                kernel_config=self.stream_kernel_config(chain.probe_id),
                conditions=conditions[chain.stream_id],
                conditions_sha256=digests[chain.stream_id],
                frames=tuple(frame.interaction for frame in chain.frames),
                loop_frames=tuple(chain.frames),
                episodes=episodes[chain.stream_id],
                commands=world.commands(chain.probe_id),
                stream_id=chain.stream_id,
                loop_state=state,
                abort_reason=abort_reason,
                stage_table=stage_table,
                substeps_per_control=self.kernel_config.timing.substeps_per_control,
                wall_clock_s=wall_clock,
            )
        return runs

    def effective_runtime(self) -> dict:
        config = self.kernel_config
        return {
            "schema": "p2-multi-effective-runtime/v1",
            "loop_schema_version": LOOP_SCHEMA_VERSION,
            "backend_id": config.backend_id,
            "backend_version": config.backend_version,
            "spec_id": self.data["spec_id"],
            "spec_sha256": self.data["spec_sha256"],
            "controller_mode": MULTI_TRAJECTORY,
            "controller_structure": "one TrajectoryStateMachine per probe, each on its own delayed feedback",
            "probes": [
                {"id": probe.id, "shape": probe.shape, "mass_kg": probe.mass_kg,
                 "radius_m": probe.radius_m, "size_m": list(probe.size_m) if probe.size_m else None}
                for probe in config.probes
            ],
            "streams": [stream["id"] for stream in self.streams],
            "timing": {
                "physics_dt_s": config.timing.physics_dt_s,
                "control_dt_s": config.timing.control_dt_s,
                "output_dt_s": config.timing.output_dt_s,
                "substeps_per_control": config.timing.substeps_per_control,
            },
            "field_policy": classify_runtime(self.spec),
            "realised_drive_authority": None,
            "isaac_sim_parity": (
                "not compared: multi-probe and box probes are outside the "
                "2026-11-25 Isaac comparison, which covered the single-sphere K1/K2 only; "
                "behaviour and conservation gates only, no claim of numeric equality"
            ),
        }
