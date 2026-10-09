"""K1 experiment assembly: single sphere probe, fixed top face, five layers.

The run is executed twice on purpose.

Pass 1 *realises* the run. Trajectory-mode commands depend on when contact
actually happens, and a force-mode command is produced by feedback, so the
command timeline cannot honestly be declared in advance. Pass 1 records what
the controller really issued.

Pass 2 *emits* it. The realised timeline becomes the manifest ``conditions``,
whose digest is stamped into every ``InteractionState``. Pass 2 re-runs the
loop from a clean state and its realised timeline must match pass 1 exactly,
which is also the determinism evidence the K1 replay gate needs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from actuators import SingleAxisActuatorConfig, SingleAxisForceActuator
from contact_kernel import ContactEpisode, KernelConfig, RigidContactKernel
from interaction import KernelInteractionStage
from mechanics.material import RigidBaselineMaterial
from std_tactile import NormalForceTargetStage
from tactile_contract import ExperimentSpec, InteractionState
from tactile_contract._common import digest
from tactile_contract.pipeline import LOOP_SCHEMA_VERSION, FiveLayerLoop, LoopFrame
from virtual_reality import (
    KernelWorldStage,
    assess_proportional_stability,
    assess_regulator_authority,
    classify_runtime,
    controller_from_spec,
    kernel_config_from_spec,
    measure_force_ceiling,
    step_budget_from_spec,
)

#: Structure version of the ``conditions`` block, not a milestone label. K2
#: emits the same structure, so the identifier is unchanged; bumping it to
#: read "k2" would alter every K1 conditions digest to rename a string.
CONDITIONS_SCHEMA = "k1-conditions/v1"
SCENARIO_ID = "k1_single_probe_rigid"
SCENARIO_VERSION = "k1.0"
#: Verbatim, because the K1 manifest hash is frozen and the committed run
#: index records it. A reworded description is a changed manifest.
SCENARIO_DESCRIPTION = (
    "K1 single sphere probe against a fixed cube top face, rigid "
    "impulse mode, CPU reference kernel"
)
K2_SCENARIO_ID = "k2_force_regulation"
K2_SCENARIO_VERSION = "k2.0"
K2_SCENARIO_DESCRIPTION = (
    "K2 single sphere probe against a fixed cube top face, rigid impulse "
    "mode, CPU reference kernel, PI normal-force regulation at the declared "
    "target"
)
PLACEHOLDER_DIGEST = "0" * 64
TRAJECTORY_COLUMNS = (
    "time_s",
    "probe_x_m",
    "probe_y_m",
    "probe_z_m",
    "command_vx_mps",
    "command_vy_mps",
    "command_vz_mps",
)
TRAJECTORY_KIND = "realised_command_velocity_and_probe_pose"


@dataclass(frozen=True)
class K1Run:
    """Everything one K1 run produced, in memory."""

    spec: ExperimentSpec
    run_id: str
    scenario_id: str
    scenario_version: str
    scenario_description: str
    controller_mode: str
    kernel_config: KernelConfig
    conditions: dict
    conditions_sha256: str
    frames: tuple[InteractionState, ...]
    loop_frames: tuple[LoopFrame, ...]
    episodes: tuple[ContactEpisode, ...]
    commands: tuple[dict, ...]
    stream_id: str
    loop_state: str
    abort_reason: str
    stage_table: tuple[dict, ...]
    substeps_per_control: int
    wall_clock_s: float

    @property
    def phase_sequence(self) -> tuple[str, ...]:
        return tuple(frame.metadata["motion_phase"] for frame in self.frames)

    @property
    def contact_sequence(self) -> tuple[bool, ...]:
        return tuple(frame.contact_present for frame in self.frames)

    @property
    def start_time_s(self) -> float:
        return self.conditions["timing"]["start_time_s"]

    @property
    def end_time_s(self) -> float:
        return self.conditions["timing"]["end_time_s"]


#: Controller modes with a segment machine and an acceptance story.
SUPPORTED_MODES = ("trajectory", "force")


class RigidProbeExperiment:
    """Build and run the five-layer loop from a validated ``ExperimentSpec``.

    One spherical probe against the fixed top face, in whichever controller
    mode the specification declares. The assembly is mode independent: layer 1
    selects its segment machine from the spec, and layers 2 to 5 read the same
    kernel output either way. Only the controller differs, which is the point
    — a force run that needed its own pipeline would not be comparable with a
    trajectory run.
    """

    def __init__(self, spec: ExperimentSpec, *, run_id: str = "k1-single-probe-cpu",
                 scenario_id: str = SCENARIO_ID, scenario_version: str = SCENARIO_VERSION,
                 scenario_description: str = SCENARIO_DESCRIPTION,
                 substeps_per_control: int | None = None):
        data = spec.to_dict()
        if data["kind"] != "cpu_reference":
            raise ValueError("This experiment requires an explicit cpu_reference ExperimentSpec")
        if len(data["probes"]) != 1 or data["probes"][0]["shape"] != "sphere":
            raise ValueError("This experiment requires exactly one spherical probe")
        mode = data["controller"]["mode"]
        if mode not in SUPPORTED_MODES:
            raise ValueError(
                f"Controller mode {mode!r} is not implemented; implemented modes "
                f"are {list(SUPPORTED_MODES)} and 'multi_trajectory' arrives at K3"
            )
        self.spec = spec
        self.data = data
        self.mode = mode
        self.run_id = run_id
        self.scenario_id = scenario_id
        self.scenario_version = scenario_version
        self.scenario_description = scenario_description
        self.kernel_config = kernel_config_from_spec(spec, substeps_per_control=substeps_per_control)
        self.probe_id = data["probes"][0]["id"]
        self.surface_id = data["surface"]["id"]
        self.stream_id = data["streams"][0]["id"]
        self.pair_key = RigidContactKernel.pair_key(self.probe_id, self.surface_id)
        self.step_budget = step_budget_from_spec(spec)

    @classmethod
    def from_json(cls, path: str | Path, **kwargs) -> "RigidProbeExperiment":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(ExperimentSpec.from_dict(data), **kwargs)

    def _build_loop(self, conditions_sha256: str) -> FiveLayerLoop:
        kernel = RigidContactKernel(self.kernel_config)
        world = KernelWorldStage(kernel, controller_from_spec(self.spec), probe_id=self.probe_id)
        contact = KernelInteractionStage(
            run_id=self.run_id,
            scenario_id=self.scenario_id,
            conditions_sha256=conditions_sha256,
            backend_id=self.kernel_config.backend_id,
            pair_key=self.pair_key,
            kinematics_source=self.kernel_config.sampling.kinematics_source,
        )
        actuator = SingleAxisForceActuator(SingleAxisActuatorConfig(
            actuator_id=f"{self.probe_id}/normal_force",
            control_dt_s=self.kernel_config.timing.control_dt_s,
        ))
        return FiveLayerLoop(
            world,
            contact,
            RigidBaselineMaterial(),
            NormalForceTargetStage(target_id=f"{self.scenario_id}/{self.stream_id}"),
            actuator,
            max_steps=self.step_budget,
        )

    def _execute(self, conditions_sha256: str):
        loop = self._build_loop(conditions_sha256)
        started = time.perf_counter()
        loop.run()
        wall_clock = time.perf_counter() - started
        kernel = loop.world.kernel
        episodes = list(
            episode
            for frame in loop.frames
            for episode in frame.world.kernel_frame.pairs[self.pair_key].ended_episodes
        )
        episodes.extend(kernel.finalize("run_end")[self.pair_key])
        episodes.sort(key=lambda episode: (episode.first_step, episode.patch_id))
        return loop, tuple(episodes), wall_clock

    def _conditions(self, loop, frames) -> dict:
        data = self.data
        timing = self.kernel_config.timing
        runtime = {name: setting["value"] for name, setting in data["runtime"].items()}
        probe = data["probes"][0]
        samples = [[
            0.0,
            *probe["initial_position_world_m"],
            *runtime["initial_linear_velocity_mps"],
        ]]
        for record, frame in zip(loop.world.commands, loop.frames, strict=True):
            if record["control_step"] != frame.control_step:
                raise RuntimeError("Command stream and loop frames are out of step")
            position = frame.world.body_states[self.probe_id]["position_world_m"]
            samples.append([record["time_s"], *position, *record["command_velocity_world_mps"]])
        return {
            "schema": CONDITIONS_SCHEMA,
            "spec_id": data["spec_id"],
            "spec_sha256": data["spec_sha256"],
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
                "end_time_s": frames[-1].time_s,
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

    def run(self) -> K1Run:
        realisation, _, _ = self._execute(PLACEHOLDER_DIGEST)
        frames = [frame.interaction for frame in realisation.frames]
        conditions = self._conditions(realisation, frames)
        conditions_sha256 = digest(conditions)

        emission, episodes, wall_clock = self._execute(conditions_sha256)
        if [record["command_velocity_world_mps"] for record in emission.world.commands] != [
            record["command_velocity_world_mps"] for record in realisation.world.commands
        ]:
            raise RuntimeError("K1 emission pass diverged from the realisation pass")
        emitted = tuple(frame.interaction for frame in emission.frames)
        if len(emitted) != len(frames):
            raise RuntimeError("K1 emission pass produced a different frame count")
        return K1Run(
            spec=self.spec,
            run_id=self.run_id,
            scenario_id=self.scenario_id,
            scenario_version=self.scenario_version,
            scenario_description=self.scenario_description,
            controller_mode=self.mode,
            kernel_config=self.kernel_config,
            conditions=conditions,
            conditions_sha256=conditions_sha256,
            frames=emitted,
            loop_frames=tuple(emission.frames),
            episodes=episodes,
            commands=tuple(emission.world.commands),
            stream_id=self.stream_id,
            loop_state=emission.state,
            abort_reason=emission.abort_reason,
            stage_table=tuple(emission.stage_table()),
            substeps_per_control=self.kernel_config.timing.substeps_per_control,
            wall_clock_s=wall_clock,
        )

    def effective_runtime(self) -> dict:
        """Everything a reader needs to reproduce this run, with provenance."""
        config = self.kernel_config
        return {
            "schema": "k1-effective-runtime/v1",
            "loop_schema_version": LOOP_SCHEMA_VERSION,
            "backend_id": config.backend_id,
            "backend_version": config.backend_version,
            "spec_id": self.data["spec_id"],
            "spec_sha256": self.data["spec_sha256"],
            "solver": {
                "type": config.solver.solver_type,
                "gravity_mps2": list(config.solver.gravity_mps2),
                "restitution": config.solver.restitution,
                "static_friction": config.solver.static_friction,
                "dynamic_friction": config.solver.dynamic_friction,
                "linear_damping_per_s": config.solver.linear_damping_per_s,
                "contact_offset_m": config.solver.contact_offset_m,
                "rest_offset_m": config.solver.rest_offset_m,
                "max_depenetration_velocity_mps": config.solver.max_depenetration_velocity_mps,
                "normal_impulse_rule": "J = m * max(0, v_target_n - v_free_n), v_target_n = -gap/h clamped at the declared depenetration limit",
                "penetration_handling": "geometric projection recorded separately from contact impulse",
            },
            "timing": {
                "physics_dt_s": config.timing.physics_dt_s,
                "control_dt_s": config.timing.control_dt_s,
                "output_dt_s": config.timing.output_dt_s,
                "substeps_per_control": config.timing.substeps_per_control,
            },
            "drive": {
                "semantics": config.drive.semantics,
                "gravity_compensation_mps2": config.drive.gravity_compensation_mps2,
                "gravity_compensation_mode": config.drive.gravity_compensation_mode,
                "relocation_note": (
                    "Step 6 folded gravity compensation into the velocity command; "
                    "K1 applies it per substep in the drive so the realised "
                    "trajectory and impulse stay invariant under substep refinement"
                ),
                "force_scale_note": (
                    "hold force equals mass * contact_maintain_speed / output_dt: it "
                    "is set by the once-per-control-step velocity overwrite, not by weight"
                ),
            },
            "grid": {
                "rows": config.grid.rows,
                "cols": config.grid.cols,
                "cell_size_m": config.grid.cell_size_m,
                "cell_area_m2": config.grid.cell_area_m2,
                "origin_local_m": list(config.grid.origin_local_m),
                "edge_policy": config.grid.edge_policy,
                "connectivity": config.grid.connectivity,
            },
            "sampling": {
                "force_threshold_n": config.sampling.force_threshold_n,
                "point_match_distance_m": config.sampling.point_match_distance_m,
                "max_gap_frames": config.sampling.max_gap_frames,
                "slide_threshold_mps": config.sampling.slide_threshold_mps,
                "direction_threshold_m": config.sampling.direction_threshold_m,
                "filter_alpha": config.sampling.filter_alpha,
                "kinematics_source": config.sampling.kinematics_source,
                "force_definition": self.data["sampling"]["force_definition"],
            },
            "unresolved_settings": [
                name for name, setting in self.data["runtime"].items() if setting["status"] == "unresolved"
            ],
            "field_policy": classify_runtime(self.spec),
            "controller_mode": self.mode,
            # None in trajectory mode. In force mode this records both
            # ceilings on the regulator and which one binds, so a reader can
            # see why the declared gains are what they are without re-deriving
            # them. See virtual_reality/drive_limits.py.
            "regulator_authority": assess_regulator_authority(self.spec),
            # Convergence, not reach. The admissible proportional gain scales
            # with output_dt, so this records the margin at the declared rate.
            "proportional_stability": assess_proportional_stability(self.spec),
            # The same question asked of the kernel instead of the ideal drive
            # law. The two agree only when every term the law omits is zero,
            # which is true for this CPU reference and is exactly why the
            # agreement is worth recording rather than assuming.
            "realised_drive_authority": measure_force_ceiling(
                self.spec, kernel_config=self.kernel_config
            ),
            "isaac_sim_parity": (
                "behaviour and conservation gates only; no claim of numeric "
                "equality with unresolved PhysX settings"
            ),
        }


#: The original name, kept so K1 call sites and docs stay valid.
K1Experiment = RigidProbeExperiment
