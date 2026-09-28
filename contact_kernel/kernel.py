"""The decoupled contact kernel.

``RigidContactKernel`` is the only owner of dynamics for the bodies it holds.
It depends on nothing but the standard library, so it can be driven from a
test, from the five-layer loop, or from a future front end without importing a
simulator or the contract package.

One control step produces one :class:`KernelFrame` per contact pair, carrying
the contact point, the impulse and force, the three area layers, the
cell/patch/episode chain and the conservation residuals that the acceptance
gates read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import SurfaceGrid, TopFaceGeometry, dot
from .spec import (
    KERNEL_SCHEMA_VERSION,
    KINEMATICS_FINITE_DIFFERENCE,
    KINEMATICS_METHOD,
    KernelConfig,
    Vec3,
)
from .tracking import (
    GEOMETRIC_FOOTPRINT,
    POINT_OCCUPANCY,
    SWEPT_PATH,
    ContactCell,
    ContactEpisode,
    ContactPatch,
    ContactPatchTracker,
    ContactPoint,
    ContactPointTracker,
    aggregate_cells,
)


@dataclass(frozen=True)
class AreaLayers:
    """The three area definitions the K1 design requires in every frame.

    ``point_occupancy_m2`` reproduces Step 6 and changes with output
    resolution. ``swept_path_m2`` closes rasterisation gaps along the contact
    trajectory and is not an instantaneous area. ``geometric_footprint_m2`` is
    the undeformed shape intersection, which is zero for a non-penetrating
    rigid contact. ``model_footprint_m2`` stays ``None`` until the K4 material
    models supply an elastic contact area.
    """

    point_occupancy_m2: float
    swept_path_m2: float
    geometric_footprint_m2: float
    footprint_radius_m: float
    swept_cell_count: int
    model_footprint_m2: float | None = None
    note: str = (
        "point_occupancy is a resolution-dependent proxy; a rigid "
        "non-penetrating contact has a zero geometric footprint"
    )

    def to_dict(self) -> dict:
        return {
            "point_occupancy_m2": self.point_occupancy_m2,
            "swept_path_m2": self.swept_path_m2,
            "geometric_footprint_m2": self.geometric_footprint_m2,
            "footprint_radius_m": self.footprint_radius_m,
            "swept_cell_count": self.swept_cell_count,
            "model_footprint_m2": self.model_footprint_m2,
            "methods": [POINT_OCCUPANCY, SWEPT_PATH, GEOMETRIC_FOOTPRINT],
            "note": self.note,
        }


@dataclass(frozen=True)
class Conservation:
    """Force sums along point -> cell -> patch, with their residuals."""

    point_force_n: float
    cell_force_n: float
    patch_force_n: float
    point_impulse_ns: float
    cell_impulse_ns: float
    patch_impulse_ns: float
    solver_impulse_ns: float

    @property
    def point_to_cell_error_n(self) -> float:
        return abs(self.point_force_n - self.cell_force_n)

    @property
    def cell_to_patch_error_n(self) -> float:
        return abs(self.cell_force_n - self.patch_force_n)

    @property
    def solver_to_point_error_ns(self) -> float:
        return abs(self.solver_impulse_ns - self.point_impulse_ns)

    @property
    def max_force_error_n(self) -> float:
        return max(self.point_to_cell_error_n, self.cell_to_patch_error_n)

    def to_dict(self) -> dict:
        return {
            "point_force_n": self.point_force_n,
            "cell_force_n": self.cell_force_n,
            "patch_force_n": self.patch_force_n,
            "point_impulse_ns": self.point_impulse_ns,
            "cell_impulse_ns": self.cell_impulse_ns,
            "patch_impulse_ns": self.patch_impulse_ns,
            "solver_impulse_ns": self.solver_impulse_ns,
            "point_to_cell_error_n": self.point_to_cell_error_n,
            "cell_to_patch_error_n": self.cell_to_patch_error_n,
            "solver_to_point_error_ns": self.solver_to_point_error_ns,
        }


@dataclass(frozen=True)
class PairFrame:
    """One contact pair at one output step, contact or not."""

    body0: str
    body1: str
    phase: str
    contact_present: bool
    solver_contact: bool
    filtered_reason: str
    contact_point_world_m: Vec3
    contact_point_local_m: Vec3
    normal_world: Vec3 | None
    tangent_basis_world: tuple[Vec3, Vec3]
    gap_m: float
    penetration_m: float
    position_correction_m: float
    normal_impulse_ns: float
    normal_force_n: float
    tangent_force_world_n: Vec3
    total_force_world_n: Vec3
    normal_relative_velocity_mps: float
    tangent_velocity_world_mps: Vec3
    tangent_speed_mps: float
    kinematics_source: str
    kinematics_method: str
    kinematics_available: bool
    rigid_body_tangent_velocity_mps: Vec3
    probe_position_world_m: Vec3
    probe_velocity_world_mps: Vec3
    command_velocity_world_mps: Vec3
    points: tuple[ContactPoint, ...]
    cells: dict[tuple[int, int], ContactCell]
    patches: tuple[ContactPatch, ...]
    started_patch_ids: tuple[int, ...]
    ended_episodes: tuple[ContactEpisode, ...]
    area_layers: AreaLayers
    conservation: Conservation
    solution: object
    inside_face: bool


@dataclass(frozen=True)
class KernelFrame:
    """All contact pairs at one output step."""

    schema_version: str
    backend_id: str
    backend_version: str
    control_step: int
    time_s: float
    dt_s: float
    phase: str
    pairs: dict[str, PairFrame]
    body_states: dict[str, dict]
    substeps_per_control: int
    physics_dt_s: float
    wall_clock_s: float = 0.0
    pair_order: tuple[str, ...] = field(default_factory=tuple)


class RigidContactKernel:
    """Rigid impulse mode of the contact kernel: sphere probes, fixed top face."""

    def __init__(self, config: KernelConfig):
        from .solver import RigidImpulseSolver

        self.config = config
        self.solver = RigidImpulseSolver(config)
        self.geometry = TopFaceGeometry(config.surface)
        self.grid = SurfaceGrid(config.grid, config.surface)
        self._probes = {probe.id: probe for probe in config.probes}
        self._point_trackers: dict[str, ContactPointTracker] = {}
        self._patch_trackers: dict[str, ContactPatchTracker] = {}
        self._last_contact_local: dict[str, Vec3 | None] = {}
        self.reset()

    @staticmethod
    def pair_key(body0: str, body1: str) -> str:
        return f"{body0}|{body1}"

    def reset(self) -> None:
        self.solver.reset()
        sampling = self.config.sampling
        self._point_trackers = {}
        self._patch_trackers = {}
        self._last_contact_local = {}
        for probe in self.config.probes:
            key = self.pair_key(probe.id, self.config.surface.id)
            self._point_trackers[key] = ContactPointTracker(
                match_distance_m=sampling.point_match_distance_m,
                max_gap_frames=sampling.max_gap_frames,
                filter_alpha=sampling.filter_alpha,
                direction_threshold_m=sampling.direction_threshold_m,
            )
            self._patch_trackers[key] = ContactPatchTracker(
                cell_size_m=self.config.grid.cell_size_m,
                cell_area_m2=self.config.grid.cell_area_m2,
                max_gap_frames=sampling.max_gap_frames,
                slide_threshold_mps=sampling.slide_threshold_mps,
            )
            self._last_contact_local[key] = None

    @property
    def time_s(self) -> float:
        return self.solver.time_s

    @property
    def control_step(self) -> int:
        return self.solver.control_step

    def body_states(self) -> dict[str, dict]:
        states = {body_id: state.snapshot() for body_id, state in self.solver.states.items()}
        states[self.config.surface.id] = {
            "id": self.config.surface.id,
            "position_world_m": list(self.config.surface.center_world_m),
            "linear_velocity_mps": [0.0, 0.0, 0.0],
            "angular_velocity_rps": [0.0, 0.0, 0.0],
        }
        return states

    def step(self, commands: dict[str, Vec3], *, phase: str) -> KernelFrame:
        solutions = self.solver.advance_control_step(commands)
        time_s = self.solver.time_s
        step_index = self.solver.control_step
        pairs: dict[str, PairFrame] = {}
        order = []
        for solution in solutions:
            key = self.pair_key(solution.probe_id, solution.surface_id)
            order.append(key)
            pairs[key] = self._build_pair(key, solution, phase, time_s, step_index)
        return KernelFrame(
            schema_version=KERNEL_SCHEMA_VERSION,
            backend_id=self.config.backend_id,
            backend_version=self.config.backend_version,
            control_step=step_index,
            time_s=time_s,
            dt_s=self.config.timing.output_dt_s,
            phase=phase,
            pairs=pairs,
            body_states=self.body_states(),
            substeps_per_control=self.config.timing.substeps_per_control,
            physics_dt_s=self.config.timing.physics_dt_s,
            pair_order=tuple(order),
        )

    def finalize(self, reason: str = "run_end") -> dict[str, tuple[ContactEpisode, ...]]:
        return {
            key: tuple(tracker.finalize(reason))
            for key, tracker in self._patch_trackers.items()
        }

    @staticmethod
    def _apply_finite_difference(points, tangent_basis) -> bool:
        """Promote the differenced contact-point motion to the authoritative values.

        The rigid-body twist stays on the point as
        ``rigid_body_tangent_speed_mps``. Returns whether every point had a
        predecessor to difference against; the first frame of a contact has
        none, and that is reported rather than filled with zeros.
        """
        available = True
        for point in points:
            if not point.finite_difference_available:
                available = False
                continue
            tangent = point.finite_difference_tangent_world_mps
            point.normal_relative_velocity_mps = point.finite_difference_normal_velocity_mps
            point.tangent_velocity_world_mps = tangent
            point.tangent_speed_mps = point.finite_difference_tangent_speed_mps
            point.tangent_u_speed_mps = dot(tangent, tangent_basis[0])
            point.tangent_v_speed_mps = dot(tangent, tangent_basis[1])
        return available

    def _build_pair(self, key: str, solution, phase: str, time_s: float, step_index: int) -> PairFrame:
        probe = self._probes[solution.probe_id]
        normal = self.geometry.normal_world
        tangent_basis = (self.geometry.tangent_u_world, self.geometry.tangent_v_world)
        point_world = solution.contact_point_world_m
        point_local = self.geometry.to_local(point_world)
        velocity = solution.linear_velocity_mps
        # Rigid-body twist. With rotation locked this is the probe's own linear
        # velocity; it is computed on every frame because it is either the
        # authoritative source or the cross-check for the other one.
        rigid_normal_speed = dot(velocity, normal)
        rigid_tangent_velocity = tuple(v - rigid_normal_speed * n for v, n in zip(velocity, normal))
        rigid_tangent_speed = math.sqrt(sum(value * value for value in rigid_tangent_velocity))
        cell_index = self.grid.cell_index(point_local)

        threshold = self.config.sampling.force_threshold_n
        # A probe can be constrained early in the interval and slide off the
        # finite face before it ends. That impulse is real but its contact
        # point is no longer on the face, so the frame says so instead of
        # reporting a contact the geometry no longer supports.
        left_face = solution.contact_substeps > 0 and not solution.inside_face
        solver_contact = solution.contact and solution.inside_face
        above_threshold = solution.normal_force_n >= threshold
        reported = solver_contact and above_threshold and cell_index is not None
        reason = ""
        if left_face:
            reason = "support_point_left_top_face_during_interval"
        elif solver_contact and not above_threshold:
            reason = "normal_force_below_sampling_threshold"
        elif solver_contact and cell_index is None:
            reason = "contact_point_outside_output_grid"
        elif not solution.inside_face and solution.rejected_reason:
            reason = solution.rejected_reason

        points: list[ContactPoint] = []
        if reported:
            curvature_1, curvature_2, curvature_valid = self.geometry.curvature(point_local)
            point = ContactPoint(
                time_s=time_s,
                control_step=step_index,
                phase=phase,
                body0=solution.probe_id,
                body1=solution.surface_id,
                point_world_m=point_world,
                point_local_m=point_local,
                normal_world=normal,
                normal_local=normal,
                tangent_u_world=tangent_basis[0],
                tangent_v_world=tangent_basis[1],
                gap_m=solution.gap_m,
                penetration_m=solution.penetration_m,
                normal_impulse_ns=solution.normal_impulse_ns,
                normal_force_n=solution.normal_force_n,
                tangent_force_world_n=(0.0, 0.0, 0.0),
                grid_row=cell_index[0],
                grid_col=cell_index[1],
                curvature_1_per_m=curvature_1,
                curvature_2_per_m=curvature_2,
                curvature_valid=curvature_valid,
                normal_relative_velocity_mps=rigid_normal_speed,
                tangent_velocity_world_mps=rigid_tangent_velocity,
                tangent_speed_mps=rigid_tangent_speed,
                tangent_u_speed_mps=dot(rigid_tangent_velocity, tangent_basis[0]),
                tangent_v_speed_mps=dot(rigid_tangent_velocity, tangent_basis[1]),
                kinematics_source=KINEMATICS_METHOD[self.config.sampling.kinematics_source],
                rigid_body_tangent_speed_mps=rigid_tangent_speed,
            )
            points.append(point)

        # The tracker assigns IDs and derives the finite-difference values, so
        # source selection happens after it and before any aggregation. Cells
        # and patches then see one consistent set of velocities.
        self._point_trackers[key].update(points, time_s, step_index)
        kinematics_available = True
        if self.config.sampling.kinematics_source == KINEMATICS_FINITE_DIFFERENCE:
            kinematics_available = bool(points) and self._apply_finite_difference(points, tangent_basis)
        normal_speed = points[0].normal_relative_velocity_mps if points else rigid_normal_speed
        tangent_velocity = points[0].tangent_velocity_world_mps if points else rigid_tangent_velocity
        tangent_speed = points[0].tangent_speed_mps if points else rigid_tangent_speed
        if not kinematics_available:
            # Differencing a contact point needs a contact point in two frames.
            # Without one there is no value, and a zero would be a fabrication.
            normal_speed, tangent_velocity, tangent_speed = 0.0, (0.0, 0.0, 0.0), 0.0
        cells = aggregate_cells(points, self.grid, time_s, step_index, phase)
        patches, started, ended = self._patch_trackers[key].update(
            points, cells, time_s, step_index, phase
        )

        previous_local = self._last_contact_local[key]
        swept = self.grid.swept_cells(previous_local, point_local) if points else ()
        self._last_contact_local[key] = point_local if points else None
        footprint_radius = self.geometry.footprint_radius_m(solution.penetration_m, probe.radius_m)
        area_layers = AreaLayers(
            point_occupancy_m2=len(cells) * self.config.grid.cell_area_m2,
            swept_path_m2=len(swept) * self.config.grid.cell_area_m2,
            geometric_footprint_m2=math.pi * footprint_radius**2,
            footprint_radius_m=footprint_radius,
            swept_cell_count=len(swept),
        )

        point_force = sum(item.normal_force_n for item in points)
        conservation = Conservation(
            point_force_n=point_force,
            cell_force_n=sum(cell.normal_force_n for cell in cells.values()),
            patch_force_n=sum(patch.normal_force_n for patch in patches),
            point_impulse_ns=sum(item.normal_impulse_ns for item in points),
            cell_impulse_ns=sum(cell.normal_impulse_ns for cell in cells.values()),
            patch_impulse_ns=sum(patch.normal_impulse_ns for patch in patches),
            # The raw solver impulse is kept even when sampling drops the point,
            # so a threshold-filtered frame is visible instead of silently zero.
            solver_impulse_ns=solution.normal_impulse_ns,
        )
        total_force = (0.0, 0.0, point_force) if reported else (0.0, 0.0, 0.0)
        return PairFrame(
            body0=solution.probe_id,
            body1=solution.surface_id,
            phase=phase,
            contact_present=reported,
            solver_contact=solver_contact,
            filtered_reason=reason,
            contact_point_world_m=point_world,
            contact_point_local_m=point_local,
            normal_world=normal if reported else None,
            tangent_basis_world=tangent_basis,
            gap_m=solution.gap_m,
            penetration_m=solution.penetration_m,
            position_correction_m=solution.position_correction_m,
            normal_impulse_ns=solution.normal_impulse_ns if reported else 0.0,
            normal_force_n=point_force,
            tangent_force_world_n=(0.0, 0.0, 0.0),
            total_force_world_n=total_force,
            normal_relative_velocity_mps=normal_speed,
            tangent_velocity_world_mps=tangent_velocity,
            tangent_speed_mps=tangent_speed,
            kinematics_source=self.config.sampling.kinematics_source,
            kinematics_method=KINEMATICS_METHOD[self.config.sampling.kinematics_source],
            kinematics_available=kinematics_available,
            rigid_body_tangent_velocity_mps=rigid_tangent_velocity,
            probe_position_world_m=solution.position_world_m,
            probe_velocity_world_mps=velocity,
            command_velocity_world_mps=solution.command_velocity_mps,
            points=tuple(points),
            cells=cells,
            patches=tuple(patches),
            started_patch_ids=tuple(started),
            ended_episodes=tuple(ended),
            area_layers=area_layers,
            conservation=conservation,
            solution=solution,
            inside_face=solution.inside_face,
        )
