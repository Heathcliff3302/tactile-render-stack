"""Point identity, cell aggregation and tracked contact patches.

The Step 6 chain is reproduced here without Isaac Sim:

    contact point -> grid cell -> connected patch -> episode

Cross-frame matching produces trajectories. A trajectory never restores an
instantaneous contact area, so every area a cell or patch reports stays
labelled with the method that produced it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import SurfaceGrid

POINT_OCCUPANCY = "point_occupancy"
SWEPT_PATH = "swept_path"
GEOMETRIC_FOOTPRINT = "geometric_footprint"

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]


@dataclass
class ContactPoint:
    """One solved contact point at one output step."""

    time_s: float
    control_step: int
    phase: str
    body0: str
    body1: str
    point_world_m: Vec3
    point_local_m: Vec3
    normal_world: Vec3
    normal_local: Vec3
    tangent_u_world: Vec3
    tangent_v_world: Vec3
    gap_m: float
    penetration_m: float
    normal_impulse_ns: float
    normal_force_n: float
    tangent_force_world_n: Vec3 = (0.0, 0.0, 0.0)
    grid_row: int = -1
    grid_col: int = -1
    curvature_1_per_m: float = math.nan
    curvature_2_per_m: float = math.nan
    curvature_valid: bool = False
    contact_id: int = 0
    patch_id: int = 0
    normal_relative_velocity_mps: float = 0.0
    tangent_velocity_world_mps: Vec3 = (0.0, 0.0, 0.0)
    tangent_speed_mps: float = 0.0
    tangent_u_speed_mps: float = 0.0
    tangent_v_speed_mps: float = 0.0
    kinematics_source: str = "rigid_body_twist"
    motion_direction_uv: Vec2 = (0.0, 0.0)
    direction_valid: bool = False
    # Both kinematics sources are always computed. The declared source decides
    # which one is authoritative; the other stays as a cross-check, so a parity
    # comparison can see the difference instead of only one of the two.
    rigid_body_tangent_speed_mps: float = math.nan
    finite_difference_available: bool = False
    finite_difference_velocity_world_mps: Vec3 = (0.0, 0.0, 0.0)
    finite_difference_normal_velocity_mps: float = math.nan
    finite_difference_tangent_world_mps: Vec3 = (0.0, 0.0, 0.0)
    finite_difference_tangent_speed_mps: float = math.nan


@dataclass
class ContactCell:
    """Sparse state of one occupied grid cell at one output step."""

    time_s: float
    control_step: int
    phase: str
    row: int
    col: int
    center_local_m: Vec3
    cell_area_m2: float
    coverage_alpha: float
    area_method: str
    point_count: int
    contact_ids: list[int]
    normal_force_n: float
    normal_impulse_ns: float
    tangent_force_world_n: Vec3
    normal_local: Vec3
    curvature_1_per_m: float
    curvature_2_per_m: float
    curvature_valid: bool
    mean_normal_relative_velocity_mps: float
    mean_tangent_speed_mps: float
    motion_direction_uv: Vec2
    direction_valid: bool
    patch_id: int = 0
    interaction_state: str = "contact"


@dataclass
class ContactPatch:
    """One connected component of occupied cells at one output step."""

    time_s: float
    control_step: int
    phase: str
    patch_id: int
    interaction_state: str
    lifecycle: str
    active_cells: list[tuple[int, int]]
    mapped_area_m2: float
    area_method: str
    contact_point_count: int
    contact_ids: list[int]
    centroid_local_m: Vec3
    geometric_centroid_local_m: Vec3
    normal_force_n: float
    normal_impulse_ns: float
    tangent_force_world_n: Vec3
    mean_normal_local: Vec3
    curvature_1_per_m: float
    curvature_2_per_m: float
    curvature_valid: bool
    mean_normal_relative_velocity_mps: float
    mean_tangent_speed_mps: float
    motion_direction_uv: Vec2
    direction_valid: bool
    merged_from_patch_ids: list[int] = field(default_factory=list)
    split_from_patch_id: int | None = None


@dataclass
class ContactEpisode:
    """Closed history of one tracked patch."""

    patch_id: int
    start_time_s: float
    end_time_s: float
    first_step: int
    last_step: int
    samples: int
    trajectory_length_m: float
    max_normal_force_n: float
    cumulative_normal_impulse_ns: float
    max_mapped_area_m2: float
    swept_area_m2: float
    sliding_detected: bool
    end_reason: str

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_finite(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _normalize2(vector: Vec2) -> Vec2:
    length = math.hypot(vector[0], vector[1])
    return (vector[0] / length, vector[1] / length) if length > 1.0e-12 else (0.0, 0.0)


def _normalize3(vector: Vec3) -> Vec3:
    length = math.sqrt(sum(value * value for value in vector))
    return tuple(value / length for value in vector) if length > 1.0e-12 else (0.0, 0.0, 0.0)


@dataclass
class _PointTrack:
    last_point_world_m: Vec3
    last_filtered_local_m: Vec3
    last_time_s: float
    last_step: int
    samples: int = 0


class ContactPointTracker:
    """Nearest-neighbour point association with deterministic ID assignment."""

    def __init__(self, *, match_distance_m: float, max_gap_frames: int,
                 filter_alpha: float, direction_threshold_m: float):
        self.match_distance_m = float(match_distance_m)
        self.max_gap_frames = int(max_gap_frames)
        self.filter_alpha = float(filter_alpha)
        self.direction_threshold_m = float(direction_threshold_m)
        self._tracks: dict[int, _PointTrack] = {}
        self._next_id = 1

    def update(self, points: list[ContactPoint], time_s: float, step: int) -> list[ContactPoint]:
        available = set(self._tracks)
        assignments = []
        for point in points:
            candidates = []
            for contact_id in sorted(available):
                track = self._tracks[contact_id]
                if step - track.last_step > self.max_gap_frames:
                    continue
                distance = math.dist(point.point_world_m, track.last_point_world_m)
                if distance <= self.match_distance_m:
                    candidates.append((distance, contact_id))
            if candidates:
                _, contact_id = min(candidates)
                available.discard(contact_id)
            else:
                contact_id = self._next_id
                self._next_id += 1
                self._tracks[contact_id] = _PointTrack(
                    last_point_world_m=point.point_world_m,
                    last_filtered_local_m=point.point_local_m,
                    last_time_s=time_s,
                    last_step=step,
                )
            assignments.append((point, contact_id))

        for point, contact_id in assignments:
            track = self._tracks[contact_id]
            point.contact_id = contact_id
            if track.samples and time_s > track.last_time_s:
                interval = time_s - track.last_time_s
                difference = tuple(
                    (a - b) / interval for a, b in zip(point.point_world_m, track.last_point_world_m)
                )
                normal_component = sum(a * b for a, b in zip(difference, point.normal_world))
                tangent = tuple(
                    d - normal_component * n for d, n in zip(difference, point.normal_world)
                )
                point.finite_difference_available = True
                point.finite_difference_velocity_world_mps = difference
                point.finite_difference_normal_velocity_mps = normal_component
                point.finite_difference_tangent_world_mps = tangent
                point.finite_difference_tangent_speed_mps = math.sqrt(sum(v * v for v in tangent))
            filtered = tuple(
                self.filter_alpha * current + (1.0 - self.filter_alpha) * previous
                for current, previous in zip(point.point_local_m, track.last_filtered_local_m)
            )
            delta = tuple(a - b for a, b in zip(filtered, track.last_filtered_local_m))
            normal_delta = sum(a * b for a, b in zip(delta, point.normal_local))
            tangent_delta = tuple(d - normal_delta * n for d, n in zip(delta, point.normal_local))
            if math.sqrt(sum(v * v for v in tangent_delta)) >= self.direction_threshold_m:
                direction = _normalize3(tangent_delta)
                point.motion_direction_uv = (
                    sum(a * b for a, b in zip(direction, point.tangent_u_world)),
                    sum(a * b for a, b in zip(direction, point.tangent_v_world)),
                )
                point.direction_valid = True
            track.last_point_world_m = point.point_world_m
            track.last_filtered_local_m = filtered
            track.last_time_s = time_s
            track.last_step = step
            track.samples += 1

        for contact_id in sorted(self._tracks):
            if step - self._tracks[contact_id].last_step > self.max_gap_frames:
                del self._tracks[contact_id]
        return points


def aggregate_cells(points: list[ContactPoint], grid: SurfaceGrid, time_s: float,
                    step: int, phase: str) -> dict[tuple[int, int], ContactCell]:
    """Sum point quantities into the cells they occupy. Total force is conserved."""
    groups: dict[tuple[int, int], list[ContactPoint]] = {}
    for point in points:
        groups.setdefault((point.grid_row, point.grid_col), []).append(point)

    cells: dict[tuple[int, int], ContactCell] = {}
    for (row, col), members in sorted(groups.items()):
        force = sum(point.normal_force_n for point in members)
        weighted_normal = (0.0, 0.0, 0.0)
        tangent_force = (0.0, 0.0, 0.0)
        for point in members:
            weighted_normal = tuple(
                acc + component * point.normal_force_n
                for acc, component in zip(weighted_normal, point.normal_local)
            )
            tangent_force = tuple(
                acc + component for acc, component in zip(tangent_force, point.tangent_force_world_n)
            )
        directions = [point.motion_direction_uv for point in members if point.direction_valid]
        direction = _normalize2((
            sum(item[0] for item in directions),
            sum(item[1] for item in directions),
        )) if directions else (0.0, 0.0)
        curvature_valid = all(point.curvature_valid for point in members)
        cells[(row, col)] = ContactCell(
            time_s=time_s,
            control_step=step,
            phase=phase,
            row=row,
            col=col,
            center_local_m=grid.cell_center_local(row, col),
            cell_area_m2=grid.cell_area_m2,
            coverage_alpha=1.0,
            area_method=POINT_OCCUPANCY,
            point_count=len(members),
            contact_ids=sorted({point.contact_id for point in members}),
            normal_force_n=force,
            normal_impulse_ns=sum(point.normal_impulse_ns for point in members),
            tangent_force_world_n=tangent_force,
            normal_local=_normalize3(weighted_normal),
            curvature_1_per_m=_mean_finite([point.curvature_1_per_m for point in members]) if curvature_valid else math.nan,
            curvature_2_per_m=_mean_finite([point.curvature_2_per_m for point in members]) if curvature_valid else math.nan,
            curvature_valid=curvature_valid,
            mean_normal_relative_velocity_mps=_mean([point.normal_relative_velocity_mps for point in members]),
            mean_tangent_speed_mps=_mean([point.tangent_speed_mps for point in members]),
            motion_direction_uv=direction,
            direction_valid=bool(directions),
        )
    return cells


def connected_components(cell_keys) -> list[set[tuple[int, int]]]:
    """Four-neighbour connected components, matching the Step 6 baseline."""
    remaining = set(cell_keys)
    components = []
    while remaining:
        seed = min(remaining)
        remaining.discard(seed)
        component = {seed}
        frontier = [seed]
        while frontier:
            row, col = frontier.pop()
            for neighbour in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbour in remaining:
                    remaining.discard(neighbour)
                    component.add(neighbour)
                    frontier.append(neighbour)
        components.append(component)
    return sorted(components, key=min)


@dataclass
class _PatchTrack:
    start_time_s: float
    last_time_s: float
    first_step: int
    last_step: int
    last_cells: set
    last_centroid_local_m: Vec3
    swept_cells: set
    samples: int = 0
    trajectory_length_m: float = 0.0
    max_normal_force_n: float = 0.0
    cumulative_normal_impulse_ns: float = 0.0
    max_mapped_area_m2: float = 0.0
    sliding_detected: bool = False


class ContactPatchTracker:
    """Overlap-and-proximity patch matching with stable IDs and episodes."""

    def __init__(self, *, cell_size_m: float, cell_area_m2: float,
                 max_gap_frames: int, slide_threshold_mps: float):
        self.cell_size_m = float(cell_size_m)
        self.cell_area_m2 = float(cell_area_m2)
        self.max_gap_frames = int(max_gap_frames)
        self.slide_threshold_mps = float(slide_threshold_mps)
        self._tracks: dict[int, _PatchTrack] = {}
        self._next_id = 1

    def update(self, points, cells, time_s, step, phase):
        ended = self._expire(step)
        components = connected_components(cells)
        points_by_cell: dict[tuple[int, int], list[ContactPoint]] = {}
        for point in points:
            points_by_cell.setdefault((point.grid_row, point.grid_col), []).append(point)

        # Candidate overlaps are collected for every component before any ID is
        # taken, so merges and splits are observable instead of implied.
        overlaps = []
        for component in components:
            matches = {
                patch_id: len(component & track.last_cells)
                for patch_id, track in self._tracks.items()
                if component & track.last_cells
            }
            overlaps.append(matches)
        track_usage: dict[int, int] = {}
        for matches in overlaps:
            for patch_id in matches:
                track_usage[patch_id] = track_usage.get(patch_id, 0) + 1

        patches, started = [], []
        available = set(self._tracks)
        for component, matches in zip(components, overlaps):
            component_cells = sorted(component)
            members = [point for cell in component_cells for point in points_by_cell[cell]]
            force = sum(point.normal_force_n for point in members)
            if force > 1.0e-12:
                centroid = tuple(
                    sum(point.point_local_m[axis] * point.normal_force_n for point in members) / force
                    for axis in range(3)
                )
            else:
                centroid = tuple(_mean([point.point_local_m[axis] for point in members]) for axis in range(3))
            geometric_centroid = tuple(
                _mean([point.point_local_m[axis] for point in members]) for axis in range(3)
            )

            candidates = []
            for patch_id in sorted(available):
                track = self._tracks[patch_id]
                overlap = matches.get(patch_id, 0)
                distance = math.dist(centroid, track.last_centroid_local_m)
                if overlap > 0 or distance <= 2.0 * self.cell_size_m:
                    candidates.append((overlap, -distance, -patch_id))
            merged_from, split_from, lifecycle = [], None, "continue"
            if candidates:
                overlap, _, negative_id = max(candidates)
                patch_id = -negative_id
                available.discard(patch_id)
                if len(matches) > 1:
                    merged_from = sorted(matches)
                    lifecycle = "merge"
            else:
                patch_id = self._next_id
                self._next_id += 1
                lifecycle = "start"
                started.append(patch_id)
                self._tracks[patch_id] = _PatchTrack(
                    start_time_s=time_s,
                    last_time_s=time_s,
                    first_step=step,
                    last_step=step,
                    last_cells=set(component),
                    last_centroid_local_m=centroid,
                    swept_cells=set(component),
                )
                parents = [source for source, count in track_usage.items() if count > 1 and source in matches]
                if parents:
                    split_from = min(parents)
                    lifecycle = "split"

            tangent_speed = _mean([point.tangent_speed_mps for point in members])
            normal_speed = _mean([point.normal_relative_velocity_mps for point in members])
            directions = [point.motion_direction_uv for point in members if point.direction_valid]
            direction = _normalize2((
                sum(item[0] for item in directions),
                sum(item[1] for item in directions),
            )) if directions else (0.0, 0.0)
            tangent_force = (0.0, 0.0, 0.0)
            weighted_normal = (0.0, 0.0, 0.0)
            for point in members:
                tangent_force = tuple(
                    acc + value for acc, value in zip(tangent_force, point.tangent_force_world_n)
                )
                weighted_normal = tuple(
                    acc + value * point.normal_force_n
                    for acc, value in zip(weighted_normal, point.normal_local)
                )
            curvature_valid = all(point.curvature_valid for point in members)
            interaction_state = self._interaction_state(phase, lifecycle == "start", tangent_speed)
            mapped_area = len(component_cells) * self.cell_area_m2

            for point in members:
                point.patch_id = patch_id
            for cell in component_cells:
                cells[cell].patch_id = patch_id
                cells[cell].interaction_state = interaction_state

            patch = ContactPatch(
                time_s=time_s,
                control_step=step,
                phase=phase,
                patch_id=patch_id,
                interaction_state=interaction_state,
                lifecycle=lifecycle,
                active_cells=component_cells,
                mapped_area_m2=mapped_area,
                area_method=POINT_OCCUPANCY,
                contact_point_count=len(members),
                contact_ids=sorted({point.contact_id for point in members}),
                centroid_local_m=centroid,
                geometric_centroid_local_m=geometric_centroid,
                normal_force_n=force,
                normal_impulse_ns=sum(point.normal_impulse_ns for point in members),
                tangent_force_world_n=tangent_force,
                mean_normal_local=_normalize3(weighted_normal),
                curvature_1_per_m=_mean_finite([point.curvature_1_per_m for point in members]) if curvature_valid else math.nan,
                curvature_2_per_m=_mean_finite([point.curvature_2_per_m for point in members]) if curvature_valid else math.nan,
                curvature_valid=curvature_valid,
                mean_normal_relative_velocity_mps=normal_speed,
                mean_tangent_speed_mps=tangent_speed,
                motion_direction_uv=direction,
                direction_valid=bool(directions),
                merged_from_patch_ids=merged_from,
                split_from_patch_id=split_from,
            )
            patches.append(patch)
            self._update_track(patch_id, patch)
        return patches, started, ended

    def finalize(self, reason: str = "run_end") -> list[ContactEpisode]:
        episodes = [self._to_episode(patch_id, track, reason)
                    for patch_id, track in sorted(self._tracks.items())]
        self._tracks.clear()
        return episodes

    def _interaction_state(self, phase: str, is_new: bool, tangent_speed_mps: float) -> str:
        if phase in ("lateral_slide", "lateral_slide_force") or tangent_speed_mps >= self.slide_threshold_mps:
            return "sliding"
        if is_new:
            return "init_contact"
        if phase in ("hold", "regulate_force"):
            return "hold"
        return "press"

    def _update_track(self, patch_id: int, patch: ContactPatch) -> None:
        track = self._tracks[patch_id]
        track.trajectory_length_m += math.dist(patch.centroid_local_m, track.last_centroid_local_m)
        track.last_time_s = patch.time_s
        track.last_step = patch.control_step
        track.last_cells = set(patch.active_cells)
        track.last_centroid_local_m = patch.centroid_local_m
        track.swept_cells |= set(patch.active_cells)
        track.samples += 1
        track.max_normal_force_n = max(track.max_normal_force_n, patch.normal_force_n)
        track.cumulative_normal_impulse_ns += patch.normal_impulse_ns
        track.max_mapped_area_m2 = max(track.max_mapped_area_m2, patch.mapped_area_m2)
        track.sliding_detected |= patch.mean_tangent_speed_mps >= self.slide_threshold_mps

    def _expire(self, step: int) -> list[ContactEpisode]:
        ended = []
        for patch_id in sorted(self._tracks):
            track = self._tracks[patch_id]
            if step - track.last_step > self.max_gap_frames:
                ended.append(self._to_episode(patch_id, track, "contact_lost"))
                del self._tracks[patch_id]
        return ended

    def _to_episode(self, patch_id: int, track: _PatchTrack, reason: str) -> ContactEpisode:
        return ContactEpisode(
            patch_id=patch_id,
            start_time_s=track.start_time_s,
            end_time_s=track.last_time_s,
            first_step=track.first_step,
            last_step=track.last_step,
            samples=track.samples,
            trajectory_length_m=track.trajectory_length_m,
            max_normal_force_n=track.max_normal_force_n,
            cumulative_normal_impulse_ns=track.cumulative_normal_impulse_ns,
            max_mapped_area_m2=track.max_mapped_area_m2,
            swept_area_m2=len(track.swept_cells) * self.cell_area_m2,
            sliding_detected=track.sliding_detected,
            end_reason=reason,
        )
