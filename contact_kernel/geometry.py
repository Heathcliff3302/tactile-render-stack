"""Analytic geometry queries and surface discretisation.

Everything here is closed form for the K1 pair: a sphere probe against the top
face of an axis-aligned fixed box. The grid helpers are kept separate from the
solver because the grid is an *output* discretisation. Changing the grid must
not change the dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .spec import EDGE_CLAMP, GridSpec, SurfaceSpec, Vec3

# Surface-local frame of the Step 6 cube: origin at the cube centre, axes
# aligned with world axes, so local vectors equal world vectors.
LOCAL_FRAME_NOTE = "surface_local = world - surface_center; axes identical to world"


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def scale(a: Vec3, factor: float) -> Vec3:
    return (a[0] * factor, a[1] * factor, a[2] * factor)


def dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


@dataclass(frozen=True)
class ContactQuery:
    """Signed separation of one probe support point from the top face.

    ``gap_m`` is positive when separated, zero when touching and negative when
    interpenetrating. ``inside_face`` reports whether the support point lies on
    the finite top face; edge and side contact are outside the K1 scope and are
    reported rather than silently approximated.
    """

    present: bool
    gap_m: float
    point_world: Vec3
    normal_world: Vec3
    inside_face: bool
    rejected_reason: str = ""


class TopFaceGeometry:
    """Sphere-versus-finite-top-face queries in world coordinates."""

    def __init__(self, surface: SurfaceSpec):
        self.surface = surface
        self.top_z_m = surface.top_z_world_m
        self.half_x_m, self.half_y_m = surface.half_extent_m
        self.normal_world: Vec3 = (0.0, 0.0, 1.0)
        self.tangent_u_world: Vec3 = (1.0, 0.0, 0.0)
        self.tangent_v_world: Vec3 = (0.0, 1.0, 0.0)

    def query_sphere(self, center_world: Vec3, radius_m: float, *, surface_offset_m: float = 0.0) -> ContactQuery:
        x, y, z = center_world
        plane_z = self.top_z_m + surface_offset_m
        gap = z - (plane_z + radius_m)
        point = (x, y, plane_z)
        cx, cy = self.surface.center_world_m[0], self.surface.center_world_m[1]
        inside = abs(x - cx) <= self.half_x_m and abs(y - cy) <= self.half_y_m
        reason = "" if inside else "support_point_outside_top_face"
        return ContactQuery(
            present=gap <= 0.0 and inside,
            gap_m=gap,
            point_world=point,
            normal_world=self.normal_world,
            inside_face=inside,
            rejected_reason=reason,
        )

    def to_local(self, point_world: Vec3) -> Vec3:
        return sub(point_world, self.surface.center_world_m)

    def curvature(self, point_local: Vec3, tolerance: float = 1.0e-4) -> tuple[float, float, bool]:
        """Top-face principal curvatures. Invalid near an edge, as in Step 6."""
        half_z = self.surface.size_m[2] / 2.0
        on_top = abs(point_local[2] - half_z) <= tolerance
        away_from_edge = (
            abs(point_local[0]) < self.half_x_m - tolerance
            and abs(point_local[1]) < self.half_y_m - tolerance
        )
        if on_top and away_from_edge:
            return 0.0, 0.0, True
        return math.nan, math.nan, False

    def footprint_radius_m(self, penetration_m: float, radius_m: float) -> float:
        """Geometric section radius of a sphere sunk ``penetration_m`` into a plane.

        This is the intersection of the undeformed shapes, not a Hertz contact
        radius. A non-penetrating rigid contact has a zero-radius footprint,
        which is exactly why an elastic contact area needs the K4 material
        models instead of this quantity.
        """
        depth = min(max(penetration_m, 0.0), radius_m)
        return math.sqrt(max(0.0, 2.0 * radius_m * depth - depth * depth))


class SurfaceGrid:
    """Regular grid over the top face, indexed in surface-local XY."""

    def __init__(self, grid: GridSpec, surface: SurfaceSpec):
        self.spec = grid
        self.surface = surface
        self.cell_size_m = grid.cell_size_m
        self.cell_area_m2 = grid.cell_area_m2
        self.origin_local_m = grid.origin_local_m
        self.top_local_z_m = surface.size_m[2] / 2.0

    def cell_index(self, point_local: Vec3) -> tuple[int, int] | None:
        """Return ``(row, col)`` or ``None`` when the point misses the face."""
        col_raw = math.floor((point_local[0] - self.origin_local_m[0]) / self.cell_size_m)
        row_raw = math.floor((point_local[1] - self.origin_local_m[1]) / self.cell_size_m)
        inside = 0 <= row_raw < self.spec.rows and 0 <= col_raw < self.spec.cols
        if inside:
            return int(row_raw), int(col_raw)
        if self.spec.edge_policy == EDGE_CLAMP:
            row = min(self.spec.rows - 1, max(0, int(row_raw)))
            col = min(self.spec.cols - 1, max(0, int(col_raw)))
            return row, col
        return None

    def cell_center_local(self, row: int, col: int) -> Vec3:
        return (
            self.origin_local_m[0] + (col + 0.5) * self.cell_size_m,
            self.origin_local_m[1] + (row + 0.5) * self.cell_size_m,
            self.top_local_z_m,
        )

    def swept_cells(self, start_local: Vec3 | None, end_local: Vec3) -> tuple[tuple[int, int], ...]:
        """Supercover rasterisation of the contact-point path between two frames.

        Point occupancy alone drops cells when the contact point moves more than
        one cell per output step. This layer closes those gaps without claiming
        the swept set is an instantaneous contact area.
        """
        end_cell = self.cell_index(end_local)
        if start_local is None:
            return () if end_cell is None else (end_cell,)
        distance = math.hypot(end_local[0] - start_local[0], end_local[1] - start_local[1])
        steps = max(1, int(math.ceil(distance / (0.5 * self.cell_size_m))))
        cells: dict[tuple[int, int], None] = {}
        for index in range(steps + 1):
            ratio = index / steps
            sample = (
                start_local[0] + (end_local[0] - start_local[0]) * ratio,
                start_local[1] + (end_local[1] - start_local[1]) * ratio,
                end_local[2],
            )
            cell = self.cell_index(sample)
            if cell is not None:
                cells[cell] = None
        return tuple(sorted(cells))

    def disc_coverage(
        self,
        center_local: Vec3,
        radius_m: float,
        *,
        subdivisions: int = 8,
    ) -> dict[tuple[int, int], float]:
        """Per-cell area fraction covered by a disc, by uniform sub-cell quadrature.

        Returns ``{(row, col): alpha}`` with ``alpha`` in ``(0, 1]``. Accuracy is
        set by ``subdivisions``; the convergence test in ``tests`` pins it. Only
        the region of interest around the disc is visited.
        """
        if radius_m <= 0.0:
            return {}
        coverage: dict[tuple[int, int], float] = {}
        weight = 1.0 / (subdivisions * subdivisions)
        col_lo = math.floor((center_local[0] - radius_m - self.origin_local_m[0]) / self.cell_size_m)
        col_hi = math.floor((center_local[0] + radius_m - self.origin_local_m[0]) / self.cell_size_m)
        row_lo = math.floor((center_local[1] - radius_m - self.origin_local_m[1]) / self.cell_size_m)
        row_hi = math.floor((center_local[1] + radius_m - self.origin_local_m[1]) / self.cell_size_m)
        radius_squared = radius_m * radius_m
        for row in range(row_lo, row_hi + 1):
            if not 0 <= row < self.spec.rows:
                continue
            for col in range(col_lo, col_hi + 1):
                if not 0 <= col < self.spec.cols:
                    continue
                x0 = self.origin_local_m[0] + col * self.cell_size_m
                y0 = self.origin_local_m[1] + row * self.cell_size_m
                covered = 0
                for sub_y in range(subdivisions):
                    py = y0 + (sub_y + 0.5) * self.cell_size_m / subdivisions
                    dy = py - center_local[1]
                    for sub_x in range(subdivisions):
                        px = x0 + (sub_x + 0.5) * self.cell_size_m / subdivisions
                        dx = px - center_local[0]
                        if dx * dx + dy * dy <= radius_squared:
                            covered += 1
                if covered:
                    coverage[(row, col)] = covered * weight
        return coverage
