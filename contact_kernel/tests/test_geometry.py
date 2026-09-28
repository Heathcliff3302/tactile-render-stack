"""Analytic geometry and output-grid behaviour of the kernel."""

import math

import pytest

from contact_kernel.geometry import SurfaceGrid, TopFaceGeometry
from contact_kernel.spec import EDGE_CLAMP, GridSpec, SurfaceSpec

SURFACE = SurfaceSpec(id="/World/Cube", center_world_m=(0, 0, 0.25), size_m=(0.5, 0.5, 0.5))
GRID = GridSpec(rows=64, cols=64, cell_size_m=0.5 / 64, origin_local_m=(-0.25, -0.25, 0.25))
RADIUS = 0.12


def test_sphere_gap_is_signed_and_contact_point_sits_on_the_top_face():
    geometry = TopFaceGeometry(SURFACE)
    separated = geometry.query_sphere((0.0, 0.0, 1.2), RADIUS)
    touching = geometry.query_sphere((0.0, 0.0, 0.62), RADIUS)
    sunk = geometry.query_sphere((0.0, 0.0, 0.61), RADIUS)
    assert separated.gap_m == pytest.approx(0.58)
    assert separated.present is False
    assert touching.gap_m == pytest.approx(0.0, abs=1e-15)
    assert touching.present is True
    assert sunk.gap_m == pytest.approx(-0.01)
    assert sunk.present is True
    assert touching.point_world == (0.0, 0.0, 0.5)
    assert touching.normal_world == (0.0, 0.0, 1.0)


def test_support_point_beyond_the_face_is_reported_not_approximated():
    geometry = TopFaceGeometry(SURFACE)
    outside = geometry.query_sphere((0.30, 0.0, 0.62), RADIUS)
    assert outside.inside_face is False
    assert outside.present is False
    assert outside.rejected_reason == "support_point_outside_top_face"


def test_curvature_is_zero_on_the_face_and_invalid_at_the_edge():
    geometry = TopFaceGeometry(SURFACE)
    assert geometry.curvature((0.0, 0.0, 0.25)) == (0.0, 0.0, True)
    edge = geometry.curvature((0.25, 0.0, 0.25))
    assert edge[2] is False and math.isnan(edge[0])


def test_geometric_footprint_is_zero_without_penetration():
    geometry = TopFaceGeometry(SURFACE)
    assert geometry.footprint_radius_m(0.0, RADIUS) == 0.0
    # Section radius of a sphere sunk by delta, not a Hertz contact radius.
    assert geometry.footprint_radius_m(0.001, RADIUS) == pytest.approx(
        math.sqrt(2 * RADIUS * 0.001 - 0.001**2)
    )


def test_cell_index_rejects_or_clamps_according_to_the_declared_policy():
    reject = SurfaceGrid(GRID, SURFACE)
    clamp = SurfaceGrid(
        GridSpec(rows=64, cols=64, cell_size_m=0.5 / 64,
                 origin_local_m=(-0.25, -0.25, 0.25), edge_policy=EDGE_CLAMP),
        SURFACE,
    )
    assert reject.cell_index((0.0, 0.0, 0.25)) == (32, 32)
    assert reject.cell_index((-0.25, -0.25, 0.25)) == (0, 0)
    assert reject.cell_index((0.30, 0.0, 0.25)) is None
    assert clamp.cell_index((0.30, 0.0, 0.25)) == (32, 63)


def test_cell_center_round_trips_through_cell_index():
    grid = SurfaceGrid(GRID, SURFACE)
    for row, col in ((0, 0), (32, 32), (63, 63), (7, 40)):
        assert grid.cell_index(grid.cell_center_local(row, col)) == (row, col)


def test_swept_path_closes_gaps_a_single_point_sample_would_miss():
    grid = SurfaceGrid(GRID, SURFACE)
    start = grid.cell_center_local(32, 32)
    end = grid.cell_center_local(32, 37)
    swept = grid.swept_cells(start, end)
    assert swept == tuple((32, col) for col in range(32, 38))
    assert grid.swept_cells(None, end) == ((32, 37),)


def test_disc_coverage_converges_to_the_analytic_circle_area():
    grid = SurfaceGrid(GRID, SURFACE)
    radius = 0.03
    exact = math.pi * radius**2
    errors = []
    for subdivisions in (2, 8, 32):
        coverage = grid.disc_coverage((0.0, 0.0, 0.25), radius, subdivisions=subdivisions)
        area = sum(coverage.values()) * grid.cell_area_m2
        errors.append(abs(area - exact) / exact)
        assert all(0.0 < alpha <= 1.0 for alpha in coverage.values())
    assert errors[-1] < errors[0]
    assert errors[-1] < 2e-3


def test_disc_coverage_is_empty_for_a_zero_radius_contact():
    grid = SurfaceGrid(GRID, SURFACE)
    assert grid.disc_coverage((0.0, 0.0, 0.25), 0.0) == {}
