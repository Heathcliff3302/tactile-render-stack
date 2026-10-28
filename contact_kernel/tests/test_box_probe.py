"""Flat-bottom box probe (P2.2.1).

Two things are being pinned here, and they are different in kind:

* the *dynamics* of a box are single-gap, because rotation is locked and the
  bottom face therefore stays parallel to the top face;
* the *footprint* of a box is genuinely multi-cell, and its cell count scales
  with grid resolution while the physical contact does not.

Conflating those two is the mistake this file exists to prevent. Each positive
assertion is paired with a counter-example that must fail, so that a gate which
can never go red is not mistaken for a gate that passes.
"""

from __future__ import annotations

import pytest

from contact_kernel import (
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
from contact_kernel.geometry import SurfaceGrid, TopFaceGeometry

BOX_ID = "/World/BoxProbe"
SURFACE_ID = "/World/Cube"
DT = 1.0 / 60.0
SURFACE = SurfaceSpec(id=SURFACE_ID, center_world_m=(0, 0, 0.25), size_m=(0.5, 0.5, 0.5))
PAIR = RigidContactKernel.pair_key(BOX_ID, SURFACE_ID)
# Top face sits at z = 0.5; a 0.05 m tall box resting on it has centre z = 0.525.
TOP_Z = 0.5
HALF_H = 0.025


def make_box_kernel(*, size_m=(0.10, 0.10, 0.05), start_z=0.6, rows=64):
    config = KernelConfig(
        surface=SURFACE,
        probes=(ProbeSpec(id=BOX_ID, shape="box", mass_kg=0.08,
                          initial_position_world_m=(0, 0, start_z), size_m=size_m),),
        timing=TimingSpec(physics_dt_s=DT, control_dt_s=DT, output_dt_s=DT,
                          substeps_per_control=1),
        grid=GridSpec(rows=rows, cols=rows, cell_size_m=0.5 / rows,
                      origin_local_m=(-0.25, -0.25, 0.25)),
        solver=SolverSpec(),
        sampling=SamplingSpec(),
        drive=DriveSpec(),
    )
    return RigidContactKernel(config)


def grid_of(rows):
    return SurfaceGrid(
        GridSpec(rows=rows, cols=rows, cell_size_m=0.5 / rows,
                 origin_local_m=(-0.25, -0.25, 0.25)),
        SURFACE,
    )


# ----------------------------------------------------------------- dynamics

def test_box_gap_is_measured_from_the_bottom_face():
    geometry = TopFaceGeometry(SURFACE)
    query = geometry.query_box((0.0, 0.0, TOP_Z + HALF_H + 0.01), (0.05, 0.05), HALF_H)
    assert query.gap_m == pytest.approx(0.01)
    assert query.present is False
    resting = geometry.query_box((0.0, 0.0, TOP_Z + HALF_H), (0.05, 0.05), HALF_H)
    assert resting.gap_m == pytest.approx(0.0)
    assert resting.present is True


def test_box_overhanging_the_face_is_reported_not_approximated():
    geometry = TopFaceGeometry(SURFACE)
    # Entirely clear of the 0.5 m face: centre 0.4 m out with 0.05 m half extent.
    off = geometry.query_box((0.4, 0.0, TOP_Z + HALF_H), (0.05, 0.05), HALF_H)
    assert off.inside_face is False
    assert off.present is False
    assert off.rejected_reason == "support_point_outside_top_face"
    # Straddling the edge still counts as contact, and the reported point is
    # clamped onto the face rather than left hanging in mid-air beyond it.
    straddle = geometry.query_box((0.24, 0.0, TOP_Z + HALF_H), (0.05, 0.05), HALF_H)
    assert straddle.inside_face is True
    assert straddle.point_world[0] <= 0.25


def test_box_contact_conserves_force_against_impulse():
    kernel = make_box_kernel()
    pair = kernel.step({BOX_ID: (0.0, 0.0, -0.3)}, phase="approach").pairs[PAIR]
    for _ in range(30):
        pair = kernel.step({BOX_ID: (0.0, 0.0, -0.3)}, phase="press").pairs[PAIR]
    assert pair.contact_present is True
    assert pair.normal_force_n == pytest.approx(pair.normal_impulse_ns / DT)


# ---------------------------------------------------------------- footprint

def test_box_footprint_activates_multiple_cells():
    grid = grid_of(64)
    coverage = grid.rect_coverage((0.0, 0.0, 0.25), (0.05, 0.05))
    assert len(coverage) > 1
    assert all(0.0 < alpha <= 1.0 for alpha in coverage.values())


def test_box_footprint_area_is_exact_and_resolution_independent():
    # The whole point of the box case: the *physical* footprint is a fixed
    # 0.1 x 0.1 m rectangle, so summing alpha * cell_area must recover it at
    # every resolution, even though the cell count changes with resolution.
    expected = 0.10 * 0.10
    counts = {}
    for rows in (32, 64, 128):
        grid = grid_of(rows)
        coverage = grid.rect_coverage((0.0, 0.0, 0.25), (0.05, 0.05))
        area = sum(alpha * grid.cell_area_m2 for alpha in coverage.values())
        assert area == pytest.approx(expected, rel=1e-9)
        counts[rows] = len(coverage)
    # Cell count is a proxy that scales with resolution; the area above is not.
    assert counts[32] < counts[64] < counts[128]


def test_rect_coverage_is_exact_where_disc_coverage_is_quadrature():
    # A rectangle aligned to cell boundaries covers whole cells exactly, with
    # no quadrature residue. This is what makes rect_coverage free of the
    # convergence parameter that disc_coverage needs.
    grid = grid_of(64)
    cell = 0.5 / 64
    coverage = grid.rect_coverage((0.0, 0.0, 0.25), (cell, cell))
    assert len(coverage) == 4
    assert all(alpha == pytest.approx(1.0) for alpha in coverage.values())


# ----------------------------------------------------- counter-examples

def test_counterexample_a_lifted_box_activates_no_cells_and_no_contact():
    """The multi-cell claim must be able to go red."""
    kernel = make_box_kernel(start_z=1.2)
    pair = kernel.step({BOX_ID: (0.0, 0.0, 0.0)}, phase="pre_contact").pairs[PAIR]
    assert pair.contact_present is False
    assert pair.normal_impulse_ns == 0.0


def test_counterexample_a_degenerate_footprint_covers_nothing():
    """A zero-extent rectangle must not silently report coverage."""
    grid = grid_of(64)
    assert grid.rect_coverage((0.0, 0.0, 0.25), (0.0, 0.05)) == {}
    assert grid.rect_coverage((0.0, 0.0, 0.25), (-0.01, 0.05)) == {}


def test_counterexample_footprint_off_the_face_covers_nothing():
    grid = grid_of(64)
    assert grid.rect_coverage((10.0, 10.0, 0.25), (0.05, 0.05)) == {}


# ------------------------------------------- P2.2.1 uniform-proxy load spread

def test_box_load_spreads_over_cells_and_conserves_exactly():
    kernel = make_box_kernel()
    pair = None
    for index in range(40):
        pair = kernel.step({BOX_ID: (0.0, 0.0, -0.3)},
                           phase="approach" if index < 10 else "press").pairs[PAIR]
    assert pair.contact_present is True
    assert len(pair.cells) > 1
    # The shares are normalised, so the per-cell forces must add back to the
    # solver's own number. This is what keeps the conservation gate honest
    # while the distribution itself is only a proxy.
    assert pair.conservation.point_force_n == pytest.approx(pair.normal_force_n)
    assert pair.conservation.cell_force_n == pytest.approx(pair.normal_force_n)
    assert pair.conservation.point_impulse_ns == pytest.approx(pair.normal_impulse_ns)


def test_box_cell_count_grows_with_resolution_but_force_does_not():
    forces, counts = [], []
    for rows in (32, 64, 128):
        kernel = make_box_kernel(rows=rows)
        pair = None
        for index in range(40):
            pair = kernel.step({BOX_ID: (0.0, 0.0, -0.3)},
                               phase="approach" if index < 10 else "press").pairs[PAIR]
        counts.append(len(pair.cells))
        forces.append(pair.normal_force_n)
    assert counts[0] < counts[1] < counts[2]
    assert forces[0] == pytest.approx(forces[1]) == pytest.approx(forces[2])


def test_box_area_layers_name_the_proxy():
    """The number must not travel without the method that produced it."""
    from contact_kernel.tracking import BOX_UNIFORM_PROXY

    kernel = make_box_kernel()
    pair = None
    for index in range(40):
        pair = kernel.step({BOX_ID: (0.0, 0.0, -0.3)},
                           phase="approach" if index < 10 else "press").pairs[PAIR]
    note = pair.area_layers.to_dict()["note"]
    assert BOX_UNIFORM_PROXY in note
    assert "singularity" in note
    assert pair.area_layers.footprint_radius_m is None


def test_counterexample_uniform_proxy_is_not_a_pressure_field():
    """Pin the known defect so nobody later reads the proxy as physics.

    Measured, not assumed: the per-cell loads are NOT all equal. Fully covered
    interior cells carry one identical share; cells the footprint only clips
    carry strictly less, in proportion to the covered fraction.

    That makes the proxy wrong at the rim in *two* compounding ways:

    * physics says a rigid flat punch loads the rim HARDER than the centre
      (edge pressure singularity);
    * this discretisation makes clipped rim cells look WEAKER than interior
      cells, because partial coverage scales their share down.

    So the rim error is not a wash — both effects push the same direction, and
    the rim is understated twice over. Downstream code must treat rim cells of
    a box footprint as unusable for anything pressure-like until a real
    distribution exists (P2.5.2 Winkler at the earliest).
    """
    kernel = make_box_kernel(rows=64)
    pair = None
    for index in range(40):
        pair = kernel.step({BOX_ID: (0.0, 0.0, -0.3)},
                           phase="approach" if index < 10 else "press").pairs[PAIR]
    loads = sorted(cell.normal_force_n for cell in pair.cells.values())
    full = max(loads)
    interior = [value for value in loads if value == pytest.approx(full, rel=1e-9)]
    clipped = [value for value in loads if value < full * (1.0 - 1e-9)]

    # Interior cells are uniform: that is the proxy's defining property.
    assert len(interior) > 1
    assert max(interior) == pytest.approx(min(interior), rel=1e-9)

    # Clipped rim cells exist and are strictly weaker, never stronger.
    assert clipped, "expected partially covered rim cells at this box/grid ratio"
    assert max(clipped) < full
    # The rim is understated, which is the opposite of the real punch solution.
    assert min(clipped) < full
