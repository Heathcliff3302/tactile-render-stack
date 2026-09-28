"""Point identity, cell aggregation, patch tracking and episode closure."""

import pytest

from contact_kernel.geometry import SurfaceGrid
from contact_kernel.spec import GridSpec, SurfaceSpec
from contact_kernel.tracking import (
    POINT_OCCUPANCY,
    ContactPatchTracker,
    ContactPoint,
    ContactPointTracker,
    aggregate_cells,
    connected_components,
)

SURFACE = SurfaceSpec(id="/World/Cube", center_world_m=(0, 0, 0.25), size_m=(0.5, 0.5, 0.5))
GRID_SPEC = GridSpec(rows=64, cols=64, cell_size_m=0.5 / 64, origin_local_m=(-0.25, -0.25, 0.25))
GRID = SurfaceGrid(GRID_SPEC, SURFACE)
DT = 1.0 / 60.0


def make_point(row, col, *, force=0.5, time_s=DT, step=1, tangent_speed=0.0, phase="hold"):
    center = GRID.cell_center_local(row, col)
    return ContactPoint(
        time_s=time_s,
        control_step=step,
        phase=phase,
        body0="/World/Fingertip",
        body1=SURFACE.id,
        point_world_m=(center[0], center[1], 0.5),
        point_local_m=center,
        normal_world=(0.0, 0.0, 1.0),
        normal_local=(0.0, 0.0, 1.0),
        tangent_u_world=(1.0, 0.0, 0.0),
        tangent_v_world=(0.0, 1.0, 0.0),
        gap_m=0.0,
        penetration_m=0.0,
        normal_impulse_ns=force * DT,
        normal_force_n=force,
        grid_row=row,
        grid_col=col,
        tangent_velocity_world_mps=(tangent_speed, 0.0, 0.0),
        tangent_speed_mps=tangent_speed,
    )


def make_patch_tracker(max_gap_frames=3):
    return ContactPatchTracker(
        cell_size_m=GRID_SPEC.cell_size_m,
        cell_area_m2=GRID_SPEC.cell_area_m2,
        max_gap_frames=max_gap_frames,
        slide_threshold_mps=0.005,
    )


def test_cell_aggregation_conserves_force_and_impulse():
    points = [make_point(32, 32, force=0.3), make_point(32, 32, force=0.2), make_point(32, 33, force=0.5)]
    cells = aggregate_cells(points, GRID, DT, 1, "hold")
    assert sorted(cells) == [(32, 32), (32, 33)]
    assert cells[(32, 32)].point_count == 2
    assert cells[(32, 32)].normal_force_n == pytest.approx(0.5)
    assert sum(cell.normal_force_n for cell in cells.values()) == pytest.approx(
        sum(point.normal_force_n for point in points)
    )
    assert sum(cell.normal_impulse_ns for cell in cells.values()) == pytest.approx(
        sum(point.normal_impulse_ns for point in points)
    )
    assert cells[(32, 32)].area_method == POINT_OCCUPANCY
    assert cells[(32, 32)].coverage_alpha == 1.0


def test_connected_components_uses_four_neighbour_connectivity():
    assert connected_components({(0, 0), (0, 1), (1, 0)}) == [{(0, 0), (0, 1), (1, 0)}]
    # Diagonal neighbours are two separate patches in the Step 6 baseline.
    assert connected_components({(0, 0), (1, 1)}) == [{(0, 0)}, {(1, 1)}]


def test_patch_id_is_stable_while_contact_continues():
    tracker = make_patch_tracker()
    ids = []
    for step in range(1, 11):
        col = 32 + step // 4
        points = [make_point(32, col, time_s=step * DT, step=step)]
        cells = aggregate_cells(points, GRID, step * DT, step, "hold")
        patches, started, ended = tracker.update(points, cells, step * DT, step, "hold")
        ids.append(patches[0].patch_id)
        assert ended == []
        assert points[0].patch_id == patches[0].patch_id
        assert cells[(32, col)].patch_id == patches[0].patch_id
    assert set(ids) == {1}


def test_patch_conserves_force_from_its_cells():
    tracker = make_patch_tracker()
    points = [make_point(32, 32, force=0.3), make_point(32, 33, force=0.7)]
    cells = aggregate_cells(points, GRID, DT, 1, "hold")
    patches, _, _ = tracker.update(points, cells, DT, 1, "hold")
    assert len(patches) == 1
    assert patches[0].normal_force_n == pytest.approx(1.0)
    assert patches[0].mapped_area_m2 == pytest.approx(2 * GRID_SPEC.cell_area_m2)
    assert sum(patch.normal_force_n for patch in patches) == pytest.approx(
        sum(cell.normal_force_n for cell in cells.values())
    )


def test_two_separated_contacts_produce_two_patches():
    tracker = make_patch_tracker()
    points = [make_point(10, 10), make_point(50, 50)]
    cells = aggregate_cells(points, GRID, DT, 1, "hold")
    patches, started, _ = tracker.update(points, cells, DT, 1, "hold")
    assert len(patches) == 2
    assert sorted(started) == [1, 2]
    assert {patch.lifecycle for patch in patches} == {"start"}


def _update(tracker, cols, step, phase="hold"):
    points = [make_point(32, col, step=step, time_s=step * DT) for col in cols]
    cells = aggregate_cells(points, GRID, step * DT, step, phase)
    return tracker.update(points, cells, step * DT, step, phase)


def test_two_patches_merging_into_one_component_record_their_sources():
    tracker = make_patch_tracker()
    # Columns 10 and 12 are not four-neighbours, so they start as two patches.
    patches, started, _ = _update(tracker, (10, 12), 1)
    assert len(patches) == 2 and sorted(started) == [1, 2]

    # Filling column 11 joins them into a single connected component.
    patches, started, _ = _update(tracker, (10, 11, 12), 2)
    assert len(patches) == 1
    assert started == []
    assert patches[0].lifecycle == "merge"
    assert patches[0].merged_from_patch_ids == [1, 2]
    assert patches[0].normal_force_n == pytest.approx(1.5)


def test_a_component_splitting_records_the_patch_it_came_from():
    tracker = make_patch_tracker()
    _update(tracker, (10, 11, 12), 1)
    patches, started, _ = _update(tracker, (10, 12), 2)
    assert len(patches) == 2
    lifecycles = {patch.patch_id: patch.lifecycle for patch in patches}
    split = [patch for patch in patches if patch.lifecycle == "split"]
    assert lifecycles[1] == "continue"
    assert len(split) == 1
    assert split[0].split_from_patch_id == 1
    assert started == [split[0].patch_id]


def test_episode_closes_after_the_declared_gap_and_reports_history():
    tracker = make_patch_tracker(max_gap_frames=2)
    for step in range(1, 6):
        points = [make_point(32, 32, time_s=step * DT, step=step, tangent_speed=0.05)]
        cells = aggregate_cells(points, GRID, step * DT, step, "lateral_slide")
        tracker.update(points, cells, step * DT, step, "lateral_slide")
    # Two empty frames are inside the gap budget, the third closes the episode.
    for step in (6, 7):
        assert tracker.update([], {}, step * DT, step, "retract")[2] == []
    _, _, ended = tracker.update([], {}, 8 * DT, 8, "retract")
    assert len(ended) == 1
    episode = ended[0]
    assert episode.patch_id == 1
    assert episode.samples == 5
    assert episode.sliding_detected is True
    assert episode.end_reason == "contact_lost"
    assert episode.max_normal_force_n == pytest.approx(0.5)
    assert episode.cumulative_normal_impulse_ns == pytest.approx(5 * 0.5 * DT)


def test_finalize_closes_a_patch_still_in_contact_at_run_end():
    tracker = make_patch_tracker()
    points = [make_point(32, 32)]
    cells = aggregate_cells(points, GRID, DT, 1, "hold")
    tracker.update(points, cells, DT, 1, "hold")
    episodes = tracker.finalize("run_end")
    assert [episode.end_reason for episode in episodes] == ["run_end"]
    assert tracker.finalize("run_end") == []


def test_point_ids_are_stable_within_the_match_distance():
    tracker = ContactPointTracker(match_distance_m=0.04, max_gap_frames=3,
                                  filter_alpha=0.5, direction_threshold_m=1e-4)
    ids = []
    for step in range(1, 6):
        point = make_point(32, 32 + step, time_s=step * DT, step=step)
        tracker.update([point], step * DT, step)
        ids.append(point.contact_id)
    assert set(ids) == {1}


def test_point_beyond_the_match_distance_gets_a_new_id():
    tracker = ContactPointTracker(match_distance_m=0.01, max_gap_frames=3,
                                  filter_alpha=0.5, direction_threshold_m=1e-4)
    first = make_point(32, 32, step=1)
    tracker.update([first], DT, 1)
    far = make_point(32, 60, step=2, time_s=2 * DT)
    tracker.update([far], 2 * DT, 2)
    assert first.contact_id == 1
    assert far.contact_id == 2


def test_sliding_point_reports_a_finite_difference_cross_check():
    tracker = ContactPointTracker(match_distance_m=0.04, max_gap_frames=3,
                                  filter_alpha=1.0, direction_threshold_m=1e-6)
    for step in (1, 2):
        point = make_point(32, 31 + step, time_s=step * DT, step=step, tangent_speed=0.05)
        tracker.update([point], step * DT, step)
    expected = GRID_SPEC.cell_size_m / DT
    assert point.finite_difference_tangent_speed_mps == pytest.approx(expected)
    assert point.direction_valid is True
    assert point.motion_direction_uv[0] == pytest.approx(1.0)
