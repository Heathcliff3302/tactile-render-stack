"""The dense tensor must carry the sparse cells and change nothing else.

Densification is a storage change. The risk is that it quietly becomes a
numerical change too — a dropped frame, an averaged value, a zero standing in
for an undefined one — after which every downstream consumer reads something
the sparse evidence does not say.

Two failures from Phase 1 are pinned here. A timeline built from the cell rows
alone drops ``pre_contact`` and ``released``, and with them the evidence that
force returns to zero after release. And writing ``0.0`` where curvature or
direction was undefined makes "undefined" indistinguishable from "flat" and
"not moving".
"""

import numpy as np
import pytest

from tactile_contract.dense_field import (
    CHANNEL_SEMANTICS,
    CHANNELS,
    DENSE_SCHEMA,
    build_dense_field,
    dense_metadata,
    read_dense_field,
    write_dense_field,
)


@pytest.fixture(scope="module")
def dense(k1_run):
    return build_dense_field(k1_run)


def test_the_channel_layout_is_the_phase_1_contract():
    # An existing Step 6.4 consumer must read this file unchanged.
    assert CHANNELS == (
        "contact_mask", "normal_force_N", "normal_impulse_Ns",
        "tangent_force_x_N", "tangent_force_y_N", "tangent_force_z_N",
        "normal_velocity_mps", "tangent_speed_mps",
        "direction_u", "direction_v",
        "curvature_1_per_m", "curvature_2_per_m", "patch_id",
    )
    assert len(CHANNELS) == 13
    # Every channel says what it is and whether summing it is meaningful.
    assert set(CHANNEL_SEMANTICS) == set(CHANNELS)
    for name, entry in CHANNEL_SEMANTICS.items():
        assert entry["unit"] and entry["description"], name
        assert entry["role"] in {"occupancy", "additive", "intensive", "label"}


def test_the_time_axis_is_every_control_step(dense, k1_run):
    """The Phase 1 failure: a cells-derived timeline drops the empty segments."""
    assert dense["field"].shape[0] == len(k1_run.loop_frames)
    assert list(dense["control_step"]) == [f.control_step for f in k1_run.loop_frames]
    phases = [str(value) for value in dense["phase"]]
    assert phases == [frame.phase for frame in k1_run.loop_frames]
    # The frames with no contact at all are present, which is the point.
    assert phases.count("pre_contact") == 30
    assert phases.count("released") == 30
    contact_frames = {
        index for index, frame in enumerate(k1_run.loop_frames)
        if frame.interaction.contact_present
    }
    assert len(contact_frames) < len(phases)


def test_released_frames_are_zero_not_absent(dense):
    mask_index = CHANNELS.index("contact_mask")
    force_index = CHANNELS.index("normal_force_N")
    released = dense["field"][np.asarray([str(p) for p in dense["phase"]]) == "released"]
    assert released.shape[0] == 30
    assert released[..., mask_index].sum() == 0.0
    assert released[..., force_index].sum() == 0.0


def test_every_sparse_cell_appears_once_at_its_grid_position(dense, k1_run):
    from contact_kernel import RigidContactKernel

    config = k1_run.kernel_config
    key = RigidContactKernel.pair_key(config.probes[0].id, config.surface.id)
    field = dense["field"]
    mask_index = CHANNELS.index("contact_mask")
    force_index = CHANNELS.index("normal_force_N")
    total_sparse = 0
    for index, loop_frame in enumerate(k1_run.loop_frames):
        pair = loop_frame.world.kernel_frame.pairs[key]
        occupied = {(int(r), int(c)) for r, c in zip(*np.nonzero(field[index, :, :, mask_index]))}
        assert occupied == {(cell.row, cell.col) for cell in pair.cells.values()}
        for cell in pair.cells.values():
            total_sparse += 1
            assert field[index, cell.row, cell.col, force_index] == pytest.approx(
                cell.normal_force_n, abs=1e-6
            )
    assert total_sparse == int(dense["active_cell_count"].sum())


def test_the_force_total_is_conserved_frame_by_frame(dense, k1_run):
    force_index = CHANNELS.index("normal_force_N")
    for index, loop_frame in enumerate(k1_run.loop_frames):
        reported = loop_frame.interaction.normal_force_n or 0.0
        assert float(dense["field"][index, ..., force_index].sum()) == pytest.approx(
            reported, abs=1e-6
        )


def test_validity_travels_beside_the_tensor_not_inside_it(dense):
    """Phase 1 wrote 0.0 for undefined; a zero must not read as a measurement."""
    assert dense["curvature_valid"].dtype == bool
    assert dense["direction_valid"].dtype == bool
    assert dense["curvature_valid"].shape == dense["field"].shape[:3]
    assert dense["direction_valid"].shape == dense["field"].shape[:3]
    # Validity is only claimed where a cell exists.
    mask = dense["field"][..., CHANNELS.index("contact_mask")] > 0
    assert not dense["curvature_valid"][~mask].any()
    assert not dense["direction_valid"][~mask].any()


def test_mapped_area_is_the_grid_proxy_and_says_so(dense, k1_run):
    cell_area = k1_run.kernel_config.grid.cell_area_m2
    expected = dense["active_cell_count"].astype(np.float64) * cell_area
    assert np.allclose(dense["mapped_area_m2"], expected)
    metadata = dense_metadata(k1_run, dense)
    assert "not physical compliant contact area" in metadata["mapped_area_semantics"]
    assert metadata["physical_contact_area_m2"] is None


def test_the_metadata_lets_a_reader_interpret_the_tensor_alone(dense, k1_run):
    metadata = dense_metadata(k1_run, dense)
    assert metadata["schema"] == DENSE_SCHEMA
    assert metadata["shape"] == list(dense["field"].shape)
    assert metadata["shape_order"] == ["time", "row", "col", "channel"]
    assert metadata["channels"] == list(CHANNELS)
    assert metadata["units"] == "SI"
    assert metadata["grid"]["rows"] == k1_run.kernel_config.grid.rows
    assert metadata["grid"]["cell_area_m2"] == k1_run.kernel_config.grid.cell_area_m2
    assert metadata["timing"]["output_dt_s"] == k1_run.kernel_config.timing.output_dt_s
    assert metadata["timing"]["frames"] == dense["field"].shape[0]
    assert sum(metadata["segments"].values()) == dense["field"].shape[0]
    assert metadata["conditions_sha256"] == k1_run.conditions_sha256
    assert "does not add spatial precision" in metadata["densification_note"]


def test_a_written_file_reloads_without_pickle(tmp_path, k1_run):
    path = tmp_path / "dense_contact_field.npz"
    metadata = write_dense_field(path, k1_run)
    assert path.is_file()
    data = read_dense_field(path)
    assert data["field"].shape == tuple(metadata["shape"])
    assert data["field"].dtype == np.float32
    assert [str(value) for value in data["channels"]] == list(CHANNELS)
    assert list(data["control_step"]) == [f.control_step for f in k1_run.loop_frames]


def test_a_corrupted_channel_layout_is_refused(tmp_path, k1_run):
    # A consumer must not silently read a tensor whose columns moved.
    arrays = build_dense_field(k1_run)
    arrays["channels"] = np.asarray(CHANNELS[::-1])
    path = tmp_path / "scrambled.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="Unexpected channel layout"):
        read_dense_field(path)


def test_a_truncated_time_axis_is_refused(tmp_path, k1_run):
    arrays = build_dense_field(k1_run)
    arrays["time_s"] = arrays["time_s"][:-1]
    path = tmp_path / "truncated.npz"
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="time_s has"):
        read_dense_field(path)


def test_force_mode_densifies_the_same_way(k2_run):
    # The tensor is mode independent: only the segment names differ.
    arrays = build_dense_field(k2_run)
    assert arrays["field"].shape[0] == len(k2_run.loop_frames)
    phases = [str(value) for value in arrays["phase"]]
    assert "regulate_force" in phases
    assert phases.count("pre_contact") == 30
    assert phases.count("released") == 30
    force_index = CHANNELS.index("normal_force_N")
    released = arrays["field"][np.asarray(phases) == "released"]
    assert released[..., force_index].sum() == 0.0
