"""Dense contact field: the same data as the sparse cells, laid out as a tensor.

The live pipeline writes sparse rows — only occupied cells. This module writes
the identical information as a fixed-shape ``[T, H, W, C]`` tensor, which is
what a visualiser, a batch job, a learning pipeline or an actuator array wants
to read. Nothing here is a new measurement: densification changes the storage
layout and must not change a single physical semantic.

Two properties are load-bearing, and both were learned in Phase 1.

**The time axis comes from the control stream, not from the cells.** Sparse
rows exist only while contact exists, so a timeline derived from them silently
drops ``pre_contact`` and ``released``. Those frames are exactly where the
evidence that contact force returns to zero lives, so the axis is built from
every control step the run produced and cells are filled into it.

**Mapped area stays a grid proxy.** ``active_cells * cell_area`` is a
discretisation artefact that changes with output resolution; it is not
physical compliant contact area. The metadata says so next to the number.

One deliberate difference from Phase 1. Phase 1 wrote ``0.0`` into the
curvature and direction channels when the source row had no valid value,
which makes "undefined" indistinguishable from "flat" and "no motion". The
13-channel layout is kept for compatibility, but the validity flags travel
beside the tensor as their own ``[T, H, W]`` boolean arrays, so a consumer can
tell the two apart instead of inferring a zero means measured.
"""

from __future__ import annotations

import numpy as np

DENSE_SCHEMA = "dense-contact-field/v1"

#: Channel order of the last tensor axis. Identical to the Phase 1 Step 6.4
#: layout so an existing consumer reads this file unchanged. Consumers should
#: read the ``channels`` array rather than hardcoding indices.
CHANNELS = (
    "contact_mask",
    "normal_force_N",
    "normal_impulse_Ns",
    "tangent_force_x_N",
    "tangent_force_y_N",
    "tangent_force_z_N",
    "normal_velocity_mps",
    "tangent_speed_mps",
    "direction_u",
    "direction_v",
    "curvature_1_per_m",
    "curvature_2_per_m",
    "patch_id",
)

#: Unit and meaning of every channel. Written into the metadata so a reader
#: never has to guess what a number is, and never has to read this source.
CHANNEL_SEMANTICS: dict[str, dict[str, str]] = {
    "contact_mask": {
        "unit": "boolean", "role": "occupancy",
        "description": "1 where the cell was reported in contact this step, else 0",
    },
    "normal_force_N": {
        "unit": "N", "role": "additive",
        "description": "cell normal force; summing the grid gives the frame total",
    },
    "normal_impulse_Ns": {
        "unit": "N*s", "role": "additive",
        "description": "cell normal impulse over the output interval",
    },
    "tangent_force_x_N": {"unit": "N", "role": "additive",
                          "description": "world X component of cell tangential force"},
    "tangent_force_y_N": {"unit": "N", "role": "additive",
                          "description": "world Y component of cell tangential force"},
    "tangent_force_z_N": {"unit": "N", "role": "additive",
                          "description": "world Z component of cell tangential force"},
    "normal_velocity_mps": {
        "unit": "m/s", "role": "intensive",
        "description": "mean normal relative velocity of the points in the cell; "
                       "do not sum across cells",
    },
    "tangent_speed_mps": {
        "unit": "m/s", "role": "intensive",
        "description": "mean tangential speed of the points in the cell; "
                       "do not sum across cells",
    },
    "direction_u": {
        "unit": "dimensionless", "role": "intensive",
        "description": "surface-local U component of motion direction; "
                       "meaningful only where direction_valid is true",
    },
    "direction_v": {
        "unit": "dimensionless", "role": "intensive",
        "description": "surface-local V component of motion direction; "
                       "meaningful only where direction_valid is true",
    },
    "curvature_1_per_m": {
        "unit": "1/m", "role": "intensive",
        "description": "first principal curvature; meaningful only where "
                       "curvature_valid is true. Analytically 0 on the flat top face",
    },
    "curvature_2_per_m": {
        "unit": "1/m", "role": "intensive",
        "description": "second principal curvature; same validity rule",
    },
    "patch_id": {
        "unit": "identifier", "role": "label",
        "description": "stable patch identity, 0 where no contact. Never sum or "
                       "interpolate this channel",
    },
}


def _cell_values(cell) -> tuple[float, ...]:
    """One cell as the 13 channel values, in ``CHANNELS`` order."""
    direction_u, direction_v = cell.motion_direction_uv
    curvature_1 = cell.curvature_1_per_m if cell.curvature_valid else 0.0
    curvature_2 = cell.curvature_2_per_m if cell.curvature_valid else 0.0
    if not cell.direction_valid:
        direction_u = direction_v = 0.0
    return (
        1.0,
        cell.normal_force_n,
        cell.normal_impulse_ns,
        *cell.tangent_force_world_n,
        cell.mean_normal_relative_velocity_mps,
        cell.mean_tangent_speed_mps,
        direction_u,
        direction_v,
        curvature_1,
        curvature_2,
        float(cell.patch_id),
    )


def build_dense_field(run) -> dict:
    """Arrays for one run. Frame count equals the control-step count exactly."""
    from contact_kernel import RigidContactKernel

    config = run.kernel_config
    rows, cols = config.grid.rows, config.grid.cols
    key = RigidContactKernel.pair_key(config.probes[0].id, config.surface.id)
    loop_frames = run.loop_frames

    field = np.zeros((len(loop_frames), rows, cols, len(CHANNELS)), dtype=np.float32)
    curvature_valid = np.zeros((len(loop_frames), rows, cols), dtype=bool)
    direction_valid = np.zeros((len(loop_frames), rows, cols), dtype=bool)
    times = np.zeros(len(loop_frames), dtype=np.float64)
    steps = np.zeros(len(loop_frames), dtype=np.int64)
    phases = np.empty(len(loop_frames), dtype="U32")

    for index, loop_frame in enumerate(loop_frames):
        pair = loop_frame.world.kernel_frame.pairs[key]
        times[index] = loop_frame.time_s
        steps[index] = loop_frame.control_step
        phases[index] = loop_frame.phase
        seen = set()
        for cell in pair.cells.values():
            if not (0 <= cell.row < rows and 0 <= cell.col < cols):
                raise ValueError(
                    f"Cell ({cell.row}, {cell.col}) is outside the declared "
                    f"{rows}x{cols} grid at control step {loop_frame.control_step}"
                )
            position = (cell.row, cell.col)
            if position in seen:
                # The kernel aggregates per cell before this point, so a repeat
                # would mean two cell records claiming the same grid position.
                # Accumulating silently would hide that; Phase 1's += could not
                # tell the two cases apart.
                raise ValueError(
                    f"Cell {position} appears twice at control step "
                    f"{loop_frame.control_step}; cells are aggregated upstream "
                    f"and must be unique per step"
                )
            seen.add(position)
            field[index, cell.row, cell.col, :] = _cell_values(cell)
            curvature_valid[index, cell.row, cell.col] = bool(cell.curvature_valid)
            direction_valid[index, cell.row, cell.col] = bool(cell.direction_valid)

    mask = field[..., CHANNELS.index("contact_mask")]
    active_cells = mask.sum(axis=(1, 2)).astype(np.int32)
    normal_force = field[..., CHANNELS.index("normal_force_N")].sum(axis=(1, 2)).astype(np.float64)
    cell_area = float(config.grid.cell_area_m2)
    return {
        "field": field,
        "time_s": times,
        "control_step": steps,
        "phase": phases,
        "curvature_valid": curvature_valid,
        "direction_valid": direction_valid,
        "active_cell_count": active_cells,
        "normal_force_sum_N": normal_force,
        "mapped_area_m2": (active_cells.astype(np.float64) * cell_area),
        "cell_area_m2": np.float64(cell_area),
        "channels": np.asarray(CHANNELS),
        "rows": np.int32(rows),
        "cols": np.int32(cols),
    }


def dense_metadata(run, arrays: dict) -> dict:
    """Everything a reader needs to interpret the tensor without this source."""
    config = run.kernel_config
    timing = config.timing
    phases = [str(value) for value in arrays["phase"]]
    return {
        "schema": DENSE_SCHEMA,
        "run_id": run.run_id,
        "scenario_id": run.scenario_id,
        "controller_mode": run.controller_mode,
        "conditions_sha256": run.conditions_sha256,
        "shape": list(arrays["field"].shape),
        "shape_order": ["time", "row", "col", "channel"],
        "dtype": str(arrays["field"].dtype),
        "channels": list(CHANNELS),
        "channel_semantics": CHANNEL_SEMANTICS,
        "grid": {
            "rows": config.grid.rows,
            "cols": config.grid.cols,
            "cell_size_m": config.grid.cell_size_m,
            "cell_area_m2": config.grid.cell_area_m2,
            "origin_local_m": list(config.grid.origin_local_m),
            "row_axis": "y",
            "col_axis": "x",
            "edge_policy": config.grid.edge_policy,
        },
        "timing": {
            "output_dt_s": timing.output_dt_s,
            "control_dt_s": timing.control_dt_s,
            "physics_dt_s": timing.physics_dt_s,
            "substeps_per_control": timing.substeps_per_control,
            "frames": int(arrays["field"].shape[0]),
            "start_time_s": float(arrays["time_s"][0]) if len(arrays["time_s"]) else None,
            "end_time_s": float(arrays["time_s"][-1]) if len(arrays["time_s"]) else None,
            "timestamp": "interval_end",
        },
        "units": "SI",
        "time_axis_source": "every control step of the run, including frames with no contact",
        "segments": {phase: phases.count(phase) for phase in dict.fromkeys(phases)},
        "mapped_area_semantics": (
            "active_cell_count * cell_area_m2. A grid occupancy proxy that "
            "changes with output resolution, not physical compliant contact area"
        ),
        "physical_contact_area_m2": None,
        "validity_arrays": {
            "curvature_valid": (
                "boolean [T, H, W]; where false the curvature channels carry 0 "
                "as a placeholder and carry no measurement"
            ),
            "direction_valid": (
                "boolean [T, H, W]; where false the direction channels carry 0 "
                "as a placeholder and carry no measurement"
            ),
        },
        "densification_note": (
            "identical information to the sparse cell records, re-laid out. "
            "Densification does not add spatial precision and does not turn a "
            "grid proxy area into a physical one"
        ),
    }


def write_dense_field(path, run) -> dict:
    """Write ``dense_contact_field.npz``. Returns the metadata block."""
    arrays = build_dense_field(run)
    # allow_pickle stays off on read, so every array must be a plain numeric or
    # unicode type; savez_compressed keeps the file reviewable in size.
    np.savez_compressed(path, **arrays)
    return dense_metadata(run, arrays)


def read_dense_field(path) -> dict:
    """Load the tensor and its side arrays. Never unpickles."""
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    channels = [str(value) for value in data["channels"]]
    if tuple(channels) != CHANNELS:
        raise ValueError(f"Unexpected channel layout: {channels}")
    field = data["field"]
    if field.ndim != 4 or field.shape[3] != len(CHANNELS):
        raise ValueError(f"Expected a [T, H, W, {len(CHANNELS)}] tensor, got {field.shape}")
    for name in ("time_s", "control_step", "phase", "active_cell_count"):
        if len(data[name]) != field.shape[0]:
            raise ValueError(f"{name} has {len(data[name])} entries for {field.shape[0]} frames")
    return data
