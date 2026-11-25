"""P2.2.1 box-probe demonstration run.

    cd Phase_2
    ./.venv/Scripts/python.exe -m mechanics.run_k3

This is a *verification entry point*, not a milestone run. It drives one
rotation-locked flat-bottom box onto the Step 6 top face and reports the three
things P2.2.1 claims, each next to the thing it must not be confused with:

  1. a flat contact activates more than one cell          (multi-cell)
  2. the cell count scales with grid resolution           (proxy)
  3. the geometric footprint area does not                (physical)

Claims 2 and 3 are printed together on purpose. "Contact point count tracks
resolution" is the sentence this run exists to make falsifiable, and it is
only meaningful beside a quantity that stays put.

What this run does NOT establish: it is a single scenario on the CPU kernel
with one probe. Multi-probe independent streams are P2.2.2 and the full
32/64/128 regression with the deferred thresholds is P2.2.3. No acceptance
gate is emitted here and no milestone field is written.

Flags:
    --grid N          output grid rows/cols (repeatable); default 32 64 128
    --probe-z Z       starting height; raise it for the counter-example
    --size X Y Z      box extents in metres
"""

from __future__ import annotations

import argparse
import json

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

PROBE_ID = "/World/BoxProbe"
SURFACE_ID = "/World/Cube"
DT = 1.0 / 60.0


def build(rows: int, probe_z: float, size: tuple[float, float, float]) -> RigidContactKernel:
    config = KernelConfig(
        surface=SurfaceSpec(id=SURFACE_ID, center_world_m=(0, 0, 0.25), size_m=(0.5, 0.5, 0.5)),
        probes=(ProbeSpec(id=PROBE_ID, shape="box", mass_kg=0.08,
                          initial_position_world_m=(0.0, 0.0, probe_z), size_m=size),),
        timing=TimingSpec(physics_dt_s=DT, control_dt_s=DT, output_dt_s=DT,
                          substeps_per_control=1),
        grid=GridSpec(rows=rows, cols=rows, cell_size_m=0.5 / rows,
                      origin_local_m=(-0.25, -0.25, 0.25)),
        solver=SolverSpec(),
        sampling=SamplingSpec(),
        drive=DriveSpec(),
    )
    return RigidContactKernel(config)


def press(kernel: RigidContactKernel, steps: int = 40):
    key = RigidContactKernel.pair_key(PROBE_ID, SURFACE_ID)
    pair = None
    for index in range(steps):
        phase = "approach" if index < 10 else "press"
        pair = kernel.step({PROBE_ID: (0.0, 0.0, -0.3)}, phase=phase).pairs[key]
    return pair


def multi_cell_claim(rows) -> bool:
    """True only if some grid saw contact and every contacting grid saw >1 cell."""
    contacting = [row for row in rows if row["contact_present"]]
    return bool(contacting) and all(row["active_cells"] > 1 for row in contacting)


def main() -> int:
    parser = argparse.ArgumentParser(description="P2.2.1 box probe verification run")
    parser.add_argument("--grid", type=int, nargs="*", default=[32, 64, 128])
    parser.add_argument("--probe-z", type=float, default=0.60)
    parser.add_argument("--size", type=float, nargs=3, default=[0.10, 0.10, 0.05])
    args = parser.parse_args()

    size = (args.size[0], args.size[1], args.size[2])
    rows_out = []
    for rows in args.grid:
        pair = press(build(rows, args.probe_z, size))
        area = pair.area_layers
        rows_out.append({
            "grid": f"{rows}x{rows}",
            "contact_present": pair.contact_present,
            "active_cells": area.swept_cell_count if not pair.contact_present else len(
                pair.cells
            ),
            "point_occupancy_m2": area.point_occupancy_m2,
            "geometric_footprint_m2": area.geometric_footprint_m2,
            "footprint_radius_m": area.footprint_radius_m,
            "normal_force_n": pair.normal_force_n,
            "force_conserved": abs(pair.conservation.point_force_n - pair.conservation.cell_force_n) < 1e-9,
        })

    report = {
        "run": "p2.2.1-box-probe",
        "note": "verification run, not a milestone; no acceptance gate is emitted",
        "probe": {"shape": "box", "size_m": list(size), "start_z_m": args.probe_z},
        "rows": rows_out,
        "claims": {
            # any() first: without it a run with no contact at all reports
            # multi_cell = true vacuously (found 2026-11-04 with --probe-z 1.2).
            "multi_cell": multi_cell_claim(rows_out),
            "cell_count_scales_with_resolution": [row["active_cells"] for row in rows_out],
            "geometric_footprint_is_resolution_independent": len(
                {round(row["geometric_footprint_m2"], 12) for row in rows_out}
            ) == 1,
            "force_conserved_everywhere": all(row["force_conserved"] for row in rows_out),
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
