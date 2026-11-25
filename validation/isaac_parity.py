"""B3: score the Phase 2 kernel against Isaac Sim on the same closed loop.

    python -m validation.isaac_parity --isaac-dir experiments/runs/isaac-b2

Inputs: the kernel's own run records (``experiments/runs/k1-single-probe-cpu``,
``k2-force-cpu``) and the B2 records ``out_<spec>_<variant>.jsonl``.

Thresholds are frozen below. They were proposed on 2026-11-25 and accepted by the
user as the Phase 2 close-out criteria; changing one is a scope change that needs
the user's approval and a line in the change log, never a fix to make a run pass.
A gate either passes, fails, or is reported only. A failed gate is a result.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Phase 2 close-out gates (user-accepted 2026-11-25). Do not edit to pass a run.
THRESHOLDS = {
    "contact_onset_max_steps": 1,
    "contact_point_max_m": 1e-4,
    "contact_normal_max_deg": 1.0,
    "force_band_fraction": 0.05,
    "force_hold_frames": 30,
    "momentum_balance_max_ns": 1e-6,
}
TARGET_FORCE_N = 1.0
MASS_KG = 0.05
SURFACE_Z = 0.5
RUNS = {"k1_single_probe_cpu": "k1-single-probe-cpu", "k2_force_cpu": "k2-force-cpu"}
VARIANTS = ("aligned", "phase1_defaults")


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_kernel(run_dir: Path) -> list[dict]:
    """Per-frame kernel record in the same shape as the Isaac lines."""
    frames = []
    for row in _jsonl(run_dir / "kinematics.jsonl"):
        impulse = sum(s["normal_impulse_ns"] for s in row["substeps"])
        frames.append({
            "control_step": row["control_step"], "time_s": row["time_s"], "phase": row["phase"],
            "command_velocity_world_mps": row["command_velocity_world_mps"],
            "probe_position_world_m": row["probe_position_world_m"],
            "probe_velocity_world_mps": row["probe_velocity_world_mps"],
            "normal_impulse_ns": impulse,
            "normal_force_n": impulse / row["substeps"][0]["physics_dt_s"] / len(row["substeps"]),
            "contact_points": [],
        })
    by_step = {f["control_step"]: f for f in frames}
    import csv
    with open(run_dir / "points.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame = by_step[int(row["control_step"])]
            frame["contact_points"].append({
                "position_world_m": [float(row[f"point_world_{a}_m"]) for a in "xyz"],
                "normal_world": [float(row[f"normal_world_{a}"]) for a in "xyz"],
            })
    for frame in frames:
        frame["contact_present"] = bool(frame["contact_points"])
    return frames


def load_isaac(path: Path) -> tuple[list[dict], dict]:
    lines = _jsonl(path)
    summary = lines[-1]["summary"]
    return lines[:-1], summary


def _onset(frames) -> int | None:
    return next((f["control_step"] for f in frames if f["contact_present"]), None)


def _centroid(points) -> list[float]:
    return [sum(p["position_world_m"][i] for p in points) / len(points) for i in range(3)]


def _angle_deg(normal) -> float:
    z = abs(normal[2]) / math.sqrt(sum(c * c for c in normal))
    return math.degrees(math.acos(min(1.0, z)))


def _momentum_residual(frames) -> float:
    """max |m (v_after_n - v_cmd_n) - J_n| over contact frames: the probe's own momentum
    balance under the velocity-overwrite drive with gravity cancelled."""
    worst = 0.0
    for f in frames:
        if not f["contact_present"]:
            continue
        dv = f["probe_velocity_world_mps"][2] - f["command_velocity_world_mps"][2]
        worst = max(worst, abs(MASS_KG * dv - f["normal_impulse_ns"]))
    return worst


def _force_hold(frames, phase: str) -> dict:
    band = THRESHOLDS["force_band_fraction"] * TARGET_FORCE_N
    reg = [f for f in frames if f["phase"] == phase]
    if not reg:
        return {"frames": 0, "entered": False, "held_last_n": False}
    inside = [abs(f["normal_force_n"] - TARGET_FORCE_N) <= band for f in reg]
    first = next((i for i, ok in enumerate(inside) if ok), None)
    tail = inside[-THRESHOLDS["force_hold_frames"]:]
    return {
        "frames": len(reg), "entered": first is not None,
        "first_entry_frame": first,
        "held_last_n": len(tail) == THRESHOLDS["force_hold_frames"] and all(tail),
        "final_force_n": reg[-1]["normal_force_n"],
    }


def compare(kernel: list[dict], isaac: list[dict], spec: str) -> dict:
    gates, reported = {}, {}
    k_on, i_on = _onset(kernel), _onset(isaac)
    gates["contact_onset"] = {
        "kernel_step": k_on, "isaac_step": i_on,
        "delta_steps": None if None in (k_on, i_on) else i_on - k_on,
        "pass": None not in (k_on, i_on) and abs(i_on - k_on) <= THRESHOLDS["contact_onset_max_steps"],
    }
    k_by = {f["control_step"]: f for f in kernel}
    point_err, normal_err, common = [], [], 0
    for f in isaac:
        k = k_by.get(f["control_step"])
        if not (k and f["contact_present"] and k["contact_present"] and f["contact_points"]):
            continue
        common += 1
        ci, ck = _centroid(f["contact_points"]), _centroid(k["contact_points"])
        point_err.append(math.dist(ci, ck))
        normal_err.append(max(_angle_deg(p["normal_world"]) for p in f["contact_points"]))
    gates["contact_point"] = {
        "common_contact_frames": common,
        "max_m": max(point_err) if point_err else None,
        "pass": bool(point_err) and max(point_err) <= THRESHOLDS["contact_point_max_m"],
    }
    gates["contact_normal"] = {
        "max_deg": max(normal_err) if normal_err else None,
        "pass": bool(normal_err) and max(normal_err) <= THRESHOLDS["contact_normal_max_deg"],
    }
    gates["momentum_balance"] = {
        "kernel_max_ns": _momentum_residual(kernel), "isaac_max_ns": _momentum_residual(isaac),
    }
    gates["momentum_balance"]["pass"] = (
        gates["momentum_balance"]["kernel_max_ns"] <= THRESHOLDS["momentum_balance_max_ns"]
        and gates["momentum_balance"]["isaac_max_ns"] <= THRESHOLDS["momentum_balance_max_ns"])
    if spec == "k2_force_cpu":
        k_hold, i_hold = _force_hold(kernel, "regulate_force"), _force_hold(isaac, "regulate_force")
        gates["force_steady_state"] = {
            "kernel": k_hold, "isaac": i_hold,
            "pass": k_hold["entered"] and k_hold["held_last_n"] and i_hold["entered"] and i_hold["held_last_n"],
        }
    # Reported only: force transient, phase sequence, frame count.
    diffs = [abs(f["normal_force_n"] - k_by[f["control_step"]]["normal_force_n"])
             for f in isaac if f["control_step"] in k_by]
    reported["force_abs_diff_n"] = {
        "max": max(diffs) if diffs else None,
        "rms": math.sqrt(sum(d * d for d in diffs) / len(diffs)) if diffs else None,
    }
    reported["frames"] = {"kernel": len(kernel), "isaac": len(isaac)}
    reported["phase_sequence_equal"] = _phases(kernel) == _phases(isaac)
    reported["phases"] = {"kernel": _phases(kernel), "isaac": _phases(isaac)}
    return {"gates": gates, "reported_only": reported,
            "all_gates_pass": all(g["pass"] for g in gates.values())}


def _phases(frames) -> list[str]:
    out = []
    for f in frames:
        if not out or out[-1] != f["phase"]:
            out.append(f["phase"])
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-dir", type=Path, default=ROOT / "experiments/runs/isaac-b2")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = {"schema": "isaac-parity-report/v1", "thresholds": THRESHOLDS, "runs": {}}
    for spec, run_dir in RUNS.items():
        kernel = load_kernel(ROOT / "experiments/runs" / run_dir)
        for variant in VARIANTS:
            path = args.isaac_dir / f"out_{spec}_{variant}.jsonl"
            if not path.exists():
                report["runs"][f"{spec}/{variant}"] = {"missing": str(path)}
                continue
            isaac, summary = load_isaac(path)
            result = compare(kernel, isaac, spec)
            result["isaac_summary"] = {k: v for k, v in summary.items() if k != "errors"}
            result["isaac_errors"] = summary.get("errors", [])
            report["runs"][f"{spec}/{variant}"] = result
    output = args.output or (args.isaac_dir / "parity_report.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, result in report["runs"].items():
        if "missing" in result:
            print(f"{name}: MISSING")
            continue
        print(f"\n[{name}] all_gates_pass={result['all_gates_pass']}")
        for gate, value in result["gates"].items():
            detail = {k: v for k, v in value.items() if k != "pass"}
            print(f"  {gate:20s} {'PASS' if value['pass'] else 'FAIL'}  {json.dumps(detail)[:220]}")
        print(f"  reported: {json.dumps(result['reported_only'])[:300]}")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
