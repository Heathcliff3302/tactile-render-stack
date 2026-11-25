"""B2: run a K1 / K2 spec closed-loop in Isaac Sim, driven by Phase 2's own controller.

Run on the GPU machine with the Isaac Sim Python, with this repository on disk:

    OMNI_KIT_ACCEPT_EULA=YES python scenarios/isaac_replay.py \\
        --spec scenarios/k0/k1_single_probe_cpu.json --variant aligned --output isaac_k1_aligned.jsonl

What is shared and what is swapped. Layer 1's segment machine
(``controller_from_spec``) is imported unchanged, with the same one-step feedback
delay and the same 0.05 N sampling threshold the kernel applies. Only the physics
step is replaced: PhysX advances the probe instead of ``RigidContactKernel``.
So any difference in the record comes from the physics backend, not the controller.

Drive mapping. The kernel overwrites the probe's linear velocity once per control
step and cancels gravity at every substep (``per_substep_feedforward``), so the
probe's net free acceleration is zero. The Isaac equivalent is to overwrite the
velocity the same way and disable gravity on the probe. Rotation is locked in the
kernel; here every rotation axis is locked if the schema accepts it, and angular
velocity is zeroed each step in any case.

Variants:

``phase1_defaults``
    the scene as Phase 1's step6_3.py built it: Isaac's default materials
    (probe/cube static 0.2, dynamic 1.0, restitution 0) and automatic contact offsets.
``aligned``
    the kernel's declared physics: zero friction, zero restitution, rest offset 0,
    a 1 mm contact offset. This isolates solver differences from parameter differences.

Output: one JSON line per control step, then one ``summary`` line. Written before
the app closes, because ``SimulationApp.close()`` may end the process.

Two conventions to read the record by (found in B3, 2026-11-25):

* PhysX generates contacts before integrating, so ``contact_points`` describe the
  start of the step; the kernel reports the end. During sliding they differ by
  one step of travel.
* The contact report's ``impulse`` carries the normal impulse only, not friction.
  ``tangent_force_world_n`` is therefore not a friction measurement; treat it as
  unavailable. Friction shows up only indirectly, as probe lag in the slide.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DT = 1.0 / 60.0
CUBE_CENTER = (0.0, 0.0, 0.25)
CUBE_SIZE = 0.50
TIP_RADIUS = 0.12
TIP_MASS = 0.05
Z_HOME = 1.20
PROBE_PATH = "/World/Fingertip"
CUBE_PATH = "/World/Cube"
SURFACE_NORMAL = (0.0, 0.0, 1.0)
VARIANTS = ("phase1_defaults", "aligned")
SCHEMA = "isaac-closed-loop-record/v1"


def _vec(value) -> list[float]:
    return [float(value[0]), float(value[1]), float(value[2])]


def _configure(stage, variant: str) -> dict:
    """Apply drive and variant settings; return what was actually set, for the record."""
    from pxr import PhysxSchema, Sdf, UsdPhysics, UsdShade

    probe = stage.GetPrimAtPath(PROBE_PATH)
    cube = stage.GetPrimAtPath(CUBE_PATH)
    applied = {"variant": variant}

    body = PhysxSchema.PhysxRigidBodyAPI.Apply(probe)
    body.CreateDisableGravityAttr().Set(True)
    applied["probe_disable_gravity"] = True
    locked = probe.CreateAttribute("physxRigidBody:lockedRotAxis", Sdf.ValueTypeNames.Int)
    applied["probe_locked_rot_axis_set"] = bool(locked.Set(7))

    report = PhysxSchema.PhysxContactReportAPI.Apply(probe)
    report.CreateThresholdAttr().Set(0.0)

    if variant == "aligned":
        material_path = "/World/Physics_Materials/kernel_aligned"
        material = UsdShade.Material.Define(stage, material_path)
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr().Set(0.0)
        api.CreateDynamicFrictionAttr().Set(0.0)
        api.CreateRestitutionAttr().Set(0.0)
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx_material.CreateFrictionCombineModeAttr().Set("min")
        physx_material.CreateRestitutionCombineModeAttr().Set("min")
        for prim in (probe, cube):
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr().Set(0.001)
            collision.CreateRestOffsetAttr().Set(0.0)
        applied.update({
            "friction_static": 0.0, "friction_dynamic": 0.0, "restitution": 0.0,
            "combine_modes": "min", "contact_offset_m": 0.001, "rest_offset_m": 0.0,
        })
    else:
        applied["materials"] = "Isaac defaults from DynamicSphere / FixedCuboid (see B1 capture)"
    return applied


def _read_contacts(dt: float) -> dict:
    """Probe-cube contacts PhysX reported for the last step."""
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools

    headers, data = get_physx_simulation_interface().get_contact_report()
    points, impulse_n, impulse_vec = [], 0.0, [0.0, 0.0, 0.0]
    for header in headers:
        paths = {str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                 str(PhysicsSchemaTools.intToSdfPath(header.actor1))}
        if paths != {PROBE_PATH, CUBE_PATH}:
            continue
        for item in data[header.contact_data_offset: header.contact_data_offset + header.num_contact_data]:
            impulse = _vec(item.impulse)
            normal = _vec(item.normal)
            # Sign-free: the normal-direction share of the impulse the pair exchanged.
            j_n = abs(sum(a * b for a, b in zip(impulse, SURFACE_NORMAL)))
            impulse_n += j_n
            impulse_vec = [a + b for a, b in zip(impulse_vec, impulse)]
            points.append({
                "position_world_m": _vec(item.position),
                "normal_world": normal,
                "separation_m": float(item.separation),
                "impulse_ns": impulse,
            })
    return {
        "points": points,
        "normal_impulse_ns": impulse_n,
        "normal_force_n": impulse_n / dt,
        "tangent_force_world_n": [impulse_vec[0] / dt, impulse_vec[1] / dt, 0.0],
    }


def run(spec_path: Path, variant: str, output: Path) -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    lines: list[dict] = []
    summary = {"schema": SCHEMA, "spec": str(spec_path.name), "variant": variant, "errors": []}
    try:
        import numpy as np
        import omni.usd
        from importlib.metadata import version
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicSphere, FixedCuboid

        from contact_kernel.spec import SamplingSpec
        from tactile_contract.experiment_spec import ExperimentSpec
        from tactile_contract.pipeline import LoopFeedback
        from virtual_reality.world import controller_from_spec, step_budget_from_spec

        spec = ExperimentSpec.from_dict(json.loads(spec_path.read_text(encoding="utf-8")))
        sampling = spec.to_dict().get("sampling", {})
        threshold = float(sampling.get("force_threshold_n", SamplingSpec().force_threshold_n))
        controller = controller_from_spec(spec)
        budget = step_budget_from_spec(spec)
        summary.update({"isaac_sim_version": version("isaacsim"), "step_budget": budget,
                        "force_threshold_n": threshold})

        world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
        world.scene.add_default_ground_plane()
        world.scene.add(FixedCuboid(prim_path=CUBE_PATH, name="cube", position=np.array(CUBE_CENTER),
                                    scale=np.array([CUBE_SIZE] * 3), size=1.0))
        probe = world.scene.add(DynamicSphere(prim_path=PROBE_PATH, name="fingertip",
                                              position=np.array([0.0, 0.0, Z_HOME]),
                                              radius=TIP_RADIUS, mass=TIP_MASS))
        stage = omni.usd.get_context().get_stage()
        summary["applied"] = _configure(stage, variant)
        world.reset()
        probe.set_linear_velocity(np.zeros(3))
        probe.set_angular_velocity(np.zeros(3))

        controller.reset()
        feedback = LoopFeedback()
        for step in range(1, budget + 1):
            phase = controller.phase
            command = controller.command(feedback)
            probe.set_linear_velocity(np.array(command, dtype=float))
            probe.set_angular_velocity(np.zeros(3))
            world.step(render=False)
            time_s = step * DT

            position = _vec(probe.get_world_pose()[0])
            velocity = _vec(probe.get_linear_velocity())
            angular = _vec(probe.get_angular_velocity())
            contacts = _read_contacts(DT)
            solver_contact = bool(contacts["points"]) and contacts["normal_impulse_ns"] > 0.0
            reported = solver_contact and contacts["normal_force_n"] >= threshold
            normal_speed = sum(a * b for a, b in zip(velocity, SURFACE_NORMAL))
            tangent = [v - normal_speed * n for v, n in zip(velocity, SURFACE_NORMAL)]
            tangent_speed = math.sqrt(sum(v * v for v in tangent))

            controller.observe(time_s, reported, contacts["normal_force_n"] if reported else 0.0)
            reported_force = contacts["normal_force_n"] if reported else 0.0
            feedback = LoopFeedback(control_step=step, contact_present=reported,
                                    normal_force_n=reported_force, tangent_speed_mps=tangent_speed,
                                    actuator_measured_force_n=reported_force)
            lines.append({
                "control_step": step, "time_s": time_s, "phase": phase,
                "phase_after_observation": controller.phase,
                "command_velocity_world_mps": list(command),
                "probe_position_world_m": position, "probe_velocity_world_mps": velocity,
                "probe_angular_velocity_rps": angular,
                "gap_m": position[2] - TIP_RADIUS - (CUBE_CENTER[2] + CUBE_SIZE / 2),
                "solver_contact": solver_contact, "contact_present": reported,
                "normal_force_n": contacts["normal_force_n"], "normal_impulse_ns": contacts["normal_impulse_ns"],
                "tangent_force_world_n": contacts["tangent_force_world_n"],
                "tangent_speed_mps": tangent_speed, "contact_points": contacts["points"],
            })
            if controller.done:
                break
        summary.update({"frames": len(lines), "controller_done": bool(controller.done),
                        "abort_reason": controller.abort_reason,
                        "phases": sorted({line["phase"] for line in lines})})
    except Exception:
        summary["errors"].append(traceback.format_exc())

    with open(output, "w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")
        handle.write(json.dumps({"summary": summary}, ensure_ascii=False) + "\n")
    print("REPLAY_SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "applied"},
                                         ensure_ascii=False), flush=True)
    status = 1 if summary["errors"] else 0
    try:
        app.close()
    except Exception:
        pass
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return run(args.spec, args.variant, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
