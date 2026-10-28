"""Capture the runtime settings PhysX actually used, on a machine with Isaac Sim.

Run this on the GPU box. It is the only step of K2 that needs one, and it
needs it for minutes, not hours: it rebuilds the Step 6 scene, resets the
world, steps a handful of times, reads the effective values off the stage, and
exits. It does not run the 472-frame experiment.

Why this script exists at all. Phase 1's ``step6_3.py`` builds its scene from
``DynamicSphere`` / ``FixedCuboid`` / ``add_default_ground_plane()`` and never
sets friction, restitution, damping or solver iteration counts. So eighteen
fields of the K0 inventory are marked ``unresolved``: the values exist, but
only inside the engine. Until they are read, Phase 2 cannot claim numeric
equivalence with Isaac Sim — only behaviour and conservation gates. That is
the single thing standing between K2-A and K2.

What it will NOT do. It never guesses. For each field it records where the
number came from:

``authored``
    the attribute is authored on the prim; this is what the scene set.
``schema_fallback``
    the attribute exists but is unauthored, so this is the USD schema's
    fallback. PhysX may still use its own internal default, so a fallback is
    weaker evidence than an authored value and is labelled as such.
``absent``
    no such attribute on this prim. Recorded as null, never as zero.
``error``
    the read raised. The exception text is kept.

A field that cannot be read stays ``unresolved`` in the output. Half a capture
honestly labelled is useful; a capture with plausible numbers of unknown
origin is worse than none, because the parity claim would rest on it.

Usage on the rented machine:

    # from the Isaac Sim installation directory
    ./python.sh /path/to/capture_isaac_runtime.py --output isaac_runtime.json

    # Isaac Sim 5.x may instead expose:
    #   ~/.local/share/ov/pkg/isaac-sim-*/python.sh
    # or, inside the official container, simply:
    #   /isaac-sim/python.sh

Then send back the single JSON file it writes. Nothing else from that machine
is needed.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from datetime import datetime, timezone

# Geometry and timing copied verbatim from Phase 1 scripts/step6_3.py so the
# scene whose defaults we read is the scene that produced the baseline. These
# must not drift; the capture records them so a reader can check.
DT = 1.0 / 60.0
CUBE_CENTER = (0.0, 0.0, 0.25)
CUBE_SIZE = 0.50
TIP_RADIUS = 0.12
TIP_MASS = 0.05
PROBE_PATH = "/World/Fingertip"
CUBE_PATH = "/World/Cube"
Z_HOME = 1.20

#: The eighteen fields K0 leaves unresolved, and where each is expected to
#: live. Keeping the expectation explicit means a field we fail to locate is
#: reported as not found, rather than quietly omitted from the output.
EXPECTED_FIELDS = (
    "gravity_mps2",
    "static_friction",
    "dynamic_friction",
    "restitution",
    "friction_combine_mode",
    "restitution_combine_mode",
    "linear_damping_per_s",
    "angular_damping_per_s",
    "initial_linear_velocity_mps",
    "initial_orientation_wxyz",
    "contact_offset_m",
    "rest_offset_m",
    "max_depenetration_velocity_mps",
    "position_iterations",
    "velocity_iterations",
    "sleep_enabled",
    "solver_type",
    "ccd_mode",
)


def _record(value, source, origin, note=""):
    return {"value": value, "source": source, "origin": origin, "note": note}


def _absent(source, note=""):
    return _record(None, source, "absent", note)


def _read_attr(prim, name, *, scale=1.0, cast=None):
    """One USD attribute, with its authored-or-fallback provenance."""
    try:
        attribute = prim.GetAttribute(name)
        if not attribute or not attribute.IsValid():
            return _absent(f"{prim.GetPath()}.{name}", "attribute not present on this prim")
        value = attribute.Get()
        if value is None:
            return _absent(f"{prim.GetPath()}.{name}", "attribute present but returned None")
        origin = "authored" if attribute.HasAuthoredValue() else "schema_fallback"
        if cast is not None:
            value = cast(value)
        elif hasattr(value, "__len__") and not isinstance(value, str):
            value = [float(component) * scale for component in value]
        elif isinstance(value, bool):
            pass
        elif isinstance(value, (int, float)):
            value = type(value)(value * scale) if scale != 1.0 else value
        return _record(value, f"{prim.GetPath()}.{name}", origin)
    except Exception as error:  # pragma: no cover - engine-side failure
        return _record(None, f"{prim.GetPath()}.{name}", "error", repr(error))


def capture(stage, world, probe, cube) -> dict:
    """Read every expected field off the live stage."""
    from pxr import PhysxSchema, UsdPhysics

    fields: dict[str, dict] = {}
    probe_prim = stage.GetPrimAtPath(PROBE_PATH)
    cube_prim = stage.GetPrimAtPath(CUBE_PATH)

    # --- scene-level -------------------------------------------------------
    scene_prim = None
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            scene_prim = prim
            break
    if scene_prim is None:
        fields["gravity_mps2"] = _absent("UsdPhysics.Scene", "no physics scene prim on the stage")
        fields["solver_type"] = _absent("UsdPhysics.Scene")
    else:
        direction = _read_attr(scene_prim, "physics:gravityDirection")
        magnitude = _read_attr(scene_prim, "physics:gravityMagnitude")
        gravity = None
        if direction["value"] is not None and magnitude["value"] is not None:
            gravity = [component * float(magnitude["value"]) for component in direction["value"]]
        fields["gravity_mps2"] = _record(
            gravity, f"{scene_prim.GetPath()} gravityDirection * gravityMagnitude",
            direction["origin"] if direction["origin"] == magnitude["origin"] else "mixed",
            f"direction={direction['value']} magnitude={magnitude['value']}",
        )
        fields["solver_type"] = _read_attr(scene_prim, "physxScene:solverType")
        # Extra scene context worth having for a parity comparison.
        for name, attribute in (
            ("scene_enable_ccd", "physxScene:enableCCD"),
            ("scene_enable_gpu_dynamics", "physxScene:enableGPUDynamics"),
            ("scene_broadphase_type", "physxScene:broadphaseType"),
            ("scene_time_steps_per_second", "physxScene:timeStepsPerSecond"),
            ("scene_bounce_threshold_velocity", "physxScene:bounceThresholdVelocity"),
            ("scene_friction_offset_threshold", "physxScene:frictionOffsetThreshold"),
        ):
            fields[name] = _read_attr(scene_prim, attribute)

    # --- rigid body on the probe -------------------------------------------
    fields["linear_damping_per_s"] = _read_attr(probe_prim, "physxRigidBody:linearDamping")
    fields["angular_damping_per_s"] = _read_attr(probe_prim, "physxRigidBody:angularDamping")
    fields["initial_linear_velocity_mps"] = _read_attr(probe_prim, "physics:velocity")
    fields["max_depenetration_velocity_mps"] = _read_attr(
        probe_prim, "physxRigidBody:maxDepenetrationVelocity"
    )
    fields["position_iterations"] = _read_attr(
        probe_prim, "physxRigidBody:solverPositionIterationCount", cast=int
    )
    fields["velocity_iterations"] = _read_attr(
        probe_prim, "physxRigidBody:solverVelocityIterationCount", cast=int
    )
    fields["sleep_enabled"] = _read_attr(probe_prim, "physxRigidBody:disableGravity")
    fields["sleep_threshold"] = _read_attr(probe_prim, "physxRigidBody:sleepThreshold")
    fields["stabilization_threshold"] = _read_attr(
        probe_prim, "physxRigidBody:stabilizationThreshold"
    )
    fields["ccd_mode"] = _read_attr(probe_prim, "physxRigidBody:enableCCD")

    orientation = _read_attr(probe_prim, "xformOp:orient")
    if orientation["value"] is None:
        # The constructor may author the pose as a quatf rather than an op.
        orientation = _absent(
            f"{PROBE_PATH}.xformOp:orient",
            "no orientation op authored; the sphere was created axis aligned",
        )
    fields["initial_orientation_wxyz"] = orientation

    # --- collision offsets --------------------------------------------------
    for key, attribute in (
        ("contact_offset_m", "physxCollision:contactOffset"),
        ("rest_offset_m", "physxCollision:restOffset"),
    ):
        probe_value = _read_attr(probe_prim, attribute)
        if probe_value["value"] is None:
            probe_value = _read_attr(cube_prim, attribute)
        fields[key] = probe_value

    # --- materials ----------------------------------------------------------
    material_fields = {
        "static_friction": "physics:staticFriction",
        "dynamic_friction": "physics:dynamicFriction",
        "restitution": "physics:restitution",
    }
    materials = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MaterialAPI) or "Material" in str(prim.GetTypeName()):
            if prim.GetAttribute("physics:staticFriction"):
                materials.append(prim)
    if not materials:
        for key in material_fields:
            fields[key] = _absent(
                "UsdPhysics.MaterialAPI",
                "no physics material found on the stage; PhysX used its own "
                "internal default, which USD does not expose",
            )
        fields["friction_combine_mode"] = _absent("PhysxSchema material")
        fields["restitution_combine_mode"] = _absent("PhysxSchema material")
    else:
        primary = materials[0]
        for key, attribute in material_fields.items():
            fields[key] = _read_attr(primary, attribute)
        fields["friction_combine_mode"] = _read_attr(
            primary, "physxMaterial:frictionCombineMode"
        )
        fields["restitution_combine_mode"] = _read_attr(
            primary, "physxMaterial:restitutionCombineMode"
        )
        if len(materials) > 1:
            fields["material_prim_count"] = _record(
                len(materials), "stage traversal", "authored",
                "more than one physics material on the stage; the contact pair "
                "may not use the one reported here. Paths: "
                + ", ".join(str(prim.GetPath()) for prim in materials),
            )

    # --- mass and geometry, as a cross-check of scene identity -------------
    fields["probe_mass_kg"] = _read_attr(probe_prim, "physics:mass")
    fields["probe_radius_m"] = _read_attr(probe_prim, "radius")
    return fields


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="isaac_runtime.json")
    parser.add_argument("--settle-steps", type=int, default=5,
                        help="physics steps before reading, so the engine has applied its defaults")
    parser.add_argument("--headless", default=True, action="store_true")
    args = parser.parse_args(argv)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": bool(args.headless), "active_gpu": 0, "physics_gpu": 0})
    result = {
        "schema": "isaac-effective-runtime/v1",
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "scene_source": "Phase 1 scripts/step6_3.py create_scene, geometry copied verbatim",
        "declared_geometry": {
            "cube_center_world_m": list(CUBE_CENTER),
            "cube_size_m": CUBE_SIZE,
            "probe_radius_m": TIP_RADIUS,
            "probe_mass_kg": TIP_MASS,
            "probe_home_z_m": Z_HOME,
            "physics_dt_s": DT,
        },
        "host": {},
        "fields": {},
        "capture_errors": [],
    }
    try:
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicSphere, FixedCuboid

        try:
            import isaacsim
            result["host"]["isaac_sim_version"] = getattr(isaacsim, "__version__", "unknown")
        except Exception:
            result["host"]["isaac_sim_version"] = "unknown"
        result["host"]["python"] = sys.version.split()[0]
        result["host"]["platform"] = platform.platform()

        world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
        world.scene.add_default_ground_plane()
        cube = world.scene.add(FixedCuboid(
            prim_path=CUBE_PATH, name="cube",
            position=np.array(CUBE_CENTER),
            scale=np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]),
            size=1.0, color=np.array([0.2, 0.5, 1.0]),
        ))
        probe = world.scene.add(DynamicSphere(
            prim_path=PROBE_PATH, name="fingertip",
            position=np.array([0.0, 0.0, Z_HOME]),
            radius=TIP_RADIUS, mass=TIP_MASS, color=np.array([1.0, 0.3, 0.3]),
        ))
        world.reset()
        # A few steps so anything the engine fills in at simulation start is
        # present on the stage before we read it.
        for _ in range(max(0, args.settle_steps)):
            world.step(render=False)

        import omni.usd
        stage = omni.usd.get_context().get_stage()
        result["fields"] = capture(stage, world, probe, cube)

        # Dump every physics-relevant attribute we did not ask for, so a field
        # this script looked for in the wrong place can still be recovered from
        # the capture without renting the machine a second time.
        inventory = {}
        for path in (PROBE_PATH, CUBE_PATH):
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            entries = {}
            for attribute in prim.GetAttributes():
                name = attribute.GetName()
                if not any(token in name for token in ("physics", "physx", "radius", "size")):
                    continue
                try:
                    value = attribute.Get()
                    if hasattr(value, "__len__") and not isinstance(value, str):
                        value = [float(component) for component in value]
                    elif isinstance(value, (int, float, bool)) or value is None:
                        pass
                    else:
                        value = str(value)
                    entries[name] = {
                        "value": value,
                        "authored": bool(attribute.HasAuthoredValue()),
                    }
                except Exception as error:
                    entries[name] = {"value": None, "error": repr(error)}
            inventory[path] = entries
        # Scene and material prims too.
        from pxr import UsdPhysics
        for prim in stage.Traverse():
            is_scene = prim.IsA(UsdPhysics.Scene)
            is_material = bool(prim.GetAttribute("physics:staticFriction"))
            if not (is_scene or is_material):
                continue
            entries = {}
            for attribute in prim.GetAttributes():
                name = attribute.GetName()
                if not any(token in name for token in ("physics", "physx")):
                    continue
                try:
                    value = attribute.Get()
                    if hasattr(value, "__len__") and not isinstance(value, str):
                        value = [float(component) for component in value]
                    elif not isinstance(value, (int, float, bool)) and value is not None:
                        value = str(value)
                    entries[name] = {
                        "value": value,
                        "authored": bool(attribute.HasAuthoredValue()),
                    }
                except Exception as error:
                    entries[name] = {"value": None, "error": repr(error)}
            inventory[str(prim.GetPath())] = entries
        result["attribute_inventory"] = inventory

    except Exception:
        result["capture_errors"].append(traceback.format_exc())
    finally:
        try:
            app.close()
        except Exception:
            pass

    fields = result.get("fields", {})
    missing = [name for name in EXPECTED_FIELDS if name not in fields]
    unresolved = [
        name for name in EXPECTED_FIELDS
        if fields.get(name, {}).get("origin") in (None, "absent", "error")
    ]
    result["summary"] = {
        "expected_fields": len(EXPECTED_FIELDS),
        "fields_read": len(EXPECTED_FIELDS) - len(unresolved) - len(missing),
        "fields_not_located": missing,
        "fields_still_unresolved": unresolved,
        "authored": sorted(
            name for name, entry in fields.items() if entry.get("origin") == "authored"
        ),
        "schema_fallback": sorted(
            name for name, entry in fields.items() if entry.get("origin") == "schema_fallback"
        ),
        "note": (
            "a schema_fallback value is the USD fallback for an unauthored "
            "attribute, which is weaker evidence than an authored value: PhysX "
            "may use an internal default USD does not expose. Fields listed as "
            "still unresolved must stay unresolved in the K0 inventory"
        ),
    }
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"\nwrote {args.output}")
    if result["capture_errors"]:
        print("\nCAPTURE ERRORS (the JSON still holds whatever was read):")
        for error in result["capture_errors"]:
            print(error)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
