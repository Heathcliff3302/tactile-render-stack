"""Isaac Sim smoke test for a rented GPU machine (run with the Isaac Sim Python).

    OMNI_KIT_ACCEPT_EULA=YES python isaac_smoke.py

Passes only if, headless: the app starts, a sphere falls onto a ground plane
under PhysX, the RTX renderer steps every frame, and the app exits cleanly.
Prints one JSON line prefixed with SMOKE_RESULT; exit code 0 on pass.
"""

import json
import sys
import time

t0 = time.time()
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
t_boot = time.time() - t0

import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import DynamicSphere  # noqa: E402

result = {"boot_s": round(t_boot, 1)}
try:
    world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    world.scene.add_default_ground_plane()
    sphere = world.scene.add(DynamicSphere(prim_path="/World/sphere", name="sphere",
                                           position=np.array([0.0, 0.0, 0.5]), radius=0.05))
    world.reset()
    t1 = time.time()
    for _ in range(180):
        world.step(render=True)
    z = float(sphere.get_world_pose()[0][2])
    result.update({
        "steps": 180,
        "step_s": round(time.time() - t1, 2),
        "sphere_final_z_m": round(z, 4),
        # resting on the plane: centre at radius above ground, within a few mm
        "physics_ok": abs(z - 0.05) < 0.005,
    })
except Exception as exc:  # report, then fail
    result.update({"error": f"{type(exc).__name__}: {exc}", "physics_ok": False})

print("SMOKE_RESULT " + json.dumps(result), flush=True)
app.close()
sys.exit(0 if result.get("physics_ok") else 1)
