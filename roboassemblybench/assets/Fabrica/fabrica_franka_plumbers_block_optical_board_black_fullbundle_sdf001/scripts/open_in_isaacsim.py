#!/usr/bin/env python3
"""Open this packaged scene in Isaac Sim.

Run with Isaac Sim's Python, for example:

    /isaac-sim/python.sh scripts/open_in_isaacsim.py

The script loads ``scene/scene.usda`` from the package root, sets a readable
camera view, and keeps the Isaac Sim window alive for inspection.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from isaacsim import SimulationApp


def _parse_vec3(text: str) -> list[float]:
    parts = [p.strip() for p in text.replace(",", " ").split() if p.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three numbers, e.g. '4.5 3.8 3.2'")
    return [float(p) for p in parts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open the packaged Fabrica scene in Isaac Sim.")
    parser.add_argument("--scene", default="scene/scene.usda", help="Scene path relative to package root or absolute path.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
    parser.add_argument("--play", action="store_true", help="Start simulation playback after loading.")
    parser.add_argument("--steps", type=int, default=-1, help="Number of render steps to keep alive; -1 means until window closes.")
    parser.add_argument("--eye", type=_parse_vec3, default=None, help="Camera eye position, e.g. '5.2 4.2 4.4'.")
    parser.add_argument("--target", type=_parse_vec3, default=None, help="Camera target position, e.g. '0.5 -0.1 1.4'.")
    return parser.parse_args()


args = parse_args()
simulation_app = SimulationApp({"headless": args.headless})

import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.viewports import set_camera_view  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402


def compute_bbox(stage: Usd.Stage, prim_path: str = "/World") -> tuple[Gf.Vec3d, Gf.Vec3d]:
    prim = stage.GetPrimAtPath(prim_path)
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=True,
    )
    rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if rng.IsEmpty():
        return Gf.Vec3d(0, 0, 1), Gf.Vec3d(3, 2, 2)
    center = (rng.GetMin() + rng.GetMax()) * 0.5
    size = rng.GetMax() - rng.GetMin()
    return center, size


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    scene = Path(args.scene)
    if not scene.is_absolute():
        scene = package_root / scene
    if not scene.exists():
        raise FileNotFoundError(scene)

    print(f"[open_in_isaacsim] Loading scene: {scene}")
    omni.usd.get_context().open_stage(str(scene))
    stage = omni.usd.get_context().get_stage()
    world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
    world.reset()

    center, size = compute_bbox(stage)
    if args.eye is not None and args.target is not None:
        eye = args.eye
        target = args.target
    else:
        radius = max(float(size[0]), float(size[1]), float(size[2]), 1.0)
        target = [float(center[0]), float(center[1]), float(center[2])]
        eye = [
            target[0] + radius * 1.45,
            target[1] + radius * 1.25,
            target[2] + radius * 0.85,
        ]
    set_camera_view(eye=eye, target=target)
    print(f"[open_in_isaacsim] Camera eye={eye} target={target}")

    if args.play:
        world.play()
        print("[open_in_isaacsim] Simulation playing.")
    else:
        world.pause()
        print("[open_in_isaacsim] Simulation paused.")

    step = 0
    while simulation_app.is_running():
        world.step(render=True)
        step += 1
        if args.steps >= 0 and step >= args.steps:
            break
        time.sleep(0.001)

    simulation_app.close()


if __name__ == "__main__":
    main()
