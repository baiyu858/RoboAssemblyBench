#!/usr/bin/env python3
"""Convert RoboFactory mesh assets to USD with authored physics proxies."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "third_part" / "RoboFactory" / "robofactory" / "assets" / "objects"
DEFAULT_OUTPUT_ROOT = ROOT / "roboassemblybench" / "assets" / "robofactory_converted"


def _parse_obj_bounds(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return _parse_obj_collection_bounds([path])


def _parse_obj_collection_bounds(paths: list[Path]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    mins = [math.inf, math.inf, math.inf]
    maxs = [-math.inf, -math.inf, -math.inf]
    for path in paths:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("v "):
                    continue
                parts = line.split()
                if len(parts) < 4:
                    continue
                values = [float(parts[1]), float(parts[2]), float(parts[3])]
                for idx, value in enumerate(values):
                    mins[idx] = min(mins[idx], value)
                    maxs[idx] = max(maxs[idx], value)
    if any(not math.isfinite(v) for v in mins + maxs):
        raise ValueError(f"No vertices found in OBJ collection: {paths}")
    return tuple(mins), tuple(maxs)


def _center_and_normalize(mins, maxs):
    center = tuple((mins[i] + maxs[i]) * 0.5 for i in range(3))
    extents = tuple(max(maxs[i] - mins[i], 1e-6) for i in range(3))
    scale = tuple(1.0 / extents[i] for i in range(3))
    return center, extents, scale


def _pot_obj_sources() -> list[Path]:
    names = (
        "original-1.obj",
        "original-3.obj",
        "original-4.obj",
        "original-5.obj",
        "original-6.obj",
        "original-7.obj",
        "original-8.obj",
        "original-10.obj",
        "original-11.obj",
        "original-13.obj",
        "original-14.obj",
    )
    pot_obj_root = SOURCE_ROOT / "pot_annotated" / "textured_objs"
    return [pot_obj_root / name for name in names]


def _vec(values) -> str:
    return "(" + ", ".join(f"{float(v):.9g}" for v in values) + ")"


def _rel_reference(from_file: Path, to_file: Path) -> str:
    return os.path.relpath(to_file.resolve(), start=from_file.resolve().parent).replace(os.sep, "/")


async def _convert_obj(src: Path, dst: Path, *, load_materials: bool = True) -> None:
    import omni.kit.asset_converter

    dst.parent.mkdir(parents=True, exist_ok=True)
    context = omni.kit.asset_converter.AssetConverterContext()
    context.ignore_materials = not load_materials
    context.export_preview_surface = True
    context.use_meter_as_world_unit = True
    context.convert_stage_up_z = True
    context.create_world_as_default_root_prim = False
    context.smooth_normals = True
    context.ignore_animations = True
    context.ignore_camera = True
    context.ignore_light = True

    def progress_callback(progress, total_steps):
        return True

    task = omni.kit.asset_converter.get_instance().create_converter_task(
        str(src),
        str(dst),
        progress_callback,
        context,
    )
    while True:
        success = await task.wait_until_finished()
        if success:
            break
        await asyncio.sleep(0.1)
    if not dst.exists():
        raise RuntimeError(f"OBJ conversion did not produce expected USD: {dst}")


def _cube(name: str, translate, scale, color=(0.15, 0.70, 1.0), visible: bool = False) -> str:
    visibility = "" if visible else '        token visibility = "invisible"\n'
    return f"""    def Cube "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
{visibility}        token purpose = "guide"
        bool physics:collisionEnabled = 1
        double size = 1
        color3f[] primvars:displayColor = [{_vec(color)}]
        double3 xformOp:translate = {_vec(translate)}
        double3 xformOp:scale = {_vec(scale)}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
"""


def _write_meat_wrapper(wrapper_path: Path, visual_usd: Path, obj_src: Path) -> dict:
    mins, maxs = _parse_obj_bounds(obj_src)
    center, extents, normalize_scale = _center_and_normalize(mins, maxs)
    reference = _rel_reference(wrapper_path, visual_usd)
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(
        f"""#usda 1.0
(
    defaultPrim = "RoboFactoryMeat"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "RoboFactoryMeat"
{{
    def Xform "Visual"
    {{
        token purpose = "render"
        double3 xformOp:scale = {_vec(normalize_scale)}
        uniform token[] xformOpOrder = ["xformOp:scale"]

        def Xform "Raw" (
            prepend references = @{reference}@
        )
        {{
            double3 xformOp:translate = {_vec([-center[0], -center[1], -center[2]])}
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
    }}

{_cube("collision_core", (0.00, 0.00, -0.02), (0.88, 0.74, 0.62))}
{_cube("collision_left_lobe", (-0.22, 0.05, 0.06), (0.52, 0.58, 0.54))}
{_cube("collision_right_lobe", (0.24, -0.03, 0.02), (0.48, 0.54, 0.50))}
}}
""",
        encoding="utf-8",
    )
    return {"source": str(obj_src), "visual_usd": str(visual_usd), "wrapper": str(wrapper_path), "bbox": [mins, maxs]}


def _write_barrier_wrapper(wrapper_path: Path, visual_usd: Path, obj_src: Path) -> dict:
    mins, maxs = _parse_obj_bounds(obj_src)
    center, extents, normalize_scale = _center_and_normalize(mins, maxs)
    reference = _rel_reference(wrapper_path, visual_usd)
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(
        f"""#usda 1.0
(
    defaultPrim = "RoboFactoryLiftBarrier"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "RoboFactoryLiftBarrier"
{{
    def Xform "Visual"
    {{
        token purpose = "render"

        def Xform "SourceXToTaskY"
        {{
            double xformOp:rotateZ = 90
            uniform token[] xformOpOrder = ["xformOp:rotateZ"]

            def Xform "Normalized"
            {{
                double3 xformOp:scale = {_vec(normalize_scale)}
                uniform token[] xformOpOrder = ["xformOp:scale"]

                def Xform "Raw" (
                    prepend references = @{reference}@
                )
                {{
                    double3 xformOp:translate = {_vec([-center[0], -center[1], -center[2]])}
                    uniform token[] xformOpOrder = ["xformOp:translate"]
                }}
            }}
        }}
    }}

{_cube("collision_body", (0.00, 0.00, 0.00), (0.96, 0.96, 0.92))}
{_cube("collision_left_grip_face", (0.00, -0.46, 0.00), (0.92, 0.08, 0.96))}
{_cube("collision_right_grip_face", (0.00, 0.46, 0.00), (0.92, 0.08, 0.96))}
}}
""",
        encoding="utf-8",
    )
    return {"source": str(obj_src), "visual_usd": str(visual_usd), "wrapper": str(wrapper_path), "bbox": [mins, maxs]}


def _write_pot_visual(visual_path: Path, part_usds: list[Path], obj_sources: list[Path]) -> dict:
    mins, maxs = _parse_obj_collection_bounds(obj_sources)
    center, extents, normalize_scale = _center_and_normalize(mins, maxs)
    visual_path.parent.mkdir(parents=True, exist_ok=True)
    part_refs = []
    for idx, part_usd in enumerate(part_usds):
        part_refs.append(
            f"""            def Xform "part_{idx:02d}" (
                prepend references = @{_rel_reference(visual_path, part_usd)}@
            )
            {{
            }}
"""
        )
    visual_path.write_text(
        f"""#usda 1.0
(
    defaultPrim = "RoboFactoryPotVisual"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "RoboFactoryPotVisual"
{{
    token purpose = "render"
    double3 xformOp:scale = {_vec(normalize_scale)}
    uniform token[] xformOpOrder = ["xformOp:scale"]

    def Xform "Centered"
    {{
        double3 xformOp:translate = {_vec([-center[0], -center[1], -center[2]])}
        uniform token[] xformOpOrder = ["xformOp:translate"]
{''.join(part_refs)}    }}
}}
""",
        encoding="utf-8",
    )
    return {"visual_usd": str(visual_path), "bbox": [mins, maxs], "parts": [str(path) for path in part_usds]}


def _write_pot_wrapper(wrapper_path: Path, visual_usd: Path, obj_sources: list[Path]) -> dict:
    reference = _rel_reference(wrapper_path, visual_usd)
    wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper_path.write_text(
        f"""#usda 1.0
(
    defaultPrim = "RoboFactoryPot"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "RoboFactoryPot"
{{
    def Xform "Visual" (
        prepend references = @{reference}@
    )
    {{
        token purpose = "render"
        double3 xformOp:translate = (0, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }}

{_cube("collision_bottom", (0.00, 0.00, -0.42), (0.92, 0.92, 0.14))}
{_cube("collision_left_wall", (-0.46, 0.00, 0.00), (0.10, 0.92, 0.82))}
{_cube("collision_right_wall", (0.46, 0.00, 0.00), (0.10, 0.92, 0.82))}
{_cube("collision_front_wall", (0.00, -0.46, 0.00), (0.82, 0.10, 0.82))}
{_cube("collision_back_wall", (0.00, 0.46, 0.00), (0.82, 0.10, 0.82))}
}}
""",
        encoding="utf-8",
    )
    return {
        "source_parts": [str(path) for path in obj_sources],
        "visual_usd": str(visual_usd),
        "wrapper": str(wrapper_path),
    }


def _write_manifest(output_root: Path, records: list[dict]) -> None:
    manifest = {
        "asset_family": "robofactory_converted",
        "source_root": str(SOURCE_ROOT),
        "notes": [
            "visual USDs are converted from RoboFactory OBJ mesh assets",
            "physics USDs reference converted visuals and author simplified collision proxies",
            "recipes should set auto_collider: false so authored proxies are preserved",
        ],
        "records": records,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _asset_paths(output_root: Path) -> dict[str, Path]:
    visual_root = output_root / "visual"
    physics_root = output_root / "physics"
    return {
        "meat_obj": SOURCE_ROOT / "meat_annotated" / "textured.obj",
        "barrier_obj": SOURCE_ROOT / "steel_barrier_annotated" / "textured.obj",
        "meat_visual": visual_root / "place_food" / "meat_visual.usd",
        "barrier_visual": visual_root / "lift_barrier" / "barrier_visual.usd",
        "pot_visual": visual_root / "place_food" / "pot_visual.usda",
        "pot_parts_root": visual_root / "place_food" / "pot_parts",
        "meat_wrapper": physics_root / "place_food" / "meat.usda",
        "barrier_wrapper": physics_root / "lift_barrier" / "barrier.usda",
        "pot_wrapper": physics_root / "place_food" / "pot.usda",
    }


def _write_wrapper_outputs(output_root: Path) -> None:
    paths = _asset_paths(output_root)
    for name in ("meat_visual", "barrier_visual", "pot_visual"):
        if not paths[name].exists():
            raise FileNotFoundError(f"Missing converted visual USD for {name}: {paths[name]}")
    records = [
        _write_meat_wrapper(paths["meat_wrapper"], paths["meat_visual"], paths["meat_obj"]),
        _write_barrier_wrapper(paths["barrier_wrapper"], paths["barrier_visual"], paths["barrier_obj"]),
        _write_pot_wrapper(paths["pot_wrapper"], paths["pot_visual"], _pot_obj_sources()),
    ]
    _write_manifest(output_root, records)


def _pot_part_usds(output_root: Path) -> list[Path]:
    paths = _asset_paths(output_root)
    return [
        paths["pot_parts_root"] / f"{source.stem}.usd"
        for source in _pot_obj_sources()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--skip-conversion", action="store_true", help="Only rewrite physics wrapper USDs.")
    parser.add_argument("--force", action="store_true", help="Reconvert visual USDs even when outputs already exist.")
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    paths = _asset_paths(output_root)
    visual_usds_exist = all(
        paths[name].exists()
        for name in ("meat_visual", "barrier_visual", "pot_visual")
    ) and all(path.exists() for path in _pot_part_usds(output_root))
    if args.skip_conversion or (visual_usds_exist and not args.force):
        _write_wrapper_outputs(output_root)
        return

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("omni.kit.asset_converter")

        if args.force or not paths["meat_visual"].exists():
            asyncio.get_event_loop().run_until_complete(
                _convert_obj(paths["meat_obj"], paths["meat_visual"], load_materials=True)
            )
        if args.force or not paths["barrier_visual"].exists():
            asyncio.get_event_loop().run_until_complete(
                _convert_obj(paths["barrier_obj"], paths["barrier_visual"], load_materials=True)
            )
        pot_sources = _pot_obj_sources()
        pot_part_usds = _pot_part_usds(output_root)
        for src, dst in zip(pot_sources, pot_part_usds, strict=True):
            if args.force or not dst.exists():
                asyncio.get_event_loop().run_until_complete(_convert_obj(src, dst, load_materials=True))
        _write_pot_visual(paths["pot_visual"], pot_part_usds, pot_sources)
        _write_wrapper_outputs(output_root)
    finally:
        app.close()


if __name__ == "__main__":
    main()
