from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "roboassemblybench/assets/Fabrica/official_logs/codex_plumbers_block_ur5e_official/plumbers_block"
DEFAULT_ASSEMBLY_DIR = REPO_ROOT / "roboassemblybench/assets/Fabrica/official_replay_assets/fabrica/plumbers_block"
DEFAULT_ASSET_DIR = REPO_ROOT / "roboassemblybench/assets/Fabrica/official_replay_assets"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/fabrica_official_isaacsim/plumbers_block_ur5e_official_traj_replay.mp4"

UNIT_SCALE = 0.01
SCENE_PROFILE_DIR = REPO_ROOT / "roboassemblybench/scenes/profiles"

UR5E_LINK_MESHES = {
    "base_link": "ur5e/visual/base.obj",
    "shoulder_link": "ur5e/visual/shoulder.obj",
    "upper_arm_link": "ur5e/visual/upperarm.obj",
    "forearm_link": "ur5e/visual/forearm.obj",
    "wrist_1_link": "ur5e/visual/wrist1.obj",
    "wrist_2_link": "ur5e/visual/wrist2.obj",
    "wrist_3_link": "ur5e/visual/wrist3.obj",
}

ROBOTIQ_85_LINK_MESHES = {
    "robotiq_base": "robotiq_85/visual/robotiq_base_fine.obj",
    "robotiq_left_outer_knuckle": "robotiq_85/visual/outer_knuckle_fine.obj",
    "robotiq_left_outer_finger": "robotiq_85/visual/outer_finger_fine.obj",
    "robotiq_left_inner_knuckle": "robotiq_85/visual/inner_knuckle_fine.obj",
    "robotiq_left_inner_finger": "robotiq_85/visual/inner_finger_fine.obj",
    "robotiq_right_outer_knuckle": "robotiq_85/visual/outer_knuckle_fine.obj",
    "robotiq_right_outer_finger": "robotiq_85/visual/outer_finger_fine.obj",
    "robotiq_right_inner_knuckle": "robotiq_85/visual/inner_knuckle_fine.obj",
    "robotiq_right_inner_finger": "robotiq_85/visual/inner_finger_fine.obj",
}

UR5E_BUNDLE_ASSET_DIR = (
    REPO_ROOT
    / "roboassemblybench/assets/Fabrica/fabrica_ur5e_cooling_optical_board_black_fullbundle_sdf001/assets"
)

UR5E_LINK_USD_PRIMS = {
    "base_link": "/ur5e/base_link",
    "shoulder_link": "/ur5e/shoulder_link",
    "upper_arm_link": "/ur5e/upper_arm_link",
    "forearm_link": "/ur5e/forearm_link",
    "wrist_1_link": "/ur5e/wrist_1_link",
    "wrist_2_link": "/ur5e/wrist_2_link",
    "wrist_3_link": "/ur5e/wrist_3_link",
}

ROBOTIQ_85_PART_USD = {
    "robotiq_base": "isaac_official/Isaac/Robots/Robotiq/2F-85/parts/Defeatured_2F_85_PAD_OPEN_basestep_JFX.usd",
    "robotiq_left_outer_knuckle": "isaac_official/Isaac/Robots/Robotiq/2F-85/parts/Defeatured_2F_85_PAD_OPEN_finger4step_JFH.usd",
    "robotiq_left_outer_finger": "isaac_official/Isaac/Robots/Robotiq/2F-85/parts/Defeatured_2F_85_PAD_OPEN_Finger1step_JFT.usd",
    "robotiq_left_inner_knuckle": "isaac_official/Isaac/Robots/Robotiq/2F-85/parts/Defeatured_2F_85_PAD_OPEN_finger3step_JFL.usd",
    "robotiq_left_inner_finger": "isaac_official/Isaac/Robots/Robotiq/2F-85/parts/Defeatured_2F_85_PAD_OPEN_fingertipsstep_JFD.usd",
    "robotiq_right_outer_knuckle": "isaac_official/Isaac/Robots/Robotiq/2F-85/parts/Defeatured_2F_85_PAD_OPEN_finger4step_JFH.usd",
    "robotiq_right_outer_finger": "isaac_official/Isaac/Robots/Robotiq/2F-85/parts/Defeatured_2F_85_PAD_OPEN_finger2step_JFP.usd",
    "robotiq_right_inner_knuckle": "isaac_official/Isaac/Robots/Robotiq/2F-85/parts/Defeatured_2F_85_PAD_OPEN_finger3step_JFL.usd",
    "robotiq_right_inner_finger": "isaac_official/Isaac/Robots/Robotiq/2F-85/parts/Defeatured_2F_85_PAD_OPEN_fingertipsstep_JFD.usd",
}

PART_COLORS = [
    (0.95, 0.38, 0.28),
    (0.25, 0.55, 0.95),
    (0.33, 0.75, 0.47),
    (0.92, 0.70, 0.22),
    (0.72, 0.45, 0.90),
    (0.27, 0.78, 0.78),
]


@dataclass
class ReplayPrim:
    body_key: str
    translate_op: object
    orient_op: object


def _encode_mp4(frames_dir: Path, output_path: Path, fps: int) -> list[str]:
    png_paths = sorted(str(path) for path in frames_dir.glob("*.png"))
    if not png_paths:
        raise RuntimeError(f"No PNG frames were written to {frames_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
    try:
        for png_path in png_paths:
            writer.append_data(imageio.imread(png_path))
    finally:
        writer.close()
    return png_paths


def _to_uint8_rgba(frame) -> np.ndarray:
    frame_array = np.asarray(frame)
    if frame_array.size == 0:
        raise RuntimeError("Camera annotator returned an empty frame")
    if frame_array.ndim != 3:
        raise RuntimeError(f"Expected an HxWxC camera frame, got shape {frame_array.shape}")
    if frame_array.shape[-1] == 3:
        alpha = np.full((*frame_array.shape[:2], 1), 255, dtype=frame_array.dtype)
        frame_array = np.concatenate([frame_array, alpha], axis=-1)
    if np.issubdtype(frame_array.dtype, np.floating) and float(np.nanmax(frame_array)) <= 1.0 + 1e-6:
        frame_array = frame_array * 255.0
    frame_array = np.nan_to_num(frame_array, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(frame_array[..., :4], 0, 255).astype(np.uint8)


def _safe_name(name: str) -> str:
    return name.replace("-", "_").replace(".", "_")


def _load_fabrica_traj(traj_path: Path) -> np.ndarray:
    try:
        return np.load(traj_path, allow_pickle=True)
    except ModuleNotFoundError as exc:
        if exc.name != "numpy._core":
            raise
        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
        sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
        return np.load(traj_path, allow_pickle=True)


def _quat_to_matrix_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=float)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0:
        return np.eye(3)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _matrix_to_quat_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    m = np.asarray(rotation, dtype=float)
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / norm, x / norm, y / norm, z / norm)


def _iter_mesh_parts(path: Path) -> list[tuple[trimesh.Trimesh, tuple[float, float, float] | None]]:
    loaded = trimesh.load(path, force="scene", process=False)
    if isinstance(loaded, trimesh.Trimesh):
        return [(loaded, _mesh_material_color(loaded))]
    if not isinstance(loaded, trimesh.Scene):
        raise TypeError(f"Unsupported mesh type {type(loaded)} for {path}")
    mesh_parts = [
        (geom, _mesh_material_color(geom))
        for geom in loaded.geometry.values()
        if isinstance(geom, trimesh.Trimesh)
    ]
    if not mesh_parts:
        raise ValueError(f"No mesh geometry found in {path}")
    return mesh_parts


def _mesh_material_color(mesh: trimesh.Trimesh) -> tuple[float, float, float] | None:
    material = getattr(getattr(mesh, "visual", None), "material", None)
    diffuse = getattr(material, "diffuse", None)
    if diffuse is None:
        return None
    rgba = np.asarray(diffuse, dtype=float).reshape(-1)
    if rgba.size < 3:
        return None
    if float(np.nanmax(rgba[:3])) > 1.0 + 1e-6:
        rgba = rgba / 255.0
    rgb = np.clip(rgba[:3], 0.0, 1.0)
    return tuple(float(value) for value in rgb)


def _transformed_vertices_m(
    mesh: trimesh.Trimesh,
    *,
    local_pos_cm: tuple[float, float, float] | None = None,
    local_quat_wxyz: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    if local_quat_wxyz is not None:
        vertices = vertices @ _quat_to_matrix_wxyz(np.asarray(local_quat_wxyz, dtype=float)).T
    if local_pos_cm is not None:
        vertices += np.asarray(local_pos_cm, dtype=float)
    return vertices * UNIT_SCALE


def _create_replay_mesh(
    stage,
    *,
    body_key: str,
    mesh_path: Path,
    color: tuple[float, float, float] | None,
    local_pos_cm: tuple[float, float, float] | None = None,
    local_quat_wxyz: tuple[float, float, float, float] | None = None,
) -> ReplayPrim:
    from pxr import Gf, Sdf, UsdGeom

    xform_path = f"/World/replay/{_safe_name(body_key)}"
    xform = UsdGeom.Xform.Define(stage, xform_path)
    translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)

    for part_index, (mesh, material_color) in enumerate(_iter_mesh_parts(mesh_path)):
        mesh_prim = UsdGeom.Mesh.Define(stage, f"{xform_path}/mesh_{part_index:03d}")
        mesh_prim.CreatePointsAttr([Gf.Vec3f(*point) for point in _transformed_vertices_m(
            mesh,
            local_pos_cm=local_pos_cm,
            local_quat_wxyz=local_quat_wxyz,
        )])
        faces = np.asarray(mesh.faces, dtype=np.int64)
        mesh_prim.CreateFaceVertexCountsAttr([int(len(face)) for face in faces])
        mesh_prim.CreateFaceVertexIndicesAttr([int(index) for index in faces.reshape(-1)])
        mesh_prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)

        display_color = color or material_color or (0.75, 0.75, 0.75)
        gprim = UsdGeom.Gprim(mesh_prim.GetPrim())
        gprim.CreateDisplayColorAttr([Gf.Vec3f(*display_color)])
        gprim.CreateDisplayOpacityAttr([1.0])
        mesh_prim.GetPrim().CreateAttribute("doubleSided", Sdf.ValueTypeNames.Bool).Set(True)
    return ReplayPrim(body_key=body_key, translate_op=translate_op, orient_op=orient_op)


def _create_replay_reference(
    stage,
    *,
    body_key: str,
    asset_path: Path,
    prim_path: str | None = None,
) -> ReplayPrim:
    from pxr import UsdGeom

    xform_path = f"/World/replay/{_safe_name(body_key)}"
    xform = UsdGeom.Xform.Define(stage, xform_path)
    translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    reference_prim = UsdGeom.Xform.Define(stage, f"{xform_path}/asset").GetPrim()
    if prim_path:
        reference_prim.GetReferences().AddReference(str(asset_path), prim_path)
    else:
        reference_prim.GetReferences().AddReference(str(asset_path))
    return ReplayPrim(body_key=body_key, translate_op=translate_op, orient_op=orient_op)


def _create_replay_placeholder(
    stage,
    *,
    body_key: str,
    scale: tuple[float, float, float] = (0.05, 0.05, 0.05),
    color: tuple[float, float, float] = (0.85, 0.22, 0.18),
) -> ReplayPrim:
    from pxr import Gf, UsdGeom

    xform_path = f"/World/replay/{_safe_name(body_key)}"
    xform = UsdGeom.Xform.Define(stage, xform_path)
    translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)
    orient_op = xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble)
    cube = UsdGeom.Cube.Define(stage, f"{xform_path}/missing_asset")
    cube.CreateSizeAttr(1.0)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(Gf.Vec3d(*scale))
    UsdGeom.Gprim(cube.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return ReplayPrim(body_key=body_key, translate_op=translate_op, orient_op=orient_op)


def _create_replay_asset(
    stage,
    *,
    body_key: str,
    asset_path: Path,
    color: tuple[float, float, float] | None,
    usd_prim_path: str | None = None,
    local_pos_cm: tuple[float, float, float] | None = None,
    local_quat_wxyz: tuple[float, float, float, float] | None = None,
) -> ReplayPrim:
    suffix = asset_path.suffix.lower()
    if asset_path.exists() and suffix in {".usd", ".usda", ".usdc"}:
        return _create_replay_reference(
            stage,
            body_key=body_key,
            asset_path=asset_path,
            prim_path=usd_prim_path,
        )
    if asset_path.exists():
        return _create_replay_mesh(
            stage,
            body_key=body_key,
            mesh_path=asset_path,
            color=color,
            local_pos_cm=local_pos_cm,
            local_quat_wxyz=local_quat_wxyz,
        )
    print(f"[render_fabrica_traj_replay] warning: missing replay asset for {body_key}: {asset_path}")
    return _create_replay_placeholder(stage, body_key=body_key)


def _set_replay_prim(prim: ReplayPrim, matrix_cm: np.ndarray, *, world_offset_m: np.ndarray) -> None:
    from pxr import Gf

    matrix_cm = np.asarray(matrix_cm, dtype=float)
    position_m = matrix_cm[:3, 3] * UNIT_SCALE + world_offset_m
    quat_wxyz = _matrix_to_quat_wxyz(matrix_cm[:3, :3])
    prim.translate_op.Set(Gf.Vec3d(*position_m.tolist()))
    prim.orient_op.Set(Gf.Quatd(quat_wxyz[0], Gf.Vec3d(quat_wxyz[1], quat_wxyz[2], quat_wxyz[3])))


def _create_stage_lighting(stage) -> None:
    from pxr import Gf, UsdLux

    dome = UsdLux.DomeLight.Define(stage, "/World/lights/dome")
    dome.CreateIntensityAttr(450.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    key = UsdLux.DistantLight.Define(stage, "/World/lights/key")
    key.CreateIntensityAttr(1800.0)
    key.CreateAngleAttr(0.45)
    xform = key.GetPrim()
    xformable = __import__("pxr").UsdGeom.Xformable(xform)
    xformable.AddRotateXYZOp().Set(Gf.Vec3f(-55.0, 0.0, 35.0))


def _parse_vector3(value: str | tuple[float, float, float] | list[float]) -> np.ndarray:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) != 3:
            raise ValueError(f"Expected vector as x,y,z, got {value!r}")
        return np.asarray([float(part) for part in parts], dtype=float)
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"Expected 3-vector, got shape {vector.shape}")
    return vector


def _resolve_profile_path(scene_profile: str) -> Path:
    path = Path(scene_profile)
    if path.exists():
        return path
    profile_path = SCENE_PROFILE_DIR / f"{scene_profile}.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"Cannot resolve scene profile {scene_profile!r}; tried {profile_path}")
    return profile_path


def _resolve_placeholder_path(value: str) -> Path:
    replacements = {
        "${BENCHMARK_ROOT}": str(REPO_ROOT / "roboassemblybench"),
        "${ASSET_PATH}": str(REPO_ROOT / "roboassemblybench/assets"),
    }
    resolved = value
    for token, replacement in replacements.items():
        resolved = resolved.replace(token, replacement)
    return Path(resolved)


def _load_scene_profile(scene_profile: str) -> dict:
    import yaml

    profile_path = _resolve_profile_path(scene_profile)
    payload = yaml.safe_load(profile_path.read_text(encoding="utf-8")) or {}
    payload["scene_profile_path"] = str(profile_path)
    return payload


def _add_visual_cube(
    stage,
    *,
    prim_path: str,
    position: tuple[float, float, float] | list[float],
    scale: tuple[float, float, float] | list[float],
    color: tuple[float, float, float] | list[float],
) -> None:
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, f"/World/factory_scene{prim_path}")
    cube.CreateSizeAttr(1.0)
    xformable = UsdGeom.Xformable(cube.GetPrim())
    xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*position))
    xformable.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*scale))
    gprim = UsdGeom.Gprim(cube.GetPrim())
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])


def _add_factory_scene(stage, *, scene_profile: str | None, include_profile_objects: bool) -> dict | None:
    if scene_profile in {None, "", "none"}:
        return None

    from pxr import UsdGeom

    profile = _load_scene_profile(scene_profile)
    UsdGeom.Xform.Define(stage, "/World/factory_scene")

    referenced_scene_path = None
    for key in ("scene_asset_path", "scene_asset_fallback_path"):
        path_value = profile.get(key)
        if not path_value:
            continue
        scene_path = _resolve_placeholder_path(str(path_value))
        if scene_path.exists():
            stage.GetPrimAtPath("/World/factory_scene").GetReferences().AddReference(str(scene_path))
            referenced_scene_path = str(scene_path)
            break

    added_profile_object_count = 0
    if include_profile_objects:
        for object_spec in profile.get("objects", []):
            if object_spec.get("kind") not in {"visual_cube", "static_cube"}:
                continue
            _add_visual_cube(
                stage,
                prim_path=str(object_spec["prim_path"]),
                position=object_spec.get("position", [0.0, 0.0, 0.0]),
                scale=object_spec.get("scale", [1.0, 1.0, 1.0]),
                color=object_spec.get("color", [0.3, 0.3, 0.3]),
            )
            added_profile_object_count += 1

    return {
        "scene_profile": profile.get("profile_name", scene_profile),
        "scene_profile_path": profile.get("scene_profile_path"),
        "referenced_scene_path": referenced_scene_path,
        "include_profile_objects": include_profile_objects,
        "added_profile_object_count": added_profile_object_count,
    }


def _add_all_replay_prims(stage, *, assembly_dir: Path, asset_dir: Path, log_dir: Path) -> list[ReplayPrim]:
    replay_prims: list[ReplayPrim] = []

    robot_asset_dir = UR5E_BUNDLE_ASSET_DIR
    ur5e_usd_path = robot_asset_dir / "isaac_official/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
    optical_board_path = asset_dir / "optical_board.obj"
    if not optical_board_path.exists():
        optical_board_path = asset_dir / "fabrica_support/optical_board.obj"

    replay_prims.append(
        _create_replay_asset(
            stage,
            body_key="optical_board",
            asset_path=optical_board_path,
            color=(0.02, 0.02, 0.02),
        )
    )
    replay_prims.append(
        _create_replay_asset(
            stage,
            body_key="fixture",
            asset_path=log_dir / "fixture/fixture.obj",
            color=(0.86, 0.84, 0.76),
        )
    )

    obj_parts = sorted(assembly_dir.glob("*.obj"), key=lambda path: int(path.stem))
    usd_parts = sorted(
        assembly_dir.glob("*.usd"),
        key=lambda path: int(path.stem.rsplit("_", maxsplit=1)[-1]),
    )
    part_paths = obj_parts or usd_parts
    if not part_paths:
        print(f"[render_fabrica_traj_replay] warning: no OBJ/USD assembly parts found in {assembly_dir}")

    for part_path in part_paths:
        part_id = int(part_path.stem.rsplit("_", maxsplit=1)[-1])
        replay_prims.append(
            _create_replay_asset(
                stage,
                body_key=f"part{part_id}",
                asset_path=part_path,
                color=PART_COLORS[part_id % len(PART_COLORS)],
            )
        )

    for motion_type in ("move", "hold"):
        for link_name, relative_path in UR5E_LINK_MESHES.items():
            mesh_path = asset_dir / relative_path
            usd_prim_path = None
            asset_path = mesh_path
            if not mesh_path.exists():
                asset_path = ur5e_usd_path
                usd_prim_path = UR5E_LINK_USD_PRIMS[link_name]
            replay_prims.append(
                _create_replay_asset(
                    stage,
                    body_key=f"{link_name}_{motion_type}",
                    asset_path=asset_path,
                    color=None,
                    usd_prim_path=usd_prim_path,
                )
            )
        for link_name, relative_path in ROBOTIQ_85_LINK_MESHES.items():
            is_knuckle = "knuckle" in link_name
            mesh_path = asset_dir / relative_path
            asset_path = mesh_path
            if not mesh_path.exists():
                asset_path = robot_asset_dir / ROBOTIQ_85_PART_USD[link_name]
            replay_prims.append(
                _create_replay_asset(
                    stage,
                    body_key=f"{link_name}_{motion_type}",
                    asset_path=asset_path,
                    color=(0.42, 0.42, 0.42) if is_knuckle else (0.05, 0.05, 0.05),
                )
            )

    return replay_prims


def _camera_pose(option: str, *, world_offset_m: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if option == "close":
        position = np.asarray([1.25, -1.45, 1.15], dtype=float)
        look_at = np.asarray([0.0, 0.03, 0.16], dtype=float)
    elif option == "front":
        position = np.asarray([1.45, -1.85, 1.35], dtype=float)
        look_at = np.asarray([0.0, 0.04, 0.18], dtype=float)
    elif option == "far":
        position = np.asarray([1.85, -2.25, 1.65], dtype=float)
        look_at = np.asarray([0.0, 0.05, 0.22], dtype=float)
    else:
        raise ValueError(f"Unsupported camera option: {option}")
    position = position + world_offset_m
    look_at = look_at + world_offset_m
    return tuple(position.tolist()), tuple(look_at.tolist())


def _enable_webrtc_streaming(simulation_app) -> None:
    from omni.isaac.core.utils.extensions import enable_extension

    simulation_app.set_setting("/app/window/drawMouse", True)
    enable_extension("omni.kit.livestream.webrtc")


def render_fabrica_traj_replay(
    *,
    log_dir: Path,
    assembly_dir: Path,
    asset_dir: Path,
    output_path: Path,
    frames_dir: Path,
    width: int,
    height: int,
    fps: int,
    stride: int,
    max_frames: int | None,
    camera_option: str,
    factory_scene: str | None,
    include_profile_objects: bool,
    world_offset: str | tuple[float, float, float] | list[float],
    headless: bool,
    webrtc: bool = False,
) -> dict:
    from isaacsim import SimulationApp

    simulation_app = SimulationApp(
        {
            "headless": headless,
            "width": width,
            "height": height,
            "renderer": "RaytracedLighting",
        }
    )
    if webrtc:
        _enable_webrtc_streaming(simulation_app)

    try:
        import omni.replicator.core as rep
        import omni.usd
        from pxr import UsdGeom

        context = omni.usd.get_context()
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(stage, "/World")
        UsdGeom.Xform.Define(stage, "/World/replay")
        UsdGeom.Xform.Define(stage, "/World/lights")
        factory_scene_summary = _add_factory_scene(
            stage,
            scene_profile=factory_scene,
            include_profile_objects=include_profile_objects,
        )
        _create_stage_lighting(stage)

        replay_prims = _add_all_replay_prims(stage, assembly_dir=assembly_dir, asset_dir=asset_dir, log_dir=log_dir)

        world_offset_m = _parse_vector3(world_offset)
        camera_position, look_at = _camera_pose(camera_option, world_offset_m=world_offset_m)
        camera = rep.create.camera(position=camera_position, look_at=look_at)
        render_product = rep.create.render_product(camera, (width, height))
        annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")
        annotator.attach([render_product])
        rep.orchestrator.set_capture_on_play(False)

        if frames_dir.exists():
            shutil.rmtree(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

        traj_path = log_dir / "traj.npy"
        traj = _load_fabrica_traj(traj_path)
        render_indices = list(range(0, len(traj), max(stride, 1)))
        if render_indices[-1] != len(traj) - 1:
            render_indices.append(len(traj) - 1)
        if max_frames is not None:
            render_indices = render_indices[: max(max_frames, 1)]

        captured_indices: list[int] = []
        for _ in range(4):
            simulation_app.update()
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)

        for output_index, source_index in enumerate(render_indices):
            frame = traj[source_index]
            for prim in replay_prims:
                if prim.body_key not in frame:
                    continue
                _set_replay_prim(prim, frame[prim.body_key], world_offset_m=world_offset_m)

            simulation_app.update()
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)
            imageio.imwrite(frames_dir / f"rgb_{output_index:05d}.png", _to_uint8_rgba(annotator.get_data()))
            captured_indices.append(source_index)

        rep.orchestrator.wait_until_complete()
        png_paths = _encode_mp4(frames_dir=frames_dir, output_path=output_path, fps=fps)

        summary = {
            "mode": "official_fabrica_traj_replay_in_isaacsim",
            "assembly_name": assembly_dir.name,
            "arm": "ur5e",
            "gripper": "robotiq-85",
            "log_dir": str(log_dir),
            "traj_path": str(traj_path),
            "assembly_dir": str(assembly_dir),
            "asset_dir": str(asset_dir),
            "output_path": str(output_path),
            "frames_dir": str(frames_dir),
            "source_frame_count": int(len(traj)),
            "captured_frame_count": len(captured_indices),
            "written_png_count": len(png_paths),
            "captured_source_indices": captured_indices,
            "stride": stride,
            "fps": fps,
            "camera_width": width,
            "camera_height": height,
            "camera_option": camera_option,
            "camera_position": camera_position,
            "camera_look_at": look_at,
            "factory_scene": factory_scene_summary,
            "world_offset_m": world_offset_m.tolist(),
            "headless": bool(headless),
            "webrtc": bool(webrtc),
            "unit_scale_m_per_fabrica_unit": UNIT_SCALE,
            "limitations": [
                "This replays Fabrica's official RedMax traj.npy body matrices inside Isaac Sim.",
                "It is a kinematic visual replay, not an Isaac PhysX contact simulation.",
                "It does not retarget through the RoboAssemblyBench UR5e articulation controller.",
            ],
        }
        output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    finally:
        simulation_app.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an official Fabrica traj.npy body-matrix replay in Isaac Sim.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--assembly-dir", type=Path, default=DEFAULT_ASSEMBLY_DIR)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.stem + "_frames"))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--camera-option", choices=("close", "front", "far"), default="close")
    parser.add_argument("--factory-scene", default="none")
    parser.add_argument("--include-profile-objects", action="store_true")
    parser.add_argument("--world-offset", default="0,0,0")
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", action="store_false", dest="headless")
    parser.add_argument("--webrtc", action="store_true", help="Enable Isaac Sim WebRTC remote visualization.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = render_fabrica_traj_replay(
        log_dir=args.log_dir,
        assembly_dir=args.assembly_dir,
        asset_dir=args.asset_dir,
        output_path=args.output,
        frames_dir=args.frames_dir,
        width=args.width,
        height=args.height,
        fps=args.fps,
        stride=args.stride,
        max_frames=args.max_frames,
        camera_option=args.camera_option,
        factory_scene=args.factory_scene,
        include_profile_objects=args.include_profile_objects,
        world_offset=args.world_offset,
        headless=args.headless,
        webrtc=bool(args.webrtc),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
