from __future__ import annotations

import argparse
import json
import pickle
import shutil
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from internutopia.core.config import Config, SimConfig
from internutopia.core.vec_env import Env
from internutopia_extension import import_extensions
from internutopia_extension.configs.objects import UsdObjCfg
from toolkits.factory_dual_franka_assembly.planner_primitives import euler_xyz_to_quat, quat_multiply, quat_rotate
from toolkits.factory_dual_franka_assembly.scene_builder import build_dual_franka_assembly_episode


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "third_part/Fabrica/logs/codex_cooling_manifold_official/cooling_manifold"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "roboassemblybench/assets/Fabrica/fabrica_franka_cooling_optical_board_black_fullbundle_sdf001"
    / "assets/fabrica_original_usd_sdf_margin_001/aligned/cooling_manifold/manifest.json"
)
DEFAULT_SCENE_SPEC = (
    REPO_ROOT
    / "roboassemblybench/assets/Fabrica/fabrica_franka_cooling_optical_board_black_fullbundle_sdf001"
    / "scene/scene_spec.json"
)
DEFAULT_FIXTURE_USD = (
    REPO_ROOT
    / "roboassemblybench/assets/Fabrica/fabrica_franka_cooling_optical_board_black_fullbundle_sdf001"
    / "assets/fabrica_fixture/cooling_manifold/fixture_pickup_tray.usda"
)
DEFAULT_ASSEMBLY_NAME = "cooling_manifold"
DEFAULT_OBJECT_PREFIX = "fabrica_cooling_manifold"
DEFAULT_BASE_PART_ID = 1
ARM_JOINTS = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]
FINGER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]


def _encode_mp4(frames_dir: Path, output_path: Path, fps: int) -> list[str]:
    png_paths = sorted(str(path) for path in frames_dir.rglob("*.png"))
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


def _build_env(task_cfg, *, headless: bool) -> Env:
    config = Config(
        simulator=SimConfig(
            physics_dt=1 / 240,
            rendering_dt=1 / 240,
            use_fabric=False,
            headless=headless,
            native=False,
            webrtc=False,
        ),
        env_num=1,
        metrics_save_path="none",
        task_configs=[task_cfg],
    )
    import_extensions()
    return Env(config)


def _load_pickle(path: Path):
    with path.open("rb") as fp:
        return pickle.load(fp)


def _matrix_to_pos_quat(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    matrix = np.asarray(matrix, dtype=float)
    position = matrix[:3, 3].copy()
    quat_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return position, quat_xyzw[[3, 0, 1, 2]]


def _quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    return Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]]).as_matrix()


def _pose_to_matrix(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = _quat_to_matrix(quat_wxyz)
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix


def _matrix_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = np.asarray(rotation, dtype=float)
    matrix[:3, 3] = np.asarray(translation, dtype=float)
    return matrix


def _rigid_transform_from_points(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError(f"Need matching Nx3 point arrays with N>=3, got {source.shape} and {target.shape}")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    u, _, vt = np.linalg.svd(source_zero.T @ target_zero)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _transform_quat(rotation: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    world_rot = Rotation.from_matrix(rotation) * Rotation.from_quat(np.asarray(quat_wxyz)[[1, 2, 3, 0]])
    quat_xyzw = world_rot.as_quat()
    return quat_xyzw[[3, 0, 1, 2]]


def _part_id_from_manifest_part(part: dict, fallback_index: int) -> int:
    default_prim = str(part.get("default_prim") or "")
    if default_prim:
        suffix = default_prim.rsplit("_", maxsplit=1)[-1]
        if suffix.isdigit():
            return int(suffix)
    part_name = str(part.get("part_name") or "")
    suffix = part_name.rsplit("_", maxsplit=1)[-1]
    return int(suffix) if suffix.isdigit() else int(fallback_index)


def _part_object_name(object_prefix: str, part_id: int) -> str:
    return f"{object_prefix}_{part_id}"


def _load_variant_offsets(manifest_path: Path) -> dict[int, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    offsets = {}
    for index, part in enumerate(manifest["parts"]):
        part_id = _part_id_from_manifest_part(part, index)
        offsets[part_id] = np.asarray(part["raw_to_variant_translation_m"], dtype=float)
    return offsets


def _load_raw_bbox_centers(manifest_path: Path) -> dict[int, np.ndarray]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    centers = {}
    for index, part in enumerate(manifest["parts"]):
        part_id = _part_id_from_manifest_part(part, index)
        centers[part_id] = np.asarray(part["raw_bbox_center_m"], dtype=float)
    return centers


def _load_scene_spec(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Cannot find Fabrica scene spec: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _scene_spec_target_positions(scene_spec: dict, object_prefix: str) -> dict[int, np.ndarray]:
    target_poses = scene_spec.get("assembled_display", {}).get("assembly_target_poses", {})
    positions = {}
    for object_name, target_pose in target_poses.items():
        if not object_name.startswith(f"{object_prefix}_"):
            continue
        part_id = int(object_name.rsplit("_", maxsplit=1)[-1])
        positions[part_id] = np.asarray(target_pose["target_translation_world_m"], dtype=float)
    return positions


def _scene_spec_robot_layout(scene_spec: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    robot_layout = {}
    for robot_spec in scene_spec.get("robots", []):
        robot_id = str(robot_spec.get("id", ""))
        if robot_id == "robot_left":
            robot_name = "franka_left"
        elif robot_id == "robot_right":
            robot_name = "franka_right"
        else:
            continue
        robot_layout[robot_name] = (
            np.asarray(robot_spec["translation"], dtype=float),
            np.asarray(robot_spec["rotation_wxyz"], dtype=float),
        )
    return robot_layout


def _apply_scene_spec_robot_layout(task_cfg, scene_spec: dict) -> None:
    robot_layout = _scene_spec_robot_layout(scene_spec)
    if not robot_layout:
        return

    updated_robots = []
    for robot_cfg in task_cfg.robots:
        if robot_cfg.name not in robot_layout:
            updated_robots.append(robot_cfg)
            continue
        position, orientation = robot_layout[robot_cfg.name]
        updated_robots.append(
            robot_cfg.update(
                position=tuple(float(value) for value in position),
                orientation=tuple(float(value) for value in orientation),
            )
        )
    task_cfg.robots = updated_robots

    updated_metadata = []
    for metadata in task_cfg.robot_metadata:
        metadata = dict(metadata)
        robot_name = metadata.get("name")
        if robot_name in robot_layout:
            position, orientation = robot_layout[robot_name]
            metadata["position"] = position.tolist()
            metadata["orientation"] = orientation.tolist()
        updated_metadata.append(metadata)
    task_cfg.robot_metadata = updated_metadata


def _transformed_raw_pose(
    raw_matrix_cm: np.ndarray,
    map_rotation: np.ndarray,
    map_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_pos_cm, raw_quat = _matrix_to_pos_quat(raw_matrix_cm)
    raw_pos_m = raw_pos_cm * 0.01
    position = map_rotation @ raw_pos_m + map_translation
    orientation = _transform_quat(map_rotation, raw_quat)
    return position, orientation


def _apply_fabrica_workcell_robot_layout(
    task_cfg,
    traj_frame: dict,
    map_rotation: np.ndarray,
    map_translation: np.ndarray,
) -> dict[str, dict[str, list[float]]]:
    raw_robot_frames = {
        "franka_right": "panda_link0_move",
        "franka_left": "panda_link0_hold",
    }
    robot_layout: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    diagnostics: dict[str, dict[str, list[float]]] = {}
    for robot_name, traj_key in raw_robot_frames.items():
        position, orientation = _transformed_raw_pose(
            traj_frame[traj_key],
            map_rotation,
            map_translation,
        )
        robot_layout[robot_name] = (position, orientation)
        diagnostics[robot_name] = {
            "source_traj_key": traj_key,
            "position": position.tolist(),
            "orientation_wxyz": orientation.tolist(),
        }

    updated_robots = []
    for robot_cfg in task_cfg.robots:
        if robot_cfg.name not in robot_layout:
            updated_robots.append(robot_cfg)
            continue
        position, orientation = robot_layout[robot_cfg.name]
        updated_robots.append(
            robot_cfg.update(
                position=tuple(float(value) for value in position),
                orientation=tuple(float(value) for value in orientation),
            )
        )
    task_cfg.robots = updated_robots

    updated_metadata = []
    for metadata in task_cfg.robot_metadata:
        metadata = dict(metadata)
        robot_name = metadata.get("name")
        if robot_name in robot_layout:
            position, orientation = robot_layout[robot_name]
            metadata["position"] = position.tolist()
            metadata.pop("orientation_euler", None)
            metadata["orientation"] = orientation.tolist()
        updated_metadata.append(metadata)
    task_cfg.robot_metadata = updated_metadata
    return diagnostics


def _force_replay_rigid_parts(
    task_cfg,
    *,
    object_prefix: str,
    dynamic_part_ids: set[int],
    kinematic_part_ids: set[int],
) -> None:
    updated_objects = []
    for object_cfg in task_cfg.objects:
        name = getattr(object_cfg, "name", "")
        if not name.startswith(f"{object_prefix}_"):
            updated_objects.append(object_cfg)
            continue
        part_id = int(name.rsplit("_", maxsplit=1)[-1])
        if part_id in dynamic_part_ids and hasattr(object_cfg, "rigid_body"):
            updated_objects.append(object_cfg.update(rigid_body=True))
        elif part_id in kinematic_part_ids and hasattr(object_cfg, "rigid_body"):
            updated_objects.append(object_cfg.update(rigid_body=False))
        else:
            updated_objects.append(object_cfg)
    task_cfg.objects = updated_objects

    updated_metadata = []
    for metadata in task_cfg.object_metadata:
        metadata = dict(metadata)
        name = metadata.get("name", "")
        if str(name).startswith(f"{object_prefix}_"):
            part_id = int(str(name).rsplit("_", maxsplit=1)[-1])
            if part_id in dynamic_part_ids:
                metadata["rigid_body"] = True
            elif part_id in kinematic_part_ids:
                metadata["rigid_body"] = False
        updated_metadata.append(metadata)
    task_cfg.object_metadata = updated_metadata


def _add_fabrica_fixture(task_cfg, fixture_usd_path: Path | None) -> bool:
    if fixture_usd_path is None:
        return False
    if not fixture_usd_path.exists():
        raise FileNotFoundError(f"Cannot find Fabrica fixture USD: {fixture_usd_path}")
    if any(getattr(object_cfg, "name", "") == "fabrica_fixture" for object_cfg in task_cfg.objects):
        return True

    fixture_cfg = UsdObjCfg(
        name="fabrica_fixture",
        prim_path="/fabrica_fixture",
        usd_path=str(fixture_usd_path),
        position=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        collider=False,
        auto_collider=False,
        rigid_body=False,
        static_friction=None,
        dynamic_friction=None,
        restitution=0.0,
    )
    task_cfg.objects.append(fixture_cfg)
    task_cfg.object_metadata.append(
        {
            "name": "fabrica_fixture",
            "kind": "usd",
            "prim_path": "/fabrica_fixture",
            "usd_path": str(fixture_usd_path),
            "position": [0.0, 0.0, 0.0],
            "sampled_position": [0.0, 0.0, 0.0],
            "orientation": [1.0, 0.0, 0.0, 0.0],
            "sampled_orientation": [1.0, 0.0, 0.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "collider": False,
            "auto_collider": False,
            "rigid_body": False,
            "tracked": False,
            "replay_note": "Loaded as a visual static reference in this renderer; task recipe enables collider for UI/physics setup.",
            "source": "third_part/Fabrica/logs/.../fixture/fixture.obj converted to USD in meters",
        }
    )
    return True


def _initial_object_poses(task_cfg, object_prefix: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    poses = {}
    for metadata in task_cfg.object_metadata:
        name = metadata.get("name", "")
        if not name.startswith(f"{object_prefix}_"):
            continue
        part_id = int(name.rsplit("_", maxsplit=1)[-1])
        poses[part_id] = (
            np.asarray(metadata["sampled_position"], dtype=float),
            np.asarray(metadata["sampled_orientation"], dtype=float),
        )
    return poses


def _target_variant_positions(task_cfg, object_prefix: str) -> dict[int, np.ndarray]:
    positions: dict[int, np.ndarray] = {}
    for part_id in sorted(_initial_object_poses(task_cfg, object_prefix)):
        target = task_cfg.target_poses.get(f"part_{part_id}_seated")
        if target is not None:
            positions[part_id] = np.asarray(target["position"], dtype=float)
    return positions


def _derive_pose_mapping(
    task_cfg,
    traj: np.ndarray,
    part_offsets: dict[int, np.ndarray],
    *,
    mode: str,
    object_prefix: str,
    scene_spec: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if mode not in {"final_targets", "initial", "scene_spec_raw_center"}:
        raise ValueError(f"Unsupported mapping mode: {mode}")

    if mode == "initial":
        initial_poses = _initial_object_poses(task_cfg, object_prefix)
        source_ids = sorted(initial_poses)
        target_positions = {part_id: pose[0] for part_id, pose in initial_poses.items()}
        source_frame = traj[0]
    elif mode == "scene_spec_raw_center":
        if scene_spec is None:
            raise ValueError("scene_spec_raw_center mapping requires --scene-spec")
        target_positions = _scene_spec_target_positions(scene_spec, object_prefix)
        source_ids = sorted(target_positions)
        source_frame = traj[-1]
    else:
        target_positions = _target_variant_positions(task_cfg, object_prefix)
        source_ids = sorted(target_positions)
        source_frame = traj[-1]

    source_points = []
    target_points = []
    for part_id in source_ids:
        raw_pos_m, raw_quat = _matrix_to_pos_quat(source_frame[f"part{part_id}"])
        raw_pos_m = raw_pos_m * 0.01
        raw_rot = _quat_to_matrix(raw_quat)
        source_points.append(raw_pos_m + raw_rot @ part_offsets[part_id])
        target_points.append(target_positions[part_id])

    rotation, translation = _rigid_transform_from_points(np.asarray(source_points), np.asarray(target_points))
    fitted = (np.asarray(source_points) @ rotation.T) + translation
    errors = np.linalg.norm(fitted - np.asarray(target_points), axis=1)
    diagnostics = {
        "mode": mode,
        "part_ids": source_ids,
        "mean_fit_error_m": float(errors.mean()),
        "max_fit_error_m": float(errors.max()),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "per_part_error_m": {str(part_id): float(error) for part_id, error in zip(source_ids, errors)},
    }
    return rotation, translation, diagnostics


def _motion_frames(motion: list) -> list[dict]:
    rest_q = np.asarray([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4], dtype=float)
    current = {
        "move_arm": rest_q.copy(),
        "hold_arm": rest_q.copy(),
        "move_gripper": 0.5,
        "hold_gripper": 0.5,
        "description": "initial",
        "active_part": None,
        "motion_type": None,
    }
    frames = [dict(current)]

    for motion_type, body_type, path, active_part, description in motion:
        if body_type == "arm":
            for arm_q in path:
                current = dict(current)
                current[f"{motion_type}_arm"] = np.asarray(arm_q, dtype=float)
                current["description"] = str(description)
                current["active_part"] = None if active_part is None else int(active_part)
                current["motion_type"] = str(motion_type)
                frames.append(current)
        elif body_type == "gripper":
            current = dict(current)
            current[f"{motion_type}_gripper"] = float(path)
            current["description"] = str(description)
            current["active_part"] = None
            current["motion_type"] = str(motion_type)
            frames.append(current)
        else:
            raise ValueError(f"Unsupported Fabrica motion body type: {body_type}")
    return frames


def _gripper_opening_to_franka_joint(open_ratio: float) -> np.ndarray:
    # Fabrica's Panda gripper uses a 0..4 cm prismatic finger range. Isaac's
    # bundled Franka exposes each finger in meters.
    opening = float(np.clip(open_ratio, 0.0, 1.0)) * 0.04
    return np.asarray([opening, opening], dtype=float)


def _set_robot_state(robot, arm_q: np.ndarray, gripper_ratio: float):
    joint_positions = np.asarray(robot.articulation.get_joint_positions(), dtype=float)
    for joint_name, value in zip(ARM_JOINTS, arm_q):
        joint_positions[robot.articulation.get_dof_index(joint_name)] = float(value)
    finger_positions = _gripper_opening_to_franka_joint(gripper_ratio)
    for joint_name, value in zip(FINGER_JOINTS, finger_positions):
        joint_positions[robot.articulation.get_dof_index(joint_name)] = float(value)

    try:
        robot.articulation.set_joint_velocities(np.zeros(len(robot.articulation.dof_names), dtype=float))
    except Exception:
        pass
    robot.articulation.set_joint_positions(joint_positions)


def _zero_object_velocity(rigid_body):
    zero3 = np.zeros(3, dtype=float)
    try:
        rigid_body.set_linear_velocity(zero3)
    except Exception:
        pass
    try:
        rigid_body.set_angular_velocity(zero3)
    except Exception:
        pass
    try:
        rigid_body.unwrap().set_linear_velocity(zero3)
        rigid_body.unwrap().set_angular_velocity(zero3)
    except Exception:
        pass


def _set_part_pose(
    task,
    part_id: int,
    position: np.ndarray,
    orientation: np.ndarray,
    *,
    object_prefix: str,
    kinematic_part_ids: set[int],
):
    object_name = _part_object_name(object_prefix, part_id)
    if part_id in kinematic_part_ids:
        _set_object_xform_pose(
            task,
            object_name,
            np.asarray(position, dtype=float),
            np.asarray(orientation, dtype=float),
        )
        return
    rigid_body = task._resolve_object(object_name)  # noqa: SLF001
    _zero_object_velocity(rigid_body)
    rigid_body.set_pose(np.asarray(position, dtype=float), np.asarray(orientation, dtype=float))
    _zero_object_velocity(rigid_body)


def _get_part_pose(
    task,
    part_id: int,
    *,
    object_prefix: str,
    kinematic_part_ids: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    object_name = _part_object_name(object_prefix, part_id)
    if part_id in kinematic_part_ids:
        return _get_object_xform_pose(task, object_name)
    rigid_body = task._resolve_object(object_name)  # noqa: SLF001
    position, orientation = rigid_body.get_pose()
    return np.asarray(position, dtype=float), np.asarray(orientation, dtype=float)


def _get_object_prim(task, object_name: str):
    from isaacsim.core.utils.prims import get_prim_at_path

    scene_object = task.objects[object_name]
    prim = get_prim_at_path(scene_object.config.prim_path)
    if prim is None or not prim.IsValid():
        raise RuntimeError(f"Cannot resolve prim for {object_name}: {scene_object.config.prim_path}")
    return prim


def _get_object_xform_pose(task, object_name: str) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Usd, UsdGeom

    prim = _get_object_prim(task, object_name)
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    position = np.asarray(transform.ExtractTranslation(), dtype=float)
    quat = transform.ExtractRotationQuat()
    imaginary = quat.GetImaginary()
    orientation = np.asarray(
        [quat.GetReal(), imaginary[0], imaginary[1], imaginary[2]],
        dtype=float,
    )
    return position, orientation


def _set_object_xform_pose(task, object_name: str, position: np.ndarray, orientation: np.ndarray):
    from pxr import Gf, UsdGeom

    def _set_or_add_translate(xformable, value: np.ndarray):
        for op in xformable.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:translate":
                op.Set(Gf.Vec3d(*(float(component) for component in value)))
                return
        xformable.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*(float(component) for component in value))
        )

    def _set_or_add_orient(xformable, value: np.ndarray):
        quat_wxyz = [float(component) for component in value]
        for op in xformable.GetOrderedXformOps():
            if op.GetOpName() == "xformOp:orient":
                if op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat:
                    op.Set(Gf.Quatf(quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]))
                else:
                    op.Set(Gf.Quatd(quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]))
                return
        xformable.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3])
        )

    prim = _get_object_prim(task, object_name)
    xformable = UsdGeom.Xformable(prim)
    _set_or_add_translate(xformable, np.asarray(position, dtype=float))
    _set_or_add_orient(xformable, np.asarray(orientation, dtype=float))


def _set_fixture_pose(task, position: np.ndarray, orientation: np.ndarray):
    _set_object_xform_pose(task, "fabrica_fixture", position, orientation)


def _set_fixture_pose_via_scene_registry(task, position: np.ndarray, orientation: np.ndarray):
    rigid_body = task._resolve_object("fabrica_fixture")  # noqa: SLF001
    try:
        rigid_body.set_pose(np.asarray(position, dtype=float), np.asarray(orientation, dtype=float))
    except AttributeError:
        rigid_body.set_world_pose(np.asarray(position, dtype=float), np.asarray(orientation, dtype=float))


def _get_robot_hand_pose(task, motion_type: str) -> tuple[np.ndarray, np.ndarray]:
    robot_name = "franka_right" if motion_type == "move" else "franka_left"
    robot = task.robots[robot_name]
    position, orientation = robot.articulation.end_effector.get_pose()
    return np.asarray(position, dtype=float), np.asarray(orientation, dtype=float)


def _fabrica_hand_key(motion_type: str) -> str:
    if motion_type == "move":
        return "panda_hand_move"
    if motion_type == "hold":
        return "panda_hand_hold"
    raise ValueError(f"Unsupported motion type for hand pose: {motion_type}")


def _transformed_part_pose(
    raw_matrix_cm: np.ndarray,
    part_offset_m: np.ndarray,
    map_rotation: np.ndarray,
    map_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    raw_pos_cm, raw_quat = _matrix_to_pos_quat(raw_matrix_cm)
    raw_pos_m = raw_pos_cm * 0.01
    raw_rot = _quat_to_matrix(raw_quat)
    part_origin_pos_m = raw_pos_m + raw_rot @ part_offset_m
    position = map_rotation @ part_origin_pos_m + map_translation
    orientation = _transform_quat(map_rotation, raw_quat)
    return position, orientation


def _set_parts_from_official_traj(
    task,
    traj_frame: dict,
    part_offsets: dict[int, np.ndarray],
    map_rotation: np.ndarray,
    map_translation: np.ndarray,
    *,
    part_ids: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6),
    object_prefix: str,
    kinematic_part_ids: set[int],
):
    for part_id in part_ids:
        position, orientation = _transformed_part_pose(
            traj_frame[f"part{part_id}"],
            part_offsets[part_id],
            map_rotation,
            map_translation,
        )
        _set_part_pose(
            task,
            part_id,
            position,
            orientation,
            object_prefix=object_prefix,
            kinematic_part_ids=kinematic_part_ids,
        )


def _set_fixture_from_official_traj(
    task,
    traj_frame: dict,
    map_rotation: np.ndarray,
    map_translation: np.ndarray,
):
    position, orientation = _transformed_raw_pose(
        traj_frame["fixture"],
        map_rotation,
        map_translation,
    )
    _set_fixture_pose(task, position, orientation)


def _update_part_with_gripper_attach(
    task,
    *,
    part_id: int,
    motion_type: str,
    attach_offsets: dict[tuple[str, int], np.ndarray],
    object_prefix: str,
    kinematic_part_ids: set[int],
) -> bool:
    key = (motion_type, part_id)
    hand_position, hand_orientation = _get_robot_hand_pose(task, motion_type)
    hand_matrix = _pose_to_matrix(hand_position, hand_orientation)
    if key not in attach_offsets:
        part_position, part_orientation = _get_part_pose(
            task,
            part_id,
            object_prefix=object_prefix,
            kinematic_part_ids=kinematic_part_ids,
        )
        attach_offsets[key] = np.linalg.inv(hand_matrix) @ _pose_to_matrix(part_position, part_orientation)
    part_matrix = hand_matrix @ attach_offsets[key]
    position, orientation = _matrix_to_pos_quat(part_matrix)
    _set_part_pose(
        task,
        part_id,
        position,
        orientation,
        object_prefix=object_prefix,
        kinematic_part_ids=kinematic_part_ids,
    )
    return True


def _flush_world_for_capture(env):
    world = getattr(env.runner, "_world", None)
    if world is None:
        return
    try:
        world.step(render=True)
        return
    except Exception:
        pass
    try:
        world.render()
    except Exception:
        pass


def _to_uint8_rgba(frame) -> np.ndarray:
    frame_array = np.asarray(frame)
    if frame_array.size == 0:
        raise RuntimeError("Camera annotator returned an empty frame")
    if frame_array.ndim != 3:
        raise RuntimeError(f"Expected an HxWxC camera frame, got shape {frame_array.shape}")
    if frame_array.shape[-1] == 3:
        alpha = np.full((*frame_array.shape[:2], 1), 255, dtype=frame_array.dtype)
        frame_array = np.concatenate([frame_array, alpha], axis=-1)
    if np.issubdtype(frame_array.dtype, np.floating):
        if float(np.nanmax(frame_array)) <= 1.0 + 1e-6:
            frame_array = frame_array * 255.0
    frame_array = np.nan_to_num(frame_array, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(frame_array[..., :4], 0, 255).astype(np.uint8)


def _transformed_variant_pose(
    raw_matrix_cm: np.ndarray,
    part_offset_m: np.ndarray,
    map_rotation: np.ndarray,
    map_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return _transformed_part_pose(raw_matrix_cm, part_offset_m, map_rotation, map_translation)


def _camera_pose(
    task_cfg,
    *,
    option: str,
    object_prefix: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    object_positions = [
        np.asarray(metadata["sampled_position"], dtype=float)
        for metadata in task_cfg.object_metadata
        if str(metadata.get("name", "")).startswith(f"{object_prefix}_")
        or metadata.get("name") in {"optical_board", "assembled_manifold_preview", "fabrica_fixture"}
    ]
    center = np.mean(object_positions, axis=0) if object_positions else np.asarray([0.5, 0.0, 1.05])
    if option == "front":
        position = center + np.asarray([1.15, -1.35, 0.85], dtype=float)
    elif option == "right":
        position = center + np.asarray([1.05, 1.05, 0.75], dtype=float)
    elif option == "official_like":
        position = center + np.asarray([1.15, -1.25, 0.95], dtype=float)
    else:
        raise ValueError(f"Unsupported camera option: {option}")
    look_at = center + np.asarray([0.02, 0.0, 0.08], dtype=float)
    return tuple(position.tolist()), tuple(look_at.tolist())


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_official_motion(
    *,
    assembly_name: str,
    object_prefix: str,
    base_part_id: int | None,
    log_dir: Path,
    manifest_path: Path,
    scene_spec_path: Path | None,
    fixture_usd_path: Path | None,
    output_path: Path,
    frames_dir: Path,
    recipe: str,
    scene_profile: str,
    headless: bool,
    width: int,
    height: int,
    fps: int,
    stride: int,
    max_frames: int | None,
    mapping_mode: str,
    camera_option: str,
    robot_layout: str,
    part_replay_mode: str,
):
    motion = _load_pickle(log_dir / "motion.pkl")
    traj = np.load(log_dir / "traj.npy", allow_pickle=True)
    frames = _motion_frames(motion)
    if len(frames) != len(traj):
        raise RuntimeError(f"Expanded motion has {len(frames)} frames, but traj.npy has {len(traj)} frames")

    scene_spec = _load_scene_spec(scene_spec_path)
    if mapping_mode == "scene_spec_raw_center":
        part_offsets = _load_raw_bbox_centers(manifest_path)
    else:
        part_offsets = _load_variant_offsets(manifest_path)
    replay_part_ids = tuple(sorted(part_offsets))
    kinematic_part_ids = set() if base_part_id is None else {int(base_part_id)}
    dynamic_part_ids = tuple(part_id for part_id in replay_part_ids if part_id not in kinematic_part_ids)

    task_cfg = build_dual_franka_assembly_episode(
        recipe=recipe,
        seed=0,
        episode_idx=0,
        scene_profile=scene_profile,
    )
    if scene_spec is not None and robot_layout == "scene_spec":
        _apply_scene_spec_robot_layout(task_cfg, scene_spec)
    _force_replay_rigid_parts(
        task_cfg,
        object_prefix=object_prefix,
        dynamic_part_ids=set(dynamic_part_ids),
        kinematic_part_ids=kinematic_part_ids,
    )
    map_rotation, map_translation, mapping_diagnostics = _derive_pose_mapping(
        task_cfg,
        traj,
        part_offsets,
        mode=mapping_mode,
        object_prefix=object_prefix,
        scene_spec=scene_spec,
    )
    robot_layout_diagnostics = None
    if robot_layout == "fabrica_workcell":
        robot_layout_diagnostics = _apply_fabrica_workcell_robot_layout(
            task_cfg,
            traj[0],
            map_rotation,
            map_translation,
        )
    elif robot_layout == "recipe":
        robot_layout_diagnostics = {
            metadata.get("name", f"robot_{index}"): {
                "position": metadata.get("position"),
                "orientation": metadata.get("orientation") or metadata.get("orientation_euler"),
            }
            for index, metadata in enumerate(task_cfg.robot_metadata)
        }
    elif robot_layout == "scene_spec":
        robot_layout_diagnostics = _scene_spec_robot_layout(scene_spec) if scene_spec is not None else {}
        robot_layout_diagnostics = {
            robot_name: {
                "position": position.tolist(),
                "orientation_wxyz": orientation.tolist(),
            }
            for robot_name, (position, orientation) in robot_layout_diagnostics.items()
        }
    else:
        raise ValueError(f"Unsupported robot layout: {robot_layout}")
    fixture_loaded = _add_fabrica_fixture(task_cfg, fixture_usd_path)

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    env = _build_env(task_cfg, headless=headless)
    env.runner.render_interval = 0
    summary = None
    captured_indices: list[int] = []
    attach_offsets: dict[tuple[str, int], np.ndarray] = {}

    try:
        env.reset()

        import omni.replicator.core as rep

        camera_position, look_at = _camera_pose(task_cfg, option=camera_option, object_prefix=object_prefix)
        camera = rep.create.camera(position=camera_position, look_at=look_at)
        render_product = rep.create.render_product(camera, (width, height))
        annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")
        annotator.attach([render_product])

        rep.orchestrator.set_capture_on_play(False)
        rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)

        task_name = next(iter(env.runner.current_tasks.keys()))
        task = env.runner.current_tasks[task_name]

        total_frames = len(frames)
        render_indices = list(range(0, total_frames, max(stride, 1)))
        if render_indices[-1] != total_frames - 1:
            render_indices.append(total_frames - 1)
        if max_frames is not None:
            render_indices = render_indices[: max(max_frames, 1)]

        for frame_index in render_indices:
            frame = frames[frame_index]
            _set_robot_state(task.robots["franka_right"], frame["move_arm"], frame["move_gripper"])
            _set_robot_state(task.robots["franka_left"], frame["hold_arm"], frame["hold_gripper"])

            traj_frame = traj[frame_index]
            _set_parts_from_official_traj(
                task,
                traj_frame,
                part_offsets,
                map_rotation,
                map_translation,
                part_ids=replay_part_ids,
                object_prefix=object_prefix,
                kinematic_part_ids=kinematic_part_ids,
            )
            if fixture_loaded:
                _set_fixture_from_official_traj(
                    task,
                    traj_frame,
                    map_rotation,
                    map_translation,
                )
            if part_replay_mode == "isaac_gripper_attach" and frame["active_part"] is not None:
                _update_part_with_gripper_attach(
                    task,
                    part_id=frame["active_part"],
                    motion_type=frame["motion_type"],
                    attach_offsets=attach_offsets,
                    object_prefix=object_prefix,
                    kinematic_part_ids=kinematic_part_ids,
                )
            elif part_replay_mode == "official_traj":
                pass
            else:
                if part_replay_mode != "isaac_gripper_attach":
                    raise ValueError(f"Unsupported part replay mode: {part_replay_mode}")

            _flush_world_for_capture(env)
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)
            frame_rgba = _to_uint8_rgba(annotator.get_data())
            imageio.imwrite(frames_dir / f"rgb_{len(captured_indices):05d}.png", frame_rgba)
            captured_indices.append(frame_index)

        rep.orchestrator.wait_until_complete()
        png_paths = _encode_mp4(frames_dir=frames_dir, output_path=output_path, fps=fps)

        summary = {
            "mode": "official_fabrica_nonrl_motion_replay_in_isaacsim",
            "assembly_name": assembly_name,
            "object_prefix": object_prefix,
            "recipe": recipe,
            "scene_profile": scene_profile,
            "log_dir": str(log_dir),
            "motion_path": str(log_dir / "motion.pkl"),
            "traj_path": str(log_dir / "traj.npy"),
            "manifest_path": str(manifest_path),
            "scene_spec_path": None if scene_spec_path is None else str(scene_spec_path),
            "fixture_usd_path": None if fixture_usd_path is None else str(fixture_usd_path),
            "fixture_loaded": fixture_loaded,
            "output_path": str(output_path),
            "frames_dir": str(frames_dir),
            "captured_frame_count": len(captured_indices),
            "written_png_count": len(png_paths),
            "source_frame_count": total_frames,
            "captured_source_indices": captured_indices,
            "fps": fps,
            "stride": stride,
            "camera_width": width,
            "camera_height": height,
            "camera_option": camera_option,
            "camera_position": camera_position,
            "camera_look_at": look_at,
            "mapping": mapping_diagnostics,
            "robot_layout": robot_layout,
            "robot_layout_diagnostics": robot_layout_diagnostics,
            "part_replay_mode": part_replay_mode,
            "replay_part_ids": list(replay_part_ids),
            "dynamic_replay_part_ids": list(dynamic_part_ids),
            "kinematic_replay_part_ids": sorted(kinematic_part_ids),
            "base_part_id": base_part_id,
            "part_pose_source": (
                "Fabrica raw OBJ replay pose plus raw_bbox_center_m from manifest"
                if mapping_mode == "scene_spec_raw_center"
                else "Fabrica replay pose plus aligned-variant offset from manifest"
            ),
            "limitations": [
                "This is a replay of Fabrica's official non-RL motion plan inside Isaac Sim.",
                "During active-part motion the part pose is replayed as a gripper-fixed kinematic relation; this is not a friction-only grasp.",
                "The replay preserves the task asset scales and does not resize the board/base.",
                "The Fabrica pickup fixture is loaded from the official generated fixture mesh and replayed kinematically.",
            ],
        }
        _write_json(output_path.with_suffix(".json"), summary)
        return summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Replay Fabrica's official non-RL motion plan in Isaac Sim.")
    parser.add_argument("--assembly-name", default=DEFAULT_ASSEMBLY_NAME)
    parser.add_argument("--object-prefix", default=DEFAULT_OBJECT_PREFIX)
    parser.add_argument("--base-part-id", type=int, default=DEFAULT_BASE_PART_ID)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scene-spec", type=Path, default=DEFAULT_SCENE_SPEC)
    parser.add_argument("--fixture-usd", type=Path, default=DEFAULT_FIXTURE_USD)
    parser.add_argument("--recipe", default="fabrica_cooling_manifold")
    parser.add_argument("--scene-profile", default="taoyuan_grscenes_tabletop")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--mapping-mode",
        choices=["final_targets", "initial", "scene_spec_raw_center"],
        default="scene_spec_raw_center",
    )
    parser.add_argument("--camera-option", choices=["front", "right", "official_like"], default="official_like")
    parser.add_argument(
        "--robot-layout",
        choices=["fabrica_workcell", "scene_spec", "recipe"],
        default="fabrica_workcell",
        help="Robot base layout used to replay Fabrica joint paths.",
    )
    parser.add_argument(
        "--part-replay-mode",
        choices=["isaac_gripper_attach", "official_traj"],
        default="isaac_gripper_attach",
        help="How to replay active parts. isaac_gripper_attach keeps active parts fixed to Isaac panda_hand.",
    )
    args = parser.parse_args()

    summary = render_official_motion(
        assembly_name=args.assembly_name,
        object_prefix=args.object_prefix,
        base_part_id=args.base_part_id,
        log_dir=args.log_dir.resolve(),
        manifest_path=args.manifest.resolve(),
        scene_spec_path=None if args.scene_spec is None else args.scene_spec.resolve(),
        fixture_usd_path=None if args.fixture_usd is None else args.fixture_usd.resolve(),
        output_path=args.output.resolve(),
        frames_dir=args.frames_dir.resolve(),
        recipe=args.recipe,
        scene_profile=args.scene_profile,
        headless=args.headless,
        width=args.width,
        height=args.height,
        fps=args.fps,
        stride=max(args.stride, 1),
        max_frames=args.max_frames,
        mapping_mode=args.mapping_mode,
        camera_option=args.camera_option,
        robot_layout=args.robot_layout,
        part_replay_mode=args.part_replay_mode,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
