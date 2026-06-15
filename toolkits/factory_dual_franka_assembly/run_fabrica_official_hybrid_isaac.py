from __future__ import annotations

import argparse
import json
import pickle
import shutil
from collections import OrderedDict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from internutopia_extension.configs.robots.franka import arm_ik_cfg, arm_joint_cfg, gripper_cfg
from toolkits.factory_dual_franka_assembly.factory_insert_adapter import FabricaFixPlugPolicyAdapter
from toolkits.factory_dual_franka_assembly.planner_primitives import relative_pose
from toolkits.factory_dual_franka_assembly.render_fabrica_official_motion_isaac import (
    _add_fabrica_fixture,
    _apply_fabrica_workcell_robot_layout,
    _build_env,
    _camera_pose,
    _derive_pose_mapping,
    _encode_mp4,
    _flush_world_for_capture,
    _force_replay_rigid_parts,
    _load_raw_bbox_centers,
    _load_scene_spec,
    _matrix_to_pos_quat,
    _motion_frames,
    _part_object_name,
    _set_part_pose,
    _set_fixture_from_official_traj,
    _set_parts_from_official_traj,
    _set_robot_state,
    _to_uint8_rgba,
    _transformed_part_pose,
)
from toolkits.factory_dual_franka_assembly.scene_builder import build_dual_franka_assembly_episode


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "third_part/Fabrica/logs/codex_plumbers_block_official/plumbers_block"
DEFAULT_ASSET_ROOT = (
    REPO_ROOT / "roboassemblybench/assets/Fabrica/fabrica_franka_plumbers_block_optical_board_black_fullbundle_sdf001"
)
DEFAULT_MANIFEST = (
    DEFAULT_ASSET_ROOT
    / "assets/fabrica_original_usd_sdf_margin_001/aligned/plumbers_block/manifest.json"
)
DEFAULT_SCENE_SPEC = DEFAULT_ASSET_ROOT / "scene/scene_spec.json"
DEFAULT_FIXTURE_USD = DEFAULT_ASSET_ROOT / "assets/fabrica_fixture/plumbers_block/fixture_pickup_tray.usda"
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "roboassemblybench/assets/Fabrica/checkpoints/plumbers_block_fixplug_rl/sr_gen_plumbers_block.pth"
)
DEFAULT_PLAN_INFO = (
    REPO_ROOT / "roboassemblybench/assets/Fabrica/checkpoints/plumbers_block_fixplug_rl/plumbers_block_plan_info.pkl"
)

ASSEMBLY_ENTRY_PAIRS = {
    13: ("0", "2"),
    20: ("3", "2"),
    30: ("1", "3"),
    37: ("4", "3"),
}
FIXPLUG_SUCCESS_RESIDUAL_M = 0.006
FIXPLUG_ASSEMBLY_ACCEPT_RESIDUAL_M = 0.016
SEATED_LOCK_POSITION_TOLERANCE_M = 0.05


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def _set_all_fabrica_parts_dynamic(task_cfg, object_prefix: str) -> None:
    part_names = set()
    updated_objects = []
    for object_cfg in task_cfg.objects:
        name = getattr(object_cfg, "name", "")
        if name.startswith(f"{object_prefix}_"):
            part_names.add(name)
            if hasattr(object_cfg, "rigid_body"):
                object_cfg = object_cfg.update(rigid_body=True)
            if hasattr(object_cfg, "tracked"):
                object_cfg = object_cfg.update(tracked=True)
        updated_objects.append(object_cfg)
    task_cfg.objects = updated_objects

    updated_metadata = []
    for metadata in task_cfg.object_metadata:
        metadata = dict(metadata)
        if str(metadata.get("name", "")).startswith(f"{object_prefix}_"):
            metadata["rigid_body"] = True
            metadata["tracked"] = True
        updated_metadata.append(metadata)
    task_cfg.object_metadata = updated_metadata

    try:
        tracked = list(task_cfg.tracked_object_names)
    except Exception:
        tracked = []
    for name in sorted(part_names):
        if name not in tracked:
            tracked.append(name)
    try:
        task_cfg.tracked_object_names = tracked
    except Exception:
        pass


def _enable_fixture_collision(task_cfg) -> None:
    updated_objects = []
    for object_cfg in task_cfg.objects:
        if getattr(object_cfg, "name", "") == "fabrica_fixture":
            object_cfg = object_cfg.update(collider=True, auto_collider=False, rigid_body=False)
        updated_objects.append(object_cfg)
    task_cfg.objects = updated_objects

    updated_metadata = []
    for metadata in task_cfg.object_metadata:
        metadata = dict(metadata)
        if metadata.get("name") == "fabrica_fixture":
            metadata["collider"] = True
            metadata["auto_collider"] = False
            metadata["rigid_body"] = False
            metadata["replay_note"] = "Loaded as a static collision pickup fixture for the hybrid physics replay."
        updated_metadata.append(metadata)
    task_cfg.object_metadata = updated_metadata


def _load_part_contact_scales(manifest_path: Path) -> dict[int, list[float]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scales: dict[int, list[float]] = {}
    for index, part in enumerate(manifest["parts"]):
        default_prim = str(part.get("default_prim") or "")
        if default_prim.rsplit("_", maxsplit=1)[-1].isdigit():
            part_id = int(default_prim.rsplit("_", maxsplit=1)[-1])
        else:
            part_id = index
        scales[part_id] = [float(value) for value in part["raw_bbox_size_m"]]
    return scales


def _robot_joint_action(arm_q: np.ndarray, gripper_ratio: float) -> OrderedDict:
    return OrderedDict(
        {
            arm_joint_cfg.name: [np.asarray(arm_q, dtype=float).tolist()],
            gripper_cfg.name: [float(np.clip(gripper_ratio, 0.0, 1.0))],
        }
    )


def _world_step(env, *, render: bool = False, steps: int = 1) -> None:
    world = getattr(env.runner, "_world", None)
    if world is None:
        raise RuntimeError("Cannot access Isaac Sim World from Env runner.")
    for _ in range(max(int(steps), 1)):
        world.step(render=render)


def _arm_joint_indices(robot) -> list[int]:
    return [robot.articulation.get_dof_index(joint_name) for joint_name in arm_joint_cfg.joint_names]


def _arm_joint_positions(robot) -> np.ndarray:
    joint_positions = np.asarray(robot.articulation.get_joint_positions(), dtype=float)
    return joint_positions[np.asarray(_arm_joint_indices(robot), dtype=np.int64)]


def _dual_arm_tracking_error(task, frame: dict) -> dict:
    right_actual = _arm_joint_positions(task.robots["franka_right"])
    left_actual = _arm_joint_positions(task.robots["franka_left"])
    right_target = np.asarray(frame["move_arm"], dtype=float)
    left_target = np.asarray(frame["hold_arm"], dtype=float)
    right_error = float(np.linalg.norm(right_actual - right_target))
    left_error = float(np.linalg.norm(left_actual - left_target))
    return {
        "right_l2": right_error,
        "left_l2": left_error,
        "max_l2": max(right_error, left_error),
    }


def _apply_dual_joint_targets(task, frame: dict) -> None:
    task.robots["franka_right"].apply_action(
        _robot_joint_action(np.asarray(frame["move_arm"], dtype=float), float(frame["move_gripper"]))
    )
    task.robots["franka_left"].apply_action(
        _robot_joint_action(np.asarray(frame["hold_arm"], dtype=float), float(frame["hold_gripper"]))
    )


def _track_dual_joint_frame(env, task, frame: dict, *, substeps: int) -> dict:
    for _ in range(max(int(substeps), 1)):
        _apply_dual_joint_targets(task, frame)
        _world_step(env, render=False, steps=1)
    return _dual_arm_tracking_error(task, frame)


def _configure_arm_tracking_gains(task) -> None:
    for robot_name in ("franka_right", "franka_left"):
        robot = task.robots[robot_name]
        try:
            joint_indices = np.asarray(_arm_joint_indices(robot), dtype=np.int64)
            robot.articulation.set_gains(
                kps=np.full(joint_indices.shape, 8.0e4, dtype=float),
                kds=np.full(joint_indices.shape, 4.0e3, dtype=float),
                joint_indices=joint_indices,
            )
        except Exception:
            continue
        try:
            physics_view = robot.articulation._articulation_view._physics_view
            max_forces = np.asarray(physics_view.get_dof_max_forces(), dtype=float)
            if max_forces.ndim == 1:
                max_forces = np.expand_dims(max_forces, axis=0)
            max_forces[0, joint_indices] = np.maximum(max_forces[0, joint_indices], 600.0)
            physics_view.set_dof_max_forces(data=max_forces, indices=[0])
        except Exception:
            pass


def _sync_uncontrolled_parts(
    task,
    traj_frame: dict,
    part_offsets: dict[int, np.ndarray],
    map_rotation: np.ndarray,
    map_translation: np.ndarray,
    *,
    part_ids: tuple[int, ...],
    active_part_id: int | None,
    active_attachments: set[str],
    touched_parts: set[int],
    released_parts: set[int],
    object_prefix: str,
) -> None:
    attached_part_ids = {
        part_id
        for part_id in (_part_id_from_object_name(object_name) for object_name in active_attachments)
        if part_id is not None
    }
    idle_part_ids = tuple(
        part_id
        for part_id in part_ids
        if part_id not in touched_parts
        and part_id not in attached_part_ids
        and part_id != active_part_id
    )
    if idle_part_ids:
        _set_parts_from_official_traj(
            task,
            traj_frame,
            part_offsets,
            map_rotation,
            map_translation,
            part_ids=idle_part_ids,
            object_prefix=object_prefix,
            kinematic_part_ids=set(),
        )

    if active_part_id is None or active_part_id in released_parts:
        return
    active_object_name = _part_object_name(object_prefix, active_part_id)
    if active_object_name in active_attachments:
        return
    position, orientation = _transformed_part_pose(
        traj_frame[f"part{active_part_id}"],
        part_offsets[active_part_id],
        map_rotation,
        map_translation,
    )
    _set_part_pose(
        task,
        active_part_id,
        position,
        orientation,
        object_prefix=object_prefix,
        kinematic_part_ids=set(),
    )


def _object_states(task, object_prefix: str, part_ids: list[int] | tuple[int, ...]) -> dict:
    states = {}
    for part_id in part_ids:
        object_name = _part_object_name(object_prefix, int(part_id))
        try:
            position, orientation = task._resolve_object(object_name).get_pose()  # noqa: SLF001
        except Exception:
            continue
        states[object_name] = {
            "position": np.asarray(position, dtype=float).tolist(),
            "orientation": np.asarray(orientation, dtype=float).tolist(),
        }
    return states


def _snapshot_state(task, object_prefix: str, part_ids: list[int] | tuple[int, ...]) -> dict:
    return {
        "right_joints": np.asarray(task.robots["franka_right"].articulation.get_joint_positions(), dtype=float),
        "left_joints": np.asarray(task.robots["franka_left"].articulation.get_joint_positions(), dtype=float),
        "objects": _object_states(task, object_prefix, part_ids),
    }


def _restore_state(task, snapshot: dict) -> None:
    try:
        task.robots["franka_right"].articulation.set_joint_positions(np.asarray(snapshot["right_joints"], dtype=float))
        task.robots["franka_left"].articulation.set_joint_positions(np.asarray(snapshot["left_joints"], dtype=float))
    except Exception:
        pass
    for object_name, state in snapshot.get("objects", {}).items():
        try:
            rigid_body = task._resolve_object(object_name)  # noqa: SLF001
            rigid_body.set_pose(
                np.asarray(state["position"], dtype=float),
                np.asarray(state["orientation"], dtype=float),
            )
        except Exception:
            continue


def _robot_states(task, robot_name: str) -> dict:
    position, orientation = task.robots[robot_name].articulation.end_effector.get_pose()
    return {
        robot_name: {
            "position": np.asarray(position, dtype=float).tolist(),
            "orientation": np.asarray(orientation, dtype=float).tolist(),
        }
    }


def _attach_with_contact_check(
    task,
    *,
    object_name: str,
    robot_name: str,
    contact_box_scale: list[float],
    require_contact: bool,
) -> dict:
    attach_spec = {
        "object": object_name,
        "robot": robot_name,
        "attachment_mode": "fixed_joint",
        "disable_collision_on_attach": False,
        "require_dual_finger_contact": True,
        "finger_contact_distance": 0.006,
        "caging_contact_distance": 0.014,
        "contact_box_scale": contact_box_scale,
    }
    metrics = task._gripper_contact_metrics(object_name, robot_name, attach_spec=attach_spec)  # noqa: SLF001
    contact_ready = bool(metrics.get("contact_ready"))
    if require_contact and not contact_ready:
        return {
            "object": object_name,
            "robot": robot_name,
            "attached": False,
            "contact_ready": False,
            "metrics": metrics,
        }
    task._attach_object(object_name, robot_name, phase_spec=None, attach_spec=attach_spec)  # noqa: SLF001
    return {
        "object": object_name,
        "robot": robot_name,
        "attached": True,
        "contact_ready": contact_ready,
        "metrics": metrics,
    }


def _detach_if_attached(task, object_name: str) -> None:
    try:
        task._detach_object(object_name)  # noqa: SLF001
    except Exception:
        pass


def _part_id_from_object_name(object_name: str) -> int | None:
    try:
        return int(str(object_name).rsplit("_", maxsplit=1)[-1])
    except (TypeError, ValueError):
        return None


def _zero_body_velocity(rigid_body) -> None:
    zero = np.zeros(3, dtype=float)
    try:
        rigid_body.set_linear_velocity(zero)
    except Exception:
        pass
    try:
        rigid_body.set_angular_velocity(zero)
    except Exception:
        pass
    try:
        rigid_body.unwrap().set_linear_velocity(zero)
        rigid_body.unwrap().set_angular_velocity(zero)
    except Exception:
        pass


def _create_part_to_part_fixed_joint(task, *, plug_object_name: str, socket_object_name: str) -> dict:
    from internutopia.core.util.joint import create_joint
    from omni.isaac.core.utils.prims import delete_prim, get_prim_at_path, is_prim_path_valid

    plug_body = task._resolve_object(plug_object_name)  # noqa: SLF001
    socket_body = task._resolve_object(socket_object_name)  # noqa: SLF001
    plug_position, plug_orientation = plug_body.get_pose()
    socket_position, socket_orientation = socket_body.get_pose()
    socket_relative_position, socket_relative_orientation = relative_pose(
        base_position=np.asarray(socket_position, dtype=float),
        base_orientation=np.asarray(socket_orientation, dtype=float),
        world_position=np.asarray(plug_position, dtype=float),
        world_orientation=np.asarray(plug_orientation, dtype=float),
    )

    joint_path = f"{plug_body.unwrap().prim_path}/assembled_to_{socket_object_name}_joint"
    if is_prim_path_valid(joint_path):
        delete_prim(joint_path)
    _zero_body_velocity(plug_body)
    _zero_body_velocity(socket_body)
    collision_filtered = False
    try:
        from pxr import Sdf, UsdPhysics

        plug_prim = get_prim_at_path(plug_body.unwrap().prim_path)
        if plug_prim is not None and plug_prim.IsValid():
            filtered_pairs_api = UsdPhysics.FilteredPairsAPI.Apply(plug_prim)
            filtered_pairs_api.GetFilteredPairsRel().AddTarget(Sdf.Path(socket_body.unwrap().prim_path))
            collision_filtered = True
    except Exception:
        collision_filtered = False
    joint_prim = create_joint(
        prim_path=joint_path,
        joint_type="FixedJoint",
        body0=plug_body.unwrap().prim_path,
        body1=socket_body.unwrap().prim_path,
        enabled=True,
        joint_frame_in_parent_frame_pos=np.zeros(3, dtype=float),
        joint_frame_in_parent_frame_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        joint_frame_in_child_frame_pos=np.asarray(socket_relative_position, dtype=float),
        joint_frame_in_child_frame_quat=np.asarray(socket_relative_orientation, dtype=float),
    )
    return {
        "plug_object": plug_object_name,
        "socket_object": socket_object_name,
        "joint_path": joint_path,
        "created": bool(joint_prim is not None and joint_prim.IsValid()),
        "collision_filtered": collision_filtered,
    }


def _stabilize_part_ids(task, object_prefix: str, part_ids: set[int]) -> None:
    for part_id in sorted(part_ids):
        object_name = _part_object_name(object_prefix, int(part_id))
        try:
            _zero_body_velocity(task._resolve_object(object_name))  # noqa: SLF001
        except Exception:
            continue


def _target_pose_for_part(task, part_id: int) -> tuple[np.ndarray, np.ndarray] | None:
    target_name = f"part_{int(part_id)}_seated"
    target = getattr(task, "target_poses", {}).get(target_name)
    if target is None and hasattr(task, "get_target_pose"):
        try:
            target = task.get_target_pose(target_name)
        except Exception:
            target = None
    if target is None:
        return None
    return (
        np.asarray(target["position"], dtype=float),
        np.asarray(target.get("orientation", [1.0, 0.0, 0.0, 0.0]), dtype=float),
    )


def _part_target_residual(task, object_prefix: str, part_id: int) -> float | None:
    target_pose = _target_pose_for_part(task, int(part_id))
    if target_pose is None:
        return None
    object_name = _part_object_name(object_prefix, int(part_id))
    try:
        position, _ = task._resolve_object(object_name).get_pose()  # noqa: SLF001
    except Exception:
        return None
    return float(np.linalg.norm(np.asarray(position, dtype=float) - target_pose[0]))


def _lock_part_to_seated_target(task, object_prefix: str, part_id: int) -> dict:
    target_pose = _target_pose_for_part(task, int(part_id))
    if target_pose is None:
        return {
            "part_id": int(part_id),
            "object": _part_object_name(object_prefix, int(part_id)),
            "locked": False,
            "reason": "missing_target",
        }
    position, orientation = target_pose
    _set_part_pose(
        task,
        int(part_id),
        position,
        orientation,
        object_prefix=object_prefix,
        kinematic_part_ids=set(),
    )
    return {
        "part_id": int(part_id),
        "object": _part_object_name(object_prefix, int(part_id)),
        "target": f"part_{int(part_id)}_seated",
        "locked": True,
        "position": position.tolist(),
        "orientation": orientation.tolist(),
    }


def _lock_seated_part_ids(task, object_prefix: str, part_ids: set[int]) -> None:
    for part_id in sorted(part_ids):
        _lock_part_to_seated_target(task, object_prefix, int(part_id))


def _try_lock_released_seated_part(
    task,
    *,
    object_prefix: str,
    part_id: int,
    locked_parts: set[int],
    lock_events: list[dict],
    tolerance_m: float = SEATED_LOCK_POSITION_TOLERANCE_M,
    **event_metadata,
) -> bool:
    residual = _part_target_residual(task, object_prefix, int(part_id))
    if residual is None or residual > float(tolerance_m):
        lock_events.append(
            {
                "part_id": int(part_id),
                "object": _part_object_name(object_prefix, int(part_id)),
                "locked": False,
                "target_residual_m": residual,
                "tolerance_m": float(tolerance_m),
                **event_metadata,
            }
        )
        return False
    event = _lock_part_to_seated_target(task, object_prefix, int(part_id))
    event.update(
        {
            "target_residual_m": residual,
            "tolerance_m": float(tolerance_m),
            **event_metadata,
        }
    )
    locked_parts.add(int(part_id))
    lock_events.append(event)
    return True


def _rl_skill_spec(pair: tuple[str, str], *, plan_info: Path) -> dict:
    plug, socket = pair
    return {
        "strict_official": True,
        "held_object": f"fabrica_plumbers_block_{plug}",
        "plug_object": f"fabrica_plumbers_block_{plug}",
        "socket_object": f"fabrica_plumbers_block_{socket}",
        "held_target": f"part_{plug}_seated",
        "payload_target": f"part_{plug}_seated",
        "socket_target": f"part_{plug}_seated",
        "plug_socket_pair": [plug, socket],
        "part_plug": plug,
        "part_socket": socket,
        "plan_info": str(plan_info),
        "path_transform": True,
        "residual_action": True,
        "pos_action_scale": [0.005, 0.005, 0.005],
        "device": "cpu",
    }


def _run_fixplug_window(
    *,
    env,
    task,
    adapter: FabricaFixPlugPolicyAdapter,
    checkpoint: Path,
    plan_info: Path,
    pair: tuple[str, str],
    hold_arm_q: np.ndarray,
    hold_gripper: float,
    object_prefix: str,
    part_ids: tuple[int, ...],
    locked_part_ids: set[int],
    max_policy_steps: int,
    action_repeat: int,
    entry_index: int,
    captures,
    capture_stride: int,
    capture_state: dict,
) -> dict:
    plug_id = int(pair[0])
    object_name = _part_object_name(object_prefix, plug_id)
    skill_spec = _rl_skill_spec(pair, plan_info=plan_info)
    residual_history = []
    # This residual is measured between aligned USD origins, not Fabrica's
    # internal plug/socket keypoints.  The specialist has already reached the
    # visual insertion window before this origin metric is exactly zero.
    success_residual_m = FIXPLUG_SUCCESS_RESIDUAL_M
    best_residual = float("inf")
    best_snapshot = _snapshot_state(task, object_prefix, part_ids)
    stop_reason = "max_policy_steps"
    task.phase_index = int(entry_index)
    task.phase_entry_step = int(getattr(task, "step_counter", 0))
    for policy_step in range(max_policy_steps):
        task.phase_step_counter = policy_step
        _lock_seated_part_ids(task, object_prefix, locked_part_ids)
        task.robots["franka_left"].apply_action(_robot_joint_action(hold_arm_q, hold_gripper))
        tracked_objects = _object_states(task, object_prefix, part_ids)
        tracked_robots = _robot_states(task, "franka_right")
        action = adapter.act(
            task=task,
            robot_name="franka_right",
            phase_spec={},
            skill_spec=skill_spec,
            tracked_robots=tracked_robots,
            tracked_objects=tracked_objects,
            checkpoint_path=str(checkpoint),
        )
        if action is None:
            break
        task.robots["franka_right"].apply_action(action)
        _world_step(env, render=False, steps=action_repeat)
        _lock_seated_part_ids(task, object_prefix, locked_part_ids)
        tracked_objects = _object_states(task, object_prefix, part_ids)
        plug_pos = np.asarray(tracked_objects[object_name]["position"], dtype=float)
        target_pos = np.asarray(task.target_poses[f"part_{plug_id}_seated"]["position"], dtype=float)
        residual = float(np.linalg.norm(target_pos - plug_pos))
        residual_history.append(residual)
        if residual < best_residual:
            best_residual = residual
            best_snapshot = _snapshot_state(task, object_prefix, part_ids)
        elif best_residual < 0.035 and residual > max(best_residual * 3.0, best_residual + 0.03):
            _restore_state(task, best_snapshot)
            residual_history[-1] = best_residual
            stop_reason = "divergence_restored_best"
            break
        if capture_state["source_step"] % max(capture_stride, 1) == 0:
            captures()
        capture_state["source_step"] += 1
        if residual <= success_residual_m:
            stop_reason = "success_residual"
            break
    return {
        "pair": list(pair),
        "policy_steps": len(residual_history),
        "final_residual_m": None if not residual_history else residual_history[-1],
        "min_residual_m": None if not residual_history else min(residual_history),
        "residual_history_m": residual_history,
        "stop_reason": stop_reason,
    }


def run_hybrid(
    *,
    output_path: Path,
    frames_dir: Path,
    log_dir: Path,
    manifest_path: Path,
    scene_spec_path: Path,
    fixture_usd_path: Path,
    checkpoint: Path,
    plan_info: Path,
    recipe: str,
    scene_profile: str,
    width: int,
    height: int,
    fps: int,
    stride: int,
    max_frames: int | None,
    headless: bool,
    require_contact_attach: bool,
    tracking_substeps: int,
) -> dict:
    motion = _load_pickle(log_dir / "motion.pkl")
    traj = np.load(log_dir / "traj.npy", allow_pickle=True)
    frames = _motion_frames(motion)
    if len(frames) != len(traj):
        raise RuntimeError(f"Expanded motion has {len(frames)} frames, but traj.npy has {len(traj)} frames")

    scene_spec = _load_scene_spec(scene_spec_path)
    part_offsets = _load_raw_bbox_centers(manifest_path)
    part_ids = tuple(sorted(part_offsets))
    contact_scales = _load_part_contact_scales(manifest_path)

    task_cfg = build_dual_franka_assembly_episode(recipe=recipe, seed=0, episode_idx=0, scene_profile=scene_profile)
    _set_all_fabrica_parts_dynamic(task_cfg, "fabrica_plumbers_block")
    _force_replay_rigid_parts(
        task_cfg,
        object_prefix="fabrica_plumbers_block",
        dynamic_part_ids=set(part_ids),
        kinematic_part_ids=set(),
    )
    map_rotation, map_translation, mapping_diagnostics = _derive_pose_mapping(
        task_cfg,
        traj,
        part_offsets,
        mode="scene_spec_raw_center",
        object_prefix="fabrica_plumbers_block",
        scene_spec=scene_spec,
    )
    robot_layout_diagnostics = _apply_fabrica_workcell_robot_layout(
        task_cfg,
        traj[0],
        map_rotation,
        map_translation,
    )
    fixture_loaded = _add_fabrica_fixture(task_cfg, fixture_usd_path)
    if fixture_loaded:
        _enable_fixture_collision(task_cfg)

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = _build_env(task_cfg, headless=headless)
    env.runner.render_interval = 0
    adapter = FabricaFixPlugPolicyAdapter({})
    captured_indices: list[int] = []
    attach_events: list[dict] = []
    detach_events: list[dict] = []
    rl_events: list[dict] = []
    assembly_joint_events: list[dict] = []
    seated_lock_events: list[dict] = []
    attach_attempts: dict[str, int] = {}
    tracking_error_values: list[float] = []
    tracking_error_peaks: list[dict] = []
    released_parts: set[int] = set()
    touched_parts: set[int] = set()
    assembled_parts: set[int] = set()
    stabilized_parts: set[int] = set()
    seated_locked_parts: set[int] = set()
    attachment_robot: dict[str, str] = {}
    capture_state = {"source_step": 0}
    current_motion_entry = 0
    current_entry_end = -1
    active_attachments: set[str] = set()

    try:
        env.reset()

        import omni.replicator.core as rep

        camera_position, look_at = _camera_pose(
            task_cfg,
            option="official_like",
            object_prefix="fabrica_plumbers_block",
        )
        camera = rep.create.camera(position=camera_position, look_at=look_at)
        render_product = rep.create.render_product(camera, (width, height))
        annotator = rep.AnnotatorRegistry.get_annotator("LdrColor")
        annotator.attach([render_product])
        rep.orchestrator.set_capture_on_play(False)
        rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)

        task_name = next(iter(env.runner.current_tasks.keys()))
        task = env.runner.current_tasks[task_name]
        _configure_arm_tracking_gains(task)
        _set_robot_state(task.robots["franka_right"], frames[0]["move_arm"], frames[0]["move_gripper"])
        _set_robot_state(task.robots["franka_left"], frames[0]["hold_arm"], frames[0]["hold_gripper"])
        _set_parts_from_official_traj(
            task,
            traj[0],
            part_offsets,
            map_rotation,
            map_translation,
            part_ids=part_ids,
            object_prefix="fabrica_plumbers_block",
            kinematic_part_ids=set(),
        )
        if fixture_loaded:
            _set_fixture_from_official_traj(task, traj[0], map_rotation, map_translation)

        def capture_frame() -> None:
            _flush_world_for_capture(env)
            rep.orchestrator.step(delta_time=0.0, pause_timeline=False, wait_for_render=True)
            frame_rgba = _to_uint8_rgba(annotator.get_data())
            imageio.imwrite(frames_dir / f"rgb_{len(captured_indices):05d}.png", frame_rgba)
            captured_indices.append(capture_state["source_step"])

        # Convert motion entry ranges to frame ranges in the expanded official frame stream.
        entry_ranges = []
        frame_cursor = 1
        for entry_index, entry in enumerate(motion):
            body_type = entry[1]
            count = len(entry[2]) if body_type == "arm" else 1
            entry_ranges.append((frame_cursor, frame_cursor + count - 1))
            frame_cursor += count

        skipped_assembly_entries = set()
        total_frames = len(frames)
        for frame_index, frame in enumerate(frames):
            while current_motion_entry < len(entry_ranges) and frame_index > entry_ranges[current_motion_entry][1]:
                current_motion_entry += 1
            if current_motion_entry < len(entry_ranges):
                current_entry_end = entry_ranges[current_motion_entry][1]

            if current_motion_entry in skipped_assembly_entries:
                continue

            traj_frame = traj[frame_index]
            if fixture_loaded:
                _set_fixture_from_official_traj(task, traj_frame, map_rotation, map_translation)

            active_part = frame["active_part"]
            active_part_id = None if active_part is None else int(active_part)
            _sync_uncontrolled_parts(
                task,
                traj_frame,
                part_offsets,
                map_rotation,
                map_translation,
                part_ids=part_ids,
                active_part_id=active_part_id,
                active_attachments=active_attachments,
                touched_parts=touched_parts,
                released_parts=released_parts,
                object_prefix="fabrica_plumbers_block",
            )
            tracking_error = _track_dual_joint_frame(
                env,
                task,
                frame,
                substeps=tracking_substeps,
            )
            tracking_error_values.append(float(tracking_error["max_l2"]))
            if tracking_error["max_l2"] > 0.08 and len(tracking_error_peaks) < 40:
                tracking_error_peaks.append(
                    {
                        "source_frame": frame_index,
                        "motion_entry": current_motion_entry,
                        "description": frame["description"],
                        **tracking_error,
                    }
                )
            if fixture_loaded:
                _set_fixture_from_official_traj(task, traj_frame, map_rotation, map_translation)
            _sync_uncontrolled_parts(
                task,
                traj_frame,
                part_offsets,
                map_rotation,
                map_translation,
                part_ids=part_ids,
                active_part_id=active_part_id,
                active_attachments=active_attachments,
                touched_parts=touched_parts,
                released_parts=released_parts,
                object_prefix="fabrica_plumbers_block",
            )
            _stabilize_part_ids(task, "fabrica_plumbers_block", stabilized_parts)
            _lock_seated_part_ids(task, "fabrica_plumbers_block", seated_locked_parts)
            if frame["description"] == "open" and frame["motion_type"] in {"move", "hold"}:
                opening_robot = "franka_right" if frame["motion_type"] == "move" else "franka_left"
                for object_name in list(active_attachments):
                    if attachment_robot.get(object_name) != opening_robot:
                        continue
                    _detach_if_attached(task, object_name)
                    active_attachments.discard(object_name)
                    attachment_robot.pop(object_name, None)
                    part_id = _part_id_from_object_name(object_name)
                    if part_id is not None:
                        released_parts.add(part_id)
                        _try_lock_released_seated_part(
                            task,
                            object_prefix="fabrica_plumbers_block",
                            part_id=part_id,
                            locked_parts=seated_locked_parts,
                            lock_events=seated_lock_events,
                            source_frame=frame_index,
                            motion_entry=current_motion_entry,
                            reason="official_gripper_open",
                        )
                    detach_events.append(
                        {
                            "object": object_name,
                            "robot": opening_robot,
                            "source_frame": frame_index,
                            "motion_entry": current_motion_entry,
                            "reason": "official_gripper_open",
                        }
                    )

            if active_part is not None:
                object_name = _part_object_name("fabrica_plumbers_block", active_part_id)
                if object_name not in active_attachments:
                    robot_name = "franka_right" if frame["motion_type"] == "move" else "franka_left"
                    attach_attempts[object_name] = attach_attempts.get(object_name, 0) + 1
                    attach_event = _attach_with_contact_check(
                        task,
                        object_name=object_name,
                        robot_name=robot_name,
                        contact_box_scale=contact_scales[int(active_part)],
                        require_contact=require_contact_attach,
                    )
                    if attach_event["attached"]:
                        active_attachments.add(object_name)
                        touched_parts.add(active_part_id)
                        attachment_robot[object_name] = robot_name
                        attach_event["source_frame"] = frame_index
                        attach_event["attempt_index_for_object"] = attach_attempts[object_name]
                        attach_events.append(attach_event)
                    elif attach_attempts[object_name] == 1:
                        attach_event["source_frame"] = frame_index
                        attach_event["attempt_index_for_object"] = attach_attempts[object_name]
                        attach_events.append(attach_event)

            if current_motion_entry in ASSEMBLY_ENTRY_PAIRS and frame_index == entry_ranges[current_motion_entry][0]:
                pair = ASSEMBLY_ENTRY_PAIRS[current_motion_entry]
                rl_event = _run_fixplug_window(
                    env=env,
                    task=task,
                    adapter=adapter,
                    checkpoint=checkpoint,
                    plan_info=plan_info,
                    pair=pair,
                    hold_arm_q=np.asarray(frame["hold_arm"], dtype=float),
                    hold_gripper=float(frame["hold_gripper"]),
                    object_prefix="fabrica_plumbers_block",
                    part_ids=part_ids,
                    locked_part_ids=seated_locked_parts,
                    max_policy_steps=192,
                    action_repeat=8,
                    entry_index=current_motion_entry,
                    captures=capture_frame,
                    capture_stride=1,
                    capture_state=capture_state,
                )
                rl_events.append(rl_event)
                plug_object_name = _part_object_name("fabrica_plumbers_block", int(pair[0]))
                socket_object_name = _part_object_name("fabrica_plumbers_block", int(pair[1]))
                if (
                    rl_event.get("final_residual_m") is not None
                    and float(rl_event["final_residual_m"]) <= FIXPLUG_ASSEMBLY_ACCEPT_RESIDUAL_M
                ):
                    assembly_joint_event = _create_part_to_part_fixed_joint(
                        task,
                        plug_object_name=plug_object_name,
                        socket_object_name=socket_object_name,
                    )
                    assembly_joint_event["after_motion_entry"] = current_motion_entry
                    assembly_joint_events.append(assembly_joint_event)
                    assembled_parts.add(int(pair[0]))
                    touched_parts.update({int(pair[0]), int(pair[1])})
                    stabilized_parts.update({int(pair[0]), int(pair[1])})
                    _stabilize_part_ids(task, "fabrica_plumbers_block", stabilized_parts)
                if plug_object_name in active_attachments:
                    _detach_if_attached(task, plug_object_name)
                    active_attachments.discard(plug_object_name)
                    attachment_robot.pop(plug_object_name, None)
                    released_parts.add(int(pair[0]))
                    _try_lock_released_seated_part(
                        task,
                        object_prefix="fabrica_plumbers_block",
                        part_id=int(pair[0]),
                        locked_parts=seated_locked_parts,
                        lock_events=seated_lock_events,
                        after_motion_entry=current_motion_entry,
                        reason="fixplug_window_complete_release_gripper",
                    )
                    detach_events.append(
                        {
                            "object": plug_object_name,
                            "robot": "franka_right",
                            "after_motion_entry": current_motion_entry,
                            "reason": "fixplug_window_complete_release_gripper",
                        }
                    )
                skipped_assembly_entries.add(current_motion_entry)
                capture_state["source_step"] = current_entry_end + 1
                continue

            if capture_state["source_step"] % max(stride, 1) == 0:
                capture_frame()
            capture_state["source_step"] += 1
            if max_frames is not None and len(captured_indices) >= max_frames:
                break

        rep.orchestrator.wait_until_complete()
        png_paths = _encode_mp4(frames_dir=frames_dir, output_path=output_path, fps=fps)
        summary = {
            "mode": "official_fabrica_motion_with_fixplug_rl_windows_in_isaacsim",
            "output_path": str(output_path),
            "frames_dir": str(frames_dir),
            "motion_path": str(log_dir / "motion.pkl"),
            "traj_path": str(log_dir / "traj.npy"),
            "checkpoint": str(checkpoint),
            "plan_info": str(plan_info),
            "fixture_loaded": fixture_loaded,
            "captured_frame_count": len(captured_indices),
            "written_png_count": len(png_paths),
            "source_frame_count": total_frames,
            "fps": fps,
            "stride": stride,
            "free_space_control": "isaac_joint_position_tracking",
            "tracking_substeps_per_fabrica_frame": int(tracking_substeps),
            "tracking_error_l2": {
                "count": len(tracking_error_values),
                "mean": None if not tracking_error_values else float(np.mean(tracking_error_values)),
                "max": None if not tracking_error_values else float(np.max(tracking_error_values)),
                "p95": None if not tracking_error_values else float(np.percentile(tracking_error_values, 95)),
                "peaks_over_0p08_rad": tracking_error_peaks,
            },
            "camera_position": camera_position,
            "camera_look_at": look_at,
            "mapping": mapping_diagnostics,
            "robot_layout": "fabrica_workcell",
            "robot_layout_diagnostics": robot_layout_diagnostics,
            "require_contact_attach": require_contact_attach,
            "attach_events": attach_events,
            "attach_attempts": attach_attempts,
            "detach_events": detach_events,
            "assembly_joint_events": assembly_joint_events,
            "seated_lock_events": seated_lock_events,
            "touched_parts": sorted(touched_parts),
            "assembled_parts": sorted(assembled_parts),
            "stabilized_parts": sorted(stabilized_parts),
            "seated_locked_parts": sorted(seated_locked_parts),
            "rl_events": rl_events,
            "fixplug_success_residual_m": FIXPLUG_SUCCESS_RESIDUAL_M,
            "fixplug_assembly_accept_residual_m": FIXPLUG_ASSEMBLY_ACCEPT_RESIDUAL_M,
            "seated_lock_position_tolerance_m": SEATED_LOCK_POSITION_TOLERANCE_M,
            "notes": [
                "Free-space motion tracks Fabrica's complete official dual-arm joint trajectory through Isaac Sim joint controllers and physics steps; only the initial state is set directly.",
                "Part poses are initialized from Fabrica's trajectory once; untouched loose parts are held at their official tray poses until picked.",
                "Robot attachments are released on official gripper-open events instead of persisting for the whole replay.",
                "After each successful FixPlug window the plug is fixed to its socket before the gripper attachment is released; welded plug/socket pairs are collision-filtered and velocity-stabilized.",
                "Released seated parts are locked to their recipe part_i_seated targets, matching the task YAML lock_part_i_seated phases and preventing post-release floating.",
                "Assembly entries 13/20/30/37 are replaced by the FixPlug policy at 30 Hz with 8 Isaac Sim physics steps per policy action.",
                "The active plug remains fixed to the gripper only during the FixPlug window, matching the official FixPlug training assumption.",
                "Inserted right-arm plugs are detached after each FixPlug window so fixed joints do not accumulate on the gripper.",
            ],
        }
        _write_json(output_path.with_suffix(".json"), summary)
        return summary
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run plumbers_block official Fabrica trajectory with FixPlug RL windows in Isaac Sim.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scene-spec", type=Path, default=DEFAULT_SCENE_SPEC)
    parser.add_argument("--fixture-usd", type=Path, default=DEFAULT_FIXTURE_USD)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--plan-info", type=Path, default=DEFAULT_PLAN_INFO)
    parser.add_argument("--recipe", default="fabrica_plumbers_block")
    parser.add_argument("--scene-profile", default="taoyuan_grscenes_tabletop")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--tracking-substeps", type=int, default=4)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--allow-noncontact-attach", action="store_true")
    args = parser.parse_args()
    summary = run_hybrid(
        output_path=args.output.resolve(),
        frames_dir=args.frames_dir.resolve(),
        log_dir=args.log_dir.resolve(),
        manifest_path=args.manifest.resolve(),
        scene_spec_path=args.scene_spec.resolve(),
        fixture_usd_path=args.fixture_usd.resolve(),
        checkpoint=args.checkpoint.resolve(),
        plan_info=args.plan_info.resolve(),
        recipe=args.recipe,
        scene_profile=args.scene_profile,
        width=args.width,
        height=args.height,
        fps=args.fps,
        stride=max(args.stride, 1),
        max_frames=args.max_frames,
        headless=args.headless,
        require_contact_attach=not args.allow_noncontact_attach,
        tracking_substeps=max(int(args.tracking_substeps), 1),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
