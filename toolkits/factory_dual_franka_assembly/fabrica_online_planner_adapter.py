from __future__ import annotations

import json
import pickle
import sys
import importlib
import types
import contextlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_ARM_JOINT_CONTROLLER = "arm_joint_controller"
_GRIPPER_CONTROLLER = "gripper_controller"
_OFFICIAL_TCP_ADAPTER_CACHE: dict[int, "FabricaOnlinePickupTransportAdapter"] = {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _quat_to_matrix_wxyz(quat_wxyz) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=float)
    norm = np.linalg.norm([w, x, y, z])
    if norm <= 0:
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


def _matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    m = np.asarray(rotation, dtype=float)
    trace = float(np.trace(m))
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ],
            dtype=float,
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        quat = np.array(
            [
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            ],
            dtype=float,
        )
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        quat = np.array(
            [
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
            ],
            dtype=float,
        )
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        quat = np.array(
            [
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s,
            ],
            dtype=float,
        )
    norm = np.linalg.norm(quat)
    return quat / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0], dtype=float)


def _pose_matrix(position, quat_wxyz) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = _quat_to_matrix_wxyz(quat_wxyz)
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix


def _rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError(f"Need matching Nx3 point arrays with N>=3, got {source.shape} and {target.shape}")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def _prepare_fabrica_imports(fabrica_root: Path):
    root_text = str(fabrica_root.resolve())
    if root_text in sys.path:
        sys.path.remove(root_text)
    sys.path.insert(0, root_text)
    importlib.invalidate_caches()

    # Isaac Sim may preload unrelated top-level packages named `utils`,
    # `assets`, or `planning`. Fabrica uses those names as top-level modules,
    # so remove shadowing modules before importing the official planner.
    for top_name in ("assets", "planning", "utils"):
        for module_name, module in list(sys.modules.items()):
            if module_name != top_name and not module_name.startswith(f"{top_name}."):
                continue
            module_file = getattr(module, "__file__", None)
            if module_file and str(Path(module_file).resolve()).startswith(root_text):
                continue
            sys.modules.pop(module_name, None)

    for top_name in ("assets", "planning", "utils"):
        package_dir = fabrica_root / top_name
        if not package_dir.is_dir():
            continue
        package = types.ModuleType(top_name)
        package.__path__ = [str(package_dir)]
        package.__package__ = top_name
        package.__file__ = str(package_dir)
        sys.modules[top_name] = package


@dataclass
class _PlannerContext:
    motion: list
    pickup_pose: dict[str, np.ndarray]
    pickup_gripper_pose: dict[str, np.ndarray]
    part_meshes: dict[str, Any]
    fixture_mesh: Any
    arm_chain: Any
    planner: Any
    motion_type: str
    arm_type: str
    gripper_type: str
    has_ft_sensor: bool
    map_rotation: np.ndarray
    map_translation: np.ndarray
    part_offsets_m: dict[str, np.ndarray]


class FabricaOnlinePickupTransportAdapter:
    """Run Fabrica's arm IK/RRT planner for pickup and transport phases.

    This adapter intentionally plans joint paths from the current task state.
    It does not replay `traj.npy` or `motion.pkl` paths; `motion.pkl` is used
    only as a source of official target segment ids and nominal end joints.
    """

    def __init__(self, spec: dict):
        self.spec = dict(spec)
        self._contexts: dict[tuple[str, str], _PlannerContext] = {}
        self._plans: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._commanded_arm_q: dict[tuple[int, str], np.ndarray] = {}

    def act(
        self,
        *,
        task,
        robot_name: str,
        phase_spec: dict,
        skill_spec: dict,
        tracked_robots: dict,
        tracked_objects: dict,
        checkpoint_path=None,
    ) -> dict | None:
        spec = {**self.spec, **skill_spec}
        phase_key = (
            id(task),
            getattr(task, "phase_index", None),
            getattr(task, "phase_entry_step", None),
            robot_name,
            int(spec["motion_index"]),
        )
        plan_state = self._plans.get(phase_key)
        if plan_state is None:
            plan_state = self._build_plan(
                task=task,
                robot_name=robot_name,
                phase_spec=phase_spec,
                spec=spec,
                tracked_objects=tracked_objects,
            )
            self._plans[phase_key] = plan_state

        current_q = self._current_arm_joint_positions(
            task,
            robot_name,
            spec={**spec, "prefer_commanded_joint_state": False},
        )
        if current_q is not None:
            while plan_state["index"] < len(plan_state["path"]) - 1:
                active_target = plan_state["path"][plan_state["index"]]
                joint_error = self._active_joint_abs_error(
                    plan_state["context"],
                    current_q,
                    active_target,
                    spec=spec,
                )
                if float(np.max(joint_error)) > float(spec.get("joint_tolerance", 0.035)):
                    break
                plan_state["index"] += 1
            final_error = float(
                np.max(
                    self._active_joint_abs_error(
                        plan_state["context"],
                        current_q,
                        plan_state["path"][-1],
                        spec=spec,
                    )
                )
            )
            if plan_state["index"] >= len(plan_state["path"]) - 1 and plan_state.get("final_command_phase_step") is None:
                plan_state["final_command_phase_step"] = int(getattr(task, "phase_step_counter", 0))
            final_command_age = (
                None
                if plan_state.get("final_command_phase_step") is None
                else int(getattr(task, "phase_step_counter", 0)) - int(plan_state["final_command_phase_step"])
            )
            completion_settle_steps = max(int(spec.get("completion_settle_steps", 24)), 0)
            completion_by_joint_error = final_error <= float(spec.get("joint_tolerance", 0.035))
            completion_by_final_hold = (
                plan_state["index"] >= len(plan_state["path"]) - 1
                and final_command_age is not None
                and final_command_age >= completion_settle_steps
            )
            force_complete_after_steps = spec.get("force_complete_after_steps")
            force_complete = (
                force_complete_after_steps is not None
                and int(getattr(task, "phase_step_counter", 0)) >= int(force_complete_after_steps)
            )
            if completion_by_joint_error or completion_by_final_hold or force_complete:
                self._mark_task_skill_complete(
                    task=task,
                    robot_name=robot_name,
                    skill_name=str(spec.get("name", "fabrica_online_pickup_transport")),
                    detail={
                        "motion_index": int(spec["motion_index"]),
                        "part_id": plan_state.get("part_id"),
                        "path_length": len(plan_state["path"]),
                        "path_index": int(plan_state["index"]),
                        "final_joint_error": final_error,
                        "final_command_age": final_command_age,
                        "force_complete": bool(force_complete),
                    },
                )

        target_q = plan_state["path"][int(plan_state["index"])]
        self._commanded_arm_q[(id(task), robot_name)] = np.asarray(target_q, dtype=float).copy()
        self._publish_official_tcp_pose_override(
            task=task,
            robot_name=robot_name,
            context=plan_state["context"],
            q_active=target_q,
        )
        action = {_ARM_JOINT_CONTROLLER: [target_q.tolist()]}
        gripper_command = None if self._disable_gripper_actions(task, spec) else self._gripper_command(phase_spec, robot_name, spec)
        if gripper_command is not None:
            action[_GRIPPER_CONTROLLER] = [gripper_command]
        return action

    @staticmethod
    def _mark_task_skill_complete(*, task, robot_name: str, skill_name: str, detail: dict[str, Any]) -> None:
        marker = getattr(task, "mark_local_skill_complete", None)
        if callable(marker):
            marker(robot_name=robot_name, skill_name=skill_name, detail=detail)
            return
        completions = getattr(task, "_local_skill_completions", None)
        if completions is None:
            completions = {}
            setattr(task, "_local_skill_completions", completions)
        completions[
            (
                getattr(task, "phase_index", None),
                getattr(task, "phase_entry_step", None),
                robot_name,
                skill_name,
            )
        ] = dict(detail)

    def _build_plan(self, *, task, robot_name: str, phase_spec: dict, spec: dict, tracked_objects: dict) -> dict:
        context = self._context(task=task, spec=spec, robot_name=robot_name)
        motion_index = int(spec["motion_index"])
        motion_entry = context.motion[motion_index]
        if motion_entry[1] != "arm":
            raise ValueError(f"Fabrica motion index {motion_index} is not an arm segment: {motion_entry}")

        q_start_active = self._current_arm_joint_positions(
            task,
            robot_name,
            spec={**spec, "prefer_commanded_joint_state": False},
        )
        if q_start_active is None:
            raise RuntimeError(f"Cannot read current arm joints for {robot_name}")
        q_start = self._clip_full_q_to_bounds(
            context,
            context.arm_chain.active_to_full(q_start_active),
        )

        q_goal_active_nominal = np.asarray(motion_entry[2][-1], dtype=float)
        q_goal = self._clip_full_q_to_bounds(
            context,
            context.arm_chain.active_to_full(q_goal_active_nominal),
        )
        part_id = self._part_id(spec.get("part_id") or spec.get("active_part") or motion_entry[3])
        current_part_pose = None
        if part_id is not None:
            current_part_pose = self._current_part_pose_plan(
                task,
                tracked_objects,
                part_id,
                context=context,
            )

        if bool(spec.get("recompute_pickup_goal", False)):
            if part_id is None or current_part_pose is None:
                raise ValueError("recompute_pickup_goal requires part_id and current part pose")
            q_goal = self._recompute_pickup_goal(
                context,
                part_id=part_id,
                q_nominal=q_goal,
                current_part_pose=current_part_pose,
                allow_nominal_fallback=bool(
                    spec.get("fallback_to_nominal_pickup_goal_on_ik_failure", True)
                ),
            )

        open_ratio = float(
            spec.get(
                "open_ratio",
                spec.get(
                    "gripper_open_ratio",
                    self._open_ratio_before_motion(
                        context.motion,
                        motion_index=motion_index,
                        motion_type=context.motion_type,
                        default=0.65,
                    ),
                ),
            )
        )
        max_speed = float(spec.get("max_speed", 4.0))
        planner_task = str(spec.get("planner_task", motion_entry[4]))

        plan_with_grasp = part_id is not None and bool(spec.get("plan_with_grasp", motion_entry[3] is not None))
        still_meshes = self._current_still_meshes(
            task,
            tracked_objects,
            active_part=part_id if plan_with_grasp else None,
            context=context,
        )
        if bool(spec.get("include_fixture", True)):
            still_meshes.append(context.fixture_mesh)

        other_context, other_q, other_open_ratio = self._other_arm_state(
            task=task,
            spec=spec,
            robot_name=robot_name,
            motion=motion_entry,
            motion_index=motion_index,
        )
        other_kwargs = {}
        if other_context is not None and other_q is not None and other_open_ratio is not None:
            other_kwargs = {
                "arm_chain_other": other_context.arm_chain,
                "arm_q_other": other_context.arm_chain.active_to_full(np.asarray(other_q, dtype=float)),
                "open_ratio_other": float(other_open_ratio),
                "has_ft_sensor_other": other_context.has_ft_sensor,
            }

        try:
            path = self._plan_path(
                context=context,
                spec=spec,
                planner_task=planner_task,
                q_start=q_start,
                q_goal=q_goal,
                part_id=part_id,
                current_part_pose=current_part_pose,
                still_meshes=still_meshes,
                open_ratio=open_ratio,
                max_speed=max_speed,
                plan_with_grasp=plan_with_grasp,
                other_kwargs=other_kwargs,
            )
        except AssertionError as exc:
            if bool(spec.get("fallback_to_official_motion_segment", context.arm_type == "ur5e")):
                fallback_active = [np.asarray(q_start_active, dtype=float).copy()]
                for waypoint in motion_entry[2]:
                    waypoint = np.asarray(waypoint, dtype=float)
                    if fallback_active and np.max(np.abs(fallback_active[-1] - waypoint)) <= 1e-6:
                        continue
                    fallback_active.append(waypoint.copy())
                fallback_goal = context.arm_chain.active_from_full(np.asarray(q_goal, dtype=float))
                if not fallback_active or np.max(np.abs(fallback_active[-1] - fallback_goal)) > 1e-6:
                    fallback_active.append(fallback_goal)
                path = [context.arm_chain.active_to_full(q_active) for q_active in fallback_active]
            else:
                diagnostic = self._collision_diagnostic(
                    context=context,
                    q_start=q_start,
                    q_goal=q_goal,
                    still_meshes=still_meshes,
                    open_ratio=open_ratio,
                    move_pickup_mesh=None,
                    gripper_pickup_transform=None,
                    other_kwargs=other_kwargs,
                )
                raise AssertionError(
                    f"{exc}; motion_index={motion_index}; robot={robot_name}; "
                    f"motion_type={context.motion_type}; q_start_active={q_start_active.tolist()}; "
                    f"q_goal_active={context.arm_chain.active_from_full(q_goal).tolist()}; "
                    f"open_ratio={open_ratio}; still_mesh_count={len(still_meshes)}; "
                    f"diagnostic={diagnostic}"
                ) from exc

        if path is None:
            raise RuntimeError(f"Fabrica online planner failed for phase {phase_spec.get('name')}")
        q_goal_active = context.arm_chain.active_from_full(np.asarray(q_goal, dtype=float))
        path_goal_active = context.arm_chain.active_from_full(np.asarray(path[-1], dtype=float))
        path_goal_error = float(np.max(np.abs(path_goal_active - q_goal_active)))
        if path_goal_error > float(spec.get("planner_goal_tolerance", 0.08)):
            if bool(spec.get("append_official_goal_on_short_path", True)):
                path = list(path)
                path.append(np.asarray(q_goal, dtype=float).copy())
            else:
                raise RuntimeError(
                    f"Fabrica planner returned a path that does not reach official goal for "
                    f"phase {phase_spec.get('name')}: max joint error {path_goal_error:.6f}"
                )
        stride = max(int(spec.get("path_stride", 6)), 1)
        active_path = [context.arm_chain.active_from_full(np.asarray(q, dtype=float)) for q in path[::stride]]
        final_active = context.arm_chain.active_from_full(np.asarray(path[-1], dtype=float))
        if not active_path or np.max(np.abs(active_path[-1] - final_active)) > 1e-6:
            active_path.append(final_active)
        if not active_path or np.max(np.abs(active_path[0] - q_start_active)) > 1e-6:
            active_path.insert(0, np.asarray(q_start_active, dtype=float).copy())
        preferred_abs_limit = spec.get("preferred_joint_abs_limit")
        if preferred_abs_limit is None and context.arm_type == "ur5e":
            preferred_abs_limit = 3.05
        hard_preferred_abs_limit = bool(
            spec.get(
                "hard_preferred_joint_abs_limit",
                context.arm_type == "ur5e" and preferred_abs_limit is not None,
            )
        )
        active_path = self._unwrap_active_path(
            active_path,
            reference=q_start_active,
            preferred_abs_limit=preferred_abs_limit,
            hard_preferred_abs_limit=hard_preferred_abs_limit,
            reference_weight=float(spec.get("joint_unwrap_reference_weight", 0.35)),
            limit_weight=float(spec.get("joint_unwrap_limit_weight", 2.0)),
        )
        if bool(spec.get("filter_joint_limit_waypoints", context.arm_type == "ur5e")):
            active_path = self._filter_joint_limit_waypoints(
                active_path,
                preferred_abs_limit=preferred_abs_limit,
                margin=float(spec.get("joint_limit_filter_margin", 0.55)),
            )
        active_path = self._densify_active_path(
            active_path,
            max_joint_step=float(spec.get("max_joint_step", 0.025 if context.arm_type == "ur5e" else 0.05)),
        )
        return {
            "path": active_path,
            "index": 0,
            "final_command_phase_step": None,
            "motion_index": motion_index,
            "part_id": part_id,
            "context": context,
        }

    @staticmethod
    def _active_joint_abs_error(context: _PlannerContext, current_q, target_q, *, spec: dict) -> np.ndarray:
        diff = np.asarray(current_q, dtype=float) - np.asarray(target_q, dtype=float)
        if bool(spec.get("wrap_revolute_joint_error", context.arm_type == "ur5e")):
            diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
        return np.abs(diff)

    @staticmethod
    def _densify_active_path(active_path: list[np.ndarray], *, max_joint_step: float) -> list[np.ndarray]:
        if len(active_path) <= 1 or max_joint_step <= 0:
            return active_path
        dense: list[np.ndarray] = [np.asarray(active_path[0], dtype=float)]
        for waypoint in active_path[1:]:
            waypoint = np.asarray(waypoint, dtype=float)
            previous = dense[-1]
            max_delta = float(np.max(np.abs(waypoint - previous)))
            segment_count = max(int(np.ceil(max_delta / float(max_joint_step))), 1)
            for segment_index in range(1, segment_count + 1):
                alpha = segment_index / segment_count
                dense.append((1.0 - alpha) * previous + alpha * waypoint)
        return dense

    def _plan_path(
        self,
        *,
        context: _PlannerContext,
        spec: dict,
        planner_task: str,
        q_start,
        q_goal,
        part_id: str | None,
        current_part_pose,
        still_meshes: list,
        open_ratio: float,
        max_speed: float,
        plan_with_grasp: bool,
        other_kwargs: dict,
    ):
        if plan_with_grasp:
            if current_part_pose is None:
                raise RuntimeError(f"Cannot read current pose for active part {part_id}")
            move_mesh = context.part_meshes[part_id].copy()
            move_mesh.apply_transform(current_part_pose)
            gripper_pickup_transform = self._pickup_gripper_transform(
                context,
                part_id=part_id,
                current_part_pose=current_part_pose,
                fallback_q=q_start,
            )
            path = context.planner.plan_path_with_grasp(
                q_start,
                q_goal,
                move_pickup_mesh=move_mesh,
                gripper_pickup_transform=gripper_pickup_transform,
                still_meshes=still_meshes,
                open_ratio=open_ratio,
                retract_start=np.array(spec.get("retract_start", [0.0, 0.0, 1.0]), dtype=float),
                retract_goal=np.array(spec.get("retract_goal", [0.0, 0.0, 1.0]), dtype=float),
                retract_delta=float(spec.get("retract_delta", 0.5)),
                max_speed=max_speed,
                verbose=bool(spec.get("verbose", False)),
                **other_kwargs,
            )
        elif planner_task == "assembly" and bool(spec.get("straight_line", False)):
            path = context.planner.plan_path_straight(
                q_start,
                q_goal,
                open_ratio=open_ratio,
                max_speed=max_speed,
                sanity_check=bool(spec.get("sanity_check", False)),
                verbose=bool(spec.get("verbose", False)),
            )
        else:
            path = context.planner.plan_path(
                q_start,
                q_goal,
                part_meshes=still_meshes,
                open_ratio=open_ratio,
                retract_start=np.array(spec.get("retract_start", [0.0, 0.0, 1.0]), dtype=float),
                retract_goal=np.array(spec.get("retract_goal", [0.0, 0.0, 1.0]), dtype=float),
                retract_delta=float(spec.get("retract_delta", 0.5)),
                max_speed=max_speed,
                verbose=bool(spec.get("verbose", False)),
                **other_kwargs,
            )
        return path

    def _collision_diagnostic(
        self,
        *,
        context: _PlannerContext,
        q_start,
        q_goal,
        still_meshes: list,
        open_ratio: float,
        move_pickup_mesh,
        gripper_pickup_transform,
        other_kwargs: dict,
    ) -> dict:
        try:
            _, _, _, collision_fn = context.planner.get_fns(
                move_pickup_mesh,
                gripper_pickup_transform,
                still_meshes,
                open_ratio,
                verbose=True,
                **other_kwargs,
            )
            result = {}
            for label, q_full in (("start", q_start), ("goal", q_goal)):
                q_active = context.arm_chain.active_from_full(np.asarray(q_full, dtype=float))
                for buffered, move_ground_buffer in ((True, True), (False, False)):
                    stream = io.StringIO()
                    with contextlib.redirect_stdout(stream):
                        colliding = bool(
                            collision_fn(q_active, buffered=buffered, move_ground_buffer=move_ground_buffer)
                        )
                    result[f"{label}_{'buffered' if buffered else 'unbuffered'}"] = {
                        "colliding": colliding,
                        "message": stream.getvalue().strip(),
                    }
            return result
        except Exception as diagnostic_exc:
            return {"error": repr(diagnostic_exc)}

    def _context(self, *, task, spec: dict, robot_name: str) -> _PlannerContext:
        motion_type = str(spec.get("motion_type") or ("hold" if robot_name.endswith("left") else "move"))
        arm_type = str(spec.get("arm_type", "ur5e"))
        key = (motion_type, arm_type)
        if key in self._contexts:
            return self._contexts[key]

        fabrica_root = Path(spec.get("fabrica_root", _repo_root() / "third_part/Fabrica")).expanduser()
        if not fabrica_root.is_absolute():
            fabrica_root = (_repo_root() / fabrica_root).resolve()
        if not fabrica_root.exists():
            raise FileNotFoundError(f"Cannot find Fabrica root: {fabrica_root}")
        _prepare_fabrica_imports(fabrica_root)

        import trimesh
        from assets.transform import get_transform_matrix
        from planning.robot.geometry import load_part_meshes
        from planning.robot.motion_plan_arm import ArmMotionPlanner
        from planning.robot.util_arm import get_arm_chain
        from planning.robot.workcell import get_dual_arm_box
        from utils.common import TimeStamp

        log_dir = self._resolve_path(spec.get("log_dir", "roboassemblybench/assets/Fabrica/official_logs/codex_plumbers_block_ur5e_official/plumbers_block"))
        assembly_dir = self._resolve_path(spec.get("assembly_dir", "third_part/Fabrica/assets/fabrica/plumbers_block"))
        motion = pickle.load(open(log_dir / "motion.pkl", "rb"))
        pickup_payload = json.load(open(log_dir / "fixture/pickup.json", encoding="utf-8"))
        traj = np.load(log_dir / "traj.npy", allow_pickle=True)
        pickup_pose = {str(part_id): get_transform_matrix(values) for part_id, values in pickup_payload.items()}
        raw_meshes = load_part_meshes(str(assembly_dir), transform="none")
        part_meshes = {str(key)[4:] if str(key).startswith("part") else str(key): mesh for key, mesh in raw_meshes.items()}
        fixture_mesh = trimesh.load_mesh(log_dir / "fixture/fixture.obj")
        manifest_path = self._resolve_manifest_path(task=task, spec=spec)
        scene_spec_path = self._resolve_scene_spec_path(task=task, spec=spec)
        part_offsets_m = self._load_manifest_part_offsets(manifest_path)
        map_rotation, map_translation = self._derive_scene_spec_raw_center_mapping(
            task=task,
            traj=traj,
            part_offsets_m=part_offsets_m,
            scene_spec_path=scene_spec_path,
        )
        arm_chain = get_arm_chain(arm_type, motion_type)
        arm_box_move, arm_box_hold = get_dual_arm_box(arm_type)
        arm_box = arm_box_hold if motion_type == "hold" else arm_box_move
        planner = ArmMotionPlanner(
            arm_chain,
            str(spec.get("gripper_type", "robotiq-85")),
            bool(spec.get("has_ft_sensor", False)),
            arm_box,
            stamp=TimeStamp(),
        )
        pickup_gripper_pose = self._official_pickup_gripper_poses(
            motion=motion,
            context_arm_chain=arm_chain,
            gripper_type=str(spec.get("gripper_type", "robotiq-85")),
            has_ft_sensor=bool(spec.get("has_ft_sensor", False)),
            pickup_motion_indices=spec.get("pickup_motion_indices"),
        )
        context = _PlannerContext(
            motion=motion,
            pickup_pose=pickup_pose,
            pickup_gripper_pose=pickup_gripper_pose,
            part_meshes=part_meshes,
            fixture_mesh=fixture_mesh,
            arm_chain=arm_chain,
            planner=planner,
            motion_type=motion_type,
            arm_type=arm_type,
            gripper_type=str(spec.get("gripper_type", "robotiq-85")),
            has_ft_sensor=bool(spec.get("has_ft_sensor", False)),
            map_rotation=map_rotation,
            map_translation=map_translation,
            part_offsets_m=part_offsets_m,
        )
        self._contexts[key] = context
        return context

    def _other_arm_state(self, *, task, spec: dict, robot_name: str, motion, motion_index: int):
        motion_type = str(spec.get("motion_type") or ("hold" if robot_name.endswith("left") else "move"))
        if motion_type == "hold":
            other_motion_type = "move"
            other_robot_name = str(spec.get("other_robot") or "franka_right")
        else:
            other_motion_type = "hold"
            other_robot_name = str(spec.get("other_robot") or "franka_left")
        if other_robot_name == robot_name:
            return None, None, None
        other_q = self._current_arm_joint_positions(
            task,
            other_robot_name,
            spec={**spec, "prefer_commanded_joint_state": False},
        )
        if other_q is None:
            return None, None, None
        other_spec = dict(spec)
        other_spec["motion_type"] = other_motion_type
        other_context = self._context(task=task, spec=other_spec, robot_name=other_robot_name)
        other_open_ratio = self._open_ratio_before_motion(
            other_context.motion,
            motion_index=motion_index,
            motion_type=other_motion_type,
            default=0.5,
        )
        return other_context, other_q, other_open_ratio

    @classmethod
    def _official_pickup_gripper_poses(
        cls,
        *,
        motion: list,
        context_arm_chain,
        gripper_type: str,
        has_ft_sensor: bool,
        pickup_motion_indices,
    ) -> dict[str, np.ndarray]:
        if pickup_motion_indices is None:
            # Official plumbers_block sequence: base part 2 is picked by the hold arm,
            # then parts 0/3/1/4 are picked by the move arm.
            pickup_motion_indices = {"2": 5, "0": 10, "3": 17, "1": 27, "4": 34}
        poses: dict[str, np.ndarray] = {}
        for part_id, motion_index in dict(pickup_motion_indices).items():
            try:
                entry = motion[int(motion_index)]
                q_active = np.asarray(entry[2][-1], dtype=float)
            except Exception:
                continue
            q_full = context_arm_chain.active_to_full(q_active)
            poses[str(part_id)] = cls._gripper_matrix_from_chain(
                context_arm_chain,
                q_full,
                gripper_type=gripper_type,
                has_ft_sensor=has_ft_sensor,
            )
        return poses

    def _pickup_gripper_transform(
        self,
        context: _PlannerContext,
        *,
        part_id: str,
        current_part_pose: np.ndarray,
        fallback_q,
    ) -> np.ndarray:
        official_part_pose = context.pickup_pose.get(str(part_id))
        official_gripper_pose = context.pickup_gripper_pose.get(str(part_id))
        if official_part_pose is None or official_gripper_pose is None:
            return self._gripper_matrix_from_q(context, fallback_q)
        # Preserve Fabrica's part-to-gripper pickup offset. If the loose part was
        # shifted in Isaac, shift the gripper pickup frame by the same rigid delta.
        delta = current_part_pose @ np.linalg.inv(official_part_pose)
        return delta @ official_gripper_pose

    def _world_to_plan_transform(self, task, context: _PlannerContext) -> tuple[np.ndarray, np.ndarray]:
        cfg = getattr(task, "config", getattr(task, "cfg", None))
        object_metadata = getattr(cfg, "object_metadata", []) if cfg is not None else []
        source_world_cm = []
        target_plan = []
        for metadata in object_metadata:
            name = str(metadata.get("name", ""))
            if not name.startswith("fabrica_plumbers_block_"):
                continue
            part_id = name.rsplit("_", maxsplit=1)[-1]
            if part_id not in context.pickup_pose:
                continue
            source_world_cm.append(np.asarray(metadata["sampled_position"], dtype=float) * 100.0)
            target_plan.append(context.pickup_pose[part_id][:3, 3])
        return _rigid_transform(np.asarray(source_world_cm), np.asarray(target_plan))

    def _current_part_pose_plan(
        self,
        task,
        tracked_objects: dict,
        part_id: str,
        *,
        context: _PlannerContext,
    ) -> np.ndarray | None:
        object_name = f"fabrica_plumbers_block_{part_id}"
        state = tracked_objects.get(object_name)
        if state is None:
            cfg = getattr(task, "config", getattr(task, "cfg", None))
            for metadata in getattr(cfg, "object_metadata", []):
                if metadata.get("name") == object_name:
                    state = {
                        "position": metadata["sampled_position"],
                        "orientation": metadata["sampled_orientation"],
                    }
                    break
        if not state:
            return None
        position = state.get("position")
        orientation = state.get("orientation", [1.0, 0.0, 0.0, 0.0])
        locked_target = state.get("locked_target")
        target_position = state.get("target_position")
        target_orientation = state.get("target_orientation")
        if locked_target is not None and target_position is not None:
            # Locked fixture parts are bookkeeping poses, not live physics poses.
            # Use the lock target for planning so Fabrica sees the same still
            # geometry it used when generating the official motion plan.
            position = target_position
            if target_orientation is not None:
                orientation = target_orientation
        position_m = np.asarray(position, dtype=float)
        orientation = np.asarray(orientation, dtype=float)
        world_rotation = _quat_to_matrix_wxyz(orientation)
        raw_rotation = context.map_rotation.T @ world_rotation
        part_offset_m = context.part_offsets_m.get(str(part_id), np.zeros(3, dtype=float))
        raw_position_m = context.map_rotation.T @ (position_m - context.map_translation) - raw_rotation @ part_offset_m
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = raw_rotation
        pose[:3, 3] = raw_position_m * 100.0
        return pose

    def _plan_pose_to_world_pose(self, task, context: _PlannerContext, pose_plan: np.ndarray):
        pose_plan = np.asarray(pose_plan, dtype=float)
        world_rotation = context.map_rotation @ pose_plan[:3, :3]
        world_position = context.map_rotation @ (pose_plan[:3, 3] * 0.01) + context.map_translation
        return world_position, _matrix_to_quat_wxyz(world_rotation)

    def _publish_official_tcp_pose_override(
        self,
        *,
        task,
        robot_name: str,
        context: _PlannerContext,
        q_active,
    ) -> None:
        setter = getattr(task, "set_robot_task_pose_override", None)
        if not callable(setter):
            return
        try:
            q_full = context.arm_chain.active_to_full(np.asarray(q_active, dtype=float))
            tcp_pose_plan = self._gripper_matrix_from_q(context, q_full)
            position, orientation = self._plan_pose_to_world_pose(task, context, tcp_pose_plan)
            setter(
                robot_name=robot_name,
                position=position,
                orientation=orientation,
                source="fabrica_official_gripper_tcp",
                phase_index=getattr(task, "phase_index", None),
                phase_entry_step=getattr(task, "phase_entry_step", None),
            )
        except Exception:
            return

    def _current_still_meshes(
        self,
        task,
        tracked_objects: dict,
        *,
        active_part: str | None,
        context: _PlannerContext,
    ) -> list[Any]:
        meshes = []
        for part_id, mesh in context.part_meshes.items():
            if active_part is not None and str(part_id) == str(active_part):
                continue
            pose = self._current_part_pose_plan(
                task,
                tracked_objects,
                str(part_id),
                context=context,
            )
            if pose is None:
                pose = context.pickup_pose[str(part_id)]
            transformed = mesh.copy()
            transformed.apply_transform(pose)
            meshes.append(transformed)
        return meshes

    def _recompute_pickup_goal(
        self,
        context: _PlannerContext,
        *,
        part_id: str,
        q_nominal,
        current_part_pose: np.ndarray,
        allow_nominal_fallback: bool = True,
    ):
        official_part_pose = context.pickup_pose[str(part_id)]
        official_gripper_pose = self._gripper_matrix_from_q(context, q_nominal)
        delta = current_part_pose @ np.linalg.inv(official_part_pose)
        target_gripper_pose = delta @ official_gripper_pose
        target_pos = target_gripper_pose[:3, 3]
        target_ori = target_gripper_pose[:3, :3]
        q_goal = context.planner.inverse_kinematics(
            target_pos,
            target_ori,
            q_init=self._clip_full_q_to_bounds(context, q_nominal),
            optimizer="least_squares",
            regularization_parameter=1.0,
        )
        if q_goal is None:
            if allow_nominal_fallback:
                return q_nominal
            raise RuntimeError(f"Fabrica IK failed for shifted pickup part {part_id}")
        return q_goal

    @staticmethod
    def _clip_full_q_to_bounds(context: _PlannerContext, q, *, margin: float = 1e-7) -> np.ndarray:
        clipped = np.asarray(q, dtype=float).copy()
        for index, link in enumerate(context.arm_chain.links):
            if index >= clipped.shape[0]:
                break
            lower, upper = getattr(link, "bounds", (-np.inf, np.inf))
            if np.isfinite(lower):
                clipped[index] = max(clipped[index], float(lower) + margin)
            if np.isfinite(upper):
                clipped[index] = min(clipped[index], float(upper) - margin)
        return clipped

    @staticmethod
    def _unwrap_active_path(
        active_path: list[np.ndarray],
        *,
        reference: np.ndarray,
        preferred_abs_limit=None,
        hard_preferred_abs_limit: bool = False,
        reference_weight: float = 0.35,
        limit_weight: float = 2.0,
    ) -> list[np.ndarray]:
        """Choose stable 2*pi-equivalent joint angles for Isaac articulation control.

        Fabrica can return UR-style revolute joints in equivalent branches such as
        -6.4 rad. That is geometrically valid, but it makes Isaac's position
        controller rotate a full turn. We choose each equivalent angle using both
        continuity from the previous command and drift from the phase start.
        """
        if not active_path:
            return active_path
        unwrapped: list[np.ndarray] = []
        previous = np.asarray(reference, dtype=float).copy()
        reference = previous.copy()
        period = 2.0 * np.pi
        preferred_abs_limit = None if preferred_abs_limit is None else float(preferred_abs_limit)
        for target in active_path:
            target = np.asarray(target, dtype=float).copy()
            if target.shape != previous.shape:
                unwrapped.append(target)
                previous = target
                continue
            selected = target.copy()
            for joint_index, joint_target in enumerate(target):
                candidates = joint_target + np.arange(-8, 9, dtype=float) * period
                if hard_preferred_abs_limit and preferred_abs_limit is not None and preferred_abs_limit > 0:
                    preferred_mask = np.abs(candidates) <= preferred_abs_limit
                    if np.any(preferred_mask):
                        candidates = candidates[preferred_mask]
                continuity_cost = np.abs(candidates - previous[joint_index])
                reference_cost = max(float(reference_weight), 0.0) * np.abs(candidates - reference[joint_index])
                limit_cost = 0.0
                if preferred_abs_limit is not None and preferred_abs_limit > 0:
                    limit_cost = max(float(limit_weight), 0.0) * np.maximum(
                        np.abs(candidates) - preferred_abs_limit,
                        0.0,
                    )
                selected[joint_index] = candidates[int(np.argmin(continuity_cost + reference_cost + limit_cost))]
                if hard_preferred_abs_limit and preferred_abs_limit is not None and preferred_abs_limit > 0:
                    selected[joint_index] = float(
                        np.clip(selected[joint_index], -preferred_abs_limit, preferred_abs_limit)
                    )
            unwrapped.append(selected)
            previous = selected
        return unwrapped

    @staticmethod
    def _filter_joint_limit_waypoints(
        active_path: list[np.ndarray],
        *,
        preferred_abs_limit=None,
        margin: float = 0.02,
    ) -> list[np.ndarray]:
        if len(active_path) <= 1 or preferred_abs_limit is None or preferred_abs_limit <= 0:
            return active_path
        limit = float(preferred_abs_limit) - max(float(margin), 0.0)
        filtered: list[np.ndarray] = []
        for index, waypoint in enumerate(active_path):
            waypoint = np.asarray(waypoint, dtype=float)
            is_endpoint = index == len(active_path) - 1
            near_limit = bool(np.any(np.abs(waypoint) >= limit))
            if near_limit and not is_endpoint:
                continue
            if filtered and np.max(np.abs(filtered[-1] - waypoint)) <= 1e-6:
                continue
            filtered.append(waypoint)
        return filtered or [np.asarray(active_path[-1], dtype=float)]

    @staticmethod
    def _gripper_matrix_from_q(context: _PlannerContext, q) -> np.ndarray:
        return FabricaOnlinePickupTransportAdapter._gripper_matrix_from_chain(
            context.arm_chain,
            np.asarray(q, dtype=float),
            gripper_type=context.gripper_type,
            has_ft_sensor=context.has_ft_sensor,
        )

    @staticmethod
    def _gripper_matrix_from_chain(arm_chain, q, *, gripper_type: str, has_ft_sensor: bool) -> np.ndarray:
        from assets.transform import get_transform_matrix_quat
        from planning.robot.util_arm import get_gripper_pos_quat_from_arm_q

        pos, quat = get_gripper_pos_quat_from_arm_q(
            arm_chain,
            np.asarray(q, dtype=float),
            gripper_type,
            has_ft_sensor=has_ft_sensor,
        )
        return get_transform_matrix_quat(pos, quat)

    def _current_arm_joint_positions(self, task, robot_name: str, *, spec: dict | None = None) -> np.ndarray | None:
        robot = task.robots.get(robot_name)
        if robot is None:
            return None
        spec = spec or {}
        commanded = self._commanded_arm_q.get((id(task), robot_name))
        prefer_commanded = bool(
            spec.get(
                "prefer_commanded_joint_state",
                str(spec.get("arm_type", "ur5e")) == "ur5e",
            )
        )
        if prefer_commanded and commanded is not None:
            return np.asarray(commanded, dtype=float).copy()
        for controller_name in (_ARM_JOINT_CONTROLLER, "arm_ik_controller"):
            controller = robot.controllers.get(controller_name)
            if controller is None:
                continue
            subset = controller.get_joint_subset()
            if subset is None:
                continue
            try:
                joint_positions = np.asarray(subset.get_joint_positions(), dtype=float)
            except Exception:
                continue
            if not np.all(np.isfinite(joint_positions)):
                break
            if np.max(np.abs(joint_positions)) > 20.0:
                break
            return joint_positions
        if commanded is not None:
            return np.asarray(commanded, dtype=float).copy()
        return self._initial_arm_joint_positions(task, robot_name)

    @staticmethod
    def _initial_arm_joint_positions(task, robot_name: str) -> np.ndarray | None:
        cfg = getattr(task, "config", getattr(task, "cfg", None))
        robot_metadata = getattr(cfg, "robot_metadata", []) if cfg is not None else []
        joint_order = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        for robot_spec in robot_metadata:
            if robot_spec.get("name") != robot_name:
                continue
            initial = robot_spec.get("initial_joint_positions")
            if isinstance(initial, dict) and all(name in initial for name in joint_order):
                return np.asarray([float(initial[name]) for name in joint_order], dtype=float)
        return None

    @staticmethod
    def _part_id(value) -> str | None:
        if value is None:
            return None
        text = str(value)
        if text.startswith("fabrica_plumbers_block_"):
            return text.rsplit("_", maxsplit=1)[-1]
        return text

    @staticmethod
    def _open_ratio_before_motion(motion: list, *, motion_index: int, motion_type: str, default: float) -> float:
        open_ratio = float(default)
        for entry in motion[: max(int(motion_index), 0)]:
            try:
                entry_motion_type, body_type, value, _, _ = entry
            except Exception:
                continue
            if entry_motion_type != motion_type or body_type != "gripper":
                continue
            try:
                open_ratio = float(value)
            except (TypeError, ValueError):
                continue
        return open_ratio

    @staticmethod
    def _gripper_command(phase_spec: dict, robot_name: str, spec: dict):
        if "gripper_command" in spec:
            command = spec["gripper_command"]
        else:
            command = phase_spec.get("gripper_commands", {}).get(robot_name)
        if command is None:
            return None
        if isinstance(command, str):
            lowered = command.lower()
            if lowered == "open":
                return 1.0
            if lowered == "close":
                return 0.0
        return float(command) if isinstance(command, (int, float)) else command

    @staticmethod
    def _disable_gripper_actions(task, spec: dict) -> bool:
        if spec.get("disable_gripper_actions") is not None:
            return bool(spec.get("disable_gripper_actions"))
        cfg = getattr(task, "config", getattr(task, "cfg", None))
        metadata = getattr(cfg, "task_metadata", {}) if cfg is not None else {}
        return isinstance(metadata, dict) and bool(metadata.get("disable_gripper_actions", False))

    def _resolve_manifest_path(self, *, task, spec: dict) -> Path:
        explicit = spec.get("manifest_path") or spec.get("manifest")
        if explicit:
            return self._resolve_path(explicit)
        cfg = getattr(task, "config", getattr(task, "cfg", None))
        metadata = getattr(cfg, "task_metadata", {}) if cfg is not None else {}
        if isinstance(metadata, dict):
            for reference in metadata.get("asset_references", []) or []:
                if not isinstance(reference, dict):
                    continue
                if reference.get("kind") == "manifest" or str(reference.get("name", "")).endswith("manifest"):
                    return self._resolve_path(str(reference["path"]).replace("${BENCHMARK_ROOT}", "roboassemblybench"))
        return (
            _repo_root()
            / "roboassemblybench/assets/Fabrica/fabrica_franka_plumbers_block_optical_board_black_fullbundle_sdf001"
            / "assets/fabrica_original_usd_sdf_margin_001/aligned/plumbers_block/manifest.json"
        )

    def _resolve_scene_spec_path(self, *, task, spec: dict) -> Path:
        explicit = spec.get("scene_spec_path") or spec.get("scene_spec")
        if explicit:
            return self._resolve_path(explicit)
        cfg = getattr(task, "config", getattr(task, "cfg", None))
        metadata = getattr(cfg, "task_metadata", {}) if cfg is not None else {}
        if isinstance(metadata, dict):
            source_scene_spec = metadata.get("source_scene_spec")
            if source_scene_spec:
                return self._resolve_path(str(source_scene_spec).replace("${BENCHMARK_ROOT}", "roboassemblybench"))
        return (
            _repo_root()
            / "roboassemblybench/assets/Fabrica/fabrica_franka_plumbers_block_optical_board_black_fullbundle_sdf001"
            / "scene/scene_spec.json"
        )

    @staticmethod
    def _manifest_part_id(part: dict, fallback_index: int) -> str:
        default_prim = str(part.get("default_prim") or "")
        if default_prim:
            suffix = default_prim.rsplit("_", maxsplit=1)[-1]
            if suffix.isdigit():
                return suffix
        part_name = str(part.get("part_name") or "")
        suffix = part_name.rsplit("_", maxsplit=1)[-1]
        return suffix if suffix.isdigit() else str(fallback_index)

    @classmethod
    def _load_manifest_part_offsets(cls, manifest_path: Path) -> dict[str, np.ndarray]:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        offsets: dict[str, np.ndarray] = {}
        for index, part in enumerate(payload.get("parts", [])):
            part_id = cls._manifest_part_id(part, index)
            offsets[str(part_id)] = np.asarray(part.get("raw_bbox_center_m", [0.0, 0.0, 0.0]), dtype=float)
        if not offsets:
            raise ValueError(f"No part offsets found in manifest: {manifest_path}")
        return offsets

    @staticmethod
    def _scene_spec_target_positions(scene_spec_path: Path, *, object_prefix: str) -> dict[str, np.ndarray]:
        payload = json.loads(scene_spec_path.read_text(encoding="utf-8"))
        target_poses = payload.get("assembled_display", {}).get("assembly_target_poses", {})
        positions: dict[str, np.ndarray] = {}
        for object_name, target_pose in target_poses.items():
            if not str(object_name).startswith(f"{object_prefix}_"):
                continue
            part_id = str(object_name).rsplit("_", maxsplit=1)[-1]
            positions[part_id] = np.asarray(target_pose["target_translation_world_m"], dtype=float)
        if not positions:
            raise ValueError(f"No assembly target positions found in scene spec: {scene_spec_path}")
        return positions

    def _derive_scene_spec_raw_center_mapping(
        self,
        *,
        task,
        traj: np.ndarray,
        part_offsets_m: dict[str, np.ndarray],
        scene_spec_path: Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        target_positions = self._scene_spec_target_positions(
            scene_spec_path,
            object_prefix="fabrica_plumbers_block",
        )
        source_points = []
        target_points = []
        source_frame = traj[-1]
        for part_id in sorted(target_positions, key=lambda value: int(value) if str(value).isdigit() else str(value)):
            if str(part_id) not in part_offsets_m:
                continue
            raw_matrix = np.asarray(source_frame[f"part{part_id}"], dtype=float)
            raw_rotation = raw_matrix[:3, :3]
            raw_position_m = raw_matrix[:3, 3] * 0.01
            source_points.append(raw_position_m + raw_rotation @ part_offsets_m[str(part_id)])
            target_points.append(target_positions[str(part_id)])
        if len(source_points) < 3:
            raise ValueError("Need at least three part correspondences to derive Fabrica scene mapping")
        return _rigid_transform(np.asarray(source_points), np.asarray(target_points))

    @staticmethod
    def _resolve_path(value) -> Path:
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return path
        return (_repo_root() / path).resolve()


def official_fabrica_tcp_pose_from_task(task, robot_name: str):
    cfg = getattr(task, "config", getattr(task, "cfg", None))
    metadata = getattr(cfg, "task_metadata", {}) if cfg is not None else {}
    if not isinstance(metadata, dict):
        return None
    defaults = metadata.get("local_skills", {}).get("fabrica_online_pickup_transport", {})
    if not isinstance(defaults, dict):
        return None
    spec = dict(defaults)
    spec.setdefault("arm_type", "ur5e")
    spec.setdefault("gripper_type", "robotiq-85")
    spec["motion_type"] = "hold" if str(robot_name).endswith("left") else "move"

    adapter = _OFFICIAL_TCP_ADAPTER_CACHE.get(id(task))
    if adapter is None:
        adapter = FabricaOnlinePickupTransportAdapter(spec)
        _OFFICIAL_TCP_ADAPTER_CACHE[id(task)] = adapter
    context = adapter._context(task=task, spec=spec, robot_name=robot_name)
    q_active = adapter._current_arm_joint_positions(task, robot_name, spec={**spec, "prefer_commanded_joint_state": False})
    if q_active is None:
        return None
    q_full = context.arm_chain.active_to_full(np.asarray(q_active, dtype=float))
    tcp_pose_plan = adapter._gripper_matrix_from_q(context, q_full)
    position, orientation = adapter._plan_pose_to_world_pose(task, context, tcp_pose_plan)
    return np.asarray(position, dtype=float), np.asarray(orientation, dtype=float)


def official_fabrica_attachment_relative_pose(task, robot_name: str, object_name: str):
    """Return Fabrica's official object-center-to-gripper pickup transform.

    Fabrica plans grasped collision with raw mesh frames, while the Isaac USD
    objects are placed at their aligned/raw bbox centers. The attachment state
    is consumed by `compose_pose(robot_tcp, local_object_pose)`, so this helper
    returns the part-center pose in the official gripper TCP frame, in meters.
    """

    cfg = getattr(task, "config", getattr(task, "cfg", None))
    metadata = getattr(cfg, "task_metadata", {}) if cfg is not None else {}
    if not isinstance(metadata, dict):
        return None
    defaults = metadata.get("local_skills", {}).get("fabrica_online_pickup_transport", {})
    if not isinstance(defaults, dict):
        return None

    part_id = str(object_name).rsplit("_", maxsplit=1)[-1]
    if not part_id:
        return None

    phase_spec = {}
    getter = getattr(task, "get_current_phase_spec", None)
    if callable(getter):
        try:
            phase_spec = getter() or {}
        except Exception:
            phase_spec = {}
    local_skill = phase_spec.get("local_skill") if isinstance(phase_spec, dict) else {}

    spec = dict(defaults)
    if isinstance(local_skill, dict) and local_skill.get("robot") == robot_name:
        spec.update(local_skill)
    spec.setdefault("arm_type", "ur5e")
    spec.setdefault("gripper_type", "robotiq-85")
    spec["motion_type"] = str(spec.get("motion_type") or ("hold" if str(robot_name).endswith("left") else "move"))

    adapter = _OFFICIAL_TCP_ADAPTER_CACHE.get(id(task))
    if adapter is None:
        adapter = FabricaOnlinePickupTransportAdapter(spec)
        _OFFICIAL_TCP_ADAPTER_CACHE[id(task)] = adapter
    context = adapter._context(task=task, spec=spec, robot_name=robot_name)
    official_part_pose = context.pickup_pose.get(part_id)
    official_gripper_pose = context.pickup_gripper_pose.get(part_id)
    if official_part_pose is None or official_gripper_pose is None:
        return None

    part_center_pose = np.asarray(official_part_pose, dtype=float).copy()
    part_offset_m = context.part_offsets_m.get(part_id, np.zeros(3, dtype=float))
    part_center_pose[:3, 3] = (
        np.asarray(official_part_pose[:3, 3], dtype=float)
        + np.asarray(official_part_pose[:3, :3], dtype=float) @ (part_offset_m * 100.0)
    )
    relative_matrix = np.linalg.inv(np.asarray(official_gripper_pose, dtype=float)) @ part_center_pose
    relative_position_m = np.asarray(relative_matrix[:3, 3], dtype=float) * 0.01
    relative_orientation = _matrix_to_quat_wxyz(relative_matrix[:3, :3])
    return relative_position_m, relative_orientation
