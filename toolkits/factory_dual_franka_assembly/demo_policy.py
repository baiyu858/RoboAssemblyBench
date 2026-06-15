from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np

from internutopia_extension.configs.robots.franka import arm_ik_cfg, arm_joint_cfg, gripper_cfg
from toolkits.factory_dual_franka_assembly.local_skills import LocalSkillExecutor
from toolkits.factory_dual_franka_assembly.robofactory_planner import (
    FrankaRobofactoryPlanner,
    PlannerWaypoint,
)


class DualFrankaAssemblyDemoPolicy:
    _PHYSICS_DT = 1.0 / 240.0
    _SAFE_TRANSIT_Z = 0.48
    _HIGH_TRANSIT_Z = 0.56
    _MOVE_XY_THRESHOLD = 0.07
    _VERTICAL_THRESHOLD = 0.025
    _PRECISION_DESCENT_THRESHOLD = 0.035
    _INTERARM_CLEARANCE = 0.26
    _CENTER_X_THRESHOLD = 0.32
    _CENTER_Y_THRESHOLD = 0.22
    _CENTER_APPROACH_Y = 0.14
    _LANE_ENTRY_THRESHOLD = 0.08
    _CENTERLINE_THRESHOLD = 0.10
    _ATTACHED_OBJECT_CLEARANCE_BONUS = 0.07
    _MAX_POSITION_STEP = {
        'wait': 0.12,
        'move': 0.12,
        'pick': 0.08,
        'hold': 0.035,
        'insert': 0.020,
        'retreat': 0.12,
    }
    _MAX_VERTICAL_STEP = {
        'wait': 0.12,
        'move': 0.12,
        'pick': 0.08,
        'hold': 0.030,
        'insert': 0.018,
        'retreat': 0.12,
    }
    _MAX_ORIENTATION_STEP = {
        'wait': 0.28,
        'move': 0.24,
        'pick': 0.18,
        'hold': 0.08,
        'insert': 0.06,
        'retreat': 0.24,
    }
    _ACTION_UPDATE_INTERVAL_STEPS = 8
    _WAYPOINT_HYSTERESIS_RATIO = 0.6
    _TRAJECTORY_POINT_SPACING = {
        'wait': 0.12,
        'move': 0.14,
        'pick': 0.08,
        'hold': 0.04,
        'insert': 0.025,
        'retreat': 0.12,
    }
    _COMMAND_POSITION_TOLERANCE = {
        'wait': 0.04,
        'move': 0.05,
        'pick': 0.035,
        'hold': 0.03,
        'insert': 0.015,
        'retreat': 0.035,
    }
    _COMMAND_ORIENTATION_TOLERANCE = {
        'wait': None,
        'move': 0.55,
        'pick': 0.40,
        'hold': 0.30,
        'insert': 0.20,
        'retreat': 0.50,
    }
    _TRAJECTORY_MAX_SEGMENT_SAMPLES = 24
    _MAX_JOINT_STEP = {
        'wait': 0.14,
        'move': 0.12,
        'pick': 0.08,
        'hold': 0.04,
        'insert': 0.025,
        'retreat': 0.12,
    }
    _MAX_JOINT_VELOCITY = {
        'wait': 1.8,
        'move': 1.5,
        'pick': 1.0,
        'hold': 0.45,
        'insert': 0.25,
        'retreat': 1.5,
    }
    _PRECISION_LOCK_XY_THRESHOLD = 0.018
    _PRECISION_LOCK_Z_MARGIN = 0.06
    _PRECISION_LOCK_ORIENTATION_THRESHOLD = 0.20
    _GRASP_SEARCH_STEP_PER_PHASE_STEP = 0.00015
    _GRASP_SEARCH_MAX_DEPTH = 0.04
    _GRASP_CLOSE_RAMP_STEPS = 18
    _GRASP_CLOSE_HOLD_STEPS = 12
    _GRASP_CLOSE_DEFAULT_OPENNESS = 1.0

    @staticmethod
    def _gripper_controller_action(command):
        """Controller-level convention: 1=open, 0=close."""
        if command is None:
            return None
        if isinstance(command, (bool, np.bool_)):
            return 1 if bool(command) else 0
        if isinstance(command, (int, float, np.integer, np.floating)):
            value = float(command)
            if 0.0 <= value <= 1.0:
                return float(np.clip(value, 0.0, 1.0))
            return 1.0 if value > 0.0 else 0.0
        command_name = str(command).strip().lower()
        if command_name == 'open':
            return 1.0
        if command_name == 'close':
            return 0.0
        return command

    def __init__(self):
        self._task_signature = None
        self._policy_step = 0
        self._robot_execution_state: dict[str, dict] = {}
        self._planner_bridge = FrankaRobofactoryPlanner()
        self._local_skill_executor = LocalSkillExecutor()

    def _reset_policy_state(self):
        self._policy_step = 0
        self._robot_execution_state = {}
        self._local_skill_executor.reset()

    @property
    def local_skill_diagnostics(self):
        return self._local_skill_executor.diagnostics

    def _task_signature_for(self, task):
        cfg = getattr(task, 'cfg', None)
        return (
            id(task),
            getattr(cfg, 'recipe', None),
            getattr(cfg, 'seed', None),
            getattr(cfg, 'episode_idx', None),
        )

    def _ensure_task_context(self, task):
        task_signature = self._task_signature_for(task)
        if task_signature != self._task_signature:
            self._reset_policy_state()
            self._task_signature = task_signature
        self._policy_step += 1

    def _robot_state(self, robot_name: str) -> dict:
        return self._robot_execution_state.setdefault(
            robot_name,
            {
                'phase_key': None,
                'latched_target_name': None,
                'latched_waypoint_chain': [],
                'trajectory': [],
                'trajectory_index': 0,
                'last_action': None,
                'last_joint_action': None,
                'last_update_step': -self._ACTION_UPDATE_INTERVAL_STEPS,
                'active_command_pose': None,
                'active_target_name': None,
                'gripper_close_latched_phase_key': None,
                'grasp_close_started_step': None,
                'grasp_close_hold_until_step': None,
                'grasp_close_pose': None,
                'precision_lock': None,
                'planned_joint_trajectory': [],
                'planned_joint_index': 0,
                'planner_status': None,
            },
        )

    def _phase_key(self, task, phase_spec: dict, raw_target_spec, robot_name: str):
        return (
            getattr(task, 'phase_index', None),
            getattr(task, 'phase_entry_step', None),
            phase_spec.get('name'),
            self._target_descriptor(raw_target_spec, fallback=robot_name),
        )

    @staticmethod
    def _phase_attach_spec_for_robot(phase_spec: dict, robot_name: str) -> dict | None:
        attach_entries = phase_spec.get('attach', [])
        if isinstance(attach_entries, dict):
            attach_entries = [attach_entries]
        for attach_spec in attach_entries:
            if isinstance(attach_spec, dict) and attach_spec.get('robot') == robot_name:
                return attach_spec
        return None

    @staticmethod
    def _attachment_mode(attach_spec: dict | None) -> str:
        if not isinstance(attach_spec, dict):
            return 'fixed_joint'
        return str(attach_spec.get('attachment_mode', 'fixed_joint')).lower()

    def _phase_requires_grasp_orientation_lock(self, *, task, phase_spec: dict, robot_name: str, target_like) -> bool:
        if target_like is None:
            return False
        attach_spec = self._phase_attach_spec_for_robot(phase_spec, robot_name)
        if self._attachment_mode(attach_spec) not in {
            'physical_hold',
            'physical',
            'contact_hold',
            'physical_grasp',
            'contact_physical_grasp',
            'pure_physical_grasp',
            'contact_pure_physical_grasp',
        }:
            return False
        if attach_spec is not None and bool(attach_spec.get('allow_orientation_free_grasp', False)):
            return False
        try:
            _, _, target_orientation, _ = task.resolve_robot_target_pose(robot_name, target_like)
        except Exception:
            return False
        return target_orientation is not None

    def _hold_pose(self):
        return [None, None]

    @staticmethod
    def _normalize_quat(quat) -> np.ndarray:
        quat = np.asarray(quat, dtype=float)
        norm = np.linalg.norm(quat)
        if norm == 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        return quat / norm

    def _quat_angle(self, lhs, rhs) -> float:
        lhs = self._normalize_quat(lhs)
        rhs = self._normalize_quat(rhs)
        dot = float(np.dot(lhs, rhs))
        dot = abs(float(np.clip(dot, -1.0, 1.0)))
        return float(2.0 * math.acos(dot))

    def _slerp_orientation(self, lhs, rhs, ratio: float) -> np.ndarray:
        lhs = self._normalize_quat(lhs)
        rhs = self._normalize_quat(rhs)
        dot = float(np.dot(lhs, rhs))
        if dot < 0.0:
            rhs = -rhs
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 0.9995:
            return self._normalize_quat((1.0 - ratio) * lhs + ratio * rhs)

        theta_0 = math.acos(dot)
        sin_theta_0 = math.sin(theta_0)
        if abs(sin_theta_0) < 1e-6:
            return self._normalize_quat((1.0 - ratio) * lhs + ratio * rhs)
        theta = theta_0 * ratio
        sin_theta = math.sin(theta)
        coeff_lhs = math.sin(theta_0 - theta) / sin_theta_0
        coeff_rhs = sin_theta / sin_theta_0
        return self._normalize_quat(coeff_lhs * lhs + coeff_rhs * rhs)

    @staticmethod
    def _target_descriptor(target_spec, *, fallback: str | None = None) -> str:
        if fallback:
            return str(fallback)
        if isinstance(target_spec, str):
            return target_spec
        if isinstance(target_spec, dict):
            for key in ('target', 'target_name', 'name'):
                value = target_spec.get(key)
                if value:
                    return str(value)
            if target_spec.get('pose') is not None or target_spec.get('position') is not None:
                return 'custom_pose'
        return ''

    def _resolve_task_target_pose(self, *, task, robot_name: str, phase_spec: dict, tracked_robots: dict):
        raw_target_spec = phase_spec.get('robot_targets', {}).get(robot_name)
        if raw_target_spec is None:
            return None, None

        tracking = tracked_robots.get(robot_name, {})
        tracked_position = tracking.get('task_target')
        tracked_orientation = tracking.get('task_target_orientation')
        target_name = self._target_descriptor(raw_target_spec, fallback=tracking.get('target_name'))
        if tracked_position is not None:
            return target_name, {
                'position': np.asarray(tracked_position, dtype=float),
                'orientation': None if tracked_orientation is None else np.asarray(tracked_orientation, dtype=float),
            }

        if isinstance(raw_target_spec, str):
            target_pose = task.get_target_pose(raw_target_spec)
            return target_name, {
                'position': np.asarray(target_pose['position'], dtype=float),
                'orientation': np.asarray(target_pose['orientation'], dtype=float),
            }

        if isinstance(raw_target_spec, dict) and hasattr(task, 'resolve_robot_target_pose'):
            target_name, target_position, target_orientation, _ = task.resolve_robot_target_pose(
                robot_name,
                raw_target_spec,
            )
            return self._target_descriptor(raw_target_spec, fallback=target_name), {
                'position': np.asarray(target_position, dtype=float),
                'orientation': None if target_orientation is None else np.asarray(target_orientation, dtype=float),
            }

        if isinstance(raw_target_spec, dict) and hasattr(task, '_resolve_target_pose_spec'):
            target_name, target_position, target_orientation, _ = task._resolve_target_pose_spec(raw_target_spec)
            return self._target_descriptor(raw_target_spec, fallback=target_name), {
                'position': np.asarray(target_position, dtype=float),
                'orientation': None if target_orientation is None else np.asarray(target_orientation, dtype=float),
            }

        raise TypeError(f'Unsupported robot target specification for {robot_name}: {type(raw_target_spec)!r}')

    def _resolve_pose_spec(self, *, task, robot_name: str, pose_spec):
        if isinstance(pose_spec, str):
            if hasattr(task, 'resolve_robot_target_pose'):
                _, target_position, target_orientation, _ = task.resolve_robot_target_pose(robot_name, pose_spec)
                target_pose = {
                    'position': np.asarray(target_position, dtype=float),
                    'orientation': np.asarray(target_orientation, dtype=float),
                }
            else:
                target_pose = task.get_target_pose(pose_spec)
            return str(pose_spec), {
                'position': np.asarray(target_pose['position'], dtype=float),
                'orientation': np.asarray(target_pose['orientation'], dtype=float),
            }, 0.05

        if isinstance(pose_spec, dict) and hasattr(task, '_resolve_target_pose_spec'):
            if hasattr(task, 'resolve_robot_target_pose'):
                target_name, target_position, target_orientation, resolved_spec = task.resolve_robot_target_pose(
                    robot_name,
                    pose_spec,
                )
            else:
                target_name, target_position, target_orientation, resolved_spec = task._resolve_target_pose_spec(pose_spec)
            tolerance = float(
                pose_spec.get(
                    'waypoint_tolerance',
                    pose_spec.get('position_tolerance', pose_spec.get('tolerance', 0.05)),
                )
            )
            return self._target_descriptor(pose_spec, fallback=target_name), {
                'position': np.asarray(target_position, dtype=float),
                'orientation': None if target_orientation is None else np.asarray(target_orientation, dtype=float),
            }, tolerance

        raise TypeError(f'Unsupported waypoint specification: {type(pose_spec)!r}')

    def _resolve_waypoint_chain(self, *, task, robot_name: str, raw_target_spec, final_target_pose):
        chain = []
        if isinstance(raw_target_spec, dict):
            waypoint_specs = raw_target_spec.get('via') or raw_target_spec.get('waypoints') or raw_target_spec.get('path') or []
            for waypoint_spec in waypoint_specs:
                waypoint_name, waypoint_pose, waypoint_tolerance = self._resolve_pose_spec(
                    task=task,
                    robot_name=robot_name,
                    pose_spec=waypoint_spec,
                )
                if waypoint_pose['orientation'] is None:
                    waypoint_pose['orientation'] = np.asarray(final_target_pose['orientation'], dtype=float)
                chain.append(
                    {
                        'name': waypoint_name,
                        'pose': waypoint_pose,
                        'tolerance': waypoint_tolerance,
                    }
                )
        chain.append(
            {
                'name': self._target_descriptor(raw_target_spec),
                'pose': {
                    'position': np.asarray(final_target_pose['position'], dtype=float),
                    'orientation': np.asarray(final_target_pose['orientation'], dtype=float),
                },
                'tolerance': float(
                    raw_target_spec.get('waypoint_tolerance', raw_target_spec.get('position_tolerance', raw_target_spec.get('tolerance', 0.04)))
                )
                if isinstance(raw_target_spec, dict)
                else 0.04,
            }
        )
        return chain

    @staticmethod
    def _as_array(value, *, default):
        if value is None:
            return np.asarray(default, dtype=float)
        return np.asarray(value, dtype=float)

    @staticmethod
    def _xy_distance(lhs, rhs) -> float:
        lhs = np.asarray(lhs, dtype=float)
        rhs = np.asarray(rhs, dtype=float)
        return float(np.linalg.norm(lhs[:2] - rhs[:2]))

    @staticmethod
    def _distance(lhs, rhs) -> float:
        lhs = np.asarray(lhs, dtype=float)
        rhs = np.asarray(rhs, dtype=float)
        return float(np.linalg.norm(lhs - rhs))

    def _pose_matches(self, lhs_pose: dict, rhs_pose: dict, *, position_tolerance: float = 1e-3, orientation_tolerance: float = 1e-2) -> bool:
        if self._distance(lhs_pose['position'], rhs_pose['position']) > position_tolerance:
            return False
        lhs_orientation = lhs_pose.get('orientation')
        rhs_orientation = rhs_pose.get('orientation')
        if lhs_orientation is None or rhs_orientation is None:
            return lhs_orientation is None and rhs_orientation is None
        return self._quat_angle(lhs_orientation, rhs_orientation) <= orientation_tolerance

    def _dynamic_payload_target_changed(
        self,
        *,
        state: dict,
        task,
        robot_name: str,
        phase_spec: dict,
        tracked_robots: dict,
    ) -> bool:
        raw_target_spec = phase_spec.get('robot_targets', {}).get(robot_name)
        if not isinstance(raw_target_spec, dict):
            return False
        if raw_target_spec.get('payload_object') is None or raw_target_spec.get('payload_target') is None:
            return False

        _, resolved_target_pose = self._resolve_task_target_pose(
            task=task,
            robot_name=robot_name,
            phase_spec=phase_spec,
            tracked_robots=tracked_robots,
        )
        if resolved_target_pose is None:
            return False

        waypoint_chain = state.get('latched_waypoint_chain') or []
        if not waypoint_chain:
            return True
        latched_final_pose = waypoint_chain[-1].get('pose')
        if latched_final_pose is None:
            return True
        return not self._pose_matches(
            latched_final_pose,
            resolved_target_pose,
            position_tolerance=5e-3,
            orientation_tolerance=8e-2,
        )

    @staticmethod
    def _side_sign(robot_name: str) -> float:
        return -1.0 if robot_name == 'franka_left' else 1.0

    @staticmethod
    def _descriptor(target_name: str, phase_spec: dict) -> str:
        return f"{phase_spec.get('name', '')} {target_name}".lower()

    def _attached_object_names(self, tracked_objects: dict, robot_name: str) -> list[str]:
        return [
            object_name
            for object_name, object_state in tracked_objects.items()
            if object_state.get('attached_to') == robot_name
        ]

    def _attached_object_state(self, tracked_objects: dict, robot_name: str):
        for object_name, object_state in tracked_objects.items():
            if object_state.get('attached_to') == robot_name:
                return object_name, object_state
        return None, None

    def _robot_release_targets(self, phase_spec: dict, robot_name: str) -> set[str]:
        release_targets = set()
        for object_entry in phase_spec.get('detach', []) if isinstance(phase_spec.get('detach'), list) else [phase_spec.get('detach')]:
            if isinstance(object_entry, dict):
                object_name = object_entry.get('object') or object_entry.get('name')
            else:
                object_name = object_entry
            if object_name:
                release_targets.add(str(object_name))
        for lock_spec in phase_spec.get('lock', []) if isinstance(phase_spec.get('lock'), list) else [phase_spec.get('lock')]:
            if isinstance(lock_spec, dict):
                object_name = lock_spec.get('object') or lock_spec.get('name')
                if object_name:
                    release_targets.add(str(object_name))
        return release_targets

    def _robot_attach_targets(self, phase_spec: dict, robot_name: str) -> set[str]:
        attach_targets = set()
        attach_entries = phase_spec.get('attach', [])
        if not isinstance(attach_entries, list):
            attach_entries = [attach_entries]
        for attach_spec in attach_entries:
            if not isinstance(attach_spec, dict) or attach_spec.get('robot') != robot_name:
                continue
            object_name = attach_spec.get('object') or attach_spec.get('name')
            if object_name:
                attach_targets.add(str(object_name))
        return attach_targets

    def _should_freeze_for_release(self, phase_spec: dict, robot_name: str, tracked_objects: dict) -> bool:
        if str(phase_spec.get('gripper_commands', {}).get(robot_name, '')).lower() != 'open':
            return False
        attached_object_name, _ = self._attached_object_state(tracked_objects, robot_name)
        if attached_object_name is None:
            return bool(phase_spec.get('freeze_after_detach', False) and self._robot_release_targets(phase_spec, robot_name))
        attached_state = tracked_objects.get(attached_object_name, {})
        attachment = attached_state.get('attachment') or {}
        if str(attachment.get('mode', '')).lower() == 'pure_physical_grasp':
            return True
        return attached_object_name in self._robot_release_targets(phase_spec, robot_name)

    def _should_freeze_for_attached_grasp(self, phase_spec: dict, robot_name: str, tracked_objects: dict) -> bool:
        attached_object_name, _ = self._attached_object_state(tracked_objects, robot_name)
        if attached_object_name is None:
            return False
        if str(phase_spec.get('gripper_commands', {}).get(robot_name, '')).lower() != 'close':
            return False
        if attached_object_name not in self._robot_attach_targets(phase_spec, robot_name):
            return False
        phase_name = str(phase_spec.get('name', '')).lower()
        return 'grasp' in phase_name or 'pick' in phase_name

    @staticmethod
    def _clear_grasp_close_state(state: dict):
        state['grasp_close_started_step'] = None
        state['grasp_close_hold_until_step'] = None
        state['grasp_close_pose'] = None

    def _should_freeze_for_grasp_close(
        self,
        phase_spec: dict,
        robot_name: str,
        tracked_objects: dict,
        state: dict,
    ) -> bool:
        attached_object_name, _ = self._attached_object_state(tracked_objects, robot_name)
        if attached_object_name is not None:
            return False
        if str(phase_spec.get('gripper_commands', {}).get(robot_name, '')).lower() != 'close':
            return False
        if state.get('grasp_close_started_step') is None:
            return False
        phase_name = str(phase_spec.get('name', '')).lower()
        return 'grasp' in phase_name or 'pick' in phase_name

    def _motion_mode(self, phase_spec: dict, robot_name: str, target_name: str, tracked_objects: dict) -> str:
        descriptor = self._descriptor(target_name, phase_spec)
        attached_objects = self._attached_object_names(tracked_objects, robot_name)
        if 'seat' in descriptor:
            return 'insert'
        if 'insert' in descriptor or 'preinsert' in descriptor:
            return 'insert'
        if 'pick' in descriptor or 'grasp' in descriptor:
            return 'pick'
        if 'hold' in descriptor or attached_objects:
            return 'hold'
        if 'wait' in descriptor:
            return 'wait'
        if 'retreat' in descriptor:
            return 'retreat'
        return 'move'

    def _approach_clearance(self, mode: str) -> float:
        if mode == 'insert':
            return 0.12
        if mode in {'pick', 'hold'}:
            return 0.10
        return 0.06

    def _safe_transit_height(self, current_position, target_position, *, mode: str, near_center: bool) -> float:
        clearance = self._approach_clearance(mode)
        base_height = max(float(current_position[2]), float(target_position[2]) + clearance, self._SAFE_TRANSIT_Z)
        if near_center or mode in {'pick', 'insert', 'hold'}:
            base_height = max(base_height, self._HIGH_TRANSIT_Z if near_center else self._SAFE_TRANSIT_Z)
        if mode in {'hold', 'insert'}:
            base_height = min(base_height, self._HIGH_TRANSIT_Z)
        return float(base_height)

    def _attached_object_clearance_bonus(self, tracked_objects: dict, robot_name: str) -> float:
        if not self._attached_object_names(tracked_objects, robot_name):
            return 0.0
        return self._ATTACHED_OBJECT_CLEARANCE_BONUS

    def _wait_pose(self, task, robot_name: str, target_orientation):
        wait_target_name = 'left_wait' if robot_name == 'franka_left' else 'right_wait'
        if wait_target_name in task.target_poses:
            wait_pose = task.get_target_pose(wait_target_name)
            return {
                'position': np.asarray(wait_pose['position'], dtype=float),
                'orientation': np.asarray(wait_pose['orientation'], dtype=float),
            }
        side = -1.0 if robot_name == 'franka_left' else 1.0
        return {
            'position': np.array([0.30, 0.34 * side, self._HIGH_TRANSIT_Z], dtype=float),
            'orientation': np.asarray(target_orientation, dtype=float),
        }

    def _conflict_escape_pose(
        self,
        *,
        target_position,
        target_orientation,
        robot_name: str,
        safe_z: float,
        near_center: bool,
        tracked_objects: dict,
    ):
        target_position = np.asarray(target_position, dtype=float)
        conflict_position = target_position.copy()
        if near_center or abs(float(conflict_position[1])) < self._CENTER_APPROACH_Y:
            conflict_position[1] = self._side_sign(robot_name) * max(abs(float(conflict_position[1])), self._CENTER_APPROACH_Y)
        conflict_position[2] = max(float(conflict_position[2]), self._HIGH_TRANSIT_Z if near_center else safe_z)
        min_payload_height = self._payload_floor_height(tracked_objects, robot_name)
        if min_payload_height is not None:
            conflict_position[2] = max(float(conflict_position[2]), min_payload_height)
        return {
            'position': conflict_position,
            'orientation': np.asarray(target_orientation, dtype=float),
        }

    def _motion_priority(self, phase_spec: dict, robot_name: str, target_name: str, tracked_objects: dict) -> int:
        descriptor = self._descriptor(target_name, phase_spec)
        score = 0
        if self._attached_object_names(tracked_objects, robot_name):
            score += 4
        if 'hold' in descriptor:
            score += 3
        if 'insert' in descriptor or 'preinsert' in descriptor:
            score += 2
        if phase_spec.get('gripper_commands', {}).get(robot_name) == 'close':
            score += 1
        if robot_name == 'franka_left':
            score += 1
        return score

    def _lane_pose(self, target_position, *, robot_name: str, safe_z: float, orientation):
        target_position = np.asarray(target_position, dtype=float)
        lane_y = target_position[1]
        if abs(lane_y) < self._CENTER_Y_THRESHOLD:
            lane_y = self._side_sign(robot_name) * self._CENTER_Y_THRESHOLD
        return {
            'position': np.array([target_position[0], lane_y, safe_z], dtype=float),
            'orientation': np.asarray(orientation, dtype=float),
        }

    def _requires_center_lane(self, target_position, mode: str) -> bool:
        target_position = np.asarray(target_position, dtype=float)
        return bool(
            target_position[0] > self._CENTER_X_THRESHOLD
            and abs(target_position[1]) < self._CENTER_Y_THRESHOLD
            and mode in {'pick', 'insert', 'hold', 'move'}
        )

    def _approach_position(self, target_position, *, robot_name: str, mode: str):
        target_position = np.asarray(target_position, dtype=float)
        approach_position = target_position.copy()
        if self._requires_center_lane(target_position, mode) and mode in {'hold', 'insert', 'move'}:
            if abs(approach_position[1]) < self._CENTER_APPROACH_Y:
                approach_position[1] = self._side_sign(robot_name) * self._CENTER_APPROACH_Y
        return approach_position

    def _payload_floor_height(self, tracked_objects: dict, robot_name: str) -> float | None:
        _, attached_object_state = self._attached_object_state(tracked_objects, robot_name)
        if attached_object_state is None:
            return None
        attachment = attached_object_state.get('attachment') or {}
        local_position = attachment.get('position')
        scale = attached_object_state.get('scale')
        if local_position is None or scale is None:
            return None
        local_position = np.asarray(local_position, dtype=float)
        scale = np.asarray(scale, dtype=float)
        half_height = float(max(scale[2] * 0.5, 0.01))
        return float(half_height + 0.015 - local_position[2])

    def _command_position_tolerance(self, mode: str, *, waypoint_tolerance: float | None = None) -> float:
        base_tolerance = float(self._COMMAND_POSITION_TOLERANCE.get(mode, 0.02))
        if waypoint_tolerance is None:
            return base_tolerance
        return max(base_tolerance, float(waypoint_tolerance) * self._WAYPOINT_HYSTERESIS_RATIO)

    def _command_orientation_tolerance(self, mode: str) -> float | None:
        value = self._COMMAND_ORIENTATION_TOLERANCE.get(mode)
        return None if value is None else float(value)

    def _trajectory_point_spacing(self, mode: str) -> float:
        return float(self._TRAJECTORY_POINT_SPACING.get(mode, 0.04))

    def _pose_within_command_tolerance(
        self,
        *,
        current_position,
        current_orientation,
        target_pose: dict,
        mode: str,
        waypoint_tolerance: float | None = None,
        ignore_orientation: bool = False,
    ) -> bool:
        position_tolerance = self._command_position_tolerance(mode, waypoint_tolerance=waypoint_tolerance)
        if self._distance(current_position, target_pose['position']) > position_tolerance:
            return False
        if ignore_orientation:
            return True
        target_orientation = target_pose.get('orientation')
        orientation_tolerance = self._command_orientation_tolerance(mode)
        if target_orientation is None or orientation_tolerance is None:
            return True
        return self._quat_angle(current_orientation, target_orientation) <= orientation_tolerance

    def _ik_pose_feasible(self, task, robot_name: str, pose: dict) -> bool:
        try:
            controller = task.robots[robot_name].controllers.get(arm_ik_cfg.name)
        except Exception:
            controller = None
        if controller is None or not hasattr(controller, '_kinematics_solver'):
            return True

        try:
            ik_base_pose = controller.get_ik_base_world_pose()
            controller._kinematics_solver.set_robot_base_pose(
                robot_position=ik_base_pose[0] / controller._robot_scale,
                robot_orientation=ik_base_pose[1],
            )
            _, success = controller._kinematics_solver.compute_inverse_kinematics(
                target_position=np.asarray(pose['position'], dtype=float) / controller._robot_scale,
                target_orientation=np.asarray(pose['orientation'], dtype=float),
            )
        except Exception:
            return True
        return bool(success)

    def _current_arm_joint_positions(self, task, robot_name: str) -> np.ndarray | None:
        robot = task.robots.get(robot_name)
        if robot is None:
            return None
        controller = robot.controllers.get(arm_joint_cfg.name)
        if controller is not None:
            subset = controller.get_joint_subset()
            if subset is not None:
                try:
                    return np.asarray(subset.get_joint_positions(), dtype=float)
                except Exception:
                    pass
        controller = robot.controllers.get(arm_ik_cfg.name)
        if controller is not None:
            subset = controller.get_joint_subset()
            if subset is not None:
                try:
                    return np.asarray(subset.get_joint_positions(), dtype=float)
                except Exception:
                    pass
        return None

    def _joint_trajectory_action_for_pose(self, task, robot_name: str, pose: dict, *, mode: str):
        robot = task.robots.get(robot_name)
        if robot is None:
            return None

        joint_controller = robot.controllers.get(arm_joint_cfg.name)
        ik_controller = robot.controllers.get(arm_ik_cfg.name)
        if joint_controller is None or ik_controller is None or not hasattr(ik_controller, '_kinematics_solver'):
            return None

        current_joint_positions = self._current_arm_joint_positions(task, robot_name)
        if current_joint_positions is None:
            return None

        try:
            ik_base_pose = ik_controller.get_ik_base_world_pose()
            ik_controller._kinematics_solver.set_robot_base_pose(
                robot_position=ik_base_pose[0] / ik_controller._robot_scale,
                robot_orientation=ik_base_pose[1],
            )
            goal_action, success = ik_controller._kinematics_solver.compute_inverse_kinematics(
                target_position=np.asarray(pose['position'], dtype=float) / ik_controller._robot_scale,
                target_orientation=np.asarray(pose['orientation'], dtype=float),
            )
        except Exception:
            return None

        if not success or goal_action is None or goal_action.joint_positions is None:
            return None

        target_joint_positions = np.asarray(goal_action.joint_positions, dtype=float)
        if target_joint_positions.shape != current_joint_positions.shape:
            return None

        # The Cartesian target has already been smoothed, down-sampled, and routed
        # through waypoint hysteresis. Sending another heavily-interpolated joint
        # step on top of that makes the arm stall far from goal, so the joint
        # controller should track the IK solution for the current smoothed pose
        # directly.
        return [target_joint_positions.tolist()]

    def _joint_action_disabled_for_phase(
        self,
        *,
        task,
        robot_name: str,
        phase_spec: dict,
        tracked_objects: dict,
    ) -> bool:
        cfg = getattr(task, 'config', getattr(task, 'cfg', None))
        task_metadata = getattr(cfg, 'task_metadata', {}) if cfg is not None else {}
        raw_target_spec = phase_spec.get('robot_targets', {}).get(robot_name)
        attached_objects = self._attached_object_names(tracked_objects, robot_name)

        if isinstance(task_metadata, dict):
            if bool(task_metadata.get('disable_joint_action', False)):
                return True
            if attached_objects and bool(task_metadata.get('disable_joint_action_when_attached', False)):
                return True

        if isinstance(raw_target_spec, dict):
            if any(
                bool(raw_target_spec.get(key, False))
                for key in (
                    'disable_joint_action',
                    'disable_joint_controller',
                    'cartesian_only',
                )
            ):
                return True
            if raw_target_spec.get('payload_object') is not None:
                return True

        return False

    def _planner_joint_tolerance(self, mode: str) -> float:
        return max(float(self._MAX_JOINT_STEP.get(mode, 0.06)) * 0.8, 0.025)

    def _build_planner_waypoints(
        self,
        *,
        task,
        robot_name: str,
        phase_spec: dict,
        waypoint_chain: list[dict],
        current_position,
        current_orientation,
        tracked_objects: dict,
    ) -> list[PlannerWaypoint]:
        planner_waypoints: list[PlannerWaypoint] = []
        cursor_pose = {
            'position': np.asarray(current_position, dtype=float),
            'orientation': self._normalize_quat(current_orientation),
        }
        for waypoint_index, waypoint in enumerate(waypoint_chain):
            explicit_waypoint = waypoint_index < len(waypoint_chain) - 1
            target_name = waypoint['name'] or self._target_descriptor(waypoint['pose'])
            mode = self._motion_mode(phase_spec, robot_name, target_name, tracked_objects)
            checkpoints = self._build_cartesian_checkpoints(
                task=task,
                robot_name=robot_name,
                start_pose=cursor_pose,
                target_pose=waypoint['pose'],
                target_name=target_name,
                phase_spec=phase_spec,
                tracked_objects=tracked_objects,
                explicit_waypoint=explicit_waypoint,
            )
            if not checkpoints:
                checkpoints = [{'name': target_name, 'pose': waypoint['pose']}]
            for checkpoint_index, checkpoint in enumerate(checkpoints):
                pose = {
                    'position': np.asarray(checkpoint['pose']['position'], dtype=float),
                    'orientation': self._normalize_quat(checkpoint['pose']['orientation']),
                }
                if planner_waypoints:
                    previous = planner_waypoints[-1]
                    previous_pose = {
                        'position': previous.position,
                        'orientation': previous.orientation,
                    }
                    if self._pose_matches(previous_pose, pose):
                        continue
                tolerance = waypoint['tolerance'] if checkpoint_index == len(checkpoints) - 1 else None
                planner_waypoints.append(
                    PlannerWaypoint(
                        name=checkpoint['name'],
                        mode=mode,
                        position=pose['position'],
                        orientation=pose['orientation'],
                        tolerance=tolerance,
                    )
                )
                cursor_pose = pose
        return planner_waypoints

    def _build_planned_joint_trajectory(
        self,
        *,
        task,
        robot_name: str,
        phase_spec: dict,
        waypoint_chain: list[dict],
        current_position,
        current_orientation,
        tracked_objects: dict,
    ) -> list[dict] | None:
        cfg = getattr(task, 'config', getattr(task, 'cfg', None))
        task_metadata = getattr(cfg, 'task_metadata', {}) if cfg is not None else {}
        if isinstance(task_metadata, dict) and bool(task_metadata.get('disable_joint_planner', False)):
            return None
        if (
            isinstance(task_metadata, dict)
            and bool(task_metadata.get('disable_joint_planner_when_attached', False))
            and self._attached_object_names(tracked_objects, robot_name)
        ):
            return None
        raw_target_spec = phase_spec.get('robot_targets', {}).get(robot_name)
        if isinstance(raw_target_spec, dict) and bool(raw_target_spec.get('disable_joint_planner', False)):
            return None
        if isinstance(raw_target_spec, dict) and raw_target_spec.get('payload_object') is not None:
            return None
        if not self._planner_bridge.available:
            return None
        robot = task.robots.get(robot_name)
        if robot is None:
            return None
        start_qpos = self._current_arm_joint_positions(task, robot_name)
        if start_qpos is None:
            return None
        planner_waypoints = self._build_planner_waypoints(
            task=task,
            robot_name=robot_name,
            phase_spec=phase_spec,
            waypoint_chain=waypoint_chain,
            current_position=current_position,
            current_orientation=current_orientation,
            tracked_objects=tracked_objects,
        )
        if not planner_waypoints:
            return None
        try:
            base_position, base_orientation = robot.articulation.get_pose()
            return self._planner_bridge.plan_waypoints(
                base_position=base_position,
                base_orientation=base_orientation,
                start_qpos=start_qpos,
                waypoints=planner_waypoints,
            )
        except Exception:
            return None

    def _advance_planned_joint_trajectory_index(self, *, state: dict, current_joint_positions):
        joint_trajectory = state.get('planned_joint_trajectory') or []
        if current_joint_positions is None:
            return
        while state['planned_joint_index'] < len(joint_trajectory):
            active_entry = joint_trajectory[state['planned_joint_index']]
            error = float(
                np.max(np.abs(np.asarray(current_joint_positions, dtype=float) - active_entry['joint_positions']))
            )
            if error > self._planner_joint_tolerance(active_entry['mode']):
                break
            state['planned_joint_index'] += 1
        if state['planned_joint_index'] >= len(joint_trajectory) and joint_trajectory:
            state['planned_joint_index'] = len(joint_trajectory) - 1

    def _planned_joint_target(self, state: dict):
        joint_trajectory = state.get('planned_joint_trajectory') or []
        if not joint_trajectory:
            return None
        index = int(np.clip(state.get('planned_joint_index', 0), 0, len(joint_trajectory) - 1))
        return joint_trajectory[index]

    def _append_checkpoint(
        self,
        checkpoints: list[dict],
        *,
        task,
        robot_name: str,
        pose: dict,
        name: str,
        allow_infeasible: bool,
    ):
        normalized_pose = {
            'position': np.asarray(pose['position'], dtype=float),
            'orientation': self._normalize_quat(pose['orientation']),
        }
        if checkpoints and self._pose_matches(checkpoints[-1]['pose'], normalized_pose):
            return
        if not allow_infeasible and not self._ik_pose_feasible(task, robot_name, normalized_pose):
            return
        checkpoints.append({'name': name, 'pose': normalized_pose})

    def _build_cartesian_checkpoints(
        self,
        *,
        task,
        robot_name: str,
        start_pose: dict,
        target_pose: dict,
        target_name: str,
        phase_spec: dict,
        tracked_objects: dict,
        explicit_waypoint: bool,
    ) -> list[dict]:
        mode = self._motion_mode(phase_spec, robot_name, target_name, tracked_objects)
        raw_target_spec = phase_spec.get('robot_targets', {}).get(robot_name)
        precision_lock_phase = self._is_precision_lock_phase(
            phase_spec=phase_spec,
            robot_name=robot_name,
            target_name=target_name,
            mode=mode,
        )
        start_position = np.asarray(start_pose['position'], dtype=float)
        start_orientation = self._normalize_quat(start_pose['orientation'])
        target_position = np.asarray(target_pose['position'], dtype=float)
        target_orientation = self._normalize_quat(target_pose['orientation'])
        near_center = target_position[0] > self._CENTER_X_THRESHOLD and abs(target_position[1]) < self._CENTER_Y_THRESHOLD
        safe_z = self._safe_transit_height(start_position, target_position, mode=mode, near_center=near_center)
        approach_position = self._approach_position(target_position, robot_name=robot_name, mode=mode)
        min_payload_height = self._payload_floor_height(tracked_objects, robot_name)
        transit_orientation = (
            start_orientation.copy()
            if mode in {'hold', 'move'}
            else self._normalize_quat(target_orientation)
        )

        ignore_payload_floor = bool(
            isinstance(raw_target_spec, dict)
            and raw_target_spec.get('direct_payload_ignore_floor_height', False)
        )

        def payload_safe(position):
            result = np.asarray(position, dtype=float).copy()
            if min_payload_height is not None and not ignore_payload_floor:
                result[2] = max(float(result[2]), float(min_payload_height))
            return result

        checkpoints: list[dict] = []
        if isinstance(raw_target_spec, dict) and bool(raw_target_spec.get('direct_payload_motion', False)):
            self._append_checkpoint(
                checkpoints,
                task=task,
                robot_name=robot_name,
                pose={'position': payload_safe(target_position), 'orientation': target_orientation},
                name=target_name,
                allow_infeasible=True,
            )
            return checkpoints

        if explicit_waypoint:
            self._append_checkpoint(
                checkpoints,
                task=task,
                robot_name=robot_name,
                pose={'position': payload_safe(target_position), 'orientation': target_orientation},
                name=target_name,
                allow_infeasible=True,
            )
            return checkpoints

        significant_xy_motion = self._xy_distance(start_position, target_position) > self._MOVE_XY_THRESHOLD
        requires_center_lane = (not precision_lock_phase) and self._requires_center_lane(target_position, mode)
        if (
            not precision_lock_phase
            and
            significant_xy_motion
            and (
                start_position[2] < safe_z - self._VERTICAL_THRESHOLD
                or target_position[2] < safe_z - self._VERTICAL_THRESHOLD
                or requires_center_lane
            )
        ):
            self._append_checkpoint(
                checkpoints,
                task=task,
                robot_name=robot_name,
                pose={
                    'position': payload_safe([start_position[0], start_position[1], safe_z]),
                    'orientation': transit_orientation,
                },
                name=f'{target_name}_lift',
                allow_infeasible=False,
            )

        if requires_center_lane:
            lane_pose = self._lane_pose(
                target_position,
                robot_name=robot_name,
                safe_z=safe_z,
                orientation=transit_orientation,
            )
            lane_pose['position'] = payload_safe(lane_pose['position'])
            self._append_checkpoint(
                checkpoints,
                task=task,
                robot_name=robot_name,
                pose=lane_pose,
                name=f'{target_name}_lane',
                allow_infeasible=False,
            )

        hover_pose = {
            'position': payload_safe([approach_position[0], approach_position[1], safe_z]),
            'orientation': transit_orientation,
        }
        if (
            (significant_xy_motion and not precision_lock_phase) or requires_center_lane
        ) and (
            self._xy_distance(start_position, hover_pose['position']) > self._MOVE_XY_THRESHOLD
            or abs(float(start_position[2]) - float(hover_pose['position'][2])) > self._VERTICAL_THRESHOLD
        ):
            self._append_checkpoint(
                checkpoints,
                task=task,
                robot_name=robot_name,
                pose=hover_pose,
                name=f'{target_name}_hover',
                allow_infeasible=False,
            )

        self._append_checkpoint(
            checkpoints,
            task=task,
            robot_name=robot_name,
            pose={'position': payload_safe(target_position), 'orientation': target_orientation},
            name=target_name,
            allow_infeasible=True,
        )
        return checkpoints

    def _sample_segment(self, *, start_pose: dict, end_pose: dict, mode: str, waypoint_tolerance: float | None, label: str) -> list[dict]:
        start_position = np.asarray(start_pose['position'], dtype=float)
        end_position = np.asarray(end_pose['position'], dtype=float)
        start_orientation = self._normalize_quat(start_pose['orientation'])
        end_orientation = self._normalize_quat(end_pose['orientation'])
        distance = self._distance(start_position, end_position)
        angle = self._quat_angle(start_orientation, end_orientation)
        spacing_steps = int(math.ceil(distance / max(self._trajectory_point_spacing(mode), 1e-6)))
        angle_step_limit = max(float(self._MAX_ORIENTATION_STEP.get(mode, 0.2)), 1e-3)
        orientation_steps = int(math.ceil(angle / angle_step_limit))
        step_count = int(max(1, spacing_steps, orientation_steps))
        coarse_segment_cap = 4 if mode == 'insert' else 3
        step_count = min(step_count, coarse_segment_cap, self._TRAJECTORY_MAX_SEGMENT_SAMPLES)

        samples = []
        for step_index in range(1, step_count + 1):
            ratio = float(step_index) / float(step_count)
            pose = {
                'position': start_position + (end_position - start_position) * ratio,
                'orientation': self._slerp_orientation(start_orientation, end_orientation, ratio),
            }
            samples.append(
                {
                    'name': label,
                    'pose': pose,
                    'mode': mode,
                    'ignore_orientation': waypoint_tolerance is None or step_index < step_count,
                    'tolerance': self._command_position_tolerance(
                        mode,
                        waypoint_tolerance=waypoint_tolerance if step_index == step_count else None,
                    ),
                }
            )
        return samples

    def _build_latched_trajectory(
        self,
        *,
        task,
        robot_name: str,
        phase_spec: dict,
        waypoint_chain: list[dict],
        current_position,
        current_orientation,
        tracked_objects: dict,
    ) -> list[dict]:
        if not waypoint_chain:
            return []

        trajectory = []
        cursor_pose = {
            'position': np.asarray(current_position, dtype=float),
            'orientation': self._normalize_quat(current_orientation),
        }
        for waypoint_index, waypoint in enumerate(waypoint_chain):
            explicit_waypoint = waypoint_index < len(waypoint_chain) - 1
            target_name = waypoint['name'] or self._target_descriptor(waypoint['pose'])
            mode = self._motion_mode(phase_spec, robot_name, target_name, tracked_objects)
            checkpoints = self._build_cartesian_checkpoints(
                task=task,
                robot_name=robot_name,
                start_pose=cursor_pose,
                target_pose=waypoint['pose'],
                target_name=target_name,
                phase_spec=phase_spec,
                tracked_objects=tracked_objects,
                explicit_waypoint=explicit_waypoint,
            )
            if not checkpoints:
                checkpoints = [{'name': target_name, 'pose': waypoint['pose']}]
            for checkpoint_index, checkpoint in enumerate(checkpoints):
                checkpoint_tolerance = waypoint['tolerance'] if checkpoint_index == len(checkpoints) - 1 else None
                segment = self._sample_segment(
                    start_pose=cursor_pose,
                    end_pose=checkpoint['pose'],
                    mode=mode,
                    waypoint_tolerance=checkpoint_tolerance,
                    label=checkpoint['name'],
                )
                trajectory.extend(segment)
                cursor_pose = checkpoint['pose']

        if not trajectory:
            final_waypoint = waypoint_chain[-1]
            trajectory.append(
                {
                    'name': final_waypoint['name'],
                    'pose': {
                        'position': np.asarray(final_waypoint['pose']['position'], dtype=float),
                        'orientation': self._normalize_quat(final_waypoint['pose']['orientation']),
                    },
                    'mode': self._motion_mode(phase_spec, robot_name, final_waypoint['name'], tracked_objects),
                    'tolerance': self._command_position_tolerance(
                        self._motion_mode(phase_spec, robot_name, final_waypoint['name'], tracked_objects),
                        waypoint_tolerance=final_waypoint['tolerance'],
                    ),
                }
            )
        return trajectory

    def _refresh_robot_plan_if_needed(
        self,
        *,
        task,
        robot_name: str,
        phase_spec: dict,
        tracked_robots: dict,
        tracked_objects: dict,
    ):
        state = self._robot_state(robot_name)
        raw_target_spec = phase_spec.get('robot_targets', {}).get(robot_name)
        phase_key = self._phase_key(task, phase_spec, raw_target_spec, robot_name)
        if (
            state['phase_key'] == phase_key
            and not self._dynamic_payload_target_changed(
                state=state,
                task=task,
                robot_name=robot_name,
                phase_spec=phase_spec,
                tracked_robots=tracked_robots,
            )
        ):
            return state

        state['phase_key'] = phase_key
        state['trajectory'] = []
        state['trajectory_index'] = 0
        state['active_command_pose'] = None
        state['active_target_name'] = None
        state['last_action'] = None
        state['last_joint_action'] = None
        state['last_update_step'] = -self._ACTION_UPDATE_INTERVAL_STEPS
        state['latched_target_name'] = None
        state['latched_waypoint_chain'] = []
        state['gripper_close_latched_phase_key'] = None
        self._clear_grasp_close_state(state)
        state['precision_lock'] = None
        state['planned_joint_trajectory'] = []
        state['planned_joint_index'] = 0
        state['planner_status'] = None

        target_name, target_pose = self._resolve_task_target_pose(
            task=task,
            robot_name=robot_name,
            phase_spec=phase_spec,
            tracked_robots=tracked_robots,
        )
        if target_pose is None:
            return state

        robot_tracking = tracked_robots.get(robot_name, {})
        current_position = self._as_array(robot_tracking.get('position'), default=target_pose['position'])
        current_orientation = self._as_array(robot_tracking.get('orientation'), default=target_pose['orientation'])
        waypoint_chain = self._resolve_waypoint_chain(
            task=task,
            robot_name=robot_name,
            raw_target_spec=raw_target_spec,
            final_target_pose=target_pose,
        )
        if (
            isinstance(raw_target_spec, dict)
            and (raw_target_spec.get('ignore_orientation') or raw_target_spec.get('position_only'))
            and not self._phase_requires_grasp_orientation_lock(
                task=task,
                phase_spec=phase_spec,
                robot_name=robot_name,
                target_like=raw_target_spec,
            )
        ):
            for waypoint in waypoint_chain:
                waypoint['pose']['orientation'] = current_orientation.copy()

        state['latched_target_name'] = target_name
        state['latched_waypoint_chain'] = waypoint_chain
        state['trajectory'] = self._build_latched_trajectory(
            task=task,
            robot_name=robot_name,
            phase_spec=phase_spec,
            waypoint_chain=waypoint_chain,
            current_position=current_position,
            current_orientation=current_orientation,
            tracked_objects=tracked_objects,
        )
        planned_joint_trajectory = self._build_planned_joint_trajectory(
            task=task,
            robot_name=robot_name,
            phase_spec=phase_spec,
            waypoint_chain=waypoint_chain,
            current_position=current_position,
            current_orientation=current_orientation,
            tracked_objects=tracked_objects,
        )
        if planned_joint_trajectory:
            state['planned_joint_trajectory'] = planned_joint_trajectory
            state['planner_status'] = 'planned'
        else:
            state['planner_status'] = 'fallback'
        return state

    def _advance_trajectory_index(self, *, state: dict, current_position, current_orientation):
        trajectory = state.get('trajectory') or []
        while state['trajectory_index'] < len(trajectory):
            active_entry = trajectory[state['trajectory_index']]
            if not self._pose_within_command_tolerance(
                current_position=current_position,
                current_orientation=current_orientation,
                target_pose=active_entry['pose'],
                mode=active_entry['mode'],
                waypoint_tolerance=active_entry.get('tolerance'),
                ignore_orientation=bool(active_entry.get('ignore_orientation', False)),
            ):
                break
            state['trajectory_index'] += 1
        if state['trajectory_index'] >= len(trajectory) and trajectory:
            state['trajectory_index'] = len(trajectory) - 1

    def _trajectory_target(self, state: dict):
        trajectory = state.get('trajectory') or []
        if not trajectory:
            return None
        index = int(np.clip(state.get('trajectory_index', 0), 0, len(trajectory) - 1))
        return trajectory[index]

    def _is_precision_lock_phase(self, *, phase_spec: dict, robot_name: str, target_name: str, mode: str) -> bool:
        gripper_command = str(phase_spec.get('gripper_commands', {}).get(robot_name, '')).lower()
        if gripper_command != 'close':
            return False
        raw_target_spec = phase_spec.get('robot_targets', {}).get(robot_name)
        if (
            isinstance(raw_target_spec, dict)
            and raw_target_spec.get('direct_payload_motion')
            and raw_target_spec.get('direct_payload_disable_precision_lock', True)
        ):
            return False
        descriptor = self._descriptor(target_name, phase_spec)
        if mode == 'pick' and 'grasp' in descriptor:
            return True
        if mode == 'insert' and 'insert' in descriptor:
            return True
        return False

    def _maybe_capture_precision_lock(
        self,
        *,
        state: dict,
        phase_spec: dict,
        robot_name: str,
        target_name: str,
        mode: str,
        current_position,
        current_orientation,
        target_pose: dict,
    ):
        if not self._is_precision_lock_phase(
            phase_spec=phase_spec,
            robot_name=robot_name,
            target_name=target_name,
            mode=mode,
        ):
            state['precision_lock'] = None
            return None

        existing_lock = state.get('precision_lock')
        if existing_lock is not None and existing_lock.get('target_name') == target_name:
            return existing_lock

        xy_error = self._xy_distance(current_position, target_pose['position'])
        z_error = abs(float(np.asarray(current_position, dtype=float)[2] - np.asarray(target_pose['position'], dtype=float)[2]))
        if xy_error > self._PRECISION_LOCK_XY_THRESHOLD or z_error > self._PRECISION_LOCK_Z_MARGIN:
            return None

        locked_orientation = np.asarray(target_pose['orientation'], dtype=float)
        if self._quat_angle(current_orientation, target_pose['orientation']) <= self._PRECISION_LOCK_ORIENTATION_THRESHOLD:
            locked_orientation = self._normalize_quat(current_orientation)

        precision_lock = {
            'target_name': target_name,
            'xy': np.asarray(current_position, dtype=float)[:2].copy(),
            'orientation': locked_orientation.copy(),
        }
        state['precision_lock'] = precision_lock
        return precision_lock

    def _should_wait_for_other_robot(
        self,
        *,
        task,
        phase_spec: dict,
        robot_name: str,
        other_robot_name: str,
        target_name: str,
        other_target_name: str,
        tracked_robots: dict,
        tracked_objects: dict,
    ) -> bool:
        other_tracking = tracked_robots.get(other_robot_name, {})
        other_descriptor = self._descriptor(other_target_name, phase_spec)
        own_descriptor = self._descriptor(target_name, phase_spec)
        other_attached = self._attached_object_names(tracked_objects, other_robot_name)
        if not other_attached and 'hold' not in other_descriptor:
            return False
        if other_tracking.get('target_reached'):
            return False
        if 'wait' in own_descriptor or 'retreat' in own_descriptor:
            return False
        own_priority = self._motion_priority(phase_spec, robot_name, target_name, tracked_objects)
        other_priority = self._motion_priority(phase_spec, other_robot_name, other_target_name, tracked_objects)
        return bool(own_priority < other_priority)

    def _interarm_conflict(self, proposed_position, other_position, other_target_position) -> bool:
        if proposed_position is None or other_position is None:
            return False

        proposed_position = np.asarray(proposed_position, dtype=float)
        other_position = np.asarray(other_position, dtype=float)
        other_target_position = None if other_target_position is None else np.asarray(other_target_position, dtype=float)

        if self._distance(proposed_position, other_position) < self._INTERARM_CLEARANCE:
            return True
        if other_target_position is not None and self._distance(proposed_position, other_target_position) < self._INTERARM_CLEARANCE:
            return True

        in_center_corridor = (
            proposed_position[0] > self._CENTER_X_THRESHOLD and abs(proposed_position[1]) < self._CENTER_Y_THRESHOLD
        )
        other_in_center = other_position[0] > self._CENTER_X_THRESHOLD and abs(other_position[1]) < self._CENTER_Y_THRESHOLD
        return bool(in_center_corridor and other_in_center and proposed_position[2] < self._HIGH_TRANSIT_Z)

    def _effective_clearance(self, tracked_objects: dict, robot_name: str, other_robot_name: str) -> float:
        clearance = self._INTERARM_CLEARANCE
        clearance += self._attached_object_clearance_bonus(tracked_objects, robot_name)
        clearance += self._attached_object_clearance_bonus(tracked_objects, other_robot_name)
        return float(clearance)

    def _interarm_conflict_with_clearance(
        self,
        proposed_position,
        other_position,
        other_target_position,
        *,
        clearance: float,
    ) -> bool:
        if proposed_position is None or other_position is None:
            return False

        proposed_position = np.asarray(proposed_position, dtype=float)
        other_position = np.asarray(other_position, dtype=float)
        other_target_position = None if other_target_position is None else np.asarray(other_target_position, dtype=float)

        if self._distance(proposed_position, other_position) < clearance:
            return True
        if other_target_position is not None and self._distance(proposed_position, other_target_position) < clearance:
            return True

        in_center_corridor = (
            proposed_position[0] > self._CENTER_X_THRESHOLD and abs(proposed_position[1]) < self._CENTER_Y_THRESHOLD
        )
        other_in_center = other_position[0] > self._CENTER_X_THRESHOLD and abs(other_position[1]) < self._CENTER_Y_THRESHOLD
        return bool(in_center_corridor and other_in_center and proposed_position[2] < self._HIGH_TRANSIT_Z)

    def _rate_limit_position(
        self,
        current_position,
        target_position,
        *,
        mode: str,
        max_step: float | None = None,
        max_vertical: float | None = None,
    ):
        current_position = np.asarray(current_position, dtype=float)
        target_position = np.asarray(target_position, dtype=float)
        delta = target_position - current_position
        max_vertical = self._MAX_VERTICAL_STEP.get(mode, 0.04) if max_vertical is None else float(max_vertical)
        max_step = self._MAX_POSITION_STEP.get(mode, 0.04) if max_step is None else float(max_step)

        limited_delta = delta.copy()
        limited_delta[2] = float(np.clip(limited_delta[2], -max_vertical, max_vertical))
        max_xy_step = max(max_step**2 - float(limited_delta[2] ** 2), 0.0) ** 0.5
        xy_norm = float(np.linalg.norm(limited_delta[:2]))
        if xy_norm > max_xy_step > 0.0:
            limited_delta[:2] *= max_xy_step / xy_norm
        elif max_xy_step == 0.0:
            limited_delta[:2] = 0.0

        limited_norm = float(np.linalg.norm(limited_delta))
        if limited_norm > max_step > 0.0:
            limited_delta *= max_step / limited_norm
        return current_position + limited_delta

    def _blend_orientation(self, current_orientation, target_orientation, *, mode: str):
        current = self._normalize_quat(current_orientation)
        target = self._normalize_quat(target_orientation)
        dot = float(np.dot(current, target))
        if dot < 0.0:
            target = -target
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        angle = float(2.0 * math.acos(dot))
        max_angle = self._MAX_ORIENTATION_STEP.get(mode, 0.2)
        if angle <= max_angle or angle < 1e-6:
            return target
        ratio = max_angle / angle
        sin_total = math.sin(angle * 0.5)
        if abs(sin_total) < 1e-6:
            blended = (1.0 - ratio) * current + ratio * target
            return self._normalize_quat(blended)
        theta = angle * 0.5
        coeff_current = math.sin((1.0 - ratio) * theta) / sin_total
        coeff_target = math.sin(ratio * theta) / sin_total
        return self._normalize_quat(coeff_current * current + coeff_target * target)

    def _plan_constrained_pose(
        self,
        *,
        task,
        robot_name: str,
        phase_spec: dict,
        target_name: str,
        current_position,
        current_orientation,
        target_position,
        target_orientation,
        other_position,
        other_target_position,
        other_target_name,
        other_target_reached,
        tracked_robots: dict,
        tracked_objects: dict,
        other_robot_name: str,
        explicit_waypoint: bool = False,
        precision_lock: dict | None = None,
    ):
        mode = self._motion_mode(phase_spec, robot_name, target_name, tracked_objects)
        descriptor = self._descriptor(target_name, phase_spec)
        current_position = np.asarray(current_position, dtype=float)
        current_orientation = np.asarray(current_orientation, dtype=float)
        target_position = np.asarray(target_position, dtype=float).copy()
        target_orientation = np.asarray(target_orientation, dtype=float)
        gripper_command = str(phase_spec.get('gripper_commands', {}).get(robot_name, '')).lower()
        attach_spec = self._phase_attach_spec_for_robot(phase_spec, robot_name) or {}
        attach_mode = str(attach_spec.get('attachment_mode', '')).lower()
        physical_grasp_mode = attach_mode in {
            'physical_hold',
            'physical',
            'contact_hold',
            'physical_grasp',
            'contact_physical_grasp',
            'pure_physical_grasp',
            'contact_pure_physical_grasp',
        }
        if (
            mode == 'pick'
            and 'grasp' in descriptor
            and gripper_command == 'close'
            and not self._attached_object_names(tracked_objects, robot_name)
        ):
            configured_search_depth = attach_spec.get(
                'grasp_search_max_depth',
                0.0 if physical_grasp_mode else self._GRASP_SEARCH_MAX_DEPTH,
            )
            grasp_search_max_depth = max(float(configured_search_depth), 0.0)
            if grasp_search_max_depth > 0.0:
                phase_step_counter = float(getattr(task, 'phase_step_counter', 0))
                grasp_search_depth = min(
                    grasp_search_max_depth,
                    max(phase_step_counter, 0.0) * self._GRASP_SEARCH_STEP_PER_PHASE_STEP,
                )
                target_position[2] -= grasp_search_depth
        near_center = target_position[0] > self._CENTER_X_THRESHOLD and abs(target_position[1]) < self._CENTER_Y_THRESHOLD
        clearance = self._effective_clearance(tracked_objects, robot_name, other_robot_name)
        raw_target_spec = phase_spec.get('robot_targets', {}).get(robot_name)
        disable_interarm_gating = bool(
            isinstance(raw_target_spec, dict)
            and raw_target_spec.get('disable_interarm_gating', False)
        )

        if explicit_waypoint:
            planned_pose = {
                'position': target_position.copy(),
                'orientation': target_orientation.copy(),
            }
            if precision_lock is not None:
                planned_pose['position'][:2] = np.asarray(precision_lock['xy'], dtype=float)
                planned_pose['orientation'] = np.asarray(precision_lock['orientation'], dtype=float)
            ignore_payload_floor = bool(
                isinstance(raw_target_spec, dict)
                and raw_target_spec.get('direct_payload_ignore_floor_height', False)
            )
            min_payload_height = self._payload_floor_height(tracked_objects, robot_name)
            if min_payload_height is not None and not ignore_payload_floor:
                planned_pose['position'][2] = max(float(planned_pose['position'][2]), min_payload_height)
            if not disable_interarm_gating and self._interarm_conflict_with_clearance(
                planned_pose['position'],
                other_position,
                other_target_position,
                clearance=clearance,
            ):
                own_priority = self._motion_priority(phase_spec, robot_name, target_name, tracked_objects)
                other_priority = self._motion_priority(
                    phase_spec,
                    other_robot_name,
                    other_target_name,
                    tracked_objects,
                )
                if own_priority < other_priority:
                    return self._wait_pose(task, robot_name, target_orientation)
            max_step = None
            max_vertical = None
            if isinstance(raw_target_spec, dict) and bool(raw_target_spec.get('direct_payload_motion', False)):
                if raw_target_spec.get('direct_payload_max_position_step') is not None:
                    max_step = float(raw_target_spec['direct_payload_max_position_step'])
                if raw_target_spec.get('direct_payload_max_vertical_step') is not None:
                    max_vertical = float(raw_target_spec['direct_payload_max_vertical_step'])
            limited_position = self._rate_limit_position(
                current_position,
                planned_pose['position'],
                mode=mode,
                max_step=max_step,
                max_vertical=max_vertical,
            )
            planned_orientation = self._blend_orientation(
                current_orientation,
                planned_pose['orientation'],
                mode=mode,
            )
            return {
                'position': limited_position,
                'orientation': planned_orientation,
            }

        safe_z = self._safe_transit_height(current_position, target_position, mode=mode, near_center=near_center)
        approach_position = self._approach_position(target_position, robot_name=robot_name, mode=mode)
        if mode in {'hold', 'move'}:
            transit_orientation = self._normalize_quat(current_orientation)
        else:
            transit_orientation = self._blend_orientation(
                current_orientation,
                target_orientation,
                mode=mode,
            )
        lane_y = float(target_position[1])
        if abs(lane_y) < self._CENTER_Y_THRESHOLD:
            lane_y = self._side_sign(robot_name) * self._CENTER_Y_THRESHOLD
        lane_position = np.array([target_position[0], lane_y, safe_z], dtype=float)

        if not disable_interarm_gating and self._should_wait_for_other_robot(
            task=task,
            phase_spec=phase_spec,
            robot_name=robot_name,
            other_robot_name=other_robot_name,
            target_name=target_name,
            other_target_name=other_target_name,
            tracked_robots=tracked_robots,
            tracked_objects=tracked_objects,
        ):
            return self._wait_pose(task, robot_name, target_orientation)

        stage = 'final'
        xy_error = self._xy_distance(current_position, target_position)
        approach_xy_error = self._xy_distance(current_position, approach_position)
        if mode in {'pick', 'insert', 'hold'}:
            if (
                current_position[2] < safe_z - self._VERTICAL_THRESHOLD
                and xy_error > self._MOVE_XY_THRESHOLD
            ):
                stage = 'lift'
            elif self._requires_center_lane(target_position, mode) and self._xy_distance(current_position, lane_position) > self._LANE_ENTRY_THRESHOLD:
                stage = 'lane'
            elif approach_xy_error > self._MOVE_XY_THRESHOLD:
                stage = 'hover'
            elif (
                abs(current_position[2] - target_position[2]) > self._PRECISION_DESCENT_THRESHOLD
                or xy_error > self._PRECISION_DESCENT_THRESHOLD
            ):
                stage = 'descend'
        else:
            if (
                target_position[2] < safe_z - self._VERTICAL_THRESHOLD
                and current_position[2] < safe_z - self._VERTICAL_THRESHOLD
                and self._xy_distance(current_position, target_position) > 0.12
            ):
                stage = 'lift'

        lift_pose = {
            'position': np.array([current_position[0], current_position[1], safe_z], dtype=float),
            'orientation': transit_orientation,
        }
        lane_pose = self._lane_pose(
            target_position,
            robot_name=robot_name,
            safe_z=safe_z,
            orientation=transit_orientation,
        )
        hover_pose = {
            'position': np.array([approach_position[0], approach_position[1], safe_z], dtype=float),
            'orientation': transit_orientation,
        }
        final_pose = {
            'position': target_position,
            'orientation': target_orientation,
        }
        stage_to_pose = {
            'lift': lift_pose,
            'lane': lane_pose,
            'hover': hover_pose,
            'descend': final_pose,
            'final': final_pose,
        }
        planned_pose = stage_to_pose[stage]

        min_payload_height = self._payload_floor_height(tracked_objects, robot_name)
        if min_payload_height is not None and stage in {'lift', 'lane', 'hover'}:
            planned_pose = {
                'position': np.array(
                    [
                        planned_pose['position'][0],
                        planned_pose['position'][1],
                        max(float(planned_pose['position'][2]), min_payload_height),
                    ],
                    dtype=float,
                ),
                'orientation': planned_pose['orientation'],
            }

        if (
            not disable_interarm_gating
            and
            other_position is not None
            and other_target_reached is not True
            and self._requires_center_lane(target_position, mode)
            and np.asarray(other_position, dtype=float)[0] > self._CENTER_X_THRESHOLD
            and abs(np.asarray(other_position, dtype=float)[1]) < self._CENTERLINE_THRESHOLD
            and float(np.asarray(other_position, dtype=float)[2]) < safe_z
            and stage in {'lane', 'hover', 'descend'}
        ):
            return self._wait_pose(task, robot_name, target_orientation)

        if not disable_interarm_gating and self._interarm_conflict_with_clearance(
            planned_pose['position'],
            other_position,
            other_target_position,
            clearance=clearance,
        ):
            own_priority = self._motion_priority(phase_spec, robot_name, target_name, tracked_objects)
            other_priority = self._motion_priority(
                phase_spec,
                other_robot_name,
                other_target_name,
                tracked_objects,
            )
            if own_priority < other_priority:
                return self._wait_pose(task, robot_name, target_orientation)
            planned_pose = self._conflict_escape_pose(
                target_position=planned_pose['position'],
                target_orientation=planned_pose['orientation'],
                robot_name=robot_name,
                safe_z=safe_z,
                near_center=near_center,
                tracked_objects=tracked_objects,
            )

        limited_position = planned_pose['position']
        if stage in {'descend', 'final'} and mode in {'pick', 'insert', 'hold'}:
            limited_position = self._rate_limit_position(
                current_position,
                planned_pose['position'],
                mode=mode,
            )
        planned_orientation = self._blend_orientation(
            current_orientation,
            planned_pose['orientation'],
            mode=mode,
        )
        return {
            'position': limited_position,
            'orientation': planned_orientation,
        }

    def _compose_robot_action(self, task, robot_name: str, phase_spec: dict, tracked_robots: dict, tracked_objects: dict):
        action = OrderedDict()
        state = self._refresh_robot_plan_if_needed(
            task=task,
            robot_name=robot_name,
            phase_spec=phase_spec,
            tracked_robots=tracked_robots,
            tracked_objects=tracked_objects,
        )
        trajectory_target = self._trajectory_target(state)
        current_joint_positions = self._current_arm_joint_positions(task, robot_name)

        def _joint_hold_action():
            if current_joint_positions is None:
                return None
            return [current_joint_positions.tolist()]

        def _resolved_gripper_command(current_position=None, target_entry: dict | None = None):
            phase_gripper_command = phase_spec.get('gripper_commands', {}).get(robot_name)
            if phase_gripper_command is None:
                self._clear_grasp_close_state(state)
                return None
            normalized_command = str(phase_gripper_command).lower()
            if (
                normalized_command == 'close'
                and state.get('gripper_close_latched_phase_key') == state.get('phase_key')
            ):
                return 0.0
            attach_spec = self._phase_attach_spec_for_robot(phase_spec, robot_name) or {}
            preclose_open_steps = max(int(attach_spec.get('grasp_preclose_open_steps', 0)), 0)
            phase_name = str(phase_spec.get('name', '')).lower()
            if normalized_command == 'close' and preclose_open_steps > 0 and 'grasp' in phase_name:
                if self._attached_object_names(tracked_objects, robot_name):
                    state['gripper_close_latched_phase_key'] = state.get('phase_key')
                    self._clear_grasp_close_state(state)
                    return 0.0
                if state.get('grasp_close_started_step') is None:
                    final_waypoint_chain = state.get('latched_waypoint_chain') or []
                    final_pose = (
                        target_entry.get('pose')
                        if target_entry is not None and not final_waypoint_chain
                        else (
                            final_waypoint_chain[-1].get('pose')
                            if final_waypoint_chain
                            else None
                        )
                    )
                    if final_pose is not None:
                        state['grasp_close_pose'] = {
                            'position': np.asarray(final_pose['position'], dtype=float),
                            'orientation': self._normalize_quat(final_pose['orientation']),
                        }
                    close_ramp_steps = max(
                        int(attach_spec.get('grasp_close_ramp_steps', self._GRASP_CLOSE_RAMP_STEPS)),
                        0,
                    )
                    close_hold_steps = max(
                        int(attach_spec.get('grasp_close_hold_steps', self._GRASP_CLOSE_HOLD_STEPS)),
                        0,
                    )
                    state['grasp_close_started_step'] = self._policy_step
                    state['grasp_close_hold_until_step'] = (
                        self._policy_step + preclose_open_steps + close_ramp_steps + close_hold_steps
                    )
                close_started_step = int(state.get('grasp_close_started_step') or self._policy_step)
                elapsed_since_close_start = max(self._policy_step - close_started_step, 0)
                if elapsed_since_close_start < preclose_open_steps:
                    return self._GRASP_CLOSE_DEFAULT_OPENNESS
                elapsed_close_steps = max(elapsed_since_close_start - preclose_open_steps, 0)
                close_ramp_steps = max(
                    int(attach_spec.get('grasp_close_ramp_steps', self._GRASP_CLOSE_RAMP_STEPS)),
                    0,
                )
                if close_ramp_steps <= 0:
                    gripper_openness = 0.0
                else:
                    gripper_openness = float(
                        np.clip(1.0 - (elapsed_close_steps / float(close_ramp_steps)), 0.0, 1.0)
                    )
                if gripper_openness <= 1e-3:
                    state['gripper_close_latched_phase_key'] = state.get('phase_key')
                return gripper_openness
            if normalized_command != 'close' or current_position is None or target_entry is None:
                if normalized_command != 'close':
                    self._clear_grasp_close_state(state)
                return phase_gripper_command
            if self._attached_object_names(tracked_objects, robot_name):
                state['gripper_close_latched_phase_key'] = state.get('phase_key')
                self._clear_grasp_close_state(state)
                return 0.0
            descriptor = self._descriptor(
                target_entry.get('name') or state.get('latched_target_name') or robot_name,
                phase_spec,
            )
            if target_entry.get('mode') != 'pick' or 'grasp' not in descriptor:
                self._clear_grasp_close_state(state)
                return phase_gripper_command
            final_waypoint_chain = state.get('latched_waypoint_chain') or []
            final_pose = (
                target_entry.get('pose')
                if not final_waypoint_chain
                else final_waypoint_chain[-1].get('pose', target_entry.get('pose'))
            )
            if final_pose is None:
                self._clear_grasp_close_state(state)
                return phase_gripper_command
            if state.get('grasp_close_started_step') is None:
                current_position_array = np.asarray(current_position, dtype=float)
                final_position = np.asarray(final_pose['position'], dtype=float)
                xy_error = self._xy_distance(current_position_array, final_position)
                z_error = abs(float(current_position_array[2] - final_position[2]))
                attach_tolerance = float(
                    attach_spec.get(
                        'position_tolerance',
                        target_entry.get('tolerance', 0.025),
                    )
                    or 0.025
                )
                close_xy_threshold = float(
                    attach_spec.get(
                        'close_xy_threshold',
                        min(0.012, max(0.008, attach_tolerance * 0.6)),
                    )
                )
                close_z_threshold = float(
                    attach_spec.get(
                        'close_z_threshold',
                        min(0.010, max(0.006, attach_tolerance * 0.5)),
                    )
                )
                if xy_error > close_xy_threshold or z_error > close_z_threshold:
                    return self._GRASP_CLOSE_DEFAULT_OPENNESS
                state['grasp_close_started_step'] = self._policy_step
                state['grasp_close_pose'] = {
                    'position': final_position.copy(),
                    'orientation': self._normalize_quat(final_pose['orientation']),
                }
                close_ramp_steps = max(
                    int(attach_spec.get('grasp_close_ramp_steps', self._GRASP_CLOSE_RAMP_STEPS)),
                    0,
                )
                close_hold_steps = max(
                    int(attach_spec.get('grasp_close_hold_steps', self._GRASP_CLOSE_HOLD_STEPS)),
                    0,
                )
                preclose_open_steps = max(int(attach_spec.get('grasp_preclose_open_steps', 0)), 0)
                state['grasp_close_hold_until_step'] = (
                    self._policy_step + preclose_open_steps + close_ramp_steps + close_hold_steps
                )
            preclose_open_steps = max(int(attach_spec.get('grasp_preclose_open_steps', 0)), 0)
            close_ramp_steps = max(
                int(attach_spec.get('grasp_close_ramp_steps', self._GRASP_CLOSE_RAMP_STEPS)),
                0,
            )
            close_started_step = int(state.get('grasp_close_started_step') or self._policy_step)
            elapsed_since_close_start = max(self._policy_step - close_started_step, 0)
            if elapsed_since_close_start < preclose_open_steps:
                return self._GRASP_CLOSE_DEFAULT_OPENNESS
            elapsed_close_steps = max(elapsed_since_close_start - preclose_open_steps, 0)
            if close_ramp_steps <= 0:
                gripper_openness = 0.0
            else:
                gripper_openness = float(
                    np.clip(1.0 - (elapsed_close_steps / float(close_ramp_steps)), 0.0, 1.0)
                )
            if gripper_openness <= 1e-3:
                state['gripper_close_latched_phase_key'] = state.get('phase_key')
            return gripper_openness

        if trajectory_target is None:
            action[arm_ik_cfg.name] = self._hold_pose()
            joint_hold = _joint_hold_action()
            if joint_hold is not None:
                action[arm_joint_cfg.name] = joint_hold
        else:
            robot_tracking = tracked_robots.get(robot_name, {})
            current_position = self._as_array(robot_tracking.get('position'), default=trajectory_target['pose']['position'])
            current_orientation = self._as_array(robot_tracking.get('orientation'), default=trajectory_target['pose']['orientation'])
            if (
                self._should_freeze_for_release(phase_spec, robot_name, tracked_objects)
                or self._should_freeze_for_attached_grasp(
                    phase_spec,
                    robot_name,
                    tracked_objects,
                )
                or self._should_freeze_for_grasp_close(
                    phase_spec,
                    robot_name,
                    tracked_objects,
                    state,
                )
            ):
                hold_grasp_target = bool(
                    (self._phase_attach_spec_for_robot(phase_spec, robot_name) or {}).get(
                        'hold_grasp_target_during_close',
                        True,
                    )
                )
                grasp_close_pose = state.get('grasp_close_pose') if hold_grasp_target else None
                if isinstance(grasp_close_pose, dict):
                    frozen_position = np.asarray(grasp_close_pose['position'], dtype=float)
                    frozen_orientation = self._normalize_quat(grasp_close_pose['orientation'])
                else:
                    frozen_position = current_position
                    frozen_orientation = current_orientation
                frozen_pose = [frozen_position.tolist(), frozen_orientation.tolist()]
                action[arm_ik_cfg.name] = frozen_pose
                joint_hold = _joint_hold_action()
                if joint_hold is not None:
                    action[arm_joint_cfg.name] = joint_hold
                state['last_action'] = frozen_pose
                state['last_joint_action'] = joint_hold
                state['active_command_pose'] = {
                    'position': frozen_position.copy(),
                    'orientation': frozen_orientation.copy(),
                }
                gripper_command = _resolved_gripper_command(current_position=current_position, target_entry=trajectory_target)
                if gripper_command is not None:
                    action[gripper_cfg.name] = [self._gripper_controller_action(gripper_command)]
                return action

            self._advance_planned_joint_trajectory_index(
                state=state,
                current_joint_positions=current_joint_positions,
            )
            planned_joint_target = self._planned_joint_target(state)
            if planned_joint_target is not None:
                arm_action = [
                    planned_joint_target['pose']['position'].tolist(),
                    planned_joint_target['pose']['orientation'].tolist(),
                ]
                joint_action = [planned_joint_target['joint_positions'].tolist()]
                if (
                    state['last_action'] is not None
                    and state.get('last_joint_action') is not None
                    and self._policy_step - int(state['last_update_step']) < self._ACTION_UPDATE_INTERVAL_STEPS
                ):
                    action[arm_ik_cfg.name] = state['last_action']
                    action[arm_joint_cfg.name] = state['last_joint_action']
                else:
                    action[arm_ik_cfg.name] = arm_action
                    action[arm_joint_cfg.name] = joint_action
                    state['last_action'] = arm_action
                    state['last_joint_action'] = joint_action
                    state['last_update_step'] = self._policy_step
                state['active_command_pose'] = {
                    'position': np.asarray(planned_joint_target['pose']['position'], dtype=float),
                    'orientation': np.asarray(planned_joint_target['pose']['orientation'], dtype=float),
                }
                state['active_target_name'] = planned_joint_target['name'] or state.get('latched_target_name') or robot_name
                gripper_command = _resolved_gripper_command(
                    current_position=current_position,
                    target_entry=planned_joint_target,
                )
                if gripper_command is not None:
                    action[gripper_cfg.name] = [self._gripper_controller_action(gripper_command)]
                return action

            self._advance_trajectory_index(
                state=state,
                current_position=current_position,
                current_orientation=current_orientation,
            )
            trajectory_target = self._trajectory_target(state)
            if trajectory_target is None:
                action[arm_ik_cfg.name] = self._hold_pose()
                joint_hold = _joint_hold_action()
                if joint_hold is not None:
                    action[arm_joint_cfg.name] = joint_hold
                gripper_command = _resolved_gripper_command(current_position=current_position, target_entry=None)
                if gripper_command is not None:
                    action[gripper_cfg.name] = [self._gripper_controller_action(gripper_command)]
                return action

            if (
                state['last_action'] is not None
                and self._policy_step - int(state['last_update_step']) < self._ACTION_UPDATE_INTERVAL_STEPS
            ):
                action[arm_ik_cfg.name] = state['last_action']
                if state.get('last_joint_action') is not None:
                    action[arm_joint_cfg.name] = state['last_joint_action']
                gripper_command = _resolved_gripper_command(
                    current_position=current_position,
                    target_entry=trajectory_target,
                )
                if gripper_command is not None:
                    action[gripper_cfg.name] = [self._gripper_controller_action(gripper_command)]
                return action

            other_robot_name = 'franka_right' if robot_name == 'franka_left' else 'franka_left'
            other_tracking = tracked_robots.get(other_robot_name, {})
            other_state = self._robot_state(other_robot_name)
            other_active_command_pose = other_state.get('active_command_pose')
            other_target_position = (
                None
                if other_active_command_pose is None
                else other_active_command_pose.get('position')
            )
            if other_target_position is None:
                other_target_position = other_tracking.get('task_target')
            precision_lock = self._maybe_capture_precision_lock(
                state=state,
                phase_spec=phase_spec,
                robot_name=robot_name,
                target_name=trajectory_target['name'] or state.get('latched_target_name') or robot_name,
                mode=trajectory_target['mode'],
                current_position=current_position,
                current_orientation=current_orientation,
                target_pose=trajectory_target['pose'],
            )
            constrained_pose = self._plan_constrained_pose(
                task=task,
                robot_name=robot_name,
                phase_spec=phase_spec,
                target_name=trajectory_target['name'] or state.get('latched_target_name') or robot_name,
                current_position=current_position,
                current_orientation=current_orientation,
                target_position=trajectory_target['pose']['position'],
                target_orientation=trajectory_target['pose']['orientation'],
                other_position=other_tracking.get('position'),
                other_target_position=other_target_position,
                other_target_name=other_state.get('active_target_name') or other_tracking.get('target_name') or '',
                other_target_reached=other_tracking.get('target_reached'),
                tracked_robots=tracked_robots,
                tracked_objects=tracked_objects,
                other_robot_name=other_robot_name,
                explicit_waypoint=True,
                precision_lock=precision_lock,
            )
            arm_action = [constrained_pose['position'].tolist(), constrained_pose['orientation'].tolist()]
            action[arm_ik_cfg.name] = arm_action
            precision_lock_phase = self._is_precision_lock_phase(
                phase_spec=phase_spec,
                robot_name=robot_name,
                target_name=trajectory_target['name'] or state.get('latched_target_name') or robot_name,
                mode=trajectory_target['mode'],
            )
            joint_action = self._joint_trajectory_action_for_pose(
                task,
                robot_name,
                constrained_pose,
                mode=trajectory_target['mode'],
            ) if not self._joint_action_disabled_for_phase(
                task=task,
                robot_name=robot_name,
                phase_spec=phase_spec,
                tracked_objects=tracked_objects,
            ) else None
            if precision_lock_phase:
                joint_action = None
            if joint_action is not None:
                action[arm_joint_cfg.name] = joint_action
            state['last_action'] = arm_action
            state['last_joint_action'] = joint_action
            state['last_update_step'] = self._policy_step
            state['active_command_pose'] = {
                'position': np.asarray(constrained_pose['position'], dtype=float),
                'orientation': np.asarray(constrained_pose['orientation'], dtype=float),
            }
            state['active_target_name'] = trajectory_target['name'] or state.get('latched_target_name') or robot_name

        active_pose = state.get('active_command_pose')
        resolved_position = None if active_pose is None else active_pose.get('position')
        gripper_command = _resolved_gripper_command(
            current_position=current_position if trajectory_target is not None else resolved_position,
            target_entry=trajectory_target,
        )
        if gripper_command is not None:
            action[gripper_cfg.name] = [self._gripper_controller_action(gripper_command)]
        return action

    def act(self, task):
        self._ensure_task_context(task)
        phase_spec = task.get_current_phase_spec()
        tracked_robots = task.get_tracked_robot_states(phase_spec=phase_spec)
        tracked_objects = task.get_tracked_object_states()
        actions = {}
        for robot_name in task.config.robot_names:
            action = self._local_skill_executor.action_for(
                task=task,
                robot_name=robot_name,
                phase_spec=phase_spec,
                tracked_robots=tracked_robots,
                tracked_objects=tracked_objects,
            )
            if isinstance(action, dict) and action.get('__local_skill_failure__'):
                if hasattr(task, '_set_terminal_state'):
                    task._set_terminal_state(
                        'failed',
                        reason=f"local-skill-failure:{action.get('reason', 'unknown')}",
                        status='failed',
                        detail=action.get('diagnostics', {}),
                    )
                else:
                    task.failed = True
                    task.terminal_reason = f"local-skill-failure:{action.get('reason', 'unknown')}"
                action = OrderedDict()
            if action is None:
                action = self._compose_robot_action(
                    task=task,
                    robot_name=robot_name,
                    phase_spec=phase_spec,
                    tracked_robots=tracked_robots,
                    tracked_objects=tracked_objects,
                )
            actions[robot_name] = action
        return actions
