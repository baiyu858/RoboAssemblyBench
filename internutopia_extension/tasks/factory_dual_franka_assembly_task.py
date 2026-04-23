from __future__ import annotations

import copy
import json
import os
from collections import OrderedDict

import numpy as np

from internutopia.core.scene.scene import IScene
from internutopia.core.task import BaseTask
from internutopia.core.util.joint import create_joint
from internutopia.core.util.physics import activate_collider, deactivate_collider
from internutopia_extension.configs.tasks.factory_dual_franka_assembly_task import (
    FactoryDualFrankaAssemblyTaskCfg,
)
from toolkits.factory_dual_franka_assembly.planner_primitives import (
    compose_pose,
    pose_error,
    pose_within_tolerance,
    quat_conjugate,
    quat_multiply,
    quat_rotate,
    relative_pose,
)


@BaseTask.register('FactoryDualFrankaAssemblyTask')
class FactoryDualFrankaAssemblyTask(BaseTask):
    _ARM_IK_CONTROLLER_NAME = 'arm_ik_controller'
    _GRIPPER_CONTROLLER_NAME = 'gripper_controller'
    _HAND_LINK_NAME = 'panda_hand'
    _LEFT_FINGER_LINK_NAME = 'panda_leftfinger'
    _RIGHT_FINGER_LINK_NAME = 'panda_rightfinger'
    _FINGERTIP_LOCAL_POSITION = (0.0, 0.0, 0.045)
    _FINGER_CONTACT_SAMPLE_POINTS = (
        (0.0, 0.0, 0.045),
        (0.0, 0.0, 0.032),
        (0.0, 0.0, 0.020),
        (0.0, 0.0075, 0.040),
        (0.0, -0.0075, 0.040),
        (0.0, 0.0075, 0.028),
        (0.0, -0.0075, 0.028),
        (0.006, 0.0, 0.040),
        (-0.006, 0.0, 0.040),
    )
    _ROBOT_TARGET_TOLERANCE_FLOOR = 0.03
    _ATTACH_POSITION_TOLERANCE = 0.035
    _LOCK_POSITION_TOLERANCE = 0.025
    _DETACH_POSITION_TOLERANCE = 0.05
    _ATTACH_DISTANCE_MARGIN = 0.025
    _ATTACH_LATERAL_MARGIN = 0.02
    _ATTACH_VERTICAL_MARGIN = 0.025
    _ATTACH_TOP_CLEARANCE = 0.006
    _ATTACH_SUPPORT_HEIGHT_MARGIN = 0.012
    _GRIPPER_CLOSED_THRESHOLD = 0.03
    _FINGER_CONTACT_FORCE_THRESHOLD = 0.5
    _FINGER_CONTACT_DISTANCE = 0.003
    _CONTACT_FORCE_DT = 1.0 / 240.0
    _RELEASE_LOCK_MIN_STEPS = 6
    _PHYSICAL_HOLD_POSITION_SLIP = 0.03
    _PHYSICAL_HOLD_ORIENTATION_SLIP = 0.75
    _LEGACY_RIGHT_GRIPPER_LOCAL_POSITION = np.array([0.0, 0.0, 0.1], dtype=float)
    _LEGACY_RIGHT_GRIPPER_LOCAL_ORIENTATION = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    _JOINT_ATTACHMENT_MODES = frozenset(
        {
            'fixed_joint',
            'joint',
            'constraint',
            'contact_fixed_joint',
            'physical_joint',
            'contact_joint',
            'contact_constraint',
        }
    )

    def __init__(self, config: FactoryDualFrankaAssemblyTaskCfg, scene: IScene):
        super().__init__(config, scene)
        self.step_counter = 0
        self.max_steps = config.max_steps
        self.phase_specs = list(config.phase_specs)
        self.target_poses = {
            name: {
                'position': np.asarray(pose['position'], dtype=float),
                'orientation': np.asarray(pose['orientation'], dtype=float),
            }
            for name, pose in config.target_poses.items()
        }
        self.phase_index = 0
        self.phase = self.phase_specs[0]['name'] if self.phase_specs else 'complete'
        self.phase_history = [self.phase]
        self.phase_transition_history = (
            [
                {
                    'event': 'initialize',
                    'phase': self.phase,
                    'phase_index': self.phase_index,
                    'step_counter': 0,
                    'status': 'running' if self.phase_specs else 'complete',
                }
            ]
            if self.phase_specs
            else []
        )
        self.phase_step_counter = 0
        self.phase_entry_step = 0
        self.phase_status = 'running' if self.phase_specs else 'complete'
        self.phase_attempts = {self.phase: 1} if self.phase_specs else {}
        self.phase_timeout_count = 0
        self.phase_recovery_count = 0
        self.success = False
        self.failed = False
        self.terminal_reason = None
        self.last_transition_reason = None
        self._phase_initialized = False
        self._resolved_objects = {}
        self._object_prims = {}
        self._attachments = {}
        self._attachment_joints = {}
        self._locked_targets = {}
        self._object_collision_enabled = {}
        self._contact_probes = {}
        self._handoff_history = []
        self._recovery_history = []
        self._object_metadata_map = {
            metadata['name']: copy.deepcopy(metadata)
            for metadata in config.object_metadata
            if isinstance(metadata, dict) and metadata.get('name') is not None
        }

    @property
    def cfg(self) -> FactoryDualFrankaAssemblyTaskCfg:
        return self.config

    def get_current_phase_spec(self) -> dict:
        if 0 <= self.phase_index < len(self.phase_specs):
            return self.phase_specs[self.phase_index]
        return {}

    def get_target_pose(self, target_name: str) -> dict:
        pose = self.target_poses[target_name]
        return {
            'position': pose['position'].tolist(),
            'orientation': pose['orientation'].tolist(),
        }

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _phase_index_for_name(self, phase_name: str | None):
        if phase_name is None:
            return None
        for index, phase_spec in enumerate(self.phase_specs):
            if phase_spec.get('name') == phase_name:
                return index
        return None

    def _resolve_target_pose_spec(self, target_like):
        if target_like is None:
            return None, None, None, {}
        if isinstance(target_like, str):
            pose = self.target_poses[target_like]
            return target_like, pose['position'], pose['orientation'], {}
        if not isinstance(target_like, dict):
            raise TypeError(f'Unsupported target specification: {type(target_like)!r}')

        target_spec = dict(target_like)
        target_name = target_spec.get('target') or target_spec.get('target_name') or target_spec.get('name')
        pose_spec = target_spec.get('pose')
        if pose_spec is None:
            pose_spec = target_spec
        if not isinstance(pose_spec, dict):
            raise TypeError(f'Unsupported pose specification: {type(pose_spec)!r}')

        position = pose_spec.get('position')
        orientation = pose_spec.get('orientation')
        if position is None and target_name is not None and target_name in self.target_poses:
            position = self.target_poses[target_name]['position']
        if orientation is None and target_name is not None and target_name in self.target_poses:
            orientation = self.target_poses[target_name]['orientation']
        if position is None:
            raise ValueError('A target specification must define position or reference a named target.')

        position = np.asarray(position, dtype=float)
        orientation_array = None if orientation is None else np.asarray(orientation, dtype=float)
        return target_name, position, orientation_array, target_spec

    def _robot_end_effector_prim_name(self, robot_name: str) -> str | None:
        robot = self.robots.get(robot_name)
        if robot is None:
            return None
        config = getattr(robot, 'config', None)
        prim_name = getattr(config, 'end_effector_prim_name', None)
        if prim_name:
            return str(prim_name)
        articulation = getattr(robot, 'articulation', None)
        return getattr(articulation, '_end_effector_prim_name', None)

    def _compensate_robot_target_pose(
        self,
        robot_name: str,
        target_position,
        target_orientation,
        *,
        target_spec: dict | None = None,
    ):
        if target_position is None:
            return target_position, target_orientation
        if self._robot_end_effector_prim_name(robot_name) != self._HAND_LINK_NAME:
            return target_position, target_orientation

        compensation_mode = 'full' if target_spec is None else target_spec.get('ik_frame_compensation', 'full')
        if compensation_mode in {'none', 'disabled', False}:
            return target_position, target_orientation

        position = np.asarray(target_position, dtype=float)
        orientation = (
            np.asarray(target_orientation, dtype=float)
            if target_orientation is not None
            else np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        )
        compensated_position = position - quat_rotate(orientation, self._LEGACY_RIGHT_GRIPPER_LOCAL_POSITION)

        if compensation_mode in {'full', 'legacy_right_gripper_full'}:
            compensated_orientation = quat_multiply(
                orientation,
                quat_conjugate(self._LEGACY_RIGHT_GRIPPER_LOCAL_ORIENTATION),
            )
        else:
            compensated_orientation = target_orientation

        return compensated_position, compensated_orientation

    def resolve_robot_target_pose(self, robot_name: str, target_like):
        target_name, target_position, target_orientation, target_spec = self._resolve_target_pose_spec(target_like)
        target_position, target_orientation = self._compensate_robot_target_pose(
            robot_name,
            target_position,
            target_orientation,
            target_spec=target_spec,
        )
        return target_name, target_position, target_orientation, target_spec

    @staticmethod
    def _target_tolerances(target_spec: dict | None, default_position_tolerance: float, default_orientation_tolerance):
        if target_spec is None:
            return float(default_position_tolerance), default_orientation_tolerance

        position_tolerance = target_spec.get('position_tolerance', target_spec.get('tolerance', default_position_tolerance))
        orientation_tolerance = target_spec.get('orientation_tolerance', default_orientation_tolerance)
        if target_spec.get('position_only') or target_spec.get('ignore_orientation'):
            orientation_tolerance = None
        return float(position_tolerance), None if orientation_tolerance is None else float(orientation_tolerance)

    @classmethod
    def _effective_robot_target_tolerance(cls, tolerance: float) -> float:
        return max(float(tolerance), cls._ROBOT_TARGET_TOLERANCE_FLOOR)

    def _transition_details(self, *, reason: str, transition_type: str, from_phase: str, to_phase: str, from_phase_index: int, to_phase_index: int, status: str, detail: dict | None = None):
        payload = {
            'from_phase': from_phase,
            'from_phase_index': from_phase_index,
            'to_phase': to_phase,
            'to_phase_index': to_phase_index,
            'transition_type': transition_type,
            'reason': reason,
            'status': status,
            'step_counter': int(self.step_counter),
            'phase_step_counter': int(self.phase_step_counter),
        }
        if detail:
            payload.update(copy.deepcopy(detail))
        return payload

    def _set_phase(self, new_phase_index: int, *, reason: str = 'advance', transition_type: str = 'advance', status: str = 'running', detail: dict | None = None):
        previous_phase = self.phase
        previous_index = self.phase_index
        self.phase_index = new_phase_index
        self.phase = self.phase_specs[new_phase_index]['name']
        self.phase_history.append(self.phase)
        self.phase_step_counter = 0
        self.phase_entry_step = self.step_counter
        self.phase_status = status
        self.phase_attempts[self.phase] = self.phase_attempts.get(self.phase, 0) + 1
        self.last_transition_reason = reason
        self.phase_transition_history.append(
            self._transition_details(
                reason=reason,
                transition_type=transition_type,
                from_phase=previous_phase,
                to_phase=self.phase,
                from_phase_index=previous_index,
                to_phase_index=new_phase_index,
                status=status,
                detail=detail,
            )
        )
        self._phase_initialized = False

    def _extract_object_name(self, entry):
        if isinstance(entry, dict):
            return entry.get('object') or entry.get('name')
        return entry

    def _apply_phase_actions(self, phase_spec: dict | None):
        if not phase_spec:
            return

        for object_entry in self._as_list(phase_spec.get('unlock')):
            object_name = self._extract_object_name(object_entry)
            if object_name is not None:
                self._unlock_object(object_name)

    def _resolve_timeout_spec(self, phase_spec: dict) -> dict:
        timeout_spec: dict = {}
        raw_timeout = phase_spec.get('timeout')
        if isinstance(raw_timeout, dict):
            timeout_spec.update(copy.deepcopy(raw_timeout))
        elif raw_timeout is not None:
            timeout_spec['steps'] = raw_timeout

        for key in ('on_timeout', 'recovery', 'failure'):
            if key not in phase_spec or phase_spec[key] is None:
                continue
            value = phase_spec[key]
            if isinstance(value, dict):
                timeout_spec.update(copy.deepcopy(value))
            elif isinstance(value, str):
                timeout_spec['phase'] = value
                if key == 'failure':
                    timeout_spec.setdefault('action', 'fail')
            elif isinstance(value, int):
                timeout_spec['phase_index'] = int(value)

        if 'steps' not in timeout_spec:
            timeout_steps = phase_spec.get('timeout_steps')
            if timeout_steps is None and isinstance(raw_timeout, dict):
                timeout_steps = raw_timeout.get('steps')
            if timeout_steps is None:
                timeout_steps = self.cfg.phase_timeout_steps
            if timeout_steps is not None:
                timeout_spec['steps'] = int(timeout_steps)

        timeout_spec.setdefault('action', phase_spec.get('timeout_action', self.cfg.phase_timeout_action))
        return timeout_spec

    def _resolve_transition_target(self, transition_spec: dict, default_index: int | None = None):
        if transition_spec is None:
            return default_index
        if isinstance(transition_spec, str):
            return self._phase_index_for_name(transition_spec)
        if isinstance(transition_spec, int):
            return int(transition_spec)

        for key in ('phase_index', 'index'):
            if transition_spec.get(key) is not None:
                return int(transition_spec[key])

        for key in ('phase', 'target_phase', 'next_phase', 'recovery_phase', 'retry_phase', 'fallback_phase'):
            value = transition_spec.get(key)
            if value is None:
                continue
            if isinstance(value, int):
                return int(value)
            if isinstance(value, str):
                return self._phase_index_for_name(value)

        return default_index

    def _resolve_handoff_spec(self, handoff_spec):
        if handoff_spec is None:
            return None
        if isinstance(handoff_spec, str):
            return {'object': handoff_spec}
        if not isinstance(handoff_spec, dict):
            return None
        return dict(handoff_spec)

    def _handoff_object(self, handoff_spec):
        handoff_spec = self._resolve_handoff_spec(handoff_spec)
        if not handoff_spec:
            return

        object_name = handoff_spec.get('object')
        target_robot = handoff_spec.get('to_robot') or handoff_spec.get('target_robot') or handoff_spec.get('destination_robot') or handoff_spec.get('to')
        source_robot = handoff_spec.get('from_robot') or handoff_spec.get('source_robot') or handoff_spec.get('from')
        if object_name is None or target_robot is None:
            return

        align_target = handoff_spec.get('target') or handoff_spec.get('handoff_target') or handoff_spec.get('align_to') or handoff_spec.get('pose_target')
        if align_target is not None:
            _, target_position, target_orientation, _ = self._resolve_target_pose_spec(align_target)
            if target_position is not None:
                _, current_orientation = self._resolve_object(object_name).get_pose()
                if target_orientation is None:
                    target_orientation = current_orientation
                self._set_object_pose(object_name, target_position, target_orientation)

        current_attachment = self._clear_attachment_state(object_name, enable_collision=False)
        if current_attachment is not None and source_robot is not None and current_attachment.get('robot_name') != source_robot:
            source_robot = current_attachment.get('robot_name')

        attach_phase_spec = None
        attach_spec = None
        if align_target is not None:
            attach_phase_spec = {
                'robot_targets': {
                    target_robot: align_target,
                }
            }
            attach_spec = {
                'object': object_name,
                'robot': target_robot,
                'target': align_target,
            }
        self._attach_object(
            object_name,
            target_robot,
            phase_spec=attach_phase_spec,
            attach_spec=attach_spec,
        )
        handoff_record = {
            'object': object_name,
            'from_robot': source_robot,
            'to_robot': target_robot,
            'align_target': align_target,
            'step_counter': int(self.step_counter),
            'phase': self.phase,
        }
        self._handoff_history.append(handoff_record)
        self.phase_transition_history.append(
            self._transition_details(
                reason='handoff',
                transition_type='handoff',
                from_phase=self.phase,
                to_phase=self.phase,
                from_phase_index=self.phase_index,
                to_phase_index=self.phase_index,
                status=self.phase_status,
                detail=handoff_record,
            )
        )

    def _phase_timeout_steps(self, phase_spec: dict) -> int | None:
        timeout_spec = self._resolve_timeout_spec(phase_spec)
        timeout_steps = timeout_spec.get('steps')
        if timeout_steps is None:
            return None
        return int(timeout_steps)

    def _handle_phase_timeout(self, phase_spec: dict) -> bool:
        timeout_spec = self._resolve_timeout_spec(phase_spec)
        timeout_steps = timeout_spec.get('steps')
        if timeout_steps is None:
            return False
        if self.phase_step_counter < int(timeout_steps):
            return False

        self.phase_timeout_count += 1
        timeout_record = {
            'phase': self.phase,
            'phase_index': self.phase_index,
            'step_counter': int(self.step_counter),
            'phase_step_counter': int(self.phase_step_counter),
            'timeout_steps': int(timeout_steps),
            'action': timeout_spec.get('action'),
        }
        self._recovery_history.append(timeout_record)
        self.phase_transition_history.append(
            self._transition_details(
                reason='timeout',
                transition_type='timeout',
                from_phase=self.phase,
                to_phase=self.phase,
                from_phase_index=self.phase_index,
                to_phase_index=self.phase_index,
                status='timeout',
                detail=timeout_record,
            )
        )

        reset_spec = timeout_spec.get('reset')
        if isinstance(reset_spec, dict):
            self._apply_phase_actions(reset_spec)

        action = str(timeout_spec.get('action', self.cfg.phase_timeout_action)).lower()
        target_index = self._resolve_transition_target(timeout_spec)
        if action in {'retry', 'reset'}:
            self.phase_recovery_count += 1
            self._set_phase(self.phase_index, reason='timeout-retry', transition_type='retry', status='running', detail=timeout_record)
            return True

        if action in {'recover', 'recovery'}:
            if target_index is None and self.cfg.phase_timeout_recovery_phase:
                target_index = self._phase_index_for_name(self.cfg.phase_timeout_recovery_phase)
            if target_index is not None:
                self.phase_recovery_count += 1
                self._set_phase(target_index, reason='timeout-recovery', transition_type='recovery', status='running', detail=timeout_record)
                return True

        if action in {'advance', 'next'}:
            next_index = self.phase_index + 1 if self.phase_index + 1 < len(self.phase_specs) else None
            if next_index is not None:
                self.phase_recovery_count += 1
                self._set_phase(next_index, reason='timeout-advance', transition_type='timeout', status='running', detail=timeout_record)
                return True

        # Some assembly phases, especially the final retreat, are bookkeeping after all objects are
        # already locked into their success targets. If that cleanup motion times out, prefer a
        # completed benchmark episode over mislabeling the whole rollout as failed.
        if self._check_success():
            self._set_terminal_state('complete', reason='success-criteria-met', status='success', detail=timeout_record)
            return True

        self.failed = True
        self.success = False
        self._set_terminal_state('failed', reason='timeout-failure', status='failed', detail=timeout_record)
        return True

    def _resolve_phase_target_pose(self, robot_name: str, robot_target_spec, default_position_tolerance: float, default_orientation_tolerance):
        target_name, target_position, target_orientation, target_spec = self.resolve_robot_target_pose(
            robot_name,
            robot_target_spec,
        )
        position_tolerance, orientation_tolerance = self._target_tolerances(
            target_spec=target_spec,
            default_position_tolerance=default_position_tolerance,
            default_orientation_tolerance=default_orientation_tolerance,
        )
        current_position, current_orientation = self._get_robot_task_pose(robot_name)
        position_error, orientation_error = pose_error(
            current_position=current_position,
            current_orientation=current_orientation,
            target_position=target_position,
            target_orientation=target_orientation,
        )
        target_reached = pose_within_tolerance(
            current_position=current_position,
            current_orientation=current_orientation,
            target_position=target_position,
            target_orientation=target_orientation,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )
        return {
            'target_name': target_name,
            'target_position': None if target_position is None else np.asarray(target_position, dtype=float),
            'target_orientation': None if target_orientation is None else np.asarray(target_orientation, dtype=float),
            'position_tolerance': position_tolerance,
            'orientation_tolerance': orientation_tolerance,
            'position_error': position_error,
            'orientation_error': orientation_error,
            'target_reached': target_reached,
        }

    def _set_terminal_state(self, terminal_phase: str, *, reason: str, status: str, transition_type: str = 'terminal', detail: dict | None = None):
        previous_phase = self.phase
        previous_index = self.phase_index
        self.phase_index = len(self.phase_specs)
        self.phase = terminal_phase
        if not self.phase_history or self.phase_history[-1] != terminal_phase:
            self.phase_history.append(terminal_phase)
        self.phase_status = status
        self.phase_entry_step = self.step_counter
        self.phase_attempts[terminal_phase] = self.phase_attempts.get(terminal_phase, 0) + 1
        self.success = status == 'success'
        self.failed = status == 'failed'
        self.last_transition_reason = reason
        self.terminal_reason = reason
        self.phase_transition_history.append(
            self._transition_details(
                reason=reason,
                transition_type=transition_type,
                from_phase=previous_phase,
                to_phase=terminal_phase,
                from_phase_index=previous_index,
                to_phase_index=self.phase_index,
                status=status,
                detail=detail,
            )
        )
        self._phase_initialized = False

    def _resolve_object(self, object_name: str):
        if object_name in self._resolved_objects:
            return self._resolved_objects[object_name]

        from isaacsim.core.utils.prims import get_prim_at_path

        scene_object = self.objects[object_name]
        rigid_body = self._scene.get(scene_object.name)
        self._resolved_objects[object_name] = rigid_body
        self._object_prims[object_name] = get_prim_at_path(rigid_body.unwrap().prim_path)
        return rigid_body

    def _set_object_collision(self, object_name: str, enabled: bool):
        prim = self._object_prims.get(object_name)
        if prim is None:
            self._resolve_object(object_name)
            prim = self._object_prims.get(object_name)
        if prim is None:
            return
        self._object_collision_enabled[object_name] = bool(enabled)
        try:
            if enabled:
                activate_collider(prim)
            else:
                deactivate_collider(prim)
        except Exception:
            return

    def _robot_rigid_body_by_suffix(self, robot_name: str, suffix: str):
        robot = self.robots.get(robot_name)
        if robot is None:
            return None
        suffix = f'/{suffix}'
        for prim_path, rigid_body in robot._rigid_body_map.items():
            if prim_path.endswith(suffix):
                return rigid_body
        return None

    def _get_robot_hand_rigid_body(self, robot_name: str):
        return self._robot_rigid_body_by_suffix(robot_name, self._HAND_LINK_NAME)

    def _get_robot_finger_rigid_bodies(self, robot_name: str):
        return {
            'left': self._robot_rigid_body_by_suffix(robot_name, self._LEFT_FINGER_LINK_NAME),
            'right': self._robot_rigid_body_by_suffix(robot_name, self._RIGHT_FINGER_LINK_NAME),
        }

    def _get_contact_probe(self, prim_path: str, filter_prim_path: str):
        from omni.isaac.core.prims import RigidPrim
        from omni.isaac.core.utils.prims import is_prim_path_valid

        if not is_prim_path_valid(prim_path) or not is_prim_path_valid(filter_prim_path):
            return None

        key = (prim_path, filter_prim_path)
        probe = self._contact_probes.get(key)
        if probe is not None and getattr(probe, 'is_valid', lambda: False)():
            return probe

        probe_name = (
            f"{self.name or 'dual_franka'}_"
            f"{prim_path.split('/')[-1]}_to_{filter_prim_path.split('/')[-1]}_contact"
        )
        try:
            probe = RigidPrim(
                prim_paths_expr=prim_path,
                name=probe_name,
                track_contact_forces=True,
                prepare_contact_sensors=True,
                contact_filter_prim_paths_expr=[filter_prim_path],
                max_contact_count=8,
            )
        except Exception:
            return None
        self._contact_probes[key] = probe
        return probe

    @staticmethod
    def _tensor_to_numpy(value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        if hasattr(value, 'detach'):
            value = value.detach()
        if hasattr(value, 'cpu'):
            value = value.cpu()
        if hasattr(value, 'numpy'):
            return np.asarray(value.numpy())
        return np.asarray(value)

    def _contact_force_between(self, prim_path: str, filter_prim_path: str) -> float:
        probe = self._get_contact_probe(prim_path, filter_prim_path)
        if probe is None:
            return 0.0
        if hasattr(probe, 'is_physics_handle_valid') and not probe.is_physics_handle_valid():
            return 0.0
        try:
            force_matrix = probe.get_contact_force_matrix(dt=self._CONTACT_FORCE_DT)
        except Exception:
            return 0.0
        force_matrix = self._tensor_to_numpy(force_matrix)
        if force_matrix is None or force_matrix.size == 0:
            return 0.0
        flattened = np.asarray(force_matrix, dtype=float).reshape(-1, 3)
        if flattened.size == 0:
            return 0.0
        return float(np.max(np.linalg.norm(flattened, axis=1)))

    @staticmethod
    def _point_to_world_aabb_gap(point: np.ndarray, center: np.ndarray, half_extents: np.ndarray) -> float:
        outside = np.maximum(np.abs(point - center) - half_extents, 0.0)
        return float(np.linalg.norm(outside))

    def _finger_contact_point(self, rigid_body) -> np.ndarray | None:
        if rigid_body is None:
            return None
        finger_position, finger_orientation = rigid_body.get_pose()
        contact_position, _ = compose_pose(
            base_position=np.asarray(finger_position, dtype=float),
            base_orientation=np.asarray(finger_orientation, dtype=float),
            local_position=np.asarray(self._FINGERTIP_LOCAL_POSITION, dtype=float),
            local_orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
        )
        return np.asarray(contact_position, dtype=float)

    def _finger_contact_sample_points(self, rigid_body) -> list[np.ndarray]:
        if rigid_body is None:
            return []
        finger_position, finger_orientation = rigid_body.get_pose()
        sample_points = []
        for local_position in self._FINGER_CONTACT_SAMPLE_POINTS:
            world_position, _ = compose_pose(
                base_position=np.asarray(finger_position, dtype=float),
                base_orientation=np.asarray(finger_orientation, dtype=float),
                local_position=np.asarray(local_position, dtype=float),
                local_orientation=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
            sample_points.append(np.asarray(world_position, dtype=float))
        return sample_points

    @staticmethod
    def _point_to_local_frame(world_point, *, origin, orientation) -> np.ndarray:
        return quat_rotate(
            quat_conjugate(np.asarray(orientation, dtype=float)),
            np.asarray(world_point, dtype=float) - np.asarray(origin, dtype=float),
        )

    def _local_surface_contact_metrics(
        self,
        local_point: np.ndarray,
        *,
        half_extents: np.ndarray,
        contact_distance: float,
    ) -> dict:
        axis_labels = ('x', 'y', 'z')
        axis_margins = np.asarray(
            [
                self._ATTACH_DISTANCE_MARGIN,
                self._ATTACH_LATERAL_MARGIN,
                self._ATTACH_VERTICAL_MARGIN,
            ],
            dtype=float,
        )
        axis_metrics = {}
        best_axis = None
        best_gap = None
        for axis_index, axis_name in enumerate(axis_labels):
            other_axes = [index for index in range(3) if index != axis_index]
            within_patch = all(
                abs(float(local_point[other_axis])) <= float(half_extents[other_axis] + axis_margins[other_axis])
                for other_axis in other_axes
            )
            surface_gap = abs(abs(float(local_point[axis_index])) - float(half_extents[axis_index]))
            contact = bool(within_patch and surface_gap <= float(contact_distance))
            axis_metrics[axis_name] = {
                'axis': axis_name,
                'surface_gap': surface_gap,
                'signed_coordinate': float(local_point[axis_index]),
                'within_patch': within_patch,
                'contact': contact,
            }
            if within_patch and (best_gap is None or surface_gap < best_gap):
                best_axis = axis_name
                best_gap = surface_gap
        return {
            'local_point': np.asarray(local_point, dtype=float).tolist(),
            'best_axis': best_axis,
            'best_surface_gap': None if best_gap is None else float(best_gap),
            'axes': axis_metrics,
        }

    def _gripper_contact_metrics(self, object_name: str, robot_name: str, attach_spec: dict | None = None) -> dict:
        attach_spec = attach_spec or {}
        object_rigid_body = self._resolve_object(object_name)
        object_position, object_orientation = object_rigid_body.get_pose()
        object_position = np.asarray(object_position, dtype=float)
        object_orientation = np.asarray(object_orientation, dtype=float)
        half_extents = np.maximum(self._get_object_scale(object_name) * 0.5, np.array([0.01, 0.01, 0.01], dtype=float))
        contact_distance = float(attach_spec.get('finger_contact_distance', self._FINGER_CONTACT_DISTANCE))
        contact_force_threshold = float(
            attach_spec.get('contact_force_threshold', self._FINGER_CONTACT_FORCE_THRESHOLD)
        )
        require_dual_contact = bool(attach_spec.get('require_dual_finger_contact', True))

        object_prim_path = object_rigid_body.unwrap().prim_path
        finger_rigid_bodies = self._get_robot_finger_rigid_bodies(robot_name)
        finger_metrics = {}
        finger_contacts = []
        contact_available = False
        for finger_name, rigid_body in finger_rigid_bodies.items():
            if rigid_body is None:
                finger_metrics[finger_name] = {
                    'prim_path': None,
                    'force': 0.0,
                    'surface_gap': None,
                    'has_contact': False,
                    'force_contact': False,
                    'geometric_contact': False,
                    'local_contact': None,
                }
                finger_contacts.append(False)
                continue

            finger_position, _ = rigid_body.get_pose()
            finger_position = np.asarray(finger_position, dtype=float)
            fingertip_position = self._finger_contact_point(rigid_body)
            sample_positions = self._finger_contact_sample_points(rigid_body)
            if not sample_positions:
                fallback_sample = fingertip_position if fingertip_position is not None else finger_position
                sample_positions = [np.asarray(fallback_sample, dtype=float)]
            origin_gap = self._point_to_world_aabb_gap(finger_position, object_position, half_extents)
            best_sample_position = None
            best_sample_gap = None
            best_local_contact = None
            geometric_contact = False
            for sample_position in sample_positions:
                local_sample_position = self._point_to_local_frame(
                    sample_position,
                    origin=object_position,
                    orientation=object_orientation,
                )
                local_contact = self._local_surface_contact_metrics(
                    local_sample_position,
                    half_extents=half_extents,
                    contact_distance=contact_distance,
                )
                sample_gap = local_contact.get('best_surface_gap')
                if sample_gap is None:
                    sample_gap = self._point_to_world_aabb_gap(sample_position, object_position, half_extents)
                sample_gap = float(sample_gap)
                if best_sample_gap is None or sample_gap < best_sample_gap:
                    best_sample_gap = sample_gap
                    best_sample_position = np.asarray(sample_position, dtype=float)
                    best_local_contact = local_contact
                geometric_contact = geometric_contact or any(
                    axis_info['contact'] for axis_info in local_contact['axes'].values()
                )
            surface_gap = origin_gap if best_sample_gap is None else min(origin_gap, best_sample_gap)
            local_contact = best_local_contact or {
                'local_point': None,
                'best_axis': None,
                'best_surface_gap': None,
                'axes': {},
            }
            force = self._contact_force_between(rigid_body.unwrap().prim_path, object_prim_path)
            force_contact = force >= contact_force_threshold
            has_contact = force_contact or geometric_contact
            contact_available = contact_available or force_contact
            finger_metrics[finger_name] = {
                'prim_path': rigid_body.unwrap().prim_path,
                'force': force,
                'surface_gap': surface_gap,
                'origin_gap': origin_gap,
                'fingertip_gap': None
                if fingertip_position is None
                else self._point_to_world_aabb_gap(fingertip_position, object_position, half_extents),
                'fingertip_position': None if fingertip_position is None else fingertip_position.tolist(),
                'sample_count': len(sample_positions),
                'best_sample_position': None if best_sample_position is None else best_sample_position.tolist(),
                'local_contact': local_contact,
                'has_contact': has_contact,
                'force_contact': force_contact,
                'geometric_contact': geometric_contact,
            }
            finger_contacts.append(has_contact)

        pinch_axis = None
        left_axes = ((finger_metrics.get('left') or {}).get('local_contact') or {}).get('axes') or {}
        right_axes = ((finger_metrics.get('right') or {}).get('local_contact') or {}).get('axes') or {}
        pinch_candidates = []
        caging_axis = None
        caging_candidates = []
        caging_contact_distance = float(
            attach_spec.get(
                'caging_contact_distance',
                max(contact_distance * 4.0, 0.01),
            )
        )
        for axis_name in ('x', 'y', 'z'):
            left_axis = left_axes.get(axis_name)
            right_axis = right_axes.get(axis_name)
            if left_axis is None or right_axis is None:
                continue
            if float(left_axis.get('signed_coordinate', 0.0)) * float(right_axis.get('signed_coordinate', 0.0)) >= 0.0:
                continue
            combined_surface_gap = float(left_axis['surface_gap']) + float(right_axis['surface_gap'])
            if bool(left_axis.get('contact')) and bool(right_axis.get('contact')):
                pinch_candidates.append(
                    {
                        'axis': axis_name,
                        'combined_surface_gap': combined_surface_gap,
                    }
                )
            if (
                bool(left_axis.get('within_patch'))
                and bool(right_axis.get('within_patch'))
                and float(max(left_axis['surface_gap'], right_axis['surface_gap'])) <= caging_contact_distance
            ):
                caging_candidates.append(
                    {
                        'axis': axis_name,
                        'combined_surface_gap': combined_surface_gap,
                    }
                )
        if pinch_candidates:
            pinch_axis = min(pinch_candidates, key=lambda item: item['combined_surface_gap'])['axis']
        if caging_candidates:
            caging_axis = min(caging_candidates, key=lambda item: item['combined_surface_gap'])['axis']

        dual_finger_contact = bool(
            bool((finger_metrics.get('left') or {}).get('has_contact'))
            and bool((finger_metrics.get('right') or {}).get('has_contact'))
        )
        if require_dual_contact:
            contact_ready = bool(
                pinch_axis is not None
                or (
                    bool((finger_metrics.get('left') or {}).get('force_contact'))
                    and bool((finger_metrics.get('right') or {}).get('force_contact'))
                )
                or dual_finger_contact
                or caging_axis is not None
            )
        else:
            contact_ready = any(finger_contacts)

        return {
            'object': object_name,
            'robot': robot_name,
            'contact_available': contact_available,
            'require_dual_finger_contact': require_dual_contact,
            'contact_force_threshold': contact_force_threshold,
            'contact_distance': contact_distance,
            'dual_finger_contact': dual_finger_contact,
            'left_finger': finger_metrics['left'],
            'right_finger': finger_metrics['right'],
            'pinch_axis': pinch_axis,
            'caging_axis': caging_axis,
            'caging_contact_distance': caging_contact_distance,
            'contact_ready': contact_ready,
        }

    def _attachment_joint_path(self, object_name: str) -> str:
        object_rigid_body = self._resolve_object(object_name)
        return f"{object_rigid_body.unwrap().prim_path}/assembly_attachment_joint"

    def _remove_attachment_joint(self, object_name: str):
        from omni.isaac.core.utils.prims import delete_prim

        joint_path = self._attachment_joints.pop(object_name, None)
        if joint_path is None:
            return
        try:
            delete_prim(joint_path)
        except Exception:
            return

    def _create_attachment_joint(self, object_name: str, robot_name: str) -> str | None:
        object_rigid_body = self._resolve_object(object_name)
        hand_rigid_body = self._get_robot_hand_rigid_body(robot_name)
        if hand_rigid_body is None:
            return None

        joint_path = self._attachment_joint_path(object_name)
        self._remove_attachment_joint(object_name)
        object_position, object_orientation = object_rigid_body.get_pose()
        hand_position, hand_orientation = hand_rigid_body.get_pose()
        hand_relative_position, hand_relative_orientation = relative_pose(
            base_position=np.asarray(hand_position, dtype=float),
            base_orientation=np.asarray(hand_orientation, dtype=float),
            world_position=np.asarray(object_position, dtype=float),
            world_orientation=np.asarray(object_orientation, dtype=float),
        )
        object_rigid_body.set_linear_velocity(np.zeros(3))
        try:
            object_rigid_body.unwrap().set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        try:
            # Pin the fixed-joint anchor to the object's current world pose so both bodies agree on
            # the same frame when the joint is created. Leaving these frames implicit allows PhysX
            # to infer slightly disjoint anchors, which shows up as snap / hover artifacts.
            create_joint(
                prim_path=joint_path,
                joint_type='FixedJoint',
                body0=object_rigid_body.unwrap().prim_path,
                body1=hand_rigid_body.unwrap().prim_path,
                enabled=True,
                joint_frame_in_parent_frame_pos=np.zeros(3, dtype=float),
                joint_frame_in_parent_frame_quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
                joint_frame_in_child_frame_pos=np.asarray(hand_relative_position, dtype=float),
                joint_frame_in_child_frame_quat=np.asarray(hand_relative_orientation, dtype=float),
            )
        except Exception:
            return None
        self._attachment_joints[object_name] = joint_path
        return joint_path

    def _clear_attachment_state(self, object_name: str, *, enable_collision: bool = True):
        attachment_state = self._attachments.pop(object_name, None)
        self._remove_attachment_joint(object_name)
        if enable_collision:
            self._set_object_collision(object_name, True)
        return attachment_state

    def _get_robot_eef_pose(self, robot_name: str):
        return self.robots[robot_name].articulation.end_effector.get_pose()

    def _get_robot_task_pose(self, robot_name: str):
        robot = self.robots[robot_name]
        controller = robot.controllers.get(self._ARM_IK_CONTROLLER_NAME)
        if controller is not None:
            try:
                controller_obs = controller.get_obs()
            except Exception:
                controller_obs = None
            if controller_obs:
                position = controller_obs.get('eef_position')
                orientation = controller_obs.get('eef_orientation')
                if position is not None and orientation is not None:
                    return np.asarray(position, dtype=float), np.asarray(orientation, dtype=float)
        return self._get_robot_eef_pose(robot_name)

    def _get_robot_attach_reference_position(self, robot_name: str) -> np.ndarray:
        finger_rigid_bodies = self._get_robot_finger_rigid_bodies(robot_name)
        contact_points = []
        for rigid_body in finger_rigid_bodies.values():
            fingertip_position = self._finger_contact_point(rigid_body)
            if fingertip_position is not None:
                contact_points.append(np.asarray(fingertip_position, dtype=float))
                continue
            if rigid_body is not None:
                finger_position, _ = rigid_body.get_pose()
                contact_points.append(np.asarray(finger_position, dtype=float))
        if contact_points:
            return np.mean(np.asarray(contact_points, dtype=float), axis=0)
        robot_position, _ = self._get_robot_task_pose(robot_name)
        return np.asarray(robot_position, dtype=float)

    def _get_robot_gripper_opening(self, robot_name: str) -> float | None:
        robot = self.robots[robot_name]
        controller = robot.controllers.get(self._GRIPPER_CONTROLLER_NAME)
        if controller is None:
            return None
        try:
            controller_obs = controller.get_obs()
        except Exception:
            return None
        gripper_positions = controller_obs.get('gripper_pos')
        if gripper_positions is None:
            return None
        gripper_positions = np.asarray(gripper_positions, dtype=float)
        if gripper_positions.size == 0:
            return None
        return float(np.max(np.abs(gripper_positions)))

    def _get_object_scale(self, object_name: str) -> np.ndarray:
        metadata = self._object_metadata_map.get(object_name, {})
        return np.asarray(metadata.get('scale', [1.0, 1.0, 1.0]), dtype=float)

    def _is_slender_attach_object(self, object_name: str, attach_spec: dict | None = None) -> bool:
        attach_spec = attach_spec or {}
        if attach_spec.get('slender_object') is not None:
            return bool(attach_spec['slender_object'])
        scale = self._get_object_scale(object_name)
        if scale.size == 0:
            return False
        lateral_dims = scale[:2] if scale.size >= 2 else scale
        lateral_threshold = float(attach_spec.get('slender_lateral_threshold', 0.04))
        return float(np.max(lateral_dims)) <= lateral_threshold

    def _sampled_object_position(self, object_name: str):
        metadata = self._object_metadata_map.get(object_name, {})
        sampled_position = metadata.get('sampled_position')
        if sampled_position is None:
            return None
        return np.asarray(sampled_position, dtype=float)

    def _attach_proximity_metrics(self, object_name: str, robot_name: str, *, target_position=None) -> dict:
        object_position, _ = self._resolve_object(object_name).get_pose()
        robot_position = self._get_robot_attach_reference_position(robot_name)
        object_position = np.asarray(object_position, dtype=float)
        delta = robot_position - object_position
        scale = self._get_object_scale(object_name)
        half_extents = np.maximum(scale * 0.5, np.array([0.01, 0.01, 0.01], dtype=float))
        if target_position is not None:
            desired_delta = np.asarray(target_position, dtype=float) - object_position
            delta_error = delta - desired_delta
            center_distance = float(np.linalg.norm(delta_error))
            lateral_distance = float(np.linalg.norm(delta_error[:2]))
            vertical_offset = float(abs(delta_error[2]))
            distance_limit = float(self._ATTACH_DISTANCE_MARGIN + 0.01)
            lateral_limit = float(self._ATTACH_LATERAL_MARGIN + 0.01)
            vertical_limit = float(self._ATTACH_VERTICAL_MARGIN + 0.01)
        else:
            center_distance = float(np.linalg.norm(delta))
            lateral_distance = float(np.linalg.norm(delta[:2]))
            vertical_offset = float(abs(delta[2]))
            distance_limit = float(np.linalg.norm(half_extents) + self._ATTACH_DISTANCE_MARGIN)
            lateral_limit = float(np.linalg.norm(half_extents[:2]) + self._ATTACH_LATERAL_MARGIN)
            vertical_limit = float(half_extents[2] + self._ATTACH_VERTICAL_MARGIN)
        return {
            'center_distance': center_distance,
            'lateral_distance': lateral_distance,
            'vertical_offset': vertical_offset,
            'distance_limit': distance_limit,
            'lateral_limit': lateral_limit,
            'vertical_limit': vertical_limit,
            'within_proximity': (
                center_distance <= distance_limit
                and lateral_distance <= lateral_limit
                and vertical_offset <= vertical_limit
            ),
        }

    def _set_object_pose(self, object_name: str, position, orientation):
        rigid_body = self._resolve_object(object_name)
        rigid_body.set_linear_velocity(np.zeros(3))
        rigid_body.set_pose(np.asarray(position, dtype=float), np.asarray(orientation, dtype=float))

    def _current_gripper_command(self, phase_spec: dict, robot_name: str) -> str | None:
        gripper_command = phase_spec.get('gripper_commands', {}).get(robot_name)
        if gripper_command is None:
            return None
        return str(gripper_command).lower()

    def _default_robot_orientation_tolerance(self, phase_spec: dict, robot_name: str, robot_target_spec) -> float | None:
        try:
            target_name, _, target_orientation, _ = self._resolve_target_pose_spec(robot_target_spec)
        except Exception:
            return None
        if target_orientation is None:
            return None
        descriptor = f"{phase_spec.get('name', '')} {target_name or ''}".lower()
        if 'wait' in descriptor or 'retreat' in descriptor or 'hover' in descriptor:
            return None
        if 'insert' in descriptor or 'preinsert' in descriptor:
            return 0.30
        if 'pick' in descriptor or 'grasp' in descriptor:
            return 0.40
        return None

    def _resolve_phase_robot_target(
        self,
        phase_spec: dict,
        robot_name: str,
        *,
        default_tolerance: float,
        default_orientation_tolerance: float | None = None,
    ):
        robot_target_spec = phase_spec.get('robot_targets', {}).get(robot_name)
        if robot_target_spec is None:
            return None
        if default_orientation_tolerance is None:
            default_orientation_tolerance = self._default_robot_orientation_tolerance(
                phase_spec,
                robot_name,
                robot_target_spec,
            )
        return self._resolve_phase_target_pose(
            robot_name=robot_name,
            robot_target_spec=robot_target_spec,
            default_position_tolerance=default_tolerance,
            default_orientation_tolerance=default_orientation_tolerance,
        )

    def _resolve_attach_target_info(self, phase_spec: dict, attach_spec: dict):
        object_name = attach_spec['object']
        robot_name = attach_spec['robot']
        target_like = attach_spec.get('target') or phase_spec.get('robot_targets', {}).get(robot_name)
        if target_like is not None:
            return self._resolve_phase_target_pose(
                robot_name=robot_name,
                robot_target_spec=target_like,
                default_position_tolerance=float(
                    attach_spec.get('position_tolerance', attach_spec.get('tolerance', self._ATTACH_POSITION_TOLERANCE))
                ),
                default_orientation_tolerance=attach_spec.get('orientation_tolerance')
                if attach_spec.get('orientation_tolerance') is not None
                else self._default_robot_orientation_tolerance(phase_spec, robot_name, target_like),
            )
        elif phase_spec.get('robot_targets', {}).get(robot_name) is not None:
            return self._resolve_phase_robot_target(
                phase_spec=phase_spec,
                robot_name=robot_name,
                default_tolerance=float(
                    attach_spec.get('position_tolerance', attach_spec.get('tolerance', self._ATTACH_POSITION_TOLERANCE))
                ),
                default_orientation_tolerance=attach_spec.get('orientation_tolerance'),
            )
        return None

    def _attach_object(self, object_name: str, robot_name: str, *, phase_spec: dict | None = None, attach_spec: dict | None = None):
        rigid_body = self._resolve_object(object_name)
        object_pose = rigid_body.get_pose()
        robot_pose = self._get_robot_task_pose(robot_name)
        object_position = np.asarray(object_pose[0], dtype=float)
        object_orientation = np.asarray(object_pose[1], dtype=float)
        desired_object_position = object_position
        desired_object_orientation = object_orientation

        relative_position, relative_orientation = relative_pose(
            base_position=robot_pose[0],
            base_orientation=robot_pose[1],
            world_position=desired_object_position,
            world_orientation=desired_object_orientation,
        )
        attach_spec = attach_spec or {}
        contact_metrics = self._gripper_contact_metrics(object_name, robot_name, attach_spec=attach_spec)
        attach_mode = self._attachment_mode(attach_spec)
        uses_joint_attachment = attach_mode in self._JOINT_ATTACHMENT_MODES
        if attach_mode in {'physical_hold', 'physical', 'contact_hold'}:
            self._attachments[object_name] = {
                'robot_name': robot_name,
                'position': relative_position.tolist(),
                'orientation': relative_orientation.tolist(),
                'mode': 'physical_hold',
                'joint_path': None,
                'contact_metrics': copy.deepcopy(contact_metrics),
                'collision_disabled': False,
                'attach_spec': copy.deepcopy(attach_spec),
                'phase': None if phase_spec is None else phase_spec.get('name'),
                'attach_step': int(self.step_counter),
            }
            self._locked_targets.pop(object_name, None)
            return
        collision_disabled = bool(
            attach_spec.get('disable_collision_on_attach', uses_joint_attachment)
        )
        if collision_disabled:
            self._set_object_collision(object_name, False)
        joint_path = None
        if uses_joint_attachment:
            joint_path = self._create_attachment_joint(object_name, robot_name)
            if joint_path is None and collision_disabled:
                self._set_object_collision(object_name, True)
        self._attachments[object_name] = {
            'robot_name': robot_name,
            'position': relative_position.tolist(),
            'orientation': relative_orientation.tolist(),
            'mode': 'fixed_joint' if joint_path is not None else 'symbolic',
            'joint_path': joint_path,
            'contact_metrics': copy.deepcopy(contact_metrics),
            'collision_disabled': collision_disabled,
            'attach_spec': copy.deepcopy(attach_spec),
            'phase': None if phase_spec is None else phase_spec.get('name'),
            'attach_step': int(self.step_counter),
        }
        self._locked_targets.pop(object_name, None)
        if collision_disabled and joint_path is None:
            self._set_object_collision(object_name, False)

    def _normalized_attach_spec(self, attach_spec):
        if isinstance(attach_spec, dict):
            return dict(attach_spec)
        if isinstance(attach_spec, str):
            return {'object': attach_spec}
        return None

    @staticmethod
    def _attachment_mode(attach_spec: dict | None) -> str:
        if not isinstance(attach_spec, dict):
            return 'fixed_joint'
        return str(attach_spec.get('attachment_mode', 'fixed_joint')).lower()

    def _current_relative_pose(self, object_name: str, robot_name: str):
        object_pose = self._resolve_object(object_name).get_pose()
        robot_pose = self._get_robot_task_pose(robot_name)
        return relative_pose(
            base_position=robot_pose[0],
            base_orientation=robot_pose[1],
            world_position=object_pose[0],
            world_orientation=object_pose[1],
        )

    def _physical_hold_valid(self, object_name: str, attachment_state: dict) -> bool:
        robot_name = attachment_state.get('robot_name')
        if robot_name is None:
            return False

        attach_spec = attachment_state.get('attach_spec') or {}
        gripper_opening = self._get_robot_gripper_opening(robot_name)
        gripper_closed_threshold = float(
            attach_spec.get('gripper_closed_threshold', self._GRIPPER_CLOSED_THRESHOLD)
        )
        gripper_closed_margin = float(attach_spec.get('gripper_closed_margin', 0.004))
        if gripper_opening is not None and gripper_opening > gripper_closed_threshold + gripper_closed_margin:
            return False

        relative_position, relative_orientation = self._current_relative_pose(object_name, robot_name)
        anchor_position = np.asarray(attachment_state.get('position', relative_position), dtype=float)
        anchor_orientation = np.asarray(attachment_state.get('orientation', relative_orientation), dtype=float)
        position_slip = float(np.linalg.norm(np.asarray(relative_position, dtype=float) - anchor_position))
        _, orientation_slip = pose_error(
            current_position=np.zeros(3, dtype=float),
            current_orientation=np.asarray(relative_orientation, dtype=float),
            target_position=np.zeros(3, dtype=float),
            target_orientation=anchor_orientation,
        )
        max_position_slip = float(
            attach_spec.get('hold_position_slip_tolerance', self._PHYSICAL_HOLD_POSITION_SLIP)
        )
        max_orientation_slip = float(
            attach_spec.get('hold_orientation_slip_tolerance', self._PHYSICAL_HOLD_ORIENTATION_SLIP)
        )

        contact_metrics = self._gripper_contact_metrics(object_name, robot_name, attach_spec=attach_spec)
        strict_contact = bool(contact_metrics.get('pinch_axis')) or bool(contact_metrics.get('dual_finger_contact'))
        return bool(
            position_slip <= max_position_slip
            and (orientation_slip is None or orientation_slip <= max_orientation_slip)
            and (bool(contact_metrics.get('contact_ready')) or strict_contact)
        )

    def _detach_object(self, object_name: str):
        self._clear_attachment_state(object_name, enable_collision=True)

    def _lock_object(self, object_name: str, target_name: str):
        self._clear_attachment_state(object_name, enable_collision=False)
        self._locked_targets[object_name] = target_name
        target_pose = self.target_poses[target_name]
        self._set_object_pose(object_name, target_pose['position'], target_pose['orientation'])
        self._set_object_collision(object_name, True)

    def _unlock_object(self, object_name: str):
        self._locked_targets.pop(object_name, None)

    def _maybe_write_attach_debug(self, payload: dict):
        debug_path = os.environ.get('DUAL_FRANKA_ATTACH_DEBUG_PATH')
        if not debug_path:
            return
        try:
            with open(debug_path, 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(payload) + '\n')
        except Exception:
            return

    def _attach_ready(self, phase_spec: dict, attach_spec: dict) -> bool:
        object_name = attach_spec['object']
        robot_name = attach_spec['robot']
        attachment_state = self._attachments.get(object_name)
        if attachment_state is not None and attachment_state.get('robot_name') == robot_name:
            return True

        if self._current_gripper_command(phase_spec, robot_name) != 'close':
            return False

        target_info = self._resolve_attach_target_info(phase_spec=phase_spec, attach_spec=attach_spec)
        if target_info is None:
            return False
        debug_payload = {
            'step_counter': int(self.step_counter),
            'phase_step_counter': int(self.phase_step_counter),
            'phase': phase_spec.get('name'),
            'object': object_name,
            'robot': robot_name,
        }
        gripper_closed_threshold = float(
            attach_spec.get('gripper_closed_threshold', self._GRIPPER_CLOSED_THRESHOLD)
        )
        gripper_closed_margin = float(attach_spec.get('gripper_closed_margin', 0.002))
        gripper_opening = self._get_robot_gripper_opening(robot_name)
        debug_payload.update(
            {
                'target_reached': bool(target_info['target_reached']),
                'position_error': float(target_info['position_error']),
                'orientation_error': None
                if target_info['orientation_error'] is None
                else float(target_info['orientation_error']),
                'gripper_opening': None if gripper_opening is None else float(gripper_opening),
                'gripper_closed_threshold': gripper_closed_threshold,
                'gripper_closed_margin': gripper_closed_margin,
            }
        )
        if gripper_opening is not None and gripper_opening > gripper_closed_threshold + gripper_closed_margin:
            debug_payload['blocked_by'] = 'gripper_open'
            self._maybe_write_attach_debug(debug_payload)
            return False
        min_gripper_opening = attach_spec.get('min_gripper_opening')
        if min_gripper_opening is not None and gripper_opening is not None:
            if float(gripper_opening) < float(min_gripper_opening):
                debug_payload['blocked_by'] = 'gripper_too_closed'
                self._maybe_write_attach_debug(debug_payload)
                return False
        proximity_metrics = self._attach_proximity_metrics(object_name=object_name, robot_name=robot_name)
        debug_payload['proximity'] = copy.deepcopy(proximity_metrics)
        support_height_tolerance = attach_spec.get('support_height_tolerance', self._ATTACH_SUPPORT_HEIGHT_MARGIN)
        sampled_object_position = self._sampled_object_position(object_name)
        if support_height_tolerance is not None:
            support_height_tolerance = max(float(support_height_tolerance), 0.0)
            if sampled_object_position is not None:
                object_position, _ = self._resolve_object(object_name).get_pose()
                debug_payload['support_height_delta'] = float(object_position[2] - sampled_object_position[2])
                if float(object_position[2]) > float(sampled_object_position[2]) + support_height_tolerance:
                    debug_payload['blocked_by'] = 'support_height'
                    self._maybe_write_attach_debug(debug_payload)
                    return False
        slender_attach = self._is_slender_attach_object(object_name, attach_spec=attach_spec)
        allow_top_contact_fallback = bool(
            attach_spec.get('allow_top_contact_fallback', not slender_attach)
        )
        require_target_reached = bool(
            attach_spec.get('require_target_reached_for_attach', slender_attach)
        )
        top_contact_ready = False
        max_top_clearance = attach_spec.get('top_clearance', self._ATTACH_TOP_CLEARANCE)
        if max_top_clearance is not None:
            max_top_clearance = max(float(max_top_clearance), 0.0)
            object_position, _ = self._resolve_object(object_name).get_pose()
            robot_position = self._get_robot_attach_reference_position(robot_name)
            object_position = np.asarray(object_position, dtype=float)
            robot_position = np.asarray(robot_position, dtype=float)
            half_height = float(max(self._get_object_scale(object_name)[2] * 0.5, 0.01))
            top_clearance = float(robot_position[2] - (object_position[2] + half_height))
            debug_payload['top_clearance'] = top_clearance
            if top_clearance > max_top_clearance:
                debug_payload['blocked_by'] = 'top_clearance'
                self._maybe_write_attach_debug(debug_payload)
                return False
            top_contact_ready = top_clearance <= min(max_top_clearance, 0.003)
        contact_metrics = self._gripper_contact_metrics(object_name, robot_name, attach_spec=attach_spec)
        debug_payload['contact_metrics'] = copy.deepcopy(contact_metrics)
        contact_ready = bool(contact_metrics['contact_ready'])
        attach_mode = str(attach_spec.get('attachment_mode', 'fixed_joint')).lower()
        uses_physical_joint = attach_mode in self._JOINT_ATTACHMENT_MODES
        left_finger_metrics = contact_metrics.get('left_finger') or {}
        right_finger_metrics = contact_metrics.get('right_finger') or {}

        def _surface_gap(metric: dict) -> float | None:
            local_contact = metric.get('local_contact') or {}
            best_surface_gap = local_contact.get('best_surface_gap')
            if best_surface_gap is not None:
                return float(best_surface_gap)
            surface_gap = metric.get('surface_gap')
            return None if surface_gap is None else float(surface_gap)

        left_surface_gap = _surface_gap(left_finger_metrics)
        right_surface_gap = _surface_gap(right_finger_metrics)
        strict_surface_gap = float(
            attach_spec.get(
                'physical_attach_surface_gap',
                0.0025 if slender_attach else 0.004,
            )
        )
        strict_dual_finger_contact = bool(
            bool(left_finger_metrics.get('geometric_contact'))
            and bool(right_finger_metrics.get('geometric_contact'))
            and left_surface_gap is not None
            and right_surface_gap is not None
            and max(left_surface_gap, right_surface_gap) <= strict_surface_gap
        )
        require_physical_contact = bool(attach_spec.get('require_physical_contact', False))
        debug_payload['strict_dual_finger_contact'] = strict_dual_finger_contact
        debug_payload['strict_surface_gap_limit'] = strict_surface_gap
        debug_payload['left_surface_gap'] = left_surface_gap
        debug_payload['right_surface_gap'] = right_surface_gap
        if allow_top_contact_fallback:
            contact_ready = bool(contact_ready or top_contact_ready)
        enclosure_ready = False
        allow_enclosure_fallback = bool(attach_spec.get('allow_enclosure_fallback', not slender_attach))
        if (
            allow_enclosure_fallback
            and not contact_ready
            and not slender_attach
            and gripper_opening is not None
            and target_info['target_reached']
            and proximity_metrics['within_proximity']
        ):
            enclosure_gripper_threshold = float(
                attach_spec.get(
                    'enclosure_gripper_threshold',
                    min(gripper_closed_threshold * 0.35, 0.012),
                )
            )
            if float(gripper_opening) <= enclosure_gripper_threshold:
                object_position, _ = self._resolve_object(object_name).get_pose()
                object_position = np.asarray(object_position, dtype=float)
                lateral_sample_delta = 0.0
                if sampled_object_position is not None:
                    lateral_sample_delta = float(
                        np.linalg.norm(object_position[:2] - np.asarray(sampled_object_position[:2], dtype=float))
                    )
                enclosure_lateral_tolerance = float(
                    attach_spec.get(
                        'enclosure_lateral_tolerance',
                        max(float(np.min(self._get_object_scale(object_name)[:2])) * 0.5, 0.015),
                    )
                )
                enclosure_ready = lateral_sample_delta <= enclosure_lateral_tolerance
        contact_ready = bool(contact_ready or enclosure_ready)
        if uses_physical_joint and (slender_attach or require_physical_contact):
            contact_ready = strict_dual_finger_contact
        debug_payload['enclosure_ready'] = enclosure_ready
        debug_payload['contact_ready'] = contact_ready
        if bool(attach_spec.get('require_contact', True)) and not contact_ready:
            debug_payload['blocked_by'] = 'contact'
            self._maybe_write_attach_debug(debug_payload)
            return False
        if require_target_reached and not target_info['target_reached']:
            debug_payload['blocked_by'] = 'target'
            self._maybe_write_attach_debug(debug_payload)
            return False
        attach_ready = bool(proximity_metrics['within_proximity'] and (target_info['target_reached'] or contact_ready))
        debug_payload['attach_ready'] = attach_ready
        if not attach_ready:
            debug_payload['blocked_by'] = 'proximity_or_target'
        self._maybe_write_attach_debug(debug_payload)
        return attach_ready

    def _detach_ready(self, phase_spec: dict, object_name: str) -> bool:
        attachment_state = self._attachments.get(object_name)
        if attachment_state is None:
            return True
        robot_name = attachment_state.get('robot_name')
        if robot_name is None:
            return True
        if self._current_gripper_command(phase_spec, robot_name) != 'open':
            return False
        target_info = self._resolve_phase_robot_target(
            phase_spec=phase_spec,
            robot_name=robot_name,
            default_tolerance=self._DETACH_POSITION_TOLERANCE,
            default_orientation_tolerance=None,
        )
        if target_info is None:
            return True
        return bool(target_info['target_reached'])

    def _lock_ready(self, phase_spec: dict, lock_spec: dict) -> bool:
        object_name = lock_spec['object']
        target_name = lock_spec.get('target') or lock_spec.get('target_name')
        if target_name is None:
            return False
        if self._locked_targets.get(object_name) == target_name:
            return True

        attachment_state = self._attachments.get(object_name)
        if attachment_state is not None:
            robot_name = attachment_state.get('robot_name')
            if robot_name is not None and self._current_gripper_command(phase_spec, robot_name) != 'open':
                return False

        object_position, object_orientation = self._resolve_object(object_name).get_pose()
        _, target_position, target_orientation, target_spec = self._resolve_target_pose_spec(target_name)
        position_tolerance, orientation_tolerance = self._target_tolerances(
            target_spec=target_spec,
            default_position_tolerance=float(
                lock_spec.get('position_tolerance', lock_spec.get('tolerance', self._LOCK_POSITION_TOLERANCE))
            ),
            default_orientation_tolerance=lock_spec.get('orientation_tolerance'),
        )
        if pose_within_tolerance(
            current_position=object_position,
            current_orientation=object_orientation,
            target_position=target_position,
            target_orientation=target_orientation,
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        ):
            return True

        phase_name = str(phase_spec.get('name', '')).lower()
        snap_on_open = lock_spec.get('snap_on_open')
        if snap_on_open is None:
            snap_on_open = 'release' in phase_name or 'finalize' in phase_name
        if not snap_on_open:
            return False

        if attachment_state is None:
            return False
        robot_name = attachment_state.get('robot_name')
        if robot_name is None or self._current_gripper_command(phase_spec, robot_name) != 'open':
            return False

        min_release_steps = int(lock_spec.get('release_snap_steps', self._RELEASE_LOCK_MIN_STEPS))
        return self.phase_step_counter >= min_release_steps

    def _handoff_ready(self, phase_spec: dict, handoff_spec) -> bool:
        handoff_spec = self._resolve_handoff_spec(handoff_spec)
        if not handoff_spec:
            return True
        object_name = handoff_spec.get('object')
        target_robot = handoff_spec.get('to_robot') or handoff_spec.get('target_robot') or handoff_spec.get('destination_robot') or handoff_spec.get('to')
        source_robot = handoff_spec.get('from_robot') or handoff_spec.get('source_robot') or handoff_spec.get('from')
        if object_name is None or target_robot is None:
            return True

        attachment_state = self._attachments.get(object_name)
        if attachment_state is not None and attachment_state.get('robot_name') == target_robot:
            return True

        for robot_name in [source_robot, target_robot]:
            if robot_name is None:
                continue
            target_info = self._resolve_phase_robot_target(
                phase_spec=phase_spec,
                robot_name=robot_name,
                default_tolerance=self._ATTACH_POSITION_TOLERANCE,
            )
            if target_info is not None and not target_info['target_reached']:
                return False
        return True

    def _process_phase_interactions(self, phase_spec: dict):
        if not phase_spec:
            return

        lock_targets = {
            lock_spec['object']: lock_spec.get('target') or lock_spec.get('target_name')
            for lock_spec in self._as_list(phase_spec.get('lock'))
            if isinstance(lock_spec, dict) and lock_spec.get('object') is not None
        }

        for attach_spec in self._as_list(phase_spec.get('attach')):
            attach_spec = self._normalized_attach_spec(attach_spec)
            if not attach_spec or 'object' not in attach_spec or 'robot' not in attach_spec:
                continue
            if self._attach_ready(phase_spec, attach_spec):
                self._attach_object(
                    object_name=attach_spec['object'],
                    robot_name=attach_spec['robot'],
                    phase_spec=phase_spec,
                    attach_spec=attach_spec,
                )

        for handoff_spec in self._as_list(phase_spec.get('handoff')) + self._as_list(phase_spec.get('transfer')):
            if self._handoff_ready(phase_spec, handoff_spec):
                self._handoff_object(handoff_spec)

        for lock_spec in self._as_list(phase_spec.get('lock')):
            if not isinstance(lock_spec, dict):
                continue
            if self._lock_ready(phase_spec, lock_spec):
                target_name = lock_spec.get('target') or lock_spec.get('target_name')
                if target_name is not None:
                    self._lock_object(object_name=lock_spec['object'], target_name=target_name)

        for object_entry in self._as_list(phase_spec.get('detach')):
            object_name = self._extract_object_name(object_entry)
            if object_name is None or object_name in lock_targets:
                continue
            if self._detach_ready(phase_spec, object_name):
                self._detach_object(object_name)

    def _phase_interactions_complete(self, phase_spec: dict) -> bool:
        if not phase_spec:
            return True

        for attach_spec in self._as_list(phase_spec.get('attach')):
            attach_spec = self._normalized_attach_spec(attach_spec)
            if not attach_spec or 'object' not in attach_spec or 'robot' not in attach_spec:
                continue
            attachment_state = self._attachments.get(attach_spec['object'])
            if attachment_state is None or attachment_state.get('robot_name') != attach_spec['robot']:
                return False
            if attachment_state.get('mode') == 'physical_hold' and not self._physical_hold_valid(
                attach_spec['object'],
                attachment_state,
            ):
                return False

        for handoff_spec in self._as_list(phase_spec.get('handoff')) + self._as_list(phase_spec.get('transfer')):
            handoff_spec = self._resolve_handoff_spec(handoff_spec)
            if not handoff_spec:
                continue
            object_name = handoff_spec.get('object')
            target_robot = handoff_spec.get('to_robot') or handoff_spec.get('target_robot') or handoff_spec.get('destination_robot') or handoff_spec.get('to')
            if object_name is None or target_robot is None:
                continue
            attachment_state = self._attachments.get(object_name)
            if attachment_state is None or attachment_state.get('robot_name') != target_robot:
                return False

        lock_objects = set()
        for lock_spec in self._as_list(phase_spec.get('lock')):
            if not isinstance(lock_spec, dict):
                continue
            object_name = lock_spec.get('object')
            target_name = lock_spec.get('target') or lock_spec.get('target_name')
            if object_name is None or target_name is None:
                continue
            lock_objects.add(object_name)
            if self._locked_targets.get(object_name) != target_name:
                return False

        for object_entry in self._as_list(phase_spec.get('detach')):
            object_name = self._extract_object_name(object_entry)
            if object_name is None or object_name in lock_objects:
                continue
            if object_name in self._attachments:
                return False

        return True

    def _initialize_phase(self):
        if self._phase_initialized:
            return

        phase_spec = self.get_current_phase_spec()
        self._apply_phase_actions(phase_spec)

        self._phase_initialized = True

    def _sync_object_states(self):
        phase_spec = self.get_current_phase_spec()
        pending_detach_objects = {
            object_name
            for object_entry in self._as_list((phase_spec or {}).get('detach'))
            for object_name in [self._extract_object_name(object_entry)]
            if object_name is not None
        }
        for object_name, attach_state in list(self._attachments.items()):
            if attach_state.get('mode') != 'physical_hold':
                continue
            if self._physical_hold_valid(object_name, attach_state):
                continue
            if object_name in pending_detach_objects:
                continue
            self._clear_attachment_state(object_name, enable_collision=True)

        for object_name, target_name in self._locked_targets.items():
            target_pose = self.target_poses[target_name]
            self._set_object_pose(object_name, target_pose['position'], target_pose['orientation'])

        for object_name, attach_state in self._attachments.items():
            if attach_state.get('mode') in {'fixed_joint', 'physical_hold'}:
                continue
            robot_pose = self._get_robot_task_pose(attach_state['robot_name'])
            position, orientation = compose_pose(
                base_position=robot_pose[0],
                base_orientation=robot_pose[1],
                local_position=attach_state['position'],
                local_orientation=attach_state['orientation'],
            )
            self._set_object_pose(object_name, position, orientation)

    def _robot_targets_reached(self, phase_spec: dict, tolerance: float, orientation_tolerance: float | None) -> bool:
        effective_tolerance = self._effective_robot_target_tolerance(tolerance)
        for robot_name in phase_spec.get('robot_targets', {}):
            if phase_spec.get('robot_targets', {}).get(robot_name) is None:
                continue
            target_info = self._resolve_phase_robot_target(
                phase_spec=phase_spec,
                robot_name=robot_name,
                default_tolerance=effective_tolerance,
                default_orientation_tolerance=orientation_tolerance,
            )
            if not target_info['target_reached']:
                return False
        return True

    def _object_targets_reached(
        self,
        object_targets: list[dict],
        tolerance: float,
        orientation_tolerance: float | None,
    ) -> bool:
        for object_target in object_targets:
            object_name = object_target['object']
            target_name = object_target.get('target') or object_target.get('target_name')
            target_like = object_target.get('target_pose', target_name)
            object_position, object_orientation = self._resolve_object(object_name).get_pose()
            _, target_position, target_orientation, target_spec = self._resolve_target_pose_spec(target_like)
            if target_position is None:
                return False
            position_tolerance, orientation_tolerance = self._target_tolerances(
                target_spec=target_spec,
                default_position_tolerance=float(object_target.get('tolerance', object_target.get('position_tolerance', tolerance))),
                default_orientation_tolerance=object_target.get('orientation_tolerance', orientation_tolerance),
            )
            if not pose_within_tolerance(
                current_position=object_position,
                current_orientation=object_orientation,
                target_position=target_position,
                target_orientation=target_orientation,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
            ):
                return False
        return True

    def _advance_condition_met(self, phase_spec: dict) -> bool:
        if not self._phase_interactions_complete(phase_spec):
            return False

        advance = phase_spec.get('advance', {})
        return self._evaluate_advance_condition(phase_spec=phase_spec, advance=advance)

    def _evaluate_advance_condition(self, *, phase_spec: dict, advance: dict) -> bool:
        if not advance:
            return True
        min_steps = int(advance.get('min_steps', 0))
        if self.phase_step_counter < min_steps:
            return False

        advance_type = advance.get('type', 'timer')
        if advance_type == 'all_of':
            return all(
                self._evaluate_advance_condition(phase_spec=phase_spec, advance=condition)
                for condition in advance.get('conditions', advance.get('all', []))
            )
        if advance_type == 'any_of':
            conditions = advance.get('conditions', advance.get('any', []))
            if not conditions:
                return False
            return any(
                self._evaluate_advance_condition(phase_spec=phase_spec, advance=condition)
                for condition in conditions
            )
        if advance_type == 'timer':
            return True
        orientation_tolerance = advance.get('orientation_tolerance')
        if orientation_tolerance is not None:
            orientation_tolerance = float(orientation_tolerance)
        if advance_type in {'robot_targets_reached', 'robot_pose_reached'}:
            return self._robot_targets_reached(
                phase_spec=phase_spec,
                tolerance=float(advance.get('tolerance', 0.04)),
                orientation_tolerance=orientation_tolerance,
            )
        if advance_type in {'object_targets_reached', 'object_pose_reached'}:
            return self._object_targets_reached(
                object_targets=advance.get('objects', []),
                tolerance=float(advance.get('tolerance', 0.04)),
                orientation_tolerance=orientation_tolerance,
            )
        if advance_type == 'success_criteria_met':
            return self._check_success()
        raise ValueError(f'Unsupported advance condition: {advance_type}')

    def _check_success(self) -> bool:
        for success_criterion in self.cfg.success_criteria:
            object_name = success_criterion['object']
            target_name = success_criterion.get('target') or success_criterion.get('target_name')
            target_like = success_criterion.get('target_pose', target_name)
            position_tolerance = float(success_criterion.get('position_tolerance', success_criterion.get('tolerance', 0.03)))
            orientation_tolerance = success_criterion.get('orientation_tolerance')
            object_position, object_orientation = self._resolve_object(object_name).get_pose()
            _, target_position, target_orientation, target_spec = self._resolve_target_pose_spec(target_like)
            position_tolerance, orientation_tolerance = self._target_tolerances(
                target_spec=target_spec,
                default_position_tolerance=position_tolerance,
                default_orientation_tolerance=orientation_tolerance,
            )
            if not pose_within_tolerance(
                current_position=object_position,
                current_orientation=object_orientation,
                target_position=target_position,
                target_orientation=target_orientation,
                position_tolerance=position_tolerance,
                orientation_tolerance=orientation_tolerance,
            ):
                return False
        return True

    def get_tracked_robot_states(self, phase_spec: dict | None = None) -> dict:
        phase_spec = phase_spec or self.get_current_phase_spec()
        tracked_states = {}
        robot_targets = phase_spec.get('robot_targets', {})
        advance = phase_spec.get('advance', {})
        default_position_tolerance = self._effective_robot_target_tolerance(float(advance.get('tolerance', 0.04)))
        default_orientation_tolerance = advance.get('orientation_tolerance')
        if default_orientation_tolerance is not None:
            default_orientation_tolerance = float(default_orientation_tolerance)

        for robot_name in self.cfg.robot_names:
            current_position, current_orientation = self._get_robot_task_pose(robot_name)
            robot_target_spec = robot_targets.get(robot_name)
            if robot_target_spec is None:
                tracked_states[robot_name] = {
                    'position': np.asarray(current_position).tolist(),
                    'orientation': np.asarray(current_orientation).tolist(),
                    'target_name': None,
                    'task_target': None,
                    'task_target_orientation': None,
                    'position_error': None,
                    'orientation_error': None,
                    'position_tolerance': None,
                    'orientation_tolerance': None,
                    'target_reached': None,
                }
                continue

            target_info = self._resolve_phase_robot_target(
                phase_spec=phase_spec,
                robot_name=robot_name,
                default_tolerance=default_position_tolerance,
                default_orientation_tolerance=default_orientation_tolerance,
            )
            tracked_states[robot_name] = {
                'position': np.asarray(current_position).tolist(),
                'orientation': np.asarray(current_orientation).tolist(),
                'target_name': target_info['target_name'],
                'task_target': None if target_info['target_position'] is None else np.asarray(target_info['target_position']).tolist(),
                'task_target_orientation': None if target_info['target_orientation'] is None else np.asarray(target_info['target_orientation']).tolist(),
                'position_error': target_info['position_error'],
                'orientation_error': target_info['orientation_error'],
                'position_tolerance': target_info['position_tolerance'],
                'orientation_tolerance': target_info['orientation_tolerance'],
                'target_reached': target_info['target_reached'],
            }
        return tracked_states

    def get_tracked_object_states(self) -> dict:
        tracked_states = {}
        for object_name in self.cfg.tracked_object_names:
            position, orientation = self._resolve_object(object_name).get_pose()
            attachment_state = self._attachments.get(object_name)
            locked_target = self._locked_targets.get(object_name)
            target_position = None
            target_orientation = None
            position_error = None
            orientation_error = None
            target_reached = None
            if locked_target is not None:
                target_pose = self.target_poses[locked_target]
                target_position = target_pose['position']
                target_orientation = target_pose['orientation']
                position_error, orientation_error = pose_error(
                    current_position=position,
                    current_orientation=orientation,
                    target_position=target_position,
                    target_orientation=target_orientation,
                )
                target_reached = pose_within_tolerance(
                    current_position=position,
                    current_orientation=orientation,
                    target_position=target_position,
                    target_orientation=target_orientation,
                    position_tolerance=0.0,
                    orientation_tolerance=0.0,
                )
            tracked_states[object_name] = {
                'position': np.asarray(position).tolist(),
                'orientation': np.asarray(orientation).tolist(),
                'scale': self._get_object_scale(object_name).tolist(),
                'attached_to': None if attachment_state is None else attachment_state.get('robot_name'),
                'attachment': None if attachment_state is None else copy.deepcopy(attachment_state),
                'locked_target': locked_target,
                'target_position': None if target_position is None else np.asarray(target_position).tolist(),
                'target_orientation': None if target_orientation is None else np.asarray(target_orientation).tolist(),
                'position_error': position_error,
                'orientation_error': orientation_error,
                'target_reached': target_reached,
                'collision_enabled': self._object_collision_enabled.get(object_name),
                'status': 'attached' if attachment_state is not None else ('locked' if locked_target is not None else 'free'),
            }
        return tracked_states

    def get_phase_runtime_state(self) -> dict:
        phase_spec = self.get_current_phase_spec()
        timeout_steps = self._phase_timeout_steps(phase_spec)
        timeout_remaining = None
        if timeout_steps is not None:
            timeout_remaining = max(int(timeout_steps) - int(self.phase_step_counter), 0)
        return {
            'phase': self.phase,
            'phase_index': self.phase_index,
            'phase_status': self.phase_status,
            'phase_step_counter': self.phase_step_counter,
            'phase_elapsed_steps': int(self.step_counter - self.phase_entry_step),
            'phase_entry_step': self.phase_entry_step,
            'phase_attempt': self.phase_attempts.get(self.phase, 0),
            'timeout_steps': timeout_steps,
            'timeout_remaining': timeout_remaining,
            'phase_history': list(self.phase_history),
            'phase_transition_history': copy.deepcopy(self.phase_transition_history[-32:]),
            'phase_timeout_count': self.phase_timeout_count,
            'phase_recovery_count': self.phase_recovery_count,
            'success': self.success,
            'failed': self.failed,
            'terminal_reason': self.terminal_reason,
            'last_transition_reason': self.last_transition_reason,
            'handoff_count': len(self._handoff_history),
            'recovery_events': copy.deepcopy(self._recovery_history[-16:]),
        }

    def _update_task_state(self):
        if self.success or self.failed or not self.phase_specs:
            return

        self._initialize_phase()
        self._sync_object_states()
        phase_spec = self.get_current_phase_spec()
        self._process_phase_interactions(phase_spec)
        self._sync_object_states()

        if self._advance_condition_met(phase_spec):
            if self.phase_index + 1 < len(self.phase_specs):
                self._set_phase(self.phase_index + 1, reason='advance', transition_type='advance', status='running')
                self._initialize_phase()
                self._sync_object_states()
            else:
                self.success = self._check_success()
                if self.success and self.phase != 'complete':
                    self._set_terminal_state('complete', reason='success-criteria-met', status='success')
        elif self._handle_phase_timeout(phase_spec):
            if not self.failed:
                self._initialize_phase()
                self._sync_object_states()

        self.phase_step_counter += 1

    def get_observations(self):
        self._update_task_state()
        obs: OrderedDict = super().get_observations()
        phase_spec = self.get_current_phase_spec()
        tracked_objects = self.get_tracked_object_states()
        tracked_robots = self.get_tracked_robot_states(phase_spec=phase_spec)
        runtime_state = self.get_phase_runtime_state()

        for robot_name in self.cfg.robot_names:
            if robot_name not in obs:
                continue
            robot_tracking = tracked_robots.get(robot_name, {})
            target_position = robot_tracking.get('task_target')
            target_orientation = robot_tracking.get('task_target_orientation')
            if target_position is None:
                target_position = None
                target_orientation = None

            obs[robot_name]['task_phase'] = self.phase
            obs[robot_name]['phase_step'] = self.phase_step_counter
            obs[robot_name]['task_target'] = target_position
            obs[robot_name]['task_target_orientation'] = target_orientation
            obs[robot_name]['task_target_position_error'] = robot_tracking.get('position_error')
            obs[robot_name]['task_target_orientation_error'] = robot_tracking.get('orientation_error')
            obs[robot_name]['task_target_reached'] = robot_tracking.get('target_reached')
            obs[robot_name]['tracked_objects'] = tracked_objects
            obs[robot_name]['tracked_robots'] = tracked_robots
            obs[robot_name]['task_runtime'] = runtime_state
            obs[robot_name]['recipe'] = self.cfg.recipe
        return obs

    def cleanup(self) -> None:
        for object_name in list(self._attachment_joints.keys()):
            self._remove_attachment_joint(object_name)
        self._contact_probes.clear()
        super().cleanup()

    def is_done(self) -> bool:
        self.step_counter += 1
        return self.success or self.failed or self.step_counter >= self.max_steps

    def calculate_metrics(self) -> dict:
        return {
            'recipe': self.cfg.recipe,
            'seed': self.cfg.seed,
            'episode_idx': self.cfg.episode_idx,
            'success': self.success or self._check_success(),
            'failed': self.failed,
            'phase_status': self.phase_status,
            'phase_history': self.phase_history,
            'phase_transition_history': self.phase_transition_history,
            'phase_attempts': self.phase_attempts,
            'steps': self.step_counter,
            'tracked_objects': self.get_tracked_object_states(),
            'tracked_robots': self.get_tracked_robot_states(),
            'timeout_count': self.phase_timeout_count,
            'recovery_count': self.phase_recovery_count,
            'handoff_history': self._handoff_history,
            'recovery_history': self._recovery_history,
            'terminal_reason': self.terminal_reason,
        }
