from __future__ import annotations

import math
import os
from collections import OrderedDict
from typing import Any

import numpy as np

from toolkits.factory_dual_franka_assembly.planner_primitives import (
    compose_pose,
    euler_xyz_to_quat,
    normalize_quat,
    pose_error,
    quat_multiply,
    quat_rotate,
)

_ARM_JOINT_CONTROLLER = 'arm_joint_controller'
_GRIPPER_CONTROLLER = 'gripper_controller'
_ARM_IK_CONTROLLER = 'arm_ik_controller'
_UR5E_ARM_JOINT_NAMES = (
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
)
_FRANKA_ARM_JOINT_NAMES = tuple(f'panda_joint{index}' for index in range(1, 8))
_SUPPORTED_ARM_JOINT_NAMES = (_FRANKA_ARM_JOINT_NAMES, _UR5E_ARM_JOINT_NAMES)
_CARTESIAN_SERVO_SKILLS = frozenset(
    {
        'ur5e_move_above_part',
        'ur5e_descend_to_grasp',
        'ur5e_retreat_vertical',
        'ur5e_move_part_to_staging',
        'ur5e_move_part_to_table_hover',
        'ur5e_hold_part_end',
    }
)


class UR5eAssemblyAtomicSkillAdapter:
    """Scripted atomic skills for continuous, bounded UR5e assembly motion.

    Each skill computes a Cartesian target, solves IK with the existing Lula solver,
    then sends the result through the joint-position controller.  Completion is
    reported through task.mark_local_skill_complete so recipes can chain the
    five atoms explicitly.
    """

    def __init__(self, spec: dict[str, Any]):
        del spec
        self.spec: dict[str, Any] = {}
        self._last_targets: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._phase_locks: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}
        self._close_gate_state: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._preshape_state: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._grasp_slip_state: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._completion_state: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._ik_failure_state: dict[tuple[Any, ...], int] = {}
        self._last_arm_command_q: dict[str, np.ndarray] = {}
        self._cartesian_command_positions: dict[tuple[Any, ...], np.ndarray] = {}
        self._cartesian_command_orientations: dict[tuple[Any, ...], np.ndarray] = {}
        self._insertion_axial_anchors: dict[tuple[Any, ...], float] = {}
        self._insertion_lateral_alignment_active: dict[tuple[Any, ...], bool] = {}
        self._insertion_lateral_alignment_stable_steps: dict[tuple[Any, ...], int] = {}
        self._insertion_lateral_orientation_anchors: dict[tuple[Any, ...], np.ndarray] = {}
        self._insertion_lateral_clearance_required: dict[tuple[Any, ...], bool] = {}
        self._insertion_lateral_clearance_anchors: dict[tuple[Any, ...], float] = {}
        self._target_object_settle_state: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._physically_relaxed_insertion_objects: set[tuple[int, str]] = set()
        self._insertion_compliance_transition_state: dict[tuple[int, str], dict[str, Any]] = {}
        self._compliant_motion_recovery_state: dict[tuple[int, str], dict[str, Any]] = {}

    def act(  # noqa: C901
        self,
        *,
        task,
        robot_name: str,
        phase_spec: dict,
        skill_spec: dict,
        tracked_robots: dict,
        tracked_objects: dict,
        checkpoint_path: str | None = None,
    ) -> dict | None:
        del checkpoint_path
        spec = {**self.spec, **dict(skill_spec)}
        skill_name = str(spec.get('name', ''))
        phase_key = (
            id(task),
            getattr(task, 'phase_index', None),
            getattr(task, 'phase_entry_step', None),
            robot_name,
            skill_name,
        )

        if skill_name in {'ur5e_preshape_gripper', 'preshape_gripper'}:
            return self._preshape_gripper_action(
                phase_key=phase_key,
                task=task,
                robot_name=robot_name,
                skill_name=skill_name,
                spec=spec,
                tracked_objects=tracked_objects,
            )

        if skill_name in {'ur5e_close_gripper', 'close_gripper'}:
            action = self._hold_joint_action(task=task, robot_name=robot_name)
            close_started_step = 0
            if bool(spec.get('require_close_pose_gate', False)):
                gate_ready, gate_action, gate_detail = self._close_pose_gate_action(
                    phase_key=phase_key,
                    task=task,
                    robot_name=robot_name,
                    spec=spec,
                    tracked_robots=tracked_robots,
                    tracked_objects=tracked_objects,
                )
                if not gate_ready:
                    timeout_steps = spec.get('close_pose_gate_timeout_steps')
                    phase_step_counter = int(getattr(task, 'phase_step_counter', 0))
                    stable_progress = int(gate_detail.get('ready_steps', 0))
                    required_stable_steps = max(int(gate_detail.get('required_ready_steps', 1)), 1)
                    timeout_grace_active = bool(
                        stable_progress > 0
                        and timeout_steps is not None
                        and phase_step_counter < int(timeout_steps) + required_stable_steps
                    )
                    if (
                        timeout_steps is not None
                        and phase_step_counter >= int(timeout_steps)
                        and not timeout_grace_active
                    ):
                        return self._failure_or_hold(
                            task,
                            robot_name,
                            spec,
                            'close_pose_gate_timeout',
                            diagnostics=gate_detail,
                        )
                    return gate_action
                state = self._close_gate_state.setdefault(phase_key, {})
                close_started_step = int(state.get('close_started_step', getattr(task, 'phase_step_counter', 0)))
                hold_q = state.get('hold_q')
                action = OrderedDict()
                if hold_q is not None:
                    action[_ARM_JOINT_CONTROLLER] = [np.asarray(hold_q, dtype=float).tolist()]

            hold_steps = max(int(spec.get('hold_steps', spec.get('close_steps', 36))), 0)
            close_elapsed_steps = max(int(getattr(task, 'phase_step_counter', 0)) - int(close_started_step), 0)
            ramp_steps = max(int(spec.get('close_ramp_steps', hold_steps)), 1)
            preclose_openness = float(spec.get('preclose_openness', spec.get('open_openness', 1.0)))
            closed_openness = float(spec.get('closed_openness', spec.get('close_openness', 0.0)))
            ramp_ratio = min(max(float(close_elapsed_steps) / float(ramp_steps), 0.0), 1.0)
            gripper_openness = preclose_openness + ramp_ratio * (closed_openness - preclose_openness)
            action[_GRIPPER_CONTROLLER] = [gripper_openness]
            if bool(spec.get('close_until_contact', False)):
                state = self._close_gate_state.setdefault(phase_key, {})
                close_ready, close_detail = self._close_until_contact_ready(
                    state=state,
                    task=task,
                    robot_name=robot_name,
                    spec=spec,
                    tracked_objects=tracked_objects,
                    close_elapsed_steps=close_elapsed_steps,
                    gripper_openness=gripper_openness,
                )
                close_detail['recenter'] = self._update_close_recenter_offset(
                    state=state,
                    close_detail=close_detail,
                    spec=spec,
                    close_ready=close_ready,
                )
                self._debug_close_step(
                    task=task,
                    robot_name=robot_name,
                    skill_name=skill_name,
                    close_elapsed_steps=close_elapsed_steps,
                    gripper_openness=gripper_openness,
                    close_ready=close_ready,
                    close_detail=close_detail,
                )
                hold_openness = state.get('hold_gripper_openness')
                if hold_openness is not None:
                    action[_GRIPPER_CONTROLLER] = [float(hold_openness)]
                if close_ready:
                    if hold_openness is None:
                        self._remember_gripper_hold_openness(
                            task=task,
                            robot_name=robot_name,
                            openness=gripper_openness,
                        )
                    self._mark_complete(
                        task=task,
                        robot_name=robot_name,
                        skill_name=skill_name,
                        detail=close_detail,
                    )
                elif self._close_object_motion_abort(
                    close_detail=close_detail,
                    spec=spec,
                    close_elapsed_steps=close_elapsed_steps,
                ):
                    return self._failure_or_hold(
                        task,
                        robot_name,
                        spec,
                        'close_object_knocked',
                        diagnostics=close_detail,
                    )
                else:
                    timeout_steps = spec.get('close_until_contact_timeout_steps')
                    if timeout_steps is not None and close_elapsed_steps >= int(timeout_steps):
                        return self._failure_or_hold(
                            task,
                            robot_name,
                            spec,
                            'close_until_contact_timeout',
                            diagnostics=close_detail,
                        )
                return action
            if close_elapsed_steps >= hold_steps:
                if bool(spec.get('require_grasp_contact', False)):
                    grasp_ready, grasp_detail = self._grasp_contact_ready(
                        task=task,
                        robot_name=robot_name,
                        spec=spec,
                    )
                    if not grasp_ready:
                        return self._failure_or_hold(
                            task,
                            robot_name,
                            spec,
                            'grasp_contact_not_ready',
                            diagnostics=grasp_detail,
                        )
                self._mark_complete(
                    task=task,
                    robot_name=robot_name,
                    skill_name=skill_name,
                    detail={
                        'closed': True,
                        'hold_steps': hold_steps,
                        'close_elapsed_steps': close_elapsed_steps,
                    },
                )
                self._remember_gripper_hold_openness(
                    task=task,
                    robot_name=robot_name,
                    openness=gripper_openness,
                )
            return action

        if skill_name in {'move_arm_to_joint_positions', 'ur5e_move_arm_to_joint_positions'}:
            return self._move_arm_to_joint_positions_action(
                phase_key=phase_key,
                task=task,
                robot_name=robot_name,
                skill_name=skill_name,
                spec=spec,
            )

        target_pose = self._target_pose(
            phase_key=phase_key,
            task=task,
            robot_name=robot_name,
            spec=spec,
            tracked_robots=tracked_robots,
            tracked_objects=tracked_objects,
        )
        if target_pose is None:
            return self._failure_or_hold(task, robot_name, spec, 'target_pose_unavailable')
        target_pose = self._locked_target_pose(phase_key=phase_key, target_pose=target_pose, spec=spec)

        prealign_action = self._prealign_action(
            task=task,
            robot_name=robot_name,
            target_pose=target_pose,
            spec=spec,
        )
        if prealign_action is not None:
            return prealign_action

        raw_current_pose = self._current_robot_pose(task=task, robot_name=robot_name, tracked_robots=tracked_robots)
        current_pose = self._current_tcp_pose(current_pose=raw_current_pose, spec=spec)
        slip_failure = self._object_tcp_slip_failure(
            phase_key=phase_key,
            task=task,
            robot_name=robot_name,
            spec=spec,
            tracked_objects=tracked_objects,
            current_pose=current_pose,
        )
        if slip_failure is not None:
            return self._failure_or_hold(
                task,
                robot_name,
                spec,
                'object_tcp_slip',
                diagnostics=slip_failure,
            )
        self._debug_transport_step(
            task=task,
            robot_name=robot_name,
            skill_name=skill_name,
            spec=spec,
            target_pose=target_pose,
            current_pose=current_pose,
            tracked_objects=tracked_objects,
        )
        compliance_transition_hold = self._maybe_relax_insertion_attachment(
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        if compliance_transition_hold:
            return self._held_transport_action(
                task=task,
                robot_name=robot_name,
                spec=spec,
            )
        recovery_allows_settle = self._compliant_recovery_allows_target_settle(
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        if not recovery_allows_settle and self._compliant_motion_requires_hold(
            phase_key=phase_key,
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        ):
            return self._held_transport_action(
                task=task,
                robot_name=robot_name,
                spec=spec,
            )
        if self._target_object_settle_ready(
            phase_key=phase_key,
            task=task,
            spec=spec,
            target_pose=target_pose,
            current_pose=current_pose,
            tracked_objects=tracked_objects,
        ):
            current_q = self._current_arm_q(task, robot_name)
            action = self._hold_joint_action(task=task, robot_name=robot_name)
            gripper_command = spec.get('gripper_command')
            if gripper_command is None:
                gripper_command = (
                    'close' if skill_name in {'ur5e_move_part_to_staging', 'ur5e_hold_part_end'} else 'open'
                )
            action[_GRIPPER_CONTROLLER] = [
                self._gripper_command_value(
                    task=task,
                    robot_name=robot_name,
                    command=gripper_command,
                )
            ]
            ik_target_pose = self._ik_target_pose(target_pose=target_pose, spec=spec)
            self._maybe_mark_complete(
                phase_key=phase_key,
                task=task,
                robot_name=robot_name,
                skill_name=skill_name,
                spec=spec,
                target_pose=target_pose,
                ik_target_pose=ik_target_pose,
                current_pose=current_pose,
                tracked_objects=tracked_objects,
                current_q=current_q,
                target_q=current_q,
            )
            return action
        if recovery_allows_settle:
            return self._held_transport_action(
                task=task,
                robot_name=robot_name,
                spec=spec,
            )
        cartesian_servo = bool(spec.get('cartesian_servo', skill_name in _CARTESIAN_SERVO_SKILLS))
        command_target_pose = target_pose
        if cartesian_servo and current_pose is not None:
            orientation_first_pose = self._orientation_first_servo_pose(
                skill_name=skill_name,
                spec=spec,
                current_pose=current_pose,
                target_pose=target_pose,
                phase_step_counter=int(getattr(task, 'phase_step_counter', 0)),
            )
            if orientation_first_pose is not None:
                command_target_pose = orientation_first_pose
            else:
                command_target_pose = self._target_object_servo_pose(
                    phase_key=phase_key,
                    task=task,
                    robot_name=robot_name,
                    spec=spec,
                    tracked_robots=tracked_robots,
                    tracked_objects=tracked_objects,
                    current_pose=current_pose,
                    target_pose=target_pose,
                )

        ik_target_pose = self._ik_target_pose(target_pose=command_target_pose, spec=spec)
        use_arm_ik_controller = bool(spec.get('use_arm_ik_controller', False))
        # Raw Cartesian IK can choose a different UR5e branch between frames.  Keep it
        # opt-in and otherwise route through the joint-limited IK path below.
        if use_arm_ik_controller and bool(spec.get('allow_direct_arm_ik_controller', False)):
            current_q = self._current_arm_q(task, robot_name)
            action = OrderedDict()
            action[_ARM_IK_CONTROLLER] = [
                np.asarray(ik_target_pose['position'], dtype=float).tolist(),
                np.asarray(ik_target_pose['orientation'], dtype=float).tolist(),
            ]
            gripper_command = spec.get('gripper_command')
            if gripper_command is None:
                gripper_command = (
                    'close' if skill_name in {'ur5e_move_part_to_staging', 'ur5e_hold_part_end'} else 'open'
                )
            action[_GRIPPER_CONTROLLER] = [
                self._gripper_command_value(task=task, robot_name=robot_name, command=gripper_command)
            ]
            self._debug_joint_step(
                task=task,
                robot_name=robot_name,
                skill_name=skill_name,
                spec=spec,
                current_pose=current_pose,
                target_pose=target_pose,
                command_target_pose=command_target_pose,
                ik_target_pose=ik_target_pose,
                current_q=current_q,
                reference_q=current_q,
                target_q=None,
                command_q=None,
            )
            self._last_targets[phase_key] = {
                'position': target_pose['position'].copy(),
                'orientation': target_pose['orientation'].copy(),
            }
            self._remember_cartesian_command_position(
                phase_key=phase_key,
                command_target_pose=command_target_pose,
            )
            self._remember_cartesian_command_orientation(
                phase_key=phase_key,
                command_target_pose=command_target_pose,
            )
            self._maybe_mark_complete(
                phase_key=phase_key,
                task=task,
                robot_name=robot_name,
                skill_name=skill_name,
                spec=spec,
                target_pose=target_pose,
                ik_target_pose=ik_target_pose,
                current_pose=current_pose,
                tracked_objects=tracked_objects,
                current_q=current_q,
                target_q=None,
            )
            return action

        current_q = self._current_arm_q(task, robot_name)
        reference_q = self._command_reference_q(task=task, robot_name=robot_name, current_q=current_q, spec=spec)
        ik_result = self._solve_ik(
            task=task,
            robot_name=robot_name,
            target_pose=ik_target_pose,
            warm_start=reference_q,
            spec=spec,
        )
        retried_with_measured_state = False
        measured_q = self._coerce_arm_q(current_q)
        if (
            ik_result is None
            and bool(spec.get('ik_retry_with_measured_state', True))
            and measured_q is not None
            and (
                reference_q is None
                or reference_q.shape != measured_q.shape
                or float(np.max(np.abs(reference_q - measured_q))) > 1e-6
            )
        ):
            retried_with_measured_state = True
            ik_result = self._solve_ik(
                task=task,
                robot_name=robot_name,
                target_pose=ik_target_pose,
                warm_start=measured_q,
                spec=spec,
            )
        ik_backtrack_ratio = None
        if ik_result is None and cartesian_servo and current_pose is not None:
            for raw_ratio in spec.get('ik_backtrack_ratios', (0.5, 0.25, 0.125)):
                try:
                    ratio = float(raw_ratio)
                except (TypeError, ValueError):
                    continue
                if not 0.0 < ratio < 1.0:
                    continue
                candidate_command_pose = self._cartesian_pose_fraction(
                    current_pose=current_pose,
                    target_pose=command_target_pose,
                    ratio=ratio,
                )
                candidate_ik_pose = self._ik_target_pose(
                    target_pose=candidate_command_pose,
                    spec=spec,
                )
                candidate_result = self._solve_ik(
                    task=task,
                    robot_name=robot_name,
                    target_pose=candidate_ik_pose,
                    warm_start=reference_q,
                    spec=spec,
                )
                if (
                    candidate_result is None
                    and bool(spec.get('ik_retry_with_measured_state', True))
                    and measured_q is not None
                    and (
                        reference_q is None
                        or reference_q.shape != measured_q.shape
                        or float(np.max(np.abs(reference_q - measured_q))) > 1e-6
                    )
                ):
                    retried_with_measured_state = True
                    candidate_result = self._solve_ik(
                        task=task,
                        robot_name=robot_name,
                        target_pose=candidate_ik_pose,
                        warm_start=measured_q,
                        spec=spec,
                    )
                if candidate_result is None:
                    continue
                ik_result = candidate_result
                command_target_pose = candidate_command_pose
                ik_target_pose = candidate_ik_pose
                ik_backtrack_ratio = ratio
                break
        if ik_result is None:
            self._debug_motion_blocked(
                task=task,
                robot_name=robot_name,
                skill_name=skill_name,
                reason='ik_failed',
                spec=spec,
                target_pose=target_pose,
                ik_target_pose=ik_target_pose,
                reference_q=reference_q,
            )
            failure_streak = int(self._ik_failure_state.get(phase_key, 0)) + 1
            self._ik_failure_state[phase_key] = failure_streak
            tolerance_steps = max(
                int(spec.get('ik_failure_tolerance_steps', 48 if cartesian_servo else 0)),
                0,
            )
            if failure_streak <= tolerance_steps:
                action = self._hold_joint_action(task=task, robot_name=robot_name)
                gripper_command = spec.get('gripper_command')
                if gripper_command is None:
                    gripper_command = (
                        'close' if skill_name in {'ur5e_move_part_to_staging', 'ur5e_hold_part_end'} else 'open'
                    )
                action[_GRIPPER_CONTROLLER] = [
                    self._gripper_command_value(
                        task=task,
                        robot_name=robot_name,
                        command=gripper_command,
                    )
                ]
                return action
            return self._failure_or_hold(
                task,
                robot_name,
                spec,
                'ik_failed',
                diagnostics={
                    'consecutive_failures': failure_streak,
                    'tolerance_steps': tolerance_steps,
                    'retried_with_measured_state': retried_with_measured_state,
                    'last_backtrack_ratio': ik_backtrack_ratio,
                },
            )
        target_q = self._unwrap_to_reference(
            target_q=ik_result,
            reference_q=reference_q,
            preferred_abs_limit=spec.get('preferred_joint_abs_limit', 3.05),
            hard_preferred_abs_limit=bool(spec.get('hard_preferred_joint_abs_limit', True)),
        )
        if reference_q is None:
            return self._failure_or_hold(
                task,
                robot_name,
                spec,
                'current_joint_state_unavailable',
                diagnostics={
                    'target_position': target_pose['position'].tolist(),
                    'target_orientation': target_pose['orientation'].tolist(),
                },
            )
        guard_branch_jump = bool(spec.get('guard_ik_branch_jump', cartesian_servo))
        branch_jump = bool(
            guard_branch_jump
            and self._ik_branch_jump_detected(
                reference_q=reference_q,
                target_q=target_q,
                spec=spec,
            )
        )
        branch_recovery_mode = None
        if branch_jump and cartesian_servo and current_pose is not None:
            recovery = self._recover_ik_branch_jump(
                task=task,
                robot_name=robot_name,
                skill_name=skill_name,
                spec=spec,
                current_pose=current_pose,
                command_target_pose=command_target_pose,
                reference_q=reference_q,
                measured_q=measured_q,
            )
            if recovery is not None:
                ik_result = recovery['ik_result']
                target_q = recovery['target_q']
                command_target_pose = recovery['command_target_pose']
                ik_target_pose = recovery['ik_target_pose']
                branch_recovery_mode = recovery['mode']
                branch_jump = False
        if branch_jump:
            self._debug_motion_blocked(
                task=task,
                robot_name=robot_name,
                skill_name=skill_name,
                reason='ik_branch_jump_guard',
                spec=spec,
                target_pose=target_pose,
                ik_target_pose=ik_target_pose,
                reference_q=reference_q,
                target_q=target_q,
            )
            failure_streak = int(self._ik_failure_state.get(phase_key, 0)) + 1
            self._ik_failure_state[phase_key] = failure_streak
            tolerance_steps = max(int(spec.get('ik_branch_jump_tolerance_steps', 48)), 0)
            diagnostics = {
                'target_position': target_pose['position'].tolist(),
                'target_orientation': target_pose['orientation'].tolist(),
                'reference_q': reference_q.tolist(),
                'target_q': target_q.tolist(),
                'consecutive_failures': failure_streak,
                'tolerance_steps': tolerance_steps,
            }
            if failure_streak > tolerance_steps:
                return self._failure_or_hold(
                    task,
                    robot_name,
                    spec,
                    'ik_branch_jump_guard',
                    diagnostics=diagnostics,
                )
            return self._failure_or_hold(
                task,
                robot_name,
                {**spec, 'require_success': False},
                'ik_branch_jump_guard',
                diagnostics=diagnostics,
            )
        self._ik_failure_state.pop(phase_key, None)
        joint_step_limits = self._command_joint_step_limits(
            spec=spec,
            joint_count=reference_q.shape[0],
        )
        if joint_step_limits is None:
            joint_step_limits = float(spec.get('max_joint_step', 0.035))
        command_q = self._limited_joint_target(
            current_q=reference_q,
            target_q=target_q,
            max_joint_step=joint_step_limits,
        )
        command_q = self._continuous_command_q(
            task=task,
            robot_name=robot_name,
            command_q=command_q,
            spec=spec,
        )
        command_q = self._limit_command_to_measured_state(
            current_q=current_q,
            command_q=command_q,
            spec=spec,
        )
        self._debug_joint_step(
            task=task,
            robot_name=robot_name,
            skill_name=skill_name,
            spec=spec,
            current_pose=current_pose,
            target_pose=target_pose,
            command_target_pose=command_target_pose,
            ik_target_pose=ik_target_pose,
            current_q=current_q,
            reference_q=reference_q,
            target_q=target_q,
            command_q=command_q,
        )

        self._last_targets[phase_key] = {
            'position': target_pose['position'].copy(),
            'orientation': target_pose['orientation'].copy(),
            'target_q': target_q.copy(),
        }
        self._remember_cartesian_command_position(
            phase_key=phase_key,
            command_target_pose=command_target_pose,
        )
        self._remember_cartesian_command_orientation(
            phase_key=phase_key,
            command_target_pose=command_target_pose,
        )
        if branch_recovery_mode is not None:
            self._last_targets[phase_key]['ik_branch_recovery_mode'] = branch_recovery_mode

        action = OrderedDict()
        if use_arm_ik_controller:
            action[_ARM_IK_CONTROLLER] = [
                np.asarray(ik_target_pose['position'], dtype=float).tolist(),
                np.asarray(ik_target_pose['orientation'], dtype=float).tolist(),
            ]
        action[_ARM_JOINT_CONTROLLER] = [command_q.tolist()]
        self._remember_arm_command(task, robot_name, command_q)
        gripper_command = spec.get('gripper_command')
        if gripper_command is None:
            gripper_command = 'close' if skill_name in {'ur5e_move_part_to_staging', 'ur5e_hold_part_end'} else 'open'
        action[_GRIPPER_CONTROLLER] = [
            self._gripper_command_value(task=task, robot_name=robot_name, command=gripper_command)
        ]

        self._maybe_mark_complete(
            phase_key=phase_key,
            task=task,
            robot_name=robot_name,
            skill_name=skill_name,
            spec=spec,
            target_pose=target_pose,
            ik_target_pose=ik_target_pose,
            current_pose=current_pose,
            tracked_objects=tracked_objects,
            current_q=current_q,
            target_q=target_q,
        )
        return action

    def _maybe_mark_complete(
        self,
        *,
        phase_key=None,
        task,
        robot_name: str,
        skill_name: str,
        spec: dict,
        target_pose: dict,
        ik_target_pose: dict,
        current_pose: dict | None,
        tracked_objects: dict,
        current_q,
        target_q,
    ) -> None:
        position_tolerance = float(spec.get('position_tolerance', 0.025))
        relaxed_position_tolerance = spec.get('relaxed_position_tolerance')
        relaxed_after_steps = spec.get('relaxed_position_tolerance_after_steps')
        relaxed_tolerance_active = bool(
            relaxed_position_tolerance is not None
            and relaxed_after_steps is not None
            and int(getattr(task, 'phase_step_counter', 0)) >= int(relaxed_after_steps)
        )
        effective_position_tolerance = position_tolerance
        if relaxed_tolerance_active:
            effective_position_tolerance = max(
                position_tolerance,
                float(relaxed_position_tolerance),
            )
        orientation_tolerance = spec.get('orientation_tolerance')
        orientation_tolerance = None if orientation_tolerance is None else float(orientation_tolerance)
        object_name = self._object_name_from_spec(spec)
        compliant_linear_allowance, compliant_angular_allowance, attachment_mode = self._compliant_attachment_allowance(
            object_name=object_name,
            tracked_objects=tracked_objects,
        )
        effective_position_tolerance += compliant_linear_allowance
        if orientation_tolerance is not None:
            orientation_tolerance += compliant_angular_allowance
        force_complete_after_steps = spec.get('force_complete_after_steps')
        complete = False
        completion_detail = {
            'target_position': target_pose['position'].tolist(),
            'target_orientation': target_pose['orientation'].tolist(),
            'ik_target_position': ik_target_pose['position'].tolist(),
            'ik_target_orientation': ik_target_pose['orientation'].tolist(),
            'attachment_mode': attachment_mode,
            'compliant_linear_allowance': compliant_linear_allowance,
            'compliant_angular_allowance': compliant_angular_allowance,
        }
        tcp_offset = self._tcp_offset(spec)
        if tcp_offset is not None:
            completion_detail['grasp_tcp_offset'] = tcp_offset.tolist()
            completion_detail['grasp_tcp_offset_frame'] = self._tcp_offset_frame(spec)
        if current_pose is not None:
            position_error, orientation_error = pose_error(
                current_position=current_pose['position'],
                current_orientation=current_pose['orientation'],
                target_position=target_pose['position'],
                target_orientation=target_pose['orientation'],
            )
            completion_detail.update(
                {
                    'position_error': position_error,
                    'orientation_error': orientation_error,
                    'position_tolerance': effective_position_tolerance,
                    'primary_position_tolerance': position_tolerance,
                    'relaxed_position_tolerance': relaxed_position_tolerance,
                    'relaxed_position_tolerance_after_steps': relaxed_after_steps,
                    'relaxed_position_tolerance_active': relaxed_tolerance_active,
                    'orientation_tolerance': orientation_tolerance,
                }
            )
            complete = position_error <= effective_position_tolerance and (
                orientation_tolerance is None or orientation_error is None or orientation_error <= orientation_tolerance
            )
            if attachment_mode == 'compliant_joint' and bool(spec.get('require_target_object_pose_convergence', False)):
                complete = True
                completion_detail['tcp_pose_required_for_completion'] = False

        if bool(spec.get('require_target_object_pose_convergence', False)):
            object_pose = None
            target_object_pose = None
            if object_name is not None:
                object_pose = self._object_pose(
                    task=task,
                    object_name=object_name,
                    tracked_objects=tracked_objects,
                )
                target_object_pose = self._target_object_pose(task=task, spec=spec)

            object_position_tolerance = float(
                spec.get('target_object_position_tolerance', effective_position_tolerance)
            )
            primary_object_position_tolerance = object_position_tolerance
            relaxed_object_position_tolerance = spec.get('relaxed_target_object_position_tolerance')
            if relaxed_tolerance_active and relaxed_object_position_tolerance is not None:
                object_position_tolerance = max(
                    object_position_tolerance,
                    float(relaxed_object_position_tolerance),
                )
            object_orientation_tolerance = spec.get(
                'target_object_orientation_tolerance',
                orientation_tolerance,
            )
            object_orientation_tolerance = (
                None if object_orientation_tolerance is None else float(object_orientation_tolerance)
            )
            object_pose_complete = False
            object_position_error = None
            object_orientation_error = None
            object_axial_position_error = None
            object_lateral_position_error = None
            object_lateral_position_tolerance = spec.get('target_object_lateral_position_tolerance')
            object_convergence_axis = spec.get('target_object_convergence_axis')
            lateral_alignment_complete = True
            if (
                phase_key is not None
                and object_convergence_axis is not None
                and object_lateral_position_tolerance is not None
            ):
                lateral_alignment_complete = not bool(self._insertion_lateral_alignment_active.get(phase_key, False))
            if object_pose is not None and target_object_pose is not None:
                object_position_delta = np.asarray(object_pose['position'], dtype=float) - np.asarray(
                    target_object_pose['position'], dtype=float
                )
                target_object_orientation = target_object_pose.get('orientation')
                if target_object_orientation is None:
                    object_position_error = float(np.linalg.norm(object_position_delta))
                else:
                    object_position_error, object_orientation_error = pose_error(
                        current_position=object_pose['position'],
                        current_orientation=object_pose['orientation'],
                        target_position=target_object_pose['position'],
                        target_orientation=target_object_orientation,
                    )
                lateral_complete = True
                if object_convergence_axis is not None:
                    convergence_axis = np.asarray(object_convergence_axis, dtype=float)
                    axis_norm = float(np.linalg.norm(convergence_axis))
                    if convergence_axis.shape != (3,) or not np.isfinite(axis_norm) or axis_norm <= 1e-9:
                        raise ValueError('target_object_convergence_axis must be a finite non-zero 3D vector.')
                    convergence_axis = convergence_axis / axis_norm
                    signed_axial_error = float(np.dot(object_position_delta, convergence_axis))
                    lateral_error = object_position_delta - signed_axial_error * convergence_axis
                    object_axial_position_error = abs(signed_axial_error)
                    object_lateral_position_error = float(np.linalg.norm(lateral_error))
                    if object_lateral_position_tolerance is not None:
                        lateral_complete = bool(
                            object_lateral_position_error <= float(object_lateral_position_tolerance)
                        )
                object_pose_complete = bool(
                    object_position_error <= object_position_tolerance
                    and lateral_complete
                    and (
                        object_orientation_tolerance is None
                        or object_orientation_error is None
                        or object_orientation_error <= object_orientation_tolerance
                    )
                )
            completion_detail.update(
                {
                    'target_object_pose_required': True,
                    'target_object_name': object_name,
                    'target_object_position': (
                        None
                        if target_object_pose is None
                        else np.asarray(target_object_pose['position'], dtype=float).tolist()
                    ),
                    'target_object_orientation': (
                        None
                        if target_object_pose is None or target_object_pose.get('orientation') is None
                        else np.asarray(target_object_pose['orientation'], dtype=float).tolist()
                    ),
                    'object_position_error': object_position_error,
                    'object_orientation_error': object_orientation_error,
                    'target_object_convergence_axis': object_convergence_axis,
                    'object_axial_position_error': object_axial_position_error,
                    'object_lateral_position_error': object_lateral_position_error,
                    'target_object_lateral_alignment_complete': (lateral_alignment_complete),
                    'target_object_lateral_position_tolerance': (
                        None if object_lateral_position_tolerance is None else float(object_lateral_position_tolerance)
                    ),
                    'target_object_position_tolerance': object_position_tolerance,
                    'primary_target_object_position_tolerance': primary_object_position_tolerance,
                    'relaxed_target_object_position_tolerance': relaxed_object_position_tolerance,
                    'target_object_orientation_tolerance': object_orientation_tolerance,
                    'target_object_pose_complete': object_pose_complete,
                }
            )
            complete = bool(complete and object_pose_complete)

        if bool(spec.get('require_target_object_static', False)):
            object_name = self._object_name_from_spec(spec)
            motion_detail = self._object_motion_detail(
                task=task,
                object_name=object_name,
                tracked_objects=tracked_objects,
            )
            max_linear_speed = float(spec.get('target_object_max_linear_speed', 0.03))
            max_angular_speed = float(spec.get('target_object_max_angular_speed', 2.0))
            configured_stable_steps = max(
                int(spec.get('target_object_stable_steps', 8)),
                1,
            )
            velocity_motion_ready = bool(
                motion_detail.get('valid')
                and motion_detail.get('linear_speed') is not None
                and motion_detail.get('angular_speed') is not None
                and float(motion_detail['linear_speed']) <= max_linear_speed
                and float(motion_detail['angular_speed']) <= max_angular_speed
            )
            pose_stable_override_used = bool(
                spec.get('target_object_allow_pose_stable_override', True)
                and motion_detail.get('is_static') is True
                and motion_detail.get('pose_stable_override') is True
                and motion_detail.get('linear_speed') is not None
                and motion_detail.get('angular_speed') is not None
                and float(motion_detail['linear_speed'])
                <= float(
                    spec.get(
                        'target_object_pose_stable_override_max_linear_speed',
                        max(
                            2.0 * max_linear_speed,
                            float(
                                spec.get(
                                    'compliant_servo_settle_max_linear_speed',
                                    max_linear_speed,
                                )
                            ),
                        ),
                    )
                )
                and float(motion_detail['angular_speed'])
                <= float(
                    spec.get(
                        'target_object_pose_stable_override_max_angular_speed',
                        spec.get(
                            'compliant_servo_settle_max_angular_speed',
                            max_angular_speed,
                        ),
                    )
                )
            )
            pose_history_velocity_override_used = self._pose_history_velocity_override_ready(
                spec=spec,
                motion_detail=motion_detail,
            )
            motion_ready = bool(velocity_motion_ready or pose_stable_override_used)
            motion_ready = bool(motion_ready or pose_history_velocity_override_used)
            entry_capture_max_steps = max(
                int(spec.get('target_object_entry_capture_max_steps', 0)),
                0,
            )
            entry_capture_active = bool(
                entry_capture_max_steps > 0
                and int(getattr(task, 'phase_step_counter', 0)) <= entry_capture_max_steps
                and complete
                and velocity_motion_ready
            )
            required_stable_steps = 1 if entry_capture_active else configured_stable_steps
            if phase_key is None:
                phase_key = (
                    id(task),
                    getattr(task, 'phase_index', None),
                    getattr(task, 'phase_entry_step', None),
                    robot_name,
                    skill_name,
                )
            state = self._completion_state.setdefault(phase_key, {})
            if complete and motion_ready:
                state['target_object_stable_steps'] = int(state.get('target_object_stable_steps', 0)) + 1
            else:
                state['target_object_stable_steps'] = 0
            stable_steps = int(state['target_object_stable_steps'])
            completion_detail.update(
                {
                    'target_object_static_required': True,
                    'target_object_motion_detail': motion_detail,
                    'target_object_motion_ready': motion_ready,
                    'target_object_velocity_motion_ready': velocity_motion_ready,
                    'target_object_pose_stable_override_used': (pose_stable_override_used),
                    'target_object_pose_history_velocity_override_used': (pose_history_velocity_override_used),
                    'target_object_max_linear_speed': max_linear_speed,
                    'target_object_max_angular_speed': max_angular_speed,
                    'target_object_stable_steps': stable_steps,
                    'configured_target_object_stable_steps': (configured_stable_steps),
                    'required_target_object_stable_steps': required_stable_steps,
                    'target_object_entry_capture_max_steps': (entry_capture_max_steps),
                    'target_object_entry_capture_active': entry_capture_active,
                }
            )
            complete = bool(complete and motion_ready and stable_steps >= required_stable_steps)

        joint_position_tolerance = spec.get('joint_position_tolerance')
        if joint_position_tolerance is not None and current_q is not None and target_q is not None:
            joint_error = float(np.max(np.abs(np.asarray(target_q, dtype=float) - np.asarray(current_q, dtype=float))))
            completion_detail.update(
                {
                    'joint_error': joint_error,
                    'joint_position_tolerance': float(joint_position_tolerance),
                }
            )
            complete = bool(complete and joint_error <= float(joint_position_tolerance))

        if force_complete_after_steps is not None and int(getattr(task, 'phase_step_counter', 0)) >= int(
            force_complete_after_steps
        ):
            complete = True
            completion_detail['force_complete'] = True

        if complete:
            self._mark_complete(
                task=task,
                robot_name=robot_name,
                skill_name=skill_name,
                detail=completion_detail,
            )

    @staticmethod
    def _debug_grasp_enabled() -> bool:
        return os.environ.get('UR5E_DEBUG_GRASP', '0').strip().lower() in {'1', 'true', 'yes'}

    @classmethod
    def _debug_close_enabled(cls) -> bool:
        return cls._debug_grasp_enabled() or os.environ.get('UR5E_DEBUG_CLOSE', '0').strip().lower() in {
            '1',
            'true',
            'yes',
        }

    def _debug_close_step(
        self,
        *,
        task,
        robot_name: str,
        skill_name: str,
        close_elapsed_steps: int,
        gripper_openness: float,
        close_ready: bool,
        close_detail: dict[str, Any],
    ) -> None:
        if not self._debug_close_enabled():
            return
        should_print = (
            int(close_elapsed_steps) <= 5
            or int(close_elapsed_steps) % 12 == 0
            or bool(close_ready)
            or bool(close_detail.get('detected_clamp'))
            or not bool(close_detail.get('motion_ready', True))
        )
        if not should_print:
            return
        motion_detail = close_detail.get('motion_detail') or {}
        recenter_detail = close_detail.get('recenter') or {}
        contact_detail = close_detail.get('contact_detail') or {}
        contact_metrics = contact_detail.get('contact_metrics') if isinstance(contact_detail, dict) else {}
        left_gap = None
        right_gap = None
        if isinstance(contact_metrics, dict):
            left_finger = contact_metrics.get('left_finger') or {}
            right_finger = contact_metrics.get('right_finger') or {}
            left_gap = left_finger.get('surface_gap')
            right_gap = right_finger.get('surface_gap')
            left_local = (left_finger.get('local_contact') or {}).get('local_point')
            right_local = (right_finger.get('local_contact') or {}).get('local_point')
        else:
            left_local = None
            right_local = None
        print(
            '[ur5e-grasp-debug] '
            f"step={getattr(task, 'step_counter', None)} "
            f"phase_step={getattr(task, 'phase_step_counter', None)} "
            f'robot={robot_name} skill={skill_name} close_elapsed={int(close_elapsed_steps)} '
            f'cmd_open={float(gripper_openness):.4f} '
            f"q={close_detail.get('gripper_joint_position')} "
            f"target_q={close_detail.get('target_gripper_joint_position')} "
            f"ready={bool(close_ready)} reason={close_detail.get('completion_reason')} "
            f"detected_clamp={close_detail.get('detected_clamp')} "
            f"contact_ready={close_detail.get('contact_ready')} stable={close_detail.get('stable_steps')}/"
            f"{close_detail.get('required_stable_steps')} "
            f"motion_ready={close_detail.get('motion_ready')} motion_stable="
            f"{close_detail.get('motion_stable_steps')}/{close_detail.get('required_motion_stable_steps')} "
            f"lin={motion_detail.get('linear_speed')} ang={motion_detail.get('angular_speed')} "
            f"is_static={motion_detail.get('is_static')} pose_static={motion_detail.get('pose_stable_override')} "
            f"hold_open={close_detail.get('hold_gripper_openness')} "
            f"strict={contact_detail.get('strict_contact_ready')} "
            f"pinch_axis={contact_metrics.get('pinch_axis') if isinstance(contact_metrics, dict) else None} "
            f"recenter_side={recenter_detail.get('single_finger_side')} "
            f"recenter_updated={recenter_detail.get('updated')} "
            f"recenter_offset={recenter_detail.get('offset_world')} "
            f'left_gap={left_gap} right_gap={right_gap} '
            f'left_local={left_local} right_local={right_local}',
            flush=True,
        )

    def _debug_transport_step(
        self,
        *,
        task,
        robot_name: str,
        skill_name: str,
        spec: dict,
        target_pose: dict,
        current_pose: dict | None,
        tracked_objects: dict,
    ) -> None:
        if not self._debug_grasp_enabled():
            return
        if not (
            spec.get('target_object_position') is not None
            or spec.get('target_object_target') is not None
            or spec.get('target_object') is not None
            or spec.get('object_target') is not None
            or skill_name in {'ur5e_move_part_to_table_hover', 'ur5e_move_part_to_staging', 'ur5e_hold_part_end'}
        ):
            return
        phase_step = int(getattr(task, 'phase_step_counter', 0))
        every = max(int(os.environ.get('UR5E_DEBUG_TRANSPORT_EVERY', '15')), 1)
        if phase_step > 5 and phase_step % every != 0:
            return

        object_name = self._object_name_from_spec(spec)
        object_pose = None
        if object_name:
            object_pose = self._object_pose(
                task=task,
                object_name=object_name,
                tracked_objects=tracked_objects,
            )
        target_object_position = self._target_object_position(task=task, spec=spec)
        object_error = None
        object_axial_error = None
        object_lateral_error = None
        relative_world = None
        if object_pose is not None and target_object_position is not None:
            object_delta = np.asarray(object_pose['position'], dtype=float) - np.asarray(
                target_object_position, dtype=float
            )
            object_error = float(np.linalg.norm(object_delta))
            convergence_axis = spec.get('target_object_convergence_axis')
            if convergence_axis is not None:
                convergence_axis = np.asarray(convergence_axis, dtype=float)
                axis_norm = float(np.linalg.norm(convergence_axis))
                if convergence_axis.shape == (3,) and np.isfinite(axis_norm) and axis_norm > 1e-9:
                    convergence_axis = convergence_axis / axis_norm
                    signed_axial_error = float(np.dot(object_delta, convergence_axis))
                    object_axial_error = abs(signed_axial_error)
                    object_lateral_error = float(np.linalg.norm(object_delta - signed_axial_error * convergence_axis))
        if object_pose is not None and current_pose is not None:
            relative_world = np.asarray(object_pose['position'], dtype=float) - np.asarray(
                current_pose['position'], dtype=float
            )

        tcp_error = None
        if current_pose is not None:
            tcp_error = float(
                np.linalg.norm(
                    np.asarray(current_pose['position'], dtype=float) - np.asarray(target_pose['position'], dtype=float)
                )
            )
        motion_detail = self._object_motion_detail(
            task=task,
            object_name=object_name,
            tracked_objects=tracked_objects,
        )
        print(
            '[ur5e-transport-debug] '
            f"step={getattr(task, 'step_counter', None)} phase={getattr(task, 'phase', None)} "
            f'phase_step={phase_step} robot={robot_name} skill={skill_name} '
            f'target_object_position={None if target_object_position is None else np.asarray(target_object_position, dtype=float).tolist()} '
            f"object_position={None if object_pose is None else np.asarray(object_pose['position'], dtype=float).tolist()} "
            f'object_error={object_error} '
            f'object_axial_error={object_axial_error} '
            f'object_lateral_error={object_lateral_error} '
            f"target_tcp={np.asarray(target_pose['position'], dtype=float).tolist()} "
            f"current_tcp={None if current_pose is None else np.asarray(current_pose['position'], dtype=float).tolist()} "
            f'tcp_error={tcp_error} '
            f'relative_world={None if relative_world is None else relative_world.tolist()} '
            f'gripper_q={self._current_gripper_q(task=task, robot_name=robot_name)} '
            f'hold_open={self._last_gripper_hold_openness(task=task, robot_name=robot_name)} '
            f"lin={motion_detail.get('linear_speed')} ang={motion_detail.get('angular_speed')} "
            f"is_static={motion_detail.get('is_static')} pose_static={motion_detail.get('pose_stable_override')}",
            flush=True,
        )

    def _debug_motion_blocked(
        self,
        *,
        task,
        robot_name: str,
        skill_name: str,
        reason: str,
        spec: dict,
        target_pose: dict,
        ik_target_pose: dict,
        reference_q=None,
        target_q=None,
    ) -> None:
        if not self._debug_grasp_enabled():
            return
        phase_step = int(getattr(task, 'phase_step_counter', 0))
        every = max(int(os.environ.get('UR5E_DEBUG_TRANSPORT_EVERY', '15')), 1)
        if phase_step > 5 and phase_step % every != 0:
            return
        print(
            '[ur5e-motion-blocked] '
            f"step={getattr(task, 'step_counter', None)} phase={getattr(task, 'phase', None)} "
            f'phase_step={phase_step} robot={robot_name} skill={skill_name} reason={reason} '
            f"target_tcp={np.asarray(target_pose['position'], dtype=float).tolist()} "
            f"ik_target={np.asarray(ik_target_pose['position'], dtype=float).tolist()} "
            f'reference_q={None if reference_q is None else np.asarray(reference_q, dtype=float).tolist()} '
            f'target_q={None if target_q is None else np.asarray(target_q, dtype=float).tolist()} '
            f"max_joint_step={spec.get('max_joint_step')} "
            f"guard_ik_branch_jump={spec.get('guard_ik_branch_jump', spec.get('cartesian_servo', False))}",
            flush=True,
        )

    def _debug_joint_step(
        self,
        *,
        task,
        robot_name: str,
        skill_name: str,
        spec: dict,
        current_pose: dict | None,
        target_pose: dict,
        command_target_pose: dict,
        ik_target_pose: dict,
        current_q,
        reference_q,
        target_q,
        command_q,
    ) -> None:
        if not self._debug_grasp_enabled():
            return
        if not (
            bool(spec.get('cartesian_servo', False))
            or spec.get('target_object_position') is not None
            or spec.get('target_object_target') is not None
            or skill_name in {'ur5e_move_part_to_table_hover', 'ur5e_move_part_to_staging', 'ur5e_hold_part_end'}
        ):
            return
        phase_step = int(getattr(task, 'phase_step_counter', 0))
        every = max(int(os.environ.get('UR5E_DEBUG_TRANSPORT_EVERY', '15')), 1)
        if phase_step > 5 and phase_step % every != 0:
            return

        current_q = self._coerce_arm_q(current_q)
        reference_q = self._coerce_arm_q(reference_q)
        target_q = self._coerce_arm_q(target_q)
        command_q = self._coerce_arm_q(command_q)

        def _max_abs_delta(a, b):
            if a is None or b is None:
                return None
            if a.shape != b.shape:
                return None
            return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))

        tcp_error = None
        tcp_orientation_error = None
        command_tcp_step = None
        command_orientation_step = None
        if current_pose is not None:
            tcp_error, tcp_orientation_error = pose_error(
                current_pose['position'],
                current_pose['orientation'],
                target_pose['position'],
                target_pose['orientation'],
            )
            command_tcp_step, command_orientation_step = pose_error(
                current_pose['position'],
                current_pose['orientation'],
                command_target_pose['position'],
                command_target_pose['orientation'],
            )
        ik_position_tolerance, ik_orientation_tolerance = self._ik_solver_tolerances(spec)
        arm_dynamics = self._current_arm_dynamics(task=task, robot_name=robot_name)
        print(
            '[ur5e-joint-debug] '
            f"step={getattr(task, 'step_counter', None)} phase={getattr(task, 'phase', None)} "
            f'phase_step={phase_step} robot={robot_name} skill={skill_name} '
            f"ik_reference_mode={spec.get('ik_reference_mode', spec.get('reference_mode'))} "
            f"use_command_warm_start={spec.get('use_command_warm_start', True)} "
            f'tcp_error={tcp_error} tcp_orientation_error={tcp_orientation_error} '
            f'command_tcp_step={command_tcp_step} command_orientation_step={command_orientation_step} '
            f'ik_position_tolerance={ik_position_tolerance} '
            f'ik_orientation_tolerance={ik_orientation_tolerance} '
            f"target_tcp={np.asarray(target_pose['position'], dtype=float).tolist()} "
            f"command_tcp={np.asarray(command_target_pose['position'], dtype=float).tolist()} "
            f"ik_tcp={np.asarray(ik_target_pose['position'], dtype=float).tolist()} "
            f'current_q={None if current_q is None else current_q.tolist()} '
            f'reference_q={None if reference_q is None else reference_q.tolist()} '
            f'target_q={None if target_q is None else target_q.tolist()} '
            f'command_q={None if command_q is None else command_q.tolist()} '
            f'ref_to_target_max={_max_abs_delta(reference_q, target_q)} '
            f'ref_to_cmd_max={_max_abs_delta(reference_q, command_q)} '
            f'current_to_cmd_max={_max_abs_delta(current_q, command_q)} '
            f'current_to_ref_max={_max_abs_delta(current_q, reference_q)} '
            f"joint_velocity={arm_dynamics.get('joint_velocity')} "
            f"measured_effort={arm_dynamics.get('measured_effort')} "
            f"applied_effort={arm_dynamics.get('applied_effort')} "
            f"stiffness={arm_dynamics.get('stiffness')} "
            f"damping={arm_dynamics.get('damping')} "
            f"max_force={arm_dynamics.get('max_force')}",
            flush=True,
        )

    @staticmethod
    def _cartesian_pose_fraction(*, current_pose: dict, target_pose: dict, ratio: float) -> dict:
        current_position = np.asarray(current_pose['position'], dtype=float)
        target_position = np.asarray(target_pose['position'], dtype=float)
        position_distance = float(np.linalg.norm(target_position - current_position))
        current_orientation = normalize_quat(current_pose['orientation'])
        target_orientation = normalize_quat(target_pose['orientation'])
        orientation_dot = float(np.dot(current_orientation, target_orientation))
        orientation_angle = float(2.0 * math.acos(np.clip(abs(orientation_dot), 0.0, 1.0)))
        return UR5eAssemblyAtomicSkillAdapter._cartesian_servo_target_pose(
            current_pose=current_pose,
            target_pose=target_pose,
            max_position_step=position_distance * float(ratio),
            max_orientation_step=orientation_angle * float(ratio),
        )

    def _recover_ik_branch_jump(
        self,
        *,
        task,
        robot_name: str,
        skill_name: str,
        spec: dict,
        current_pose: dict,
        command_target_pose: dict,
        reference_q: np.ndarray,
        measured_q: np.ndarray | None,
    ) -> dict | None:
        candidates: list[tuple[str, dict]] = []
        for raw_ratio in spec.get(
            'ik_branch_backtrack_ratios',
            spec.get('ik_backtrack_ratios', (0.5, 0.25, 0.125)),
        ):
            try:
                ratio = float(raw_ratio)
            except (TypeError, ValueError):
                continue
            if not 0.0 < ratio < 1.0:
                continue
            candidates.append(
                (
                    f'cartesian_backtrack_{ratio:g}',
                    self._cartesian_pose_fraction(
                        current_pose=current_pose,
                        target_pose=command_target_pose,
                        ratio=ratio,
                    ),
                )
            )

        translation_only = bool(
            spec.get(
                'ik_branch_translation_only_recovery',
                skill_name == 'ur5e_move_above_part',
            )
        )
        translation_distance = float(
            np.linalg.norm(
                np.asarray(command_target_pose['position'], dtype=float)
                - np.asarray(current_pose['position'], dtype=float)
            )
        )
        minimum_translation = max(
            float(spec.get('ik_branch_translation_only_min_step', 1e-5)),
            0.0,
        )
        if translation_only and translation_distance > minimum_translation:
            candidates.append(
                (
                    'translation_only',
                    {
                        'position': np.asarray(command_target_pose['position'], dtype=float),
                        'orientation': normalize_quat(current_pose['orientation']),
                    },
                )
            )

        warm_starts = [('command', reference_q)]
        measured_q = self._coerce_arm_q(measured_q)
        if (
            measured_q is not None
            and measured_q.shape == reference_q.shape
            and float(np.max(np.abs(measured_q - reference_q))) > 1e-6
        ):
            warm_starts.append(('measured', measured_q))

        for mode, candidate_command_pose in candidates:
            candidate_ik_pose = self._ik_target_pose(
                target_pose=candidate_command_pose,
                spec=spec,
            )
            for warm_start_name, warm_start in warm_starts:
                candidate_result = self._solve_ik(
                    task=task,
                    robot_name=robot_name,
                    target_pose=candidate_ik_pose,
                    warm_start=warm_start,
                    spec=spec,
                )
                if candidate_result is None:
                    continue
                candidate_q = self._unwrap_to_reference(
                    target_q=candidate_result,
                    reference_q=reference_q,
                    preferred_abs_limit=spec.get('preferred_joint_abs_limit', 3.05),
                    hard_preferred_abs_limit=bool(spec.get('hard_preferred_joint_abs_limit', True)),
                )
                if self._ik_branch_jump_detected(
                    reference_q=reference_q,
                    target_q=candidate_q,
                    spec=spec,
                ):
                    continue
                return {
                    'ik_result': np.asarray(candidate_result, dtype=float),
                    'target_q': candidate_q,
                    'command_target_pose': candidate_command_pose,
                    'ik_target_pose': candidate_ik_pose,
                    'mode': f'{mode}_{warm_start_name}_warm_start',
                }
        return None

    @staticmethod
    def _orientation_first_servo_pose(
        *,
        skill_name: str,
        spec: dict,
        current_pose: dict,
        target_pose: dict,
        phase_step_counter: int = 0,
    ) -> dict | None:
        orientation_tolerance = spec.get('orientation_tolerance')
        enabled = bool(spec.get('orientation_first_before_translation', False))
        if not enabled:
            return None
        max_steps = max(int(spec.get('orientation_first_max_steps', 48)), 0)
        if int(phase_step_counter) >= max_steps:
            return None
        _, orientation_error = pose_error(
            current_position=current_pose['position'],
            current_orientation=current_pose['orientation'],
            target_position=target_pose['position'],
            target_orientation=target_pose['orientation'],
        )
        default_threshold = 0.035 if orientation_tolerance is None else max(float(orientation_tolerance) * 0.8, 0.035)
        threshold = max(float(spec.get('orientation_first_tolerance', default_threshold)), 0.0)
        if orientation_error is None or float(orientation_error) <= threshold:
            return None
        alignment_target = {
            'position': np.asarray(current_pose['position'], dtype=float),
            'orientation': np.asarray(target_pose['orientation'], dtype=float),
        }
        return UR5eAssemblyAtomicSkillAdapter._cartesian_servo_target_pose(
            current_pose=current_pose,
            target_pose=alignment_target,
            max_position_step=0.0,
            max_orientation_step=float(spec.get('cartesian_orientation_step', 0.01)),
        )

    @staticmethod
    def _cartesian_servo_target_pose(
        *,
        current_pose: dict,
        target_pose: dict,
        max_position_step: float,
        max_orientation_step: float,
    ) -> dict:
        current_position = np.asarray(current_pose['position'], dtype=float)
        target_position = np.asarray(target_pose['position'], dtype=float)
        delta = target_position - current_position
        distance = float(np.linalg.norm(delta))
        if max_position_step > 0.0 and distance > max_position_step:
            command_position = current_position + delta * (max_position_step / distance)
        else:
            command_position = target_position

        current_orientation = normalize_quat(current_pose['orientation'])
        target_orientation = normalize_quat(target_pose['orientation'])
        dot = float(np.dot(current_orientation, target_orientation))
        if dot < 0.0:
            target_orientation = -target_orientation
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        orientation_angle = float(2.0 * math.acos(dot))
        if max_orientation_step > 0.0 and orientation_angle > max_orientation_step:
            ratio = max_orientation_step / orientation_angle
            if dot > 0.9995:
                command_orientation = normalize_quat((1.0 - ratio) * current_orientation + ratio * target_orientation)
            else:
                half_angle = math.acos(dot)
                sin_half_angle = math.sin(half_angle)
                if abs(sin_half_angle) < 1e-8:
                    command_orientation = current_orientation
                else:
                    command_orientation = normalize_quat(
                        (math.sin((1.0 - ratio) * half_angle) / sin_half_angle) * current_orientation
                        + (math.sin(ratio * half_angle) / sin_half_angle) * target_orientation
                    )
        else:
            command_orientation = target_orientation
        return {
            'position': command_position,
            'orientation': command_orientation,
        }

    def _target_object_servo_pose(  # noqa: C901
        self,
        *,
        phase_key,
        task,
        robot_name: str,
        spec: dict,
        tracked_robots: dict,
        tracked_objects: dict,
        current_pose: dict,
        target_pose: dict,
    ) -> dict:
        max_position_step = float(spec.get('cartesian_position_step', 0.01))
        max_orientation_step = float(spec.get('cartesian_orientation_step', 0.01))
        default_object_servo = any(
            spec.get(name) is not None
            for name in ('target_object_position', 'target_object_target', 'target_object', 'object_target')
        )
        if not bool(spec.get('servo_target_object_pose', default_object_servo)):
            return self._cartesian_command_servo_target_pose(
                phase_key=phase_key,
                current_pose=current_pose,
                target_pose=target_pose,
                max_position_step=max_position_step,
                max_orientation_step=max_orientation_step,
                spec=spec,
            )

        object_name = self._object_name_from_spec(spec)
        object_pose = (
            None
            if object_name is None
            else self._object_pose(
                task=task,
                object_name=object_name,
                tracked_objects=tracked_objects,
            )
        )
        target_object_pose = self._target_object_pose(task=task, spec=spec)
        if object_pose is None or target_object_pose is None:
            return self._cartesian_command_servo_target_pose(
                phase_key=phase_key,
                current_pose=current_pose,
                target_pose=target_pose,
                max_position_step=max_position_step,
                max_orientation_step=max_orientation_step,
                spec=spec,
            )

        current_object_position = np.asarray(object_pose['position'], dtype=float)
        target_object_position = np.asarray(target_object_pose['position'], dtype=float)
        object_delta = target_object_position - current_object_position
        _, _, attachment_mode = self._compliant_attachment_allowance(
            object_name=object_name,
            tracked_objects=tracked_objects,
        )
        if attachment_mode == 'compliant_joint':
            motion_detail = self._object_motion_detail(
                task=task,
                object_name=object_name,
                tracked_objects=tracked_objects,
            )
            compliant_step_scale = self._compliant_servo_step_scale(
                spec=spec,
                motion_detail=motion_detail,
            )
            compliant_max_position_step = float(spec.get('compliant_servo_max_position_step', 0.0005))
            compliant_max_orientation_step = float(spec.get('compliant_servo_max_orientation_step', 0.002))
            if (
                not np.isfinite(compliant_max_position_step)
                or compliant_max_position_step <= 0.0
                or not np.isfinite(compliant_max_orientation_step)
                or compliant_max_orientation_step <= 0.0
            ):
                raise ValueError('Compliant-servo Cartesian step limits must be finite and positive.')
            max_position_step = (
                min(
                    max_position_step,
                    compliant_max_position_step,
                )
                * compliant_step_scale
            )
            max_orientation_step = (
                min(
                    max_orientation_step,
                    compliant_max_orientation_step,
                )
                * compliant_step_scale
            )
        orientation_step = self._cartesian_command_servo_target_pose(
            phase_key=phase_key,
            current_pose=current_pose,
            target_pose=target_pose,
            max_position_step=0.0,
            max_orientation_step=max_orientation_step,
            spec=spec,
        )
        compliant_tcp_orientation = None
        target_object_orientation = target_object_pose.get('orientation')
        if (
            attachment_mode == 'compliant_joint'
            and bool(spec.get('compliant_servo_track_object_orientation', False))
            and target_object_orientation is not None
            and object_pose.get('orientation') is not None
        ):
            orientation_deadband = float(
                spec.get(
                    'compliant_servo_orientation_correction_deadband',
                    0.005,
                )
            )
            object_orientation_tolerance = spec.get('target_object_orientation_tolerance')
            if object_orientation_tolerance is not None:
                object_orientation_tolerance = float(object_orientation_tolerance)
                activation_ratio = float(
                    spec.get(
                        'compliant_servo_orientation_tolerance_activation_ratio',
                        0.5,
                    )
                )
                if not np.isfinite(activation_ratio) or activation_ratio <= 0.0 or activation_ratio > 1.0:
                    raise ValueError('Compliant-servo orientation activation ratio must be ' 'finite and in (0, 1].')
                orientation_deadband = max(
                    orientation_deadband,
                    activation_ratio * object_orientation_tolerance,
                )
            if not np.isfinite(orientation_deadband) or orientation_deadband < 0.0 or orientation_deadband > math.pi:
                raise ValueError('Compliant-servo orientation correction deadband must be finite ' 'and in [0, pi].')
            _, object_orientation_error = pose_error(
                current_object_position,
                object_pose['orientation'],
                current_object_position,
                target_object_orientation,
            )
            compliant_tcp_orientation = normalize_quat(orientation_step['orientation'])
            object_orientation_correction = quat_multiply(
                normalize_quat(target_object_orientation),
                self._quat_conjugate(object_pose['orientation']),
            )
            if object_orientation_error is not None and object_orientation_error > orientation_deadband:
                compliant_tcp_orientation = normalize_quat(
                    quat_multiply(
                        object_orientation_correction,
                        current_pose['orientation'],
                    )
                )
                compliant_tcp_orientation = self._cartesian_servo_target_pose(
                    current_pose=current_pose,
                    target_pose={
                        'position': np.asarray(current_pose['position'], dtype=float),
                        'orientation': compliant_tcp_orientation,
                    },
                    max_position_step=0.0,
                    max_orientation_step=max_orientation_step,
                )['orientation']
        convergence_axis = spec.get('target_object_convergence_axis')
        lateral_tolerance = spec.get('target_object_lateral_position_tolerance')
        axial_delta = None
        if convergence_axis is not None and lateral_tolerance is not None:
            convergence_axis = np.asarray(convergence_axis, dtype=float)
            axis_norm = float(np.linalg.norm(convergence_axis))
            if convergence_axis.shape != (3,) or not np.isfinite(axis_norm) or axis_norm <= 1e-9:
                raise ValueError('target_object_convergence_axis must be a finite non-zero 3D vector.')
            convergence_axis = convergence_axis / axis_norm
            lateral_tolerance = float(lateral_tolerance)
            if not np.isfinite(lateral_tolerance) or lateral_tolerance < 0.0:
                raise ValueError('target_object_lateral_position_tolerance must be finite and non-negative.')
            alignment_enter_tolerance = float(
                spec.get(
                    'target_object_lateral_alignment_enter_tolerance',
                    lateral_tolerance,
                )
            )
            alignment_exit_tolerance = float(
                spec.get(
                    'target_object_lateral_alignment_exit_tolerance',
                    alignment_enter_tolerance,
                )
            )
            if (
                not np.isfinite(alignment_enter_tolerance)
                or alignment_enter_tolerance < 0.0
                or not np.isfinite(alignment_exit_tolerance)
                or alignment_exit_tolerance < alignment_enter_tolerance
            ):
                raise ValueError(
                    'target-object lateral alignment tolerances must be finite, '
                    'non-negative, and ordered enter <= exit.'
                )
            lateral_alignment_step = float(
                spec.get(
                    'target_object_lateral_alignment_cartesian_position_step',
                    max_position_step,
                )
            )
            if not np.isfinite(lateral_alignment_step) or lateral_alignment_step <= 0.0:
                raise ValueError(
                    'target_object_lateral_alignment_cartesian_position_step must be ' 'finite and positive.'
                )
            if attachment_mode == 'compliant_joint':
                compliant_max_lateral_step = float(spec.get('compliant_servo_max_lateral_step', 0.0005))
                if not np.isfinite(compliant_max_lateral_step) or compliant_max_lateral_step <= 0.0:
                    raise ValueError('compliant_servo_max_lateral_step must be finite and positive.')
                lateral_alignment_step = (
                    min(
                        lateral_alignment_step,
                        compliant_max_lateral_step,
                    )
                    * compliant_step_scale
                )
            axial_delta = float(np.dot(object_delta, convergence_axis))
            lateral_delta = object_delta - axial_delta * convergence_axis
            lateral_error = float(np.linalg.norm(lateral_delta))
            axial_recovery_step = None
            axial_recovery_deadband = 0.0
            axial_clearance = 0.0
            clearance_required = False
            clearance_ready = True
            if attachment_mode == 'compliant_joint':
                axial_recovery_step = spec.get('target_object_axial_recovery_cartesian_position_step')
                if axial_recovery_step is not None:
                    axial_recovery_step = float(axial_recovery_step)
                    axial_recovery_deadband = float(spec.get('target_object_axial_recovery_deadband', 0.0))
                    axial_clearance = float(
                        spec.get(
                            'target_object_lateral_alignment_axial_clearance',
                            0.0,
                        )
                    )
                    max_alignment_retraction = spec.get('compliant_servo_max_alignment_retraction')
                    if max_alignment_retraction is not None:
                        max_alignment_retraction = float(max_alignment_retraction)
                        insertion_path_depth = float(spec.get('target_object_insertion_path_depth', 0.0))
                        if (
                            not np.isfinite(max_alignment_retraction)
                            or max_alignment_retraction <= 0.0
                            or not np.isfinite(insertion_path_depth)
                            or insertion_path_depth < 0.0
                        ):
                            raise ValueError(
                                'Compliant alignment retraction must be finite and '
                                'positive, and insertion path depth must be finite '
                                'and non-negative.'
                            )
                    if (
                        not np.isfinite(axial_recovery_step)
                        or axial_recovery_step <= 0.0
                        or not np.isfinite(axial_recovery_deadband)
                        or axial_recovery_deadband < 0.0
                        or not np.isfinite(axial_clearance)
                        or axial_clearance < 0.0
                    ):
                        raise ValueError(
                            'target-object axial recovery step must be finite and '
                            'positive, and its deadband and lateral-alignment axial '
                            'clearance must be finite and non-negative.'
                        )
                    clearance_required = bool(
                        self._insertion_lateral_clearance_required.get(
                            phase_key,
                            False,
                        )
                        or (axial_clearance > 0.0 and lateral_error > alignment_exit_tolerance)
                    )
                    if clearance_required and max_alignment_retraction is not None:
                        axial_clearance = self._insertion_lateral_clearance_anchors.setdefault(
                            phase_key,
                            min(
                                axial_clearance,
                                axial_delta + max_alignment_retraction,
                            ),
                        )
                    elif not clearance_required:
                        self._insertion_lateral_clearance_anchors.pop(
                            phase_key,
                            None,
                        )
                    clearance_ready = bool(
                        not clearance_required or axial_delta >= axial_clearance - axial_recovery_deadband
                    )
            previous_alignment_active = self._insertion_lateral_alignment_active.get(
                phase_key,
                True,
            )
            alignment_active = previous_alignment_active
            required_alignment_stable_steps = max(
                int(spec.get('target_object_lateral_alignment_stable_steps', 1)),
                1,
            )
            if alignment_active:
                if lateral_error <= alignment_enter_tolerance and clearance_ready:
                    alignment_stable_steps = (
                        int(
                            self._insertion_lateral_alignment_stable_steps.get(
                                phase_key,
                                0,
                            )
                        )
                        + 1
                    )
                    self._insertion_lateral_alignment_stable_steps[phase_key] = alignment_stable_steps
                    alignment_active = alignment_stable_steps < required_alignment_stable_steps
                else:
                    self._insertion_lateral_alignment_stable_steps[phase_key] = 0
            else:
                alignment_active = lateral_error > alignment_exit_tolerance
                if alignment_active:
                    self._insertion_lateral_alignment_stable_steps[phase_key] = 0
            self._insertion_lateral_alignment_active[phase_key] = alignment_active
            if alignment_active != previous_alignment_active:
                self._cartesian_command_positions.pop(phase_key, None)
            if alignment_active and clearance_required:
                self._insertion_lateral_clearance_required[phase_key] = True
            else:
                self._insertion_lateral_clearance_required.pop(phase_key, None)
                self._insertion_lateral_clearance_anchors.pop(phase_key, None)
            if alignment_active:
                if attachment_mode == 'compliant_joint':
                    gate_object_position = current_object_position + lateral_delta
                    if lateral_error > lateral_alignment_step:
                        command_object_position = current_object_position + lateral_delta * (
                            lateral_alignment_step / lateral_error
                        )
                    else:
                        command_object_position = current_object_position + lateral_delta
                    if axial_recovery_step is not None:
                        required_recovery = axial_clearance - axial_delta if clearance_required else -axial_delta
                        if required_recovery > axial_recovery_deadband:
                            gate_object_position = gate_object_position - required_recovery * convergence_axis
                            recovery_distance = min(
                                required_recovery,
                                axial_recovery_step * compliant_step_scale,
                            )
                            command_object_position = command_object_position - recovery_distance * convergence_axis
                    if bool(
                        spec.get(
                            'compliant_servo_hold_orientation_during_lateral_alignment',
                            True,
                        )
                    ):
                        orientation_anchor = self._insertion_lateral_orientation_anchors.setdefault(
                            phase_key,
                            normalize_quat(current_pose['orientation']),
                        )
                        command_orientation = self._cartesian_servo_target_pose(
                            current_pose=current_pose,
                            target_pose={
                                'position': np.asarray(
                                    current_pose['position'],
                                    dtype=float,
                                ),
                                'orientation': orientation_anchor,
                            },
                            max_position_step=0.0,
                            max_orientation_step=max_orientation_step,
                        )['orientation']
                    else:
                        command_orientation = (
                            compliant_tcp_orientation
                            if compliant_tcp_orientation is not None
                            else normalize_quat(orientation_step['orientation'])
                        )
                    relative_world = current_object_position - np.asarray(
                        current_pose['position'],
                        dtype=float,
                    )
                    relative_position = quat_rotate(
                        self._quat_conjugate(current_pose['orientation']),
                        relative_world,
                    )
                    if relative_position.shape == (3,) and np.all(np.isfinite(relative_position)):
                        position_servo_orientation = command_orientation
                        if bool(
                            spec.get(
                                'target_object_use_measured_orientation_for_position_servo',
                                True,
                            )
                        ):
                            position_servo_orientation = normalize_quat(current_pose['orientation'])
                        one_step_tcp_position = command_object_position - quat_rotate(
                            position_servo_orientation,
                            relative_position,
                        )
                        gate_tcp_position = gate_object_position - quat_rotate(
                            position_servo_orientation,
                            relative_position,
                        )
                        one_step_tcp_position = self._accumulated_cartesian_command_position(
                            phase_key=phase_key,
                            current_position=np.asarray(current_pose['position'], dtype=float),
                            one_step_position=one_step_tcp_position,
                            gate_position=gate_tcp_position,
                            enabled=bool(
                                spec.get(
                                    'compliant_servo_position_command_warm_start',
                                    True,
                                )
                            ),
                            allow_gate_overdrive=bool(
                                spec.get(
                                    'compliant_servo_position_command_gate_overdrive',
                                    True,
                                )
                            ),
                            accumulation_step=float(
                                spec.get(
                                    'compliant_servo_position_command_accumulation_step',
                                    0.0001,
                                )
                            ),
                            lookahead=float(
                                spec.get(
                                    'compliant_servo_position_command_lookahead',
                                    0.004,
                                )
                            ),
                        )
                        return {
                            'position': one_step_tcp_position,
                            'orientation': command_orientation,
                        }
                # Shift the stable final TCP target back to the current insertion depth.
                # Rebuilding this target from the live, loaded object/TCP transform can
                # reverse the lateral correction when the grasp has a large lever arm.
                current_axial_position = float(np.dot(current_object_position, convergence_axis))
                axial_anchor = self._insertion_axial_anchors.setdefault(
                    phase_key,
                    current_axial_position,
                )
                target_axial_position = float(np.dot(target_object_position, convergence_axis))
                gate_target_pose = {
                    'position': np.asarray(target_pose['position'], dtype=float)
                    + (axial_anchor - target_axial_position) * convergence_axis,
                    'orientation': normalize_quat(target_pose['orientation']),
                }
                command_pose = self._cartesian_command_servo_target_pose(
                    phase_key=phase_key,
                    current_pose=current_pose,
                    target_pose=gate_target_pose,
                    max_position_step=lateral_alignment_step,
                    max_orientation_step=max_orientation_step,
                    spec=spec,
                )
                command_pose['position'] = self._accumulated_cartesian_command_position(
                    phase_key=phase_key,
                    current_position=np.asarray(current_pose['position'], dtype=float),
                    one_step_position=np.asarray(command_pose['position'], dtype=float),
                    gate_position=np.asarray(gate_target_pose['position'], dtype=float),
                    enabled=bool(
                        spec.get(
                            'target_object_servo_position_command_warm_start',
                            False,
                        )
                    ),
                    lookahead=float(
                        spec.get(
                            'target_object_servo_position_command_lookahead',
                            0.004,
                        )
                    ),
                    allow_gate_overdrive=bool(
                        spec.get(
                            'target_object_servo_position_command_gate_overdrive',
                            True,
                        )
                    ),
                    accumulation_step=float(
                        spec.get(
                            'target_object_servo_position_command_accumulation_step',
                            0.0001,
                        )
                    ),
                )
                return command_pose
            else:
                self._insertion_axial_anchors.pop(phase_key, None)
                self._insertion_lateral_alignment_stable_steps.pop(phase_key, None)
                self._insertion_lateral_orientation_anchors.pop(phase_key, None)
                self._insertion_lateral_clearance_anchors.pop(phase_key, None)
        effective_position_step = max_position_step
        if axial_delta is not None and spec.get('target_object_axial_recovery_cartesian_position_step') is not None:
            axial_recovery_step = float(spec['target_object_axial_recovery_cartesian_position_step'])
            axial_recovery_deadband = float(spec.get('target_object_axial_recovery_deadband', 0.0))
            if (
                not np.isfinite(axial_recovery_step)
                or axial_recovery_step <= 0.0
                or not np.isfinite(axial_recovery_deadband)
                or axial_recovery_deadband < 0.0
            ):
                raise ValueError(
                    'target-object axial recovery step must be finite and positive, '
                    'and its deadband must be finite and non-negative.'
                )
            if axial_delta < -axial_recovery_deadband:
                effective_position_step = max(
                    effective_position_step,
                    axial_recovery_step,
                )
        object_distance = float(np.linalg.norm(object_delta))
        if attachment_mode == 'compliant_joint' and axial_delta is not None:
            axial_step = float(
                np.clip(
                    axial_delta,
                    -effective_position_step,
                    effective_position_step,
                )
            )
            if lateral_error > lateral_alignment_step:
                lateral_step = lateral_delta * (lateral_alignment_step / lateral_error)
            else:
                lateral_step = lateral_delta
            command_object_position = current_object_position + axial_step * convergence_axis + lateral_step
        elif effective_position_step > 0.0 and object_distance > effective_position_step:
            command_object_position = current_object_position + object_delta * (
                effective_position_step / object_distance
            )
        else:
            command_object_position = current_object_position + object_delta
        command_orientation = (
            compliant_tcp_orientation
            if compliant_tcp_orientation is not None
            else normalize_quat(orientation_step['orientation'])
        )
        position_servo_orientation = command_orientation
        if bool(
            spec.get(
                'target_object_use_measured_orientation_for_position_servo',
                attachment_mode == 'compliant_joint',
            )
        ):
            position_servo_orientation = normalize_quat(current_pose['orientation'])
        relative_world = current_object_position - np.asarray(current_pose['position'], dtype=float)
        relative_position = quat_rotate(
            self._quat_conjugate(current_pose['orientation']),
            relative_world,
        )
        if relative_position.shape != (3,) or not np.all(np.isfinite(relative_position)):
            return self._cartesian_command_servo_target_pose(
                phase_key=phase_key,
                current_pose=current_pose,
                target_pose=target_pose,
                max_position_step=max_position_step,
                max_orientation_step=max_orientation_step,
                spec=spec,
            )
        one_step_tcp_position = command_object_position - quat_rotate(
            position_servo_orientation,
            relative_position,
        )
        gate_tcp_position = target_object_position - quat_rotate(
            position_servo_orientation,
            relative_position,
        )
        one_step_tcp_position = self._accumulated_cartesian_command_position(
            phase_key=phase_key,
            current_position=np.asarray(current_pose['position'], dtype=float),
            one_step_position=one_step_tcp_position,
            gate_position=gate_tcp_position,
            enabled=bool(
                spec.get(
                    'target_object_servo_position_command_warm_start',
                    False,
                )
            ),
            lookahead=float(
                spec.get(
                    'target_object_servo_position_command_lookahead',
                    0.004,
                )
            ),
            allow_gate_overdrive=bool(
                spec.get(
                    'target_object_servo_position_command_gate_overdrive',
                    True,
                )
            ),
            accumulation_step=float(
                spec.get(
                    'target_object_servo_position_command_accumulation_step',
                    0.0001,
                )
            ),
        )
        return {
            'position': one_step_tcp_position,
            'orientation': command_orientation,
        }

    @staticmethod
    def _compliant_servo_step_scale(
        *,
        spec: dict,
        motion_detail: dict[str, Any],
    ) -> float:
        if not bool(spec.get('compliant_servo_velocity_rate_limit', True)):
            return 1.0
        minimum_scale = float(spec.get('compliant_servo_minimum_step_scale', 0.2))
        if not np.isfinite(minimum_scale) or not 0.0 < minimum_scale <= 1.0:
            raise ValueError('compliant_servo_minimum_step_scale must be finite and in (0, 1].')
        if not motion_detail.get('valid'):
            return 1.0
        pause_linear_speed = float(spec.get('compliant_servo_pause_linear_speed', 0.15))
        pause_angular_speed = float(spec.get('compliant_servo_pause_angular_speed', 5.0))
        if (
            not np.isfinite(pause_linear_speed)
            or pause_linear_speed <= 0.0
            or not np.isfinite(pause_angular_speed)
            or pause_angular_speed <= 0.0
        ):
            raise ValueError('Compliant-servo pause speeds must be finite and positive.')
        speed_ratio = max(
            float(motion_detail['linear_speed']) / pause_linear_speed,
            float(motion_detail['angular_speed']) / pause_angular_speed,
        )
        return float(np.clip(1.0 - speed_ratio, minimum_scale, 1.0))

    def _target_object_settle_ready(
        self,
        *,
        phase_key: tuple[Any, ...] | None = None,
        task,
        spec: dict,
        target_pose: dict,
        current_pose: dict | None,
        tracked_objects: dict,
    ) -> bool:
        detail = self._target_object_settle_capture_detail(
            task=task,
            spec=spec,
            target_pose=target_pose,
            current_pose=current_pose,
            tracked_objects=tracked_objects,
        )
        if phase_key is None or not bool(detail['configured']):
            return bool(detail['capture_ready'])

        hold_steps = int(spec.get('target_object_settle_hold_steps', 48))
        retry_servo_steps = int(spec.get('target_object_settle_retry_servo_steps', 8))
        if hold_steps <= 0 or retry_servo_steps < 0:
            raise ValueError(
                'target-object settle hold steps must be positive and retry servo ' 'steps must be non-negative.'
            )
        state = self._target_object_settle_state.setdefault(phase_key, {})
        event = 'idle'
        hold_active = False
        if bool(state.get('hold_active', False)):
            hold_active = True
            remaining_steps = max(int(state.get('remaining_steps', 1)) - 1, 0)
            state['remaining_steps'] = remaining_steps
            event = 'hold'
            if remaining_steps == 0:
                state['hold_active'] = False
                state['retry_servo_steps'] = retry_servo_steps
                event = 'hold_complete'
        elif int(state.get('retry_servo_steps', 0)) > 0:
            state['retry_servo_steps'] = int(state['retry_servo_steps']) - 1
            event = 'retry_servo'
        elif bool(detail['capture_ready']):
            hold_active = True
            state['remaining_steps'] = hold_steps - 1
            state['hold_active'] = bool(state['remaining_steps'] > 0)
            if not state['hold_active']:
                state['retry_servo_steps'] = retry_servo_steps
            state['capture_count'] = int(state.get('capture_count', 0)) + 1
            event = 'capture'

        self._debug_target_object_settle(
            task=task,
            detail=detail,
            state=state,
            event=event,
            hold_active=hold_active,
            tracked_objects=tracked_objects,
        )
        return hold_active

    def _maybe_relax_insertion_attachment(
        self,
        *,
        task,
        spec: dict,
        tracked_objects: dict,
    ) -> bool:
        """Return whether motion should hold while switching to compliance."""

        final_target_name = spec.get('target_object_final_target')
        position_tolerance = spec.get('relax_fixed_attachment_within_final_position_tolerance')
        if final_target_name is None or position_tolerance is None:
            return False
        object_name = self._object_name_from_spec(spec)
        if object_name is None:
            return False
        relaxation_key = (id(task), object_name)
        locked_linear_world_direction = None
        compliance_mode = 'isotropic'
        minimum_gravity_alignment = float(spec.get('relax_fixed_attachment_minimum_gravity_alignment', 0.0))
        if not np.isfinite(minimum_gravity_alignment) or not 0.0 <= minimum_gravity_alignment <= 1.0:
            raise ValueError('relax_fixed_attachment_minimum_gravity_alignment must be ' 'finite and in [0, 1].')
        if minimum_gravity_alignment > 0.0:
            convergence_axis = np.asarray(
                spec.get('target_object_convergence_axis'),
                dtype=float,
            )
            gravity_direction = np.asarray(
                spec.get('relax_fixed_attachment_gravity_direction', [0.0, 0.0, -1.0]),
                dtype=float,
            )
            convergence_norm = float(np.linalg.norm(convergence_axis))
            gravity_norm = float(np.linalg.norm(gravity_direction))
            if (
                convergence_axis.shape != (3,)
                or gravity_direction.shape != (3,)
                or not np.isfinite(convergence_norm)
                or not np.isfinite(gravity_norm)
                or convergence_norm <= 1e-9
                or gravity_norm <= 1e-9
            ):
                raise ValueError(
                    'Gravity-gated insertion compliance requires finite non-zero ' 'convergence and gravity directions.'
                )
            gravity_alignment = abs(
                float(
                    np.dot(
                        convergence_axis / convergence_norm,
                        gravity_direction / gravity_norm,
                    )
                )
            )
            if gravity_alignment < minimum_gravity_alignment:
                locked_linear_world_direction = (gravity_direction / gravity_norm).tolist()
                compliance_mode = 'gravity_locked'
        if relaxation_key in self._physically_relaxed_insertion_objects:
            return False
        relax_after_steps = int(spec.get('relax_fixed_attachment_after_steps', 0))
        if relax_after_steps < 0:
            raise ValueError('relax_fixed_attachment_after_steps must be non-negative.')
        if int(getattr(task, 'phase_step_counter', 0)) < relax_after_steps:
            self._insertion_compliance_transition_state.pop(relaxation_key, None)
            return False

        object_pose = self._object_pose(
            task=task,
            object_name=object_name,
            tracked_objects=tracked_objects,
        )
        target_poses = getattr(task, 'target_poses', {})
        final_target_pose = target_poses.get(str(final_target_name))
        if object_pose is None or final_target_pose is None:
            return False
        position_error, final_orientation_error = pose_error(
            object_pose['position'],
            object_pose['orientation'],
            final_target_pose['position'],
            final_target_pose['orientation'],
        )
        position_tolerance = float(position_tolerance)
        orientation_tolerance = float(spec.get('relax_fixed_attachment_final_orientation_tolerance', 0.15))
        orientation_error = final_orientation_error
        waypoint_position_error = None
        waypoint_position_tolerance = None
        waypoint_axial_position_error = None
        waypoint_axial_position_tolerance = None
        waypoint_lateral_position_error = None
        waypoint_lateral_position_tolerance = None
        waypoint_proximity_ready = True
        if bool(spec.get('relax_fixed_attachment_require_waypoint_proximity', False)):
            waypoint_pose = self._target_object_pose(task=task, spec=spec)
            if waypoint_pose is None:
                self._insertion_compliance_transition_state.pop(relaxation_key, None)
                return False
            waypoint_position_error, orientation_error = pose_error(
                object_pose['position'],
                object_pose['orientation'],
                waypoint_pose['position'],
                waypoint_pose.get('orientation'),
            )
            waypoint_position_tolerance = float(spec.get('relax_fixed_attachment_waypoint_position_tolerance', 0.010))
            if not np.isfinite(waypoint_position_tolerance) or waypoint_position_tolerance <= 0.0:
                raise ValueError('relax_fixed_attachment_waypoint_position_tolerance must be ' 'finite and positive.')
            split_axial_tolerance = spec.get('relax_fixed_attachment_waypoint_axial_position_tolerance')
            split_lateral_tolerance = spec.get('relax_fixed_attachment_waypoint_lateral_position_tolerance')
            strict_waypoint_proximity_ready = bool(waypoint_position_error <= waypoint_position_tolerance)
            if split_axial_tolerance is not None or split_lateral_tolerance is not None:
                if split_axial_tolerance is None or split_lateral_tolerance is None:
                    raise ValueError('Waypoint axial and lateral capture tolerances must be ' 'configured together.')
                convergence_axis = np.asarray(
                    spec.get('target_object_convergence_axis'),
                    dtype=float,
                )
                axis_norm = float(np.linalg.norm(convergence_axis))
                waypoint_axial_position_tolerance = float(split_axial_tolerance)
                waypoint_lateral_position_tolerance = float(split_lateral_tolerance)
                if (
                    convergence_axis.shape != (3,)
                    or not np.isfinite(axis_norm)
                    or axis_norm <= 1e-9
                    or not np.isfinite(waypoint_axial_position_tolerance)
                    or waypoint_axial_position_tolerance <= 0.0
                    or not np.isfinite(waypoint_lateral_position_tolerance)
                    or waypoint_lateral_position_tolerance <= 0.0
                ):
                    raise ValueError(
                        'Waypoint split capture requires a finite non-zero convergence '
                        'axis and finite positive axial/lateral tolerances.'
                    )
                convergence_axis = convergence_axis / axis_norm
                waypoint_delta = np.asarray(
                    waypoint_pose['position'],
                    dtype=float,
                ) - np.asarray(object_pose['position'], dtype=float)
                signed_axial_error = float(np.dot(waypoint_delta, convergence_axis))
                waypoint_axial_position_error = abs(signed_axial_error)
                waypoint_lateral_position_error = float(
                    np.linalg.norm(waypoint_delta - signed_axial_error * convergence_axis)
                )
                geometric_capture_after_steps = int(
                    spec.get(
                        'relax_fixed_attachment_geometric_capture_after_steps',
                        0,
                    )
                )
                if geometric_capture_after_steps < 0:
                    raise ValueError('Geometric waypoint capture delay must be non-negative.')
                split_waypoint_proximity_ready = bool(
                    waypoint_axial_position_error <= waypoint_axial_position_tolerance
                    and waypoint_lateral_position_error <= waypoint_lateral_position_tolerance
                )
                waypoint_proximity_ready = bool(
                    split_waypoint_proximity_ready
                    and (
                        strict_waypoint_proximity_ready
                        or int(getattr(task, 'phase_step_counter', 0)) >= geometric_capture_after_steps
                    )
                )
            else:
                waypoint_proximity_ready = strict_waypoint_proximity_ready
        if (
            position_error > position_tolerance
            or not waypoint_proximity_ready
            or orientation_error is None
            or orientation_error > orientation_tolerance
        ):
            self._insertion_compliance_transition_state.pop(relaxation_key, None)
            return False

        motion_detail = self._object_motion_detail(
            task=task,
            object_name=object_name,
            tracked_objects=tracked_objects,
        )
        max_linear_speed = float(
            spec.get(
                'relax_fixed_attachment_max_linear_speed',
                spec.get('target_object_max_linear_speed', 0.03),
            )
        )
        max_angular_speed = float(
            spec.get(
                'relax_fixed_attachment_max_angular_speed',
                spec.get('target_object_max_angular_speed', 2.0),
            )
        )
        transition_state = self._insertion_compliance_transition_state.setdefault(
            relaxation_key,
            {'stable_steps': 0},
        )
        velocity_ready = bool(
            motion_detail.get('valid')
            and motion_detail.get('linear_speed') is not None
            and motion_detail.get('angular_speed') is not None
            and float(motion_detail['linear_speed']) <= max_linear_speed
            and float(motion_detail['angular_speed']) <= max_angular_speed
        )
        pose_stable_ready = bool(
            spec.get('relax_fixed_attachment_allow_pose_stable_override', True)
            and motion_detail.get('is_static') is True
            and motion_detail.get('pose_stable_override') is True
            and motion_detail.get('linear_speed') is not None
            and motion_detail.get('angular_speed') is not None
            and float(motion_detail['linear_speed']) <= float(spec.get('compliant_servo_pause_linear_speed', 0.15))
            and float(motion_detail['angular_speed']) <= float(spec.get('compliant_servo_pause_angular_speed', 5.0))
        )
        pose_history_velocity_override_ready = self._pose_history_velocity_override_ready(
            spec=spec,
            motion_detail=motion_detail,
        )
        motion_ready = bool(velocity_ready or pose_stable_ready or pose_history_velocity_override_ready)
        if motion_ready:
            transition_state['stable_steps'] = int(transition_state.get('stable_steps', 0)) + 1
        else:
            transition_state['stable_steps'] = 0
        required_stable_steps = max(
            int(spec.get('relax_fixed_attachment_stable_steps', 8)),
            1,
        )
        if int(transition_state['stable_steps']) < required_stable_steps:
            return True

        relax_fn = getattr(task, 'relax_fixed_attachment_to_physical_hold', None)
        if not callable(relax_fn) or not bool(
            relax_fn(
                object_name,
                locked_linear_world_direction=locked_linear_world_direction,
            )
        ):
            return True

        self._physically_relaxed_insertion_objects.add(relaxation_key)
        self._insertion_compliance_transition_state.pop(relaxation_key, None)
        print(
            '[ur5e-insertion-compliance] '
            f"step={getattr(task, 'step_counter', None)} "
            f"phase={getattr(task, 'phase', None)} object={object_name} "
            f'final_target={final_target_name} '
            f'position_error={position_error} position_tolerance={position_tolerance} '
            f'waypoint_position_error={waypoint_position_error} '
            f'waypoint_position_tolerance={waypoint_position_tolerance} '
            f'waypoint_axial_position_error={waypoint_axial_position_error} '
            f'waypoint_axial_position_tolerance={waypoint_axial_position_tolerance} '
            f'waypoint_lateral_position_error={waypoint_lateral_position_error} '
            f'waypoint_lateral_position_tolerance={waypoint_lateral_position_tolerance} '
            f'orientation_error={orientation_error} '
            f'orientation_tolerance={orientation_tolerance} '
            f'stable_steps={required_stable_steps} mode={compliance_mode}_compliant_joint ',
            f'locked_linear_world_direction={locked_linear_world_direction}',
            flush=True,
        )
        return True

    def _compliant_motion_requires_hold(
        self,
        *,
        phase_key: tuple[Any, ...] | None = None,
        task,
        spec: dict,
        tracked_objects: dict,
    ) -> bool:
        object_name = self._object_name_from_spec(spec)
        if object_name is None:
            return False
        _, _, attachment_mode = self._compliant_attachment_allowance(
            object_name=object_name,
            tracked_objects=tracked_objects,
        )
        recovery_key = (id(task), object_name)
        if attachment_mode != 'compliant_joint':
            self._compliant_motion_recovery_state.pop(recovery_key, None)
            return False

        motion_detail = self._object_motion_detail(
            task=task,
            object_name=object_name,
            tracked_objects=tracked_objects,
        )
        if not motion_detail.get('valid'):
            return False
        linear_speed = float(motion_detail['linear_speed'])
        angular_speed = float(motion_detail['angular_speed'])
        pause_linear_speed = float(spec.get('compliant_servo_pause_linear_speed', 0.15))
        pause_angular_speed = float(spec.get('compliant_servo_pause_angular_speed', 5.0))
        resume_linear_speed = float(
            spec.get(
                'compliant_servo_resume_linear_speed',
                spec.get('target_object_max_linear_speed', 0.03),
            )
        )
        resume_angular_speed = float(
            spec.get(
                'compliant_servo_resume_angular_speed',
                spec.get('target_object_max_angular_speed', 2.0),
            )
        )
        required_stable_steps = max(
            int(spec.get('compliant_servo_resume_stable_steps', 8)),
            1,
        )
        state = self._compliant_motion_recovery_state.setdefault(
            recovery_key,
            {'active': False, 'stable_steps': 0},
        )
        pose_stable_ready = bool(
            spec.get('compliant_servo_allow_pose_stable_resume', True)
            and motion_detail.get('is_static') is True
            and motion_detail.get('pose_stable_override') is True
            and linear_speed <= resume_linear_speed
            and angular_speed <= resume_angular_speed
        )
        pose_history_velocity_override_ready = self._pose_history_velocity_override_ready(
            spec=spec,
            motion_detail=motion_detail,
        )
        if not bool(state.get('active', False)):
            if (
                (linear_speed <= pause_linear_speed and angular_speed <= pause_angular_speed)
                or pose_stable_ready
                or pose_history_velocity_override_ready
            ):
                return False
            state['active'] = True
            state['stable_steps'] = 0
            state['settle_stable_steps'] = 0
            if phase_key is not None:
                self._cartesian_command_positions.pop(phase_key, None)
                self._cartesian_command_orientations.pop(phase_key, None)
            print(
                '[ur5e-compliant-recovery] '
                f"event=enter step={getattr(task, 'step_counter', None)} "
                f"phase={getattr(task, 'phase', None)} object={object_name} "
                f'linear_speed={linear_speed} angular_speed={angular_speed}',
                flush=True,
            )
            return True

        velocity_ready = bool(linear_speed <= resume_linear_speed and angular_speed <= resume_angular_speed)
        motion_ready = bool(velocity_ready or pose_stable_ready or pose_history_velocity_override_ready)
        if motion_ready:
            state['stable_steps'] = int(state.get('stable_steps', 0)) + 1
        else:
            state['stable_steps'] = 0
        if int(state['stable_steps']) < required_stable_steps:
            return True

        self._compliant_motion_recovery_state.pop(recovery_key, None)
        print(
            '[ur5e-compliant-recovery] '
            f"event=resume step={getattr(task, 'step_counter', None)} "
            f"phase={getattr(task, 'phase', None)} object={object_name} "
            f'linear_speed={linear_speed} angular_speed={angular_speed} '
            f'stable_steps={required_stable_steps}',
            flush=True,
        )
        return False

    def _compliant_recovery_allows_target_settle(
        self,
        *,
        task,
        spec: dict,
        tracked_objects: dict,
    ) -> bool:
        object_name = self._object_name_from_spec(spec)
        if object_name is None:
            return False
        state = self._compliant_motion_recovery_state.get((id(task), object_name))
        if state is None or not bool(state.get('active', False)):
            return False
        motion_detail = self._object_motion_detail(
            task=task,
            object_name=object_name,
            tracked_objects=tracked_objects,
        )
        max_linear_speed = float(
            spec.get(
                'compliant_servo_settle_max_linear_speed',
                spec.get('relax_fixed_attachment_max_linear_speed', 0.10),
            )
        )
        max_angular_speed = float(
            spec.get(
                'compliant_servo_settle_max_angular_speed',
                spec.get('target_object_max_angular_speed', 2.0),
            )
        )
        settle_ready = bool(
            motion_detail.get('valid')
            and motion_detail.get('pose_stable_override') is True
            and motion_detail.get('linear_speed') is not None
            and motion_detail.get('angular_speed') is not None
            and float(motion_detail['linear_speed']) <= max_linear_speed
            and float(motion_detail['angular_speed']) <= max_angular_speed
        )
        settle_ready = bool(
            settle_ready
            or self._pose_history_velocity_override_ready(
                spec=spec,
                motion_detail=motion_detail,
            )
        )
        if settle_ready:
            state['settle_stable_steps'] = int(state.get('settle_stable_steps', 0)) + 1
        else:
            state['settle_stable_steps'] = 0
            state['settle_ready_logged'] = False
        required_steps = max(
            int(spec.get('compliant_servo_settle_stable_steps', 24)),
            1,
        )
        ready = int(state['settle_stable_steps']) >= required_steps
        maximum_target_settle_steps = max(
            int(
                spec.get(
                    'compliant_recovery_target_settle_max_steps',
                    spec.get('target_object_settle_hold_steps', 48),
                )
            ),
            1,
        )
        if ready:
            state['target_settle_steps'] = int(state.get('target_settle_steps', 0)) + 1
        else:
            state['target_settle_steps'] = 0
        if ready and not bool(state.get('settle_ready_logged', False)):
            state['settle_ready_logged'] = True
            print(
                '[ur5e-compliant-recovery] '
                f"event=settle_ready step={getattr(task, 'step_counter', None)} "
                f"phase={getattr(task, 'phase', None)} object={object_name} "
                f"linear_speed={motion_detail.get('linear_speed')} "
                f"angular_speed={motion_detail.get('angular_speed')} "
                f'stable_steps={required_steps}',
                flush=True,
            )
        if ready and int(state['target_settle_steps']) > maximum_target_settle_steps:
            self._compliant_motion_recovery_state.pop(
                (id(task), object_name),
                None,
            )
            print(
                '[ur5e-compliant-recovery] '
                f"event=retry_servo step={getattr(task, 'step_counter', None)} "
                f"phase={getattr(task, 'phase', None)} object={object_name} "
                f'target_settle_steps={maximum_target_settle_steps}',
                flush=True,
            )
            return False
        return ready

    def _held_transport_action(self, *, task, robot_name: str, spec: dict) -> OrderedDict:
        action = self._hold_joint_action(task=task, robot_name=robot_name)
        gripper_command = spec.get('gripper_command', 'contact_hold')
        action[_GRIPPER_CONTROLLER] = [
            self._gripper_command_value(
                task=task,
                robot_name=robot_name,
                command=gripper_command,
            )
        ]
        return action

    def _target_object_settle_capture_detail(
        self,
        *,
        task,
        spec: dict,
        target_pose: dict,
        current_pose: dict | None,
        tracked_objects: dict,
    ) -> dict[str, Any]:
        detail: dict[str, Any] = {
            'configured': bool(
                spec.get('hold_for_target_object_settle', False) and spec.get('require_target_object_static', False)
            ),
            'capture_ready': False,
            'tcp_position_ready': False,
            'tcp_orientation_ready': False,
            'object_position_ready': False,
            'object_orientation_ready': False,
            'lateral_ready': False,
        }
        if not detail['configured'] or current_pose is None:
            return detail
        object_name = self._object_name_from_spec(spec)
        if object_name is None:
            return detail
        object_pose = self._object_pose(
            task=task,
            object_name=object_name,
            tracked_objects=tracked_objects,
        )
        target_object_pose = self._target_object_pose(task=task, spec=spec)
        if object_pose is None or target_object_pose is None:
            return detail
        relaxed_active = bool(
            spec.get('relaxed_position_tolerance') is not None
            and spec.get('relaxed_position_tolerance_after_steps') is not None
            and int(getattr(task, 'phase_step_counter', 0)) >= int(spec['relaxed_position_tolerance_after_steps'])
        )
        tcp_tolerance = float(spec.get('position_tolerance', 0.025))
        object_tolerance = float(spec.get('target_object_position_tolerance', tcp_tolerance))
        if relaxed_active:
            tcp_tolerance = max(
                tcp_tolerance,
                float(spec['relaxed_position_tolerance']),
            )
            if spec.get('relaxed_target_object_position_tolerance') is not None:
                object_tolerance = max(
                    object_tolerance,
                    float(spec['relaxed_target_object_position_tolerance']),
                )

        compliant_linear_allowance, compliant_angular_tolerance, attachment_mode = self._compliant_attachment_allowance(
            object_name=object_name,
            tracked_objects=tracked_objects,
        )
        tcp_tolerance += compliant_linear_allowance

        tcp_position_error, tcp_orientation_error = pose_error(
            current_pose['position'],
            current_pose['orientation'],
            target_pose['position'],
            target_pose['orientation'],
        )
        orientation_tolerance = spec.get('orientation_tolerance')
        if orientation_tolerance is not None:
            orientation_tolerance = float(orientation_tolerance) + compliant_angular_tolerance
        compliant_object_gated = bool(
            attachment_mode == 'compliant_joint' and spec.get('require_target_object_pose_convergence', False)
        )
        tcp_position_ready = bool(compliant_object_gated or tcp_position_error <= tcp_tolerance)
        tcp_orientation_ready = bool(
            compliant_object_gated
            or orientation_tolerance is None
            or tcp_orientation_error is None
            or tcp_orientation_error <= float(orientation_tolerance)
        )

        target_object_orientation = target_object_pose.get('orientation')
        if target_object_orientation is None:
            object_position_error = float(
                np.linalg.norm(
                    np.asarray(object_pose['position'], dtype=float)
                    - np.asarray(target_object_pose['position'], dtype=float)
                )
            )
            object_orientation_error = None
        else:
            object_position_error, object_orientation_error = pose_error(
                object_pose['position'],
                object_pose['orientation'],
                target_object_pose['position'],
                target_object_orientation,
            )
        object_orientation_tolerance = spec.get(
            'target_object_orientation_tolerance',
            orientation_tolerance,
        )
        object_position_ready = bool(object_position_error <= object_tolerance)
        object_orientation_ready = bool(
            object_orientation_tolerance is None
            or object_orientation_error is None
            or object_orientation_error <= float(object_orientation_tolerance)
        )

        convergence_axis = spec.get('target_object_convergence_axis')
        lateral_tolerance = spec.get('target_object_lateral_position_tolerance')
        lateral_error = None
        lateral_ready = True
        if convergence_axis is not None and lateral_tolerance is not None:
            convergence_axis = np.asarray(convergence_axis, dtype=float)
            axis_norm = float(np.linalg.norm(convergence_axis))
            if convergence_axis.shape != (3,) or not np.isfinite(axis_norm) or axis_norm <= 1e-9:
                lateral_ready = False
            else:
                convergence_axis = convergence_axis / axis_norm
                object_delta = np.asarray(object_pose['position'], dtype=float) - np.asarray(
                    target_object_pose['position'],
                    dtype=float,
                )
                axial_delta = float(np.dot(object_delta, convergence_axis))
                lateral_error = float(np.linalg.norm(object_delta - axial_delta * convergence_axis))
                lateral_ready = bool(lateral_error <= float(lateral_tolerance))

        detail.update(
            {
                'object_name': object_name,
                'attachment_mode': attachment_mode,
                'tcp_pose_required_for_capture': not compliant_object_gated,
                'relaxed_active': relaxed_active,
                'tcp_position_error': tcp_position_error,
                'tcp_position_tolerance': tcp_tolerance,
                'tcp_orientation_error': tcp_orientation_error,
                'tcp_orientation_tolerance': orientation_tolerance,
                'tcp_position_ready': tcp_position_ready,
                'tcp_orientation_ready': tcp_orientation_ready,
                'object_position_error': object_position_error,
                'object_position_tolerance': object_tolerance,
                'object_orientation_error': object_orientation_error,
                'object_orientation_tolerance': object_orientation_tolerance,
                'object_position_ready': object_position_ready,
                'object_orientation_ready': object_orientation_ready,
                'lateral_error': lateral_error,
                'lateral_tolerance': lateral_tolerance,
                'lateral_ready': lateral_ready,
            }
        )
        detail['capture_ready'] = bool(
            tcp_position_ready
            and tcp_orientation_ready
            and object_position_ready
            and object_orientation_ready
            and lateral_ready
        )
        return detail

    @staticmethod
    def _compliant_attachment_allowance(
        *,
        object_name: str | None,
        tracked_objects: dict,
    ) -> tuple[float, float, str | None]:
        if object_name is None:
            return 0.0, 0.0, None
        tracked_object = tracked_objects.get(str(object_name), {})
        attachment_state = tracked_object.get('attachment') or {}
        attachment_mode = attachment_state.get('mode')
        if attachment_mode != 'compliant_joint':
            return 0.0, 0.0, attachment_mode
        attach_spec = attachment_state.get('attach_spec') or {}
        linear_limit = max(
            float(attach_spec.get('compliant_hold_linear_limit', 0.006)),
            0.0,
        )
        angular_limit_degrees = max(
            float(
                attach_spec.get(
                    'compliant_hold_angular_limit_degrees',
                    6.0,
                )
            ),
            0.0,
        )
        return (
            math.sqrt(3.0) * linear_limit,
            math.radians(angular_limit_degrees),
            attachment_mode,
        )

    def _debug_target_object_settle(
        self,
        *,
        task,
        detail: dict[str, Any],
        state: dict[str, Any],
        event: str,
        hold_active: bool,
        tracked_objects: dict,
    ) -> None:
        if not self._debug_grasp_enabled() or not bool(detail.get('configured')):
            return
        phase_step = int(getattr(task, 'phase_step_counter', 0))
        every = max(int(os.environ.get('UR5E_DEBUG_TRANSPORT_EVERY', '15')), 1)
        if phase_step > 5 and phase_step % every != 0 and event not in {'capture', 'hold_complete'}:
            return
        motion_detail = self._object_motion_detail(
            task=task,
            object_name=detail.get('object_name'),
            tracked_objects=tracked_objects,
        )
        print(
            '[ur5e-settle-debug] '
            f"step={getattr(task, 'step_counter', None)} "
            f"phase={getattr(task, 'phase', None)} phase_step={phase_step} "
            f"event={event} capture_ready={detail.get('capture_ready')} "
            f"hold_active={hold_active} hold_remaining={state.get('remaining_steps', 0)} "
            f"retry_servo_remaining={state.get('retry_servo_steps', 0)} "
            f"capture_count={state.get('capture_count', 0)} "
            f"tcp_pos={detail.get('tcp_position_error')}/{detail.get('tcp_position_tolerance')} "
            f"tcp_pos_ready={detail.get('tcp_position_ready')} "
            f"tcp_rot={detail.get('tcp_orientation_error')}/{detail.get('tcp_orientation_tolerance')} "
            f"tcp_rot_ready={detail.get('tcp_orientation_ready')} "
            f"object_pos={detail.get('object_position_error')}/{detail.get('object_position_tolerance')} "
            f"object_pos_ready={detail.get('object_position_ready')} "
            f"object_rot={detail.get('object_orientation_error')}/{detail.get('object_orientation_tolerance')} "
            f"object_rot_ready={detail.get('object_orientation_ready')} "
            f"lateral={detail.get('lateral_error')}/{detail.get('lateral_tolerance')} "
            f"lateral_ready={detail.get('lateral_ready')} "
            f"lin={motion_detail.get('linear_speed')} ang={motion_detail.get('angular_speed')} "
            f"relaxed={detail.get('relaxed_active')}",
            flush=True,
        )

    def _cartesian_command_servo_target_pose(
        self,
        *,
        phase_key,
        current_pose: dict,
        target_pose: dict,
        max_position_step: float,
        max_orientation_step: float,
        spec: dict,
    ) -> dict:
        command_pose = self._cartesian_servo_target_pose(
            current_pose=current_pose,
            target_pose=target_pose,
            max_position_step=max_position_step,
            max_orientation_step=max_orientation_step,
        )
        if not bool(spec.get('cartesian_orientation_command_warm_start', False)) or max_orientation_step <= 0.0:
            return command_pose

        previous_orientation = self._cartesian_command_orientations.get(phase_key)
        if previous_orientation is None:
            previous_orientation = normalize_quat(current_pose['orientation'])
        orientation_reference_pose = {
            'position': np.zeros(3, dtype=float),
            'orientation': normalize_quat(previous_orientation),
        }
        orientation_target_pose = {
            'position': np.zeros(3, dtype=float),
            'orientation': normalize_quat(target_pose['orientation']),
        }
        accumulated = self._cartesian_servo_target_pose(
            current_pose=orientation_reference_pose,
            target_pose=orientation_target_pose,
            max_position_step=0.0,
            max_orientation_step=max_orientation_step,
        )
        lookahead = max(
            float(
                spec.get(
                    'cartesian_orientation_command_lookahead',
                    max_orientation_step * 4.0,
                )
            ),
            max_orientation_step,
        )
        bounded = self._cartesian_servo_target_pose(
            current_pose={
                'position': np.zeros(3, dtype=float),
                'orientation': normalize_quat(current_pose['orientation']),
            },
            target_pose=accumulated,
            max_position_step=0.0,
            max_orientation_step=lookahead,
        )
        command_pose['orientation'] = normalize_quat(bounded['orientation'])
        return command_pose

    def _accumulated_cartesian_command_position(
        self,
        *,
        phase_key,
        current_position: np.ndarray,
        one_step_position: np.ndarray,
        gate_position: np.ndarray,
        enabled: bool,
        lookahead: float,
        allow_gate_overdrive: bool = False,
        accumulation_step: float | None = None,
    ) -> np.ndarray:
        current_position = np.asarray(current_position, dtype=float)
        one_step_position = np.asarray(one_step_position, dtype=float)
        gate_position = np.asarray(gate_position, dtype=float)
        if not enabled:
            return one_step_position
        if (
            current_position.shape != (3,)
            or one_step_position.shape != (3,)
            or gate_position.shape != (3,)
            or not np.all(np.isfinite(current_position))
            or not np.all(np.isfinite(one_step_position))
            or not np.all(np.isfinite(gate_position))
            or not np.isfinite(lookahead)
            or lookahead <= 0.0
            or (accumulation_step is not None and (not np.isfinite(accumulation_step) or accumulation_step <= 0.0))
        ):
            raise ValueError(
                'Cartesian position command accumulation requires finite 3D '
                'positions and a finite positive lookahead.'
            )

        increment = one_step_position - current_position
        previous_position = self._cartesian_command_positions.get(phase_key)
        if previous_position is None:
            candidate_position = one_step_position
        else:
            previous_position = np.asarray(previous_position, dtype=float)
            if previous_position.shape != (3,) or not np.all(np.isfinite(previous_position)):
                previous_position = current_position
            increment_distance = float(np.linalg.norm(increment))
            if accumulation_step is not None and increment_distance > accumulation_step:
                increment = increment * (accumulation_step / increment_distance)
            candidate_position = previous_position + increment

        gate_delta = gate_position - current_position
        candidate_delta = candidate_position - current_position
        gate_distance = float(np.linalg.norm(gate_delta))
        candidate_distance = float(np.linalg.norm(candidate_delta))
        if gate_distance <= 1e-12 and not allow_gate_overdrive:
            candidate_position = gate_position
        elif not allow_gate_overdrive:
            gate_direction = gate_delta / gate_distance
            candidate_progress = float(np.dot(candidate_delta, gate_direction))
            if candidate_progress >= gate_distance:
                candidate_position = gate_position

        candidate_delta = candidate_position - current_position
        candidate_distance = float(np.linalg.norm(candidate_delta))
        if candidate_distance > lookahead:
            candidate_position = current_position + candidate_delta * (lookahead / candidate_distance)
        return np.asarray(candidate_position, dtype=float)

    def _remember_cartesian_command_position(
        self,
        *,
        phase_key,
        command_target_pose: dict,
    ) -> None:
        position = command_target_pose.get('position')
        if position is None:
            return
        position = np.asarray(position, dtype=float)
        if position.shape == (3,) and np.all(np.isfinite(position)):
            self._cartesian_command_positions[phase_key] = position.copy()

    def _remember_cartesian_command_orientation(
        self,
        *,
        phase_key,
        command_target_pose: dict,
    ) -> None:
        orientation = command_target_pose.get('orientation')
        if orientation is None:
            return
        try:
            orientation = normalize_quat(orientation)
        except Exception:
            return
        if orientation.shape == (4,) and np.all(np.isfinite(orientation)):
            self._cartesian_command_orientations[phase_key] = orientation.copy()

    def _target_pose(  # noqa: C901
        self,
        *,
        phase_key=None,
        task,
        robot_name: str,
        spec: dict,
        tracked_robots: dict,
        tracked_objects: dict,
    ):
        if bool(spec.get('relative_to_current_tcp', False)):
            raw_current_pose = self._current_robot_pose(
                task=task,
                robot_name=robot_name,
                tracked_robots=tracked_robots,
            )
            current_pose = self._current_tcp_pose(current_pose=raw_current_pose, spec=spec)
            if current_pose is None:
                return None
            offset = np.asarray(spec.get('offset', [0.0, 0.0, 0.0]), dtype=float)
            offset_frame = str(spec.get('offset_frame', 'world')).lower()
            if offset_frame in {'target', 'eef', 'tcp', 'gripper'}:
                offset = quat_rotate(current_pose['orientation'], offset)
            target_position = np.asarray(current_pose['position'], dtype=float) + offset
            minimum_planar_radius = spec.get('workspace_minimum_planar_radius')
            if minimum_planar_radius is not None:
                workspace_center = spec.get('workspace_center')
                if workspace_center is None:
                    return None
                try:
                    target_position = self._bound_relative_target_to_planar_workspace(
                        current_position=current_pose['position'],
                        target_position=target_position,
                        workspace_center=workspace_center,
                        minimum_planar_radius=float(minimum_planar_radius),
                    )
                except (TypeError, ValueError):
                    return None
            return {
                'position': target_position,
                'orientation': normalize_quat(current_pose['orientation']),
            }

        grasp_relative_pose = self._configured_grasp_relative_pose(spec)
        if grasp_relative_pose is not None:
            object_name = self._object_name_from_spec(spec)
            if object_name is None:
                return None
            object_pose = self._object_pose(
                task=task,
                object_name=object_name,
                tracked_objects=tracked_objects,
            )
            if object_pose is None:
                return None
            relative_position, relative_orientation = grasp_relative_pose
            orientation = normalize_quat(
                quat_multiply(
                    object_pose['orientation'],
                    self._quat_conjugate(relative_orientation),
                )
            )
            position = np.asarray(object_pose['position'], dtype=float) - quat_rotate(
                orientation,
                relative_position,
            )
            approach_clearance = float(spec.get('approach_clearance', 0.0))
            if approach_clearance < 0.0 or not np.isfinite(approach_clearance):
                return None
            relative_distance = float(np.linalg.norm(relative_position))
            if approach_clearance > 0.0 and relative_distance > 1e-8:
                retreat_in_target = -relative_position * (approach_clearance / relative_distance)
                position = position + quat_rotate(orientation, retreat_in_target)
            approach_offset = np.asarray(spec.get('offset', [0.0, 0.0, 0.0]), dtype=float)
            offset_frame = str(spec.get('offset_frame', 'world')).lower()
            if offset_frame in {'object', 'local', 'part'}:
                position = position + quat_rotate(object_pose['orientation'], approach_offset)
            elif offset_frame in {'target', 'eef', 'tcp', 'gripper'}:
                position = position + quat_rotate(orientation, approach_offset)
            else:
                position = position + approach_offset
            return {
                'position': np.asarray(position, dtype=float),
                'orientation': orientation,
            }

        object_pose = None
        direct_target_pose = None
        target_object_pose = self._target_object_pose(task=task, spec=spec)
        target_object_position = None if target_object_pose is None else target_object_pose['position']
        if target_object_position is not None:
            object_name = self._object_name_from_spec(spec)
            if object_name is None:
                return None
            object_pose = self._object_pose(
                task=task,
                object_name=object_name,
                tracked_objects=tracked_objects,
            )
            if object_pose is None:
                return None
        elif spec.get('target_pose_target') is not None or spec.get('target_pose') is not None:
            target_name = spec.get('target_pose_target') or spec.get('target_pose')
            direct_target_pose = self._target_pose_by_name(
                task=task,
                target_name=None if target_name is None else str(target_name),
            )
            if direct_target_pose is None:
                return None
            position = direct_target_pose['position']
        elif spec.get('target_position') is not None:
            position = np.asarray(spec['target_position'], dtype=float)
        else:
            object_name = self._object_name_from_spec(spec)
            if object_name is None:
                return None
            object_pose = self._object_pose(
                task=task,
                object_name=object_name,
                tracked_objects=tracked_objects,
            )
            if object_pose is None:
                return None
            offset = np.asarray(spec.get('offset', [0.0, 0.0, 0.0]), dtype=float)
            if str(spec.get('offset_frame', 'world')).lower() in {'object', 'local', 'part'}:
                position, _ = compose_pose(
                    base_position=object_pose['position'],
                    base_orientation=object_pose['orientation'],
                    local_position=offset,
                    local_orientation=[1.0, 0.0, 0.0, 0.0],
                )
            else:
                position = object_pose['position'] + offset

        has_explicit_orientation = any(
            spec.get(name) is not None for name in ('target_orientation', 'orientation', 'orientation_euler')
        )
        derive_tcp_orientation = bool(
            target_object_position is not None and self._derive_tcp_orientation_from_target_object(spec=spec)
        )
        relative_pose = None
        if derive_tcp_orientation:
            relative_pose = self._object_tcp_relative_pose(
                phase_key=phase_key,
                task=task,
                robot_name=robot_name,
                object_name=object_name,
                spec=spec,
                tracked_robots=tracked_robots,
                object_pose=object_pose,
                tracked_objects=tracked_objects,
            )
            if relative_pose is None or target_object_pose.get('orientation') is None:
                return None
            _, relative_orientation = relative_pose
            orientation = normalize_quat(
                quat_multiply(
                    target_object_pose['orientation'],
                    self._quat_conjugate(relative_orientation),
                )
            )
        elif direct_target_pose is not None and not has_explicit_orientation:
            orientation = direct_target_pose['orientation']
        else:
            orientation = self._target_orientation(
                task=task,
                robot_name=robot_name,
                spec=spec,
                tracked_robots=tracked_robots,
                object_pose=object_pose,
            )
        if orientation is None:
            return None
        if target_object_position is not None:
            if relative_pose is None:
                relative_pose = self._object_tcp_relative_pose(
                    phase_key=phase_key,
                    task=task,
                    robot_name=robot_name,
                    object_name=object_name,
                    spec=spec,
                    tracked_robots=tracked_robots,
                    object_pose=object_pose,
                    tracked_objects=tracked_objects,
                )
            if relative_pose is None:
                return None
            relative_position, _ = relative_pose
            position = np.asarray(target_object_position, dtype=float) - quat_rotate(
                normalize_quat(orientation),
                relative_position,
            )
        return {
            'position': np.asarray(position, dtype=float),
            'orientation': normalize_quat(orientation),
        }

    @staticmethod
    def _bound_relative_target_to_planar_workspace(
        *,
        current_position,
        target_position,
        workspace_center,
        minimum_planar_radius: float,
    ) -> np.ndarray:
        """Stop an inward Cartesian segment at the workspace's inner radial boundary."""

        current = np.asarray(current_position, dtype=float)
        target = np.asarray(target_position, dtype=float)
        center = np.asarray(workspace_center, dtype=float)
        if current.shape != (3,) or target.shape != (3,) or center.shape != (3,):
            raise ValueError('Workspace-bounded Cartesian positions must be 3-vectors.')
        if not (
            np.all(np.isfinite(current))
            and np.all(np.isfinite(target))
            and np.all(np.isfinite(center))
            and np.isfinite(minimum_planar_radius)
            and minimum_planar_radius > 0.0
        ):
            raise ValueError('Workspace-bounded Cartesian positions must be finite.')

        current_radial = current[:2] - center[:2]
        target_radial = target[:2] - center[:2]
        radius = float(minimum_planar_radius)
        if float(np.linalg.norm(target_radial)) >= radius:
            return target.copy()

        result = target.copy()
        current_radius = float(np.linalg.norm(current_radial))
        if current_radius <= radius:
            result[:2] = current[:2]
            return result

        planar_delta = target[:2] - current[:2]
        quadratic_a = float(np.dot(planar_delta, planar_delta))
        quadratic_b = 2.0 * float(np.dot(current_radial, planar_delta))
        quadratic_c = float(np.dot(current_radial, current_radial) - radius * radius)
        discriminant = quadratic_b * quadratic_b - 4.0 * quadratic_a * quadratic_c
        if quadratic_a <= 1e-16 or discriminant < 0.0:
            result[:2] = current[:2]
            return result

        square_root = math.sqrt(max(discriminant, 0.0))
        intersections = sorted(
            (
                (-quadratic_b - square_root) / (2.0 * quadratic_a),
                (-quadratic_b + square_root) / (2.0 * quadratic_a),
            )
        )
        segment_ratio = next(
            (value for value in intersections if 0.0 <= value <= 1.0),
            None,
        )
        if segment_ratio is None:
            result[:2] = current[:2]
            return result
        result[:2] = current[:2] + float(segment_ratio) * planar_delta
        return result

    @staticmethod
    def _configured_grasp_relative_pose(spec: dict) -> tuple[np.ndarray, np.ndarray] | None:
        position = spec.get(
            'grasp_relative_position',
            spec.get('object_in_tcp_position', spec.get('object_in_gripper_position')),
        )
        orientation = spec.get(
            'grasp_relative_orientation',
            spec.get('object_in_tcp_orientation', spec.get('object_in_gripper_orientation')),
        )
        if position is None and orientation is None:
            return None
        if position is None or orientation is None:
            return None
        relative_position = np.asarray(position, dtype=float)
        relative_orientation = normalize_quat(orientation)
        if relative_position.shape != (3,) or not np.all(np.isfinite(relative_position)):
            return None
        if relative_orientation.shape != (4,) or not np.all(np.isfinite(relative_orientation)):
            return None
        return relative_position, relative_orientation

    @staticmethod
    def _quat_conjugate(quat) -> np.ndarray:
        quat = normalize_quat(quat)
        return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=float)

    @staticmethod
    def _target_pose_by_name(*, task, target_name: str | None) -> dict | None:
        if not target_name:
            return None
        target_poses = getattr(task, 'target_poses', None)
        if not isinstance(target_poses, dict) or target_name not in target_poses:
            return None
        target_pose = target_poses[target_name]
        if target_pose.get('position') is None:
            return None
        return {
            'position': np.asarray(target_pose['position'], dtype=float),
            'orientation': normalize_quat(target_pose.get('orientation', [1.0, 0.0, 0.0, 0.0])),
        }

    def _target_object_pose(self, *, task, spec: dict) -> dict | None:
        if spec.get('target_object_position') is not None:
            position = np.asarray(spec['target_object_position'], dtype=float)
            orientation = self._target_object_orientation_from_spec(task=task, spec=spec)
        else:
            target_name = spec.get('target_object_target') or spec.get('target_object') or spec.get('object_target')
            target_pose = self._target_pose_by_name(
                task=task,
                target_name=None if target_name is None else str(target_name),
            )
            if target_pose is None:
                return None
            position = target_pose['position']
            orientation = target_pose['orientation']
            override_orientation = self._target_object_orientation_from_spec(task=task, spec=spec)
            if override_orientation is not None:
                orientation = override_orientation
        offset = spec.get('target_object_offset')
        if offset is not None:
            offset = np.asarray(offset, dtype=float)
            offset_frame = str(spec.get('target_object_offset_frame', 'world')).lower()
            if offset_frame in {'target', 'local', 'object_target'}:
                if orientation is None:
                    return None
                position = position + quat_rotate(orientation, offset)
            else:
                position = position + offset
        return {
            'position': np.asarray(position, dtype=float),
            'orientation': None if orientation is None else normalize_quat(orientation),
        }

    @staticmethod
    def _target_object_orientation_from_spec(*, task, spec: dict) -> np.ndarray | None:
        if spec.get('target_object_orientation') is not None:
            return normalize_quat(spec['target_object_orientation'])
        if spec.get('target_object_orientation_euler') is not None:
            return euler_xyz_to_quat(spec['target_object_orientation_euler'])
        target_name = spec.get('target_object_orientation_target') or spec.get('object_orientation_target')
        if target_name is None:
            return None
        target_pose = UR5eAssemblyAtomicSkillAdapter._target_pose_by_name(
            task=task,
            target_name=str(target_name),
        )
        if target_pose is None:
            return None
        return normalize_quat(target_pose['orientation'])

    def _target_object_position(self, *, task, spec: dict) -> np.ndarray | None:
        target_pose = self._target_object_pose(task=task, spec=spec)
        if target_pose is None:
            return None
        return np.asarray(target_pose['position'], dtype=float)

    def _object_tcp_relative_position(
        self,
        *,
        phase_key,
        task,
        robot_name: str,
        object_name: str,
        spec: dict,
        tracked_robots: dict,
        object_pose: dict,
    ) -> np.ndarray | None:
        relative_pose = self._object_tcp_relative_pose(
            phase_key=phase_key,
            task=task,
            robot_name=robot_name,
            object_name=object_name,
            spec=spec,
            tracked_robots=tracked_robots,
            object_pose=object_pose,
        )
        if relative_pose is None:
            return None
        return relative_pose[0].copy()

    def _object_tcp_relative_pose(
        self,
        *,
        phase_key,
        task,
        robot_name: str,
        object_name: str,
        spec: dict,
        tracked_robots: dict,
        object_pose: dict,
        tracked_objects: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        cache = getattr(task, '_ur5e_plumbers_object_tcp_relative_poses', None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(task, '_ur5e_plumbers_object_tcp_relative_poses', cache)
        cache_key = (str(robot_name), str(object_name))
        tracked_objects = tracked_objects or {}
        object_state = tracked_objects.get(object_name, {})
        attachment = object_state.get('attachment') or {}
        attachment_robot = attachment.get('robot_name')
        if attachment_robot is None:
            attachment_robot = object_state.get('attached_to')
        attachment_mode = str(attachment.get('mode', '')).strip().lower()
        attachment_identity = None
        if attachment_robot == robot_name and attachment_mode:
            attachment_identity = (
                str(attachment_robot),
                attachment_mode,
                attachment.get('attach_step'),
                attachment.get('joint_path'),
            )

        if attachment_identity is not None and attachment_mode == 'fixed_joint' and self._tcp_offset(spec) is None:
            try:
                attachment_position = np.asarray(attachment['position'], dtype=float)
                attachment_orientation = normalize_quat(attachment['orientation'])
            except (KeyError, TypeError, ValueError):
                attachment_position = None
                attachment_orientation = None
            if (
                attachment_position is not None
                and attachment_position.shape == (3,)
                and attachment_orientation is not None
                and attachment_orientation.shape == (4,)
                and np.all(np.isfinite(attachment_position))
                and np.all(np.isfinite(attachment_orientation))
            ):
                cache[cache_key] = {
                    'position': attachment_position.copy(),
                    'orientation': attachment_orientation.copy(),
                    'attachment_identity': attachment_identity,
                    'source': 'observed_attachment',
                }
                return (
                    attachment_position.copy(),
                    attachment_orientation.copy(),
                )

        cached = cache.get(cache_key)
        if cached is not None:
            if cached.get('attachment_identity') != attachment_identity:
                cache.pop(cache_key, None)
                cached = None
        if cached is not None:
            try:
                cached_position = np.asarray(cached['position'], dtype=float)
                cached_orientation = normalize_quat(cached['orientation'])
                if (
                    cached_position.shape == (3,)
                    and cached_orientation.shape == (4,)
                    and np.all(np.isfinite(cached_position))
                    and np.all(np.isfinite(cached_orientation))
                ):
                    return cached_position.copy(), cached_orientation.copy()
            except Exception:
                pass

        raw_current_pose = self._current_robot_pose(task=task, robot_name=robot_name, tracked_robots=tracked_robots)
        current_tcp_pose = self._current_tcp_pose(current_pose=raw_current_pose, spec=spec)
        if current_tcp_pose is None:
            return None
        relative_world = np.asarray(object_pose['position'], dtype=float) - np.asarray(
            current_tcp_pose['position'],
            dtype=float,
        )
        relative_tcp = quat_rotate(
            self._quat_conjugate(current_tcp_pose['orientation']),
            relative_world,
        )
        if relative_tcp.shape != (3,) or not np.all(np.isfinite(relative_tcp)):
            return None
        relative_orientation = normalize_quat(
            quat_multiply(
                self._quat_conjugate(current_tcp_pose['orientation']),
                object_pose['orientation'],
            )
        )
        if relative_orientation.shape != (4,) or not np.all(np.isfinite(relative_orientation)):
            return None
        cache[cache_key] = {
            'position': relative_tcp.copy(),
            'orientation': relative_orientation.copy(),
            'attachment_identity': attachment_identity,
            'source': 'observed_tcp_pose',
        }
        if self._debug_grasp_enabled():
            print(
                '[ur5e-grasp-debug] '
                f'captured_object_tcp_relative robot={robot_name} object={object_name} '
                f'phase_key={phase_key} relative_tcp={relative_tcp.tolist()} '
                f'relative_orientation={relative_orientation.tolist()} '
                f"object_position={np.asarray(object_pose['position'], dtype=float).tolist()} "
                f"tcp_position={np.asarray(current_tcp_pose['position'], dtype=float).tolist()}",
                flush=True,
            )
        return relative_tcp.copy(), relative_orientation.copy()

    @staticmethod
    def _derive_tcp_orientation_from_target_object(*, spec: dict) -> bool:
        for name in (
            'derive_tcp_orientation_from_target_object',
            'target_orientation_from_object_target',
            'use_target_object_orientation',
        ):
            if name in spec:
                return bool(spec.get(name))
        return False

    def _locked_target_pose(self, *, phase_key, target_pose: dict, spec: dict) -> dict:
        lock_orientation = bool(spec.get('lock_target_orientation', True))
        lock_position = bool(spec.get('lock_target_position', False))
        if not lock_orientation and not lock_position:
            return target_pose

        locked = self._phase_locks.setdefault(phase_key, {})
        result = {
            'position': np.asarray(target_pose['position'], dtype=float).copy(),
            'orientation': normalize_quat(target_pose['orientation']).copy(),
        }
        if lock_position:
            if 'position' not in locked:
                locked['position'] = result['position'].copy()
            result['position'] = locked['position'].copy()
        if lock_orientation:
            if 'orientation' not in locked:
                locked['orientation'] = result['orientation'].copy()
            result['orientation'] = locked['orientation'].copy()
        return result

    def _close_pose_gate_action(
        self,
        *,
        phase_key,
        task,
        robot_name: str,
        spec: dict,
        tracked_robots: dict,
        tracked_objects: dict,
    ) -> tuple[bool, OrderedDict, dict[str, Any]]:
        state = self._close_gate_state.setdefault(phase_key, {'ready_steps': 0})
        gate_spec = dict(spec)
        for name in (
            'offset',
            'offset_frame',
            'target_orientation',
            'target_orientation_frame',
            'orientation',
            'orientation_frame',
            'orientation_euler',
            'ik_target_offset',
            'ik_target_offset_frame',
            'grasp_tcp_offset',
            'grasp_tcp_offset_frame',
            'tcp_offset',
            'tcp_offset_frame',
        ):
            override_name = f'close_gate_{name}'
            if override_name in spec:
                gate_spec[name] = spec[override_name]

        target_pose = self._target_pose(
            phase_key=phase_key,
            task=task,
            robot_name=robot_name,
            spec=gate_spec,
            tracked_robots=tracked_robots,
            tracked_objects=tracked_objects,
        )
        if target_pose is None:
            action = self._hold_joint_action(task=task, robot_name=robot_name)
            action[_GRIPPER_CONTROLLER] = [
                float(gate_spec.get('preclose_openness', gate_spec.get('open_openness', 1.0)))
            ]
            state['ready_steps'] = 0
            return False, action, {'reason': 'close_gate_target_pose_unavailable'}
        current_pose = self._current_tcp_pose(
            current_pose=self._current_robot_pose(
                task=task,
                robot_name=robot_name,
                tracked_robots=tracked_robots,
            ),
            spec=gate_spec,
        )
        recenter_offset = np.asarray(
            state.get('recenter_offset_world', np.zeros(3, dtype=float)),
            dtype=float,
        )
        if recenter_offset.shape == (3,) and np.all(np.isfinite(recenter_offset)):
            target_position = np.asarray(target_pose['position'], dtype=float)
            if (
                bool(gate_spec.get('close_gate_recenter_single_finger_contact', False))
                and float(np.linalg.norm(recenter_offset)) > 1e-12
            ):
                if 'recenter_anchor_position_world' not in state:
                    state['recenter_anchor_position_world'] = (
                        target_position.copy()
                        if current_pose is None
                        else np.asarray(current_pose['position'], dtype=float).copy()
                    )
                target_position = (
                    np.asarray(
                        state['recenter_anchor_position_world'],
                        dtype=float,
                    )
                    + recenter_offset
                )
            target_pose = {
                'position': target_position,
                'orientation': normalize_quat(target_pose['orientation']).copy(),
            }
        lock_spec = gate_spec
        if (
            bool(gate_spec.get('close_gate_recenter_single_finger_contact', False))
            and float(np.linalg.norm(recenter_offset)) > 1e-12
        ):
            lock_spec = {**gate_spec, 'lock_target_position': False}
        target_pose = self._locked_target_pose(
            phase_key=phase_key,
            target_pose=target_pose,
            spec=lock_spec,
        )
        command_target_pose = target_pose
        if bool(gate_spec.get('close_gate_cartesian_servo', True)) and current_pose is not None:
            command_target_pose = self._cartesian_servo_target_pose(
                current_pose=current_pose,
                target_pose=target_pose,
                max_position_step=float(
                    gate_spec.get(
                        'close_gate_cartesian_position_step',
                        gate_spec.get('cartesian_position_step', 0.004),
                    )
                ),
                max_orientation_step=float(
                    gate_spec.get(
                        'close_gate_cartesian_orientation_step',
                        gate_spec.get('cartesian_orientation_step', 0.015),
                    )
                ),
            )

        current_q = self._current_arm_q(task, robot_name)
        reference_q = self._command_reference_q(task=task, robot_name=robot_name, current_q=current_q, spec=gate_spec)
        ik_target_pose = self._ik_target_pose(target_pose=command_target_pose, spec=gate_spec)
        ik_result = self._solve_ik(
            task=task,
            robot_name=robot_name,
            target_pose=ik_target_pose,
            warm_start=reference_q,
            spec=gate_spec,
        )
        if ik_result is None:
            action = self._hold_joint_action(task=task, robot_name=robot_name)
            action[_GRIPPER_CONTROLLER] = [
                float(gate_spec.get('preclose_openness', gate_spec.get('open_openness', 1.0)))
            ]
            state['ready_steps'] = 0
            return (
                False,
                action,
                {
                    'reason': 'close_gate_ik_failed',
                    'target_position': target_pose['position'].tolist(),
                    'target_orientation': target_pose['orientation'].tolist(),
                },
            )

        target_q = self._unwrap_to_reference(
            target_q=ik_result,
            reference_q=reference_q,
            preferred_abs_limit=gate_spec.get('preferred_joint_abs_limit', 3.05),
            hard_preferred_abs_limit=bool(gate_spec.get('hard_preferred_joint_abs_limit', True)),
        )
        if reference_q is None:
            action = self._hold_joint_action(task=task, robot_name=robot_name)
            action[_GRIPPER_CONTROLLER] = [
                float(gate_spec.get('preclose_openness', gate_spec.get('open_openness', 1.0)))
            ]
            state['ready_steps'] = 0
            return (
                False,
                action,
                {
                    'reason': 'close_gate_current_joint_state_unavailable',
                    'target_position': target_pose['position'].tolist(),
                    'target_orientation': target_pose['orientation'].tolist(),
                },
            )
        if bool(gate_spec.get('close_gate_guard_ik_branch_jump', True)) and self._ik_branch_jump_detected(
            reference_q=reference_q,
            target_q=target_q,
            spec=gate_spec,
        ):
            action = self._hold_joint_action(task=task, robot_name=robot_name)
            action[_GRIPPER_CONTROLLER] = [
                float(gate_spec.get('preclose_openness', gate_spec.get('open_openness', 1.0)))
            ]
            state['ready_steps'] = 0
            return (
                False,
                action,
                {
                    'reason': 'close_gate_ik_branch_jump_guard',
                    'target_position': target_pose['position'].tolist(),
                    'target_orientation': target_pose['orientation'].tolist(),
                    'reference_q': reference_q.tolist(),
                    'target_q': target_q.tolist(),
                },
            )
        command_q = self._limited_joint_target(
            current_q=reference_q,
            target_q=target_q,
            max_joint_step=float(gate_spec.get('close_gate_max_joint_step', gate_spec.get('max_joint_step', 0.025))),
        )
        command_q = self._continuous_command_q(
            task=task,
            robot_name=robot_name,
            command_q=command_q,
            spec={
                **gate_spec,
                'max_joint_step': gate_spec.get(
                    'close_gate_max_joint_step',
                    gate_spec.get('max_joint_step', 0.025),
                ),
            },
        )
        command_q = self._limit_command_to_measured_state(
            current_q=current_q,
            command_q=command_q,
            spec=gate_spec,
        )

        action = OrderedDict()
        action[_ARM_JOINT_CONTROLLER] = [command_q.tolist()]
        self._remember_arm_command(task, robot_name, command_q)
        action[_GRIPPER_CONTROLLER] = [float(gate_spec.get('preclose_openness', gate_spec.get('open_openness', 1.0)))]

        detail = {
            'target_position': target_pose['position'].tolist(),
            'target_orientation': target_pose['orientation'].tolist(),
            'command_target_position': command_target_pose['position'].tolist(),
            'command_target_orientation': command_target_pose['orientation'].tolist(),
            'ik_target_position': ik_target_pose['position'].tolist(),
            'ik_target_orientation': ik_target_pose['orientation'].tolist(),
            'recenter_offset_world': recenter_offset.tolist(),
        }
        ready = False
        if current_pose is not None:
            position_tolerance = float(
                gate_spec.get('close_position_tolerance', gate_spec.get('position_tolerance', 0.01))
            )
            orientation_tolerance = gate_spec.get('close_orientation_tolerance', gate_spec.get('orientation_tolerance'))
            orientation_tolerance = None if orientation_tolerance is None else float(orientation_tolerance)
            position_error, orientation_error = pose_error(
                current_position=current_pose['position'],
                current_orientation=current_pose['orientation'],
                target_position=target_pose['position'],
                target_orientation=target_pose['orientation'],
            )
            ready = position_error <= position_tolerance and (
                orientation_tolerance is None or orientation_error is None or orientation_error <= orientation_tolerance
            )
            detail.update(
                {
                    'position_error': position_error,
                    'orientation_error': orientation_error,
                    'position_tolerance': position_tolerance,
                    'orientation_tolerance': orientation_tolerance,
                }
            )
            recenter_target_tolerance = float(gate_spec.get('close_gate_recenter_target_tolerance', 0.00035))
            state['recenter_target_ready'] = bool(position_error <= recenter_target_tolerance)
            detail.update(
                {
                    'recenter_target_ready': state['recenter_target_ready'],
                    'recenter_target_tolerance': recenter_target_tolerance,
                }
            )

        joint_tolerance = gate_spec.get('close_joint_position_tolerance', gate_spec.get('joint_position_tolerance'))
        if joint_tolerance is not None and current_q is not None:
            joint_error = float(np.max(np.abs(np.asarray(target_q, dtype=float) - np.asarray(current_q, dtype=float))))
            ready = bool(ready and joint_error <= float(joint_tolerance))
            detail.update(
                {
                    'joint_error': joint_error,
                    'joint_position_tolerance': float(joint_tolerance),
                }
            )

        if ready:
            state['ready_steps'] = int(state.get('ready_steps', 0)) + 1
        else:
            state['ready_steps'] = 0
            state.pop('close_started_step', None)
            state.pop('hold_q', None)

        required_ready_steps = max(int(gate_spec.get('close_ready_stable_steps', 4)), 1)
        gate_ready = int(state.get('ready_steps', 0)) >= required_ready_steps
        detail['ready_steps'] = int(state.get('ready_steps', 0))
        detail['required_ready_steps'] = required_ready_steps
        detail['gate_ready'] = gate_ready
        if gate_ready:
            if 'close_started_step' not in state:
                state['close_started_step'] = int(getattr(task, 'phase_step_counter', 0))
                if bool(gate_spec.get('close_gate_hold_refined_command', False)):
                    hold_q = command_q
                else:
                    hold_q = current_q if current_q is not None else command_q
                state['hold_q'] = np.asarray(hold_q, dtype=float).copy()
            elif bool(gate_spec.get('close_gate_track_object_during_close', False)):
                # Closing fingers can slide a free part. Keep servoing the planned
                # object-in-TCP relation until contact instead of freezing the arm.
                state['hold_q'] = np.asarray(command_q, dtype=float).copy()
        return gate_ready, action, detail

    def _prealign_action(self, *, task, robot_name: str, target_pose: dict, spec: dict):
        prealign_steps = int(spec.get('prealign_steps', 0) or 0)
        if prealign_steps <= 0:
            return None
        if int(getattr(task, 'phase_step_counter', 0)) >= prealign_steps:
            return None

        current_q = self._current_arm_q(task, robot_name)
        reference_q = self._command_reference_q(task=task, robot_name=robot_name, current_q=current_q, spec=spec)
        if reference_q is None or reference_q.shape[0] < 1:
            return None

        desired_q = reference_q.copy()
        if spec.get('prealign_joint_positions') is not None:
            joint_values = np.asarray(spec['prealign_joint_positions'], dtype=float)
            desired_q[: min(desired_q.shape[0], joint_values.shape[0])] = joint_values[: desired_q.shape[0]]
        else:
            shoulder_pan = spec.get('prealign_shoulder_pan')
            if shoulder_pan is None and bool(spec.get('prealign_shoulder_pan_from_target', False)):
                shoulder_pan = self._target_facing_shoulder_pan(
                    task=task,
                    robot_name=robot_name,
                    target_position=target_pose['position'],
                    yaw_offset=float(spec.get('prealign_shoulder_pan_yaw_offset', -0.47)),
                )
            if shoulder_pan is None:
                return None
            desired_q[0] = float(shoulder_pan)

        command_q = self._limited_joint_target(
            current_q=reference_q,
            target_q=desired_q,
            max_joint_step=float(spec.get('prealign_max_joint_step', spec.get('max_joint_step', 0.035))),
        )
        command_q = self._continuous_command_q(
            task=task,
            robot_name=robot_name,
            command_q=command_q,
            spec={
                **spec,
                'max_joint_step': spec.get(
                    'prealign_max_joint_step',
                    spec.get('max_joint_step', 0.035),
                ),
            },
        )
        action = OrderedDict()
        action[_ARM_JOINT_CONTROLLER] = [command_q.tolist()]
        self._remember_arm_command(task, robot_name, command_q)
        action[_GRIPPER_CONTROLLER] = [
            self._gripper_command_value(task=task, robot_name=robot_name, command=spec.get('gripper_command', 'open'))
        ]
        return action

    @staticmethod
    def _target_facing_shoulder_pan(*, task, robot_name: str, target_position, yaw_offset: float) -> float | None:
        robot = task.robots.get(robot_name)
        if robot is None:
            return None
        try:
            base_position, _ = robot.articulation.get_pose()
        except Exception:
            return None
        target_position = np.asarray(target_position, dtype=float)
        base_position = np.asarray(base_position, dtype=float)
        delta = target_position[:2] - base_position[:2]
        if float(np.linalg.norm(delta)) < 1e-6:
            return None
        return float(np.arctan2(delta[1], delta[0]) + yaw_offset)

    @staticmethod
    def _current_tcp_pose(*, current_pose: dict | None, spec: dict) -> dict | None:
        if current_pose is None:
            return None
        position = np.asarray(current_pose['position'], dtype=float).copy()
        orientation = normalize_quat(current_pose['orientation'])
        tcp_offset = UR5eAssemblyAtomicSkillAdapter._tcp_offset(spec)
        if tcp_offset is None:
            return {'position': position, 'orientation': orientation}
        position = UR5eAssemblyAtomicSkillAdapter._position_with_offset(
            position=position,
            orientation=orientation,
            offset=tcp_offset,
            offset_frame=UR5eAssemblyAtomicSkillAdapter._tcp_offset_frame(spec),
            sign=1.0,
        )
        return {'position': position, 'orientation': orientation}

    @staticmethod
    def _ik_target_pose(*, target_pose: dict, spec: dict) -> dict:
        position = np.asarray(target_pose['position'], dtype=float).copy()
        orientation = normalize_quat(target_pose['orientation'])
        tcp_offset = UR5eAssemblyAtomicSkillAdapter._tcp_offset(spec)
        if tcp_offset is not None:
            position = UR5eAssemblyAtomicSkillAdapter._position_with_offset(
                position=position,
                orientation=orientation,
                offset=tcp_offset,
                offset_frame=UR5eAssemblyAtomicSkillAdapter._tcp_offset_frame(spec),
                sign=-1.0,
            )
        offset = spec.get('ik_target_offset')
        if offset is not None:
            offset = np.asarray(offset, dtype=float)
            offset_frame = str(spec.get('ik_target_offset_frame', 'world')).lower()
            if offset_frame in {'target', 'local', 'eef', 'gripper'}:
                position = position + quat_rotate(orientation, offset)
            else:
                position = position + offset
        return {'position': position, 'orientation': orientation}

    @staticmethod
    def _tcp_offset(spec: dict) -> np.ndarray | None:
        offset = spec.get('grasp_tcp_offset', spec.get('tcp_offset'))
        if offset is None:
            return None
        offset = np.asarray(offset, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            return None
        return offset

    @staticmethod
    def _tcp_offset_frame(spec: dict) -> str:
        return str(spec.get('grasp_tcp_offset_frame', spec.get('tcp_offset_frame', 'target'))).lower()

    @staticmethod
    def _position_with_offset(
        *,
        position: np.ndarray,
        orientation: np.ndarray,
        offset: np.ndarray,
        offset_frame: str,
        sign: float,
    ) -> np.ndarray:
        if offset_frame in {'target', 'local', 'eef', 'tool', 'gripper'}:
            return position + float(sign) * quat_rotate(orientation, offset)
        return position + float(sign) * offset

    @staticmethod
    def _target_orientation(
        *, task, robot_name: str, spec: dict, tracked_robots: dict, object_pose: dict | None = None
    ):
        del task
        if spec.get('target_orientation') is not None:
            orientation = np.asarray(spec['target_orientation'], dtype=float)
        elif spec.get('orientation') is not None:
            orientation = np.asarray(spec['orientation'], dtype=float)
        elif spec.get('orientation_euler') is not None:
            orientation = euler_xyz_to_quat(spec['orientation_euler'])
        else:
            robot_state = tracked_robots.get(robot_name, {})
            if robot_state.get('orientation') is not None:
                orientation = np.asarray(robot_state['orientation'], dtype=float)
            else:
                orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        orientation_frame = str(spec.get('target_orientation_frame', spec.get('orientation_frame', 'world'))).lower()
        if orientation_frame in {'object', 'local', 'part'}:
            if object_pose is None:
                return None
            return normalize_quat(quat_multiply(object_pose['orientation'], orientation))
        return normalize_quat(orientation)

    @staticmethod
    def _object_pose(*, task, object_name: str, tracked_objects: dict):
        object_state = tracked_objects.get(object_name, {})
        if object_state.get('position') is not None and object_state.get('orientation') is not None:
            return {
                'position': np.asarray(object_state['position'], dtype=float),
                'orientation': normalize_quat(object_state['orientation']),
            }
        try:
            position, orientation = task._resolve_object(object_name).get_pose()  # noqa: SLF001
        except Exception:
            return None
        return {
            'position': np.asarray(position, dtype=float),
            'orientation': normalize_quat(orientation),
        }

    @staticmethod
    def _current_robot_pose(*, task, robot_name: str, tracked_robots: dict):
        robot_state = tracked_robots.get(robot_name, {})
        if robot_state.get('position') is not None and robot_state.get('orientation') is not None:
            return {
                'position': np.asarray(robot_state['position'], dtype=float),
                'orientation': normalize_quat(robot_state['orientation']),
            }
        try:
            position, orientation = task._get_robot_task_pose(robot_name)  # noqa: SLF001
        except Exception:
            return None
        return {
            'position': np.asarray(position, dtype=float),
            'orientation': normalize_quat(orientation),
        }

    def _current_arm_q(self, task, robot_name: str) -> np.ndarray | None:
        robot = task.robots.get(robot_name)
        if robot is None:
            return None
        controller = robot.controllers.get(_ARM_JOINT_CONTROLLER)
        arm_joint_names = self._arm_joint_names(robot=robot, controller=controller)
        arm_joint_count = len(arm_joint_names)
        if controller is not None and hasattr(controller, 'get_joint_subset'):
            subset = controller.get_joint_subset()
        else:
            subset = getattr(controller, 'joint_subset', None) if controller is not None else None
        if subset is not None:
            try:
                joint_positions = self._coerce_arm_q(
                    subset.get_joint_positions(),
                    joint_count=arm_joint_count,
                )
                if joint_positions is not None:
                    return joint_positions
            except Exception:
                pass
        articulation = getattr(robot, 'articulation', None)
        if articulation is not None:
            try:
                indices = np.asarray([articulation.get_dof_index(name) for name in arm_joint_names], dtype=np.int64)
                joint_positions = self._coerce_arm_q(
                    articulation.get_joint_positions(joint_indices=indices),
                    joint_count=arm_joint_count,
                )
                if joint_positions is not None:
                    return joint_positions
            except Exception:
                pass
            try:
                all_joint_positions = np.asarray(articulation.get_joint_positions(), dtype=float)
                dof_names = list(getattr(articulation, 'dof_names', []) or [])
                if dof_names:
                    indices = [dof_names.index(name) for name in arm_joint_names if name in dof_names]
                    if len(indices) == arm_joint_count:
                        joint_positions = self._coerce_arm_q(
                            all_joint_positions[np.asarray(indices, dtype=np.int64)],
                            joint_count=arm_joint_count,
                        )
                        if joint_positions is not None:
                            return joint_positions
                joint_positions = self._coerce_arm_q(
                    all_joint_positions[:arm_joint_count],
                    joint_count=arm_joint_count,
                )
                if joint_positions is not None:
                    return joint_positions
            except Exception:
                pass
        last_q = self._last_arm_command_q.get(robot_name)
        if last_q is not None:
            return np.asarray(last_q, dtype=float).copy()
        try:
            obs = robot.get_obs()
            for control in obs.get('joint_action', []) or []:
                joint_positions = self._coerce_arm_q(control.get('joint_positions'))
                if joint_positions is not None:
                    return joint_positions
        except Exception:
            pass
        return None

    def _current_arm_dynamics(self, *, task, robot_name: str) -> dict[str, list[float]]:
        """Read optional PhysX drive diagnostics without affecting control."""

        robot = task.robots.get(robot_name)
        articulation = getattr(robot, 'articulation', None)
        if articulation is None:
            return {}
        controller = robot.controllers.get(_ARM_JOINT_CONTROLLER)
        joint_names = self._arm_joint_names(robot=robot, controller=controller)
        try:
            indices = np.asarray([articulation.get_dof_index(name) for name in joint_names], dtype=np.int64)
        except Exception:
            return {}

        result: dict[str, list[float]] = {}

        def _store(name: str, values) -> bool:
            try:
                array = np.asarray(values, dtype=float)
                if array.ndim > 1:
                    array = array[0]
                array = array.reshape(-1)
                if array.size != indices.size:
                    array = array[indices]
                if array.size != indices.size or not np.all(np.isfinite(array)):
                    return False
                result[name] = array.tolist()
                return True
            except Exception:
                return False

        try:
            _store('joint_velocity', articulation.get_joint_velocities(joint_indices=indices))
        except Exception:
            pass

        try:
            unwrapped = articulation.unwrap()
        except Exception:
            unwrapped = None
        for result_name, method_names in (
            ('measured_effort', ('get_measured_joint_efforts', 'get_joint_efforts')),
            ('applied_effort', ('get_applied_joint_efforts',)),
        ):
            for candidate in (unwrapped, articulation):
                if candidate is None:
                    continue
                for method_name in method_names:
                    method = getattr(candidate, method_name, None)
                    if not callable(method):
                        continue
                    try:
                        values = method(joint_indices=indices)
                    except TypeError:
                        try:
                            values = method()
                        except Exception:
                            continue
                    except Exception:
                        continue
                    if _store(result_name, values):
                        break
                if result_name in result:
                    break

        try:
            physics_view = articulation._articulation_view._physics_view
        except Exception:
            physics_view = None
        if physics_view is not None:
            for result_name, method_name in (
                ('stiffness', 'get_dof_stiffnesses'),
                ('damping', 'get_dof_dampings'),
                ('max_force', 'get_dof_max_forces'),
            ):
                method = getattr(physics_view, method_name, None)
                if callable(method):
                    try:
                        _store(result_name, method())
                    except Exception:
                        pass
        return result

    @staticmethod
    def _arm_joint_names(*, robot, controller=None) -> tuple[str, ...]:
        configured_names = tuple(getattr(controller, 'joint_names', None) or ())
        if len(configured_names) in {6, 7}:
            return configured_names
        articulation = getattr(robot, 'articulation', None)
        dof_names = set(getattr(articulation, 'dof_names', []) or [])
        for names in _SUPPORTED_ARM_JOINT_NAMES:
            if all(name in dof_names for name in names):
                return names
        robot_type = str(getattr(getattr(robot, 'config', None), 'type', '')).lower()
        if 'franka' in robot_type or 'panda' in robot_type:
            return _FRANKA_ARM_JOINT_NAMES
        return _UR5E_ARM_JOINT_NAMES

    @staticmethod
    def _coerce_arm_q(
        joint_positions,
        *,
        bound_revolute: bool = True,
        joint_count: int | None = None,
    ) -> np.ndarray | None:
        if joint_positions is None:
            return None
        try:
            values = np.asarray(joint_positions, dtype=float).reshape(-1)
        except Exception:
            return None
        if joint_count is None:
            joint_count = 7 if values.shape[0] >= 7 else 6
        if joint_count not in {6, 7} or values.shape[0] < joint_count:
            return None
        values = values[:joint_count]
        if not np.all(np.isfinite(values)):
            return None
        if not bound_revolute:
            return values.copy()
        return UR5eAssemblyAtomicSkillAdapter._bounded_revolute_joint_values(values)

    @staticmethod
    def _bounded_revolute_joint_values(values) -> np.ndarray:
        values = np.asarray(values, dtype=float).copy()
        wrapped = (values + np.pi) % (2.0 * np.pi) - np.pi
        # UR wrists can legitimately cross multiple pi turns during continuous
        # motion.  Wrapping too early makes the cached command state jump across
        # branches, so only fold back values that are far outside a nearby branch.
        return np.where(np.abs(values) > 4.0 * np.pi + 0.25, wrapped, values)

    def _remember_arm_command(self, task, robot_name: str, command_q: np.ndarray) -> None:
        joint_positions = self._coerce_arm_q(command_q, bound_revolute=False)
        if joint_positions is not None:
            self._last_arm_command_q[robot_name] = joint_positions
            task_cache = getattr(task, '_ur5e_plumbers_last_arm_command_q', None)
            if task_cache is None:
                task_cache = {}
                setattr(task, '_ur5e_plumbers_last_arm_command_q', task_cache)
            task_cache[robot_name] = joint_positions.copy()

    def _last_command_q(self, *, task, robot_name: str) -> np.ndarray | None:
        task_cache = getattr(task, '_ur5e_plumbers_last_arm_command_q', None)
        last_q = None
        if isinstance(task_cache, dict):
            last_q = task_cache.get(robot_name)
        if last_q is None:
            last_q = self._last_arm_command_q.get(robot_name)
        return self._coerce_arm_q(last_q, bound_revolute=False)

    def _command_reference_q(
        self, *, task, robot_name: str, current_q: np.ndarray | None, spec: dict
    ) -> np.ndarray | None:
        current_q = self._coerce_arm_q(current_q)
        reference_mode = str(spec.get('ik_reference_mode', spec.get('reference_mode', ''))).strip().lower()
        last_q = self._last_command_q(task=task, robot_name=robot_name)
        if reference_mode in {'current', 'current_q', 'actual', 'measured'}:
            if current_q is not None and last_q is not None:
                return self._unwrap_to_reference(
                    target_q=current_q,
                    reference_q=last_q,
                    preferred_abs_limit=None,
                    hard_preferred_abs_limit=False,
                )
            return current_q
        if reference_mode in {'hybrid', 'bounded_command', 'command_bounded'}:
            if last_q is None:
                return current_q
            if current_q is None:
                return last_q
            measured_reference = self._unwrap_to_reference(
                target_q=current_q,
                reference_q=last_q,
                preferred_abs_limit=None,
                hard_preferred_abs_limit=False,
            )
            try:
                raw_limit = np.asarray(
                    spec.get('ik_reference_command_max_tracking_error', 0.12),
                    dtype=float,
                ).reshape(-1)
            except Exception:
                return measured_reference
            if raw_limit.size == 0 or not np.all(np.isfinite(raw_limit)):
                return measured_reference
            if raw_limit.size == 1:
                limits = np.full(last_q.shape[0], float(raw_limit[0]), dtype=float)
            else:
                limits = np.full(last_q.shape[0], float(raw_limit[-1]), dtype=float)
                limits[: min(raw_limit.size, last_q.shape[0])] = raw_limit[: last_q.shape[0]]
            if np.any(limits <= 0.0):
                return measured_reference
            if np.any(np.abs(last_q - measured_reference) > limits):
                return measured_reference
            return last_q
        if last_q is None:
            return current_q
        if current_q is None:
            return last_q
        reset_threshold = spec.get('ik_reference_reset_threshold')
        if reset_threshold is not None:
            try:
                if float(np.max(np.abs(last_q - current_q))) > float(reset_threshold):
                    return current_q
            except Exception:
                return current_q
        return last_q

    def _continuous_command_q(
        self,
        *,
        task,
        robot_name: str,
        command_q: np.ndarray,
        spec: dict,
    ) -> np.ndarray:
        command_q = self._coerce_arm_q(command_q, bound_revolute=False)
        if command_q is None:
            return command_q
        if not bool(spec.get('enforce_continuous_joint_commands', True)):
            return command_q
        last_q = self._last_command_q(task=task, robot_name=robot_name)
        if last_q is None:
            return command_q
        command_q = self._unwrap_to_reference(
            target_q=command_q,
            reference_q=last_q,
            preferred_abs_limit=None,
            hard_preferred_abs_limit=False,
        )
        max_command_step = self._command_joint_step_limits(spec=spec, joint_count=command_q.shape[0])
        if max_command_step is None:
            return command_q
        return self._limited_joint_target(
            current_q=last_q,
            target_q=command_q,
            max_joint_step=max_command_step,
        )

    @staticmethod
    def _command_joint_step_limits(*, spec: dict, joint_count: int) -> np.ndarray | float | None:
        explicit = spec.get('max_command_joint_step')
        raw_limit = explicit if explicit is not None else spec.get('max_joint_step')
        if raw_limit is None:
            return None
        try:
            limit_values = np.asarray(raw_limit, dtype=float).reshape(-1)
        except Exception:
            return None
        if limit_values.size == 0 or not np.all(np.isfinite(limit_values)):
            return None
        if limit_values.size == 1:
            scalar_limit = float(limit_values[0])
            if scalar_limit <= 0.0:
                return scalar_limit
            if explicit is None:
                default_cap = float(spec.get('default_max_command_joint_step', 0.08))
                if default_cap > 0.0:
                    scalar_limit = min(scalar_limit, default_cap)
                limits = np.full(int(joint_count), scalar_limit, dtype=float)
                wrist_cap = float(spec.get('default_max_command_wrist_joint_step', 0.025))
                if wrist_cap > 0.0 and limits.shape[0] >= 6:
                    limits[3:] = np.minimum(limits[3:], wrist_cap)
                return limits
            return scalar_limit
        limits = np.full(int(joint_count), float(limit_values[-1]), dtype=float)
        copy_count = min(int(joint_count), int(limit_values.size))
        limits[:copy_count] = limit_values[:copy_count]
        return limits

    @staticmethod
    def _limit_command_to_measured_state(
        *,
        current_q: np.ndarray | None,
        command_q: np.ndarray,
        spec: dict,
    ) -> np.ndarray:
        command_q = np.asarray(command_q, dtype=float)
        if not bool(spec.get('limit_command_to_measured_state', True)):
            return command_q
        current_q = UR5eAssemblyAtomicSkillAdapter._coerce_arm_q(current_q, bound_revolute=False)
        if current_q is None or current_q.shape != command_q.shape:
            return command_q
        command_q = UR5eAssemblyAtomicSkillAdapter._unwrap_to_reference(
            target_q=command_q,
            reference_q=current_q,
            preferred_abs_limit=None,
            hard_preferred_abs_limit=False,
        )
        raw_limit = spec.get('max_command_tracking_error', 0.18)
        if raw_limit is None:
            return command_q
        try:
            values = np.asarray(raw_limit, dtype=float).reshape(-1)
        except Exception:
            return command_q
        if values.size == 0 or not np.all(np.isfinite(values)):
            return command_q
        if values.size == 1:
            limits = np.full(command_q.shape[0], float(values[0]), dtype=float)
        else:
            limits = np.full(command_q.shape[0], float(values[-1]), dtype=float)
            limits[: min(values.size, command_q.shape[0])] = values[: command_q.shape[0]]
        wrist_limit = spec.get('max_wrist_command_tracking_error', 0.12)
        if wrist_limit is not None and limits.shape[0] >= 6:
            try:
                wrist_limit = float(wrist_limit)
                if np.isfinite(wrist_limit) and wrist_limit > 0.0:
                    limits[3:] = np.minimum(limits[3:], wrist_limit)
            except Exception:
                pass
        if np.any(limits <= 0.0):
            return command_q
        return UR5eAssemblyAtomicSkillAdapter._limited_joint_target(
            current_q=current_q,
            target_q=command_q,
            max_joint_step=limits,
        )

    @staticmethod
    def _ik_branch_jump_detected(*, reference_q: np.ndarray, target_q: np.ndarray, spec: dict) -> bool:
        reference_q = np.asarray(reference_q, dtype=float)
        target_q = np.asarray(target_q, dtype=float)
        if reference_q.shape != target_q.shape:
            return False
        max_joint_step = float(spec.get('max_joint_step', spec.get('close_gate_max_joint_step', 0.035)))
        default_limit = max(0.18, max_joint_step * 4.0)
        jump_limit = float(spec.get('ik_branch_jump_limit', default_limit))
        if jump_limit <= 0.0:
            return False
        default_branch_joint_indices = [0, 3, reference_q.shape[0] - 1]
        branch_joint_indices = spec.get('ik_branch_guard_joint_indices', default_branch_joint_indices)
        try:
            indices = [int(index) for index in branch_joint_indices]
        except Exception:
            indices = default_branch_joint_indices
        deltas = []
        for index in indices:
            if 0 <= index < reference_q.shape[0]:
                deltas.append(abs(float(target_q[index] - reference_q[index])))
        if not deltas:
            return False
        return max(deltas) > jump_limit

    def _solve_ik(
        self,
        *,
        task,
        robot_name: str,
        target_pose: dict,
        warm_start: np.ndarray | None = None,
        spec: dict | None = None,
    ) -> np.ndarray | None:
        robot = task.robots.get(robot_name)
        if robot is None:
            return None
        ik_controller = robot.controllers.get(_ARM_IK_CONTROLLER)
        if ik_controller is None or not hasattr(ik_controller, '_kinematics_solver'):
            return None
        spec = spec or {}
        target_position = np.asarray(target_pose['position'], dtype=float) / ik_controller._robot_scale  # noqa: SLF001
        target_orientation = np.asarray(target_pose['orientation'], dtype=float)
        warm_start = self._coerce_arm_q(warm_start)
        position_tolerance, orientation_tolerance = self._ik_solver_tolerances(spec)
        try:
            ik_base_pose = ik_controller.get_ik_base_world_pose()
            ik_controller._kinematics_solver.set_robot_base_pose(  # noqa: SLF001
                robot_position=ik_base_pose[0] / ik_controller._robot_scale,  # noqa: SLF001
                robot_orientation=ik_base_pose[1],
            )
            used_warm_start = warm_start is not None and bool(spec.get('use_command_warm_start', True))
            if used_warm_start:
                solver_wrapper = ik_controller._kinematics_solver  # noqa: SLF001
                raw_solver = None
                get_raw_solver = getattr(solver_wrapper, 'get_kinematics_solver', None)
                if callable(get_raw_solver):
                    raw_solver = get_raw_solver()
                if raw_solver is None:
                    raw_solver = getattr(solver_wrapper, '_kinematics_solver', None)
                if raw_solver is None:
                    raw_solver = getattr(solver_wrapper, '_kinematics', None)
                get_ee_frame = getattr(solver_wrapper, 'get_end_effector_frame', None)
                ee_frame = get_ee_frame() if callable(get_ee_frame) else getattr(solver_wrapper, '_ee_frame', None)
                if raw_solver is not None and ee_frame is not None:
                    ik_result, success = raw_solver.compute_inverse_kinematics(
                        ee_frame,
                        target_position,
                        target_orientation,
                        warm_start=warm_start,
                        position_tolerance=position_tolerance,
                        orientation_tolerance=orientation_tolerance,
                    )
                    if success and ik_result is not None:
                        ik_result = np.asarray(ik_result, dtype=float)
                        if np.all(np.isfinite(ik_result)):
                            return ik_result
            if used_warm_start and bool(spec.get('require_warm_start_ik', False)):
                return None
            goal_action, success = ik_controller._kinematics_solver.compute_inverse_kinematics(  # noqa: SLF001
                target_position=target_position,
                target_orientation=target_orientation,
            )
        except Exception:
            return None
        if not success or goal_action is None or goal_action.joint_positions is None:
            return None
        joint_positions = np.asarray(goal_action.joint_positions, dtype=float)
        if not np.all(np.isfinite(joint_positions)):
            return None
        return joint_positions

    @staticmethod
    def _ik_solver_tolerances(spec: dict) -> tuple[float | None, float | None]:
        """Keep Lula's convergence deadband smaller than one Cartesian servo step."""

        position_tolerance = spec.get('ik_position_tolerance')
        orientation_tolerance = spec.get('ik_orientation_tolerance')
        if not bool(spec.get('cartesian_servo', False)):
            return position_tolerance, orientation_tolerance

        if position_tolerance is None:
            position_step = float(spec.get('cartesian_position_step', 0.01))
            if np.isfinite(position_step) and position_step > 0.0:
                position_tolerance = max(min(position_step * 0.05, 5e-4), 1e-5)
        if orientation_tolerance is None:
            orientation_step = float(spec.get('cartesian_orientation_step', 0.01))
            if np.isfinite(orientation_step) and orientation_step > 0.0:
                orientation_tolerance = max(min(orientation_step * 0.1, 2e-3), 1e-4)
        return position_tolerance, orientation_tolerance

    @staticmethod
    def _limited_joint_target(*, current_q: np.ndarray, target_q: np.ndarray, max_joint_step) -> np.ndarray:
        current_q = np.asarray(current_q, dtype=float)
        target_q = np.asarray(target_q, dtype=float)
        if not np.all(np.isfinite(current_q)):
            return target_q
        if not np.all(np.isfinite(target_q)):
            return current_q
        if current_q.shape != target_q.shape:
            return target_q
        try:
            step_limits = np.asarray(max_joint_step, dtype=float).reshape(-1)
        except Exception:
            return target_q
        if step_limits.size == 0 or not np.all(np.isfinite(step_limits)):
            return target_q
        if step_limits.size == 1:
            scalar_limit = float(step_limits[0])
            if scalar_limit <= 0.0:
                return target_q
            delta = target_q - current_q
            max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
            if max_abs <= scalar_limit:
                return target_q
            return current_q + delta * (scalar_limit / max_abs)
        if step_limits.size < current_q.shape[0]:
            padded = np.full(current_q.shape[0], float(step_limits[-1]), dtype=float)
            padded[: step_limits.size] = step_limits
            step_limits = padded
        else:
            step_limits = step_limits[: current_q.shape[0]]
        if np.any(step_limits <= 0.0):
            return target_q
        delta = target_q - current_q
        if np.all(np.abs(delta) <= step_limits):
            return target_q
        nonzero = np.abs(delta) > 1e-12
        scale = float(np.min(step_limits[nonzero] / np.abs(delta[nonzero]))) if np.any(nonzero) else 1.0
        return current_q + delta * min(scale, 1.0)

    @staticmethod
    def _unwrap_to_reference(
        *,
        target_q: np.ndarray,
        reference_q: np.ndarray | None,
        preferred_abs_limit=None,
        hard_preferred_abs_limit: bool = True,
    ) -> np.ndarray:
        target_q = np.asarray(target_q, dtype=float).copy()
        if reference_q is None:
            return target_q
        reference_q = np.asarray(reference_q, dtype=float)
        if target_q.shape != reference_q.shape:
            return target_q

        period = 2.0 * np.pi
        preferred_abs_limit = None if preferred_abs_limit is None else float(preferred_abs_limit)
        for joint_index, target_value in enumerate(target_q):
            candidates = target_value + np.arange(-4, 5, dtype=float) * period
            nearest_candidate = candidates[int(np.argmin(np.abs(candidates - reference_q[joint_index])))]
            if hard_preferred_abs_limit and preferred_abs_limit is not None and preferred_abs_limit > 0.0:
                bounded = candidates[np.abs(candidates) <= preferred_abs_limit]
                if bounded.size:
                    bounded_candidate = bounded[int(np.argmin(np.abs(bounded - reference_q[joint_index])))]
                    nearest_delta = abs(float(nearest_candidate - reference_q[joint_index]))
                    bounded_delta = abs(float(bounded_candidate - reference_q[joint_index]))
                    # A preferred range must not turn an equivalent +/-pi wrap
                    # into an almost 2*pi command jump.
                    if bounded_delta <= max(0.5, nearest_delta * 4.0):
                        candidates = bounded
            cost = np.abs(candidates - reference_q[joint_index])
            if not hard_preferred_abs_limit and preferred_abs_limit is not None and preferred_abs_limit > 0.0:
                cost = cost + 2.0 * np.maximum(np.abs(candidates) - preferred_abs_limit, 0.0)
            target_q[joint_index] = candidates[int(np.argmin(cost))]
        return target_q

    def _hold_joint_action(self, *, task, robot_name: str) -> OrderedDict:
        action = OrderedDict()
        current_q = self._current_arm_q(task, robot_name)
        hold_q = self._command_reference_q(task=task, robot_name=robot_name, current_q=current_q, spec={})
        if hold_q is not None:
            action[_ARM_JOINT_CONTROLLER] = [hold_q.tolist()]
        return action

    def _move_arm_to_joint_positions_action(
        self,
        *,
        phase_key,
        task,
        robot_name: str,
        skill_name: str,
        spec: dict,
    ) -> OrderedDict:
        current_q = self._current_arm_q(task, robot_name)
        if current_q is None:
            return self._failure_or_hold(task, robot_name, spec, 'current_joint_state_unavailable')

        target_q = self._coerce_arm_q(
            spec.get('joint_positions'),
            bound_revolute=False,
            joint_count=current_q.shape[0],
        )
        if target_q is None or target_q.shape != current_q.shape:
            return self._failure_or_hold(
                task,
                robot_name,
                spec,
                'invalid_joint_target',
                diagnostics={'joint_positions': spec.get('joint_positions')},
            )

        reference_q = self._command_reference_q(
            task=task,
            robot_name=robot_name,
            current_q=current_q,
            spec=spec,
        )
        if reference_q is None or reference_q.shape != target_q.shape:
            reference_q = current_q
        target_q = self._unwrap_to_reference(
            target_q=target_q,
            reference_q=reference_q,
            preferred_abs_limit=spec.get('preferred_joint_abs_limit'),
            hard_preferred_abs_limit=False,
        )
        step_limits = self._command_joint_step_limits(
            spec=spec,
            joint_count=current_q.shape[0],
        )
        if step_limits is None:
            step_limits = float(spec.get('max_joint_step', 0.020))
        command_q = self._limited_joint_target(
            current_q=reference_q,
            target_q=target_q,
            max_joint_step=step_limits,
        )
        command_q = self._continuous_command_q(
            task=task,
            robot_name=robot_name,
            command_q=command_q,
            spec=spec,
        )
        command_q = self._limit_command_to_measured_state(
            current_q=current_q,
            command_q=command_q,
            spec=spec,
        )

        action = OrderedDict()
        action[_ARM_JOINT_CONTROLLER] = [command_q.tolist()]
        action[_GRIPPER_CONTROLLER] = [
            self._gripper_command_value(
                task=task,
                robot_name=robot_name,
                command=spec.get('gripper_command', 'open'),
            )
        ]
        self._remember_arm_command(task, robot_name, command_q)

        joint_error = float(np.max(np.abs(target_q - current_q)))
        tolerance = float(spec.get('joint_position_tolerance', 0.025))
        state = self._completion_state.setdefault(phase_key, {})
        if joint_error <= tolerance:
            state['joint_target_stable_steps'] = int(state.get('joint_target_stable_steps', 0)) + 1
        else:
            state['joint_target_stable_steps'] = 0
        stable_steps = int(state['joint_target_stable_steps'])
        required_stable_steps = max(int(spec.get('joint_target_stable_steps', 8)), 1)
        if stable_steps >= required_stable_steps:
            self._mark_complete(
                task=task,
                robot_name=robot_name,
                skill_name=skill_name,
                detail={
                    'target_joint_positions': target_q.tolist(),
                    'joint_error': joint_error,
                    'joint_position_tolerance': tolerance,
                    'stable_steps': stable_steps,
                    'required_stable_steps': required_stable_steps,
                },
            )
        return action

    def _preshape_gripper_action(
        self,
        *,
        phase_key,
        task,
        robot_name: str,
        skill_name: str,
        spec: dict,
        tracked_objects: dict,
    ) -> OrderedDict:
        action = self._hold_joint_action(task=task, robot_name=robot_name)
        openness = self._gripper_value(spec.get('gripper_openness', spec.get('gripper_command', 1.0)))
        action[_GRIPPER_CONTROLLER] = [openness]

        state = self._preshape_state.setdefault(phase_key, {'stable_steps': 0})
        gripper_q = self._current_gripper_q(task=task, robot_name=robot_name)
        open_q, closed_q = self._gripper_open_closed_q(task=task, robot_name=robot_name)
        target_q = None
        if open_q is not None and closed_q is not None:
            target_q = float(open_q) + (1.0 - openness) * (float(closed_q) - float(open_q))
        gripper_ready = bool(
            gripper_q is not None
            and target_q is not None
            and abs(float(gripper_q) - target_q) <= float(spec.get('gripper_position_tolerance', 0.015))
        )

        motion_detail = {'valid': True, 'checked': False}
        motion_ready = True
        if bool(spec.get('require_object_static', True)):
            motion_detail = self._object_motion_detail(
                task=task,
                object_name=self._object_name_from_spec(spec),
                tracked_objects=tracked_objects,
            )
            motion_detail['checked'] = True
            linear_speed = motion_detail.get('linear_speed')
            angular_speed = motion_detail.get('angular_speed')
            velocity_ready = bool(
                motion_detail.get('valid')
                and linear_speed is not None
                and angular_speed is not None
                and float(linear_speed) <= float(spec.get('max_object_linear_speed', 0.01))
                and float(angular_speed) <= float(spec.get('max_object_angular_speed', 0.10))
            )
            pose_stable_ready = bool(
                spec.get('allow_pose_stable_override', True) and motion_detail.get('pose_stable_override') is True
            )
            motion_ready = bool(velocity_ready or pose_stable_ready)

        if gripper_ready and motion_ready:
            state['stable_steps'] = int(state.get('stable_steps', 0)) + 1
        else:
            state['stable_steps'] = 0
        required_stable_steps = max(int(spec.get('stable_steps', 12)), 1)
        detail = {
            'gripper_openness': openness,
            'gripper_joint_position': None if gripper_q is None else float(gripper_q),
            'target_gripper_joint_position': target_q,
            'gripper_ready': gripper_ready,
            'motion_ready': motion_ready,
            'motion_detail': motion_detail,
            'stable_steps': int(state['stable_steps']),
            'required_stable_steps': required_stable_steps,
        }
        if int(state['stable_steps']) >= required_stable_steps:
            self._mark_complete(
                task=task,
                robot_name=robot_name,
                skill_name=skill_name,
                detail=detail,
            )
        elif int(getattr(task, 'phase_step_counter', 0)) >= int(spec.get('preshape_timeout_steps', 240)):
            return self._failure_or_hold(
                task,
                robot_name,
                spec,
                'preshape_not_stable',
                diagnostics=detail,
            )
        return action

    @staticmethod
    def _gripper_value(command) -> float:
        if isinstance(command, str):
            lowered = command.strip().lower()
            if lowered == 'open':
                return 1.0
            if lowered == 'close':
                return 0.0
        try:
            return float(np.clip(float(command), 0.0, 1.0))
        except Exception:
            return 1.0

    def _gripper_command_value(self, *, task, robot_name: str, command) -> float:
        if isinstance(command, str) and command.strip().lower() in {'contact_hold', 'hold_contact', 'grasp_hold'}:
            hold_openness = self._last_gripper_hold_openness(task=task, robot_name=robot_name)
            if hold_openness is not None:
                return float(hold_openness)
            return 0.0
        return self._gripper_value(command)

    @staticmethod
    def _last_gripper_hold_openness(*, task, robot_name: str) -> float | None:
        cache = getattr(task, '_ur5e_plumbers_gripper_hold_openness', None)
        if not isinstance(cache, dict):
            return None
        value = cache.get(robot_name)
        if value is None:
            return None
        try:
            return float(np.clip(float(value), 0.0, 1.0))
        except Exception:
            return None

    @staticmethod
    def _remember_gripper_hold_openness(*, task, robot_name: str, openness: float) -> None:
        cache = getattr(task, '_ur5e_plumbers_gripper_hold_openness', None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(task, '_ur5e_plumbers_gripper_hold_openness', cache)
        cache[robot_name] = float(np.clip(float(openness), 0.0, 1.0))

    @staticmethod
    def _gripper_openness_from_q(
        *,
        gripper_q: float | None,
        open_q: float | None,
        closed_q: float | None,
        squeeze_margin: float,
    ) -> float | None:
        if gripper_q is None or open_q is None or closed_q is None:
            return None
        denom = float(open_q) - float(closed_q)
        if abs(denom) < 1e-8:
            return None
        openness = (float(gripper_q) - float(closed_q)) / denom
        return float(np.clip(openness - float(squeeze_margin), 0.0, 1.0))

    @staticmethod
    def _object_name_from_spec(spec: dict) -> str | None:
        object_name = spec.get('held_object') or spec.get('object') or spec.get('object_name')
        if object_name is None:
            return None
        return str(object_name)

    def _object_motion_detail(
        self,
        *,
        task,
        object_name: str | None,
        tracked_objects: dict,
    ) -> dict[str, Any]:
        if not object_name:
            return {'valid': False, 'reason': 'missing_object_name'}
        state = tracked_objects.get(object_name, {}) if isinstance(tracked_objects, dict) else {}
        linear_speed = state.get('linear_speed')
        angular_speed = state.get('angular_speed')
        linear_velocity = state.get('linear_velocity')
        angular_velocity = state.get('angular_velocity')
        is_static = state.get('is_static')
        pose_stable_override = state.get('pose_stable_override')
        pose_stability_position_drift = state.get('pose_stability_position_drift')
        pose_stability_orientation_drift = state.get('pose_stability_orientation_drift')

        if linear_speed is None and linear_velocity is not None:
            try:
                linear_speed = float(np.linalg.norm(np.asarray(linear_velocity, dtype=float)))
            except Exception:
                linear_speed = None
        if angular_speed is None and angular_velocity is not None:
            try:
                angular_speed = float(np.linalg.norm(np.asarray(angular_velocity, dtype=float)))
            except Exception:
                angular_speed = None

        if (
            linear_speed is None
            or angular_speed is None
            or pose_stable_override is None
            or pose_stability_position_drift is None
            or pose_stability_orientation_drift is None
        ):
            velocity_fn = getattr(task, '_object_velocity_metrics', None)
            if callable(velocity_fn):
                try:
                    metrics = velocity_fn(object_name)
                    linear_speed = metrics.get('linear_speed', linear_speed)
                    angular_speed = metrics.get('angular_speed', angular_speed)
                    linear_velocity = metrics.get('linear_velocity', linear_velocity)
                    angular_velocity = metrics.get('angular_velocity', angular_velocity)
                    is_static = metrics.get('is_static', is_static)
                    pose_stable_override = metrics.get('pose_stable_override', pose_stable_override)
                    pose_stability_position_drift = metrics.get(
                        'pose_stability_position_drift',
                        pose_stability_position_drift,
                    )
                    pose_stability_orientation_drift = metrics.get(
                        'pose_stability_orientation_drift',
                        pose_stability_orientation_drift,
                    )
                except Exception:
                    pass

        valid = linear_speed is not None and angular_speed is not None
        return {
            'valid': bool(valid),
            'object': object_name,
            'linear_speed': None if linear_speed is None else float(linear_speed),
            'angular_speed': None if angular_speed is None else float(angular_speed),
            'linear_velocity': linear_velocity,
            'angular_velocity': angular_velocity,
            'is_static': None if is_static is None else bool(is_static),
            'pose_stable_override': None if pose_stable_override is None else bool(pose_stable_override),
            'pose_stability_position_drift': (
                None if pose_stability_position_drift is None else float(pose_stability_position_drift)
            ),
            'pose_stability_orientation_drift': (
                None if pose_stability_orientation_drift is None else float(pose_stability_orientation_drift)
            ),
        }

    @staticmethod
    def _pose_history_velocity_override_ready(
        *,
        spec: dict,
        motion_detail: dict,
    ) -> bool:
        if not bool(spec.get('target_object_allow_pose_history_velocity_override', False)):
            return False
        position_drift = motion_detail.get('pose_stability_position_drift')
        orientation_drift = motion_detail.get('pose_stability_orientation_drift')
        if position_drift is None or orientation_drift is None:
            return False
        position_tolerance = float(spec.get('pose_history_velocity_override_position_tolerance', 0.0005))
        orientation_tolerance = float(spec.get('pose_history_velocity_override_orientation_tolerance', 0.01))
        if (
            not np.isfinite(position_tolerance)
            or position_tolerance <= 0.0
            or not np.isfinite(orientation_tolerance)
            or orientation_tolerance <= 0.0
        ):
            return False
        return bool(
            motion_detail.get('valid')
            and motion_detail.get('is_static') is True
            and motion_detail.get('pose_stable_override') is True
            and np.isfinite(float(position_drift))
            and np.isfinite(float(orientation_drift))
            and float(position_drift) <= position_tolerance
            and float(orientation_drift) <= orientation_tolerance
        )

    def _object_tcp_slip_failure(
        self,
        *,
        phase_key,
        task,
        robot_name: str,
        spec: dict,
        tracked_objects: dict,
        current_pose: dict | None,
    ) -> dict[str, Any] | None:
        max_slip = spec.get('max_object_tcp_slip')
        if max_slip is None or current_pose is None:
            return None
        object_name = self._object_name_from_spec(spec)
        if object_name is None:
            return None
        object_pose = self._object_pose(
            task=task,
            object_name=object_name,
            tracked_objects=tracked_objects,
        )
        if object_pose is None:
            return None

        current_relative_world = np.asarray(object_pose['position'], dtype=float) - np.asarray(
            current_pose['position'], dtype=float
        )
        current_relative_tcp = quat_rotate(
            self._quat_conjugate(current_pose['orientation']),
            current_relative_world,
        )
        if current_relative_tcp.shape != (3,) or not np.all(np.isfinite(current_relative_tcp)):
            return None
        state = self._grasp_slip_state.setdefault(phase_key, {})
        if 'initial_relative_tcp_position' not in state:
            state['initial_relative_tcp_position'] = current_relative_tcp.copy()
            state['initial_relative_world_position'] = current_relative_world.copy()
            state['initial_object_position'] = np.asarray(object_pose['position'], dtype=float).copy()
            state['initial_tcp_position'] = np.asarray(current_pose['position'], dtype=float).copy()
            state['initial_tcp_orientation'] = normalize_quat(current_pose['orientation']).copy()
            return None

        initial_relative_tcp = np.asarray(state['initial_relative_tcp_position'], dtype=float)
        slip = float(np.linalg.norm(current_relative_tcp - initial_relative_tcp))
        threshold = float(max_slip)
        if slip <= threshold:
            return None
        initial_relative_world = np.asarray(
            state.get('initial_relative_world_position', current_relative_world),
            dtype=float,
        )
        return {
            'object': object_name,
            'robot': robot_name,
            'slip': slip,
            'slip_frame': 'tcp',
            'max_object_tcp_slip': threshold,
            'initial_relative_position': initial_relative_tcp.tolist(),
            'current_relative_position': current_relative_tcp.tolist(),
            'initial_relative_tcp_position': initial_relative_tcp.tolist(),
            'current_relative_tcp_position': current_relative_tcp.tolist(),
            'initial_relative_world_position': initial_relative_world.tolist(),
            'current_relative_world_position': current_relative_world.tolist(),
            'world_relative_delta': float(np.linalg.norm(current_relative_world - initial_relative_world)),
            'initial_tcp_orientation': np.asarray(state.get('initial_tcp_orientation'), dtype=float).tolist(),
            'current_tcp_orientation': normalize_quat(current_pose['orientation']).tolist(),
            'initial_object_position': np.asarray(state.get('initial_object_position'), dtype=float).tolist(),
            'initial_tcp_position': np.asarray(state.get('initial_tcp_position'), dtype=float).tolist(),
            'current_object_position': np.asarray(object_pose['position'], dtype=float).tolist(),
            'current_tcp_position': np.asarray(current_pose['position'], dtype=float).tolist(),
        }

    @staticmethod
    def _close_object_motion_abort(
        *,
        close_detail: dict[str, Any],
        spec: dict,
        close_elapsed_steps: int,
    ) -> bool:
        if not bool(spec.get('close_abort_on_object_motion', False)):
            return False
        min_steps = int(spec.get('close_abort_min_steps', 0))
        if int(close_elapsed_steps) < min_steps:
            return False
        motion_detail = close_detail.get('motion_detail')
        if not isinstance(motion_detail, dict) or not bool(motion_detail.get('valid')):
            return False
        linear_speed = motion_detail.get('linear_speed')
        angular_speed = motion_detail.get('angular_speed')
        max_linear_speed = spec.get('close_abort_max_object_speed', spec.get('close_contact_max_object_speed'))
        max_angular_speed = spec.get('close_abort_max_angular_speed', spec.get('close_contact_max_angular_speed'))
        linear_abort = (
            max_linear_speed is not None and linear_speed is not None and float(linear_speed) > float(max_linear_speed)
        )
        angular_abort = (
            max_angular_speed is not None
            and angular_speed is not None
            and float(angular_speed) > float(max_angular_speed)
        )
        return bool(linear_abort or angular_abort)

    @staticmethod
    def _update_close_recenter_offset(
        *,
        state: dict[str, Any],
        close_detail: dict[str, Any],
        spec: dict,
        close_ready: bool,
    ) -> dict[str, Any]:
        """Bias the TCP toward a prematurely contacting finger during closure."""
        enabled = bool(spec.get('close_gate_recenter_single_finger_contact', False))
        result: dict[str, Any] = {
            'enabled': enabled,
            'updated': False,
            'offset_world': np.asarray(
                state.get('recenter_offset_world', np.zeros(3, dtype=float)),
                dtype=float,
            ).tolist(),
        }
        if not enabled or close_ready:
            return result

        contact_detail = close_detail.get('contact_detail') or {}
        metrics = contact_detail.get('contact_metrics') or {}
        left = metrics.get('left_finger') or {}
        right = metrics.get('right_finger') or {}

        def surface_gap(finger: dict[str, Any]) -> float | None:
            local_contact = finger.get('local_contact') or {}
            value = local_contact.get('best_surface_gap', finger.get('surface_gap'))
            if value is None:
                return None
            value = float(value)
            return value if math.isfinite(value) else None

        left_gap = surface_gap(left)
        right_gap = surface_gap(right)
        result.update({'left_gap': left_gap, 'right_gap': right_gap})
        if left_gap is None or right_gap is None:
            state['recenter_single_finger_steps'] = 0
            state.pop('recenter_single_finger_side', None)
            return result

        contact_distance = float(
            spec.get(
                'close_gate_recenter_contact_distance',
                spec.get('finger_contact_distance', 0.006),
            )
        )
        minimum_gap_imbalance = float(spec.get('close_gate_recenter_min_gap_imbalance', 0.004))
        left_near = left_gap <= contact_distance
        right_near = right_gap <= contact_distance
        if left_near == right_near or abs(left_gap - right_gap) < minimum_gap_imbalance:
            state['recenter_single_finger_steps'] = 0
            state.pop('recenter_single_finger_side', None)
            return result

        side = 'left' if left_near else 'right'
        finger = left if left_near else right
        if state.get('recenter_single_finger_side') == side:
            stable_steps = int(state.get('recenter_single_finger_steps', 0)) + 1
        else:
            stable_steps = 1
        state['recenter_single_finger_side'] = side
        state['recenter_single_finger_steps'] = stable_steps
        required_steps = max(
            int(spec.get('close_gate_recenter_stable_steps', 2)),
            1,
        )
        result.update(
            {
                'single_finger_side': side,
                'stable_steps': stable_steps,
                'required_stable_steps': required_steps,
            }
        )
        if stable_steps < required_steps:
            return result

        local_contact = finger.get('local_contact') or {}
        axis_name = local_contact.get('best_axis')
        local_point = local_contact.get('local_point')
        contact_box_orientation = metrics.get('contact_box_orientation')
        if axis_name not in {'x', 'y', 'z'} or local_point is None or contact_box_orientation is None:
            result['reason'] = 'contact_frame_unavailable'
            return result
        axis_index = {'x': 0, 'y': 1, 'z': 2}[str(axis_name)]
        signed_coordinate = float(local_point[axis_index])
        if not math.isfinite(signed_coordinate) or abs(signed_coordinate) <= 1e-9:
            result['reason'] = 'contact_side_unavailable'
            return result

        local_direction = np.zeros(3, dtype=float)
        local_direction[axis_index] = math.copysign(1.0, signed_coordinate)
        world_direction = quat_rotate(
            normalize_quat(np.asarray(contact_box_orientation, dtype=float)),
            local_direction,
        )
        world_direction = np.asarray(world_direction, dtype=float)
        norm = float(np.linalg.norm(world_direction))
        if norm <= 1e-9 or not np.all(np.isfinite(world_direction)):
            result['reason'] = 'contact_direction_unavailable'
            return result
        world_direction /= norm

        step = max(float(spec.get('close_gate_recenter_step', 0.00075)), 0.0)
        maximum_offset = max(
            float(spec.get('close_gate_recenter_max_offset', 0.025)),
            0.0,
        )
        previous = np.asarray(
            state.get('recenter_offset_world', np.zeros(3, dtype=float)),
            dtype=float,
        )
        if float(np.linalg.norm(previous)) > 1e-12 and not bool(state.get('recenter_target_ready', False)):
            result['reason'] = 'previous_offset_servo_in_progress'
            return result
        updated = previous + world_direction * step
        updated_norm = float(np.linalg.norm(updated))
        if maximum_offset > 0.0 and updated_norm > maximum_offset:
            updated *= maximum_offset / updated_norm
        state['recenter_offset_world'] = updated
        result.update(
            {
                'updated': bool(np.linalg.norm(updated - previous) > 1e-12),
                'axis': axis_name,
                'step': step,
                'maximum_offset': maximum_offset,
                'offset_world': updated.tolist(),
            }
        )
        return result

    def _close_until_contact_ready(
        self,
        *,
        state: dict[str, Any],
        task,
        robot_name: str,
        spec: dict,
        tracked_objects: dict,
        close_elapsed_steps: int,
        gripper_openness: float,
    ) -> tuple[bool, dict[str, Any]]:
        min_steps = max(int(spec.get('close_until_contact_min_steps', spec.get('close_ramp_steps', 24))), 0)
        required_stable_steps = max(int(spec.get('close_contact_stable_steps', 8)), 1)
        gripper_q = self._current_gripper_q(task=task, robot_name=robot_name)
        open_q, closed_q = self._gripper_open_closed_q(task=task, robot_name=robot_name)
        joint_range = None
        if open_q is not None and closed_q is not None:
            joint_range = abs(float(open_q) - float(closed_q))

        # The Robotiq drive spans about 0.8 while each Panda finger spans only
        # 0.04. Scale implicit gates to the configured joint range so a valid
        # short-stroke gripper can satisfy the same normalized grasp checks.
        default_stall_delta = 0.0015 if joint_range is None else min(0.0015, 0.05 * joint_range)
        default_blocked_margin = 0.025 if joint_range is None else min(0.025, 0.25 * joint_range)
        default_min_closure = 0.05 if joint_range is None else min(0.05, 0.10 * joint_range)
        default_target_tolerance = 0.025 if joint_range is None else min(0.025, 0.05 * joint_range)
        stall_delta = float(spec.get('close_contact_stall_joint_delta', default_stall_delta))
        blocked_margin = float(spec.get('close_contact_blocked_joint_margin', default_blocked_margin))
        min_closure = float(spec.get('close_contact_min_joint_closure', default_min_closure))
        target_tolerance = float(spec.get('close_gripper_target_tolerance', default_target_tolerance))
        hold_squeeze_margin = float(spec.get('close_contact_hold_squeeze_margin', 0.04))

        last_q = state.get('last_gripper_q')
        joint_delta = None
        if gripper_q is not None and last_q is not None:
            joint_delta = abs(float(gripper_q) - float(last_q))
        if gripper_q is not None:
            state['last_gripper_q'] = float(gripper_q)

        contact_ready = False
        contact_detail: dict[str, Any] = {'contact_checked': False}
        if bool(spec.get('use_contact_for_close_until_contact', True)):
            contact_ready, contact_detail = self._grasp_contact_ready(
                task=task,
                robot_name=robot_name,
                spec={**spec, 'require_dual_finger_contact': spec.get('require_dual_finger_contact', True)},
            )
            contact_detail['contact_checked'] = True

        blocked_before_full_close = False
        moved_from_open = False
        target_q = None
        reached_gripper_target = False
        if gripper_q is not None and open_q is not None and closed_q is not None:
            blocked_before_full_close = abs(float(gripper_q) - float(closed_q)) >= blocked_margin
            moved_from_open = abs(float(gripper_q) - float(open_q)) >= min_closure
            target_q = float(closed_q) + float(gripper_openness) * (float(open_q) - float(closed_q))
            reached_gripper_target = abs(float(gripper_q) - target_q) <= target_tolerance
        stalled = bool(joint_delta is not None and joint_delta <= stall_delta)
        contact_candidate = bool(contact_ready and moved_from_open)
        stall_contact = bool(
            bool(spec.get('use_joint_stall_for_close_until_contact', True))
            and stalled
            and blocked_before_full_close
            and moved_from_open
        )
        detected_clamp = bool(contact_candidate or stall_contact)

        configured_latch_after_stable = spec.get('close_contact_latch_after_stable')
        use_transient_hold_candidate = bool(
            configured_latch_after_stable is None and not spec.get('require_strict_physical_contact', False)
        )
        if detected_clamp:
            state['close_contact_stable_steps'] = int(state.get('close_contact_stable_steps', 0)) + 1
            if (
                use_transient_hold_candidate
                and close_elapsed_steps >= min_steps
                and state.get('hold_gripper_openness') is None
            ):
                hold_openness = self._gripper_openness_from_q(
                    gripper_q=gripper_q,
                    open_q=open_q,
                    closed_q=closed_q,
                    squeeze_margin=hold_squeeze_margin,
                )
                if hold_openness is not None:
                    max_hold_openness = spec.get('max_hold_gripper_openness', spec.get('closed_openness'))
                    if max_hold_openness is not None:
                        hold_openness = min(float(hold_openness), float(max_hold_openness))
                    state['hold_gripper_openness'] = float(hold_openness)
        else:
            state['close_contact_stable_steps'] = 0
            if use_transient_hold_candidate:
                state.pop('hold_gripper_openness', None)

        motion_ready = True
        motion_detail: dict[str, Any] = {'checked': False}
        motion_stable_steps = 0
        max_linear_speed = spec.get('close_contact_max_object_speed')
        max_angular_speed = spec.get('close_contact_max_angular_speed')
        if max_linear_speed is not None or max_angular_speed is not None:
            motion_detail = self._object_motion_detail(
                task=task,
                object_name=self._object_name_from_spec(spec),
                tracked_objects=tracked_objects,
            )
            motion_detail['checked'] = True
            linear_speed = motion_detail.get('linear_speed')
            angular_speed = motion_detail.get('angular_speed')
            linear_ok = max_linear_speed is None or (
                linear_speed is not None and float(linear_speed) <= float(max_linear_speed)
            )
            angular_ok = max_angular_speed is None or (
                angular_speed is not None and float(angular_speed) <= float(max_angular_speed)
            )
            raw_velocity_thresholds_passed = bool(linear_ok and angular_ok)
            pose_stable_override_used = bool(
                spec.get('close_contact_allow_pose_stable_override', False)
                and motion_detail.get('pose_stable_override') is True
            )
            velocity_thresholds_passed = bool(raw_velocity_thresholds_passed or pose_stable_override_used)
            motion_detail['raw_velocity_thresholds_passed'] = raw_velocity_thresholds_passed
            motion_detail['pose_stable_override_used'] = pose_stable_override_used
            motion_detail['velocity_thresholds_passed'] = velocity_thresholds_passed
            motion_ready = bool(motion_detail.get('valid') and velocity_thresholds_passed)
            if detected_clamp and motion_ready:
                state['close_contact_motion_stable_steps'] = int(state.get('close_contact_motion_stable_steps', 0)) + 1
            else:
                state['close_contact_motion_stable_steps'] = 0
            motion_stable_steps = int(state.get('close_contact_motion_stable_steps', 0))

        stable_steps = int(state.get('close_contact_stable_steps', 0))
        required_motion_stable_steps = 0
        if max_linear_speed is not None or max_angular_speed is not None:
            required_motion_stable_steps = max(int(spec.get('close_contact_motion_stable_steps', 1)), 1)
        motion_stable_ready = bool(
            required_motion_stable_steps <= 0 or motion_stable_steps >= required_motion_stable_steps
        )
        closed_target_candidate = bool(
            bool(spec.get('allow_closed_gripper_completion', False)) and reached_gripper_target and moved_from_open
        )
        if closed_target_candidate and motion_ready:
            state['close_closed_target_stable_steps'] = int(state.get('close_closed_target_stable_steps', 0)) + 1
        else:
            state['close_closed_target_stable_steps'] = 0
        closed_target_stable_steps = int(state.get('close_closed_target_stable_steps', 0))
        required_closed_target_stable_steps = max(
            int(spec.get('close_closed_target_stable_steps', required_motion_stable_steps or 1)),
            1,
        )
        closed_target_ready = bool(
            closed_target_candidate
            and motion_ready
            and closed_target_stable_steps >= required_closed_target_stable_steps
        )
        ready = bool(
            close_elapsed_steps >= min_steps
            and (stable_steps >= required_stable_steps or closed_target_ready)
            and motion_ready
            and motion_stable_ready
        )
        latch_after_stable = configured_latch_after_stable
        if latch_after_stable is None:
            max_deferred_latch_steps = max(
                int(spec.get('close_contact_deferred_latch_max_steps', 8)),
                0,
            )
            latch_after_stable = bool(
                spec.get('require_strict_physical_contact', False) and required_stable_steps <= max_deferred_latch_steps
            )
        latch_after_stable = bool(latch_after_stable)
        latch_ready = bool(ready or not latch_after_stable)
        if use_transient_hold_candidate and ready:
            hold_openness = state.get('hold_gripper_openness')
            if hold_openness is not None:
                self._remember_gripper_hold_openness(
                    task=task,
                    robot_name=robot_name,
                    openness=float(hold_openness),
                )
        elif (
            latch_ready
            and detected_clamp
            and close_elapsed_steps >= min_steps
            and state.get('hold_gripper_openness') is None
        ):
            hold_openness = self._gripper_openness_from_q(
                gripper_q=gripper_q,
                open_q=open_q,
                closed_q=closed_q,
                squeeze_margin=hold_squeeze_margin,
            )
            if hold_openness is not None:
                max_hold_openness = spec.get('max_hold_gripper_openness', spec.get('closed_openness'))
                if max_hold_openness is not None:
                    hold_openness = min(float(hold_openness), float(max_hold_openness))
                state['hold_gripper_openness'] = float(hold_openness)
                self._remember_gripper_hold_openness(
                    task=task,
                    robot_name=robot_name,
                    openness=float(hold_openness),
                )
        reason = (
            'contact'
            if contact_candidate
            else 'joint_stall'
            if stall_contact
            else 'closed_target'
            if closed_target_ready
            else 'closing'
        )
        return ready, {
            'closed': ready,
            'close_until_contact': True,
            'completion_reason': reason,
            'close_elapsed_steps': int(close_elapsed_steps),
            'min_steps': min_steps,
            'stable_steps': stable_steps,
            'required_stable_steps': required_stable_steps,
            'motion_ready': motion_ready,
            'motion_stable_steps': motion_stable_steps,
            'required_motion_stable_steps': required_motion_stable_steps,
            'motion_detail': motion_detail,
            'allow_closed_gripper_completion': bool(spec.get('allow_closed_gripper_completion', False)),
            'target_gripper_joint_position': target_q,
            'reached_gripper_target': reached_gripper_target,
            'closed_target_candidate': closed_target_candidate,
            'closed_target_stable_steps': closed_target_stable_steps,
            'required_closed_target_stable_steps': required_closed_target_stable_steps,
            'closed_target_ready': closed_target_ready,
            'gripper_openness_command': float(gripper_openness),
            'hold_gripper_openness': state.get('hold_gripper_openness'),
            'gripper_joint_position': None if gripper_q is None else float(gripper_q),
            'gripper_joint_delta': None if joint_delta is None else float(joint_delta),
            'gripper_open_position': None if open_q is None else float(open_q),
            'gripper_closed_position': None if closed_q is None else float(closed_q),
            'gripper_joint_range': joint_range,
            'close_contact_stall_joint_delta': stall_delta,
            'close_contact_blocked_joint_margin': blocked_margin,
            'close_contact_min_joint_closure': min_closure,
            'close_gripper_target_tolerance': target_tolerance,
            'blocked_before_full_close': blocked_before_full_close,
            'moved_from_open': moved_from_open,
            'stalled': stalled,
            'contact_ready': contact_ready,
            'contact_candidate': contact_candidate,
            'stall_contact': stall_contact,
            'detected_clamp': detected_clamp,
            'contact_detail': contact_detail,
        }

    @staticmethod
    def _current_gripper_q(*, task, robot_name: str) -> float | None:
        robot = task.robots.get(robot_name)
        if robot is None:
            return None
        dof_name = str(getattr(getattr(robot, 'config', None), 'gripper_dof_name', None) or 'finger_joint')
        articulation = getattr(robot, 'articulation', None)
        if articulation is not None:
            try:
                index = articulation.get_dof_index(dof_name)
                values = articulation.get_joint_positions(joint_indices=np.asarray([index], dtype=np.int64))
                values = np.asarray(values, dtype=float).reshape(-1)
                if values.size:
                    return float(values[0])
            except Exception:
                pass
            try:
                values = np.asarray(articulation.get_joint_positions(), dtype=float).reshape(-1)
                dof_names = list(getattr(articulation, 'dof_names', []) or [])
                if dof_name in dof_names:
                    return float(values[dof_names.index(dof_name)])
            except Exception:
                pass
        try:
            controller = robot.controllers.get(_GRIPPER_CONTROLLER)
            obs = controller.get_obs()
            values = np.asarray(obs.get('gripper_pos'), dtype=float).reshape(-1)
            if values.size:
                return float(values[0])
        except Exception:
            pass
        return None

    @staticmethod
    def _gripper_open_closed_q(*, task, robot_name: str) -> tuple[float | None, float | None]:
        robot = task.robots.get(robot_name)
        config = getattr(robot, 'config', None) if robot is not None else None
        try:
            open_q = float(getattr(config, 'gripper_open_position'))
            closed_q = float(getattr(config, 'gripper_closed_position'))
            return open_q, closed_q
        except Exception:
            return None, None

    @staticmethod
    def _mark_complete(*, task, robot_name: str, skill_name: str, detail: dict[str, Any]) -> None:
        marker = getattr(task, 'mark_local_skill_complete', None)
        if callable(marker):
            marker(robot_name=robot_name, skill_name=skill_name, detail=detail)

    def _grasp_contact_ready(self, *, task, robot_name: str, spec: dict) -> tuple[bool, dict[str, Any]]:
        object_name = str(spec.get('object', spec.get('object_name', spec.get('held_object', ''))))
        if not object_name:
            return False, {'reason': 'missing_object_for_grasp_check'}
        metrics_fn = getattr(task, '_gripper_contact_metrics', None)
        if not callable(metrics_fn):
            return False, {'reason': 'gripper_contact_metrics_unavailable', 'object': object_name}
        attach_spec = {
            'require_dual_finger_contact': bool(spec.get('require_dual_finger_contact', True)),
            'require_force_contact': bool(spec.get('require_force_contact', spec.get('require_contact_report', False))),
            'finger_contact_distance': float(spec.get('finger_contact_distance', 0.006)),
            'contact_force_threshold': float(spec.get('contact_force_threshold', 0.2)),
            'physical_attach_surface_gap': float(spec.get('physical_attach_surface_gap', 0.006)),
        }
        for key in (
            'require_dual_force_contact',
            'measure_force_contact',
            'allow_cross_axis_dual_finger_contact',
            'contact_box_scale',
            'contact_box_half_extents',
            'contact_box_offset',
            'caging_contact_distance',
            'physical_grasp_min_opening_ratio',
            'physical_contact_axes',
            'physical_contact_interior_scale',
            'physical_contact_interior_margin',
            'strict_finger_surface_gap',
            'strict_finger_contact_distance',
        ):
            if key in spec:
                attach_spec[key] = spec[key]
        try:
            metrics = metrics_fn(object_name, robot_name, attach_spec=attach_spec)
        except Exception as exc:
            return False, {
                'reason': 'gripper_contact_metrics_error',
                'object': object_name,
                'error': str(exc),
            }

        strict_ready = False
        strict_fn = getattr(task, '_strict_physical_grasp_contact', None)
        if callable(strict_fn):
            try:
                strict_ready = bool(
                    strict_fn(object_name, metrics, attach_spec=attach_spec).get('physical_contact_ready')
                )
            except Exception:
                strict_ready = False
        if bool(spec.get('require_strict_physical_contact', False)):
            contact_ready = strict_ready
        else:
            contact_ready = bool(metrics.get('contact_ready') or strict_ready)
        return contact_ready, {
            'object': object_name,
            'contact_ready': contact_ready,
            'strict_contact_ready': strict_ready,
            'contact_metrics': metrics,
        }

    def _failure_or_hold(
        self,
        task,
        robot_name: str,
        spec: dict,
        reason: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict:
        if bool(spec.get('require_success', False)):
            return {
                '__local_skill_failure__': True,
                'reason': reason,
                'diagnostics': {
                    'skill': spec.get('name'),
                    'robot': robot_name,
                    **(diagnostics or {}),
                },
            }
        action = self._hold_joint_action(task=task, robot_name=robot_name)
        gripper_command = spec.get('gripper_command')
        if gripper_command is not None:
            action[_GRIPPER_CONTROLLER] = [
                self._gripper_command_value(task=task, robot_name=robot_name, command=gripper_command)
            ]
        return action


# Backward-compatible import for existing recipes and downstream code.
AssemblyAtomicSkillAdapter = UR5eAssemblyAtomicSkillAdapter
UR5ePlumbersBlockAtomicSkillAdapter = UR5eAssemblyAtomicSkillAdapter
