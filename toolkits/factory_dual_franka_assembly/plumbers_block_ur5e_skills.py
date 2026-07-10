from __future__ import annotations

from collections import OrderedDict
import math
import os
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


_ARM_JOINT_CONTROLLER = "arm_joint_controller"
_GRIPPER_CONTROLLER = "gripper_controller"
_ARM_IK_CONTROLLER = "arm_ik_controller"
_UR5E_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


class UR5ePlumbersBlockAtomicSkillAdapter:
    """Scripted UR5e atomic skills for the first plumbers-block base pickup step.

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
        self._grasp_slip_state: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._last_arm_command_q: dict[str, np.ndarray] = {}

    def act(
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
        skill_name = str(spec.get("name", ""))
        phase_key = (
            id(task),
            getattr(task, "phase_index", None),
            getattr(task, "phase_entry_step", None),
            robot_name,
            skill_name,
        )

        if skill_name in {"ur5e_close_gripper", "close_gripper"}:
            action = self._hold_joint_action(task=task, robot_name=robot_name)
            close_started_step = 0
            if bool(spec.get("require_close_pose_gate", False)):
                gate_ready, gate_action, gate_detail = self._close_pose_gate_action(
                    phase_key=phase_key,
                    task=task,
                    robot_name=robot_name,
                    spec=spec,
                    tracked_robots=tracked_robots,
                    tracked_objects=tracked_objects,
                )
                if not gate_ready:
                    timeout_steps = spec.get("close_pose_gate_timeout_steps")
                    if timeout_steps is not None and int(getattr(task, "phase_step_counter", 0)) >= int(timeout_steps):
                        return self._failure_or_hold(
                            task,
                            robot_name,
                            spec,
                            "close_pose_gate_timeout",
                            diagnostics=gate_detail,
                        )
                    return gate_action
                state = self._close_gate_state.setdefault(phase_key, {})
                close_started_step = int(state.get("close_started_step", getattr(task, "phase_step_counter", 0)))
                hold_q = state.get("hold_q")
                action = OrderedDict()
                if hold_q is not None:
                    action[_ARM_JOINT_CONTROLLER] = [np.asarray(hold_q, dtype=float).tolist()]

            hold_steps = max(int(spec.get("hold_steps", spec.get("close_steps", 36))), 0)
            close_elapsed_steps = max(int(getattr(task, "phase_step_counter", 0)) - int(close_started_step), 0)
            ramp_steps = max(int(spec.get("close_ramp_steps", hold_steps)), 1)
            preclose_openness = float(spec.get("preclose_openness", spec.get("open_openness", 1.0)))
            closed_openness = float(spec.get("closed_openness", spec.get("close_openness", 0.0)))
            ramp_ratio = min(max(float(close_elapsed_steps) / float(ramp_steps), 0.0), 1.0)
            gripper_openness = preclose_openness + ramp_ratio * (closed_openness - preclose_openness)
            action[_GRIPPER_CONTROLLER] = [gripper_openness]
            if bool(spec.get("close_until_contact", False)):
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
                self._debug_close_step(
                    task=task,
                    robot_name=robot_name,
                    skill_name=skill_name,
                    close_elapsed_steps=close_elapsed_steps,
                    gripper_openness=gripper_openness,
                    close_ready=close_ready,
                    close_detail=close_detail,
                )
                hold_openness = state.get("hold_gripper_openness")
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
                        "close_object_knocked",
                        diagnostics=close_detail,
                    )
                else:
                    timeout_steps = spec.get("close_until_contact_timeout_steps")
                    if timeout_steps is not None and close_elapsed_steps >= int(timeout_steps):
                        return self._failure_or_hold(
                            task,
                            robot_name,
                            spec,
                            "close_until_contact_timeout",
                            diagnostics=close_detail,
                        )
                return action
            if close_elapsed_steps >= hold_steps:
                if bool(spec.get("require_grasp_contact", False)):
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
                            "grasp_contact_not_ready",
                            diagnostics=grasp_detail,
                        )
                self._mark_complete(
                    task=task,
                    robot_name=robot_name,
                    skill_name=skill_name,
                    detail={
                        "closed": True,
                        "hold_steps": hold_steps,
                        "close_elapsed_steps": close_elapsed_steps,
                    },
                )
                self._remember_gripper_hold_openness(
                    task=task,
                    robot_name=robot_name,
                    openness=gripper_openness,
                )
            return action

        target_pose = self._target_pose(
            phase_key=phase_key,
            task=task,
            robot_name=robot_name,
            spec=spec,
            tracked_robots=tracked_robots,
            tracked_objects=tracked_objects,
        )
        if target_pose is None:
            return self._failure_or_hold(task, robot_name, spec, "target_pose_unavailable")
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
                "object_tcp_slip",
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
        command_target_pose = target_pose
        if bool(spec.get("cartesian_servo", False)) and current_pose is not None:
            command_target_pose = self._cartesian_servo_target_pose(
                current_pose=current_pose,
                target_pose=target_pose,
                max_position_step=float(spec.get("cartesian_position_step", 0.01)),
                max_orientation_step=float(spec.get("cartesian_orientation_step", 0.01)),
            )

        ik_target_pose = self._ik_target_pose(target_pose=command_target_pose, spec=spec)
        use_arm_ik_controller = bool(spec.get("use_arm_ik_controller", False))
        # Raw Cartesian IK can choose a different UR5e branch between frames.  Keep it
        # opt-in and otherwise route through the joint-limited IK path below.
        if use_arm_ik_controller and bool(spec.get("allow_direct_arm_ik_controller", False)):
            current_q = self._current_arm_q(task, robot_name)
            action = OrderedDict()
            action[_ARM_IK_CONTROLLER] = [
                np.asarray(ik_target_pose["position"], dtype=float).tolist(),
                np.asarray(ik_target_pose["orientation"], dtype=float).tolist(),
            ]
            gripper_command = spec.get("gripper_command")
            if gripper_command is None:
                gripper_command = "close" if skill_name in {"ur5e_move_part_to_staging", "ur5e_hold_part_end"} else "open"
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
                "position": target_pose["position"].copy(),
                "orientation": target_pose["orientation"].copy(),
            }
            self._maybe_mark_complete(
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
        if ik_result is None:
            self._debug_motion_blocked(
                task=task,
                robot_name=robot_name,
                skill_name=skill_name,
                reason="ik_failed",
                spec=spec,
                target_pose=target_pose,
                ik_target_pose=ik_target_pose,
                reference_q=reference_q,
            )
            return self._failure_or_hold(task, robot_name, spec, "ik_failed")

        target_q = self._unwrap_to_reference(
            target_q=ik_result,
            reference_q=reference_q,
            preferred_abs_limit=spec.get("preferred_joint_abs_limit", 3.05),
            hard_preferred_abs_limit=bool(spec.get("hard_preferred_joint_abs_limit", True)),
        )
        if reference_q is None:
            return self._failure_or_hold(
                task,
                robot_name,
                spec,
                "current_joint_state_unavailable",
                diagnostics={
                    "target_position": target_pose["position"].tolist(),
                    "target_orientation": target_pose["orientation"].tolist(),
                },
            )
        guard_branch_jump = bool(spec.get("guard_ik_branch_jump", bool(spec.get("cartesian_servo", False))))
        if guard_branch_jump and self._ik_branch_jump_detected(reference_q=reference_q, target_q=target_q, spec=spec):
            self._debug_motion_blocked(
                task=task,
                robot_name=robot_name,
                skill_name=skill_name,
                reason="ik_branch_jump_guard",
                spec=spec,
                target_pose=target_pose,
                ik_target_pose=ik_target_pose,
                reference_q=reference_q,
                target_q=target_q,
            )
            return self._failure_or_hold(
                task,
                robot_name,
                {**spec, "require_success": False},
                "ik_branch_jump_guard",
                diagnostics={
                    "target_position": target_pose["position"].tolist(),
                    "target_orientation": target_pose["orientation"].tolist(),
                    "reference_q": reference_q.tolist(),
                    "target_q": target_q.tolist(),
                },
            )
        joint_step_limits = self._command_joint_step_limits(
            spec=spec,
            joint_count=reference_q.shape[0],
        )
        if joint_step_limits is None:
            joint_step_limits = float(spec.get("max_joint_step", 0.035))
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
            "position": target_pose["position"].copy(),
            "orientation": target_pose["orientation"].copy(),
            "target_q": target_q.copy(),
        }

        action = OrderedDict()
        if use_arm_ik_controller:
            action[_ARM_IK_CONTROLLER] = [
                np.asarray(ik_target_pose["position"], dtype=float).tolist(),
                np.asarray(ik_target_pose["orientation"], dtype=float).tolist(),
            ]
        action[_ARM_JOINT_CONTROLLER] = [command_q.tolist()]
        self._remember_arm_command(task, robot_name, command_q)
        gripper_command = spec.get("gripper_command")
        if gripper_command is None:
            gripper_command = "close" if skill_name in {"ur5e_move_part_to_staging", "ur5e_hold_part_end"} else "open"
        action[_GRIPPER_CONTROLLER] = [
            self._gripper_command_value(task=task, robot_name=robot_name, command=gripper_command)
        ]

        self._maybe_mark_complete(
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
        position_tolerance = float(spec.get("position_tolerance", 0.025))
        orientation_tolerance = spec.get("orientation_tolerance")
        orientation_tolerance = None if orientation_tolerance is None else float(orientation_tolerance)
        force_complete_after_steps = spec.get("force_complete_after_steps")
        complete = False
        completion_detail = {
            "target_position": target_pose["position"].tolist(),
            "target_orientation": target_pose["orientation"].tolist(),
            "ik_target_position": ik_target_pose["position"].tolist(),
            "ik_target_orientation": ik_target_pose["orientation"].tolist(),
        }
        tcp_offset = self._tcp_offset(spec)
        if tcp_offset is not None:
            completion_detail["grasp_tcp_offset"] = tcp_offset.tolist()
            completion_detail["grasp_tcp_offset_frame"] = self._tcp_offset_frame(spec)
        if current_pose is not None:
            position_error, orientation_error = pose_error(
                current_position=current_pose["position"],
                current_orientation=current_pose["orientation"],
                target_position=target_pose["position"],
                target_orientation=target_pose["orientation"],
            )
            completion_detail.update(
                {
                    "position_error": position_error,
                    "orientation_error": orientation_error,
                    "position_tolerance": position_tolerance,
                    "orientation_tolerance": orientation_tolerance,
                }
            )
            complete = position_error <= position_tolerance and (
                orientation_tolerance is None
                or orientation_error is None
                or orientation_error <= orientation_tolerance
            )

        if bool(spec.get("require_target_object_pose_convergence", False)):
            object_name = self._object_name_from_spec(spec)
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
                spec.get("target_object_position_tolerance", position_tolerance)
            )
            object_orientation_tolerance = spec.get(
                "target_object_orientation_tolerance",
                orientation_tolerance,
            )
            object_orientation_tolerance = (
                None
                if object_orientation_tolerance is None
                else float(object_orientation_tolerance)
            )
            object_pose_complete = False
            object_position_error = None
            object_orientation_error = None
            if object_pose is not None and target_object_pose is not None:
                target_object_orientation = target_object_pose.get("orientation")
                if target_object_orientation is None:
                    object_position_error = float(
                        np.linalg.norm(
                            np.asarray(object_pose["position"], dtype=float)
                            - np.asarray(target_object_pose["position"], dtype=float)
                        )
                    )
                else:
                    object_position_error, object_orientation_error = pose_error(
                        current_position=object_pose["position"],
                        current_orientation=object_pose["orientation"],
                        target_position=target_object_pose["position"],
                        target_orientation=target_object_orientation,
                    )
                object_pose_complete = bool(
                    object_position_error <= object_position_tolerance
                    and (
                        object_orientation_tolerance is None
                        or object_orientation_error is None
                        or object_orientation_error <= object_orientation_tolerance
                    )
                )
            completion_detail.update(
                {
                    "target_object_pose_required": True,
                    "target_object_name": object_name,
                    "target_object_position": (
                        None
                        if target_object_pose is None
                        else np.asarray(target_object_pose["position"], dtype=float).tolist()
                    ),
                    "target_object_orientation": (
                        None
                        if target_object_pose is None or target_object_pose.get("orientation") is None
                        else np.asarray(target_object_pose["orientation"], dtype=float).tolist()
                    ),
                    "object_position_error": object_position_error,
                    "object_orientation_error": object_orientation_error,
                    "target_object_position_tolerance": object_position_tolerance,
                    "target_object_orientation_tolerance": object_orientation_tolerance,
                    "target_object_pose_complete": object_pose_complete,
                }
            )
            complete = bool(complete and object_pose_complete)

        joint_position_tolerance = spec.get("joint_position_tolerance")
        if joint_position_tolerance is not None and current_q is not None and target_q is not None:
            joint_error = float(np.max(np.abs(np.asarray(target_q, dtype=float) - np.asarray(current_q, dtype=float))))
            completion_detail.update(
                {
                    "joint_error": joint_error,
                    "joint_position_tolerance": float(joint_position_tolerance),
                }
            )
            complete = bool(complete and joint_error <= float(joint_position_tolerance))

        if force_complete_after_steps is not None and int(getattr(task, "phase_step_counter", 0)) >= int(force_complete_after_steps):
            complete = True
            completion_detail["force_complete"] = True

        if complete:
            self._mark_complete(
                task=task,
                robot_name=robot_name,
                skill_name=skill_name,
                detail=completion_detail,
            )

    @staticmethod
    def _debug_grasp_enabled() -> bool:
        return os.environ.get("UR5E_DEBUG_GRASP", "0").strip().lower() in {"1", "true", "yes"}

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
        if not self._debug_grasp_enabled():
            return
        should_print = (
            int(close_elapsed_steps) <= 5
            or int(close_elapsed_steps) % 12 == 0
            or bool(close_ready)
            or bool(close_detail.get("detected_clamp"))
            or not bool(close_detail.get("motion_ready", True))
        )
        if not should_print:
            return
        motion_detail = close_detail.get("motion_detail") or {}
        contact_detail = close_detail.get("contact_detail") or {}
        contact_metrics = contact_detail.get("contact_metrics") if isinstance(contact_detail, dict) else {}
        left_gap = None
        right_gap = None
        if isinstance(contact_metrics, dict):
            left_gap = (contact_metrics.get("left_finger") or {}).get("surface_gap")
            right_gap = (contact_metrics.get("right_finger") or {}).get("surface_gap")
        print(
            "[ur5e-grasp-debug] "
            f"step={getattr(task, 'step_counter', None)} "
            f"phase_step={getattr(task, 'phase_step_counter', None)} "
            f"robot={robot_name} skill={skill_name} close_elapsed={int(close_elapsed_steps)} "
            f"cmd_open={float(gripper_openness):.4f} "
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
            f"left_gap={left_gap} right_gap={right_gap}",
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
            spec.get("target_object_position") is not None
            or spec.get("target_object_target") is not None
            or spec.get("target_object") is not None
            or spec.get("object_target") is not None
            or skill_name in {"ur5e_move_part_to_table_hover", "ur5e_move_part_to_staging", "ur5e_hold_part_end"}
        ):
            return
        phase_step = int(getattr(task, "phase_step_counter", 0))
        every = max(int(os.environ.get("UR5E_DEBUG_TRANSPORT_EVERY", "15")), 1)
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
        relative_world = None
        if object_pose is not None and target_object_position is not None:
            object_error = float(
                np.linalg.norm(
                    np.asarray(object_pose["position"], dtype=float)
                    - np.asarray(target_object_position, dtype=float)
                )
            )
        if object_pose is not None and current_pose is not None:
            relative_world = (
                np.asarray(object_pose["position"], dtype=float)
                - np.asarray(current_pose["position"], dtype=float)
            )

        tcp_error = None
        if current_pose is not None:
            tcp_error = float(
                np.linalg.norm(
                    np.asarray(current_pose["position"], dtype=float)
                    - np.asarray(target_pose["position"], dtype=float)
                )
            )
        motion_detail = self._object_motion_detail(
            task=task,
            object_name=object_name,
            tracked_objects=tracked_objects,
        )
        print(
            "[ur5e-transport-debug] "
            f"step={getattr(task, 'step_counter', None)} phase={getattr(task, 'phase', None)} "
            f"phase_step={phase_step} robot={robot_name} skill={skill_name} "
            f"target_object_position={None if target_object_position is None else np.asarray(target_object_position, dtype=float).tolist()} "
            f"object_position={None if object_pose is None else np.asarray(object_pose['position'], dtype=float).tolist()} "
            f"object_error={object_error} "
            f"target_tcp={np.asarray(target_pose['position'], dtype=float).tolist()} "
            f"current_tcp={None if current_pose is None else np.asarray(current_pose['position'], dtype=float).tolist()} "
            f"tcp_error={tcp_error} "
            f"relative_world={None if relative_world is None else relative_world.tolist()} "
            f"gripper_q={self._current_gripper_q(task=task, robot_name=robot_name)} "
            f"hold_open={self._last_gripper_hold_openness(task=task, robot_name=robot_name)} "
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
        phase_step = int(getattr(task, "phase_step_counter", 0))
        every = max(int(os.environ.get("UR5E_DEBUG_TRANSPORT_EVERY", "15")), 1)
        if phase_step > 5 and phase_step % every != 0:
            return
        print(
            "[ur5e-motion-blocked] "
            f"step={getattr(task, 'step_counter', None)} phase={getattr(task, 'phase', None)} "
            f"phase_step={phase_step} robot={robot_name} skill={skill_name} reason={reason} "
            f"target_tcp={np.asarray(target_pose['position'], dtype=float).tolist()} "
            f"ik_target={np.asarray(ik_target_pose['position'], dtype=float).tolist()} "
            f"reference_q={None if reference_q is None else np.asarray(reference_q, dtype=float).tolist()} "
            f"target_q={None if target_q is None else np.asarray(target_q, dtype=float).tolist()} "
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
            bool(spec.get("cartesian_servo", False))
            or spec.get("target_object_position") is not None
            or spec.get("target_object_target") is not None
            or skill_name in {"ur5e_move_part_to_table_hover", "ur5e_move_part_to_staging", "ur5e_hold_part_end"}
        ):
            return
        phase_step = int(getattr(task, "phase_step_counter", 0))
        every = max(int(os.environ.get("UR5E_DEBUG_TRANSPORT_EVERY", "15")), 1)
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
        command_tcp_step = None
        if current_pose is not None:
            tcp_error = float(
                np.linalg.norm(
                    np.asarray(current_pose["position"], dtype=float)
                    - np.asarray(target_pose["position"], dtype=float)
                )
            )
            command_tcp_step = float(
                np.linalg.norm(
                    np.asarray(command_target_pose["position"], dtype=float)
                    - np.asarray(current_pose["position"], dtype=float)
                )
            )
        print(
            "[ur5e-joint-debug] "
            f"step={getattr(task, 'step_counter', None)} phase={getattr(task, 'phase', None)} "
            f"phase_step={phase_step} robot={robot_name} skill={skill_name} "
            f"ik_reference_mode={spec.get('ik_reference_mode', spec.get('reference_mode'))} "
            f"use_command_warm_start={spec.get('use_command_warm_start', True)} "
            f"tcp_error={tcp_error} command_tcp_step={command_tcp_step} "
            f"target_tcp={np.asarray(target_pose['position'], dtype=float).tolist()} "
            f"command_tcp={np.asarray(command_target_pose['position'], dtype=float).tolist()} "
            f"ik_tcp={np.asarray(ik_target_pose['position'], dtype=float).tolist()} "
            f"current_q={None if current_q is None else current_q.tolist()} "
            f"reference_q={None if reference_q is None else reference_q.tolist()} "
            f"target_q={None if target_q is None else target_q.tolist()} "
            f"command_q={None if command_q is None else command_q.tolist()} "
            f"ref_to_target_max={_max_abs_delta(reference_q, target_q)} "
            f"ref_to_cmd_max={_max_abs_delta(reference_q, command_q)} "
            f"current_to_cmd_max={_max_abs_delta(current_q, command_q)} "
            f"current_to_ref_max={_max_abs_delta(current_q, reference_q)}",
            flush=True,
        )

    @staticmethod
    def _cartesian_servo_target_pose(
        *,
        current_pose: dict,
        target_pose: dict,
        max_position_step: float,
        max_orientation_step: float,
    ) -> dict:
        current_position = np.asarray(current_pose["position"], dtype=float)
        target_position = np.asarray(target_pose["position"], dtype=float)
        delta = target_position - current_position
        distance = float(np.linalg.norm(delta))
        if max_position_step > 0.0 and distance > max_position_step:
            command_position = current_position + delta * (max_position_step / distance)
        else:
            command_position = target_position

        current_orientation = normalize_quat(current_pose["orientation"])
        target_orientation = normalize_quat(target_pose["orientation"])
        dot = float(np.dot(current_orientation, target_orientation))
        if dot < 0.0:
            target_orientation = -target_orientation
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        orientation_angle = float(2.0 * math.acos(dot))
        if max_orientation_step > 0.0 and orientation_angle > max_orientation_step:
            ratio = max_orientation_step / orientation_angle
            if dot > 0.9995:
                command_orientation = normalize_quat(
                    (1.0 - ratio) * current_orientation + ratio * target_orientation
                )
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
            "position": command_position,
            "orientation": command_orientation,
        }

    def _target_pose(
        self,
        *,
        phase_key=None,
        task,
        robot_name: str,
        spec: dict,
        tracked_robots: dict,
        tracked_objects: dict,
    ):
        grasp_relative_pose = self._configured_grasp_relative_pose(spec)
        if grasp_relative_pose is not None:
            object_name = str(spec.get("object", spec.get("object_name", "fabrica_plumbers_block_2")))
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
                    object_pose["orientation"],
                    self._quat_conjugate(relative_orientation),
                )
            )
            position = np.asarray(object_pose["position"], dtype=float) - quat_rotate(
                orientation,
                relative_position,
            )
            approach_offset = np.asarray(spec.get("offset", [0.0, 0.0, 0.0]), dtype=float)
            offset_frame = str(spec.get("offset_frame", "world")).lower()
            if offset_frame in {"object", "local", "part"}:
                position = position + quat_rotate(object_pose["orientation"], approach_offset)
            elif offset_frame in {"target", "eef", "tcp", "gripper"}:
                position = position + quat_rotate(orientation, approach_offset)
            else:
                position = position + approach_offset
            return {
                "position": np.asarray(position, dtype=float),
                "orientation": orientation,
            }

        object_pose = None
        direct_target_pose = None
        target_object_pose = self._target_object_pose(task=task, spec=spec)
        target_object_position = None if target_object_pose is None else target_object_pose["position"]
        if target_object_position is not None:
            object_name = str(spec.get("object", spec.get("object_name", "fabrica_plumbers_block_2")))
            object_pose = self._object_pose(
                task=task,
                object_name=object_name,
                tracked_objects=tracked_objects,
            )
            if object_pose is None:
                return None
        elif spec.get("target_pose_target") is not None or spec.get("target_pose") is not None:
            target_name = spec.get("target_pose_target") or spec.get("target_pose")
            direct_target_pose = self._target_pose_by_name(
                task=task,
                target_name=None if target_name is None else str(target_name),
            )
            if direct_target_pose is None:
                return None
            position = direct_target_pose["position"]
        elif spec.get("target_position") is not None:
            position = np.asarray(spec["target_position"], dtype=float)
        else:
            object_name = str(spec.get("object", spec.get("object_name", "fabrica_plumbers_block_2")))
            object_pose = self._object_pose(
                task=task,
                object_name=object_name,
                tracked_objects=tracked_objects,
            )
            if object_pose is None:
                return None
            offset = np.asarray(spec.get("offset", [0.0, 0.0, 0.0]), dtype=float)
            if str(spec.get("offset_frame", "world")).lower() in {"object", "local", "part"}:
                position, _ = compose_pose(
                    base_position=object_pose["position"],
                    base_orientation=object_pose["orientation"],
                    local_position=offset,
                    local_orientation=[1.0, 0.0, 0.0, 0.0],
                )
            else:
                position = object_pose["position"] + offset

        has_explicit_orientation = any(
            spec.get(name) is not None
            for name in ("target_orientation", "orientation", "orientation_euler")
        )
        derive_tcp_orientation = bool(
            target_object_position is not None
            and self._derive_tcp_orientation_from_target_object(spec=spec)
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
            )
            if relative_pose is None or target_object_pose.get("orientation") is None:
                return None
            _, relative_orientation = relative_pose
            orientation = normalize_quat(
                quat_multiply(
                    target_object_pose["orientation"],
                    self._quat_conjugate(relative_orientation),
                )
            )
        elif direct_target_pose is not None and not has_explicit_orientation:
            orientation = direct_target_pose["orientation"]
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
                )
            if relative_pose is None:
                return None
            relative_position, _ = relative_pose
            position = np.asarray(target_object_position, dtype=float) - quat_rotate(
                normalize_quat(orientation),
                relative_position,
            )
        return {
            "position": np.asarray(position, dtype=float),
            "orientation": normalize_quat(orientation),
        }

    @staticmethod
    def _configured_grasp_relative_pose(spec: dict) -> tuple[np.ndarray, np.ndarray] | None:
        position = spec.get(
            "grasp_relative_position",
            spec.get("object_in_tcp_position", spec.get("object_in_gripper_position")),
        )
        orientation = spec.get(
            "grasp_relative_orientation",
            spec.get("object_in_tcp_orientation", spec.get("object_in_gripper_orientation")),
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
        target_poses = getattr(task, "target_poses", None)
        if not isinstance(target_poses, dict) or target_name not in target_poses:
            return None
        target_pose = target_poses[target_name]
        if target_pose.get("position") is None:
            return None
        return {
            "position": np.asarray(target_pose["position"], dtype=float),
            "orientation": normalize_quat(target_pose.get("orientation", [1.0, 0.0, 0.0, 0.0])),
        }

    def _target_object_pose(self, *, task, spec: dict) -> dict | None:
        if spec.get("target_object_position") is not None:
            position = np.asarray(spec["target_object_position"], dtype=float)
            orientation = self._target_object_orientation_from_spec(task=task, spec=spec)
        else:
            target_name = spec.get("target_object_target") or spec.get("target_object") or spec.get("object_target")
            target_pose = self._target_pose_by_name(
                task=task,
                target_name=None if target_name is None else str(target_name),
            )
            if target_pose is None:
                return None
            position = target_pose["position"]
            orientation = target_pose["orientation"]
            override_orientation = self._target_object_orientation_from_spec(task=task, spec=spec)
            if override_orientation is not None:
                orientation = override_orientation
        offset = spec.get("target_object_offset")
        if offset is not None:
            offset = np.asarray(offset, dtype=float)
            offset_frame = str(spec.get("target_object_offset_frame", "world")).lower()
            if offset_frame in {"target", "local", "object_target"}:
                if orientation is None:
                    return None
                position = position + quat_rotate(orientation, offset)
            else:
                position = position + offset
        return {
            "position": np.asarray(position, dtype=float),
            "orientation": None if orientation is None else normalize_quat(orientation),
        }

    @staticmethod
    def _target_object_orientation_from_spec(*, task, spec: dict) -> np.ndarray | None:
        if spec.get("target_object_orientation") is not None:
            return normalize_quat(spec["target_object_orientation"])
        if spec.get("target_object_orientation_euler") is not None:
            return euler_xyz_to_quat(spec["target_object_orientation_euler"])
        target_name = spec.get("target_object_orientation_target") or spec.get("object_orientation_target")
        if target_name is None:
            return None
        target_pose = UR5ePlumbersBlockAtomicSkillAdapter._target_pose_by_name(
            task=task,
            target_name=str(target_name),
        )
        if target_pose is None:
            return None
        return normalize_quat(target_pose["orientation"])

    def _target_object_position(self, *, task, spec: dict) -> np.ndarray | None:
        target_pose = self._target_object_pose(task=task, spec=spec)
        if target_pose is None:
            return None
        return np.asarray(target_pose["position"], dtype=float)

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
    ) -> tuple[np.ndarray, np.ndarray] | None:
        cache = getattr(task, "_ur5e_plumbers_object_tcp_relative_poses", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(task, "_ur5e_plumbers_object_tcp_relative_poses", cache)
        cache_key = (str(robot_name), str(object_name))
        cached = cache.get(cache_key)
        if cached is not None:
            try:
                cached_position = np.asarray(cached["position"], dtype=float)
                cached_orientation = normalize_quat(cached["orientation"])
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
        relative_world = np.asarray(object_pose["position"], dtype=float) - np.asarray(
            current_tcp_pose["position"],
            dtype=float,
        )
        relative_tcp = quat_rotate(
            self._quat_conjugate(current_tcp_pose["orientation"]),
            relative_world,
        )
        if relative_tcp.shape != (3,) or not np.all(np.isfinite(relative_tcp)):
            return None
        relative_orientation = normalize_quat(
            quat_multiply(
                self._quat_conjugate(current_tcp_pose["orientation"]),
                object_pose["orientation"],
            )
        )
        if relative_orientation.shape != (4,) or not np.all(np.isfinite(relative_orientation)):
            return None
        cache[cache_key] = {
            "position": relative_tcp.copy(),
            "orientation": relative_orientation.copy(),
        }
        if self._debug_grasp_enabled():
            print(
                "[ur5e-grasp-debug] "
                f"captured_object_tcp_relative robot={robot_name} object={object_name} "
                f"phase_key={phase_key} relative_tcp={relative_tcp.tolist()} "
                f"relative_orientation={relative_orientation.tolist()} "
                f"object_position={np.asarray(object_pose['position'], dtype=float).tolist()} "
                f"tcp_position={np.asarray(current_tcp_pose['position'], dtype=float).tolist()}",
                flush=True,
            )
        return relative_tcp.copy(), relative_orientation.copy()

    @staticmethod
    def _derive_tcp_orientation_from_target_object(*, spec: dict) -> bool:
        for name in (
            "derive_tcp_orientation_from_target_object",
            "target_orientation_from_object_target",
            "use_target_object_orientation",
        ):
            if name in spec:
                return bool(spec.get(name))
        return False

    def _locked_target_pose(self, *, phase_key, target_pose: dict, spec: dict) -> dict:
        lock_orientation = bool(spec.get("lock_target_orientation", True))
        lock_position = bool(spec.get("lock_target_position", False))
        if not lock_orientation and not lock_position:
            return target_pose

        locked = self._phase_locks.setdefault(phase_key, {})
        result = {
            "position": np.asarray(target_pose["position"], dtype=float).copy(),
            "orientation": normalize_quat(target_pose["orientation"]).copy(),
        }
        if lock_position:
            if "position" not in locked:
                locked["position"] = result["position"].copy()
            result["position"] = locked["position"].copy()
        if lock_orientation:
            if "orientation" not in locked:
                locked["orientation"] = result["orientation"].copy()
            result["orientation"] = locked["orientation"].copy()
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
        state = self._close_gate_state.setdefault(phase_key, {"ready_steps": 0})
        gate_spec = dict(spec)
        for name in (
            "offset",
            "offset_frame",
            "target_orientation",
            "target_orientation_frame",
            "orientation",
            "orientation_frame",
            "orientation_euler",
            "ik_target_offset",
            "ik_target_offset_frame",
            "grasp_tcp_offset",
            "grasp_tcp_offset_frame",
            "tcp_offset",
            "tcp_offset_frame",
        ):
            override_name = f"close_gate_{name}"
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
            action[_GRIPPER_CONTROLLER] = [float(gate_spec.get("preclose_openness", gate_spec.get("open_openness", 1.0)))]
            state["ready_steps"] = 0
            return False, action, {"reason": "close_gate_target_pose_unavailable"}
        target_pose = self._locked_target_pose(phase_key=phase_key, target_pose=target_pose, spec=gate_spec)

        current_q = self._current_arm_q(task, robot_name)
        reference_q = self._command_reference_q(task=task, robot_name=robot_name, current_q=current_q, spec=gate_spec)
        ik_target_pose = self._ik_target_pose(target_pose=target_pose, spec=gate_spec)
        ik_result = self._solve_ik(
            task=task,
            robot_name=robot_name,
            target_pose=ik_target_pose,
            warm_start=reference_q,
            spec=gate_spec,
        )
        if ik_result is None:
            action = self._hold_joint_action(task=task, robot_name=robot_name)
            action[_GRIPPER_CONTROLLER] = [float(gate_spec.get("preclose_openness", gate_spec.get("open_openness", 1.0)))]
            state["ready_steps"] = 0
            return False, action, {
                "reason": "close_gate_ik_failed",
                "target_position": target_pose["position"].tolist(),
                "target_orientation": target_pose["orientation"].tolist(),
            }

        target_q = self._unwrap_to_reference(
            target_q=ik_result,
            reference_q=reference_q,
            preferred_abs_limit=gate_spec.get("preferred_joint_abs_limit", 3.05),
            hard_preferred_abs_limit=bool(gate_spec.get("hard_preferred_joint_abs_limit", True)),
        )
        if reference_q is None:
            action = self._hold_joint_action(task=task, robot_name=robot_name)
            action[_GRIPPER_CONTROLLER] = [float(gate_spec.get("preclose_openness", gate_spec.get("open_openness", 1.0)))]
            state["ready_steps"] = 0
            return False, action, {
                "reason": "close_gate_current_joint_state_unavailable",
                "target_position": target_pose["position"].tolist(),
                "target_orientation": target_pose["orientation"].tolist(),
            }
        if bool(gate_spec.get("close_gate_guard_ik_branch_jump", True)) and self._ik_branch_jump_detected(
            reference_q=reference_q,
            target_q=target_q,
            spec=gate_spec,
        ):
            action = self._hold_joint_action(task=task, robot_name=robot_name)
            action[_GRIPPER_CONTROLLER] = [float(gate_spec.get("preclose_openness", gate_spec.get("open_openness", 1.0)))]
            state["ready_steps"] = 0
            return False, action, {
                "reason": "close_gate_ik_branch_jump_guard",
                "target_position": target_pose["position"].tolist(),
                "target_orientation": target_pose["orientation"].tolist(),
                "reference_q": reference_q.tolist(),
                "target_q": target_q.tolist(),
            }
        command_q = self._limited_joint_target(
            current_q=reference_q,
            target_q=target_q,
            max_joint_step=float(gate_spec.get("close_gate_max_joint_step", gate_spec.get("max_joint_step", 0.025))),
        )
        command_q = self._continuous_command_q(
            task=task,
            robot_name=robot_name,
            command_q=command_q,
            spec={
                **gate_spec,
                "max_joint_step": gate_spec.get(
                    "close_gate_max_joint_step",
                    gate_spec.get("max_joint_step", 0.025),
                ),
            },
        )

        action = OrderedDict()
        action[_ARM_JOINT_CONTROLLER] = [command_q.tolist()]
        self._remember_arm_command(task, robot_name, command_q)
        action[_GRIPPER_CONTROLLER] = [float(gate_spec.get("preclose_openness", gate_spec.get("open_openness", 1.0)))]

        detail = {
            "target_position": target_pose["position"].tolist(),
            "target_orientation": target_pose["orientation"].tolist(),
            "ik_target_position": ik_target_pose["position"].tolist(),
            "ik_target_orientation": ik_target_pose["orientation"].tolist(),
        }
        ready = False
        current_pose = self._current_tcp_pose(
            current_pose=self._current_robot_pose(task=task, robot_name=robot_name, tracked_robots=tracked_robots),
            spec=gate_spec,
        )
        if current_pose is not None:
            position_tolerance = float(gate_spec.get("close_position_tolerance", gate_spec.get("position_tolerance", 0.01)))
            orientation_tolerance = gate_spec.get("close_orientation_tolerance", gate_spec.get("orientation_tolerance"))
            orientation_tolerance = None if orientation_tolerance is None else float(orientation_tolerance)
            position_error, orientation_error = pose_error(
                current_position=current_pose["position"],
                current_orientation=current_pose["orientation"],
                target_position=target_pose["position"],
                target_orientation=target_pose["orientation"],
            )
            ready = position_error <= position_tolerance and (
                orientation_tolerance is None
                or orientation_error is None
                or orientation_error <= orientation_tolerance
            )
            detail.update(
                {
                    "position_error": position_error,
                    "orientation_error": orientation_error,
                    "position_tolerance": position_tolerance,
                    "orientation_tolerance": orientation_tolerance,
                }
            )

        joint_tolerance = gate_spec.get("close_joint_position_tolerance", gate_spec.get("joint_position_tolerance"))
        if joint_tolerance is not None and current_q is not None:
            joint_error = float(np.max(np.abs(np.asarray(target_q, dtype=float) - np.asarray(current_q, dtype=float))))
            ready = bool(ready and joint_error <= float(joint_tolerance))
            detail.update(
                {
                    "joint_error": joint_error,
                    "joint_position_tolerance": float(joint_tolerance),
                }
            )

        if ready:
            state["ready_steps"] = int(state.get("ready_steps", 0)) + 1
        else:
            state["ready_steps"] = 0
            state.pop("close_started_step", None)
            state.pop("hold_q", None)

        required_ready_steps = max(int(gate_spec.get("close_ready_stable_steps", 4)), 1)
        gate_ready = int(state.get("ready_steps", 0)) >= required_ready_steps
        detail["ready_steps"] = int(state.get("ready_steps", 0))
        detail["required_ready_steps"] = required_ready_steps
        detail["gate_ready"] = gate_ready
        if gate_ready and "close_started_step" not in state:
            state["close_started_step"] = int(getattr(task, "phase_step_counter", 0))
            state["hold_q"] = np.asarray(current_q if current_q is not None else command_q, dtype=float).copy()
        return gate_ready, action, detail

    def _prealign_action(self, *, task, robot_name: str, target_pose: dict, spec: dict):
        prealign_steps = int(spec.get("prealign_steps", 0) or 0)
        if prealign_steps <= 0:
            return None
        if int(getattr(task, "phase_step_counter", 0)) >= prealign_steps:
            return None

        current_q = self._current_arm_q(task, robot_name)
        reference_q = self._command_reference_q(task=task, robot_name=robot_name, current_q=current_q, spec=spec)
        if reference_q is None or reference_q.shape[0] < 1:
            return None

        desired_q = reference_q.copy()
        if spec.get("prealign_joint_positions") is not None:
            joint_values = np.asarray(spec["prealign_joint_positions"], dtype=float)
            desired_q[: min(desired_q.shape[0], joint_values.shape[0])] = joint_values[: desired_q.shape[0]]
        else:
            shoulder_pan = spec.get("prealign_shoulder_pan")
            if shoulder_pan is None and bool(spec.get("prealign_shoulder_pan_from_target", False)):
                shoulder_pan = self._target_facing_shoulder_pan(
                    task=task,
                    robot_name=robot_name,
                    target_position=target_pose["position"],
                    yaw_offset=float(spec.get("prealign_shoulder_pan_yaw_offset", -0.47)),
                )
            if shoulder_pan is None:
                return None
            desired_q[0] = float(shoulder_pan)

        command_q = self._limited_joint_target(
            current_q=reference_q,
            target_q=desired_q,
            max_joint_step=float(spec.get("prealign_max_joint_step", spec.get("max_joint_step", 0.035))),
        )
        command_q = self._continuous_command_q(
            task=task,
            robot_name=robot_name,
            command_q=command_q,
            spec={
                **spec,
                "max_joint_step": spec.get(
                    "prealign_max_joint_step",
                    spec.get("max_joint_step", 0.035),
                ),
            },
        )
        action = OrderedDict()
        action[_ARM_JOINT_CONTROLLER] = [command_q.tolist()]
        self._remember_arm_command(task, robot_name, command_q)
        action[_GRIPPER_CONTROLLER] = [
            self._gripper_command_value(task=task, robot_name=robot_name, command=spec.get("gripper_command", "open"))
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
        position = np.asarray(current_pose["position"], dtype=float).copy()
        orientation = normalize_quat(current_pose["orientation"])
        tcp_offset = UR5ePlumbersBlockAtomicSkillAdapter._tcp_offset(spec)
        if tcp_offset is None:
            return {"position": position, "orientation": orientation}
        position = UR5ePlumbersBlockAtomicSkillAdapter._position_with_offset(
            position=position,
            orientation=orientation,
            offset=tcp_offset,
            offset_frame=UR5ePlumbersBlockAtomicSkillAdapter._tcp_offset_frame(spec),
            sign=1.0,
        )
        return {"position": position, "orientation": orientation}

    @staticmethod
    def _ik_target_pose(*, target_pose: dict, spec: dict) -> dict:
        position = np.asarray(target_pose["position"], dtype=float).copy()
        orientation = normalize_quat(target_pose["orientation"])
        tcp_offset = UR5ePlumbersBlockAtomicSkillAdapter._tcp_offset(spec)
        if tcp_offset is not None:
            position = UR5ePlumbersBlockAtomicSkillAdapter._position_with_offset(
                position=position,
                orientation=orientation,
                offset=tcp_offset,
                offset_frame=UR5ePlumbersBlockAtomicSkillAdapter._tcp_offset_frame(spec),
                sign=-1.0,
            )
        offset = spec.get("ik_target_offset")
        if offset is not None:
            offset = np.asarray(offset, dtype=float)
            offset_frame = str(spec.get("ik_target_offset_frame", "world")).lower()
            if offset_frame in {"target", "local", "eef", "gripper"}:
                position = position + quat_rotate(orientation, offset)
            else:
                position = position + offset
        return {"position": position, "orientation": orientation}

    @staticmethod
    def _tcp_offset(spec: dict) -> np.ndarray | None:
        offset = spec.get("grasp_tcp_offset", spec.get("tcp_offset"))
        if offset is None:
            return None
        offset = np.asarray(offset, dtype=float)
        if offset.shape != (3,) or not np.all(np.isfinite(offset)):
            return None
        return offset

    @staticmethod
    def _tcp_offset_frame(spec: dict) -> str:
        return str(spec.get("grasp_tcp_offset_frame", spec.get("tcp_offset_frame", "target"))).lower()

    @staticmethod
    def _position_with_offset(
        *,
        position: np.ndarray,
        orientation: np.ndarray,
        offset: np.ndarray,
        offset_frame: str,
        sign: float,
    ) -> np.ndarray:
        if offset_frame in {"target", "local", "eef", "tool", "gripper"}:
            return position + float(sign) * quat_rotate(orientation, offset)
        return position + float(sign) * offset

    @staticmethod
    def _target_orientation(*, task, robot_name: str, spec: dict, tracked_robots: dict, object_pose: dict | None = None):
        del task
        if spec.get("target_orientation") is not None:
            orientation = np.asarray(spec["target_orientation"], dtype=float)
        elif spec.get("orientation") is not None:
            orientation = np.asarray(spec["orientation"], dtype=float)
        elif spec.get("orientation_euler") is not None:
            orientation = euler_xyz_to_quat(spec["orientation_euler"])
        else:
            robot_state = tracked_robots.get(robot_name, {})
            if robot_state.get("orientation") is not None:
                orientation = np.asarray(robot_state["orientation"], dtype=float)
            else:
                orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        orientation_frame = str(
            spec.get("target_orientation_frame", spec.get("orientation_frame", "world"))
        ).lower()
        if orientation_frame in {"object", "local", "part"}:
            if object_pose is None:
                return None
            return normalize_quat(quat_multiply(object_pose["orientation"], orientation))
        return normalize_quat(orientation)

    @staticmethod
    def _object_pose(*, task, object_name: str, tracked_objects: dict):
        object_state = tracked_objects.get(object_name, {})
        if object_state.get("position") is not None and object_state.get("orientation") is not None:
            return {
                "position": np.asarray(object_state["position"], dtype=float),
                "orientation": normalize_quat(object_state["orientation"]),
            }
        try:
            position, orientation = task._resolve_object(object_name).get_pose()  # noqa: SLF001
        except Exception:
            return None
        return {
            "position": np.asarray(position, dtype=float),
            "orientation": normalize_quat(orientation),
        }

    @staticmethod
    def _current_robot_pose(*, task, robot_name: str, tracked_robots: dict):
        robot_state = tracked_robots.get(robot_name, {})
        if robot_state.get("position") is not None and robot_state.get("orientation") is not None:
            return {
                "position": np.asarray(robot_state["position"], dtype=float),
                "orientation": normalize_quat(robot_state["orientation"]),
            }
        try:
            position, orientation = task._get_robot_task_pose(robot_name)  # noqa: SLF001
        except Exception:
            return None
        return {
            "position": np.asarray(position, dtype=float),
            "orientation": normalize_quat(orientation),
        }

    def _current_arm_q(self, task, robot_name: str) -> np.ndarray | None:
        robot = task.robots.get(robot_name)
        if robot is None:
            return None
        controller = robot.controllers.get(_ARM_JOINT_CONTROLLER)
        if controller is not None and hasattr(controller, "get_joint_subset"):
            subset = controller.get_joint_subset()
        else:
            subset = getattr(controller, "joint_subset", None) if controller is not None else None
        if subset is not None:
            try:
                joint_positions = self._coerce_arm_q(subset.get_joint_positions())
                if joint_positions is not None:
                    return joint_positions
            except Exception:
                pass
        articulation = getattr(robot, "articulation", None)
        if articulation is not None:
            try:
                indices = np.asarray([articulation.get_dof_index(name) for name in _UR5E_ARM_JOINT_NAMES], dtype=np.int64)
                joint_positions = self._coerce_arm_q(articulation.get_joint_positions(joint_indices=indices))
                if joint_positions is not None:
                    return joint_positions
            except Exception:
                pass
            try:
                all_joint_positions = np.asarray(articulation.get_joint_positions(), dtype=float)
                dof_names = list(getattr(articulation, "dof_names", []) or [])
                if dof_names:
                    indices = [dof_names.index(name) for name in _UR5E_ARM_JOINT_NAMES if name in dof_names]
                    if len(indices) == len(_UR5E_ARM_JOINT_NAMES):
                        joint_positions = self._coerce_arm_q(all_joint_positions[np.asarray(indices, dtype=np.int64)])
                        if joint_positions is not None:
                            return joint_positions
                joint_positions = self._coerce_arm_q(all_joint_positions[: len(_UR5E_ARM_JOINT_NAMES)])
                if joint_positions is not None:
                    return joint_positions
            except Exception:
                pass
        last_q = self._last_arm_command_q.get(robot_name)
        if last_q is not None:
            return np.asarray(last_q, dtype=float).copy()
        try:
            obs = robot.get_obs()
            for control in obs.get("joint_action", []) or []:
                joint_positions = self._coerce_arm_q(control.get("joint_positions"))
                if joint_positions is not None:
                    return joint_positions
        except Exception:
            pass
        return None

    @staticmethod
    def _coerce_arm_q(joint_positions, *, bound_revolute: bool = True) -> np.ndarray | None:
        if joint_positions is None:
            return None
        try:
            values = np.asarray(joint_positions, dtype=float).reshape(-1)
        except Exception:
            return None
        if values.shape[0] < len(_UR5E_ARM_JOINT_NAMES):
            return None
        values = values[: len(_UR5E_ARM_JOINT_NAMES)]
        if not np.all(np.isfinite(values)):
            return None
        if not bound_revolute:
            return values.copy()
        return UR5ePlumbersBlockAtomicSkillAdapter._bounded_revolute_joint_values(values)

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
            task_cache = getattr(task, "_ur5e_plumbers_last_arm_command_q", None)
            if task_cache is None:
                task_cache = {}
                setattr(task, "_ur5e_plumbers_last_arm_command_q", task_cache)
            task_cache[robot_name] = joint_positions.copy()

    def _last_command_q(self, *, task, robot_name: str) -> np.ndarray | None:
        task_cache = getattr(task, "_ur5e_plumbers_last_arm_command_q", None)
        last_q = None
        if isinstance(task_cache, dict):
            last_q = task_cache.get(robot_name)
        if last_q is None:
            last_q = self._last_arm_command_q.get(robot_name)
        return self._coerce_arm_q(last_q, bound_revolute=False)

    def _command_reference_q(self, *, task, robot_name: str, current_q: np.ndarray | None, spec: dict) -> np.ndarray | None:
        current_q = self._coerce_arm_q(current_q)
        reference_mode = str(spec.get("ik_reference_mode", spec.get("reference_mode", ""))).strip().lower()
        last_q = self._last_command_q(task=task, robot_name=robot_name)
        if reference_mode in {"current", "current_q", "actual", "measured"}:
            if current_q is not None and last_q is not None:
                return self._unwrap_to_reference(
                    target_q=current_q,
                    reference_q=last_q,
                    preferred_abs_limit=None,
                    hard_preferred_abs_limit=False,
                )
            return current_q
        if last_q is None:
            return current_q
        if current_q is None:
            return last_q
        reset_threshold = spec.get("ik_reference_reset_threshold")
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
        if not bool(spec.get("enforce_continuous_joint_commands", True)):
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
        explicit = spec.get("max_command_joint_step")
        raw_limit = explicit if explicit is not None else spec.get("max_joint_step")
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
                default_cap = float(spec.get("default_max_command_joint_step", 0.08))
                if default_cap > 0.0:
                    scalar_limit = min(scalar_limit, default_cap)
                limits = np.full(int(joint_count), scalar_limit, dtype=float)
                wrist_cap = float(spec.get("default_max_command_wrist_joint_step", 0.025))
                if wrist_cap > 0.0 and limits.shape[0] >= 6:
                    limits[3:6] = np.minimum(limits[3:6], wrist_cap)
                return limits
            return scalar_limit
        limits = np.full(int(joint_count), float(limit_values[-1]), dtype=float)
        copy_count = min(int(joint_count), int(limit_values.size))
        limits[:copy_count] = limit_values[:copy_count]
        return limits

    @staticmethod
    def _ik_branch_jump_detected(*, reference_q: np.ndarray, target_q: np.ndarray, spec: dict) -> bool:
        reference_q = np.asarray(reference_q, dtype=float)
        target_q = np.asarray(target_q, dtype=float)
        if reference_q.shape != target_q.shape:
            return False
        max_joint_step = float(spec.get("max_joint_step", spec.get("close_gate_max_joint_step", 0.035)))
        default_limit = max(0.18, max_joint_step * 4.0)
        jump_limit = float(spec.get("ik_branch_jump_limit", default_limit))
        if jump_limit <= 0.0:
            return False
        branch_joint_indices = spec.get("ik_branch_guard_joint_indices", [0, 3, 5])
        try:
            indices = [int(index) for index in branch_joint_indices]
        except Exception:
            indices = [0, 3, 5]
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
        if ik_controller is None or not hasattr(ik_controller, "_kinematics_solver"):
            return None
        spec = spec or {}
        target_position = np.asarray(target_pose["position"], dtype=float) / ik_controller._robot_scale  # noqa: SLF001
        target_orientation = np.asarray(target_pose["orientation"], dtype=float)
        warm_start = self._coerce_arm_q(warm_start)
        try:
            ik_base_pose = ik_controller.get_ik_base_world_pose()
            ik_controller._kinematics_solver.set_robot_base_pose(  # noqa: SLF001
                robot_position=ik_base_pose[0] / ik_controller._robot_scale,  # noqa: SLF001
                robot_orientation=ik_base_pose[1],
            )
            used_warm_start = warm_start is not None and bool(spec.get("use_command_warm_start", True))
            if used_warm_start:
                solver_wrapper = ik_controller._kinematics_solver  # noqa: SLF001
                raw_solver = None
                get_raw_solver = getattr(solver_wrapper, "get_kinematics_solver", None)
                if callable(get_raw_solver):
                    raw_solver = get_raw_solver()
                if raw_solver is None:
                    raw_solver = getattr(solver_wrapper, "_kinematics_solver", None)
                if raw_solver is None:
                    raw_solver = getattr(solver_wrapper, "_kinematics", None)
                get_ee_frame = getattr(solver_wrapper, "get_end_effector_frame", None)
                ee_frame = get_ee_frame() if callable(get_ee_frame) else getattr(solver_wrapper, "_ee_frame", None)
                if raw_solver is not None and ee_frame is not None:
                    ik_result, success = raw_solver.compute_inverse_kinematics(
                        ee_frame,
                        target_position,
                        target_orientation,
                        warm_start=warm_start,
                        position_tolerance=spec.get("ik_position_tolerance"),
                        orientation_tolerance=spec.get("ik_orientation_tolerance"),
                    )
                    if success and ik_result is not None:
                        ik_result = np.asarray(ik_result, dtype=float)
                        if np.all(np.isfinite(ik_result)):
                            return ik_result
            if used_warm_start and bool(spec.get("require_warm_start_ik", False)):
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
                    bounded_candidate = bounded[
                        int(np.argmin(np.abs(bounded - reference_q[joint_index])))
                    ]
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

    @staticmethod
    def _gripper_value(command) -> float:
        if isinstance(command, str):
            lowered = command.strip().lower()
            if lowered == "open":
                return 1.0
            if lowered == "close":
                return 0.0
        try:
            return float(np.clip(float(command), 0.0, 1.0))
        except Exception:
            return 1.0

    def _gripper_command_value(self, *, task, robot_name: str, command) -> float:
        if isinstance(command, str) and command.strip().lower() in {"contact_hold", "hold_contact", "grasp_hold"}:
            hold_openness = self._last_gripper_hold_openness(task=task, robot_name=robot_name)
            if hold_openness is not None:
                return float(hold_openness)
            return 0.0
        return self._gripper_value(command)

    @staticmethod
    def _last_gripper_hold_openness(*, task, robot_name: str) -> float | None:
        cache = getattr(task, "_ur5e_plumbers_gripper_hold_openness", None)
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
        cache = getattr(task, "_ur5e_plumbers_gripper_hold_openness", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(task, "_ur5e_plumbers_gripper_hold_openness", cache)
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
        object_name = spec.get("held_object") or spec.get("object") or spec.get("object_name")
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
            return {"valid": False, "reason": "missing_object_name"}
        state = tracked_objects.get(object_name, {}) if isinstance(tracked_objects, dict) else {}
        linear_speed = state.get("linear_speed")
        angular_speed = state.get("angular_speed")
        linear_velocity = state.get("linear_velocity")
        angular_velocity = state.get("angular_velocity")
        is_static = state.get("is_static")
        pose_stable_override = state.get("pose_stable_override")

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

        if linear_speed is None or angular_speed is None:
            velocity_fn = getattr(task, "_object_velocity_metrics", None)
            if callable(velocity_fn):
                try:
                    metrics = velocity_fn(object_name)
                    linear_speed = metrics.get("linear_speed", linear_speed)
                    angular_speed = metrics.get("angular_speed", angular_speed)
                    linear_velocity = metrics.get("linear_velocity", linear_velocity)
                    angular_velocity = metrics.get("angular_velocity", angular_velocity)
                    is_static = metrics.get("is_static", is_static)
                    pose_stable_override = metrics.get("pose_stable_override", pose_stable_override)
                except Exception:
                    pass

        valid = linear_speed is not None and angular_speed is not None
        return {
            "valid": bool(valid),
            "object": object_name,
            "linear_speed": None if linear_speed is None else float(linear_speed),
            "angular_speed": None if angular_speed is None else float(angular_speed),
            "linear_velocity": linear_velocity,
            "angular_velocity": angular_velocity,
            "is_static": None if is_static is None else bool(is_static),
            "pose_stable_override": None if pose_stable_override is None else bool(pose_stable_override),
        }

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
        max_slip = spec.get("max_object_tcp_slip")
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

        current_relative_world = np.asarray(object_pose["position"], dtype=float) - np.asarray(
            current_pose["position"], dtype=float
        )
        current_relative_tcp = quat_rotate(
            self._quat_conjugate(current_pose["orientation"]),
            current_relative_world,
        )
        if current_relative_tcp.shape != (3,) or not np.all(np.isfinite(current_relative_tcp)):
            return None
        state = self._grasp_slip_state.setdefault(phase_key, {})
        if "initial_relative_tcp_position" not in state:
            state["initial_relative_tcp_position"] = current_relative_tcp.copy()
            state["initial_relative_world_position"] = current_relative_world.copy()
            state["initial_object_position"] = np.asarray(object_pose["position"], dtype=float).copy()
            state["initial_tcp_position"] = np.asarray(current_pose["position"], dtype=float).copy()
            state["initial_tcp_orientation"] = normalize_quat(current_pose["orientation"]).copy()
            return None

        initial_relative_tcp = np.asarray(state["initial_relative_tcp_position"], dtype=float)
        slip = float(np.linalg.norm(current_relative_tcp - initial_relative_tcp))
        threshold = float(max_slip)
        if slip <= threshold:
            return None
        initial_relative_world = np.asarray(
            state.get("initial_relative_world_position", current_relative_world),
            dtype=float,
        )
        return {
            "object": object_name,
            "robot": robot_name,
            "slip": slip,
            "slip_frame": "tcp",
            "max_object_tcp_slip": threshold,
            "initial_relative_position": initial_relative_tcp.tolist(),
            "current_relative_position": current_relative_tcp.tolist(),
            "initial_relative_tcp_position": initial_relative_tcp.tolist(),
            "current_relative_tcp_position": current_relative_tcp.tolist(),
            "initial_relative_world_position": initial_relative_world.tolist(),
            "current_relative_world_position": current_relative_world.tolist(),
            "world_relative_delta": float(np.linalg.norm(current_relative_world - initial_relative_world)),
            "initial_tcp_orientation": np.asarray(state.get("initial_tcp_orientation"), dtype=float).tolist(),
            "current_tcp_orientation": normalize_quat(current_pose["orientation"]).tolist(),
            "initial_object_position": np.asarray(state.get("initial_object_position"), dtype=float).tolist(),
            "initial_tcp_position": np.asarray(state.get("initial_tcp_position"), dtype=float).tolist(),
            "current_object_position": np.asarray(object_pose["position"], dtype=float).tolist(),
            "current_tcp_position": np.asarray(current_pose["position"], dtype=float).tolist(),
        }

    @staticmethod
    def _close_object_motion_abort(
        *,
        close_detail: dict[str, Any],
        spec: dict,
        close_elapsed_steps: int,
    ) -> bool:
        if not bool(spec.get("close_abort_on_object_motion", False)):
            return False
        min_steps = int(spec.get("close_abort_min_steps", 0))
        if int(close_elapsed_steps) < min_steps:
            return False
        motion_detail = close_detail.get("motion_detail")
        if not isinstance(motion_detail, dict) or not bool(motion_detail.get("valid")):
            return False
        linear_speed = motion_detail.get("linear_speed")
        angular_speed = motion_detail.get("angular_speed")
        max_linear_speed = spec.get("close_abort_max_object_speed", spec.get("close_contact_max_object_speed"))
        max_angular_speed = spec.get("close_abort_max_angular_speed", spec.get("close_contact_max_angular_speed"))
        linear_abort = (
            max_linear_speed is not None
            and linear_speed is not None
            and float(linear_speed) > float(max_linear_speed)
        )
        angular_abort = (
            max_angular_speed is not None
            and angular_speed is not None
            and float(angular_speed) > float(max_angular_speed)
        )
        return bool(linear_abort or angular_abort)

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
        min_steps = max(int(spec.get("close_until_contact_min_steps", spec.get("close_ramp_steps", 24))), 0)
        required_stable_steps = max(int(spec.get("close_contact_stable_steps", 8)), 1)
        stall_delta = float(spec.get("close_contact_stall_joint_delta", 0.0015))
        blocked_margin = float(spec.get("close_contact_blocked_joint_margin", 0.025))
        min_closure = float(spec.get("close_contact_min_joint_closure", 0.05))
        hold_squeeze_margin = float(spec.get("close_contact_hold_squeeze_margin", 0.04))

        gripper_q = self._current_gripper_q(task=task, robot_name=robot_name)
        open_q, closed_q = self._gripper_open_closed_q(task=task, robot_name=robot_name)
        last_q = state.get("last_gripper_q")
        joint_delta = None
        if gripper_q is not None and last_q is not None:
            joint_delta = abs(float(gripper_q) - float(last_q))
        if gripper_q is not None:
            state["last_gripper_q"] = float(gripper_q)

        contact_ready = False
        contact_detail: dict[str, Any] = {"contact_checked": False}
        if bool(spec.get("use_contact_for_close_until_contact", True)):
            contact_ready, contact_detail = self._grasp_contact_ready(
                task=task,
                robot_name=robot_name,
                spec={**spec, "require_dual_finger_contact": spec.get("require_dual_finger_contact", True)},
            )
            contact_detail["contact_checked"] = True

        blocked_before_full_close = False
        moved_from_open = False
        target_q = None
        reached_gripper_target = False
        if gripper_q is not None and open_q is not None and closed_q is not None:
            blocked_before_full_close = abs(float(gripper_q) - float(closed_q)) >= blocked_margin
            moved_from_open = abs(float(gripper_q) - float(open_q)) >= min_closure
            target_q = float(closed_q) + float(gripper_openness) * (float(open_q) - float(closed_q))
            reached_gripper_target = abs(float(gripper_q) - target_q) <= float(
                spec.get("close_gripper_target_tolerance", 0.025)
            )
        stalled = bool(joint_delta is not None and joint_delta <= stall_delta)
        contact_candidate = bool(contact_ready and moved_from_open)
        stall_contact = bool(
            bool(spec.get("use_joint_stall_for_close_until_contact", True))
            and stalled
            and blocked_before_full_close
            and moved_from_open
        )
        detected_clamp = bool(contact_candidate or stall_contact)

        if detected_clamp and close_elapsed_steps >= min_steps and state.get("hold_gripper_openness") is None:
            hold_openness = self._gripper_openness_from_q(
                gripper_q=gripper_q,
                open_q=open_q,
                closed_q=closed_q,
                squeeze_margin=hold_squeeze_margin,
            )
            if hold_openness is not None:
                max_hold_openness = spec.get("max_hold_gripper_openness", spec.get("closed_openness"))
                if max_hold_openness is not None:
                    hold_openness = min(float(hold_openness), float(max_hold_openness))
                state["hold_gripper_openness"] = float(hold_openness)
                self._remember_gripper_hold_openness(
                    task=task,
                    robot_name=robot_name,
                    openness=float(hold_openness),
                )

        if detected_clamp:
            state["close_contact_stable_steps"] = int(state.get("close_contact_stable_steps", 0)) + 1
        else:
            state["close_contact_stable_steps"] = 0

        motion_ready = True
        motion_detail: dict[str, Any] = {"checked": False}
        motion_stable_steps = 0
        max_linear_speed = spec.get("close_contact_max_object_speed")
        max_angular_speed = spec.get("close_contact_max_angular_speed")
        if max_linear_speed is not None or max_angular_speed is not None:
            motion_detail = self._object_motion_detail(
                task=task,
                object_name=self._object_name_from_spec(spec),
                tracked_objects=tracked_objects,
            )
            motion_detail["checked"] = True
            linear_speed = motion_detail.get("linear_speed")
            angular_speed = motion_detail.get("angular_speed")
            linear_ok = max_linear_speed is None or (
                linear_speed is not None and float(linear_speed) <= float(max_linear_speed)
            )
            angular_ok = max_angular_speed is None or (
                angular_speed is not None and float(angular_speed) <= float(max_angular_speed)
            )
            static_override = bool(motion_detail.get("is_static") or motion_detail.get("pose_stable_override"))
            motion_ready = bool(motion_detail.get("valid") and (static_override or (linear_ok and angular_ok)))
            if detected_clamp and motion_ready:
                state["close_contact_motion_stable_steps"] = int(
                    state.get("close_contact_motion_stable_steps", 0)
                ) + 1
            else:
                state["close_contact_motion_stable_steps"] = 0
            motion_stable_steps = int(state.get("close_contact_motion_stable_steps", 0))

        stable_steps = int(state.get("close_contact_stable_steps", 0))
        required_motion_stable_steps = 0
        if max_linear_speed is not None or max_angular_speed is not None:
            required_motion_stable_steps = max(int(spec.get("close_contact_motion_stable_steps", 1)), 1)
        motion_stable_ready = bool(
            required_motion_stable_steps <= 0 or motion_stable_steps >= required_motion_stable_steps
        )
        closed_target_candidate = bool(
            bool(spec.get("allow_closed_gripper_completion", False))
            and reached_gripper_target
            and moved_from_open
        )
        if closed_target_candidate and motion_ready:
            state["close_closed_target_stable_steps"] = int(
                state.get("close_closed_target_stable_steps", 0)
            ) + 1
        else:
            state["close_closed_target_stable_steps"] = 0
        closed_target_stable_steps = int(state.get("close_closed_target_stable_steps", 0))
        required_closed_target_stable_steps = max(
            int(spec.get("close_closed_target_stable_steps", required_motion_stable_steps or 1)),
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
        reason = (
            "contact"
            if contact_candidate
            else "joint_stall"
            if stall_contact
            else "closed_target"
            if closed_target_ready
            else "closing"
        )
        return ready, {
            "closed": ready,
            "close_until_contact": True,
            "completion_reason": reason,
            "close_elapsed_steps": int(close_elapsed_steps),
            "min_steps": min_steps,
            "stable_steps": stable_steps,
            "required_stable_steps": required_stable_steps,
            "motion_ready": motion_ready,
            "motion_stable_steps": motion_stable_steps,
            "required_motion_stable_steps": required_motion_stable_steps,
            "motion_detail": motion_detail,
            "allow_closed_gripper_completion": bool(spec.get("allow_closed_gripper_completion", False)),
            "target_gripper_joint_position": target_q,
            "reached_gripper_target": reached_gripper_target,
            "closed_target_candidate": closed_target_candidate,
            "closed_target_stable_steps": closed_target_stable_steps,
            "required_closed_target_stable_steps": required_closed_target_stable_steps,
            "closed_target_ready": closed_target_ready,
            "gripper_openness_command": float(gripper_openness),
            "hold_gripper_openness": state.get("hold_gripper_openness"),
            "gripper_joint_position": None if gripper_q is None else float(gripper_q),
            "gripper_joint_delta": None if joint_delta is None else float(joint_delta),
            "gripper_open_position": None if open_q is None else float(open_q),
            "gripper_closed_position": None if closed_q is None else float(closed_q),
            "blocked_before_full_close": blocked_before_full_close,
            "moved_from_open": moved_from_open,
            "stalled": stalled,
            "contact_ready": contact_ready,
            "contact_candidate": contact_candidate,
            "stall_contact": stall_contact,
            "detected_clamp": detected_clamp,
            "contact_detail": contact_detail,
        }

    @staticmethod
    def _current_gripper_q(*, task, robot_name: str) -> float | None:
        robot = task.robots.get(robot_name)
        if robot is None:
            return None
        dof_name = str(getattr(getattr(robot, "config", None), "gripper_dof_name", None) or "finger_joint")
        articulation = getattr(robot, "articulation", None)
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
                dof_names = list(getattr(articulation, "dof_names", []) or [])
                if dof_name in dof_names:
                    return float(values[dof_names.index(dof_name)])
            except Exception:
                pass
        try:
            controller = robot.controllers.get(_GRIPPER_CONTROLLER)
            obs = controller.get_obs()
            values = np.asarray(obs.get("gripper_pos"), dtype=float).reshape(-1)
            if values.size:
                return float(values[0])
        except Exception:
            pass
        return None

    @staticmethod
    def _gripper_open_closed_q(*, task, robot_name: str) -> tuple[float | None, float | None]:
        robot = task.robots.get(robot_name)
        config = getattr(robot, "config", None) if robot is not None else None
        try:
            open_q = float(getattr(config, "gripper_open_position"))
            closed_q = float(getattr(config, "gripper_closed_position"))
            return open_q, closed_q
        except Exception:
            return None, None

    @staticmethod
    def _mark_complete(*, task, robot_name: str, skill_name: str, detail: dict[str, Any]) -> None:
        marker = getattr(task, "mark_local_skill_complete", None)
        if callable(marker):
            marker(robot_name=robot_name, skill_name=skill_name, detail=detail)

    def _grasp_contact_ready(self, *, task, robot_name: str, spec: dict) -> tuple[bool, dict[str, Any]]:
        object_name = str(spec.get("object", spec.get("object_name", spec.get("held_object", ""))))
        if not object_name:
            return False, {"reason": "missing_object_for_grasp_check"}
        metrics_fn = getattr(task, "_gripper_contact_metrics", None)
        if not callable(metrics_fn):
            return False, {"reason": "gripper_contact_metrics_unavailable", "object": object_name}
        attach_spec = {
            "require_dual_finger_contact": bool(spec.get("require_dual_finger_contact", True)),
            "require_force_contact": bool(
                spec.get("require_force_contact", spec.get("require_contact_report", False))
            ),
            "finger_contact_distance": float(spec.get("finger_contact_distance", 0.006)),
            "contact_force_threshold": float(spec.get("contact_force_threshold", 0.2)),
            "physical_attach_surface_gap": float(spec.get("physical_attach_surface_gap", 0.006)),
        }
        for key in (
            "contact_box_scale",
            "contact_box_half_extents",
            "contact_box_offset",
            "caging_contact_distance",
            "physical_grasp_min_opening_ratio",
            "strict_finger_surface_gap",
            "strict_finger_contact_distance",
        ):
            if key in spec:
                attach_spec[key] = spec[key]
        try:
            metrics = metrics_fn(object_name, robot_name, attach_spec=attach_spec)
        except Exception as exc:
            return False, {
                "reason": "gripper_contact_metrics_error",
                "object": object_name,
                "error": str(exc),
            }

        strict_ready = False
        strict_fn = getattr(task, "_strict_physical_grasp_contact", None)
        if callable(strict_fn):
            try:
                strict_ready = bool(
                    strict_fn(object_name, metrics, attach_spec=attach_spec).get("physical_contact_ready")
                )
            except Exception:
                strict_ready = False
        if bool(spec.get("require_strict_physical_contact", False)):
            contact_ready = strict_ready
        else:
            contact_ready = bool(metrics.get("contact_ready") or strict_ready)
        return contact_ready, {
            "object": object_name,
            "contact_ready": contact_ready,
            "strict_contact_ready": strict_ready,
            "contact_metrics": metrics,
        }

    def _failure_or_hold(
        self,
        task,
        robot_name: str,
        spec: dict,
        reason: str,
        diagnostics: dict[str, Any] | None = None,
    ) -> dict:
        if bool(spec.get("require_success", False)):
            return {
                "__local_skill_failure__": True,
                "reason": reason,
                "diagnostics": {
                    "skill": spec.get("name"),
                    "robot": robot_name,
                    **(diagnostics or {}),
                },
            }
        action = self._hold_joint_action(task=task, robot_name=robot_name)
        gripper_command = spec.get("gripper_command")
        if gripper_command is not None:
            action[_GRIPPER_CONTROLLER] = [
                self._gripper_command_value(task=task, robot_name=robot_name, command=gripper_command)
            ]
        return action
