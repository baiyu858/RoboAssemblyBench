from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any, List, Optional

import numpy as np

from internutopia.core.robot.isaacsim.articulation import IsaacsimArticulation
from internutopia.core.robot.rigid_body import IRigidBody
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia.core.util import log
from internutopia_extension.configs.robots.ur5e import (
    DEFAULT_UR5E_READY_JOINTS,
    UR5eRobotCfg,
)


class UR5e(IsaacsimArticulation):
    def __init__(
        self,
        prim_path: str,
        name: str = "ur5e_robot",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        gripper_dof_name: Optional[str] = None,
        gripper_open_position: Optional[float] = None,
        gripper_closed_position: Optional[float] = None,
        deltas: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
    ) -> None:
        from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper

        self._end_effector = None
        self._gripper = None
        self._root_prim_path = prim_path
        self._end_effector_prim_name = end_effector_prim_name or "tool0"
        self._end_effector_prim_path = prim_path + "/" + self._end_effector_prim_name
        self._gripper_dof_name = gripper_dof_name or "finger_joint"
        if gripper_open_position is None:
            gripper_open_position = 0.0
        if gripper_closed_position is None:
            gripper_closed_position = 0.80
        if deltas is None:
            deltas = np.array([0.04], dtype=float)
        self._gripper_open_position = float(gripper_open_position)
        self._gripper_closed_position = float(gripper_closed_position)
        self._gripper_deltas = np.asarray(deltas, dtype=float)

        super().__init__(
            usd_path=usd_path,
            prim_path=prim_path,
            name=name,
            position=position,
            orientation=orientation,
            scale=scale,
        )
        self._resolve_end_effector_prim_path()
        self._gripper = self._make_gripper(ParallelGripper)

    def _make_gripper(self, gripper_cls=None):
        if gripper_cls is None:
            from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper as gripper_cls

        gripper = gripper_cls(
            end_effector_prim_path=self._end_effector_prim_path,
            joint_prim_names=[self._gripper_dof_name],
            joint_opened_positions=np.array([self._gripper_open_position], dtype=float),
            joint_closed_positions=np.array([self._gripper_closed_position], dtype=float),
            action_deltas=self._gripper_deltas,
            use_mimic_joints=True,
        )
        try:
            gripper.set_default_state(np.array([self._gripper_open_position], dtype=float))
        except Exception:
            pass
        return gripper

    def _resolve_end_effector_prim_path(self) -> str:
        try:
            from isaacsim.core.utils.prims import is_prim_path_valid
        except Exception:
            try:
                from omni.isaac.core.utils.prims import is_prim_path_valid
            except Exception:
                return self._end_effector_prim_path

        candidates = [
            self._end_effector_prim_name,
            "tool0",
            "flange",
            "wrist_3_link",
            "wrist_3_link/ft_frame",
        ]
        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            candidate_path = f"{self._root_prim_path}/{candidate}"
            try:
                if is_prim_path_valid(candidate_path):
                    self._end_effector_prim_name = candidate
                    self._end_effector_prim_path = candidate_path
                    return candidate_path
            except Exception:
                continue
        return self._end_effector_prim_path

    @property
    def end_effector(self) -> IRigidBody:
        return self._end_effector

    @property
    def gripper(self):
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        self.unwrap().initialize(physics_sim_view)
        previous_end_effector_path = self._end_effector_prim_path
        self._resolve_end_effector_prim_path()
        if self._end_effector_prim_path != previous_end_effector_path:
            self._gripper = self._make_gripper()
        self._end_effector = IRigidBody.create(prim_path=self._end_effector_prim_path, name=self.name + "_end_effector")
        self._end_effector.unwrap().initialize(physics_sim_view)
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )

    def post_reset(self) -> None:
        self.unwrap().post_reset()
        self._gripper.post_reset()
        for dof_index in self.gripper.active_joint_indices:
            self._articulation_controller.switch_dof_control_mode(dof_index=dof_index, mode="position")


class _ArticulationPoseProxy:
    def __init__(self, articulation: UR5e):
        self._articulation = articulation

    def get_pose(self):
        return self._articulation.get_pose()

    def get_local_pose(self):
        return self._articulation.get_local_pose()


@BaseRobot.register("UR5eRobot")
class UR5eRobot(BaseRobot):
    def __init__(self, config: UR5eRobotCfg, scene: IScene):
        super().__init__(config, scene)
        self._robot_ik_base = None
        self._start_position = np.array(config.position) if config.position is not None else None
        self._start_orientation = np.array(config.orientation) if config.orientation is not None else None
        self._robot_scale = np.array([1.0, 1.0, 1.0])
        if config.scale is not None:
            self._robot_scale = np.array(config.scale)

        if config.usd_path is None:
            raise ValueError("UR5eRobotCfg.usd_path must be set to a UR5e USD asset.")

        log.debug(f"ur5e {config.name}: position    : {self._start_position}")
        log.debug(f"ur5e {config.name}: orientation : {self._start_orientation}")
        log.debug(f"ur5e {config.name}: usd_path    : {config.usd_path}")

        self.articulation = UR5e(
            prim_path=config.prim_path,
            name=config.name,
            position=self._start_position,
            orientation=self._start_orientation,
            usd_path=os.path.abspath(config.usd_path),
            end_effector_prim_name=config.end_effector_prim_name,
            gripper_dof_name=config.gripper_dof_name,
            gripper_open_position=config.gripper_open_position,
            gripper_closed_position=config.gripper_closed_position,
            scale=self._robot_scale,
        )
        self.last_action = []

    def get_robot_scale(self):
        return self._robot_scale

    def get_robot_ik_base(self):
        return self._robot_ik_base

    def post_reset(self):
        super().post_reset()
        self._robot_ik_base = self._resolve_ik_base_rigid_body()
        self._apply_initial_joint_positions()
        self._configure_drive_gains()
        self._apply_gripper_contact_material()

    def _robot_rigid_body_by_suffix(self, suffix: str):
        suffix = f"/{suffix}"
        for prim_path, rigid_body in self._rigid_body_map.items():
            if prim_path.endswith(suffix):
                return rigid_body
        return None

    def _resolve_ik_base_rigid_body(self):
        candidate_names = [
            self.config.ik_base_prim_name,
            "base_link",
            "base",
            "base_link_inertia",
            "shoulder_link",
        ]
        seen = set()
        for candidate_name in candidate_names:
            if not candidate_name or candidate_name in seen:
                continue
            seen.add(candidate_name)
            rigid_body = self._robot_rigid_body_by_suffix(str(candidate_name))
            if rigid_body is not None:
                return rigid_body

        if self._rigid_body_map:
            first_path, first_body = next(iter(self._rigid_body_map.items()))
            log.warn(
                f"ur5e {self.config.name}: failed to resolve IK base {self.config.ik_base_prim_name!r}; "
                f"falling back to first rigid body {first_path!r}."
            )
            return first_body

        log.warn(
            f"ur5e {self.config.name}: no rigid bodies were found for IK base resolution; "
            "using the articulation root pose as IK base."
        )
        return _ArticulationPoseProxy(self.articulation)

    def _apply_initial_joint_positions(self) -> None:
        joint_targets = dict(DEFAULT_UR5E_READY_JOINTS)
        if self.config.initial_joint_positions:
            joint_targets.update({str(k): float(v) for k, v in self.config.initial_joint_positions.items()})
        for joint_name, joint_pos in joint_targets.items():
            try:
                joint_index = self.articulation.get_dof_index(joint_name)
                self.articulation.set_joint_positions(
                    np.array([float(joint_pos)], dtype=float),
                    joint_indices=np.array([joint_index], dtype=np.int64),
                )
            except Exception:
                continue

    def _configure_drive_gains(self) -> None:
        arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        try:
            arm_indices = np.asarray([self.articulation.get_dof_index(name) for name in arm_joint_names], dtype=np.int64)
            self.articulation.set_gains(
                kps=np.full(arm_indices.shape, 8.0e4, dtype=float),
                kds=np.full(arm_indices.shape, 4.0e3, dtype=float),
                joint_indices=arm_indices,
            )
        except Exception:
            pass
        try:
            gripper_index = np.asarray([self.articulation.get_dof_index(self.config.gripper_dof_name or "finger_joint")], dtype=np.int64)
            self.articulation.set_gains(
                kps=np.asarray([7.5e3], dtype=float),
                kds=np.asarray([1.73e2], dtype=float),
                joint_indices=gripper_index,
            )
        except Exception:
            pass

    def _apply_gripper_contact_material(self):
        try:
            from isaacsim.core.api.materials import PhysicsMaterial
        except Exception:
            try:
                from omni.isaac.core.materials import PhysicsMaterial
            except Exception:
                return

        try:
            material_name = f"{self.config.name}_robotiq_high_friction"
            physics_material = PhysicsMaterial(
                prim_path=f"/World/Physics_Materials/{material_name}",
                name=material_name,
                static_friction=3.0,
                dynamic_friction=2.5,
                restitution=0.0,
            )
        except Exception:
            return

        for link_name in (self.config.left_finger_link_name, self.config.right_finger_link_name):
            if not link_name:
                continue
            rigid_body = self._robot_rigid_body_by_suffix(str(link_name))
            if rigid_body is None:
                continue
            try:
                rigid_body.unwrap().apply_physics_material(physics_material)
            except Exception:
                pass

    @staticmethod
    def action_to_dict(action):
        def numpy_to_list(array):
            return array.tolist() if isinstance(array, np.ndarray) else array

        return {
            "joint_efforts": numpy_to_list(action.joint_efforts),
            "joint_indices": numpy_to_list(action.joint_indices),
            "joint_positions": numpy_to_list(action.joint_positions),
            "joint_velocities": numpy_to_list(action.joint_velocities),
        }

    def apply_action(self, action: dict):
        self.last_action = []
        deferred_controls = []
        has_joint_override = "arm_joint_controller" in action and "arm_ik_controller" in action
        for controller_name, controller_action in action.items():
            if controller_name not in self.controllers:
                log.warn(f"unknown controller {controller_name} in action")
                continue
            controller = self.controllers[controller_name]
            control = controller.action_to_control(controller_action)
            if has_joint_override and controller_name == "arm_ik_controller":
                self.last_action.append(self.action_to_dict(control))
                continue
            deferred_controls.append(control)
            self.last_action.append(self.action_to_dict(control))
        for control in deferred_controls:
            self.articulation.apply_action(control)

    def get_last_action(self):
        return self.last_action

    def get_obs(self) -> OrderedDict[str, Any]:
        position, orientation = self.articulation.get_pose()
        obs = {
            "position": position,
            "orientation": orientation,
            "joint_action": self.get_last_action(),
            "controllers": {},
            "sensors": {},
        }
        eef_pose = self.articulation.end_effector.get_pose()
        obs["eef_body_position"] = eef_pose[0]
        obs["eef_body_orientation"] = eef_pose[1]
        if "arm_ik_controller" in self.controllers:
            ik_obs = self.controllers["arm_ik_controller"].get_obs()
            obs["eef_position"] = ik_obs.get("eef_position", eef_pose[0])
            obs["eef_orientation"] = ik_obs.get("eef_orientation", eef_pose[1])
        else:
            obs["eef_position"] = eef_pose[0]
            obs["eef_orientation"] = eef_pose[1]

        for c_obs_name, controller_obs in self.controllers.items():
            obs["controllers"][c_obs_name] = controller_obs.get_obs()
        for sensor_name, sensor_obs in self.sensors.items():
            obs["sensors"][sensor_name] = sensor_obs.get_data()
        return self._make_ordered(obs)
