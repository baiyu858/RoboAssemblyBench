# Copyright (c) 2021-2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
import os
from collections import OrderedDict
from typing import Any, List, Optional

import numpy as np

from internutopia.core.robot.isaacsim.articulation import IsaacsimArticulation
from internutopia.core.robot.rigid_body import IRigidBody
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia.core.util import log
from internutopia_extension.configs.robots.franka import FrankaRobotCfg


class Franka(IsaacsimArticulation):
    # TODO: change IsaacsimArticulation to IArticulation
    def __init__(
        self,
        prim_path: str,
        name: str = 'franka_robot',
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        end_effector_prim_name: Optional[str] = None,
        gripper_dof_names: Optional[List[str]] = None,
        gripper_open_position: Optional[np.ndarray] = None,
        gripper_closed_position: Optional[np.ndarray] = None,
        deltas: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
    ) -> None:
        from isaacsim.core.utils.stage import get_stage_units
        from isaacsim.robot.manipulators.grippers.parallel_gripper import (
            ParallelGripper,
        )

        self._end_effector = None
        self._gripper = None
        self._end_effector_prim_name = end_effector_prim_name
        if self._end_effector_prim_name is None:
            self._end_effector_prim_path = prim_path + '/panda_hand'
        else:
            self._end_effector_prim_path = prim_path + '/' + end_effector_prim_name
        if gripper_dof_names is None:
            gripper_dof_names = ['panda_finger_joint1', 'panda_finger_joint2']
        if gripper_open_position is None:
            # Fabrica's Panda model and the Franka joint limits both use a
            # 4 cm maximum displacement per finger.  A 5 cm range changes all
            # official grasp ratios and leaves the jaws visibly off the part.
            gripper_open_position = np.array([0.04, 0.04]) / get_stage_units()
        if gripper_closed_position is None:
            gripper_closed_position = np.array([0.0, 0.0])
        if deltas is None:
            # Match RoboFactory's visible multi-step open/close behavior instead of
            # snapping the fingers directly to the final command on a single step.
            deltas = np.array([0.0025, 0.0025]) / get_stage_units()
        super().__init__(
            usd_path=usd_path,
            prim_path=prim_path,
            name=name,
            position=position,
            orientation=orientation,
            scale=scale,
        )
        if gripper_dof_names is not None:
            self._gripper = ParallelGripper(
                end_effector_prim_path=self._end_effector_prim_path,
                joint_prim_names=gripper_dof_names,
                joint_opened_positions=gripper_open_position,
                joint_closed_positions=gripper_closed_position,
                action_deltas=deltas,
                # The bundled Franka USD exposes both finger joints directly. Driving only
                # the first mimic/drive joint clamps the observed opening to ~3mm.
                use_mimic_joints=False,
            )
            try:
                self._gripper.set_default_state(np.asarray(gripper_open_position, dtype=float))
            except Exception:
                pass
        return

    @property
    def end_effector(self) -> IRigidBody:
        return self._end_effector

    @property
    def gripper(self):
        return self._gripper

    def initialize(self, physics_sim_view=None) -> None:
        self.unwrap().initialize(physics_sim_view)
        self._end_effector = IRigidBody.create(prim_path=self._end_effector_prim_path, name=self.name + '_end_effector')
        self._end_effector.unwrap().initialize(physics_sim_view)
        self._gripper.initialize(
            physics_sim_view=physics_sim_view,
            articulation_apply_action_func=self.apply_action,
            get_joint_positions_func=self.get_joint_positions,
            set_joint_positions_func=self.set_joint_positions,
            dof_names=self.dof_names,
        )
        return

    def post_reset(self) -> None:
        self.unwrap().post_reset()
        self._gripper.post_reset()
        for dof_index in self.gripper.active_joint_indices:
            self._articulation_controller.switch_dof_control_mode(dof_index=dof_index, mode='position')
        return


@BaseRobot.register('FrankaRobot')
class FrankaRobot(BaseRobot):
    def __init__(self, config: FrankaRobotCfg, scene: IScene):
        super().__init__(config, scene)
        self._robot_ik_base = None
        self._start_position = np.array(config.position) if config.position is not None else None
        self._start_orientation = np.array(config.orientation) if config.orientation is not None else None

        log.debug(f'franka {config.name}: position    : ' + str(self._start_position))
        log.debug(f'franka {config.name}: orientation : ' + str(self._start_orientation))

        usd_path = config.usd_path

        log.debug(f'franka {config.name}: usd_path         : ' + str(usd_path))
        log.debug(f'franka {config.name}: config.prim_path : ' + str(config.prim_path))
        self._robot_scale = np.array([1.0, 1.0, 1.0])
        if config.scale is not None:
            self._robot_scale = np.array(config.scale)
        self.articulation = Franka(
            prim_path=config.prim_path,
            name=config.name,
            position=self._start_position,
            orientation=self._start_orientation,
            usd_path=os.path.abspath(usd_path),
            end_effector_prim_name=config.end_effector_prim_name,
            scale=self._robot_scale,
        )

        self.last_action = []

    def get_robot_scale(self):
        return self._robot_scale

    def get_robot_ik_base(self):
        return self._robot_ik_base

    def post_reset(self):
        super().post_reset()
        self._robot_ik_base = self._rigid_body_map[self.config.prim_path + '/panda_link0']
        try:
            self.articulation.set_solver_position_iteration_count(32)
            self.articulation.set_solver_velocity_iteration_count(16)
        except Exception:
            pass

        finger_joint_names = ['panda_finger_joint1', 'panda_finger_joint2']
        try:
            finger_indices = np.asarray(
                [self.articulation.get_dof_index(name) for name in finger_joint_names],
                dtype=np.int64,
            )
        except Exception:
            finger_indices = None
        if finger_indices is not None and finger_indices.size == 2:
            try:
                self.articulation.set_gains(
                    kps=np.asarray([2.0e4, 2.0e4], dtype=float),
                    kds=np.asarray([1.0e3, 1.0e3], dtype=float),
                    joint_indices=finger_indices,
                )
            except Exception:
                pass
            try:
                physics_view = self.articulation._articulation_view._physics_view
                max_forces = np.asarray(physics_view.get_dof_max_forces(), dtype=float)
                if max_forces.ndim == 1:
                    max_forces = np.expand_dims(max_forces, axis=0)
                max_forces[0, finger_indices] = np.maximum(max_forces[0, finger_indices], 300.0)
                physics_view.set_dof_max_forces(data=max_forces, indices=[0])

                friction_coefficients = np.asarray(
                    physics_view.get_dof_friction_coefficients(),
                    dtype=float,
                )
                if friction_coefficients.ndim == 1:
                    friction_coefficients = np.expand_dims(friction_coefficients, axis=0)
                friction_coefficients[0, finger_indices] = np.maximum(
                    friction_coefficients[0, finger_indices],
                    5.0,
                )
                physics_view.set_dof_friction_coefficients(data=friction_coefficients, indices=[0])
            except Exception:
                pass
        self._apply_gripper_contact_material()

    def _apply_gripper_contact_material(self):
        """Give the Franka fingers enough surface friction for real PhysX grasps."""
        try:
            from isaacsim.core.api.materials import PhysicsMaterial
        except Exception:
            try:
                from omni.isaac.core.materials import PhysicsMaterial
            except Exception:
                return

        try:
            material_name = f'{self.config.name}_finger_high_friction'
            physics_material = PhysicsMaterial(
                prim_path=f'/World/Physics_Materials/{material_name}',
                name=material_name,
                static_friction=3.0,
                dynamic_friction=2.5,
                restitution=0.0,
            )
        except Exception:
            return

        for link_name in ('panda_leftfinger', 'panda_rightfinger'):
            rigid_body = self._rigid_body_map.get(f'{self.config.prim_path}/{link_name}')
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
            'joint_efforts': numpy_to_list(action.joint_efforts),
            'joint_indices': numpy_to_list(action.joint_indices),
            'joint_positions': numpy_to_list(action.joint_positions),
            'joint_velocities': numpy_to_list(action.joint_velocities),
        }

    def apply_action(self, action: dict):
        """
        Args:
            action (dict): inputs for controllers.
        """
        self.last_action = []
        deferred_controls = []
        has_joint_override = 'arm_joint_controller' in action and 'arm_ik_controller' in action
        for controller_name, controller_action in action.items():
            if controller_name not in self.controllers:
                log.warn(f'unknown controller {controller_name} in action')
                continue
            controller = self.controllers[controller_name]
            control = controller.action_to_control(controller_action)
            if has_joint_override and controller_name == 'arm_ik_controller':
                # Keep IK solver state / controller observations fresh, but let the joint
                # controller own the actual arm execution to avoid conflicting commands.
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

        # custom
        obs = {
            'position': position,
            'orientation': orientation,
            'joint_action': self.get_last_action(),
            'controllers': {},
            'sensors': {},
        }

        eef_pose = self.articulation.end_effector.get_pose()
        obs['eef_body_position'] = eef_pose[0]
        obs['eef_body_orientation'] = eef_pose[1]
        if 'arm_ik_controller' in self.controllers:
            ik_obs = self.controllers['arm_ik_controller'].get_obs()
            obs['eef_position'] = ik_obs.get('eef_position', eef_pose[0])
            obs['eef_orientation'] = ik_obs.get('eef_orientation', eef_pose[1])
        else:
            obs['eef_position'] = eef_pose[0]
            obs['eef_orientation'] = eef_pose[1]

        # common
        for c_obs_name, controller_obs in self.controllers.items():
            obs['controllers'][c_obs_name] = controller_obs.get_obs()
        for sensor_name, sensor_obs in self.sensors.items():
            obs['sensors'][sensor_name] = sensor_obs.get_data()
        return self._make_ordered(obs)
