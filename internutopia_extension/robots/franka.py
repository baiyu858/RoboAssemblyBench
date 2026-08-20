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

from internutopia.core.robot.articulation_action import ArticulationAction
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
            gripper_open_position=np.full(2, float(config.gripper_open_position), dtype=float),
            gripper_closed_position=np.full(2, float(config.gripper_closed_position), dtype=float),
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
        self._apply_configured_arm_drive()
        self._apply_configured_gripper_drive()
        self._apply_configured_initial_joint_state()
        try:
            self.articulation.set_solver_position_iteration_count(32)
            self.articulation.set_solver_velocity_iteration_count(16)
        except Exception:
            pass

        self._apply_gripper_contact_material()

    def _apply_configured_initial_joint_state(self):
        """Atomically initialize measured joints and their PhysX drive targets."""

        configured = self.config.initial_joint_positions or {}
        if not configured:
            return
        joint_names = list(configured)
        try:
            joint_indices = np.asarray(
                [self.articulation.get_dof_index(name) for name in joint_names],
                dtype=np.int64,
            )
            joint_positions = np.asarray([configured[name] for name in joint_names], dtype=float)
            if not np.all(np.isfinite(joint_positions)):
                raise ValueError(f'non-finite joint positions: {joint_positions.tolist()}')
            joint_velocities = np.zeros_like(joint_positions)

            self.articulation.set_joint_positions(
                positions=joint_positions,
                joint_indices=joint_indices,
            )
            self.articulation.set_joint_velocities(
                velocities=joint_velocities,
                joint_indices=joint_indices,
            )
            self.articulation.apply_action(
                ArticulationAction(
                    joint_positions=joint_positions.copy(),
                    joint_velocities=joint_velocities,
                    joint_indices=joint_indices.copy(),
                )
            )

            measured_positions = np.asarray(
                self.articulation.get_joint_positions(joint_indices=joint_indices),
                dtype=float,
            ).reshape(-1)
            measured_velocities = np.asarray(
                self.articulation.get_joint_velocities(joint_indices=joint_indices),
                dtype=float,
            ).reshape(-1)
            if measured_positions.shape != joint_positions.shape or not np.allclose(
                measured_positions,
                joint_positions,
                rtol=1.0e-5,
                atol=1.0e-5,
            ):
                raise RuntimeError(
                    f'position readback {measured_positions.tolist()} does not match '
                    f'target {joint_positions.tolist()}'
                )
            if measured_velocities.shape != joint_velocities.shape or not np.allclose(
                measured_velocities,
                joint_velocities,
                rtol=0.0,
                atol=1.0e-5,
            ):
                raise RuntimeError(
                    f'velocity readback {measured_velocities.tolist()} is not zero'
                )
        except Exception as exc:
            message = f'franka {self.config.name}: failed to initialize configured joint state: {exc}'
            log.warning(message)
            raise RuntimeError(message) from exc

        log.info(
            f'franka {self.config.name}: initialized joint state '
            f'{dict(zip(joint_names, measured_positions.tolist()))}'
        )

    def _apply_configured_arm_drive(self):
        """Replace the asset's rigid tracking drive when a task profile requests it."""

        stiffness = self.config.arm_joint_stiffness
        damping = self.config.arm_joint_damping
        max_force = self.config.arm_joint_max_force
        if stiffness is None and damping is None and max_force is None:
            return
        joint_names = [f'panda_joint{index}' for index in range(1, 8)]
        try:
            joint_indices = np.asarray(
                [self.articulation.get_dof_index(name) for name in joint_names],
                dtype=np.int64,
            )
        except Exception as exc:
            raise RuntimeError(f'franka {self.config.name}: failed to resolve arm DOFs') from exc

        requested = {
            'stiffness': stiffness,
            'damping': damping,
            'max_force': max_force,
        }
        physics_properties = {
            'stiffness': ('get_dof_stiffnesses', 'set_dof_stiffnesses'),
            'damping': ('get_dof_dampings', 'set_dof_dampings'),
            'max_force': ('get_dof_max_forces', 'set_dof_max_forces'),
        }
        try:
            physics_view = self.articulation._articulation_view._physics_view
            for property_name, configured_value in requested.items():
                if configured_value is None:
                    continue
                getter_name, setter_name = physics_properties[property_name]
                values = np.asarray(getattr(physics_view, getter_name)()).copy()
                if values.ndim == 1:
                    values = np.expand_dims(values, axis=0)
                values[0, joint_indices] = configured_value
                getattr(physics_view, setter_name)(data=values, indices=[0])

            readback = {}
            for property_name, configured_value in requested.items():
                if configured_value is None:
                    continue
                getter_name, _ = physics_properties[property_name]
                values = np.asarray(getattr(physics_view, getter_name)())
                if values.ndim == 1:
                    values = np.expand_dims(values, axis=0)
                arm_values = np.asarray(values[0, joint_indices], dtype=float)
                if not np.allclose(arm_values, float(configured_value), rtol=1.0e-5, atol=1.0e-4):
                    raise RuntimeError(
                        f'{property_name} readback {arm_values.tolist()} '
                        f'does not match requested value {configured_value}'
                    )
                readback[property_name] = arm_values.tolist()
        except Exception as exc:
            message = f'franka {self.config.name}: failed to configure PhysX arm drive: {exc}'
            log.warning(message)
            raise RuntimeError(message) from exc

        log.info(f'franka {self.config.name}: configured PhysX arm drive {readback}')

    def _apply_configured_gripper_drive(self):
        """Configure both Panda finger drives with their prismatic-drive values."""

        requested = {
            'stiffness': self.config.gripper_joint_stiffness,
            'damping': self.config.gripper_joint_damping,
            'max_force': self.config.gripper_joint_max_force,
            'friction': self.config.gripper_joint_friction,
        }
        if all(value is None for value in requested.values()):
            return

        joint_names = ['panda_finger_joint1', 'panda_finger_joint2']
        try:
            joint_indices = np.asarray(
                [self.articulation.get_dof_index(name) for name in joint_names],
                dtype=np.int64,
            )
        except Exception as exc:
            raise RuntimeError(f'franka {self.config.name}: failed to resolve gripper DOFs') from exc

        physics_properties = {
            'stiffness': ('get_dof_stiffnesses', 'set_dof_stiffnesses'),
            'damping': ('get_dof_dampings', 'set_dof_dampings'),
            'max_force': ('get_dof_max_forces', 'set_dof_max_forces'),
            'friction': ('get_dof_friction_coefficients', 'set_dof_friction_coefficients'),
        }
        try:
            physics_view = self.articulation._articulation_view._physics_view
            for property_name, configured_value in requested.items():
                if configured_value is None:
                    continue
                getter_name, setter_name = physics_properties[property_name]
                values = np.asarray(getattr(physics_view, getter_name)()).copy()
                if values.ndim == 1:
                    values = np.expand_dims(values, axis=0)
                values[0, joint_indices] = configured_value
                getattr(physics_view, setter_name)(data=values, indices=[0])

            readback = {}
            for property_name, configured_value in requested.items():
                if configured_value is None:
                    continue
                getter_name, _ = physics_properties[property_name]
                values = np.asarray(getattr(physics_view, getter_name)())
                if values.ndim == 1:
                    values = np.expand_dims(values, axis=0)
                gripper_values = np.asarray(values[0, joint_indices], dtype=float)
                if not np.allclose(gripper_values, float(configured_value), rtol=1.0e-5, atol=1.0e-4):
                    raise RuntimeError(
                        f'{property_name} readback {gripper_values.tolist()} '
                        f'does not match requested value {configured_value}'
                    )
                readback[property_name] = gripper_values.tolist()
        except Exception as exc:
            message = f'franka {self.config.name}: failed to configure PhysX gripper drive: {exc}'
            log.warning(message)
            raise RuntimeError(message) from exc

        log.info(f'franka {self.config.name}: configured PhysX gripper drive {readback}')

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
        # Dataset state must describe the simulated Panda hand.  Lula FK is
        # retained separately as an IK-frame diagnostic rather than silently
        # replacing the physical pose used by grasp/contact code.
        obs['eef_position'] = eef_pose[0]
        obs['eef_orientation'] = eef_pose[1]
        if 'arm_ik_controller' in self.controllers:
            ik_obs = self.controllers['arm_ik_controller'].get_obs()
            obs['eef_kinematics_position'] = ik_obs.get('eef_position')
            obs['eef_kinematics_orientation'] = ik_obs.get('eef_orientation')

        # common
        for c_obs_name, controller_obs in self.controllers.items():
            obs['controllers'][c_obs_name] = controller_obs.get_obs()
        for sensor_name, sensor_obs in self.sensors.items():
            obs['sensors'][sensor_name] = sensor_obs.get_data()
        return self._make_ordered(obs)
