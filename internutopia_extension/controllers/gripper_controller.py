from collections import OrderedDict
from typing import Any, List

import numpy as np

from internutopia.core.robot.articulation_action import ArticulationAction
from internutopia.core.robot.controller import BaseController
from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia_extension.configs.controllers import GripperControllerCfg


@BaseController.register('GripperController')
class GripperController(BaseController):
    def __init__(self, config: GripperControllerCfg, robot: BaseRobot, scene: IScene):
        self._gripper = robot.articulation.gripper  # for franka is OK
        self._robot_config = getattr(robot, 'config', None)
        self._command_index = 0
        self.last_numeric_command = None
        super().__init__(config, robot, scene)

    @staticmethod
    def _normalize_action(action: Any) -> str | float:
        if isinstance(action, str):
            lowered = action.strip().lower()
            if lowered in {'open', '1', '1.0', 'true'}:
                return 'open'
            if lowered in {'close', '0', '0.0', 'false'}:
                return 'close'

        if isinstance(action, (bool, np.bool_)):
            return 'open' if bool(action) else 'close'

        if isinstance(action, (int, float, np.integer, np.floating)):
            value = float(action)
            if 0.0 <= value <= 1.0:
                if value <= 1e-6:
                    return 'close'
                if value >= 1.0 - 1e-6:
                    return 'open'
                return value

        raise AssertionError(
            'gripper action must be one of "open"/"close" or a scalar in [0, 1] '
            f'where 1=open and 0=close, but got {action!r}'
        )

    def _continuous_forward(self, openness: float) -> ArticulationAction:
        requested_openness = float(openness)
        opened_positions = np.asarray(self._gripper.joint_opened_positions, dtype=float)
        closed_positions = np.asarray(self._gripper.joint_closed_positions, dtype=float)
        physical_minimum = getattr(self._robot_config, 'gripper_min_openness', None)
        min_openness = float(
            (getattr(self._robot_config, 'gripper_close_openness', 0.0) or 0.0)
            if physical_minimum is None
            else physical_minimum
        )
        if not np.isfinite(min_openness) or not 0.0 <= min_openness < 1.0:
            raise ValueError('gripper_min_openness must be finite and in [0, 1)')
        openness = float(np.clip(openness, max(min_openness, 0.0), 1.0))
        joint_positions = closed_positions + openness * (opened_positions - closed_positions)
        joint_indices = getattr(self._gripper, 'active_joint_indices', None)
        if joint_indices is not None:
            joint_indices = np.asarray(joint_indices, dtype=np.int64)
            if joint_positions.shape[0] != joint_indices.shape[0]:
                joint_positions = joint_positions[: joint_indices.shape[0]]
        self._command_index += 1
        self.last_numeric_command = {
            'command_index': self._command_index,
            'requested_openness': requested_openness,
            'clipped_openness': openness,
            'minimum_openness': min_openness,
            'joint_positions': joint_positions.tolist(),
            'joint_indices': None if joint_indices is None else joint_indices.tolist(),
        }
        return ArticulationAction(joint_positions=joint_positions, joint_indices=joint_indices)

    def forward(self, action: Any) -> ArticulationAction:
        normalized_action = self._normalize_action(action)
        if isinstance(normalized_action, str):
            if normalized_action == 'close':
                return self._continuous_forward(0.0)
            # ParallelGripper.forward("open") uses a relative action_delta
            # when one is configured. On Isaac Sim 5.1 that full-articulation
            # action can be dropped by the indexed controller path, leaving a
            # Panda closed for the entire release phase. Use the same explicit
            # joint target as continuous commands for both endpoints.
            return self._continuous_forward(1.0)
        return self._continuous_forward(normalized_action)

    def action_to_control(self, action: List | np.ndarray) -> ArticulationAction:
        """
        Args:
            action (List | np.ndarray): 1-element 1d array.
        """
        assert len(action) == 1, f'action must be a 1-element list/array, but got {action}'
        return self.forward(action[0])

    def get_obs(self) -> OrderedDict[str, Any]:
        reader = getattr(self.robot, 'get_measured_joint_positions', None)
        indices = getattr(self._gripper, 'active_joint_indices', None)
        if reader is not None and indices is not None:
            return OrderedDict({'gripper_pos': reader(joint_indices=np.asarray(indices, dtype=np.int64))})
        return OrderedDict({'gripper_pos': self._gripper.get_joint_positions()})
