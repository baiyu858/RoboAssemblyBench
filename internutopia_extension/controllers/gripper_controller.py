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
        super().__init__(config, robot, scene)

    @staticmethod
    def _normalize_action(action: Any) -> str:
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
                return 'open' if value >= 0.5 else 'close'

        raise AssertionError(
            'gripper action must be one of "open"/"close" or a binary scalar '
            f'where 1=open and 0=close, but got {action!r}'
        )

    def forward(self, action: Any) -> ArticulationAction:
        return self._gripper.forward(self._normalize_action(action))

    def action_to_control(self, action: List | np.ndarray) -> ArticulationAction:
        """
        Args:
            action (List | np.ndarray): 1-element 1d array.
        """
        assert len(action) == 1, f'action must be a 1-element list/array, but got {action}'
        return self.forward(action[0])

    def get_obs(self) -> OrderedDict[str, Any]:
        return OrderedDict({'gripper_pos': self._gripper.get_joint_positions()})
