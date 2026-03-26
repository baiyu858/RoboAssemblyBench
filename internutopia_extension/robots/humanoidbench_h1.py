from internutopia.core.robot.robot import BaseRobot
from internutopia.core.scene.scene import IScene
from internutopia_extension.configs.robots.humanoidbench_h1 import (
    HumanoidBenchH1RobotCfg,
)
from internutopia_extension.robots.h1_with_hand import H1WithHandRobot


@BaseRobot.register('HumanoidBenchH1Robot')
class HumanoidBenchH1Robot(H1WithHandRobot):
    """Isaac Sim compatible wrapper for the HumanoidBench H1 embodiment."""

    def __init__(self, config: HumanoidBenchH1RobotCfg, scene: IScene):
        super().__init__(config, scene)
        self._robot_left_hand = None
        self._robot_right_hand = None

    def post_reset(self):
        super().post_reset()
        self._robot_left_hand = self._rigid_body_map.get(self.config.prim_path + '/left_hand_link')
        self._robot_right_hand = self._rigid_body_map.get(self.config.prim_path + '/right_hand_link')

    def get_left_hand_pose(self):
        if self._robot_left_hand is not None:
            return self._robot_left_hand.get_pose()
        return self.get_pose()

    def get_right_hand_pose(self):
        if self._robot_right_hand is not None:
            return self._robot_right_hand.get_pose()
        return self.get_pose()
