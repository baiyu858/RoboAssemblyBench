from __future__ import annotations

from collections import OrderedDict

import numpy as np

from internutopia.core.scene.scene import IScene
from internutopia.core.task import BaseTask
from internutopia_extension.configs.tasks.factory_box_carry_task import (
    FactoryBoxCarryTaskCfg,
)


@BaseTask.register('FactoryBoxCarryTask')
class FactoryBoxCarryTask(BaseTask):
    """
    A lightweight cooperative carry task for two robots.

    The first version deliberately focuses on a stable benchmark/demo pipeline:
    robots approach a shared box, attach when both are aligned, transport it
    cooperatively, and place it at the target zone.
    """

    def __init__(self, config: FactoryBoxCarryTaskCfg, scene: IScene):
        super().__init__(config, scene)
        self.step_counter = 0
        self.max_steps = config.max_steps
        self.phase = 'approach_pickup'
        self.phase_history = [self.phase]
        self.success = False
        self.attached = False
        self._box = None
        self._box_half_height = 0.15

    @property
    def cfg(self) -> FactoryBoxCarryTaskCfg:
        return self.config

    def _set_phase(self, phase: str):
        if phase != self.phase:
            self.phase = phase
            self.phase_history.append(phase)

    def _resolve_box(self):
        if self._box is not None:
            return
        box_obj = self.objects[self.cfg.box_name]
        self._box = self._scene.get(box_obj.name)
        if box_obj.config.scale is not None and len(box_obj.config.scale) >= 3:
            self._box_half_height = float(box_obj.config.scale[2]) * 0.5

    def _get_box_pose(self):
        self._resolve_box()
        return self._box.get_pose()

    def _get_robot_pair(self):
        left_name, right_name = self.cfg.robot_names
        return self.robots[left_name], self.robots[right_name]

    def _compute_standoff_targets(self, center: np.ndarray):
        left_target = np.array([center[0], center[1] - self.cfg.standoff_distance, 0.0], dtype=float)
        right_target = np.array([center[0], center[1] + self.cfg.standoff_distance, 0.0], dtype=float)
        return left_target, right_target

    def get_pickup_targets(self):
        box_position, _ = self._get_box_pose()
        return self._compute_standoff_targets(np.array(box_position))

    def get_drop_targets(self):
        return self._compute_standoff_targets(np.array(self.cfg.goal_position))

    def get_reach_targets(self):
        box_position, _ = self._get_box_pose()
        left_target = np.array([box_position[0], box_position[1] - self.cfg.attach_distance, self.cfg.carry_height])
        right_target = np.array([box_position[0], box_position[1] + self.cfg.attach_distance, self.cfg.carry_height])
        return left_target, right_target

    def _xy_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(np.array(a[:2]) - np.array(b[:2])))

    def _robots_reached(self, targets) -> bool:
        left_robot, right_robot = self._get_robot_pair()
        left_position, _ = left_robot.get_pose()
        right_position, _ = right_robot.get_pose()
        return (
            self._xy_distance(left_position, targets[0]) < self.cfg.goal_tolerance
            and self._xy_distance(right_position, targets[1]) < self.cfg.goal_tolerance
        )

    def _lift_box_with_team(self):
        left_robot, right_robot = self._get_robot_pair()
        left_position, _ = left_robot.get_pose()
        right_position, _ = right_robot.get_pose()

        midpoint = (np.array(left_position) + np.array(right_position)) * 0.5
        box_position = np.array(
            [
                midpoint[0],
                midpoint[1],
                max(self.cfg.carry_height, self._box_half_height + 0.05),
            ],
            dtype=float,
        )
        self._box.set_linear_velocity(np.zeros(3))
        self._box.set_pose(box_position, np.array([1.0, 0.0, 0.0, 0.0]))

    def _update_task_state(self):
        self._resolve_box()

        if self.success:
            return

        if self.phase == 'approach_pickup' and self._robots_reached(self.get_pickup_targets()):
            self.attached = True
            self._set_phase('co_carry')

        if self.attached:
            self._lift_box_with_team()
            if self._robots_reached(self.get_drop_targets()):
                placed_position = np.array(
                    [self.cfg.goal_position[0], self.cfg.goal_position[1], self._box_half_height],
                    dtype=float,
                )
                self._box.set_linear_velocity(np.zeros(3))
                self._box.set_pose(placed_position, np.array([1.0, 0.0, 0.0, 0.0]))
                self.attached = False
                self.success = True
                self._set_phase('complete')

    def get_observations(self):
        self._update_task_state()
        obs: OrderedDict = super().get_observations()
        box_position, _ = self._get_box_pose()
        pickup_targets = self.get_pickup_targets()
        drop_targets = self.get_drop_targets()

        for robot_index, robot_name in enumerate(self.cfg.robot_names):
            if robot_name not in obs:
                continue
            target = pickup_targets[robot_index] if self.phase == 'approach_pickup' else drop_targets[robot_index]
            obs[robot_name]['task_phase'] = self.phase
            obs[robot_name]['box_position'] = np.array(box_position)
            obs[robot_name]['goal_position'] = np.array(self.cfg.goal_position)
            obs[robot_name]['task_target'] = np.array(target)
            obs[robot_name]['box_attached'] = self.attached
        return obs

    def is_done(self) -> bool:
        self.step_counter += 1
        return self.success or self.step_counter >= self.max_steps

    def calculate_metrics(self) -> dict:
        box_position, _ = self._get_box_pose()
        return {
            'seed': self.cfg.seed,
            'success': self.success,
            'phase_history': self.phase_history,
            'steps': self.step_counter,
            'box_position': np.asarray(box_position).tolist(),
            'goal_position': list(self.cfg.goal_position),
            'attached': self.attached,
        }
