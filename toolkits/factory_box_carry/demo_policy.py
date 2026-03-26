from __future__ import annotations

from collections import OrderedDict

import numpy as np

from internutopia_extension.configs.robots.humanoidbench_h1 import (
    humanoidbench_arm_pose_cfg,
    humanoidbench_walk_to_cfg,
    humanoidbench_wholebody_pose_cfg,
)


_PREGRASP_ARM_POSE = [
    0.72,
    0.72,
    0.18,
    -0.18,
    0.0,
    0.0,
    1.10,
    1.10,
]

_CARRY_ARM_POSE = [
    0.78,
    0.78,
    0.16,
    -0.16,
    0.0,
    0.0,
    1.16,
    1.16,
]

_SQUAT_WHOLEBODY_POSE = [
    0.08,
    -0.56,
    -0.56,
    1.10,
    1.10,
    -0.48,
    -0.48,
    0.88,
    0.88,
    0.18,
    -0.18,
    0.0,
    0.0,
    1.48,
    1.48,
]

_STAND_CARRY_WHOLEBODY_POSE = [
    0.02,
    -0.08,
    -0.08,
    0.18,
    0.18,
    -0.10,
    -0.10,
    0.78,
    0.78,
    0.16,
    -0.16,
    0.0,
    0.0,
    1.16,
    1.16,
]


class HumanoidBenchCarryDemoPolicy:
    """Closed-loop scripted demo policy using HumanoidBench-style atomic skills."""

    def _compose_action(self, walk_target):
        return {humanoidbench_walk_to_cfg.name: [np.asarray(walk_target, dtype=float).tolist()]}

    def _compose_arm_pose(self, pose):
        return {humanoidbench_arm_pose_cfg.name: [list(pose)]}

    def _compose_wholebody_pose(self, pose):
        return {humanoidbench_wholebody_pose_cfg.name: [list(pose)]}

    def _compose_walk_and_arm_pose(self, walk_target, pose):
        action = OrderedDict()
        action[humanoidbench_walk_to_cfg.name] = [np.asarray(walk_target, dtype=float).tolist()]
        action[humanoidbench_arm_pose_cfg.name] = [list(pose)]
        return action

    def act(self, task):
        left_name, right_name = task.config.robot_names
        if task.phase == 'approach_pickup':
            walk_targets = task.get_pickup_targets()
            return {
                left_name: self._compose_action(walk_targets[0]),
                right_name: self._compose_action(walk_targets[1]),
            }

        if task.phase == 'grasp_setup':
            walk_targets = task.get_grasp_targets()
            return {
                left_name: self._compose_walk_and_arm_pose(walk_targets[0], _PREGRASP_ARM_POSE),
                right_name: self._compose_walk_and_arm_pose(walk_targets[1], _PREGRASP_ARM_POSE),
            }

        if task.phase == 'squat_grasp':
            return {
                left_name: self._compose_wholebody_pose(_SQUAT_WHOLEBODY_POSE),
                right_name: self._compose_wholebody_pose(_SQUAT_WHOLEBODY_POSE),
            }

        if task.phase == 'stand_lift':
            return {
                left_name: self._compose_wholebody_pose(_STAND_CARRY_WHOLEBODY_POSE),
                right_name: self._compose_wholebody_pose(_STAND_CARRY_WHOLEBODY_POSE),
            }

        if task.phase == 'co_carry':
            walk_targets = task.get_drop_targets()
            return {
                left_name: self._compose_walk_and_arm_pose(walk_targets[0], _CARRY_ARM_POSE),
                right_name: self._compose_walk_and_arm_pose(walk_targets[1], _CARRY_ARM_POSE),
            }

        if task.phase == 'lower_place':
            return {
                left_name: self._compose_wholebody_pose(_SQUAT_WHOLEBODY_POSE),
                right_name: self._compose_wholebody_pose(_SQUAT_WHOLEBODY_POSE),
            }

        return {
            left_name: {},
            right_name: {},
        }
