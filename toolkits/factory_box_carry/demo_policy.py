from __future__ import annotations

import numpy as np

from internutopia_extension.configs.robots.humanoidbench_h1 import humanoidbench_walk_to_cfg


class HumanoidBenchCarryDemoPolicy:
    """Closed-loop scripted demo policy using HumanoidBench-style atomic skills."""

    def _compose_action(self, walk_target):
        return {humanoidbench_walk_to_cfg.name: [np.asarray(walk_target, dtype=float).tolist()]}

    def act(self, task):
        left_name, right_name = task.config.robot_names
        if task.phase == 'approach_pickup':
            walk_targets = task.get_pickup_targets()
            return {
                left_name: self._compose_action(walk_targets[0]),
                right_name: self._compose_action(walk_targets[1]),
            }

        if task.phase == 'co_carry':
            walk_targets = task.get_drop_targets()
            return {
                left_name: self._compose_action(walk_targets[0]),
                right_name: self._compose_action(walk_targets[1]),
            }

        return {
            left_name: {},
            right_name: {},
        }
