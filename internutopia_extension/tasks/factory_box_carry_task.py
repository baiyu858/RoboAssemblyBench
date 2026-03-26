from __future__ import annotations

from collections import OrderedDict

import numpy as np

from internutopia.core.scene.scene import IScene
from internutopia.core.task import BaseTask
from internutopia.core.util.physics import activate_collider, deactivate_collider
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
        self.phase_step_counter = 0
        self.success = False
        self.attached = False
        self._box = None
        self._box_prim = None
        self._box_collision_enabled = True
        self._box_half_height = 0.15

    @property
    def cfg(self) -> FactoryBoxCarryTaskCfg:
        return self.config

    def _set_phase(self, phase: str):
        if phase != self.phase:
            self.phase = phase
            self.phase_history.append(phase)
            self.phase_step_counter = 0

    def _resolve_box(self):
        if self._box is not None:
            return
        from isaacsim.core.utils.prims import get_prim_at_path

        box_obj = self.objects[self.cfg.box_name]
        self._box = self._scene.get(box_obj.name)
        self._box_prim = get_prim_at_path(self._box.unwrap().prim_path)
        if box_obj.config.scale is not None and len(box_obj.config.scale) >= 3:
            self._box_half_height = float(box_obj.config.scale[2]) * 0.5

    def _set_box_collision(self, enabled: bool):
        self._resolve_box()
        if self._box_prim is None or self._box_collision_enabled == enabled:
            return
        if enabled:
            activate_collider(self._box_prim)
        else:
            deactivate_collider(self._box_prim)
        self._box_collision_enabled = enabled

    def _get_box_pose(self):
        self._resolve_box()
        return self._box.get_pose()

    def _get_robot_pair(self):
        left_name, right_name = self.cfg.robot_names
        return self.robots[left_name], self.robots[right_name]

    def _compute_team_targets(self, center: np.ndarray, rear_offset: float):
        left_target = np.array(
            [center[0] - rear_offset, center[1] - self.cfg.formation_half_width, 0.0],
            dtype=float,
        )
        right_target = np.array(
            [center[0] - rear_offset, center[1] + self.cfg.formation_half_width, 0.0],
            dtype=float,
        )
        return left_target, right_target

    def get_pickup_targets(self):
        box_position, _ = self._get_box_pose()
        return self._compute_team_targets(np.array(box_position), self.cfg.standoff_distance)

    def get_grasp_targets(self):
        box_position, _ = self._get_box_pose()
        return self._compute_team_targets(np.array(box_position), self.cfg.attach_distance)

    def get_drop_targets(self):
        return self._compute_team_targets(np.array(self.cfg.goal_position), self.cfg.carry_forward_offset)

    def _phase_alpha(self, duration_steps: int) -> float:
        if duration_steps <= 0:
            return 1.0
        return min(float(self.phase_step_counter + 1) / float(duration_steps), 1.0)

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

    def _box_reached_goal(self) -> bool:
        box_position, _ = self._get_box_pose()
        return self._xy_distance(box_position, self.cfg.goal_position) < self.cfg.box_goal_tolerance

    def _get_team_box_pose(self, box_height: float):
        left_robot, right_robot = self._get_robot_pair()
        left_position, _ = left_robot.get_pose()
        right_position, _ = right_robot.get_pose()

        midpoint = (np.array(left_position) + np.array(right_position)) * 0.5
        return np.array(
            [
                midpoint[0] + self.cfg.carry_forward_offset,
                midpoint[1],
                max(box_height, self._box_half_height),
            ],
            dtype=float,
        )

    def _set_box_pose(self, box_height: float):
        box_position = self._get_team_box_pose(box_height)
        self._box.set_linear_velocity(np.zeros(3))
        self._box.set_pose(box_position, np.array([1.0, 0.0, 0.0, 0.0]))

    def _update_task_state(self):
        self._resolve_box()

        if self.success:
            return

        if self.phase == 'approach_pickup':
            if self._robots_reached(self.get_pickup_targets()):
                self._set_phase('grasp_setup')
        elif self.phase == 'grasp_setup':
            if self._robots_reached(self.get_grasp_targets()) and self.phase_step_counter >= self.cfg.grasp_settle_steps:
                self._set_phase('squat_grasp')
        elif self.phase == 'squat_grasp':
            if self.phase_step_counter >= self.cfg.squat_steps:
                self.attached = True
                self._set_box_collision(False)
                self._set_phase('stand_lift')
        elif self.phase == 'stand_lift':
            self.attached = True
            lift_alpha = self._phase_alpha(self.cfg.lift_steps)
            current_height = self.cfg.squat_carry_height + lift_alpha * (self.cfg.carry_height - self.cfg.squat_carry_height)
            self._set_box_pose(current_height)
            if lift_alpha >= 1.0:
                self._set_phase('co_carry')
        elif self.phase == 'co_carry':
            self.attached = True
            self._set_box_pose(self.cfg.carry_height)
            if self._robots_reached(self.get_drop_targets()) or self._box_reached_goal():
                self._set_phase('lower_place')
        elif self.phase == 'lower_place':
            self.attached = True
            place_alpha = self._phase_alpha(self.cfg.place_steps)
            current_height = self.cfg.carry_height + place_alpha * (self._box_half_height - self.cfg.carry_height)
            self._set_box_pose(current_height)
            if place_alpha >= 1.0:
                placed_position = np.array(
                    [self.cfg.goal_position[0], self.cfg.goal_position[1], self._box_half_height],
                    dtype=float,
                )
                self._box.set_linear_velocity(np.zeros(3))
                self._box.set_pose(placed_position, np.array([1.0, 0.0, 0.0, 0.0]))
                self.attached = False
                self._set_box_collision(True)
                self.success = True
                self._set_phase('complete')

        self.phase_step_counter += 1

    def get_observations(self):
        self._update_task_state()
        obs: OrderedDict = super().get_observations()
        box_position, _ = self._get_box_pose()
        pickup_targets = self.get_pickup_targets()
        grasp_targets = self.get_grasp_targets()
        drop_targets = self.get_drop_targets()

        for robot_index, robot_name in enumerate(self.cfg.robot_names):
            if robot_name not in obs:
                continue
            if self.phase == 'approach_pickup':
                target = pickup_targets[robot_index]
            elif self.phase in {'grasp_setup', 'squat_grasp', 'stand_lift'}:
                target = grasp_targets[robot_index]
            else:
                target = drop_targets[robot_index]
            obs[robot_name]['task_phase'] = self.phase
            obs[robot_name]['phase_step'] = self.phase_step_counter
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
