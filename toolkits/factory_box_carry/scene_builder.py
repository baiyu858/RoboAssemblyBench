from __future__ import annotations

import os
import random
from typing import List

from internutopia.macros import gm
from internutopia_extension.configs.objects import DynamicCubeCfg, VisualCubeCfg
from internutopia_extension.configs.robots.humanoidbench_h1 import (
    HumanoidBenchH1RobotCfg,
    humanoidbench_arm_pose_cfg,
    humanoidbench_reach_single_cfg,
    humanoidbench_recover_cfg,
    humanoidbench_rotate_cfg,
    humanoidbench_walk_to_cfg,
    humanoidbench_wholebody_pose_cfg,
)
from internutopia_extension.configs.tasks.factory_box_carry_task import (
    FactoryBoxCarryTaskCfg,
)


def _humanoidbench_reach_available() -> bool:
    return os.path.exists(humanoidbench_reach_single_cfg.robot_description_path) and os.path.exists(
        humanoidbench_reach_single_cfg.robot_urdf_path
    )


def build_humanoidbench_carrier(
    name: str,
    prim_path: str,
    position,
    enable_reach: bool = False,
) -> HumanoidBenchH1RobotCfg:
    controllers: List = [
        humanoidbench_walk_to_cfg.update(),
        humanoidbench_rotate_cfg.update(),
        humanoidbench_recover_cfg.update(),
        humanoidbench_arm_pose_cfg.update(),
        humanoidbench_wholebody_pose_cfg.update(),
    ]
    if enable_reach and _humanoidbench_reach_available():
        controllers.append(humanoidbench_reach_single_cfg.update())

    return HumanoidBenchH1RobotCfg(
        name=name,
        prim_path=prim_path,
        position=position,
        controllers=controllers,
        sensors=[],
    )


def build_factory_box_carry_episode(seed: int, episode_idx: int = 0) -> FactoryBoxCarryTaskCfg:
    rng = random.Random(seed)

    formation_half_width = 0.34
    pickup_distance = 0.86
    grasp_distance = 0.60
    carry_forward_offset = 0.62

    box_scale = (0.55, 0.35, 0.30)
    box_position = (
        rng.uniform(2.2, 2.8),
        rng.uniform(-0.15, 0.15),
        box_scale[2] * 0.5,
    )
    goal_position = (
        rng.uniform(4.5, 5.2),
        rng.uniform(-0.4, 0.4),
        box_scale[2] * 0.5,
    )

    left_start = (box_position[0] - 1.45, box_position[1] - formation_half_width, 1.05)
    right_start = (box_position[0] - 1.45, box_position[1] + formation_half_width, 1.05)

    robots = [
        build_humanoidbench_carrier(
            name='carrier_left',
            prim_path='/carrier_left',
            position=left_start,
        ),
        build_humanoidbench_carrier(
            name='carrier_right',
            prim_path='/carrier_right',
            position=right_start,
        ),
    ]

    objects = [
        DynamicCubeCfg(
            name='carry_box',
            prim_path='/carry_box',
            position=box_position,
            scale=box_scale,
            color=(0.73, 0.47, 0.28),
        ),
        VisualCubeCfg(
            name='pickup_zone',
            prim_path='/pickup_zone',
            position=(box_position[0], box_position[1], 0.01),
            scale=(0.9, 0.9, 0.02),
            color=(0.20, 0.55, 0.20),
        ),
        VisualCubeCfg(
            name='drop_zone',
            prim_path='/drop_zone',
            position=(goal_position[0], goal_position[1], 0.01),
            scale=(0.9, 0.9, 0.02),
            color=(0.20, 0.20, 0.65),
        ),
        VisualCubeCfg(
            name='assembly_lane',
            prim_path='/assembly_lane',
            position=((box_position[0] + goal_position[0]) * 0.5, 0.0, 0.005),
            scale=(4.2, 2.6, 0.01),
            color=(0.32, 0.32, 0.32),
        ),
    ]

    return FactoryBoxCarryTaskCfg(
        prompt='Two humanoid robots cooperatively carry the box from the pickup zone to the drop zone.',
        seed=seed,
        max_steps=1200,
        scene_asset_path=gm.ASSET_PATH + '/scenes/empty.usd',
        robots=robots,
        objects=objects,
        box_name='carry_box',
        robot_names=('carrier_left', 'carrier_right'),
        goal_position=goal_position,
        standoff_distance=pickup_distance,
        attach_distance=grasp_distance,
        formation_half_width=formation_half_width,
        carry_forward_offset=carry_forward_offset,
        squat_carry_height=0.74,
        carry_height=1.02,
        goal_tolerance=0.24,
        box_goal_tolerance=0.42,
        grasp_settle_steps=20,
        squat_steps=28,
        lift_steps=30,
        place_steps=16,
        episode_idx=episode_idx,
    )
