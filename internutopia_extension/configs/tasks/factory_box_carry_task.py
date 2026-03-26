from typing import Optional, Tuple

from internutopia.core.config.task import TaskCfg


class FactoryBoxCarryTaskCfg(TaskCfg):
    type: Optional[str] = 'FactoryBoxCarryTask'
    max_steps: int = 1200
    prompt: Optional[str] = ''
    seed: int = 0
    box_name: str = 'carry_box'
    robot_names: Tuple[str, str] = ('carrier_left', 'carrier_right')
    goal_position: Tuple[float, float, float] = (4.5, 0.0, 0.15)
    standoff_distance: float = 0.75
    attach_distance: float = 0.45
    formation_half_width: float = 0.35
    carry_forward_offset: float = 0.62
    squat_carry_height: float = 0.72
    carry_height: float = 1.0
    goal_tolerance: float = 0.35
    box_goal_tolerance: float = 0.42
    grasp_settle_steps: int = 18
    squat_steps: int = 26
    lift_steps: int = 30
    place_steps: int = 22
