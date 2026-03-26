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
    carry_height: float = 1.0
    goal_tolerance: float = 0.35
