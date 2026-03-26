from typing import Optional

from internutopia_extension.configs.robots.h1_with_hand import (
    H1WithHandRobotCfg,
    move_to_point_cfg,
    recover_cfg,
    right_arm_ik_controller_cfg,
    rotate_cfg,
)

# Reuse Isaac Sim compatible H1-with-hand embodiment, but expose HumanoidBench-style
# atomic skills so higher-level policies can share the same interface.
humanoidbench_walk_to_cfg = move_to_point_cfg.update(name='humanoidbench_walk_to')
humanoidbench_rotate_cfg = rotate_cfg.update(name='humanoidbench_rotate')
humanoidbench_reach_single_cfg = right_arm_ik_controller_cfg.update(name='humanoidbench_reach_single')
humanoidbench_recover_cfg = recover_cfg.update(name='humanoidbench_recover')


class HumanoidBenchH1RobotCfg(H1WithHandRobotCfg):
    name: Optional[str] = 'humanoidbench_h1'
    type: Optional[str] = 'HumanoidBenchH1Robot'
    prim_path: Optional[str] = '/humanoidbench_h1'
