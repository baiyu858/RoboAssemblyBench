from internutopia_extension.configs.robots.humanoidbench_h1 import (
    humanoidbench_arm_pose_cfg,
    HumanoidBenchH1RobotCfg,
    humanoidbench_reach_single_cfg,
    humanoidbench_recover_cfg,
    humanoidbench_rotate_cfg,
    humanoidbench_walk_to_cfg,
    humanoidbench_wholebody_pose_cfg,
)


def __getattr__(name):
    """Load UR5e configuration only when a caller explicitly requests it."""

    if name in {
        'UR5eRobotCfg',
        'ur5e_arm_ik_cfg',
        'ur5e_arm_joint_cfg',
        'ur5e_gripper_cfg',
    }:
        from internutopia_extension.configs.robots.ur5e import (
            UR5eRobotCfg,
            arm_ik_cfg,
            arm_joint_cfg,
            gripper_cfg,
        )

        values = {
            'UR5eRobotCfg': UR5eRobotCfg,
            'ur5e_arm_ik_cfg': arm_ik_cfg,
            'ur5e_arm_joint_cfg': arm_joint_cfg,
            'ur5e_gripper_cfg': gripper_cfg,
        }
        globals().update(values)
        return values[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
