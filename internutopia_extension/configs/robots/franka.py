from typing import Optional

from internutopia.core.config import RobotCfg
from internutopia.macros import gm
from internutopia_extension.configs.controllers import (
    GripperControllerCfg,
    InverseKinematicsControllerCfg,
    JointControllerCfg,
)

arm_ik_cfg = InverseKinematicsControllerCfg(
    name='arm_ik_controller',
    robot_description_path=gm.ASSET_PATH + '/robots/franka/rmpflow/robot_descriptor.yaml',
    robot_urdf_path=gm.ASSET_PATH + '/robots/franka/lula_franka_gen.urdf',
    end_effector_frame_name='panda_hand',
    threshold=0.01,
)

arm_joint_cfg = JointControllerCfg(
    name='arm_joint_controller',
    joint_names=[
        'panda_joint1',
        'panda_joint2',
        'panda_joint3',
        'panda_joint4',
        'panda_joint5',
        'panda_joint6',
        'panda_joint7',
    ],
)

gripper_cfg = GripperControllerCfg(
    name='gripper_controller',
)


class FrankaRobotCfg(RobotCfg):
    # meta info
    name: Optional[str] = 'franka'
    type: Optional[str] = 'FrankaRobot'
    prim_path: Optional[str] = '/franka'
    usd_path: Optional[str] = gm.ASSET_PATH + '/robots/franka/franka.usd'
    end_effector_prim_name: Optional[str] = 'panda_hand'
