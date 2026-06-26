import os
from typing import Dict, Optional

import numpy as np

from internutopia.core.config import RobotCfg
from internutopia_extension.configs.controllers import (
    GripperControllerCfg,
    InverseKinematicsControllerCfg,
    JointControllerCfg,
)


_ISAAC_SIM_ROOT = os.environ.get("ISAAC_SIM_ROOT") or os.environ.get("ISAAC_PATH") or "/home/baiyu24/APP/isaac-smi"
_UR5E_MOTION_CFG_ROOT = os.path.join(
    _ISAAC_SIM_ROOT,
    "exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur5e",
)


arm_ik_cfg = InverseKinematicsControllerCfg(
    name="arm_ik_controller",
    robot_description_path=os.path.join(_UR5E_MOTION_CFG_ROOT, "rmpflow/ur5e_robot_description.yaml"),
    robot_urdf_path=os.path.join(_UR5E_MOTION_CFG_ROOT, "ur5e.urdf"),
    end_effector_frame_name="tool0",
    threshold=0.01,
)

arm_joint_cfg = JointControllerCfg(
    name="arm_joint_controller",
    joint_names=[
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ],
)

gripper_cfg = GripperControllerCfg(
    name="gripper_controller",
)


class UR5eRobotCfg(RobotCfg):
    name: Optional[str] = "ur5e"
    type: Optional[str] = "UR5eRobot"
    prim_path: Optional[str] = "/ur5e"
    usd_path: Optional[str] = None
    end_effector_prim_name: Optional[str] = "tool0"
    ik_base_prim_name: Optional[str] = "base_link"
    gripper_dof_name: Optional[str] = "finger_joint"
    gripper_open_position: float = 0.0
    gripper_closed_position: float = 0.80
    gripper_close_openness: float = 0.08
    hand_link_name: Optional[str] = "wrist_3_link"
    left_finger_link_name: Optional[str] = "left_inner_finger"
    right_finger_link_name: Optional[str] = "right_inner_finger"
    initial_joint_positions: Optional[Dict[str, float]] = None
    gripper_xform_orient: Optional[list[float]] = None
    gripper_mount_local_pos0: Optional[list[float]] = None
    gripper_mount_local_pos1: Optional[list[float]] = None
    gripper_mount_local_rot0: Optional[list[float]] = None
    gripper_mount_local_rot1: Optional[list[float]] = None


DEFAULT_UR5E_READY_JOINTS = {
    "shoulder_pan_joint": -np.pi / 2.0,
    "shoulder_lift_joint": -np.pi / 2.0,
    "elbow_joint": np.pi / 2.0,
    "wrist_1_joint": -np.pi / 2.0,
    "wrist_2_joint": -np.pi / 2.0,
    "wrist_3_joint": 0.0,
    "finger_joint": 0.0,
}
