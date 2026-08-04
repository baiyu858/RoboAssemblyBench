import os
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from internutopia.core.config import RobotCfg
from internutopia_extension.configs.controllers import (
    GripperControllerCfg,
    InverseKinematicsControllerCfg,
    JointControllerCfg,
)

_MOTION_CFG_RELATIVE = Path('exts/isaacsim.robot_motion.motion_generation/motion_policy_configs/universal_robots/ur5e')


def _resolve_ur5e_motion_cfg_root() -> Path:
    candidates = []
    configured_root = os.environ.get('ISAAC_SIM_ROOT')
    if configured_root:
        candidates.append(Path(configured_root).expanduser() / _MOTION_CFG_RELATIVE)
    candidates.append(Path('/home/baiyu24/APP/isaac-smi') / _MOTION_CFG_RELATIVE)
    for entry in sys.path:
        if not entry:
            continue
        package_root = Path(entry).expanduser()
        candidates.append(package_root / 'isaacsim' / _MOTION_CFG_RELATIVE)
        candidates.append(package_root / _MOTION_CFG_RELATIVE)

    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if (candidate / 'rmpflow/ur5e_robot_description.yaml').is_file() and (candidate / 'ur5e.urdf').is_file():
            return candidate
    raise FileNotFoundError(
        'Cannot locate Isaac Sim UR5e motion-policy configuration. Set ISAAC_SIM_ROOT or install '
        f'isaacsim.robot_motion.motion_generation. Checked: {[str(path) for path in checked]}'
    )


_UR5E_MOTION_CFG_ROOT = _resolve_ur5e_motion_cfg_root()


arm_ik_cfg = InverseKinematicsControllerCfg(
    name='arm_ik_controller',
    robot_description_path=str(_UR5E_MOTION_CFG_ROOT / 'rmpflow/ur5e_robot_description.yaml'),
    robot_urdf_path=str(_UR5E_MOTION_CFG_ROOT / 'ur5e.urdf'),
    end_effector_frame_name='tool0',
    threshold=0.01,
)

arm_joint_cfg = JointControllerCfg(
    name='arm_joint_controller',
    joint_names=[
        'shoulder_pan_joint',
        'shoulder_lift_joint',
        'elbow_joint',
        'wrist_1_joint',
        'wrist_2_joint',
        'wrist_3_joint',
    ],
)

gripper_cfg = GripperControllerCfg(
    name='gripper_controller',
)


class UR5eRobotCfg(RobotCfg):
    name: Optional[str] = 'ur5e'
    type: Optional[str] = 'UR5eRobot'
    prim_path: Optional[str] = '/ur5e'
    usd_path: Optional[str] = None
    end_effector_prim_name: Optional[str] = 'tool0'
    ik_base_prim_name: Optional[str] = 'base_link'
    gripper_dof_name: Optional[str] = 'finger_joint'
    gripper_open_position: float = 0.0
    gripper_closed_position: float = 0.80
    gripper_close_openness: float = 0.08
    hand_link_name: Optional[str] = 'wrist_3_link'
    left_finger_link_name: Optional[str] = 'left_inner_finger'
    right_finger_link_name: Optional[str] = 'right_inner_finger'
    initial_joint_positions: Optional[Dict[str, float]] = None
    gripper_xform_orient: Optional[list[float]] = None
    gripper_mount_local_pos0: Optional[list[float]] = None
    gripper_mount_local_pos1: Optional[list[float]] = None
    gripper_mount_local_rot0: Optional[list[float]] = None
    gripper_mount_local_rot1: Optional[list[float]] = None
    configure_gripper_mount_joint: bool = True
    gripper_base_link_path: Optional[str] = None
    gripper_container_path: Optional[str] = None
    gripper_container_orient: Optional[list[float]] = None
    author_gripper_collision_pads: bool = True


DEFAULT_UR5E_READY_JOINTS = {
    'shoulder_pan_joint': -np.pi / 2.0,
    'shoulder_lift_joint': -np.pi / 2.0,
    'elbow_joint': np.pi / 2.0,
    'wrist_1_joint': -np.pi / 2.0,
    'wrist_2_joint': -np.pi / 2.0,
    'wrist_3_joint': 0.0,
    'finger_joint': 0.0,
}
