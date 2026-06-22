"""Generated IsaacLab scene config for the Fabrica dual-Franka plumbers-block workcell.

This is a scene-level bridge, not a complete RL environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = PACKAGE_ROOT / "scene"
FABRICA_PART_ROOT = PACKAGE_ROOT / "assets" / "fabrica_original_usd_sdf_margin_001" / "aligned" / "plumbers_block" / "parts"
FIXTURE_ASSET = PACKAGE_ROOT / "assets" / "fabrica_fixture" / "plumbers_block" / "fixture_pickup_tray.usda"
ISAAC_ASSET_ROOT = Path(os.environ.get("ISAAC_ASSET_ROOT", str(PACKAGE_ROOT / "assets" / "isaac_official" / "Isaac")))
FRANKA_USD = ISAAC_ASSET_ROOT / "Robots" / "FrankaRobotics" / "FrankaPanda" / "franka.usd"
ENV_REGEX_NS = "{ENV_REGEX_NS}"


def make_franka_panda_cfg(name: str, pos: tuple[float, float, float], rot: list[float]) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=f"{ENV_REGEX_NS}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FRANKA_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos,
            rot=rot,
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.7853981633974483,
                "panda_joint3": 0.0,
                "panda_joint4": -2.356194490192345,
                "panda_joint5": 0.0,
                "panda_joint6": 1.5707963267948966,
                "panda_joint7": 0.7853981633974483,
                "panda_finger_joint1": 0.04,
                "panda_finger_joint2": 0.04,
            },
        ),
        actuators={
            "panda_arm_1": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[1-4]"],
                effort_limit_sim=87.0,
                velocity_limit_sim=2.175,
                stiffness=400.0,
                damping=40.0,
                armature=0.01,
                friction=0.0,
            ),
            "panda_arm_2": ImplicitActuatorCfg(
                joint_names_expr=["panda_joint[5-7]"],
                effort_limit_sim=12.0,
                velocity_limit_sim=2.175,
                stiffness=400.0,
                damping=40.0,
                armature=0.01,
                friction=0.0,
            ),
            "panda_hand": ImplicitActuatorCfg(
                joint_names_expr=["panda_finger_joint.*"],
                effort_limit_sim=200.0,
                velocity_limit_sim=0.2,
                stiffness=7500.0,
                damping=173.0,
            ),
        },
    )


@configclass
class FabricaDualFrankaWorkcellSceneCfg(InteractiveSceneCfg):
    """Scene-only dual-arm Fabrica workcell."""

    main_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/MainTable",
        spawn=sim_utils.UsdFileCfg(usd_path=str(SCENE_ROOT / "clean_packing_table.usda")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[1, 0, 0, 0]),
    )

    robot_left = make_franka_panda_cfg("RobotLeft", [0.05, 0.25, 0.998051], [0.707106781, 0.0, 0.0, -0.707106781])
    robot_right = make_franka_panda_cfg("RobotRight", [0.95, 0.25, 0.998051], [0.707106781, 0.0, 0.0, -0.707106781])

    # Official IsaacLab-style cameras. Wrist cameras are mounted under
    # each Franka panda_hand, matching IsaacLab's Franka visuomotor stack task.
    table_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_cam",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(1.35, -1.05, 1.55),
            rot=(0.792856387, 0.513037814, 0.178675043, 0.27612711),
            convention="opengl",
        ),
    )

    table_high_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/table_high_cam",
        update_period=0.0,
        height=704,
        width=1280,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(2.15, -1.55, 2.35),
            rot=(0.781470042, 0.477571127, 0.2093824, 0.342621369),
            convention="opengl",
        ),
    )

    left_wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RobotLeft/panda_hand/wrist_cam",
        update_period=0.0,
        height=128,
        width=128,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.121622418, 0.022627431, -0.095948721),
            rot=(-0.24124222, 0.774542423, 0.558258727, -0.173877601),
            convention="opengl",
        ),
    )

    right_wrist_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RobotRight/panda_hand/wrist_cam",
        update_period=0.0,
        height=128,
        width=128,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.022627446, 0.121622358, -0.095948723),
            rot=(-0.019874242, 0.112964149, 0.978373958, -0.172129353),
            convention="opengl",
        ),
    )


    fixture_tray = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/FabricaPickupFixtureTray",
        spawn=sim_utils.UsdFileCfg(usd_path=str(FIXTURE_ASSET)),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.699975526, -0.429500145, 1.003051339],
            rot=[1, 0, 0, 0],
        ),
    )

    assembled_display_part_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/assembled_display_part_0",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.32, -0.18, 1.05746], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_plumbers_block_0.usd"),
            scale=[1.0, 1.0, 1.0],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
    )

    assembled_display_part_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/assembled_display_part_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.28, -0.178734, 1.075551], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_plumbers_block_1.usd"),
            scale=[1.0, 1.0, 1.0],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
    )

    assembled_display_part_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/assembled_display_part_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.32, -0.178734, 1.038051], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_plumbers_block_2.usd"),
            scale=[1.0, 1.0, 1.0],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
    )

    assembled_display_part_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/assembled_display_part_3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.32, -0.178734, 1.083625], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_plumbers_block_3.usd"),
            scale=[1.0, 1.0, 1.0],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
    )

    assembled_display_part_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/assembled_display_part_4",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.36, -0.178734, 1.075551], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_plumbers_block_4.usd"),
            scale=[1.0, 1.0, 1.0],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
    )

    fabrica_plumbers_block_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/fabrica_plumbers_block_0",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.729975526, -0.300890634, 1.049781015], rot=[0.964164607, -0.265304749, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_plumbers_block_0.usd"),
            scale=[1.0, 1.0, 1.0],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
    )

    fabrica_plumbers_block_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/fabrica_plumbers_block_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.661099095, -0.191089954, 1.043147345], rot=[0.702445846, -0.087475325, 0.087286216, 0.700927256]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_plumbers_block_1.usd"),
            scale=[1.0, 1.0, 1.0],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
    )

    fabrica_plumbers_block_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/fabrica_plumbers_block_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.668614066, -0.293987146, 1.093188624], rot=[0.617614866, 0.344313632, 0.344313632, -0.617614866]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_plumbers_block_2.usd"),
            scale=[1.0, 1.0, 1.0],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
    )

    fabrica_plumbers_block_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/fabrica_plumbers_block_3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.728614066, -0.192175925, 1.047898473], rot=[0.672893101, 0.21728984, -0.21728984, 0.672893101]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_plumbers_block_3.usd"),
            scale=[1.0, 1.0, 1.0],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
    )

    fabrica_plumbers_block_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/fabrica_plumbers_block_4",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.661099095, -0.136444319, 1.04337142], rot=[0.706415653, -0.045376459, 0.045278362, 0.704888482]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_plumbers_block_4.usd"),
            scale=[1.0, 1.0, 1.0],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
    )
