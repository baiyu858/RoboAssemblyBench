"""Generated IsaacLab scene config for the Fabrica dual-UR5E workcell.

This is a scene-level bridge, not a complete RL environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCENE_ROOT = PACKAGE_ROOT / "scene"
FABRICA_PART_ROOT = PACKAGE_ROOT / "assets" / "fabrica_original_usd_sdf_margin_001" / "aligned" / "cooling_manifold" / "parts"
ISAAC_ASSET_ROOT = Path(os.environ.get("ISAAC_ASSET_ROOT", str(PACKAGE_ROOT / "assets" / "isaac_official" / "Isaac")))
UR5E_USD = ISAAC_ASSET_ROOT / "Robots" / "UniversalRobots" / "ur5e" / "ur5e.usd"
ENV_REGEX_NS = "{ENV_REGEX_NS}"


def make_ur5e_robotiq_cfg(name: str, pos: tuple[float, float, float], rot: list[float]) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path=f"{ENV_REGEX_NS}/{name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(UR5E_USD),
            variants={'Physics': 'PhysX', 'Sensor': 'Sensors', 'Gripper': 'Robotiq_2f_85'},
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
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
                "shoulder_pan_joint": -1.5707963267948966,
                "shoulder_lift_joint": -1.5707963267948966,
                "elbow_joint": 1.5707963267948966,
                "wrist_1_joint": -1.5707963267948966,
                "wrist_2_joint": -1.5707963267948966,
                "wrist_3_joint": 0.0,
                "finger_joint": 0.0,
                ".*_inner_finger_joint": 0.0,
                ".*_inner_finger_knuckle_joint": 0.0,
                ".*_outer_.*_joint": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_.*", "elbow_joint", "wrist_.*"],
                effort_limit_sim=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["finger_joint", ".*_inner_finger.*", ".*_outer_.*"],
                effort_limit_sim=40.0,
                stiffness=7500.0,
                damping=173.0,
            ),
        },
    )


@configclass
class FabricaDualUR5EWorkcellSceneCfg(InteractiveSceneCfg):
    """Scene-only dual-arm Fabrica workcell."""

    main_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/MainTable",
        spawn=sim_utils.UsdFileCfg(usd_path=str(SCENE_ROOT / "clean_packing_table.usda")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[1, 0, 0, 0]),
    )

    robot_left = make_ur5e_robotiq_cfg("RobotLeft", [0.05, 0.25, 0.998051], [0.707106781, 0.0, 0.0, -0.707106781])
    robot_right = make_ur5e_robotiq_cfg("RobotRight", [0.95, 0.25, 0.998051], [0.707106781, 0.0, 0.0, -0.707106781])

    assembled_display_part_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/assembled_display_part_0",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.248778, -0.169903, 1.035301], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_cooling_manifold_0.usd"),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.279951, -0.169903, 1.014551], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_cooling_manifold_1.usd"),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.229951, -0.219903, 1.014551], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_cooling_manifold_2.usd"),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.329951, -0.119903, 1.014551], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_cooling_manifold_3.usd"),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.311124, -0.169903, 1.035301], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_cooling_manifold_4.usd"),
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

    assembled_display_part_5 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/assembled_display_part_5",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.329951, -0.219903, 1.014551], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_cooling_manifold_5.usd"),
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

    assembled_display_part_6 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/assembled_display_part_6",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.229951, -0.119903, 1.014551], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_cooling_manifold_6.usd"),
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

    fabrica_cooling_manifold_0 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/fabrica_cooling_manifold_0",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.66, -0.32, 1.020301], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_cooling_manifold_0.usd"),
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

    fabrica_cooling_manifold_1 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/fabrica_cooling_manifold_1",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.81, -0.32, 1.014551], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_cooling_manifold_1.usd"),
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

    fabrica_cooling_manifold_2 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/fabrica_cooling_manifold_2",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.96, -0.32, 1.014551], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_cooling_manifold_2.usd"),
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

    fabrica_cooling_manifold_3 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/fabrica_cooling_manifold_3",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.66, -0.17, 1.014551], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_cooling_manifold_3.usd"),
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

    fabrica_cooling_manifold_4 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/fabrica_cooling_manifold_4",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.81, -0.17, 1.020301], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_cooling_manifold_4.usd"),
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

    fabrica_cooling_manifold_5 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/fabrica_cooling_manifold_5",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.96, -0.17, 1.014551], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_cooling_manifold_5.usd"),
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

    fabrica_cooling_manifold_6 = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/fabrica_cooling_manifold_6",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.66, -0.02, 1.014551], rot=[1, 0, 0, 0]),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(FABRICA_PART_ROOT / "fabrica_cooling_manifold_6.usd"),
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
