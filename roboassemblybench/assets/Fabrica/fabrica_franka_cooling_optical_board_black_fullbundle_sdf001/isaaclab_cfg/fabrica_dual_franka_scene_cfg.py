"""Generated IsaacLab scene config for the Fabrica dual-Franka workcell.

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
