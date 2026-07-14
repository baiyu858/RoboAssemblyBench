"""Pick up the red cube using the official PickPlaceController.

Reuses the shared scene (table + Franka + container + red cube + camera) from
`src.scene.build_scene`. The PickPlaceController drives the arm to the cube,
grasps it, lifts, and places it at a target position on the tabletop, then the
program ends.

Run:
    /mnt/SSD_7T/panxubei/isaac-sim5.1/python.sh demos/pick_cube.py
"""
import sys
import os

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import numpy as np

from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": False})

from src.scene import build_scene, CUBE_POS, TABLE_TOP_Z

try:
    from isaacsim.robot.manipulators.examples.franka.controllers import (
        PickPlaceController,
    )
except ImportError:
    from omni.isaac.franka.controllers import PickPlaceController


def main():
    print("[pick_cube] building scene ...")
    world, franka, cube, camera = build_scene()

    # Ensure the gripper starts open.
    franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
    world.reset()
    camera.initialize()

    # Pre-grasp / lift height in world Z. Franka base sits on the tabletop
    # (z = TABLE_TOP_Z = 0.40), so we go ~0.25 m above the table.
    ee_initial_height = TABLE_TOP_Z + 0.25  # 0.65

    controller = PickPlaceController(
        name="pick_place_controller",
        gripper=franka.gripper,
        robot_articulation=franka,
        end_effector_initial_height=ee_initial_height,
    )
    articulation_controller = franka.get_articulation_controller()

    # Place target: on the open tabletop, in front of the robot, away from the
    # container.
    placing_position = np.array(
        [CUBE_POS[0] - 0.20, CUBE_POS[1] - 0.20, TABLE_TOP_Z + 0.05 / 2.0],
        dtype=np.float32,
    )

    print(
        f"[pick_cube] picking_position={CUBE_POS}, "
        f"placing_position={placing_position}"
    )
    print(f"[pick_cube] ee_initial_height={ee_initial_height}")

    reset_needed = False
    while sim_app.is_running():
        world.step(render=True)
        if world.is_stopped() and not reset_needed:
            reset_needed = True
        if world.is_playing():
            if reset_needed:
                world.reset()
                controller.reset()
                reset_needed = False

            picking_position = cube.get_world_pose()[0].astype(np.float32)

            actions = controller.forward(
                picking_position=picking_position,
                placing_position=placing_position,
                current_joint_positions=franka.get_joint_positions(),
                end_effector_offset=np.array([0.0, 0.005, 0.0]),
            )
            articulation_controller.apply_action(actions)

            if controller.is_done():
                print("[pick_cube] done picking and placing.")
                break

    # Hold a moment so the result is visible.
    for _ in range(120):
        world.step(render=True)

    print("[pick_cube] closing.")
    sim_app.close()


if __name__ == "__main__":
    main()