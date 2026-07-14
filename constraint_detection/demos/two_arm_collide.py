"""Two Franka arms moving toward the same target point on the table.

Collision detection uses the shared capsule-based CollisionDetector
(src/collision.py): each arm link is a capsule (segment + radius) and the
reported distance is the true surface-to-surface clearance, not a coarse
link-center Euclidean distance.

Run:  conda deactivate
      /mnt/SSD_7T/panxubei/isaac-sim5.1/python.sh demos/two_arm_collide.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": False})

import numpy as np
from scipy.spatial.transform import Rotation as R

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid

try:
    from isaacsim.robot.manipulators.examples.franka import Franka
except ImportError:
    from omni.isaac.franka import Franka

try:
    from isaacsim.robot.manipulators.examples.franka.controllers import RMPFlowController
except ImportError:
    from omni.isaac.franka.controllers import RMPFlowController

try:
    from isaacsim.core.prims import SingleXFormPrim
except ImportError:
    from omni.isaac.core.prims import SingleXFormPrim

try:
    from isaacsim.sensors.camera import Camera
except ImportError:
    from omni.isaac.sensor import Camera

from src.collision import CollisionDetector, setup_link_xforms

# ---- Scene ----
TABLE_CENTER = np.array([0.30, 0.00, 0.20], dtype=np.float32)
TABLE_SIZE   = np.array([2.00, 1.40, 0.40], dtype=np.float32)
TABLE_TOP_Z  = TABLE_CENTER[2] + TABLE_SIZE[2] / 2  # = 0.40

ARM_A_POS = np.array([0.00,  0.30, TABLE_TOP_Z], dtype=np.float32)
ARM_B_POS = np.array([0.00, -0.30, TABLE_TOP_Z], dtype=np.float32)

CUBE_SIZE = 0.05
CUBE_POS = np.array([0.50, 0.00, TABLE_TOP_Z + CUBE_SIZE / 2], dtype=np.float32)

EE_TARGET_A = np.array([0.50,  0.04, TABLE_TOP_Z + 0.12], dtype=np.float32)
EE_TARGET_B = np.array([0.50, -0.04, TABLE_TOP_Z + 0.12], dtype=np.float32)

CAM_POS    = np.array([1.20, 0.00, TABLE_TOP_Z + 0.65], dtype=np.float32)
CAM_TARGET = np.array([0.40, 0.00, TABLE_TOP_Z + 0.05], dtype=np.float32)

NUM_STEPS = 500

# Collision detection: surface-to-surface clearance (m) below which a
# capsule pair is reported as a collision/proximity violation.
COLLISION_THRESHOLD = 0.05  # meters


def camera_look_at_orn(eye, target):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    direction = target - eye
    direction /= np.linalg.norm(direction) + 1e-8
    yaw = np.arctan2(-direction[1], -direction[0])
    pitch = np.arcsin(-direction[2])
    r = R.from_euler('YX', [yaw, pitch])
    q = r.as_quat()
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def build_scene(world):
    world.scene.add_default_ground_plane()
    world.scene.add(FixedCuboid(
        prim_path="/World/Table", name="table",
        position=TABLE_CENTER, scale=TABLE_SIZE,
        color=np.array([0.78, 0.62, 0.45]),
    ))
    arm_a = world.scene.add(Franka(
        prim_path="/World/FrankaA", name="franka_a", position=ARM_A_POS,
    ))
    arm_b = world.scene.add(Franka(
        prim_path="/World/FrankaB", name="franka_b", position=ARM_B_POS,
    ))
    camera = Camera(
        prim_path="/World/Camera",
        position=CAM_POS, frequency=20, resolution=(256, 256),
        orientation=camera_look_at_orn(CAM_POS, CAM_TARGET),
    )
    return arm_a, arm_b, camera


def main():
    print("[two_arm] building scene …")
    world = World(stage_units_in_meters=1.0)
    arm_a, arm_b, camera = build_scene(world)
    world.reset()

    world.scene.add(DynamicCuboid(
        prim_path="/World/Cube", name="target_cube",
        position=CUBE_POS,
        scale=np.array([CUBE_SIZE] * 3, dtype=np.float32),
        color=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        mass=0.05,
    ))

    print("[two_arm] warmup …")
    for _ in range(80):
        world.step(render=True)

    # Setup link position readers + shared capsule-based detector
    print("[two_arm] setting up link position readers …")
    link_xf_a = setup_link_xforms("/World/FrankaA")
    link_xf_b = setup_link_xforms("/World/FrankaB")
    print(f"[two_arm]   FrankaA links: {list(link_xf_a.keys())}")
    print(f"[two_arm]   FrankaB links: {list(link_xf_b.keys())}")

    detector = CollisionDetector(threshold=COLLISION_THRESHOLD)
    detector.add_agent("franka_a", link_xf_a)
    detector.add_agent("franka_b", link_xf_b)

    rmp_a = RMPFlowController(name="rmpflow_a", robot_articulation=arm_a)
    rmp_b = RMPFlowController(name="rmpflow_b", robot_articulation=arm_b)
    rmp_a.reset()
    rmp_b.reset()

    xf_a = SingleXFormPrim("/World/FrankaA/panda_hand")
    xf_b = SingleXFormPrim("/World/FrankaB/panda_hand")
    _, orn_a = xf_a.get_world_pose()
    _, orn_b = xf_b.get_world_pose()
    orn_a = np.asarray(orn_a, dtype=np.float32)
    orn_b = np.asarray(orn_b, dtype=np.float32)

    print(f"[two_arm] cube at {CUBE_POS}")
    print(f"[two_arm] EE target A = {EE_TARGET_A}")
    print(f"[two_arm] EE target B = {EE_TARGET_B}")
    print(f"[two_arm] collision threshold = {COLLISION_THRESHOLD*100:.0f}cm")
    print("[two_arm] driving both arms …")

    collision_was_on = False

    for step in range(NUM_STEPS):
        action_a = rmp_a.forward(
            target_end_effector_position=EE_TARGET_A,
            target_end_effector_orientation=orn_a,
        )
        action_b = rmp_b.forward(
            target_end_effector_position=EE_TARGET_B,
            target_end_effector_orientation=orn_b,
        )
        arm_a.apply_action(action_a)
        arm_b.apply_action(action_b)
        world.step(render=True)

        # Collision detection: capsule-vs-capsule surface clearance
        events = detector.check_inter_agent(step)
        collision_now = len(events) > 0
        min_dist = min((e.distance for e in events), default=float("inf"))

        if collision_now and not collision_was_on:
            print(f"\n===== COLLISION START step {step} (min clearance = {min_dist*100:.1f}cm) =====")
            for e in events:
                print(f"  {e.entity_a}  <->  {e.entity_b}  ({e.distance*100:.1f}cm)")
            print()
        elif not collision_now and collision_was_on:
            print(f"\n===== COLLISION END   step {step} =====")
        collision_was_on = collision_now

        if step % 50 == 0:
            pos_a, _ = xf_a.get_world_pose()
            pos_b, _ = xf_b.get_world_pose()
            err_a = np.linalg.norm(EE_TARGET_A - np.asarray(pos_a)) * 100
            err_b = np.linalg.norm(EE_TARGET_B - np.asarray(pos_b)) * 100
            flag = "COLLIDING" if collision_now else "OK"
            clr = f"{min_dist*100:.1f}cm" if collision_now else "clear"
            print(f"  step {step:03d} | A err={err_a:.1f}cm | B err={err_b:.1f}cm | min_clearance={clr} [{flag}]")

    pos_a, _ = xf_a.get_world_pose()
    pos_b, _ = xf_b.get_world_pose()
    err_a = np.linalg.norm(EE_TARGET_A - np.asarray(pos_a)) * 100
    err_b = np.linalg.norm(EE_TARGET_B - np.asarray(pos_b)) * 100
    print(f"\n[two_arm] final: A err={err_a:.1f}cm  B err={err_b:.1f}cm")
    print("[two_arm] done.")
    sim_app.close()


if __name__ == "__main__":
    main()
