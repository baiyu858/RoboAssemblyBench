"""Arm-vs-container collision detection demo.

Scene mirrors pick_cube.py (table + 4-wall concave container + red cube),
but here the arm is told to drive STRAIGHT toward the cube without going
above the container first. It will collide with the container walls on
the way down. Every physics step we run agent-environment collision
detection against the 5 container parts (base + 4 walls). Whenever any
arm link breaches the threshold against any container part, we print
the event — otherwise we stay silent.

Run:
    /mnt/SSD_7T/panxubei/isaac-sim5.1/python.sh demos/test_env_collision.py
"""
import numpy as np
from isaacsim import SimulationApp
from scipy.spatial.transform import Rotation as R

sim_app = SimulationApp({'headless': False})

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid

try:
    from isaacsim.robot.manipulators.examples.franka import Franka
except ImportError:
    from omni.isaac.franka import Franka

try:
    from isaacsim.robot.manipulators.examples.franka.controllers import (
        RMPFlowController,
    )
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

from toolkits.constraint_checking.detector.collision import (
    CollisionDetector,
    setup_link_xforms,
)

# ═══════════════════════════════════════════════════════════════════════════
# scene layout
# ═══════════════════════════════════════════════════════════════════════════

TABLE_CENTER = np.array([0.30, 0.00, 0.20], dtype=np.float32)
TABLE_SIZE = np.array([1.40, 1.00, 0.40], dtype=np.float32)
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] / 2  # = 0.40

FRANKA_POS = np.array([0.00, 0.00, TABLE_TOP_Z], dtype=np.float32)

# Container — real concave container (base + 4 walls)
CONTAINER_XY = np.array([0.50, 0.00], dtype=np.float32)
CONTAINER_BASE_THK = 0.005
CONTAINER_WALL_THK = 0.008
CONTAINER_WALL_H = 0.06
CONTAINER_INNER_XY = np.array([0.16, 0.16], dtype=np.float32)
CONTAINER_OUTER_XY = CONTAINER_INNER_XY + 2 * CONTAINER_WALL_THK
CONTAINER_TOP_Z = TABLE_TOP_Z + CONTAINER_BASE_THK + CONTAINER_WALL_H

CUBE_SIZE = 0.05
CUBE_POS = np.array(
    [CONTAINER_XY[0], CONTAINER_XY[1], TABLE_TOP_Z + CONTAINER_BASE_THK + CUBE_SIZE / 2 + 0.001],
    dtype=np.float32,
)

# Motion — drive STRAIGHT to the cube; no waypoint above the container.
MOVE_STEPS = 240
# capsule surface-to-box/environment clearance threshold (m)
# 2 cm: only report when capsule body is nearly in contact with environment
COLLISION_THRESHOLD = 0.02

CAM_POS = np.array([1.20, 0.00, TABLE_TOP_Z + 0.65], dtype=np.float32)
CAM_TARGET = CUBE_POS


# ═══════════════════════════════════════════════════════════════════════════
# scene build
# ═══════════════════════════════════════════════════════════════════════════


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


def _container_parts():
    """Return [(name, center, half_extents)] for all 5 container parts."""
    base_z = TABLE_TOP_Z + CONTAINER_BASE_THK / 2
    wall_z = TABLE_TOP_Z + CONTAINER_BASE_THK + CONTAINER_WALL_H / 2
    half_in_x = CONTAINER_INNER_XY[0] / 2
    half_in_y = CONTAINER_INNER_XY[1] / 2
    half_thk = CONTAINER_WALL_THK / 2
    return [
        (
            'container_base',
            np.array([CONTAINER_XY[0], CONTAINER_XY[1], base_z]),
            np.array([CONTAINER_OUTER_XY[0] / 2, CONTAINER_OUTER_XY[1] / 2, CONTAINER_BASE_THK / 2]),
        ),
        (
            'container_wall_posx',
            np.array([CONTAINER_XY[0] + half_in_x + half_thk, CONTAINER_XY[1], wall_z]),
            np.array([CONTAINER_WALL_THK / 2, CONTAINER_OUTER_XY[1] / 2, CONTAINER_WALL_H / 2]),
        ),
        (
            'container_wall_negx',
            np.array([CONTAINER_XY[0] - half_in_x - half_thk, CONTAINER_XY[1], wall_z]),
            np.array([CONTAINER_WALL_THK / 2, CONTAINER_OUTER_XY[1] / 2, CONTAINER_WALL_H / 2]),
        ),
        (
            'container_wall_posy',
            np.array([CONTAINER_XY[0], CONTAINER_XY[1] + half_in_y + half_thk, wall_z]),
            np.array([CONTAINER_INNER_XY[0] / 2, CONTAINER_WALL_THK / 2, CONTAINER_WALL_H / 2]),
        ),
        (
            'container_wall_negy',
            np.array([CONTAINER_XY[0], CONTAINER_XY[1] - half_in_y - half_thk, wall_z]),
            np.array([CONTAINER_INNER_XY[0] / 2, CONTAINER_WALL_THK / 2, CONTAINER_WALL_H / 2]),
        ),
    ]


def build_scene():
    """Create the scene: table, 4-wall container, cube, Franka, camera."""
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    world.scene.add(
        FixedCuboid(
            prim_path='/World/Table',
            name='table',
            position=TABLE_CENTER,
            scale=TABLE_SIZE,
            color=np.array([0.78, 0.62, 0.45]),
        )
    )

    container_color = np.array([0.20, 0.40, 0.70], dtype=np.float32)
    for name, center, half in _container_parts():
        scale = (half * 2.0).astype(np.float32)
        world.scene.add(
            FixedCuboid(
                prim_path=f'/World/Container/{name}',
                name=name,
                position=center.astype(np.float32),
                scale=scale,
                color=container_color,
            )
        )

    cube = world.scene.add(
        DynamicCuboid(
            prim_path='/World/Cube',
            name='red_cube',
            position=CUBE_POS,
            scale=np.array([CUBE_SIZE] * 3, dtype=np.float32),
            color=np.array([1.0, 0.0, 0.0]),
            mass=0.05,
        )
    )

    franka = world.scene.add(
        Franka(
            prim_path='/World/Franka',
            name='franka',
            position=FRANKA_POS,
        )
    )

    camera = Camera(
        prim_path='/World/Camera',
        position=CAM_POS,
        frequency=20,
        resolution=(256, 256),
        orientation=camera_look_at_orn(CAM_POS, CAM_TARGET),
    )
    return world, franka, cube, camera


# ═══════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════


def main():
    print('[env_collision] building scene ...')
    world, franka, cube, camera = build_scene()
    world.reset()
    camera.initialize()

    print(f'[env_collision] table top z     = {TABLE_TOP_Z:.3f}')
    print(f'[env_collision] container top z = {CONTAINER_TOP_Z:.3f}')
    print(f'[env_collision] cube pos        = {CUBE_POS}')
    print(f'[env_collision] collision threshold = {COLLISION_THRESHOLD*100:.1f} cm')
    print()

    # ── Collision detector: register Franka links + 5 container parts ──
    detector = CollisionDetector(threshold=COLLISION_THRESHOLD)
    link_xforms = setup_link_xforms('/World/Franka')
    detector.add_agent('franka', link_xforms)
    for name, center, half in _container_parts():
        detector.add_box(name, center=center, half_extents=half)

    # ── Drive EE straight to the cube (no waypoint above container) ──
    ee_prim = SingleXFormPrim('/World/Franka/panda_hand')
    rmp = RMPFlowController(name='env_rmpflow', robot_articulation=franka)
    rmp.reset()

    _, ee_orn = ee_prim.get_world_pose()
    ee_orn = np.asarray(ee_orn, dtype=np.float32)

    # Target = directly at the cube, ignoring the container in the path.
    target = CUBE_POS.astype(np.float32)
    print(f'[env_collision] driving EE straight to {target} (ignoring container)')

    reported_pairs = set()  # de-dupe noisy frame-by-frame logs
    for step in range(MOVE_STEPS):
        art_action = rmp.forward(
            target_end_effector_position=target,
            target_end_effector_orientation=ee_orn,
        )
        franka.apply_action(art_action)
        world.step(render=True)

        # Only print on collision, only print each (link, part) pair once.
        events = detector.check_agent_env(step=step)
        for e in events:
            key = (e.entity_a, e.entity_b)
            if key in reported_pairs:
                continue
            reported_pairs.add(key)
            print(f'[COLLISION] {e}')

    if not reported_pairs:
        print('[env_collision] no collision detected.')
    else:
        print(f'[env_collision] total unique colliding pairs: {len(reported_pairs)}')

    print('[env_collision] done. closing.')
    sim_app.close()


if __name__ == '__main__':
    main()
