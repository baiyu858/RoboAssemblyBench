"""LIBERO-style scene: large tabletop with Franka and red cube on top."""
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from scipy.spatial.transform import Rotation as R

# ---- Layout constants -------------------------------------------------------
TABLE_CENTER = np.array([0.30, 0.00, 0.20], dtype=np.float32)
TABLE_SIZE = np.array([1.40, 1.00, 0.40], dtype=np.float32)  # X, Y, Z
TABLE_TOP_Z = TABLE_CENTER[2] + TABLE_SIZE[2] / 2  # = 0.40

FRANKA_POS = np.array([0.00, 0.00, TABLE_TOP_Z], dtype=np.float32)

# ---- Container (4 walls + thin base, open top) -----------------------------
CONTAINER_XY = np.array([0.50, 0.00], dtype=np.float32)
CONTAINER_BASE_THK = 0.005
CONTAINER_WALL_THK = 0.008
CONTAINER_WALL_H = 0.06
CONTAINER_INNER_XY = np.array([0.16, 0.16], dtype=np.float32)
CONTAINER_OUTER_XY = CONTAINER_INNER_XY + 2 * CONTAINER_WALL_THK

CUBE_SIZE = 0.05
# Cube sits on the container's inner floor.
CUBE_POS = np.array(
    [
        CONTAINER_XY[0],
        CONTAINER_XY[1],
        TABLE_TOP_Z + CONTAINER_BASE_THK + CUBE_SIZE / 2 + 0.001,
    ],
    dtype=np.float32,
)

# LIBERO-style agentview: camera in front looking back at the workspace
# CAM_POS    = np.array([1.40, 0.00, TABLE_TOP_Z + 0.50], dtype=np.float32)
# CAM_TARGET = np.array([0.40, 0.00, TABLE_TOP_Z + 0.05], dtype=np.float32)

CAM_POS = np.array([1.20, 0.00, TABLE_TOP_Z + 0.65], dtype=np.float32)
CAM_TARGET = CUBE_POS


def camera_look_at_orn(eye, target):
    """Return wxyz quaternion for Isaac Sim camera looking from eye to target."""
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    direction = target - eye
    direction /= np.linalg.norm(direction) + 1e-8
    yaw = np.arctan2(-direction[1], -direction[0])
    pitch = np.arcsin(-direction[2])
    r = R.from_euler('YX', [yaw, pitch])
    q = r.as_quat()
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def build_scene():
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # 1) Big wood-colored tabletop, both robot and cube live on top of it
    world.scene.add(
        FixedCuboid(
            prim_path='/World/Table',
            name='table',
            position=TABLE_CENTER,
            scale=TABLE_SIZE,
            color=np.array([0.78, 0.62, 0.45]),
        )
    )

    # 2) Franka mounted on the tabletop (base sits on z = TABLE_TOP_Z)
    try:
        from isaacsim.robot.manipulators.examples.franka import Franka
    except ImportError:
        from omni.isaac.franka import Franka
    franka = world.scene.add(
        Franka(
            prim_path='/World/Franka',
            name='franka',
            position=FRANKA_POS,
        )
    )

    # 3) A real concave container on the tabletop: thin base + 4 thin walls.
    container_color = np.array([0.55, 0.40, 0.25], dtype=np.float32)
    base_z = TABLE_TOP_Z + CONTAINER_BASE_THK / 2
    wall_z = TABLE_TOP_Z + CONTAINER_BASE_THK + CONTAINER_WALL_H / 2
    half_inner_x = CONTAINER_INNER_XY[0] / 2
    half_inner_y = CONTAINER_INNER_XY[1] / 2
    half_thk = CONTAINER_WALL_THK / 2

    # base
    world.scene.add(
        FixedCuboid(
            prim_path='/World/Container/Base',
            name='container_base',
            position=np.array([CONTAINER_XY[0], CONTAINER_XY[1], base_z], dtype=np.float32),
            scale=np.array([CONTAINER_OUTER_XY[0], CONTAINER_OUTER_XY[1], CONTAINER_BASE_THK], dtype=np.float32),
            color=container_color,
        )
    )
    # +X wall
    world.scene.add(
        FixedCuboid(
            prim_path='/World/Container/WallPosX',
            name='container_wall_posx',
            position=np.array([CONTAINER_XY[0] + half_inner_x + half_thk, CONTAINER_XY[1], wall_z], dtype=np.float32),
            scale=np.array([CONTAINER_WALL_THK, CONTAINER_OUTER_XY[1], CONTAINER_WALL_H], dtype=np.float32),
            color=container_color,
        )
    )
    # -X wall
    world.scene.add(
        FixedCuboid(
            prim_path='/World/Container/WallNegX',
            name='container_wall_negx',
            position=np.array([CONTAINER_XY[0] - half_inner_x - half_thk, CONTAINER_XY[1], wall_z], dtype=np.float32),
            scale=np.array([CONTAINER_WALL_THK, CONTAINER_OUTER_XY[1], CONTAINER_WALL_H], dtype=np.float32),
            color=container_color,
        )
    )
    # +Y wall
    world.scene.add(
        FixedCuboid(
            prim_path='/World/Container/WallPosY',
            name='container_wall_posy',
            position=np.array([CONTAINER_XY[0], CONTAINER_XY[1] + half_inner_y + half_thk, wall_z], dtype=np.float32),
            scale=np.array([CONTAINER_INNER_XY[0], CONTAINER_WALL_THK, CONTAINER_WALL_H], dtype=np.float32),
            color=container_color,
        )
    )
    # -Y wall
    world.scene.add(
        FixedCuboid(
            prim_path='/World/Container/WallNegY',
            name='container_wall_negy',
            position=np.array([CONTAINER_XY[0], CONTAINER_XY[1] - half_inner_y - half_thk, wall_z], dtype=np.float32),
            scale=np.array([CONTAINER_INNER_XY[0], CONTAINER_WALL_THK, CONTAINER_WALL_H], dtype=np.float32),
            color=container_color,
        )
    )

    # 4) Red cube sitting inside the container.
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

    # 4) LIBERO-style frontview camera (256x256, looking down at the workspace)
    try:
        from isaacsim.sensors.camera import Camera
    except ImportError:
        from omni.isaac.sensor import Camera

    camera = Camera(
        prim_path='/World/Camera',
        position=CAM_POS,
        frequency=20,
        resolution=(256, 256),
        orientation=camera_look_at_orn(CAM_POS, CAM_TARGET),
    )
    return world, franka, cube, camera
