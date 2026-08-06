"""Trajectory precheck demo — no VLA, hand-crafted test targets.

Feeds a sequence of EE target positions into TrajectoryPrechecker to
validate IK reachability and collision-freeness (against table, container
walls, and ground plane). Demonstrates:
  1) safe target above container   → PASS
  2) straight to cube through wall → REJECTED (collision)
  3) far unreachable position      → REJECTED (IK infeasible)
  4) below table                   → REJECTED (ground collision)
  5) two-step safe trajectory      → both PASS (validates path planning)

Run:
    conda deactivate
    /mnt/SSD_7T/panxubei/isaac-sim5.1/python.sh demos/main.py
"""
from isaacsim import SimulationApp

sim_app = SimulationApp({'headless': False})

import numpy as np
from scipy.spatial.transform import Rotation as R

from toolkits.constraint_checking.detector.collision import CollisionDetector
from toolkits.constraint_checking.detector.precheck import TrajectoryPrechecker
from toolkits.constraint_checking.detector.scene import (
    CONTAINER_BASE_THK,
    CONTAINER_INNER_XY,
    CONTAINER_OUTER_XY,
    CONTAINER_WALL_H,
    CONTAINER_WALL_THK,
    CONTAINER_XY,
    CUBE_POS,
    TABLE_CENTER,
    TABLE_SIZE,
    TABLE_TOP_Z,
    build_scene,
)

WARMUP_STEPS = 60
PRECHECK_WAYPOINTS = 12  # trajectory discrete-interpolation count
COLLISION_THRESHOLD = 0.02  # 2cm clearance threshold
CONTAINER_TOP_Z = TABLE_TOP_Z + CONTAINER_BASE_THK + CONTAINER_WALL_H


def _container_parts():
    """Return [(name, center, half_extents)] for all container obstacles.

    Mirrors the geometry built in src/scene.py — 1 base + 4 walls.
    """
    base_z = TABLE_TOP_Z + CONTAINER_BASE_THK / 2.0
    wall_z = TABLE_TOP_Z + CONTAINER_BASE_THK + CONTAINER_WALL_H / 2.0
    half_in_x = CONTAINER_INNER_XY[0] / 2.0
    half_in_y = CONTAINER_INNER_XY[1] / 2.0
    half_thk = CONTAINER_WALL_THK / 2.0
    return [
        (
            'container_base',
            np.array([CONTAINER_XY[0], CONTAINER_XY[1], base_z], dtype=np.float64),
            np.array(
                [CONTAINER_OUTER_XY[0] / 2.0, CONTAINER_OUTER_XY[1] / 2.0, CONTAINER_BASE_THK / 2.0], dtype=np.float64
            ),
        ),
        (
            'container_wall_posx',
            np.array([CONTAINER_XY[0] + half_in_x + half_thk, CONTAINER_XY[1], wall_z], dtype=np.float64),
            np.array([CONTAINER_WALL_THK / 2.0, CONTAINER_OUTER_XY[1] / 2.0, CONTAINER_WALL_H / 2.0], dtype=np.float64),
        ),
        (
            'container_wall_negx',
            np.array([CONTAINER_XY[0] - half_in_x - half_thk, CONTAINER_XY[1], wall_z], dtype=np.float64),
            np.array([CONTAINER_WALL_THK / 2.0, CONTAINER_OUTER_XY[1] / 2.0, CONTAINER_WALL_H / 2.0], dtype=np.float64),
        ),
        (
            'container_wall_posy',
            np.array([CONTAINER_XY[0], CONTAINER_XY[1] + half_in_y + half_thk, wall_z], dtype=np.float64),
            np.array([CONTAINER_INNER_XY[0] / 2.0, CONTAINER_WALL_THK / 2.0, CONTAINER_WALL_H / 2.0], dtype=np.float64),
        ),
        (
            'container_wall_negy',
            np.array([CONTAINER_XY[0], CONTAINER_XY[1] - half_in_y - half_thk, wall_z], dtype=np.float64),
            np.array([CONTAINER_INNER_XY[0] / 2.0, CONTAINER_WALL_THK / 2.0, CONTAINER_WALL_H / 2.0], dtype=np.float64),
        ),
    ]


def _default_orn_wxyz():
    """Default EE orientation: pointing down (flipped) for grasping."""
    r = R.from_euler('xyz', [np.pi, 0.0, 0.0])
    q = r.as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def main():
    world, franka, _, _ = build_scene()
    world.reset()

    # ── collision detector (offline, for precheck) ──
    detector = CollisionDetector(threshold=COLLISION_THRESHOLD)
    detector.add_box('table', center=TABLE_CENTER, half_extents=TABLE_SIZE / 2.0)
    for name, center, half in _container_parts():
        detector.add_box(name, center=center, half_extents=half)
    detector.add_ground(z=0.0)

    # ── trajectory prechecker (Lula IK + collision) ──
    print(f'[main] warm-up {WARMUP_STEPS} physics steps ...')
    for _ in range(WARMUP_STEPS):
        world.step(render=True)

    prechecker = TrajectoryPrechecker(
        franka,
        detector,
        num_waypoints=PRECHECK_WAYPOINTS,
    )

    ee_orn = _default_orn_wxyz()

    # ── test targets ──
    #  1) above container — safe, should PASS
    #  2) straight to cube — goes through container wall, should REJECT
    #  3) far unreachable — outside workspace, IK should fail
    #  4) below table — penetrates ground, should REJECT
    #  5) two-step: first go above container (safe), then down into it

    print('\n' + '=' * 68)
    print('[precheck demo] Testing target waypoints against:')
    print(f'  table       center={TABLE_CENTER}, size={TABLE_SIZE}')
    print(f'  container   top_z={CONTAINER_TOP_Z:.3f}')
    print(f'  cube        pos={CUBE_POS}')
    print(f'  threshold   ={COLLISION_THRESHOLD*100:.0f}cm')
    print(f'  waypoints   ={PRECHECK_WAYPOINTS}')
    print('=' * 68)

    # ── single-target tests ──
    test_targets = [
        (
            'above_container (safe)',
            np.array([CONTAINER_XY[0], CONTAINER_XY[1], CONTAINER_TOP_Z + 0.15], dtype=np.float64),
        ),
        ('straight_to_cube (wall collision)', np.array([CUBE_POS[0], CUBE_POS[1], CUBE_POS[2]], dtype=np.float64)),
        ('far_left (unreachable)', np.array([1.5, 0.0, TABLE_TOP_Z + 0.3], dtype=np.float64)),
        ('below_table (ground collision)', np.array([0.3, 0.0, -0.1], dtype=np.float64)),
    ]

    for target_no, (label, target_pos) in enumerate(test_targets, 1):
        print(f'\n── test {target_no}: {label}')
        print(f'   target = {np.round(target_pos, 3)}')

        report = prechecker.check_trajectory(target_pos, ee_orn, step=target_no)
        status = 'PASS' if report.feasible else 'REJECTED'
        print(f'   result = {status}')
        print(f'   {report}')
        if report.events:
            for e in report.events:
                print(f'   -> {e}')

    # ── two-step safe trajectory ──
    print('\n── test 5: two-step safe trajectory')
    waypoint_above = np.array(
        [CONTAINER_XY[0], CONTAINER_XY[1], CONTAINER_TOP_Z + 0.15],
        dtype=np.float64,
    )
    waypoint_cube = np.array(
        [CUBE_POS[0], CUBE_POS[1], CUBE_POS[2]],
        dtype=np.float64,
    )

    for i, (label, wp) in enumerate(
        [('step1: above container', waypoint_above), ('step2: down to cube', waypoint_cube)]
    ):
        report = prechecker.check_trajectory(wp, ee_orn, step=5)
        status = 'PASS' if report.feasible else 'REJECTED'
        print(f'   {label} → {status}  {report}')

    print('\n[main] done. closing.')
    sim_app.close()


if __name__ == '__main__':
    main()
