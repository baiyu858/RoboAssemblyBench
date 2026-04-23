from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PlannerWaypoint:
    name: str
    mode: str
    position: np.ndarray
    orientation: np.ndarray
    tolerance: float | None = None


class FrankaRobofactoryPlanner:
    """Small mplib-based planner bridge for the dual Franka assembly demos.

    This is intentionally narrower than RoboFactory's original planner stack:
    it only plans 7-DoF Franka arm trajectories to `panda_hand`, using a
    simplified planning URDF/SRDF that avoids external mesh dependencies.
    """

    JOINT_NAMES = [f'panda_joint{i}' for i in range(1, 8)]
    LINK_NAMES = ['base_link'] + [f'panda_link{i}' for i in range(0, 9)] + ['panda_hand']

    def __init__(self) -> None:
        self._mplib = None
        self._planner = None
        self._planner_root = Path(__file__).resolve().parent / 'planning_assets'
        self._urdf_path = self._planner_root / 'franka_mplib.urdf'
        self._srdf_path = self._planner_root / 'franka_mplib.srdf'

    @property
    def available(self) -> bool:
        try:
            self._ensure_planner()
        except Exception:
            return False
        return True

    def _ensure_planner(self):
        if self._planner is not None and self._mplib is not None:
            return self._planner

        import mplib

        self._mplib = mplib
        self._planner = mplib.Planner(
            str(self._urdf_path),
            'panda_hand',
            srdf=str(self._srdf_path),
            user_joint_names=self.JOINT_NAMES,
            user_link_names=self.LINK_NAMES,
            verbose=False,
        )
        return self._planner

    @staticmethod
    def _as_pose(mplib_module, position, orientation):
        return mplib_module.Pose(
            np.asarray(position, dtype=float).tolist(),
            np.asarray(orientation, dtype=float).tolist(),
        )

    def _plan_single(self, planner, waypoint: PlannerWaypoint, current_qpos: np.ndarray) -> dict | None:
        goal_pose = self._as_pose(self._mplib, waypoint.position, waypoint.orientation)
        result = None
        if waypoint.mode in {'pick', 'hold', 'insert'}:
            result = planner.plan_screw(
                goal_pose,
                current_qpos,
                qpos_step=0.08 if waypoint.mode != 'insert' else 0.04,
                time_step=1.0 / 30.0,
                wrt_world=True,
                verbose=False,
            )
            if result.get('status') == 'Success':
                return result

        result = planner.plan_pose(
            goal_pose,
            current_qpos,
            time_step=1.0 / 30.0,
            planning_time=1.0,
            rrt_range=0.12,
            wrt_world=True,
            simplify=True,
            verbose=False,
        )
        if result.get('status') == 'Success':
            return result
        return None

    def plan_waypoints(
        self,
        *,
        base_position,
        base_orientation,
        start_qpos,
        waypoints: list[PlannerWaypoint],
    ) -> list[dict] | None:
        if not waypoints:
            return []

        planner = self._ensure_planner()
        planner.set_base_pose(self._as_pose(self._mplib, base_position, base_orientation))

        current_qpos = np.asarray(start_qpos, dtype=float).copy()
        trajectory: list[dict] = []

        for waypoint in waypoints:
            result = self._plan_single(planner, waypoint, current_qpos)
            if result is None:
                return None
            planned_qpos = np.asarray(result.get('position'), dtype=float)
            if planned_qpos.ndim != 2 or planned_qpos.shape[1] != len(self.JOINT_NAMES):
                return None
            for qpos in planned_qpos:
                if trajectory and np.allclose(trajectory[-1]['joint_positions'], qpos, atol=1e-6):
                    continue
                trajectory.append(
                    {
                        'name': waypoint.name,
                        'mode': waypoint.mode,
                        'pose': {
                            'position': waypoint.position.copy(),
                            'orientation': waypoint.orientation.copy(),
                        },
                        'tolerance': waypoint.tolerance,
                        'joint_positions': np.asarray(qpos, dtype=float).copy(),
                    }
                )
            current_qpos = planned_qpos[-1]

        return trajectory
