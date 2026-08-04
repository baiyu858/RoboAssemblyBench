"""Execution-time trajectory precheck — IK reachability + collision detection.

Before any EE target is actually driven by the controller, the entire path
(current EE pose → target) is validated waypoint-by-waypoint:
  1. Lula IK: solve inverse kinematics for each waypoint → reachability.
  2. FK: compute link world positions from the solved joints.
  3. CollisionDetector.check_config(): check capsules vs boxes/ground.

If any waypoint fails, the trajectory is rejected as a whole before any
action touches the physics simulation. Complements src/checker.py (which
validates task-level semantics) with geometric-level feasibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

try:
    from isaacsim.robot_motion.motion_generation import (
        LulaKinematicsSolver,
        interface_config_loader,
    )
except ImportError:
    from omni.isaac.motion_generation import (
        LulaKinematicsSolver,
        interface_config_loader,
    )


@dataclass
class PrecheckReport:
    """Structured result of a trajectory precheck."""

    feasible: bool
    reason: str = 'ok'  # "ok" | "ik_unreachable" | "joint_limit" | "collision"
    first_fail_idx: int = -1  # 0-based index of the first failing waypoint
    num_waypoints: int = 0
    events: list = field(default_factory=list)  # CollisionEvent list if collision

    def __str__(self):
        if self.feasible:
            return f'[precheck OK] {self.num_waypoints} waypoints feasible'
        detail = ''
        if self.events:
            detail = ' | ' + '; '.join(str(e) for e in self.events[:3])
        return f'[precheck FAIL] {self.reason} ' f'at waypoint {self.first_fail_idx}/{self.num_waypoints}{detail}'


class TrajectoryPrechecker:
    """Validate an EE trajectory (current pose → target) before execution.

    Parameters:
        franka:          The Franka articulation object (for initial joint state).
        collision_detector: CollisionDetector with boxes/ground already registered.
        robot_name:      Lula robot config name (default "Franka").
        ee_frame:        End-effector frame name for IK.
        num_waypoints:   How many intermediate poses to sample along the path.
    """

    def __init__(
        self,
        franka,
        collision_detector,
        robot_name: str = 'Franka',
        ee_frame: str = 'panda_hand',
        num_waypoints: int = 12,
    ):
        self.franka = franka
        self.detector = collision_detector
        self.ee_frame = ee_frame
        self.num_waypoints = max(2, num_waypoints)

        # ── Lula KinematicsSolver (one-shot IK, no physics) ──
        cfg = interface_config_loader.load_supported_lula_kinematics_solver_config(
            robot_name,
        )
        self.ik = LulaKinematicsSolver(**cfg)

        # ── set robot base pose from the Franka articulation ──
        base_pos, base_orn = franka.get_world_pose()
        self.ik.set_robot_base_pose(
            np.asarray(base_pos, dtype=np.float64),
            np.asarray(base_orn, dtype=np.float64),
        )

        # ── identify which Franka links the solver supports for FK ──
        all_frames = set(self.ik.get_all_frame_names())
        self._fk_links = [ln for ln in self.detector.franka_links if ln in all_frames]

        # ── joint limits (read from articulation for fine-grained cutoff) ──
        self._joint_lower = None
        self._joint_upper = None
        try:
            dof = self.franka.num_dof if hasattr(self.franka, 'num_dof') else 7
            self._joint_lower = np.array(
                [self.franka.get_dof_limits()[0][i] for i in range(dof)],
                dtype=np.float64,
            )
            self._joint_upper = np.array(
                [self.franka.get_dof_limits()[1][i] for i in range(dof)],
                dtype=np.float64,
            )
        except Exception:
            pass

    def _current_joints(self) -> np.ndarray:
        """Read current joint positions from the Franka articulation.
        Returns only the arm joints (first 7), since Lula solver expects exactly 7 DOF.
        """
        try:
            if hasattr(self.franka, 'get_joint_positions'):
                q = np.asarray(self.franka.get_joint_positions(), dtype=np.float64)
            else:
                q = np.asarray(self.franka.joint_positions, dtype=np.float64)
            return q[:7] if len(q) > 7 else q
        except Exception:
            return np.zeros(7, dtype=np.float64)

    def _interpolate_waypoints(
        self, start_pos, start_orn, target_pos, target_orn
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Linear position interpolation + constant orientation.
        Returns list of (pos, orn) waypoints, including both endpoints.
        Uses the provided target_orn (guaranteed quaternion) for all waypoints
        to avoid FK returning non-quaternion orientation formats.
        """
        sp = np.asarray(start_pos, dtype=np.float64)
        tp = np.asarray(target_pos, dtype=np.float64)
        orn = np.asarray(target_orn, dtype=np.float64)  # use input quaternion directly

        waypoints = []
        for frac in np.linspace(0.0, 1.0, self.num_waypoints):
            pos = sp + (tp - sp) * frac
            waypoints.append((pos, orn))
        return waypoints

    def _check_joint_limits(self, q: np.ndarray) -> bool:
        """Return True if *q* is within joint limits."""
        if self._joint_lower is None:
            return True
        dof = min(len(q), len(self._joint_lower))
        return bool(np.all(q[:dof] >= self._joint_lower[:dof]) and np.all(q[:dof] <= self._joint_upper[:dof]))

    def check_trajectory(self, target_pos, target_orn, step: int = 0) -> PrecheckReport:
        """Run precheck on the full path from current EE pose to target.

        Args:
            target_pos:  3D target EE position (world frame, meters).
            target_orn:  4D target EE orientation [w, x, y, z].
            step:        Current step index (for log context).

        Returns:
            PrecheckReport with .feasible and .reason.
        """
        # ── 1. get current EE pose as start ──
        cur_q = self._current_joints()
        start_pos, start_orn = self.ik.compute_forward_kinematics(
            self.ee_frame,
            cur_q,
        )
        start_pos = np.asarray(start_pos, dtype=np.float64)
        start_orn = np.asarray(start_orn, dtype=np.float64)

        # ── 2. generate waypoints ──
        waypoints = self._interpolate_waypoints(
            start_pos,
            start_orn,
            target_pos,
            target_orn,
        )

        # ── 3. validate each waypoint ──
        warm_start = cur_q[:7].copy()
        for i, (pos, orn) in enumerate(waypoints):
            # 3a. IK solve
            result, success = self.ik.compute_inverse_kinematics(
                self.ee_frame,
                np.asarray(pos, dtype=np.float64),
                np.asarray(orn, dtype=np.float64),
                warm_start,
            )
            if not success or result is None:
                return PrecheckReport(
                    False,
                    'ik_unreachable',
                    i,
                    self.num_waypoints,
                )

            # result is already a numpy array of joint positions
            q = np.asarray(result, dtype=np.float64)[:7]
            warm_start = q.copy()

            # 3b. joint limit check
            if not self._check_joint_limits(q):
                return PrecheckReport(
                    False,
                    'joint_limit',
                    i,
                    self.num_waypoints,
                )

            # 3c. FK → link world positions
            link_positions: Dict[str, np.ndarray] = {}
            for ln in self._fk_links:
                lp, _ = self.ik.compute_forward_kinematics(ln, q)
                link_positions[ln] = np.asarray(lp, dtype=np.float64)

            # 3d. collision check
            events = self.detector.check_config(
                link_positions,
                step=step,
                agent_name='franka',
            )
            if events:
                return PrecheckReport(
                    False,
                    'collision',
                    i,
                    self.num_waypoints,
                    events,
                )

        return PrecheckReport(True, 'ok', -1, self.num_waypoints)
