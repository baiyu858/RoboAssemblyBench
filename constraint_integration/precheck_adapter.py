"""Execution-time trajectory precheck adapter.

This file keeps the precheck algorithm independent from the current UR5e skill
implementation.  It expects callers to provide IK and FK callbacks from the
controller/skill layer, so no existing RoboAssemblyBench files need to import
or depend on it yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np


IKCallback = Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], Tuple[Optional[np.ndarray], bool]]
FKCallback = Callable[[np.ndarray], Dict[str, np.ndarray]]


@dataclass
class PrecheckReport:
    feasible: bool
    reason: str = "ok"
    first_fail_idx: int = -1
    num_waypoints: int = 0
    events: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "feasible": bool(self.feasible),
            "reason": str(self.reason),
            "first_fail_idx": int(self.first_fail_idx),
            "num_waypoints": int(self.num_waypoints),
            "events": [str(event) for event in self.events],
        }


class LinearPosePrechecker:
    """Precheck a straight-line EE pose path with caller-provided IK/FK.

    The existing constraint_detection prechecker is tied to Franka Lula config.
    This adapter is robot-agnostic: the UR5e skill layer can pass its own IK and
    FK functions later, after the baseline task is confirmed working.
    """

    def __init__(self, detector, num_waypoints: int = 12):
        self.detector = detector
        self.num_waypoints = max(int(num_waypoints), 2)

    def check(
        self,
        *,
        start_position: Iterable[float],
        target_position: Iterable[float],
        target_orientation: Iterable[float],
        solve_ik: IKCallback,
        forward_kinematics: FKCallback,
        warm_start: Optional[np.ndarray] = None,
        step: int = 0,
    ) -> PrecheckReport:
        target_orientation = np.asarray(target_orientation, dtype=float)
        waypoints = self._linear_waypoints(start_position, target_position)
        q_ref = None if warm_start is None else np.asarray(warm_start, dtype=float)

        for index, position in enumerate(waypoints):
            q, ok = solve_ik(np.asarray(position, dtype=float), target_orientation, q_ref)
            if not ok or q is None:
                return PrecheckReport(
                    feasible=False,
                    reason="ik_unreachable",
                    first_fail_idx=index,
                    num_waypoints=self.num_waypoints,
                )
            q_ref = np.asarray(q, dtype=float)
            link_positions = forward_kinematics(q_ref)
            events = self.detector.check_config(
                link_positions,
                step=int(step),
                agent_name="precheck_robot",
            )
            if events:
                return PrecheckReport(
                    feasible=False,
                    reason="collision",
                    first_fail_idx=index,
                    num_waypoints=self.num_waypoints,
                    events=events,
                )

        return PrecheckReport(
            feasible=True,
            reason="ok",
            first_fail_idx=-1,
            num_waypoints=self.num_waypoints,
        )

    def _linear_waypoints(self, start_position, target_position) -> list[np.ndarray]:
        start = np.asarray(start_position, dtype=float)
        target = np.asarray(target_position, dtype=float)
        return [start + (target - start) * frac for frac in np.linspace(0.0, 1.0, self.num_waypoints)]
