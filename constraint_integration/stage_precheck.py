"""Passive precheck of commanded UR5e joint segments using Lula FK."""

from __future__ import annotations

from collections import Counter
import time

import numpy as np

from .models import get_robot_collision_model
from .runtime_monitor import RuntimeConstraintConfig, RuntimeConstraintMonitor


class _StaticXform:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)

    def get_world_pose(self):
        return self.position, np.asarray([1.0, 0.0, 0.0, 0.0])


class StageTrajectoryPrechecker:
    """Check actual policy joint commands without changing or rejecting them."""

    def __init__(
        self,
        *,
        check_stride: int = 64,
        num_waypoints: int = 8,
        threshold: float | None = None,
        include_ground: bool = False,
        ignore_pairs=(),
    ):
        self.check_stride = max(int(check_stride), 1)
        self.num_waypoints = max(int(num_waypoints), 2)
        self.model = get_robot_collision_model("ur5e_robotiq_2f85")
        self.monitor = RuntimeConstraintMonitor(
            RuntimeConstraintConfig(
                threshold=threshold,
                include_ground=include_ground,
                ignore_pairs=list(ignore_pairs),
            )
        )
        self.detector = self.monitor.detector
        self._observations = 0
        self._checks = 0
        self._segments = 0
        self._waypoints = 0
        self._violations = []
        self._errors = []
        self._reasons = Counter()
        self._total_seconds = 0.0
        self._ignore_pairs = list(ignore_pairs)

    def observe(self, task, actions: dict) -> dict:
        step = int(getattr(task, "step_counter", 0))
        self._observations += 1
        if step % self.check_stride:
            return {"checked": False, "step": step, "reason": "stride_skip"}
        started = time.perf_counter()
        self._checks += 1
        try:
            if self.detector is None:
                return self._result(step, "detector_unavailable")
            self.monitor.refresh_environment_from_task(task)
            paths = {}
            for robot_name, robot_actions in (actions or {}).items():
                path, reason = self._joint_path(task, str(robot_name), robot_actions)
                if path is None:
                    self._reasons[reason] += 1
                    continue
                paths[str(robot_name)] = path
                self._segments += 1
            if not paths:
                return self._result(step, "no_joint_targets")

            sample_count = max(len(path) for path in paths.values())
            events = []
            for waypoint_index in range(sample_count):
                link_positions_by_robot = {}
                for robot_name, path in paths.items():
                    q = path[min(waypoint_index, len(path) - 1)]
                    positions = self._forward_link_positions(task, robot_name, q)
                    if positions:
                        link_positions_by_robot[robot_name] = positions
                        events.extend(
                            self.detector.check_config(
                                positions,
                                step=step,
                                agent_name=f"{robot_name}@waypoint_{waypoint_index}",
                            )
                        )
                events.extend(self._check_inter_robot(link_positions_by_robot, step, waypoint_index))
                self._waypoints += 1
            serialized = [self.monitor._event_to_dict(event) for event in events]
            serialized = [
                event
                for event in serialized
                if not any(
                    rule.matches(event["entity_a"], event["entity_b"])
                    for rule in self._ignore_pairs
                )
            ]
            self._violations.extend(serialized)
            return {
                "checked": True,
                "step": step,
                "reason": "ok",
                "segments": len(paths),
                "violations": serialized,
            }
        except Exception as exc:
            error = self._error("observe", exc, step)
            self._record_error(error)
            return {"checked": True, "step": step, "reason": "error", "monitor_error": error}
        finally:
            self._total_seconds += max(time.perf_counter() - started, 0.0)

    def finalize(self) -> dict:
        minimum_distance = min(
            (float(event["distance"]) for event in self._violations),
            default=None,
        )
        return {
            "enabled": True,
            "mode": "passive",
            "check_stride": self.check_stride,
            "num_waypoints": self.num_waypoints,
            "observed_steps": self._observations,
            "checks": self._checks,
            "segments_checked": self._segments,
            "waypoints_checked": self._waypoints,
            "violation_total": len(self._violations),
            "violations_by_kind": dict(Counter(event["kind"] for event in self._violations)),
            "minimum_distance": minimum_distance,
            "events": list(self._violations),
            "skip_reasons": dict(self._reasons),
            "monitor_error": list(self._errors),
            "total_check_seconds": self._total_seconds,
        }

    def _joint_path(self, task, robot_name: str, robot_actions):
        if not isinstance(robot_actions, dict):
            return None, "invalid_robot_action"
        robot = (getattr(task, "robots", {}) or {}).get(robot_name)
        if robot is None:
            return None, "missing_robot"
        joint_controller = self._controller(robot, "joint")
        ik_controller = self._controller(robot, "ik")
        if joint_controller is None or ik_controller is None:
            return None, "missing_controller"
        target = self._joint_target(robot_actions, joint_controller)
        if target is None:
            return None, "missing_joint_target"
        subset = joint_controller.get_joint_subset()
        current = np.asarray(subset.get_joint_positions(), dtype=float).reshape(-1)
        target = np.asarray(target, dtype=float).reshape(-1)
        if current.shape != target.shape or not np.all(np.isfinite(target)):
            return None, "invalid_joint_target"
        return [
            current + (target - current) * fraction
            for fraction in np.linspace(0.0, 1.0, self.num_waypoints)
        ], "ok"

    @staticmethod
    def _controller(robot, kind: str):
        controllers = getattr(robot, "controllers", {}) or {}
        try:
            from internutopia_extension.configs.robots.franka import arm_ik_cfg, arm_joint_cfg

            configured_name = arm_joint_cfg.name if kind == "joint" else arm_ik_cfg.name
            if configured_name in controllers:
                return controllers[configured_name]
        except Exception:
            pass
        for name, controller in controllers.items():
            lowered = str(name).lower()
            if kind == "joint" and "joint" in lowered and "gripper" not in lowered:
                return controller
            if kind == "ik" and ("ik" in lowered or "inverse" in lowered):
                return controller
        return None

    @staticmethod
    def _joint_target(robot_actions: dict, joint_controller):
        configured_name = getattr(getattr(joint_controller, "config", None), "name", None)
        candidates = []
        if configured_name:
            candidates.append(configured_name)
        candidates.extend(name for name in robot_actions if "joint" in str(name).lower())
        for name in candidates:
            value = robot_actions.get(name)
            if value is None:
                continue
            array = np.asarray(value, dtype=float)
            if array.ndim >= 2:
                array = array[0]
            return array
        return None

    def _forward_link_positions(self, task, robot_name: str, q: np.ndarray) -> dict:
        robot = task.robots[robot_name]
        controller = self._controller(robot, "ik")
        solver = getattr(controller, "_kinematics_solver")
        lula_solver = getattr(solver, "_kinematics", solver)
        base_pose = controller.get_ik_base_world_pose()
        solver.set_robot_base_pose(
            robot_position=np.asarray(base_pose[0], dtype=float) / controller._robot_scale,
            robot_orientation=np.asarray(base_pose[1], dtype=float),
        )
        available = set(lula_solver.get_all_frame_names())
        result = {}
        for link_name in self.model.link_names:
            frame_name = link_name.rsplit("/", 1)[-1]
            if frame_name not in available:
                continue
            position, _ = lula_solver.compute_forward_kinematics(frame_name, joint_positions=q)
            result[link_name] = np.asarray(position, dtype=float)
        return result

    def _check_inter_robot(self, positions_by_robot, step, waypoint_index):
        if len(positions_by_robot) < 2:
            return []
        self.detector._agents.clear()
        for robot_name, positions in positions_by_robot.items():
            self.detector.add_agent(
                f"{robot_name}@waypoint_{waypoint_index}",
                {name: _StaticXform(position) for name, position in positions.items()},
            )
        return self.detector.check_inter_agent(step=step)

    def _result(self, step, reason):
        self._reasons[reason] += 1
        return {"checked": True, "step": step, "reason": reason, "violations": []}

    def _record_error(self, error: dict) -> None:
        key = (error["stage"], error["type"], error["message"])
        for existing in self._errors:
            existing_key = (existing["stage"], existing["type"], existing["message"])
            if existing_key == key:
                existing["last_step"] = error["step"]
                existing["occurrences"] = int(existing.get("occurrences", 1)) + 1
                return
        error["last_step"] = error["step"]
        error["occurrences"] = 1
        self._errors.append(error)

    @staticmethod
    def _error(stage, exc, step):
        return {
            "stage": stage,
            "step": int(step),
            "type": type(exc).__name__,
            "message": str(exc),
        }
