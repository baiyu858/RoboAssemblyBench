"""Small, testable lifecycle hook for passive rollout monitoring."""

from __future__ import annotations

from typing import Callable


class _StaticReportMonitor:
    def __init__(self, report: dict):
        self.report = report

    def observe(self, task) -> dict:
        return {
            "checked": False,
            "step": int(getattr(task, "step_counter", 0)),
            "violations": [],
            "monitor_error": self.report.get("monitor_error", []),
        }

    def finalize(self) -> dict:
        return dict(self.report)


class RuntimeConstraintEpisodeHook:
    """Own exactly one monitor instance per active episode."""

    metric_key = "runtime_constraint_monitor"

    def __init__(
        self,
        *,
        enabled: bool = False,
        check_stride: int = 8,
        threshold: float | None = None,
        collision_threshold: float = 0.0,
        include_ground: bool = False,
        ignore_pairs: list[str] | None = None,
        monitor_factory: Callable[[], object] | None = None,
    ):
        self.enabled = bool(enabled)
        self.check_stride = max(int(check_stride), 1)
        self.threshold = threshold
        self.collision_threshold = float(collision_threshold)
        self.include_ground = bool(include_ground)
        self.ignore_pairs = list(ignore_pairs or [])
        self._monitor_factory = monitor_factory
        self._monitor = None

    def observe(self, task) -> dict | None:
        if not self.enabled:
            return None
        monitor = self._get_monitor()
        try:
            return monitor.observe(task)
        except Exception as exc:
            self._replace_with_failure_report(monitor, "observe", exc, task=task)
            return self._monitor.observe(task)

    def attach_metrics(self, metrics: dict) -> dict:
        if not self.enabled:
            return metrics
        metrics[self.metric_key] = self.finalize()
        return metrics

    def finalize(self) -> dict | None:
        if not self.enabled:
            return None
        monitor = self._get_monitor()
        try:
            return monitor.finalize()
        except Exception as exc:
            self._replace_with_failure_report(monitor, "finalize", exc)
            return self._monitor.finalize()

    def reset_episode(self) -> None:
        self._monitor = None

    def _get_monitor(self):
        if self._monitor is not None:
            return self._monitor
        try:
            self._monitor = self._monitor_factory() if self._monitor_factory else self._build_monitor()
        except Exception as exc:
            self._monitor = _StaticReportMonitor(self._failure_report("initialize", exc))
        return self._monitor

    def _build_monitor(self):
        from .runtime_monitor import PairFilter, RuntimeConstraintConfig, RuntimeConstraintMonitor

        return RuntimeConstraintMonitor(
            RuntimeConstraintConfig(
                check_stride=self.check_stride,
                threshold=self.threshold,
                collision_threshold=self.collision_threshold,
                include_ground=self.include_ground,
                ignore_pairs=[PairFilter.parse(value) for value in self.ignore_pairs],
            )
        )

    def _replace_with_failure_report(self, monitor, stage: str, exc: Exception, *, task=None) -> None:
        try:
            report = dict(monitor.finalize())
        except Exception:
            report = {}
        errors = list(report.get("monitor_error") or [])
        errors.append(self._error(stage, exc, task=task))
        report.update(
            {
                "enabled": True,
                "checks": int(report.get("checks", 0)),
                "violation_total": int(report.get("violation_total", 0)),
                "events": list(report.get("events") or []),
                "monitor_error": errors,
            }
        )
        self._monitor = _StaticReportMonitor(report)

    @classmethod
    def _failure_report(cls, stage: str, exc: Exception) -> dict:
        return {
            "enabled": True,
            "checks": 0,
            "violation_total": 0,
            "events": [],
            "monitor_error": [cls._error(stage, exc)],
        }

    @staticmethod
    def _error(stage: str, exc: Exception, *, task=None) -> dict:
        return {
            "stage": str(stage),
            "step": None if task is None else int(getattr(task, "step_counter", 0)),
            "type": type(exc).__name__,
            "message": str(exc),
        }
