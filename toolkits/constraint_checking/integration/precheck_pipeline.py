"""Episode lifecycle hooks for passive sequence and stage prechecks."""

from __future__ import annotations

from .runtime_monitor import PairFilter
from .sequence_precheck import AssemblySequencePrechecker
from .stage_precheck import StageTrajectoryPrechecker


class PassivePrecheckEpisodeHook:
    def __init__(
        self,
        *,
        sequence_enabled: bool = False,
        stage_enabled: bool = False,
        stage_check_stride: int = 64,
        stage_waypoints: int = 8,
        threshold: float | None = None,
        include_ground: bool = False,
        ignore_pairs: list[str] | None = None,
        sequence_factory=None,
        stage_factory=None,
    ):
        self.sequence_enabled = bool(sequence_enabled)
        self.stage_enabled = bool(stage_enabled)
        self.stage_check_stride = max(int(stage_check_stride), 1)
        self.stage_waypoints = max(int(stage_waypoints), 2)
        self.threshold = threshold
        self.include_ground = bool(include_ground)
        self.ignore_pairs = list(ignore_pairs or [])
        self.sequence_factory = sequence_factory
        self.stage_factory = stage_factory
        self.reset_episode()

    def observe_before_step(self, task, actions: dict) -> None:
        if self.sequence_enabled and self._sequence_report is None:
            try:
                checker = self.sequence_factory() if self.sequence_factory else AssemblySequencePrechecker()
                config = getattr(task, 'config', None)
                phases = getattr(config, 'phase_specs', None)
                if phases is None:
                    phases = getattr(task, 'phase_specs', [])
                self._sequence_report = checker.check(
                    phases=phases or [],
                    robot_names=getattr(config, 'robot_names', []) or [],
                    object_names=getattr(config, 'object_names', []) or [],
                ).to_dict()
            except Exception as exc:
                self._sequence_report = self._failure('sequence', exc)
        if self.stage_enabled:
            try:
                self._get_stage_checker().observe(task, actions)
            except Exception as exc:
                self._stage_failure = self._failure('stage', exc)

    def attach_metrics(self, metrics: dict) -> dict:
        if self.sequence_enabled:
            metrics['assembly_sequence_precheck'] = self._sequence_report or {
                'enabled': True,
                'mode': 'passive',
                'feasible': None,
                'errors': [],
                'warnings': [],
                'monitor_error': [{'stage': 'sequence', 'message': 'precheck_not_observed'}],
            }
        if self.stage_enabled:
            metrics['stage_trajectory_precheck'] = (
                self._stage_failure if self._stage_failure is not None else self._get_stage_checker().finalize()
            )
        return metrics

    def reset_episode(self) -> None:
        self._sequence_report = None
        self._stage_checker = None
        self._stage_failure = None

    def _get_stage_checker(self):
        if self._stage_checker is None:
            filters = [PairFilter.parse(value) for value in self.ignore_pairs]
            self._stage_checker = (
                self.stage_factory()
                if self.stage_factory
                else StageTrajectoryPrechecker(
                    check_stride=self.stage_check_stride,
                    num_waypoints=self.stage_waypoints,
                    threshold=self.threshold,
                    include_ground=self.include_ground,
                    ignore_pairs=filters,
                )
            )
        return self._stage_checker

    @staticmethod
    def _failure(stage: str, exc: Exception) -> dict:
        return {
            'enabled': True,
            'mode': 'passive',
            'checks': 0,
            'violation_total': 0,
            'events': [],
            'monitor_error': [
                {
                    'stage': stage,
                    'type': type(exc).__name__,
                    'message': str(exc),
                }
            ],
        }
