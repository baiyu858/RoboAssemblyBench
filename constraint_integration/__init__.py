"""Isolated constraint-detection bridge for RoboAssemblyBench.

This package is intentionally not imported by the main benchmark code.  It
contains adapters that can be wired into demo generation or atomic skills once
the baseline assembly rollout is verified.
"""

from .models import RobotCollisionModel, get_robot_collision_model
from .pipeline import RuntimeConstraintEpisodeHook
from .runtime_monitor import PairFilter, RuntimeConstraintConfig, RuntimeConstraintMonitor
from .precheck_adapter import LinearPosePrechecker, PrecheckReport
from .precheck_pipeline import PassivePrecheckEpisodeHook
from .sequence_precheck import AssemblySequencePrechecker, SequencePrecheckReport
from .stage_precheck import StageTrajectoryPrechecker

__all__ = [
    "LinearPosePrechecker",
    "PrecheckReport",
    "PairFilter",
    "PassivePrecheckEpisodeHook",
    "RobotCollisionModel",
    "RuntimeConstraintConfig",
    "RuntimeConstraintEpisodeHook",
    "RuntimeConstraintMonitor",
    "AssemblySequencePrechecker",
    "SequencePrecheckReport",
    "StageTrajectoryPrechecker",
    "get_robot_collision_model",
]
