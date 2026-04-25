from __future__ import annotations

from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = BENCHMARK_ROOT / 'tasks'
SHARED_TASK_DIR = TASKS_DIR / '_shared'
SCENE_PROFILE_DIR = BENCHMARK_ROOT / 'scenes' / 'profiles'
LEGACY_TOOLKIT_DIR = BENCHMARK_ROOT.parent / 'toolkits' / 'factory_dual_franka_assembly'
