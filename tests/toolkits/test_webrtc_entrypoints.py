from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _help_text(script: str) -> str:
    completed = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout + completed.stderr


def test_isaac_python_entrypoints_expose_webrtc_flag():
    scripts = [
        "toolkits/factory_dual_franka_assembly/render_fabrica_traj_replay_isaac.py",
        "toolkits/factory_dual_franka_assembly/render_fabrica_official_motion_isaac.py",
        "toolkits/factory_dual_franka_assembly/render_task_scene_preview.py",
        "toolkits/factory_dual_franka_assembly/view_task_scene.py",
    ]

    for script in scripts:
        assert "--webrtc" in _help_text(script), script


def test_isaac_shell_entrypoints_accept_webrtc_env():
    scripts = [
        "roboassemblybench/scripts/render_fabrica_official_plumbers_block_ur5e_traj_isaacsim.sh",
        "roboassemblybench/scripts/render_fabrica_official_plumbers_block_ur5e_traj_factory_isaacsim.sh",
        "roboassemblybench/scripts/render_fabrica_plumbers_block_ur5e_scene_preview.sh",
        "roboassemblybench/scripts/render_fabrica_official_cooling_manifold_isaacsim.sh",
        "roboassemblybench/scripts/render_fabrica_official_plumbers_block_isaacsim.sh",
        "roboassemblybench/scripts/view_fabrica_plumbers_block_ur5e_scene_ui.sh",
        "roboassemblybench/scripts/view_fabrica_cooling_manifold_ur5e_scene_ui.sh",
    ]

    for script in scripts:
        text = (REPO_ROOT / script).read_text(encoding="utf-8")
        assert "WEBRTC" in text, script
