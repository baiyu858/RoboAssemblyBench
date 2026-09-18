"""Run AutoMate assembly training/evaluation from the RoboAssemblyBench tree.

The upstream ``run_w_id.py`` assumes that AutoMate lives inside
``isaaclab_tasks`` and therefore does not import the vendored Gym registration
used here.  This small adapter imports the vendored package first and then
delegates to Isaac Lab's unmodified RL-Games ``train.py`` or ``play.py``.
Assembly-specific values are passed as environment variables and are resolved
when the vendored config module is imported; no source config is rewritten.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_ISAACLAB_ROOT = _REPO_ROOT / "IsaacLab"
_TASK_ID = "RoboAssemblyBench-AutoMate-Assembly-Direct-v0"


def _bootstrap_pythonpath() -> None:
    candidates = (
        _REPO_ROOT,
        _ISAACLAB_ROOT / "source" / "isaaclab",
        _ISAACLAB_ROOT / "source" / "isaaclab_tasks",
        _ISAACLAB_ROOT / "source" / "isaaclab_rl",
        _ISAACLAB_ROOT / "source" / "isaaclab_assets",
    )
    for candidate in reversed(candidates):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run vendored AutoMate assembly training or checkpoint evaluation."
    )
    parser.add_argument("--mode", choices=("train", "play"), default="train")
    parser.add_argument("--assembly-id", default="00015")
    parser.add_argument(
        "--disassembly-json",
        type=Path,
        default=None,
        help="Optional generated AutoMate trajectory JSON for imitation reward.",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=None,
        help="HDF5 path used by AutoMate evaluation logging.",
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--max-iterations", type=int, default=1500)
    parser.add_argument("--checkpoint", type=Path, default=None)
    # Any Isaac Lab AppLauncher flags (such as --headless/--device) remain in
    # the list passed to train.py/play.py.
    return parser.parse_known_args()


def _get_passthrough_option(args: list[str], name: str) -> str | None:
    """Return a value passed as either ``--name value`` or ``--name=value``."""
    prefix = f"{name}="
    for index, arg in enumerate(args):
        if arg.startswith(prefix):
            return arg[len(prefix) :]
        if arg == name and index + 1 < len(args):
            return args[index + 1]
    return None


def main() -> None:
    _bootstrap_pythonpath()
    args, passthrough = _parse_args()
    if args.num_envs < 1:
        raise SystemExit("--num-envs must be positive")
    if args.mode == "train" and args.max_iterations < 1:
        raise SystemExit("--max-iterations must be positive")
    if args.mode == "play" and args.checkpoint is None:
        raise SystemExit("--checkpoint is required with --mode play")

    os.environ["ROBOASSEMBLYBENCH_AUTOMATE_ASSEMBLY_ID"] = str(args.assembly_id)
    if args.disassembly_json is not None:
        os.environ["ROBOASSEMBLYBENCH_AUTOMATE_DISASSEMBLY_TRAJ"] = str(args.disassembly_json.resolve())
    if args.eval_output is not None:
        os.environ["ROBOASSEMBLYBENCH_AUTOMATE_EVAL_FILENAME"] = str(args.eval_output.resolve())
        os.environ["ROBOASSEMBLYBENCH_AUTOMATE_LOG_EVAL"] = "1"

    # Import first: Isaac Lab's hydra_task_config resolves env_cfg_entry_point
    # from Gym's registry after train.py/play.py has launched the app.
    import roboassemblybench.library.automate  # noqa: F401

    if args.mode == "train":
        script = _ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rl_games" / "train.py"
        command = [
            str(script),
            f"--task={_TASK_ID}",
            f"--num_envs={args.num_envs}",
            f"--seed={args.seed}",
            f"--max_iterations={args.max_iterations}",
        ]
        if args.checkpoint is not None:
            command.append(f"--checkpoint={args.checkpoint.resolve()}")
    else:
        script = _ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rl_games" / "play.py"
        command = [
            str(script),
            f"--task={_TASK_ID}",
            f"--num_envs={args.num_envs}",
            f"--checkpoint={args.checkpoint.resolve()}",
            f"--seed={args.seed}",
        ]
        # RL-Games' player reads ``device_name`` while its trainer reads
        # ``device``.  Keep both on the AppLauncher-selected CUDA device.
        player_device = _get_passthrough_option(passthrough, "--device")
        if player_device is not None and not any("agent.params.config.device_name" in arg for arg in passthrough):
            command.append(f"+agent.params.config.device_name={player_device}")

    sys.argv = command + passthrough
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
