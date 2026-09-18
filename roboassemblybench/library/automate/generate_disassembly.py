"""Generate AutoMate disassembly demonstrations without editing upstream configs.

This is the RoboAssemblyBench-side entry point for AutoMate's original data
generation path.  The upstream :class:`DisassemblyEnv` performs the grasp,
pull-out and randomization motions and writes ``*_disassemble_traj.json``
when an episode times out.  This launcher only supplies the Isaac Lab app,
registers the vendored Gym task, and overrides the assembly-specific paths in
memory.

Run it with Isaac Lab's Python wrapper, for example::

    cd /path/to/InternUtopia/IsaacLab
    TERM=xterm ./isaaclab.sh -p \
      ../roboassemblybench/library/automate/generate_disassembly.py \
      --assembly-id 00015 --output-dir \
      ../outputs/automate/00015/disassembly --num-envs 32 \
      --num-trajectories 100 --headless

The vendored environment currently follows the upstream behavior and exits
with status 0 immediately after writing the requested JSON file.  Therefore a
successful run ends with the ``Trajectory collection complete!`` message.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


TASK_ID = "RoboAssemblyBench-AutoMate-Disassembly-Direct-v0"
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _bootstrap_pythonpath() -> None:
    """Make a source checkout runnable before editable installs are created."""

    candidates = (
        _REPO_ROOT,
        _REPO_ROOT / "IsaacLab" / "source" / "isaaclab",
        _REPO_ROOT / "IsaacLab" / "source" / "isaaclab_tasks",
        _REPO_ROOT / "IsaacLab" / "source" / "isaaclab_rl",
        _REPO_ROOT / "IsaacLab" / "source" / "isaaclab_assets",
    )
    for candidate in reversed(candidates):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


_bootstrap_pythonpath()


def _build_parser() -> argparse.ArgumentParser:
    # Importing AppLauncher is delayed until this function so importing helper
    # functions from this module does not start Isaac Sim by itself.
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(
        description=(
            "Generate AutoMate disassembly JSON demonstrations using the "
            "vendored Isaac Lab environment."
        )
    )
    parser.add_argument("--assembly-id", default="00015", help="AutoMate assembly directory, e.g. 00015.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "outputs" / "automate" / "00015" / "disassembly",
        help="Directory for <assembly-id>_disassemble_traj.json.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=32,
        help="Number of parallel AutoMate environments (default: 32).",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=100,
        help="Number of trajectories to retain in the JSON output.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Environment reset seed.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Safety limit for environment steps; 0 means no extra limit.",
    )
    # Isaac Lab adds --headless, --device, --enable_cameras, etc.  Keep those
    # flags identical to the official train/play scripts.
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _set_usd_path(cfg_task, field_name: str, assembly_dir: Path) -> None:
    """Point an Isaac Lab asset config at the selected local assembly."""

    asset_cfg = getattr(cfg_task, field_name)
    usd_name = str(getattr(asset_cfg, "usd_path", ""))
    if not usd_name:
        raise ValueError(f"AutoMate task config has no usd_path for {field_name}.")
    # The nested spawn config is a configclass, not a plain dict, so assigning
    # in memory preserves the rest of the upstream asset settings.
    asset_cfg.spawn.usd_path = str(assembly_dir / usd_name)


def _configure_task(cfg, assembly_id: str, output_dir: Path, num_trajectories: int, asset_root: Path) -> None:
    """Override assembly-dependent fields without touching source files."""

    assembly_dir = asset_root / assembly_id
    required = (
        assembly_dir / "plug.usd",
        assembly_dir / "socket.usd",
        assembly_dir / "plug.obj",
        assembly_dir / "socket.obj",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "AutoMate assembly is incomplete. Missing files:\n  " + "\n  ".join(missing)
        )

    task_cfg = cfg.tasks[cfg.task_name]
    task_cfg.assembly_id = assembly_id
    task_cfg.assembly_dir = str(assembly_dir) + os.sep
    task_cfg.disassembly_dir = str(output_dir.resolve())
    task_cfg.num_log_traj = int(num_trajectories)
    task_cfg.plug_grasp_json = str(asset_root / "plug_grasps.json")
    task_cfg.disassembly_dist_json = str(asset_root / "disassembly_dist.json")
    _set_usd_path(task_cfg, "fixed_asset", assembly_dir)
    _set_usd_path(task_cfg, "held_asset", assembly_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.num_trajectories < 1:
        parser.error("--num-trajectories must be positive")
    if args.max_steps < 0:
        parser.error("--max-steps cannot be negative")

    # Launch Isaac Sim before importing Isaac Lab task modules.  This mirrors
    # Isaac Lab's official RL-Games scripts and is required by Isaac Sim 5.x.
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import torch

        # Importing the vendored package registers TASK_ID.  It is deliberately
        # not added to IsaacLab's task package, so the RoboAssemblyBench tree
        # remains isolated from the upstream checkout.
        import roboassemblybench.library.automate  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        from roboassemblybench.library.automate.asset_root import get_asset_root

        asset_root = Path(get_asset_root()).expanduser().resolve()
        cfg = load_cfg_from_registry(TASK_ID, "env_cfg_entry_point")
        cfg.scene.num_envs = int(args.num_envs)
        if args.device is not None:
            cfg.sim.device = args.device
        _configure_task(
            cfg,
            assembly_id=str(args.assembly_id),
            output_dir=args.output_dir,
            num_trajectories=int(args.num_trajectories),
            asset_root=asset_root,
        )

        output_file = args.output_dir.resolve() / f"{args.assembly_id}_disassemble_traj.json"
        print(f"[AutoMate] task: {TASK_ID}")
        print(f"[AutoMate] assembly: {args.assembly_id}")
        print(f"[AutoMate] assets: {asset_root}")
        print(f"[AutoMate] output: {output_file}")
        print(f"[AutoMate] parallel envs: {args.num_envs}; target trajectories: {args.num_trajectories}")

        env = gym.make(TASK_ID, cfg=cfg)
        try:
            env.reset(seed=int(args.seed))
            # DirectRLEnv accepts a torch action tensor.  Zero actions are
            # intentional: AutoMate's disassembly logger executes its scripted
            # extraction/randomization sequence at timeout, independently of a
            # learned policy.
            device = getattr(env.unwrapped, "device", cfg.sim.device)
            action = torch.zeros((int(args.num_envs), 6), device=device)
            steps = 0
            while True:
                env.step(action)
                steps += 1
                if output_file.is_file():
                    break
                if args.max_steps and steps >= args.max_steps:
                    raise RuntimeError(
                        f"AutoMate did not write {output_file} within {args.max_steps} environment steps."
                    )
        finally:
            env.close()
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
