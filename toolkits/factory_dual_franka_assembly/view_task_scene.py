from __future__ import annotations

import argparse
import time

from internutopia.core.config import Config, SimConfig
from internutopia.core.util import has_display
from internutopia.core.vec_env import Env
from internutopia_extension import import_extensions

from toolkits.factory_dual_franka_assembly.scene_builder import build_dual_franka_assembly_batch
from toolkits.factory_dual_franka_assembly.scene_profiles import DEFAULT_SCENE_PROFILE, list_scene_profiles
from toolkits.factory_dual_franka_assembly.task_specs import list_task_recipes


def _build_env(task_configs, *, headless: bool) -> Env:
    config = Config(
        simulator=SimConfig(
            physics_dt=1 / 240,
            rendering_dt=1 / 240,
            use_fabric=False,
            headless=headless,
            native=False,
            webrtc=False,
        ),
        env_num=1,
        metrics_save_path='none',
        task_configs=task_configs,
    )
    import_extensions()
    return Env(config)


def _pause_timeline():
    try:
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        timeline.pause()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description='Open a task scene in Isaac Sim UI without running robot actions.')
    parser.add_argument('--recipe', default='peg_insertion', choices=list_task_recipes())
    parser.add_argument('--scene-profile', default=DEFAULT_SCENE_PROFILE, choices=list_scene_profiles())
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument(
        '--warmup-render-steps',
        type=int,
        default=8,
        help='Render-only warmup steps after reset. Physics is not stepped.',
    )
    parser.add_argument(
        '--attach-runtime-cameras',
        action='store_true',
        help='Attach task cameras as runtime sensors for camera inspection. Disabled by default for a lighter scene view.',
    )
    args = parser.parse_args()

    if not args.headless and not has_display():
        raise RuntimeError('No display detected. Use a desktop session or pass --headless.')

    env = _build_env(
        task_configs=build_dual_franka_assembly_batch(
            recipe=args.recipe,
            seeds=[int(args.seed)],
            scene_profile=args.scene_profile,
            attach_runtime_cameras=bool(args.attach_runtime_cameras),
        ),
        headless=bool(args.headless),
    )

    try:
        obs_list, task_cfgs = env.reset()
        if not task_cfgs or task_cfgs[0] is None or not env.runner.current_tasks:
            raise RuntimeError(f'No task scene was loaded for recipe {args.recipe!r}.')

        _pause_timeline()
        env.runner.warm_up(steps=max(int(args.warmup_render_steps), 1), render=True, physics=False)
        _pause_timeline()

        task_name = next(iter(env.runner.current_tasks.keys()))
        task = env.runner.current_tasks[task_name]
        print('Task scene viewer is live.')
        print(f'Recipe        : {getattr(task.config, "recipe", args.recipe)}')
        print(f'Scene profile : {getattr(task.config, "scene_profile", args.scene_profile)}')
        print(f'Seed          : {getattr(task.config, "seed", args.seed)}')
        print(f'Scene asset   : {getattr(task.config, "scene_asset_path", "")}')
        print('No demo policy or robot actions are running. Press Ctrl+C to exit.')

        while env.simulation_app.is_running():
            env.simulation_app.update()
            time.sleep(0.02)
    except KeyboardInterrupt:
        print('\nTask scene viewer interrupted by user.')
    finally:
        env.close()


if __name__ == '__main__':
    main()
