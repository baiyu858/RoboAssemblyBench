from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from internutopia.core.config import Config, SimConfig
from internutopia.core.util import has_display
from internutopia.core.vec_env import Env
from internutopia_extension import import_extensions

from toolkits.factory_box_carry.demo_policy import HumanoidBenchCarryDemoPolicy
from toolkits.factory_box_carry.scene_builder import build_factory_box_carry_episode


def _to_jsonable(value: Any):
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


class EpisodeRecorder:
    def __init__(self, seed: int, output_dir: Path):
        self.seed = seed
        self.output_dir = output_dir
        self.steps = []

    def record(self, task, obs: dict, actions: dict):
        box_position = task._get_box_pose()[0]  # noqa: SLF001 - convenient for demo recording
        self.steps.append(
            {
                'phase': task.phase,
                'actions': _to_jsonable(actions),
                'observations': _to_jsonable(obs),
                'box_position': _to_jsonable(box_position),
            }
        )

    def save(self, episode_idx: int, metrics: dict):
        payload = {
            'episode_idx': episode_idx,
            'seed': self.seed,
            'metrics': _to_jsonable(metrics),
            'steps': self.steps,
        }
        path = self.output_dir / f'episode_{episode_idx:04d}.json'
        path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _build_env(task_configs, headless: bool):
    config = Config(
        simulator=SimConfig(
            physics_dt=1 / 240,
            rendering_dt=1 / 240,
            use_fabric=False,
            headless=headless,
            # Demo generation is fully offline; streaming only slows startup.
            native=False,
            webrtc=False,
        ),
        env_num=1,
        metrics_save_path='none',
        task_configs=task_configs,
    )
    import_extensions()
    return Env(config)


def _run_task_sequence(task_configs, headless: bool, output_dir: Path | None = None):
    env = _build_env(task_configs, headless=headless)
    policy = HumanoidBenchCarryDemoPolicy()
    obs_list, task_cfgs = env.reset()
    results = []
    episode_idx = 0
    recorder = None

    try:
        while env.simulation_app.is_running() and task_cfgs and task_cfgs[0] is not None:
            task_name = next(iter(env.runner.current_tasks.keys()))
            task = env.runner.current_tasks[task_name]

            if output_dir is not None and recorder is None:
                recorder = EpisodeRecorder(seed=task.config.seed, output_dir=output_dir)

            env_actions = policy.act(task)
            obs_list, _, terminated, _, _ = env.step([env_actions])

            if output_dir is not None:
                recorder.record(task, obs_list[0], env_actions)

            if terminated[0]:
                metrics = task.calculate_metrics()
                results.append(metrics)
                if output_dir is not None and recorder is not None:
                    recorder.save(episode_idx, metrics)
                episode_idx += 1
                recorder = None
                obs_list, task_cfgs = env.reset([0])
                if not task_cfgs or task_cfgs[0] is None:
                    break
    finally:
        env.close()

    return results


def search_successful_seeds(num_demos: int, start_seed: int, max_trials: int, headless: bool):
    candidate_seeds = list(range(start_seed, start_seed + max_trials))
    candidate_cfgs = [build_factory_box_carry_episode(seed=seed, episode_idx=index) for index, seed in enumerate(candidate_seeds)]
    results = _run_task_sequence(candidate_cfgs, headless=headless)
    success_seeds = [result['seed'] for result in results if result['success']]
    return success_seeds[:num_demos]


def collect_demos(seeds, output_dir: Path, headless: bool):
    task_cfgs = [build_factory_box_carry_episode(seed=seed, episode_idx=index) for index, seed in enumerate(seeds)]
    return _run_task_sequence(task_cfgs, headless=headless, output_dir=output_dir)


def main():
    parser = argparse.ArgumentParser(description='Generate cooperative box-carry demos in InternUtopia.')
    parser.add_argument('--num-demos', type=int, default=4)
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument('--max-trials', type=int, default=20)
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Path(__file__).resolve().parent / 'outputs' / 'factory_box_carry'),
    )
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()

    headless = args.headless or not has_display()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    successful_seeds = search_successful_seeds(
        num_demos=args.num_demos,
        start_seed=args.start_seed,
        max_trials=args.max_trials,
        headless=headless,
    )
    if not successful_seeds:
        raise RuntimeError('No successful demo seeds were found. Increase max_trials or inspect the task setup.')

    (output_dir / 'seed.txt').write_text(' '.join(str(seed) for seed in successful_seeds), encoding='utf-8')
    results = collect_demos(successful_seeds, output_dir=output_dir, headless=headless)
    manifest = {
        'num_requested': args.num_demos,
        'num_collected': len(results),
        'headless': headless,
        'successful_seeds': successful_seeds,
        'results': results,
    }
    (output_dir / 'manifest.json').write_text(json.dumps(_to_jsonable(manifest), indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
