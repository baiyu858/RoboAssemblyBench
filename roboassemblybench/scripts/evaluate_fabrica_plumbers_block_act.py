from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboassemblybench.datasets.cartesian_episode import (
    ACTION_NAMES,
    CartesianObservationEncoder,
)
from roboassemblybench.policies.act_rpc import PolicyRPCClient
from toolkits.factory_dual_franka_assembly.generate_demos import _build_env
from toolkits.factory_dual_franka_assembly.scene_builder import build_dual_franka_assembly_batch
from toolkits.factory_dual_franka_assembly.task_specs import load_task_recipe


DEFAULT_RECIPE = 'fabrica_plumbers_block_ur5e_right_base_prepare'
DEFAULT_SCENE_PROFILE = 'taoyuan_grscenes_tabletop'


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'{path.suffix}.tmp')
    temporary.write_text(json.dumps(_jsonable(payload), indent=2), encoding='utf-8')
    temporary.replace(path)


def _normalize_quaternion(quaternion, *, reference: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        return reference.copy()
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-8:
        return reference.copy()
    quaternion = quaternion / norm
    if float(np.dot(quaternion, reference)) < 0.0:
        quaternion = -quaternion
    return quaternion


def _bounded_quaternion(target, current, *, max_angle: float) -> np.ndarray:
    current = _normalize_quaternion(current, reference=np.asarray([1.0, 0.0, 0.0, 0.0]))
    target = _normalize_quaternion(target, reference=current)
    dot = float(np.clip(np.dot(current, target), -1.0, 1.0))
    angle = 2.0 * float(np.arccos(dot))
    if angle <= max(float(max_angle), 0.0) or angle <= 1e-8:
        return target
    fraction = float(max_angle) / angle
    sin_half_angle = float(np.sqrt(max(1.0 - dot * dot, 0.0)))
    if sin_half_angle <= 1e-6:
        blended = current + fraction * (target - current)
    else:
        half_angle = float(np.arccos(dot))
        blended = (
            np.sin((1.0 - fraction) * half_angle) / sin_half_angle * current
            + np.sin(fraction * half_angle) / sin_half_angle * target
        )
    return _normalize_quaternion(blended, reference=current)


def sanitize_absolute_cartesian_action(
    action,
    current_state,
    *,
    max_translation_step: float,
    max_rotation_step: float,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64)
    current_state = np.asarray(current_state, dtype=np.float64)
    if action.shape != (len(ACTION_NAMES),) or current_state.shape != (len(ACTION_NAMES),):
        raise ValueError(f'Expected 16D action/state, got {action.shape} and {current_state.shape}.')
    if not np.all(np.isfinite(action)) or not np.all(np.isfinite(current_state)):
        raise ValueError('Policy action and current Cartesian state must be finite.')

    bounded = action.copy()
    for offset in (0, 8):
        delta = bounded[offset : offset + 3] - current_state[offset : offset + 3]
        distance = float(np.linalg.norm(delta))
        if distance > float(max_translation_step) > 0.0:
            bounded[offset : offset + 3] = (
                current_state[offset : offset + 3]
                + delta * (float(max_translation_step) / distance)
            )
        bounded[offset + 3 : offset + 7] = _bounded_quaternion(
            bounded[offset + 3 : offset + 7],
            current_state[offset + 3 : offset + 7],
            max_angle=float(max_rotation_step),
        )
        bounded[offset + 7] = float(np.clip(bounded[offset + 7], 0.0, 1.0))
    return bounded.astype(np.float32)


def _env_action(action: np.ndarray) -> dict[str, dict[str, list]]:
    result = {}
    for robot_name, offset in (('franka_left', 0), ('franka_right', 8)):
        result[robot_name] = {
            'arm_ik_controller': [
                action[offset : offset + 3].tolist(),
                action[offset + 3 : offset + 7].tolist(),
            ],
            'gripper_controller': [float(action[offset + 7])],
        }
    return result


def _task_description(task) -> str:
    config = getattr(task, 'config', None)
    return str(
        getattr(config, 'task_description', None)
        or getattr(config, 'prompt', None)
        or DEFAULT_RECIPE
    )


def apply_max_steps_override(task_configs, max_steps: int | None) -> None:
    if max_steps is None:
        return
    max_steps = int(max_steps)
    if max_steps <= 0:
        raise ValueError('max_steps must be positive when provided.')
    for task_config in task_configs:
        task_config.max_steps = max_steps


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate an ACT policy online in Isaac Sim.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--num-episodes', type=int, default=50)
    parser.add_argument('--start-seed', type=int, default=10000)
    parser.add_argument('--layout-seeds', type=int, nargs='+', default=None)
    parser.add_argument('--recipe', default=DEFAULT_RECIPE)
    parser.add_argument('--scene-profile', default=DEFAULT_SCENE_PROFILE)
    parser.add_argument('--output-dir', default='outputs/fabrica_plumbers_block_ur5e_act_eval_50ep')
    parser.add_argument('--control-stride', type=int, default=8)
    parser.add_argument('--rendering-fps', type=int, default=240)
    parser.add_argument('--max-translation-step', type=float, default=0.04)
    parser.add_argument('--max-rotation-step', type=float, default=0.35)
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Override the recipe episode limit; intended for short end-to-end smoke evaluations.',
    )
    parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.num_episodes <= 0 or args.control_stride <= 0:
        parser.error('num-episodes and control-stride must be positive.')
    output_dir = Path(args.output_dir).resolve()
    results_path = output_dir / 'episode_results.json'
    summary_path = output_dir / 'success_rate.json'
    seeds = list(range(int(args.start_seed), int(args.start_seed) + int(args.num_episodes)))
    if args.layout_seeds is None:
        recipe_spec = load_task_recipe(str(args.recipe), scene_profile=str(args.scene_profile))
        args.layout_seeds = [
            int(seed) for seed in (recipe_spec.get('collection') or {}).get('layout_seeds', [])
        ]
    if not args.layout_seeds:
        parser.error('--layout-seeds requires at least one seed or recipe collection.layout_seeds.')
    layout_seeds = [
        int(args.layout_seeds[index % len(args.layout_seeds)])
        for index in range(int(args.num_episodes))
    ]

    client = PolicyRPCClient(host=args.host, port=args.port)
    server_info = client.ping()
    task_configs = build_dual_franka_assembly_batch(
        recipe=str(args.recipe),
        seeds=seeds,
        layout_seeds=layout_seeds,
        scene_profile=str(args.scene_profile),
        attach_runtime_cameras=True,
        domain_randomization_enabled=True,
        policy_evaluation_mode=True,
    )
    apply_max_steps_override(task_configs, args.max_steps)
    env = _build_env(
        task_configs=task_configs,
        headless=bool(args.headless),
        rendering_fps=int(args.rendering_fps),
        rendering_interval=int(args.control_stride) - 1,
    )
    obs_list, task_cfgs = env.reset()
    results = []
    episode_index = 0
    encoder = None
    held_env_action = None
    episode_started = time.time()

    try:
        while task_cfgs and task_cfgs[0] is not None and not env.finished():
            task = next(iter(env.runner.current_tasks.values()))
            if encoder is None:
                encoder = CartesianObservationEncoder(task=task, output_resolution=(640, 480))
                held_env_action = None
                episode_started = time.time()
                client.reset()

            if held_env_action is None or int(task.step_counter) % int(args.control_stride) == 0:
                policy_observation = encoder.encode(task=task, obs=obs_list[0])
                if policy_observation is not None:
                    current_state = policy_observation['observation.state']
                    action = client.predict(policy_observation, task=_task_description(task))
                    action = sanitize_absolute_cartesian_action(
                        action,
                        current_state,
                        max_translation_step=float(args.max_translation_step),
                        max_rotation_step=float(args.max_rotation_step),
                    )
                    held_env_action = _env_action(action)
            if held_env_action is None:
                held_env_action = _env_action(encoder.state(task, obs_list[0]))

            obs_list, _, terminated, _, _ = env.step([held_env_action])
            if not terminated[0]:
                continue

            metrics = task.calculate_metrics()
            episode_result = {
                'episode_index': episode_index,
                'seed': int(getattr(task.config, 'seed', seeds[episode_index])),
                'layout_seed': int(
                    layout_seeds[episode_index]
                    if getattr(task.config, 'layout_seed', None) is None
                    else task.config.layout_seed
                ),
                'success': bool(metrics.get('success', False)),
                'terminal_reason': metrics.get('terminal_reason'),
                'steps': int(getattr(task, 'step_counter', 0)),
                'wall_seconds': time.time() - episode_started,
                'domain_randomization': metrics.get('domain_randomization') or {},
                'success_detector': metrics.get('success_detector') or {},
                'policy_interaction_history': metrics.get('policy_interaction_history') or [],
            }
            results.append(episode_result)
            _write_json_atomic(results_path, results)
            successes = sum(int(item['success']) for item in results)
            _write_json_atomic(
                summary_path,
                {
                    'complete': len(results) == int(args.num_episodes),
                    'num_episodes': len(results),
                    'target_episodes': int(args.num_episodes),
                    'num_successes': successes,
                    'success_rate': successes / len(results),
                    'start_seed': int(args.start_seed),
                    'layout_seeds': list(dict.fromkeys(layout_seeds)),
                    'policy_server': server_info,
                    'results_path': str(results_path),
                },
            )
            print(
                f"episode={episode_index} seed={episode_result['seed']} "
                f"layout_seed={episode_result['layout_seed']} "
                f"success={episode_result['success']} rate={successes}/{len(results)}",
                flush=True,
            )
            episode_index += 1
            encoder = None
            held_env_action = None
            obs_list, task_cfgs = env.reset([0])
            if not task_cfgs or task_cfgs[0] is None:
                break
    finally:
        client.close()
        env.close()

    if len(results) != int(args.num_episodes):
        raise RuntimeError(f'Online evaluation ended after {len(results)}/{args.num_episodes} episodes.')


if __name__ == '__main__':
    main()
