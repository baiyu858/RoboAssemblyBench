from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'{path.suffix}.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


def aggregate_evaluation(
    *,
    episodes_dir: Path,
    output_dir: Path,
    expected_episodes: int,
    start_seed: int,
    layout_seeds: list[int] | None = None,
) -> dict[str, Any]:
    expected_seeds = set(range(int(start_seed), int(start_seed) + int(expected_episodes)))
    results_by_seed = {}
    summaries = []
    for results_path in episodes_dir.glob('episode_*/episode_results.json'):
        results = _load_json(results_path)
        summary_path = results_path.parent / 'success_rate.json'
        if not summary_path.is_file() or len(results) != 1:
            continue
        summary = _load_json(summary_path)
        result = results[0]
        seed = int(result.get('seed', -1))
        if not bool(summary.get('complete')) or int(summary.get('num_episodes', -1)) != 1:
            continue
        if seed in results_by_seed:
            raise ValueError(f'Duplicate online-evaluation result for seed {seed}.')
        results_by_seed[seed] = result
        summaries.append(summary)

    actual_seeds = set(results_by_seed)
    if actual_seeds != expected_seeds:
        missing = sorted(expected_seeds - actual_seeds)
        unexpected = sorted(actual_seeds - expected_seeds)
        raise ValueError(f'Online evaluation seed mismatch: missing={missing}, unexpected={unexpected}.')

    if layout_seeds is not None:
        layout_seeds = [int(seed) for seed in layout_seeds]
        allowed_layout_seeds = set(layout_seeds)
        unexpected_layout_seeds = sorted(
            {
                int(result.get('layout_seed', -1))
                for result in results_by_seed.values()
                if int(result.get('layout_seed', -1)) not in allowed_layout_seeds
            }
        )
        if unexpected_layout_seeds:
            raise ValueError(f'Online evaluation layout seed mismatch: unexpected={unexpected_layout_seeds}.')

    results = [results_by_seed[seed] for seed in sorted(results_by_seed)]
    successes = sum(int(bool(item.get('success'))) for item in results)
    summary = {
        'complete': True,
        'num_episodes': len(results),
        'target_episodes': int(expected_episodes),
        'num_successes': successes,
        'success_rate': successes / len(results),
        'start_seed': int(start_seed),
        'end_seed': int(start_seed) + int(expected_episodes) - 1,
        'layout_seeds': layout_seeds or [],
        'single_isaac_process_per_episode': True,
        'policy_server': summaries[0].get('policy_server') if summaries else None,
        'results_path': str((output_dir / 'episode_results.json').resolve()),
    }
    _write_json_atomic(output_dir / 'episode_results.json', results)
    _write_json_atomic(output_dir / 'success_rate.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Aggregate isolated ACT online-evaluation episodes.')
    parser.add_argument('--episodes-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--expected-episodes', type=int, required=True)
    parser.add_argument('--start-seed', type=int, required=True)
    parser.add_argument('--layout-seeds', type=int, nargs='+', default=None)
    args = parser.parse_args()
    if args.expected_episodes <= 0:
        parser.error('expected-episodes must be positive.')
    summary = aggregate_evaluation(
        episodes_dir=Path(args.episodes_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        expected_episodes=int(args.expected_episodes),
        start_seed=int(args.start_seed),
        layout_seeds=args.layout_seeds,
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
