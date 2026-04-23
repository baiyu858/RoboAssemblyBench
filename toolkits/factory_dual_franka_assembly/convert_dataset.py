from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from collections import defaultdict
from pathlib import Path


def collect_episode_paths(input_dir: Path) -> list[Path]:
    return sorted(input_dir.rglob('episode_*.json'))


def load_episode_payloads(input_dir: Path) -> list[dict]:
    payloads = []
    for episode_path in collect_episode_paths(input_dir):
        payload = json.loads(episode_path.read_text(encoding='utf-8'))
        payload['source_path'] = str(episode_path)
        payloads.append(payload)
    return payloads


def build_dataset_entries(episodes: list[dict], success_only: bool = True) -> list[dict]:
    entries = []
    for episode in episodes:
        if success_only and not episode.get('metrics', {}).get('success', False):
            continue
        steps = episode.get('steps', [])
        phase_trace = [step.get('phase') for step in steps]
        instruction = episode.get('task_description') or episode.get('annotation_description') or episode.get('prompt', '')
        entries.append(
            {
                'episode_id': f"{episode['recipe']}_{episode['episode_idx']:04d}",
                'recipe': episode['recipe'],
                'seed': episode['seed'],
                'prompt': episode.get('prompt', ''),
                'instruction': instruction,
                'task_description': episode.get('task_description') or instruction,
                'annotation_name': episode.get('annotation_name', ''),
                'annotation_path': episode.get('annotation_path'),
                'annotation_title': episode.get('annotation_title', ''),
                'annotation_summary': episode.get('annotation_summary', ''),
                'annotation_description': episode.get('annotation_description', ''),
                'annotation_tags': episode.get('annotation_tags', []),
                'annotation_metadata': episode.get('annotation_metadata', {}),
                'annotation_object_roles': episode.get('annotation_object_roles', {}),
                'annotation_target_roles': episode.get('annotation_target_roles', {}),
                'annotation_phase_notes': episode.get('annotation_phase_notes', []),
                'target_annotations': episode.get('target_annotations', {}),
                'phase_annotations': episode.get('phase_annotations', []),
                'scene_profile': episode.get('scene_profile'),
                'spec_path': episode.get('spec_path'),
                'scene_profile_path': episode.get('scene_profile_path'),
                'scene_asset_path': episode.get('scene_asset_path'),
                'workspace_offset': episode.get('workspace_offset', []),
                'asset_references': episode.get('asset_references', []),
                'metadata': episode.get('metadata', {}),
                'task_metadata': episode.get('task_metadata', {}),
                'scene_profile_metadata': episode.get('scene_profile_metadata', {}),
                'source_benchmark': episode.get('source_benchmark', 'factory_dual_franka_assembly'),
                'source_config_path': episode.get('source_config_path'),
                'camera_metadata': episode.get('camera_metadata', []),
                'robot_metadata': episode.get('robot_metadata', []),
                'object_metadata': episode.get('object_metadata', []),
                'success': episode.get('metrics', {}).get('success', False),
                'num_steps': len(steps),
                'phase_trace': phase_trace,
                'phase_segments': build_phase_segments(phase_trace),
                'observations': [step.get('observations', {}) for step in steps],
                'actions': [step.get('actions', {}) for step in steps],
                'objects': [step.get('objects', {}) for step in steps],
                'metrics': episode.get('metrics', {}),
                'source_path': episode.get('source_path'),
            }
        )
    return entries


def build_phase_segments(phase_trace: list[str | None]) -> list[dict]:
    if not phase_trace:
        return []

    segments = []
    current_phase = phase_trace[0]
    segment_start = 0
    for index, phase in enumerate(phase_trace[1:], start=1):
        if phase == current_phase:
            continue
        segments.append(
            {
                'phase': current_phase,
                'start_step': segment_start,
                'end_step': index - 1,
                'num_steps': index - segment_start,
            }
        )
        current_phase = phase
        segment_start = index

    segments.append(
        {
            'phase': current_phase,
            'start_step': segment_start,
            'end_step': len(phase_trace) - 1,
            'num_steps': len(phase_trace) - segment_start,
        }
    )
    return segments


def split_entries(
    entries: list[dict],
    val_ratio: float,
    seed: int,
    group_key: str = 'recipe',
) -> tuple[list[dict], list[dict]]:
    grouped_entries = defaultdict(list)
    for entry in entries:
        grouped_entries[entry.get(group_key) or 'ungrouped'].append(entry)

    rng = random.Random(seed)
    train_entries = []
    val_entries = []
    for group_name in sorted(grouped_entries):
        group_items = list(grouped_entries[group_name])
        rng.shuffle(group_items)
        val_count = int(round(len(group_items) * val_ratio))
        val_entries.extend(group_items[:val_count])
        train_entries.extend(group_items[val_count:])

    return train_entries, val_entries


def write_jsonl(path: Path, entries: list[dict]):
    path.write_text(
        ''.join(json.dumps(entry, ensure_ascii=False) + '\n' for entry in entries),
        encoding='utf-8',
    )


def _counter_to_dict(counter: Counter) -> dict:
    return {key: counter[key] for key in sorted(counter)}


def main():
    parser = argparse.ArgumentParser(description='Convert assembly episode JSONs into train/val JSONL splits.')
    parser.add_argument(
        '--input-dir',
        type=str,
        default=str(Path(__file__).resolve().parent / 'outputs' / 'factory_dual_franka_assembly'),
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Path(__file__).resolve().parent / 'outputs' / 'factory_dual_franka_assembly_dataset'),
    )
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--include-failures', action='store_true')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = load_episode_payloads(input_dir=input_dir)
    entries = build_dataset_entries(episodes=episodes, success_only=not args.include_failures)
    train_entries, val_entries = split_entries(entries=entries, val_ratio=args.val_ratio, seed=args.split_seed)

    write_jsonl(output_dir / 'train.jsonl', train_entries)
    write_jsonl(output_dir / 'val.jsonl', val_entries)

    recipe_counter = Counter(entry['recipe'] for entry in entries)
    scene_profile_counter = Counter((entry.get('scene_profile') or 'raw') for entry in entries)
    recipe_profile_counter = Counter(
        f"{entry.get('scene_profile') or 'raw'}::{entry['recipe']}"
        for entry in entries
    )

    summary = {
        'input_dir': str(input_dir),
        'num_episodes': len(episodes),
        'num_entries': len(entries),
        'num_train': len(train_entries),
        'num_val': len(val_entries),
        'success_only': not args.include_failures,
        'split_grouping': 'recipe',
        'recipes': sorted({entry['recipe'] for entry in entries}),
        'scene_profiles': sorted({(entry.get('scene_profile') or 'raw') for entry in entries}),
        'counts_by_recipe': _counter_to_dict(recipe_counter),
        'counts_by_scene_profile': _counter_to_dict(scene_profile_counter),
        'counts_by_recipe_scene_profile': _counter_to_dict(recipe_profile_counter),
    }
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
