from __future__ import annotations

import argparse
import copy
import json
import pickle
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from toolkits.robofactory_compat.config_adapter import slugify_task_name

try:
    import zarr
    from numcodecs import Blosc
except ImportError:  # pragma: no cover - optional dependency
    zarr = None
    Blosc = None


DEFAULT_EXPORT_ROOT = Path(__file__).resolve().parents[2] / 'toolkits' / 'robofactory_compat' / 'outputs'
DEFAULT_PKL_DIRNAME = 'pkl_data'
DEFAULT_ZARR_DIRNAME = 'zarr_data'


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, set):
        return [_to_jsonable(item) for item in sorted(value, key=lambda item: str(item))]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def collect_episode_paths(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    return sorted(root.rglob('episode_*.json'))


def load_episode_payloads(input_dir: str | Path) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for episode_path in collect_episode_paths(input_dir):
        payload = json.loads(episode_path.read_text(encoding='utf-8'))
        payload['source_path'] = str(episode_path)
        payloads.append(payload)
    return payloads


def _ordered_unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def discover_agent_names(episode: dict[str, Any]) -> list[str]:
    discovered: list[str] = []

    for top_level_key in ('agent_names', 'robot_names', 'agents'):
        raw_agents = episode.get(top_level_key)
        if isinstance(raw_agents, (list, tuple)):
            for item in raw_agents:
                if isinstance(item, str):
                    discovered.append(item)
                elif isinstance(item, dict) and isinstance(item.get('name'), str):
                    discovered.append(item['name'])

    for step in episode.get('steps', []) or []:
        for section_name in ('observations', 'actions'):
            section = step.get(section_name) or {}
            if isinstance(section, dict):
                discovered.extend(str(key) for key in section.keys())

    return _ordered_unique(discovered)


def build_phase_segments(phase_trace: list[str | None]) -> list[dict[str, Any]]:
    if not phase_trace:
        return []

    segments: list[dict[str, Any]] = []
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


def _build_episode_metadata(episode: dict[str, Any], *, source_index: int, agent_names: list[str]) -> dict[str, Any]:
    steps = episode.get('steps', []) or []
    phase_trace = [step.get('phase') for step in steps]
    base_metadata = {key: copy.deepcopy(value) for key, value in episode.items() if key != 'steps'}
    episode_idx = int(episode.get('episode_idx', source_index))
    episode_id = f'episode{episode_idx}'
    base_metadata['step_count'] = len(steps)
    base_metadata['agent_names'] = agent_names
    base_metadata['phase_trace'] = phase_trace
    base_metadata['phase_segments'] = build_phase_segments(phase_trace)
    base_metadata['episode_idx'] = episode_idx
    base_metadata['episode_id'] = episode_id
    return base_metadata


def _extract_preferred_value(payload: Any, preferred_keys: tuple[str, ...]) -> Any:
    if isinstance(payload, dict):
        for key in preferred_keys:
            if key in payload:
                return copy.deepcopy(payload[key])
    return copy.deepcopy(payload)


def _build_step_record(
    episode: dict[str, Any],
    *,
    agent_name: str,
    agent_id: int,
    step_index: int,
    step: dict[str, Any],
    episode_metadata: dict[str, Any],
) -> dict[str, Any]:
    observations = step.get('observations') or {}
    actions = step.get('actions') or {}
    step_context = {key: copy.deepcopy(value) for key, value in step.items() if key not in {'observations', 'actions'}}
    observation = copy.deepcopy(observations.get(agent_name))
    action = copy.deepcopy(actions.get(agent_name))

    return {
        'pointcloud': None,
        'observation': observation,
        'action': action,
        'joint_action': _extract_preferred_value(action, ('joint_action', 'action', 'endpose')),
        'endpose': _extract_preferred_value(action, ('endpose', 'action', 'joint_action')),
        'metadata': {
            'episode_idx': episode_metadata['episode_idx'],
            'episode_id': episode_metadata['episode_id'],
            'step_idx': step_index,
            'agent_id': agent_id,
            'agent_name': agent_name,
            'phase': step.get('phase'),
            'source_path': episode.get('source_path'),
            'step_context': step_context,
        },
    }


def build_episode_export(
    episode: dict[str, Any],
    *,
    source_index: int = 0,
    agent_names: list[str] | None = None,
) -> dict[str, Any]:
    agent_names = list(agent_names) if agent_names is not None else discover_agent_names(episode)
    episode_metadata = _build_episode_metadata(episode, source_index=source_index, agent_names=agent_names)
    steps = episode.get('steps', []) or []

    agent_exports = []
    for agent_id, agent_name in enumerate(agent_names):
        agent_steps = [
            _build_step_record(
                episode,
                agent_name=agent_name,
                agent_id=agent_id,
                step_index=step_index,
                step=step,
                episode_metadata=episode_metadata,
            )
            for step_index, step in enumerate(steps)
        ]
        agent_exports.append(
            {
                'agent_id': agent_id,
                'agent_name': agent_name,
                'episode_dir': f"episode{episode_metadata['episode_idx']}",
                'steps': agent_steps,
            }
        )

    return {
        'episode_idx': episode_metadata['episode_idx'],
        'episode_id': episode_metadata['episode_id'],
        'source_path': episode.get('source_path'),
        'metadata': episode_metadata,
        'agent_exports': agent_exports,
    }


def build_export_manifest(episodes: list[dict[str, Any]], *, task_name: str | None = None) -> dict[str, Any]:
    all_agent_names = _ordered_unique(
        [agent_name for episode in episodes for agent_name in discover_agent_names(episode)]
    )
    exports = [
        build_episode_export(episode, source_index=index, agent_names=all_agent_names)
        for index, episode in enumerate(episodes)
    ]
    recipe_counter = Counter((episode.get('recipe') or 'unknown') for episode in episodes)
    scene_profile_counter = Counter((episode.get('scene_profile') or 'raw') for episode in episodes)

    resolved_task_name = task_name
    if resolved_task_name is None:
        resolved_task_name = episodes[0].get('source_benchmark') if episodes else None
    resolved_task_name = resolved_task_name or 'internutopia_episodes'
    task_slug = slugify_task_name(resolved_task_name, 'internutopia_episodes')

    return {
        'schema_version': 'robofactory-episode-export/v1',
        'task_name': resolved_task_name,
        'task_slug': task_slug,
        'input_episode_count': len(episodes),
        'agent_names': all_agent_names,
        'counts_by_recipe': {key: recipe_counter[key] for key in sorted(recipe_counter)},
        'counts_by_scene_profile': {key: scene_profile_counter[key] for key in sorted(scene_profile_counter)},
        'episodes': exports,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False), encoding='utf-8')


def _write_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def export_pickle_tree(
    episodes: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    task_name: str | None = None,
    clean: bool = True,
) -> dict[str, Any]:
    manifest = build_export_manifest(episodes, task_name=task_name)
    output_root = Path(output_dir)
    pkl_root = output_root / DEFAULT_PKL_DIRNAME
    if clean and pkl_root.exists():
        shutil.rmtree(pkl_root)
    pkl_root.mkdir(parents=True, exist_ok=True)

    for agent_id, _agent_name in enumerate(manifest['agent_names']):
        agent_root = pkl_root / f"{manifest['task_slug']}_Agent{agent_id}"
        agent_root.mkdir(parents=True, exist_ok=True)

    agent_summary: dict[str, dict[str, Any]] = {}
    for episode_export in manifest['episodes']:
        episode_idx = episode_export['episode_idx']
        episode_metadata = episode_export['metadata']
        episode_dir_name = f'episode{episode_idx}'
        for agent_export in episode_export['agent_exports']:
            agent_id = agent_export['agent_id']
            agent_name = agent_export['agent_name']
            agent_root = pkl_root / f"{manifest['task_slug']}_Agent{agent_id}"
            episode_dir = agent_root / episode_dir_name
            episode_dir.mkdir(parents=True, exist_ok=True)

            for step_index, step_record in enumerate(agent_export['steps']):
                _write_pickle(episode_dir / f'{step_index}.pkl', step_record)

            _write_json(episode_dir / 'manifest.json', episode_metadata)

            summary = agent_summary.setdefault(
                str(agent_id),
                {
                    'agent_id': agent_id,
                    'agent_name': agent_name,
                    'episode_dirs': [],
                    'step_count': 0,
                },
            )
            summary['episode_dirs'].append(str(episode_dir.relative_to(pkl_root)))
            summary['step_count'] += len(agent_export['steps'])

    manifest['pickle_tree'] = {
        'root': str(pkl_root),
        'agents': [agent_summary[key] for key in sorted(agent_summary, key=int)],
    }
    _write_json(pkl_root / 'manifest.json', manifest)
    _write_json(output_root / 'manifest.json', manifest)
    return manifest


def _extract_camera_rgb(observation: Any) -> np.ndarray | None:
    if not isinstance(observation, dict):
        return None

    direct_candidates = [
        observation.get('head_camera'),
        observation.get('camera'),
        observation.get('rgb_camera'),
    ]
    for candidate in direct_candidates:
        if isinstance(candidate, dict) and 'rgb' in candidate:
            rgb = np.asarray(candidate['rgb'])
            if rgb.ndim == 3:
                return rgb

    def _search(node: Any) -> np.ndarray | None:
        if isinstance(node, dict):
            if 'rgb' in node:
                rgb = np.asarray(node['rgb'])
                if rgb.ndim == 3:
                    return rgb
            for key, value in node.items():
                if isinstance(key, str) and 'camera' in key.lower():
                    found = _search(value)
                    if found is not None:
                        return found
                if isinstance(value, (dict, list, tuple)):
                    found = _search(value)
                    if found is not None:
                        return found
        elif isinstance(node, (list, tuple)):
            for item in node:
                found = _search(item)
                if found is not None:
                    return found
        return None

    return _search(observation)


def _extract_numeric_vector(payload: Any) -> np.ndarray | None:
    if payload is None:
        return None
    array = np.asarray(payload)
    if array.dtype.kind not in {'b', 'i', 'u', 'f'}:
        return None
    if array.ndim == 0:
        return array.reshape(1).astype(np.float32)
    if array.ndim == 1:
        return array.astype(np.float32)
    return None


def export_zarr_tree(
    episodes: list[dict[str, Any]],
    output_dir: str | Path,
    *,
    task_name: str | None = None,
    clean: bool = True,
) -> dict[str, Any]:
    if zarr is None or Blosc is None:  # pragma: no cover - optional dependency
        raise RuntimeError('zarr is not installed; cannot write zarr output')

    manifest = build_export_manifest(episodes, task_name=task_name)
    output_root = Path(output_dir)
    zarr_root = output_root / DEFAULT_ZARR_DIRNAME
    if clean and zarr_root.exists():
        shutil.rmtree(zarr_root)
    zarr_root.mkdir(parents=True, exist_ok=True)

    compressor = Blosc(cname='zstd', clevel=3, shuffle=1)
    for agent_index, agent_name in enumerate(manifest['agent_names']):
        agent_store = zarr_root / f"{manifest['task_slug']}_Agent{agent_index}_{len(manifest['episodes'])}.zarr"
        root = zarr.open_group(str(agent_store), mode='w')
        data_group = root.create_group('data')
        meta_group = root.create_group('meta')

        step_json: list[str] = []
        episode_ends: list[int] = []
        head_cameras: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        states: list[np.ndarray] = []
        step_count = 0

        for episode_export in manifest['episodes']:
            agent_exports = episode_export['agent_exports']
            agent_export = next(item for item in agent_exports if item['agent_name'] == agent_name)
            for step_record in agent_export['steps']:
                step_json.append(json.dumps(_to_jsonable(step_record), ensure_ascii=False))
                observation = step_record.get('observation')
                rgb = _extract_camera_rgb(observation)
                if rgb is not None:
                    head_cameras.append(np.moveaxis(rgb.astype(np.uint8), -1, 0))

                action_vector = _extract_numeric_vector(step_record.get('endpose'))
                state_vector = _extract_numeric_vector(step_record.get('joint_action'))
                if action_vector is not None:
                    actions.append(action_vector)
                if state_vector is not None:
                    states.append(state_vector)
                step_count += 1
            episode_ends.append(step_count)

        meta_group.attrs['task_name'] = manifest['task_name']
        meta_group.attrs['task_slug'] = manifest['task_slug']
        meta_group.attrs['agent_name'] = agent_name
        meta_group.attrs['agent_index'] = agent_index
        meta_group.attrs['episode_ends'] = episode_ends
        meta_group.attrs['step_json'] = step_json
        meta_group.attrs['manifest'] = {
            'task_name': manifest['task_name'],
            'task_slug': manifest['task_slug'],
            'agent_name': agent_name,
            'agent_index': agent_index,
            'step_count': step_count,
            'episode_ends': episode_ends,
        }

        meta_group.create_dataset('episode_ends', data=np.asarray(episode_ends, dtype=np.int64), overwrite=True)

        if head_cameras and len(head_cameras) == len(step_json):
            head_camera_array = np.asarray(head_cameras, dtype=np.uint8)
            data_group.create_dataset(
                'head_camera',
                data=head_camera_array,
                chunks=(min(100, len(head_cameras)), *head_camera_array.shape[1:]),
                overwrite=True,
                compressor=compressor,
            )

        if actions and len(actions) == len(step_json):
            action_array = np.asarray(actions, dtype=np.float32)
            if action_array.ndim == 1:
                action_array = action_array[:, None]
            data_group.create_dataset(
                'tcp_action',
                data=action_array,
                chunks=(min(100, len(actions)), *action_array.shape[1:]),
                overwrite=True,
                compressor=compressor,
            )
            data_group.create_dataset(
                'action',
                data=action_array,
                chunks=(min(100, len(actions)), *action_array.shape[1:]),
                overwrite=True,
                compressor=compressor,
            )

        if states and len(states) == len(step_json):
            state_array = np.asarray(states, dtype=np.float32)
            if state_array.ndim == 1:
                state_array = state_array[:, None]
            data_group.create_dataset(
                'state',
                data=state_array,
                chunks=(min(100, len(states)), *state_array.shape[1:]),
                overwrite=True,
                compressor=compressor,
            )

    _write_json(zarr_root / 'manifest.json', manifest)
    _write_json(output_root / 'manifest.json', manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Export InternUtopia episode JSONs into RoboFactory-style artifacts.')
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing episode_*.json files.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DEFAULT_EXPORT_ROOT),
        help='Directory that will receive pkl_data/ and optional zarr_data/ outputs.',
    )
    parser.add_argument(
        '--task-name',
        type=str,
        default=None,
        help='Dataset name used for the export tree. Defaults to source_benchmark or internutopia_episodes.',
    )
    parser.add_argument(
        '--zarr',
        action='store_true',
        help='Also write zarr output when the dependency is installed.',
    )
    args = parser.parse_args(argv)

    episodes = load_episode_payloads(args.input_dir)
    output_dir = Path(args.output_dir)
    export_pickle_tree(episodes, output_dir=output_dir, task_name=args.task_name)
    if args.zarr:
        export_zarr_tree(episodes, output_dir=output_dir, task_name=args.task_name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
