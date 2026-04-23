from __future__ import annotations

import json
import pickle

from toolkits.robofactory_compat.export_dataset import (
    build_episode_export,
    build_export_manifest,
    discover_agent_names,
    export_pickle_tree,
    load_episode_payloads,
)


def _sample_episode() -> dict:
    return {
        'episode_idx': 0,
        'recipe': 'demo_recipe',
        'source_benchmark': 'demo_benchmark',
        'prompt': 'demo prompt',
        'steps': [
            {
                'phase': 'reach',
                'observations': {
                    'robot_a': {'state': [1, 2, 3]},
                    'robot_b': {'state': [4, 5, 6]},
                },
                'actions': {
                    'robot_a': {'joint_action': [0.1, 0.2], 'endpose': [0.3, 0.4]},
                    'robot_b': {'joint_action': [0.5, 0.6], 'endpose': [0.7, 0.8]},
                },
                'objects': {'cube': {'position': [0, 0, 0]}},
            },
            {
                'phase': 'place',
                'observations': {
                    'robot_a': {'state': [7, 8, 9]},
                    'robot_b': {'state': [10, 11, 12]},
                },
                'actions': {
                    'robot_a': {'joint_action': [0.9, 1.0], 'endpose': [1.1, 1.2]},
                    'robot_b': {'joint_action': [1.3, 1.4], 'endpose': [1.5, 1.6]},
                },
                'objects': {'cube': {'position': [1, 0, 0]}},
            },
        ],
    }


def test_discover_agent_names_reads_steps():
    assert discover_agent_names(_sample_episode()) == ['robot_a', 'robot_b']


def test_build_episode_export_preserves_phase_metadata():
    export = build_episode_export(_sample_episode())

    assert export['episode_idx'] == 0
    assert export['metadata']['phase_trace'] == ['reach', 'place']
    assert export['metadata']['phase_segments'][0]['phase'] == 'reach'
    assert export['agent_exports'][0]['agent_name'] == 'robot_a'
    assert export['agent_exports'][1]['steps'][1]['endpose'] == [1.5, 1.6]


def test_build_export_manifest_uses_source_benchmark_as_task_name():
    manifest = build_export_manifest([_sample_episode()])

    assert manifest['task_name'] == 'demo_benchmark'
    assert manifest['task_slug'] == 'demo_benchmark'
    assert manifest['agent_names'] == ['robot_a', 'robot_b']
    assert manifest['counts_by_recipe'] == {'demo_recipe': 1}


def test_export_pickle_tree_writes_agent_episode_directories(tmp_path):
    output_dir = tmp_path / 'compat_export'
    manifest = export_pickle_tree([_sample_episode()], output_dir)

    assert manifest['pickle_tree']['agents'][0]['agent_name'] == 'robot_a'
    pkl_root = output_dir / 'pkl_data'
    assert (pkl_root / 'demo_benchmark_Agent0' / 'episode0' / '0.pkl').exists()
    assert (pkl_root / 'demo_benchmark_Agent1' / 'episode0' / '1.pkl').exists()

    with (pkl_root / 'demo_benchmark_Agent0' / 'episode0' / '0.pkl').open('rb') as handle:
        step = pickle.load(handle)
    assert step['metadata']['phase'] == 'reach'
    assert step['joint_action'] == [0.1, 0.2]

    manifest_json = json.loads((pkl_root / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest_json['task_slug'] == 'demo_benchmark'


def test_load_episode_payloads_discovers_episode_jsons(tmp_path):
    episode_dir = tmp_path / 'episodes'
    episode_dir.mkdir()
    payload = _sample_episode()
    (episode_dir / 'episode_0000.json').write_text(json.dumps(payload), encoding='utf-8')

    loaded = load_episode_payloads(episode_dir)

    assert len(loaded) == 1
    assert loaded[0]['source_path'].endswith('episode_0000.json')
    assert loaded[0]['prompt'] == 'demo prompt'
