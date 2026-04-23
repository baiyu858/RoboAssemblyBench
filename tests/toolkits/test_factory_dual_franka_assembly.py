import json

import pytest

from toolkits.factory_dual_franka_assembly.convert_dataset import (
    build_dataset_entries,
    load_episode_payloads,
    split_entries,
)
from toolkits.factory_dual_franka_assembly.export_lerobot import export_lerobot_dataset
from toolkits.factory_dual_franka_assembly.scene_builder import (
    build_dual_franka_assembly_episode,
)
from toolkits.factory_dual_franka_assembly.task_specs import list_task_recipes


def test_task_specs_are_discoverable():
    recipes = list_task_recipes()
    assert 'screw_fastening' in recipes
    assert 'peg_insertion' in recipes
    assert 'panel_alignment' in recipes
    assert 'bracket_latching' in recipes


def test_scene_builder_builds_screw_fastening_episode():
    task_cfg = build_dual_franka_assembly_episode(recipe='screw_fastening', seed=3, episode_idx=0)
    assert task_cfg.recipe == 'screw_fastening'
    assert task_cfg.robot_names == ('franka_left', 'franka_right')
    assert 'part_pick' in task_cfg.target_poses
    assert 'screw_insert' in task_cfg.target_poses
    assert len(task_cfg.phase_specs) >= 8
    assert {'assembly_part', 'screw'}.issubset(set(task_cfg.tracked_object_names))
    assert len(task_cfg.camera_metadata) == 3
    assert {camera['video_key'] for camera in task_cfg.camera_metadata} == {
        'observation.images.left_wrist',
        'observation.images.right_wrist',
        'observation.images.third_person',
    }
    assert len(task_cfg.robots[0].sensors) == 2
    assert len(task_cfg.robots[1].sensors) == 1


def test_scene_builder_builds_peg_insertion_episode():
    task_cfg = build_dual_franka_assembly_episode(recipe='peg_insertion', seed=7, episode_idx=1)
    assert task_cfg.recipe == 'peg_insertion'
    assert 'housing_hold' in task_cfg.target_poses
    assert 'peg_insert' in task_cfg.target_poses
    assert len(task_cfg.success_criteria) == 2
    assert len(task_cfg.camera_metadata) == 3


@pytest.mark.parametrize(
    ('recipe', 'required_target_names', 'required_object_names'),
    [
        ('panel_alignment', {'panel_fixture_hold', 'pin_insert'}, {'panel', 'locating_pin'}),
        ('bracket_latching', {'bracket_fixture_hold', 'latch_insert'}, {'bracket', 'latch'}),
    ],
)
def test_scene_builder_builds_additional_dual_franka_episodes(
    recipe,
    required_target_names,
    required_object_names,
):
    task_cfg = build_dual_franka_assembly_episode(recipe=recipe, seed=13, episode_idx=2)
    assert task_cfg.recipe == recipe
    assert required_target_names.issubset(task_cfg.target_poses)
    assert required_object_names.issubset(set(task_cfg.tracked_object_names))
    assert len(task_cfg.phase_specs) >= 8
    assert len(task_cfg.success_criteria) == 2


def test_convert_dataset_pipeline(tmp_path):
    recipe_dir = tmp_path / 'screw_fastening'
    recipe_dir.mkdir(parents=True)
    episode_payload = {
        'episode_idx': 0,
        'seed': 11,
        'recipe': 'screw_fastening',
        'prompt': 'demo prompt',
        'metrics': {'success': True},
        'steps': [
            {'phase': 'left_approach_part', 'observations': {'a': 1}, 'actions': {'b': 2}, 'objects': {'c': 3}},
            {'phase': 'retreat', 'observations': {'d': 4}, 'actions': {'e': 5}, 'objects': {'f': 6}},
        ],
    }
    (recipe_dir / 'episode_0000.json').write_text(json.dumps(episode_payload), encoding='utf-8')

    episodes = load_episode_payloads(tmp_path)
    entries = build_dataset_entries(episodes)
    train_entries, val_entries = split_entries(entries, val_ratio=0.5, seed=0)

    assert len(episodes) == 1
    assert len(entries) == 1
    assert len(train_entries) + len(val_entries) == 1
    assert entries[0]['episode_id'] == 'screw_fastening_0000'
    assert entries[0]['phase_trace'] == ['left_approach_part', 'retreat']


def test_lerobot_export_supports_multi_view_video_keys(tmp_path, monkeypatch):
    input_dir = tmp_path / 'input'
    episode_dir = input_dir / 'proxy_factory_cell' / 'screw_fastening'
    episode_dir.mkdir(parents=True)
    episode_payload = {
        'episode_idx': 0,
        'seed': 11,
        'recipe': 'screw_fastening',
        'prompt': 'demo prompt',
        'task_description': 'demo task',
        'scene_profile': 'proxy_factory_cell',
        'metrics': {'success': True},
        'camera_metadata': [
            {
                'name': 'left_wrist_camera',
                'owner': 'franka_left',
                'video_key': 'observation.images.left_wrist',
            },
            {
                'name': 'right_wrist_camera',
                'owner': 'franka_right',
                'video_key': 'observation.images.right_wrist',
            },
            {
                'name': 'third_person_camera',
                'owner': 'franka_left',
                'video_key': 'observation.images.third_person',
            },
        ],
        'steps': [
            {
                'phase': 'left_approach_part',
                'observations': {
                    'franka_left': {
                        'position': [0.0, 0.0, 0.0],
                        'orientation': [1.0, 0.0, 0.0, 0.0],
                        'eef_position': [0.0, 0.0, 0.0],
                        'eef_orientation': [1.0, 0.0, 0.0, 0.0],
                        'joint_action': [],
                    },
                    'franka_right': {
                        'position': [0.0, 0.0, 0.0],
                        'orientation': [1.0, 0.0, 0.0, 0.0],
                        'eef_position': [0.0, 0.0, 0.0],
                        'eef_orientation': [1.0, 0.0, 0.0, 0.0],
                        'joint_action': [],
                    },
                },
                'actions': {'franka_left': {}, 'franka_right': {}},
                'objects': {},
            }
        ],
    }
    (episode_dir / 'episode_0000.json').write_text(json.dumps(episode_payload), encoding='utf-8')

    rendered = []

    def fake_render_episode_video(*, episode, output_path, fps, width, height, video_mode, keep_video_frames=False, camera_video_key=None):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b'fake-mp4')
        rendered.append((output_path, camera_video_key))
        return {
            'image_stats': {'min': [0], 'max': [0], 'mean': [0], 'std': [0], 'count': [1]},
            'video_path': str(output_path),
            'width': 8,
            'height': 6,
            'fps': fps,
            'renderer': 'stub',
        }

    monkeypatch.setattr('toolkits.factory_dual_franka_assembly.export_lerobot._render_episode_video', fake_render_episode_video)

    summary = export_lerobot_dataset(
        input_dir=input_dir,
        output_dir=tmp_path / 'output',
        include_failures=False,
        fps=10,
        video_width=8,
        video_height=6,
        chunk_size=1000,
        video_key='observation.images.isaac_3d',
        video_mode='isaac_replay',
    )

    info = json.loads((tmp_path / 'output' / 'meta' / 'info.json').read_text(encoding='utf-8'))
    episode_row = json.loads((tmp_path / 'output' / 'meta' / 'episodes.jsonl').read_text(encoding='utf-8').splitlines()[0])

    assert summary['total_video_streams'] == 3
    assert summary['video_keys'] == [
        'observation.images.front',
        'observation.images.left_wrist',
        'observation.images.right_wrist',
    ]
    assert len(rendered) == 3
    assert set(info['features']) >= {
        'observation.images.front',
        'observation.images.left_wrist',
        'observation.images.right_wrist',
    }
    assert set(episode_row['video_paths']) == {
        'observation.images.front',
        'observation.images.left_wrist',
        'observation.images.right_wrist',
    }


def test_lerobot_export_defaults_to_front_and_dual_wrist_views_when_camera_metadata_missing(tmp_path, monkeypatch):
    input_dir = tmp_path / 'input'
    episode_dir = input_dir / 'proxy_factory_cell' / 'peg_insertion'
    episode_dir.mkdir(parents=True)
    episode_payload = {
        'episode_idx': 0,
        'seed': 7,
        'recipe': 'peg_insertion',
        'prompt': 'demo prompt',
        'task_description': 'demo task',
        'scene_profile': 'proxy_factory_cell',
        'metrics': {'success': True},
        'camera_metadata': [],
        'steps': [
            {
                'phase': 'left_wait',
                'observations': {
                    'franka_left': {
                        'position': [0.0, 0.0, 0.0],
                        'orientation': [1.0, 0.0, 0.0, 0.0],
                        'eef_position': [0.0, 0.0, 0.0],
                        'eef_orientation': [1.0, 0.0, 0.0, 0.0],
                        'joint_action': [],
                    },
                    'franka_right': {
                        'position': [0.0, 0.0, 0.0],
                        'orientation': [1.0, 0.0, 0.0, 0.0],
                        'eef_position': [0.0, 0.0, 0.0],
                        'eef_orientation': [1.0, 0.0, 0.0, 0.0],
                        'joint_action': [],
                    },
                },
                'actions': {'franka_left': {}, 'franka_right': {}},
                'objects': {},
            }
        ],
    }
    (episode_dir / 'episode_0000.json').write_text(json.dumps(episode_payload), encoding='utf-8')

    rendered = []

    def fake_render_episode_video(*, episode, output_path, fps, width, height, video_mode, keep_video_frames=False, camera_video_key=None):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b'fake-mp4')
        rendered.append(camera_video_key)
        return {
            'image_stats': {'min': [0], 'max': [0], 'mean': [0], 'std': [0], 'count': [1]},
            'video_path': str(output_path),
            'width': 8,
            'height': 6,
            'fps': fps,
            'renderer': 'stub',
        }

    monkeypatch.setattr('toolkits.factory_dual_franka_assembly.export_lerobot._render_episode_video', fake_render_episode_video)

    summary = export_lerobot_dataset(
        input_dir=input_dir,
        output_dir=tmp_path / 'output',
        include_failures=False,
        fps=10,
        video_width=8,
        video_height=6,
        chunk_size=1000,
        video_key='observation.images.isaac_3d',
        video_mode='isaac_replay',
    )

    assert summary['video_keys'] == [
        'observation.images.front',
        'observation.images.left_wrist',
        'observation.images.right_wrist',
    ]
    assert rendered == [
        'observation.images.front',
        'observation.images.left_wrist',
        'observation.images.right_wrist',
    ]
