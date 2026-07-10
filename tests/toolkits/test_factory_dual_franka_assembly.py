import json
from types import SimpleNamespace

import numpy as np
import pytest

from toolkits.factory_dual_franka_assembly.convert_dataset import (
    build_dataset_entries,
    load_episode_payloads,
    split_entries,
)
from toolkits.factory_dual_franka_assembly.export_lerobot import export_lerobot_dataset
from toolkits.factory_dual_franka_assembly.demo_policy import DualFrankaAssemblyDemoPolicy
from toolkits.factory_dual_franka_assembly.plumbers_block_ur5e_skills import (
    UR5ePlumbersBlockAtomicSkillAdapter,
)
from toolkits.factory_dual_franka_assembly.scene_builder import (
    build_dual_franka_assembly_episode,
)
from toolkits.factory_dual_franka_assembly.task_specs import list_task_recipes


def test_idle_arm_clearance_accounts_for_raised_robot_base_and_gripper_envelope():
    policy = DualFrankaAssemblyDemoPolicy()
    task = SimpleNamespace(
        config=SimpleNamespace(
            robot_names=('franka_left', 'franka_right'),
            robot_metadata=[
                {'name': 'franka_left', 'position': [0.50, 0.30, 0.998]},
                {'name': 'franka_right', 'position': [0.58, -0.80, 0.998]},
            ]
        ),
        robots={},
        is_local_skill_complete=lambda robot_name, skill_name: False,
    )
    phase_spec = {
        'name': 'left_transport_payload',
        'robot_targets': {},
        'gripper_commands': {'franka_right': 'open'},
        'local_skill': {'name': 'ur5e_move_part_to_staging', 'robot': 'franka_left'},
    }
    tracked_robots = {
        'franka_left': {
            'position': [0.60, -0.15, 1.25],
            'orientation': [1.0, 0.0, 0.0, 0.0],
        },
        'franka_right': {
            'position': [0.699, -0.397, 1.114],
            'orientation': [1.0, 0.0, 0.0, 0.0],
            'gripper_opening': 1.0,
        },
    }

    clearance_pose = policy._idle_clearance_pose(
        task=task,
        robot_name='franka_right',
        phase_spec=phase_spec,
        tracked_robots=tracked_robots,
        tracked_objects={},
    )

    assert clearance_pose is not None
    assert clearance_pose['position'][2] > tracked_robots['franka_right']['position'][2]
    assert np.linalg.norm(
        clearance_pose['position'] - np.asarray(tracked_robots['franka_right']['position'], dtype=float)
    ) <= policy._MAX_POSITION_STEP['retreat'] + 1e-9


def test_transport_completion_requires_carried_object_pose_when_enabled():
    completions = []
    task = SimpleNamespace(
        phase_step_counter=20,
        target_poses={
            'assembly_target': {
                'position': np.array([0.7, -0.2, 1.1]),
                'orientation': np.array([1.0, 0.0, 0.0, 0.0]),
            }
        },
        mark_local_skill_complete=lambda **kwargs: completions.append(kwargs),
    )
    adapter = UR5ePlumbersBlockAtomicSkillAdapter({})
    spec = {
        'object': 'part',
        'target_object_target': 'assembly_target',
        'position_tolerance': 0.01,
        'orientation_tolerance': 0.1,
        'require_target_object_pose_convergence': True,
        'target_object_position_tolerance': 0.012,
        'target_object_orientation_tolerance': 0.12,
    }
    target_pose = {
        'position': np.array([0.8, -0.2, 1.2]),
        'orientation': np.array([1.0, 0.0, 0.0, 0.0]),
    }
    tracked_objects = {
        'part': {
            'position': [0.73, -0.2, 1.1],
            'orientation': [1.0, 0.0, 0.0, 0.0],
        }
    }

    adapter._maybe_mark_complete(
        task=task,
        robot_name='franka_left',
        skill_name='ur5e_move_part_to_staging',
        spec=spec,
        target_pose=target_pose,
        ik_target_pose=target_pose,
        current_pose=target_pose,
        tracked_objects=tracked_objects,
        current_q=None,
        target_q=None,
    )
    assert completions == []

    tracked_objects['part']['position'] = [0.705, -0.2, 1.1]
    adapter._maybe_mark_complete(
        task=task,
        robot_name='franka_left',
        skill_name='ur5e_move_part_to_staging',
        spec=spec,
        target_pose=target_pose,
        ik_target_pose=target_pose,
        current_pose=target_pose,
        tracked_objects=tracked_objects,
        current_q=None,
        target_q=None,
    )
    assert len(completions) == 1
    assert completions[0]['detail']['target_object_pose_complete'] is True


def test_task_specs_are_discoverable():
    recipes = list_task_recipes()
    assert 'screw_fastening' in recipes
    assert 'peg_insertion' in recipes
    assert 'panel_alignment' in recipes
    assert 'bracket_latching' in recipes


def test_scene_builder_builds_screw_fastening_episode():
    task_cfg = build_dual_franka_assembly_episode(
        recipe='screw_fastening', seed=3, episode_idx=0, attach_runtime_cameras=True
    )
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
        'observation.images.isaac_3d',
    }
    assert len(task_cfg.robots[0].sensors) == 2
    assert len(task_cfg.robots[1].sensors) == 1


def test_scene_builder_builds_peg_insertion_episode():
    task_cfg = build_dual_franka_assembly_episode(recipe='peg_insertion', seed=7, episode_idx=1)
    assert task_cfg.recipe == 'peg_insertion'
    assert 'peg_pick' in task_cfg.target_poses
    assert 'peg_insert' in task_cfg.target_poses
    assert len(task_cfg.success_criteria) == 1
    assert len(task_cfg.camera_metadata) == 3


def test_peg_insertion_uses_contact_gated_physical_joint_and_release_success():
    task_cfg = build_dual_franka_assembly_episode(recipe='peg_insertion', seed=7, episode_idx=1)
    attach_specs = [
        attach_spec
        for phase_spec in task_cfg.phase_specs
        for attach_spec in phase_spec.get('attach', [])
        if isinstance(attach_spec, dict)
    ]
    assert attach_specs
    assert {attach_spec['attachment_mode'] for attach_spec in attach_specs} == {'physical_joint'}
    assert all(attach_spec.get('require_physical_contact') for attach_spec in attach_specs)
    assert all(not phase_spec.get('lock') for phase_spec in task_cfg.phase_specs)
    assert all(success_spec.get('require_released') for success_spec in task_cfg.success_criteria)
    assert all(success_spec.get('require_static') for success_spec in task_cfg.success_criteria)
    assert {'static_friction', 'dynamic_friction', 'restitution'}.issubset(
        set(next(obj for obj in task_cfg.object_metadata if obj['name'] == 'workbench'))
    )


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
