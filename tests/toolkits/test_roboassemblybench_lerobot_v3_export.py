import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('lerobot')

from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset

from roboassemblybench.datasets.cartesian_episode import (
    CAMERA_KEYS,
    CompactCartesianEpisodeRecorder,
)
from roboassemblybench.scripts.export_fabrica_plumbers_block_lerobot_v3 import (
    export_dataset,
)


def _task():
    cameras = [
        {'name': 'front', 'owner': 'franka_left', 'view_type': 'front'},
        {'name': 'left_wrist', 'owner': 'franka_left', 'robot': 'franka_left', 'view_type': 'wrist'},
        {'name': 'right_wrist', 'owner': 'franka_right', 'robot': 'franka_right', 'view_type': 'wrist'},
    ]
    robot_config = SimpleNamespace(gripper_open_position=0.0, gripper_closed_position=0.8)
    return SimpleNamespace(
        step_counter=0,
        phase_index=0,
        phase_step_counter=0,
        phase='test',
        robots={
            'franka_left': SimpleNamespace(config=robot_config),
            'franka_right': SimpleNamespace(config=robot_config),
        },
        config=SimpleNamespace(
            camera_metadata=cameras,
            seed=11,
            recipe='test_recipe',
            task_description='test dual arm assembly',
            prompt='test dual arm assembly',
            domain_randomization={'enabled': True, 'groups': {}},
        ),
    )


def _observation(frame_index: int):
    front = np.full((36, 64, 4), 30 + frame_index, dtype=np.uint8)
    wrist = np.full((48, 64, 4), 90 + frame_index, dtype=np.uint8)
    robot_obs = {
        'eef_position': np.asarray([0.1 + frame_index * 0.01, 0.0, 1.0], dtype=np.float32),
        'eef_orientation': np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        'controllers': {'gripper_controller': {'gripper_pos': [0.0]}},
    }
    return {
        'franka_left': {
            **robot_obs,
            'sensors': {'front': {'rgba': front}, 'left_wrist': {'rgba': wrist}},
        },
        'franka_right': {
            **robot_obs,
            'sensors': {'right_wrist': {'rgba': wrist}},
        },
    }


def test_compact_episode_exports_and_resumes_as_lerobot_v3(tmp_path):
    raw_dir = tmp_path / 'raw'
    task = _task()
    recorder = CompactCartesianEpisodeRecorder(
        output_dir=raw_dir,
        episode_idx=0,
        task=task,
        fps=30,
        frame_stride=1,
        output_resolution=(64, 48),
    )
    actions = {
        'franka_left': {'gripper_controller': [1.0]},
        'franka_right': {'gripper_controller': [1.0]},
    }
    for frame_index in range(4):
        task.step_counter = frame_index
        assert recorder.record(task=task, obs=_observation(frame_index), actions=actions)
    recorder.finalize(task=task, metrics={'success': True, 'terminal_reason': 'test-success'})

    dataset_dir = tmp_path / 'lerobot_v3'
    summary = export_dataset(
        input_dir=raw_dir,
        output_dir=dataset_dir,
        repo_id='test/roboassemblybench-v3',
        max_episodes=None,
        resume=False,
        streaming_encoding=True,
        encoder_threads=1,
        vcodec='h264',
    )
    assert summary['codebase_version'] == 'v3.0'
    assert summary['added_episodes'] == 1
    assert summary['added_frames'] == 4

    resumed = export_dataset(
        input_dir=raw_dir,
        output_dir=dataset_dir,
        repo_id='test/roboassemblybench-v3',
        max_episodes=None,
        resume=True,
        streaming_encoding=True,
        encoder_threads=1,
        vcodec='h264',
    )
    assert resumed['added_episodes'] == 0
    assert resumed['total_episodes'] == 1

    assert CODEBASE_VERSION == 'v3.0'
    dataset = LeRobotDataset(
        repo_id='test/roboassemblybench-v3',
        root=dataset_dir,
        video_backend='pyav',
    )
    assert dataset.meta.total_episodes == 1
    assert dataset.meta.total_frames == 4
    assert set(dataset.meta.video_keys) == set(CAMERA_KEYS)
    sample = dataset[0]
    assert tuple(sample['observation.state'].shape) == (16,)
    assert tuple(sample['action'].shape) == (16,)
    assert {tuple(sample[key].shape) for key in CAMERA_KEYS} == {(3, 48, 64)}

    manifest = json.loads((dataset_dir / 'roboassemblybench_conversion_manifest.json').read_text(encoding='utf-8'))
    assert manifest['total_episodes'] == 1
