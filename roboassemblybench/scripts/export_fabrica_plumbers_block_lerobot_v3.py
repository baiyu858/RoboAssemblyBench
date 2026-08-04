from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from roboassemblybench.datasets.cartesian_episode import (
    ACTION_NAMES,
    ACTION_SEMANTICS,
    CAMERA_KEYS,
    STATE_NAMES,
    cartesian_trajectory_errors,
)

DEFAULT_REPO_ID = 'baiyu858/roboassemblybench_fabrica_plumbers_block_ur5e_2k'
CONVERSION_MANIFEST = 'roboassemblybench_conversion_manifest.json'
COLLECTION_MANIFEST = 'collection_manifest.json'


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'{path.suffix}.tmp')
    temporary.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    temporary.replace(path)


def _discover_successful_episodes(input_dir: Path) -> list[dict[str, Any]]:
    collection_manifest_path = input_dir / COLLECTION_MANIFEST
    metadata_paths = None
    if collection_manifest_path.is_file():
        collection_manifest = _load_json(collection_manifest_path)
        successful = collection_manifest.get('successful_episodes') or {}
        metadata_paths = [Path(item['metadata_path']) for item in successful.values()]

    episodes = []
    candidates = (
        metadata_paths if metadata_paths is not None else input_dir.rglob('episode_*_cartesian_raw/metadata.json')
    )
    for metadata_path in candidates:
        if not metadata_path.is_file():
            raise FileNotFoundError(f'Collection manifest references missing metadata: {metadata_path}.')
        metadata = _load_json(metadata_path)
        if metadata.get('schema_version') != 'roboassemblybench_raw_cartesian_v1':
            continue
        if not bool((metadata.get('metrics') or {}).get('success', False)):
            continue
        metadata['metadata_path'] = str(metadata_path.resolve())
        episodes.append(metadata)
    episodes = sorted(
        episodes,
        key=lambda item: (int(item.get('seed', -1)), str(item['metadata_path'])),
    )
    seeds = [int(item.get('seed', -1)) for item in episodes]
    if len(seeds) != len(set(seeds)):
        raise ValueError('Successful source episodes contain duplicate seeds.')
    return episodes


def _validate_timing(metadata: dict[str, Any]) -> None:
    fps = int(metadata.get('fps', 0))
    simulation_fps = int(metadata.get('simulation_fps', 0))
    frame_stride = int(metadata.get('frame_stride', 0))
    timing = metadata.get('timing') or {}
    expected = {
        'physics_fps': simulation_fps,
        'control_fps': simulation_fps,
        'dataset_fps': fps,
        'dataset_frame_stride': frame_stride,
        'rendering_interval': frame_stride - 1,
        'camera_render_period_steps': frame_stride,
    }
    try:
        matches = all(int(timing.get(key, -1)) == value for key, value in expected.items())
    except (TypeError, ValueError):
        matches = False
    if (
        fps <= 0
        or frame_stride <= 0
        or simulation_fps != fps * frame_stride
        or not matches
        or not bool(timing.get('camera_state_action_aligned', False))
    ):
        raise ValueError(f"Camera/state/action timing mismatch in {metadata['metadata_path']}.")


def _validate_metadata(metadata: dict[str, Any]) -> None:
    _validate_timing(metadata)
    if list(metadata.get('state_names') or []) != list(STATE_NAMES):
        raise ValueError(f"State schema mismatch in {metadata['metadata_path']}.")
    if list(metadata.get('action_names') or []) != list(ACTION_NAMES):
        raise ValueError(f"Action schema mismatch in {metadata['metadata_path']}.")
    if metadata.get('action_semantics') != ACTION_SEMANTICS:
        raise ValueError(f"Action semantics mismatch in {metadata['metadata_path']}.")
    videos = metadata.get('videos') or {}
    missing = [key for key in CAMERA_KEYS if not videos.get(key) or not Path(videos[key]).is_file()]
    if missing:
        raise FileNotFoundError(f"Episode {metadata['metadata_path']} is missing videos: {missing}.")
    trajectory_path = Path(metadata.get('trajectory_path') or '')
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"Episode {metadata['metadata_path']} is missing {trajectory_path}.")


def _conversion_entry(metadata: dict[str, Any], episode_index: int) -> dict[str, Any]:
    domain_randomization = metadata.get('domain_randomization') or {}
    return {
        'episode_index': int(episode_index),
        'seed': int(metadata['seed']),
        'layout_seed': int(metadata.get('layout_seed', domain_randomization.get('seed', metadata['seed']))),
        'source_metadata': str(Path(metadata['metadata_path']).resolve()),
        'frame_count': int(metadata['frame_count']),
        'domain_randomization': domain_randomization,
    }


def _reconcile_conversion_manifest(
    manifest: dict[str, Any],
    episodes: list[dict[str, Any]],
    *,
    dataset_episode_count: int,
) -> bool:
    manifest_entries = manifest.setdefault('episodes', [])
    manifest_count = len(manifest_entries)
    if manifest_count > dataset_episode_count:
        raise RuntimeError(
            f'Conversion manifest has {manifest_count} episodes but LeRobot has {dataset_episode_count}.'
        )
    if dataset_episode_count > len(episodes):
        raise RuntimeError(
            f'LeRobot has {dataset_episode_count} episodes but only {len(episodes)} authoritative sources exist.'
        )
    expected_prefix = [str(Path(metadata['metadata_path']).resolve()) for metadata in episodes[:manifest_count]]
    actual_prefix = [str(item.get('source_metadata')) for item in manifest_entries]
    if actual_prefix != expected_prefix:
        raise RuntimeError('Conversion manifest is not a prefix of the authoritative source episode order.')

    changed = False
    for episode_index in range(manifest_count, dataset_episode_count):
        manifest_entries.append(_conversion_entry(episodes[episode_index], episode_index))
        changed = True
    if changed:
        manifest['total_episodes'] = len(manifest_entries)
        manifest['total_frames'] = sum(int(item['frame_count']) for item in manifest_entries)
    return changed


def _features(first_episode: dict[str, Any]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        'observation.state': {
            'dtype': 'float32',
            'shape': (len(STATE_NAMES),),
            'names': list(STATE_NAMES),
        },
        'action': {
            'dtype': 'float32',
            'shape': (len(ACTION_NAMES),),
            'names': list(ACTION_NAMES),
        },
    }
    video_shapes = first_episode.get('video_shapes') or {}
    common_shapes = {tuple(int(value) for value in video_shapes[camera_key]) for camera_key in CAMERA_KEYS}
    if len(common_shapes) != 1:
        raise ValueError(
            'ACT requires all camera streams to have one common shape; got '
            f'{dict((key, video_shapes.get(key)) for key in CAMERA_KEYS)}.'
        )
    for camera_key in CAMERA_KEYS:
        shape = tuple(int(value) for value in video_shapes[camera_key])
        if len(shape) != 3 or shape[-1] != 3:
            raise ValueError(f'Invalid RGB shape for {camera_key}: {shape}.')
        features[camera_key] = {
            'dtype': 'video',
            'shape': shape,
            'names': ['height', 'width', 'channels'],
            'info': {'is_depth_map': False},
        }
    return features


def _open_video_captures(metadata: dict[str, Any]) -> dict[str, cv2.VideoCapture]:
    captures = {}
    for camera_key in CAMERA_KEYS:
        capture = cv2.VideoCapture(str(metadata['videos'][camera_key]))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {camera_key} video: {metadata['videos'][camera_key]}")
        captures[camera_key] = capture
    return captures


def _release_video_captures(captures: dict[str, cv2.VideoCapture]) -> None:
    for capture in captures.values():
        capture.release()


def _append_episode(dataset, metadata: dict[str, Any]) -> int:
    trajectory = np.load(metadata['trajectory_path'])
    states = np.asarray(trajectory['observation_state'], dtype=np.float32)
    actions = np.asarray(trajectory['action'], dtype=np.float32)
    expected_frames = int(metadata['frame_count'])
    if states.shape != (expected_frames, len(STATE_NAMES)):
        raise ValueError(f"Invalid state shape in {metadata['trajectory_path']}: {states.shape}.")
    if actions.shape != (expected_frames, len(ACTION_NAMES)):
        raise ValueError(f"Invalid action shape in {metadata['trajectory_path']}: {actions.shape}.")
    trajectory_errors = cartesian_trajectory_errors(
        states,
        actions,
        simulation_steps=trajectory.get('simulation_step'),
        frame_stride=int(metadata['frame_stride']),
    )
    if trajectory_errors:
        raise ValueError(f"Invalid Cartesian trajectory in {metadata['trajectory_path']}: {trajectory_errors}.")

    captures = _open_video_captures(metadata)
    try:
        for frame_index in range(expected_frames):
            frame = {
                'observation.state': states[frame_index],
                'action': actions[frame_index],
                'task': str(metadata['task']),
            }
            for camera_key, capture in captures.items():
                ok, bgr = capture.read()
                if not ok or bgr is None:
                    raise RuntimeError(
                        f'{camera_key} ended at frame {frame_index}/{expected_frames} for '
                        f"{metadata['metadata_path']}."
                    )
                frame[camera_key] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dataset.add_frame(frame)
    finally:
        _release_video_captures(captures)

    dataset.save_episode()
    return expected_frames


def export_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    repo_id: str,
    max_episodes: int | None,
    resume: bool,
    streaming_encoding: bool,
    encoder_threads: int | None,
    vcodec: str,
) -> dict[str, Any]:
    try:
        import lerobot
        from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            'LeRobot with v3 dataset support is required. Run this script in the ' 'roboassemblybench-act environment.'
        ) from exc
    if str(CODEBASE_VERSION) != 'v3.0':
        raise RuntimeError(f'Expected LeRobot dataset codebase v3.0, got {CODEBASE_VERSION!r}.')

    episodes = _discover_successful_episodes(input_dir)
    if max_episodes is not None:
        episodes = episodes[: max(int(max_episodes), 0)]
    if not episodes:
        raise RuntimeError(f'No successful compact Cartesian episodes found under {input_dir}.')
    for metadata in episodes:
        _validate_metadata(metadata)
    expected_video_shapes = episodes[0].get('video_shapes') or {}
    for metadata in episodes[1:]:
        if (metadata.get('video_shapes') or {}) != expected_video_shapes:
            raise ValueError(
                f"Camera shape mismatch between {episodes[0]['metadata_path']} and " f"{metadata['metadata_path']}."
            )

    fps_values = {int(metadata['fps']) for metadata in episodes}
    if len(fps_values) != 1:
        raise ValueError(f'All source episodes must use one FPS, got {sorted(fps_values)}.')
    fps = fps_values.pop()

    manifest_path = output_dir / CONVERSION_MANIFEST
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
    else:
        manifest = {
            'schema_version': 'roboassemblybench_lerobot_v3_conversion_v1',
            'repo_id': repo_id,
            'input_dir': str(input_dir.resolve()),
            'output_dir': str(output_dir.resolve()),
            'episodes': [],
        }
    if manifest.get('repo_id') != repo_id:
        raise ValueError(
            f"Existing conversion manifest repo_id={manifest.get('repo_id')!r} does not match {repo_id!r}."
        )
    has_dataset = (output_dir / 'meta' / 'info.json').is_file()
    if has_dataset:
        if not resume:
            raise FileExistsError(f'LeRobot dataset already exists at {output_dir}; pass --resume.')
        resume_method = getattr(LeRobotDataset, 'resume', None)
        if callable(resume_method):
            dataset = resume_method(
                repo_id,
                root=output_dir,
                streaming_encoding=streaming_encoding,
                encoder_threads=encoder_threads,
            )
        else:
            resume_kwargs = {
                'repo_id': repo_id,
                'root': output_dir,
                'streaming_encoding': streaming_encoding,
                'encoder_threads': encoder_threads,
            }
            if 'vcodec' in inspect.signature(LeRobotDataset).parameters:
                resume_kwargs['vcodec'] = vcodec
            dataset = LeRobotDataset(**resume_kwargs)
    else:
        create_kwargs = {
            'repo_id': repo_id,
            'root': output_dir,
            'fps': fps,
            'robot_type': 'dual_ur5e_robotiq_2f85',
            'features': _features(episodes[0]),
            'use_videos': True,
            'streaming_encoding': streaming_encoding,
            'encoder_threads': encoder_threads,
            'metadata_buffer_size': 10,
        }
        if 'vcodec' in inspect.signature(LeRobotDataset.create).parameters:
            create_kwargs['vcodec'] = vcodec
        dataset = LeRobotDataset.create(
            **create_kwargs,
        )

    dataset_episode_count = int(dataset.meta.total_episodes)
    if _reconcile_conversion_manifest(
        manifest,
        episodes,
        dataset_episode_count=dataset_episode_count,
    ):
        _write_json_atomic(manifest_path, manifest)
    processed_sources = {str(item['source_metadata']) for item in manifest.get('episodes', [])}

    added_episodes = 0
    added_frames = 0
    try:
        for metadata in episodes:
            source_metadata = str(Path(metadata['metadata_path']).resolve())
            if source_metadata in processed_sources:
                continue
            frame_count = _append_episode(dataset, metadata)
            episode_index = len(manifest['episodes'])
            manifest['episodes'].append(_conversion_entry(metadata, episode_index))
            processed_sources.add(source_metadata)
            added_episodes += 1
            added_frames += frame_count
            manifest['total_episodes'] = len(manifest['episodes'])
            manifest['total_frames'] = sum(int(item['frame_count']) for item in manifest['episodes'])
            _write_json_atomic(manifest_path, manifest)
    finally:
        dataset.finalize()

    summary = {
        'repo_id': repo_id,
        'input_dir': str(input_dir.resolve()),
        'output_dir': str(output_dir.resolve()),
        'codebase_version': 'v3.0',
        'lerobot_version': str(getattr(lerobot, '__version__', 'unknown')),
        'fps': fps,
        'source_successful_episodes': len(episodes),
        'added_episodes': added_episodes,
        'added_frames': added_frames,
        'total_episodes': len(manifest['episodes']),
        'total_frames': sum(int(item['frame_count']) for item in manifest['episodes']),
        'camera_keys': list(CAMERA_KEYS),
        'state_names': list(STATE_NAMES),
        'action_names': list(ACTION_NAMES),
    }
    _write_json_atomic(output_dir / 'roboassemblybench_export_summary.json', summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Export compact RoboAssemblyBench episodes to LeRobot v3.')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--repo-id', default=DEFAULT_REPO_ID)
    parser.add_argument('--max-episodes', type=int, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-streaming-encoding', action='store_true')
    parser.add_argument('--encoder-threads', type=int, default=2)
    parser.add_argument('--vcodec', default='h264')
    args = parser.parse_args()

    summary = export_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        repo_id=str(args.repo_id),
        max_episodes=args.max_episodes,
        resume=bool(args.resume),
        streaming_encoding=not bool(args.no_streaming_encoding),
        encoder_threads=max(int(args.encoder_threads), 1),
        vcodec=str(args.vcodec),
    )
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
