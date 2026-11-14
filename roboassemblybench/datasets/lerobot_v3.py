"""Direct LeRobot v3 recording in an isolated, bounded background process.

The Isaac process never imports LeRobot or encodes media. Each episode is a
transaction: frames go to a private staging dataset, finalize closes all Parquet
and MP4 writers, and only then is the dataset published. Existing small replay
receipts remain compatible with collection qualification and recovery.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from multiprocessing.connection import Connection, Pipe
from pathlib import Path

import numpy as np

from roboassemblybench.datasets.cartesian_episode import (
    ACTION_NAMES,
    ACTION_SEMANTICS,
    CAMERA_KEYS,
    ROBOT_NAMES,
    STATE_NAMES,
    CompactCartesianEpisodeRecorder,
    _jsonable,
)

WRITER_PYTHON_ENV = 'ROBOASSEMBLYBENCH_LEROBOT_PYTHON'
COMPLETE_MARKER = '.roboassemblybench_v3_complete'


class BackgroundV3Writer:
    """At most queue_size copied frames plus pipe buffers; never drop frames."""

    def __init__(self, *, root: Path, config: dict, python: str | None = None, queue_size: int = 2):
        self.root = Path(root).resolve()
        if self.root.exists():
            raise FileExistsError(f'Direct v3 output already exists: {self.root}')
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = self.root.with_suffix('.writer.log')
        self._log = self.log_path.open('wb')
        self._connection, child = Pipe(duplex=True)
        self._queue = queue.Queue(maxsize=max(1, int(queue_size)))
        self._error = None
        self._closed = False
        self.queue_high_watermark = 0
        self.backpressure_seconds = 0.0
        executable = python or os.environ.get(WRITER_PYTHON_ENV)
        if executable is None:
            # Standard deployment env and the existing local ACT data env.
            environments = Path(sys.prefix).parent
            executable = next(
                (
                    str(environments / name / 'bin/python')
                    for name in ('robo-assembly-data', 'roboassemblybench-act')
                    if (environments / name / 'bin/python').is_file()
                ),
                sys.executable,
            )
        executable = str(executable)
        # Do not expose the Isaac SDK's CPython modules/libraries to the data env.
        env = os.environ.copy()
        for key in (
            'PYTHONPATH',
            'LD_LIBRARY_PATH',
            'LD_PRELOAD',
            'PYTHONHOME',
            'ISAAC_PATH',
            'CARB_APP_PATH',
            'EXP_PATH',
        ):
            env.pop(key, None)
        repo = Path(__file__).resolve().parents[2]
        env.update(
            PYTHONPATH=str(repo),
            PYTHONNOUSERSITE='1',
            HF_HUB_OFFLINE='1',
            OMP_NUM_THREADS='1',
            MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',
        )
        try:
            self._process = subprocess.Popen(
                [executable, '-m', 'roboassemblybench.datasets.lerobot_v3', '--worker-fd', str(child.fileno())],
                pass_fds=(child.fileno(),),
                env=env,
                cwd=repo,
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
            )
        except BaseException:
            self._connection.close()
            self._log.close()
            raise
        finally:
            child.close()
        try:
            self._connection.send(dict(config, root=str(self.root)))
            ready = self._receive(timeout=120)
            if not ready.get('ready'):
                raise RuntimeError(ready.get('error', 'LeRobot writer did not initialize'))
        except BaseException:
            self.abort()
            raise
        self._thread = threading.Thread(target=self._send_loop, name='lerobot-v3-sender', daemon=True)
        self._thread.start()

    def _failure(self):
        detail = self.log_path.read_text(errors='replace')[-3000:]
        return RuntimeError(f'Background v3 writer failed; {self.log_path}\n{detail}')

    def _receive(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._connection.poll(0.2):
                try:
                    return self._connection.recv()
                except EOFError:
                    raise self._failure() from None
            if self._process.poll() is not None:
                raise self._failure()
            if self._error is not None:
                raise self._failure() from self._error
        raise TimeoutError(f'Background v3 writer timeout; inspect {self.log_path}')

    def _send_loop(self):
        try:
            while True:
                packet = self._queue.get()
                if packet is None:
                    return
                self._connection.send(packet)
                if packet['kind'] == 'end':
                    return
        except BaseException as error:
            self._error = error

    def submit(self, packet):
        if self._closed:
            raise RuntimeError('Background v3 writer is closed')
        started = time.monotonic()
        while True:
            if self._error is not None or self._process.poll() is not None:
                raise self._failure() from self._error
            try:
                self._queue.put(packet, timeout=0.2)
                self.queue_high_watermark = max(self.queue_high_watermark, self._queue.qsize())
                self.backpressure_seconds += time.monotonic() - started
                return
            except queue.Full:
                if time.monotonic() - started > 120:
                    self.abort()
                    raise TimeoutError(f'Background writer stalled; inspect {self.log_path}')

    def finish(self, *, metrics, keep_failed=False):
        try:
            self.submit({'kind': 'end', 'metrics': _jsonable(metrics), 'keep_failed': bool(keep_failed)})
            result = self._receive(timeout=300)
            self._thread.join(timeout=5)
            if self._process.wait(timeout=15) != 0 or result.get('error'):
                raise self._failure()
            result.update(
                queue_high_watermark=self.queue_high_watermark,
                backpressure_seconds=self.backpressure_seconds,
                dropped_frames=0,
            )
            return result
        finally:
            self.abort()

    def abort(self):
        if self._closed:
            return
        self._closed = True
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)
        self._connection.close()
        self._log.close()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        thread = getattr(self, '_thread', None)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)


def _cached_finger_forces(task):
    """Read existing same-step pair probes, without creating probes or changing control."""
    forces = np.zeros(4, np.float32)
    valid = np.zeros(4, np.int64)
    for attachment in (getattr(task, '_attachments', {}) or {}).values():
        robot = attachment.get('robot_name')
        if robot not in ROBOT_NAMES:
            continue
        metrics = attachment.get('current_contact_metrics') or {}
        if attachment.get('current_contact_step') != int(task.step_counter):
            continue
        offset = ROBOT_NAMES.index(robot) * 2
        for i, name in enumerate(('left_finger', 'right_finger')):
            finger = metrics.get(name) or {}
            value = float(finger.get('force') or 0.0)
            if finger.get('force_probe_valid') and np.isfinite(value):
                forces[offset + i], valid[offset + i] = value, 1
    return forces, valid


def _wrist_joint_reaction(task):
    """Solver reaction at wrist_3_link, expressed in that link's local frame.

    This includes gravity/inertia and is not the RL pair-contact wrench or an
    applied motor torque. Fixed/missing sensors are represented by a validity mask.
    """
    values, valid = np.zeros(12, np.float32), np.zeros(2, np.int64)
    for i, name in enumerate(ROBOT_NAMES):
        articulation = getattr(getattr(task, 'robots', {}).get(name), 'articulation', None)
        try:
            native = articulation.unwrap()
            indices = native._articulation_view._metadata.joint_indices
            # Isaac's incoming-joint force array includes the root at row 0.
            body = int(indices['wrist_3_joint']) + 1
            forces = np.asarray(native.get_measured_joint_forces(), dtype=np.float32)
            sample = forces[int(body)]
            if sample.shape == (6,) and np.all(np.isfinite(sample)):
                values[i * 6 : (i + 1) * 6], valid[i] = sample, 1
        except (AttributeError, TypeError, ValueError, IndexError, KeyError, RuntimeError):
            pass
    return values, valid


def _measured_joint_effort(task):
    values, valid = np.zeros(14, np.float32), np.zeros(2, np.int64)
    for i, name in enumerate(ROBOT_NAMES):
        articulation = getattr(getattr(task, 'robots', {}).get(name), 'articulation', None)
        try:
            native = articulation.unwrap()
            sample = np.asarray(native.get_measured_joint_efforts(), dtype=np.float32).reshape(-1)[:7]
            if sample.shape == (7,) and np.all(np.isfinite(sample)):
                values[i * 7 : (i + 1) * 7], valid[i] = sample, 1
        except (AttributeError, TypeError, ValueError, IndexError, RuntimeError):
            pass
    return values, valid


class DirectLeRobotEpisodeRecorder(CompactCartesianEpisodeRecorder):
    """Same sampling and next-sample action contract as the raw recorder, no raw RGB pass."""

    def __init__(self, *, writer_python=None, queue_size=2, keep_failed=False, **kwargs):
        if kwargs.get('record_media') is False:
            raise ValueError('Direct v3 recording requires runtime RGB-D cameras')
        super().__init__(**kwargs)
        self._keep_failed = bool(keep_failed or os.environ.get('ROBOASSEMBLYBENCH_LEROBOT_KEEP_FAILED') == '1')
        self._direct_finalized = False
        self.dataset_root = self.output_dir / f'episode_{self.episode_idx:06d}_lerobot_v3'
        self._background = BackgroundV3Writer(
            root=self.dataset_root,
            python=writer_python,
            queue_size=int(os.environ.get('ROBOASSEMBLYBENCH_LEROBOT_QUEUE_SIZE', queue_size)),
            config=dict(
                fps=self.fps,
                video_codec=self.video_codec,
                video_crf=self.video_crf,
                video_preset=self.video_preset,
                depth_compression_level=self.depth_compression_level,
                simulation_fps=self.simulation_fps,
                frame_stride=self.frame_stride,
                repo_id='local/roboassemblybench-direct-v3',
                task=str(self._task_text(kwargs['task'])),
            ),
        )

    @staticmethod
    def _task_text(task):
        return task.config.task_description or task.config.prompt

    def _record_media_frame(self, obs):
        frames, depths = {}, {}
        for key, binding in self._camera_bindings.items():
            rgb = self._rgb_frame(obs, binding)
            depth = self._observation_encoder.depth_frame(obs, binding)
            if rgb is None or depth is None:
                raise RuntimeError(f'Missing synchronized RGB/depth for {key}')
            self._source_video_shapes.setdefault(key, tuple(int(x) for x in rgb.shape))
            frames[key] = np.ascontiguousarray(self._letterbox_frame(rgb, key)).copy()
            # Depth must share its own camera's RGB geometry, including front 16:9.
            h, w = frames[key].shape[:2]
            import cv2

            dh, dw = depth.shape
            scale = min(w / dw, h / dh)
            rw, rh = max(1, round(dw * scale)), max(1, round(dh * scale))
            resized = cv2.resize(np.asarray(depth, np.float32), (rw, rh), interpolation=cv2.INTER_NEAREST)
            output = np.zeros((h, w), np.float32)
            x, y = (w - rw) // 2, (h - rh) // 2
            output[y : y + rh, x : x + rw] = resized
            depths[key] = output
            self._video_shapes[key] = tuple(frames[key].shape)
            self._frame_counts[key] += 1
        self._pending_media = (frames, depths)
        return True

    def record(self, *, task, obs, actions):
        if not super().record(task=task, obs=obs, actions=actions):
            return False
        # Low dimensional values were sampled once by the common recorder.
        fields = {
            'observation.state': self._states[-1],
            'observation.eef_pose': self._states[-1][[*range(7), *range(8, 15)]],
            'observation.gripper_state': self._states[-1][[7, 15]],
            'observation.joint_state': self._joint_states[-1],
            'observation.joint_velocity': self._joint_velocities[-1],
            'observation.joint_effort': self._joint_efforts[-1],
            'observation.wrist_wrench': self._wrist_wrenches[-1],
            'observation.collision_signal': self._collision_signals[-1],
            'observation.full_joint_state': np.concatenate(self._replay_joint_states[-1]),
        }
        for name, values in (
            ('simulation_step', self._simulation_steps),
            ('phase_index', self._phase_indices),
            ('phase_step', self._phase_steps),
            ('subtask_index', self._subtask_indices),
            ('substage_index', self._substage_indices),
            ('waiting_state', self._waiting_states),
            ('handoff_state', self._handoff_states),
            ('joint_state_available', self._joint_state_available),
            ('joint_velocity_available', self._joint_velocity_available),
            ('joint_effort_available', self._joint_effort_available),
            ('wrist_wrench_available', self._wrist_wrench_available),
        ):
            fields[f'observation.{name}'] = np.array([values[-1]], np.int64)
        for name, value in (
            ('tracked_object_position', self._tracked_object_positions[-1]),
            ('tracked_object_orientation', self._tracked_object_orientations[-1]),
        ):
            if value.size:
                fields[f'observation.{name}'] = value
        forces, valid = _cached_finger_forces(task)
        fields['observation.finger_contact_force'] = forces
        fields['observation.finger_contact_force_available'] = valid
        reaction, valid = _wrist_joint_reaction(task)
        fields['observation.wrist_joint_reaction'] = reaction
        fields['observation.wrist_joint_reaction_available'] = valid
        measured, valid = _measured_joint_effort(task)
        fields['observation.joint_measured_effort'] = measured
        fields['observation.joint_measured_effort_available'] = valid
        frames, depths = self._pending_media
        self._background.submit(
            dict(
                kind='frame',
                fields={k: v.copy() for k, v in fields.items()},
                rgb=frames,
                depth=depths,
                gripper=self._gripper_commands[-1].copy(),
            )
        )
        self._pending_media = None
        return True

    def finalize(self, *, task, metrics):
        if self._direct_finalized:
            raise RuntimeError('Direct v3 episode finalized twice')
        self._direct_finalized = True
        result = self._background.finish(metrics=metrics, keep_failed=self._keep_failed)
        if result['committed']:
            self._video_paths = {k: Path(v) for k, v in result['videos'].items()}
            self._depth_metadata = result['depth']
            self._video_encoding_metadata = {
                k: dict(
                    codec=self.video_codec,
                    crf=self.video_crf,
                    preset=self.video_preset,
                    fps=self.fps,
                    frame_count=len(self._states),
                    backend='background_lerobot_v3_stream',
                )
                for k in CAMERA_KEYS
            }
        receipt = super().finalize(task=task, metrics=metrics)
        # Chunk indexes already live in metadata.depth and the depth sidecar.
        # Do not duplicate thousands of per-frame offsets in collection manifests.
        result = {key: value for key, value in result.items() if key not in {'videos', 'depth'}}
        metadata_path = Path(receipt['metadata_path'])
        metadata = json.loads(metadata_path.read_text())
        metadata['recording_mode'] = 'direct_lerobot_v3' if result['committed'] else 'direct_lerobot_v3_discarded'
        metadata['lerobot_v3'] = result
        metadata_path.write_text(json.dumps(_jsonable(metadata), indent=2))
        if result['committed']:
            for source, relative in (
                (receipt['trajectory_path'], 'replay/episode_000000.npz'),
                (receipt['annotation_path'], 'annotations/episode_000000.json'),
                (str(metadata_path), 'roboassemblybench_episode.json'),
            ):
                dest = self.dataset_root / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
            info_path = self.dataset_root / 'meta/info.json'
            info = json.loads(info_path.read_text())
            info['roboassemblybench_recording'].update(
                full_joint_names={
                    name: list(getattr(getattr(robot, 'articulation', None), 'dof_names', []) or [])
                    for name, robot in task.robots.items()
                },
                tracked_object_names=self._tracked_object_names or [],
                robot_metadata=metadata['robot_metadata'],
                image_preprocessing=metadata['image_preprocessing'],
                wrist_joint_reaction=dict(
                    source='PhysX incoming joint solver reaction at wrist_3_link',
                    frame='wrist_3_link local frame',
                    units=['N', 'N', 'N', 'Nm', 'Nm', 'Nm'],
                ),
            )
            info_path.write_text(json.dumps(_jsonable(info), indent=2))
            temporary_marker = self.dataset_root / (COMPLETE_MARKER + '.tmp')
            temporary_marker.write_text(
                json.dumps(
                    dict(
                        codebase_version='v3.0',
                        total_episodes=1,
                        total_frames=receipt['frame_count'],
                        success=bool(metrics.get('success', False)),
                        diagnostic_only=not bool(metrics.get('success', False)),
                    )
                )
            )
            temporary_marker.replace(self.dataset_root / COMPLETE_MARKER)
        receipt['lerobot_v3'] = result
        return receipt


def _run_writer(connection):
    """Worker-only imports keep LeRobot and its torch stack out of Isaac."""
    import av
    import lerobot
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
    from lerobot.datasets.video_utils import StreamingVideoEncoder

    from roboassemblybench.datasets.media_codecs import ChunkedDepthWriter

    if str(CODEBASE_VERSION) != 'v3.0' or lerobot.__version__ != '0.4.4':
        raise RuntimeError('Direct writer requires validated lerobot==0.4.4 (dataset v3.0)')

    class BlockingEncoder(StreamingVideoEncoder):
        # LeRobot 0.4.4 otherwise silently drops frames after a 100 ms timeout.
        def feed_frame(self, video_key, image):
            while self._threads[video_key].is_alive():
                try:
                    self._frame_queues[video_key].put(image.copy(), timeout=0.2)
                    return
                except queue.Full:
                    continue
            raise RuntimeError(f'RGB encoder stopped for {video_key}')

        def finish_episode(self):
            result = super().finish_episode()
            if any(stats is None for _, stats in result.values()):
                raise RuntimeError('RGB encoder did not finish cleanly')
            return result

    config = connection.recv()
    root = Path(config['root'])
    staging = root.with_name(f'.{root.name}.pending-{uuid.uuid4().hex}')
    dataset, depths, pending, count = None, {}, None, 0
    connection.send({'ready': True})
    try:
        while True:
            packet = connection.recv()
            if packet['kind'] == 'end':
                break
            fields, rgb = packet['fields'], packet['rgb']
            if dataset is None:
                features = {k: dict(dtype=str(v.dtype), shape=tuple(v.shape), names=None) for k, v in fields.items()}
                features['action'] = dict(dtype='float32', shape=(16,), names=None)
                features['observation.state']['names'] = list(STATE_NAMES)
                features['action']['names'] = list(ACTION_NAMES)
                for k, v in rgb.items():
                    features[k] = dict(dtype='video', shape=tuple(v.shape), names=['height', 'width', 'channels'])
                dataset = LeRobotDataset.create(
                    repo_id=config['repo_id'],
                    fps=int(config['fps']),
                    root=staging,
                    robot_type='dual_ur5e_robotiq_2f85',
                    features=features,
                    use_videos=True,
                    streaming_encoding=True,
                    video_backend='pyav',
                    encoder_threads=1,
                    encoder_queue_maxsize=2,
                    vcodec=config['video_codec'],
                )
                dataset._streaming_encoder.close()
                dataset._streaming_encoder = BlockingEncoder(
                    fps=config['fps'],
                    vcodec=config['video_codec'],
                    crf=config['video_crf'],
                    preset=config['video_preset'],
                    queue_maxsize=2,
                    encoder_threads=1,
                )
            if pending is not None:
                _add_pending(dataset, pending, packet, config['task'])
            for k, v in packet['depth'].items():
                if k not in depths:
                    depths[k] = ChunkedDepthWriter(
                        staging / 'sensors/depth' / f'{k}.u16.bshuf.zst',
                        shape=tuple(v.shape),
                        compression_level=config['depth_compression_level'],
                    )
                depths[k].write(v)
            pending = packet
            count += 1
        commit = bool(packet['metrics'].get('success', False) or packet['keep_failed'])
        if not commit:
            if dataset is not None:
                dataset.clear_episode_buffer()
                dataset.finalize()
                dataset = None
            connection.send(dict(committed=False, frame_count=count, diagnostic_only=False))
            return
        if pending is None:
            raise RuntimeError('Cannot publish empty v3 episode')
        _add_pending(dataset, pending, pending, config['task'])
        for key, feature in dataset.features.items():
            if feature['dtype'] == 'int64' and tuple(feature['shape']) == (1,) and key.startswith('observation.'):
                dataset.episode_buffer[key] = [int(np.asarray(v).item()) for v in dataset.episode_buffer[key]]
        dataset.save_episode(parallel_encoding=False)
        dataset.finalize()
        info_path = staging / 'meta/info.json'
        info = json.loads(info_path.read_text())
        depth = {k: writer.close() for k, writer in depths.items()}
        depth_index = {k: dict(v, path=Path(v['path']).relative_to(staging).as_posix()) for k, v in depth.items()}
        (staging / 'sensors/depth/index.json').write_text(json.dumps(depth_index, indent=2))
        info['roboassemblybench_sidecars'] = dict(
            metric_depth=dict(
                index_path='sensors/depth/index.json',
                dtype='uint16',
                units='millimeters',
                depth_scale=0.001,
                invalid_value=0,
                compression='zstd',
                filter='bitshuffle',
                frame_alignment='frame_index',
            )
        )
        info['roboassemblybench_recording'] = dict(
            mode='direct_background_v3',
            action_semantics=ACTION_SEMANTICS,
            quaternion_order='wxyz',
            physics_fps=config['simulation_fps'],
            dataset_fps=config['fps'],
            frame_stride=config['frame_stride'],
            finger_contact_force_units='N',
            wrist_wrench_units=['N', 'N', 'N', 'Nm', 'Nm', 'Nm'],
            unavailable_values='zero with corresponding availability flag; never treat as a measured zero',
            success=bool(packet['metrics'].get('success', False)),
            diagnostic_only=not bool(packet['metrics'].get('success', False)),
        )
        info_path.write_text(json.dumps(info, indent=2))
        videos = {}
        for k in CAMERA_KEYS:
            path = staging / info['video_path'].format(video_key=k, chunk_index=0, file_index=0)
            with av.open(str(path)) as video:
                if video.streams.video[0].frames != count:
                    raise RuntimeError(f'Video frame count mismatch for {k}')
            videos[k] = str(root / path.relative_to(staging))
        if root.exists():
            raise FileExistsError(root)
        staging.rename(root)
        connection.send(
            dict(
                committed=True,
                root=str(root),
                repo_id=config['repo_id'],
                episode_index=0,
                frame_count=count,
                videos=videos,
                depth={k: dict(v, path=str(root / Path(v['path']).relative_to(staging))) for k, v in depth.items()},
                diagnostic_only=not bool(packet['metrics'].get('success', False)),
            )
        )
    finally:
        if dataset is not None:
            dataset.finalize()
        for writer in depths.values():
            writer.close()
        if staging.exists():
            shutil.rmtree(staging)


def _add_pending(dataset, pending, following, task):
    state = following['fields']['observation.state'].copy()
    state[[7, 15]] = following['gripper']
    frame = dict(pending['fields'], **pending['rgb'], action=state, task=task)
    dataset.add_frame(frame)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-fd', type=int, required=True)
    args = parser.parse_args()
    connection = Connection(args.worker_fd)
    try:
        _run_writer(connection)
    except BaseException:
        traceback.print_exc()
        try:
            connection.send({'error': traceback.format_exc()})
        except (OSError, EOFError):
            pass
        raise
    finally:
        connection.close()


if __name__ == '__main__':
    main()
