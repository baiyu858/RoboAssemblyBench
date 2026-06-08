from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(value: Any):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def _to_uint8_rgb(frame) -> np.ndarray | None:
    if frame is None:
        return None
    frame_array = np.asarray(frame)
    if frame_array.size == 0 or frame_array.ndim < 3:
        return None
    if frame_array.shape[-1] >= 4:
        frame_array = frame_array[..., :3]
    elif frame_array.shape[-1] == 1:
        frame_array = np.repeat(frame_array, 3, axis=-1)
    if np.issubdtype(frame_array.dtype, np.floating):
        if float(np.nanmax(frame_array)) <= 1.0 + 1e-6:
            frame_array = frame_array * 255.0
    frame_array = np.nan_to_num(frame_array, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(frame_array, 0, 255).astype(np.uint8)


def _obs_camera_frames(obs: dict) -> list[tuple[str, np.ndarray]]:
    frames = []
    for robot_name, robot_obs in (obs or {}).items():
        if not isinstance(robot_obs, dict):
            continue
        for sensor_name, sensor_obs in ((robot_obs.get('sensors') or {}).items()):
            if not isinstance(sensor_obs, dict):
                continue
            frame = _to_uint8_rgb(sensor_obs.get('rgba'))
            if frame is None:
                continue
            frames.append((f'{robot_name}.{sensor_name}', frame))
    return frames


class RuntimeRoboChecker:
    """Runtime RoboChecker for online rollout monitoring and re-planning feedback."""

    def __init__(
        self,
        *,
        output_dir: Path | None = None,
        feedback_path: Path | None = None,
        check_stride: int = 8,
        capture_rgb: bool = True,
        rgb_frame_stride: int = 24,
        stop_on_violation: bool = True,
        min_steps_before_stop: int = 24,
        interarm_min_distance: float = 0.045,
        attach_grace_steps: int = 72,
        progress_window_steps: int = 96,
        progress_min_delta: float = 0.003,
    ):
        self.output_dir = None if output_dir is None else Path(output_dir).resolve()
        self.feedback_path = None if feedback_path is None else Path(feedback_path).resolve()
        self.check_stride = max(int(check_stride), 1)
        self.capture_rgb = bool(capture_rgb)
        self.rgb_frame_stride = max(int(rgb_frame_stride), 1)
        self.stop_on_violation = bool(stop_on_violation)
        self.min_steps_before_stop = max(int(min_steps_before_stop), 0)
        self.interarm_min_distance = float(interarm_min_distance)
        self.attach_grace_steps = max(int(attach_grace_steps), 0)
        self.progress_window_steps = max(int(progress_window_steps), 1)
        self.progress_min_delta = float(progress_min_delta)
        self.feedback: list[dict[str, Any]] = []
        self.observation_count = 0
        self._last_phase = None
        self._phase_start_errors: dict[str, float] = {}
        self._latest_images: list[str] = []
        self._latest_state_paths: list[str] = []
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.feedback_path is not None:
            self.feedback_path.parent.mkdir(parents=True, exist_ok=True)

    def observe(self, *, task, obs: dict, actions: dict | None = None, episode_idx: int = 0) -> dict[str, Any]:
        step_index = int(getattr(task, 'step_counter', 0))
        should_check = step_index % self.check_stride == 0
        should_capture = self.capture_rgb and step_index % self.rgb_frame_stride == 0
        if not should_check and not should_capture:
            return {'ok': True, 'blocking': False, 'feedback': []}

        snapshot = self._snapshot(task=task, obs=obs, actions=actions, episode_idx=episode_idx)
        if should_capture:
            snapshot['image_paths'] = self._write_rgb_observation(
                obs=obs,
                episode_idx=episode_idx,
                step_index=step_index,
            )
            if snapshot['image_paths']:
                self._latest_images = list(snapshot['image_paths'])
        elif self._latest_images:
            snapshot['image_paths'] = list(self._latest_images)

        feedback = self._validate_snapshot(task=task, snapshot=snapshot)
        if self.output_dir is not None and (should_check or should_capture):
            self._write_snapshot(snapshot=snapshot, feedback=feedback)
        if feedback:
            self.feedback.extend(feedback)
            self._write_feedback()

        blocking = any(item.get('severity') == 'error' for item in feedback)
        return {
            'ok': not blocking,
            'blocking': bool(blocking and self.stop_on_violation and step_index >= self.min_steps_before_stop),
            'feedback': feedback,
            'snapshot': snapshot,
        }

    def finalize(self) -> dict[str, Any]:
        self._write_feedback()
        return {
            'feedback_count': len(self.feedback),
            'feedback': _jsonable(self.feedback),
            'feedback_path': None if self.feedback_path is None else str(self.feedback_path),
            'observation_dir': None if self.output_dir is None else str(self.output_dir),
            'latest_images': list(self._latest_images),
            'latest_state_paths': list(self._latest_state_paths),
        }

    def _snapshot(self, *, task, obs: dict, actions: dict | None, episode_idx: int) -> dict[str, Any]:
        phase_spec = task.get_current_phase_spec() if hasattr(task, 'get_current_phase_spec') else {}
        tracked_objects = task.get_tracked_object_states() if hasattr(task, 'get_tracked_object_states') else {}
        tracked_robots = (
            task.get_tracked_robot_states(phase_spec=phase_spec)
            if hasattr(task, 'get_tracked_robot_states')
            else {}
        )
        runtime_state = task.get_phase_runtime_state() if hasattr(task, 'get_phase_runtime_state') else {}
        return {
            'episode_idx': int(episode_idx),
            'step_index': int(getattr(task, 'step_counter', 0)),
            'phase': getattr(task, 'phase', None),
            'phase_spec': _jsonable(phase_spec),
            'runtime_state': _jsonable(runtime_state),
            'tracked_robots': _jsonable(tracked_robots),
            'tracked_objects': _jsonable(tracked_objects),
            'actions': _jsonable(actions or {}),
            'camera_keys': [key for key, _ in _obs_camera_frames(obs)],
        }

    def _write_rgb_observation(self, *, obs: dict, episode_idx: int, step_index: int) -> list[str]:
        if self.output_dir is None:
            return []
        try:
            import cv2
        except ImportError:
            return []

        image_paths = []
        frame_dir = self.output_dir / f'episode_{episode_idx:04d}' / 'rgb'
        frame_dir.mkdir(parents=True, exist_ok=True)
        for camera_key, frame_rgb in _obs_camera_frames(obs):
            safe_key = camera_key.replace('/', '_').replace('.', '_')
            path = frame_dir / f'step_{step_index:06d}_{safe_key}.png'
            cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            image_paths.append(str(path))
        return image_paths

    def _write_snapshot(self, *, snapshot: dict[str, Any], feedback: list[dict[str, Any]]):
        self.observation_count += 1
        snapshot_dir = self.output_dir / f"episode_{int(snapshot.get('episode_idx', 0)):04d}" / 'state'
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        path = snapshot_dir / f"step_{int(snapshot.get('step_index', 0)):06d}.json"
        payload = {**snapshot, 'feedback': feedback}
        path.write_text(json.dumps(_jsonable(payload), indent=2), encoding='utf-8')
        self._latest_state_paths.append(str(path))
        self._latest_state_paths = self._latest_state_paths[-8:]

    def _write_feedback(self):
        if self.feedback_path is None:
            return
        payload = {
            'feedback_count': len(self.feedback),
            'feedback': _jsonable(self.feedback),
            'latest_images': list(self._latest_images),
            'latest_state_paths': list(self._latest_state_paths),
            'state_observation_count': self.observation_count,
            'observation_dir': None if self.output_dir is None else str(self.output_dir),
        }
        self.feedback_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')

    def _validate_snapshot(self, *, task, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        feedback = []
        feedback.extend(self._validate_scheduling(snapshot))
        feedback.extend(self._validate_spatial(snapshot))
        feedback.extend(self._validate_interaction(snapshot))
        feedback.extend(self._validate_progress(snapshot))
        for item in feedback:
            item.setdefault('phase', snapshot.get('phase'))
            item.setdefault('step_index', snapshot.get('step_index'))
            item.setdefault('image_paths', snapshot.get('image_paths', []))
            item.setdefault('runtime_state', snapshot.get('runtime_state', {}))
        return feedback

    def _feedback(self, *, severity: str, validation: str, message: str, evidence: dict | None = None) -> dict[str, Any]:
        return {
            'severity': severity,
            'validation': validation,
            'message': message,
            'evidence': _jsonable(evidence or {}),
        }

    def _validate_scheduling(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        runtime_state = snapshot.get('runtime_state') or {}
        feedback = []
        if runtime_state.get('failed') or runtime_state.get('phase_status') == 'failed':
            feedback.append(
                self._feedback(
                    severity='error',
                    validation='Validate_Scheduling',
                    message=f"Execution reached failed state: {runtime_state.get('terminal_reason') or 'unknown reason'}.",
                    evidence={'phase_history': runtime_state.get('phase_history')},
                )
            )
        timeout_remaining = runtime_state.get('timeout_remaining')
        if timeout_remaining == 0:
            feedback.append(
                self._feedback(
                    severity='error',
                    validation='Validate_Scheduling',
                    message='Current phase exhausted its timeout budget before satisfying its advance condition.',
                    evidence={'phase_elapsed_steps': runtime_state.get('phase_elapsed_steps')},
                )
            )
        return feedback

    def _validate_spatial(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        tracked_robots = snapshot.get('tracked_robots') or {}
        robot_items = []
        for robot_name, state in tracked_robots.items():
            position = state.get('position') if isinstance(state, dict) else None
            if position is not None:
                robot_items.append((robot_name, np.asarray(position, dtype=float)))
        feedback = []
        for index, (lhs_name, lhs_position) in enumerate(robot_items):
            for rhs_name, rhs_position in robot_items[index + 1 :]:
                distance = float(np.linalg.norm(lhs_position - rhs_position))
                if distance < self.interarm_min_distance:
                    feedback.append(
                        self._feedback(
                            severity='error',
                            validation='Validate_Spatial_Occupancy',
                            message=(
                                f'End-effectors for {lhs_name} and {rhs_name} are too close '
                                f'({distance:.3f} m < {self.interarm_min_distance:.3f} m).'
                            ),
                            evidence={'distance': distance, 'robots': [lhs_name, rhs_name]},
                        )
                    )
        return feedback

    def _validate_interaction(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        phase_spec = snapshot.get('phase_spec') or {}
        runtime_state = snapshot.get('runtime_state') or {}
        phase_step_counter = int(runtime_state.get('phase_step_counter') or 0)
        tracked_objects = snapshot.get('tracked_objects') or {}
        feedback = []

        for object_name, object_state in tracked_objects.items():
            attachment = object_state.get('attachment') if isinstance(object_state, dict) else None
            if isinstance(attachment, dict) and attachment.get('physical_hold_valid') is False:
                feedback.append(
                    self._feedback(
                        severity='error',
                        validation='Validate_Interaction',
                        message=f'Physical hold for object {object_name} is no longer valid.',
                        evidence={'object': object_name, 'attachment': attachment},
                    )
                )

        for attach_spec in self._as_list(phase_spec.get('attach')):
            if not isinstance(attach_spec, dict):
                continue
            object_name = attach_spec.get('object')
            robot_name = attach_spec.get('robot')
            object_state = tracked_objects.get(object_name, {}) if object_name else {}
            if not object_name or phase_step_counter < self.attach_grace_steps:
                continue
            attached_to = object_state.get('attached_to')
            if attached_to != robot_name:
                feedback.append(
                    self._feedback(
                        severity='error',
                        validation='Validate_Interaction',
                        message=(
                            f'Expected {robot_name} to attach {object_name} in phase '
                            f"{phase_spec.get('name')}, but object is attached to {attached_to}."
                        ),
                        evidence={'attach_spec': attach_spec, 'object_state': object_state},
                    )
                )
        return feedback

    def _validate_progress(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        phase = (snapshot.get('episode_idx'), snapshot.get('phase'))
        tracked_robots = snapshot.get('tracked_robots') or {}
        runtime_state = snapshot.get('runtime_state') or {}
        phase_step_counter = int(runtime_state.get('phase_step_counter') or 0)
        if phase != self._last_phase:
            self._last_phase = phase
            self._phase_start_errors = {
                robot_name: float(state.get('position_error'))
                for robot_name, state in tracked_robots.items()
                if isinstance(state, dict) and state.get('position_error') is not None
            }
            return []
        if phase_step_counter < self.progress_window_steps:
            return []
        feedback = []
        for robot_name, state in tracked_robots.items():
            if not isinstance(state, dict):
                continue
            current_error = state.get('position_error')
            start_error = self._phase_start_errors.get(robot_name)
            if current_error is None or start_error is None:
                continue
            improvement = float(start_error) - float(current_error)
            if improvement < self.progress_min_delta and not bool(state.get('target_reached')):
                feedback.append(
                    self._feedback(
                        severity='warning',
                        validation='Validate_Scheduling',
                        message=(
                            f'{robot_name} made little progress toward target {state.get("target_name")} '
                            f'in the current phase.'
                        ),
                        evidence={
                            'start_error': start_error,
                            'current_error': current_error,
                            'improvement': improvement,
                        },
                    )
                )
        return feedback

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]
