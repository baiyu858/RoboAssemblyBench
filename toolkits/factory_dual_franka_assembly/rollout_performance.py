"""Small aggregate timings and cached safety evidence, without extra sensor reads."""
import json
import math
import time
from pathlib import Path


class RolloutPerformance:
    def __init__(self, output_dir, *, physics_fps, rendering_interval):
        self.path = None if output_dir is None else Path(output_dir) / 'rollout_performance.json'
        self.fps = physics_fps
        self.interval = rendering_interval
        self.started = time.perf_counter()
        self.steps = 0
        self.phases = {}
        self.holds = {}
        self.previous_poses = {}
        self.physics_read_cache = {}
        self.read_owners = {}

    @staticmethod
    def _pose_delta(a, b):
        distance = math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a[0], b[0])))
        norm = math.sqrt(sum(float(x) ** 2 for x in a[1]) * sum(float(x) ** 2 for x in b[1]))
        cosine = abs(sum(float(x) * float(y) for x, y in zip(a[1], b[1]))) / max(norm, 1e-12)
        return distance, 2 * math.acos(min(1.0, cosine))

    def record(self, *, phase, timings, task, observations=None):
        self.steps += 1
        self.physics_read_cache = getattr(task, '_physics_read_cache_counts', {})
        self.read_owners = getattr(task, 'robots', {})
        row = self.phases.setdefault(phase, {'steps': 0, 'seconds': {}})
        row['steps'] += 1
        for key, value in timings.items():
            row['seconds'][key] = row['seconds'].get(key, 0.0) + value
        for robot, obs in (observations or {}).items():
            if not isinstance(obs, dict):
                continue
            pose = (obs.get('eef_kinematics_position'), obs.get('eef_kinematics_orientation'))
            if any(x is None for x in pose):
                continue
            pose = ([float(x) for x in pose[0]], [float(x) for x in pose[1]])
            tracking = row.setdefault('controller_fk_tracking', {}).setdefault(
                robot,
                {
                    'max_linear_speed_m_s': 0.0,
                    'max_angular_speed_rad_s': 0.0,
                    'max_published_position_lead_m': 0.0,
                    'max_published_angle_lead_rad': 0.0,
                },
            )
            previous = self.previous_poses.get(robot)
            if previous and previous[0] == int(task.step_counter) - 1:
                distance, angle = self._pose_delta(pose, previous[1])
                tracking['max_linear_speed_m_s'] = max(tracking['max_linear_speed_m_s'], distance * self.fps)
                tracking['max_angular_speed_rad_s'] = max(tracking['max_angular_speed_rad_s'], angle * self.fps)
            self.previous_poses[robot] = (int(task.step_counter), pose)
            command = getattr(task, '_ur5e_contact_hold_commands', {}).get(robot, {})
            if command.get('pose_source') == 'published_joint_fk' and command.get('object') in getattr(
                task, '_attachments', {}
            ):
                target = command['pose']
                distance, angle = self._pose_delta(pose, (target['position'], target['orientation']))
                tracking['max_published_position_lead_m'] = max(tracking['max_published_position_lead_m'], distance)
                tracking['max_published_angle_lead_rad'] = max(tracking['max_published_angle_lead_rad'], angle)
        # Read only results already produced by mandatory safety checks.
        for name, attachment in getattr(task, '_attachments', {}).items():
            sample = attachment.get('last_hold_validation')
            if not sample:
                continue
            summary = self.holds.setdefault(
                name,
                {
                    'max_position_slip': 0.0,
                    'max_orientation_slip': 0.0,
                    'invalid_samples': 0,
                    'last': None,
                    'min_force_N': {},
                    'max_force_N': {},
                    'contact_missing_samples': 0,
                },
            )
            if summary['last'] and sample['step'] == summary['last']['step']:
                continue
            summary['max_position_slip'] = max(summary['max_position_slip'], sample.get('position_slip', 0.0))
            summary['max_orientation_slip'] = max(summary['max_orientation_slip'], sample.get('orientation_slip', 0.0))
            summary['invalid_samples'] += not sample['valid']
            summary['contact_missing_samples'] += sample.get('contact_ready') is False
            for side, finger in sample.get('fingers', {}).items():
                force = finger.get('force')
                if force is not None and math.isfinite(force):
                    summary['min_force_N'][side] = min(summary['min_force_N'].get(side, force), force)
                    summary['max_force_N'][side] = max(summary['max_force_N'].get(side, force), force)
            summary['last'] = sample.copy()
        if self.steps % 240 == 0:
            self.flush()

    def flush(self, *, finalized=False):
        elapsed = time.perf_counter() - self.started
        state_caches, image_caches = {}, {}
        for name, robot in self.read_owners.items():
            state_caches[name] = {
                'robot': getattr(robot, '_physics_read_cache_counts', {}),
                'controllers': {
                    key: getattr(c, '_physics_read_cache_counts', {})
                    for key, c in getattr(robot, 'controllers', {}).items()
                },
            }
            for key, sensor in getattr(robot, 'sensors', {}).items():
                cache = getattr(sensor, '_render_frame_cache', None)
                if cache is not None:
                    image_caches[f'{name}/{key}'] = cache.counts
        result = {
            'finalized': finalized,
            'steps': self.steps,
            'physics_fps': self.fps,
            'physics_dt': 1 / self.fps,
            'rendering_interval': self.interval,
            'render_period_steps': self.interval + 1,
            'wall_seconds': elapsed,
            'simulated_seconds': self.steps / self.fps,
            'real_time_factor': self.steps / self.fps / max(elapsed, 1e-9),
            'phases': self.phases,
            'cached_hold_summary': self.holds,
            'physics_read_cache': self.physics_read_cache,
            'state_read_caches': state_caches,
            'image_read_caches': image_caches,
            'timing_scope': 'CPU wall times including synchronization; not an exclusive GPU profile.',
        }
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(result, indent=2))
            temporary.replace(self.path)
        return result
