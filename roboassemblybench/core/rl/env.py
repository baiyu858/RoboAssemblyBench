"""A single native UR5e scene exposed through the Gymnasium API.

The first version deliberately uses the benchmark's existing simulator and
grasp controller. It is not a vectorized Isaac Lab port. Object/robot state
writes are confined to reset; step applies joint/gripper targets only.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from roboassemblybench.core.rl.control import (
    InsertionConfig, OBSERVATION_SIZE, insertion_reward,
)
from .runtime import InsertionController
from .task import InsertionTask
from roboassemblybench.core.rl.snapshot import read_snapshot


class NativeInsertionEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, snapshot_path, *, config=None, output_dir=None, record_video=False, recipe=None):
        from toolkits.factory_dual_franka_assembly.scene_builder import build_dual_franka_assembly_episode
        from toolkits.factory_dual_franka_assembly.generate_demos import _build_env, _warm_up_runtime_cameras

        self.snapshot = read_snapshot(snapshot_path)
        self.binding = InsertionTask.from_dict(self.snapshot['binding'])
        self.passive_actions = {}
        self.config = config or InsertionConfig(**self.snapshot.get('stage_settings', {}).get('controller_config', {}))
        self.torque_soft_limit = self.snapshot.get('stage_settings', {}).get('torque_soft_limit')
        self.torque_hard_limit = self.snapshot.get('stage_settings', {}).get('torque_hard_limit')
        self.axis_alignment = self.snapshot.get('stage_settings', {}).get('axis_alignment')
        self.output_dir = Path(output_dir or Path(snapshot_path).parent / 'local')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.record_video = record_video
        self.video_recorder = None
        self.episode_index = -1
        self.action_space = gym.spaces.Box(-1., 1., (6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-10., 10., (OBSERVATION_SIZE,), dtype=np.float32)
        cfg = build_dual_franka_assembly_episode(
            recipe=recipe or self.snapshot['recipe'], seed=self.snapshot['seed'], layout_seed=self.snapshot['layout_seed'],
            scene_profile=self.snapshot.get('scene_profile'), attach_runtime_cameras=record_video,
            recipe_overrides=self.snapshot.get('recipe_overrides'),
            control_fps=self.config.physics_hz)
        if recipe:
            from roboassemblybench.core.task_registry import load_task_recipe
            from .capture import request_binding
            payload = load_task_recipe(recipe, scene_profile=self.snapshot.get('scene_profile'),
                                       overrides=self.snapshot.get('recipe_overrides'))
            current = request_binding(payload, self.binding.stage_id)
            if current.policy_contract() != self.binding.policy_contract():
                raise ValueError('Requested recipe changes snapshot bodies, roles or insertion frames')
            self.binding = current
            self.torque_soft_limit = payload.get('precision_insertion_resolved', {}).get(
                self.binding.stage_id, {}).get('torque_soft_limit', self.torque_soft_limit)
            self.torque_hard_limit = payload.get('precision_insertion_resolved', {}).get(
                self.binding.stage_id, {}).get('torque_hard_limit', self.torque_hard_limit)
            self.axis_alignment = payload.get('precision_insertion_resolved', {}).get(
                self.binding.stage_id, {}).get('axis_alignment', self.axis_alignment)
        # Keep previously assembled components, fixtures and obstacles.
        cfg.tracked_object_names = tuple(dict.fromkeys((*cfg.tracked_object_names, *self.binding.objects)))
        cfg.max_steps = 10**9
        for obj in cfg.objects:
            if obj.name in self.snapshot['objects']:
                state = self.snapshot['objects'][obj.name]
                obj.position = tuple(state['position'])
                obj.orientation = tuple(state['orientation'])
        for robot in cfg.robots:
            state = self.snapshot['robots'][robot.name]
            robot.initial_joint_positions = dict(zip(state['dof_names'], state['joint_positions']))
        if not record_video:
            for robot in cfg.robots:
                robot.sensors = []
            cfg.camera_metadata = []
        # Original attach specs prepare finger/part contact reporting before physics.
        original_phases = copy.deepcopy(cfg.phase_specs)
        self.original_phases = original_phases
        self.native_env = _build_env([cfg], headless=True, rendering_fps=self.config.physics_hz,
                                     rendering_interval=15 if record_video else 999999)
        self.native_env.reset()
        startup_task = next(iter(self.native_env.runner.current_tasks.values()))
        for name, saved in self.snapshot['hold_commands'].items():
            startup_task.robots[name].apply_action({
                'arm_joint_controller': [saved['arm_q']],
                'gripper_controller': [saved['gripper_openness']]})
        # Flush initial USD/PhysX/renderer loading before writing reset tensor
        # states. Otherwise the first deferred asset update can replace them.
        self.native_env.warm_up(steps=16, render=True, physics=True)
        print('[insertion-env] native scene reset', flush=True)
        self.task = next(iter(self.native_env.runner.current_tasks.values()))
        if record_video:
            _warm_up_runtime_cameras(self.native_env)
        self.task.phase_specs = [{
            'name': 'insertion_physical_local', 'robot_targets': {},
            'gripper_commands': {r: 'contact_hold' for r, _ in self.binding.participants},
            'advance': {'type': 'local_skill_complete', 'robot': self.binding.insertion_robot, 'skill': '__external_policy__'},
            'timeout_steps': 10**9,
        }]
        self.task.phase_index = 0
        self.task.phase = 'insertion_physical_local'
        self.task._phase_initialized = True
        self.task.max_steps = 10**9
        self.task.config.phase_specs = original_phases
        native_done = self.task.is_done

        def retained_task():
            native_done()  # Preserve the benchmark physics-step clock.
            return False  # Gym reset owns cleanup; the runner must retain this scene.

        self.task.is_done = retained_task
        self.controller = None
        self.last_obs = np.zeros(OBSERVATION_SIZE, np.float32)
        self.policy_steps = 0
        self.needs_reset = True
        print('[insertion-env] local Gym scene ready', flush=True)

    def _finish_video(self):
        if self.video_recorder is not None:
            self.video_recorder.finalize()
            self.video_recorder = None

    def _physics_step(self, actions):
        actions = {**self.passive_actions, **actions}
        obs, _, _, _, _ = self.native_env.step([actions])
        if self.video_recorder is not None:
            self.video_recorder.record(self.task, obs[0], step_index=self.task.step_counter)

    def reset(self, *, seed=None, options=None):
        print(f'[insertion-reset] start episode={self.episode_index + 1}', flush=True)
        super().reset(seed=seed)
        self.needs_reset = True
        self._finish_video()
        self.episode_index += 1
        self.policy_steps = 0
        t, snapshot = self.task, self.snapshot
        t.failed = t.success = False
        t.terminal_reason = None
        t.phase = 'insertion_physical_local'
        t.phase_index = 0
        t.phase_status = 'running'
        t.phase_step_counter = 0
        t.phase_entry_step = t.step_counter
        t._phase_initialized = True
        t._local_skill_completions = {}
        t._phase_transition_request = None
        t._attachments = {}
        t._locked_targets.clear()
        t._frozen_lock_poses.clear()
        t._object_pose_history.clear()
        t._native_closure_clearance_reports = {}
        t._ur5e_contact_hold_commands = copy.deepcopy(snapshot['hold_commands'])
        t.target_poses = {name: {k: np.asarray(v, dtype=float) for k, v in pose.items()}
                          for name, pose in snapshot['target_poses'].items()}
        for name, robot in t.robots.items():
            state = snapshot['robots'][name]
            arm = robot.articulation
            if list(arm.dof_names) != state['dof_names']:
                raise ValueError(f'{name}: snapshot DOF ordering differs from the loaded asset')
            arm.set_joint_positions(np.asarray(state['joint_positions'], dtype=float))
            arm.set_joint_velocities(np.zeros_like(state['joint_velocities'], dtype=float))
            saved = t._ur5e_contact_hold_commands.get(name)
            if saved is None:
                continue
            saved['arm_q'] = np.asarray(saved['arm_q'], dtype=float)
            saved['pose'] = {k: np.asarray(v, dtype=float) for k, v in saved['pose'].items()}
            force_state = saved.get('loaded_force_state', {})
            for key in ('step', 'last_control', 'filtered_forces', 'slip_history'):
                force_state.pop(key, None)
            force_state.update(stable_steps=0, resume_count=0, paused_steps=0)
        for name, state in snapshot['objects'].items():
            body = t._resolve_object(name)
            from isaacsim.core.simulation_manager import SimulationManager
            if state.get('linear_velocity') is not None or name in {o for _, o in self.binding.participants}:
                body.unwrap().initialize(physics_sim_view=SimulationManager.get_physics_sim_view())
            body.set_pose(np.asarray(state['position']), np.asarray(state['orientation']))
            if not (name == self.binding.socket_object and self.binding.socket_is_static):
                t._zero_object_velocity(name)
        from .backends.isaac_ur5e import IsaacUR5eBridge
        atomic = IsaacUR5eBridge(t).atomic
        active = {r for r, _ in self.binding.participants}
        self.passive_actions = {}
        for name in t.robots:
            if name not in active:
                q = atomic._current_arm_q(t, name)
                openness = t._gripper_normalized_openness(name)
                if q is None or openness is None:
                    raise ValueError(f'{name}: cannot initialize passive arm hold')
                self.passive_actions[name] = dict(arm_joint_controller=[q.tolist()], gripper_controller=[float(openness)])
        # Deliberately no random pose teleport after this boundary. The initial
        # contact manifold is rebuilt by PhysX, never restored as a fake force.
        actions = {robot: {'arm_joint_controller': [saved['arm_q'].tolist()],
                           'gripper_controller': [float(saved['gripper_openness'])]}
                   for robot, saved in t._ur5e_contact_hold_commands.items()}
        for robot, name in self.binding.participants:
            t._gripper_contact_metrics(name, robot, attach_spec=snapshot['attachments'][name]['attach_spec'])
        for _ in range(32):
            self._physics_step(actions)
        print('[insertion-reset] physical manifolds rebuilt', flush=True)
        from .flow import refresh_grasp_force_control
        t._attachments = {name: refresh_grasp_force_control(attachment, self.original_phases, self.binding.rl_phase)
                          for name, attachment in snapshot['attachments'].items()}
        for attachment in t._attachments.values():
            for key in ('position', 'orientation'):
                attachment[key] = np.asarray(attachment[key], dtype=float)
        self.controller = InsertionController(t, binding=self.binding, config=self.config,
                                              torque_soft_limit=self.torque_soft_limit,
                                              torque_hard_limit=self.torque_hard_limit, axis_alignment=self.axis_alignment)
        print('[insertion-reset] controller created', flush=True)
        self.controller.guidance = False
        reset_attachments = dict(t._attachments)
        try:
            for _ in range(self.config.reset_warmup_steps):
                self._physics_step(self.controller.tick())
        except (ValueError, RuntimeError):
            from roboassemblybench.core.rl.snapshot import json_value
            detail = {'last_sample': self.controller.last_sample,
                      'attachment_validations': {n: a.get('last_hold_validation') for n, a in reset_attachments.items()},
                      'current_objects': {n: list(t._resolve_object(n).get_pose()) for n in self.binding.objects},
                      'robot_joint_positions': {n: r.articulation.get_joint_positions() for n, r in t.robots.items()}}
            (self.output_dir / 'reset_failure.json').write_text(json.dumps(detail, default=json_value, indent=2))
            raise
        initial = self.controller.sample()
        if initial['failure'] or not all(x.get('motion_ready') for x in initial['controls']):
            raise RuntimeError('Reset failed real-contact qualification: ' + json.dumps(self._info(initial)))
        self.controller.guidance = not bool((options or {}).get('disable_guidance', False))
        self.last_obs = initial['observation']
        self.needs_reset = False
        if self.record_video:
            from toolkits.factory_dual_franka_assembly.generate_demos import LiveRolloutVideoRecorder
            self.video_recorder = LiveRolloutVideoRecorder(output_dir=self.output_dir, episode_idx=self.episode_index,
                                                           task=t, fps=15, frame_stride=16, encoder_backend='stream')
        return self.last_obs.copy(), self._info(initial)

    def _info(self, sample):
        from roboassemblybench.core.rl.snapshot import json_value
        return json.loads(json.dumps({
            'scope': 'local_held_insertion', 'task_contract': self.binding.policy_contract(), 'benchmark_success': False,
            'policy_steps': self.policy_steps, 'physics_step': self.task.step_counter,
            'failure': sample.get('failure'), 'local_seated': sample.get('local_seated', False),
            'measurement': sample.get('measurement'), 'grasps': sample.get('grasps'),
            'controls': sample.get('controls'), 'wrench_world': sample.get('wrench_world'),
            'safety_wrench_world': sample.get('safety_wrench_world'),
            'torque_soft_limit': sample.get('torque_soft_limit'),
            'torque_hard_limit': sample.get('torque_hard_limit'),
            'axis_alignment': sample.get('axis_alignment'),
            'torque_soft_active': sample.get('torque_soft_active'),
            'wrist_torque_norm': sample.get('wrist_torque_norm'),
            'arbitration': self.controller.arbitration if self.controller else None,
            'admittance_offset': self.controller.admittance.offset if self.controller else None,
            'ik_failure': sample.get('ik_failure'),
            'ik_accepted_fraction': sample.get('ik_accepted_fraction'),
        }, default=json_value))

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError('Call reset before stepping a completed/failed episode')
        before = self.controller.sample()['measurement']
        previous = self.controller.action.copy()
        self.controller.set_action(action)
        try:
            for _ in range(self.config.decimation):
                self._physics_step(self.controller.tick())
                observed = self.controller.sample()
                if observed['failure'] or (self.binding.allow_seated_load_transfer and observed['local_seated']):
                    break
            sample = self.controller.sample()
            if self.task.failed:
                sample = {**sample, 'failure': self.task.terminal_reason or 'native_task_failure'}
        except (ValueError, RuntimeError) as exc:
            sample = {**self.controller.last_sample, 'failure': str(exc), 'local_seated': False}
        self.policy_steps += 1
        self.last_obs = sample['observation']
        terminated = bool(sample['failure'] or sample['local_seated'])
        truncated = self.policy_steps >= self.config.horizon and not terminated
        reward = insertion_reward(before, sample['measurement'], sample['wrench_world'],
                                  action, previous, sample['failure'], sample['local_seated'], self.config)
        self.needs_reset = terminated or truncated
        return self.last_obs.copy(), reward, terminated, truncated, self._info(sample)

    def close(self):
        self._finish_video()
        self.native_env.close()
