from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import pickle
import re
import sys
from typing import Any

import numpy as np

from internutopia_extension.configs.robots.franka import arm_ik_cfg, arm_joint_cfg, gripper_cfg


class _FactoryPegInsertNetwork:
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        import torch
        import torch.nn as nn

        self._torch = torch
        self.device = torch.device(device if device != 'auto' else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_state = checkpoint['model']

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = nn.LSTM(input_size=19, hidden_size=1024, num_layers=2)
                self.layer_norm = nn.LayerNorm(1024)
                self.actor_mlp = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ELU(),
                    nn.Linear(512, 128),
                    nn.ELU(),
                    nn.Linear(128, 64),
                    nn.ELU(),
                )
                self.mu = nn.Linear(64, 6)

            def forward(self, obs, state):
                output, next_state = self.rnn(obs.view(1, 1, 19), state)
                latent = self.layer_norm(output.view(1, 1024))
                latent = self.actor_mlp(latent)
                return self.mu(latent).view(6), next_state

        self.network = Network().to(self.device)
        self.network.eval()
        translated_state = {}
        for key, value in model_state.items():
            if not key.startswith('a2c_network.'):
                continue
            translated_key = key.removeprefix('a2c_network.')
            if translated_key.startswith('rnn.rnn.'):
                translated_key = 'rnn.' + translated_key.removeprefix('rnn.rnn.')
            if translated_key.startswith(('rnn.', 'layer_norm.', 'actor_mlp.', 'mu.')):
                translated_state[translated_key] = value
        self.network.load_state_dict(translated_state, strict=True)
        self.obs_mean = model_state['running_mean_std.running_mean'].to(self.device, dtype=torch.float32)
        self.obs_var = model_state['running_mean_std.running_var'].to(self.device, dtype=torch.float32)
        self.state = self._zero_state()

    def _zero_state(self):
        torch = self._torch
        return (
            torch.zeros(2, 1, 1024, dtype=torch.float32, device=self.device),
            torch.zeros(2, 1, 1024, dtype=torch.float32, device=self.device),
        )

    def reset(self):
        self.state = self._zero_state()

    def act(self, obs: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            obs_tensor = (obs_tensor - self.obs_mean) / torch.sqrt(self.obs_var + 1e-5)
            action, self.state = self.network(obs_tensor, self.state)
            return torch.clamp(action, -1.0, 1.0).detach().cpu().numpy()


class FactoryPegInsertPolicyAdapter:
    """Experimental Fabrica adapter for the IsaacLab Factory peg-insertion policy.

    The checkpoint was trained on an 8 mm cylindrical peg/hole pair.  For Fabrica
    cooling manifold phases, this adapter treats the current robot seat target as
    the fixed insertion frame and emits small IK deltas only.  It is intentionally
    conservative: geometry, object scale, board scale, and collisions remain
    controlled by the surrounding task.
    """

    def __init__(self, spec: dict[str, Any]):
        self.spec = dict(spec)
        self._network_cache: dict[str, _FactoryPegInsertNetwork] = {}
        self._prev_actions: dict[tuple[Any, ...], np.ndarray] = {}
        self._last_phase_key: tuple[Any, ...] | None = None

    def act(
        self,
        *,
        task,
        robot_name: str,
        phase_spec: dict,
        skill_spec: dict,
        tracked_robots: dict,
        tracked_objects: dict,
        checkpoint_path: str,
    ) -> dict | None:
        robot_state = tracked_robots.get(robot_name, {})
        current_position = _array_or_none(robot_state.get('position'))
        current_orientation = _array_or_none(robot_state.get('orientation'))
        target_position = _array_or_none(robot_state.get('task_target'))
        if current_position is None or current_orientation is None or target_position is None:
            return None
        payload_delta = self._payload_target_delta(
            task=task,
            skill_spec=skill_spec,
            tracked_objects=tracked_objects,
        )
        activation_distance = float(skill_spec.get('activation_distance', 0.04))
        activation_payload_distance = float(skill_spec.get('activation_payload_distance', activation_distance))
        robot_ready = activation_distance <= 0.0 or float(np.linalg.norm(current_position - target_position)) <= activation_distance
        payload_ready = (
            payload_delta is not None
            and (
                activation_payload_distance <= 0.0
                or float(np.linalg.norm(payload_delta)) <= activation_payload_distance
            )
        )

        phase_key = (
            id(task),
            getattr(task, 'phase_index', None),
            getattr(task, 'phase_entry_step', None),
            robot_name,
        )
        network = self._network(checkpoint_path, skill_spec)
        if phase_key != self._last_phase_key:
            network.reset()
            self._last_phase_key = phase_key

        prev_action = self._prev_actions.get(phase_key, np.zeros(6, dtype=np.float32))
        obs = np.concatenate(
            [
                current_position - target_position,
                _normalize_quat(current_orientation),
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                prev_action,
            ]
        ).astype(np.float32)
        rl_action = network.act(obs).astype(np.float32)
        self._prev_actions[phase_key] = rl_action

        rl_position_step = float(skill_spec.get('rl_position_step', 0.012))
        target_bias_step = float(skill_spec.get('target_bias_step', 0.004))
        max_position_step = float(skill_spec.get('max_position_step', 0.012))
        if payload_ready:
            rl_position_step = float(skill_spec.get('payload_rl_position_step', rl_position_step))
            target_bias_step = float(skill_spec.get('payload_target_bias_step', target_bias_step))
            max_position_step = float(skill_spec.get('payload_max_position_step', max_position_step))
        action_frame_radius = float(skill_spec.get('action_frame_radius', 0.05))

        rl_delta = np.clip(rl_action[:3], -1.0, 1.0) * rl_position_step
        target_bias_source = payload_delta if payload_delta is not None else target_position - current_position
        if payload_delta is not None and not payload_ready and not robot_ready:
            target_bias_step = float(skill_spec.get('preactivation_target_bias_step', target_bias_step))
            max_position_step = float(skill_spec.get('preactivation_max_position_step', max_position_step))
        target_bias = _clip_norm(target_bias_source, target_bias_step)
        delta = _clip_norm(rl_delta + target_bias, max_position_step)
        commanded_position = current_position + delta
        commanded_position = target_position + np.clip(
            commanded_position - target_position,
            -action_frame_radius,
            action_frame_radius,
        )

        action = OrderedDict()
        action[arm_ik_cfg.name] = [commanded_position.tolist(), current_orientation.tolist()]
        action[gripper_cfg.name] = [0.0]
        return action

    def _payload_target_delta(self, *, task, skill_spec: dict, tracked_objects: dict) -> np.ndarray | None:
        object_name = skill_spec.get('held_object') or skill_spec.get('object')
        target_name = skill_spec.get('held_target') or skill_spec.get('payload_target') or skill_spec.get('target')
        if not object_name or not target_name:
            return None

        object_state = tracked_objects.get(str(object_name), {})
        object_position = _array_or_none(object_state.get('position'))
        if object_position is None:
            try:
                object_position = _array_or_none(task._resolve_object(str(object_name)).get_pose()[0])
            except Exception:
                object_position = None
        if object_position is None:
            return None

        try:
            _, target_position, _, _ = task._resolve_target_pose_spec(target_name)
        except Exception:
            return None
        target_position = _array_or_none(target_position)
        if target_position is None:
            return None
        return target_position - object_position

    def _network(self, checkpoint_path: str, spec: dict[str, Any]) -> _FactoryPegInsertNetwork:
        resolved_path = str(Path(checkpoint_path).expanduser().resolve())
        network = self._network_cache.get(resolved_path)
        if network is None:
            network = _FactoryPegInsertNetwork(resolved_path, device=str(spec.get('device', 'cpu')))
            self._network_cache[resolved_path] = network
        return network


class _FabricaFixPlugNetwork:
    """rl-games MLP actor used by FabricaFixPlugTaskAssemble specialist policies."""

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        import torch
        import torch.nn as nn

        self._torch = torch
        self.device = torch.device(device if device != 'auto' else ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_state = checkpoint['model']

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.actor_mlp = nn.Sequential(
                    nn.Linear(3, 256),
                    nn.ELU(),
                    nn.Linear(256, 128),
                    nn.ELU(),
                    nn.Linear(128, 64),
                    nn.ELU(),
                )
                self.mu = nn.Linear(64, 3)

            def forward(self, obs):
                return self.mu(self.actor_mlp(obs.view(1, 3))).view(3)

        def key(name: str) -> str:
            if name in model_state:
                return name
            no_orig = name.replace('_orig_mod.', '')
            if no_orig in model_state:
                return no_orig
            raise KeyError(name)

        translated_state = {}
        prefix_options = ('_orig_mod.a2c_network.', 'a2c_network.')
        for raw_key, value in model_state.items():
            for prefix in prefix_options:
                if raw_key.startswith(prefix):
                    translated_key = raw_key.removeprefix(prefix)
                    if translated_key.startswith(('actor_mlp.', 'mu.')):
                        translated_state[translated_key] = value
                    break

        self.network = Network().to(self.device)
        self.network.eval()
        self.network.load_state_dict(translated_state, strict=True)
        self.obs_mean = model_state[key('_orig_mod.running_mean_std.running_mean')].to(
            self.device, dtype=torch.float32
        )
        self.obs_var = model_state[key('_orig_mod.running_mean_std.running_var')].to(
            self.device, dtype=torch.float32
        )

    def reset(self):
        return None

    def act(self, obs: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            obs_tensor = (obs_tensor - self.obs_mean) / torch.sqrt(self.obs_var + 1e-5)
            action = self.network(obs_tensor)
            return torch.clamp(action, -1.0, 1.0).detach().cpu().numpy()


class FabricaFixPlugPolicyAdapter:
    """Adapter for official Fabrica FixPlug specialist insertion checkpoints.

    This mirrors FabricaFixPlugTaskAssemble's 3D observation/action contract:
    observation is path-aligned socket-minus-plug position, and action is a
    path-aligned residual Cartesian displacement before the official open-loop
    insertion direction is added.
    """

    def __init__(self, spec: dict[str, Any]):
        self.spec = dict(spec)
        self._network_cache: dict[str, _FabricaFixPlugNetwork] = {}
        self._plan_cache: dict[str, dict] = {}
        self._last_phase_key: tuple[Any, ...] | None = None
        self._phase_residual_state: dict[tuple[Any, ...], dict[str, Any]] = {}

    def act(
        self,
        *,
        task,
        robot_name: str,
        phase_spec: dict,
        skill_spec: dict,
        tracked_robots: dict,
        tracked_objects: dict,
        checkpoint_path: str,
    ) -> dict | None:
        robot_state = tracked_robots.get(robot_name, {})
        current_position = _array_or_none(robot_state.get('position'))
        current_orientation = _array_or_none(robot_state.get('orientation'))
        if current_position is None or current_orientation is None:
            return None

        plug_object = (
            skill_spec.get('held_object')
            or skill_spec.get('plug_object')
            or skill_spec.get('object')
            or skill_spec.get('payload_object')
        )
        socket_object = skill_spec.get('socket_object') or skill_spec.get('fixed_object')
        plug_pos = self._object_position(task, tracked_objects, str(plug_object)) if plug_object else None
        if plug_pos is None:
            plug_pos = current_position
        socket_pos = self._socket_position(
            task=task,
            skill_spec=skill_spec,
            tracked_objects=tracked_objects,
            socket_object=str(socket_object) if socket_object else None,
        )
        if socket_pos is None:
            return None

        pair = self._plug_socket_pair(skill_spec, plug_object=plug_object, socket_object=socket_object)
        if pair is None:
            return None
        plan = self._plan_info(skill_spec, checkpoint_path)
        if pair not in plan:
            return None
        path = np.asarray(plan[pair]['path'], dtype=np.float32)
        path_start = path[-1, :3]
        path_end = path[0, :3]
        path_scale = max(float(np.linalg.norm(path[0, :3] - path[-1, :3]) / 0.02), 1e-6)

        phase_key = (
            id(task),
            getattr(task, 'phase_index', None),
            getattr(task, 'phase_entry_step', None),
            robot_name,
            pair,
        )
        network = self._network(checkpoint_path, skill_spec)
        if phase_key != self._last_phase_key:
            network.reset()
            self._last_phase_key = phase_key
            self._phase_residual_state[phase_key] = {
                'best_norm': float('inf'),
                'stagnant_steps': 0,
                'force_residual': False,
            }

        delta_pos = socket_pos - plug_pos
        obs = _do_deltapos_path_transform(delta_pos, path_start, path_end)
        obs[2] /= path_scale
        rl_action = network.act(obs.astype(np.float32)).astype(np.float32)

        action_world = rl_action.copy()
        if bool(skill_spec.get('path_transform', True)):
            action_world[2] *= path_scale
            action_world = _undo_deltapos_path_transform(action_world, path_start, path_end)

        strict_official = bool(skill_spec.get('strict_official', False))
        residual = socket_pos - plug_pos
        residual_norm_world = float(np.linalg.norm(residual))
        residual_state = self._phase_residual_state.setdefault(
            phase_key,
            {
                'best_norm': float('inf'),
                'stagnant_steps': 0,
                'force_residual': False,
            },
        )
        progress_epsilon = float(skill_spec.get('residual_progress_epsilon', 5e-4))
        if residual_norm_world + progress_epsilon < float(residual_state.get('best_norm', float('inf'))):
            residual_state['best_norm'] = residual_norm_world
            residual_state['stagnant_steps'] = 0
        else:
            residual_state['stagnant_steps'] = int(residual_state.get('stagnant_steps', 0)) + 1
        residual_patience = max(int(skill_spec.get('residual_stagnation_patience', 32)), 1)
        if int(residual_state.get('stagnant_steps', 0)) >= residual_patience:
            residual_state['force_residual'] = True

        if bool(skill_spec.get('residual_action', True)):
            residual_for_norm = residual.copy()
            if bool(skill_spec.get('path_transform', True)):
                residual_for_norm = _do_deltapos_path_transform(residual_for_norm, path_start, path_end)
                residual_for_norm[2] /= path_scale
            residual_norm = float(np.linalg.norm(residual_for_norm))
            if residual_norm > 1e-8:
                action_world = action_world + residual / residual_norm

        pos_scale = np.asarray(skill_spec.get('pos_action_scale', [0.005, 0.005, 0.005]), dtype=np.float32)
        delta = np.asarray(action_world, dtype=np.float32) * pos_scale
        if not strict_official:
            max_position_step = float(skill_spec.get('max_position_step', 0.006))
            delta = _clip_norm(delta, max_position_step)
        action_orientation = current_orientation
        if (
            not strict_official
            and bool(residual_state.get('force_residual', False))
            and bool(
            skill_spec.get('use_payload_target_residual_fallback', False)
            )
        ):
            solved_position, solved_orientation = self._payload_robot_target_pose(
                task=task,
                robot_name=robot_name,
                payload_object=str(plug_object) if plug_object else None,
                payload_target=skill_spec.get('held_target')
                or skill_spec.get('payload_target')
                or skill_spec.get('target'),
                payload_relative_pose_source=skill_spec.get('payload_relative_pose_source'),
            )
            if solved_position is not None:
                delta = _clip_norm(np.asarray(solved_position, dtype=np.float32) - current_position, max_position_step)
                if solved_orientation is not None:
                    action_orientation = _normalize_quat(solved_orientation)
            else:
                delta = _clip_norm(residual, max_position_step)
        elif not strict_official and bool(residual_state.get('force_residual', False)):
            delta = _clip_norm(residual, max_position_step)
        elif not strict_official and bool(skill_spec.get('residual_guard', True)):
            residual_norm = residual_norm_world
            candidate_residual_norm = float(np.linalg.norm(residual - delta))
            if residual_norm > 1e-8 and candidate_residual_norm >= residual_norm - 1e-6:
                delta = _clip_norm(residual, max_position_step)
        min_activation_distance = float(skill_spec.get('activation_payload_distance', 0.0))
        if min_activation_distance > 0.0 and float(np.linalg.norm(socket_pos - plug_pos)) > min_activation_distance:
            return None

        commanded_position = current_position + delta
        action = OrderedDict()
        action[arm_ik_cfg.name] = [commanded_position.tolist(), _normalize_quat(action_orientation).tolist()]
        action[gripper_cfg.name] = [0.0]
        return action

    @staticmethod
    def _payload_robot_target_pose(
        *,
        task,
        robot_name: str,
        payload_object: str | None,
        payload_target,
        payload_relative_pose_source=None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if payload_object is None or payload_target is None or not hasattr(task, 'resolve_robot_target_pose'):
            return None, None
        target_spec = {
            'target': payload_target,
            'payload_object': payload_object,
            'payload_target': payload_target,
            'ik_frame_compensation': 'none',
        }
        if payload_relative_pose_source is not None:
            target_spec['payload_relative_pose_source'] = payload_relative_pose_source
        try:
            _, solved_position, solved_orientation, _ = task.resolve_robot_target_pose(robot_name, target_spec)
        except Exception:
            return None, None
        return _array_or_none(solved_position), _array_or_none(solved_orientation)

    def _network(self, checkpoint_path: str, spec: dict[str, Any]) -> _FabricaFixPlugNetwork:
        resolved_path = str(Path(checkpoint_path).expanduser().resolve())
        network = self._network_cache.get(resolved_path)
        if network is None:
            network = _FabricaFixPlugNetwork(resolved_path, device=str(spec.get('device', 'cpu')))
            self._network_cache[resolved_path] = network
        return network

    def _plan_info(self, spec: dict[str, Any], checkpoint_path: str) -> dict:
        plan_path = spec.get('plan_info') or spec.get('plan_info_path')
        if plan_path:
            resolved = _resolve_existing_path(plan_path)
        else:
            resolved = Path(checkpoint_path).expanduser().resolve().with_name('plumbers_block_plan_info.pkl')
        key = str(resolved)
        if key not in self._plan_cache:
            _install_numpy_pickle_compat()
            with open(resolved, 'rb') as handle:
                self._plan_cache[key] = pickle.load(handle)
        return self._plan_cache[key]

    @staticmethod
    def _socket_position(*, task, skill_spec: dict, tracked_objects: dict, socket_object: str | None) -> np.ndarray | None:
        target_name = skill_spec.get('socket_target') or skill_spec.get('held_target') or skill_spec.get('payload_target')
        if target_name:
            try:
                _, target_position, _, _ = task._resolve_target_pose_spec(target_name)
                target_position = _array_or_none(target_position)
                if target_position is not None:
                    return target_position
            except Exception:
                pass
        if socket_object:
            return FabricaFixPlugPolicyAdapter._object_position(task, tracked_objects, socket_object)
        target_name = skill_spec.get('target')
        if target_name:
            try:
                _, target_position, _, _ = task._resolve_target_pose_spec(target_name)
                return _array_or_none(target_position)
            except Exception:
                return None
        return None

    @staticmethod
    def _object_position(task, tracked_objects: dict, object_name: str) -> np.ndarray | None:
        object_state = tracked_objects.get(object_name, {})
        position = _array_or_none(object_state.get('position'))
        if position is not None:
            return position
        try:
            return _array_or_none(task._resolve_object(object_name).get_pose()[0])
        except Exception:
            return None

    @staticmethod
    def _plug_socket_pair(
        skill_spec: dict,
        *,
        plug_object: Any,
        socket_object: Any,
    ) -> tuple[str, str] | None:
        explicit = skill_spec.get('plug_socket_pair') or skill_spec.get('part_pair')
        if explicit is not None and len(explicit) == 2:
            return (str(explicit[0]), str(explicit[1]))
        plug_id = skill_spec.get('part_plug') or skill_spec.get('plug')
        socket_id = skill_spec.get('part_socket') or skill_spec.get('socket')
        if plug_id is None:
            plug_id = _trailing_int(plug_object)
        if socket_id is None:
            socket_id = _trailing_int(socket_object)
        if plug_id is None or socket_id is None:
            return None
        return (str(plug_id), str(socket_id))


class FabricaOfficialJointPoseAdapter:
    """Send a Fabrica official planned Franka arm configuration as a joint target."""

    def __init__(self, spec: dict[str, Any]):
        self.spec = dict(spec)
        self._motion_cache: dict[str, list] = {}

    def act(
        self,
        *,
        task,
        robot_name: str,
        phase_spec: dict,
        skill_spec: dict,
        tracked_robots: dict,
        tracked_objects: dict,
        checkpoint_path: str | None = None,
    ) -> dict | None:
        joint_positions = self._motion_joint_positions(task=task, skill_spec=skill_spec)
        if joint_positions is None:
            joint_positions = skill_spec.get('joint_positions') or skill_spec.get('arm_q')
        if joint_positions is None:
            plan_info = skill_spec.get('plan_info') or skill_spec.get('plan_info_path')
            pair = FabricaFixPlugPolicyAdapter._plug_socket_pair(
                skill_spec,
                plug_object=skill_spec.get('held_object') or skill_spec.get('plug_object') or skill_spec.get('object'),
                socket_object=skill_spec.get('socket_object') or skill_spec.get('fixed_object'),
            )
            if plan_info and pair is not None:
                _install_numpy_pickle_compat()
                with open(_resolve_existing_path(plan_info), 'rb') as handle:
                    plan = pickle.load(handle)
                entry = plan[pair]
                key = str(skill_spec.get('pose', skill_spec.get('which', 'preassembly')))
                if key in {'preassembly', 'pre', 'grasp', 'start'}:
                    joint_positions = entry['arm_q_plug'][0]
                elif key in {'assembled', 'assembly', 'seat', 'end'}:
                    joint_positions = entry['arm_q_plug'][1]
                else:
                    raise ValueError(f'Unsupported Fabrica joint pose selector: {key!r}')
        if joint_positions is None:
            return None

        robot_state = tracked_robots.get(robot_name, {})
        current_position = _array_or_none(robot_state.get('position'))
        current_orientation = _array_or_none(robot_state.get('orientation'))
        if current_position is None or current_orientation is None:
            try:
                current_position, current_orientation = task._get_robot_eef_pose(robot_name)
            except Exception:
                return None
        joint_positions = np.asarray(joint_positions, dtype=np.float32)
        if joint_positions.shape[0] > 7:
            joint_positions = joint_positions[:7]
        action = OrderedDict()
        action[arm_ik_cfg.name] = [np.asarray(current_position, dtype=np.float32).tolist(), _normalize_quat(current_orientation).tolist()]
        action[arm_joint_cfg.name] = [joint_positions.tolist()]
        gripper_command = skill_spec.get('gripper_command')
        if gripper_command is not None:
            action[gripper_cfg.name] = [float(gripper_command)]
        elif skill_spec.get('gripper_ratio') is not None:
            action[gripper_cfg.name] = [float(np.clip(skill_spec['gripper_ratio'], 0.0, 1.0))]
        return action

    def _motion_joint_positions(self, *, task, skill_spec: dict) -> np.ndarray | None:
        motion_path = skill_spec.get('motion_path')
        motion_index = skill_spec.get('motion_index')
        if motion_path is None or motion_index is None:
            return None

        resolved = _resolve_existing_path(motion_path)
        key = str(resolved)
        if key not in self._motion_cache:
            _install_numpy_pickle_compat()
            with open(resolved, 'rb') as handle:
                self._motion_cache[key] = pickle.load(handle)

        motion = self._motion_cache[key]
        entry = motion[int(motion_index)]
        if len(entry) < 3 or entry[1] != 'arm':
            raise ValueError(f'Motion entry {motion_index} in {resolved} is not an arm trajectory.')

        trajectory = np.asarray(entry[2], dtype=np.float32)
        if trajectory.ndim == 1:
            return trajectory
        if trajectory.shape[0] == 0:
            return None

        if str(skill_spec.get('trajectory_endpoint', '')).lower() in {'last', 'end', 'final'}:
            return trajectory[-1]

        stride = max(int(skill_spec.get('trajectory_stride', 1)), 1)
        step_counter = int(getattr(task, 'phase_step_counter', 0))
        trajectory_index = min(step_counter * stride, trajectory.shape[0] - 1)
        return trajectory[trajectory_index]


def _array_or_none(value) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def _normalize_quat(quat) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def _clip_norm(vector, max_norm: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    max_norm = max(float(max_norm), 0.0)
    norm = float(np.linalg.norm(vector))
    if max_norm <= 0.0 or norm <= max_norm or norm <= 1e-8:
        return vector
    return vector * (max_norm / norm)


def _trailing_int(value) -> str | None:
    if value is None:
        return None
    match = re.search(r'(\d+)$', str(value))
    return match.group(1) if match else None


def _resolve_existing_path(path_like) -> Path:
    path = Path(str(path_like)).expanduser()
    if path.is_absolute():
        return path
    roots = [Path.cwd(), Path(__file__).resolve().parents[2]]
    for root in roots:
        candidate = (root / path).resolve()
        if candidate.exists():
            return candidate
    return (roots[0] / path).resolve()


def _install_numpy_pickle_compat():
    try:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric
    except Exception:
        return
    sys.modules.setdefault('numpy._core', numpy_core)
    sys.modules.setdefault('numpy._core.multiarray', numpy_multiarray)
    sys.modules.setdefault('numpy._core.numeric', numpy_numeric)


def _path_rotation(path_start: np.ndarray, path_end: np.ndarray, *, inverse: bool = False) -> np.ndarray:
    path_start = np.asarray(path_start, dtype=np.float32)
    path_end = np.asarray(path_end, dtype=np.float32)
    path_vector = path_end - path_start
    norm = float(np.linalg.norm(path_vector))
    if norm <= 1e-8:
        return np.eye(3, dtype=np.float32)
    path_direction = path_vector / norm
    target_direction = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    if inverse:
        v = np.cross(target_direction, path_direction)
        c = float(np.dot(target_direction, path_direction))
    else:
        v = np.cross(path_direction, target_direction)
        c = float(np.dot(path_direction, target_direction))
    s = float(np.linalg.norm(v))
    v_cross = np.asarray(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ],
        dtype=np.float32,
    )
    return np.eye(3, dtype=np.float32) + v_cross + (v_cross @ v_cross) * ((1.0 - c) / (s * s + 1e-8))


def _do_deltapos_path_transform(delta_position, path_start, path_end) -> np.ndarray:
    return _path_rotation(path_start, path_end, inverse=False) @ np.asarray(delta_position, dtype=np.float32)


def _undo_deltapos_path_transform(delta_transformed, path_start, path_end) -> np.ndarray:
    return _path_rotation(path_start, path_end, inverse=True) @ np.asarray(delta_transformed, dtype=np.float32)


class HardcodedFabricaInsertAdapter:
    """Deterministic payload insertion adapter for Fabrica fixed-layout demos."""

    def __init__(self, spec: dict[str, Any]):
        self.spec = dict(spec)

    def act(
        self,
        *,
        task,
        robot_name: str,
        phase_spec: dict,
        skill_spec: dict,
        tracked_robots: dict,
        tracked_objects: dict,
        checkpoint_path: str | None = None,
    ) -> dict | None:
        robot_state = tracked_robots.get(robot_name, {})
        current_position = _array_or_none(robot_state.get('position'))
        current_orientation = _array_or_none(robot_state.get('orientation'))
        if current_position is None or current_orientation is None:
            return None

        payload_object = skill_spec.get('held_object') or skill_spec.get('object') or skill_spec.get('payload_object')
        payload_target = skill_spec.get('held_target') or skill_spec.get('payload_target') or skill_spec.get('target')
        if not payload_object or not payload_target:
            return None

        payload_position = self._object_position(task, tracked_objects, str(payload_object))
        if payload_position is None:
            return None

        try:
            _, target_position, target_orientation, _ = task._resolve_target_pose_spec(payload_target)
        except Exception:
            return None
        target_position = _array_or_none(target_position)
        if target_position is None:
            return None

        delta = target_position - payload_position
        distance = float(np.linalg.norm(delta))
        coarse_distance = float(skill_spec.get('coarse_distance', 0.035))
        fine_distance = float(skill_spec.get('fine_distance', 0.012))
        coarse_step = float(skill_spec.get('coarse_step', 0.008))
        fine_step = float(skill_spec.get('fine_step', 0.0025))
        final_step = float(skill_spec.get('final_step', 0.0008))
        max_step = coarse_step if distance > coarse_distance else fine_step
        if distance <= fine_distance:
            max_step = final_step
        commanded_position = current_position + _clip_norm(delta, max_step)

        if bool(skill_spec.get('snap_when_close', False)) and distance <= float(skill_spec.get('snap_distance', 0.0015)):
            try:
                _, solved_position, solved_orientation, _ = task.resolve_robot_target_pose(
                    robot_name,
                    {
                        'target': phase_spec.get('robot_targets', {}).get(robot_name, {}).get('target'),
                        'payload_object': str(payload_object),
                        'payload_target': payload_target,
                    },
                )
                if solved_position is not None:
                    commanded_position = np.asarray(solved_position, dtype=np.float32)
                if solved_orientation is not None:
                    current_orientation = _normalize_quat(solved_orientation)
            except Exception:
                pass

        if target_orientation is not None and not bool(skill_spec.get('keep_current_orientation', True)):
            current_orientation = _normalize_quat(target_orientation)

        action = OrderedDict()
        action[arm_ik_cfg.name] = [commanded_position.tolist(), _normalize_quat(current_orientation).tolist()]
        action[gripper_cfg.name] = [0.0]
        return action

    @staticmethod
    def _object_position(task, tracked_objects: dict, object_name: str) -> np.ndarray | None:
        object_state = tracked_objects.get(object_name, {})
        position = _array_or_none(object_state.get('position'))
        if position is not None:
            return position
        try:
            return _array_or_none(task._resolve_object(object_name).get_pose()[0])
        except Exception:
            return None
