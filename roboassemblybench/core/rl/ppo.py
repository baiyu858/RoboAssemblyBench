"""Small CPU PPO pilot for the native single-environment validation bridge.

The checkpoint format is separate from old Fabrica FixPlug policies. A pilot
checkpoint is not a demonstrated successful insertion policy.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from roboassemblybench.core.rl.control import (
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    InsertionConfig,
)

FORMAT = 'roboassemblybench-contact-insertion-ppo-v2'


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 6)
        )
        self.critic = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.log_std = nn.Parameter(torch.full((6,), -1.0))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, math.sqrt(2.0))
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor[-1].weight, 0.01)
        nn.init.orthogonal_(self.critic[-1].weight, 1.0)

    def distribution(self, obs):
        return torch.distributions.Normal(self.actor(obs), self.log_std.clamp(-4.0, 1.0).exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def evaluate(self, obs, raw_action):
        dist = self.distribution(obs)
        # Numerically stable log determinant of tanh, including near saturation.
        log_det = 2.0 * (math.log(2.0) - raw_action - torch.nn.functional.softplus(-2.0 * raw_action))
        log_prob = (dist.log_prob(raw_action) - log_det).sum(-1)
        return log_prob, dist.entropy().sum(-1), self.value(obs)

    @torch.no_grad()
    def sample(self, obs):
        raw = self.distribution(obs).sample()
        log_prob, _, value = self.evaluate(obs, raw)
        return raw.tanh(), raw, log_prob, value

    @torch.no_grad()
    def act(self, observation):
        return self.actor(torch.as_tensor(observation, dtype=torch.float32)).tanh().cpu().numpy()


def generalized_advantage(rewards, values, next_values, terminated, done, gamma=0.99, lam=0.95):
    """Time limits bootstrap; neither timeout nor terminal leaks into next reset."""
    advantages = np.zeros(len(rewards), dtype=np.float32)
    trace = 0.0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * (1.0 - terminated[i]) * next_values[i] - values[i]
        trace = delta + gamma * lam * (1.0 - done[i]) * trace
        advantages[i] = trace
    return advantages, advantages + np.asarray(values, dtype=np.float32)


def train(env, *, steps, output_dir, qualification_path, snapshot_path, seed=4906, checkpoint=None):
    if steps < 1:
        raise ValueError('Training steps must be positive')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    qualification = json.loads(Path(qualification_path).read_text())
    reset_hash = hashlib.sha256(Path(snapshot_path).read_bytes()).hexdigest()
    if qualification.get('passed') is not True or qualification.get('snapshot_sha256') != reset_hash:
        raise ValueError('Training requires a passing native qualification for this reset snapshot')
    if qualification.get('controller_config') != asdict(env.config):
        raise ValueError('Re-run native qualification for this controller configuration')
    if qualification.get('torque_soft_limit') != getattr(env, 'torque_soft_limit', None):
        raise ValueError('Re-run native qualification for this torque_soft_limit')
    if qualification.get('torque_hard_limit') != getattr(env, 'torque_hard_limit', None):
        raise ValueError('Re-run native qualification for this torque_hard_limit')
    if qualification.get('torque_relief', False) != getattr(env, 'torque_relief', False):
        raise ValueError('Re-run native qualification for this torque_relief')
    if qualification.get('axis_alignment') != getattr(env, 'axis_alignment', None):
        raise ValueError('Re-run native qualification for this axis_alignment')
    if qualification.get('task_contract') != env.binding.policy_contract():
        raise ValueError('Qualification belongs to a different insertion binding; re-run qualification')
    model = ActorCritic()
    already_trained = 0
    payload = None
    if checkpoint is not None:
        payload = load_checkpoint(checkpoint, binding=env.binding)
        if payload['config'] != asdict(env.config):
            raise ValueError('Checkpoint and environment controller configurations differ')
        if payload.get('torque_soft_limit') != getattr(env, 'torque_soft_limit', None):
            raise ValueError('Checkpoint and environment torque_soft_limit differ')
        if payload.get('torque_hard_limit') != getattr(env, 'torque_hard_limit', None):
            raise ValueError('Checkpoint and environment torque_hard_limit differ')
        if payload.get('torque_relief', False) != getattr(env, 'torque_relief', False):
            raise ValueError('Checkpoint and environment torque_relief differ')
        if payload.get('axis_alignment') != getattr(env, 'axis_alignment', None):
            raise ValueError('Checkpoint and environment axis_alignment differ')
        model.load_state_dict(payload['model'])
        already_trained = payload['training_steps']
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    if payload is not None:
        if 'optimizer' in payload:
            optimizer.load_state_dict(payload['optimizer'])
        if 'torch_rng_state' in payload:
            torch.set_rng_state(payload['torch_rng_state'])
        if 'numpy_rng_state_json' in payload:
            rng.bit_generator.state = json.loads(payload['numpy_rng_state_json'])
    obs, _ = env.reset(seed=seed)
    collected = 0
    episode_return = 0.0
    episode_count = 0
    local_successes = 0
    history = []
    while collected < steps:
        buf = {k: [] for k in ('obs', 'raw', 'logp', 'value', 'reward', 'next_value', 'terminated', 'done')}
        for _ in range(min(256, steps - collected)):
            tensor = torch.as_tensor(obs, dtype=torch.float32)
            action, raw, logp, value = model.sample(tensor)
            next_obs, reward, terminated, truncated, info = env.step(action.numpy())
            with torch.no_grad():
                next_value = model.value(torch.as_tensor(next_obs, dtype=torch.float32)).item()
            for key, item in dict(
                obs=obs,
                raw=raw.numpy(),
                logp=logp.item(),
                value=value.item(),
                reward=reward,
                next_value=next_value,
                terminated=float(terminated),
                done=float(terminated or truncated),
            ).items():
                buf[key].append(item)
            collected += 1
            episode_return += reward
            obs = next_obs
            if terminated or truncated:
                episode_count += 1
                local_successes += int(info['local_seated'])
                with (output / 'training_episodes.jsonl').open('a') as handle:
                    handle.write(
                        json.dumps(
                            dict(
                                training_step=already_trained + collected,
                                episode=episode_count,
                                reward=episode_return,
                                **info,
                            )
                        )
                        + '\n'
                    )
                episode_return = 0.0
                obs, _ = env.reset()
        advantages, returns = generalized_advantage(
            buf['reward'], buf['value'], buf['next_value'], buf['terminated'], buf['done']
        )
        advantage = torch.as_tensor(advantages)
        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
        tensors = {k: torch.as_tensor(np.asarray(buf[k]), dtype=torch.float32) for k in ('obs', 'raw', 'logp')}
        target = torch.as_tensor(returns)
        losses = []
        for _ in range(4):
            order = rng.permutation(len(advantages))
            for start in range(0, len(order), 64):
                index = order[start : start + 64]
                logp, entropy, value = model.evaluate(tensors['obs'][index], tensors['raw'][index])
                ratio = (logp - tensors['logp'][index]).exp()
                clipped = ratio.clamp(0.8, 1.2) * advantage[index]
                actor_loss = -torch.minimum(ratio * advantage[index], clipped).mean()
                value_loss = 0.5 * (value - target[index]).square().mean()
                loss = actor_loss + 0.5 * value_loss - 0.001 * entropy.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError('PPO produced a non-finite loss')
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                losses.append(float(loss.detach()))
        report = dict(
            training_steps=already_trained + collected,
            episodes=episode_count,
            local_held_successes=local_successes,
            mean_loss=float(np.mean(losses)),
            scope='single_env_pilot',
            benchmark_success=False,
        )
        history.append(report)
        print('[insertion-ppo] ' + json.dumps(report), flush=True)
        payload = dict(
            format=FORMAT,
            observation_version=OBSERVATION_VERSION,
            observation_size=OBSERVATION_SIZE,
            action_size=6,
            config=asdict(env.config),
            model=model.state_dict(),
            torque_soft_limit=getattr(env, 'torque_soft_limit', None),
            torque_hard_limit=getattr(env, 'torque_hard_limit', None),
            torque_relief=getattr(env, 'torque_relief', False),
            axis_alignment=getattr(env, 'axis_alignment', None),
            task_contract=env.binding.policy_contract(),
            optimizer=optimizer.state_dict(),
            torch_rng_state=torch.get_rng_state(),
            numpy_rng_state_json=json.dumps(rng.bit_generator.state),
            training_steps=already_trained + collected,
            snapshot_sha256=reset_hash,
            qualification_sha256=hashlib.sha256(Path(qualification_path).read_bytes()).hexdigest(),
            deployment_validated=False,
            report=report,
        )
        temporary = output / 'policy.tmp'
        torch.save(payload, temporary)
        temporary.replace(output / 'policy.pt')
        (output / 'training_summary.json').write_text(json.dumps(history, indent=2))
    return output / 'policy.pt'


def load_checkpoint(path, *, binding=None):
    from . import legacy

    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    old = payload.get('format') == legacy.CHECKPOINT_FORMAT
    expected_observation = legacy.OBSERVATION_VERSION if old else OBSERVATION_VERSION
    if (
        payload.get('format') not in (FORMAT, legacy.CHECKPOINT_FORMAT)
        or payload.get('observation_version') != expected_observation
        or payload.get('observation_size') != OBSERVATION_SIZE
        or payload.get('action_size') != 6
        or payload.get('training_steps', 0) < 1
    ):
        raise ValueError('Not a trained contact insertion checkpoint')
    if payload.get('config', {}).get('version') != InsertionConfig().version:
        raise ValueError('Checkpoint uses an older controller; do not mix controller versions')
    InsertionConfig(**payload['config'])
    from .control import validate_torque_soft_limit

    validate_torque_soft_limit(payload.get('torque_soft_limit'), InsertionConfig(**payload['config']))
    from .control import validate_torque_hard_limit

    validate_torque_hard_limit(
        payload.get('torque_hard_limit'), InsertionConfig(**payload['config']), payload.get('torque_soft_limit')
    )
    from .torque_relief import relief_enabled

    relief_enabled(payload.get('torque_relief', False), payload.get('torque_soft_limit'))
    if old:
        payload = dict(payload, task_contract=legacy.legacy_task().policy_contract(), legacy_artifact=True)
    if not isinstance(payload.get('task_contract'), dict):
        raise ValueError('Checkpoint has no insertion task contract')
    if binding is not None and payload['task_contract'] != binding.policy_contract():
        raise ValueError('Checkpoint task contract differs from requested insertion; train a policy for this binding')
    return payload


def load_policy(path, *, binding):
    payload = load_checkpoint(path, binding=binding)
    model = ActorCritic()
    model.load_state_dict(payload['model'], strict=True)
    model.eval()
    model.torque_soft_limit = payload.get('torque_soft_limit')
    from .axis_alignment import alignment_settings

    model.axis_alignment = alignment_settings(payload.get('axis_alignment'))
    from .torque_relief import relief_enabled

    model.torque_relief = relief_enabled(payload.get('torque_relief', False), model.torque_soft_limit)
    model.torque_hard_limit = payload.get('torque_hard_limit')
    return model, InsertionConfig(**payload['config'])
