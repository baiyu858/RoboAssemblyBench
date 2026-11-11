"""Native qualification, kept separate from reward/episode success."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class QualificationConfig:
    reset_trials: int = 3
    hold_decisions: int = 60
    probe_decisions: int = 80
    probe_action: float = 0.5
    probe_contact_N: float = 2.0
    dwell_decisions: int = 24
    retreat_decisions: int = 40
    lead_in_decisions: int = 0
    align_decisions: int = 0
    probe_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self):
        for field in ('reset_trials', 'hold_decisions', 'probe_decisions', 'dwell_decisions', 'retreat_decisions'):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(f'{field} must be a positive integer')
        if not 0 < self.probe_action <= 1 or not np.isfinite(self.probe_contact_N) or self.probe_contact_N <= 0:
            raise ValueError('Invalid qualification probe action/force')
        for field in ('lead_in_decisions', 'align_decisions'):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise ValueError(f'{field} must be a nonnegative integer')
        axis = np.asarray(self.probe_axis, dtype=float)
        if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) < 1e-9:
            raise ValueError('probe_axis must be a finite nonzero three-vector')
        object.__setattr__(self, 'probe_axis', tuple((axis / np.linalg.norm(axis)).tolist()))


def qualify(env, snapshot_path, output_path, *, config=None):
    config = config or QualificationConfig()
    output = Path(output_path)
    records = []
    holds = []
    contact = []
    relief = []
    alignment, lead_in = [], []
    info = None
    failure = None
    try:
        # Repeated reset is part of qualification: warm-start forces cannot be
        # faked by restoring an old contact buffer or grasp-valid boolean.
        for trial in range(config.reset_trials):
            _, info = env.reset(seed=4906 + trial, options={'disable_guidance': True})
            for _ in range(config.hold_decisions):
                _, _, terminal, truncated, info = env.step(np.zeros(6))
                holds.append(info)
                if terminal or truncated:
                    raise RuntimeError(info.get('failure') or 'Hold test terminated unexpectedly')
            records.append({'test': 'bilateral_hold', 'trial': trial, 'passed': True, 'last': info})
        env.reset(options={'disable_guidance': True})
        if config.align_decisions:
            # Resolve the coarse precontact entry error before a straight
            # unlearned probe. No pose writes or safety bypasses are involved.
            env.controller.guidance = True
            for _ in range(config.align_decisions):
                _, _, terminal, truncated, info = env.step(np.zeros(6))
                alignment.append(info)
                if terminal or truncated:
                    raise RuntimeError(info.get('failure') or 'Precontact alignment terminated unexpectedly')
            env.controller.guidance = False
        # Open sockets need a bounded lead-in before a lateral sidewall probe.
        # The ordinary physical force/contact/slip guards remain active.
        for _ in range(config.lead_in_decisions):
            action = np.zeros(6)
            action[2] = config.probe_action
            _, _, terminal, truncated, info = env.step(action)
            lead_in.append(info)
            if terminal or truncated:
                raise RuntimeError(info.get('failure') or 'Probe lead-in terminated unexpectedly')
        # Probe along the configured axis, then withdraw from contact. This
        # validates compliance separately from assembly success.
        for _ in range(config.probe_decisions):
            action = np.zeros(6)
            action[:3] = config.probe_action * np.asarray(config.probe_axis)
            _, _, terminal, truncated, info = env.step(action)
            contact.append(info)
            if terminal or truncated:
                raise RuntimeError(info.get('failure') or 'Contact test terminated unexpectedly')
            if np.linalg.norm(info['wrench_world'][:3]) >= config.probe_contact_N:
                break
        for _ in range(config.dwell_decisions):
            _, _, terminal, truncated, info = env.step(np.zeros(6))
            contact.append(info)
            if terminal or truncated:
                raise RuntimeError(info.get('failure') or 'Compliance dwell terminated unexpectedly')
        if config.align_decisions:
            # A long pin must retain lateral/angular alignment while backing
            # out. Do not add the guide's forward feed against the backoff.
            env.controller.guidance = True
            env.controller.alignment_only = True
        for _ in range(config.retreat_decisions):
            action = np.zeros(6)
            action[2] = -config.probe_action
            _, _, terminal, truncated, info = env.step(action)
            relief.append(info)
            if terminal or truncated:
                raise RuntimeError(info.get('failure') or 'Retreat test terminated unexpectedly')
    except (ValueError, RuntimeError, KeyError) as exc:
        failure = str(exc)
    excitation = lead_in + contact
    peak = max((np.linalg.norm(x['wrench_world'][:3]) for x in excitation), default=0.0)
    last_force = np.linalg.norm(relief[-1]['wrench_world'][:3]) if relief else None
    response = max((float(np.dot(x['wrench_world'][:3], x['admittance_offset'][:3])) for x in excitation), default=0.0)
    # Offset response alone validates the force-feedback implementation; native
    # force reduction during backoff additionally checks the mechanical sign.
    checks = {
        'repeated_real_contact_holds': len(records) == config.reset_trials,
        'real_part_base_contact_excited': bool(peak >= 0.5),
        'admittance_responds_along_reaction_force': response > 1e-7,
        'backoff_reduces_measured_contact_force': bool(
            last_force is not None and peak >= 0.5 and last_force < max(0.5, 0.8 * peak)
        ),
        'no_safety_failure': failure is None,
    }
    result = dict(
        passed=all(checks.values()),
        checks=checks,
        failure=failure,
        failure_detail=info if failure else None,
        controller_config=asdict(env.config),
        torque_soft_limit=getattr(env, 'torque_soft_limit', None),
        torque_hard_limit=getattr(env, 'torque_hard_limit', None),
        torque_relief=getattr(env, 'torque_relief', False),
        axis_alignment=getattr(env, 'axis_alignment', None),
        task_contract=env.binding.policy_contract(),
        qualification_config=asdict(config),
        snapshot_sha256=hashlib.sha256(Path(snapshot_path).read_bytes()).hexdigest(),
        scope='real_grasp_and_outer_admittance_only',
        benchmark_success=False,
        peak_insertion_force_N=float(peak),
        final_retreat_force_N=None if last_force is None else float(last_force),
        max_force_offset_dot_Nm=response,
        holds=records,
        alignment=alignment,
        lead_in=lead_in,
        contact_probe=contact,
        retreat=relief,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(
        '[insertion-qualification] '
        + json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ('holds', 'alignment', 'lead_in', 'contact_probe', 'retreat', 'failure_detail')
            }
        ),
        flush=True,
    )
    return result


def evaluate(env, model, *, episodes, output_path):
    results = []
    for i in range(episodes):
        obs, _ = env.reset(seed=10000 + i)
        terminal = truncated = False
        reward_sum = 0.0
        while not (terminal or truncated):
            obs, reward, terminal, truncated, info = env.step(model.act(obs))
            reward_sum += reward
            if info['policy_steps'] % 4 == 0 or terminal or truncated:
                with (Path(output_path).parent / 'evaluation_trace.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(episode=i, **info)) + '\n')
        results.append(dict(episode=i, reward=reward_sum, **info))
        print(f'[insertion-eval] episode={i} local_seated={info["local_seated"]} failure={info["failure"]}', flush=True)
        Path(output_path).with_name('evaluation_episodes.json').write_text(json.dumps(results, indent=2))
    payload = dict(
        episodes=results,
        local_held_successes=sum(int(x['local_seated']) for x in results),
        episode_count=episodes,
        benchmark_success=False,
        scope='local_held_insertion; release and unloaded stability still required',
    )
    Path(output_path).write_text(json.dumps(payload, indent=2))
    return payload
