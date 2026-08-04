import socket
import json
from pathlib import Path

import numpy as np
import pytest

from roboassemblybench.policies.act_rpc import receive_message, send_message
from roboassemblybench.scripts.aggregate_fabrica_plumbers_block_act_eval import aggregate_evaluation
from roboassemblybench.scripts.evaluate_fabrica_plumbers_block_act import (
    _env_action,
    apply_max_steps_override,
    sanitize_absolute_cartesian_action,
)
from roboassemblybench.scripts.serve_fabrica_plumbers_block_act import _portable_action_payload


def _state() -> np.ndarray:
    state = np.zeros(16, dtype=np.float32)
    state[3] = 1.0
    state[11] = 1.0
    state[7] = 0.5
    state[15] = 0.5
    return state


def test_policy_rpc_round_trip():
    sender, receiver = socket.socketpair()
    try:
        payload = {'command': 'predict', 'state': np.arange(16, dtype=np.float32)}
        send_message(sender, payload)
        received = receive_message(receiver)
    finally:
        sender.close()
        receiver.close()

    assert received['command'] == 'predict'
    np.testing.assert_array_equal(received['state'], payload['state'])


def test_policy_rpc_action_payload_uses_only_portable_python_scalars():
    action = np.arange(16, dtype=np.float32)
    payload = _portable_action_payload(action)

    assert isinstance(payload, list)
    assert len(payload) == 16
    assert all(isinstance(value, float) for value in payload)

    sender, receiver = socket.socketpair()
    try:
        send_message(sender, {'ok': True, 'action': payload})
        received = receive_message(receiver)
    finally:
        sender.close()
        receiver.close()

    assert received['action'] == payload


def test_cartesian_action_limits_translation_rotation_and_gripper():
    current = _state()
    target = current.copy()
    target[0:3] = [1.0, 0.0, 0.0]
    target[3:7] = [0.0, 1.0, 0.0, 0.0]
    target[7] = -2.0
    target[8:11] = [0.0, -2.0, 0.0]
    target[11:15] = [0.0, 0.0, 1.0, 0.0]
    target[15] = 3.0

    bounded = sanitize_absolute_cartesian_action(
        target,
        current,
        max_translation_step=0.04,
        max_rotation_step=0.35,
    )

    assert np.linalg.norm(bounded[0:3] - current[0:3]) == pytest.approx(0.04)
    assert np.linalg.norm(bounded[8:11] - current[8:11]) == pytest.approx(0.04)
    assert np.linalg.norm(bounded[3:7]) == pytest.approx(1.0)
    assert np.linalg.norm(bounded[11:15]) == pytest.approx(1.0)
    assert bounded[7] == 0.0
    assert bounded[15] == 1.0

    env_action = _env_action(bounded)
    assert set(env_action) == {'franka_left', 'franka_right'}
    assert len(env_action['franka_left']['arm_ik_controller'][0]) == 3
    assert len(env_action['franka_right']['arm_ik_controller'][1]) == 4


def test_cartesian_action_rejects_nonfinite_output():
    action = _state()
    action[0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        sanitize_absolute_cartesian_action(
            action,
            _state(),
            max_translation_step=0.04,
            max_rotation_step=0.35,
        )


def test_online_evaluation_can_override_episode_limit_for_smoke_runs():
    configs = [type('TaskConfig', (), {'max_steps': 18000})() for _ in range(2)]

    apply_max_steps_override(configs, 200)

    assert [config.max_steps for config in configs] == [200, 200]
    with pytest.raises(ValueError, match='positive'):
        apply_max_steps_override(configs, 0)


def _write_eval_episode(
    root: Path,
    *,
    index: int,
    seed: int,
    success: bool,
    layout_seed: int | None = None,
) -> None:
    layout_seed = seed if layout_seed is None else int(layout_seed)
    episode_dir = root / f'episode_{index:04d}_seed_{seed:06d}'
    episode_dir.mkdir(parents=True)
    (episode_dir / 'episode_results.json').write_text(
        json.dumps([{'seed': seed, 'layout_seed': layout_seed, 'success': success}]),
        encoding='utf-8',
    )
    (episode_dir / 'success_rate.json').write_text(
        json.dumps(
            {
                'complete': True,
                'num_episodes': 1,
                'start_seed': seed,
                'policy_server': {'policy_type': 'act'},
            }
        ),
        encoding='utf-8',
    )


def test_online_evaluation_aggregates_unique_isolated_episodes(tmp_path: Path):
    episodes_dir = tmp_path / 'episodes'
    _write_eval_episode(episodes_dir, index=0, seed=100, success=True)
    _write_eval_episode(episodes_dir, index=1, seed=101, success=False)

    summary = aggregate_evaluation(
        episodes_dir=episodes_dir,
        output_dir=tmp_path,
        expected_episodes=2,
        start_seed=100,
    )

    assert summary['complete'] is True
    assert summary['num_successes'] == 1
    assert summary['success_rate'] == 0.5
    assert summary['single_isaac_process_per_episode'] is True
    results = json.loads((tmp_path / 'episode_results.json').read_text(encoding='utf-8'))
    assert [item['seed'] for item in results] == [100, 101]


def test_online_evaluation_aggregation_rejects_missing_seed(tmp_path: Path):
    episodes_dir = tmp_path / 'episodes'
    _write_eval_episode(episodes_dir, index=0, seed=100, success=True)
    with pytest.raises(ValueError, match='missing='):
        aggregate_evaluation(
            episodes_dir=episodes_dir,
            output_dir=tmp_path,
            expected_episodes=2,
            start_seed=100,
        )


def test_online_evaluation_enforces_fixed_layout_seed_contract(tmp_path: Path):
    episodes_dir = tmp_path / 'episodes'
    _write_eval_episode(
        episodes_dir,
        index=0,
        seed=100,
        layout_seed=4906,
        success=True,
    )
    _write_eval_episode(
        episodes_dir,
        index=1,
        seed=101,
        layout_seed=17,
        success=False,
    )

    with pytest.raises(ValueError, match='layout seed mismatch'):
        aggregate_evaluation(
            episodes_dir=episodes_dir,
            output_dir=tmp_path,
            expected_episodes=2,
            start_seed=100,
            layout_seeds=[4906, 485, 34, 12],
        )
