from types import SimpleNamespace

from roboassemblybench.scripts import validate_fabrica_canonical_ur5e as validation


def test_worker_command_uses_one_seed_for_a_multi_recipe_queue(monkeypatch, tmp_path):
    monkeypatch.setattr(validation.shutil, 'which', lambda name: '/conda')
    args = SimpleNamespace(
        conda_env='env',
        num_threads=2,
        seed=4,
        layout_seed=9,
        scene_profile='profile',
        output_dir=tmp_path,
        rendering_interval=2399,
        record_live_video=False,
        live_video_fps=30,
        live_video_frame_stride=8,
        constraint_check_stride=32,
        stage_precheck_stride=128,
        stage_precheck_waypoints=4,
    )

    command = validation._worker_command(args, ['beam', 'car'], tmp_path / 'results.json')

    assert command[command.index('--worker-recipes') + 1 : command.index('--worker-seeds')] == ['beam', 'car']
    assert command[command.index('--worker-seeds') + 1] == '4'
    assert '--skip-episode-steps' in command
    assert command[command.index('--worker-rendering-interval') + 1] == '2399'
    assert '--runtime-constraint-monitor' in command
    assert '--stage-trajectory-precheck' in command


def test_worker_command_enables_video_without_rendering_override(monkeypatch, tmp_path):
    monkeypatch.setattr(validation.shutil, 'which', lambda name: '/conda')
    args = SimpleNamespace(
        conda_env='env',
        num_threads=2,
        seed=0,
        layout_seed=0,
        scene_profile='profile',
        output_dir=tmp_path,
        rendering_interval=2399,
        record_live_video=True,
        live_video_fps=24,
        live_video_frame_stride=12,
        constraint_check_stride=32,
        stage_precheck_stride=128,
        stage_precheck_waypoints=4,
    )

    command = validation._worker_command(args, ['beam'], tmp_path / 'results.json')

    assert '--record-live-video' in command
    assert command[command.index('--live-video-fps') + 1] == '24'
    assert command[command.index('--live-video-frame-stride') + 1] == '12'
    assert '--worker-rendering-interval' not in command
    assert '--skip-episode-steps' in command


def test_conda_executable_falls_back_to_current_environment(monkeypatch, tmp_path):
    monkeypatch.delenv('CONDA_EXE', raising=False)
    monkeypatch.setattr(validation.shutil, 'which', lambda name: None)
    fake_prefix = tmp_path / 'envs' / 'internutopia311'
    conda = fake_prefix.parent.parent / 'bin' / 'conda'
    conda.parent.mkdir(parents=True)
    conda.touch()
    conda.chmod(0o755)
    monkeypatch.setattr(validation.sys, 'prefix', str(fake_prefix))

    assert validation._conda_executable() == str(conda)


def test_summary_identifies_terminal_phase_and_collision_phase():
    result = {
        'recipe': 'beam',
        'seed': 0,
        'success': False,
        'failed': True,
        'steps': 20,
        'terminal_reason': 'timeout-failure',
        'phase_status': 'failed',
        'phase_history': ['pick', 'place', 'failed'],
        'phase_transition_history': [
            {'from_phase': 'pick', 'to_phase': 'place', 'step_counter': 10, 'to_phase_index': 1},
            {'from_phase': 'place', 'from_phase_index': 1, 'to_phase': 'failed', 'step_counter': 20},
        ],
        'runtime_constraint_monitor': {
            'checks': 2,
            'violation_total': 1,
            'events': [{'step': 12}],
        },
        'stage_trajectory_precheck': {'checks': 1, 'violation_total': 0, 'events': []},
    }

    summary = validation._summarize_results(
        requested_recipes=['beam'],
        results=[result],
        worker_exit_code=0,
        resource_abort=None,
        resource_usage={},
    )

    task = summary['tasks'][0]
    assert task['terminal_phase'] == 'place'
    assert task['terminal_phase_index'] == 1
    assert task['runtime_collisions']['violations_by_phase'] == {'place': 1}
