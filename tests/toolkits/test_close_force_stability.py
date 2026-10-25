"""Noise tolerance must not hide missing contact, raw faults or stale samples."""
import pytest

from toolkits.factory_dual_franka_assembly.close_force_stability import (
    close_force_qualification, close_force_stability_settings,
)


CONFIG = {'enabled': True, 'window_steps': 5, 'hysteresis_margin': .5}


def sample(state, step, forces=(4.5, 5.), **kwargs):
    return close_force_qualification(state=state, config=CONFIG, step=step, forces=forces,
        entry_force=4., high_force=12., contact_threshold=.1,
        probes_valid=kwargs.get('probes_valid', True), contact_ready=kwargs.get('contact_ready', True))


def test_requires_full_fresh_window_and_raw_and_average_entry_force():
    state = {}
    for step in range(4):
        assert not sample(state, step)['qualified']
    assert sample(state, 4)['qualified']
    # Constant sub-entry force cannot create a new accepted grasp.
    state = {}
    for step in range(100):
        assert not sample(state, step, (3.9, 5.))['qualified']


def test_noise_near_entry_does_not_drop_an_established_force_band():
    state = {}
    for step in range(5):
        sample(state, step)
    for step in range(5, 110):
        r = sample(state, step, (3.7 if step % 2 else 4.3, 5.))
        assert r['qualified']
        assert r['release_force'] == 3.5
        assert r['window_samples'] == 5


def test_latest_raw_force_must_also_reach_entry_even_if_mean_is_high():
    state = {}
    for step in range(4):
        assert not sample(state, step, (6., 6.))['qualified']
    assert not sample(state, 4, (3.9, 6.))['qualified']
    assert sample(state, 5, (4.1, 6.))['qualified']


@pytest.mark.parametrize('forces,options,reason', [
    ((0., 6.), {}, 'raw_force_below_release'),
    ((3.49, 6.), {}, 'raw_force_below_release'),
    ((12.01, 6.), {}, 'raw_force_above_high'),
    ((50., 6.), {}, 'raw_force_above_high'),
    ((6., 6.), {'probes_valid': False}, 'invalid_probe'),
    ((float('nan'), 6.), {}, 'invalid_probe'),
    ((6., 6.), {'contact_ready': False}, 'target_contact_not_ready'),
])
def test_raw_faults_clear_history_immediately_despite_good_smoothed_force(forces, options, reason):
    state = {}
    for step in range(5):
        sample(state, step)
    r = sample(state, 5, forces, **options)
    assert not r['qualified'] and r['reason'] == reason
    assert state['samples'] == [] and state['latched'] is False
    for step in range(6, 10):
        assert not sample(state, step)['qualified']
    assert sample(state, 10)['qualified']


def test_same_step_is_not_a_new_sample_and_time_gap_discards_history():
    state = {}
    sample(state, 0)
    for _ in range(10):
        r = sample(state, 0)
        assert r['window_samples'] == 1 and not r['new_sample']
    for step in range(1, 5):
        sample(state, step)
    r = sample(state, 7)
    assert not r['qualified'] and r['restart'] and r['window_samples'] == 1


@pytest.mark.parametrize('config', [None, {}, {'enabled': 1}, {'enabled': True, 'extra': 1},
    {'enabled': True, 'window_steps': True}, {'enabled': True, 'window_steps': 0},
    {'enabled': True, 'window_steps': 33}, {'enabled': True, 'window_steps': 5.0},
    {'enabled': True, 'hysteresis_margin': -1.}, {'enabled': True, 'hysteresis_margin': True},
    {'enabled': True, 'hysteresis_margin': float('nan')},
    {'enabled': True, 'hysteresis_margin': float('inf')}])
def test_bad_settings_fail_closed(config):
    with pytest.raises(ValueError, match='close_force_stability'):
        close_force_stability_settings(config)


def test_release_threshold_cannot_hide_loss_of_real_contact():
    with pytest.raises(ValueError, match='release'):
        close_force_qualification(state={}, config={**CONFIG, 'hysteresis_margin': 4.}, step=1,
            forces=(6., 6.), probes_valid=True, contact_ready=True,
            entry_force=4., high_force=12., contact_threshold=.1)
