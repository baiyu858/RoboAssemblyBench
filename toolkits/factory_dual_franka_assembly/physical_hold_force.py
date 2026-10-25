"""Bounded, weak-side force maintenance for an already registered grasp.

This controller never moves the arm, changes a grasp reference, or declares a
grasp successful. The caller gates motion on ``motion_ready`` and retains the
ordinary target-contact, 3 mm and 5 degree checks. Forces are in N, joint travel
in radians, and slip in metres. Defaults are simulation experiment parameters.
"""
from __future__ import annotations

import math


DEFAULTS = {
    'low': 6.0,
    'target': 8.0,
    'soft_limit': 10.0,
    'unload': 12.0,
    'hard_limit': 40.0,
    'soft_limit_motion_scale': 0.5,
    'joint_rate': 0.025,
    'unload_rate': 0.025,
    'max_command_lead': 0.001,
    'max_seek': 0.03,
    'max_closing_travel': 0.03,
    'filter_tau': 0.02,
    'stable_steps': 24,
    'resume_steps': 48,
    'recovery_timeout_steps': 480,
    'slip_window_steps': 48,
    'slip_growth': 0.0001,
}
_COUNTS = {'stable_steps', 'resume_steps', 'recovery_timeout_steps', 'slip_window_steps'}


def hold_force_settings(config: dict) -> dict:
    """Validate the opt-in configuration at compilation and runtime."""
    if not isinstance(config, dict) or config.get('enabled') is not True:
        raise ValueError('hold_force_control must be an enabled mapping.')
    unknown = set(config) - set(DEFAULTS) - {'enabled', 'soft_limit_action'}
    if unknown:
        raise ValueError(f'Unknown hold_force_control settings: {sorted(unknown)}')
    result = {'enabled': True}
    for key, default in DEFAULTS.items():
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'hold_force_control.{key} must be a positive finite number.')
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'hold_force_control.{key} must be positive and finite.')
        if key in _COUNTS and int(value) != value:
            raise ValueError(f'hold_force_control.{key} must be an integer.')
        result[key] = int(value) if key in _COUNTS else float(value)
    if not (result['low'] < result['target'] < result['soft_limit']
            < result['unload'] < result['hard_limit']):
        raise ValueError('hold_force_control requires low < target < soft_limit < unload < hard_limit.')
    if result['recovery_timeout_steps'] <= result['stable_steps']:
        raise ValueError('hold_force_control recovery timeout must exceed its stability window.')
    if result['soft_limit_motion_scale'] > 1.0:
        raise ValueError('hold_force_control.soft_limit_motion_scale must be in (0, 1].')
    action = config.get('soft_limit_action', 'pause')
    if action not in ('pause', 'limit_and_slow'):
        raise ValueError('hold_force_control.soft_limit_action must be pause or limit_and_slow.')
    result['soft_limit_action'] = action
    return result


def hold_force_command(*, state: dict, config: dict, forces, probes_valid: bool,
                       gripper_q: float, open_q: float, closed_q: float,
                       min_openness: float, step: int, dt: float, position_slip: float,
                       contact_threshold: float = 0.1) -> dict:
    """Advance at most once per physics step, never integrating missed time.

    A trigger below ``low`` latches preload until the weak side reaches
    ``target``. The strong-side ceiling and raw overload checks take priority.
    The opt-in limit_and_slow policy separates mild force regulation from the
    arm's motion gate. Neither policy permits motion on sustained slip, weak
    contact, invalid probes or overload, or raises the target/resets slip.
    """
    c = hold_force_settings(config)
    if state.get('step') == step:
        return state['last_control']
    if state.get('step', -1) > step:
        raise ValueError('hold force control cannot reuse a previous simulation step.')
    numeric = [gripper_q, open_q, closed_q, min_openness, dt, position_slip, contact_threshold]
    if not all(math.isfinite(x) for x in numeric) or dt <= 0 or position_slip < 0:
        raise ValueError('Invalid hold force state, timestep or slip measurement.')
    if not 0 <= min_openness < 1 or contact_threshold <= 0 or abs(closed_q - open_q) < 1e-8:
        raise ValueError('Invalid hold force gripper limits.')
    span = abs(closed_q - open_q)
    direction = math.copysign(1.0, closed_q - open_q)
    measured = (gripper_q - open_q) * direction
    maximum = span * (1.0 - min_openness)
    anchor = float(state.setdefault('contact_capture_closure', measured))
    previous = float(state.get('contact_command_closure', anchor))
    if not all(math.isfinite(x) for x in (previous, anchor)) or not 0 <= anchor <= maximum:
        raise ValueError('Invalid original hold force command anchor.')
    if not 0 <= previous <= maximum:
        raise ValueError('Hold force command is outside safe gripper travel.')
    valid = bool(probes_valid and len(forces) == 2
                 and all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in forces))
    raw = [float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None for x in forces]
    finite_forces = [x for x in raw if x is not None]
    peak = max(finite_forces, default=0.0)
    gap = state.get('step') is not None and step != state['step'] + 1
    filtered = state.get('filtered_forces')
    if valid:
        alpha = -math.expm1(-dt / c['filter_tau'])
        filtered = raw[:] if filtered is None or gap else [
            old + alpha * (now - old) for old, now in zip(filtered, raw)]
        state['filtered_forces'] = filtered
    if gap:
        state.update(stable_steps=0, resume_count=0, preloading=True, slip_history=[])
    history = state.setdefault('slip_history', [])
    history.append((step, position_slip))
    while len(history) > 1 and history[1][0] <= step - c['slip_window_steps']:
        history.pop(0)
    growth = position_slip - history[0][1]
    slipping = bool(step - history[0][0] >= c['slip_window_steps'] and growth >= c['slip_growth'])
    preloading = bool(state.get('preloading', True))
    weak = min(filtered) if valid else 0.0
    strong = max(peak, max(filtered)) if valid else peak
    if valid and weak < c['low']:
        preloading = True
    if valid and weak >= c['target'] - 1e-9:
        preloading = False

    command, failure = previous, None
    if peak > c['hard_limit']:
        mode, failure = 'force_limit', 'loaded_gripper_force_limit'
    elif not valid:
        mode, failure = 'invalid_probe', 'loaded_force_probe_invalid'
    elif min(raw) < contact_threshold:
        mode, failure = 'contact_missing', 'loaded_force_contact_missing'
    elif peak > c['unload']:
        command = max(0.0, min(previous, measured) - c['unload_rate'] * dt)
        mode, preloading = 'unload', True
    elif strong >= c['soft_limit']:
        mode = 'force_limited'
        if (c['soft_limit_action'] == 'limit_and_slow' and weak >= c['target']
                and min(raw) >= c['low'] and not slipping):
            # Mild overload: relieve slowly without discarding the last command
            # or forcing the arm to stop. Do not unload a weak/slipping grasp.
            command = max(0.0, previous - c['unload_rate'] * dt)
            mode = 'force_relief'
    elif preloading:
        remaining = max(0.0, c['max_closing_travel'] - state.get('closing_travel', 0.0))
        # Do not withdraw a published preload just because measured FK lags.
        command = max(previous, min(previous + c['joint_rate'] * dt,
                                    measured + c['max_command_lead'], anchor + c['max_seek'],
                                    previous + remaining, maximum))
        mode = 'preload'
        if command == previous and (remaining <= 1e-12 or previous >= min(anchor + c['max_seek'], maximum)):
            mode, failure = 'travel_limit', 'loaded_force_travel_limit'
    else:
        mode = 'slip_hold' if slipping else 'hold'
    state['closing_travel'] = state.get('closing_travel', 0.0) + max(0.0, command - previous)
    state['contact_command_closure'] = command
    state['preloading'] = preloading
    continue_on_soft_limit = c['soft_limit_action'] == 'limit_and_slow'
    motion_modes = ('hold', 'force_limited', 'force_relief') if continue_on_soft_limit else ('hold',)
    motion_force_limit = c['unload'] if continue_on_soft_limit else c['soft_limit']
    stable = bool(valid and failure is None and mode in motion_modes and not preloading
                  and not slipping and weak >= c['low'] and strong < motion_force_limit)
    state['stable_steps'] = state.get('stable_steps', 0) + 1 if stable else 0
    ready = state['stable_steps'] >= c['stable_steps']
    state['paused_steps'] = 0 if ready else state.get('paused_steps', 0) + 1
    if failure is None and state['paused_steps'] >= c['recovery_timeout_steps']:
        failure = 'loaded_force_recovery_timeout'
        ready = False
    state['resume_count'] = min(c['resume_steps'], state.get('resume_count', 0) + 1) if ready else 0
    x = state['resume_count'] / c['resume_steps']
    force_scale = 1.0
    if continue_on_soft_limit and strong >= c['soft_limit']:
        excess = min(1.0, (strong - c['soft_limit']) / (c['unload'] - c['soft_limit']))
        force_scale -= excess * (1.0 - c['soft_limit_motion_scale'])
    result = {
        'mode': mode, 'forces': raw, 'filtered_forces': filtered if valid else None,
        'probe_valid': valid, 'force_within_limit': bool(valid and peak <= c['hard_limit']),
        'failure': failure, 'motion_ready': bool(ready and failure is None),
        'motion_scale': force_scale * x * x * (3.0 - 2.0 * x) if failure is None else 0.0,
        'force_motion_scale': force_scale, 'soft_limit_action': c['soft_limit_action'],
        'stable_steps': state['stable_steps'], 'paused_steps': state['paused_steps'],
        'target_force': c['target'], 'weak_force': weak, 'strong_force': strong,
        'command_q': float(open_q + direction * command),
        'command_openness': 1.0 - command / span,
        'closing_travel': state['closing_travel'], 'physics_dt': dt,
        'position_slip': position_slip, 'slip_growth': growth, 'slipping': slipping,
    }
    state.update(step=step, last_control=result)
    return result
