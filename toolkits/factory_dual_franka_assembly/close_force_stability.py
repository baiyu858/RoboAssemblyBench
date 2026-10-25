"""Opt-in noise tolerance for close qualification, never for drive/overload control."""
from __future__ import annotations

import math


def close_force_stability_settings(config: dict) -> dict:
    if not isinstance(config, dict) or type(config.get('enabled')) is not bool:
        raise ValueError('close_force_stability requires an explicit enabled boolean.')
    unknown = set(config) - {'enabled', 'window_steps', 'hysteresis_margin'}
    if unknown:
        raise ValueError(f'Unknown close_force_stability settings: {sorted(unknown)}')
    if not config['enabled']:
        return {'enabled': False}
    window = config.get('window_steps', 5)
    margin = config.get('hysteresis_margin', 0.5)
    if type(window) is not int or not 1 <= window <= 32:
        raise ValueError('close_force_stability.window_steps must be an integer in [1, 32].')
    if (isinstance(margin, bool) or not isinstance(margin, (int, float))
            or not math.isfinite(margin) or margin < 0):
        raise ValueError('close_force_stability.hysteresis_margin must be finite and nonnegative.')
    return {'enabled': True, 'window_steps': window, 'hysteresis_margin': float(margin)}


def close_force_qualification(*, state: dict, config: dict, step: int, forces,
                              probes_valid: bool, contact_ready: bool,
                              entry_force: float, high_force: float,
                              contact_threshold: float) -> dict:
    """Average fresh bilateral samples; release hysteresis never masks raw faults.

    Both raw and averaged forces must reach entry_force to enter the band.
    Afterwards either raw force below entry_force-margin clears the window
    immediately, as do target-contact loss, invalid probes and raw excess force.
    The caller still checks pose and requires its original stable-step count.
    """
    config = close_force_stability_settings(config)
    if not config['enabled']:
        raise ValueError('close_force_qualification requires enabled close_force_stability.')
    release_force = entry_force - config['hysteresis_margin']
    if (not all(math.isfinite(x) for x in (entry_force, high_force, contact_threshold))
            or not 0 <= contact_threshold < release_force <= entry_force <= high_force):
        raise ValueError('close_force_stability requires contact_threshold < release <= entry <= high.')
    previous_step = state.get('step')
    new_sample = previous_step != step
    signature = (config['window_steps'], release_force, entry_force, high_force, contact_threshold)
    restart = (state.get('signature') != signature
               or (previous_step is not None and step not in (previous_step, previous_step + 1)))
    if restart:
        state.clear()
    state['signature'] = signature
    state['step'] = int(step)
    raw = [float(x) for x in forces]
    valid = probes_valid and len(raw) == 2 and all(math.isfinite(x) for x in raw)
    fault = ('invalid_probe' if not valid else
             'target_contact_not_ready' if not contact_ready else
             'raw_force_above_high' if max(raw) > high_force else
             'raw_force_below_release' if min(raw) < release_force else None)
    if fault:
        state.update(samples=[], latched=False)
        detail = {'qualified': False, 'reason': fault, 'filtered_forces': None,
                  'window_samples': 0, 'restart': True}
    elif not new_sample and not restart and state.get('last_result') is not None:
        return {**state['last_result'], 'new_sample': False, 'restart': False}
    else:
        samples = (state.get('samples', []) + [raw])[-config['window_steps']:]
        state['samples'] = samples
        average = [sum(sample[i] for sample in samples) / len(samples) for i in (0, 1)]
        full = len(samples) == config['window_steps']
        was_latched = bool(state.get('latched', False))
        latched = bool(full and (min(average) >= release_force if was_latched else
                               min(raw) >= entry_force and min(average) >= entry_force))
        state['latched'] = latched
        detail = {'qualified': latched, 'filtered_forces': average, 'window_samples': len(samples),
                  'restart': restart,
                  'reason': ('warming' if not full else 'below_entry' if not latched else
                             'entered' if not was_latched else 'in_hysteresis_band')}
    detail.update(new_sample=new_sample, raw_forces=raw, entry_force=entry_force,
                  release_force=release_force, high_force=high_force,
                  window_steps=config['window_steps'])
    state['last_result'] = detail.copy()
    return detail
