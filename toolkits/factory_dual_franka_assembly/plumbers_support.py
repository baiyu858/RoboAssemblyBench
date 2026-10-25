"""Read-only cross-arm readiness for the physical plumbers base side-pinch."""
from __future__ import annotations

import json
import os

import numpy as np

from toolkits.factory_dual_franka_assembly.planner_primitives import pose_error


def _record_status(task, support: dict, result: dict) -> dict:
    path = os.environ.get('BEAM_HOLD_TRACE_PATH')
    if not path:
        return result
    previous = getattr(task, '_plumbers_support_trace_state', {})
    step = int(task.step_counter)
    stamp = (support['cycle'], task.phase, result['ready'], result['failure'], result['reason'])
    if (previous.get('stamp') != stamp or step < previous.get('step', -100)
            or step - previous.get('step', -100) >= 32):
        record = {'step': step, 'phase': task.phase, 'robot': support['robot'], 'object': support['object'],
                  'cycle': support['cycle'], **result}
        with open(os.path.join(os.path.dirname(path), 'plumbers_support_trace.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
        task._plumbers_support_trace_state = {'stamp': stamp, 'step': step}
    return result


def base_support_status(task, support: dict) -> dict:
    """Never turn missing support into a new grasp or a kinematic constraint.

    The right skill runs first each control tick. Its force-maintenance result
    is the readiness gate for the left skill, not merely an attachment label.
    The world anchor is captured once per real support grasp, never per stage.
    """
    robot, obj = support.get('robot'), support.get('object')
    result = {'ready': False, 'failure': None, 'reason': 'support_unavailable'}
    attachment = task._attachment_for(obj, robot)
    if (attachment is None or attachment.get('robot_name') != robot
            or attachment.get('mode') != 'pure_physical_grasp'):
        return _record_status(task, support, {**result, 'failure': 'support_grasp_missing'})
    if not task._physical_hold_valid(obj, attachment, validation_stage='plumbers_base_support'):
        return _record_status(task, support, {**result, 'failure': 'support_grasp_invalid'})
    reference = attachment.get('hold_grasp_reference') or {}
    point = np.asarray(reference.get('point_in_object', []), dtype=float)
    center = np.asarray(support['contact_box_offset'], dtype=float)
    half = .5 * np.asarray(support['contact_box_scale'], dtype=float)
    if (point.shape != (3,) or not np.all(np.isfinite(point))
            or np.any(np.abs(point - center) > half + .0035)):
        return _record_status(task, support, {**result, 'failure': 'support_contact_outside_low_flange'})
    saved = getattr(task, '_ur5e_contact_hold_commands', {}).get(robot)
    if saved is None or saved.get('object') != obj:
        return _record_status(task, support, {**result, 'failure': 'support_command_missing'})
    position, orientation = task._resolve_object(obj).get_pose()
    position, orientation = np.asarray(position, dtype=float), np.asarray(orientation, dtype=float)
    if (position.shape != (3,) or orientation.shape != (4,)
            or not np.all(np.isfinite(position)) or not np.all(np.isfinite(orientation))
            or np.linalg.norm(orientation) < 1e-12):
        return _record_status(task, support, {**result, 'failure': 'support_pose_invalid'})
    anchor = attachment.setdefault('plumbers_support_world_anchor', {
        'position': position.copy().tolist(), 'orientation': orientation.copy().tolist(),
        'captured_step': int(task.step_counter), 'cycle': support['cycle']})
    if anchor['cycle'] != support['cycle']:
        return _record_status(task, support, {**result, 'failure': 'support_stale_grasp'})
    distance, angle = pose_error(position, orientation, anchor['position'], anchor['orientation'])
    result.update(position_drift=float(distance), orientation_drift=float(angle),
                  anchor_capture_step=anchor['captured_step'])
    if distance > support['max_position_drift'] or angle > support['max_orientation_drift']:
        result.update(failure='support_base_moved', reason='world_anchor_drift')
    else:
        attach_spec = attachment.get('attach_spec') or {}
        config = attach_spec.get('hold_force_control') or {}
        if config.get('enabled'):
            force_state = saved.get('loaded_force_state') or {}
            control = force_state.get('last_control') or {}
            # Task phase checks may run just after the next physics step. A
            # result from earlier stages or an older controller tick is stale.
            age = int(task.step_counter) - int(force_state.get('step', -100))
            if control.get('failure'):
                result.update(failure='support_force_failure', reason=str(control['failure']))
            elif age not in (0, 1) or not control:
                result['reason'] = 'support_force_not_updated'
            else:
                result.update(ready=bool(control.get('motion_ready')), reason=control.get('mode', 'force_recovery'))
        else:
            # Controlled legacy comparison still requires bilateral clamp
            # force, even when the newer weak-side force policy is disabled.
            fingers = attachment.get('current_contact_metrics') or {}
            low, high = float(attach_spec.get('close_force_low', 4.)), float(attach_spec.get('close_force_high', 12.))
            forces = [fingers.get(side, {}).get('force', float('nan'))
                      for side in ('left_finger', 'right_finger')]
            valid = all(fingers.get(side, {}).get('force_probe_valid') for side in ('left_finger', 'right_finger'))
            result.update(ready=bool(valid and all(np.isfinite(f) and low <= f <= high for f in forces)),
                          reason='bilateral_force_band')
    return _record_status(task, support, result)
