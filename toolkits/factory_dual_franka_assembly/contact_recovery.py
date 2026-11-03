"""Track a bounded physical reclose without claiming a valid grasp or moving an arm."""
import math


def retain_contact_recovery(task, object_name, attachment):
    spec = attachment.get('attach_spec', {})
    steps = spec.get('contact_recovery_steps', 0)
    current = attachment.get('contact_recovery')
    hold = attachment.get('last_hold_validation') or {}
    metrics = attachment.get('current_contact_metrics') or {}
    # Opening is an intentional release, not a failed clamp to be recovered.
    command = getattr(task, '_current_gripper_command', None)
    phase_spec = getattr(task, 'get_current_phase_spec', None)
    if command and phase_spec and command(phase_spec(), attachment['robot_name']) == 'open':
        if current:
            current['record'].update(outcome='released', end_step=int(task.step_counter))
            attachment.pop('contact_recovery', None)
        return False
    if hold.get('valid'):
        if current:
            current['record'].update(outcome='restored', end_step=int(task.step_counter))
            attachment.pop('contact_recovery', None)
        return False
    eligible = (type(steps) is int and 0 < steps <= 48
                and attachment.get('mode') == 'pure_physical_grasp'
                and spec.get('continuous_physics') is True
                and hold.get('reasons') == ['contact_missing']
                and isinstance(metrics.get('caging_axis'), str)
                and metrics['caging_axis'] in str(spec.get('physical_contact_axes', 'xy')))
    gap_limit = float(spec.get('contact_recovery_max_surface_gap', .0005))
    eligible = eligible and 0 < gap_limit <= .001
    for side in ('left_finger', 'right_finger'):
        finger = metrics.get(side, {})
        force, gap = finger.get('force'), finger.get('surface_gap')
        eligible = bool(eligible and finger.get('force_probe_valid')
                        and isinstance(force, (int, float)) and math.isfinite(force) and force >= 0
                        and isinstance(gap, (int, float)) and math.isfinite(gap) and gap <= gap_limit)
    if not eligible:
        if current:
            current['record'].update(outcome='aborted', end_step=int(task.step_counter), reasons=hold.get('reasons'))
            attachment.pop('contact_recovery', None)
        return False
    if current is None:
        record = dict(object=object_name, robot=attachment['robot_name'], start_step=int(task.step_counter),
                      phase=getattr(task, 'phase', None), maximum_steps=steps, outcome='recovering')
        history = getattr(task, '_physical_contact_recovery_history', None)
        if history is None:
            history = task._physical_contact_recovery_history = []
        history.append(record)
        del history[:-64]
        current = attachment['contact_recovery'] = dict(start_step=int(task.step_counter), record=record)
    if int(task.step_counter) - current['start_step'] >= steps:
        current['record'].update(outcome='timeout', end_step=int(task.step_counter))
        attachment.pop('contact_recovery', None)
        return False
    return True
