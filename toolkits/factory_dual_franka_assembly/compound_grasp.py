"""Measure additional bodies held by the same jaw pair, without pose constraints."""
import numpy as np

from toolkits.factory_dual_franka_assembly.planner_primitives import pose_error, relative_pose


def compound_close_metrics(task, robot, spec, primary_metrics):
    """Measure all jaw loads during closure, without inferring a grasp.

    A cap-only feedback signal would mistake a jaw already loaded by the
    base for an empty jaw and recenter the arm into the assembly.
    """
    metrics = dict(primary_metrics)
    sides = ('left_finger', 'right_finger')
    for side in sides:
        metrics[side] = dict(primary_metrics.get(side) or {})
        observation = metrics[side].get('force_observation')
        if observation is not None:
            metrics[side]['force_observation'] = {**observation, 'contacts': list(observation.get('contacts', []))}
    primary_floor = float(spec.get('primary_minimum_finger_force', 0.))
    ready = all(bool((primary_metrics.get(s) or {}).get('force_probe_valid'))
                and float((primary_metrics.get(s) or {}).get('force', 0.)) >= primary_floor for s in sides)
    bodies = {}
    for group in spec.get('additional_grasp_contacts', []):
        extra = task._gripper_contact_metrics(group['object'], robot, attach_spec={**spec, **group})
        forces = [float((extra.get(s) or {}).get('force', float('nan'))) for s in sides]
        valid = all((extra.get(s) or {}).get('force_probe_valid') for s in sides)
        valid = bool(valid and all(np.isfinite(f) and f >= 0 for f in forces))
        contact = bool(valid and min(forces) >= group['minimum_finger_force'])
        bodies[group['object']] = dict(forces=forces, valid=valid, contact_ready=contact)
        ready = ready and contact
        for side, force in zip(sides, forces):
            metrics[side]['force'] = float(metrics[side].get('force', 0.)) + force
            metrics[side]['force_probe_valid'] = bool(metrics[side].get('force_probe_valid') and valid)
            extra_observation = (extra.get(side) or {}).get('force_observation') or {}
            metrics[side].setdefault('force_observation', {}).setdefault('contacts', []).extend(
                extra_observation.get('contacts', []))
    return metrics, dict(ready=ready, bodies=bodies)


def setup_pose_failure(task, config):
    """Keep a fixed pregrasp world reference while an assembled object is free."""
    anchors = getattr(task, '_physical_setup_pose_anchors', None)
    if anchors is None:
        anchors = task._physical_setup_pose_anchors = {}
    group = anchors.setdefault(config['group'], {})
    for name in config['objects']:
        pose = tuple(np.asarray(v).copy() for v in task._resolve_object(name).get_pose())
        reference = group.setdefault(name, pose)
        point = config.get('position_reference_points', {}).get(name)
        distance, angle = pose_error(*pose, *reference, position_reference_point=point)
        if distance > config['position_limit'] or angle > config['orientation_limit']:
            return dict(reason='setup_object_moved', object=name,
                        position_drift=float(distance), orientation_drift=float(angle),
                        root_position_drift=float(np.linalg.norm(pose[0] - np.asarray(reference[0]))),
                        position_reference_point=point)
    return None


def compound_grasp_sample(task, robot, attachment, primary_metrics):
    """Sum actual jaw loads and separately require each supported body contact.

    One primary physical attachment owns the arm command. Additional bodies
    remain dynamic; their real forces and a frozen relative pose are checked.
    """
    spec = attachment['attach_spec']
    groups = spec.get('additional_grasp_contacts', [])
    fingers = [dict(primary_metrics.get(side) or {}) for side in ('left_finger', 'right_finger')]
    floor = float(spec.get('primary_minimum_finger_force', 0.))
    status = dict(ready=(floor == 0. or all(f.get('force_probe_valid') and float(f.get('force', 0.)) >= floor for f in fingers)),
                  failure=None, bodies={})
    if not groups:
        return fingers, status
    anchors = attachment.setdefault('additional_contact_anchors', {})
    primary = task._resolve_object(spec['object']).get_pose()
    for group in groups:
        obj = group['object']
        metrics = task._gripper_contact_metrics(obj, robot, attach_spec={**spec, **group})
        actual = [metrics.get(side) or {} for side in ('left_finger', 'right_finger')]
        forces = [float(f.get('force', float('nan'))) for f in actual]
        valid = all(f.get('force_probe_valid') for f in actual) and all(np.isfinite(f) and f >= 0 for f in forces)
        contact = bool(valid and min(forces) >= group['minimum_finger_force'])
        pose = relative_pose(*primary, *task._resolve_object(obj).get_pose())
        if contact and obj not in anchors:
            anchors[obj] = {k: np.asarray(v).copy().tolist() for k, v in zip(('position', 'orientation'), pose)}
        detail = dict(forces=forces, valid=valid, contact_ready=contact)
        if obj in anchors:
            anchor = anchors[obj]
            distance, angle = pose_error(*pose, anchor['position'], anchor['orientation'])
            detail.update(position_drift=float(distance), orientation_drift=float(angle))
            if distance > group['max_position_drift'] or angle > group['max_orientation_drift']:
                status['failure'] = 'additional_grasp_body_moved'
        if not valid:
            status['failure'] = 'additional_grasp_probe_invalid'
        status['ready'] = bool(status['ready'] and contact)
        status['bodies'][obj] = detail
        for total, extra in zip(fingers, actual):
            total['force'] = float(total.get('force', 0.)) + float(extra.get('force', 0.))
            total['force_probe_valid'] = bool(total.get('force_probe_valid') and extra.get('force_probe_valid'))
    step = int(task.step_counter)
    if status['ready']:
        attachment.pop('additional_contact_unready_since', None)
    else:
        since = attachment.setdefault('additional_contact_unready_since', step)
        if step - since >= 240:
            status['failure'] = status['failure'] or 'additional_grasp_contact_timeout'
    attachment['additional_contact_validation'] = dict(step=step, **status)
    return fingers, status
