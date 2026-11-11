"""Opt-in physical execution of the canonical UR5e plumbers task.

The canonical compiler still owns grasp selection and insertion geometry. This
pass changes how those motions interact with free parts; it has no Beam phase,
grasp-axis, support-plan or robot-pose assumptions.
"""
from __future__ import annotations

import copy
import math

from toolkits.factory_dual_franka_assembly.close_force_stability import (
    close_force_stability_settings,
)
from toolkits.factory_dual_franka_assembly.physical_hold_force import (
    hold_force_settings,
)
from toolkits.factory_dual_franka_assembly.planner_primitives import (
    quat_conjugate,
    quat_rotate,
)


def apply_plumbers_wait_step_limit(payload: dict, spec: dict) -> dict:
    """Cap confirmation windows after physical and RL phase compilation.

    Movement budgets, actuator ramps, safety history and learned controller
    protocols are not dwell times. Keep those intact, including checkpoint
    configuration; RL receives a separate execution-only completion window.
    """
    limit = spec.get('physical_wait_step_limit')
    if limit is None:
        return payload
    if type(limit) is not int or limit < 2:
        raise ValueError('physical_wait_step_limit must be an integer of at least 2')
    keys = {
        'stable_steps',
        'min_steps',
        'hold_steps',
        'wait_steps',
        'target_object_stable_steps',
        'target_object_settle_hold_steps',
        'close_ready_stable_steps',
        'close_force_validation_steps',
        'close_contact_stable_steps',
        'close_until_contact_min_steps',
        'close_gate_recenter_stable_steps',
        'close_force_recenter_settle_steps',
        'compliant_servo_resume_stable_steps',
        'compliant_servo_settle_stable_steps',
        'joint_target_stable_steps',
        'min_attach_steps',
        'release_min_steps',
        'resume_steps',
    }
    changes = []

    def visit(value, path):
        if isinstance(value, list):
            for i, item in enumerate(value):
                visit(item, f'{path}[{i}]')
        elif isinstance(value, dict):
            if value.get('name') == 'contact_insert_rl':
                required = value['controller_config']['seated_stable_steps']
                current = value.get('completion_stable_steps', required)
                if current > limit:
                    value['completion_stable_steps'] = limit
                    changes.append(dict(path=path + '.completion_stable_steps', before=current, after=limit))
            for key, item in list(value.items()):
                if key == 'controller_config':
                    continue  # Checked against the original trained policy.
                if (
                    (key in keys or (key == 'steps' and value.get('type') == 'timer'))
                    and type(item) is int
                    and item > limit
                ):
                    value[key] = limit
                    changes.append(dict(path=path + '.' + key, before=item, after=limit))
                elif isinstance(item, (dict, list)):
                    visit(item, path + '.' + key)

    for phase in payload['phases']:
        visit(phase, phase['name'])
    audit = dict(steps=limit, changes=changes)
    payload.setdefault('fabrica_canonical_resolved', {})['physical_wait_step_limit'] = audit
    return payload


def align_plumbers_base_grasps_to_pads(task: dict, spec: dict) -> None:
    """Align the nominal pinch with the native Robotiq pads before IK/clearance checks.

    Fabrica's converted base frame puts the pinch about 168 mm ahead of the
    gripper. The authored Isaac pads are closer to that frame. Correct both
    directions of the rigid transform, not just the close-phase command; all
    pickup, attachment and transport geometry must describe the same grasp.
    Only the plumbers base candidates are affected, and only when opted in.
    """
    value = spec.get('physical_base_pad_center_m')
    if value is None:
        return
    pad_center = float(value)
    if not math.isfinite(pad_center) or not 0.10 <= pad_center <= 0.17:
        raise ValueError('physical_base_pad_center_m must be finite and in [0.10, 0.17] m.')
    center_offset = [float(x) for x in spec.get('physical_base_grasp_center_offset_m', [0.0, 0.0, 0.0])]
    if (
        len(center_offset) != 3
        or not all(math.isfinite(x) for x in center_offset)
        or sum(x * x for x in center_offset) > 0.02**2
    ):
        raise ValueError('physical_base_grasp_center_offset_m must be a finite 3-vector of at most 20 mm.')
    configured_id = spec.get('base_grasp_id')
    for grasp in task['base_grasp_candidates']:
        if configured_id is not None and int(grasp['grasp_id']) != int(configured_id):
            continue
        if grasp.get('target_gripper') != 'robotiq-85':
            raise ValueError('physical_base_pad_center_m requires the native Robotiq 2F-85 profile.')
        orientation = grasp['tcp_in_assembly_orientation']
        inverse = quat_conjugate(orientation)
        position = grasp['tcp_in_assembly_position']
        pinch = quat_rotate(inverse, [c - p for c, p in zip(grasp['grasp_center_m'], position)])
        shift = float(pinch[2]) - pad_center
        if abs(shift) > 0.08:
            raise ValueError('physical_base_pad_center_m would shift the base grasp by more than 80 mm.')
        delta = quat_rotate(orientation, [0.0, 0.0, shift])
        corrected = [float(p + d + offset) for p, d, offset in zip(position, delta, center_offset)]
        grasp['grasp_center_m'] = [float(c + offset) for c, offset in zip(grasp['grasp_center_m'], center_offset)]
        part = next(p for p in task['parts'] if str(p['part_id']) == str(task['base_part']))
        grasp['interior_clearance_score'] = min(
            min(c - lo, hi - c) / max((hi - lo) * 0.5, 1e-9)
            for c, lo, hi in zip(grasp['grasp_center_m'], part['bbox_min'], part['bbox_max'])
        )
        grasp['tcp_in_assembly_position'] = corrected
        grasp['object_in_tcp_position'] = [float(-x) for x in quat_rotate(inverse, corrected)]
        grasp['physical_pad_alignment'] = {
            'source_tcp_in_assembly_position': list(position),
            'source_pinch_distance_m': float(pinch[2]),
            'pad_center_m': pad_center,
            'tcp_shift_along_approach_m': shift,
            'tcp_shift_in_object_m': [float(x) for x in delta],
            'grasp_center_offset_in_object_m': center_offset,
        }


def align_plumbers_move_grasps_to_pads(task: dict, spec: dict) -> None:
    """Calibrate selected moving-part tools before grasp ranking and path IK."""
    centers = spec.get('physical_move_pad_centers_m', {})
    offsets = spec.get('physical_move_grasp_center_offsets_m', {})
    if not isinstance(centers, dict):
        raise ValueError('physical_move_pad_centers_m must map part ids to pad-center distances.')
    if not isinstance(offsets, dict):
        raise ValueError('physical_move_grasp_center_offsets_m must map part ids to object-frame offsets.')
    steps = {str(step['move_part']): step for step in task['assembly_steps']}
    if set(offsets) - set(centers):
        raise ValueError('physical_move_grasp_center_offsets_m requires an explicit native pad center for each part.')
    parts = {str(part['part_id']): part for part in task['parts']}
    for part_id, value in centers.items():
        pad = float(value)
        if str(part_id) not in steps or not math.isfinite(pad) or not 0.10 <= pad <= 0.17:
            raise ValueError('physical_move_pad_centers_m requires an existing part and distance in [0.10, 0.17] m.')
        step = steps[str(part_id)]
        offset = [float(x) for x in offsets.get(str(part_id), [0.0, 0.0, 0.0])]
        if len(offset) != 3 or not all(math.isfinite(x) for x in offset) or sum(x * x for x in offset) > 0.03**2:
            raise ValueError('physical_move_grasp_center_offsets_m requires finite offsets of at most 30 mm.')
        seen = set()
        for grasp in [step['move_grasp'], *step['move_grasp_candidates']]:
            if id(grasp) in seen:
                continue
            seen.add(id(grasp))
            if grasp.get('target_gripper') != 'robotiq-85':
                raise ValueError('physical_move_pad_centers_m requires the native Robotiq gripper.')
            p, q = grasp['tcp_in_assembly_position'], grasp['tcp_in_assembly_orientation']
            inverse = quat_conjugate(q)
            pinch = quat_rotate(inverse, [c - x for c, x in zip(grasp['grasp_center_m'], p)])
            shift = float(pinch[2]) - pad
            if abs(shift) > 0.08:
                raise ValueError('physical_move_pad_centers_m would shift a tool by more than 80 mm.')
            corrected = [float(x + d + delta) for x, d, delta in zip(p, quat_rotate(q, [0.0, 0.0, shift]), offset)]
            grasp['grasp_center_m'] = [float(x + delta) for x, delta in zip(grasp['grasp_center_m'], offset)]
            part = parts[str(part_id)]
            grasp['interior_clearance_score'] = min(
                min(c - lo, hi - c) / max((hi - lo) * 0.5, 1e-9)
                for c, lo, hi in zip(grasp['grasp_center_m'], part['bbox_min'], part['bbox_max'])
            )
            grasp['tcp_in_assembly_position'] = corrected
            grasp['object_in_tcp_position'] = [-float(x) for x in quat_rotate(inverse, corrected)]
            grasp['physical_pad_alignment'] = dict(
                source_pinch_distance_m=float(pinch[2]),
                pad_center_m=pad,
                tcp_shift_along_approach_m=shift,
                source_tcp_in_assembly_position=list(p),
                grasp_center_offset_in_object_m=offset,
            )


def raise_plumbers_insertion_entries(task, spec):
    """Place the planned entry above first contact; RL owns the contact stroke."""
    settings = spec.get('physical_insertion_entry_offsets_m', {})
    if not isinstance(settings, dict):
        raise ValueError('physical_insertion_entry_offsets_m must map part ids to axial entry distances')
    steps = {str(step['move_part']): step for step in task['assembly_steps']}
    for part_id, distance in settings.items():
        if str(part_id) not in steps or isinstance(distance, bool) or not math.isfinite(float(distance)):
            raise ValueError('Invalid physical insertion entry part or distance')
        path = steps[str(part_id)]['disassembly_path']
        start = path[0]['position']
        delta = [b - a for a, b in zip(start, path[-1]['position'])]
        length = math.sqrt(sum(v * v for v in delta))
        if not 0 < length <= float(distance) <= 0.10:
            raise ValueError('Physical insertion entry may only extend an existing path up to 100 mm')
        path[-1]['position'] = [a + float(distance) * d / length for a, d in zip(start, delta)]


def _pickup_close_settings(spec: dict, part_id: str) -> dict:
    settings = {}
    clearance = spec.get('physical_pickup_fixture_clearance', {}).get(part_id)
    if clearance is not None:
        if (
            not isinstance(clearance, dict)
            or not clearance.get('fixture')
            or not 0 < float(clearance.get('margin_m', 0.0015)) <= 0.01
            or not 0 < float(clearance.get('joint_step', 0.005)) <= 0.01
            or not 0 <= float(clearance.get('preload_joint_travel', 0.01)) <= 0.03
        ):
            raise ValueError('Invalid physical_pickup_fixture_clearance settings')
        settings['fixture_closure_clearance'] = copy.deepcopy(clearance)
    force_min = spec.get('physical_move_close_force_min_n', {}).get(part_id)
    if force_min is not None:
        force_min = float(force_min)
        if not 4.0 <= force_min < 12.0:
            raise ValueError('physical_move_close_force_min_n must be in [4, 12) N.')
        balance_target = float(spec.get('physical_move_close_balance_target_n', {}).get(part_id, force_min))
        if not force_min <= balance_target < 12.0:
            raise ValueError('physical_move_close_balance_target_n must be between the minimum and 12 N.')
        # Qualify in the carrying controller's band and balance before capture.
        settings.update(
            close_force_low=force_min,
            close_force_min_stable=force_min,
            close_force_balance_min=balance_target,
            close_force_balance_step=0.000025,
            close_force_weak_side_preload=True,
        )
    return settings


def _move_hold_force_controls(spec: dict, phases: list) -> dict:
    """Resolve opt-in force bands for moving parts without altering support."""
    overrides = spec.get('physical_move_hold_force_control', {})
    if not isinstance(overrides, dict):
        raise ValueError('physical_move_hold_force_control must map moving part ids to force settings')
    moving = {
        str(p.get('local_skill', {}).get('object', '')).removeprefix('fabrica_plumbers_block_')
        for p in phases
        if p['name'].startswith('assemble_') and p.get('local_skill', {}).get('name') == 'ur5e_close_gripper'
    }
    base = spec.get('hold_force_control') or {}
    result = {}
    for part_id, settings in overrides.items():
        if not isinstance(part_id, str) or part_id not in moving or not isinstance(settings, dict):
            raise ValueError('physical_move_hold_force_control requires an existing moving part and settings mapping')
        if base.get('enabled') is not True or settings.get('enabled', True) is not True:
            raise ValueError('physical_move_hold_force_control requires enabled force maintenance')
        result[part_id] = hold_force_settings({**base, **settings})
    return result


def _insertion_release_settings(spec, success):
    release_settings = spec.get('physical_insertion_release_tolerances', {})
    if not isinstance(release_settings, dict):
        raise ValueError('physical_insertion_release_tolerances must map part ids to tolerances')
    release_settings = {str(k): v for k, v in release_settings.items()}
    for part_id, settings in release_settings.items():
        allowed = {'position_m', 'axial_m', 'lateral_m', 'orientation_rad', 'stable_steps'}
        if not isinstance(settings, dict) or set(settings) != allowed:
            raise ValueError(f'physical_insertion_release_tolerances[{part_id}] requires {sorted(allowed)}')
        for key in allowed - {'stable_steps'}:
            value = settings[key]
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f'physical_insertion_release_tolerances[{part_id}].{key} must be positive')
        if type(settings['stable_steps']) is not int or settings['stable_steps'] < 2:
            raise ValueError('Physical insertion release needs at least two stable steps')
        entries = [e for e in success if str(e.get('object', '')).rsplit('_', 1)[-1] == str(part_id)]
        if len(entries) != 1:
            raise ValueError(f'Physical insertion release references unknown part {part_id}')
        entry = entries[0]
        entry['position_tolerance'] = min(float(entry['position_tolerance']), float(settings['position_m']))
        entry['orientation_tolerance'] = min(float(entry['orientation_tolerance']), float(settings['orientation_rad']))
    return release_settings


def _apply_seated_release_gate(skill, release_settings):
    precision = release_settings.get(str(skill.get('object', '')).rsplit('_', 1)[-1])
    if (
        precision is None
        or not skill.get('target_object_final_target')
        or skill.get('target_object_target') != skill['target_object_final_target']
    ):
        return
    # Complete the seated endpoint while still grasped. Legacy relaxed gates
    # can otherwise release several millimetres high.
    for key in list(skill):
        if key.startswith('relaxed_'):
            skill.pop(key)
    skill.update(
        position_tolerance=float(precision['position_m']),
        target_object_position_tolerance=float(precision['position_m']),
        target_object_axial_position_tolerance=float(precision['axial_m']),
        target_object_lateral_position_tolerance=float(precision['lateral_m']),
        target_object_orientation_tolerance=float(precision['orientation_rad']),
        target_object_stable_steps=precision['stable_steps'],
        cartesian_position_step=0.00008,
        target_object_servo_position_command_lookahead=0.0005,
        target_object_servo_position_command_accumulation_step=0.00008,
        target_object_servo_position_command_gate_overdrive=False,
        compliant_servo_max_position_step=0.00008,
        compliant_servo_position_command_lookahead=0.0005,
        compliant_servo_position_command_accumulation_step=0.00008,
        compliant_servo_position_command_gate_overdrive=False,
    )


def apply_plumbers_physics(phases: list[dict], *, success: list[dict], spec: dict) -> None:  # noqa: C901
    close_stability = close_force_stability_settings(spec.get('close_force_stability', {'enabled': False}))
    if close_stability.get('hysteresis_margin', 0.0) >= 4.0 - 0.1:
        raise ValueError('close_force_stability release threshold must exceed the 0.1 N contact threshold.')
    idle_hold = spec.get('physical_idle_hold_anchor', False)
    if not isinstance(idle_hold, bool):
        raise ValueError('physical_idle_hold_anchor must be an explicit boolean.')
    release_settings = _insertion_release_settings(spec, success)
    points = spec.get('physical_success_reference_points_m', {})
    if not isinstance(points, dict):
        raise ValueError('physical_success_reference_points_m must map part ids to CAD-local points')
    known = {entry['object'] for entry in success}
    hold_angle_overrides = spec.get('physical_hold_orientation_slip_tolerances_rad', {})
    if not isinstance(hold_angle_overrides, dict):
        raise ValueError('physical_hold_orientation_slip_tolerances_rad must map part ids to radians')
    hold_angle_overrides = {str(key): value for key, value in hold_angle_overrides.items()}
    for part_id, value in hold_angle_overrides.items():
        if (
            f'fabrica_plumbers_block_{part_id}' not in known
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 < value <= math.pi
        ):
            raise ValueError(
                'physical_hold_orientation_slip_tolerances_rad requires existing parts and (0, pi] radians'
            )
    for part_id, point in points.items():
        if (
            f'fabrica_plumbers_block_{part_id}' not in known
            or not isinstance(point, (list, tuple))
            or len(point) != 3
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in point)
        ):
            raise ValueError('Invalid physical success reference point')
    for entry in success:
        key = entry['object'].removeprefix('fabrica_plumbers_block_')
        if key in points:
            entry['position_reference_point'] = copy.deepcopy(points[key])
    goals = {entry['object']: copy.deepcopy(entry) for entry in success}
    stable_steps = int(spec.get('physical_settle_steps', 120))
    if stable_steps < 2:
        raise ValueError('physical_settle_steps must be at least 2.')
    motion_timeout = spec.get('physical_motion_min_timeout_steps', 0)
    if type(motion_timeout) is not int or motion_timeout < 0:
        raise ValueError('physical_motion_min_timeout_steps must be a nonnegative integer.')
    rotation_limits = {
        key: float(spec.get(key, default))
        for key, default in (
            ('physical_rotation_speed', 0.10),
            ('physical_rotation_acceleration', 0.30),
            ('physical_rotation_lookahead', 0.010),
        )
    }
    if not all(math.isfinite(value) and value > 0 for value in rotation_limits.values()):
        raise ValueError('physical_rotation limits must be finite and positive.')
    if rotation_limits['physical_rotation_lookahead'] >= math.pi:
        raise ValueError('physical_rotation_lookahead must be less than pi.')
    transport_rotation_limits = dict(rotation_limits)
    for kind in ('speed', 'acceleration'):
        key = f'physical_transport_rotation_{kind}'
        value = spec.get(key, rotation_limits[f'physical_rotation_{kind}'])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive.')
        transport_rotation_limits[f'physical_rotation_{kind}'] = float(value)
    contact_policy = {
        'continuous_physics': True,
        'hold_force_regulation': True,
        'require_force_contact': True,
        'require_dual_force_contact': True,
        'require_pair_force_contact': True,
        'require_opposing_force_contact': True,
        'measure_force_contact': True,
        'require_strict_physical_contact': True,
        'require_dual_finger_contact': True,
        'allow_cross_axis_dual_finger_contact': False,
        # Each grasp retains its own part-local contact box. Unlike Beam, these
        # parts do not all pinch along the same assembly-coordinate axis.
        'physical_contact_axes': 'x,y,z',
        'contact_box_min_half_extent': 1e-6,
        'physical_contact_interior_scale': 1.0,
        'physical_contact_interior_margin': 0.0,
        'finger_contact_distance': 0.0035,
        'physical_attach_surface_gap': 0.0035,
        'contact_force_threshold': 0.10,
        'allow_jaw_width_completion': False,
        'allow_jaw_width_contact_for_physical_grasp': False,
        'allow_closed_gripper_completion': False,
        'use_joint_stall_for_close_until_contact': False,
        'allow_initial_strict_contact_for_short_close': False,
        'allow_initial_strict_contact_without_closure': False,
        'allow_caging_contact_for_physical_grasp': False,
        'allow_caging_hold_for_physical_grasp': False,
        'hold_grace_allow_missing_contact': False,
        'hold_grace_steps': 0,
        # Freeze a material point at confirmed bilateral contact, not the raw
        # OBJ origin (which moves when a part rotates about its grasp).
        'hold_position_slip_reference': 'grasp_point',
        'hold_position_slip_tolerance': 0.003,
        'hold_orientation_slip_tolerance': 0.0872664626,
        'close_force_low': 4.0,
        'close_force_high': 12.0,
        'close_force_limit': 40.0,
        # Capture requires both fingers in band. Initial base and cloned support
        # closes must preload the weak side, just like the moving-part closes.
        'close_force_weak_side_preload': True,
        'close_force_joint_rate': 0.025,
        'close_force_unload_rate': 0.06,
        'close_force_max_command_lead': 0.001,
        'close_force_max_seek': 0.03,
        # The nominal close preset is not a physical jaw-travel limit. Thin
        # tabs may need the remaining travel, bounded by closed joint position.
        'physical_gripper_min_openness': 0.0,
    }
    hold_config = spec.get('hold_force_control')
    if hold_config is not None:
        if not isinstance(hold_config, dict) or not isinstance(hold_config.get('enabled'), bool):
            raise ValueError('hold_force_control requires an explicit enabled boolean.')
        if hold_config['enabled']:
            contact_policy['hold_force_control'] = hold_force_settings(hold_config)
    move_hold_controls = _move_hold_force_controls(spec, phases)

    def settle_phase(name: str, objects: list, robot: str | None = None) -> dict:
        return {
            'name': name,
            'timeout_steps': max(1200, stable_steps * 4),
            'robot_targets': {},
            'gripper_commands': {} if robot is None else {robot: 'open'},
            'advance': {
                'type': 'physical_settle',
                'objects': copy.deepcopy(objects),
                'stable_steps': stable_steps,
                'linear_velocity_threshold': 0.01,
                'angular_velocity_threshold': 0.1,
                'position_drift_tolerance': 0.002,
                'orientation_drift_tolerance': 0.03,
            },
        }

    result = [settle_phase('plumbers_fixture_settle', list(goals))]
    releases = {}
    for phase in phases:
        old_name = phase['name']
        # Free parts stay dynamic from reset through release. Removing the
        # initial pin alone is insufficient: close stages can re-pin the base.
        locks = phase.pop('lock', [])
        for key in ('fixture_lock', 'unlock_after_steps', 'kinematic_carry', 'gripper_sibling_filters'):
            phase.pop(key, None)
        if any(not entry.get('enabled', True) for entry in phase.get('object_collisions', [])):
            raise ValueError(f'{old_name}: continuous plumbers cannot disable object collision.')
        if any(not entry.get('enabled', True) for entry in phase.get('object_gravity', [])):
            raise ValueError(f'{old_name}: continuous plumbers cannot disable gravity.')
        if any(entry.get('enabled', True) for entry in phase.get('collision_filters', [])):
            raise ValueError(f'{old_name}: continuous plumbers cannot filter contact pairs.')

        skill = phase.get('local_skill', {})
        for attach in phase.get('attach', []):
            attach.update(contact_policy)
            part_id = str(attach.get('object', '')).removeprefix('fabrica_plumbers_block_')
            if part_id in hold_angle_overrides:
                attach['hold_orientation_slip_tolerance'] = float(hold_angle_overrides[part_id])
                attach['hold_orientation_slip_limit_override'] = True
            if part_id in move_hold_controls:
                attach['hold_force_control'] = copy.deepcopy(move_hold_controls[part_id])
            recovery = spec.get('physical_contact_recovery', {}).get(
                str(attach.get('object', '')).removeprefix('fabrica_plumbers_block_')
            )
            if recovery:
                steps, gap = recovery['steps'], float(recovery['maximum_surface_gap_m'])
                if type(steps) is not int or not 0 < steps <= 48 or not 0 < gap <= 0.001:
                    raise ValueError('Invalid bounded physical_contact_recovery settings')
                attach.update(contact_recovery_steps=steps, contact_recovery_max_surface_gap=gap)
            attach.update(
                attachment_mode='pure_physical_grasp',
                attachment_relative_pose_source='current',
                disable_collision_on_attach=False,
                filter_gripper_collisions_on_attach=False,
                release_lock_on_attach=True,
                release_open_openness=0.95,
            )
            # Attach/close use the same part-specific geometry and force gate.
            for key in ('contact_box_offset', 'contact_box_scale'):
                skill[key] = copy.deepcopy(attach[key])

        if skill.get('name') == 'ur5e_close_gripper':
            part_id = str(skill.get('object', '')).removeprefix('fabrica_plumbers_block_')
            skill.update(contact_policy)
            if part_id in hold_angle_overrides:
                skill['hold_orientation_slip_tolerance'] = float(hold_angle_overrides[part_id])
            if part_id in move_hold_controls:
                skill['hold_force_control'] = copy.deepcopy(move_hold_controls[part_id])
            if close_stability['enabled']:
                skill['close_force_stability'] = copy.deepcopy(close_stability)
            else:
                skill.pop('close_force_stability', None)
            skill.update(
                close_force_regulation=True,
                close_force_seek_until_contact=True,
                close_force_require_clamp_band=True,
                close_gate_freeze_at_start=True,
                close_gate_local_ik=True,
                close_gate_local_ik_radius=0.04,
                close_gate_track_object_during_close=False,
                close_ready_stable_steps=24,
                close_position_tolerance=0.001,
                close_orientation_tolerance=0.02,
                close_gate_stationary_position_tolerance=0.0001,
                close_gate_stationary_orientation_tolerance=0.002,
                close_pose_gate_timeout_steps=720,
                ik_position_tolerance=0.000005,
                ik_orientation_tolerance=0.001,
                close_force_min_stable=4.0,
                close_force_validation_steps=96,
                close_contact_stable_steps=96,
                close_contact_hold_squeeze_margin=0.0,
                close_until_contact_timeout_steps=1200,
                close_approach_joint_rate=0.10,
                close_approach_joint_margin=0.06,
                close_gate_recenter_stable_steps=4,
                close_gate_recenter_step=0.0002,
                close_gate_recenter_max_offset=0.002,
                close_force_recenter_measure_progress=True,
                close_force_recenter_stop_on_bilateral=True,
                close_force_recenter_adaptive_step=True,
                close_force_recenter_settle_steps=8,
                close_force_recenter_timeout_steps=240,
            )
            skill.update(_pickup_close_settings(spec, part_id))
            phase['timeout_steps'] = max(int(phase['timeout_steps']), 2000)
        elif skill.get('requires_held_object'):
            # Speed up only high-clearance transport. Pickup departure, lift,
            # final hover, placement and insertion retain their original rates.
            high_transport = old_name.endswith('_assembly_clearance') or old_name.rpartition('_transport_')[2].isdigit()
            phase_rotation_limits = transport_rotation_limits if high_transport else rotation_limits
            skill.update(
                continuous_physics=True,
                regulate_physical_carry=True,
                gripper_command='contact_hold',
                track_published_grasp_command=True,
                use_arm_ik_controller=False,
                allow_initial_ik_branch_jump=False,
                ik_reference_mode='command',
                max_command_joint_step=0.008,
                max_command_wrist_joint_step=0.008,
                physical_rotation_governor=True,
                cartesian_orientation_command_warm_start=False,
                cartesian_orientation_command_lookahead=rotation_limits['physical_rotation_lookahead'],
                orientation_first_before_translation=False,
                lock_transport_ik_target=False,
                ik_retry_with_measured_state=False,
                **phase_rotation_limits,
            )
            # The governor brakes toward its measured-pose lookahead boundary.
            # Its speed cap is not the achieved speed: reserve half the usable
            # lookahead for tracking lag and budget with the same discrete
            # braking law. This is a finite scheduling allowance, not a promise
            # of convergence under contact/IK obstruction.
            acceleration = phase_rotation_limits['physical_rotation_acceleration']
            interval_dv = acceleration / 240
            usable_lookahead = 0.9 * rotation_limits['physical_rotation_lookahead']
            budget_speed = min(
                phase_rotation_limits['physical_rotation_speed'],
                math.sqrt(interval_dv**2 + acceleration * usable_lookahead) - interval_dv,
            )
            rotation_steps = math.ceil(240 * (math.pi / budget_speed + 2 * budget_speed / acceleration + 5.0))
            phase['timeout_steps'] = max(int(phase['timeout_steps']), rotation_steps, motion_timeout)
            # Fixed-joint relaxation/capture can skip insertion waypoints and
            # jump to release. A friction grasp must satisfy the ordinary goal.
            for key in list(skill):
                if key.startswith('relax_fixed_attachment_') or key in {
                    'keep_fixed_attachment_until_release_lock',
                    'insertion_capture_transition_phase',
                    'insertion_capture_transition_position_tolerance',
                }:
                    skill.pop(key)
            if skill.get('relative_to_current_tcp'):
                skill.update(
                    inherit_grasp_drive_target=True,
                    grasp_drive_position_lookahead=0.002,
                    cartesian_position_step=0.00025,
                    ik_position_tolerance=0.000005,
                    ik_orientation_tolerance=0.001,
                )
                phase['timeout_steps'] = max(int(phase['timeout_steps']), 4000)

            _apply_seated_release_gate(skill, release_settings)
            part_id = str(skill.get('object', '')).rsplit('_', 1)[-1]
            if (
                part_id in spec.get('physical_insertion_entry_offsets_m', {})
                and skill.get('target_object_target') == f'part_{part_id}_insert_00'
            ):
                for key in list(skill):
                    if key.startswith('relaxed_'):
                        skill.pop(key)
                skill.update(
                    position_tolerance=0.001,
                    target_object_position_tolerance=0.001,
                    target_object_axial_position_tolerance=0.001,
                    target_object_lateral_position_tolerance=0.001,
                    target_object_orientation_tolerance=0.03,
                    target_object_stable_steps=24,
                    target_object_capture_requires_tcp=False,
                    target_object_servo_position_command_lookahead=0.0005,
                    compliant_servo_position_command_lookahead=0.0005,
                )

        if old_name.endswith('_release_and_lock'):
            prefix = old_name.removesuffix('_release_and_lock')
            lock = locks[0]
            robot = next(iter(phase['gripper_commands']))
            assembled = phase['advance']['objects']
            releases[prefix] = (lock, robot, assembled)
            phase['name'] = f'{prefix}_release'
            phase['advance'] = {
                'type': 'all_of',
                'min_steps': 24,
                'conditions': [
                    {'type': 'objects_released', 'objects': assembled},
                    {'type': 'gripper_open', 'robot': robot, 'minimum_openness': 0.95},
                ],
            }
        result.append(phase)
        if old_name.endswith('_park') and old_name.removesuffix('_park') in releases:
            prefix = old_name.removesuffix('_park')
            lock, robot, assembled = releases[prefix]
            result.append(settle_phase(f'{prefix}_unloaded_settle', [goals[name] for name in assembled], robot))
            if lock.get('rebase_targets'):
                result.append(
                    {
                        'name': f'{prefix}_rebase_targets',
                        'timeout_steps': 120,
                        'robot_targets': {},
                        'gripper_commands': {robot: 'open'},
                        'rebase_from_object': {
                            'object': lock['object'],
                            'anchor_target': lock['target'],
                            'targets': copy.deepcopy(lock['rebase_targets']),
                            'position_tolerance': lock['position_tolerance'],
                            'orientation_tolerance': lock['orientation_tolerance'],
                            'require_released': True,
                        },
                        'advance': {'type': 'targets_rebased', 'object': lock['object']},
                    }
                )
    if not result[-1]['name'].endswith('_unloaded_settle'):
        raise ValueError('Physical plumbers must retain its final retreat, park and unloaded settling window.')
    from roboassemblybench.core.plumbers_support import (
        add_plumbers_base_support,
        support_grasp_settings,
    )

    add_plumbers_base_support(result, spec=spec, settings=support_grasp_settings(spec))
    from roboassemblybench.core.plumbers_junction import apply_junction_support

    apply_junction_support(result, spec=spec)
    from roboassemblybench.core.plumbers_release import apply_pin_release_sequence

    apply_pin_release_sequence(result, spec=spec)
    result[-1]['physical_unloaded_verification'] = True
    idle_robot = str(spec.get('assembly_robot', 'franka_left'))
    for index, phase in enumerate(result):
        # Anti-droop is only for the empty inserting arm while it waits. Leave
        # initial settling and the base arm's entire control path unchanged.
        skill = phase.get('local_skill') or {}
        active = (
            skill.get('robot') == idle_robot
            or idle_robot in (phase.get('local_skills') or {})
            or (phase.get('robot_targets') or {}).get(idle_robot) is not None
        )
        enabled = bool(idle_hold and index > 0 and not active)
        phase['physical_idle_hold_anchor'] = enabled
        phase['physical_idle_hold_robot'] = idle_robot if enabled else None
    from toolkits.factory_dual_franka_assembly.manipulation_stabilization import (
        annotate_active_parts,
    )

    annotate_active_parts(result)
    phases[:] = result
