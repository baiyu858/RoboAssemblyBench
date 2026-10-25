"""Opt-in physical execution of the canonical UR5e plumbers task.

The canonical compiler still owns grasp selection and insertion geometry. This
pass changes how those motions interact with free parts; it has no Beam phase,
grasp-axis, support-plan or robot-pose assumptions.
"""
from __future__ import annotations

import copy
import math

from toolkits.factory_dual_franka_assembly.physical_hold_force import hold_force_settings
from toolkits.factory_dual_franka_assembly.close_force_stability import close_force_stability_settings
from toolkits.factory_dual_franka_assembly.planner_primitives import quat_conjugate, quat_rotate


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
    if (len(center_offset) != 3 or not all(math.isfinite(x) for x in center_offset)
            or sum(x * x for x in center_offset) > 0.02 ** 2):
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
            min(c - lo, hi - c) / max((hi - lo) * .5, 1e-9)
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


def apply_plumbers_physics(phases: list[dict], *, success: list[dict], spec: dict) -> None:
    close_stability = close_force_stability_settings(
        spec.get('close_force_stability', {'enabled': False}))
    if close_stability.get('hysteresis_margin', 0.) >= 4.0 - 0.1:
        raise ValueError('close_force_stability release threshold must exceed the 0.1 N contact threshold.')
    idle_hold = spec.get('physical_idle_hold_anchor', False)
    if not isinstance(idle_hold, bool):
        raise ValueError('physical_idle_hold_anchor must be an explicit boolean.')
    goals = {entry['object']: copy.deepcopy(entry) for entry in success}
    stable_steps = int(spec.get('physical_settle_steps', 120))
    if stable_steps < 2:
        raise ValueError('physical_settle_steps must be at least 2.')
    rotation_limits = {
        key: float(spec.get(key, default)) for key, default in (
            ('physical_rotation_speed', 0.10),
            ('physical_rotation_acceleration', 0.30),
            ('physical_rotation_lookahead', 0.010),
        )
    }
    if not all(math.isfinite(value) and value > 0 for value in rotation_limits.values()):
        raise ValueError('physical_rotation limits must be finite and positive.')
    if rotation_limits['physical_rotation_lookahead'] >= math.pi:
        raise ValueError('physical_rotation_lookahead must be less than pi.')
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
        'close_force_joint_rate': 0.025,
        'close_force_unload_rate': 0.06,
        'close_force_max_command_lead': 0.001,
        'close_force_max_seek': 0.03,
    }
    hold_config = spec.get('hold_force_control')
    if hold_config is not None:
        if not isinstance(hold_config, dict) or not isinstance(hold_config.get('enabled'), bool):
            raise ValueError('hold_force_control requires an explicit enabled boolean.')
        if hold_config['enabled']:
            contact_policy['hold_force_control'] = hold_force_settings(hold_config)

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
            skill.update(contact_policy)
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
            phase['timeout_steps'] = max(int(phase['timeout_steps']), 2000)
        elif skill.get('requires_held_object'):
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
                **rotation_limits,
            )
            # At 240 Hz, even a pi rotation at the configured speed must fit.
            # This extends the budget, never the physical success tolerances.
            rotation_steps = math.ceil(240 * (
                math.pi / rotation_limits['physical_rotation_speed']
                + 2 * rotation_limits['physical_rotation_speed'] / rotation_limits['physical_rotation_acceleration']
                + 5.0))
            phase['timeout_steps'] = max(int(phase['timeout_steps']), rotation_steps)
            # Fixed-joint relaxation/capture can skip insertion waypoints and
            # jump to release. A friction grasp must satisfy the ordinary goal.
            for key in list(skill):
                if key.startswith('relax_fixed_attachment_') or key in {
                    'keep_fixed_attachment_until_release_lock', 'insertion_capture_transition_phase',
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
                result.append({
                    'name': f'{prefix}_rebase_targets',
                    'timeout_steps': 120,
                    'robot_targets': {},
                    'gripper_commands': {robot: 'open'},
                    'rebase_from_object': {
                        'object': lock['object'], 'anchor_target': lock['target'],
                        'targets': copy.deepcopy(lock['rebase_targets']),
                        'position_tolerance': lock['position_tolerance'],
                        'orientation_tolerance': lock['orientation_tolerance'],
                        'require_released': True,
                    },
                    'advance': {'type': 'targets_rebased', 'object': lock['object']},
                })
    if not result[-1]['name'].endswith('_unloaded_settle'):
        raise ValueError('Physical plumbers must retain its final retreat, park and unloaded settling window.')
    from roboassemblybench.core.plumbers_support import add_plumbers_base_support, support_grasp_settings
    add_plumbers_base_support(result, spec=spec, settings=support_grasp_settings(spec))
    result[-1]['physical_unloaded_verification'] = True
    idle_robot = str(spec.get('assembly_robot', 'franka_left'))
    for index, phase in enumerate(result):
        # Anti-droop is only for the empty inserting arm while it waits. Leave
        # initial settling and the base arm's entire control path unchanged.
        skill = phase.get('local_skill') or {}
        active = (skill.get('robot') == idle_robot
                  or idle_robot in (phase.get('local_skills') or {})
                  or (phase.get('robot_targets') or {}).get(idle_robot) is not None)
        enabled = bool(idle_hold and index > 0 and not active)
        phase['physical_idle_hold_anchor'] = enabled
        phase['physical_idle_hold_robot'] = idle_robot if enabled else None
    phases[:] = result
