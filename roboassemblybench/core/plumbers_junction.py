"""Physical right-hand regrasp across the cap/base seam before the two pins."""
from __future__ import annotations

import copy
import math

from toolkits.factory_dual_franka_assembly.planner_primitives import quat_conjugate, quat_rotate


def junction_support_settings(spec):
    raw = spec.get('physical_junction_support', {'enabled': False})
    if not isinstance(raw, dict) or type(raw.get('enabled')) is not bool:
        raise ValueError('physical_junction_support requires an explicit enabled boolean')
    if not raw['enabled']:
        return {'enabled': False}
    if not (spec.get('physical_base_support') or {}).get('hold_between_insertions'):
        raise ValueError('Junction regrasp requires continuous physical base support')
    center = [float(v) for v in raw.get('grasp_center_m', [.045, .0012658, .072])]
    if (len(center) != 3 or not all(math.isfinite(v) for v in center)
            or not .025 <= center[0] <= .047 or abs(center[1] - .0012658) > .002
            or not .065 <= center[2] <= .075):
        raise ValueError('physical_junction_support.grasp_center_m must overlap the positive-X cap/base seam')
    tilt = float(raw.get('approach_tilt_degrees', 30.))
    pad = float(raw.get('pad_center_m', .143))
    if not (math.isfinite(tilt) and 20 <= tilt <= 40 and math.isfinite(pad) and .13 <= pad <= .15):
        raise ValueError('Invalid physical_junction_support tilt or native pad center')
    angle = .5 * math.radians(90 - tilt)
    q = [0., math.cos(angle), 0., -math.sin(angle)]
    direction = quat_rotate(q, [0., 0., 1.])
    tcp = [c - pad * d for c, d in zip(center, direction)]
    inv = quat_conjugate(q)
    from toolkits.factory_dual_franka_assembly.physical_hold_force import hold_force_settings
    force_control = hold_force_settings({**spec.get('hold_force_control', {}), **raw.get('hold_force_control', {})})
    return dict(enabled=True, grasp_center_m=center, tcp=tcp, orientation=q,
                hold_force_control=force_control,
                relative_position=[-float(x) for x in quat_rotate(inv, tcp)], relative_orientation=list(inv),
                contact_box_offset=[center[0], center[1], .073], contact_box_scale=[.028, .040, .016],
                additional_grasp_contacts=[dict(object='fabrica_plumbers_block_2',
                                                contact_box_offset=[center[0], center[1], .057], contact_box_scale=[.028, .040, .016],
                                                minimum_finger_force=1., max_position_drift=.001, max_orientation_drift=math.radians(2.))])


def _temporary_left_support(by_name, additions, *, right, left, new_support, new_hold):
    """A real handover keeps the cap supported while the right jaw is empty."""
    from roboassemblybench.core.fabrica_canonical import _skill_phase

    cap = 'fabrica_plumbers_block_3'
    prefix = 'support_junction_left_temporary'
    old = by_name['assemble_01_part_3_supported_settle']
    old_hold = copy.deepcopy(old['local_skills'][right])
    q = [.10271068179660436, 0., .9947112726037022, 0.]
    center = [-.045, .0012658, .0843]
    tcp = [c - .143 * d for c, d in zip(center, quat_rotate(q, [0., 0., 1.]))]
    inv = quat_conjugate(q)
    geometry = dict(grasp_relative_position=[-float(x) for x in quat_rotate(inv, tcp)],
                    grasp_relative_orientation=list(inv))
    contact = dict(contact_box_offset=[-.041, .0012658, .080], contact_box_scale=[.025, .040, .025],
                   physical_contact_axes='y', jaw_width_thickness_m=.040)
    common = dict(object=cap, **geometry, use_arm_ik_controller=False, require_success=True, fallback='fail',
                  lock_target_position=True, lock_target_orientation=True, position_tolerance=.001,
                  orientation_tolerance=.02, free_space_trajectory=True, free_space_stage='approach',
                  free_space_linear_speed=.06, free_space_angular_speed=.16,
                  free_space_joint_speed=.4, free_space_joint_acceleration=1.,
                  free_space_position_lookahead=.005, free_space_orientation_lookahead=.03,
                  free_space_joint_lookahead=.06, gripper_command='open')
    approach = _skill_phase(skill='move_above_part', robot=left, phase_name=prefix + '_approach', timeout_steps=8000,
                            parameters=dict(common, offset=[0., 0., .08], offset_frame='world'))
    descend = _skill_phase(skill='descend_to_grasp', robot=left, phase_name=prefix + '_descend', timeout_steps=4000,
                           parameters=dict(common, offset=[0., 0., 0.], offset_frame='world', free_space_linear_speed=.01,
                                           free_space_angular_speed=.08, free_space_joint_speed=.25,
                                           free_space_joint_acceleration=.6, free_space_position_lookahead=.003,
                                           free_space_orientation_lookahead=.02, gripper_command=.75))
    preshape = _skill_phase(skill='preshape_gripper', robot=left, phase_name=prefix + '_preshape', timeout_steps=1200,
                            parameters=dict(object=cap, gripper_openness=.75, minimum_gripper_openness=.60))
    close = copy.deepcopy(by_name['support_00_part_0_base_2_close_and_attach'])
    close['name'] = prefix + '_close'
    close.pop('plumbers_support_cycle', None)
    close['local_skill'].update(robot=left, object=cap, **geometry, **contact)
    close['attach'][0].update(robot=left, object=cap, **contact, allow_shared_physical_grasp=True)
    # This hand only supports the cap during handover. Use a slower, lower
    # static preload than the hand resisting the later insertion force.
    close['attach'][0]['hold_force_control'].update(low=4., target=6., joint_rate=.005, unload_rate=.01)
    close['attach'][0]['target'] = dict(reference=cap, offset=tcp, orientation=q, ik_frame_compensation='none')
    close['advance'] = dict(type='all_of', conditions=[
        dict(type='local_skill_complete', robot=left, skill='ur5e_close_gripper', min_steps=24),
        dict(type='object_attached', object=cap, robot=left), dict(type='plumbers_support_ready')])
    close['gripper_commands'] = {left: 'close', right: 'contact_hold'}
    left_hold = dict(new_hold, robot=left, object=cap)
    temporary = dict(new_support, robot=left, inserting_robot=right, cycle=prefix,
                     contact_box_offset=contact['contact_box_offset'], contact_box_scale=contact['contact_box_scale'])
    align = _skill_phase(skill='ur5e_move_part_to_staging', robot=left, phase_name=prefix + '_align_seam',
                         timeout_steps=2400, parameters=dict(
                             left_hold, target_object_target='part_3_assembled', target_object_reference_object='fabrica_plumbers_block_2',
                             derive_tcp_orientation_from_target_object=True,
                             physical_rotation_governor=True, physical_rotation_speed=.03,
                             physical_rotation_acceleration=.1, physical_rotation_lookahead=.003,
                             orientation_first_before_translation=False, ik_retry_with_measured_state=False,
                             ik_reference_mode='command', use_command_warm_start=True, require_warm_start_ik=True,
                             ik_position_tolerance=1e-6, ik_orientation_tolerance=1e-8,
                             target_object_servo_position_command_warm_start=True,
                             target_object_servo_position_command_gate_overdrive=False,
                             target_object_servo_position_command_lookahead=.0005,
                             target_object_servo_position_command_accumulation_step=.00005,
                             hold_previous_grasp_command=False, position_tolerance=.0002, orientation_tolerance=.003,
                             target_object_position_tolerance=.0002, target_object_orientation_tolerance=.003,
                             require_target_object_pose_convergence=True, target_object_capture_requires_tcp=False,
                             cartesian_position_step=.00005, cartesian_orientation_step=.0003,
                             max_command_position_lead=.0005, max_command_joint_step=.002,
                             max_command_wrist_joint_step=.002, gripper_command='contact_hold',
                             lock_target_position=True, lock_target_orientation=True, target_object_stable_steps=24))
    initial = [approach, preshape, descend, close, align]
    for phase in initial:
        phase['plumbers_base_support'] = copy.deepcopy(old['plumbers_base_support'])
        phase.setdefault('local_skills', {})[right] = copy.deepcopy(old_hold)
        phase.setdefault('gripper_commands', {})[right] = 'contact_hold'
    for phase in additions:
        phase.setdefault('local_skills', {})[left] = copy.deepcopy(left_hold)
        phase.setdefault('gripper_commands', {})[left] = 'contact_hold'
        if not phase['name'].endswith('_ready'):
            phase['plumbers_base_support'] = copy.deepcopy(temporary)
        for attach in phase.get('attach', []):
            if attach['object'] == cap:
                attach['allow_shared_physical_grasp'] = True
    release = dict(name=prefix + '_release', robot_targets={}, timeout_steps=1200,
                   local_skill=dict(name='ur5e_release_supported_part', robot=left, object=cap,
                                    requires_held_object=False, require_success=True, fallback='fail'),
                   local_skills={right: copy.deepcopy(new_hold)}, plumbers_base_support=copy.deepcopy(new_support),
                   gripper_commands={left: 'open', right: 'contact_hold'},
                   detach=[dict(object=cap, robot=left, release_min_steps=0)],
                   advance=dict(type='all_of', min_steps=24, conditions=[
                       dict(type='gripper_open', robot=left, minimum_openness=.95), dict(type='plumbers_support_ready')]))
    retreat = _skill_phase(skill='retreat_vertical', robot=left, phase_name=prefix + '_retreat', timeout_steps=4000,
                           parameters=dict(common, requires_held_object=False, relative_to_current_tcp=True,
                                           offset=[0., 0., .06], offset_frame='world', gripper_command='open'))
    park = copy.deepcopy(by_name['assemble_01_part_3_park'])
    park['name'] = prefix + '_park'
    for phase in (retreat, park):
        phase['plumbers_base_support'] = copy.deepcopy(new_support)
        phase.setdefault('local_skills', {})[right] = copy.deepcopy(new_hold)
        phase.setdefault('gripper_commands', {}).update({left: 'open', right: 'contact_hold'})
    return initial + additions + [release, retreat, park]


def apply_junction_support(phases, *, spec):  # noqa: C901
    settings = junction_support_settings(spec)
    if not settings['enabled']:
        return
    from roboassemblybench.core.fabrica_canonical import _skill_phase

    cap, base = 'fabrica_plumbers_block_3', 'fabrica_plumbers_block_2'
    right, left = spec.get('base_robot', 'franka_right'), spec.get('assembly_robot', 'franka_left')
    by_name = {p['name']: p for p in phases}
    after = 'assemble_01_part_3_supported_settle'
    if after not in by_name:
        raise ValueError('Junction support must follow the verified part_3 assembly')
    old_cycle = by_name[after]['plumbers_base_support']['cycle']
    cycle = 'support_junction_cap_base'
    templates = [copy.deepcopy(p) for p in phases if p.get('plumbers_support_cycle') == old_cycle
                 and p['name'].removeprefix(old_cycle) in (
                     '_coarse_lift', '_coarse_orient', '_coarse_transfer', '_move_above',
                     '_preshape', '_descend', '_close_and_attach')]
    if len(templates) != 7:
        raise ValueError('Missing physical support pickup templates for junction regrasp')
    support = dict(robot=right, inserting_robot=left, object=cap, cycle=cycle,
                   max_position_drift=.003, max_orientation_drift=math.radians(5.),
                   contact_box_offset=settings['contact_box_offset'], contact_box_scale=settings['contact_box_scale'],
                   region='cap_base_junction')
    hold = copy.deepcopy(by_name[after]['local_skills'][right])
    hold['object'] = cap
    release = dict(name=cycle + '_release_low_flange', robot_targets={}, timeout_steps=1200,
                   local_skill=dict(name='ur5e_release_supported_part', robot=right, object=base,
                                    requires_held_object=False, require_success=True, fallback='fail'),
                   gripper_commands={right: 'open', left: 'open'},
                   detach=[dict(object=base, robot=right, release_min_steps=0)],
                   advance=dict(type='all_of', min_steps=24, conditions=[
                       dict(type='objects_released', objects=[base]),
                       dict(type='gripper_open', robot=right, minimum_openness=.95)]))
    outward = [-float(x) for x in quat_rotate(settings['orientation'], [0., 0., 1.])]
    retreat = _skill_phase(skill='retreat_vertical', robot=right, phase_name=cycle + '_withdraw_low_flange',
                           timeout_steps=4000, parameters=dict(object=base, requires_held_object=False, relative_to_current_tcp=True,
                                                               offset=[.08 * x for x in outward], offset_frame='world',
                                                               lock_target_position=True, lock_target_orientation=True, gripper_command='open',
                                                               use_arm_ik_controller=False, require_success=True, fallback='fail',
                                                               free_space_trajectory=True, free_space_stage='approach',
                                                               free_space_linear_speed=.02, free_space_angular_speed=.08,
                                                               free_space_joint_speed=.25, free_space_joint_acceleration=.6,
                                                               free_space_position_lookahead=.003, free_space_orientation_lookahead=.02,
                                                               free_space_joint_lookahead=.06, position_tolerance=.002, orientation_tolerance=.025))
    additions = [release, retreat]
    geometry = dict(grasp_relative_position=settings['relative_position'],
                    grasp_relative_orientation=settings['relative_orientation'])
    contact = {key: copy.deepcopy(settings[key]) for key in ('contact_box_offset', 'contact_box_scale')}
    contact.update(jaw_width_thickness_m=.040, physical_contact_axes='y')
    for phase in templates:
        phase['name'] = cycle + phase['name'].removeprefix(old_cycle)
        phase['plumbers_support_cycle'] = cycle
        skill = phase['local_skill']
        skill.update(object=cap, **copy.deepcopy(geometry))
        if phase['name'].endswith('_preshape'):
            skill.pop('grasp_relative_position', None)
            skill.pop('grasp_relative_orientation', None)
        if phase['name'].endswith(('_move_above', '_descend')) or '_coarse_' in phase['name']:
            if 'offset' in skill:
                skill['offset'] = [(.08 if not phase['name'].endswith('_descend') else 0.) * x for x in outward]
        if phase['name'].endswith('_close_and_attach'):
            phase['timeout_steps'] = 3600
            skill.update(copy.deepcopy(contact), close_force_low=8., close_force_min_stable=6.,
                         close_force_high=16., close_force_weak_side_preload=True,
                         close_force_balance_min=8., close_force_balance_peak=8.,
                         close_force_balance_step=.000025,
                         close_force_joint_rate=.01, close_force_max_command_lead=.01,
                         close_until_contact_timeout_steps=3000,
                         primary_minimum_finger_force=1.,
                         additional_grasp_contacts=copy.deepcopy(settings['additional_grasp_contacts']))
            attach = phase['attach'][0]
            attach.update(object=cap, **copy.deepcopy(contact),
                          primary_minimum_finger_force=1.,
                          hold_force_control=copy.deepcopy(settings['hold_force_control']),
                          additional_grasp_contacts=copy.deepcopy(settings['additional_grasp_contacts']))
            attach['target'] = dict(reference=cap, offset=settings['tcp'],
                                    orientation=settings['orientation'], ik_frame_compensation='none')
            phase['advance'] = dict(type='all_of', min_steps=24, conditions=[
                dict(type='local_skill_complete', robot=right, skill='ur5e_close_gripper', min_steps=24),
                dict(type='object_attached', object=cap, robot=right)])
            # The primary cap supplies the fixed grasp reference. Both body
            # loads are summed by the carrying controller after capture.
        additions.append(phase)
    additions.append(dict(name=cycle + '_ready', robot_targets={}, timeout_steps=1200,
                          local_skill=copy.deepcopy(hold), gripper_commands={right: 'contact_hold', left: 'open'},
                          plumbers_base_support=copy.deepcopy(support), advance=dict(type='plumbers_support_ready')))
    additions = _temporary_left_support(by_name, additions, right=right, left=left,
                                        new_support=support, new_hold=hold)
    # All part meshes share the CAD assembly frame. After a qualified regrasp,
    # align the insertion targets once to the measured socket frame. Object
    # poses are never rewritten and pre-rebase target tolerances still apply.
    target_names = {f'part_{i}_assembled' for i in range(5)}
    for phase in phases:
        if phase['name'].startswith(('assemble_02_part_1_', 'assemble_03_part_4_')):
            skill = phase.get('local_skill') or {}
            if any(s in phase['name'] for s in ('_assembly_clearance', '_transport_', '_insert_')):
                target_names.update(skill[key] for key in ('target_object_target', 'target_object_final_target')
                                    if skill.get(key))
    additions.append(dict(name=cycle + '_rebase_targets', robot_targets={}, timeout_steps=1200,
                          local_skill=copy.deepcopy(hold), plumbers_base_support=copy.deepcopy(support),
                          gripper_commands={right: 'contact_hold', left: 'open'},
                          rebase_from_object=dict(object=cap, anchor_target='part_3_assembled', targets=sorted(target_names),
                                                  position_tolerance=.0015, orientation_tolerance=math.radians(3.), require_released=False),
                          advance=dict(type='all_of', conditions=[dict(type='targets_rebased', object=cap),
                                                                  dict(type='plumbers_support_ready')])))
    for phase in additions:
        phase['physical_setup_pose_guard'] = dict(group=cycle, robot=right,
                                                  objects=['fabrica_plumbers_block_0', base, cap], position_limit=.0015,
                                                  position_reference_points={f'fabrica_plumbers_block_{key}': value for key, value in
                                                                             spec.get('physical_success_reference_points_m', {}).items()},
                                                  orientation_limit=math.radians(3.))
    output, switched = [], False
    for phase in phases:
        if switched and phase['name'] in ('assemble_02_part_1_preshape', 'assemble_03_part_4_preshape'):
            output.append(dict(
                name=phase['name'].removesuffix('_preshape') + '_empty_wrist_unwind',
                robot_targets={}, timeout_steps=4800,
                gripper_commands={left: 'open', right: 'contact_hold'},
                local_skill=dict(name='ur5e_unwind_empty_wrist', robot=left, winding='negative',
                                 adapter='toolkits.factory_dual_franka_assembly.empty_wrist:EmptyWristUnwindAdapter',
                                 requires_held_object=False, require_success=True, fallback='fail'),
                local_skills={right: copy.deepcopy(hold)}, plumbers_base_support=copy.deepcopy(support),
                advance=dict(type='local_skill_complete', robot=left, skill='ur5e_unwind_empty_wrist', min_steps=24),
                physical_idle_hold_anchor=False))
        if switched:
            if phase.get('plumbers_base_support'):
                phase['plumbers_base_support'] = copy.deepcopy(support)
                if right in (phase.get('local_skills') or {}):
                    phase['local_skills'][right] = copy.deepcopy(hold)
                # The cap is now the right hand's primary support body. A pin
                # release must not also demand that this continuing grasp open.

                def keep_support_attached(condition):
                    if condition.get('type') == 'objects_released':
                        condition['objects'] = [o for o in condition['objects'] if o != cap]
                    for child in condition.get('conditions', []):
                        keep_support_attached(child)
                keep_support_attached(phase.get('advance') or {})
            if phase.get('physical_supported_verification'):
                for entry in phase['advance']['objects']:
                    if entry['object'] == cap:
                        entry['require_released'] = False
            # Final opening releases the new primary cap grasp. The green
            # base stays dynamic throughout; no symbolic secondary weld exists.
            if phase['name'].startswith('support_03_part_4_base_2_'):
                suffix = phase['name'].removeprefix('support_03_part_4_base_2')
                if suffix in ('_release', '_retreat', '_park'):
                    phase['name'] = cycle + suffix
                    for detach in phase.get('detach', []):
                        if detach.get('object') == base:
                            detach['object'] = cap
                    advance = phase.get('advance') or {}
                    for condition in advance.get('conditions', []):
                        if condition.get('type') == 'objects_released':
                            condition['objects'] = [cap if x == base else x for x in condition['objects']]
        output.append(phase)
        if phase['name'] == after:
            output.extend(additions)
            switched = True
    phases[:] = output
