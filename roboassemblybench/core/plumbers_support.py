"""Side-pinch the exposed base flange while the partner assembles plumbers.

This pass consumes already-physical phases. It does not pin the base, change
assembly targets, or borrow Beam's grasps/choreography. Every insertion gets a
fresh support grasp by default. Continuous support can span all insertions,
with free-part stability checks between them and an unloaded final check.
"""
from __future__ import annotations

import copy
import math

from toolkits.factory_dual_franka_assembly.planner_primitives import quat_conjugate, quat_rotate
from toolkits.factory_dual_franka_assembly.ur5e_skill_api import UR5E_ASSEMBLY_ADAPTER


def add_terminal_joint_home(phases, *, spec, robots):
    """Replace only final parking with real joint-home moves after clearance."""
    raw = spec.get('physical_terminal_home', {'enabled': False})
    if not isinstance(raw, dict) or type(raw.get('enabled')) is not bool:
        raise ValueError('physical_terminal_home requires an enabled boolean')
    if not raw['enabled']:
        return
    from roboassemblybench.core.fabrica_canonical import _initial_arm_joint_positions, _skill_phase
    tolerance, step = float(raw.get('joint_tolerance', .025)), float(raw.get('max_joint_step', .001))
    if not (math.isfinite(tolerance) and 0 < tolerance <= .05 and math.isfinite(step) and 0 < step <= .004):
        raise ValueError('Invalid terminal home joint tolerance or step')
    left, right = spec.get('assembly_robot', 'franka_left'), spec.get('base_robot', 'franka_right')
    authored = {r['name']: r for r in robots}
    home = {r: _initial_arm_joint_positions(authored[r]) for r in (left, right)}
    final = phases[-1]
    if not final.get('physical_unloaded_verification'):
        raise ValueError('Terminal home must precede the final unloaded verification')
    park_indices = {}
    for i, phase in enumerate(phases[:-1]):
        skill = phase.get('local_skill') or {}
        if phase['name'].endswith('_park') and skill.get('robot') in home:
            park_indices[skill['robot']] = i
    if set(park_indices) != set(home):
        raise ValueError('Terminal home requires both final parking phases')

    def home_skill(robot):
        return _skill_phase(skill='move_arm_to_joint_positions', robot=robot, phase_name='home', timeout_steps=4800,
                            parameters=dict(joint_positions=home[robot], joint_position_tolerance=tolerance,
                                            joint_target_stable_steps=24, max_joint_step=step, max_command_joint_step=step,
                                            max_command_wrist_joint_step=step, use_command_warm_start=True,
                                            ik_reference_mode='command', limit_command_to_measured_state=True,
                                            max_command_tracking_error=.06, max_wrist_command_tracking_error=.06,
                                            gripper_command='open', requires_empty_arm=True, requires_held_object=False,
                                            require_success=True, fallback='fail'))['local_skill']

    for robot, index in park_indices.items():
        phase = phases[index]
        phase['local_skill'] = home_skill(robot)
        phase['advance'] = dict(type='local_skill_complete', robot=robot, skill='move_arm_to_joint_positions', min_steps=24)
        phase.update(timeout_steps=4800, terminal_joint_home=True, physical_idle_hold_anchor=False,
                     physical_idle_hold_robot=None)
    # Only the final support withdrawal changes; intermediate assembly paths
    # keep their existing contact choreography and pose targets.
    right_park = park_indices[right]
    retreat = phases[right_park - 1]
    if not retreat['name'].endswith('_retreat'):
        raise ValueError('Terminal right home requires a preceding clearance retreat')
    retreat['local_skill'].update(object=retreat['local_skill']['offset_reference_object'],
                                  requires_held_object=False, free_space_trajectory=True, free_space_stage='approach',
                                  free_space_linear_speed=.01, free_space_angular_speed=.08,
                                  free_space_joint_speed=.25, free_space_joint_acceleration=.6,
                                  free_space_position_lookahead=.003, free_space_orientation_lookahead=.02,
                                  free_space_joint_lookahead=.06)
    # Keep the already homed left arm there while the right opens, withdraws
    # and goes home. Never replace the right hand's earlier support command.
    for phase in phases[park_indices[left] + 1:]:
        phase.setdefault('local_skills', {})[left] = home_skill(left)
        phase['physical_idle_hold_anchor'] = False
        phase['physical_idle_hold_robot'] = None
    verify = dict(name='terminal_both_arms_home', timeout_steps=4800, robot_targets={},
                  local_skills={r: home_skill(r) for r in home}, gripper_commands={r: 'open' for r in home},
                  physical_idle_hold_anchor=False, physical_idle_hold_robot=None,
                  advance=dict(type='all_of', min_steps=24, conditions=[
                      *[dict(type='local_skill_complete', robot=r, skill='move_arm_to_joint_positions') for r in home],
                      *[dict(type='gripper_open', robot=r, minimum_openness=.95) for r in home]]))
    phases.insert(len(phases) - 1, verify)
    final.setdefault('local_skills', {}).update({r: home_skill(r) for r in home})
    final.setdefault('gripper_commands', {}).update({r: 'open' for r in home})


def support_grasp_settings(spec: dict) -> dict:
    raw = spec.get('physical_base_support', {'enabled': False})
    if not isinstance(raw, dict) or type(raw.get('enabled')) is not bool:
        raise ValueError('physical_base_support requires an explicit enabled boolean.')
    if not raw['enabled']:
        return {'enabled': False}
    continuous = raw.get('hold_between_insertions', False)
    if type(continuous) is not bool:
        raise ValueError('physical_base_support.hold_between_insertions must be a boolean.')
    try:
        center = [float(v) for v in raw.get('grasp_center_m', [.074, .0012658, .022])]
        tilt = float(raw.get('approach_tilt_degrees', 30.))
        pad = float(raw.get('pad_center_m', .143))
        approach = float(raw.get('approach_distance_m', .08))
        clearance = float(raw.get('coarse_tcp_height_above_base_origin_m', .30))
        coarse_speed = float(raw.get('coarse_transfer_speed_m_s', .06))
    except (ValueError, TypeError) as exc:
        raise ValueError('physical_base_support geometry must contain finite numeric vectors/scalars.') from exc
    if (len(center) != 3 or not all(math.isfinite(v) for v in center)
            or not .060 <= center[0] <= .078 or abs(center[1] - .0012658) > .002
            or not .020 <= center[2] <= .028):
        raise ValueError('physical_base_support.grasp_center_m must target the exposed positive-X low flange.')
    if not math.isfinite(tilt) or not 20. <= tilt <= 40.:
        raise ValueError('physical_base_support.approach_tilt_degrees must be in [20, 40].')
    if not math.isfinite(pad) or not .13 <= pad <= .15:
        raise ValueError('physical_base_support.pad_center_m must be in [0.13, 0.15] m.')
    if not math.isfinite(approach) or not .04 <= approach <= .12:
        raise ValueError('physical_base_support.approach_distance_m must be in [0.04, 0.12] m.')
    if not math.isfinite(clearance) or not .25 <= clearance <= .35:
        raise ValueError('physical_base_support.coarse_tcp_height_above_base_origin_m must be in [0.25, 0.35] m.')
    if not math.isfinite(coarse_speed) or not .02 <= coarse_speed <= .12:
        raise ValueError('physical_base_support.coarse_transfer_speed_m_s must be in [0.02, 0.12] m/s.')
    # Native tool +Z aims inward/down; native Y is the closing axis. The pad
    # centre sits slightly above the 20 mm flange so the tilted distal pad
    # overlaps its side without putting the fingertip below the board plane.
    half = .5 * math.radians(90. - tilt)
    orientation = [0., math.cos(half), 0., -math.sin(half)]
    inward = quat_rotate(orientation, [0., 0., 1.])
    tcp = [c - pad * a for c, a in zip(center, inward)]
    inverse = quat_conjugate(orientation)
    return {
        'enabled': True, 'hold_between_insertions': continuous,
        'grasp_center_m': center, 'approach_tilt_degrees': tilt,
        'pad_center_m': pad, 'approach_distance_m': approach,
        'coarse_tcp_height_above_base_origin_m': clearance,
        'coarse_transfer_speed_m_s': coarse_speed,
        'tcp_in_object_position': tcp, 'tcp_in_object_orientation': orientation,
        'object_in_tcp_position': [-float(x) for x in quat_rotate(inverse, tcp)],
        'object_in_tcp_orientation': list(inverse),
        'contact_box_offset': [.065, .0012658, .010],
        'contact_box_scale': [.030, .040, .020],
        'jaw_width_thickness_m': .040,
        'geometry_status': 'native_pad_geometric_candidate_requires_physics_validation',
    }


def add_plumbers_base_support(phases: list[dict], *, spec: dict, settings: dict) -> None:
    if not settings['enabled']:
        return
    from roboassemblybench.core.fabrica_canonical import _skill_phase

    right, left = str(spec.get('base_robot', 'franka_right')), str(spec.get('assembly_robot', 'franka_left'))
    if left == right:
        raise ValueError('physical_base_support requires distinct base_robot and assembly_robot.')
    base = 'fabrica_plumbers_block_2'
    by_name = {p['name']: p for p in phases}
    suffixes = ('move_above', 'preshape', 'descend', 'close_and_attach')
    if any('base_2_' + s not in by_name for s in suffixes):
        raise ValueError('physical_base_support requires the canonical plumbers base-2 phase graph.')
    # Reuse the same physically qualified close/pose-gate parameters as base
    # pickup, but replace *all* grasp geometry and remove pickup prealignment.
    templates = {s: copy.deepcopy(by_name['base_2_' + s]) for s in suffixes}
    outward = [-float(x) for x in quat_rotate(settings['tcp_in_object_orientation'], [0., 0., 1.])]
    distance = settings['approach_distance_m']
    clearance = settings['coarse_tcp_height_above_base_origin_m']
    offset = [distance * x for x in outward]
    grasp = {'grasp_relative_position': settings['object_in_tcp_position'],
             'grasp_relative_orientation': settings['object_in_tcp_orientation']}
    contact = {key: copy.deepcopy(settings[key]) for key in
               ('contact_box_offset', 'contact_box_scale', 'jaw_width_thickness_m')}
    contact['physical_contact_axes'] = 'y'
    result = []
    active = None

    def hold_skill():
        return {
            'name': 'ur5e_hold_part_end', 'adapter': UR5E_ASSEMBLY_ADAPTER, 'robot': right, 'object': base,
            'requires_held_object': True, 'hold_previous_grasp_command': True,
            'regulate_physical_carry': True, 'continuous_physics': True,
            'track_published_grasp_command': True, 'use_arm_ik_controller': False,
            'gripper_command': 'contact_hold', 'require_success': True, 'fallback': 'fail',
            'max_command_tracking_error': .18, 'max_wrist_command_tracking_error': .12,
        }

    def support_pick(prefix):
        output = []
        for suffix in suffixes:
            p = copy.deepcopy(templates[suffix])
            p.update(name=prefix + '_' + suffix, plumbers_support_cycle=prefix)
            skill = p['local_skill']
            skill.update(robot=right, object=base, require_success=True, fallback='fail',
                         align_grasp_axis_to_world_down=False, use_arm_ik_controller=False)
            if suffix != 'preshape':
                skill.update(copy.deepcopy(grasp))
            if suffix == 'move_above':
                for key in list(skill):
                    if key.startswith('prealign_'):
                        skill.pop(key)
                skill.update(offset=offset, offset_frame='object', orientation_first_before_translation=False,
                             cartesian_position_step=.001, cartesian_orientation_step=.004,
                             max_command_joint_step=.006, max_command_wrist_joint_step=.006,
                             position_tolerance=.002, orientation_tolerance=.025)
                # Last-published FK drives each reference, rather than a tiny
                # step continually restarted from the measured position.
                for stage in ('lift', 'orient', 'transfer'):
                    coarse = copy.deepcopy(p)
                    coarse.update(name=prefix + '_coarse_' + stage, timeout_steps=4000)
                    coarse['local_skill'].update(
                        free_space_trajectory=True, free_space_stage=stage,
                        coarse_tcp_height_above_base_origin_m=clearance,
                        position_tolerance=.003, orientation_tolerance=.025,
                        free_space_linear_speed=(settings['coarse_transfer_speed_m_s'] if stage == 'transfer' else .06),
                        free_space_angular_speed=.35,
                        free_space_joint_speed=.60, free_space_joint_acceleration=1.2,
                        free_space_position_lookahead=.015, free_space_orientation_lookahead=.06,
                        free_space_joint_lookahead=.12, trace_pregrasp=True)
                    output.append(coarse)
                skill.update(free_space_trajectory=True, free_space_stage='lower',
                             coarse_tcp_height_above_base_origin_m=clearance,
                             free_space_linear_speed=.025, free_space_angular_speed=.15,
                             free_space_joint_speed=.35, free_space_joint_acceleration=.8,
                             free_space_position_lookahead=.005, free_space_orientation_lookahead=.03,
                             free_space_joint_lookahead=.10, trace_pregrasp=True)
                p['timeout_steps'] = max(p['timeout_steps'], 4000)
            elif suffix == 'preshape':
                skill.update(gripper_openness=.75, minimum_gripper_openness=.60)
            elif suffix == 'descend':
                skill.update(gripper_command=.75, cartesian_position_step=.00025,
                             cartesian_orientation_step=.002, max_command_joint_step=.004,
                             max_command_wrist_joint_step=.004, position_tolerance=.001,
                             relaxed_position_tolerance=.001, orientation_tolerance=.02)
                skill.update(free_space_trajectory=True, free_space_stage='approach',
                             free_space_linear_speed=.01, free_space_angular_speed=.08,
                             free_space_joint_speed=.25, free_space_joint_acceleration=.6,
                             free_space_position_lookahead=.003, free_space_orientation_lookahead=.02,
                             free_space_joint_lookahead=.06, trace_pregrasp=True)
                p['timeout_steps'] = max(p['timeout_steps'], 4000)
            else:
                skill.update(copy.deepcopy(contact), preclose_openness=.75)
                attach = p['attach'][0]
                attach.update(copy.deepcopy(contact))
                attach['target'].update(offset=copy.deepcopy(settings['tcp_in_object_position']),
                                        orientation=copy.deepcopy(settings['tcp_in_object_orientation']))
            output.append(p)
        support = {'robot': right, 'inserting_robot': left, 'object': base, 'cycle': prefix,
                   'max_position_drift': .003, 'max_orientation_drift': .0872664626,
                   'contact_box_offset': copy.deepcopy(settings['contact_box_offset']),
                   'contact_box_scale': copy.deepcopy(settings['contact_box_scale'])}
        output.append({'name': prefix + '_ready', 'timeout_steps': 1200, 'robot_targets': {},
                       'local_skill': hold_skill(), 'gripper_commands': {right: 'contact_hold'},
                       'plumbers_base_support': support, 'plumbers_support_cycle': prefix,
                       'advance': {'type': 'plumbers_support_ready'}})
        return output, support

    def support_release(prefix):
        # The left arm has already opened, retreated and parked. Opening the
        # right jaw is a real release, not a force-free detach/velocity reset.
        release = {'name': prefix + '_release', 'timeout_steps': 1200, 'robot_targets': {},
                   'gripper_commands': {right: 'open', left: 'open'},
                   'detach': [{'object': base, 'robot': right, 'release_min_steps': 0}],
                   'plumbers_support_cycle': prefix,
                   'advance': {'type': 'all_of', 'min_steps': 24, 'conditions': [
                       {'type': 'objects_released', 'objects': [base]},
                       {'type': 'gripper_open', 'robot': right, 'minimum_openness': .95}]}}
        retreat = _skill_phase(skill='retreat_vertical', robot=right,
                               phase_name=prefix + '_retreat', timeout_steps=4000, parameters={
                                   'relative_to_current_tcp': True, 'offset': offset, 'offset_frame': 'object',
                                   'offset_reference_object': base, 'gripper_command': 'open',
                                   'lock_target_position': True, 'lock_target_orientation': True,
                                   'cartesian_position_step': .0005, 'max_command_joint_step': .004,
                                   'max_command_wrist_joint_step': .004, 'use_arm_ik_controller': False,
                                   'position_tolerance': .002, 'orientation_tolerance': .025,
                                   'use_command_warm_start': True, 'require_warm_start_ik': True,
                                   'require_success': True, 'fallback': 'fail'})
        park = copy.deepcopy(by_name['base_2_park'])
        park.update(name=prefix + '_park', plumbers_support_cycle=prefix)
        return [release, retreat, park]

    for phase in phases:
        name = phase['name']
        if name.startswith('assemble_') and name.endswith('_move_above'):
            if active is not None:
                raise ValueError('Previous plumbers support cycle did not release after the left park.')
            assembly_prefix = name.removesuffix('_move_above')
            support_prefix = 'support_' + assembly_prefix.removeprefix('assemble_') + '_base_2'
            additions, support = support_pick(support_prefix)
            result.extend(additions)
            active = assembly_prefix, support_prefix, support
        if active is not None:
            assembly_prefix, support_prefix, support = active
            if not name.startswith(assembly_prefix + '_') or name.endswith('_unloaded_settle'):
                raise ValueError(f'Unexpected stage inside plumbers support cycle: {name}')
            phase['plumbers_base_support'] = copy.deepcopy(support)
            phase.setdefault('local_skills', {})[right] = hold_skill()
            phase.setdefault('gripper_commands', {})[right] = 'contact_hold'
            phase['advance'] = {'type': 'all_of', 'conditions': [
                phase['advance'], {'type': 'plumbers_support_ready'}]}
            if name.endswith('_release'):
                # Only the just-inserted left-hand part must be released here.
                # Requiring the supported base to be released would deadlock.
                release_gate = phase['advance']['conditions'][0]['conditions'][0]
                release_gate['objects'] = [o for o in release_gate['objects'] if o != base]
                released = phase['detach'][0]['object']
                phase['local_skill'] = {
                    'name': 'ur5e_release_supported_part', 'adapter': UR5E_ASSEMBLY_ADAPTER,
                    'robot': left, 'object': released, 'requires_held_object': False,
                    'require_success': True, 'fallback': 'fail'}
            result.append(phase)
            if name == assembly_prefix + '_park':
                result.extend(support_release(support_prefix))
                active = None
        else:
            result.append(phase)
    if active is not None:
        raise ValueError('Unfinished plumbers support cycle at end of phase graph.')
    if not any(p.get('plumbers_base_support') for p in result):
        raise ValueError('physical_base_support requires at least one left assembly cycle.')
    if settings['hold_between_insertions']:
        _retain_support_across_insertions(result, base=base, robot=right, hold_skill=hold_skill)
    phases[:] = result


def _retain_support_across_insertions(phases, *, base, robot, hold_skill):
    """Keep the first real flange grasp and its fixed anchor until the end."""
    cycles = [(p['name'].removesuffix('_move_above'), copy.deepcopy(p['plumbers_base_support']))
              for p in phases if p['name'].startswith('assemble_') and p['name'].endswith('_move_above')]
    if not cycles:
        raise ValueError('Continuous support requires at least one insertion cycle.')
    first = cycles[0][1]['cycle']
    remove = set()
    intermediate = {}
    for index, (prefix, support) in enumerate(cycles):
        cycle = support['cycle']
        if index:
            remove.update(p['name'] for p in phases
                          if p.get('plumbers_support_cycle') == cycle
                          and p['name'] not in {cycle + suffix for suffix in ('_release', '_retreat', '_park')})
        if index < len(cycles) - 1:
            remove.update(cycle + suffix for suffix in ('_release', '_retreat', '_park'))
            intermediate[prefix + '_unloaded_settle'] = support
    retained = []
    for phase in phases:
        if phase['name'] in remove:
            continue
        support = intermediate.get(phase['name'])
        if support is not None:
            phase['name'] = phase['name'].removesuffix('_unloaded_settle') + '_supported_settle'
            phase['physical_supported_verification'] = True
            phase['advance']['type'] = 'plumbers_supported_settle'
            phase['advance']['objects'] = [o for o in phase['advance']['objects'] if o['object'] != base]
            # Active support is not unloaded rest: verify the released parts
            # stay seated relative to the held base. The final unloaded gate
            # still checks measured world velocities on every physics step.
            phase['advance']['position_drift_tolerance'] = .0005
            phase['advance']['orientation_drift_tolerance'] = math.radians(.5)
            phase['plumbers_base_support'] = copy.deepcopy(support)
            phase['local_skills'] = {robot: hold_skill()}
            phase.setdefault('gripper_commands', {})[robot] = 'contact_hold'
        if phase.get('plumbers_base_support'):
            # This identity belongs to the real grasp, not to an assembly step.
            phase['plumbers_base_support']['cycle'] = first
        retained.append(phase)
    phases[:] = retained
