"""Physical base-flange support, two-arm preflight and real release sequencing."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from roboassemblybench.core.fabrica_canonical import compile_fabrica_canonical_recipe
from roboassemblybench.core.plumbers_support import support_grasp_settings
from toolkits.factory_dual_franka_assembly.planner_primitives import quat_rotate
from toolkits.factory_dual_franka_assembly.plumbers_block_ur5e_skills import UR5eAssemblyAtomicSkillAdapter
from toolkits.factory_dual_franka_assembly.plumbers_support import base_support_status
from toolkits.factory_dual_franka_assembly.task_specs import load_task_recipe


@pytest.fixture(scope='module')
def recipe():
    return load_task_recipe('fabrica_plumbers_block_ur5e_staged')


def test_flange_grasp_transform_is_invertible_and_closes_across_y(recipe):
    g = recipe['fabrica_canonical_resolved']['physical_base_support']
    assert g['enabled']
    p, q = np.array(g['tcp_in_object_position']), g['tcp_in_object_orientation']
    assert np.allclose(p + quat_rotate(q, [0, 0, g['pad_center_m']]), g['grasp_center_m'])
    assert np.allclose(p + quat_rotate(q, g['object_in_tcp_position']), 0)
    assert np.allclose(quat_rotate(q, [0, 1, 0]), [0, -1, 0])
    approach = quat_rotate(q, [0, 0, -1])
    assert approach[0] > 0 and approach[2] > 0
    lo = np.array(g['contact_box_offset']) - np.array(g['contact_box_scale']) / 2
    hi = np.array(g['contact_box_offset']) + np.array(g['contact_box_scale']) / 2
    assert np.allclose(lo, [.05, -.0187342, 0.])
    assert np.allclose(hi, [.08, .0212658, .02])


def test_support_requires_new_low_flange_grasp_without_lift_or_pin(recipe):
    phases = recipe['phases']
    support_closes = [p for p in phases if p.get('plumbers_support_cycle') and p.get('attach')]
    assert len(support_closes) == 4
    g = recipe['fabrica_canonical_resolved']['physical_base_support']
    for p in support_closes:
        a, s = p['attach'][0], p['local_skill']
        assert a['robot'] == s['robot'] == 'franka_right'
        assert a['object'] == s['object'] == 'fabrica_plumbers_block_2'
        assert a['attachment_mode'] == 'pure_physical_grasp'
        assert a['target']['reference'] == 'fabrica_plumbers_block_2'
        assert a['target']['offset'] == g['tcp_in_object_position']
        assert s['grasp_relative_position'] == g['object_in_tcp_position']
        for key in ('require_force_contact', 'require_pair_force_contact', 'require_dual_force_contact',
                    'require_opposing_force_contact', 'close_force_regulation', 'close_force_require_clamp_band'):
            assert s[key] is True
        assert a['physical_contact_axes'] == s['physical_contact_axes'] == 'y'
        assert a['contact_box_scale'] == s['contact_box_scale'] == [.03, .04, .02]
        assert not a['disable_collision_on_attach'] and not a['filter_gripper_collisions_on_attach']
        assert a['hold_position_slip_tolerance'] == .003
        assert a['hold_orientation_slip_tolerance'] == .0872664626
        assert not any(p.get(key) for key in ('lock', 'fixture_lock', 'kinematic_carry', 'collision_filters'))
        idx = phases.index(p)
        assert phases[idx + 1]['name'] == p['plumbers_support_cycle'] + '_ready'
        assert phases[idx + 1]['advance']['type'] == 'plumbers_support_ready'
        assert phases[idx + 2]['name'].startswith('assemble_')
    lifts = [p for p in phases if p['name'].startswith('support_') and p['name'].endswith('_lift')]
    assert len(lifts) == 4
    assert all(p['local_skill']['free_space_stage'] == 'lift'
               and not p['local_skill'].get('requires_held_object') for p in lifts)


def test_support_coarse_to_fine_stages_do_not_change_contact_or_insertion_geometry(recipe):
    phases = recipe['phases']
    for i, p in enumerate(phases):
        if not p['name'].endswith('_coarse_lift'):
            continue
        prefix = p['plumbers_support_cycle']
        expected = ['coarse_lift', 'coarse_orient', 'coarse_transfer', 'move_above', 'preshape',
                    'descend', 'close_and_attach', 'ready']
        assert [x['name'] for x in phases[i:i+8]] == [prefix + '_' + s for s in expected]
        skills = [phases[i+j]['local_skill'] for j in (0, 1, 2, 3, 5)]
        assert [s['free_space_stage'] for s in skills] == ['lift', 'orient', 'transfer', 'lower', 'approach']
        assert all(s['free_space_trajectory'] and s['robot'] == 'franka_right' for s in skills)
        assert all(s['free_space_joint_acceleration'] > 0 for s in skills)
        assert skills[-1]['free_space_linear_speed'] < skills[0]['free_space_linear_speed']
        assert skills[-1]['free_space_position_lookahead'] == .003


@pytest.mark.parametrize('value', [0., .24, .36, float('nan'), float('inf')])
def test_invalid_coarse_clearance_fails(value):
    with pytest.raises(ValueError, match='coarse_tcp_height'):
        support_grasp_settings({'physical_base_support': {'enabled': True,
            'coarse_tcp_height_above_base_origin_m': value}})


def test_support_spans_left_pick_insert_release_retreat_and_park_then_really_unloads(recipe):
    phases = recipe['phases']
    supported = [p for p in phases if p['name'].startswith('assemble_') and p.get('plumbers_base_support')]
    assert supported
    for p in supported:
        support = p['plumbers_base_support']
        hold = p['local_skills']['franka_right']
        assert hold['hold_previous_grasp_command'] and hold['regulate_physical_carry']
        assert hold['requires_held_object'] and hold['require_success']
        assert p['gripper_commands']['franka_right'] == 'contact_hold'
        assert p['advance']['conditions'][-1]['type'] == 'plumbers_support_ready'
        if p['name'].endswith('_release'):
            assert p['local_skill']['name'] == 'ur5e_release_supported_part'
            left_gate = p['advance']['conditions'][0]
            assert 'fabrica_plumbers_block_2' not in left_gate['conditions'][0]['objects']
        if p['name'].endswith('_park'):
            i = phases.index(p)
            assert [x['name'] for x in phases[i + 1:i + 4]] == [support['cycle'] + s for s in
                                                                     ('_release', '_retreat', '_park')]
            assert phases[i + 4]['name'] == p['name'].removesuffix('_park') + '_unloaded_settle'
    for p in phases:
        if p['name'].endswith('_unloaded_settle'):
            assert not p.get('plumbers_base_support') and not p.get('local_skills')
            assert p['advance']['type'] == 'physical_settle'
    assert phases[-1]['physical_unloaded_verification']


def test_support_off_preserves_pickup_geometry_objects_targets_and_success(recipe):
    source = copy.deepcopy(recipe)
    source['fabrica_canonical']['physical_base_support']['enabled'] = False
    # Recompiling an already-resolved recipe deliberately preserves explicit
    # randomization overrides. Remove its generated reach list for this test.
    source['domain_randomization']['groups']['assembly_base'].pop('translation_constraints')
    legacy = compile_fabrica_canonical_recipe(source)
    assert legacy['fabrica_canonical_resolved']['physical_base_support'] == {'enabled': False}
    assert not any(p.get('plumbers_base_support') or p['name'].startswith('support_') for p in legacy['phases'])
    assert len([p for p in legacy['phases'] if p.get('attach')]) == 5
    for key in ('objects', 'targets', 'success'):
        assert legacy[key] == recipe[key]
    assert (legacy['fabrica_canonical_resolved']['selected_base_grasp'] ==
            recipe['fabrica_canonical_resolved']['selected_base_grasp'])
    assert legacy['phases'][-1] == recipe['phases'][-1]
    enabled_constraints = recipe['domain_randomization']['groups']['assembly_base']['translation_constraints']
    disabled_constraints = legacy['domain_randomization']['groups']['assembly_base']['translation_constraints']
    enabled_points = [c for c in enabled_constraints if c['type'] == 'points_within_distance']
    disabled_points = [c for c in disabled_constraints if c['type'] == 'points_within_distance']
    assert sum(len(c['points']) for c in enabled_points) == sum(len(c['points']) for c in disabled_points) + 2


def test_support_pose_and_approach_are_reachable_in_compiler_ur5e_model(recipe):
    from roboassemblybench.core.fabrica_canonical import _ik_chain, _initial_ik_configuration, _solve_ik_pose
    from internutopia_extension.configs.robots.ur5e import arm_ik_cfg
    from toolkits.factory_dual_franka_assembly.planner_primitives import quat_multiply
    g = recipe['fabrica_canonical_resolved']['physical_base_support']
    robot = next(r for r in recipe['robots'] if r['name'] == 'franka_right')
    target = next(t for t in recipe['targets'] if t['name'] == 'part_2_assembled')
    offset = np.array(recipe['workspace_offset'])
    robot_position = np.array(robot['position']) + (offset if robot.get('apply_workspace_offset', True) else 0)
    position = np.array(target['position']) + offset + quat_rotate(target['orientation'], g['tcp_in_object_position'])
    orientation = quat_multiply(target['orientation'], g['tcp_in_object_orientation'])
    retreat = quat_rotate(orientation, [0., 0., -g['approach_distance_m']])
    chain = _ik_chain(arm_ik_cfg.robot_urdf_path, 'ur5e')
    seed = _initial_ik_configuration(chain, robot)
    for amount in np.linspace(1., 0., 11):
        seed, distance, angle = _solve_ik_pose(chain=chain, position=position + amount * retreat,
            orientation=orientation, robot_position=robot_position, robot_orientation=robot['orientation'],
            warm_start=seed, fallback_start=seed, max_iterations=500)
        assert distance < 1e-5 and angle < 1e-4


@pytest.mark.parametrize('raw', [None, True, {}, {'enabled': 1}, {'enabled': 'true'},
    {'enabled': True, 'grasp_center_m': [0., 0., .01]},
    {'enabled': True, 'grasp_center_m': [.07, 0., float('nan')]},
    {'enabled': True, 'grasp_center_m': [.07, 0.]},
    {'enabled': True, 'approach_tilt_degrees': 0},
    {'enabled': True, 'pad_center_m': .20},
    {'enabled': True, 'approach_distance_m': -1}])
def test_bad_support_configuration_fails_instead_of_falling_back(raw):
    with pytest.raises((ValueError, TypeError), match='physical_base_support'):
        support_grasp_settings({'physical_base_support': raw})


SUPPORT = {'robot': 'right', 'inserting_robot': 'left', 'object': 'base', 'cycle': 'support_00',
           'max_position_drift': .003, 'max_orientation_drift': .0872664626,
           'contact_box_offset': [.065, .0012658, .010], 'contact_box_scale': [.03, .04, .02]}


@pytest.fixture
def state():
    task = NS(step_counter=10, phase='insert', position=np.zeros(3), orientation=np.array([1., 0, 0, 0]))
    task.attachment = {'robot_name': 'right', 'mode': 'pure_physical_grasp',
                       'hold_grasp_reference': {'point_in_object': [.065, .0012658, .010]},
                       'attach_spec': {'hold_force_control': {'enabled': True}}}
    task._attachment_for = lambda obj, robot: task.attachment
    task._physical_hold_valid = lambda *args, **kwargs: True
    task._resolve_object = lambda _: NS(get_pose=lambda: (task.position, task.orientation))
    task.control = {'motion_ready': True, 'failure': None, 'mode': 'hold'}
    task.force_state = {'step': 10, 'last_control': task.control}
    task._ur5e_contact_hold_commands = {'right': {'object': 'base', 'loaded_force_state': task.force_state}}
    return task


def test_force_recovery_or_stale_update_cannot_allow_left_motion(state):
    assert base_support_status(state, SUPPORT)['ready']
    state.control['motion_ready'] = False
    status = base_support_status(state, SUPPORT)
    assert not status['ready'] and not status['failure']
    state.control['motion_ready'] = True
    state.step_counter += 2
    assert base_support_status(state, SUPPORT)['reason'] == 'support_force_not_updated'


@pytest.mark.parametrize('bad', ['missing', 'wrong_robot', 'weld', 'invalid_contact', 'command_missing',
                               'force_failure', 'nan_pose', 'zero_quaternion', 'stale_cycle',
                               'high_grasp', 'reference_missing'])
def test_invalid_support_is_fatal_not_a_geometric_or_idle_fallback(state, bad):
    base_support_status(state, SUPPORT)
    if bad == 'missing':
        state.attachment = None
    elif bad == 'wrong_robot':
        state.attachment['robot_name'] = 'left'
    elif bad == 'weld':
        state.attachment['mode'] = 'fixed_joint'
    elif bad == 'invalid_contact':
        state._physical_hold_valid = lambda *args, **kwargs: False
    elif bad == 'command_missing':
        state._ur5e_contact_hold_commands.clear()
    elif bad == 'force_failure':
        state.control['failure'] = 'loaded_gripper_force_limit'
    elif bad == 'nan_pose':
        state.position[0] = np.nan
    elif bad == 'zero_quaternion':
        state.orientation[:] = 0
    elif bad == 'high_grasp':
        state.attachment['hold_grasp_reference']['point_in_object'][2] = .045
    elif bad == 'reference_missing':
        state.attachment.pop('hold_grasp_reference')
    else:
        state.attachment['plumbers_support_world_anchor']['cycle'] = 'old'
    result = base_support_status(state, SUPPORT)
    assert not result['ready'] and result['failure']


def test_world_anchor_is_fixed_across_stages_and_does_not_absorb_base_drift(state):
    assert base_support_status(state, SUPPORT)['ready']
    anchor = copy.deepcopy(state.attachment['plumbers_support_world_anchor'])
    state.phase = 'release'
    state.position[0] = .002
    assert base_support_status(state, SUPPORT)['ready']
    state.position[0] = .00301
    result = base_support_status(state, SUPPORT)
    assert not result['ready'] and result['failure'] == 'support_base_moved'
    assert state.attachment['plumbers_support_world_anchor'] == anchor
    state.position[:] = 0
    state.orientation[:] = [np.cos(np.radians(5.01) / 2), 0, 0, np.sin(np.radians(5.01) / 2)]
    assert base_support_status(state, SUPPORT)['failure'] == 'support_base_moved'


def act(adapter, task, spec):
    return adapter.act(task=task, robot_name='left', phase_spec={'plumbers_base_support': SUPPORT},
        skill_spec=spec, tracked_robots={}, tracked_objects={})


def test_release_opening_is_blocked_until_right_support_recovers_without_completing_skill(state):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    state.phase_index = state.phase_entry_step = 0
    state.control['motion_ready'] = False
    adapter._hold_joint_action = lambda **kw: {'arm_joint_controller': [[.1] * 6]}
    adapter._current_gripper_q = lambda **kw: .4
    adapter._gripper_open_closed_q = lambda **kw: (0., .8)
    spec = {'name': 'ur5e_release_supported_part', 'object': 'left_part'}
    original_lookup = state._attachment_for
    state._attachment_for = lambda obj, robot: original_lookup(obj, robot) if robot == 'right' else None
    a = act(adapter, state, spec)
    assert a['gripper_controller'] == [.5]
    assert not adapter._completion_state
    state.control['motion_ready'] = True
    a = act(adapter, state, spec)
    assert a['gripper_controller'] == [1.]
    assert not adapter._completion_state


def test_loaded_left_pause_keeps_last_grasp_command_and_never_calls_ik(state):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    state.phase_index = state.phase_entry_step = 0
    state.control['motion_ready'] = False
    original_lookup = state._attachment_for
    state._attachment_for = lambda obj, robot: original_lookup(obj, robot) if robot == 'right' else {'robot_name': 'left'}
    expected = {'arm_joint_controller': [[.12] * 6], 'gripper_controller': [.4]}
    adapter._hold_previous_grasp_command = lambda **kw: expected
    adapter._solve_ik = lambda **kw: pytest.fail('No IK while support is recovering')
    assert act(adapter, state, {'name': 'ur5e_move_part_to_staging', 'object': 'left_part'}) is expected
    assert not adapter._completion_state


def test_real_task_advance_gate_cannot_skip_force_recovery(state):
    path = Path(__file__).resolve().parents[2] / 'internutopia_extension/tasks/factory_dual_franka_assembly_task.py'
    node = next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == '_evaluate_advance_condition')
    scope = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), scope)
    state.phase_step_counter = 10
    state._evaluate_advance_condition = lambda **kw: scope['_evaluate_advance_condition'](state, **kw)
    phase = {'plumbers_base_support': SUPPORT}
    advance = {'type': 'all_of', 'conditions': [{'type': 'timer'}, {'type': 'plumbers_support_ready'}]}
    assert state._evaluate_advance_condition(phase_spec=phase, advance=advance)
    state.control['motion_ready'] = False
    assert not state._evaluate_advance_condition(phase_spec=phase, advance=advance)
    assert not state._evaluate_advance_condition(phase_spec={}, advance={'type': 'plumbers_support_ready'})


def test_trace_includes_early_support_loss_and_deduplicates_same_tick(state, tmp_path, monkeypatch):
    monkeypatch.setenv('BEAM_HOLD_TRACE_PATH', str(tmp_path / 'hold.jsonl'))
    base_support_status(state, SUPPORT)
    base_support_status(state, SUPPORT)
    state.attachment = None
    base_support_status(state, SUPPORT)
    records = [json.loads(s) for s in (tmp_path / 'plumbers_support_trace.jsonl').read_text().splitlines()]
    assert len(records) == 2 and records[0]['ready']
    assert records[1]['failure'] == 'support_grasp_missing' and not records[1]['ready']


def test_legacy_force_comparison_still_requires_both_target_finger_forces(state):
    state.attachment['attach_spec']['hold_force_control']['enabled'] = False
    fingers = {s: {'force': 8., 'force_probe_valid': True} for s in ('left_finger', 'right_finger')}
    state.attachment['current_contact_metrics'] = fingers
    assert base_support_status(state, SUPPORT)['ready']
    fingers['left_finger']['force'] = 1.
    assert not base_support_status(state, SUPPORT)['ready']
    fingers['left_finger']['force'] = 8.
    fingers['right_finger']['force_probe_valid'] = False
    assert not base_support_status(state, SUPPORT)['ready']
