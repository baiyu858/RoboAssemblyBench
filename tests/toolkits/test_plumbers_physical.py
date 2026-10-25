"""Physical plumbers contract and actual runtime gates, without Isaac imports."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from roboassemblybench.core.fabrica_canonical import compile_fabrica_canonical_recipe
from roboassemblybench.core.plumbers_physical import align_plumbers_base_grasps_to_pads, apply_plumbers_physics
from toolkits.factory_dual_franka_assembly.task_specs import load_task_recipe
from toolkits.factory_dual_franka_assembly import planner_primitives


ROOT = Path(__file__).resolve().parents[2]


def test_native_pad_alignment_keeps_forward_inverse_and_other_parts_consistent():
    import json
    metadata = json.loads((ROOT / 'roboassemblybench/assets/Fabrica/canonical_7_bundles/canonical_tasks.json').read_text())
    task = copy.deepcopy(metadata['tasks']['plumbers_block'])
    original = copy.deepcopy(task)
    align_plumbers_base_grasps_to_pads(task, {'physical_base_pad_center_m': .143})
    old = next(g for g in original['base_grasp_candidates'] if g['grasp_id'] == 944)
    new = next(g for g in task['base_grasp_candidates'] if g['grasp_id'] == 944)
    q = new['tcp_in_assembly_orientation']
    p = np.array(new['tcp_in_assembly_position'])
    pinch = planner_primitives.quat_rotate(planner_primitives.quat_conjugate(q), np.array(new['grasp_center_m']) - p)
    assert pinch[2] == pytest.approx(.143)
    assert new['physical_pad_alignment']['tcp_shift_along_approach_m'] == pytest.approx(.025371394704243433)
    assert np.allclose(planner_primitives.quat_rotate(q, new['object_in_tcp_position']) + p, 0)
    assert new['tcp_in_assembly_orientation'] == old['tcp_in_assembly_orientation']
    assert new['grasp_center_m'] == old['grasp_center_m']
    assert task['parts'] == original['parts'] and task['assembly_steps'] == original['assembly_steps']
    unchanged = copy.deepcopy(original)
    align_plumbers_base_grasps_to_pads(unchanged, {})
    assert unchanged == original


@pytest.mark.parametrize('value', [0., -.1, .18, float('nan'), float('inf')])
def test_invalid_native_pad_alignment_fails(value):
    with pytest.raises(ValueError, match='physical_base_pad_center_m'):
        align_plumbers_base_grasps_to_pads({}, {'physical_base_pad_center_m': value})


def test_base_center_offset_moves_only_selected_grasp_and_both_transforms():
    import json
    task = json.loads((ROOT / 'roboassemblybench/assets/Fabrica/canonical_7_bundles/canonical_tasks.json').read_text())['tasks']['plumbers_block']
    before = copy.deepcopy(task)
    align_plumbers_base_grasps_to_pads(task, {
        'physical_base_pad_center_m': .143, 'base_grasp_id': 944,
        'physical_base_grasp_center_offset_m': [.012, 0., 0.],
    })
    for old, new in zip(before['base_grasp_candidates'], task['base_grasp_candidates']):
        if old['grasp_id'] != 944:
            assert old == new
            continue
        assert np.allclose(np.array(new['grasp_center_m']) - old['grasp_center_m'], [.012, 0., 0.])
        assert -.04 < new['grasp_center_m'][0] < -.035
        assert new['interior_clearance_score'] > old['interior_clearance_score']
        assert np.allclose(planner_primitives.quat_rotate(new['tcp_in_assembly_orientation'],
                          new['object_in_tcp_position']) + new['tcp_in_assembly_position'], 0)


@pytest.mark.parametrize('offset', [[.021, 0, 0], [float('nan'), 0, 0], [0, 0]])
def test_invalid_base_center_offset_fails(offset):
    with pytest.raises(ValueError, match='physical_base_grasp_center_offset_m'):
        align_plumbers_base_grasps_to_pads({}, {
            'physical_base_pad_center_m': .143, 'physical_base_grasp_center_offset_m': offset})


def test_aligned_base_pose_is_used_for_close_attach_and_transport(recipe):
    grasp = recipe['fabrica_canonical_resolved']['selected_base_grasp']
    assert grasp['grasp_id'] == 944 and grasp['physical_pad_alignment']['pad_center_m'] == .143
    for phase in recipe['phases']:
        if phase.get('plumbers_support_cycle'):
            continue  # Flange support has an independent, lower grasp pose.
        skill = phase.get('local_skill', {})
        if skill.get('object') == 'fabrica_plumbers_block_2' and 'grasp_relative_position' in skill:
            assert skill['grasp_relative_position'] == grasp['object_in_tcp_position']
        for attach in phase.get('attach', []):
            if attach['object'] == 'fabrica_plumbers_block_2':
                assert attach['target']['offset'] == grasp['tcp_in_assembly_position']


def runtime_type(names):
    source = ast.parse((ROOT / 'internutopia_extension/tasks/factory_dual_franka_assembly_task.py').read_text())
    methods = [n for n in ast.walk(source) if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(methods) == len(names)
    scope = {**vars(planner_primitives), 'np': np, 'copy': copy}
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), '<runtime gates>', 'exec'), scope)
    return type('Runtime', (), {name: scope[name] for name in names})


@pytest.fixture(scope='module')
def recipe():
    return load_task_recipe('fabrica_plumbers_block_ur5e_staged')


def test_all_pickup_and_support_grasps_require_real_opposing_pair_forces(recipe):
    assert recipe['fabrica_canonical_resolved']['continuous_physics']
    closes = [p for p in recipe['phases'] if p.get('attach')]
    assert len(closes) == 9  # Five pickups and one base support per insertion.
    boxes = []
    for phase in closes:
        attach, skill = phase['attach'][0], phase['local_skill']
        assert attach['attachment_mode'] == 'pure_physical_grasp'
        assert not attach['disable_collision_on_attach']
        assert not attach['filter_gripper_collisions_on_attach']
        for key in ('continuous_physics', 'require_force_contact', 'require_dual_force_contact',
                    'require_pair_force_contact', 'require_opposing_force_contact'):
            assert attach[key] is True and skill[key] is True
        assert skill['close_force_regulation'] and attach['hold_force_regulation']
        assert skill['close_force_seek_until_contact']
        assert skill['close_force_require_clamp_band']
        assert skill['close_force_min_stable'] == skill['close_force_low'] == 4.0
        control = attach['hold_force_control']
        assert control['enabled']
        assert [control[key] for key in ('low', 'target', 'soft_limit', 'unload', 'hard_limit')] == [6, 8, 14, 18, 40]
        assert control['soft_limit_action'] == 'limit_and_slow'
        assert control['soft_limit_motion_scale'] == .5
        assert control['max_command_lead'] == .001 and control['max_closing_travel'] == .03
        assert not skill['allow_jaw_width_completion']
        assert not attach['hold_grace_allow_missing_contact']
        assert attach['hold_position_slip_reference'] == skill['hold_position_slip_reference'] == 'grasp_point'
        assert attach['hold_position_slip_tolerance'] == 0.003
        assert attach['hold_orientation_slip_tolerance'] == 0.0872664626
        assert skill['contact_box_scale'] == attach['contact_box_scale']
        assert skill['contact_box_offset'] == attach['contact_box_offset']
        assert skill['jaw_width_thickness_m'] == attach['jaw_width_thickness_m']
        assert skill['jaw_width_thickness_m'] > 0
        boxes.append(tuple(attach['contact_box_scale']))
    assert len(set(boxes)) > 1  # Not a copied Beam contact box.


def test_idle_anchor_is_left_only_and_skips_initialization_and_active_work(recipe):
    assert recipe['fabrica_canonical_resolved']['physical_idle_hold_anchor'] is True
    assert recipe['phases'][0]['physical_idle_hold_anchor'] is False
    enabled = []
    for phase in recipe['phases']:
        active = ((phase.get('local_skill') or {}).get('robot') == 'franka_left'
                  or 'franka_left' in (phase.get('local_skills') or {})
                  or (phase.get('robot_targets') or {}).get('franka_left') is not None)
        if active:
            assert phase['physical_idle_hold_anchor'] is False
        if phase['physical_idle_hold_anchor']:
            assert phase['physical_idle_hold_robot'] == 'franka_left'
            enabled.append(phase['name'])
        else:
            assert phase['physical_idle_hold_robot'] is None
    assert 'base_2_move_above' in enabled and 'base_2_lift' in enabled
    assert any(name.startswith('support_') for name in enabled)


def test_scoped_idle_switch_changes_no_motion_or_contact_controls(recipe):
    source = copy.deepcopy(recipe)
    source['fabrica_canonical']['physical_idle_hold_anchor'] = False
    without_hold = compile_fabrica_canonical_recipe(source)
    assert without_hold['fabrica_canonical_resolved']['physical_idle_hold_anchor'] is False
    old_phases, new_phases = copy.deepcopy(recipe['phases']), copy.deepcopy(without_hold['phases'])
    for original, disabled in zip(old_phases, new_phases):
        assert disabled.pop('physical_idle_hold_anchor') is False
        assert disabled.pop('physical_idle_hold_robot') is None
        original.pop('physical_idle_hold_anchor')
        original.pop('physical_idle_hold_robot')
    assert old_phases == new_phases


@pytest.mark.parametrize('value', [None, 'true', 1])
def test_idle_anchor_requires_explicit_boolean(value):
    with pytest.raises(ValueError, match='physical_idle_hold_anchor'):
        apply_plumbers_physics([], success=[], spec={'physical_idle_hold_anchor': value})


def test_no_part_pinning_or_collision_disabling_anywhere(recipe):
    for phase in recipe['phases']:
        assert not any(phase.get(key) for key in ('lock', 'fixture_lock', 'kinematic_carry',
                                                'gripper_sibling_filters', 'collision_filters'))
        for entry in phase.get('object_collisions', []) + phase.get('object_gravity', []):
            assert entry['enabled']
        skill = phase.get('local_skill', {})
        if skill.get('requires_held_object'):
            assert skill['continuous_physics'] and skill['track_published_grasp_command']
            assert skill['regulate_physical_carry']
            assert skill['gripper_command'] == 'contact_hold'
            assert not any(key.startswith('relax_fixed_attachment_') for key in skill)
            assert 'insertion_capture_transition_phase' not in skill
            if skill.get('hold_previous_grasp_command'):
                continue  # Static support never computes a rotation command.
            assert skill['physical_rotation_governor']
            assert skill['physical_rotation_speed'] == .1
            assert skill['physical_rotation_acceleration'] == .3
            assert skill['physical_rotation_lookahead'] == .01
            assert not skill['cartesian_orientation_command_warm_start']
            assert skill['cartesian_orientation_command_lookahead'] == .01
            assert not skill['orientation_first_before_translation']
            assert not skill['lock_transport_ik_target']
            assert not skill['ik_retry_with_measured_state']
            assert phase['timeout_steps'] >= 8800
    assert not recipe['fabrica_canonical']['stabilize_fixture_parts']


@pytest.mark.parametrize('value', [0., -1., float('nan'), float('inf')])
def test_invalid_rotation_limit_fails_compilation(recipe, value):
    source = copy.deepcopy(recipe)
    source['fabrica_canonical']['physical_rotation_speed'] = value
    with pytest.raises(ValueError, match='physical_rotation'):
        compile_fabrica_canonical_recipe(source)


def test_hold_force_can_be_disabled_for_controlled_legacy_comparison(recipe):
    source = copy.deepcopy(recipe)
    source['fabrica_canonical']['hold_force_control'] = {'enabled': False}
    result = compile_fabrica_canonical_recipe(source)
    assert all('hold_force_control' not in p['attach'][0] for p in result['phases'] if p.get('attach'))


def test_invalid_hold_force_setting_is_not_silently_ignored(recipe):
    source = copy.deepcopy(recipe)
    source['fabrica_canonical']['hold_force_control']['target'] = 15.
    with pytest.raises(ValueError, match='hold_force_control'):
        compile_fabrica_canonical_recipe(source)


def test_close_force_smoothing_is_explicit_scoped_and_can_be_disabled(recipe):
    closes = [p['local_skill'] for p in recipe['phases']
              if p.get('local_skill', {}).get('name') == 'ur5e_close_gripper']
    assert closes and all(s['close_force_stability'] == {
        'enabled': True, 'window_steps': 5, 'hysteresis_margin': .5} for s in closes)
    assert all(s['close_contact_stable_steps'] == s['close_force_validation_steps'] == 96 for s in closes)
    source = copy.deepcopy(recipe)
    source['fabrica_canonical']['close_force_stability'] = {'enabled': False}
    disabled = compile_fabrica_canonical_recipe(source)
    assert all('close_force_stability' not in p.get('local_skill', {}) for p in disabled['phases'])
    assert all('close_force_stability' not in a for p in recipe['phases'] for a in p.get('attach', []))


@pytest.mark.parametrize('config', [{'enabled': 'true'}, {'enabled': True, 'window_steps': 0},
    {'enabled': True, 'hysteresis_margin': 3.9}])
def test_bad_close_force_stability_settings_fail_compilation(config):
    with pytest.raises(ValueError, match='close_force_stability'):
        apply_plumbers_physics([], success=[], spec={'close_force_stability': config})


def test_release_retreat_park_settle_then_read_only_rebase(recipe):
    phases = recipe['phases']
    for index, phase in enumerate(phases):
        if phase['name'].endswith('_release'):
            prefix = phase['name'].removesuffix('_release')
            if prefix.startswith('support_'):
                assert [p['name'] for p in phases[index + 1:index + 3]] == [
                    f'{prefix}_retreat', f'{prefix}_park']
                assert phases[index + 3]['name'].startswith('assemble_')
                assert phases[index + 3]['name'].endswith('_unloaded_settle')
            elif phase.get('plumbers_base_support'):
                support = phase['plumbers_base_support']['cycle']
                assert [p['name'] for p in phases[index + 1:index + 7]] == [
                    f'{prefix}_retreat', f'{prefix}_park', f'{support}_release',
                    f'{support}_retreat', f'{support}_park', f'{prefix}_unloaded_settle']
            else:
                assert [p['name'] for p in phases[index + 1:index + 4]] == [
                    f'{prefix}_retreat', f'{prefix}_park', f'{prefix}_unloaded_settle']
            gate = phase['advance']['conditions'][0] if phase.get('plumbers_base_support') else phase['advance']
            assert gate['conditions'][1]['type'] == 'gripper_open'
    rebases = [p for p in phases if p.get('rebase_from_object')]
    assert len(rebases) == 1
    assert rebases[0]['rebase_from_object']['require_released']
    assert 'enable_collision' not in rebases[0]['rebase_from_object']
    assert phases[-1]['physical_unloaded_verification']
    assert len(phases[-1]['advance']['objects']) == 5
    assert phases[0]['advance']['type'] == 'physical_settle'


def test_physical_gripper_recipe_is_explicit_and_preserves_native_effort(recipe):
    for robot in recipe['robots']:
        assert robot['gripper_drive_kp'] == 7500
        assert robot['gripper_drive_kd'] == 173
        assert robot['gripper_drive_max_effort'] is None
        assert robot['gripper_require_drive_readback']
        assert robot['gripper_pad_static_friction'] == 1.0
        assert robot['gripper_pad_dynamic_friction'] == 0.8


def test_legacy_mode_remains_available_and_other_assemblies_are_not_enabled(recipe):
    source = copy.deepcopy(recipe)
    source['fabrica_canonical']['continuous_physics'] = False
    source['fabrica_canonical']['stabilize_fixture_parts'] = True
    legacy = compile_fabrica_canonical_recipe(source)
    assert all(p['attach'][0]['attachment_mode'] == 'fixed_joint' for p in legacy['phases'] if p.get('attach'))
    source['fabrica_canonical'].update(assembly='car', continuous_physics=True)
    with pytest.raises(ValueError, match='continuous_physics'):
        compile_fabrica_canonical_recipe(source)


@pytest.fixture
def runtime():
    cls = runtime_type(['_gripper_normalized_openness', '_pure_physical_release_ready',
                        '_gripper_opening_limit', '_detach_object', '_clear_attachment_state',
                        '_physical_settle_ready', '_evaluate_advance_condition', '_check_success',
                        '_rebase_targets_from_object'])
    t = cls()
    t.q = 0.0
    t.robots = {'right': NS(config=NS(gripper_open_position=0.0, gripper_closed_position=0.8))}
    t._get_robot_gripper_opening = lambda _: t.q
    t.get_current_phase_spec = lambda: {}
    t._current_gripper_command = lambda *_: 'open'
    t.phase_step_counter = 10
    t.phase_index, t.phase_entry_step, t.step_counter = 0, 0, 0
    t.phase = 'settle'
    t._attachments, t._locked_targets, t._object_collision_enabled = {}, {}, {}
    t._as_list = lambda x: x if isinstance(x, list) else [] if x is None else [x]
    t._extract_object_name = lambda x: x if isinstance(x, str) else x.get('object')
    t._object_targets_reached = lambda *a, **kw: True
    t.position = np.zeros(3)
    t._resolve_object = lambda _: NS(get_pose=lambda: (t.position, np.array([1., 0., 0., 0.])))
    t.velocity = {'valid': True, 'linear_speed': 0., 'angular_speed': 0., 'is_static': True}
    t._object_velocity_metrics = lambda _: t.velocity
    t.forces = {s: {'force_probe_valid': True, 'force': 0.} for s in ('left_finger', 'right_finger')}
    t._gripper_contact_metrics = lambda *a, **kw: t.forces
    return t


def test_robotiq_release_needs_measured_open_and_both_valid_clear_contacts(runtime):
    t = runtime
    state = {'robot_name': 'right', 'attach_spec': {'release_open_openness': .95}}
    assert t._pure_physical_release_ready('part', state)
    t.q = .8
    assert not t._pure_physical_release_ready('part', state)
    t.q = 0.
    for side in t.forces:
        t.forces[side]['force'] = 1.
        assert not t._pure_physical_release_ready('part', state)
        t.forces[side].update(force=0., force_probe_valid=False)
        assert not t._pure_physical_release_ready('part', state)
        t.forces[side]['force_probe_valid'] = True
    t.q = float('nan')
    assert not t._pure_physical_release_ready('part', state)


def test_robotiq_closing_direction_is_not_rejected_as_franka_opening(runtime):
    limit = runtime._gripper_opening_limit('part', {
        'robot': 'right', 'continuous_physics': True, 'release_open_openness': .95})
    assert .8 <= limit < .82


def test_physical_release_is_idempotent_and_preserves_pose_velocity_collision(runtime):
    t = runtime
    t._attachments['part'] = {'mode': 'pure_physical_grasp', 'robot_name': 'right',
                              'attach_spec': {'continuous_physics': True}}
    t._attachment_for = lambda name, robot: t._attachments.get(name)
    t._remove_attachment_joint = lambda _: None
    def forbidden(*a, **kw):
        pytest.fail('Physical detach must not write body state')
    t._zero_object_velocity = t._set_object_collision = t._set_object_pose = forbidden
    t._detach_object('part')
    t._detach_object('part')
    assert not t._attachments


SETTLE = {'type': 'physical_settle', 'objects': ['part'], 'stable_steps': 3}


def test_settle_requires_consecutive_unique_low_velocity_samples(runtime):
    t = runtime
    assert not t._physical_settle_ready(SETTLE)
    assert not t._physical_settle_ready(SETTLE)  # Same tick cannot count twice.
    t.step_counter = 1
    assert not t._physical_settle_ready(SETTLE)
    t.step_counter = 2
    assert t._physical_settle_ready(SETTLE)
    t.step_counter = 3
    t.velocity['linear_speed'] = .1  # Even when legacy is_static=True.
    assert not t._physical_settle_ready(SETTLE)
    assert t._physical_settle_diagnostics['reason'] == 'part:velocity'


@pytest.mark.parametrize('bad_state', ['attached', 'locked', 'collision_off', 'probe_invalid', 'nan', 'off_target'])
def test_settle_rejects_invalid_states(runtime, bad_state):
    t = runtime
    spec = {**SETTLE, 'objects': [{'object': 'part', 'target': 'goal'}]}
    if bad_state == 'attached':
        t._attachments['part'] = {}
    elif bad_state == 'locked':
        t._locked_targets['part'] = 'goal'
    elif bad_state == 'collision_off':
        t._object_collision_enabled['part'] = False
    elif bad_state == 'probe_invalid':
        t.velocity['valid'] = False
    elif bad_state == 'nan':
        t.velocity['angular_speed'] = np.nan
    else:
        t._object_targets_reached = lambda *a, **kw: False
    for step in range(4):
        t.step_counter = step
        assert not t._physical_settle_ready(spec)


def test_settle_checks_window_drift_and_restarts_after_phase_change(runtime):
    t = runtime
    for step in range(3):
        t.step_counter = step
        t.position[0] = step * .0011
        assert not t._physical_settle_ready(SETTLE)
    assert t._physical_settle_diagnostics['reason'] == 'part:pose_drift'
    for step in range(3, 6):
        t.step_counter = step
        result = t._physical_settle_ready(SETTLE)
    assert result
    t.phase_index += 1
    assert not t._physical_settle_ready(SETTLE)


def test_success_cannot_bypass_final_unloaded_window(runtime):
    t = runtime
    final = {'name': 'final_settle', 'physical_unloaded_verification': True, 'advance': SETTLE}
    t.phase_specs, t.policy_evaluation_mode, t.failed = [{}, final], False, False
    t.cfg = NS(success_criteria=[])
    assert not t._check_success()
    t.phase_index = 1
    t.phase = 'final_settle'
    assert not t._check_success()
    t.step_counter = 1
    assert not t._check_success()
    t.step_counter = 2
    assert t._check_success()
    # Terminal transition advances phase_index beyond the last phase. Metrics
    # still report the verified window, without recounting it under a new key.
    t.phase_index, t.phase, t.success = 2, 'complete', True
    assert t._check_success()
    t.failed = True
    assert not t._check_success()


def test_rebase_changes_only_targets_after_release(runtime):
    t = runtime
    t._rebased_anchors = set()
    t.target_poses = {n: {'position': np.array([x, 0., 0.]), 'orientation': np.array([1., 0., 0., 0.])}
                      for n, x in (('base', 0.), ('part', 1.))}
    t._maybe_write_attach_debug = lambda _: None
    t.position[0] = .005
    spec = {'object': 'part', 'anchor_target': 'base', 'targets': ['part'], 'require_released': True}
    t._attachments['part'] = {}
    t._rebase_targets_from_object(spec)
    assert not t._rebased_anchors
    t._attachments.clear()
    t._rebase_targets_from_object(spec)
    np.testing.assert_allclose(t.target_poses['part']['position'], [1.005, 0., 0.])
    assert t.position[0] == .005 and not t._locked_targets
    t.position[0] = .01
    t._rebase_targets_from_object(spec)
    np.testing.assert_allclose(t.target_poses['part']['position'], [1.005, 0., 0.])


def test_waiting_transport_keeps_the_published_grasp_and_force_controller():
    from toolkits.factory_dual_franka_assembly.plumbers_block_ur5e_skills import UR5eAssemblyAtomicSkillAdapter
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    calls = []
    adapter._hold_previous_grasp_command = lambda **kw: calls.append(kw) or {'regulated': True}
    result = adapter._held_transport_action(task=NS(), robot_name='right',
        spec={'continuous_physics': True, 'requires_held_object': True, 'regulate_physical_carry': True})
    assert result == {'regulated': True}
    assert len(calls) == 1 and calls[0]['robot_name'] == 'right'
