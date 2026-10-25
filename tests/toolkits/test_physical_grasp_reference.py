"""Fixed material-point slip: no recapture, no relaxation of rotation/contact gates."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from internutopia_extension.tasks.factory_dual_franka_assembly_task import FactoryDualFrankaAssemblyTask
from toolkits.factory_dual_franka_assembly.physical_grasp_reference import (
    capture_grasp_reference,
    measure_grasp_point_slip,
)
from toolkits.factory_dual_franka_assembly.planner_primitives import (
    euler_xyz_to_quat,
    quat_rotate,
    relative_pose,
)

TARGET = '/World/target_part'


def contacts(left, right):
    def finger(point):
        return {
            'force_probe_valid': True, 'force_contact': True, 'force': 6.0,
            'force_observation': {
                'valid': True, 'details_valid': True, 'details_at_capacity': False,
                'source': 'filtered_contact_probe', 'other_body': TARGET,
                'contacts': [{'normal_force': 6.0, 'point_world': np.asarray(point).tolist()}],
            },
        }
    return {'contact_ready': True, 'left_finger': finger(left), 'right_finger': finger(right)}


@pytest.fixture
def runtime(monkeypatch):
    task = FactoryDualFrankaAssemblyTask.__new__(FactoryDualFrankaAssemblyTask)
    task.step_counter, task.phase_step_counter = 96, 96
    task.phase = 'close'
    task.object_position = np.array([0.4, 0.2, 1.0])
    task.object_orientation = np.array([1., 0., 0., 0.])
    task.tcp_position = np.array([0.1, 0.2, 1.2])
    task.tcp_orientation = np.array([1., 0., 0., 0.])
    task.point_local = np.array([0.08, 0., 0.02])
    center = task.object_position + task.point_local
    task.metrics = contacts(center + [0., 0.01, 0.], center - [0., 0.01, 0.])
    task.spec = {
        'attachment_mode': 'pure_physical_grasp', 'continuous_physics': True,
        'hold_position_slip_reference': 'grasp_point', 'hold_position_slip_tolerance': 0.003,
        'hold_orientation_slip_tolerance': np.deg2rad(5.0), 'require_force_contact': True,
        'force_enable_collision_on_attach': True,
    }
    task._attachments, task._locked_targets, task._frozen_lock_poses = {}, {}, {}
    task._locked_collision_states, task._lock_pin_pose = {}, {}
    task._object_collision_enabled = {'part': True}
    task.contact_ready = True
    task.attach_debug = []
    monkeypatch.setattr(task, '_resolve_object', lambda _: NS(
        get_pose=lambda: (task.object_position, task.object_orientation), unwrap=lambda: NS(prim_path=TARGET)))
    monkeypatch.setattr(task, '_get_robot_task_pose', lambda _: (task.tcp_position, task.tcp_orientation))
    monkeypatch.setattr(task, '_gripper_contact_metrics', lambda *a, **kw: task.metrics)
    monkeypatch.setattr(task, '_strict_physical_grasp_contact', lambda *a, **kw: {
        'physical_contact_ready': task.contact_ready})
    monkeypatch.setattr(task, '_maybe_write_attach_debug', task.attach_debug.append)
    monkeypatch.setattr(task, '_physical_hold_within_grace', lambda *a: False)
    monkeypatch.setattr(task, '_get_robot_gripper_opening', lambda *a: 0.47)
    monkeypatch.setattr(task, '_gripper_opening_limit', lambda *a, **kw: 0.81)
    monkeypatch.setattr(task, '_jaw_width_contact', lambda *a: False)
    monkeypatch.setattr(task, '_write_physical_contact_trace', lambda *a, **kw: None)

    def forbidden(*a, **kw):
        pytest.fail('Capturing/checking a grasp point must not write physics state.')

    for method in ('_set_object_pose', '_zero_object_velocity', '_set_object_collision',
                   '_filter_robot_gripper_from_other_parts'):
        monkeypatch.setattr(task, method, forbidden)
    return task


def register(task):
    task._attach_object('part', 'right', phase_spec={'name': 'close'}, attach_spec=task.spec)
    return task._attachments['part']


def rotate_about_grasp(task, degrees):
    center_world = np.array([0.4, 0.2, 1.0]) + task.point_local
    task.object_orientation = euler_xyz_to_quat([0., np.deg2rad(degrees), 0.])
    task.object_position = center_world - quat_rotate(task.object_orientation, task.point_local)


def test_registration_captures_actual_contact_midpoint_in_both_frames(runtime):
    state = register(runtime)
    reference = state['hold_grasp_reference']
    np.testing.assert_allclose(reference['point_in_object'], runtime.point_local, atol=1e-12)
    expected = runtime.object_position + runtime.point_local - runtime.tcp_position
    np.testing.assert_allclose(reference['point_in_tcp_at_attach'], expected, atol=1e-12)
    assert reference['captured_step'] == state['attach_step'] == 96
    assert state['mode'] == 'pure_physical_grasp' and state['joint_path'] is None


def test_rotation_about_fixed_grasp_does_not_count_remote_origin_motion_as_translation(runtime):
    state = register(runtime)
    rotate_about_grasp(runtime, 2.65)
    assert runtime._physical_hold_valid('part', state)
    result = state['last_hold_validation']
    assert result['position_slip'] < 1e-12
    assert result['object_origin_position_slip'] > 0.003
    assert result['orientation_slip'] == pytest.approx(np.deg2rad(2.65))
    assert result['position_slip_reference'] == 'grasp_point'
    assert result['max_position_slip'] == 0.003


def test_real_translation_still_fails_even_when_current_contact_points_move(runtime):
    state = register(runtime)
    saved = copy.deepcopy(state['hold_grasp_reference'])
    for index, distance in enumerate([0.001, 0.002, 0.0031]):
        runtime.step_counter += 1
        runtime.object_position = np.array([0.4 + distance, 0.2, 1.0])
        center = runtime.object_position + runtime.point_local
        runtime.metrics = contacts(center + [0., 0.01, 0.], center - [0., 0.01, 0.])
        assert runtime._physical_hold_valid('part', state) == (index < 2)
        assert state['last_hold_validation']['position_slip'] == pytest.approx(distance)
        assert state['hold_grasp_reference'] == saved
    assert state['last_hold_validation']['reasons'] == ['position_slip']


def test_rotation_and_contact_loss_gates_are_unchanged(runtime):
    state = register(runtime)
    rotate_about_grasp(runtime, 6.0)
    assert not runtime._physical_hold_valid('part', state)
    assert state['last_hold_validation']['position_slip'] < 1e-12
    assert state['last_hold_validation']['reasons'] == ['orientation_slip']
    rotate_about_grasp(runtime, 0.0)
    runtime.contact_ready = False
    assert not runtime._physical_hold_valid('part', state)
    assert state['last_hold_validation']['reasons'] == ['contact_missing']


def test_repeated_registration_unlock_and_phase_change_never_recapture(runtime):
    state = register(runtime)
    reference = state['hold_grasp_reference']
    saved = copy.deepcopy(reference)
    runtime.step_counter = 200
    runtime.object_position += [0.004, 0., 0.]
    runtime.metrics = contacts([1., 2., 3.], [4., 5., 6.])
    runtime._attach_object('part', 'right', phase_spec={'name': 'carry'}, attach_spec=runtime.spec)
    runtime._unlock_object('part')
    runtime.phase = 'carry'
    assert runtime._attachments['part'] is state
    assert state['hold_grasp_reference'] is reference and reference == saved
    assert state['attach_step'] == 96
    assert not runtime._physical_hold_valid('part', state)
    assert state['last_hold_validation']['reasons'] == ['position_slip']


@pytest.mark.parametrize('corrupt', ['missing', 'nan', 'zero_quaternion'])
def test_bad_reference_fails_closed_instead_of_recapturing(runtime, corrupt):
    state = register(runtime)
    if corrupt == 'missing':
        state.pop('hold_grasp_reference')
    elif corrupt == 'nan':
        state['hold_grasp_reference']['point_in_object'][0] = float('nan')
    else:
        state['hold_grasp_reference']['object_orientation_in_tcp_at_attach'] = [0., 0., 0., 0.]
    runtime.step_counter += 1
    runtime._attach_object('part', 'right', attach_spec=runtime.spec)
    assert not runtime._physical_hold_valid('part', state)
    assert state['last_hold_validation']['reasons'] == ['grasp_reference_invalid']


def test_transport_anchor_rewrite_does_not_reset_hold_baseline(runtime):
    state = register(runtime)
    runtime.object_position += [0.004, 0., 0.]
    state['position'], state['orientation'] = [v.tolist() for v in runtime._current_relative_pose('part', 'right')]
    assert not runtime._physical_hold_valid('part', state)
    assert state['last_hold_validation']['position_slip'] == pytest.approx(0.004)


def test_genuine_new_grasp_gets_a_new_reference_after_release(runtime):
    old = register(runtime)['hold_grasp_reference']
    runtime._attachments.pop('part')  # Symbolic release; physics is unchanged.
    runtime.step_counter = 300
    center = runtime.object_position + runtime.point_local + [0.002, 0., 0.]
    runtime.metrics = contacts(center + [0., 0.01, 0.], center - [0., 0.01, 0.])
    new = register(runtime)['hold_grasp_reference']
    assert new is not old and new['captured_step'] == 300
    np.testing.assert_allclose(new['point_in_object'], np.array(old['point_in_object']) + [0.002, 0., 0.])


@pytest.mark.parametrize('bad', ['no_points', 'invalid_probe', 'other_object', 'aggregate', 'truncated',
                                 'invalid_point', 'invalid_force', 'not_strict'])
def test_registration_requires_complete_real_target_pair_contacts(runtime, bad):
    observation = runtime.metrics['left_finger']['force_observation']
    if bad == 'no_points':
        observation['contacts'] = []
    elif bad == 'invalid_probe':
        runtime.metrics['left_finger']['force_probe_valid'] = False
    elif bad == 'other_object':
        observation['other_body'] = '/World/table'
    elif bad == 'aggregate':
        observation['source'] = 'aggregate_contact_sensor'
    elif bad == 'truncated':
        observation['details_at_capacity'] = True
    elif bad == 'invalid_point':
        observation['contacts'][0]['point_world'][0] = float('nan')
    elif bad == 'invalid_force':
        observation['contacts'][0]['normal_force'] = float('nan')
    else:
        runtime.contact_ready = False
    runtime._attach_object('part', 'right', attach_spec=runtime.spec)
    assert not runtime._attachments
    assert runtime.attach_debug[-1]['blocked_by'] == 'grasp_reference_unavailable'


def test_capture_uses_patch_centroids_but_equal_weight_between_fingers():
    metrics = contacts([0.08, 0.01, 0.], [0.08, -0.01, 0.])
    metrics['left_finger']['force_observation']['contacts'] = [
        {'normal_force': 1.0, 'point_world': [0.07, 0.01, 0.]},
        {'normal_force': 3.0, 'point_world': [0.09, 0.01, 0.]},
    ]
    reference = capture_grasp_reference(
        object_position=[0., 0., 0.], object_orientation=[1., 0., 0., 0.],
        tcp_position=[0., 0., 0.], tcp_orientation=[1., 0., 0., 0.],
        contact_metrics=metrics, target_prim_path=TARGET, step=0,
    )
    np.testing.assert_allclose(reference['point_in_object'], [0.0825, 0., 0.])


def test_nonidentity_world_frames_and_json_roundtrip_are_supported(runtime):
    runtime.object_orientation = euler_xyz_to_quat([0.2, -0.3, 0.4])
    runtime.tcp_orientation = euler_xyz_to_quat([-0.7, 0.1, 0.9])
    center = runtime.object_position + quat_rotate(runtime.object_orientation, runtime.point_local)
    runtime.metrics = contacts(center + [0., 0.01, 0.], center - [0., 0.01, 0.])
    reference = json.loads(json.dumps(register(runtime)['hold_grasp_reference']))
    pose = relative_pose(runtime.tcp_position, runtime.tcp_orientation,
                         runtime.object_position, runtime.object_orientation)
    result = measure_grasp_point_slip(reference, relative_position=pose[0], relative_orientation=pose[1])
    assert result['position_slip'] < 1e-12
    np.testing.assert_allclose(reference['point_in_object'], runtime.point_local, atol=1e-12)


def test_rigid_carry_in_world_does_not_look_like_slip(runtime):
    state = register(runtime)
    rotation = euler_xyz_to_quat([0.4, 0.5, -0.7])
    translation = np.array([0.3, -0.2, 0.4])
    runtime.object_position = quat_rotate(rotation, runtime.object_position) + translation
    runtime.tcp_position = quat_rotate(rotation, runtime.tcp_position) + translation
    runtime.object_orientation = runtime.tcp_orientation = rotation
    assert runtime._physical_hold_valid('part', state)
    assert state['last_hold_validation']['position_slip'] < 1e-12
    assert state['last_hold_validation']['orientation_slip'] < 1e-7


def test_unopted_legacy_mode_keeps_object_origin_gate(runtime):
    runtime.spec.pop('hold_position_slip_reference')
    state = register(runtime)
    assert 'hold_grasp_reference' not in state
    rotate_about_grasp(runtime, 2.65)
    assert not runtime._physical_hold_valid('part', state)
    assert state['last_hold_validation']['position_slip_reference'] == 'object_origin'
    assert state['last_hold_validation']['reasons'] == ['position_slip']


def test_recorded_plumbers_stop_is_measured_at_same_fixed_material_point():
    data = json.loads((Path(__file__).parent / 'fixtures/plumbers_fixed_grasp_point_regression.json').read_text())
    first, last = data['capture'], data['stop']
    reference = capture_grasp_reference(**first)
    saved = copy.deepcopy(reference)
    result = measure_grasp_point_slip(reference, **last)
    assert reference == saved
    assert result['object_origin_position_slip'] == pytest.approx(0.003331166493276339, abs=1e-10)
    assert result['position_slip'] == pytest.approx(0.0005746850513293766, abs=1e-10)
    assert result['position_slip'] < 0.003 < result['object_origin_position_slip']
