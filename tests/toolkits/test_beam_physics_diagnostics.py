"""Boundary captures must preserve live buffers and never alter physics."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from toolkits.factory_dual_franka_assembly.beam_physics_diagnostics import (
    record_path_state_write, record_state_write, sample_boundary,
)


class BodyView:
    def __init__(self, valid=True, mutate=False):
        self.valid = valid
        self.positions = np.array([[1., 2., 3.]])
        self.orientations = np.array([[1., 0., 0., 0.]])
        self.velocities = np.array([[.1, .2, .3, .4, .5, .6]])
        self.mutate = mutate
        self.pose_calls = 0
        self.metadata_calls = 0

    def is_physics_handle_valid(self):
        return self.valid

    def get_world_poses(self, clone):
        self.pose_calls += 1
        return self.positions, self.orientations

    def get_velocities(self, clone):
        if self.mutate:
            self.positions[:] = 9
            self.orientations[:] = 9
        return self.velocities

    def get_masses(self, clone):
        self.metadata_calls += 1
        if self.mutate:
            self.velocities[:] = 9
        return np.array([.25])

    def get_inertias(self, clone):
        return np.array([[1., 0., 0., 0., 2., 0., 0., 0., 3.]])

    def get_coms(self, clone):
        return np.array([[.01, .02, .03]]), np.array([[1., 0., 0., 0.]])

    def get_sleep_thresholds(self):
        return np.array([1.e-5])


class JointView:
    def __init__(self, valid=True, mutate=False):
        self.valid = valid
        self.positions = np.arange(12, dtype=float).reshape(1, 12)
        self.velocities = np.arange(12, dtype=float).reshape(1, 12) / 10
        self.mutate = mutate

    def is_physics_handle_valid(self):
        return self.valid

    def get_joint_positions(self, clone):
        return self.positions

    def get_joint_velocities(self, clone):
        if self.mutate:
            self.positions[:] = 9
        return self.velocities

    def get_sleep_thresholds(self):
        return np.array([5.e-5])


def wrapper(view, path='/World/beam_6'):
    raw = NS(_rigid_prim_view=view, prim_path=path)
    return NS(unwrap=lambda: raw)


def task():
    body = BodyView()
    robot = NS(config=NS(left_finger_link_name='left_inner_finger',
                         right_finger_link_name='right_inner_finger'),
               _rigid_body_map={}, articulation=NS(_articulation_view=JointView()))
    for link in ('base_link', 'left_inner_finger', 'right_inner_finger'):
        path = '/World/right/Robotiq_2F_85/' + link
        robot._rigid_body_map[path] = wrapper(BodyView(), path)
    return NS(name='beam_test', step_counter=9000, phase='base_6_set_down',
              _resolved_objects={'fabrica_beam_6': wrapper(body)},
              robots={'right': robot}, _contact_probes={},
              _locked_targets={}, _lock_pin_pose={}, _attachment_joints={},
              _attachments={'fabrica_beam_6': {'mode': 'pure_physical_grasp',
                                            'robot_name': 'right', 'attach_step': 500}},
              _beam_shared_grasps={})


def clock():
    physics = NS(is_stabilization_enabled=lambda: True, get_solver_type=lambda: 'TGS',
                 is_gpu_dynamics_enabled=lambda: False)
    return NS(current_time_step_index=9003, current_time=9003/240,
              get_physics_dt=lambda: 1/240, get_physics_context=lambda: physics)


class PhysicsDiagnostics(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'boundaries.jsonl'
        self.env = {'BEAM_PHYSICS_DIAGNOSTICS': '1',
                    'BEAM_PHYSICS_DIAGNOSTICS_PATH': str(self.path)}
        self.patch = patch.dict(os.environ, self.env, clear=True)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.task = task()
        self.clock = clock()

    def sample(self, boundary='before_action', **kwargs):
        return sample_boundary(self.task, boundary, self.clock, **kwargs)

    def test_copies_every_component_with_units_and_live_source(self):
        record = self.sample()
        self.assertTrue(record['valid'])
        self.assertEqual(record['physics_step_index'], 9003)
        self.assertEqual(record['physics_dt'], 1/240)
        body = record['bodies'][0]
        self.assertEqual(body['linear_velocities'], [[.1, .2, .3]])
        self.assertEqual(body['angular_velocities'], [[.4, .5, .6]])
        self.assertEqual(body['velocity_reference'], 'center_of_mass')
        self.assertEqual(body['metadata']['masses'], [.25])
        self.assertEqual(len(record['robots']['right']['joint_positions'][0]), 12)
        self.assertEqual(len(record['bodies']), 4)
        self.assertTrue(record['global_stabilization']['enabled'])
        self.assertEqual(record['global_physics_settings']['solver_type']['value'], 'TGS')
        self.assertFalse(record['global_physics_settings']['gpu_dynamics_enabled']['value'])
        self.assertEqual(body['metadata']['sleep_thresholds'], [1.e-5])
        self.assertEqual(record['robots']['right']['metadata']['sleep_thresholds'], [5.e-5])
        self.assertEqual(record['task_constraints']['attachments']['fabrica_beam_6']['mode'],
                         'pure_physical_grasp')

    def test_borrowed_buffers_detach_before_next_getter_reuses_memory(self):
        raw = self.task._resolved_objects['fabrica_beam_6'].unwrap()
        raw._rigid_prim_view = BodyView(mutate=True)
        self.task.robots['right'].articulation._articulation_view = JointView(mutate=True)
        record = self.sample()
        body = record['bodies'][0]
        self.assertTrue(record['valid'])
        self.assertEqual(body['positions'], [[1., 2., 3.]])
        self.assertEqual(body['orientations'], [[1., 0., 0., 0.]])
        self.assertEqual(body['linear_velocities'], [[.1, .2, .3]])
        self.assertEqual(record['robots']['right']['joint_positions'][0], list(range(12)))

    def test_cached_pair_view_is_used_for_pose_velocity_and_metadata(self):
        pair = BodyView()
        pair.positions[0, 0] = 42
        self.task._contact_probes[('/World/beam_6', '/World/table')] = pair
        record = self.sample()
        body = record['bodies'][0]
        self.assertEqual(body['source'], 'cached_pair_probe')
        self.assertEqual(body['view_id'], id(pair))
        self.assertEqual(body['positions'][0][0], 42)
        self.assertEqual(pair.metadata_calls, 1)
        self.assertEqual(self.task._resolved_objects['fabrica_beam_6'].unwrap()._rigid_prim_view.pose_calls, 0)

    def test_missing_or_invalid_views_are_explicit_and_never_initialized(self):
        view = self.task._resolved_objects['fabrica_beam_6'].unwrap()._rigid_prim_view
        view.valid = False
        self.task.robots['right']._rigid_body_map.clear()
        self.task._resolve_object = lambda *_: self.fail('Must not resolve or construct a body')
        self.task._get_contact_probe = lambda *_: self.fail('Must not construct a contact view')
        self.task._robot_rigid_body_by_suffix = lambda *_: self.fail('Must not construct a link')
        record = self.sample()
        self.assertFalse(record['valid'])
        self.assertFalse(record['bodies'][0]['valid'])
        self.assertIn('valid live physics handle', record['bodies'][0]['error'])
        self.assertEqual(view.pose_calls, 0)
        self.assertTrue(all(not body['valid'] for body in record['bodies']))

    def test_force_only_contact_view_does_not_replace_cached_pose_view(self):
        self.task._contact_probes[('/World/beam_6', '/World/table')] = NS(
            is_physics_handle_valid=lambda: True)
        record = self.sample()
        body = record['bodies'][0]
        self.assertTrue(body['valid'])
        self.assertEqual(body['source'], 'cached_rigid_wrapper')
        self.assertEqual(body['positions'], [[1., 2., 3.]])

    def test_sampling_does_not_invoke_task_observation_completion_or_setters(self):
        for name in ('get_observations', 'is_done', '_set_object_pose'):
            setattr(self.task, name, lambda *_: self.fail('Must not advance or write task state'))
        before_step = self.task.step_counter
        before_position = self.task._resolved_objects['fabrica_beam_6'].unwrap()._rigid_prim_view.positions.copy()
        self.sample()
        self.assertEqual(self.task.step_counter, before_step)
        np.testing.assert_array_equal(before_position,
            self.task._resolved_objects['fabrica_beam_6'].unwrap()._rigid_prim_view.positions)

    def test_all_four_boundaries_keep_one_cycle_despite_task_counter_increment(self):
        os.environ['BEAM_PHYSICS_DIAGNOSTICS_START_STEP'] = '9000'
        os.environ['BEAM_PHYSICS_DIAGNOSTICS_END_STEP'] = '9000'
        for boundary in ('before_action', 'after_action', 'after_physics', 'after_observation'):
            if boundary == 'after_observation':
                self.task.step_counter += 1
            record = self.sample(boundary, cycle_step=9000)
            self.assertEqual(record['cycle_step'], 9000)
        records = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual(len(records), 4)
        self.assertEqual(records[-1]['task_step'], 9001)
        self.assertEqual([record['sequence'] for record in records], [0, 1, 2, 3])

    def test_failure_transition_does_not_drop_last_observation_boundary(self):
        self.sample(cycle_step=9000)
        self.task.phase = 'failed'
        self.task.step_counter = 9001
        record = self.sample('after_observation', cycle_step=9000)
        self.assertIsNotNone(record)
        self.assertEqual(record['phase'], 'failed')
        self.assertEqual(record['cycle_phase'], 'base_6_set_down')
        self.assertIsNone(self.sample('before_action', cycle_step=9001))

    def test_phase_step_window_and_stride_skip_outside_samples(self):
        os.environ.update({'BEAM_PHYSICS_DIAGNOSTICS_START_STEP': '9000',
                           'BEAM_PHYSICS_DIAGNOSTICS_END_STEP': '9004',
                           'BEAM_PHYSICS_DIAGNOSTICS_STRIDE': '2'})
        for step, expected in ((8999, False), (9000, True), (9001, False), (9002, True), (9005, False)):
            self.task.step_counter = step
            self.assertEqual(self.sample() is not None, expected)
        self.task.phase = 'insert_2'
        self.task.step_counter = 9002
        self.assertIsNone(self.sample())

    def test_metadata_getters_only_run_once_per_live_view(self):
        first = self.sample()
        second = self.sample('after_action')
        self.assertIn('metadata', first['bodies'][0])
        self.assertNotIn('metadata', second['bodies'][0])
        self.assertNotIn('global_stabilization', second)
        self.assertNotIn('global_physics_settings', second)
        self.assertNotIn('metadata', second['robots']['right'])
        self.assertEqual(self.task._resolved_objects['fabrica_beam_6'].unwrap()._rigid_prim_view.metadata_calls, 1)

    def test_observation_boundary_reflects_current_locks_and_attachment_modes(self):
        before = self.sample(cycle_step=9000)
        self.task._locked_targets['fabrica_beam_6'] = {'position': np.array([1., 2., 3.])}
        self.task._lock_pin_pose['fabrica_beam_6'] = {'orientation': np.array([1., 0., 0., 0.])}
        self.task._attachment_joints['fabrica_beam_6'] = '/World/joint'
        self.task._attachments['fabrica_beam_6']['mode'] = 'fixed_joint'
        after = self.sample('after_observation', cycle_step=9000)
        self.assertEqual(before['task_constraints']['locked_targets'], {})
        self.assertEqual(before['task_constraints']['attachments']['fabrica_beam_6']['mode'],
                         'pure_physical_grasp')
        self.assertEqual(after['task_constraints']['locked_targets']['fabrica_beam_6']['position'],
                         [1., 2., 3.])
        self.assertEqual(after['task_constraints']['lock_pin_pose']['fabrica_beam_6']['orientation'],
                         [1., 0., 0., 0.])
        self.assertEqual(after['task_constraints']['attachment_joint_paths']['fabrica_beam_6'], '/World/joint')
        self.assertEqual(after['task_constraints']['attachments']['fabrica_beam_6']['mode'], 'fixed_joint')

    def test_nonfinite_or_wrong_shape_physics_cannot_be_valid_evidence(self):
        view = self.task._resolved_objects['fabrica_beam_6'].unwrap()._rigid_prim_view
        view.velocities[0, 2] = float('nan')
        record = self.sample()
        self.assertFalse(record['valid'])
        self.assertIn('Non-finite', record['bodies'][0]['error'])
        json.loads(self.path.read_text())  # Strict JSON cannot contain a bare NaN.
        view.velocities = np.zeros((1, 3))
        record = self.sample('after_action')
        self.assertFalse(record['bodies'][0]['valid'])
        self.assertIn('shape', record['bodies'][0]['error'])

    def test_invalid_clock_is_recorded_and_not_invented(self):
        self.clock.get_physics_dt = lambda: (_ for _ in ()).throw(RuntimeError('clock unavailable'))
        record = self.sample()
        self.assertFalse(record['valid'])
        self.assertFalse(record['clock_valid'])
        self.assertIn('clock unavailable', record['clock_error'])
        self.assertNotIn('physics_step_index', record)

    def test_unavailable_metadata_and_stabilization_are_explicit(self):
        self.clock.get_physics_context = lambda: NS()
        view = self.task._resolved_objects['fabrica_beam_6'].unwrap()._rigid_prim_view
        view.get_coms = lambda **kw: (_ for _ in ()).throw(RuntimeError('COM unavailable'))
        record = self.sample()
        self.assertTrue(record['bodies'][0]['valid'])
        self.assertFalse(record['bodies'][0]['metadata']['valid'])
        self.assertIn('COM unavailable', record['bodies'][0]['metadata']['errors']['center_of_mass'])
        self.assertFalse(record['global_stabilization']['valid'])

    def test_scene_attribute_can_read_global_stabilization_without_setting_it(self):
        prim = NS(GetAttribute=lambda name: NS(Get=lambda: False))
        self.clock.get_physics_context = lambda: NS(_physics_scene=NS(GetPrim=lambda: prim))
        record = self.sample()
        self.assertFalse(record['global_stabilization']['enabled'])
        self.assertIn('existing_scene_attribute', record['global_stabilization']['source'])

    def test_state_write_audit_preserves_arguments_without_writing_physics(self):
        before = self.task._resolved_objects['fabrica_beam_6'].unwrap()._rigid_prim_view.positions.copy()
        arguments = {'position': np.array([4., 5., 6.]), 'velocity': [float('nan')]}
        record = record_state_write(self.task, 'before_set_pose', 'fabrica_beam_6',
                                   '/World/beam_6', arguments)
        self.assertEqual(record['kind'], 'state_write_call_site')
        self.assertEqual(record['arguments']['position'], [4., 5., 6.])
        self.assertEqual(record['arguments']['velocity'], [{'invalid_numeric': 'nan'}])
        self.assertTrue(record['callers'])
        np.testing.assert_array_equal(before,
            self.task._resolved_objects['fabrica_beam_6'].unwrap()._rigid_prim_view.positions)

    def test_low_level_paths_match_only_registered_beam_bodies_and_robots(self):
        self.task.robots['right'].config.prim_path = '/World/right'
        self.sample()
        record = record_path_state_write('/World/right/wrist_3_link', 'before_set_joint_positions', [1., 2.])
        self.assertEqual(record['body_path'], '/World/right/wrist_3_link')
        self.assertEqual(record['arguments'], [1., 2.])
        self.assertIsNone(record_path_state_write('/World/right_other/link', 'ignored'))
        os.environ['BEAM_PHYSICS_DIAGNOSTICS'] = '0'
        self.assertIsNone(record_path_state_write('/World/beam_6', 'ignored'))

    def test_other_tasks_and_disabled_capture_have_no_side_effects(self):
        self.task._resolved_objects = {'fabrica_car_6': wrapper(BodyView())}
        self.assertIsNone(self.sample())
        self.assertFalse(self.path.exists())
        self.assertFalse(hasattr(self.task, '_beam_physics_diagnostics'))
        self.task = task()
        os.environ['BEAM_PHYSICS_DIAGNOSTICS'] = '0'
        self.assertIsNone(self.sample())
        self.assertIsNone(record_state_write(self.task, 'test'))
        self.assertFalse(self.path.exists())

    def test_disabling_environment_stops_already_initialized_capture(self):
        self.sample()
        size = self.path.stat().st_size
        os.environ['BEAM_PHYSICS_DIAGNOSTICS'] = '0'
        self.assertIsNone(self.sample('after_action'))
        self.assertIsNone(record_state_write(self.task, 'before_set_pose'))
        self.assertEqual(self.path.stat().st_size, size)

    def test_output_failure_and_invalid_configuration_remain_visible(self):
        os.environ['BEAM_PHYSICS_DIAGNOSTICS_PATH'] = str(Path(self.tmp.name) / 'missing' / 'trace.jsonl')
        with self.assertRaises(FileNotFoundError):
            self.sample()
        self.task = task()
        os.environ['BEAM_PHYSICS_DIAGNOSTICS_STRIDE'] = '0'
        with self.assertRaises(ValueError):
            self.sample()

    def test_duplicate_cached_link_is_not_silently_selected(self):
        robot = self.task.robots['right']
        path = '/World/duplicate/Robotiq_2F_85/base_link'
        robot._rigid_body_map[path] = wrapper(BodyView(), path)
        record = self.sample()
        self.assertFalse(record['valid'])
        self.assertIn('found 2', record['bodies'][1]['error'])

    def test_existing_end_effector_can_supply_base_without_creating_idle_fingers(self):
        robot = self.task.robots['right']
        path = '/World/right/Robotiq_2F_85/base_link'
        robot.articulation.end_effector = robot._rigid_body_map.pop(path)
        robot._rigid_body_map.clear()
        record = self.sample()
        self.assertTrue(record['bodies'][1]['valid'])
        self.assertEqual(record['bodies'][1]['path'], path)
        self.assertTrue(all(not body['valid'] for body in record['bodies'][2:]))
        self.assertEqual(robot._rigid_body_map, {})

    def test_end_effector_fallback_cannot_label_wrist_or_invalid_view_as_gripper(self):
        robot = self.task.robots['right']
        robot._rigid_body_map.clear()
        robot.articulation.end_effector = wrapper(BodyView(), '/World/right/wrist_3_link')
        record = self.sample()
        self.assertFalse(record['bodies'][1]['valid'])
        robot.articulation.end_effector = wrapper(BodyView(valid=False),
            '/World/right/Robotiq_2F_85/base_link')
        record = self.sample('after_action')
        self.assertFalse(record['bodies'][1]['valid'])

    def test_sleep_threshold_readbacks_copy_without_clone_and_never_enable_gate(self):
        sleep = np.array([1.e-5])
        body_view = self.task._resolved_objects['fabrica_beam_6'].unwrap()._rigid_prim_view
        body_view.get_sleep_thresholds = lambda: sleep
        joint_view = self.task.robots['right'].articulation._articulation_view
        joint_view.get_sleep_thresholds = lambda: sleep
        record = self.sample()
        sleep[:] = 12
        self.assertEqual(record['bodies'][0]['metadata']['sleep_thresholds'], [1.e-5])
        self.assertEqual(record['robots']['right']['metadata']['sleep_thresholds'], [1.e-5])
        self.assertNotIn('success', record)
        self.assertNotIn('table_support_ready', record)

    def test_unavailable_solver_gpu_and_sleep_metadata_are_explicit(self):
        self.clock.get_physics_context = lambda: NS()
        self.task.robots['right'].articulation._articulation_view.get_sleep_thresholds = None
        record = self.sample()
        self.assertFalse(record['global_physics_settings']['solver_type']['valid'])
        self.assertFalse(record['global_physics_settings']['gpu_dynamics_enabled']['valid'])
        self.assertFalse(record['robots']['right']['metadata']['valid'])
        self.assertTrue(record['robots']['right']['valid'])

    def test_external_forces_reads_existing_schema_once_without_writing(self):
        reads = []
        def forbidden(*args, **kwargs):
            self.fail('Scene diagnostic must not apply a schema or write an attribute')
        attribute = NS(Get=lambda: True, Set=forbidden)
        api = NS(GetEnableExternalForcesEveryIterationAttr=lambda: reads.append('get') or attribute,
                 CreateEnableExternalForcesEveryIterationAttr=forbidden, Apply=forbidden)
        self.clock.get_physics_context()._physx_scene_api = api
        first = self.sample()
        readback = first['global_physics_settings']['external_forces_every_iteration']
        self.assertTrue(readback['valid'])
        self.assertIs(readback['value'], True)
        self.assertEqual(readback['attribute'], 'physxScene:enableExternalForcesEveryIteration')
        self.assertEqual(readback['scope'], 'existing_scene_attribute')
        self.assertIn('existing_physx_scene_api', readback['source'])
        second = self.sample('after_action')
        self.assertNotIn('global_physics_settings', second)
        self.assertEqual(reads, ['get'])

    def test_external_forces_scene_attribute_fallback_preserves_false(self):
        names = []
        prim = NS(GetAttribute=lambda name: names.append(name) or NS(Get=lambda: False))
        self.clock.get_physics_context()._physics_scene = NS(GetPrim=lambda: prim)
        readback = self.sample()['global_physics_settings']['external_forces_every_iteration']
        self.assertTrue(readback['valid'])
        self.assertIs(readback['value'], False)
        self.assertIn('physxScene:enableExternalForcesEveryIteration', names)
        self.assertIn('existing_scene_attribute:', readback['source'])

    def test_external_forces_missing_invalid_or_throwing_values_are_explicit(self):
        for value in (None, 'false', 0, np.array([True])):
            with self.subTest(value=str(value)):
                self.task = task()
                self.clock.get_physics_context()._physx_scene_api = NS(
                    GetEnableExternalForcesEveryIterationAttr=lambda: NS(Get=lambda: value))
                readback = self.sample()['global_physics_settings']['external_forces_every_iteration']
                self.assertFalse(readback['valid'])
                self.assertIn('boolean', readback['error'])
        self.task = task()
        self.clock.get_physics_context()._physx_scene_api = NS(
            GetEnableExternalForcesEveryIterationAttr=lambda: (_ for _ in ()).throw(RuntimeError('read unavailable')))
        readback = self.sample()['global_physics_settings']['external_forces_every_iteration']
        self.assertFalse(readback['valid'])
        self.assertIn('read unavailable', readback['error'])
        self.task = task()
        self.clock.get_physics_context = lambda: NS()
        readback = self.sample()['global_physics_settings']['external_forces_every_iteration']
        self.assertFalse(readback['valid'])
        self.assertIn('No existing physics scene', readback['error'])


if __name__ == '__main__':
    unittest.main()
