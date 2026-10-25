"""Native arm gain readback must validate trials without changing other drives."""
import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from toolkits.factory_dual_franka_assembly.beam_arm_drive import (
    ArmDriveReadbackError, DEFAULT_ARM_KD, DEFAULT_ARM_KP, configure_arm_drive,
)


JOINTS = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
          'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')


class NativeView:
    def __init__(self):
        self.kp = np.ones((1, 7))
        self.kd = np.ones((1, 7))
        self.efforts = np.array([[150., 150., 150., 28., 28., 28., 25.]])
        self.calls = []
        self.ignore_gains = False
        self.gain_shape = None

    def is_physics_handle_valid(self):
        return True

    def get_gains(self, joint_indices):
        self.calls.append('get_gains')
        if self.gain_shape:
            return np.zeros(self.gain_shape), np.zeros(self.gain_shape)
        return self.kp[:, joint_indices], self.kd[:, joint_indices]

    def get_max_efforts(self, joint_indices):
        self.calls.append('get_max_efforts')
        return self.efforts[:, joint_indices]

    def set_max_efforts(self, *args, **kwargs):
        raise AssertionError('The arm gain experiment must not change effort limits')


class Articulation:
    def __init__(self):
        self.view = NativeView()
        self.dof_names = (*JOINTS, 'finger_joint')
        self.setters = []

    def get_dof_index(self, name):
        return self.dof_names.index(name)

    def set_gains(self, kps, kds, joint_indices):
        self.setters.append((kps.copy(), kds.copy(), joint_indices.copy()))
        if not self.view.ignore_gains:
            self.view.kp[:, joint_indices] = kps
            self.view.kd[:, joint_indices] = kds

    def unwrap(self):
        return NS(_articulation_view=self.view)


def drive_metadata(names, indices):
    return [{'valid': True, 'joint_name': name, 'dof_index': int(index),
             'joint_prim_path': '/World/robot/joints/' + name,
             'source': 'actual_drive_schema_readback',
             'has_angular_drive_api': True, 'type': 'force'}
            for name, index in zip(names, indices)]


ROOT = Path(__file__).resolve().parents[2]
ROBOT_TREE = ast.parse((ROOT / 'internutopia_extension/robots/ur5e.py').read_text())


def robot_method(name, namespace=None):
    method = next(node for node in ast.walk(ROBOT_TREE)
                  if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {'np': np, 'os': NS(environ={}), '_UR5E_ARM_JOINT_NAMES': JOINTS,
                 'log': NS(warn=lambda *args: None, error=lambda *args: None),
                 **(namespace or {})}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                 '<real UR5e arm drive method>', 'exec'), namespace)
    return namespace[name]


class ArmDriveValidation(unittest.TestCase):
    def setUp(self):
        self.articulation = Articulation()

    def configure(self, kp=20000., kd=1000., reader=drive_metadata):
        return configure_arm_drive(self.articulation, JOINTS, kp, kd, reader)

    def test_trial_has_exact_native_readbacks_and_only_sets_the_six_arm_gains(self):
        original_efforts = self.articulation.view.efforts.copy()
        audit = self.configure()
        self.assertTrue(audit['configured'])
        self.assertTrue(audit['strict_request'])
        self.assertTrue(audit['physics_handle_valid'])
        self.assertEqual(audit['kp_readback'], [20000.] * 6)
        self.assertEqual(audit['kd_readback'], [1000.] * 6)
        kp, kd, indices = self.articulation.setters[0]
        self.assertEqual(kp.shape, (6,))
        self.assertEqual(kd.shape, (6,))
        self.assertEqual(indices.dtype, np.dtype('int64'))
        self.assertEqual(indices.tolist(), list(range(6)))
        self.assertEqual(self.articulation.view.kp[0, 6], 1.)
        np.testing.assert_array_equal(original_efforts, self.articulation.view.efforts)
        self.assertEqual(audit['max_effort_readback'], [150., 150., 150., 28., 28., 28.])

    def test_default_parameters_preserve_original_behavior_when_readback_fails(self):
        self.articulation.view.get_gains = lambda **kw: (_ for _ in ()).throw(RuntimeError('unavailable'))
        audit = self.configure(DEFAULT_ARM_KP, DEFAULT_ARM_KD)
        self.assertFalse(audit['configured'])
        self.assertFalse(audit['strict_request'])
        self.assertIn('unavailable', audit['errors']['gains'])
        np.testing.assert_array_equal(self.articulation.setters[0][0], np.full(6, 80000.))
        np.testing.assert_array_equal(self.articulation.setters[0][1], np.full(6, 4000.))

    def test_invalid_requests_fail_before_any_articulation_write(self):
        for value in (0., -1., float('nan'), float('inf'), True, [20000.], None):
            for field in ('kp', 'kd'):
                with self.subTest(value=value, field=field):
                    with self.assertRaises(ValueError):
                        self.configure(**{field: value})
                    self.assertFalse(self.articulation.setters)

    def test_duplicate_or_invalid_dof_indices_cannot_write_gains(self):
        for index in (0, -1, True, .5):
            with self.subTest(index=index):
                self.articulation.get_dof_index = lambda name: index
                with self.assertRaises(ArmDriveReadbackError):
                    self.configure()
                self.assertFalse(self.articulation.setters)

    def test_silent_ignored_trial_is_rejected_with_requested_and_actual_values(self):
        self.articulation.view.ignore_gains = True
        with self.assertRaises(ArmDriveReadbackError) as raised:
            self.configure()
        audit = raised.exception.audit
        self.assertEqual(audit['kp_requested'], 20000.)
        self.assertEqual(audit['kp_readback'], [1.] * 6)
        self.assertFalse(audit['gains_match'])
        self.assertTrue(audit['setter_completed'])

    def test_bad_native_buffer_shapes_are_not_flattened_into_apparent_success(self):
        for shape in ((2, 3), (6, 1), (2, 6), (1, 7)):
            with self.subTest(shape=shape):
                self.articulation.view.gain_shape = shape
                with self.assertRaises(ArmDriveReadbackError) as raised:
                    self.configure()
                self.assertIn('gains', raised.exception.audit['errors'])

    def test_zero_or_nonfinite_effort_limits_cannot_verify_a_trial(self):
        for value in (0., -1., float('nan'), float('inf')):
            with self.subTest(value=value):
                self.articulation.view.efforts[0, 1] = value
                with self.assertRaises(ArmDriveReadbackError) as raised:
                    self.configure()
                self.assertIn('max_efforts', raised.exception.audit['errors'])

    def test_missing_schema_api_or_unknown_type_cannot_be_assumed_force(self):
        for key, value in (('type', None), ('type', 'guessed_force'),
                           ('has_angular_drive_api', False), ('joint_prim_path', None),
                           ('dof_index', 5), ('valid', False)):
            with self.subTest(key=key):
                def reader(names, indices):
                    result = drive_metadata(names, indices)
                    result[0][key] = value
                    return result
                with self.assertRaises(ArmDriveReadbackError) as raised:
                    self.configure(reader=reader)
                self.assertIn('drive_types', raised.exception.audit['errors'])

    def test_native_buffers_are_copied_before_later_getters_reuse_them(self):
        kp, kd = np.full((1, 6), 20000.), np.full((1, 6), 1000.)
        self.articulation.view.get_gains = lambda **kw: (kp, kd)
        def efforts(**kwargs):
            kp[:] = -1.; kd[:] = -1.
            return np.ones((1, 6))
        self.articulation.view.get_max_efforts = efforts
        audit = self.configure()
        self.assertTrue(audit['configured'])
        self.assertEqual(audit['kp_readback'], [20000.] * 6)
        self.assertEqual(audit['kd_readback'], [1000.] * 6)

    def test_default_setter_failure_is_diagnostic_but_trial_failure_is_fatal(self):
        self.articulation.set_gains = lambda **kw: (_ for _ in ()).throw(RuntimeError('setter failed'))
        audit = self.configure(DEFAULT_ARM_KP, DEFAULT_ARM_KD)
        self.assertIn('setter failed', audit['errors']['set_gains'])
        with self.assertRaises(ArmDriveReadbackError):
            self.configure()

    def test_nonlive_or_missing_handle_cannot_validate_authored_or_cached_gains(self):
        for getter in (None, lambda: False, lambda: None, lambda: 1,
                       lambda: (_ for _ in ()).throw(RuntimeError('handle read failed'))):
            with self.subTest(getter=getter):
                self.articulation.view.is_physics_handle_valid = getter
                with self.assertRaises(ArmDriveReadbackError) as raised:
                    self.configure()
                audit = raised.exception.audit
                self.assertFalse(audit['configured'])
                self.assertIn('native_view', audit['errors'])
                self.assertNotIn('kp_readback', audit)
                self.assertEqual(self.articulation.view.calls, [])

    def test_default_nonlive_handle_is_reported_without_breaking_legacy_path(self):
        self.articulation.view.is_physics_handle_valid = lambda: False
        audit = self.configure(DEFAULT_ARM_KP, DEFAULT_ARM_KD)
        self.assertFalse(audit['configured'])
        self.assertFalse(audit['physics_handle_valid'])
        self.assertIn('native_view', audit['errors'])
        self.assertNotIn('kp_readback', audit)

    def test_acceleration_types_are_recorded_without_conversion_or_type_writes(self):
        def reader(names, indices):
            result = drive_metadata(names, indices)
            for item in result:
                item['type'] = 'acceleration'
            return result
        audit = self.configure(reader=reader)
        self.assertTrue(audit['configured'])
        self.assertEqual([item['type'] for item in audit['drive_types']], ['acceleration'] * 6)


class RealRobotIntegration(unittest.TestCase):
    def robot(self):
        return NS(articulation=Articulation(),
                  config=NS(name='beam_right', arm_drive_kp=80000., arm_drive_kd=4000.,
                            gripper_dof_name='finger_joint', gripper_drive_kp=7500.,
                            gripper_drive_kd=173., gripper_drive_max_effort=None),
                  _arm_drive_type_readbacks=drive_metadata,
                  _resolved_articulation_dof_name=lambda name: name)

    def test_default_diagnostic_failure_does_not_skip_existing_gripper_configuration(self):
        robot = self.robot()
        robot._arm_drive_type_readbacks = lambda *args: (_ for _ in ()).throw(RuntimeError('schema missing'))
        configure = robot_method('_configure_drive_gains')
        configure(robot)
        self.assertIn('schema missing', robot.arm_drive_diagnostics['errors']['drive_types'])
        self.assertTrue(robot.gripper_drive_diagnostics['configured'])
        self.assertEqual(robot.gripper_drive_diagnostics['kp_readback'], [[7500.]])
        self.assertEqual(robot.gripper_drive_diagnostics['kd_readback'], [[173.]])
        self.assertEqual(robot.gripper_drive_diagnostics['max_effort_readback'], [[25.]])

    def test_unconfigured_gripper_effort_never_calls_legacy_25000_setter(self):
        robot = self.robot()
        robot.config.gripper_require_drive_readback = True
        robot.articulation.set_max_efforts = lambda *a, **kw: self.fail('Unrequested effort write')
        robot_method('_configure_drive_gains')(robot)
        self.assertTrue(robot.gripper_drive_diagnostics['configured'])
        self.assertEqual(robot.gripper_drive_diagnostics['max_effort_before'], [[25.]])
        self.assertEqual(robot.gripper_drive_diagnostics['max_effort_readback'], [[25.]])

    def test_explicit_gripper_effort_changes_only_finger_and_is_verified(self):
        robot = self.robot()
        robot.config.gripper_require_drive_readback = True
        robot.config.gripper_drive_max_effort = 30.
        before = robot.articulation.view.efforts.copy()
        def set_effort(values, joint_indices):
            self.assertEqual(joint_indices.tolist(), [6])
            robot.articulation.view.efforts[:, joint_indices] = values
        robot.articulation.view.set_max_efforts = set_effort
        robot_method('_configure_drive_gains')(robot)
        np.testing.assert_array_equal(robot.articulation.view.efforts[:, :6], before[:, :6])
        self.assertEqual(robot.gripper_drive_diagnostics['max_effort_readback'], [[30.]])

    def test_silently_ignored_gripper_override_is_fatal_in_strict_mode(self):
        robot = self.robot()
        robot.config.gripper_require_drive_readback = True
        robot.config.gripper_drive_max_effort = 30.
        robot.articulation.view.set_max_efforts = lambda *a, **kw: None
        with self.assertRaisesRegex(RuntimeError, 'max_effort native readback'):
            robot_method('_configure_drive_gains')(robot)
        self.assertFalse(robot.gripper_drive_diagnostics['configured'])

    def test_strict_gripper_rejects_missing_handle_ignored_gains_and_invalid_effort(self):
        for failure in ('handle', 'gains', 'effort'):
            with self.subTest(failure=failure):
                robot = self.robot()
                robot.config.gripper_require_drive_readback = True
                if failure == 'handle':
                    robot.articulation.view.is_physics_handle_valid = lambda: False
                elif failure == 'gains':
                    robot.articulation.view.ignore_gains = True
                else:
                    robot.articulation.view.efforts[0, 6] = float('nan')
                with self.assertRaises(RuntimeError):
                    robot_method('_configure_drive_gains')(robot)

    def test_unverified_nondefault_request_aborts_instead_of_silently_using_asset_gains(self):
        robot = self.robot()
        robot.config.arm_drive_kp, robot.config.arm_drive_kd = 20000., 1000.
        robot.articulation.view.ignore_gains = True
        with self.assertRaises(ArmDriveReadbackError):
            robot_method('_configure_drive_gains')(robot)
        self.assertFalse(robot.gripper_drive_diagnostics['configured'])
        self.assertFalse(robot.arm_drive_diagnostics['gains_match'])

    def test_nonlive_arm_default_readback_does_not_skip_existing_gripper_setup(self):
        robot = self.robot()
        robot.articulation.view.is_physics_handle_valid = lambda: False
        robot_method('_configure_drive_gains')(robot)
        self.assertFalse(robot.arm_drive_diagnostics['physics_handle_valid'])
        self.assertTrue(robot.gripper_drive_diagnostics['configured'])
        self.assertEqual(robot.gripper_drive_diagnostics['kp_readback'], [[7500.]])

    def test_joint_schema_reader_reports_actual_path_and_refuses_absent_api(self):
        class Prim:
            def __init__(self, name, has_drive=True):
                self.name, self.has_drive = name, has_drive
            def IsValid(self): return True
            def IsA(self, kind): return True
            def GetName(self): return self.name
            def GetPath(self): return '/World/robot/joints/' + self.name
            def HasAPI(self, kind, instance): return self.has_drive
        prims = [Prim(name) for name in JOINTS]
        prims[2].has_drive = False
        stage = NS(GetPrimAtPath=lambda path: Prim('robot'))
        pxr = NS(Usd=NS(PrimRange=lambda *args: prims, TraverseInstanceProxies=lambda: None),
                 UsdPhysics=NS(Joint=object(), DriveAPI=lambda prim, axis:
                               NS(GetTypeAttr=lambda: NS(Get=lambda: 'acceleration'))))
        robot = self.robot()
        robot.config.prim_path = '/World/robot'
        with patch.dict(sys.modules, {'pxr': pxr, 'isaacsim.core.utils.stage':
                                      NS(get_current_stage=lambda: stage)}):
            records = robot_method('_arm_drive_type_readbacks')(robot, JOINTS, np.arange(6))
        self.assertEqual(records[0]['joint_prim_path'], '/World/robot/joints/' + JOINTS[0])
        self.assertEqual(records[0]['native_dof_name'], JOINTS[0])
        self.assertEqual(records[0]['type'], 'acceleration')
        self.assertTrue(records[0]['has_angular_drive_api'])
        self.assertFalse(records[2]['valid'])
        self.assertFalse(records[2]['has_angular_drive_api'])
        self.assertNotIn('type', records[2])


if __name__ == '__main__':
    unittest.main()
