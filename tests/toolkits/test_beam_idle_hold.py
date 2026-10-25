"""Native idle targets must stay fixed while active physical work stays untouched."""
import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from toolkits.factory_dual_franka_assembly.beam_idle_hold import (
    native_joint_vector, published_arm_target,
)

# Import the actual policy with only simulator asset-path configuration replaced.
# No Isaac installation is needed to exercise its real action branches.
ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / 'toolkits/factory_dual_franka_assembly/demo_policy.py'
tree = ast.parse(POLICY_PATH.read_text())
tree.body = [node for node in tree.body if not (
    isinstance(node, ast.ImportFrom) and node.module == 'internutopia_extension.configs.robots.franka')]
scope = {'__file__': str(POLICY_PATH),
         'arm_ik_cfg': NS(name='arm_ik_controller'),
         'arm_joint_cfg': NS(name='arm_joint_controller'),
         'gripper_cfg': NS(name='gripper_controller')}
exec(compile(ast.fix_missing_locations(tree), str(POLICY_PATH), 'exec'), scope)
Policy = scope['DualFrankaAssemblyDemoPolicy']


class NativeSubset:
    def __init__(self):
        self.q = np.array([-3.5, -.6, 1.5, -2.7, -2.6, 6.8])
        self.joint_indices = np.arange(6)

    def get_joint_positions(self):
        return self.q


class Task:
    def __init__(self, recipe='fabrica_beam_ur5e_staged'):
        self.cfg = self.config = NS(recipe=recipe, robot_names=['left'], seed=42, episode_idx=0)
        self.step_counter = 10
        self.phase_index = 0
        self.phase_entry_step = 0
        self.phase = 'right_moving_left_idle'
        self.failed = self.success = False
        self._attachments = {}
        self._beam_shared_grasps = {}
        self.subset = NativeSubset()
        self.applied = None
        self.published = []
        raw = NS(get_applied_action=lambda: self.applied)
        controller = NS(get_joint_subset=lambda: self.subset)
        self.robots = {'left': NS(controllers={'arm_joint_controller': controller,
                               'arm_ik_controller': controller},
                               articulation=NS(unwrap=lambda: raw),
                               get_last_action=lambda: self.published)}
        self.spec = {'name': self.phase, 'gripper_commands': {'left': 'open'}}
        self.phase_specs = [self.spec]
        self.tracking = {'left': {'position': [0.1, -.65, .8],
                                 'orientation': [1, 0, 0, 0], 'gripper_opening': 1.}}
        self.objects = {}

    def get_current_phase_spec(self):
        return self.spec

    def get_tracked_robot_states(self, **kwargs):
        return self.tracking

    def get_tracked_object_states(self):
        return self.objects


class BeamIdleHold(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'idle.jsonl'
        self.env = patch.dict(os.environ, {'BEAM_IDLE_HOLD_ANCHOR': '1',
            'BEAM_IDLE_HOLD_TRACE_PATH': str(self.path)}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.task = Task()
        self.policy = Policy()
        self.policy._local_skill_executor.action_for = lambda **kwargs: None

    def act(self):
        return self.policy.act(self.task)['left']

    def target(self, action):
        return np.array(action['arm_joint_controller'][0])

    def test_native_initial_target_does_not_wrap_or_absorb_drift(self):
        initial = self.task.subset.q.copy()
        first = self.act()
        np.testing.assert_array_equal(self.target(first), initial)
        self.task.subset.q += .015
        self.task.step_counter += 1
        second = self.act()
        np.testing.assert_array_equal(self.target(second), initial)
        self.assertEqual(second['gripper_controller'], [1.])
        self.assertEqual(second['arm_ik_controller'], [None, None])

    def test_idle_phase_change_keeps_anchor_when_plan_state_is_refreshed(self):
        initial = self.target(self.act()).copy()
        self.task.phase_index += 1
        self.task.phase_entry_step = 20
        self.task.phase = self.task.spec['name'] = 'next_right_motion_left_idle'
        self.task.subset.q += .1
        self.task.step_counter = 20
        np.testing.assert_array_equal(self.target(self.act()), initial)

    def test_actual_articulation_target_wins_over_telemetry_and_measurement(self):
        actual = self.task.subset.q + .02
        self.task.applied = NS(joint_positions=actual.copy(), joint_indices=np.arange(6))
        self.task.published = [{'joint_positions': (actual + .03).tolist(), 'joint_indices': list(range(6))}]
        np.testing.assert_array_equal(self.target(self.act()), actual)
        self.assertEqual(self.policy._beam_idle_anchors['left']['source'], 'articulation_applied_target')

    def test_published_native_target_fallback_ignores_gripper_commands(self):
        target = self.task.subset.q + .01
        self.task.published = [{'joint_positions': target.tolist(), 'joint_indices': list(range(6))},
                               {'joint_positions': [.6], 'joint_indices': [6]}]
        np.testing.assert_array_equal(self.target(self.act()), target)
        self.assertEqual(self.policy._beam_idle_anchors['left']['source'], 'robot_published_target')

    def test_historical_two_pi_target_is_rejected_without_modifying_its_branch(self):
        target = self.task.subset.q.copy()
        target[0] += 2*np.pi
        self.task.applied = NS(joint_positions=target.copy(), joint_indices=np.arange(6))
        initial = self.task.subset.q.copy()
        np.testing.assert_array_equal(self.target(self.act()), initial)
        np.testing.assert_array_equal(self.task.applied.joint_positions, target)
        self.assertTrue(self.policy._beam_idle_anchors['left']['rejected_published_target_branch'])

    def test_same_task_step_reset_releases_old_anchor_and_recaptures(self):
        old = self.target(self.act()).copy()
        self.task.step_counter = 100
        self.act()
        self.task.step_counter = 0
        self.task.subset.q += .2
        new = self.target(self.act())
        np.testing.assert_array_equal(new, self.task.subset.q)
        self.assertFalse(np.array_equal(old, new))
        self.assertEqual(self.policy._policy_step, 1)

    def test_new_episode_signature_or_task_identity_cannot_reuse_anchor(self):
        self.act()
        self.task.cfg.episode_idx += 1
        self.task.subset.q += .1
        np.testing.assert_array_equal(self.target(self.act()), self.task.subset.q)
        replacement = Task()
        replacement.subset.q += .3
        self.task = replacement
        np.testing.assert_array_equal(self.target(self.act()), replacement.subset.q)

    def test_other_recipes_and_explicitly_disabled_flag_keep_legacy_behavior(self):
        for recipe, flag in (('fabrica_car_ur5e_staged', '1'),
                             ('fabrica_beam_ur5e_staged', '0'),
                             ('fabrica_beam_ur5e_staged', '')):
            with self.subTest(recipe=recipe, flag=flag):
                self.task = Task(recipe)
                self.policy = Policy()
                self.policy._local_skill_executor.action_for = lambda **kwargs: None
                os.environ['BEAM_IDLE_HOLD_ANCHOR'] = flag
                expected = self.policy._normalized_joint_action(self.task.subset.q, reference=self.task.subset.q)
                np.testing.assert_array_equal(self.target(self.act()), expected[0])
                self.assertEqual(self.policy._beam_idle_anchors, {})

    def test_generated_continuous_beam_default_keeps_native_idle_target(self):
        # Execute the actual generation function without importing unrelated
        # recipe/YAML loaders or simulator configuration dependencies.
        path = ROOT / 'roboassemblybench/core/fabrica_canonical.py'
        node = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_apply_continuous_beam_physics')
        module = ast.Module(body=[ast.ImportFrom(module='__future__',
            names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
        generated_scope = {}
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), generated_scope)
        apply_physics = generated_scope['_apply_continuous_beam_physics']
        os.environ.pop('BEAM_IDLE_HOLD_ANCHOR')
        self.task.phase_specs.append({'name': 'next_right_motion_left_idle',
                                      'gripper_commands': {'left': 'open'}})
        apply_physics(self.task.phase_specs)
        self.assertTrue(all(p['beam_idle_hold_anchor'] is True for p in self.task.phase_specs))
        initial = self.task.subset.q.copy()
        np.testing.assert_array_equal(self.target(self.act()), initial)
        self.task.phase_index = 1
        self.task.phase_entry_step = 20
        self.task.phase = 'next_right_motion_left_idle'
        self.task.spec = self.task.phase_specs[1]
        self.task.step_counter = 20
        self.task.subset.q += .1
        np.testing.assert_array_equal(self.target(self.act()), initial)

    def test_zero_override_disables_generated_phase_flag(self):
        self.task.spec['beam_idle_hold_anchor'] = True
        os.environ['BEAM_IDLE_HOLD_ANCHOR'] = '0'
        self.assertFalse(self.policy._beam_idle_enabled(self.task))
        expected = self.policy._normalized_joint_action(self.task.subset.q, reference=self.task.subset.q)
        np.testing.assert_array_equal(self.target(self.act()), expected[0])
        self.assertEqual(self.policy._beam_idle_anchors, {})

    def test_final_generation_flags_inserted_coordination_phases_without_reapplying_physics(self):
        # Execute the real compile ordering with a coordinator that appends
        # an unflagged stage. Its special parameters must survive untouched.
        path = ROOT / 'roboassemblybench/core/fabrica_canonical.py'
        source = ast.parse(path.read_text())
        # Continuous Beam dispatches out of the generic compiler before this
        # region. Exercise its authoritative implementation, not the old dead
        # Beam branch that used to be copied in fabrica_canonical.py.
        beam_source = ast.parse((ROOT / 'roboassemblybench/core/beam_physical_compiler.py').read_text())
        beam_compile = next(n for n in beam_source.body if isinstance(n, ast.FunctionDef)
                            and n.name == 'compile_continuous_beam_recipe')
        beam_begin = next(i for i, n in enumerate(beam_compile.body)
                          if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                          and isinstance(n.value.func, ast.Name)
                          and n.value.func.id == '_compile_targets_and_phases')
        beam_end = next(i for i, n in enumerate(beam_compile.body)
                        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                            and t.id == 'part_names' for t in n.targets))
        beam_region = ast.Module(body=beam_compile.body[beam_begin:beam_end], type_ignores=[])
        compile_node = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                            and n.name == 'compile_fabrica_canonical_recipe')
        begin = next(i for i, n in enumerate(compile_node.body)
                     if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                     and isinstance(n.value.func, ast.Name)
                     and n.value.func.id == '_compile_targets_and_phases')
        end = next(i for i, n in enumerate(compile_node.body)
                   if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                       and t.id == 'part_names' for t in n.targets))
        apply_node = next(n for n in beam_source.body if isinstance(n, ast.FunctionDef)
                          and n.name == '_apply_continuous_beam_physics')
        scope = {}
        definition = ast.Module(body=[ast.ImportFrom(module='__future__',
            names=[ast.alias(name='annotations')], level=0), apply_node,
            *[n for n in source.body if isinstance(n, ast.FunctionDef)
              and n.name in {'_apply_idle_robot_home_policy', '_apply_ik_execution_policy'}]],
            type_ignores=[])
        exec(compile(ast.fix_missing_locations(definition), str(path), 'exec'), scope)
        apply_physics = scope['_apply_continuous_beam_physics']
        region = ast.Module(body=compile_node.body[begin:end], type_ignores=[])
        for assembly, continuous in (('beam', True), ('beam', False), ('car', False)):
            with self.subTest(assembly=assembly, continuous=continuous):
                phases = [{'name': 'initial_idle'}]
                inserted = {'name': 'coordinated_idle',
                    'local_skill': {'cartesian_position_step': .007, 'offset': [0., 0., .123]},
                    'collision_filters': [{'enabled': True}],
                    'gripper_sibling_filters': {'special_filter': True}}
                expected = json.loads(json.dumps(inserted))
                calls = []
                def physics_pass(items):
                    calls.append('physics')
                    apply_physics(items)
                def coordinate(items, *_args):
                    calls.append('coordinate')
                    self.assertEqual(items[0].get('beam_idle_hold_anchor'), True if continuous else None)
                    items.append(inserted)
                module = NS(coordinate_phases=coordinate,
                            configure_beam_velocity_solver=lambda *_args: None)
                run_scope = {'assembly': assembly, 'plumbers_physics': False,
                    'spec': {'continuous_physics': continuous, 'official_bimanual_hold': True,
                             'coordinated_support_plan': True},
                    'task': {}, 'pickup_origin': [], 'pickup_orientation': [],
                    'assembly_origin': [], 'assembly_orientation': [], 'robots': [],
                    'generated_objects': [], '_apply_continuous_beam_physics': physics_pass,
                    '_apply_idle_robot_home_policy': scope['_apply_idle_robot_home_policy'],
                    '_apply_ik_execution_policy': scope['_apply_ik_execution_policy'],
                    '_compile_targets_and_phases': lambda **_kwargs: ([], phases, {}, [], [])}
                with patch.dict('sys.modules', {'roboassemblybench.core.beam_coordinated': module}):
                    selected_region = beam_region if continuous else region
                    exec(compile(ast.fix_missing_locations(selected_region), str(path), 'exec'), run_scope)
                self.assertEqual(calls, ['physics', 'coordinate'] if continuous else ['coordinate'])
                if continuous:
                    self.assertTrue(all(p['beam_idle_hold_anchor'] is True for p in phases))
                    expected['beam_idle_hold_anchor'] = True
                else:
                    self.assertTrue(all('beam_idle_hold_anchor' not in p for p in phases))
                self.assertEqual(inserted, expected)

    def test_phase_flag_never_enables_non_beam_recipe(self):
        self.task = Task('fabrica_car_ur5e_staged')
        self.task.spec['beam_idle_hold_anchor'] = True
        for override in (None, '1'):
            with self.subTest(override=override):
                if override is None:
                    os.environ.pop('BEAM_IDLE_HOLD_ANCHOR', None)
                else:
                    os.environ['BEAM_IDLE_HOLD_ANCHOR'] = override
                self.assertFalse(self.policy._beam_idle_enabled(self.task))

    def test_unflagged_beam_keeps_legacy_behavior_without_environment_override(self):
        os.environ.pop('BEAM_IDLE_HOLD_ANCHOR')
        self.assertFalse(self.policy._beam_idle_enabled(self.task))
        expected = self.policy._normalized_joint_action(self.task.subset.q, reference=self.task.subset.q)
        np.testing.assert_array_equal(self.target(self.act()), expected[0])
        self.assertEqual(self.policy._beam_idle_anchors, {})

    def test_default_enable_reads_only_current_explicit_boolean_phase_flag(self):
        os.environ.pop('BEAM_IDLE_HOLD_ANCHOR')
        def forbidden(*args, **kwargs):
            self.fail('Enable check must not call task observation or phase getters')
        self.task.get_obs = self.task.get_current_phase_spec = forbidden
        self.task.phase_specs.append({'name': 'future', 'beam_idle_hold_anchor': True})
        for flag in (None, False, 'true', 1):
            with self.subTest(flag=flag):
                self.task.spec['beam_idle_hold_anchor'] = flag
                self.assertFalse(self.policy._beam_idle_enabled(self.task))
        self.task.spec['beam_idle_hold_anchor'] = True
        self.assertTrue(self.policy._beam_idle_enabled(self.task))
        for index in (-1, len(self.task.phase_specs), None):
            self.task.phase_index = index
            self.assertFalse(self.policy._beam_idle_enabled(self.task))

    def test_active_local_skill_passes_through_and_releases_idle_anchor(self):
        self.act()
        command = {'arm_joint_controller': [[0., .1, .2, .3, .4, .5]], 'gripper_controller': [.15]}
        self.policy._local_skill_executor.action_for = lambda **kwargs: command
        action = self.act()
        self.assertIs(action, command)
        self.assertEqual(self.policy._beam_idle_anchors, {})

    def test_active_grasp_freeze_keeps_original_joint_action(self):
        self.act()
        state = self.policy._robot_state('left')
        target = {'pose': {'position': np.array([.2, -.5, .7]),
                          'orientation': np.array([1., 0, 0, 0])}, 'name': 'grasp', 'mode': 'pick'}
        self.policy._refresh_robot_plan_if_needed = lambda **kwargs: state
        self.policy._trajectory_target = lambda *_: target
        self.policy._should_freeze_for_release = lambda *_: True
        expected = self.policy._normalized_joint_action(self.task.subset.q, reference=self.task.subset.q)
        np.testing.assert_array_equal(self.target(self.act()), expected[0])
        self.assertEqual(self.policy._beam_idle_anchors, {})

    def test_clearance_takeover_keeps_planned_action_and_releases_anchor(self):
        self.act()
        target_q = np.array([.1, .2, .3, .4, .5, .6])
        pose = {'position': np.array([.2, -.7, .9]), 'orientation': np.array([1., 0, 0, 0])}
        self.policy._idle_clearance_pose = lambda **kwargs: pose
        self.policy._joint_trajectory_action_for_pose = lambda *_a, **_kw: [target_q.tolist()]
        action = self.act()
        np.testing.assert_array_equal(self.target(action), target_q)
        self.assertEqual(action['arm_ik_controller'][0], pose['position'].tolist())
        self.assertEqual(self.policy._beam_idle_anchors, {})

    def test_primary_and_shared_support_never_use_idle_anchor(self):
        for shared in (False, True):
            with self.subTest(shared=shared):
                self.policy = Policy()
                self.policy._local_skill_executor.action_for = lambda **kwargs: None
                self.task._attachments = {}
                self.task._beam_shared_grasps = {}
                self.act()
                if shared:
                    self.task._beam_shared_grasps[('fabrica_beam_6', 'left')] = {'mode': 'pure_physical_grasp'}
                else:
                    self.task._attachments['fabrica_beam_6'] = {'robot_name': 'left', 'mode': 'pure_physical_grasp'}
                expected = self.policy._normalized_joint_action(self.task.subset.q, reference=self.task.subset.q)
                np.testing.assert_array_equal(self.target(self.act()), expected[0])
                self.assertEqual(self.policy._beam_idle_anchors, {})

    def test_disappeared_support_cannot_reactivate_an_old_idle_park_target(self):
        self.act()
        self.task._attachments['fabrica_beam_6'] = {'robot_name': 'left', 'mode': 'pure_physical_grasp'}
        self.act()
        self.task._attachments.clear()
        self.task.subset.q += .1
        self.act()
        self.assertTrue(self.policy._beam_idle_guard['left']['blocked'])
        self.assertEqual(self.policy._beam_idle_anchors, {})

    def test_closed_gripper_or_unfinished_skill_is_not_treated_as_idle(self):
        self.task.spec['gripper_commands']['left'] = 'close'
        self.act()
        self.assertEqual(self.policy._beam_idle_anchors, {})
        self.task.spec['gripper_commands']['left'] = 'open'
        self.task.spec['local_skills'] = {'left': {'name': 'physical_hold'}}
        self.act()
        self.assertEqual(self.policy._beam_idle_anchors, {})

    def test_trace_reports_fixed_target_and_measured_drift_at_bounded_frequency(self):
        initial = self.task.subset.q.copy()
        for index in range(70):
            self.task.step_counter += 1
            self.task.subset.q[:] = initial + index*.001
            self.act()
        records = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual(len(records), 3)
        self.assertEqual([r['event'] for r in records], ['anchor_captured', 'anchor_command', 'anchor_command'])
        self.assertTrue(all(r['target_native_q'] == initial.tolist() for r in records))
        self.assertAlmostEqual(records[-1]['max_abs_tracking_error_rad'], .064)

    def test_command_lists_and_sensor_buffers_cannot_mutate_saved_anchor(self):
        first = self.act()
        saved = self.policy._beam_idle_anchors['left']['target'].copy()
        first['arm_joint_controller'][0][0] += 1
        self.task.subset.q += .2
        np.testing.assert_array_equal(self.target(self.act()), saved)

    def test_policy_never_writes_robot_pose_or_joint_state(self):
        robot = self.task.robots['left']
        def forbidden(*args, **kwargs):
            self.fail('Idle hold must return controller targets, not write physical state')
        robot.set_pose = robot.set_joint_positions = forbidden
        robot.articulation.set_joint_positions = robot.articulation.set_joint_velocities = forbidden
        self.act()


class PlumbersIdleHold(unittest.TestCase):
    """Run the real policy through the separate physical-plumbers switch."""

    act = BeamIdleHold.act
    target = BeamIdleHold.target

    def setUp(self):
        BeamIdleHold.setUp(self)
        self.task = Task('fabrica_plumbers_block_ur5e_staged')
        self.task.spec['physical_idle_hold_anchor'] = True
        self.task.spec['physical_idle_hold_robot'] = 'left'

    def test_native_hold_target_stays_fixed_despite_measured_drift(self):
        initial = self.task.subset.q.copy()
        for index in range(8):
            self.task.subset.q[:] = initial + index * .01
            self.task.step_counter += 1
            action = self.act()
            np.testing.assert_array_equal(self.target(action), initial)
            np.testing.assert_array_equal(action['arm_joint_controller'][1], np.zeros(6))
        self.assertFalse(self.policy._beam_idle_enabled(self.task))
        self.assertTrue(self.policy._idle_hold_enabled(self.task))

    def test_right_motion_does_not_change_left_idle_target(self):
        right = Task('fabrica_plumbers_block_ur5e_staged')
        self.task.config.robot_names.append('right')
        self.task.robots['right'] = right.robots['left']
        self.task.tracking['right'] = dict(right.tracking['left'], position=[.1, .65, .8])
        self.task.spec['local_skills'] = {'right': {'name': 'active_motion', 'robot': 'right'}}
        initial = self.task.subset.q.copy()
        for index in range(4):
            right_target = right.subset.q + index * .02
            command = {'arm_joint_controller': [right_target.tolist()]}
            self.policy._local_skill_executor.action_for = (
                lambda *, robot_name, **kwargs: command if robot_name == 'right' else None)
            self.task.subset.q[:] = initial + index * .01
            self.task.step_counter += 1
            actions = self.policy.act(self.task)
            np.testing.assert_array_equal(self.target(actions['left']), initial)
            np.testing.assert_array_equal(actions['left']['arm_joint_controller'][1], np.zeros(6))
            self.assertIs(actions['right'], command)
            self.assertEqual(set(self.policy._beam_idle_anchors), {'left'})

    def test_disabled_plumbers_switch_keeps_legacy_idle_behavior(self):
        self.task.spec['physical_idle_hold_anchor'] = False
        for index in range(3):
            self.task.subset.q += .01
            expected = self.policy._normalized_joint_action(self.task.subset.q, reference=self.task.subset.q)
            np.testing.assert_array_equal(self.target(self.act()), expected[0])
            self.assertEqual(self.policy._beam_idle_anchors, {})

    def test_plumbers_supporter_updates_before_inserting_arm_despite_config_order(self):
        right = Task('fabrica_plumbers_block_ur5e_staged')
        self.task.config.robot_names.append('right')
        self.task.robots['right'] = right.robots['left']
        self.task.tracking['right'] = dict(right.tracking['left'], position=[.1, .65, .8])
        self.task.spec['plumbers_base_support'] = {'robot': 'right', 'inserting_robot': 'left'}
        order = []
        def local_action(*, robot_name, **kwargs):
            order.append(robot_name)
            return {'arm_joint_controller': [[0.] * 6]}
        self.policy._local_skill_executor.action_for = local_action
        self.policy.act(self.task)
        self.assertEqual(order, ['right', 'left'])

    def test_support_failure_cannot_advance_left_local_skill_or_fallback_planner(self):
        right = Task('fabrica_plumbers_block_ur5e_staged')
        self.task.config.robot_names.append('right')
        self.task.robots['right'] = right.robots['left']
        self.task.tracking['right'] = dict(right.tracking['left'], position=[.1, .65, .8])
        self.task.spec['plumbers_base_support'] = {'robot': 'right', 'inserting_robot': 'left'}
        def local_action(*, robot_name, **kwargs):
            self.assertEqual(robot_name, 'right')
            return {'__local_skill_failure__': True, 'reason': 'support_lost'}
        self.policy._local_skill_executor.action_for = local_action
        self.policy._compose_robot_action = lambda **kwargs: self.fail('Must not invoke insertion planner after support loss')
        actions = self.policy.act(self.task)
        self.assertTrue(self.task.failed)
        self.assertEqual(actions, {'left': {}, 'right': {}})

    def test_published_target_is_kept_across_gripper_updates_and_idle_phases(self):
        target = self.task.subset.q + .01
        self.task.applied = NS(joint_positions=target.copy(), joint_indices=np.arange(6))
        np.testing.assert_array_equal(self.target(self.act()), target)
        self.task.applied = NS(joint_positions=[.2], joint_indices=[6])
        self.task.subset.q += .1
        self.task.spec = {'name': 'next_idle', 'physical_idle_hold_anchor': True,
                          'physical_idle_hold_robot': 'left'}
        self.task.phase_specs.append(self.task.spec)
        self.task.phase_index = 1
        self.task.phase_entry_step = self.task.step_counter = 30
        self.task.phase = 'next_idle'
        np.testing.assert_array_equal(self.target(self.act()), target)
        self.assertEqual(self.policy._beam_idle_anchors['left']['source'], 'articulation_applied_target')

    def test_active_skill_releases_old_target_and_idle_recaptures_new_published_target(self):
        self.act()
        new_target = self.task.subset.q + .2
        command = {'arm_joint_controller': [new_target.tolist()], 'gripper_controller': [1.]}
        self.task.spec['local_skills'] = {'left': {'name': 'new_motion', 'robot': 'left'}}
        self.policy._local_skill_executor.action_for = lambda **kwargs: command
        self.assertIs(self.act(), command)
        self.assertEqual(self.policy._beam_idle_anchors, {})
        self.task.applied = NS(joint_positions=new_target.copy(), joint_indices=np.arange(6))
        self.task.spec.pop('local_skills')
        self.task.step_counter += 1
        self.task.subset.q += .15
        self.policy._local_skill_executor.action_for = lambda **kwargs: None
        np.testing.assert_array_equal(self.target(self.act()), new_target)

    def test_physical_support_never_uses_idle_hold_or_beam_rearm_state(self):
        self.act()
        self.task._attachments['part'] = {'robot_name': 'left', 'mode': 'pure_physical_grasp'}
        self.task.subset.q += .1
        self.act()
        self.assertEqual(self.policy._beam_idle_anchors, {})
        self.task._attachments.clear()
        self.task.spec['local_skills'] = {'left': {'name': 'physical_hold', 'robot': 'left'}}
        self.task.step_counter += 1
        self.act()
        self.assertEqual(self.policy._beam_idle_guard, {})
        self.assertEqual(self.policy._beam_idle_anchors, {})

    def test_initial_settle_never_captures_target(self):
        self.task.spec.update(physical_idle_hold_anchor=False, physical_idle_hold_robot=None)
        self.task.phase = self.task.spec['name'] = 'plumbers_fixture_settle'
        action = self.act()
        self.assertEqual(len(action['arm_joint_controller']), 1)
        self.assertEqual(self.policy._beam_idle_anchors, {})
        self.assertEqual(self.policy._beam_idle_guard, {})

    def test_idle_right_arm_keeps_legacy_command_even_with_left_hold_enabled(self):
        right = Task('fabrica_plumbers_block_ur5e_staged')
        self.task.config.robot_names.append('right')
        self.task.robots['right'] = right.robots['left']
        self.task.tracking['right'] = dict(right.tracking['left'], position=[.1, .65, .8])
        for _ in range(3):
            right.subset.q += .01
            actions = self.policy.act(self.task)
            expected = self.policy._normalized_joint_action(right.subset.q, reference=right.subset.q)
            self.assertEqual(actions['right']['arm_joint_controller'], expected)
            self.assertEqual(set(self.policy._beam_idle_anchors), {'left'})
        self.assertEqual(self.policy._beam_idle_guard, {})

    def test_leaving_scope_releases_anchor_before_next_action(self):
        self.act()
        self.task.spec.update(physical_idle_hold_anchor=False, physical_idle_hold_robot=None)
        def active_action(**kwargs):
            self.assertEqual(self.policy._beam_idle_anchors, {})
            return {'arm_joint_controller': [[0.] * 6]}
        self.policy._local_skill_executor.action_for = active_action
        self.act()

    def test_step_reset_to_unflagged_initial_settle_discards_previous_anchor(self):
        self.act()
        self.task.step_counter = 0
        self.task.spec.update(physical_idle_hold_anchor=False, physical_idle_hold_robot=None)
        self.task.phase = self.task.spec['name'] = 'plumbers_fixture_settle'
        self.act()
        self.assertEqual(self.policy._policy_step, 1)
        self.assertEqual(self.policy._beam_idle_anchors, {})

    def test_missing_robot_scope_fails_closed_instead_of_enabling_both_arms(self):
        self.task.spec.pop('physical_idle_hold_robot')
        self.assertFalse(self.policy._idle_hold_enabled(self.task))
        self.act()
        self.assertEqual(self.policy._beam_idle_anchors, {})

    def test_completed_motion_phase_cannot_become_idle_mid_phase(self):
        self.act()
        self.task.spec['local_skills'] = {'left': {'name': 'active_motion', 'robot': 'left'}}
        self.task.is_local_skill_complete = lambda *_: True
        self.act()
        self.assertEqual(self.policy._beam_idle_anchors, {})

    def test_clearance_motion_is_not_blocked_by_static_hold(self):
        BeamIdleHold.test_clearance_takeover_keeps_planned_action_and_releases_anchor(self)

    def test_active_target_does_not_reuse_idle_target(self):
        BeamIdleHold.test_active_grasp_freeze_keeps_original_joint_action(self)

    def test_episode_reset_and_step_restart_discard_old_anchor(self):
        self.act()
        self.task.subset.q += .1
        self.task.cfg.episode_idx += 1
        np.testing.assert_array_equal(self.target(self.act()), self.task.subset.q)
        self.task.subset.q += .1
        self.task.step_counter = 0
        np.testing.assert_array_equal(self.target(self.act()), self.task.subset.q)

    def test_only_explicit_plumbers_phase_flag_enables_hold(self):
        for value in (None, False, 'true', 1):
            with self.subTest(value=value):
                self.task.spec['physical_idle_hold_anchor'] = value
                self.assertFalse(self.policy._idle_hold_enabled(self.task))
        self.task.spec['physical_idle_hold_anchor'] = True
        for override in ('0', '1'):
            os.environ['BEAM_IDLE_HOLD_ANCHOR'] = override
            self.assertTrue(self.policy._idle_hold_enabled(self.task))
        self.task.cfg.recipe = 'fabrica_car_ur5e_staged'
        self.assertFalse(self.policy._idle_hold_enabled(self.task))

    def test_target_is_copied_and_policy_never_writes_physical_state(self):
        BeamIdleHold.test_command_lists_and_sensor_buffers_cannot_mutate_saved_anchor(self)
        BeamIdleHold.test_policy_never_writes_robot_pose_or_joint_state(self)

    def test_anchor_trace_exposes_fixed_target_and_measured_drift(self):
        BeamIdleHold.test_trace_reports_fixed_target_and_measured_drift_at_bounded_frequency(self)


class PublishedTargetExtraction(unittest.TestCase):
    def test_native_vector_copies_without_two_pi_normalization(self):
        values = np.array([-7., 7.])
        native = native_joint_vector(values)
        values[:] = 0
        np.testing.assert_array_equal(native, [-7., 7.])

    def test_sparse_ordered_arm_indices_and_partial_nonarm_none_fields(self):
        command = {'joint_indices': [4, 2, 6, 0], 'joint_positions': [.4, .2, None, .0]}
        np.testing.assert_array_equal(published_arm_target(command, [0, 2, 4]), [.0, .2, .4])
        full = {'joint_positions': [.0, None, .2, None, .4, None, None], 'joint_indices': None}
        np.testing.assert_array_equal(published_arm_target(full, [0, 2, 4]), [.0, .2, .4])

    def test_incomplete_nonfinite_and_ambiguous_targets_are_rejected(self):
        for command in ({'joint_indices': [0, 2], 'joint_positions': [0, 2]},
                        {'joint_indices': [0, 2, 4], 'joint_positions': [0, None, 4]},
                        {'joint_indices': [0, 2, 4], 'joint_positions': [0, float('nan'), 4]},
                        {'joint_indices': [0, 2.5, 4], 'joint_positions': [0, 2, 4]},
                        {'joint_indices': [0, 2, 2, 4], 'joint_positions': [0, 2, 2, 4]}):
            with self.subTest(command=command):
                self.assertIsNone(published_arm_target(command, [0, 2, 4]))


if __name__ == '__main__':
    unittest.main()
