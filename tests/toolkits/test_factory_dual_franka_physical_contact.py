from collections import deque
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from internutopia_extension.tasks.factory_dual_franka_assembly_task import (
    FactoryDualFrankaAssemblyTask,
)
from toolkits.factory_dual_franka_assembly.planner_primitives import (
    compose_pose,
    normalize_quat,
    quat_conjugate,
    quat_multiply,
    quat_rotate,
)
from toolkits.factory_dual_franka_assembly.plumbers_block_ur5e_skills import (
    UR5eAssemblyAtomicSkillAdapter,
)


def _finger(
    *,
    force_contact: bool,
    signed_x: float,
    axis: str = 'x',
    local_point: list[float] | None = None,
) -> dict:
    return {
        'force_contact': force_contact,
        'force_probe_valid': True,
        'geometric_contact': True,
        'surface_gap': 0.001,
        'local_contact': {
            'local_point': local_point,
            'axes': {
                axis: {
                    'contact': True,
                    'surface_gap': 0.001,
                    'signed_coordinate': signed_x,
                }
            },
        },
    }


def test_relative_cartesian_target_stops_at_inner_workspace_radius():
    current = np.asarray([0.60711995, 0.03117098, 1.21609908], dtype=float)
    target = np.asarray([0.51775714, 0.22468429, 1.23617826], dtype=float)
    center = np.asarray([0.50, 0.30, 0.998], dtype=float)

    bounded = UR5eAssemblyAtomicSkillAdapter._bound_relative_target_to_planar_workspace(
        current_position=current,
        target_position=target,
        workspace_center=center,
        minimum_planar_radius=0.28,
    )

    assert np.isclose(np.linalg.norm(bounded[:2] - center[:2]), 0.28)
    assert np.isclose(bounded[2], target[2])
    planar_delta = target[:2] - current[:2]
    bounded_delta = bounded[:2] - current[:2]
    assert np.dot(planar_delta, bounded_delta) > 0.0
    assert np.isclose(
        planar_delta[0] * bounded_delta[1],
        planar_delta[1] * bounded_delta[0],
    )


def test_relative_cartesian_target_keeps_pose_outside_inner_workspace_radius():
    target = np.asarray([0.72, 0.01, 1.24], dtype=float)
    bounded = UR5eAssemblyAtomicSkillAdapter._bound_relative_target_to_planar_workspace(
        current_position=[0.68, 0.04, 1.22],
        target_position=target,
        workspace_center=[0.50, 0.30, 0.998],
        minimum_planar_radius=0.28,
    )

    np.testing.assert_allclose(bounded, target)


def test_relative_park_locks_entry_position_but_tracks_measured_orientation():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    phase_key = (1, 2, 3, 'franka_left', 'ur5e_retreat_vertical')
    spec = {
        'relative_to_current_tcp': True,
        'offset': [0.20, 0.0, 0.02],
        'offset_frame': 'world',
        'lock_target_position': True,
        'lock_target_orientation': False,
    }
    tracked_robots = {
        'franka_left': {
            'position': [0.40, 0.10, 1.00],
            'orientation': [1.0, 0.0, 0.0, 0.0],
        }
    }

    first = adapter._target_pose(
        phase_key=phase_key,
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec=spec,
        tracked_robots=tracked_robots,
        tracked_objects={},
    )
    first = adapter._locked_target_pose(
        phase_key=phase_key,
        target_pose=first,
        spec=spec,
    )

    tracked_robots['franka_left'] = {
        'position': [0.50, 0.10, 1.01],
        'orientation': [0.9238795325, 0.0, 0.3826834324, 0.0],
    }
    second = adapter._target_pose(
        phase_key=phase_key,
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec=spec,
        tracked_robots=tracked_robots,
        tracked_objects={},
    )
    second = adapter._locked_target_pose(
        phase_key=phase_key,
        target_pose=second,
        spec=spec,
    )

    np.testing.assert_allclose(first['position'], [0.60, 0.10, 1.02])
    np.testing.assert_allclose(second['position'], first['position'])
    np.testing.assert_allclose(
        second['orientation'],
        normalize_quat(tracked_robots['franka_left']['orientation']),
    )
    assert not np.allclose(second['orientation'], first['orientation'])


def _task() -> FactoryDualFrankaAssemblyTask:
    task = object.__new__(FactoryDualFrankaAssemblyTask)
    task._object_metadata_map = {'part': {'scale': [0.06, 0.04, 0.05]}}
    return task


def test_attachment_collision_filters_are_authored_before_physics_initialization(monkeypatch):
    task = _task()
    task._policy_attach_specs = [
        {
            'object': 'part',
            'robot': 'franka_right',
            'filter_gripper_collisions_on_attach': True,
        },
        {
            'object': 'other_part',
            'robot': 'franka_left',
            'filter_gripper_collisions_on_attach': False,
        },
    ]
    task._preconfigured_attachment_collision_filters = {}
    calls = []
    monkeypatch.setattr(
        task,
        '_set_attachment_gripper_collision_filter',
        lambda *args, **kwargs: calls.append((args, kwargs))
        or ['/World/franka_right/panda_hand', '/World/franka_right/panda_leftfinger'],
    )

    task._preconfigure_attachment_gripper_collision_filters()

    assert calls == [(('part', 'franka_right'), {'enabled': True})]
    assert task._preconfigured_attachment_collision_filters[('part', 'franka_right')] == [
        '/World/franka_right/panda_hand',
        '/World/franka_right/panda_leftfinger',
    ]


def test_task_pose_uses_physical_hand_and_reports_lula_frame_delta():
    task = _task()
    physical_position = np.asarray([0.4, -0.2, 1.1], dtype=float)
    physical_orientation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    kinematic_position = physical_position + np.asarray([0.001, 0.0, 0.0], dtype=float)
    kinematic_orientation = np.asarray([0.9999500037, 0.0, 0.0099998333, 0.0], dtype=float)
    task.robots = {
        'franka_right': SimpleNamespace(
            articulation=SimpleNamespace(
                end_effector=SimpleNamespace(get_pose=lambda: (physical_position, physical_orientation))
            ),
            controllers={
                'arm_ik_controller': SimpleNamespace(
                    get_obs=lambda: {
                        'eef_position': kinematic_position,
                        'eef_orientation': kinematic_orientation,
                    }
                )
            },
        )
    }

    position, orientation = task._get_robot_task_pose('franka_right')
    diagnostic = task._get_robot_pose_frame_diagnostic('franka_right')

    np.testing.assert_allclose(position, physical_position)
    np.testing.assert_allclose(orientation, physical_orientation)
    np.testing.assert_allclose(diagnostic['kinematics_position'], kinematic_position)
    assert np.isclose(diagnostic['position_error'], 0.001)
    assert diagnostic['orientation_error'] > 0.0


def test_compliant_attachment_uses_a_distinct_joint_prim_path(monkeypatch):
    task = _task()
    rigid_body = SimpleNamespace(unwrap=lambda: SimpleNamespace(prim_path='/World/part'))
    monkeypatch.setattr(task, '_resolve_object', lambda _name: rigid_body)

    fixed_path = task._attachment_joint_path('part')
    compliant_path = task._attachment_joint_path('part', compliant=True)

    assert fixed_path == '/World/part/assembly_attachment_joint'
    assert compliant_path == '/World/part/assembly_compliant_attachment_joint'
    assert fixed_path != compliant_path


def test_enabled_rigid_body_lookup_ignores_disabled_nested_api(monkeypatch):
    class _Attr:
        def __init__(self, value):
            self.value = value

        def HasAuthoredValueOpinion(self):
            return True

        def Get(self):
            return self.value

    class _RigidBodyAPI:
        def __init__(self, prim):
            self.prim = prim

        def GetRigidBodyEnabledAttr(self):
            return _Attr(self.prim.rigid_body_enabled)

    class _Prim:
        def __init__(self, rigid_body_enabled, parent=None):
            self.rigid_body_enabled = rigid_body_enabled
            self.parent = parent

        def IsValid(self):
            return True

        def HasAPI(self, _api):
            return self.rigid_body_enabled is not None

        def GetParent(self):
            return self.parent

    usd_physics = SimpleNamespace(RigidBodyAPI=_RigidBodyAPI)
    monkeypatch.setitem(sys.modules, 'pxr', SimpleNamespace(UsdPhysics=usd_physics))
    dynamic_root = _Prim(True)
    nested_mesh = _Prim(False, parent=dynamic_root)

    assert FactoryDualFrankaAssemblyTask._prim_has_enabled_rigid_body(nested_mesh) is True


def test_release_detaches_lock_target_when_lock_pose_is_not_ready(monkeypatch):
    task = _task()
    task._attachments = {'part': {'robot_name': 'franka_right', 'mode': 'fixed_joint'}}
    task._locked_targets = {}
    detached = []
    monkeypatch.setattr(task, '_lock_ready', lambda *_args, **_kwargs: False)
    monkeypatch.setattr(task, '_detach_ready', lambda *_args, **_kwargs: True)
    monkeypatch.setattr(task, '_detach_object', lambda object_name: detached.append(object_name))

    task._process_phase_interactions(
        {
            'gripper_commands': {'franka_right': 'open'},
            'lock': [{'object': 'part', 'target': 'part_assembled'}],
            'detach': [{'object': 'part', 'release_min_steps': 0}],
        }
    )

    assert detached == ['part']


def test_release_does_not_detach_again_after_lock_wins(monkeypatch):
    task = _task()
    task._attachments = {'part': {'robot_name': 'franka_right', 'mode': 'fixed_joint'}}
    task._locked_targets = {}
    detached = []
    monkeypatch.setattr(task, '_lock_ready', lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        task,
        '_lock_object',
        lambda object_name, target_name, **_kwargs: task._locked_targets.update({object_name: target_name}),
    )
    monkeypatch.setattr(task, '_detach_ready', lambda *_args, **_kwargs: True)
    monkeypatch.setattr(task, '_detach_object', lambda object_name: detached.append(object_name))

    task._process_phase_interactions(
        {
            'gripper_commands': {'franka_right': 'open'},
            'lock': [{'object': 'part', 'target': 'part_assembled'}],
            'detach': [{'object': 'part', 'release_min_steps': 0}],
        }
    )

    assert task._locked_targets == {'part': 'part_assembled'}
    assert detached == []


def test_release_does_not_relock_an_already_locked_target(monkeypatch):
    task = _task()
    task._attachments = {}
    task._locked_targets = {'part': 'part_assembled'}
    lock_calls = []
    monkeypatch.setattr(task, '_lock_ready', lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        task,
        '_lock_object',
        lambda object_name, target_name, **_kwargs: lock_calls.append((object_name, target_name)),
    )

    task._process_phase_interactions(
        {'lock': [{'object': 'part', 'target': 'part_assembled'}]}
    )

    assert lock_calls == []


def test_lock_object_is_idempotent_for_the_same_target(monkeypatch):
    task = _task()
    task._locked_targets = {'part': 'part_assembled'}
    monkeypatch.setattr(
        task,
        '_resolve_object',
        lambda _name: pytest.fail('an already locked object must not be resolved again'),
    )

    task._lock_object('part', 'part_assembled')


def test_phase_entry_interactions_run_before_next_control_step(monkeypatch):
    task = _task()
    task.success = False
    task.failed = False
    task.policy_evaluation_mode = False
    task.phase_specs = [{'name': 'place'}, {'name': 'release_and_lock'}]
    task.phase_index = 0
    task.phase = 'place'
    task.phase_step_counter = 10
    task.step_counter = 20
    interaction_phases = []

    monkeypatch.setattr(task, '_initialize_phase', lambda: None)
    monkeypatch.setattr(task, '_sync_object_states', lambda: None)
    monkeypatch.setattr(
        task,
        '_process_phase_interactions',
        lambda phase_spec: interaction_phases.append(phase_spec['name']),
    )
    monkeypatch.setattr(task, '_advance_condition_met', lambda _phase_spec: True)

    def _set_phase(new_phase_index, **_kwargs):
        task.phase_index = new_phase_index
        task.phase = task.phase_specs[new_phase_index]['name']
        task.phase_step_counter = 0

    monkeypatch.setattr(task, '_set_phase', _set_phase)

    task._update_task_state()

    assert interaction_phases == ['place', 'release_and_lock']


def test_final_phase_timeout_is_checked_while_waiting_for_success_stability(monkeypatch):
    task = _task()
    task.success = False
    task.failed = False
    task.policy_evaluation_mode = False
    task.phase_specs = [{'name': 'release_and_lock', 'timeout_steps': 2}]
    task.phase_index = 0
    task.phase = 'release_and_lock'
    task.phase_step_counter = 2
    task.step_counter = 20
    timeout_calls = []

    monkeypatch.setattr(task, '_initialize_phase', lambda: None)
    monkeypatch.setattr(task, '_sync_object_states', lambda: None)
    monkeypatch.setattr(task, '_process_phase_interactions', lambda _phase_spec: None)
    monkeypatch.setattr(task, '_advance_condition_met', lambda _phase_spec: True)
    monkeypatch.setattr(task, '_check_success', lambda: False)

    def _handle_timeout(phase_spec):
        timeout_calls.append(phase_spec['name'])
        task.failed = True
        return True

    monkeypatch.setattr(task, '_handle_phase_timeout', _handle_timeout)

    task._update_task_state()

    assert timeout_calls == ['release_and_lock']
    assert task.failed is True


def test_clear_attachment_state_zeroes_release_velocity(monkeypatch):
    task = _task()
    task._attachments = {'part': {'robot_name': 'franka_right', 'mode': 'fixed_joint'}}
    calls = []
    monkeypatch.setattr(task, '_remove_attachment_joint', lambda object_name: calls.append(('remove', object_name)))
    monkeypatch.setattr(task, '_zero_object_velocity', lambda object_name: calls.append(('zero', object_name)))
    monkeypatch.setattr(task, '_set_object_collision', lambda object_name, enabled: calls.append(('collision', enabled)))

    task._clear_attachment_state('part')

    assert calls == [('remove', 'part'), ('zero', 'part'), ('collision', True)]


def test_pose_stability_checks_the_full_history_window(monkeypatch):
    task = _task()
    task._object_pose_history = {
        'part': deque(
            [
                (0, np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])),
                (1, np.asarray([0.012, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0, 0.0])),
                (2, np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])),
            ],
            maxlen=24,
        )
    }
    rigid_body = SimpleNamespace(
        get_linear_velocity=lambda: np.zeros(3),
        get_angular_velocity=lambda: np.zeros(3),
    )
    monkeypatch.setattr(task, '_resolve_object', lambda _name: rigid_body)

    metrics = task._object_velocity_metrics(
        'part',
        pose_stability_min_samples=3,
    )

    assert metrics['pose_stable_override'] is False
    assert np.isclose(metrics['pose_stability_position_drift'], 0.012)


def test_pose_stability_accepts_a_bounded_history_window(monkeypatch):
    task = _task()
    task._object_pose_history = {
        'part': deque(
            [
                (0, np.asarray([0.0002, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0, 0.0])),
                (1, np.asarray([-0.0003, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0, 0.0])),
                (2, np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])),
            ],
            maxlen=24,
        )
    }
    rigid_body = SimpleNamespace(
        get_linear_velocity=lambda: np.zeros(3),
        get_angular_velocity=lambda: np.zeros(3),
    )
    monkeypatch.setattr(task, '_resolve_object', lambda _name: rigid_body)

    metrics = task._object_velocity_metrics(
        'part',
        pose_stability_min_samples=3,
    )

    assert metrics['pose_stable_override'] is True
    assert metrics['is_static'] is True


def test_fixed_attachment_relaxes_to_contact_preserving_compliant_hold(monkeypatch):
    task = _task()
    task.step_counter = 42
    task._attachments = {
        'part': {
            'mode': 'fixed_joint',
            'robot_name': 'franka_left',
            'joint_path': '/World/part/assembly_attachment_joint',
            'attach_spec': {'require_dual_finger_contact': True},
            'filtered_gripper_collision_paths': ['/World/left', '/World/right'],
        }
    }
    removed = []
    filters = []
    compliant_specs = []
    monkeypatch.setattr(task, '_gripper_contact_metrics', lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        task,
        '_strict_physical_grasp_contact',
        lambda *_args, **_kwargs: {'physical_contact_ready': True},
    )
    monkeypatch.setattr(
        task,
        '_current_relative_pose',
        lambda *_args, **_kwargs: (
            np.asarray([0.01, 0.02, 0.03]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
        ),
    )
    monkeypatch.setattr(task, '_remove_attachment_joint', lambda name: removed.append(name))
    monkeypatch.setattr(
        task,
        '_create_compliant_attachment_joint',
        lambda *_args, **kwargs: compliant_specs.append(kwargs['attach_spec'])
        or '/World/part/assembly_attachment_joint',
    )
    monkeypatch.setattr(
        task,
        '_set_attachment_gripper_collision_filter',
        lambda *args, **kwargs: filters.append((args, kwargs)),
    )

    assert (
        task.relax_fixed_attachment_to_physical_hold(
            'part',
            locked_linear_world_direction=[0.0, 0.0, -2.0],
        )
        is True
    )
    state = task._attachments['part']
    assert removed == ['part']
    assert state['mode'] == 'compliant_joint'
    assert state['joint_path'] == '/World/part/assembly_attachment_joint'
    assert state['filtered_gripper_collision_paths'] == ['/World/left', '/World/right']
    assert state['relaxed_from_fixed_joint_step'] == 42
    assert state['attach_spec']['compliant_hold_locked_linear_world_direction'] == [
        0.0,
        0.0,
        -1.0,
    ]
    assert compliant_specs[0] is state['attach_spec']
    np.testing.assert_allclose(state['position'], [0.01, 0.02, 0.03])
    assert filters == []


def test_collision_disabled_transport_restores_world_collision_at_compliant_hold(monkeypatch):
    task = _task()
    task.step_counter = 42
    stored_contact = {'contact_ready': True, 'source': 'attach'}
    task._attachments = {
        'part': {
            'mode': 'fixed_joint',
            'robot_name': 'franka_left',
            'joint_path': '/World/part/assembly_attachment_joint',
            'attach_spec': {'require_dual_finger_contact': True},
            'contact_metrics': stored_contact,
            'collision_disabled': True,
            'filtered_gripper_collision_paths': ['/World/left', '/World/right'],
        }
    }
    collisions = []
    strict_inputs = []
    monkeypatch.setattr(
        task,
        '_gripper_contact_metrics',
        lambda *_args, **_kwargs: pytest.fail('disabled transport must reuse attach contact'),
    )
    monkeypatch.setattr(
        task,
        '_strict_physical_grasp_contact',
        lambda _object_name, metrics, **_kwargs: strict_inputs.append(metrics)
        or {'physical_contact_ready': True},
    )
    monkeypatch.setattr(
        task,
        '_current_relative_pose',
        lambda *_args, **_kwargs: (
            np.asarray([0.01, 0.02, 0.03]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
        ),
    )
    monkeypatch.setattr(task, '_remove_attachment_joint', lambda _name: None)
    monkeypatch.setattr(
        task,
        '_create_compliant_attachment_joint',
        lambda *_args, **_kwargs: '/World/part/assembly_compliant_attachment_joint',
    )
    monkeypatch.setattr(
        task,
        '_set_object_collision',
        lambda object_name, enabled: collisions.append((object_name, enabled)),
    )

    assert task.relax_fixed_attachment_to_physical_hold('part') is True

    assert strict_inputs == [stored_contact]
    assert collisions == [('part', True)]
    assert task._attachments['part']['collision_disabled'] is False


def test_compliant_hold_filters_gripper_collisions_when_fixed_hold_did_not(monkeypatch):
    task = _task()
    task.step_counter = 42
    task._attachments = {
        'part': {
            'mode': 'fixed_joint',
            'robot_name': 'franka_left',
            'joint_path': '/World/part/assembly_attachment_joint',
            'attach_spec': {'require_dual_finger_contact': True},
            'filtered_gripper_collision_paths': [],
        }
    }
    filters = []
    monkeypatch.setattr(task, '_gripper_contact_metrics', lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        task,
        '_strict_physical_grasp_contact',
        lambda *_args, **_kwargs: {'physical_contact_ready': True},
    )
    monkeypatch.setattr(
        task,
        '_current_relative_pose',
        lambda *_args, **_kwargs: (
            np.asarray([0.01, 0.02, 0.03]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
        ),
    )
    monkeypatch.setattr(task, '_remove_attachment_joint', lambda _name: None)
    monkeypatch.setattr(
        task,
        '_create_compliant_attachment_joint',
        lambda *_args, **_kwargs: '/World/part/assembly_attachment_joint',
    )

    def _set_filter(*args, **kwargs):
        filters.append((args, kwargs))
        return ['/World/left', '/World/right']

    monkeypatch.setattr(task, '_set_attachment_gripper_collision_filter', _set_filter)

    assert task.relax_fixed_attachment_to_physical_hold('part') is True
    assert task._attachments['part']['filtered_gripper_collision_paths'] == [
        '/World/left',
        '/World/right',
    ]
    assert filters[0][1] == {'enabled': True}


def test_compliant_hold_can_explicitly_restore_gripper_collisions(monkeypatch):
    task = _task()
    task.step_counter = 42
    task._attachments = {
        'part': {
            'mode': 'fixed_joint',
            'robot_name': 'franka_left',
            'joint_path': '/World/part/assembly_attachment_joint',
            'attach_spec': {
                'require_dual_finger_contact': True,
                'compliant_hold_filter_gripper_collisions': False,
            },
            'filtered_gripper_collision_paths': ['/World/left', '/World/right'],
        }
    }
    filters = []
    monkeypatch.setattr(task, '_gripper_contact_metrics', lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        task,
        '_strict_physical_grasp_contact',
        lambda *_args, **_kwargs: {'physical_contact_ready': True},
    )
    monkeypatch.setattr(
        task,
        '_current_relative_pose',
        lambda *_args, **_kwargs: (
            np.asarray([0.01, 0.02, 0.03]),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
        ),
    )
    monkeypatch.setattr(task, '_remove_attachment_joint', lambda _name: None)
    monkeypatch.setattr(
        task,
        '_create_compliant_attachment_joint',
        lambda *_args, **_kwargs: '/World/part/assembly_attachment_joint',
    )
    monkeypatch.setattr(
        task,
        '_set_attachment_gripper_collision_filter',
        lambda *args, **kwargs: filters.append((args, kwargs)),
    )

    assert task.relax_fixed_attachment_to_physical_hold('part') is True
    assert task._attachments['part']['filtered_gripper_collision_paths'] == []
    assert filters[0][1]['enabled'] is False


def test_insertion_compliance_waits_for_stable_object_motion(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(
        step_counter=42,
        phase='insert',
        target_poses={
            'part_assembled': {
                'position': np.zeros(3),
                'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
            }
        },
    )
    object_pose = {
        'position': np.asarray([0.0, 0.0, 0.010]),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    motion = {'valid': True, 'linear_speed': 0.2, 'angular_speed': 0.5}
    relaxed = []
    task.relax_fixed_attachment_to_physical_hold = lambda name, **_kwargs: relaxed.append(name) or True
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(adapter, '_object_motion_detail', lambda **_: dict(motion))
    spec = {
        'object': 'part',
        'target_object_final_target': 'part_assembled',
        'relax_fixed_attachment_within_final_position_tolerance': 0.015,
        'relax_fixed_attachment_final_orientation_tolerance': 0.15,
        'relax_fixed_attachment_stable_steps': 3,
        'target_object_max_linear_speed': 0.03,
        'target_object_max_angular_speed': 2.0,
    }

    assert (
        adapter._maybe_relax_insertion_attachment(
            task=task,
            spec=spec,
            tracked_objects={},
        )
        is True
    )
    assert relaxed == []

    motion.update(
        {
            'linear_speed': 0.10,
            'is_static': True,
            'pose_stable_override': True,
        }
    )
    for _ in range(2):
        assert (
            adapter._maybe_relax_insertion_attachment(
                task=task,
                spec=spec,
                tracked_objects={},
            )
            is True
        )
        assert relaxed == []
    assert (
        adapter._maybe_relax_insertion_attachment(
            task=task,
            spec=spec,
            tracked_objects={},
        )
        is False
    )
    assert relaxed == ['part']
    assert (id(task), 'part') in adapter._completed_insertion_compliance_transitions
    assert (
        adapter._maybe_relax_insertion_attachment(
            task=task,
            spec=spec,
            tracked_objects={},
        )
        is False
    )


def test_insertion_compliance_waits_for_current_waypoint_proximity(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    task = SimpleNamespace(
        step_counter=42,
        phase='insert_entry',
        phase_step_counter=10,
        target_poses={
            'part_insert_entry': {
                'position': np.zeros(3),
                'orientation': identity.copy(),
            },
            'part_assembled': {
                'position': np.asarray([0.0, 0.0, -0.10]),
                'orientation': identity.copy(),
            },
        },
    )
    object_pose = {
        'position': np.asarray([0.020, 0.0, 0.0]),
        'orientation': identity.copy(),
    }
    relaxed = []
    task.relax_fixed_attachment_to_physical_hold = lambda name, **_kwargs: relaxed.append(name) or True
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(
        adapter,
        '_object_motion_detail',
        lambda **_: {
            'valid': True,
            'linear_speed': 0.0,
            'angular_speed': 0.0,
        },
    )
    spec = {
        'object': 'part',
        'target_object_target': 'part_insert_entry',
        'target_object_final_target': 'part_assembled',
        'relax_fixed_attachment_within_final_position_tolerance': 0.12,
        'relax_fixed_attachment_require_waypoint_proximity': True,
        'relax_fixed_attachment_waypoint_position_tolerance': 0.005,
        'relax_fixed_attachment_final_orientation_tolerance': 0.15,
        'relax_fixed_attachment_stable_steps': 1,
    }

    assert (
        adapter._maybe_relax_insertion_attachment(
            task=task,
            spec=spec,
            tracked_objects={},
        )
        is False
    )
    assert relaxed == []

    object_pose['position'] = np.asarray([0.004, 0.0, 0.0])
    assert (
        adapter._maybe_relax_insertion_attachment(
            task=task,
            spec=spec,
            tracked_objects={},
        )
        is False
    )
    assert relaxed == ['part']


def test_insertion_compliance_uses_split_axial_and_lateral_capture(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    task = SimpleNamespace(
        step_counter=42,
        phase='insert_entry',
        phase_step_counter=10,
        target_poses={
            'part_insert_entry': {
                'position': np.zeros(3),
                'orientation': identity.copy(),
            },
            'part_assembled': {
                'position': np.asarray([0.0, 0.0, -0.10]),
                'orientation': identity.copy(),
            },
        },
    )
    object_pose = {
        'position': np.asarray([0.024, 0.0, 0.006]),
        'orientation': identity.copy(),
    }
    relaxed = []
    task.relax_fixed_attachment_to_physical_hold = lambda name, **_kwargs: relaxed.append(name) or True
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(
        adapter,
        '_object_motion_detail',
        lambda **_: {
            'valid': True,
            'linear_speed': 0.0,
            'angular_speed': 0.0,
        },
    )
    spec = {
        'object': 'part',
        'target_object_target': 'part_insert_entry',
        'target_object_final_target': 'part_assembled',
        'target_object_convergence_axis': [0.0, 0.0, -1.0],
        'relax_fixed_attachment_within_final_position_tolerance': 0.12,
        'relax_fixed_attachment_require_waypoint_proximity': True,
        'relax_fixed_attachment_waypoint_position_tolerance': 0.005,
        'relax_fixed_attachment_waypoint_axial_position_tolerance': 0.010,
        'relax_fixed_attachment_waypoint_lateral_position_tolerance': 0.030,
        'relax_fixed_attachment_geometric_capture_after_steps': 8,
        'relax_fixed_attachment_final_orientation_tolerance': 0.15,
        'relax_fixed_attachment_stable_steps': 1,
    }

    assert (
        adapter._maybe_relax_insertion_attachment(
            task=task,
            spec=spec,
            tracked_objects={},
        )
        is False
    )
    assert relaxed == ['part']

    adapter = UR5eAssemblyAtomicSkillAdapter({})
    relaxed.clear()
    object_pose['position'] = np.asarray([0.024, 0.0, 0.012])
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(
        adapter,
        '_object_motion_detail',
        lambda **_: {
            'valid': True,
            'linear_speed': 0.0,
            'angular_speed': 0.0,
        },
    )

    assert (
        adapter._maybe_relax_insertion_attachment(
            task=task,
            spec=spec,
            tracked_objects={},
        )
        is False
    )
    assert relaxed == []

    adapter = UR5eAssemblyAtomicSkillAdapter({})
    object_pose['position'] = np.asarray([0.004, 0.0, 0.0])
    task.phase_step_counter = 10
    spec['relax_fixed_attachment_waypoint_lateral_position_tolerance'] = 0.001
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(
        adapter,
        '_object_motion_detail',
        lambda **_: {
            'valid': True,
            'linear_speed': 0.0,
            'angular_speed': 0.0,
        },
    )

    assert (
        adapter._maybe_relax_insertion_attachment(
            task=task,
            spec=spec,
            tracked_objects={},
        )
        is False
    )
    assert relaxed == []

    spec['relax_fixed_attachment_waypoint_lateral_position_tolerance'] = 0.030
    object_pose['position'] = np.asarray([0.024, 0.0, 0.006])
    task.phase_step_counter = 7
    assert (
        adapter._maybe_relax_insertion_attachment(
            task=task,
            spec=spec,
            tracked_objects={},
        )
        is False
    )
    assert relaxed == []


def test_insertion_compliance_locks_gravity_for_horizontal_insertion(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(
        step_counter=42,
        phase='insert',
        phase_step_counter=10,
        target_poses={
            'part_assembled': {
                'position': np.zeros(3),
                'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
            }
        },
    )
    relaxed = []
    task.relax_fixed_attachment_to_physical_hold = lambda name, **kwargs: relaxed.append((name, kwargs)) or True
    monkeypatch.setattr(
        adapter,
        '_object_pose',
        lambda **_: {
            'position': np.zeros(3),
            'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
        },
    )
    monkeypatch.setattr(
        adapter,
        '_object_motion_detail',
        lambda **_: {
            'valid': True,
            'linear_speed': 0.0,
            'angular_speed': 0.0,
        },
    )

    assert (
        adapter._maybe_relax_insertion_attachment(
            task=task,
            spec={
                'object': 'part',
                'target_object_final_target': 'part_assembled',
                'target_object_convergence_axis': [0.0, 1.0, 0.0],
                'relax_fixed_attachment_minimum_gravity_alignment': 0.70,
                'relax_fixed_attachment_within_final_position_tolerance': 0.015,
                'relax_fixed_attachment_stable_steps': 1,
            },
            tracked_objects={},
        )
        is False
    )
    assert relaxed == [
        (
            'part',
            {'locked_linear_world_direction': [0.0, 0.0, -1.0]},
        )
    ]


def test_insertion_compliance_accepts_vertical_gravity_alignment(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(
        step_counter=42,
        phase='insert',
        phase_step_counter=10,
        target_poses={
            'part_assembled': {
                'position': np.zeros(3),
                'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
            }
        },
    )
    relaxed = []
    task.relax_fixed_attachment_to_physical_hold = lambda name, **kwargs: relaxed.append((name, kwargs)) or True
    monkeypatch.setattr(
        adapter,
        '_object_pose',
        lambda **_: {
            'position': np.zeros(3),
            'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
        },
    )
    monkeypatch.setattr(
        adapter,
        '_object_motion_detail',
        lambda **_: {
            'valid': True,
            'linear_speed': 0.0,
            'angular_speed': 0.0,
        },
    )

    assert (
        adapter._maybe_relax_insertion_attachment(
            task=task,
            spec={
                'object': 'part',
                'target_object_final_target': 'part_assembled',
                'target_object_convergence_axis': [0.0, 0.0, -1.0],
                'relax_fixed_attachment_minimum_gravity_alignment': 0.70,
                'relax_fixed_attachment_within_final_position_tolerance': 0.015,
                'relax_fixed_attachment_final_orientation_tolerance': 0.15,
                'relax_fixed_attachment_stable_steps': 1,
            },
            tracked_objects={},
        )
        is False
    )
    assert relaxed == [('part', {'locked_linear_world_direction': None})]


def test_compliant_joint_frame_aligns_local_z_with_world_gravity():
    object_orientation = normalize_quat([0.5, 0.5, 0.5, 0.5])
    hand_orientation = normalize_quat([0.8, -0.2, 0.4, 0.4])
    joint_world_orientation = FactoryDualFrankaAssemblyTask._quat_align_local_z([0.0, 0.0, -1.0])
    parent_frame_orientation = normalize_quat(
        quat_multiply(
            quat_conjugate(object_orientation),
            joint_world_orientation,
        )
    )
    child_frame_orientation = normalize_quat(
        quat_multiply(
            quat_conjugate(hand_orientation),
            joint_world_orientation,
        )
    )

    np.testing.assert_allclose(
        quat_rotate(
            quat_multiply(object_orientation, parent_frame_orientation),
            [0.0, 0.0, 1.0],
        ),
        [0.0, 0.0, -1.0],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        quat_rotate(
            quat_multiply(hand_orientation, child_frame_orientation),
            [0.0, 0.0, 1.0],
        ),
        [0.0, 0.0, -1.0],
        atol=1e-7,
    )


def test_compliant_joint_drive_scales_with_mass_and_grasp_lever_arm():
    parameters = FactoryDualFrankaAssemblyTask._compliant_attachment_drive_parameters(
        object_mass=1.0,
        grasp_lever_arm=0.24,
        attach_spec={
            'compliant_hold_linear_limit': 0.006,
            'compliant_hold_angular_limit_degrees': 6.0,
            'compliant_hold_gravity_force_multiplier': 6.0,
            'compliant_hold_drive_damping_ratio': 0.5,
            'compliant_hold_torque_force_fraction': 0.5,
        },
    )

    assert parameters['linear_max_force'] >= 6.0 * 9.81
    assert parameters['locked_linear_limit'] == 0.00025
    assert parameters['linear_stiffness'] * 2.0 * 0.006 >= parameters['linear_max_force']
    assert parameters['linear_damping'] > 10.0
    assert parameters['angular_max_force'] >= (parameters['linear_max_force'] * 0.24 * 0.5)
    assert parameters['angular_stiffness'] > 5.0
    assert parameters['angular_damping'] > 0.2

    capped = FactoryDualFrankaAssemblyTask._compliant_attachment_drive_parameters(
        object_mass=100.0,
        grasp_lever_arm=1.0,
        attach_spec={},
    )
    assert capped['linear_max_force'] == 120.0
    assert capped['angular_max_force'] == 12.0


def test_compliant_servo_pauses_on_dynamic_spike_then_resumes(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(step_counter=10, phase='insert')
    motion = {'valid': True, 'linear_speed': 0.3, 'angular_speed': 1.0}
    monkeypatch.setattr(adapter, '_object_motion_detail', lambda **_: dict(motion))
    tracked_objects = {
        'part': {
            'attachment': {
                'mode': 'compliant_joint',
                'attach_spec': {},
            }
        }
    }
    spec = {
        'object': 'part',
        'compliant_servo_pause_linear_speed': 0.15,
        'compliant_servo_pause_angular_speed': 5.0,
        'compliant_servo_resume_linear_speed': 0.03,
        'compliant_servo_resume_angular_speed': 2.0,
        'compliant_servo_resume_stable_steps': 3,
    }
    phase_key = ('insert',)
    adapter._cartesian_command_positions[phase_key] = np.ones(3)
    adapter._cartesian_command_orientations[phase_key] = np.asarray([1.0, 0.0, 0.0, 0.0])

    assert (
        adapter._compliant_motion_requires_hold(
            phase_key=phase_key,
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        is True
    )
    assert phase_key not in adapter._cartesian_command_positions
    assert phase_key not in adapter._cartesian_command_orientations
    motion.update(
        {
            'linear_speed': 0.30,
            'angular_speed': 10.0,
            'is_static': True,
            'pose_stable_override': True,
        }
    )
    for _ in range(2):
        assert (
            adapter._compliant_motion_requires_hold(
                task=task,
                spec=spec,
                tracked_objects=tracked_objects,
            )
            is True
        )
    assert (
        adapter._compliant_motion_requires_hold(
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        is True
    )
    motion.update({'linear_speed': 0.02, 'angular_speed': 1.0})
    for _ in range(2):
        assert (
            adapter._compliant_motion_requires_hold(
                task=task,
                spec=spec,
                tracked_objects=tracked_objects,
            )
            is True
        )
    assert (
        adapter._compliant_motion_requires_hold(
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        is False
    )


def test_compliant_servo_rejects_stale_velocity_when_pose_history_is_static(
    monkeypatch,
):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(step_counter=10, phase='insert')
    motion = {
        'valid': True,
        'linear_speed': 0.217,
        'angular_speed': 2.98,
        'is_static': True,
        'pose_stable_override': True,
        'pose_stability_position_drift': 3.4e-6,
        'pose_stability_orientation_drift': 7.8e-5,
    }
    monkeypatch.setattr(adapter, '_object_motion_detail', lambda **_: dict(motion))
    tracked_objects = {
        'part': {
            'attachment': {
                'mode': 'compliant_joint',
                'attach_spec': {},
            }
        }
    }
    spec = {
        'object': 'part',
        'compliant_servo_pause_linear_speed': 0.20,
        'compliant_servo_pause_angular_speed': 5.0,
        'target_object_allow_pose_history_velocity_override': True,
        'pose_history_velocity_override_position_tolerance': 0.0005,
        'pose_history_velocity_override_orientation_tolerance': 0.01,
    }

    assert (
        adapter._compliant_motion_requires_hold(
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        is False
    )

    motion['pose_stability_position_drift'] = 0.003
    assert (
        adapter._compliant_motion_requires_hold(
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        is True
    )


def test_compliant_recovery_allows_bounded_target_settle_without_servo_resume(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(step_counter=10, phase='insert')
    motion = {
        'valid': True,
        'linear_speed': 0.3,
        'angular_speed': 1.0,
        'pose_stable_override': False,
    }
    monkeypatch.setattr(adapter, '_object_motion_detail', lambda **_: dict(motion))
    tracked_objects = {
        'part': {
            'attachment': {
                'mode': 'compliant_joint',
                'attach_spec': {},
            }
        }
    }
    spec = {
        'object': 'part',
        'compliant_servo_pause_linear_speed': 0.15,
        'compliant_servo_pause_angular_speed': 5.0,
        'compliant_servo_resume_linear_speed': 0.03,
        'compliant_servo_resume_angular_speed': 2.0,
        'compliant_servo_resume_stable_steps': 2,
        'compliant_servo_settle_max_linear_speed': 0.10,
        'compliant_servo_settle_max_angular_speed': 2.0,
        'compliant_servo_settle_stable_steps': 2,
    }

    assert (
        adapter._compliant_motion_requires_hold(
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        is True
    )
    assert (
        adapter._compliant_recovery_allows_target_settle(
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        is False
    )

    motion.update(
        {
            'linear_speed': 0.07,
            'angular_speed': 0.1,
            'pose_stable_override': True,
        }
    )
    assert (
        adapter._compliant_recovery_allows_target_settle(
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        is False
    )
    assert (
        adapter._compliant_recovery_allows_target_settle(
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        is True
    )
    assert (
        adapter._compliant_motion_requires_hold(
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        is True
    )

    motion.update({'angular_speed': 3.0})
    assert (
        adapter._compliant_recovery_allows_target_settle(
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        is False
    )


def test_compliant_recovery_retries_servo_after_bounded_target_settle(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    task = SimpleNamespace(step_counter=10, phase='insert')
    motion = {
        'valid': True,
        'linear_speed': 0.3,
        'angular_speed': 1.0,
        'pose_stable_override': False,
    }
    monkeypatch.setattr(adapter, '_object_motion_detail', lambda **_: dict(motion))
    tracked_objects = {
        'part': {
            'attachment': {
                'mode': 'compliant_joint',
                'attach_spec': {},
            }
        }
    }
    spec = {
        'object': 'part',
        'compliant_servo_pause_linear_speed': 0.15,
        'compliant_servo_pause_angular_speed': 5.0,
        'compliant_servo_resume_linear_speed': 0.03,
        'compliant_servo_resume_angular_speed': 2.0,
        'compliant_servo_settle_max_linear_speed': 0.10,
        'compliant_servo_settle_max_angular_speed': 2.0,
        'compliant_servo_settle_stable_steps': 1,
        'compliant_recovery_target_settle_max_steps': 2,
    }

    assert (
        adapter._compliant_motion_requires_hold(
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        is True
    )
    motion.update(
        {
            'linear_speed': 0.07,
            'angular_speed': 0.1,
            'pose_stable_override': True,
        }
    )
    assert (
        adapter._compliant_recovery_allows_target_settle(
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        is True
    )
    assert (
        adapter._compliant_recovery_allows_target_settle(
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        is True
    )
    assert (
        adapter._compliant_recovery_allows_target_settle(
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        is False
    )
    assert (id(task), 'part') not in adapter._compliant_motion_recovery_state
    assert (
        adapter._compliant_motion_requires_hold(
            task=task,
            spec=spec,
            tracked_objects=tracked_objects,
        )
        is False
    )


def test_compliant_servo_rate_limit_scales_with_object_speed():
    scale = UR5eAssemblyAtomicSkillAdapter._compliant_servo_step_scale(
        spec={
            'compliant_servo_pause_linear_speed': 0.15,
            'compliant_servo_pause_angular_speed': 5.0,
            'compliant_servo_minimum_step_scale': 0.1,
        },
        motion_detail={
            'valid': True,
            'linear_speed': 0.12,
            'angular_speed': 1.0,
        },
    )

    assert np.isclose(scale, 0.2)


def test_insertion_waits_for_stable_lateral_alignment(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    current_pose = {
        'position': np.zeros(3),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    target_pose = {
        'position': np.asarray([-0.001, 0.0, -0.1]),
        'orientation': current_pose['orientation'].copy(),
    }
    object_pose = {
        'position': np.zeros(3),
        'orientation': current_pose['orientation'].copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(adapter, '_target_object_pose', lambda **_: target_pose)
    phase_key = ('stable-lateral',)
    spec = {
        'object': 'part',
        'target_object_target': 'target',
        'target_object_convergence_axis': [0.0, 0.0, -1.0],
        'target_object_lateral_position_tolerance': 0.002,
        'target_object_lateral_alignment_stable_steps': 3,
        'target_object_lateral_alignment_cartesian_position_step': 0.001,
        'cartesian_position_step': 0.00025,
        'cartesian_orientation_step': 0.1,
    }

    for _ in range(2):
        command_pose = adapter._target_object_servo_pose(
            phase_key=phase_key,
            task=SimpleNamespace(),
            robot_name='franka_left',
            spec=spec,
            tracked_robots={},
            tracked_objects={},
            current_pose=current_pose,
            target_pose=target_pose,
        )
        np.testing.assert_allclose(command_pose['position'][2], 0.0)

    command_pose = adapter._target_object_servo_pose(
        phase_key=phase_key,
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec=spec,
        tracked_robots={},
        tracked_objects={},
        current_pose=current_pose,
        target_pose=target_pose,
    )
    assert command_pose['position'][2] < 0.0


def test_lock_rebases_assembly_targets_to_the_observed_base_pose(monkeypatch):
    nominal_position = np.asarray([0.7, -0.15, 1.0025])
    nominal_orientation = np.asarray([2**-0.5, 0.0, 0.0, 2**-0.5])
    observed_position = np.asarray([0.69, -0.16, 1.01])
    observed_orientation = np.asarray([0.0, 0.0, 0.0, 1.0])
    local_position = np.asarray([0.03, -0.02, 0.12])
    local_orientation = np.asarray([1.0, 0.0, 0.0, 0.0])
    waypoint_position, waypoint_orientation = compose_pose(
        nominal_position,
        nominal_orientation,
        local_position,
        local_orientation,
    )
    expected_position, expected_orientation = compose_pose(
        observed_position,
        observed_orientation,
        local_position,
        local_orientation,
    )

    class _Body:
        @staticmethod
        def get_pose():
            return observed_position.copy(), observed_orientation.copy()

    task = _task()
    task.target_poses = {
        'base_target': {
            'position': nominal_position.copy(),
            'orientation': nominal_orientation.copy(),
        },
        'insertion_waypoint': {
            'position': waypoint_position.copy(),
            'orientation': waypoint_orientation.copy(),
        },
    }
    task._attachments = {}
    task._locked_targets = {}
    task._frozen_lock_poses = {}
    monkeypatch.setattr(task, '_resolve_object', lambda _: _Body())
    monkeypatch.setattr(task, '_clear_attachment_state', lambda *_, **__: None)
    monkeypatch.setattr(task, '_set_object_pose', lambda *_, **__: None)
    monkeypatch.setattr(task, '_set_object_collision', lambda *_, **__: None)

    task._lock_object(
        'base',
        'base_target',
        lock_spec={
            'freeze_current_pose': True,
            'rebase_targets': ['insertion_waypoint'],
        },
    )

    np.testing.assert_allclose(task.target_poses['base_target']['position'], observed_position)
    np.testing.assert_allclose(
        task.target_poses['base_target']['orientation'],
        observed_orientation,
    )
    np.testing.assert_allclose(
        task.target_poses['insertion_waypoint']['position'],
        expected_position,
    )
    np.testing.assert_allclose(
        task.target_poses['insertion_waypoint']['orientation'],
        expected_orientation,
    )


def test_transport_completion_requires_lateral_insertion_alignment(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    completed = []
    task = SimpleNamespace(phase_step_counter=700)
    target_pose = {
        'position': np.zeros(3),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    current_pose = {
        'position': np.zeros(3),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    object_pose = {
        'position': np.asarray([0.004, 0.0, 0.006]),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(adapter, '_target_object_pose', lambda **_: target_pose)
    monkeypatch.setattr(adapter, '_mark_complete', lambda **kwargs: completed.append(kwargs))
    spec = {
        'object': 'part',
        'require_target_object_pose_convergence': True,
        'position_tolerance': 0.015,
        'target_object_position_tolerance': 0.015,
        'target_object_convergence_axis': [0.0, 0.0, 1.0],
        'target_object_lateral_position_tolerance': 0.003,
    }

    adapter._maybe_mark_complete(
        phase_key=('insert',),
        task=task,
        robot_name='franka_left',
        skill_name='ur5e_move_part_to_staging',
        spec=spec,
        target_pose=target_pose,
        ik_target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects={},
        current_q=None,
        target_q=None,
    )
    assert completed == []

    object_pose['position'] = np.asarray([0.002, 0.0, 0.010])
    adapter._maybe_mark_complete(
        phase_key=('insert',),
        task=task,
        robot_name='franka_left',
        skill_name='ur5e_move_part_to_staging',
        spec=spec,
        target_pose=target_pose,
        ik_target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects={},
        current_q=None,
        target_q=None,
    )
    assert len(completed) == 1


def test_compliant_hold_completion_ignores_tcp_pose_but_requires_object_pose(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    completed = []
    task = SimpleNamespace(phase_step_counter=10)
    target_pose = {
        'position': np.zeros(3),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    current_pose = {
        # A 6 degree compliant rotation over a long grasp lever arm can move the
        # TCP well beyond the nominal Cartesian tolerance while the part is seated.
        'position': np.asarray([0.030, 0.0, 0.0]),
        'orientation': target_pose['orientation'].copy(),
    }
    object_pose = {
        'position': np.asarray([0.001, 0.0, 0.0]),
        'orientation': target_pose['orientation'].copy(),
    }
    tracked_objects = {
        'part': {
            'attachment': {
                'mode': 'compliant_joint',
                'attach_spec': {'compliant_hold_linear_limit': 0.006},
            }
        }
    }
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(adapter, '_target_object_pose', lambda **_: target_pose)
    monkeypatch.setattr(adapter, '_mark_complete', lambda **kwargs: completed.append(kwargs))
    spec = {
        'object': 'part',
        'require_target_object_pose_convergence': True,
        'position_tolerance': 0.006,
        'target_object_position_tolerance': 0.008,
        'orientation_tolerance': 0.1,
        'target_object_orientation_tolerance': 0.1,
    }

    adapter._maybe_mark_complete(
        phase_key=('compliant-insert',),
        task=task,
        robot_name='franka_left',
        skill_name='ur5e_move_part_to_staging',
        spec=spec,
        target_pose=target_pose,
        ik_target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects=tracked_objects,
        current_q=None,
        target_q=None,
    )
    assert len(completed) == 1
    assert completed[0]['detail']['tcp_pose_required_for_completion'] is False
    assert completed[0]['detail']['position_error'] > completed[0]['detail']['position_tolerance']

    completed.clear()
    object_pose['position'][0] = 0.009
    adapter._maybe_mark_complete(
        phase_key=('compliant-insert-object-outside',),
        task=task,
        robot_name='franka_left',
        skill_name='ur5e_move_part_to_staging',
        spec=spec,
        target_pose=target_pose,
        ik_target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects=tracked_objects,
        current_q=None,
        target_q=None,
    )
    assert completed == []


def test_insertion_completion_splits_axial_contact_and_lateral_capture_tolerances(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    completed = []
    task = SimpleNamespace(phase_step_counter=700)
    target_pose = {
        'position': np.zeros(3),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    current_pose = {
        'position': np.zeros(3),
        'orientation': target_pose['orientation'].copy(),
    }
    object_pose = {
        'position': np.asarray([0.005, 0.0, 0.020]),
        'orientation': target_pose['orientation'].copy(),
    }
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(adapter, '_target_object_pose', lambda **_: target_pose)
    monkeypatch.setattr(adapter, '_mark_complete', lambda **kwargs: completed.append(kwargs))
    spec = {
        'object': 'part',
        'require_target_object_pose_convergence': True,
        'position_tolerance': 0.015,
        'target_object_position_tolerance': 0.015,
        'target_object_convergence_axis': [0.0, 0.0, 1.0],
        'target_object_axial_position_tolerance': 0.025,
        'target_object_lateral_position_tolerance': 0.006,
    }

    adapter._maybe_mark_complete(
        phase_key=('split-insertion-tolerance',),
        task=task,
        robot_name='franka_left',
        skill_name='ur5e_move_part_to_staging',
        spec=spec,
        target_pose=target_pose,
        ik_target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects={},
        current_q=None,
        target_q=None,
    )
    assert len(completed) == 1
    assert completed[0]['detail']['object_position_error'] > 0.015
    assert completed[0]['detail']['object_axial_position_error'] == pytest.approx(0.020)
    assert completed[0]['detail']['target_object_axial_position_tolerance'] == pytest.approx(0.025)

    completed.clear()
    object_pose['position'] = np.asarray([0.007, 0.0, 0.020])
    adapter._maybe_mark_complete(
        phase_key=('split-insertion-lateral-failure',),
        task=task,
        robot_name='franka_left',
        skill_name='ur5e_move_part_to_staging',
        spec=spec,
        target_pose=target_pose,
        ik_target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects={},
        current_q=None,
        target_q=None,
    )
    assert completed == []

    object_pose['position'] = np.asarray([0.005, 0.0, 0.026])
    adapter._maybe_mark_complete(
        phase_key=('split-insertion-axial-failure',),
        task=task,
        robot_name='franka_left',
        skill_name='ur5e_move_part_to_staging',
        spec=spec,
        target_pose=target_pose,
        ik_target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects={},
        current_q=None,
        target_q=None,
    )
    assert completed == []


def test_target_object_completion_uses_declared_lateral_tolerance(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    completed = []
    phase_key = ('alignment-gated-completion',)
    task = SimpleNamespace(phase_step_counter=10)
    target_pose = {
        'position': np.zeros(3),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    current_pose = {
        'position': np.asarray([0.030, 0.0, 0.0]),
        'orientation': target_pose['orientation'].copy(),
    }
    object_pose = {
        'position': np.asarray([0.0004, 0.0, 0.0]),
        'orientation': target_pose['orientation'].copy(),
    }
    tracked_objects = {'part': {'attachment': {'mode': 'compliant_joint'}}}
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(adapter, '_target_object_pose', lambda **_: target_pose)
    monkeypatch.setattr(
        adapter,
        '_mark_complete',
        lambda **kwargs: completed.append(kwargs),
    )
    spec = {
        'object': 'part',
        'require_target_object_pose_convergence': True,
        'position_tolerance': 0.006,
        'target_object_position_tolerance': 0.008,
        'orientation_tolerance': 0.1,
        'target_object_orientation_tolerance': 0.1,
        'target_object_convergence_axis': [0.0, 0.0, -1.0],
        'target_object_lateral_position_tolerance': 0.001,
    }
    adapter._insertion_lateral_alignment_active[phase_key] = True

    adapter._maybe_mark_complete(
        phase_key=phase_key,
        task=task,
        robot_name='franka_left',
        skill_name='ur5e_move_part_to_staging',
        spec=spec,
        target_pose=target_pose,
        ik_target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects=tracked_objects,
        current_q=None,
        target_q=None,
    )
    assert len(completed) == 1
    assert completed[0]['detail']['target_object_lateral_alignment_complete'] is False


def test_transport_servo_aligns_laterally_before_advancing_insertion_depth(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    current_pose = {
        'position': np.zeros(3),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    target_pose = {
        'position': np.asarray([-0.010, 0.0, -0.100]),
        'orientation': np.asarray([2**-0.5, 0.0, 2**-0.5, 0.0]),
    }
    object_pose = {
        'position': np.zeros(3),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(adapter, '_target_object_pose', lambda **_: target_pose)

    command_pose = adapter._target_object_servo_pose(
        phase_key=('insert',),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'target_object_convergence_axis': [0.0, 0.0, -1.0],
            'target_object_lateral_position_tolerance': 0.001,
            'cartesian_position_step': 0.00025,
            'target_object_lateral_alignment_cartesian_position_step': 0.002,
            'cartesian_orientation_step': 0.1,
        },
        tracked_robots={},
        tracked_objects={},
        current_pose=current_pose,
        target_pose=target_pose,
    )

    np.testing.assert_allclose(command_pose['position'], [-0.002, 0.0, 0.0])
    assert not np.allclose(command_pose['orientation'], current_pose['orientation'])

    object_pose['position'] = np.asarray([0.0, 0.0, -0.010])
    current_pose['position'] = object_pose['position'].copy()
    command_pose = adapter._target_object_servo_pose(
        phase_key=('insert',),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'target_object_convergence_axis': [0.0, 0.0, -1.0],
            'target_object_lateral_position_tolerance': 0.001,
            'cartesian_position_step': 0.00025,
            'target_object_lateral_alignment_cartesian_position_step': 0.002,
            'cartesian_orientation_step': 0.1,
        },
        tracked_robots={},
        tracked_objects={},
        current_pose=current_pose,
        target_pose=target_pose,
    )

    assert command_pose['position'][2] > current_pose['position'][2]

    object_pose['position'] = np.zeros(3)
    current_pose['position'] = np.zeros(3)
    target_pose['position'] = np.asarray([-0.0015, 0.0, -0.100])
    command_pose = adapter._target_object_servo_pose(
        phase_key=('insert',),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'target_object_convergence_axis': [0.0, 0.0, -1.0],
            'target_object_lateral_position_tolerance': 0.001,
            'cartesian_position_step': 0.00025,
            'target_object_lateral_alignment_cartesian_position_step': 0.002,
            'cartesian_orientation_step': 0.1,
        },
        tracked_robots={},
        tracked_objects={},
        current_pose=current_pose,
        target_pose=target_pose,
    )

    np.testing.assert_allclose(command_pose['position'], [-0.0015, 0.0, 0.0])

    target_pose['position'] = np.asarray([-0.010, 0.0, -0.100])
    object_pose['position'] = np.asarray([-0.0095, 0.0, 0.0])
    current_pose['position'] = object_pose['position'].copy()
    command_pose = adapter._target_object_servo_pose(
        phase_key=('insert',),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'target_object_convergence_axis': [0.0, 0.0, -1.0],
            'target_object_lateral_position_tolerance': 0.001,
            'cartesian_position_step': 0.00025,
            'target_object_lateral_alignment_cartesian_position_step': 0.002,
            'cartesian_orientation_step': 0.1,
        },
        tracked_robots={},
        tracked_objects={},
        current_pose=current_pose,
        target_pose=target_pose,
    )

    np.testing.assert_allclose(
        np.linalg.norm(command_pose['position'] - current_pose['position']),
        0.00025,
    )
    assert command_pose['position'][2] < current_pose['position'][2]
    assert not np.allclose(command_pose['orientation'], current_pose['orientation'])


def test_compliant_hold_lateral_servo_moves_past_nominal_tcp(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    current_pose = {
        'position': np.zeros(3),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    target_pose = {
        'position': np.zeros(3),
        'orientation': current_pose['orientation'].copy(),
    }
    object_pose = {
        'position': np.zeros(3),
        'orientation': current_pose['orientation'].copy(),
    }
    target_object_pose = {
        'position': np.asarray([0.006, 0.0, 0.0]),
        'orientation': current_pose['orientation'].copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: target_object_pose,
    )

    command_pose = adapter._target_object_servo_pose(
        phase_key=('compliant-lateral',),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'target_object_convergence_axis': [0.0, 0.0, -1.0],
            'target_object_lateral_position_tolerance': 0.002,
            'target_object_lateral_alignment_cartesian_position_step': 0.002,
            'cartesian_position_step': 0.00025,
            'cartesian_orientation_step': 0.1,
        },
        tracked_robots={},
        tracked_objects={'part': {'attachment': {'mode': 'compliant_joint'}}},
        current_pose=current_pose,
        target_pose=target_pose,
    )

    np.testing.assert_allclose(command_pose['position'], [0.0005, 0.0, 0.0])


def test_compliant_hold_servo_corrects_object_orientation_past_nominal_tcp(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    object_orientation = np.asarray([np.cos(0.05), 0.0, 0.0, np.sin(0.05)])
    current_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    target_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    object_pose = {
        'position': np.zeros(3),
        'orientation': object_orientation,
    }
    target_object_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: target_object_pose,
    )

    command_pose = adapter._target_object_servo_pose(
        phase_key=('compliant-orientation',),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'cartesian_position_step': 0.00025,
            'cartesian_orientation_step': 0.2,
            'compliant_servo_track_object_orientation': True,
            'target_object_orientation_tolerance': 0.05,
        },
        tracked_robots={},
        tracked_objects={'part': {'attachment': {'mode': 'compliant_joint'}}},
        current_pose=current_pose,
        target_pose=target_pose,
    )

    expected = np.asarray([np.cos(0.001), 0.0, 0.0, -np.sin(0.001)])
    np.testing.assert_allclose(command_pose['orientation'], expected)


def test_compliant_hold_servo_leaves_orientation_within_target_tolerance(
    monkeypatch,
):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    object_orientation = np.asarray([np.cos(0.02), 0.0, 0.0, np.sin(0.02)])
    tcp_orientation = np.asarray([np.cos(0.015), 0.0, 0.0, np.sin(0.015)])
    current_pose = {
        'position': np.zeros(3),
        'orientation': tcp_orientation,
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(
        adapter,
        '_object_pose',
        lambda **_: {
            'position': np.zeros(3),
            'orientation': object_orientation,
        },
    )
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: {
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )

    command_pose = adapter._target_object_servo_pose(
        phase_key=('compliant-orientation-within-tolerance',),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'cartesian_position_step': 0.00025,
            'cartesian_orientation_step': 0.2,
            'compliant_servo_track_object_orientation': True,
            'compliant_servo_orientation_correction_deadband': 0.005,
            'target_object_orientation_tolerance': 0.1,
        },
        tracked_robots={},
        tracked_objects={'part': {'attachment': {'mode': 'compliant_joint'}}},
        current_pose=current_pose,
        target_pose={
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )

    assert abs(float(np.dot(command_pose['orientation'], identity))) > abs(
        float(np.dot(current_pose['orientation'], identity))
    )


def test_compliant_hold_servo_keeps_nominal_tcp_orientation_by_default(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    object_orientation = np.asarray([np.cos(0.05), 0.0, 0.0, np.sin(0.05)])
    current_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    object_pose = {
        'position': np.zeros(3),
        'orientation': object_orientation,
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: {
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )

    command_pose = adapter._target_object_servo_pose(
        phase_key=('compliant-nominal-orientation',),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'cartesian_position_step': 0.00025,
            'cartesian_orientation_step': 0.2,
        },
        tracked_robots={},
        tracked_objects={'part': {'attachment': {'mode': 'compliant_joint'}}},
        current_pose=current_pose,
        target_pose={
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )

    np.testing.assert_allclose(command_pose['orientation'], identity)


def test_compliant_lateral_servo_compensates_long_grasp_lever(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    current_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    object_pose = {
        'position': np.asarray([0.0, 0.0, -0.24]),
        'orientation': np.asarray([np.cos(0.05), 0.0, np.sin(0.05), 0.0]),
    }
    target_object_pose = {
        'position': np.asarray([0.006, 0.0, -0.24]),
        'orientation': identity.copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: target_object_pose,
    )

    command_pose = adapter._target_object_servo_pose(
        phase_key=('compliant-lever',),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'target_object_convergence_axis': [0.0, 0.0, -1.0],
            'target_object_lateral_position_tolerance': 0.002,
            'target_object_lateral_alignment_cartesian_position_step': 0.002,
            'cartesian_position_step': 0.00025,
            'cartesian_orientation_step': 0.1,
        },
        tracked_robots={},
        tracked_objects={'part': {'attachment': {'mode': 'compliant_joint'}}},
        current_pose=current_pose,
        target_pose={
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )

    command_orientation = command_pose['orientation']
    relative_position = np.asarray([0.0, 0.0, -0.24])
    commanded_object_position = command_pose['position'] + quat_rotate(
        command_orientation,
        relative_position,
    )
    np.testing.assert_allclose(commanded_object_position, [0.0005, 0.0, -0.24])


def test_compliant_lateral_servo_defers_long_lever_orientation_change(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    current_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    object_pose = {
        'position': np.asarray([0.0, 0.0, -0.24]),
        'orientation': identity.copy(),
    }
    target_object_pose = {
        'position': np.asarray([0.006, 0.0, -0.34]),
        'orientation': identity.copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: target_object_pose,
    )

    command_pose = adapter._target_object_servo_pose(
        phase_key=('compliant-lateral-orientation-gate',),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'target_object_convergence_axis': [0.0, 0.0, -1.0],
            'target_object_lateral_position_tolerance': 0.002,
            'target_object_lateral_alignment_cartesian_position_step': 0.002,
            'cartesian_position_step': 0.00025,
            'cartesian_orientation_step': 0.1,
        },
        tracked_robots={},
        tracked_objects={'part': {'attachment': {'mode': 'compliant_joint'}}},
        current_pose=current_pose,
        target_pose={
            'position': np.zeros(3),
            'orientation': np.asarray([2**-0.5, 0.0, 2**-0.5, 0.0]),
        },
    )

    np.testing.assert_allclose(command_pose['position'], [0.0005, 0.0, 0.0])
    np.testing.assert_allclose(command_pose['orientation'], identity)


def test_compliant_lateral_servo_holds_entry_orientation_anchor(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    current_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    object_pose = {
        'position': np.asarray([0.0, 0.0, -0.24]),
        'orientation': identity.copy(),
    }
    target_object_pose = {
        'position': np.asarray([0.006, 0.0, -0.24]),
        'orientation': identity.copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: target_object_pose,
    )
    phase_key = ('compliant-lateral-orientation-anchor',)
    spec = {
        'object': 'part',
        'target_object_target': 'target',
        'target_object_convergence_axis': [0.0, 0.0, -1.0],
        'target_object_lateral_position_tolerance': 0.002,
        'target_object_lateral_alignment_cartesian_position_step': 0.002,
        'cartesian_position_step': 0.00025,
        'cartesian_orientation_step': 0.1,
    }
    tracked_objects = {'part': {'attachment': {'mode': 'compliant_joint'}}}

    adapter._target_object_servo_pose(
        phase_key=phase_key,
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec=spec,
        tracked_robots={},
        tracked_objects=tracked_objects,
        current_pose=current_pose,
        target_pose=current_pose,
    )
    drifted_orientation = np.asarray([np.cos(0.01), 0.0, np.sin(0.01), 0.0])
    current_pose['orientation'] = drifted_orientation
    command_pose = adapter._target_object_servo_pose(
        phase_key=phase_key,
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec=spec,
        tracked_robots={},
        tracked_objects=tracked_objects,
        current_pose=current_pose,
        target_pose=current_pose,
    )

    assert abs(float(np.dot(command_pose['orientation'], identity))) > abs(float(np.dot(drifted_orientation, identity)))
    np.testing.assert_allclose(
        adapter._insertion_lateral_orientation_anchors[phase_key],
        identity,
    )


def test_compliant_lateral_servo_ignores_sub_deadband_orientation_noise(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    current_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    object_pose = {
        'position': np.asarray([0.0, 0.0, -0.24]),
        'orientation': np.asarray([np.cos(0.0005), 0.0, 0.0, np.sin(0.0005)]),
    }
    target_object_pose = {
        'position': np.asarray([0.006, 0.0, -0.24]),
        'orientation': identity.copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: target_object_pose,
    )

    command_pose = adapter._target_object_servo_pose(
        phase_key=('compliant-orientation-deadband',),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'target_object_convergence_axis': [0.0, 0.0, -1.0],
            'target_object_lateral_position_tolerance': 0.002,
            'target_object_lateral_alignment_cartesian_position_step': 0.002,
            'compliant_servo_orientation_correction_deadband': 0.005,
            'cartesian_position_step': 0.00025,
            'cartesian_orientation_step': 0.1,
        },
        tracked_robots={},
        tracked_objects={'part': {'attachment': {'mode': 'compliant_joint'}}},
        current_pose=current_pose,
        target_pose={
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )

    np.testing.assert_allclose(command_pose['position'], [0.0005, 0.0, 0.0])
    np.testing.assert_allclose(command_pose['orientation'], identity)


def test_compliant_lateral_gate_allows_only_outward_axial_recovery(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    current_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    object_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    target_object_pose = {
        'position': np.asarray([0.006, 0.0, 0.010]),
        'orientation': identity.copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: target_object_pose,
    )

    command_pose = adapter._target_object_servo_pose(
        phase_key=('compliant-axial-recovery',),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'target_object_convergence_axis': [0.0, 0.0, -1.0],
            'target_object_lateral_position_tolerance': 0.002,
            'target_object_lateral_alignment_cartesian_position_step': 0.002,
            'target_object_axial_recovery_cartesian_position_step': 0.001,
            'target_object_axial_recovery_deadband': 0.0005,
            'cartesian_position_step': 0.00025,
            'cartesian_orientation_step': 0.1,
        },
        tracked_robots={},
        tracked_objects={'part': {'attachment': {'mode': 'compliant_joint'}}},
        current_pose=current_pose,
        target_pose={
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )

    np.testing.assert_allclose(command_pose['position'], [0.0005, 0.0, 0.001])


def test_compliant_lateral_gate_retracts_to_alignment_clearance(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    current_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    object_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    target_object_pose = {
        'position': np.asarray([0.006, 0.0, -0.010]),
        'orientation': identity.copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: target_object_pose,
    )

    command_pose = adapter._target_object_servo_pose(
        phase_key=('compliant-alignment-clearance',),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'target_object_convergence_axis': [0.0, 0.0, -1.0],
            'target_object_lateral_position_tolerance': 0.002,
            'target_object_lateral_alignment_cartesian_position_step': 0.002,
            'target_object_lateral_alignment_axial_clearance': 0.020,
            'target_object_axial_recovery_cartesian_position_step': 0.001,
            'target_object_axial_recovery_deadband': 0.0005,
            'cartesian_position_step': 0.00025,
            'cartesian_orientation_step': 0.1,
        },
        tracked_robots={},
        tracked_objects={'part': {'attachment': {'mode': 'compliant_joint'}}},
        current_pose=current_pose,
        target_pose={
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )

    np.testing.assert_allclose(command_pose['position'], [0.0005, 0.0, 0.001])

    target_object_pose['position'] = np.asarray([0.001, 0.0, -0.010])
    command_pose = adapter._target_object_servo_pose(
        phase_key=('compliant-alignment-clearance',),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'target_object_convergence_axis': [0.0, 0.0, -1.0],
            'target_object_lateral_position_tolerance': 0.002,
            'target_object_lateral_alignment_cartesian_position_step': 0.002,
            'target_object_lateral_alignment_axial_clearance': 0.020,
            'target_object_axial_recovery_cartesian_position_step': 0.001,
            'target_object_axial_recovery_deadband': 0.0005,
            'cartesian_position_step': 0.00025,
            'cartesian_orientation_step': 0.1,
        },
        tracked_robots={},
        tracked_objects={'part': {'attachment': {'mode': 'compliant_joint'}}},
        current_pose=current_pose,
        target_pose={
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )

    np.testing.assert_allclose(command_pose['position'], [0.0005, 0.0, 0.001])
    assert adapter._insertion_lateral_clearance_required[('compliant-alignment-clearance',)] is True
    assert (
        adapter._insertion_lateral_alignment_stable_steps.get(
            ('compliant-alignment-clearance',),
            0,
        )
        == 0
    )


def test_compliant_lateral_hysteresis_band_does_not_latch_clearance(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    current_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    object_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    target_object_pose = {
        'position': np.asarray([0.0015, 0.0, -0.010]),
        'orientation': identity.copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: target_object_pose,
    )
    phase_key = ('compliant-hysteresis-band',)

    command_pose = adapter._target_object_servo_pose(
        phase_key=phase_key,
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'target_object_convergence_axis': [0.0, 0.0, -1.0],
            'target_object_lateral_position_tolerance': 0.002,
            'target_object_lateral_alignment_enter_tolerance': 0.001,
            'target_object_lateral_alignment_exit_tolerance': 0.002,
            'target_object_lateral_alignment_cartesian_position_step': 0.002,
            'target_object_lateral_alignment_axial_clearance': 0.020,
            'target_object_axial_recovery_cartesian_position_step': 0.001,
            'target_object_axial_recovery_deadband': 0.0005,
            'cartesian_position_step': 0.00025,
            'cartesian_orientation_step': 0.1,
        },
        tracked_robots={},
        tracked_objects={'part': {'attachment': {'mode': 'compliant_joint'}}},
        current_pose=current_pose,
        target_pose={
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )

    np.testing.assert_allclose(command_pose['position'], [0.0005, 0.0, 0.0])
    assert phase_key not in adapter._insertion_lateral_clearance_required


def test_compliant_lateral_gate_caps_full_clearance_retraction(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    current_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    object_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(
        adapter,
        '_object_pose',
        lambda **_: object_pose,
    )
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: {
            'position': np.asarray([0.006, 0.0, -0.010]),
            'orientation': identity.copy(),
        },
    )

    phase_key = ('compliant-capped-clearance',)
    spec = {
        'object': 'part',
        'target_object_target': 'target',
        'target_object_convergence_axis': [0.0, 0.0, -1.0],
        'target_object_lateral_position_tolerance': 0.002,
        'target_object_lateral_alignment_cartesian_position_step': 0.002,
        'target_object_lateral_alignment_axial_clearance': 0.020,
        'target_object_axial_recovery_cartesian_position_step': 0.001,
        'target_object_axial_recovery_deadband': 0.0005,
        'compliant_servo_max_alignment_retraction': 0.006,
        'cartesian_position_step': 0.00025,
        'cartesian_orientation_step': 0.1,
    }

    for retraction_step in range(1, 7):
        command_pose = adapter._target_object_servo_pose(
            phase_key=phase_key,
            task=SimpleNamespace(),
            robot_name='franka_left',
            spec=spec,
            tracked_robots={},
            tracked_objects={'part': {'attachment': {'mode': 'compliant_joint'}}},
            current_pose=current_pose,
            target_pose={
                'position': np.zeros(3),
                'orientation': identity.copy(),
            },
        )
        np.testing.assert_allclose(
            command_pose['position'],
            [0.0005, 0.0, retraction_step * 0.001],
        )
        current_pose['position'] = np.asarray(command_pose['position'], dtype=float)
        current_pose['position'][0] = 0.0
        object_pose['position'] = np.asarray(
            [0.0, 0.0, retraction_step * 0.001],
            dtype=float,
        )
        np.testing.assert_allclose(
            adapter._insertion_lateral_clearance_anchors[phase_key],
            0.016,
        )

    command_pose = adapter._target_object_servo_pose(
        phase_key=phase_key,
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec=spec,
        tracked_robots={},
        tracked_objects={'part': {'attachment': {'mode': 'compliant_joint'}}},
        current_pose=current_pose,
        target_pose={
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )
    np.testing.assert_allclose(command_pose['position'], [0.0005, 0.0, 0.006])
    np.testing.assert_allclose(
        adapter._insertion_lateral_clearance_anchors[phase_key],
        0.016,
    )


def test_compliant_lateral_gate_anchors_retraction_at_current_depth(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    current_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(
        adapter,
        '_object_pose',
        lambda **_: {
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: {
            'position': np.asarray([0.006, 0.0, -0.010]),
            'orientation': identity.copy(),
        },
    )

    phase_key = ('compliant-local-clearance',)
    command_pose = adapter._target_object_servo_pose(
        phase_key=phase_key,
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'target_object_convergence_axis': [0.0, 0.0, -1.0],
            'target_object_lateral_position_tolerance': 0.002,
            'target_object_lateral_alignment_cartesian_position_step': 0.002,
            'target_object_lateral_alignment_axial_clearance': 0.020,
            'target_object_insertion_path_depth': 0.012,
            'target_object_axial_recovery_cartesian_position_step': 0.001,
            'target_object_axial_recovery_deadband': 0.0005,
            'compliant_servo_max_alignment_retraction': 0.006,
            'cartesian_position_step': 0.00025,
            'cartesian_orientation_step': 0.1,
        },
        tracked_robots={},
        tracked_objects={'part': {'attachment': {'mode': 'compliant_joint'}}},
        current_pose=current_pose,
        target_pose={
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )

    np.testing.assert_allclose(command_pose['position'], [0.0005, 0.0, 0.001])
    np.testing.assert_allclose(
        adapter._insertion_lateral_clearance_anchors[phase_key],
        0.016,
    )


def test_compliant_cartesian_position_command_accumulates_with_bounded_lookahead():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    phase_key = ('compliant-position-lookahead',)
    current_position = np.zeros(3)
    one_step_position = np.asarray([0.0, 0.0, 0.0005])
    gate_position = np.asarray([0.0, 0.0, 0.020])

    for expected_height in (0.0005, 0.0010, 0.0015):
        command_position = adapter._accumulated_cartesian_command_position(
            phase_key=phase_key,
            current_position=current_position,
            one_step_position=one_step_position,
            gate_position=gate_position,
            enabled=True,
            lookahead=0.004,
        )
        np.testing.assert_allclose(
            command_position,
            [0.0, 0.0, expected_height],
        )
        adapter._remember_cartesian_command_position(
            phase_key=phase_key,
            command_target_pose={'position': command_position},
        )

    for _ in range(12):
        command_position = adapter._accumulated_cartesian_command_position(
            phase_key=phase_key,
            current_position=current_position,
            one_step_position=one_step_position,
            gate_position=gate_position,
            enabled=True,
            lookahead=0.004,
        )
        adapter._remember_cartesian_command_position(
            phase_key=phase_key,
            command_target_pose={'position': command_position},
        )

    np.testing.assert_allclose(command_position, [0.0, 0.0, 0.004])


def test_compliant_cartesian_position_command_stops_at_nearby_gate():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    phase_key = ('compliant-position-nearby-gate',)
    current_position = np.zeros(3)
    one_step_position = np.asarray([0.0, 0.0, 0.0005])
    gate_position = np.asarray([0.0, 0.0, 0.00125])

    for expected_height in (0.0005, 0.0010, 0.00125):
        command_position = adapter._accumulated_cartesian_command_position(
            phase_key=phase_key,
            current_position=current_position,
            one_step_position=one_step_position,
            gate_position=gate_position,
            enabled=True,
            lookahead=0.004,
        )
        np.testing.assert_allclose(
            command_position,
            [0.0, 0.0, expected_height],
        )
        adapter._remember_cartesian_command_position(
            phase_key=phase_key,
            command_target_pose={'position': command_position},
        )


def test_compliant_cartesian_position_command_can_overdrive_nearby_gate():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    phase_key = ('compliant-position-gate-overdrive',)
    current_position = np.zeros(3)
    one_step_position = np.asarray([0.0, 0.0, 0.0005])
    gate_position = np.asarray([0.0, 0.0, 0.00125])

    for expected_height in (0.0005, 0.0010, 0.0015, 0.0020):
        command_position = adapter._accumulated_cartesian_command_position(
            phase_key=phase_key,
            current_position=current_position,
            one_step_position=one_step_position,
            gate_position=gate_position,
            enabled=True,
            lookahead=0.004,
            allow_gate_overdrive=True,
        )
        np.testing.assert_allclose(
            command_position,
            [0.0, 0.0, expected_height],
        )
        adapter._remember_cartesian_command_position(
            phase_key=phase_key,
            command_target_pose={'position': command_position},
        )


def test_compliant_cartesian_position_lookahead_accumulates_gradually():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    phase_key = ('compliant-position-gradual-lookahead',)
    current_position = np.zeros(3)
    one_step_position = np.asarray([0.0, 0.0, 0.0005])
    gate_position = np.asarray([0.0, 0.0, 0.00125])

    for expected_height in (0.0005, 0.0006, 0.0007, 0.0008):
        command_position = adapter._accumulated_cartesian_command_position(
            phase_key=phase_key,
            current_position=current_position,
            one_step_position=one_step_position,
            gate_position=gate_position,
            enabled=True,
            lookahead=0.004,
            allow_gate_overdrive=True,
            accumulation_step=0.0001,
        )
        np.testing.assert_allclose(
            command_position,
            [0.0, 0.0, expected_height],
        )
        adapter._remember_cartesian_command_position(
            phase_key=phase_key,
            command_target_pose={'position': command_position},
        )


def test_target_object_translation_uses_measured_orientation_during_rotation(
    monkeypatch,
):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    current_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(
        adapter,
        '_object_pose',
        lambda **_: {
            'position': np.asarray([0.2, 0.0, 0.0]),
            'orientation': identity.copy(),
        },
    )
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: {
            'position': np.asarray([0.21, 0.0, 0.0]),
            'orientation': identity.copy(),
        },
    )

    command_pose = adapter._target_object_servo_pose(
        phase_key=('measured-orientation-position-servo',),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'cartesian_position_step': 0.003,
            'cartesian_orientation_step': 0.1,
            'target_object_use_measured_orientation_for_position_servo': True,
        },
        tracked_robots={},
        tracked_objects={},
        current_pose=current_pose,
        target_pose={
            'position': np.zeros(3),
            'orientation': np.asarray([0.70710678, 0.0, 0.0, 0.70710678]),
        },
    )

    np.testing.assert_allclose(command_pose['position'], [0.003, 0.0, 0.0])
    assert not np.allclose(command_pose['orientation'], identity)


def test_fixed_attachment_target_object_servo_uses_bounded_gate_overdrive(
    monkeypatch,
):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    current_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(
        adapter,
        '_object_pose',
        lambda **_: {
            'position': np.asarray([0.2, 0.0, 0.0]),
            'orientation': identity.copy(),
        },
    )
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: {
            'position': np.asarray([0.201, 0.0, 0.0]),
            'orientation': identity.copy(),
        },
    )
    phase_key = ('fixed-target-object-gate-overdrive',)

    for expected_x in (0.0005, 0.0006, 0.0007):
        command_pose = adapter._target_object_servo_pose(
            phase_key=phase_key,
            task=SimpleNamespace(),
            robot_name='franka_left',
            spec={
                'object': 'part',
                'target_object_target': 'target',
                'cartesian_position_step': 0.0005,
                'cartesian_orientation_step': 0.1,
                'target_object_servo_position_command_warm_start': True,
                'target_object_servo_position_command_gate_overdrive': True,
                'target_object_servo_position_command_lookahead': 0.004,
            },
            tracked_robots={},
            tracked_objects={},
            current_pose=current_pose,
            target_pose={
                'position': np.zeros(3),
                'orientation': identity.copy(),
            },
        )
        np.testing.assert_allclose(
            command_pose['position'],
            [expected_x, 0.0, 0.0],
        )
        adapter._remember_cartesian_command_position(
            phase_key=phase_key,
            command_target_pose=command_pose,
        )


def test_fixed_attachment_lateral_alignment_compensates_tracking_residual(
    monkeypatch,
):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    current_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(
        adapter,
        '_object_pose',
        lambda **_: {
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: {
            'position': np.asarray([0.001, 0.0, -0.100]),
            'orientation': identity.copy(),
        },
    )
    phase_key = ('fixed-lateral-alignment-gate-overdrive',)
    spec = {
        'object': 'part',
        'target_object_target': 'target',
        'target_object_convergence_axis': [0.0, 0.0, -1.0],
        'target_object_lateral_position_tolerance': 0.0001,
        'target_object_lateral_alignment_cartesian_position_step': 0.0005,
        'cartesian_position_step': 0.00025,
        'cartesian_orientation_step': 0.1,
        'target_object_servo_position_command_warm_start': True,
        'target_object_servo_position_command_gate_overdrive': True,
        'target_object_servo_position_command_lookahead': 0.004,
        'target_object_servo_position_command_accumulation_step': 0.0001,
    }
    target_pose = {
        'position': np.asarray([0.001, 0.0, -0.100]),
        'orientation': identity.copy(),
    }

    for expected_x in (0.0005, 0.0006, 0.0007):
        command_pose = adapter._target_object_servo_pose(
            phase_key=phase_key,
            task=SimpleNamespace(),
            robot_name='franka_left',
            spec=spec,
            tracked_robots={},
            tracked_objects={},
            current_pose=current_pose,
            target_pose=target_pose,
        )
        np.testing.assert_allclose(
            command_pose['position'],
            [expected_x, 0.0, 0.0],
        )
        adapter._remember_cartesian_command_position(
            phase_key=phase_key,
            command_target_pose=command_pose,
        )


def test_transport_servo_uses_lateral_alignment_hysteresis(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    current_pose = {
        'position': np.zeros(3),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    target_pose = {
        'position': np.asarray([-0.0015, 0.0, -0.100]),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    object_pose = {
        'position': np.zeros(3),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(adapter, '_target_object_pose', lambda **_: target_pose)
    spec = {
        'object': 'part',
        'target_object_target': 'target',
        'target_object_convergence_axis': [0.0, 0.0, -1.0],
        'target_object_lateral_position_tolerance': 0.015,
        'target_object_lateral_alignment_enter_tolerance': 0.002,
        'target_object_lateral_alignment_exit_tolerance': 0.004,
        'target_object_lateral_alignment_cartesian_position_step': 0.002,
        'cartesian_position_step': 0.00025,
        'cartesian_orientation_step': 0.1,
    }
    phase_key = ('base-place',)

    command_pose = adapter._target_object_servo_pose(
        phase_key=phase_key,
        task=SimpleNamespace(),
        robot_name='franka_right',
        spec=spec,
        tracked_robots={},
        tracked_objects={},
        current_pose=current_pose,
        target_pose=target_pose,
    )
    assert command_pose['position'][2] < current_pose['position'][2]

    target_pose['position'] = np.asarray([-0.003, 0.0, -0.100])
    command_pose = adapter._target_object_servo_pose(
        phase_key=phase_key,
        task=SimpleNamespace(),
        robot_name='franka_right',
        spec=spec,
        tracked_robots={},
        tracked_objects={},
        current_pose=current_pose,
        target_pose=target_pose,
    )
    assert command_pose['position'][2] < current_pose['position'][2]

    target_pose['position'] = np.asarray([-0.005, 0.0, -0.100])
    command_pose = adapter._target_object_servo_pose(
        phase_key=phase_key,
        task=SimpleNamespace(),
        robot_name='franka_right',
        spec=spec,
        tracked_robots={},
        tracked_objects={},
        current_pose=current_pose,
        target_pose=target_pose,
    )
    np.testing.assert_allclose(command_pose['position'], [-0.002, 0.0, 0.0])


def test_compliant_servo_resets_position_integrator_between_lateral_and_axial(
    monkeypatch,
):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    current_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    target_object_pose = {
        'position': np.asarray([0.0005, 0.0, -0.010]),
        'orientation': identity.copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(
        adapter,
        '_object_pose',
        lambda **_: {
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: target_object_pose,
    )
    phase_key = ('compliant-mode-transition-reset',)
    spec = {
        'object': 'part',
        'target_object_target': 'target',
        'target_object_convergence_axis': [0.0, 0.0, -1.0],
        'target_object_lateral_position_tolerance': 0.002,
        'target_object_lateral_alignment_enter_tolerance': 0.001,
        'target_object_lateral_alignment_exit_tolerance': 0.002,
        'target_object_lateral_alignment_cartesian_position_step': 0.001,
        'target_object_lateral_alignment_stable_steps': 1,
        'cartesian_position_step': 0.001,
        'cartesian_orientation_step': 0.1,
        'target_object_servo_position_command_warm_start': True,
        'target_object_servo_position_command_gate_overdrive': True,
        'target_object_servo_position_command_lookahead': 0.004,
    }
    tracked_objects = {'part': {'attachment': {'mode': 'compliant_joint'}}}

    adapter._cartesian_command_positions[phase_key] = np.asarray([0.004, 0.0, 0.0])
    command_pose = adapter._target_object_servo_pose(
        phase_key=phase_key,
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec=spec,
        tracked_robots={},
        tracked_objects=tracked_objects,
        current_pose=current_pose,
        target_pose={
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )
    np.testing.assert_allclose(command_pose['position'], [0.0005, 0.0, -0.0005])
    assert phase_key not in adapter._cartesian_command_positions
    assert adapter._insertion_lateral_alignment_active[phase_key] is False

    adapter._cartesian_command_positions[phase_key] = np.asarray([0.0, 0.0, -0.004])
    target_object_pose['position'] = np.asarray([0.003, 0.0, -0.010])
    command_pose = adapter._target_object_servo_pose(
        phase_key=phase_key,
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec=spec,
        tracked_robots={},
        tracked_objects=tracked_objects,
        current_pose=current_pose,
        target_pose={
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )
    np.testing.assert_allclose(command_pose['position'], [0.0005, 0.0, 0.0])
    assert phase_key not in adapter._cartesian_command_positions
    assert adapter._insertion_lateral_alignment_active[phase_key] is True


def test_compliant_axial_servo_keeps_independent_lateral_correction(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    current_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    target_object_pose = {
        'position': np.asarray([0.0015, 0.0, -0.010]),
        'orientation': identity.copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(
        adapter,
        '_object_pose',
        lambda **_: {
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: target_object_pose,
    )
    phase_key = ('compliant-anisotropic-axial-servo',)
    adapter._insertion_lateral_alignment_active[phase_key] = False

    command_pose = adapter._target_object_servo_pose(
        phase_key=phase_key,
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'target_object_convergence_axis': [0.0, 0.0, -1.0],
            'target_object_lateral_position_tolerance': 0.002,
            'target_object_lateral_alignment_enter_tolerance': 0.001,
            'target_object_lateral_alignment_exit_tolerance': 0.002,
            'target_object_lateral_alignment_cartesian_position_step': 0.001,
            'cartesian_position_step': 0.001,
            'cartesian_orientation_step': 0.1,
        },
        tracked_robots={},
        tracked_objects={'part': {'attachment': {'mode': 'compliant_joint'}}},
        current_pose=current_pose,
        target_pose={
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )

    np.testing.assert_allclose(command_pose['position'], [0.0005, 0.0, -0.0005])
    assert adapter._insertion_lateral_alignment_active[phase_key] is False


def test_compliant_axial_servo_can_progress_above_final_lateral_tolerance(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    current_pose = {
        'position': np.zeros(3),
        'orientation': identity.copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(
        adapter,
        '_object_pose',
        lambda **_: {
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )
    monkeypatch.setattr(
        adapter,
        '_target_object_pose',
        lambda **_: {
            'position': np.asarray([0.0028, 0.0, -0.010]),
            'orientation': identity.copy(),
        },
    )
    phase_key = ('compliant-axial-progress-gate',)

    command_pose = adapter._target_object_servo_pose(
        phase_key=phase_key,
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'target_object_convergence_axis': [0.0, 0.0, -1.0],
            'target_object_lateral_position_tolerance': 0.002,
            'target_object_lateral_alignment_enter_tolerance': 0.0035,
            'target_object_lateral_alignment_exit_tolerance': 0.0035,
            'target_object_lateral_alignment_cartesian_position_step': 0.001,
            'target_object_lateral_alignment_stable_steps': 1,
            'cartesian_position_step': 0.001,
            'cartesian_orientation_step': 0.1,
        },
        tracked_robots={},
        tracked_objects={'part': {'attachment': {'mode': 'compliant_joint'}}},
        current_pose=current_pose,
        target_pose={
            'position': np.zeros(3),
            'orientation': identity.copy(),
        },
    )

    assert command_pose['position'][2] < 0.0
    assert adapter._insertion_lateral_alignment_active[phase_key] is False


def test_transport_servo_recovers_axial_waypoint_overshoot(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    current_pose = {
        'position': np.asarray([0.0, 0.0, -0.010]),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    target_pose = {
        'position': np.zeros(3),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    object_pose = {
        'position': current_pose['position'].copy(),
        'orientation': current_pose['orientation'].copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(adapter, '_target_object_pose', lambda **_: target_pose)

    command_pose = adapter._target_object_servo_pose(
        phase_key=('insert-overshoot',),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'target_object_target': 'target',
            'target_object_convergence_axis': [0.0, 0.0, -1.0],
            'target_object_lateral_position_tolerance': 0.002,
            'target_object_lateral_alignment_cartesian_position_step': 0.001,
            'target_object_axial_recovery_cartesian_position_step': 0.001,
            'target_object_axial_recovery_deadband': 0.0005,
            'cartesian_position_step': 0.00025,
            'cartesian_orientation_step': 0.1,
        },
        tracked_robots={},
        tracked_objects={},
        current_pose=current_pose,
        target_pose=target_pose,
    )

    np.testing.assert_allclose(command_pose['position'], [0.0, 0.0, -0.009])


def test_transport_settle_hold_requires_tcp_object_and_lateral_capture(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    target_pose = {
        'position': np.zeros(3),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    current_pose = {
        'position': np.asarray([0.0, 0.0, 0.004]),
        'orientation': target_pose['orientation'].copy(),
    }
    object_pose = {
        'position': np.asarray([0.0015, 0.0, 0.004]),
        'orientation': target_pose['orientation'].copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(adapter, '_target_object_pose', lambda **_: target_pose)
    spec = {
        'object': 'part',
        'target_object_target': 'target',
        'hold_for_target_object_settle': True,
        'require_target_object_static': True,
        'position_tolerance': 0.006,
        'target_object_position_tolerance': 0.008,
        'orientation_tolerance': 0.1,
        'target_object_orientation_tolerance': 0.1,
        'target_object_convergence_axis': [0.0, 0.0, -1.0],
        'target_object_lateral_position_tolerance': 0.002,
    }

    assert adapter._target_object_settle_ready(
        task=SimpleNamespace(phase_step_counter=10),
        spec=spec,
        target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects={},
    )

    object_pose['position'][0] = 0.0025
    assert not adapter._target_object_settle_ready(
        task=SimpleNamespace(phase_step_counter=10),
        spec=spec,
        target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects={},
    )


def test_compliant_hold_expands_only_tcp_settle_tolerance(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    target_pose = {
        'position': np.zeros(3),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    current_pose = {
        'position': np.asarray([0.015, 0.0, 0.0]),
        'orientation': target_pose['orientation'].copy(),
    }
    object_pose = {
        'position': np.asarray([0.001, 0.0, 0.0]),
        'orientation': target_pose['orientation'].copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(adapter, '_target_object_pose', lambda **_: target_pose)
    spec = {
        'object': 'part',
        'target_object_target': 'target',
        'hold_for_target_object_settle': True,
        'require_target_object_static': True,
        'position_tolerance': 0.006,
        'target_object_position_tolerance': 0.008,
        'orientation_tolerance': 0.1,
        'target_object_orientation_tolerance': 0.1,
    }
    tracked_objects = {
        'part': {
            'attachment': {
                'mode': 'compliant_joint',
                'attach_spec': {
                    'compliant_hold_linear_limit': 0.006,
                    'compliant_hold_angular_limit_degrees': 6.0,
                },
            }
        }
    }

    assert adapter._target_object_settle_ready(
        task=SimpleNamespace(phase_step_counter=10),
        spec=spec,
        target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects=tracked_objects,
    )

    object_pose['position'][0] = 0.009
    assert not adapter._target_object_settle_ready(
        task=SimpleNamespace(phase_step_counter=10),
        spec=spec,
        target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects=tracked_objects,
    )


def test_transport_settle_hold_latches_then_retries_servo(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    target_pose = {
        'position': np.zeros(3),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    current_pose = {
        'position': np.asarray([0.0, 0.0, 0.004]),
        'orientation': target_pose['orientation'].copy(),
    }
    object_pose = {
        'position': np.asarray([0.0015, 0.0, 0.004]),
        'orientation': target_pose['orientation'].copy(),
    }
    monkeypatch.setattr(adapter, '_object_name_from_spec', lambda _: 'part')
    monkeypatch.setattr(adapter, '_object_pose', lambda **_: object_pose)
    monkeypatch.setattr(adapter, '_target_object_pose', lambda **_: target_pose)
    spec = {
        'object': 'part',
        'target_object_target': 'target',
        'hold_for_target_object_settle': True,
        'require_target_object_static': True,
        'position_tolerance': 0.006,
        'target_object_position_tolerance': 0.008,
        'orientation_tolerance': 0.1,
        'target_object_orientation_tolerance': 0.1,
        'target_object_convergence_axis': [0.0, 0.0, -1.0],
        'target_object_lateral_position_tolerance': 0.002,
        'target_object_settle_hold_steps': 3,
        'target_object_settle_retry_servo_steps': 2,
    }
    phase_key = ('insert-settle',)
    task = SimpleNamespace(phase_step_counter=10)
    adapter._cartesian_command_positions[phase_key] = np.ones(3)
    adapter._cartesian_command_orientations[phase_key] = np.asarray([1.0, 0.0, 0.0, 0.0])
    adapter._insertion_lateral_alignment_active[phase_key] = True
    adapter._insertion_lateral_alignment_stable_steps[phase_key] = 2
    adapter._insertion_axial_anchors[phase_key] = 0.004
    adapter._insertion_lateral_orientation_anchors[phase_key] = np.asarray([1.0, 0.0, 0.0, 0.0])
    adapter._insertion_lateral_clearance_required[phase_key] = True
    adapter._insertion_lateral_clearance_anchors[phase_key] = 0.006

    assert adapter._target_object_settle_ready(
        phase_key=phase_key,
        task=task,
        spec=spec,
        target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects={},
    )
    assert phase_key not in adapter._cartesian_command_positions
    assert phase_key not in adapter._cartesian_command_orientations
    assert adapter._insertion_lateral_alignment_active[phase_key] is False
    assert phase_key not in adapter._insertion_lateral_alignment_stable_steps
    assert phase_key not in adapter._insertion_axial_anchors
    assert phase_key not in adapter._insertion_lateral_orientation_anchors
    assert phase_key not in adapter._insertion_lateral_clearance_required
    assert phase_key not in adapter._insertion_lateral_clearance_anchors
    adapter._cartesian_command_positions[phase_key] = np.ones(3)
    adapter._cartesian_command_orientations[phase_key] = np.asarray([1.0, 0.0, 0.0, 0.0])
    object_pose['position'][0] = 0.004
    assert adapter._target_object_settle_ready(
        phase_key=phase_key,
        task=task,
        spec=spec,
        target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects={},
    )
    assert phase_key in adapter._cartesian_command_positions
    assert phase_key in adapter._cartesian_command_orientations
    assert adapter._target_object_settle_ready(
        phase_key=phase_key,
        task=task,
        spec=spec,
        target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects={},
    )
    assert phase_key not in adapter._cartesian_command_positions
    assert phase_key not in adapter._cartesian_command_orientations
    assert not adapter._target_object_settle_ready(
        phase_key=phase_key,
        task=task,
        spec=spec,
        target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects={},
    )

    assert not adapter._target_object_settle_ready(
        phase_key=phase_key,
        task=task,
        spec=spec,
        target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects={},
    )

    object_pose['position'][0] = 0.0015
    spec['target_object_settle_hold_steps'] = 1
    assert adapter._target_object_settle_ready(
        phase_key=phase_key,
        task=task,
        spec=spec,
        target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects={},
    )
    assert not adapter._target_object_settle_ready(
        phase_key=phase_key,
        task=task,
        spec=spec,
        target_pose=target_pose,
        current_pose=current_pose,
        tracked_objects={},
    )


def test_contact_metrics_find_opposed_pinch_across_all_finger_samples(monkeypatch):
    class _Body:
        def __init__(self, name: str):
            self.name = name
            self.prim_path = f'/World/{name}'

        def get_pose(self):
            return np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])

        def unwrap(self):
            return self

    task = _task()
    part = _Body('part')
    left = _Body('left')
    right = _Body('right')
    monkeypatch.setattr(task, '_resolve_object', lambda _: part)
    monkeypatch.setattr(
        task,
        '_contact_box_pose',
        lambda *_, **__: (np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])),
    )
    monkeypatch.setattr(task, '_contact_box_scale', lambda *_, **__: np.asarray([0.1, 0.04, 0.04]))
    monkeypatch.setattr(task, '_get_robot_finger_rigid_bodies', lambda _: {'left': left, 'right': right})
    monkeypatch.setattr(task, '_finger_contact_point', lambda *_, **__: None)
    monkeypatch.setattr(task, '_point_to_world_aabb_gap', lambda *_, **__: 1.0)

    samples = {
        'left': [np.asarray([0.0, 0.027, 0.019]), np.asarray([0.0, 0.024, 0.0])],
        'right': [np.asarray([0.0, -0.023, 0.019])],
    }
    monkeypatch.setattr(
        task,
        '_finger_contact_sample_points',
        lambda body, **_: samples[body.name],
    )

    metrics = task._gripper_contact_metrics(
        'part',
        'franka_left',
        attach_spec={
            'finger_contact_distance': 0.006,
            'caging_contact_distance': 0.006,
        },
    )

    assert metrics['left_finger']['local_contact']['best_axis'] == 'z'
    assert metrics['pinch_axis'] == 'y'
    assert metrics['pinch_sample_pair']['left_local_point'] == [0.0, 0.024, 0.0]
    assert metrics['contact_ready'] is True


def test_force_required_grasp_accepts_one_force_probe_with_strict_opposing_contact():
    result = _task()._strict_physical_grasp_contact(
        'part',
        {
            'pinch_axis': 'x',
            'left_finger': _finger(force_contact=True, signed_x=0.03),
            'right_finger': _finger(force_contact=False, signed_x=-0.03),
        },
        attach_spec={
            'require_force_contact': True,
            'physical_attach_surface_gap': 0.006,
        },
    )

    assert result['dual_force_contact'] is False
    assert result['any_force_contact'] is True
    assert result['force_supported_geometric_contact'] is True
    assert result['physical_contact_ready'] is True


def test_force_required_grasp_rejects_geometry_without_any_force_evidence():
    result = _task()._strict_physical_grasp_contact(
        'part',
        {
            'pinch_axis': 'x',
            'left_finger': _finger(force_contact=False, signed_x=0.03),
            'right_finger': _finger(force_contact=False, signed_x=-0.03),
        },
        attach_spec={
            'require_force_contact': True,
            'physical_attach_surface_gap': 0.006,
        },
    )

    assert result['strict_pinch_contact'] is True
    assert result['any_force_contact'] is False
    assert result['physical_contact_ready'] is False


def test_dual_force_mode_rejects_single_probe_fallback():
    result = _task()._strict_physical_grasp_contact(
        'part',
        {
            'pinch_axis': 'x',
            'left_finger': _finger(force_contact=True, signed_x=0.03),
            'right_finger': _finger(force_contact=False, signed_x=-0.03),
        },
        attach_spec={
            'require_force_contact': True,
            'require_dual_force_contact': True,
            'physical_attach_surface_gap': 0.006,
        },
    )

    assert result['force_supported_geometric_contact'] is True
    assert result['physical_contact_ready'] is False


def test_force_required_grasp_accepts_opted_in_cross_axis_dual_contact():
    result = _task()._strict_physical_grasp_contact(
        'part',
        {
            'pinch_axis': None,
            'left_finger': _finger(force_contact=False, signed_x=0.03, axis='x'),
            'right_finger': _finger(force_contact=True, signed_x=-0.03, axis='y'),
        },
        attach_spec={
            'require_force_contact': True,
            'allow_cross_axis_dual_finger_contact': True,
            'physical_attach_surface_gap': 0.006,
        },
    )

    assert result['strict_dual_finger_contact'] is True
    assert result['any_force_contact'] is True
    assert result['force_supported_geometric_contact'] is True
    assert result['physical_contact_ready'] is True


def test_force_optional_grasp_accepts_strict_geometry_with_zero_force_readings():
    result = _task()._strict_physical_grasp_contact(
        'part',
        {
            'pinch_axis': 'x',
            'left_finger': _finger(force_contact=False, signed_x=0.03),
            'right_finger': _finger(force_contact=False, signed_x=-0.03),
        },
        attach_spec={
            'require_force_contact': False,
            'physical_attach_surface_gap': 0.006,
        },
    )

    assert result['strict_pinch_contact'] is True
    assert result['any_force_contact'] is False
    assert result['physical_contact_ready'] is True


def test_strict_grasp_interior_scale_rejects_edge_contact_and_accepts_center_contact():
    task = _task()
    spec = {
        'require_force_contact': False,
        'physical_attach_surface_gap': 0.005,
        'physical_contact_axes': ['y'],
        'physical_contact_interior_scale': 0.9,
    }

    def metrics(x_position: float) -> dict:
        return {
            'pinch_axis': 'y',
            'contact_box_scale': [0.1, 0.04, 0.026],
            'left_finger': _finger(
                force_contact=False,
                signed_x=0.021,
                axis='y',
                local_point=[x_position, 0.021, 0.0],
            ),
            'right_finger': _finger(
                force_contact=False,
                signed_x=-0.021,
                axis='y',
                local_point=[x_position, -0.021, 0.0],
            ),
        }

    edge = task._strict_physical_grasp_contact('part', metrics(0.049), attach_spec=spec)
    centered = task._strict_physical_grasp_contact('part', metrics(0.02), attach_spec=spec)

    assert edge['strict_pinch_contact'] is True
    assert edge['interior_contact_ready'] is False
    assert edge['physical_contact_ready'] is False
    assert centered['interior_contact_ready'] is True
    assert centered['physical_contact_ready'] is True


def test_strict_grasp_searches_all_finger_samples_for_an_interior_pair():
    def finger(side: float) -> dict:
        metric = _finger(
            force_contact=False,
            signed_x=side * 0.021,
            axis='y',
            local_point=[0.0, side * 0.021, 0.020],
        )
        metric['sample_contacts'] = [
            {'local_contact': metric['local_contact']},
            {
                'local_contact': {
                    'local_point': [0.0, side * 0.022, 0.0],
                    'axes': {
                        'y': {
                            'contact': True,
                            'surface_gap': 0.002,
                            'signed_coordinate': side * 0.022,
                        }
                    },
                }
            },
        ]
        return metric

    result = _task()._strict_physical_grasp_contact(
        'part',
        {
            'pinch_axis': 'y',
            'contact_box_scale': [0.1, 0.04, 0.026],
            'left_finger': finger(1.0),
            'right_finger': finger(-1.0),
        },
        attach_spec={
            'physical_attach_surface_gap': 0.005,
            'physical_contact_axes': ['y'],
            'physical_contact_interior_scale': 0.9,
        },
    )

    assert result['strict_sample_pair']['axis'] == 'y'
    assert result['interior_contact_ready'] is True
    assert result['physical_contact_ready'] is True


def test_physical_joint_gates_accept_opposing_contact_from_finger_samples(monkeypatch):
    task = _task()

    def finger(side: float) -> dict:
        metric = _finger(
            force_contact=False,
            signed_x=0.02,
            axis='z',
            local_point=[0.0, 0.0, 0.02],
        )
        metric['sample_contacts'] = [
            {
                'local_contact': {
                    'local_point': [side * 0.03, 0.0, 0.0],
                    'axes': {
                        'x': {
                            'contact': True,
                            'surface_gap': 0.003,
                            'signed_coordinate': side * 0.03,
                        }
                    },
                }
            }
        ]
        return metric

    left = finger(1.0)
    right = finger(-1.0)
    contact_metrics = {
        'contact_ready': True,
        'pinch_axis': 'x',
        'contact_box_scale': [0.1, 0.04, 0.04],
        'left_finger': left,
        'right_finger': right,
    }
    attach_spec = {
        'object': 'part',
        'robot': 'franka_left',
        'attachment_mode': 'fixed_joint',
        'require_contact': True,
        'require_physical_contact': True,
        'require_target_reached_for_attach': True,
        'physical_attach_surface_gap': 0.006,
        'gripper_closed_threshold': 0.04,
        'support_height_tolerance': None,
        'top_clearance': None,
    }

    assert task._strict_dual_finger_contact('part', left, right, attach_spec=attach_spec) is False
    assert (
        task._strict_physical_grasp_contact('part', contact_metrics, attach_spec=attach_spec)['physical_contact_ready']
        is True
    )

    task._attachments = {}
    task.step_counter = 1
    task.phase_step_counter = 1
    monkeypatch.setattr(task, '_current_gripper_command', lambda *_: 'close')
    monkeypatch.setattr(
        task,
        '_resolve_attach_target_info',
        lambda **_: {
            'target_reached': True,
            'position_error': 0.005,
            'orientation_error': 0.01,
        },
    )
    monkeypatch.setattr(task, '_get_robot_gripper_opening', lambda *_: 0.03)
    monkeypatch.setattr(task, '_attach_proximity_metrics', lambda **_: {'within_proximity': True})
    monkeypatch.setattr(task, '_sampled_object_position', lambda *_: None)
    monkeypatch.setattr(task, '_gripper_contact_metrics', lambda *_, **__: contact_metrics)
    monkeypatch.setattr(task, '_contact_box_scale', lambda *_, **__: np.asarray([0.1, 0.04, 0.04]))
    monkeypatch.setattr(task, '_is_slender_attach_object', lambda *_, **__: False)

    phase_spec = {
        'name': 'close_and_attach',
        'gripper_commands': {'franka_left': 'close'},
    }
    assert task._attach_ready(phase_spec, attach_spec) is True
    assert task._robot_object_contact(attach_spec) is True


def test_attach_accepts_bounded_contact_refined_target_after_strict_close(monkeypatch):
    task = _task()
    task._attachments = {}
    task.step_counter = 100
    task.phase_step_counter = 100
    task.phase_index = 4
    task.phase_entry_step = 80
    task._local_skill_completions = {
        (4, 80, 'franka_left', 'ur5e_close_gripper'): {
            'contact_detail': {'strict_contact_ready': True},
            'recenter': {'offset_world': [0.00075, 0.0, 0.0]},
        }
    }
    phase_spec = {
        'name': 'close_and_attach',
        'gripper_commands': {'franka_left': 'close'},
        'local_skill': {
            'name': 'ur5e_close_gripper',
            'robot': 'franka_left',
        },
    }
    attach_spec = {
        'object': 'part',
        'robot': 'franka_left',
        'attachment_mode': 'fixed_joint',
        'require_contact': True,
        'require_physical_contact': True,
        'require_local_skill_complete_for_attach': True,
        'require_target_reached_for_attach': True,
        'allow_strict_contact_target_refinement': True,
        'strict_contact_target_refinement_max_distance': 0.025,
        'strict_contact_target_refinement_tracking_tolerance': 0.00035,
        'position_tolerance': 0.007,
        'orientation_tolerance': 0.10,
        'physical_attach_surface_gap': 0.006,
        'gripper_closed_threshold': 0.46,
        'support_height_tolerance': None,
        'top_clearance': None,
    }
    target_info = {
        'target_reached': False,
        'position_error': 0.00805,
        'position_tolerance': 0.007,
        'orientation_error': 0.0885,
        'orientation_tolerance': 0.10,
    }
    contact_metrics = {
        'contact_ready': True,
        'contact_box_scale': [0.04, 0.04, 0.04],
        'left_finger': {},
        'right_finger': {},
    }

    monkeypatch.setattr(task, '_current_gripper_command', lambda *_: 'close')
    monkeypatch.setattr(task, '_resolve_attach_target_info', lambda **_: target_info)
    monkeypatch.setattr(task, '_get_robot_gripper_opening', lambda *_: 0.41)
    monkeypatch.setattr(
        task,
        '_attach_proximity_metrics',
        lambda **_: {'within_proximity': True},
    )
    monkeypatch.setattr(task, '_sampled_object_position', lambda *_: None)
    monkeypatch.setattr(
        task,
        '_gripper_contact_metrics',
        lambda *_, **__: contact_metrics,
    )
    monkeypatch.setattr(
        task,
        '_strict_physical_grasp_contact',
        lambda *_, **__: {'physical_contact_ready': True},
    )
    monkeypatch.setattr(task, '_is_slender_attach_object', lambda *_, **__: False)

    assert task._attach_ready(phase_spec, attach_spec) is True

    target_info['position_error'] = 0.0082
    assert task._attach_ready(phase_spec, attach_spec) is False

    target_info['position_error'] = 0.00805
    task._local_skill_completions[(4, 80, 'franka_left', 'ur5e_close_gripper')]['contact_detail'][
        'strict_contact_ready'
    ] = False
    assert task._attach_ready(phase_spec, attach_spec) is False


def test_attach_accepts_object_blocked_franka_opening_with_strict_dual_contact(monkeypatch):
    task = _task()
    task._attachments = {}
    task.step_counter = 100
    task.phase_step_counter = 100
    phase_spec = {
        'name': 'close_and_attach',
        'gripper_commands': {'franka_left': 'close'},
    }
    attach_spec = {
        'object': 'part',
        'robot': 'franka_left',
        'attachment_mode': 'fixed_joint',
        'require_contact': True,
        'require_physical_contact': True,
        'require_target_reached_for_attach': True,
        'gripper_closed_threshold': 0.0086,
        'physical_attach_surface_gap': 0.006,
        'support_height_tolerance': None,
        'top_clearance': None,
    }
    contact_metrics = {
        'contact_ready': True,
        'pinch_axis': 'y',
        'left_finger': {},
        'right_finger': {},
    }

    monkeypatch.setattr(task, '_current_gripper_command', lambda *_: 'close')
    monkeypatch.setattr(
        task,
        '_resolve_attach_target_info',
        lambda **_: {
            'target_reached': True,
            'position_error': 0.001,
            'orientation_error': 0.01,
        },
    )
    monkeypatch.setattr(task, '_get_robot_gripper_opening', lambda *_: 0.0188)
    monkeypatch.setattr(task, '_attach_proximity_metrics', lambda **_: {'within_proximity': True})
    monkeypatch.setattr(task, '_sampled_object_position', lambda *_: None)
    monkeypatch.setattr(task, '_gripper_contact_metrics', lambda *_, **__: contact_metrics)
    monkeypatch.setattr(task, '_contact_box_scale', lambda *_, **__: np.asarray([0.012, 0.012, 0.076]))
    monkeypatch.setattr(
        task,
        '_strict_physical_grasp_contact',
        lambda *_, **__: {'physical_contact_ready': True},
    )
    monkeypatch.setattr(task, '_is_slender_attach_object', lambda *_, **__: False)

    assert task._gripper_opening_limit(
        'part',
        attach_spec,
        contact_metrics=contact_metrics,
    ) < 0.0188
    assert task._attach_ready(phase_spec, attach_spec) is True

    attach_spec['allow_contact_blocked_gripper_opening'] = False
    assert task._attach_ready(phase_spec, attach_spec) is False


def test_strict_grasp_allows_calibrated_margin_at_a_physical_edge():
    task = _task()
    contact_metrics = {
        'pinch_axis': 'z',
        'contact_box_scale': [0.1, 0.04, 0.026],
        'left_finger': _finger(
            force_contact=False,
            signed_x=0.025,
            axis='y',
            local_point=[0.016, 0.025, 0.014],
        ),
        'right_finger': _finger(
            force_contact=False,
            signed_x=-0.024,
            axis='y',
            local_point=[-0.020, -0.024, -0.010],
        ),
    }
    spec = {
        'physical_attach_surface_gap': 0.006,
        'physical_contact_axes': ['y'],
        'physical_contact_interior_scale': 0.9,
    }

    without_margin = task._strict_physical_grasp_contact(
        'part',
        contact_metrics,
        attach_spec={**spec, 'physical_contact_interior_margin': 0.0},
    )
    calibrated = task._strict_physical_grasp_contact(
        'part',
        contact_metrics,
        attach_spec=spec,
    )

    assert without_margin['physical_contact_ready'] is False
    assert calibrated['strict_sample_pair']['axis'] == 'y'
    assert calibrated['physical_contact_interior_margin'] == 0.003
    assert calibrated['physical_contact_ready'] is True


def test_force_probe_is_only_enabled_when_required_or_explicitly_requested():
    task = _task()
    assert task._force_contact_measurement_enabled({}) is False
    assert task._force_contact_measurement_enabled({'require_force_contact': False}) is False
    assert task._force_contact_measurement_enabled({'require_force_contact': True}) is True
    assert task._force_contact_measurement_enabled({'require_contact_report': True}) is True
    assert task._force_contact_measurement_enabled({'measure_force_contact': True}) is True
    assert (
        task._force_contact_measurement_enabled({'require_force_contact': True, 'measure_force_contact': False}) is True
    )


def test_grasp_check_forwards_configured_interior_margin():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    captured = []

    def contact_metrics(_object_name, _robot_name, *, attach_spec):
        captured.append(dict(attach_spec))
        return {'contact_ready': True}

    task = SimpleNamespace(
        _gripper_contact_metrics=contact_metrics,
        _strict_physical_grasp_contact=lambda *_args, **_kwargs: {'physical_contact_ready': True},
    )
    ready, _ = adapter._grasp_contact_ready(
        task=task,
        robot_name='franka_left',
        spec={
            'object': 'part',
            'require_strict_physical_contact': True,
            'physical_contact_interior_margin': 0.002,
        },
    )

    assert ready is True
    assert captured[0]['physical_contact_interior_margin'] == 0.002


def test_grasp_check_forwards_force_measurement_without_requiring_it():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    captured = []

    def contact_metrics(_object_name, _robot_name, *, attach_spec):
        captured.append(dict(attach_spec))
        return {'contact_ready': True}

    task = SimpleNamespace(
        _gripper_contact_metrics=contact_metrics,
        _strict_physical_grasp_contact=lambda *_args, **_kwargs: {'physical_contact_ready': True},
    )
    ready, _ = adapter._grasp_contact_ready(
        task=task,
        robot_name='franka_left',
        spec={
            'object': 'part',
            'measure_force_contact': True,
            'require_force_contact': False,
        },
    )

    assert ready is True
    assert captured[0]['measure_force_contact'] is True
    assert captured[0]['require_force_contact'] is False


def test_close_contact_does_not_let_static_flag_override_excess_velocity(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    monkeypatch.setattr(adapter, '_current_gripper_q', lambda **_: 0.5)
    monkeypatch.setattr(adapter, '_gripper_open_closed_q', lambda **_: (0.0, 0.8))
    monkeypatch.setattr(adapter, '_grasp_contact_ready', lambda **_: (True, {}))
    monkeypatch.setattr(
        adapter,
        '_object_motion_detail',
        lambda **_: {
            'valid': True,
            'linear_speed': 0.7,
            'angular_speed': 16.0,
            'is_static': True,
            'pose_stable_override': True,
        },
    )
    state = {'last_gripper_q': 0.5}

    ready, detail = adapter._close_until_contact_ready(
        state=state,
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'close_until_contact_min_steps': 1,
            'close_contact_stable_steps': 1,
            'close_contact_motion_stable_steps': 1,
            'close_contact_latch_after_stable': True,
            'close_contact_max_object_speed': 0.08,
            'close_contact_max_angular_speed': 3.0,
        },
        tracked_objects={},
        close_elapsed_steps=1,
        gripper_openness=0.0,
    )

    assert detail['contact_ready'] is True
    assert detail['motion_detail']['velocity_thresholds_passed'] is False
    assert detail['motion_ready'] is False
    assert 'hold_gripper_openness' not in state
    assert ready is False

    ready, detail = adapter._close_until_contact_ready(
        state={'last_gripper_q': 0.5},
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'close_until_contact_min_steps': 1,
            'close_contact_stable_steps': 1,
            'close_contact_motion_stable_steps': 1,
            'close_contact_latch_after_stable': True,
            'close_contact_max_object_speed': 0.08,
            'close_contact_max_angular_speed': 3.0,
            'close_contact_allow_pose_stable_override': True,
        },
        tracked_objects={},
        close_elapsed_steps=1,
        gripper_openness=0.0,
    )

    assert detail['motion_detail']['raw_velocity_thresholds_passed'] is False
    assert detail['motion_detail']['pose_stable_override_used'] is True
    assert detail['motion_ready'] is True
    assert ready is True


def test_close_contact_latches_hold_only_after_contact_and_motion_are_stable(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    monkeypatch.setattr(adapter, '_current_gripper_q', lambda **_: 0.4)
    monkeypatch.setattr(adapter, '_gripper_open_closed_q', lambda **_: (0.0, 0.8))
    monkeypatch.setattr(adapter, '_grasp_contact_ready', lambda **_: (True, {}))
    monkeypatch.setattr(
        adapter,
        '_object_motion_detail',
        lambda **_: {
            'valid': True,
            'linear_speed': 0.0,
            'angular_speed': 0.0,
        },
    )
    state = {'last_gripper_q': 0.4}
    task = SimpleNamespace()
    spec = {
        'object': 'part',
        'close_until_contact_min_steps': 1,
        'close_contact_stable_steps': 2,
        'close_contact_motion_stable_steps': 2,
        'close_contact_latch_after_stable': True,
        'close_contact_max_object_speed': 0.08,
        'close_contact_max_angular_speed': 3.0,
        'close_contact_hold_squeeze_margin': 0.0,
    }

    ready, _ = adapter._close_until_contact_ready(
        state=state,
        task=task,
        robot_name='franka_left',
        spec=spec,
        tracked_objects={},
        close_elapsed_steps=1,
        gripper_openness=0.5,
    )
    assert ready is False
    assert 'hold_gripper_openness' not in state

    ready, detail = adapter._close_until_contact_ready(
        state=state,
        task=task,
        robot_name='franka_left',
        spec=spec,
        tracked_objects={},
        close_elapsed_steps=2,
        gripper_openness=0.5,
    )
    assert ready is True
    assert detail['hold_gripper_openness'] == 0.5
    assert state['hold_gripper_openness'] == 0.5


def test_close_contact_scales_implicit_joint_gates_for_short_stroke_gripper(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    monkeypatch.setattr(adapter, '_current_gripper_q', lambda **_: 0.0373)
    monkeypatch.setattr(adapter, '_gripper_open_closed_q', lambda **_: (0.04, 0.0))
    monkeypatch.setattr(adapter, '_grasp_contact_ready', lambda **_: (True, {}))

    ready, detail = adapter._close_until_contact_ready(
        state={'last_gripper_q': 0.0373},
        task=SimpleNamespace(),
        robot_name='franka_right',
        spec={
            'object': 'part',
            'close_until_contact_min_steps': 1,
            'close_contact_stable_steps': 1,
            'require_strict_physical_contact': True,
        },
        tracked_objects={},
        close_elapsed_steps=1,
        gripper_openness=0.465,
    )

    assert ready is True
    assert detail['moved_from_open'] is True
    assert detail['contact_candidate'] is True
    assert detail['gripper_joint_range'] == pytest.approx(0.04)
    assert detail['close_contact_min_joint_closure'] == pytest.approx(0.0025)
    assert detail['close_contact_short_command_max_joint_closure'] == pytest.approx(0.004)
    assert detail['close_contact_blocked_joint_margin'] == pytest.approx(0.01)
    assert detail['close_gripper_target_tolerance'] == pytest.approx(0.002)


def test_close_contact_keeps_explicit_joint_gate_for_short_stroke_gripper(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    monkeypatch.setattr(adapter, '_current_gripper_q', lambda **_: 0.0186)
    monkeypatch.setattr(adapter, '_gripper_open_closed_q', lambda **_: (0.04, 0.0))
    monkeypatch.setattr(adapter, '_grasp_contact_ready', lambda **_: (True, {}))

    ready, detail = adapter._close_until_contact_ready(
        state={'last_gripper_q': 0.0186},
        task=SimpleNamespace(),
        robot_name='franka_right',
        spec={
            'object': 'part',
            'close_until_contact_min_steps': 1,
            'close_contact_stable_steps': 1,
            'close_contact_min_joint_closure': 0.05,
        },
        tracked_objects={},
        close_elapsed_steps=1,
        gripper_openness=0.465,
    )

    assert ready is False
    assert detail['moved_from_open'] is False
    assert detail['close_contact_min_joint_closure'] == pytest.approx(0.05)


def test_close_accepts_initial_strict_contact_when_commanded_stroke_is_short(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    monkeypatch.setattr(adapter, '_current_gripper_q', lambda **_: 0.04)
    monkeypatch.setattr(adapter, '_gripper_open_closed_q', lambda **_: (0.04, 0.0))
    monkeypatch.setattr(
        adapter,
        '_grasp_contact_ready',
        lambda **_: (True, {'strict_contact_ready': True}),
    )

    ready, detail = adapter._close_until_contact_ready(
        state={'last_gripper_q': 0.04},
        task=SimpleNamespace(),
        robot_name='franka_right',
        spec={
            'object': 'wide_part',
            'close_until_contact_min_steps': 1,
            'close_contact_stable_steps': 1,
            'require_strict_physical_contact': True,
            'allow_initial_strict_contact_for_short_close': True,
        },
        tracked_objects={},
        close_elapsed_steps=1,
        gripper_openness=0.9025,
    )

    assert ready is True
    assert detail['moved_from_open'] is False
    assert detail['commanded_closure_from_open'] == pytest.approx(0.0039)
    assert detail['initial_strict_contact_candidate'] is True
    assert detail['contact_candidate'] is True


@pytest.mark.parametrize(
    ('enabled', 'strict_contact_ready'),
    ((False, True), (True, False)),
)
def test_close_rejects_initial_contact_without_explicit_strict_short_close_gate(
    monkeypatch,
    enabled,
    strict_contact_ready,
):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    monkeypatch.setattr(adapter, '_current_gripper_q', lambda **_: 0.04)
    monkeypatch.setattr(adapter, '_gripper_open_closed_q', lambda **_: (0.04, 0.0))
    monkeypatch.setattr(
        adapter,
        '_grasp_contact_ready',
        lambda **_: (True, {'strict_contact_ready': strict_contact_ready}),
    )

    ready, detail = adapter._close_until_contact_ready(
        state={'last_gripper_q': 0.04},
        task=SimpleNamespace(),
        robot_name='franka_right',
        spec={
            'object': 'wide_part',
            'close_until_contact_min_steps': 1,
            'close_contact_stable_steps': 1,
            'require_strict_physical_contact': True,
            'allow_initial_strict_contact_for_short_close': enabled,
        },
        tracked_objects={},
        close_elapsed_steps=1,
        gripper_openness=0.9025,
    )

    assert ready is False
    assert detail['initial_strict_contact_candidate'] is False
    assert detail['contact_candidate'] is False


def test_strict_close_does_not_complete_from_joint_stall_without_contact(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    monkeypatch.setattr(adapter, '_current_gripper_q', lambda **_: 0.03)
    monkeypatch.setattr(adapter, '_gripper_open_closed_q', lambda **_: (0.04, 0.0))
    monkeypatch.setattr(
        adapter,
        '_grasp_contact_ready',
        lambda **_: (False, {'strict_contact_ready': False}),
    )

    ready, detail = adapter._close_until_contact_ready(
        state={'last_gripper_q': 0.03},
        task=SimpleNamespace(),
        robot_name='franka_right',
        spec={
            'object': 'part',
            'require_strict_physical_contact': True,
            'close_until_contact_min_steps': 1,
            'close_contact_stable_steps': 1,
        },
        tracked_objects={},
        close_elapsed_steps=1,
        gripper_openness=0.0,
    )

    assert ready is False
    assert detail['stalled'] is True
    assert detail['moved_from_open'] is True
    assert detail['blocked_before_full_close'] is True
    assert detail['stall_contact'] is False
    assert detail['detected_clamp'] is False
    assert detail['require_strict_physical_contact'] is True


def test_close_can_defer_latch_until_contact_is_stable(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    monkeypatch.setattr(adapter, '_current_gripper_q', lambda **_: 0.4)
    monkeypatch.setattr(adapter, '_gripper_open_closed_q', lambda **_: (0.0, 0.8))
    contact_results = iter([(True, {}), (False, {})])
    monkeypatch.setattr(adapter, '_grasp_contact_ready', lambda **_: next(contact_results))
    state = {'last_gripper_q': 0.4}
    spec = {
        'object': 'part',
        'require_strict_physical_contact': True,
        'close_until_contact_min_steps': 1,
        'close_contact_stable_steps': 2,
        'close_contact_hold_squeeze_margin': 0.0,
        'use_joint_stall_for_close_until_contact': False,
    }

    ready, _ = adapter._close_until_contact_ready(
        state=state,
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec=spec,
        tracked_objects={},
        close_elapsed_steps=1,
        gripper_openness=0.5,
    )
    assert ready is False
    assert 'hold_gripper_openness' not in state

    ready, detail = adapter._close_until_contact_ready(
        state=state,
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec=spec,
        tracked_objects={},
        close_elapsed_steps=2,
        gripper_openness=0.4,
    )
    assert ready is False
    assert detail['stable_steps'] == 0
    assert 'hold_gripper_openness' not in state


def test_close_latches_first_valid_clamp_by_default(monkeypatch):
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    monkeypatch.setattr(adapter, '_current_gripper_q', lambda **_: 0.4)
    monkeypatch.setattr(adapter, '_gripper_open_closed_q', lambda **_: (0.0, 0.8))
    monkeypatch.setattr(adapter, '_grasp_contact_ready', lambda **_: (True, {}))
    state = {'last_gripper_q': 0.4}

    ready, detail = adapter._close_until_contact_ready(
        state=state,
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'require_strict_physical_contact': True,
            'close_until_contact_min_steps': 1,
            'close_contact_stable_steps': 24,
            'close_contact_hold_squeeze_margin': 0.0,
            'use_joint_stall_for_close_until_contact': False,
        },
        tracked_objects={},
        close_elapsed_steps=1,
        gripper_openness=0.5,
    )

    assert ready is False
    assert detail['stable_steps'] == 1
    assert detail['hold_gripper_openness'] == 0.5
    assert state['hold_gripper_openness'] == 0.5


def test_descend_aligns_orientation_before_translating():
    current_pose = {
        'position': np.asarray([0.1, 0.2, 0.3]),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    target_pose = {
        'position': np.asarray([0.4, 0.5, 0.6]),
        'orientation': np.asarray([0.70710678, 0.0, 0.0, 0.70710678]),
    }

    command = UR5eAssemblyAtomicSkillAdapter._orientation_first_servo_pose(
        skill_name='ur5e_descend_to_grasp',
        spec={
            'cartesian_orientation_step': 0.02,
            'orientation_tolerance': 0.1,
            'orientation_first_before_translation': True,
        },
        current_pose=current_pose,
        target_pose=target_pose,
    )

    assert command is not None
    np.testing.assert_allclose(command['position'], current_pose['position'])
    assert not np.allclose(command['orientation'], current_pose['orientation'])
    assert not np.allclose(command['orientation'], target_pose['orientation'])
    assert (
        UR5eAssemblyAtomicSkillAdapter._orientation_first_servo_pose(
            skill_name='ur5e_move_above_part',
            spec={'orientation_tolerance': 0.1},
            current_pose=current_pose,
            target_pose=target_pose,
        )
        is None
    )
    assert (
        UR5eAssemblyAtomicSkillAdapter._orientation_first_servo_pose(
            skill_name='ur5e_descend_to_grasp',
            spec={
                'orientation_tolerance': 0.1,
                'orientation_first_before_translation': True,
                'orientation_first_max_steps': 48,
            },
            current_pose=current_pose,
            target_pose=target_pose,
            phase_step_counter=48,
        )
        is None
    )
    assert (
        UR5eAssemblyAtomicSkillAdapter._orientation_first_servo_pose(
            skill_name='ur5e_descend_to_grasp',
            spec={'orientation_tolerance': None},
            current_pose=current_pose,
            target_pose=target_pose,
        )
        is None
    )


def test_object_tcp_slip_gate_remains_strict_for_fixed_grasp():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    phase_key = ('task', 'phase')
    spec = {
        'object': 'part',
        'max_object_tcp_slip': 0.04,
        'require_target_object_pose_convergence': True,
    }
    current_pose = {
        'position': np.zeros(3),
        'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
    }
    tracked_objects = {
        'part': {
            'position': [0.0, 0.0, 0.20],
            'orientation': [1.0, 0.0, 0.0, 0.0],
            'attachment': {'mode': 'fixed_joint'},
        }
    }

    assert (
        adapter._object_tcp_slip_failure(
            phase_key=phase_key,
            task=SimpleNamespace(),
            robot_name='franka_left',
            spec=spec,
            tracked_objects=tracked_objects,
            current_pose=current_pose,
        )
        is None
    )
    tracked_objects['part']['position'] = [0.05, 0.0, 0.20]

    detail = adapter._object_tcp_slip_failure(
        phase_key=phase_key,
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec=spec,
        tracked_objects=tracked_objects,
        current_pose=current_pose,
    )

    assert detail is not None
    assert detail['slip'] == pytest.approx(0.05)


def test_object_tcp_slip_gate_allows_target_gated_compliant_insertion():
    adapter = UR5eAssemblyAtomicSkillAdapter({})
    tracked_objects = {
        'part': {
            'position': [0.05, 0.0, 0.20],
            'orientation': [1.0, 0.0, 0.0, 0.0],
            'attachment': {
                'mode': 'compliant_joint',
                'attach_spec': {'compliant_hold_linear_limit': 0.006},
            },
        }
    }

    detail = adapter._object_tcp_slip_failure(
        phase_key=('task', 'phase'),
        task=SimpleNamespace(),
        robot_name='franka_left',
        spec={
            'object': 'part',
            'max_object_tcp_slip': 0.04,
            'require_target_object_pose_convergence': True,
        },
        tracked_objects=tracked_objects,
        current_pose={
            'position': np.zeros(3),
            'orientation': np.asarray([1.0, 0.0, 0.0, 0.0]),
        },
    )

    assert detail is None


def test_release_lock_uses_explicit_geometry_tolerance(monkeypatch):
    task = _task()
    task._attachments = {}
    task._locked_targets = {}
    monkeypatch.setattr(
        task,
        '_resolve_object',
        lambda _name: SimpleNamespace(
            get_pose=lambda: (
                np.asarray([0.012, 0.0, 0.0]),
                np.asarray([1.0, 0.0, 0.0, 0.0]),
            )
        ),
    )
    monkeypatch.setattr(
        task,
        '_resolve_target_pose_spec',
        lambda _name: (
            None,
            np.zeros(3),
            np.asarray([1.0, 0.0, 0.0, 0.0]),
            {'position_tolerance': 0.008, 'orientation_tolerance': 0.1},
        ),
    )

    ready = task._lock_ready(
        {'name': 'release_and_lock'},
        {
            'object': 'part',
            'target': 'part_assembled',
            'position_tolerance': 0.015,
            'orientation_tolerance': 0.12,
        },
    )

    assert ready is True
