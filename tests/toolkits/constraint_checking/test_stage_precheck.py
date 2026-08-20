from types import SimpleNamespace

import numpy as np

from toolkits.constraint_checking.integration.stage_precheck import (
    StageTrajectoryPrechecker,
)


class JointSubset:
    def get_joint_positions(self):
        return np.zeros(6)


class JointController:
    config = SimpleNamespace(name='arm_joint')

    def get_joint_subset(self):
        return JointSubset()


class Solver:
    frames = [
        'shoulder_link',
        'upper_arm_link',
        'forearm_link',
        'wrist_1_link',
        'wrist_2_link',
        'wrist_3_link',
    ]

    def set_robot_base_pose(self, **kwargs):
        return None

    def get_all_frame_names(self):
        return self.frames

    def compute_forward_kinematics(self, frame_name, joint_positions):
        index = self.frames.index(frame_name)
        return np.asarray([joint_positions[0], 0.0, 0.2 + index * 0.1]), np.eye(3)


class IKController:
    config = SimpleNamespace(name='arm_ik')
    _kinematics_solver = Solver()
    _robot_scale = 1.0

    def get_ik_base_world_pose(self):
        return np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])


class WrappedSolver:
    def __init__(self):
        self._kinematics = Solver()

    def set_robot_base_pose(self, **kwargs):
        return self._kinematics.set_robot_base_pose(**kwargs)


class WrappedIKController(IKController):
    _kinematics_solver = WrappedSolver()


def make_task(step=0, object_states=None):
    robot = SimpleNamespace(controllers={'arm_joint': JointController(), 'arm_ik': IKController()})
    return SimpleNamespace(
        step_counter=step,
        robots={'left': robot},
        get_tracked_object_states=lambda: object_states or {},
    )


def test_project_kinematics_wrapper_uses_inner_lula_solver():
    robot = SimpleNamespace(
        controllers={
            'arm_joint': JointController(),
            'arm_ik': WrappedIKController(),
        }
    )
    current_task = SimpleNamespace(
        step_counter=0,
        robots={'left': robot},
        get_tracked_object_states=lambda: {},
    )
    checker = StageTrajectoryPrechecker(check_stride=1, num_waypoints=4)

    checker.observe(
        current_task,
        {'left': {'arm_joint': [[0.1, 0, 0, 0, 0, 0]]}},
    )

    assert checker.finalize()['waypoints_checked'] == 4


def test_safe_joint_segment_checks_all_waypoints():
    checker = StageTrajectoryPrechecker(check_stride=1, num_waypoints=5)

    result = checker.observe(
        make_task(),
        {'left': {'arm_joint': [[0.2, 0, 0, 0, 0, 0]]}},
    )
    report = checker.finalize()

    assert result['checked'] is True
    assert report['segments_checked'] == 1
    assert report['waypoints_checked'] == 5
    assert report['violation_total'] == 0


def test_environment_collision_is_serialized():
    checker = StageTrajectoryPrechecker(check_stride=1, num_waypoints=3)
    checker.observe(
        make_task(
            object_states={
                'fixture': {
                    'position': [0.0, 0.0, 0.35],
                    'size': [0.1, 0.1, 0.1],
                    'orientation': [1.0, 0.0, 0.0, 0.0],
                }
            }
        ),
        {'left': {'arm_joint': [[0.0, 0, 0, 0, 0, 0]]}},
    )
    report = checker.finalize()

    assert report['violation_total'] > 0
    assert report['events'][0]['entity_b'] == 'fixture'


def test_stride_skips_without_running_fk():
    checker = StageTrajectoryPrechecker(check_stride=8, num_waypoints=3)

    result = checker.observe(
        make_task(step=1),
        {'left': {'arm_joint': [[0.0, 0, 0, 0, 0, 0]]}},
    )

    assert result['reason'] == 'stride_skip'
    assert checker.finalize()['checks'] == 0


def test_invalid_joint_target_is_fail_open():
    checker = StageTrajectoryPrechecker(check_stride=1, num_waypoints=3)

    result = checker.observe(
        make_task(),
        {'left': {'arm_joint': [[float('nan')] * 6]}},
    )

    assert result['reason'] == 'no_joint_targets'
    assert checker.finalize()['skip_reasons']['invalid_joint_target'] == 1


def test_franka_task_uses_panda_capsules_and_seven_dof_fk():
    panda_frames = [
        'panda_link0',
        'panda_link1',
        'panda_link2',
        'panda_link3',
        'panda_link4',
        'panda_link5',
        'panda_link6',
        'panda_link7',
        'panda_hand',
        'panda_leftfinger',
        'panda_rightfinger',
    ]

    class PandaSubset:
        def get_joint_positions(self):
            return np.zeros(7)

    class PandaJointController:
        config = SimpleNamespace(name='arm_joint_controller')

        def get_joint_subset(self):
            return PandaSubset()

    class PandaSolver(Solver):
        frames = panda_frames

    class PandaIKController(IKController):
        config = SimpleNamespace(name='arm_ik_controller')
        _kinematics_solver = PandaSolver()

    robot = SimpleNamespace(
        config=SimpleNamespace(type='FrankaRobot'),
        controllers={
            'arm_joint_controller': PandaJointController(),
            'arm_ik_controller': PandaIKController(),
        },
    )
    task = SimpleNamespace(
        step_counter=0,
        robots={'franka_left': robot},
        get_tracked_object_states=lambda: {},
    )
    checker = StageTrajectoryPrechecker(check_stride=1, num_waypoints=3)

    result = checker.observe(
        task,
        {'franka_left': {'arm_joint_controller': [[0.1] + [0.0] * 6]}},
    )
    report = checker.finalize()

    assert result['reason'] == 'ok'
    assert report['robot_model'] == 'franka_panda'
    assert report['segments_checked'] == 1
    assert report['waypoints_checked'] == 3
    assert report['monitor_error'] == []
