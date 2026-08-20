from types import SimpleNamespace

import numpy as np

from internutopia_extension.controllers.gripper_controller import GripperController


class _FakeGripper:
    joint_opened_positions = np.asarray([0.04, 0.04], dtype=float)
    joint_closed_positions = np.asarray([0.0, 0.0], dtype=float)
    active_joint_indices = [7, 8]

    def forward(self, _action):
        raise AssertionError('endpoint commands must use explicit indexed joint targets')


def _controller() -> GripperController:
    controller = GripperController.__new__(GripperController)
    controller._gripper = _FakeGripper()
    controller._robot_config = SimpleNamespace(gripper_close_openness=0.0)
    return controller


def test_open_command_targets_configured_open_joint_positions():
    action = _controller().forward('open')

    np.testing.assert_allclose(action.joint_positions, [0.04, 0.04])
    np.testing.assert_array_equal(action.joint_indices, [7, 8])


def test_numeric_open_endpoint_uses_same_explicit_joint_target():
    action = _controller().forward(1.0)

    np.testing.assert_allclose(action.joint_positions, [0.04, 0.04])
    np.testing.assert_array_equal(action.joint_indices, [7, 8])
