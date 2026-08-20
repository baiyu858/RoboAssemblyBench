from types import SimpleNamespace

from toolkits.constraint_checking.integration.models import (
    FRANKA_MODEL,
    UR5E_ROBOTIQ_MODEL,
    get_robot_collision_model,
    infer_task_robot_collision_model,
)


def test_ur5e_model_has_expected_capsules_and_threshold():
    model = get_robot_collision_model('UR5eRobot')

    assert model is UR5E_ROBOTIQ_MODEL
    assert model.default_threshold == 0.03
    assert ('shoulder_link', 'upper_arm_link', 0.07) in model.detector_capsules
    assert (
        'Gripper/Robotiq_2F_85/left_outer_knuckle',
        'Gripper/Robotiq_2F_85/left_outer_finger',
        0.02,
    ) in model.detector_capsules
    assert (
        'Gripper/Robotiq_2F_85/right_outer_finger',
        'Gripper/Robotiq_2F_85/right_inner_finger',
        0.02,
    ) in model.detector_capsules
    assert (
        'wrist_3_link',
        'Gripper/Robotiq_2F_85/base_link',
        0.072,
    ) in model.detector_capsules
    assert (
        'Gripper/Robotiq_2F_85/left_outer_knuckle',
        'Gripper/Robotiq_2F_85/right_outer_knuckle',
        0.03,
    ) in model.detector_capsules


def test_gripper_links_have_root_and_wrist_mount_prim_candidates():
    paths = UR5E_ROBOTIQ_MODEL.prim_paths_for_link(
        '/ur5e_left',
        'Gripper/Robotiq_2F_85/base_link',
    )

    assert paths[0] == '/ur5e_left/Gripper/Robotiq_2F_85/base_link'
    assert '/ur5e_left/wrist_3_link/Gripper/Robotiq_2F_85/base_link' in paths
    assert (
        UR5E_ROBOTIQ_MODEL.prim_path_for_link(
            '/ur5e_left/',
            'shoulder_link',
        )
        == '/ur5e_left/shoulder_link'
    )

    outer_paths = UR5E_ROBOTIQ_MODEL.prim_paths_for_link(
        '/ur5e_left',
        'Gripper/Robotiq_2F_85/left_outer_finger',
    )
    assert outer_paths[0] == '/ur5e_left/Gripper/Robotiq_2F_85/left_outer_finger'
    assert '/ur5e_left/wrist_3_link/Gripper/Robotiq_2F_85/left_outer_finger' in outer_paths


def test_franka_robot_type_selects_panda_collision_model():
    task = SimpleNamespace(
        robots={
            'left': SimpleNamespace(config=SimpleNamespace(type='FrankaRobot', name='franka_left')),
            'right': SimpleNamespace(config=SimpleNamespace(type='FrankaRobot', name='franka_right')),
        }
    )

    assert get_robot_collision_model('FrankaRobot') is FRANKA_MODEL
    assert infer_task_robot_collision_model(task) is FRANKA_MODEL
