from types import SimpleNamespace

from toolkits.constraint_checking.integration.contact_policy import AssemblyContactPolicy


def _event(entity_a, entity_b, distance=-0.001):
    return SimpleNamespace(entity_a=entity_a, entity_b=entity_b, distance=distance)


class PhaseTask:
    phase = "left_close_gripper_on_block_4"

    def get_current_phase_spec(self):
        return {
            "name": self.phase,
            "local_skill": {
                "robot": "franka_left",
                "object": "fabrica_plumbers_block_4",
            },
        }


def test_positive_clearance_is_proximity_not_collision():
    decision = AssemblyContactPolicy().classify(
        _event("franka_left/forearm_link->wrist_1_link", "block_3", distance=0.009),
        PhaseTask(),
    )

    assert decision.classification == "proximity"
    assert decision.reason == "positive_surface_clearance"


def test_target_gripper_contact_is_allowed_during_grasp():
    decision = AssemblyContactPolicy().classify(
        _event(
            "franka_left/wrist_3_link->Gripper/Robotiq_2F_85/base_link",
            "fabrica_plumbers_block_4",
        ),
        PhaseTask(),
    )

    assert decision.classification == "allowed_contact"
    assert decision.reason == "phase_target_end_effector_contact"


def test_target_outer_finger_contact_is_allowed_during_grasp():
    decision = AssemblyContactPolicy().classify(
        _event(
            "franka_left/Gripper/Robotiq_2F_85/left_outer_knuckle"
            "->Gripper/Robotiq_2F_85/left_outer_finger",
            "fabrica_plumbers_block_4",
        ),
        PhaseTask(),
    )

    assert decision.classification == "allowed_contact"


def test_non_target_overlap_remains_abnormal_collision():
    decision = AssemblyContactPolicy().classify(
        _event(
            "franka_left/wrist_3_link->Gripper/Robotiq_2F_85/base_link",
            "fabrica_plumbers_block_3",
        ),
        PhaseTask(),
    )

    assert decision.classification == "collision"


def test_robot_robot_overlap_is_not_treated_as_assembly_contact():
    decision = AssemblyContactPolicy().classify(
        _event(
            "franka_left/wrist_3_link->Gripper/Robotiq_2F_85/base_link",
            "franka_right/wrist_3_link->Gripper/Robotiq_2F_85/base_link",
        ),
        PhaseTask(),
    )

    assert decision.classification == "collision"
