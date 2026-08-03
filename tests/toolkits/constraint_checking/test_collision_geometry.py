from __future__ import annotations

import numpy as np

from toolkits.constraint_checking.detector.collision import CollisionDetector, _closest_point_box


class FakeXform:
    def __init__(self, position):
        self.position = np.asarray(position, dtype=float)

    def get_world_pose(self):
        return self.position, np.array([1.0, 0.0, 0.0, 0.0])


def _detector() -> CollisionDetector:
    detector = CollisionDetector(threshold=0.05)
    detector.franka_links = ["link_a", "link_b"]
    detector.franka_capsules = [("link_a", "link_b", 0.03)]
    return detector


def _agent(y: float, z: float = 0.2):
    return {
        "link_a": FakeXform([0.0, y, z]),
        "link_b": FakeXform([1.0, y, z]),
    }


def test_detects_inter_arm_capsule_collision():
    detector = _detector()
    detector.add_agent("left", _agent(0.0))
    detector.add_agent("right", _agent(0.08))

    events = detector.check_inter_agent(step=12)

    assert len(events) == 1
    assert events[0].kind == "inter_agent"
    assert events[0].distance < detector.threshold


def test_detects_arm_box_and_ground_collision():
    detector = _detector()
    detector.add_agent("left", _agent(0.0, z=0.02))
    detector.add_box("fixture", center=[0.5, 0.06, 0.02], half_extents=[0.1, 0.02, 0.1])
    detector.add_ground(0.0)

    events = detector.check_agent_env(step=3)

    assert {event.entity_b for event in events} == {"fixture", "ground"}


def test_returns_no_events_when_clearances_are_safe():
    detector = _detector()
    detector.add_agent("left", _agent(0.0, z=0.5))
    detector.add_agent("right", _agent(0.4, z=0.5))
    detector.add_box("fixture", center=[0.5, 0.8, 0.5], half_extents=[0.05, 0.05, 0.05])
    detector.add_ground(0.0)

    assert detector.check_all(step=5) == []


def test_oriented_box_uses_cached_rotation_matrix_geometry():
    half_angle = np.sqrt(0.5)
    closest = _closest_point_box(
        point=np.array([0.0, 2.0, 0.0]),
        center=np.zeros(3),
        half_extents=np.array([1.0, 0.5, 0.5]),
        orient=np.array([half_angle, 0.0, 0.0, half_angle]),
    )

    assert np.allclose(closest, [0.0, 1.0, 0.0], atol=1e-9)
