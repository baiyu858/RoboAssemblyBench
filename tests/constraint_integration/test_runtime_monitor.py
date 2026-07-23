from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from constraint_integration import runtime_monitor
from constraint_integration.runtime_monitor import PairFilter, RuntimeConstraintConfig, RuntimeConstraintMonitor


class FakeXform:
    def __init__(self, prim_path):
        self.prim_path = prim_path

    def get_world_pose(self):
        if "right_inner_finger" in self.prim_path:
            raise RuntimeError("missing test prim")
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])


class FakeDetector:
    emitted_events = []
    raise_on_check = False

    def __init__(self, threshold):
        self.threshold = threshold
        self.franka_links = []
        self.franka_capsules = []
        self._agents = {}
        self._boxes = {}
        self._ground_z = None

    def add_agent(self, name, links):
        self._agents[name] = links

    def add_box(self, name, center, half_extents, orient=None):
        self._boxes[name] = {
            "center": np.asarray(center),
            "half_extents": np.asarray(half_extents),
            "orient": orient,
        }

    def add_ground(self, z):
        self._ground_z = z

    def check_all(self, step=0):
        if self.raise_on_check:
            raise RuntimeError("detector failed")
        return list(self.emitted_events)

    @staticmethod
    def summary(events):
        return f"{len(events)} events"


def _event(kind="agent_env", entity_a="left/arm", entity_b="fixture", distance=0.01):
    return SimpleNamespace(
        step=8,
        kind=kind,
        entity_a=entity_a,
        entity_b=entity_b,
        distance=distance,
        threshold=0.03,
        pos_a=np.array([1.0, 2.0, 3.0]),
        pos_b=np.array([1.0, 2.01, 3.0]),
    )


class FakeTask:
    def __init__(self, step_counter=1):
        self.step_counter = step_counter
        self.phase_specs = [
            {
                "local_skill": {
                    "object": "free_part",
                    "contact_box_scale": [0.1, 0.2, 0.3],
                }
            }
        ]
        self.robots = {
            "left": SimpleNamespace(config=SimpleNamespace(prim_path="/ur5e_left")),
            "right": SimpleNamespace(config=SimpleNamespace(prim_path="/ur5e_right")),
        }

    def get_tracked_object_states(self):
        return {
            "free_part": {
                "position": [0.4, 0.0, 1.0],
                "orientation": [1.0, 0.0, 0.0, 0.0],
                "scale": [0.1, 0.2, 0.3],
                "attached_to": None,
            },
            "held_part": {
                "position": [0.2, 0.0, 1.0],
                "scale": [0.1, 0.1, 0.1],
                "attached_to": "left",
            },
        }


def _install_fakes(monkeypatch):
    FakeDetector.emitted_events = []
    FakeDetector.raise_on_check = False
    monkeypatch.setattr(runtime_monitor, "_load_collision_detector_cls", lambda: FakeDetector)
    monkeypatch.setattr(runtime_monitor, "_load_xform_reader_cls", lambda: FakeXform)


def test_stride_registration_environment_filtering_and_summary(monkeypatch):
    _install_fakes(monkeypatch)
    FakeDetector.emitted_events = [
        _event(entity_b="fixture", distance=-0.001),
        _event(entity_a="left/gripper", entity_b="allowed_part", distance=-0.002),
    ]
    monitor = RuntimeConstraintMonitor(
        RuntimeConstraintConfig(
            check_stride=2,
            include_ground=True,
            ignore_pairs=[PairFilter("gripper", "allowed_part")],
        )
    )
    task = FakeTask(step_counter=1)

    assert monitor.observe(task)["reason"] == "stride_skip"
    task.step_counter = 2
    result = monitor.observe(task)
    report = monitor.finalize()

    assert result["checked"] is True
    assert len(result["violations"]) == 1
    assert report["checks"] == 1
    assert report["total_check_seconds"] >= 0.0
    assert report["average_check_seconds"] >= 0.0
    assert report["max_check_seconds"] >= 0.0
    assert report["violation_total"] == 1
    assert report["candidate_total"] == 2
    assert report["classifications"] == {"allowed_contact": 1, "collision": 1}
    assert report["violations_by_kind"] == {"agent_env": 1}
    assert report["minimum_distance"] == -0.001
    assert report["registered_robots"] == {"left": "/ur5e_left", "right": "/ur5e_right"}
    assert {item["link"] for item in report["missing_prims"]} == {
        "Gripper/Robotiq_2F_85/right_inner_finger"
    }
    assert set(monitor.detector._boxes) == {"free_part"}
    assert monitor.detector._boxes["free_part"]["half_extents"].tolist() == [0.05, 0.1, 0.15]
    assert monitor.detector._ground_z == 0.0
    json.dumps(report, allow_nan=False)


def test_unknown_object_scale_is_not_treated_as_metric_size(monkeypatch):
    _install_fakes(monkeypatch)
    task = FakeTask(step_counter=8)
    task.phase_specs = []
    monitor = RuntimeConstraintMonitor(RuntimeConstraintConfig())

    monitor.observe(task)
    report = monitor.finalize()

    assert monitor.detector._boxes == {}
    assert report["missing_object_geometry"] == ["free_part"]


def test_detailed_events_are_capped_while_totals_are_preserved(monkeypatch):
    _install_fakes(monkeypatch)
    FakeDetector.emitted_events = [_event(distance=-(value + 1) / 1000) for value in range(10)]
    monitor = RuntimeConstraintMonitor(RuntimeConstraintConfig(max_recorded_events=3))

    monitor.observe(FakeTask(step_counter=8))
    report = monitor.finalize()

    assert report["violation_total"] == 10
    assert len(report["events"]) == 3
    assert report["events_dropped"] == 7


def test_positive_clearance_is_kept_as_auditable_proximity(monkeypatch):
    _install_fakes(monkeypatch)
    FakeDetector.emitted_events = [_event(distance=0.009)]
    monitor = RuntimeConstraintMonitor(RuntimeConstraintConfig())

    result = monitor.observe(FakeTask(step_counter=8))
    report = monitor.finalize()

    assert result["violations"] == []
    assert result["proximity_count"] == 1
    assert report["candidate_total"] == 1
    assert report["violation_total"] == 0
    assert report["classifications"] == {"proximity": 1}
    assert report["proximity_events"][0]["classification_reason"] == "positive_surface_clearance"


def test_detector_exception_is_fail_open(monkeypatch):
    _install_fakes(monkeypatch)
    FakeDetector.raise_on_check = True
    monitor = RuntimeConstraintMonitor(RuntimeConstraintConfig(check_stride=1))

    result = monitor.observe(FakeTask(step_counter=4))
    report = monitor.finalize()

    assert result["violations"] == []
    assert result["monitor_error"]["stage"] == "observe"
    assert report["checks"] == 1
    assert report["violation_total"] == 0
    assert report["monitor_error"][0]["message"] == "detector failed"


def test_new_monitor_does_not_reuse_previous_episode_state(monkeypatch):
    _install_fakes(monkeypatch)
    FakeDetector.emitted_events = [_event(distance=-0.001)]
    first = RuntimeConstraintMonitor(RuntimeConstraintConfig())
    first.observe(FakeTask(step_counter=8))

    FakeDetector.emitted_events = []
    second = RuntimeConstraintMonitor(RuntimeConstraintConfig())

    assert first.finalize()["violation_total"] == 1
    assert second.finalize()["violation_total"] == 0
