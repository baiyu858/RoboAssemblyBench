from types import SimpleNamespace

from toolkits.constraint_checking.integration.precheck_pipeline import PassivePrecheckEpisodeHook


class FakeStageChecker:
    def __init__(self):
        self.calls = []

    def observe(self, task, actions):
        self.calls.append((task, actions))

    def finalize(self):
        return {
            "enabled": True,
            "mode": "passive",
            "checks": len(self.calls),
            "violation_total": 1,
            "events": [{"kind": "agent_env"}],
            "monitor_error": [],
        }


def task(phases):
    return SimpleNamespace(
        config=SimpleNamespace(
            phase_specs=phases,
            robot_names=["left", "right"],
            object_names=["part"],
        )
    )


def test_hook_attaches_both_reports_without_changing_actions():
    stage = FakeStageChecker()
    hook = PassivePrecheckEpisodeHook(
        sequence_enabled=True,
        stage_enabled=True,
        stage_factory=lambda: stage,
    )
    actions = {"left": {"joint": [[0.0] * 6]}}
    original = repr(actions)

    hook.observe_before_step(
        task(
            [
                {"name": "grasp", "attach": [{"robot": "left", "object": "part"}]},
                {"name": "place", "lock": [{"object": "part"}]},
            ]
        ),
        actions,
    )
    metrics = {"success": True, "status": "success"}
    hook.attach_metrics(metrics)

    assert repr(actions) == original
    assert metrics["success"] is True
    assert metrics["status"] == "success"
    assert metrics["assembly_sequence_precheck"]["feasible"] is True
    assert metrics["stage_trajectory_precheck"]["checks"] == 1


def test_sequence_runs_only_once_per_episode():
    hook = PassivePrecheckEpisodeHook(sequence_enabled=True)
    current_task = task([])

    hook.observe_before_step(current_task, {})
    first_report = hook._sequence_report
    hook.observe_before_step(current_task, {})

    assert hook._sequence_report is first_report


def test_failure_is_fail_open_and_serialized():
    class BrokenChecker:
        def check(self, **kwargs):
            raise RuntimeError("boom")

    hook = PassivePrecheckEpisodeHook(
        sequence_enabled=True,
        sequence_factory=BrokenChecker,
    )
    metrics = {"success": True}

    hook.observe_before_step(task([]), {})
    hook.attach_metrics(metrics)

    assert metrics["success"] is True
    assert metrics["assembly_sequence_precheck"]["monitor_error"][0]["message"] == "boom"


def test_reset_creates_fresh_episode_state():
    hook = PassivePrecheckEpisodeHook(sequence_enabled=True)
    current_task = task([])
    hook.observe_before_step(current_task, {})

    hook.reset_episode()

    assert hook._sequence_report is None
    assert hook._stage_checker is None
