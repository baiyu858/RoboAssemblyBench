from types import SimpleNamespace

from toolkits.constraint_checking.integration.pipeline import RuntimeConstraintEpisodeHook


class FakeMonitor:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.steps = []

    def observe(self, task):
        if self.fail:
            raise RuntimeError("synthetic collision failure")
        self.steps.append(task.step_counter)
        return {"checked": True, "step": task.step_counter, "violations": []}

    def finalize(self):
        return {
            "enabled": True,
            "checks": len(self.steps),
            "violation_total": 0,
            "events": [],
            "monitor_error": [],
        }


def test_disabled_hook_leaves_metrics_identical():
    metrics = {"success": True, "status": "success", "terminal_reason": "complete"}
    hook = RuntimeConstraintEpisodeHook(enabled=False, monitor_factory=lambda: (_ for _ in ()).throw(RuntimeError()))

    assert hook.observe(SimpleNamespace(step_counter=1)) is None
    assert hook.attach_metrics(metrics) is metrics
    assert metrics == {"success": True, "status": "success", "terminal_reason": "complete"}


def test_enabled_hook_adds_only_constraint_metrics_and_resets_episode():
    created = []

    def factory():
        monitor = FakeMonitor()
        created.append(monitor)
        return monitor

    hook = RuntimeConstraintEpisodeHook(enabled=True, monitor_factory=factory)
    task = SimpleNamespace(step_counter=8)
    hook.observe(task)
    metrics = {"success": True, "status": "success", "terminal_reason": "complete"}
    hook.attach_metrics(metrics)

    assert metrics["success"] is True
    assert metrics["status"] == "success"
    assert metrics["terminal_reason"] == "complete"
    assert metrics["runtime_constraint_monitor"]["checks"] == 1

    hook.reset_episode()
    task.step_counter = 16
    hook.observe(task)
    assert len(created) == 2
    assert created[0] is not created[1]


def test_monitor_exception_is_recorded_without_changing_success():
    hook = RuntimeConstraintEpisodeHook(enabled=True, monitor_factory=lambda: FakeMonitor(fail=True))
    metrics = {"success": True, "status": "success"}

    result = hook.observe(SimpleNamespace(step_counter=24))
    hook.attach_metrics(metrics)

    assert result["violations"] == []
    assert metrics["success"] is True
    assert metrics["status"] == "success"
    assert metrics["runtime_constraint_monitor"]["monitor_error"][0]["stage"] == "observe"
