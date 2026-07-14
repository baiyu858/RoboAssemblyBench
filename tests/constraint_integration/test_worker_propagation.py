from pathlib import Path

from toolkits.factory_dual_franka_assembly import generate_demos


def test_worker_command_propagates_constraint_monitor_options(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, check):
        captured["command"] = command
        captured["check"] = check

    monkeypatch.setattr(generate_demos.subprocess, "run", fake_run)
    generate_demos._invoke_worker(
        mode="collect",
        recipe="recipe",
        scene_profile="profile",
        headless=True,
        output_dir=tmp_path / "output",
        results_path=tmp_path / "results.json",
        start_seed=0,
        max_trials=1,
        seeds=[3],
        runtime_constraint_monitor=True,
        constraint_check_stride=11,
        constraint_threshold=0.025,
        constraint_include_ground=True,
        constraint_ignore_pairs=["gripper:held_part", "insert:slot"],
    )

    command = captured["command"]
    assert captured["check"] is True
    assert "--runtime-constraint-monitor" in command
    assert command[command.index("--constraint-check-stride") + 1] == "11"
    assert command[command.index("--constraint-threshold") + 1] == "0.025"
    assert "--constraint-include-ground" in command
    ignore_values = [command[index + 1] for index, value in enumerate(command) if value == "--constraint-ignore-pair"]
    assert ignore_values == ["gripper:held_part", "insert:slot"]


def test_worker_command_omits_constraint_options_when_disabled(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(
        generate_demos.subprocess,
        "run",
        lambda command, check: captured.update(command=command, check=check),
    )

    generate_demos._invoke_worker(
        mode="search",
        recipe="recipe",
        scene_profile=None,
        headless=True,
        output_dir=Path(tmp_path),
        results_path=tmp_path / "results.json",
        start_seed=0,
        max_trials=1,
    )

    assert "--runtime-constraint-monitor" not in captured["command"]
    assert "--constraint-check-stride" not in captured["command"]
