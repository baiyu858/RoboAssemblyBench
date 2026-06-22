from pathlib import Path

from roboassemblybench.robobrain.executor import RoboBrainRunConfig, RoboBrainRunner
from roboassemblybench.robobrain.webapp import (
    DEFAULT_TEMPLATE,
    AgentRunRequest,
    serialize_inventory,
    summarize_result,
)
from roboassemblybench.robobrain.manual_demo import build_manual_demo_payload, write_manual_demo_bundle


def test_agent_run_request_defaults_to_safe_plan_only(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)

    request = AgentRunRequest.from_payload({'task': '使用 ur5e 双臂进行零件组装'})
    config = request.to_runner_config(lambda event: None)

    assert request.mock_llm
    assert request.template == DEFAULT_TEMPLATE
    assert config.plan_only
    assert not config.capture_demo_output


def test_agent_run_request_enables_demo_streaming_when_requested():
    request = AgentRunRequest.from_payload({'task': 'assemble parts', 'run_simulation': True, 'mock_llm': True})
    config = request.to_runner_config(lambda event: None)

    assert not config.plan_only
    assert config.capture_demo_output


def test_serialize_inventory_exposes_default_template():
    inventory = serialize_inventory('taoyuan_grscenes_tabletop', DEFAULT_TEMPLATE)

    assert inventory['selected_template'] == DEFAULT_TEMPLATE
    assert any(item['name'] == DEFAULT_TEMPLATE for item in inventory['templates'])
    assert inventory['selected_recipe']['phase_count'] > 0
    assert inventory['asset_catalog']


def test_runner_event_callback_receives_static_trace(tmp_path: Path):
    events = []
    result = RoboBrainRunner(
        RoboBrainRunConfig(
            output_dir=tmp_path,
            selected_template='peg_insertion',
            mock_llm=True,
            plan_only=True,
            event_callback=events.append,
        )
    ).run('right Franka inserts a peg into the socket')

    event_types = {event['type'] for event in events}
    assert result.check_result.ok
    assert 'inventory_loaded' in event_types
    assert 'plan_generated' in event_types
    assert 'checker_completed' in event_types
    assert 'bundle_written' in event_types


def test_summarize_result_lists_artifact_files(tmp_path: Path):
    result = RoboBrainRunner(
        RoboBrainRunConfig(
            output_dir=tmp_path,
            selected_template='peg_insertion',
            mock_llm=True,
            plan_only=True,
        )
    ).run('right Franka inserts a peg into the socket')

    summary = summarize_result(result)

    assert summary['primitive_plan']
    assert summary['recipe_summary']['phase_count'] > 0
    assert {item['label'] for item in summary['artifacts']} >= {'recipe', 'plan', 'checker_report'}
    assert all(item['kind'] == 'file' for item in summary['artifacts'])


def test_manual_demo_builds_plumbers_block_trace():
    payload = build_manual_demo_payload(
        task_instruction='manual plumbers block demo',
        scene_profile='taoyuan_grscenes_tabletop',
    )

    assert payload['plan']['source'] == 'manual_demo'
    assert payload['check_result']['ok']
    assert payload['recipe_summary']['phase_count'] == 7
    assert len(payload['manual_reasoning_trace']) >= 8
    assert payload['manual_reasoning_trace'][0]['title'] == '理解用户任务'
    assert '我正在进行' in payload['manual_reasoning_trace'][0]['thinking_process']
    assert payload['manual_skill_steps'][0]['execution_skill'] == 'ur5e_move_above_part'
    assert payload['manual_skill_steps'][2]['operation_arm'] == 'franka_right'


def test_manual_demo_writes_bundle(tmp_path: Path):
    payload = write_manual_demo_bundle(
        output_root=tmp_path,
        task_instruction='manual plumbers block demo',
        scene_profile='taoyuan_grscenes_tabletop',
    )

    paths = payload['bundle_paths']
    assert Path(paths['manual_reasoning_trace']).exists()
    assert Path(paths['manual_skill_steps']).exists()
    assert Path(paths['manual_demo_report']).exists()
    assert payload['check_result']['ok']
