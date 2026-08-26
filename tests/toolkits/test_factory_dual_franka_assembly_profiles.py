import json

import pytest

from internutopia.core.scene.isaacsim.scene import IsaacsimScene
from internutopia_extension.objects.usd_object import UsdObject
from toolkits.factory_dual_franka_assembly.convert_dataset import (
    build_dataset_entries,
    load_episode_payloads,
)
from toolkits.factory_dual_franka_assembly.scene_builder import (
    build_dual_franka_assembly_episode,
)
from toolkits.factory_dual_franka_assembly.scene_profiles import list_scene_profiles
from toolkits.factory_dual_franka_assembly.task_specs import load_task_recipe

ISAAC_PACKING_TABLE_MARKER = '/Isaac/Props/PackingTable/'
FABRICA_UR5E_TASKS = (
    'beam',
    'car',
    'cooling_manifold',
    'duct',
    'gamepad',
    'plumbers_block',
    'stool_circular',
)


def test_isaac_scene_asset_root_can_be_overridden_with_a_local_directory(tmp_path, monkeypatch):
    assets_root = tmp_path / 'isaac_sim_5.1'
    warehouse_asset = assets_root / 'Isaac/Environments/Simple_Warehouse/warehouse.usd'
    warehouse_asset.parent.mkdir(parents=True)
    warehouse_asset.touch()
    monkeypatch.setenv('ISAAC_ASSETS_ROOT', str(assets_root))

    resolved = IsaacsimScene._resolve_isaac_asset_path(
        '${ISAAC_ASSETS_ROOT}/Isaac/Environments/Simple_Warehouse/warehouse.usd'
    )

    assert resolved == str(warehouse_asset)


def test_usd_object_rejects_a_missing_local_asset(tmp_path):
    missing_asset = tmp_path / 'missing.usd'

    with pytest.raises(FileNotFoundError, match='Object USD asset does not exist'):
        UsdObject._resolve_usd_path(str(missing_asset))


def test_usd_object_resolves_an_existing_local_asset(tmp_path):
    asset = tmp_path / 'part.usd'
    asset.touch()

    assert UsdObject._resolve_usd_path(str(asset)) == str(asset)


def test_scene_profiles_are_discoverable():
    profiles = list_scene_profiles()
    assert 'proxy_factory_cell' in profiles
    assert 'taoyuan_tabletop' in profiles
    assert 'taoyuan_grscenes_tabletop' in profiles


def test_taoyuan_scene_profile_injects_assets_and_workspace_offset():
    recipe_spec = load_task_recipe('screw_fastening', scene_profile='taoyuan_tabletop')
    assert recipe_spec['scene_profile'] == 'taoyuan_tabletop'
    assert any(object_spec['name'] == 'taoyuan_table' for object_spec in recipe_spec['objects'])
    assert any(
        reference['path'].endswith('/objects/table/white_big/instance.usd')
        for reference in recipe_spec['asset_references']
    )

    task_cfg = build_dual_franka_assembly_episode(
        recipe='screw_fastening',
        seed=1,
        episode_idx=0,
        scene_profile='taoyuan_tabletop',
    )
    assert task_cfg.scene_profile == 'taoyuan_tabletop'
    assert task_cfg.workspace_offset == [0.0, 0.0, 0.78]
    assert any(object_cfg.name == 'taoyuan_table' for object_cfg in task_cfg.objects)
    assert task_cfg.target_poses['left_wait']['position'][2] > 1.0


def test_taoyuan_grscenes_scene_profile_omits_isaac_factory_table_xform():
    recipe = 'fabrica_plumbers_block_ur5e_staged'
    recipe_spec = load_task_recipe(recipe, scene_profile='taoyuan_grscenes_tabletop')
    assert recipe_spec['scene_profile'] == 'taoyuan_grscenes_tabletop'
    assert recipe_spec['metadata']['scene_family'] == 'isaac_simple_warehouse_tabletop'
    assert recipe_spec['scene_asset_path'].endswith('/warehouse_with_forklifts.usd')
    assert {light['name'] for light in recipe_spec['scene_lights']} == {'warehouse_dome_fill'}
    assert not any(ISAAC_PACKING_TABLE_MARKER in reference['path'] for reference in recipe_spec['asset_references'])
    assert not any(object_spec.get('prim_path') == '/factory_packing_table' for object_spec in recipe_spec['objects'])
    assert any(object_spec['name'] == 'factory_tabletop_visual' for object_spec in recipe_spec['objects'])
    assert any(object_spec['name'] == 'factory_background_visual' for object_spec in recipe_spec['objects'])

    task_cfg = build_dual_franka_assembly_episode(
        recipe=recipe,
        seed=3,
        episode_idx=0,
        scene_profile='taoyuan_grscenes_tabletop',
    )
    assert task_cfg.scene_profile == 'taoyuan_grscenes_tabletop'
    assert task_cfg.workspace_offset == [0.0, 0.0, 0.99]
    assert {light['name'] for light in task_cfg.scene_lights} == {'warehouse_dome_fill'}
    assert not any(ISAAC_PACKING_TABLE_MARKER in reference['path'] for reference in task_cfg.asset_references)
    assert not any(object_cfg.prim_path == '/factory_packing_table' for object_cfg in task_cfg.objects)
    assert any(object_cfg.name == 'factory_tabletop_visual' for object_cfg in task_cfg.objects)
    assert any(object_cfg.name == 'factory_background_visual' for object_cfg in task_cfg.objects)
    required_renderable_names = {
        'optical_board',
        'fabrica_fixture',
        'fabrica_plumbers_block_0',
        'fabrica_plumbers_block_1',
        'fabrica_plumbers_block_2',
        'fabrica_plumbers_block_3',
        'fabrica_plumbers_block_4',
    }
    required_renderable_objects = {
        object_cfg.name: object_cfg
        for object_cfg in task_cfg.objects
        if object_cfg.name in required_renderable_names
    }
    assert set(required_renderable_objects) == required_renderable_names
    assert all(object_cfg.force_renderable is True for object_cfg in required_renderable_objects.values())


@pytest.mark.parametrize('task_name', FABRICA_UR5E_TASKS)
def test_all_fabrica_ur5e_assets_are_required_to_be_renderable(task_name):
    task_cfg = build_dual_franka_assembly_episode(
        recipe=f'fabrica_{task_name}_ur5e_staged',
        seed=0,
        episode_idx=0,
        scene_profile='taoyuan_grscenes_tabletop',
    )
    required_objects = [
        object_cfg
        for object_cfg in task_cfg.objects
        if object_cfg.name in {'optical_board', 'fabrica_fixture'}
        or object_cfg.name.startswith(f'fabrica_{task_name}_')
    ]

    assert {object_cfg.name for object_cfg in required_objects} >= {'optical_board', 'fabrica_fixture'}
    assert any(object_cfg.name.startswith(f'fabrica_{task_name}_') for object_cfg in required_objects)
    assert all(object_cfg.force_renderable is True for object_cfg in required_objects)


def test_80hz_control_scales_phase_counts_and_motion_steps_from_240hz():
    recipe = 'fabrica_plumbers_block_ur5e_staged'
    baseline = build_dual_franka_assembly_episode(recipe=recipe, seed=3, control_fps=240)
    reduced = build_dual_franka_assembly_episode(recipe=recipe, seed=3, control_fps=80)

    assert reduced.max_steps == round(baseline.max_steps / 3)
    assert reduced.phase_timeout_steps == round(baseline.phase_timeout_steps / 3)
    baseline_phase = next(
        phase for phase in baseline.phase_specs if (phase.get('local_skill') or {}).get('cartesian_position_step')
    )
    reduced_phase = next(phase for phase in reduced.phase_specs if phase['name'] == baseline_phase['name'])
    assert reduced_phase['timeout_steps'] == round(baseline_phase['timeout_steps'] / 3)
    assert reduced_phase['local_skill']['cartesian_position_step'] == (
        baseline_phase['local_skill']['cartesian_position_step'] * 3
    )

    baseline_insertion = next(
        phase
        for phase in baseline.phase_specs
        if '_insert_' in phase['name']
        and (phase.get('local_skill') or {}).get('compliant_servo_position_command_accumulation_step')
    )
    reduced_insertion = next(
        phase for phase in reduced.phase_specs if phase['name'] == baseline_insertion['name']
    )
    baseline_skill = baseline_insertion['local_skill']
    reduced_skill = reduced_insertion['local_skill']
    assert reduced_skill['target_object_servo_position_command_accumulation_step'] == (
        baseline_skill['target_object_servo_position_command_accumulation_step'] * 3
    )
    assert reduced_skill['compliant_servo_position_command_accumulation_step'] == (
        baseline_skill['compliant_servo_position_command_accumulation_step'] * 3
    )
    assert reduced_skill['compliant_servo_max_lateral_step'] == (
        baseline_skill['compliant_servo_max_lateral_step'] * 3
    )
    assert reduced_skill['target_object_lateral_alignment_enter_tolerance'] == (
        reduced_skill['target_object_lateral_position_tolerance']
    )


@pytest.mark.parametrize(
    ('recipe', 'terminal_phase'),
    [
        ('fabrica_gamepad_ur5e_staged', 'assemble_04_part_4_release_and_lock'),
        ('fabrica_plumbers_block_ur5e_staged', 'assemble_03_part_4_release_and_lock'),
    ],
)
def test_80hz_long_fabrica_tasks_reserve_terminal_phase_budget(recipe, terminal_phase):
    task = build_dual_franka_assembly_episode(recipe=recipe, seed=3, control_fps=80)

    assert task.phase_specs[-1]['name'] == terminal_phase
    assert task.max_steps >= round(len(task.phase_specs) * 420 / 3)


def test_asset_backed_recipes_default_to_taoyuan_tabletop():
    screw_fastening = load_task_recipe('screw_fastening')
    peg_insertion = load_task_recipe('peg_insertion')

    assert screw_fastening['scene_profile'] == 'taoyuan_tabletop'
    assert peg_insertion['scene_profile'] == 'taoyuan_grscenes_tabletop'
    assert any(
        reference['path'].endswith('/objects/table/white_big/instance.usd')
        for reference in screw_fastening['asset_references']
    )
    assert not any(ISAAC_PACKING_TABLE_MARKER in reference['path'] for reference in peg_insertion['asset_references'])


def test_convert_dataset_recurses_profile_directories(tmp_path):
    recipe_dir = tmp_path / 'taoyuan_tabletop' / 'screw_fastening'
    recipe_dir.mkdir(parents=True)
    episode_payload = {
        'episode_idx': 0,
        'seed': 5,
        'recipe': 'screw_fastening',
        'prompt': 'demo prompt',
        'scene_profile': 'taoyuan_tabletop',
        'scene_asset_path': '/scene.usd',
        'workspace_offset': [0.0, 0.0, 0.78],
        'asset_references': [{'path': '/table.usd'}],
        'metadata': {'scene_family': 'taoyuan'},
        'metrics': {'success': True},
        'steps': [
            {'phase': 'phase_a', 'observations': {}, 'actions': {}, 'objects': {}},
            {'phase': 'phase_a', 'observations': {}, 'actions': {}, 'objects': {}},
            {'phase': 'phase_b', 'observations': {}, 'actions': {}, 'objects': {}},
        ],
    }
    (recipe_dir / 'episode_0000.json').write_text(json.dumps(episode_payload), encoding='utf-8')

    entries = build_dataset_entries(load_episode_payloads(tmp_path))
    assert len(entries) == 1
    assert entries[0]['scene_profile'] == 'taoyuan_tabletop'
    assert entries[0]['phase_segments'] == [
        {'phase': 'phase_a', 'start_step': 0, 'end_step': 1, 'num_steps': 2},
        {'phase': 'phase_b', 'start_step': 2, 'end_step': 2, 'num_steps': 1},
    ]
