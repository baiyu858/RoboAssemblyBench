import json

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
    recipe_spec = load_task_recipe('screw_fastening', scene_profile='taoyuan_grscenes_tabletop')
    assert recipe_spec['scene_profile'] == 'taoyuan_grscenes_tabletop'
    assert recipe_spec['metadata']['scene_family'] == 'factory_tabletop_visual'
    assert {light['name'] for light in recipe_spec['scene_lights']} == {
        'warehouse_dome_fill',
        'warehouse_sun_fill',
    }
    assert not any(ISAAC_PACKING_TABLE_MARKER in reference['path'] for reference in recipe_spec['asset_references'])
    assert not any(object_spec.get('prim_path') == '/factory_packing_table' for object_spec in recipe_spec['objects'])
    assert any(object_spec['name'] == 'factory_tabletop_visual' for object_spec in recipe_spec['objects'])

    task_cfg = build_dual_franka_assembly_episode(
        recipe='screw_fastening',
        seed=3,
        episode_idx=0,
        scene_profile='taoyuan_grscenes_tabletop',
    )
    assert task_cfg.scene_profile == 'taoyuan_grscenes_tabletop'
    assert task_cfg.workspace_offset == [0.0, 0.0, 0.99]
    assert {light['name'] for light in task_cfg.scene_lights} == {
        'warehouse_dome_fill',
        'warehouse_sun_fill',
    }
    assert not any(ISAAC_PACKING_TABLE_MARKER in reference['path'] for reference in task_cfg.asset_references)
    assert not any(object_cfg.prim_path == '/factory_packing_table' for object_cfg in task_cfg.objects)
    assert any(object_cfg.name == 'factory_tabletop_visual' for object_cfg in task_cfg.objects)


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
