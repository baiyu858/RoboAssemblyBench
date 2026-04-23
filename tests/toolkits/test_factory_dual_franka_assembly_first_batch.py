import pytest

from toolkits.factory_dual_franka_assembly.scene_builder import (
    build_dual_franka_assembly_episode,
)
from toolkits.factory_dual_franka_assembly.task_specs import (
    list_task_recipes,
    load_task_recipe,
)


def test_first_batch_task_specs_are_discoverable():
    recipes = list_task_recipes()
    assert 'connector_docking' in recipes
    assert 'gear_pair_mesh' in recipes
    assert 'nut_thread_after_hold' in recipes
    assert 'handover_fastener_then_insert' in recipes


@pytest.mark.parametrize(
    ('recipe', 'required_target_names', 'required_object_names', 'expected_tag'),
    [
        (
            'connector_docking',
            {'connector_dock', 'socket_hold'},
            {'connector_body', 'socket_housing'},
            'connector',
        ),
        (
            'gear_pair_mesh',
            {'gear_a_mesh', 'gear_b_mesh'},
            {'gear_a', 'gear_b'},
            'meshing',
        ),
        (
            'nut_thread_after_hold',
            {'nut_thread', 'bracket_hold'},
            {'bracket_block', 'nut_fastener'},
            'threading',
        ),
        (
            'handover_fastener_then_insert',
            {'fastener_handoff', 'fastener_insert'},
            {'receiver_block', 'fastener_pin'},
            'handoff',
        ),
    ],
)
def test_first_batch_tasks_build_and_expose_category_metadata(
    recipe,
    required_target_names,
    required_object_names,
    expected_tag,
):
    recipe_spec = load_task_recipe(recipe, scene_profile='taoyuan_tabletop')
    assert recipe_spec['scene_profile'] == 'taoyuan_tabletop'
    assert expected_tag in recipe_spec['metadata']['tags']
    assert 'assembly' in recipe_spec['metadata']['tags']
    assert recipe_spec['supported_scene_profiles'] == [
        'taoyuan_tabletop',
        'taoyuan_grscenes_tabletop',
        'proxy_factory_cell',
    ]

    task_cfg = build_dual_franka_assembly_episode(
        recipe=recipe,
        seed=17,
        episode_idx=0,
        scene_profile='taoyuan_tabletop',
    )
    assert task_cfg.recipe == recipe
    assert required_target_names.issubset(task_cfg.target_poses)
    assert required_object_names.issubset(set(task_cfg.tracked_object_names))
    assert len(task_cfg.phase_specs) >= 8
    assert len(task_cfg.success_criteria) == 2


def test_first_batch_tasks_support_proxy_factory_cell():
    task_cfg = build_dual_franka_assembly_episode(
        recipe='connector_docking',
        seed=23,
        episode_idx=1,
        scene_profile='proxy_factory_cell',
    )
    assert task_cfg.scene_profile == 'proxy_factory_cell'
    assert task_cfg.recipe == 'connector_docking'
    assert 'connector_body' in task_cfg.tracked_object_names
