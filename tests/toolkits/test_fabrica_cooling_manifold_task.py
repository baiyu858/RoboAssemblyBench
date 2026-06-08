from roboassemblybench.core.task_registry import (
    list_task_annotations,
    list_task_recipes,
    load_task_recipe,
)
from toolkits.factory_dual_franka_assembly.scene_builder import (
    build_dual_franka_assembly_episode,
)


def test_fabrica_cooling_manifold_task_is_discoverable_and_reuses_factory_scene():
    assert 'fabrica_cooling_manifold' in list_task_recipes()
    assert 'fabrica_cooling_manifold' in list_task_annotations()

    recipe_spec = load_task_recipe('fabrica_cooling_manifold', scene_profile='taoyuan_tabletop')

    assert recipe_spec['scene_profile'] == 'taoyuan_tabletop'
    assert recipe_spec['source_benchmark'] == 'fabrica'
    assert 'fabrica' in recipe_spec['metadata']['tags']
    assert any(
        reference['name'] == 'fabrica_franka_cooling_fullbundle'
        for reference in recipe_spec['asset_references']
    )
    assert any(
        reference['name'] == 'optical_board'
        and reference['path'].endswith('assets/fabrica_support/optical_board.obj')
        for reference in recipe_spec['asset_references']
    )


def test_fabrica_cooling_manifold_episode_builds_with_fabrica_parts():
    task_cfg = build_dual_franka_assembly_episode(
        recipe='fabrica_cooling_manifold',
        seed=31,
        episode_idx=0,
        scene_profile='taoyuan_tabletop',
    )

    assert task_cfg.recipe == 'fabrica_cooling_manifold'
    assert task_cfg.annotation_name == 'fabrica_cooling_manifold'
    assert 'manifold_insert' in task_cfg.tracked_object_names
    assert 'optical_board' not in task_cfg.tracked_object_names
    assert 'insert_seated' in task_cfg.target_poses
    assert len(task_cfg.phase_specs) >= 8
    assert len(task_cfg.success_criteria) == 1
