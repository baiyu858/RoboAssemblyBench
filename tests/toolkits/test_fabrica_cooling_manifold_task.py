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
    assert any(reference['name'] == 'optical_board_source_obj' for reference in recipe_spec['asset_references'])
    assert any(
        reference['name'] == 'fabrica_cooling_assembled_preview'
        and reference['path'].endswith('assembled/fabrica_cooling_manifold_assembled.usda')
        for reference in recipe_spec['asset_references']
    )
    assert any(
        reference['name'] == 'optical_board_visual_usd'
        and reference['path'].endswith('roboassemblybench/assets/Fabrica/optical_board.usda')
        for reference in recipe_spec['asset_references']
    )
    assert any(
        reference['name'] == 'fabrica_pickup_fixture_usd'
        and reference['path'].endswith('assets/fabrica_fixture/cooling_manifold/fixture_pickup_tray.usda')
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
    assert 'optical_board' not in task_cfg.tracked_object_names
    assert 'fabrica_cooling_manifold_1' not in task_cfg.tracked_object_names
    assert 'fabrica_fixture' not in task_cfg.tracked_object_names
    for part_idx in (0, 2, 3, 4, 5, 6):
        assert f'fabrica_cooling_manifold_{part_idx}' in task_cfg.tracked_object_names
    assert not any(
        name.startswith('assembled_reference_part_')
        for name in task_cfg.tracked_object_names
    )
    assert 'assembled_manifold_preview' not in task_cfg.tracked_object_names
    for part_idx in (0, 2, 3, 4, 5, 6):
        assert f'part_{part_idx}_seated' in task_cfg.target_poses
    object_names = {metadata['name'] for metadata in task_cfg.object_metadata}
    assert {
        'fabrica_cooling_manifold_0',
        'fabrica_cooling_manifold_1',
        'fabrica_cooling_manifold_2',
        'fabrica_cooling_manifold_3',
        'fabrica_cooling_manifold_4',
        'fabrica_cooling_manifold_5',
        'fabrica_cooling_manifold_6',
        'assembled_manifold_preview',
        'optical_board',
        'fabrica_fixture',
    }.issubset(object_names)
    assert not any(name.startswith('assembled_reference_part_') for name in object_names)
    assert len(task_cfg.phase_specs) >= 49
    assert len(task_cfg.success_criteria) == 6
    assert task_cfg.task_metadata['local_skills']['factory_insert_rl']['checkpoint'].endswith(
        'checkpoints/Factory/test/nn/Factory.pth'
    )

    object_specs = {metadata['name']: metadata for metadata in task_cfg.object_metadata}
    assert 'optical_board_proxy' not in object_specs
    assert not any(name.startswith('manifold_hole_proxy_') for name in object_specs)
    assert object_specs['fabrica_cooling_manifold_1']['collider'] is True
    assert object_specs['fabrica_cooling_manifold_1']['auto_collider'] is True
    assert object_specs['fabrica_cooling_manifold_1']['scale'] == [1.0, 1.0, 1.0]
    assert object_specs['assembled_manifold_preview']['tracked'] is False
    assert object_specs['assembled_manifold_preview']['position'] != object_specs['fabrica_cooling_manifold_1']['position']
    assert object_specs['assembled_manifold_preview']['scale'] == [1.0, 1.0, 1.0]
    for part_idx in (0, 2, 3, 4, 5, 6):
        assert object_specs[f'fabrica_cooling_manifold_{part_idx}']['collider'] is True
        assert object_specs[f'fabrica_cooling_manifold_{part_idx}']['auto_collider'] is True
        assert object_specs[f'fabrica_cooling_manifold_{part_idx}']['scale'] == [1.0, 1.0, 1.0]
    assert object_specs['fabrica_cooling_manifold_6']['static_friction'] == 1.2
    assert object_specs['fabrica_cooling_manifold_6']['dynamic_friction'] == 1.0
    assert object_specs['optical_board']['collider'] is True
    assert object_specs['optical_board']['scale'] == [1.0, 1.0, 1.0]
    assert object_specs['fabrica_fixture']['collider'] is True
    assert object_specs['fabrica_fixture']['auto_collider'] is False
    assert object_specs['fabrica_fixture']['rigid_body'] is False
    assert object_specs['fabrica_fixture']['tracked'] is False
    assert object_specs['fabrica_fixture']['scale'] == [1.0, 1.0, 1.0]
    assert object_specs['fabrica_fixture']['position'] == [0.28, -0.02, 0.015551339]

    phase_specs = {phase['name']: phase for phase in task_cfg.phase_specs}
    attach_specs = [
        attach
        for phase in task_cfg.phase_specs
        for attach in phase.get('attach', [])
        if isinstance(attach, dict)
    ]
    assert len(attach_specs) == 5
    assert 'attach' not in phase_specs['right_grasp_part_6']
    assert 'detach' not in phase_specs['release_part_6']
    assert {attach['object'] for attach in attach_specs} == {
        f'fabrica_cooling_manifold_{part_idx}'
        for part_idx in (0, 2, 3, 4, 5)
    }
    assert {attach['attachment_mode'] for attach in attach_specs} == {'pure_physical_grasp'}
    assert all(attach.get('disable_collision_on_attach') is False for attach in attach_specs)
    assert all(attach.get('require_physical_contact') is True for attach in attach_specs)
    part_6_grasp_conditions = phase_specs['right_grasp_part_6']['advance']['conditions']
    assert any(
        condition.get('type') == 'robot_object_contact'
        and condition.get('object') == 'fabrica_cooling_manifold_6'
        and condition.get('robot') == 'franka_right'
        for condition in part_6_grasp_conditions
    )
    part_6_lift_conditions = phase_specs['right_lift_part_6']['advance']['conditions']
    assert any(
        condition.get('type') == 'object_lifted'
        and condition.get('object') == 'fabrica_cooling_manifold_6'
        and condition.get('require_attached') is False
        for condition in part_6_lift_conditions
    )
    assert not any(
        target_spec.get('payload_object') == 'fabrica_cooling_manifold_6'
        for phase in task_cfg.phase_specs
        for target_spec in phase.get('robot_targets', {}).values()
        if isinstance(target_spec, dict)
    )
    assert not any(
        target_spec.get('direct_payload_motion')
        for phase in task_cfg.phase_specs
        for target_spec in phase.get('robot_targets', {}).values()
        if isinstance(target_spec, dict)
    )
    assert not any(phase.get('freeze_after_detach') for phase in task_cfg.phase_specs)

    rl_insert_phases = [
        phase
        for phase in task_cfg.phase_specs
        if isinstance(phase.get('local_skill'), dict)
        and phase['local_skill'].get('name') == 'factory_insert_rl'
    ]
    assert len(rl_insert_phases) == 6
    assert {
        phase['local_skill']['held_object']
        for phase in rl_insert_phases
    } == {
        f'fabrica_cooling_manifold_{part_idx}'
        for part_idx in (0, 2, 3, 4, 5, 6)
    }
