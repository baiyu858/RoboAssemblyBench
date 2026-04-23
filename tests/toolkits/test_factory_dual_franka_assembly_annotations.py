from toolkits.factory_dual_franka_assembly.convert_dataset import build_dataset_entries
from toolkits.factory_dual_franka_assembly.scene_builder import build_dual_franka_assembly_episode
from toolkits.factory_dual_franka_assembly.task_specs import (
    list_task_annotations,
    load_task_annotation,
    load_task_recipe,
)


def test_annotation_assets_are_discoverable():
    annotations = list_task_annotations()
    assert 'screw_fastening' in annotations
    assert 'peg_insertion' in annotations
    assert 'panel_alignment' in annotations
    assert 'bracket_latching' in annotations
    assert 'connector_docking' in annotations
    assert 'gear_pair_mesh' in annotations
    assert 'nut_thread_after_hold' in annotations
    assert 'handover_fastener_then_insert' in annotations


def test_annotation_loader_exposes_description_and_roles():
    annotation = load_task_annotation('screw_fastening')
    assert annotation['annotation_name'] == 'screw_fastening'
    assert annotation['metadata']['authoring_stack'] == 'RoboFactory/RoboTwin-style'
    assert annotation['metadata']['object_roles']['assembly_part']['role'] == 'workpiece'
    assert annotation['metadata']['target_roles']['screw_insert']['object'] == 'screw'
    assert 'fasten the screw' in annotation['task_description']


def test_recipe_loader_merges_annotation_metadata():
    recipe_spec = load_task_recipe('peg_insertion', scene_profile='taoyuan_tabletop')
    assert recipe_spec['annotation_name'] == 'peg_insertion'
    assert recipe_spec['task_description'].startswith('Two Franka arms align the housing')
    assert 'dual-arm' in recipe_spec['metadata']['tags']
    assert recipe_spec['annotation_target_roles']['peg_insert']['object'] == 'peg'
    assert recipe_spec['annotation_phase_notes'][0]['name'] == 'left_approach_housing'


def test_scene_builder_uses_annotation_targets():
    task_cfg = build_dual_franka_assembly_episode(
        recipe='panel_alignment',
        seed=17,
        episode_idx=2,
        scene_profile='taoyuan_tabletop',
    )
    assert task_cfg.annotation_name == 'panel_alignment'
    assert task_cfg.task_description.startswith('Two Franka arms align the panel')
    assert task_cfg.target_annotations['pin_insert']['object'] == 'locating_pin'
    assert task_cfg.phase_annotations[0]['name'] == 'left_approach_panel'


def test_first_batch_tasks_also_load_annotations():
    recipe_spec = load_task_recipe('handover_fastener_then_insert', scene_profile='taoyuan_tabletop')
    assert recipe_spec['annotation_name'] == 'handover_fastener_then_insert'
    assert 'handoff' in recipe_spec['annotation_tags']
    assert recipe_spec['annotation_target_roles']['fastener_handoff']['object'] == 'fastener_pin'


def test_dataset_entries_preserve_annotation_metadata():
    entries = build_dataset_entries(
        [
            {
                'episode_idx': 0,
                'seed': 11,
                'recipe': 'bracket_latching',
                'prompt': 'Two Franka arms seat a bracket and latch it into the shared fixture.',
                'task_description': 'Two Franka arms seat the bracket and latch the fixture.',
                'annotation_name': 'bracket_latching',
                'annotation_path': '/tmp/annotations/bracket_latching.yaml',
                'annotation_title': 'Bracket latching',
                'annotation_summary': 'Seat a bracket and latch it.',
                'annotation_description': 'Two Franka arms seat the bracket and latch the fixture.',
                'annotation_tags': ['bracket', 'latch'],
                'annotation_metadata': {'authoring_stack': 'RoboFactory/RoboTwin-style'},
                'annotation_object_roles': {'bracket': {'role': 'workpiece'}},
                'annotation_target_roles': {'latch_insert': {'object': 'latch'}},
                'annotation_phase_notes': [{'name': 'left_approach_bracket'}],
                'target_annotations': {'latch_insert': {'object': 'latch', 'pose': {'position': [0, 0, 0], 'orientation': [1, 0, 0, 0]}}},
                'phase_annotations': [{'name': 'left_approach_bracket'}],
                'metrics': {'success': True},
                'steps': [
                    {'phase': 'left_approach_bracket', 'observations': {}, 'actions': {}, 'objects': {}},
                    {'phase': 'retreat', 'observations': {}, 'actions': {}, 'objects': {}},
                ],
            }
        ]
    )

    assert entries[0]['instruction'] == 'Two Franka arms seat the bracket and latch the fixture.'
    assert entries[0]['annotation_name'] == 'bracket_latching'
    assert entries[0]['target_annotations']['latch_insert']['object'] == 'latch'
    assert entries[0]['phase_annotations'][0]['name'] == 'left_approach_bracket'
