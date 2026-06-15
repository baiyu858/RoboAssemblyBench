from roboassemblybench.core.task_registry import (
    list_task_annotations,
    list_task_recipes,
    load_task_recipe,
)
from toolkits.factory_dual_franka_assembly.scene_builder import (
    build_dual_franka_assembly_episode,
)


def test_fabrica_plumbers_block_fixplug_rl_is_separate_from_official_replay():
    assert 'fabrica_plumbers_block' in list_task_recipes()
    assert 'fabrica_plumbers_block_fixplug_rl' in list_task_recipes()
    assert 'fabrica_plumbers_block_fixplug_rl' in list_task_annotations()

    official_replay = load_task_recipe('fabrica_plumbers_block', scene_profile='taoyuan_tabletop')
    fixplug_rl = load_task_recipe('fabrica_plumbers_block_fixplug_rl', scene_profile='taoyuan_tabletop')

    assert official_replay['phases'] == []
    assert len(fixplug_rl['phases']) == 30
    assert len(fixplug_rl['success']) == 5
    assert fixplug_rl['metadata']['fabrica_official_layout']['fixplug_pairs'] == [
        ['0', '2'],
        ['3', '2'],
        ['1', '3'],
        ['4', '3'],
    ]


def test_fabrica_plumbers_block_fixplug_rl_episode_builds_with_rl_and_joint_phases():
    task_cfg = build_dual_franka_assembly_episode(
        recipe='fabrica_plumbers_block_fixplug_rl',
        seed=31,
        episode_idx=0,
        scene_profile='taoyuan_tabletop',
    )

    assert task_cfg.recipe == 'fabrica_plumbers_block_fixplug_rl'
    assert len(task_cfg.phase_specs) == 30
    assert len(task_cfg.success_criteria) == 5
    assert task_cfg.task_metadata['local_skills']['fabrica_fixplug_rl']['backend'] == 'fabrica_fixplug'
    assert task_cfg.task_metadata['local_skills']['fabrica_fixplug_rl']['observation_space'] == 3
    assert task_cfg.task_metadata['local_skills']['fabrica_fixplug_rl']['action_space'] == 3

    rl_insert_phases = [
        phase
        for phase in task_cfg.phase_specs
        if isinstance(phase.get('local_skill'), dict)
        and phase['local_skill'].get('name') == 'fabrica_fixplug_rl'
    ]
    official_joint_phases = [
        phase
        for phase in task_cfg.phase_specs
        if isinstance(phase.get('local_skill'), dict)
        and phase['local_skill'].get('name') == 'fabrica_official_joint_pose'
    ]

    assert len(rl_insert_phases) == 4
    assert len(official_joint_phases) == 19
    assert {tuple(phase['local_skill']['plug_socket_pair']) for phase in rl_insert_phases} == {
        ('0', '2'),
        ('3', '2'),
        ('1', '3'),
        ('4', '3'),
    }
    initial_wait = task_cfg.phase_specs[0]
    assert initial_wait['name'] == 'initial_wait'
    assert {
        lock_spec['object']: lock_spec['target']
        for lock_spec in initial_wait['lock']
    } == {
        'fabrica_plumbers_block_0': 'part_0_pickup',
        'fabrica_plumbers_block_1': 'part_1_pickup',
        'fabrica_plumbers_block_2': 'part_2_pickup',
        'fabrica_plumbers_block_3': 'part_3_pickup',
        'fabrica_plumbers_block_4': 'part_4_pickup',
    }
    grasp_phases = [
        phase
        for phase in task_cfg.phase_specs
        if phase['name'].startswith('grasp_') and phase['name'].endswith('_fixed_to_gripper')
    ]
    assert all('unlock' not in phase for phase in grasp_phases)
    assert all(phase['advance']['min_steps'] == 1 for phase in rl_insert_phases)
    fixed_joint_targets = {
        phase['attach'][0]['object']: phase['attach'][0]['target']
        for phase in task_cfg.phase_specs
        if phase.get('attach')
    }
    assert fixed_joint_targets == {
        'fabrica_plumbers_block_0': 'part_0_pickup',
        'fabrica_plumbers_block_1': 'part_1_pickup',
        'fabrica_plumbers_block_2': 'part_2_pickup',
        'fabrica_plumbers_block_3': 'part_3_pickup',
        'fabrica_plumbers_block_4': 'part_4_pickup',
    }
    for phase in task_cfg.phase_specs:
        if not phase.get('attach'):
            continue
        attach_spec = phase['attach'][0]
        assert attach_spec['attachment_mode'] == 'fixed_joint'
        assert attach_spec['require_contact'] is False
        assert attach_spec['require_physical_contact'] is False
        assert attach_spec['support_height_tolerance'] is None
        assert attach_spec['gripper_closed_threshold'] == 0.05
    phase_by_name = {phase['name']: phase for phase in task_cfg.phase_specs}
    assert phase_by_name['approach_base_part_2_official']['local_skill']['robot'] == 'franka_left'
    assert phase_by_name['grasp_base_part_2_fixed_to_gripper']['local_skill']['robot'] == 'franka_left'
    assert phase_by_name['grasp_base_part_2_fixed_to_gripper']['attach'][0]['robot'] == 'franka_left'
    assert phase_by_name['place_base_part_2_official']['local_skill']['robot'] == 'franka_left'
    assert phase_by_name['approach_part_0_official']['local_skill']['robot'] == 'franka_right'
    assert 'fabrica_plumbers_block_2' in task_cfg.tracked_object_names


def test_fabrica_plumbers_block_online_plan_keeps_rl_without_joint_replay():
    assert 'fabrica_plumbers_block_online_plan_fixplug_rl' in list_task_recipes()
    assert 'fabrica_plumbers_block_online_plan_fixplug_rl' in list_task_annotations()

    recipe = load_task_recipe(
        'fabrica_plumbers_block_online_plan_fixplug_rl',
        scene_profile='taoyuan_tabletop',
    )

    assert recipe['metadata']['fabrica_official_layout']['execution_mode'] == (
        'phase_flow_with_isaacsim_online_planning_and_fixplug_rl_insertions'
    )
    assert recipe['metadata']['disable_joint_planner_when_attached'] is True
    assert recipe['metadata']['local_skills']['fabrica_fixplug_rl']['residual_guard'] is True
    assert recipe['metadata']['local_skills']['fabrica_fixplug_rl']['residual_stagnation_patience'] == 32
    assert (
        recipe['metadata']['online_planning_source']['not_used_for_runtime_control']
        == 'Fabrica motion.pkl joint trajectory replay'
    )

    target_names = {target['name'] for target in recipe['targets']}
    assert {
        'part_2_pickup_gripper',
        'part_2_place_gripper',
        'part_0_pickup_gripper',
        'part_0_preinsert_gripper',
        'part_3_preinsert_gripper',
        'part_1_preinsert_gripper',
        'part_4_preinsert_gripper',
    } <= target_names

    task_cfg = build_dual_franka_assembly_episode(
        recipe='fabrica_plumbers_block_online_plan_fixplug_rl',
        seed=31,
        episode_idx=0,
        scene_profile='taoyuan_tabletop',
    )

    assert task_cfg.recipe == 'fabrica_plumbers_block_online_plan_fixplug_rl'
    assert len(task_cfg.phase_specs) == 30
    rl_insert_phases = [
        phase
        for phase in task_cfg.phase_specs
        if isinstance(phase.get('local_skill'), dict)
        and phase['local_skill'].get('name') == 'fabrica_fixplug_rl'
    ]
    official_joint_phases = [
        phase
        for phase in task_cfg.phase_specs
        if isinstance(phase.get('local_skill'), dict)
        and phase['local_skill'].get('name') == 'fabrica_official_joint_pose'
    ]
    assert len(rl_insert_phases) == 4
    assert official_joint_phases == []

    phase_by_name = {phase['name']: phase for phase in task_cfg.phase_specs}
    assert phase_by_name['approach_base_part_2_online']['robot_targets']['franka_left']['target'] == (
        'part_2_pickup_gripper'
    )
    assert phase_by_name['place_base_part_2_online']['robot_targets']['franka_left']['target'] == (
        'part_2_place_gripper'
    )
    assert phase_by_name['preinsert_part_0_online']['robot_targets']['franka_right']['target'] == (
        'part_0_preinsert_gripper'
    )
    for phase_name in [
        'approach_part_3_online',
        'grasp_part_3_fixed_to_gripper',
        'approach_part_1_online',
        'grasp_part_1_fixed_to_gripper',
        'approach_part_4_online',
        'grasp_part_4_fixed_to_gripper',
    ]:
        assert phase_by_name[phase_name]['robot_targets']['franka_right']['disable_joint_planner'] is True
        assert phase_by_name[phase_name]['robot_targets']['franka_right']['disable_interarm_gating'] is True
    for phase in task_cfg.phase_specs:
        for target_spec in phase.get('robot_targets', {}).values():
            if (
                isinstance(target_spec, dict)
                and target_spec.get('target') in {'left_wait', 'right_wait'}
            ):
                assert target_spec.get('blocking') is False

    fixed_joint_targets = {}
    fixed_joint_target_specs = []
    for phase in task_cfg.phase_specs:
        if not phase.get('attach'):
            continue
        attach_target = phase['attach'][0]['target']
        fixed_joint_target_specs.append(attach_target)
        fixed_joint_targets[phase['attach'][0]['object']] = attach_target['target']
    assert fixed_joint_targets == {
        'fabrica_plumbers_block_0': 'part_0_pickup_gripper',
        'fabrica_plumbers_block_1': 'part_1_pickup_gripper',
        'fabrica_plumbers_block_2': 'part_2_pickup_gripper',
        'fabrica_plumbers_block_3': 'part_3_pickup_gripper',
        'fabrica_plumbers_block_4': 'part_4_pickup_gripper',
    }
    assert all(
        target_spec.get('ik_frame_compensation') == 'none'
        for target_spec in fixed_joint_target_specs
    )
    assert all(
        phase['attach'][0].get('require_target_reached_for_attach') is True
        for phase in task_cfg.phase_specs
        if phase.get('attach')
    )
    seated_locks = [
        lock_spec
        for phase in task_cfg.phase_specs
        for lock_spec in phase.get('lock', [])
        if isinstance(lock_spec, dict) and str(lock_spec.get('target', '')).endswith('_seated')
    ]
    assert {lock_spec['object'] for lock_spec in seated_locks} == {
        'fabrica_plumbers_block_0',
        'fabrica_plumbers_block_1',
        'fabrica_plumbers_block_2',
        'fabrica_plumbers_block_3',
        'fabrica_plumbers_block_4',
    }
    assert all(lock_spec.get('snap_on_open') is True for lock_spec in seated_locks)
    assert all(lock_spec.get('release_snap_steps') == 0 for lock_spec in seated_locks)
