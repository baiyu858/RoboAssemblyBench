from toolkits.constraint_checking.integration.sequence_precheck import (
    AssemblySequencePrechecker,
)

ROBOTS = ['left', 'right']
OBJECTS = ['part_a', 'part_b']


def check(phases):
    return AssemblySequencePrechecker().check(
        phases=phases,
        robot_names=ROBOTS,
        object_names=OBJECTS,
    )


def test_valid_pick_carry_place_sequence():
    report = check(
        [
            {'name': 'approach', 'local_skill': {'name': 'move_above_part', 'robot': 'left', 'object': 'part_a'}},
            {'name': 'grasp', 'attach': [{'robot': 'left', 'object': 'part_a'}]},
            {'name': 'carry', 'local_skill': {'name': 'move_part_to_target', 'robot': 'left', 'object': 'part_a'}},
            {'name': 'place', 'lock': [{'object': 'part_a', 'target': 'slot'}]},
        ]
    )

    assert report.feasible
    assert report.errors == []
    assert report.final_holding['left'] is None


def test_robot_cannot_attach_second_object_while_holding():
    report = check(
        [
            {'name': 'grasp_a', 'attach': [{'robot': 'left', 'object': 'part_a'}]},
            {'name': 'grasp_b', 'attach': [{'robot': 'left', 'object': 'part_b'}]},
        ]
    )

    assert not report.feasible
    assert {issue['code'] for issue in report.errors} == {'end_effector_occupied'}


def test_two_robots_cannot_grasp_same_object():
    report = check(
        [
            {'name': 'left_grasp', 'attach': [{'robot': 'left', 'object': 'part_a'}]},
            {'name': 'right_grasp', 'attach': [{'robot': 'right', 'object': 'part_a'}]},
        ]
    )

    assert not report.feasible
    assert 'double_grasp' in {issue['code'] for issue in report.errors}


def test_payload_motion_requires_prior_attach():
    report = check(
        [
            {
                'name': 'carry_without_grasp',
                'local_skill': {'name': 'move_part_to_target', 'robot': 'left', 'object': 'part_a'},
            }
        ]
    )

    assert not report.feasible
    assert 'payload_not_held' in {issue['code'] for issue in report.errors}


def test_release_by_non_owner_is_rejected():
    report = check(
        [
            {'name': 'grasp', 'attach': [{'robot': 'left', 'object': 'part_a'}]},
            {'name': 'bad_release', 'detach': [{'robot': 'right', 'object': 'part_a'}]},
        ]
    )

    assert not report.feasible
    assert 'release_by_non_owner' in {issue['code'] for issue in report.errors}


def test_unknown_references_are_reported():
    report = check([{'name': 'bad', 'attach': [{'robot': 'third', 'object': 'missing'}]}])

    assert not report.feasible
    codes = {issue['code'] for issue in report.errors}
    assert {'unknown_robot', 'unknown_object'}.issubset(codes)


def test_current_plumbers_block_recipe_passes_sequence_precheck():
    from toolkits.factory_dual_franka_assembly.task_specs import load_task_recipe

    recipe = load_task_recipe('fabrica_plumbers_block_ur5e_right_base_prepare')
    report = AssemblySequencePrechecker().check(
        phases=recipe['phases'],
        robot_names=[robot['name'] for robot in recipe['robots']],
        object_names=[item['name'] for item in recipe['objects']],
    )

    assert report.feasible, report.errors
    assert report.phase_count > 20
