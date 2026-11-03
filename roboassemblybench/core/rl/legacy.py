"""Read-only compatibility for the first experimental checkpoint/reset format.

This is the sole place encoding that old artifact's object/robot identities.
It does not make the old policy valid for any other insertion.
"""
from .task import InsertionTask

CHECKPOINT_FORMAT = 'ur5e-part0-contact-ppo-v1'
OBSERVATION_VERSION = 'part0-physical-55-v1'
SNAPSHOT_FORMAT = 'ur5e-part0-physical-reset-v1'
BACKEND = 'ur5e_part0_physical'


def legacy_task():
    prefix = 'assemble_00_part_0'
    return InsertionTask(
        task_family='fabrica/plumbers_block', stage_id='0',
        moving_object='fabrica_plumbers_block_0', socket_object='fabrica_plumbers_block_2',
        insertion_robot='franka_left', support_robot='franka_right',
        support_object='fabrica_plumbers_block_2',
        entry_target='part_0_insert_00', goal_target='part_0_assembled',
        entry_phase=f'{prefix}_insert_01',
        replace_phases=tuple(f'{prefix}_insert_{i:02d}' for i in range(1, 10)),
        rl_phase=f'{prefix}_physical_rl', settle_phase=f'{prefix}_unloaded_settle',
        support_retreat_phase='support_00_part_0_base_2_retreat',
        readiness_conditions=({'type': 'plumbers_support_ready'},))


def validate_legacy_snapshot(data):
    task = legacy_task()
    if (data.get('phase') != task.entry_phase
            or not str(data.get('recipe', '')).startswith('fabrica_plumbers_block_ur5e_staged')):
        raise ValueError('Legacy reset does not belong to the audited insertion task')
    return task
