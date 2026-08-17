import os


if os.environ.get('INTERNUTOPIA_EXTENSION_PROFILE', '').endswith('_assembly'):
    from internutopia_extension.tasks import factory_dual_franka_assembly_task
else:
    from internutopia_extension.tasks import (
        factory_box_carry_task,
        factory_dual_franka_assembly_task,
        finite_step_task,
        manipulation_task,
        single_inference_task,
    )
