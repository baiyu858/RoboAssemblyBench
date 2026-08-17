import os


_profile = os.environ.get('INTERNUTOPIA_EXTENSION_PROFILE')
if _profile == 'franka_assembly':
    from internutopia_extension.robots import franka
elif _profile == 'ur5e_assembly':
    from internutopia_extension.robots import ur5e
else:
    from internutopia_extension.robots import (
        aliengo,
        franka,
        g1,
        gr1,
        h1,
        h1_with_hand,
        humanoidbench_h1,
        jetbot,
        mocap_controlled_franka,
        ur5e,
    )
