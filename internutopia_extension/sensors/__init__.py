import os


if os.environ.get('INTERNUTOPIA_EXTENSION_PROFILE', '').endswith('_assembly'):
    from internutopia_extension.sensors import rep_camera
else:
    from internutopia_extension.sensors import (
        layout_edit_mocap_controlled_camera,
        mocap_controlled_camera,
        rep_camera,
    )
