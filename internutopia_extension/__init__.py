def import_extensions():
    try:
        import carb  # noqa: F401
    except ModuleNotFoundError:
        # The pip Isaac Sim distribution registers Kit modules when isaacsim is imported.
        import isaacsim  # noqa: F401

    import internutopia_extension.controllers
    import internutopia_extension.interactions
    import internutopia_extension.metrics
    import internutopia_extension.objects
    import internutopia_extension.robots
    import internutopia_extension.sensors
    import internutopia_extension.tasks


def import_fabrica_assembly_extensions(robot_platform='ur5e'):
    """Register only runtime components needed by one Fabrica robot platform."""

    import os

    platform = str(robot_platform).strip().lower()
    if platform not in {'ur5e', 'franka'}:
        raise ValueError(f'Unsupported Fabrica robot platform: {robot_platform!r}')

    os.environ['INTERNUTOPIA_EXTENSION_PROFILE'] = f'{platform}_assembly'
    try:
        import carb  # noqa: F401
    except ModuleNotFoundError:
        import isaacsim  # noqa: F401

    import importlib

    modules = (
        'internutopia_extension.controllers.gripper_controller',
        'internutopia_extension.controllers.ik_controller',
        'internutopia_extension.controllers.joint_controller',
        'internutopia_extension.objects.dynamic_compound_cuboid',
        'internutopia_extension.objects.dynamic_cube',
        'internutopia_extension.objects.static_cube',
        'internutopia_extension.objects.usd_object',
        'internutopia_extension.objects.visual_cube',
        f'internutopia_extension.robots.{platform}',
        'internutopia_extension.sensors.rep_camera',
        'internutopia_extension.tasks.factory_dual_franka_assembly_task',
    )
    for module_name in modules:
        importlib.import_module(module_name)


def import_ur5e_assembly_extensions():
    """Backward-compatible UR5e-specific registration entry point."""

    import_fabrica_assembly_extensions('ur5e')
