from importlib import import_module

__all__ = [
    'DEFAULT_SCENE_PROFILE',
    'DualFrankaAssemblyDemoPolicy',
    'FrankaRobofactoryPlanner',
    'PlannerWaypoint',
    'build_dual_franka_assembly_batch',
    'build_dual_franka_assembly_episode',
    'build_task_description',
    'list_scene_profiles',
    'list_task_annotations',
    'list_task_recipes',
    'load_scene_profile',
    'load_task_annotation',
    'load_task_recipe',
]


def __getattr__(name: str):
    if name in {
        'DualFrankaAssemblyDemoPolicy',
        'FrankaRobofactoryPlanner',
        'PlannerWaypoint',
        'build_dual_franka_assembly_batch',
        'build_dual_franka_assembly_episode',
    }:
        module = import_module('roboassemblybench.core.runtime')
        return getattr(module, name)

    if name in {
        'DEFAULT_SCENE_PROFILE',
        'list_scene_profiles',
        'load_scene_profile',
    }:
        module = import_module('roboassemblybench.core.scene_profiles')
        return getattr(module, name)

    if name in {
        'build_task_description',
        'list_task_annotations',
        'list_task_recipes',
        'load_task_annotation',
        'load_task_recipe',
    }:
        module = import_module('roboassemblybench.core.task_registry')
        return getattr(module, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
