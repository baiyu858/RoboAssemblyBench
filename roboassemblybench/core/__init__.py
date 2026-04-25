from importlib import import_module

__all__ = [
    'BENCHMARK_ROOT',
    'DEFAULT_SCENE_PROFILE',
    'SCENE_PROFILE_DIR',
    'SHARED_TASK_DIR',
    'TASKS_DIR',
    'build_dual_franka_assembly_batch',
    'build_dual_franka_assembly_episode',
    'build_task_description',
    'deep_merge',
    'list_scene_profiles',
    'list_task_annotations',
    'list_task_recipes',
    'load_scene_profile',
    'load_task_annotation',
    'load_task_recipe',
    'resolve_task_annotation_path',
    'resolve_task_spec_path',
]


def __getattr__(name: str):
    if name in {'BENCHMARK_ROOT', 'SCENE_PROFILE_DIR', 'SHARED_TASK_DIR', 'TASKS_DIR'}:
        module = import_module('roboassemblybench.core.paths')
        return getattr(module, name)

    if name in {'DEFAULT_SCENE_PROFILE', 'deep_merge', 'list_scene_profiles', 'load_scene_profile'}:
        module = import_module('roboassemblybench.core.scene_profiles')
        return getattr(module, name)

    if name in {
        'build_task_description',
        'list_task_annotations',
        'list_task_recipes',
        'load_task_annotation',
        'load_task_recipe',
        'resolve_task_annotation_path',
        'resolve_task_spec_path',
    }:
        module = import_module('roboassemblybench.core.task_registry')
        return getattr(module, name)

    if name in {'build_dual_franka_assembly_batch', 'build_dual_franka_assembly_episode'}:
        module = import_module('roboassemblybench.core.runtime')
        return getattr(module, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
