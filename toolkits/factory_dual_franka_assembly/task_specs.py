from roboassemblybench.core.paths import SHARED_TASK_DIR, TASKS_DIR
from roboassemblybench.core.task_registry import (
    build_task_description,
    list_task_annotations,
    list_task_recipes,
    load_task_annotation,
    load_task_recipe,
    resolve_task_annotation_path,
    resolve_task_spec_path,
)

TASK_SPEC_DIR = TASKS_DIR
ANNOTATION_DIR = SHARED_TASK_DIR.parent
