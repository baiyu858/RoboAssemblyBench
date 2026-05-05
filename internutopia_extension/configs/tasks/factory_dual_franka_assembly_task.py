from typing import Any, Dict, List, Optional, Tuple

from internutopia.core.config.task import TaskCfg


class FactoryDualFrankaAssemblyTaskCfg(TaskCfg):
    type: Optional[str] = 'FactoryDualFrankaAssemblyTask'
    max_steps: int = 1800
    phase_timeout_steps: Optional[int] = None
    phase_timeout_action: str = 'fail'
    phase_timeout_recovery_phase: Optional[str] = None
    prompt: Optional[str] = ''
    task_description: str = ''
    recipe: str = 'screw_fastening'
    seed: int = 0
    episode_idx: int = 0
    robot_names: Tuple[str, ...] = ('franka_left', 'franka_right')
    tracked_object_names: Tuple[str, ...] = ()
    phase_specs: List[Dict[str, Any]] = []
    target_poses: Dict[str, Dict[str, List[float]]] = {}
    success_criteria: List[Dict[str, Any]] = []
    scene_profile: str = ''
    spec_path: str = ''
    scene_profile_path: Optional[str] = None
    annotation_name: str = ''
    annotation_path: Optional[str] = None
    annotation_title: str = ''
    annotation_summary: str = ''
    annotation_description: str = ''
    annotation_metadata: Dict[str, Any] = {}
    annotation_object_roles: Dict[str, Any] = {}
    annotation_target_roles: Dict[str, Any] = {}
    annotation_phase_notes: List[Dict[str, Any]] = []
    annotation_tags: List[str] = []
    target_annotations: Dict[str, Any] = {}
    phase_annotations: List[Dict[str, Any]] = []
    workspace_offset: List[float] = []
    benchmark_metadata: Dict[str, Any] = {}
    task_metadata: Dict[str, Any] = {}
    scene_profile_metadata: Dict[str, Any] = {}
    scene_lights: List[Dict[str, Any]] = []
    asset_references: List[Dict[str, Any]] = []
    source_benchmark: str = 'factory_dual_franka_assembly'
    source_config_path: Optional[str] = None
    camera_metadata: List[Dict[str, Any]] = []
    robot_metadata: List[Dict[str, Any]] = []
    object_metadata: List[Dict[str, Any]] = []
