from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_FACTORY_INSERT_CHECKPOINT = 'checkpoints/Factory/test/nn/Factory.pth'
DEFAULT_FABRICA_FIXPLUG_PLUMBERS_BLOCK_CHECKPOINT = (
    'roboassemblybench/assets/Fabrica/checkpoints/plumbers_block_fixplug_rl/sr_gen_plumbers_block.pth'
)


@dataclass
class LocalSkillResult:
    action: dict | None = None
    consumed: bool = False
    status: str = 'inactive'
    reason: str = ''
    diagnostics: dict[str, Any] = field(default_factory=dict)


def normalize_local_skill_spec(phase_spec: dict | None, robot_name: str) -> dict | None:
    if not phase_spec:
        return None

    raw_spec = None
    local_skills = phase_spec.get('local_skills')
    if isinstance(local_skills, dict):
        raw_spec = local_skills.get(robot_name)
    elif isinstance(local_skills, list):
        for item in local_skills:
            if isinstance(item, dict) and item.get('robot') == robot_name:
                raw_spec = item
                break

    if raw_spec is None:
        raw_spec = phase_spec.get('local_skill') or phase_spec.get('skill')

    if raw_spec is None:
        return None
    if isinstance(raw_spec, str):
        raw_spec = {'name': raw_spec}
    if not isinstance(raw_spec, dict):
        return None

    spec = dict(raw_spec)
    spec.setdefault('name', spec.get('type'))
    if not spec.get('name'):
        return None

    spec_robot = spec.get('robot')
    if spec_robot is not None and str(spec_robot) != robot_name:
        return None

    spec['robot'] = robot_name
    spec.setdefault('fallback', 'native_phase_policy')
    return spec


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_roots(task=None) -> list[Path]:
    roots = [Path.cwd(), _repo_root()]
    cfg = getattr(task, 'config', getattr(task, 'cfg', None))
    spec_path = getattr(cfg, 'spec_path', None)
    if spec_path:
        roots.append(Path(spec_path).expanduser().resolve().parent)
    benchmark_root = os.environ.get('BENCHMARK_ROOT')
    if benchmark_root:
        roots.append(Path(benchmark_root).expanduser())
    unique_roots = []
    seen = set()
    for root in roots:
        root = root.expanduser().resolve()
        if root in seen:
            continue
        seen.add(root)
        unique_roots.append(root)
    return unique_roots


def resolve_checkpoint_path(spec: dict, task=None) -> str | None:
    default_checkpoint = DEFAULT_FACTORY_INSERT_CHECKPOINT
    backend = str(spec.get('backend', '')).lower()
    if backend in {'fabrica_fixplug', 'fabrica_fixplug_isaacgym', 'fabrica_isaacgym_fixplug'}:
        default_checkpoint = DEFAULT_FABRICA_FIXPLUG_PLUMBERS_BLOCK_CHECKPOINT
    checkpoint = (
        spec.get('checkpoint')
        or spec.get('checkpoint_path')
        or os.environ.get('FACTORY_DUAL_FRANKA_PEG_TRANSFER_CHECKPOINT')
        or default_checkpoint
    )
    if not checkpoint:
        return None

    path = Path(str(checkpoint)).expanduser()
    if path.is_absolute():
        return str(path) if path.exists() else None

    for root in _candidate_roots(task):
        candidate = root / path
        if candidate.exists():
            return str(candidate)
    return None


def probe_factory_insert_rl_backend(spec: dict, task=None) -> dict[str, Any]:
    backend = str(spec.get('backend', 'isaaclab_factory'))
    if backend in {'fabrica_fixplug', 'fabrica_fixplug_isaacgym', 'fabrica_isaacgym_fixplug'}:
        dependencies = ('torch',)
        default_adapter = (
            'toolkits.factory_dual_franka_assembly.factory_insert_adapter:FabricaFixPlugPolicyAdapter'
        )
        default_observation_space = 3
        default_action_space = 3
    else:
        dependencies = ('isaaclab', 'isaaclab_rl', 'isaaclab_tasks', 'rl_games')
        default_adapter = None
        default_observation_space = 19
        default_action_space = 6
    missing_dependencies = [name for name in dependencies if not _module_available(name)]
    checkpoint_path = resolve_checkpoint_path(spec, task=task)
    adapter = spec.get('adapter') or spec.get('adapter_class') or default_adapter
    ready = bool(adapter) and not missing_dependencies and bool(checkpoint_path)
    reason = 'ready'
    if not adapter:
        reason = 'adapter_required'
    elif missing_dependencies:
        reason = 'missing_dependencies'
    elif checkpoint_path is None:
        reason = 'checkpoint_missing'
    return {
        'backend': backend,
        'ready': ready,
        'reason': reason,
        'adapter': adapter,
        'checkpoint_path': checkpoint_path,
        'checkpoint_requested': spec.get('checkpoint') or spec.get('checkpoint_path') or default_checkpoint,
        'missing_dependencies': missing_dependencies,
        'observation_space': int(spec.get('observation_space', default_observation_space)),
        'action_space': int(spec.get('action_space', default_action_space)),
    }


class LocalSkillExecutor:
    def __init__(self):
        self._diagnostics: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._adapter_cache: dict[str, Any] = {}

    @property
    def diagnostics(self) -> list[dict[str, Any]]:
        return list(self._diagnostics.values())

    def reset(self):
        self._diagnostics = {}
        self._adapter_cache = {}

    def action_for(
        self,
        *,
        task,
        robot_name: str,
        phase_spec: dict | None,
        tracked_robots: dict,
        tracked_objects: dict,
    ) -> dict | None:
        spec = normalize_local_skill_spec(phase_spec, robot_name)
        if spec is None:
            return None

        spec = self._merge_task_defaults(task, spec)
        skill_name = str(spec['name'])
        if skill_name in {'scripted_grab', 'grab', 'scripted_pull_out', 'pull_out'}:
            result = LocalSkillResult(
                status='fallback',
                reason='scripted_skill_uses_native_phase_policy',
                diagnostics={'fallback': spec.get('fallback', 'native_phase_policy')},
            )
        elif skill_name in {
            'fabrica_official_joint_pose',
            'fabrica_joint_pose',
            'official_joint_pose',
            'hardcoded_insert',
            'scripted_insert',
            'deterministic_insert',
        }:
            default_adapter = (
                'toolkits.factory_dual_franka_assembly.factory_insert_adapter:FabricaOfficialJointPoseAdapter'
                if skill_name in {'fabrica_official_joint_pose', 'fabrica_joint_pose', 'official_joint_pose'}
                else 'toolkits.factory_dual_franka_assembly.factory_insert_adapter:HardcodedFabricaInsertAdapter'
            )
            result = self._adapter_action(
                task=task,
                robot_name=robot_name,
                phase_spec=phase_spec or {},
                spec=spec,
                tracked_robots=tracked_robots,
                tracked_objects=tracked_objects,
                default_adapter=default_adapter,
            )
        elif skill_name in {'factory_insert_rl', 'fabrica_fixplug_rl', 'fabrica_insert_rl', 'insert_rl', 'insert'}:
            result = self._factory_insert_rl_action(
                task=task,
                robot_name=robot_name,
                phase_spec=phase_spec or {},
                spec=spec,
                tracked_robots=tracked_robots,
                tracked_objects=tracked_objects,
            )
        else:
            result = LocalSkillResult(
                status='fallback',
                reason=f'unknown_local_skill:{skill_name}',
                diagnostics={'fallback': spec.get('fallback', 'native_phase_policy')},
            )

        self._record_diagnostic(
            task=task,
            phase_spec=phase_spec or {},
            robot_name=robot_name,
            skill_name=skill_name,
            result=result,
            spec=spec,
        )
        return result.action if result.consumed else None

    def _merge_task_defaults(self, task, spec: dict) -> dict:
        cfg = getattr(task, 'config', getattr(task, 'cfg', None))
        task_metadata = getattr(cfg, 'task_metadata', {}) if cfg is not None else {}
        skill_defaults = {}
        if isinstance(task_metadata, dict):
            local_skill_defaults = task_metadata.get('local_skills', {})
            if isinstance(local_skill_defaults, dict):
                skill_defaults = dict(local_skill_defaults.get(spec['name'], {}))
        merged = {**skill_defaults, **spec}
        merged.setdefault('fallback', 'native_phase_policy')
        return merged

    def _adapter_action(
        self,
        *,
        task,
        robot_name: str,
        phase_spec: dict,
        spec: dict,
        tracked_robots: dict,
        tracked_objects: dict,
        default_adapter: str,
    ) -> LocalSkillResult:
        held_object_failure = self._held_object_failure_diagnostics(
            spec=spec,
            robot_name=robot_name,
            tracked_objects=tracked_objects,
            task=task,
        )
        if held_object_failure is not None:
            return self._required_failure_if_configured(
                spec,
                reason='held_object_not_grasped',
                diagnostics=held_object_failure,
            )

        adapter_ref = str(spec.get('adapter') or spec.get('adapter_class') or default_adapter)
        try:
            adapter = self._load_adapter(adapter_ref, spec)
            raw_action = adapter.act(
                task=task,
                robot_name=robot_name,
                phase_spec=phase_spec,
                skill_spec=spec,
                tracked_robots=tracked_robots,
                tracked_objects=tracked_objects,
                checkpoint_path=None,
            )
        except Exception as exc:
            return self._required_failure_if_configured(
                spec,
                reason=f'adapter_error:{type(exc).__name__}',
                diagnostics={
                    'adapter': adapter_ref,
                    'fallback': spec.get('fallback', 'native_phase_policy'),
                    'error': str(exc),
                },
            )

        action = self._normalize_adapter_action(raw_action, robot_name)
        if action is None:
            return self._required_failure_if_configured(
                spec,
                reason='adapter_returned_no_action',
                diagnostics={'adapter': adapter_ref, 'fallback': spec.get('fallback', 'native_phase_policy')},
            )
        return LocalSkillResult(
            action=action,
            consumed=True,
            status='active',
            reason='adapter_action',
            diagnostics={'adapter': adapter_ref, 'backend': 'hardcoded'},
        )

    def _factory_insert_rl_action(
        self,
        *,
        task,
        robot_name: str,
        phase_spec: dict,
        spec: dict,
        tracked_robots: dict,
        tracked_objects: dict,
    ) -> LocalSkillResult:
        held_object_failure = self._held_object_failure_diagnostics(
            spec=spec,
            robot_name=robot_name,
            tracked_objects=tracked_objects,
            task=task,
        )
        if held_object_failure is not None:
            return self._required_failure_if_configured(
                spec,
                reason='held_object_not_grasped',
                diagnostics=held_object_failure,
            )

        probe = probe_factory_insert_rl_backend(spec, task=task)
        if not probe['ready']:
            return self._required_failure_if_configured(
                spec,
                reason=probe['reason'],
                diagnostics={**probe, 'fallback': spec.get('fallback', 'native_phase_policy')},
            )

        try:
            adapter = self._load_adapter(str(probe['adapter']), spec)
            raw_action = adapter.act(
                task=task,
                robot_name=robot_name,
                phase_spec=phase_spec,
                skill_spec=spec,
                tracked_robots=tracked_robots,
                tracked_objects=tracked_objects,
                checkpoint_path=probe['checkpoint_path'],
            )
        except Exception as exc:
            return self._required_failure_if_configured(
                spec,
                reason=f'adapter_error:{type(exc).__name__}',
                diagnostics={
                    **probe,
                    'fallback': spec.get('fallback', 'native_phase_policy'),
                    'error': str(exc),
                },
            )

        action = self._normalize_adapter_action(raw_action, robot_name)
        if action is None:
            return self._required_failure_if_configured(
                spec,
                reason='adapter_returned_no_action',
                diagnostics={**probe, 'fallback': spec.get('fallback', 'native_phase_policy')},
            )
        return LocalSkillResult(action=action, consumed=True, status='active', reason='adapter_action', diagnostics=probe)

    @staticmethod
    def _requires_success(spec: dict) -> bool:
        return bool(spec.get('require_success') or str(spec.get('fallback', '')).lower() == 'fail')

    def _held_object_failure_diagnostics(
        self,
        *,
        spec: dict,
        robot_name: str,
        tracked_objects: dict,
        task,
    ) -> dict[str, Any] | None:
        held_object = spec.get('held_object') or spec.get('object')
        if not held_object or not self._requires_success(spec):
            return None

        held_object = str(held_object)
        object_state = tracked_objects.get(held_object, {})
        attached_to = object_state.get('attached_to')
        grasped_by = object_state.get('grasped_by')
        if attached_to == robot_name or grasped_by == robot_name:
            return None

        attachment_state = getattr(task, '_attachments', {}).get(held_object)
        if isinstance(attachment_state, dict) and attachment_state.get('robot_name') == robot_name:
            return None

        return {
            'held_object': held_object,
            'required_robot': robot_name,
            'attached_to': attached_to,
            'grasped_by': grasped_by,
            'object_status': object_state.get('status'),
            'fallback': spec.get('fallback', 'native_phase_policy'),
        }

    @staticmethod
    def _required_failure_if_configured(spec: dict, *, reason: str, diagnostics: dict[str, Any]) -> LocalSkillResult:
        if LocalSkillExecutor._requires_success(spec):
            return LocalSkillResult(
                action={'__local_skill_failure__': True, 'reason': reason, 'diagnostics': diagnostics},
                consumed=True,
                status='failed',
                reason=reason,
                diagnostics=diagnostics,
            )
        return LocalSkillResult(
            status='fallback',
            reason=reason,
            diagnostics=diagnostics,
        )

    def _load_adapter(self, adapter_ref: str, spec: dict):
        if adapter_ref in self._adapter_cache:
            return self._adapter_cache[adapter_ref]
        module_name, _, class_name = adapter_ref.partition(':')
        if not module_name or not class_name:
            module_name, _, class_name = adapter_ref.rpartition('.')
        if not module_name or not class_name:
            raise ValueError(f'Invalid local skill adapter reference: {adapter_ref!r}')
        module = importlib.import_module(module_name)
        adapter_cls = getattr(module, class_name)
        adapter = adapter_cls(spec)
        self._adapter_cache[adapter_ref] = adapter
        return adapter

    @staticmethod
    def _normalize_adapter_action(raw_action, robot_name: str) -> dict | None:
        if raw_action is None:
            return None
        if isinstance(raw_action, dict) and robot_name in raw_action and isinstance(raw_action[robot_name], dict):
            return raw_action[robot_name]
        if isinstance(raw_action, dict):
            return raw_action
        return None

    def _record_diagnostic(
        self,
        *,
        task,
        phase_spec: dict,
        robot_name: str,
        skill_name: str,
        result: LocalSkillResult,
        spec: dict,
    ):
        key = (
            id(task),
            getattr(task, 'phase_index', None),
            getattr(task, 'phase_entry_step', None),
            robot_name,
            skill_name,
        )
        diagnostic = {
            'phase': phase_spec.get('name', getattr(task, 'phase', None)),
            'phase_index': getattr(task, 'phase_index', None),
            'robot': robot_name,
            'skill': skill_name,
            'status': result.status,
            'reason': result.reason,
            'fallback': spec.get('fallback'),
            'held_object': spec.get('held_object') or spec.get('object'),
            'socket_object': spec.get('socket_object') or spec.get('fixed_object'),
        }
        diagnostic.update(result.diagnostics)
        self._diagnostics[key] = diagnostic
