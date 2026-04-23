from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from internutopia.macros import gm
from toolkits.factory_dual_franka_assembly.scene_profiles import (
    DEFAULT_SCENE_PROFILE,
    deep_merge,
    load_scene_profile,
)

TASK_SPEC_DIR = Path(__file__).resolve().parent / 'task_specs'
ANNOTATION_DIR = Path(__file__).resolve().parent / 'annotations'


def list_task_recipes() -> list[str]:
    return sorted(path.stem for path in TASK_SPEC_DIR.glob('*.yaml') if not path.stem.startswith('_'))


def list_task_annotations() -> list[str]:
    return sorted(path.stem for path in ANNOTATION_DIR.glob('*.yaml') if not path.stem.startswith('_'))


def _replace_placeholders(value: Any):
    if isinstance(value, dict):
        return {key: _replace_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_placeholders(item) for item in value]
    if isinstance(value, str):
        return value.replace('${ASSET_PATH}', gm.ASSET_PATH)
    return value


def resolve_task_spec_path(recipe_or_path: str) -> Path:
    candidate = Path(recipe_or_path)
    if candidate.exists():
        return candidate.resolve()

    yaml_path = TASK_SPEC_DIR / f'{recipe_or_path}.yaml'
    if yaml_path.exists():
        return yaml_path.resolve()

    raise FileNotFoundError(f'Cannot find assembly task spec for {recipe_or_path!r}.')


def resolve_task_annotation_path(recipe_or_path: str) -> Path:
    candidate = Path(recipe_or_path)
    if candidate.exists() and candidate.parent.resolve() == ANNOTATION_DIR.resolve():
        return candidate.resolve()

    yaml_path = ANNOTATION_DIR / f'{Path(recipe_or_path).stem}.yaml'
    if yaml_path.exists():
        return yaml_path.resolve()

    raise FileNotFoundError(f'Cannot find assembly task annotation for {recipe_or_path!r}.')


def _resolve_extends_path(extends_ref: str, current_path: Path) -> Path:
    candidate = Path(extends_ref)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    relative_candidate = (current_path.parent / extends_ref).resolve()
    if relative_candidate.exists():
        return relative_candidate

    if candidate.suffix:
        default_candidate = (TASK_SPEC_DIR / candidate).resolve()
        if default_candidate.exists():
            return default_candidate
    else:
        default_candidate = (TASK_SPEC_DIR / f'{extends_ref}.yaml').resolve()
        if default_candidate.exists():
            return default_candidate

    raise FileNotFoundError(f'Cannot resolve inherited task spec {extends_ref!r} from {current_path}.')


def _load_yaml_with_extends(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    resolved_path = path.resolve()
    if resolved_path in stack:
        cycle = ' -> '.join(str(item) for item in (*stack, resolved_path))
        raise ValueError(f'Cycle detected while resolving task spec inheritance: {cycle}')

    payload = yaml.safe_load(resolved_path.read_text(encoding='utf-8')) or {}
    extends_ref = payload.pop('extends', None)
    if extends_ref is None:
        return payload

    parent_path = _resolve_extends_path(extends_ref=extends_ref, current_path=resolved_path)
    parent_payload = _load_yaml_with_extends(parent_path, stack=(*stack, resolved_path))
    return deep_merge(parent_payload, payload)


def _load_annotation_with_extends(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    resolved_path = path.resolve()
    if resolved_path in stack:
        cycle = ' -> '.join(str(item) for item in (*stack, resolved_path))
        raise ValueError(f'Cycle detected while resolving task annotation inheritance: {cycle}')

    payload = yaml.safe_load(resolved_path.read_text(encoding='utf-8')) or {}
    extends_ref = payload.pop('extends', None)
    if extends_ref is None:
        return payload

    parent_path = resolve_task_annotation_path(extends_ref)
    parent_payload = _load_annotation_with_extends(parent_path, stack=(*stack, resolved_path))
    return deep_merge(parent_payload, payload)


def build_task_description(annotation: dict | None, fallback_prompt: str) -> str:
    if not annotation:
        return fallback_prompt

    for key in ('task_description', 'description', 'summary'):
        value = annotation.get(key)
        if value:
            return str(value)

    metadata = annotation.get('metadata', {})
    topic_bits = []
    object_roles = metadata.get('object_roles', {})
    if object_roles:
        topic_bits.append(
            'Objects: '
            + '; '.join(
                f"{name} ({role.get('role', 'role')})"
                for name, role in sorted(object_roles.items())
                if isinstance(role, dict)
            )
        )

    target_roles = metadata.get('target_roles', {})
    if target_roles:
        topic_bits.append(
            'Targets: '
            + '; '.join(
                f"{name} ({role.get('phase', 'staging')})"
                for name, role in sorted(target_roles.items())
                if isinstance(role, dict)
            )
        )

    if topic_bits:
        return ' '.join(part for part in (fallback_prompt, *topic_bits) if part).strip()
    return fallback_prompt


def load_task_annotation(recipe_or_path: str) -> dict:
    try:
        path = resolve_task_annotation_path(recipe_or_path)
    except FileNotFoundError:
        return {}

    payload = _load_annotation_with_extends(path)
    payload = _replace_placeholders(payload)

    metadata = copy.deepcopy(payload.get('metadata', {}))
    metadata.setdefault('schema_version', payload.get('schema_version', 1))
    metadata.setdefault('task_name', payload.get('task_name', Path(path).stem))
    metadata.setdefault('authoring_stack', 'RoboFactory/RoboTwin-style')
    payload['metadata'] = metadata
    payload.setdefault('annotation_name', payload.get('task_name', Path(path).stem))
    payload['annotation_path'] = str(path)
    payload['annotation_title'] = payload.get('title') or payload['annotation_name'].replace('_', ' ')
    payload['annotation_summary'] = payload.get('summary') or payload.get('prompt') or ''
    payload['task_description'] = build_task_description(payload, payload.get('prompt', payload['annotation_summary']))
    return copy.deepcopy(payload)


def _normalize_asset_reference(entry: Any, source: str | None = None) -> dict | None:
    if isinstance(entry, str):
        return {'path': entry, 'source': source or 'metadata'}
    if isinstance(entry, dict):
        normalized = copy.deepcopy(entry)
        normalized.setdefault('source', source or 'metadata')
        if 'path' in normalized:
            return normalized
    return None


def _dedupe_asset_references(references: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[Any, ...]] = set()
    for reference in references:
        key = (
            reference.get('path'),
            reference.get('name'),
            reference.get('kind'),
            reference.get('source'),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(reference)
    return deduped


def _collect_asset_references(payload: dict, metadata: dict) -> list[dict]:
    references: list[dict] = []

    scene_asset_path = payload.get('scene_asset_path')
    if scene_asset_path:
        references.append(
            {
                'name': 'scene_asset',
                'kind': 'scene',
                'path': scene_asset_path,
                'source': 'scene_asset_path',
            }
        )

    for object_spec in payload.get('objects', []):
        usd_path = object_spec.get('usd_path')
        file_path = object_spec.get('file_path')
        annotation_path = object_spec.get('annotation_path')
        object_name = object_spec.get('name')
        if usd_path:
            references.append(
                {
                    'name': object_name,
                    'kind': object_spec.get('kind', 'usd'),
                    'path': usd_path,
                    'prim_path': object_spec.get('prim_path'),
                    'source': 'object',
                }
            )
        if file_path:
            references.append(
                {
                    'name': object_name,
                    'kind': object_spec.get('kind', 'asset'),
                    'path': file_path,
                    'prim_path': object_spec.get('prim_path'),
                    'source': 'object_file',
                }
            )
        if annotation_path:
            references.append(
                {
                    'name': f'{object_name}_annotation' if object_name else None,
                    'kind': 'annotation',
                    'path': annotation_path,
                    'source': 'object_annotation',
                }
            )

    for robot_spec in payload.get('robots', []):
        usd_path = robot_spec.get('usd_path')
        if not usd_path:
            continue
        references.append(
            {
                'name': robot_spec.get('name'),
                'kind': 'robot_usd',
                'path': usd_path,
                'prim_path': robot_spec.get('prim_path'),
                'source': 'robot',
            }
        )

    for entry in metadata.get('asset_references', []):
        normalized = _normalize_asset_reference(entry)
        if normalized is not None:
            references.append(normalized)

    return _dedupe_asset_references(references)


def load_task_recipe(recipe_or_path: str, scene_profile: str | None = None) -> dict:
    path = resolve_task_spec_path(recipe_or_path)
    payload = _load_yaml_with_extends(path)
    payload = _replace_placeholders(payload)

    task_metadata = copy.deepcopy(payload.get('metadata', {}))
    annotation_payload = load_task_annotation(path.stem)
    annotation_metadata = copy.deepcopy(annotation_payload.get('metadata', {}))
    requested_scene_profile = scene_profile or payload.get('scene_profile')
    supported_scene_profiles = payload.get('supported_scene_profiles', [])
    if requested_scene_profile and supported_scene_profiles and requested_scene_profile not in supported_scene_profiles:
        raise ValueError(
            f'Task spec {path.name} does not support scene profile {requested_scene_profile!r}. '
            f'Supported profiles: {supported_scene_profiles}'
        )

    scene_profile_payload = load_scene_profile(requested_scene_profile)
    scene_profile_metadata = copy.deepcopy(scene_profile_payload.get('metadata', {}))
    if scene_profile_payload:
        payload = deep_merge(scene_profile_payload, payload)

    merged_metadata = deep_merge(scene_profile_metadata, task_metadata)
    merged_metadata = deep_merge(merged_metadata, annotation_metadata)
    payload['metadata'] = merged_metadata
    payload['task_metadata'] = task_metadata
    payload['scene_profile_metadata'] = scene_profile_metadata
    payload['annotation_name'] = annotation_payload.get('annotation_name', '')
    payload['annotation_path'] = annotation_payload.get('annotation_path')
    payload['annotation_title'] = annotation_payload.get('annotation_title', '')
    payload['annotation_summary'] = annotation_payload.get('annotation_summary', '')
    payload['annotation_description'] = annotation_payload.get('task_description') or ''
    payload['annotation_metadata'] = annotation_metadata
    payload['annotation_object_roles'] = copy.deepcopy(annotation_metadata.get('object_roles', {}))
    payload['annotation_target_roles'] = copy.deepcopy(annotation_metadata.get('target_roles', {}))
    payload['annotation_phase_notes'] = copy.deepcopy(annotation_metadata.get('phase_notes', []))
    payload['annotation_tags'] = copy.deepcopy(annotation_metadata.get('tags', []))
    payload['task_description'] = annotation_payload.get('task_description') or payload.get('prompt', '')
    payload['scene_profile'] = requested_scene_profile
    payload['scene_profile_path'] = scene_profile_payload.get('scene_profile_path')
    payload['spec_path'] = str(path)
    payload.setdefault('benchmark_family', 'factory_dual_franka_assembly')
    payload.setdefault('default_scene_profile', DEFAULT_SCENE_PROFILE)
    payload.setdefault('source_benchmark', payload.get('benchmark_family', 'factory_dual_franka_assembly'))
    payload.setdefault('source_config_path', str(path))
    payload['asset_references'] = _collect_asset_references(payload=payload, metadata=merged_metadata)
    return copy.deepcopy(payload)
