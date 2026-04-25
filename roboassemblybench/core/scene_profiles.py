from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from internutopia.macros import gm
from roboassemblybench.core.paths import SCENE_PROFILE_DIR

DEFAULT_SCENE_PROFILE = 'proxy_factory_cell'

_NAMED_LIST_KEYS = {'robots', 'objects', 'targets', 'phases'}
_UNIQUE_APPEND_LIST_KEYS = {'asset_references', 'supported_scene_profiles', 'tags'}


def list_scene_profiles() -> list[str]:
    return sorted(path.stem for path in SCENE_PROFILE_DIR.glob('*.yaml') if not path.stem.startswith('_'))


def _replace_placeholders(value: Any):
    if isinstance(value, dict):
        return {key: _replace_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_placeholders(item) for item in value]
    if isinstance(value, str):
        return value.replace('${ASSET_PATH}', gm.ASSET_PATH)
    return value


def _resolve_extends_path(extends_ref: str, current_path: Path, default_dir: Path) -> Path:
    candidate = Path(extends_ref)
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    relative_candidate = (current_path.parent / extends_ref).resolve()
    if relative_candidate.exists():
        return relative_candidate

    if candidate.suffix:
        default_candidate = (default_dir / candidate).resolve()
        if default_candidate.exists():
            return default_candidate
    else:
        default_candidate = (default_dir / f'{extends_ref}.yaml').resolve()
        if default_candidate.exists():
            return default_candidate

    raise FileNotFoundError(f'Cannot resolve inherited scene profile {extends_ref!r} from {current_path}.')


def _merge_unique_list(base: list[Any], override: list[Any]) -> list[Any]:
    merged = [copy.deepcopy(item) for item in base]
    for item in override:
        if item not in merged:
            merged.append(copy.deepcopy(item))
    return merged


def _merge_named_list(base: list[dict], override: list[dict]) -> list[dict]:
    merged: list[dict] = [copy.deepcopy(item) for item in base]
    index_by_name = {
        item.get('name'): index
        for index, item in enumerate(merged)
        if isinstance(item, dict) and item.get('name') is not None
    }
    for item in override:
        if not isinstance(item, dict) or item.get('name') is None:
            merged.append(copy.deepcopy(item))
            continue
        name = item['name']
        if name not in index_by_name:
            index_by_name[name] = len(merged)
            merged.append(copy.deepcopy(item))
            continue
        base_item = merged[index_by_name[name]]
        merged[index_by_name[name]] = deep_merge(base_item, item, path=('named_list', name))
    return merged


def deep_merge(base: Any, override: Any, path: tuple[str, ...] = ()) -> Any:
    if base is None:
        return copy.deepcopy(override)
    if override is None:
        return copy.deepcopy(base)

    if isinstance(base, dict) and isinstance(override, dict):
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if key not in merged:
                merged[key] = copy.deepcopy(value)
                continue
            merged[key] = deep_merge(merged[key], value, path=(*path, key))
        return merged

    if isinstance(base, list) and isinstance(override, list):
        field_name = path[-1] if path else ''
        if field_name in _NAMED_LIST_KEYS:
            return _merge_named_list(base, override)
        if field_name in _UNIQUE_APPEND_LIST_KEYS:
            return _merge_unique_list(base, override)
        return copy.deepcopy(override)

    return copy.deepcopy(override)


def resolve_scene_profile_path(profile_or_path: str) -> Path:
    candidate = Path(profile_or_path)
    if candidate.exists():
        return candidate.resolve()

    yaml_path = SCENE_PROFILE_DIR / f'{profile_or_path}.yaml'
    if yaml_path.exists():
        return yaml_path.resolve()

    raise FileNotFoundError(f'Cannot find scene profile {profile_or_path!r}.')


def _load_yaml_with_extends(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    resolved_path = path.resolve()
    if resolved_path in stack:
        cycle = ' -> '.join(str(item) for item in (*stack, resolved_path))
        raise ValueError(f'Cycle detected while resolving scene profile inheritance: {cycle}')

    payload = yaml.safe_load(resolved_path.read_text(encoding='utf-8')) or {}
    extends_ref = payload.pop('extends', None)
    if extends_ref is None:
        return payload

    parent_path = _resolve_extends_path(extends_ref=extends_ref, current_path=resolved_path, default_dir=SCENE_PROFILE_DIR)
    parent_payload = _load_yaml_with_extends(parent_path, stack=(*stack, resolved_path))
    return deep_merge(parent_payload, payload)


def load_scene_profile(profile_or_path: str | None) -> dict:
    if profile_or_path in (None, '', 'none', 'raw'):
        return {}

    path = resolve_scene_profile_path(profile_or_path)
    payload = _load_yaml_with_extends(path)
    payload = _replace_placeholders(payload)
    payload.setdefault('profile_name', path.stem)
    payload['scene_profile_path'] = str(path)
    return copy.deepcopy(payload)
