from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ASSET_DIR = Path(__file__).resolve().parents[2] / 'third_part' / 'RoboFactory' / 'robofactory' / 'assets'
DEFAULT_PLACEHOLDERS = {
    '${ASSET_DIR}': str(DEFAULT_ASSET_DIR),
}


def slugify_task_name(task_name: str | None, fallback: str) -> str:
    if not task_name:
        return fallback
    slug = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', task_name)
    slug = re.sub(r'[^a-zA-Z0-9]+', '_', slug).strip('_')
    return slug.lower() or fallback


def replace_placeholders(value: Any, placeholder_map: dict[str, str] | None = None) -> Any:
    placeholder_map = placeholder_map or DEFAULT_PLACEHOLDERS
    if isinstance(value, str):
        replaced = value
        for placeholder, replacement in placeholder_map.items():
            replaced = replaced.replace(placeholder, replacement)
        return replaced
    if isinstance(value, list):
        return [replace_placeholders(item, placeholder_map) for item in value]
    if isinstance(value, tuple):
        return tuple(replace_placeholders(item, placeholder_map) for item in value)
    if isinstance(value, dict):
        return {key: replace_placeholders(item, placeholder_map) for key, item in value.items()}
    return value


def load_robofactory_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    return yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}


def _ensure_mapping(value: Any, *, section_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f'Expected {section_name!r} to be a mapping, got {type(value).__name__}')
    return value


def _section_list(config: dict[str, Any], section_name: str) -> list[dict[str, Any]]:
    section = config.get(section_name)
    if not section:
        return []
    if not isinstance(section, list):
        raise TypeError(f'Expected {section_name!r} to be a list, got {type(section).__name__}')
    return [copy.deepcopy(item) for item in section]


def _pick_identifier(entry: dict[str, Any], fields: tuple[str, ...]) -> str | None:
    for field in fields:
        value = entry.get(field)
        if value not in (None, ''):
            return str(value)
    return None


def _normalize_entries(
    entries: list[dict[str, Any]],
    *,
    section_name: str,
    fallback_prefix: str,
    identifier_fields: tuple[str, ...] = ('uid', 'name'),
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        normalized_entry = copy.deepcopy(entry)
        normalized_entry.setdefault('uid', _pick_identifier(normalized_entry, identifier_fields) or f'{fallback_prefix}_{index}')
        normalized_entry['index'] = index
        normalized_entry['section'] = section_name
        normalized.append(normalized_entry)
    return normalized


def _normalize_camera_groups(
    cameras: dict[str, Any] | None,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    normalized_groups: dict[str, list[dict[str, Any]]] = {}
    flattened: list[dict[str, Any]] = []
    for group_name, group_items in _ensure_mapping(cameras, section_name='cameras').items():
        normalized_items: list[dict[str, Any]] = []
        for index, item in enumerate(group_items or []):
            camera = copy.deepcopy(item)
            if not isinstance(camera, dict):
                raise TypeError(f'Expected camera entries in {group_name!r} to be mappings, got {type(camera).__name__}')
            camera.setdefault('uid', _pick_identifier(camera, ('uid', 'name')) or f'{group_name}_{index}')
            camera['group'] = group_name
            camera['group_index'] = index
            camera['section'] = f'cameras.{group_name}'
            normalized_items.append(camera)
            flattened.append(camera)
        normalized_groups[group_name] = normalized_items
    return normalized_groups, flattened


def normalize_robofactory_config(
    config: dict[str, Any],
    *,
    source_path: str | Path | None = None,
    asset_dir: str | Path | None = None,
    placeholder_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError(f'Expected config to be a mapping, got {type(config).__name__}')
    config_copy = copy.deepcopy(config)
    resolved_asset_dir = Path(asset_dir) if asset_dir is not None else DEFAULT_ASSET_DIR
    merged_placeholders = dict(DEFAULT_PLACEHOLDERS)
    merged_placeholders['${ASSET_DIR}'] = str(resolved_asset_dir)
    if placeholder_map:
        merged_placeholders.update({key: str(value) for key, value in placeholder_map.items()})
    normalized = replace_placeholders(config_copy, merged_placeholders)

    scene = _ensure_mapping(normalized.get('scene'), section_name='scene')
    scene = copy.deepcopy(scene)
    scene_assets = _normalize_entries(
        _section_list(scene, 'assets'),
        section_name='scene.assets',
        fallback_prefix='scene_asset',
    )
    scene_primitives = _normalize_entries(
        _section_list(scene, 'primitives'),
        section_name='scene.primitives',
        fallback_prefix='scene_primitive',
    )
    objects = _section_list(normalized, 'objects')
    agents = _section_list(normalized, 'agents')
    objects = _normalize_entries(objects, section_name='objects', fallback_prefix='object')
    agents = _normalize_entries(agents, section_name='agents', fallback_prefix='agent', identifier_fields=('uid', 'robot_uid', 'name'))
    camera_groups, all_cameras = _normalize_camera_groups(normalized.get('cameras'))

    scene['assets'] = scene_assets
    scene['primitives'] = scene_primitives

    source_path_str = str(source_path) if source_path is not None else None
    task_name = normalized.get('task_name') or normalized.get('task')
    task_slug = slugify_task_name(task_name, Path(source_path_str).stem if source_path_str else 'robofactory_task')

    metadata = {
        'schema_version': 'robofactory-config-adapter/v1',
        'source_path': source_path_str,
        'source_dir': str(Path(source_path_str).parent) if source_path_str else None,
        'task_name': task_name,
        'task_slug': task_slug,
        'scene_name': scene.get('name'),
        'asset_dir': str(resolved_asset_dir),
        'placeholder_map': merged_placeholders,
        'camera_groups': list(camera_groups.keys()),
        'sections': {
            'scene.assets': len(scene_assets),
            'scene.primitives': len(scene_primitives),
            'objects': len(objects),
            'agents': len(agents),
            'cameras': len(all_cameras),
        },
        'counts': {
            'scene_primitives': len(scene_primitives),
            'scene_assets': len(scene_assets),
            'objects': len(objects),
            'agents': len(agents),
            'cameras': len(all_cameras),
        },
    }

    return {
        'task_name': task_name,
        'task_slug': task_slug,
        'scene': scene,
        'scene_primitives': scene_primitives,
        'scene_assets': scene_assets,
        'objects': objects,
        'agents': agents,
        'cameras': {
            'groups': camera_groups,
            'all': all_cameras,
        },
        'metadata': metadata,
    }


def iter_robofactory_config_paths(input_path: str | Path) -> list[Path]:
    root = Path(input_path)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(root)
    paths = sorted({*root.rglob('*.yaml'), *root.rglob('*.yml')})
    return paths


def normalize_robofactory_config_file(
    input_path: str | Path,
    *,
    asset_dir: str | Path | None = None,
    placeholder_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    config_path = Path(input_path)
    config = load_robofactory_config(config_path)
    return normalize_robofactory_config(
        config,
        source_path=config_path,
        asset_dir=asset_dir,
        placeholder_map=placeholder_map,
    )


def write_normalized_config(path: str | Path, payload: dict[str, Any], *, output_format: str = 'json') -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == 'json':
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
        return
    if output_format == 'yaml':
        output_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding='utf-8')
        return
    raise ValueError(f'Unsupported output format: {output_format}')


def _build_output_path(input_root: Path, output_dir: Path, config_path: Path, output_format: str) -> Path:
    suffix = '.json' if output_format == 'json' else '.yaml'
    if input_root.is_file():
        return output_dir / f'{config_path.stem}{suffix}'
    relative_path = config_path.relative_to(input_root)
    return (output_dir / relative_path).with_suffix(suffix)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Normalize RoboFactory YAML configs for InternUtopia.')
    parser.add_argument(
        '--input',
        type=str,
        default=str(Path(__file__).resolve().parents[2] / 'third_part' / 'RoboFactory' / 'robofactory' / 'configs' / 'table'),
        help='Input config file or directory.',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(Path(__file__).resolve().parents[2] / 'toolkits' / 'robofactory_compat' / 'outputs'),
        help='Directory to write normalized specs into.',
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=('json', 'yaml'),
        default='json',
        help='Output serialization format.',
    )
    parser.add_argument(
        '--asset-dir',
        type=str,
        default=str(DEFAULT_ASSET_DIR),
        help='Replacement path for ${ASSET_DIR}.',
    )
    args = parser.parse_args(argv)

    input_root = Path(args.input)
    output_dir = Path(args.output_dir)
    config_paths = iter_robofactory_config_paths(input_root)
    written_paths = []
    for config_path in config_paths:
        normalized = normalize_robofactory_config_file(
            config_path,
            asset_dir=args.asset_dir,
        )
        output_path = _build_output_path(input_root, output_dir, config_path, args.format)
        write_normalized_config(output_path, normalized, output_format=args.format)
        written_paths.append(str(output_path))

    print(json.dumps({'input': str(input_root), 'output_dir': str(output_dir), 'format': args.format, 'files': written_paths}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
