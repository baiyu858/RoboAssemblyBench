from __future__ import annotations

import copy
import hashlib
import math
import random
from pathlib import Path
from typing import Any


RANDOMIZATION_PROFILES = (
    'position',
    'object_distractors',
    'texture',
    'lighting',
    'table_color',
    'scene',
)
MIXED_RANDOMIZATION_PROFILE = 'mixed'
RANDOMIZATION_PROFILE_CHOICES = (*RANDOMIZATION_PROFILES, MIXED_RANDOMIZATION_PROFILE)


def normalize_randomization_profile(profile: str | None) -> str:
    normalized = str(profile or MIXED_RANDOMIZATION_PROFILE).strip().lower().replace('-', '_')
    aliases = {
        'geometry': 'position',
        'layout': 'position',
        'position_only': 'position',
        'distractors': 'object_distractors',
        'object': 'object_distractors',
        'textures': 'texture',
        'light': 'lighting',
        'colour': 'table_color',
        'color': 'table_color',
        'table_colour': 'table_color',
        'background': 'scene',
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in RANDOMIZATION_PROFILE_CHOICES:
        raise ValueError(
            f'Unknown randomization profile {profile!r}; expected one of '
            f'{list(RANDOMIZATION_PROFILE_CHOICES)}.'
        )
    return normalized


def _derived_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f'{int(seed)}:{namespace}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], byteorder='big', signed=False)


def _axis_range(value: Any, *, group_name: str, axis_name: str) -> tuple[float, float]:
    if value is None:
        return 0.0, 0.0
    if isinstance(value, dict):
        lower = value.get('min', value.get('low'))
        upper = value.get('max', value.get('high'))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        lower, upper = value
    else:
        raise ValueError(
            f'Domain-randomization group {group_name!r} axis {axis_name!r} must be [min, max] or a mapping.'
        )

    lower = float(lower)
    upper = float(upper)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ValueError(
            f'Invalid domain-randomization range for group {group_name!r} axis {axis_name!r}: ' f'[{lower}, {upper}].'
        )
    return lower, upper


def _sample_translation(group_name: str, group_spec: dict[str, Any], rng: random.Random) -> list[float]:
    translation_spec = group_spec.get('translation', {})
    if isinstance(translation_spec, (list, tuple)):
        if len(translation_spec) != 3:
            raise ValueError(f'Domain-randomization group {group_name!r} translation must have three axes.')
        translation_spec = dict(zip(('x', 'y', 'z'), translation_spec))
    if not isinstance(translation_spec, dict):
        raise ValueError(f'Domain-randomization group {group_name!r} translation must be a mapping.')

    axis_ranges = {
        axis_name: _axis_range(
            translation_spec.get(axis_name),
            group_name=group_name,
            axis_name=axis_name,
        )
        for axis_name in ('x', 'y', 'z')
    }
    minimum_planar_distance = float(group_spec.get('minimum_planar_distance', 0.0))
    maximum_planar_distance = float(group_spec.get('maximum_planar_distance', math.inf))
    if (
        not math.isfinite(minimum_planar_distance)
        or minimum_planar_distance < 0.0
        or maximum_planar_distance <= 0.0
        or minimum_planar_distance > maximum_planar_distance
    ):
        raise ValueError(
            f'Domain-randomization group {group_name!r} planar-distance bounds are invalid: '
            f'[{minimum_planar_distance}, {maximum_planar_distance}].'
        )

    constraints = group_spec.get('translation_constraints') or []
    if not isinstance(constraints, list):
        raise ValueError(f'Domain-randomization group {group_name!r} translation_constraints must be a list.')

    def satisfies_constraints(translation: list[float]) -> bool:
        for constraint in constraints:
            if not isinstance(constraint, dict):
                raise ValueError(
                    f'Domain-randomization group {group_name!r} translation constraints ' 'must be mappings.'
                )
            constraint_type = str(constraint.get('type', '')).strip().lower()
            points = constraint.get('points') or []
            if not isinstance(points, list) or not points:
                raise ValueError(
                    f'Domain-randomization group {group_name!r} constraint '
                    f'{constraint_type!r} requires a non-empty points list.'
                )
            translated_points = []
            for point in points:
                values = [float(value) for value in point]
                if len(values) != 3 or not all(math.isfinite(value) for value in values):
                    raise ValueError(
                        f'Domain-randomization group {group_name!r} constraint points '
                        'must contain three finite values.'
                    )
                translated_points.append([values[index] + translation[index] for index in range(3)])

            if constraint_type == 'points_inside_bounds':
                lower = [float(value) for value in constraint.get('lower', [])]
                upper = [float(value) for value in constraint.get('upper', [])]
                axes = [int(value) for value in constraint.get('axes', [0, 1, 2])]
                if (
                    len(lower) != 3
                    or len(upper) != 3
                    or not all(math.isfinite(value) for value in [*lower, *upper])
                    or any(axis not in (0, 1, 2) for axis in axes)
                    or any(lower[axis] > upper[axis] for axis in axes)
                ):
                    raise ValueError(
                        f'Domain-randomization group {group_name!r} has invalid ' 'points_inside_bounds limits.'
                    )
                if any(
                    point[axis] < lower[axis] or point[axis] > upper[axis]
                    for point in translated_points
                    for axis in axes
                ):
                    return False
            elif constraint_type == 'points_within_distance':
                origin = [float(value) for value in constraint.get('origin', [])]
                maximum_distance = float(constraint.get('maximum_distance', 0.0))
                if (
                    len(origin) != 3
                    or not all(math.isfinite(value) for value in origin)
                    or not math.isfinite(maximum_distance)
                    or maximum_distance <= 0.0
                ):
                    raise ValueError(
                        f'Domain-randomization group {group_name!r} has an invalid '
                        'points_within_distance constraint.'
                    )
                if any(
                    math.sqrt(sum((point[index] - origin[index]) ** 2 for index in range(3))) > maximum_distance
                    for point in translated_points
                ):
                    return False
            else:
                raise ValueError(
                    f'Domain-randomization group {group_name!r} uses unsupported '
                    f'translation constraint type {constraint_type!r}.'
                )
        return True

    for _ in range(1024):
        sampled = [rng.uniform(lower, upper) if lower != upper else lower for lower, upper in axis_ranges.values()]
        planar_distance = math.hypot(sampled[0], sampled[1])
        if minimum_planar_distance <= planar_distance <= maximum_planar_distance and satisfies_constraints(sampled):
            return sampled
    raise ValueError(
        f'Domain-randomization group {group_name!r} cannot satisfy planar-distance bounds '
        f'[{minimum_planar_distance}, {maximum_planar_distance}] with translation ranges '
        f'{translation_spec!r}.'
    )


def _color_triplet(value: Any, *, group_name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f'Domain-randomization appearance group {group_name!r} colors must contain three channels.')
    color = [float(channel) for channel in value]
    if any(not math.isfinite(channel) or channel < 0.0 or channel > 1.0 for channel in color):
        raise ValueError(
            f'Domain-randomization appearance group {group_name!r} colors must be finite values in [0, 1].'
        )
    return color


def _sample_color(group_name: str, group_spec: dict[str, Any], rng: random.Random) -> list[float]:
    palette = group_spec.get('palette')
    if palette is not None:
        if not isinstance(palette, (list, tuple)) or not palette:
            raise ValueError(f'Domain-randomization appearance group {group_name!r} palette must be non-empty.')
        colors = [_color_triplet(value, group_name=group_name) for value in palette]
        return list(colors[rng.randrange(len(colors))])

    color_spec = group_spec.get('color')
    if isinstance(color_spec, (list, tuple)):
        return _color_triplet(color_spec, group_name=group_name)
    if not isinstance(color_spec, dict):
        raise ValueError(f'Domain-randomization appearance group {group_name!r} requires palette or color.')
    sampled = []
    for channel_name in ('r', 'g', 'b'):
        lower, upper = _axis_range(
            color_spec.get(channel_name),
            group_name=group_name,
            axis_name=channel_name,
        )
        channel = rng.uniform(lower, upper) if lower != upper else lower
        if channel < 0.0 or channel > 1.0:
            raise ValueError(
                f'Domain-randomization appearance group {group_name!r} channel {channel_name!r} ' 'must stay in [0, 1].'
            )
        sampled.append(channel)
    return sampled


def _sample_scalar(group_name: str, group_spec: dict[str, Any], key: str, rng: random.Random) -> float | None:
    value = group_spec.get(key)
    if value is None:
        return None
    lower, upper = _axis_range(value, group_name=group_name, axis_name=key)
    return float(rng.uniform(lower, upper) if lower != upper else lower)


def _sample_vector(group_name: str, group_spec: dict[str, Any], key: str, rng: random.Random) -> list[float] | None:
    value = group_spec.get(key)
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        ranges = value
    elif isinstance(value, dict):
        ranges = [value.get(axis) for axis in ('x', 'y', 'z')]
    else:
        raise ValueError(f'Domain-randomization appearance group {group_name!r} {key} must have three axes.')
    return [
        _sample_scalar(group_name, {key: axis_range}, key, rng)
        for axis_range in ranges
    ]


def _integer_range(value: Any, *, name: str, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        lower, upper = default
    elif isinstance(value, dict):
        lower = value.get('min', value.get('low'))
        upper = value.get('max', value.get('high'))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        lower, upper = value
    else:
        raise ValueError(f'{name} must be [min, max] or a mapping.')
    lower = int(lower)
    upper = int(upper)
    if lower < 0 or lower > upper:
        raise ValueError(f'Invalid {name}: [{lower}, {upper}].')
    return lower, upper


def _profiled_randomization_spec(
    randomization_spec: dict[str, Any],
    *,
    profile: str,
) -> dict[str, Any]:
    """Select one extra visual domain while keeping geometric layout randomization active."""

    selected = copy.deepcopy(randomization_spec)
    if profile == MIXED_RANDOMIZATION_PROFILE:
        distractor_spec = selected.get('visual_distractors') or {}
        if isinstance(distractor_spec, dict):
            distractor_spec.pop('count_range', None)
            distractor_spec.setdefault('count', 4)
            selected['visual_distractors'] = distractor_spec
        return selected

    appearance = selected.setdefault('appearance', {})
    appearance_groups = appearance.get('groups') or {}
    active_appearance_groups = {
        'table_color': {'table_surface'},
        'scene': {'background'},
    }.get(profile, set())
    appearance['groups'] = {
        name: group
        for name, group in appearance_groups.items()
        if str(name) in active_appearance_groups
    }

    distractor_spec = selected.get('visual_distractors') or {}
    if not isinstance(distractor_spec, dict):
        distractor_spec = {}
    distractor_spec['enabled'] = profile == 'object_distractors'
    selected['visual_distractors'] = distractor_spec
    return selected


def _apply_group_translation(
    entries: list[dict[str, Any]],
    *,
    member_names: list[str],
    group_name: str,
    translation: list[float],
    entry_kind: str,
) -> list[str]:
    by_name = {
        str(entry.get('name')): entry for entry in entries if isinstance(entry, dict) and entry.get('name') is not None
    }
    applied = []
    for member_name in member_names:
        member_name = str(member_name)
        if member_name not in by_name:
            raise KeyError(
                f'Domain-randomization group {group_name!r} references unknown {entry_kind} {member_name!r}.'
            )
        entry = by_name[member_name]
        if entry.get('position') is None:
            raise ValueError(
                f'Domain-randomization group {group_name!r} cannot translate {entry_kind} {member_name!r} '
                'because it has no explicit position.'
            )
        base_position = [float(value) for value in entry['position']]
        if len(base_position) != 3:
            raise ValueError(f'{entry_kind.title()} {member_name!r} position must contain three values.')
        entry['position'] = [base_position[index] + translation[index] for index in range(3)]
        entry['domain_randomization_group'] = group_name
        entry['domain_randomization_translation'] = list(translation)
        applied.append(member_name)
    return applied


def _apply_group_color(
    entries: list[dict[str, Any]],
    *,
    member_names: list[str],
    group_name: str,
    color: list[float],
    entry_kind: str,
) -> list[str]:
    by_name = {
        str(entry.get('name')): entry for entry in entries if isinstance(entry, dict) and entry.get('name') is not None
    }
    applied = []
    for member_name in member_names:
        member_name = str(member_name)
        if member_name not in by_name:
            raise KeyError(
                f'Domain-randomization appearance group {group_name!r} references unknown '
                f'{entry_kind} {member_name!r}.'
            )
        entry = by_name[member_name]
        entry['color'] = list(color)
        entry['domain_randomization_appearance_group'] = group_name
        entry['domain_randomization_color'] = list(color)
        applied.append(member_name)
    return applied


def _apply_group_appearance(
    entries: list[dict[str, Any]],
    *,
    member_names: list[str],
    group_name: str,
    color: list[float],
    entry_kind: str,
    intensity: float | None = None,
    exposure: float | None = None,
    rotation_euler: list[float] | None = None,
) -> list[str]:
    applied = _apply_group_color(
        entries,
        member_names=member_names,
        group_name=group_name,
        color=color,
        entry_kind=entry_kind,
    )
    by_name = {
        str(entry.get('name')): entry
        for entry in entries
        if isinstance(entry, dict) and entry.get('name') is not None
    }
    for member_name in applied:
        entry = by_name[member_name]
        if intensity is not None:
            entry['intensity'] = float(intensity)
            entry['domain_randomization_intensity'] = float(intensity)
        if exposure is not None:
            entry['exposure'] = float(exposure)
            entry['domain_randomization_exposure'] = float(exposure)
        if rotation_euler is not None:
            entry['rotation_euler'] = list(rotation_euler)
            entry['domain_randomization_rotation_euler'] = list(rotation_euler)
    return applied


def _append_visual_distractors(resolved: dict[str, Any], *, seed: int, rng: random.Random) -> list[dict[str, Any]]:
    """Add non-colliding tabletop clutter while leaving assembly geometry untouched."""

    randomization_spec = resolved.get('domain_randomization') or {}
    distractor_spec = randomization_spec.get('visual_distractors') or {}
    if distractor_spec is False or not bool(distractor_spec.get('enabled', True)):
        return []

    table = next(
        (
            entry
            for entry in resolved.get('objects', [])
            if isinstance(entry, dict) and entry.get('name') == 'factory_tabletop_visual'
        ),
        None,
    )
    if table is None:
        return []
    table_position = [float(value) for value in table.get('position', [0.47, 0.0, 1.0005])]
    table_scale = [float(value) for value in table.get('scale', [1.62, 2.12, 0.002])]
    count_range = distractor_spec.get('count_range')
    if count_range is None:
        count = max(int(distractor_spec.get('count', 6)), 0)
    else:
        count_lower, count_upper = _integer_range(
            count_range,
            name='visual_distractors.count_range',
            default=(0, 8),
        )
        count = rng.randint(count_lower, count_upper)
    colors = distractor_spec.get(
        'palette',
    ) or [
        [0.85, 0.20, 0.12],
        [0.12, 0.45, 0.80],
        [0.90, 0.65, 0.08],
        [0.18, 0.68, 0.38],
        [0.65, 0.20, 0.72],
    ]
    safe_margin = float(distractor_spec.get('safe_margin', 0.16))
    x_min = table_position[0] - table_scale[0] * 0.5 + safe_margin
    x_max = table_position[0] + table_scale[0] * 0.5 - safe_margin
    y_min = table_position[1] - table_scale[1] * 0.5 + safe_margin
    y_max = table_position[1] + table_scale[1] * 0.5 - safe_margin
    workspace_offset = [float(value) for value in resolved.get('workspace_offset', [0.0, 0.0, 0.0])]
    robot_keepout_radius = max(float(distractor_spec.get('robot_keepout_radius', 0.24)), 0.0)
    robot_positions = []
    for robot in resolved.get('robots', []):
        if not isinstance(robot, dict) or robot.get('position') is None:
            continue
        position = [float(value) for value in robot['position']]
        if bool(robot.get('apply_workspace_offset', True)):
            position = [position[index] + workspace_offset[index] for index in range(3)]
        robot_positions.append(position)
    workspace_keepout = distractor_spec.get('workspace_keepout') or {
        'x': [0.04, 0.96],
        'y': [-0.60, 0.60],
    }
    workspace_x = _axis_range(
        workspace_keepout.get('x'),
        group_name='visual_distractors',
        axis_name='workspace_keepout.x',
    )
    workspace_y = _axis_range(
        workspace_keepout.get('y'),
        group_name='visual_distractors',
        axis_name='workspace_keepout.y',
    )
    candidates = []
    minimum_spacing = max(float(distractor_spec.get('minimum_spacing', 0.08)), 0.0)
    for _ in range(max(count * 128, 128)):
        x = rng.uniform(x_min, x_max)
        y = rng.uniform(y_min, y_max)
        in_workspace = workspace_x[0] <= x <= workspace_x[1] and workspace_y[0] <= y <= workspace_y[1]
        outside_robots = all(
            math.hypot(x - position[0], y - position[1]) >= robot_keepout_radius
            for position in robot_positions
        )
        separated = all(math.hypot(x - other_x, y - other_y) >= minimum_spacing for other_x, other_y in candidates)
        if not in_workspace and outside_robots and separated:
            candidates.append((x, y))
        if len(candidates) >= count:
            break
    distractors = []
    existing_names = {
        str(entry.get('name')) for entry in resolved.get('objects', []) if isinstance(entry, dict)
    }
    shapes = [str(value) for value in distractor_spec.get('shapes', ['cube', 'flat', 'tall'])]
    if not shapes:
        raise ValueError('visual_distractors.shapes must not be empty.')
    assets = distractor_spec.get('assets') or []
    if not isinstance(assets, list):
        raise ValueError('visual_distractors.assets must be a list.')
    for index, (x, y) in enumerate(candidates):
        name = f'domain_randomization_visual_distractor_{index}'
        if name in existing_names:
            continue
        size_lower, size_upper = _axis_range(
            distractor_spec.get('size_range', [0.025, 0.045]),
            group_name='visual_distractors',
            axis_name='size_range',
        )
        height_lower, height_upper = _axis_range(
            distractor_spec.get('height_range', [0.012, 0.025]),
            group_name='visual_distractors',
            axis_name='height_range',
        )
        if assets:
            asset = copy.deepcopy(assets[rng.randrange(len(assets))])
            if not isinstance(asset, dict) or not asset.get('path'):
                raise ValueError('Every visual distractor asset requires a path.')
            scale_lower, scale_upper = _axis_range(
                asset.get('scale_range', [0.06, 0.12]),
                group_name='visual_distractors',
                axis_name='asset.scale_range',
            )
            uniform_scale = rng.uniform(scale_lower, scale_upper)
            entry = {
                'name': name,
                'kind': 'usd',
                'prim_path': f'/{name}',
                'usd_path': str(asset['path']),
                'position': [
                    float(x),
                    float(y),
                    float(table_position[2] + table_scale[2] * 0.5 + float(asset.get('z_offset', 0.0))),
                ],
                'orientation_euler': [0.0, 0.0, rng.uniform(-math.pi, math.pi)],
                'apply_workspace_offset': False,
                'scale': [uniform_scale, uniform_scale, uniform_scale],
                'tracked': False,
                'collider': False,
                'auto_collider': False,
                'rigid_body': False,
                'domain_randomization_asset_name': str(asset.get('name', 'warehouse_prop')),
                'domain_randomization_asset_path': str(asset['path']),
                'domain_randomization_visual_distractor': True,
                'domain_randomization_seed': int(seed),
            }
        else:
            scale = [
                rng.uniform(size_lower, size_upper),
                rng.uniform(size_lower, size_upper),
                rng.uniform(height_lower, height_upper),
            ]
            shape = shapes[rng.randrange(len(shapes))]
            if shape == 'flat':
                scale[0] *= rng.uniform(1.4, 2.2)
                scale[2] *= 0.5
            elif shape == 'tall':
                scale[0] *= 0.75
                scale[1] *= 0.75
                scale[2] *= rng.uniform(1.8, 3.0)
            elif shape != 'cube':
                raise ValueError(f'Unsupported visual distractor shape {shape!r}.')
            entry = {
                'name': name,
                'kind': 'visual_cube',
                'prim_path': f'/{name}',
                'position': [
                    float(x),
                    float(y),
                    float(table_position[2] + table_scale[2] * 0.5 + scale[2] * 0.5),
                ],
                'apply_workspace_offset': False,
                'scale': scale,
                'color': _color_triplet(colors[rng.randrange(len(colors))], group_name='visual_distractors'),
                'tracked': False,
                'collider': False,
                'domain_randomization_shape': shape,
                'domain_randomization_visual_distractor': True,
                'domain_randomization_seed': int(seed),
            }
        distractors.append(entry)
    resolved.setdefault('objects', []).extend(distractors)
    return distractors


def _apply_table_texture(resolved: dict[str, Any], *, rng: random.Random) -> dict[str, Any]:
    randomization_spec = resolved.get('domain_randomization') or {}
    texture_spec = randomization_spec.get('textures') or {}
    if texture_spec is False or not bool(texture_spec.get('enabled', True)):
        return {}
    default_table_paths = [str(value) for value in texture_spec.get('table_paths', [])]
    if not default_table_paths:
        raise ValueError('Texture randomization requires domain_randomization.textures.table_paths.')

    # Keep the target list data-driven so new task templates can add a floor,
    # wall, or fixture surface without changing the randomization engine.
    targets = texture_spec.get('targets')
    if targets is None:
        targets = [
            {
                'name': 'tabletop',
                'object': texture_spec.get('table_object', 'factory_tabletop_visual'),
                'paths': default_table_paths,
                'texture_scale': texture_spec.get('texture_scale', [2.0, 6.0]),
            },
        ]
        for name, object_name, path_key, fallback_paths in (
            ('back_wall', 'factory_background_visual', 'wall_paths', default_table_paths),
            ('floor', 'factory_floor_visual', 'floor_paths', default_table_paths),
        ):
            paths = [str(value) for value in texture_spec.get(path_key, fallback_paths)]
            if paths:
                targets.append(
                    {
                        'name': name,
                        'object': object_name,
                        'paths': paths,
                        'texture_scale': texture_spec.get(f'{name}_texture_scale', [2.0, 6.0]),
                    }
                )
    if not isinstance(targets, list) or not targets:
        raise ValueError('domain_randomization.textures.targets must be a non-empty list.')

    objects_by_name = {
        str(entry.get('name')): entry
        for entry in resolved.get('objects', [])
        if isinstance(entry, dict) and entry.get('name') is not None
    }
    surfaces = []
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError('Every domain_randomization.textures target must be a mapping.')
        target_name = str(target.get('name') or target.get('object') or 'surface')
        object_name = str(target.get('object') or '')
        if object_name not in objects_by_name:
            raise KeyError(
                f'Texture randomization target {target_name!r} references unknown object {object_name!r}.'
            )
        texture_paths = [str(value) for value in (target.get('paths') or default_table_paths)]
        if not texture_paths:
            raise ValueError(f'Texture randomization target {target_name!r} has no paths.')
        object_spec = objects_by_name[object_name]
        texture_path = texture_paths[rng.randrange(len(texture_paths))]
        scale_lower, scale_upper = _axis_range(
            target.get('texture_scale', texture_spec.get('texture_scale', [2.0, 6.0])),
            group_name=f'texture.{target_name}',
            axis_name='texture_scale',
        )
        texture_scale = rng.uniform(scale_lower, scale_upper)
        rotation_degrees = float(rng.choice([0.0, 90.0, 180.0, 270.0]))
        object_spec['texture_path'] = texture_path
        object_spec['texture_scale'] = [float(texture_scale), float(texture_scale)]
        object_spec['texture_rotation_degrees'] = rotation_degrees
        object_spec['domain_randomization_texture'] = texture_path
        object_spec['domain_randomization_texture_target'] = target_name
        surfaces.append(
            {
                'name': target_name,
                'object': object_name,
                'path': texture_path,
                'scale': list(object_spec['texture_scale']),
                'rotation_degrees': rotation_degrees,
            }
        )

    first_surface = surfaces[0]
    return {
        **first_surface,
        'surfaces': surfaces,
        'surface_map': {surface['object']: surface for surface in surfaces},
    }


def _apply_lighting_profile(resolved: dict[str, Any], *, rng: random.Random) -> list[dict[str, Any]]:
    randomization_spec = resolved.get('domain_randomization') or {}
    lighting_spec = randomization_spec.get('lighting') or {}
    count_lower, count_upper = _integer_range(
        lighting_spec.get('count_range'),
        name='lighting.count_range',
        default=(1, 4),
    )
    count = rng.randint(count_lower, count_upper)
    intensity_lower, intensity_upper = _axis_range(
        lighting_spec.get('intensity_multiplier', [0.5, 2.0]),
        group_name='lighting',
        axis_name='intensity_multiplier',
    )
    offset_ranges = {
        axis: _axis_range(
            (lighting_spec.get('position_offset') or {}).get(axis, [-0.5, 0.5]),
            group_name='lighting',
            axis_name=f'position_offset.{axis}',
        )
        for axis in ('x', 'y', 'z')
    }
    colors = lighting_spec.get('palette') or [
        [1.00, 0.68, 0.42],
        [0.45, 0.72, 1.00],
        [0.55, 1.00, 0.72],
        [1.00, 0.52, 0.76],
    ]
    base_intensity = float(lighting_spec.get('base_intensity', 100.0))
    exposure = _sample_scalar('lighting', lighting_spec, 'exposure', rng)
    if exposure is None:
        exposure = -0.35
    # Keep a global dome light in the randomized scene.  Replacing the scene
    # profile's dome with only low-energy sphere lights makes lighting changes
    # nearly invisible against the warehouse asset's existing illumination.
    base_lights = copy.deepcopy(resolved.get('scene_lights') or [])
    base_dome = next(
        (
            light
            for light in base_lights
            if isinstance(light, dict) and str(light.get('kind', '')).lower() in {'dome', 'dome_light', 'domelight'}
        ),
        {},
    )
    dome_multiplier = rng.uniform(intensity_lower, intensity_upper)
    dome = {
        **base_dome,
        'name': str(base_dome.get('name') or 'warehouse_dome_fill'),
        'kind': 'dome',
        'intensity': base_intensity * dome_multiplier,
        'exposure': float(exposure),
        'color': _color_triplet(colors[rng.randrange(len(colors))], group_name='lighting'),
        'domain_randomization_intensity_multiplier': float(dome_multiplier),
    }
    lights = [dome]
    for index in range(count):
        multiplier = rng.uniform(intensity_lower, intensity_upper)
        position = [
            rng.uniform(*offset_ranges[axis])
            for axis in ('x', 'y', 'z')
        ]
        position[2] += float(lighting_spec.get('base_height', 3.0))
        lights.append(
            {
                'name': f'domain_randomization_area_light_{index}',
                'kind': 'sphere',
                'position': position,
                'intensity': base_intensity * multiplier,
                'exposure': float(exposure),
                'radius': float(lighting_spec.get('radius', 0.35)),
                'color': _color_triplet(colors[rng.randrange(len(colors))], group_name='lighting'),
                'domain_randomization_intensity_multiplier': float(multiplier),
            }
        )
    resolved['scene_lights'] = lights
    return copy.deepcopy(lights)


def _apply_scene_profile(resolved: dict[str, Any], *, rng: random.Random) -> dict[str, Any]:
    randomization_spec = resolved.get('domain_randomization') or {}
    scene_spec = randomization_spec.get('scene') or {}
    variants = scene_spec.get('asset_paths') or scene_spec.get('variants') or []
    if variants:
        normalized_variants = []
        for index, variant in enumerate(variants):
            if isinstance(variant, str):
                normalized_variants.append({'name': Path(variant).stem, 'asset_path': variant})
            elif isinstance(variant, dict) and variant.get('asset_path'):
                normalized_variants.append(
                    {
                        'name': str(variant.get('name') or Path(str(variant['asset_path'])).stem),
                        'asset_path': str(variant['asset_path']),
                    }
                )
            else:
                raise ValueError(f'Invalid scene asset variant at index {index}.')
        selected_variant = normalized_variants[rng.randrange(len(normalized_variants))]
        resolved['scene_asset_path'] = selected_variant['asset_path']
    else:
        selected_variant = {
            'name': Path(str(resolved.get('scene_asset_path') or 'scene')).stem,
            'asset_path': str(resolved.get('scene_asset_path') or ''),
        }
    position_ranges = {
        axis: _axis_range(
            (scene_spec.get('position_offset') or {}).get(axis, [-0.35, 0.35] if axis != 'z' else [0.0, 0.0]),
            group_name='scene',
            axis_name=f'position_offset.{axis}',
        )
        for axis in ('x', 'y', 'z')
    }
    position = [rng.uniform(*position_ranges[axis]) for axis in ('x', 'y', 'z')]
    yaw_lower, yaw_upper = _axis_range(
        scene_spec.get('yaw_degrees', [-8.0, 8.0]),
        group_name='scene',
        axis_name='yaw_degrees',
    )
    yaw_degrees = rng.uniform(yaw_lower, yaw_upper)
    half_yaw = math.radians(yaw_degrees) * 0.5
    resolved['scene_position'] = position
    resolved['scene_orientation'] = [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]
    return {
        'asset_path': selected_variant['asset_path'],
        'variant': selected_variant['name'],
        'available_variants': [variant['name'] for variant in normalized_variants]
        if variants
        else [selected_variant['name']],
        'position': position,
        'yaw_degrees': float(yaw_degrees),
        'orientation': list(resolved['scene_orientation']),
    }


def apply_domain_randomization(
    recipe_spec: dict[str, Any],
    *,
    seed: int,
    enabled_override: bool | None = None,
    profile: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one deterministic randomized episode from a recipe.

    Every member of a group receives the exact same sampled translation. This is
    required for assembly data: pickup parts and their pickup waypoints move as a
    rigid layout, while the assembly targets move as a second rigid layout.
    """

    resolved = copy.deepcopy(recipe_spec)
    requested_profile = normalize_randomization_profile(profile)
    randomization_spec = _profiled_randomization_spec(
        copy.deepcopy(resolved.get('domain_randomization') or {}),
        profile=requested_profile,
    )
    resolved['domain_randomization'] = randomization_spec
    configured_enabled = bool(randomization_spec.get('enabled', False))
    enabled = configured_enabled if enabled_override is None else bool(enabled_override)
    namespace = str(randomization_spec.get('seed_namespace', resolved.get('task_name', 'task')))
    result: dict[str, Any] = {
        'enabled': enabled,
        'configured_enabled': configured_enabled,
        'seed': int(seed),
        'derived_seed': _derived_seed(seed, namespace),
        'seed_namespace': namespace,
        'profile': requested_profile,
        'groups': {},
        'appearance_groups': {},
    }
    if not enabled:
        resolved['resolved_domain_randomization'] = result
        return resolved, result

    groups = randomization_spec.get('groups') or {}
    appearance_groups = (randomization_spec.get('appearance') or {}).get('groups') or {}
    appearance_spec = randomization_spec.get('appearance') or {}
    if not isinstance(groups, dict):
        raise ValueError('Domain-randomization groups must be a mapping.')
    if not isinstance(appearance_groups, dict):
        raise ValueError('Domain-randomization appearance groups must be a mapping.')
    if not groups and not appearance_groups:
        raise ValueError('Enabled domain randomization requires position or appearance groups.')

    rng = random.Random(result['derived_seed'])
    fixed_objects = {str(value) for value in (randomization_spec.get('fixed_objects') or [])}
    assigned_members: dict[tuple[str, str], str] = {}
    for group_name, raw_group_spec in groups.items():
        if not isinstance(raw_group_spec, dict):
            raise ValueError(f'Domain-randomization group {group_name!r} must be a mapping.')
        group_name = str(group_name)
        group_spec = copy.deepcopy(raw_group_spec)
        translation = _sample_translation(group_name, group_spec, rng)
        group_result = {
            'translation': translation,
            'objects': [],
            'targets': [],
        }
        for entry_kind, collection_key in (('object', 'objects'), ('target', 'targets')):
            member_names = [str(value) for value in (group_spec.get(collection_key) or [])]
            for member_name in member_names:
                if entry_kind == 'object' and member_name in fixed_objects:
                    raise ValueError(
                        f'Fixed object {member_name!r} cannot belong to position-randomization '
                        f'group {group_name!r}.'
                    )
                member_key = (entry_kind, member_name)
                previous_group = assigned_members.get(member_key)
                if previous_group is not None:
                    raise ValueError(
                        f'{entry_kind.title()} {member_name!r} belongs to both domain-randomization groups '
                        f'{previous_group!r} and {group_name!r}.'
                    )
                assigned_members[member_key] = group_name
            group_result[collection_key] = _apply_group_translation(
                resolved.get(collection_key, []),
                member_names=member_names,
                group_name=group_name,
                translation=translation,
                entry_kind=entry_kind,
            )
        result['groups'][group_name] = group_result

    assigned_appearance_members: dict[tuple[str, str], str] = {}
    allowed_appearance_objects = appearance_spec.get('allowed_objects')
    allowed_appearance_lights = appearance_spec.get('allowed_lights')
    if allowed_appearance_objects is not None:
        allowed_appearance_objects = {str(value) for value in allowed_appearance_objects}
    if allowed_appearance_lights is not None:
        allowed_appearance_lights = {str(value) for value in allowed_appearance_lights}
    for group_name, raw_group_spec in appearance_groups.items():
        if not isinstance(raw_group_spec, dict):
            raise ValueError(f'Domain-randomization appearance group {group_name!r} must be a mapping.')
        group_name = str(group_name)
        group_spec = copy.deepcopy(raw_group_spec)
        color = _sample_color(group_name, group_spec, rng)
        intensity = _sample_scalar(group_name, group_spec, 'intensity', rng)
        exposure = _sample_scalar(group_name, group_spec, 'exposure', rng)
        rotation_euler = _sample_vector(group_name, group_spec, 'rotation_euler', rng)
        group_result = {
            'color': color,
            'objects': [],
            'lights': [],
            'intensity': intensity,
            'exposure': exposure,
            'rotation_euler': rotation_euler,
        }
        for entry_kind, collection_key in (('object', 'objects'), ('light', 'lights')):
            member_names = [str(value) for value in (group_spec.get(collection_key) or [])]
            allowed_members = allowed_appearance_objects if entry_kind == 'object' else allowed_appearance_lights
            for member_name in member_names:
                if allowed_members is not None and member_name not in allowed_members:
                    allowlist_key = 'allowed_objects' if entry_kind == 'object' else 'allowed_lights'
                    raise ValueError(
                        f'Domain-randomization appearance group {group_name!r} cannot modify '
                        f'{entry_kind} {member_name!r}; it is outside appearance.{allowlist_key}.'
                    )
                member_key = (entry_kind, member_name)
                previous_group = assigned_appearance_members.get(member_key)
                if previous_group is not None:
                    raise ValueError(
                        f'{entry_kind.title()} {member_name!r} belongs to both appearance groups '
                        f'{previous_group!r} and {group_name!r}.'
                    )
                assigned_appearance_members[member_key] = group_name
            entries_key = 'objects' if entry_kind == 'object' else 'scene_lights'
            group_result[collection_key] = _apply_group_appearance(
                resolved.get(entries_key, []),
                member_names=member_names,
                group_name=group_name,
                color=color,
                entry_kind=entry_kind,
                intensity=intensity if entry_kind == 'light' else None,
                exposure=exposure if entry_kind == 'light' else None,
                rotation_euler=rotation_euler if entry_kind == 'light' else None,
            )
        result['appearance_groups'][group_name] = group_result

    visual_distractors = _append_visual_distractors(resolved, seed=seed, rng=rng)
    result['visual_distractors'] = [
        {
            'name': entry['name'],
            'kind': entry['kind'],
            'position': list(entry['position']),
            'scale': list(entry['scale']),
            **({'color': list(entry['color'])} if entry.get('color') is not None else {}),
            **(
                {
                    'asset_name': entry['domain_randomization_asset_name'],
                    'asset_path': entry['domain_randomization_asset_path'],
                }
                if entry.get('domain_randomization_asset_path') is not None
                else {'shape': entry['domain_randomization_shape']}
            ),
        }
        for entry in visual_distractors
    ]
    result['table_texture'] = _apply_table_texture(resolved, rng=rng) if requested_profile == 'texture' else {}
    result['lighting'] = _apply_lighting_profile(resolved, rng=rng) if requested_profile == 'lighting' else []
    result['scene'] = _apply_scene_profile(resolved, rng=rng) if requested_profile == 'scene' else {}

    resolved['resolved_domain_randomization'] = result
    return resolved, result
