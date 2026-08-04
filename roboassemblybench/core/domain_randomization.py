from __future__ import annotations

import copy
import hashlib
import math
import random
from typing import Any


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


def apply_domain_randomization(
    recipe_spec: dict[str, Any],
    *,
    seed: int,
    enabled_override: bool | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one deterministic randomized episode from a recipe.

    Every member of a group receives the exact same sampled translation. This is
    required for assembly data: pickup parts and their pickup waypoints move as a
    rigid layout, while the assembly targets move as a second rigid layout.
    """

    resolved = copy.deepcopy(recipe_spec)
    randomization_spec = copy.deepcopy(resolved.get('domain_randomization') or {})
    configured_enabled = bool(randomization_spec.get('enabled', False))
    enabled = configured_enabled if enabled_override is None else bool(enabled_override)
    namespace = str(randomization_spec.get('seed_namespace', resolved.get('task_name', 'task')))
    result: dict[str, Any] = {
        'enabled': enabled,
        'configured_enabled': configured_enabled,
        'seed': int(seed),
        'derived_seed': _derived_seed(seed, namespace),
        'seed_namespace': namespace,
        'groups': {},
        'appearance_groups': {},
    }
    if not enabled:
        resolved['resolved_domain_randomization'] = result
        return resolved, result

    groups = randomization_spec.get('groups') or {}
    appearance_groups = (randomization_spec.get('appearance') or {}).get('groups') or {}
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
    for group_name, raw_group_spec in appearance_groups.items():
        if not isinstance(raw_group_spec, dict):
            raise ValueError(f'Domain-randomization appearance group {group_name!r} must be a mapping.')
        group_name = str(group_name)
        group_spec = copy.deepcopy(raw_group_spec)
        color = _sample_color(group_name, group_spec, rng)
        group_result = {'color': color, 'objects': [], 'lights': []}
        for entry_kind, collection_key in (('object', 'objects'), ('light', 'lights')):
            member_names = [str(value) for value in (group_spec.get(collection_key) or [])]
            for member_name in member_names:
                member_key = (entry_kind, member_name)
                previous_group = assigned_appearance_members.get(member_key)
                if previous_group is not None:
                    raise ValueError(
                        f'{entry_kind.title()} {member_name!r} belongs to both appearance groups '
                        f'{previous_group!r} and {group_name!r}.'
                    )
                assigned_appearance_members[member_key] = group_name
            entries_key = 'objects' if entry_kind == 'object' else 'scene_lights'
            group_result[collection_key] = _apply_group_color(
                resolved.get(entries_key, []),
                member_names=member_names,
                group_name=group_name,
                color=color,
                entry_kind=entry_kind,
            )
        result['appearance_groups'][group_name] = group_result

    resolved['resolved_domain_randomization'] = result
    return resolved, result
