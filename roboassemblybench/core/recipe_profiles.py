"""Small recipe overrides selected as task@profile; no simulator dependencies."""
from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path

import yaml

from roboassemblybench.core.scene_profiles import deep_merge


def split_recipe_profile(reference, aliases):
    reference = aliases.get(str(reference), str(reference))
    if '@' not in reference:
        return reference, None
    base, profile = reference.rsplit('@', 1)
    if not base or not profile:
        raise ValueError(f'Expected recipe@profile, got {reference!r}.')
    return base, profile


def apply_recipe_profile(payload, profile):
    base = copy.deepcopy(payload)
    profiles = base.pop('profiles', {})
    if not isinstance(profiles, dict):
        raise ValueError('Recipe profiles must be a mapping.')
    if profile is None:
        return base

    def overrides(name, stack=()):
        if name in stack:
            raise ValueError(f'Recipe profile inheritance cycle: {" -> ".join((*stack, name))}')
        if name not in profiles:
            raise ValueError(f'Unknown recipe profile {name!r}; available: {sorted(profiles)}')
        entry = copy.deepcopy(profiles[name])
        if not isinstance(entry, dict):
            raise ValueError(f'Recipe profile {name!r} must be a mapping.')
        parent = entry.pop('extends_profile', None)
        if parent is not None and not isinstance(parent, str):
            raise ValueError('extends_profile must name a profile in this recipe.')
        return deep_merge(overrides(parent, (*stack, name)), entry) if parent is not None else entry

    selected = overrides(profile)
    selected['task_name'] = profiles[profile].get('task_name', f'{base.get("task_name", "task")}@{profile}')
    return deep_merge(base, selected)


@lru_cache(maxsize=1)
def recipe_catalog(files):
    """Index profile names and old task-name aliases; invalidate on file changes.

    files contains (path, mtime_ns, size). Default listings omit aliases so all
    tasks are not collected twice. An explicit old name still resolves.
    """
    names = {Path(path).parent.name for path, _, _ in files}
    selectors, aliases = set(names), {}
    for path, _, _ in files:
        payload = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
        profiles = payload.get('profiles', {})
        if not isinstance(profiles, dict):
            raise ValueError(f'Recipe profiles must be a mapping: {path}')
        for profile, entry in profiles.items():
            if not isinstance(profile, str) or not profile or '@' in profile or not isinstance(entry, dict):
                raise ValueError(f'Invalid recipe profile {profile!r} in {path}')
            selector = f'{Path(path).parent.name}@{profile}'
            selectors.add(selector)
            alias = entry.get('task_name')
            if alias and alias != selector:
                if not isinstance(alias, str) or alias in names or alias in aliases:
                    raise ValueError(f'Duplicate or invalid recipe profile task_name alias {alias!r}')
                aliases[alias] = selector
    if set(aliases) & selectors:
        raise ValueError('Recipe profile task_name alias cannot shadow a recipe@profile selector.')
    return tuple(sorted(selectors)), aliases
