"""Reuse native measurements only within the same simulator physics frame."""
from __future__ import annotations

import copy
import math
import os


def physics_frame_stamp():
    try:
        from isaacsim.core.api import SimulationContext

        context = SimulationContext.instance()
        if context is None:
            return None
        index, seconds = context.current_time_step_index, float(context.current_time)
        dt = float(context.get_physics_dt())
        if index is None or not math.isfinite(seconds) or not math.isfinite(dt) or dt <= 0:
            return None
        return id(context), int(index), seconds, dt
    except (ImportError, AttributeError, TypeError, ValueError, RuntimeError):
        return None


def invalidate_physics_frame(owner):
    """Call after explicit state writes/reset, including writes without a tick."""
    owner.__dict__.pop('_physics_read_cache_stamp', None)
    owner.__dict__.pop('_physics_read_cache', None)


def read_physics_frame(owner, key, reader, *, stamp=None):
    if stamp is None:
        return reader()
    if getattr(owner, '_physics_read_cache_stamp', None) != stamp:
        owner._physics_read_cache_stamp = stamp
        owner._physics_read_cache = {}
    counters = getattr(owner, '_physics_read_cache_counts', None)
    if counters is None:
        counters = owner._physics_read_cache_counts = {}
    counts = counters.setdefault(key[0], {'hits': 0, 'misses': 0})
    cache = owner._physics_read_cache
    if key in cache:
        counts['hits'] += 1
        return copy.deepcopy(cache[key])
    counts['misses'] += 1
    result = reader()
    if result is not None and not (
        isinstance(result, dict) and (result.get('valid') is False or result.get('details_valid') is False)
    ):
        cache[key] = copy.deepcopy(result)
    return result


def read_sim_state(owner, key, reader):
    """The diagnostic switch leaves existing contact caching unchanged."""
    if os.environ.get('INTERNUTOPIA_LEGACY_OBSERVATIONS') == '1':
        return reader()
    return read_physics_frame(owner, key, reader, stamp=physics_frame_stamp())
