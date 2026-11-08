"""Reuse read-only measurements within one completed physics frame."""
from __future__ import annotations

import copy
import math


def physics_frame_stamp():
    """Use the simulator clock, never a policy/phase counter, as freshness."""
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


def read_physics_frame(owner, key, reader):
    """Return independent values; unavailable clocks and failed reads bypass storage.

    Each new physics frame, clock reset or dt change drops the previous frame.
    Namespace counters are retained for performance reports, without sensor reads.
    """
    stamp = physics_frame_stamp()
    if stamp is None:
        return reader()
    if getattr(owner, '_physics_read_cache_stamp', None) != stamp:
        owner._physics_read_cache_stamp = stamp
        owner._physics_read_cache = {}
    cache = owner._physics_read_cache
    counters = getattr(owner, '_physics_read_cache_counts', None)
    if counters is None:
        counters = owner._physics_read_cache_counts = {}
    counts = counters.setdefault(key[0], {'hits': 0, 'misses': 0})
    if key in cache:
        counts['hits'] += 1
        return copy.deepcopy(cache[key])
    counts['misses'] += 1
    value = reader()
    # A temporarily unavailable probe must be retried, even within the frame.
    if value is not None and not (
        isinstance(value, dict) and (value.get('valid') is False or value.get('details_valid') is False)
    ):
        cache[key] = copy.deepcopy(value)
    return value
