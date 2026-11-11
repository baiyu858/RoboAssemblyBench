"""One sensor readback per rendered frame; physics-only ticks reuse that frame."""
from __future__ import annotations

import copy
from collections import OrderedDict

import numpy as np


def _snapshot(value):
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.setflags(write=False)
        return result
    if isinstance(value, dict):
        return type(value)((k, _snapshot(v)) for k, v in value.items())
    if isinstance(value, list):
        return [_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(v) for v in value)
    return copy.deepcopy(value)


def _view(value):
    if isinstance(value, np.ndarray):
        return value.view()
    if isinstance(value, dict):
        return type(value)((k, _view(v)) for k, v in value.items())
    if isinstance(value, list):
        return [_view(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_view(v) for v in value)
    return value


class RenderFrameCache:
    def __init__(self):
        self.frame = self.captured_frame = None
        self.rendered = False
        self.data = None
        self.counts = dict(captures=0, hits=0, deferred=0)

    def clear(self):
        # Keep lifetime counters available to the final report after cleanup.
        self.frame = self.captured_frame = self.data = None
        self.rendered = False

    def prepare(self, frame, *, rendered):
        self.frame, self.rendered = frame, bool(rendered)

    def read(self, reader):
        if self.frame is None:
            return reader()
        if not self.rendered or self.captured_frame == self.frame:
            self.counts['hits' if self.data is not None else 'deferred'] += 1
            return OrderedDict() if self.data is None else _view(self.data)
        data = reader()
        rgba = data.get('rgba') if data else None
        if not data or ('rgba' in data and (rgba is None or np.size(rgba) == 0)):
            return data
        self.data = _snapshot(data)
        self.captured_frame = self.frame
        self.counts['captures'] += 1
        return _view(self.data)
