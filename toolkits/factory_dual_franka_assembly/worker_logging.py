"""Bound repeated known native warning output while preserving other messages."""
from __future__ import annotations

import time

MATERIAL_FACE_WARNING = (
    'PhysX warning: PxShape::getMaterialFromInternalFaceIndex received ' '0xFFFFffff as input - returning NULL.'
)


class WorkerLogWriter:
    """Keep first occurrences and periodic counts; errors are never aggregated."""

    def __init__(self, stream, *, aggregate=True, clock=time.monotonic):
        self.stream, self.aggregate, self.clock = stream, bool(aggregate), clock
        self.counts = dict(
            input_lines=0,
            written_lines=0,
            material_warning_occurrences=0,
            material_warning_suppressed=0,
            aggregation_enabled=self.aggregate,
        )
        self.next_report = self.clock() + 60.0
        self.closed = False

    def _emit(self, line):
        self.stream.write(line)
        self.counts['written_lines'] += 1

    def _summary(self):
        self._emit(
            '[validation-log] repeated_material_face_warning '
            f"occurrences={self.counts['material_warning_occurrences']} "
            f"suppressed={self.counts['material_warning_suppressed']}\n"
        )
        self.stream.flush()

    def write(self, line):
        self.counts['input_lines'] += 1
        known = (
            '[Warning] [omni.physx.plugin]' in line
            and MATERIAL_FACE_WARNING in line
            and '[Error]' not in line
            and '[Fatal]' not in line
        )
        if known:
            self.counts['material_warning_occurrences'] += 1
            if self.aggregate and self.counts['material_warning_occurrences'] > 3:
                self.counts['material_warning_suppressed'] += 1
                if self.clock() >= self.next_report:
                    self._summary()
                    self.next_report = self.clock() + 60.0
                return
        self._emit(line)
        # Keep progress/errors visible immediately; repeated warnings flush in
        # aggregate summaries instead of writing megabytes to disk every tick.
        if not known:
            self.stream.flush()

    def close(self):
        if not self.closed:
            if self.counts['material_warning_suppressed']:
                self._summary()
            self.stream.flush()
            self.closed = True
        return dict(self.counts)
