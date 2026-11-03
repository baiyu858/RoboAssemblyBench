"""Render only when needed while keeping every requested image sample fresh."""
from functools import reduce
from math import gcd


def render_interval(*, simulation_fps, live_stride=None, dataset_stride=None,
                    checker_stride=None, override=None):
    periods = [p for p in (live_stride, dataset_stride, checker_stride) if p is not None]
    if any(type(p) is not int or p <= 0 for p in periods):
        raise ValueError('Image capture periods must be positive integer physics steps.')
    if dataset_stride is not None:
        # Raw Cartesian export has a strict camera/action timing contract:
        # exactly one render for every dataset sample, not merely a fresh image.
        if any(p % dataset_stride for p in periods):
            raise ValueError('Raw dataset timing requires other capture periods to be multiples of its stride.')
        if override is not None and override != dataset_stride - 1:
            raise ValueError('Raw dataset timing requires rendering_interval + 1 == dataset stride.')
        return dataset_stride - 1
    if override is not None:
        if type(override) is not int or override < 0:
            raise ValueError('Rendering interval must be a nonnegative integer.')
        if any(p % (override + 1) for p in periods):
            raise ValueError('Rendering period must divide every enabled image capture period.')
        return override
    return reduce(gcd, periods) - 1 if periods else simulation_fps - 1
