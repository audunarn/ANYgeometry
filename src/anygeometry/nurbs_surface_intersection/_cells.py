"""Per-surface cached cell hulls and the lazy deterministic pair cover.

Hulls come from the existing exact ``NurbsSurface.patch_bounds`` (read-only
reuse; Fraction boxes, outward for the float chart map).  They are cached per
prepared surface so every partner pairing reuses them.  The pair hierarchy
starts from the two whole surfaces and splits one factor (one 2D chart box) at
a time, preferring interior knot lines (so cells are not cut artificially);
no Cartesian product of spans is ever formed.
"""
from __future__ import annotations

import math
import time
from fractions import Fraction

from ._source import SourceDefinition, WorkCounters as IntersectionStats

__all__ = ["PreparedSurface", "cover_pair"]

_ROOT = ((0.0, 1.0), (0.0, 1.0))


class PreparedSurface:
    """Trusted surface value, its exact net and a hull cache (per call, never shared across queries)."""

    def __init__(self, source: SourceDefinition, stats: IntersectionStats):
        self.surface = source.rebuild()
        self._net = None
        self.stats = stats
        self._hulls: dict = {}
        self._seams = (self._seam_values(self.surface.knots_u, self.surface.u_range),
                       self._seam_values(self.surface.knots_v, self.surface.v_range))

    @property
    def net(self):
        """Exact homogeneous net, computed lazily (after the caller's cancellation poll)."""
        if self._net is None:
            self._net = self.surface.exact_net()
        return self._net

    @staticmethod
    def _seam_values(knots, chart_range) -> tuple:
        r0, r1 = Fraction(chart_range[0]), Fraction(chart_range[1])
        low, high = sorted((r0, r1))
        values = set()
        for knot in set(knots):
            k = Fraction(knot)
            if low < k < high:
                values.add(float((k - r0) / (r1 - r0)))
        return tuple(sorted(value for value in values if 0.0 < value < 1.0))

    def hull(self, box):
        """``(lower, upper)`` exact Fraction corners bounding the surface over a chart box."""
        cached = self._hulls.get(box)
        if cached is not None:
            self.stats.hull_cache_hits += 1
            return cached
        lower, upper, _sigma2 = self.surface.patch_bounds(box, self.net)
        self.stats.hull_evaluations += 1
        self.stats.cells_prepared += 1
        made = (lower, upper)
        self._hulls[box] = made
        return made

    def split(self, box):
        """Two child boxes (knot seam nearest the middle of the longer chart side, else its midpoint)."""
        axis = 0 if box[0][1] - box[0][0] >= box[1][1] - box[1][0] else 1
        low, high = box[axis]
        seams = [value for value in self._seams[axis] if low < value < high]
        middle = (low + high) / 2
        at = min(seams, key=lambda value: (abs(value - middle), value)) if seams else middle
        if not low < at < high:
            return None
        left, right = list(box), list(box)
        left[axis], right[axis] = (low, at), (at, high)
        return tuple(left), tuple(right)


def _separated(a, b) -> bool:
    return any(a[1][k] < b[0][k] or b[1][k] < a[0][k] for k in range(3))


def _diameter(hull) -> Fraction:
    return max(hi - lo for lo, hi in zip(*hull))


def cover_pair(first: PreparedSurface, second: PreparedSurface, *, limits, resolution: Fraction,
               stats: IntersectionStats, tick, started: float):
    """Depth-first cover of the pair hierarchy.

    Returns ``(True, "", ())`` only when every node was excluded by certified
    hull separation.  The first non-excluded node that cannot be refined
    further (resolution, depth, split or box budget) is returned as
    ``(False, reason, (box_a, box_b))``: no partial discovery is ever reported
    as disjoint.  ``tick`` is called once per node (cancellation/integrity).
    """
    # max_subdivision_depth bounds the TOTAL split depth of a product branch (both factors together)
    stack = [(_ROOT, _ROOT, 0, 0)]
    while stack:
        stats.peak_stack = max(stats.peak_stack, len(stack))
        box_a, box_b, depth_a, depth_b = stack.pop()
        tick()
        if stats.boxes_visited >= limits.max_boxes_per_pair:      # authority checked before consuming
            return False, "box budget exhausted before the pair cover completed", ((box_a, box_b),)
        stats.boxes_visited += 1
        stats.max_depth = max(stats.max_depth, depth_a + depth_b)
        hull_a, hull_b = first.hull(box_a), second.hull(box_b)
        if _separated(hull_a, hull_b):
            stats.excluded_pairs += 1
            continue
        diameter_a, diameter_b = _diameter(hull_a), _diameter(hull_b)
        if max(diameter_a, diameter_b) <= resolution:
            return (False, "hulls still overlap at the length resolution (intersection, tangency or "
                           "coincidence is not classified by the foundation kernel)", ((box_a, box_b),))
        split_first = diameter_a >= diameter_b
        if depth_a + depth_b >= limits.max_subdivision_depth:
            return False, "subdivision depth exhausted with overlapping hulls", ((box_a, box_b),)
        children = (first if split_first else second).split(box_a if split_first else box_b)
        if children is None:
            return False, "box can no longer be split with overlapping hulls", ((box_a, box_b),)
        stats.splits += 1
        for child in reversed(children):
            if split_first:
                stack.append((child, box_b, depth_a + 1, depth_b))
            else:
                stack.append((box_a, child, depth_a, depth_b + 1))
    return True, "", ()
