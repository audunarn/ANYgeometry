"""Run the existing scalar plane x field topology engine on the exact oracle and freeze its answer.

The engine (plane_nurbs.plane_nurbs_section, read-only) is executed inside a COPIED ``contextvars`` context whose
``CURRENT_WORK`` is a fresh per-query worker charging only this query's limits, so no caller context or shared
global is touched.  Retries inside the engine share the worker (no resets).  Mutable engine records are copied
into frozen records; the engine's lists are never exposed.
"""
from __future__ import annotations

import contextvars
from fractions import Fraction
from types import SimpleNamespace

from ..plane_nurbs import CURRENT_WORK, PlaneNurbsNotCertified, plane_nurbs_section
from ._intervals import float_down, float_up
from ._records import ArcRecord, ComponentRecord, FrozenBox, FrozenRoot
from ._source import NurbsSurfaceIntersectionIntegrityError
from ._structural import ScalarField

__all__ = ["run_scalar_section", "freeze_section", "chart_box", "world_box", "Enclosures", "root_projection"]

_PLANE = SimpleNamespace(origin=(0.0, 0.0, 0.0), u_vector=(1.0, 0.0, 0.0), v_vector=(0.0, 1.0, 0.0))


class _Worker:
    def __init__(self, limit: int, depth: int, tick, counters):
        self.limit, self.max_depth, self.subdivisions = limit, depth, 0
        self._tick, self._counters = tick, counters

    def take(self) -> bool:
        self._tick()                              # cancellation / integrity at every engine box
        if self._counters.engine_box_charges >= self.limit:
            return False                          # the engine raises PlaneNurbsNotCertified (resource refusal)
        self._counters.engine_box_charges += 1
        return True


def run_scalar_section(source_a, certificate, limits, tick, counters):
    """The engine's PlaneNurbsSection; ``PlaneNurbsNotCertified`` signals an explicit refusal."""
    field = ScalarField(source_a, certificate)
    worker = _Worker(limits.max_boxes_per_pair, limits.max_subdivision_depth, tick, counters)

    def run():
        CURRENT_WORK.set(worker)                  # set in the copied context only
        return plane_nurbs_section(_PLANE, field)

    try:
        return contextvars.copy_context().run(run)
    finally:
        counters.engine_subdivisions = worker.subdivisions       # preserved on refusal, cancel and exceptions


def root_projection(root, slicing: int):
    """Slicing-coordinate enclosure of a (frozen or engine) root."""
    return (root.value, root.value) if root.axis == slicing else (root.lo, root.hi)


def _chart(value: Fraction, chart_range) -> Fraction:
    r0, r1 = Fraction(chart_range[0]), Fraction(chart_range[1])
    return (value - r0) / (r1 - r0)


def chart_box(native_box, u_range, v_range):
    """Outward float chart enclosure ``((u0,u1),(v0,v1))`` of a native box for one surface's selected ranges."""
    out = []
    for (lo, hi), chart_range in zip(native_box, (u_range, v_range)):
        a, b = sorted((_chart(lo, chart_range), _chart(hi, chart_range)))
        out.append((max(0.0, float_down(a)), min(1.0, float_up(b))))
    return tuple(out)


class Enclosures:
    """Per-call reuse of the originals' exact nets for ``patch_bounds`` (no cross-query cache)."""

    def __init__(self, surface_a, surface_b):
        self.surfaces = (surface_a, surface_b)
        self._nets = [None, None]
        self.bounds_calls = 0

    def world(self, charts):
        lows, highs = [], []
        for k, (surface, box) in enumerate(zip(self.surfaces, charts)):
            if self._nets[k] is None:
                self._nets[k] = surface.exact_net()
            lower, upper, _ = surface.patch_bounds(box, self._nets[k])
            self.bounds_calls += 1
            lows.append([float_down(x) for x in lower])
            highs.append([float_up(x) for x in upper])
        low = tuple(max(x, y) for x, y in zip(*lows))
        high = tuple(min(x, y) for x, y in zip(*highs))
        if any(a > b for a, b in zip(low, high)):
            # both outward hulls contain every true intersection point, so an empty intersection means the
            # descriptors/supports are inconsistent: never fall back to the union
            raise NurbsSurfaceIntersectionIntegrityError("the two original-support hulls of an arc do not meet")
        return low, high


def world_box(enclosures: Enclosures, native_box):
    charts = tuple(chart_box(native_box, s.u_range, s.v_range) for s in enclosures.surfaces)
    low, high = enclosures.world(charts)
    return charts, low, high


def freeze_section(section, enclosures: Enclosures, max_components=None, max_arcs=None):
    """``(roots, boxes, components)`` as deeply immutable records.

    The engine has already materialized its paths when this runs (no shared-source edit is possible), so the
    caps stop OUR record accumulation early (no further enclosure work); they are not engine memory limits.
    """
    roots = tuple(FrozenRoot(r.axis, r.value, r.lo, r.hi, tuple(r.poly), tuple(r.edge), r.patch, r.ident)
                  for r in section.roots)
    boxes = tuple(FrozenBox(b.patch, tuple(b.u_range), tuple(b.v_range),
                            tuple(tuple(row) for row in b.coefficients), b.kind) for b in section.boxes)
    components = []
    total_arcs = 0
    for index, path in enumerate(section.paths):
        if max_components is not None and index >= max_components:
            raise PlaneNurbsNotCertified("component limit exceeded")
        total_arcs += len(path.arcs)
        if max_arcs is not None and total_arcs > max_arcs:
            raise PlaneNurbsNotCertified("trace-segment limit exceeded")
        arcs = []
        for box_index, first, second in path.arcs:
            box = boxes[box_index]
            slicing = 0 if box.kind == "v" else 1
            spans = [root_projection(roots[first], slicing), root_projection(roots[second], slicing)]
            s_range = box.u_range if slicing == 0 else box.v_range
            s_lo = max(min(s[0] for s in spans), s_range[0])
            s_hi = min(max(s[1] for s in spans), s_range[1])
            native = ((s_lo, s_hi), box.v_range) if slicing == 0 else (box.u_range, (s_lo, s_hi))
            charts, low, high = world_box(enclosures, native)
            arcs.append(ArcRecord(box_index, first, second, slicing, native, charts[0], charts[1], low, high))
        if not arcs:
            raise PlaneNurbsNotCertified("a section path without arcs")
        comp_charts = tuple(tuple((min(a.__getattribute__(name)[k][0] for a in arcs),
                                   max(a.__getattribute__(name)[k][1] for a in arcs)) for k in range(2))
                            for name in ("first_chart", "second_chart"))
        low = tuple(min(a.world_lo[k] for a in arcs) for k in range(3))
        high = tuple(max(a.world_hi[k] for a in arcs) for k in range(3))
        components.append(ComponentRecord(index, bool(path.closed), tuple(path.roots), tuple(arcs),
                                          comp_charts[0], comp_charts[1], low, high))
    return roots, boxes, tuple(components)
