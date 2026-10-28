"""Portable replay of the first unresolved regular rational nested-loop cell.

Captured by root at source 9282169c..., artifact
first-unresolved-replay-1792853565470591400.json. Author has not run these tests.
"""
from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from anygeometry.nurbs import NurbsSurface
from anygeometry.tolerance import TolerancePolicy
from anygeometry.nurbs_surface_intersection import _atlas
from anygeometry.nurbs_surface_intersection._certificate import _Pair, certify_prepared
from anygeometry.nurbs_surface_intersection._intervals import Interval
from anygeometry.nurbs_surface_intersection._source import SourceDefinition, WorkCounters
from test_fast_path import nested_pair


CELL = tuple(tuple(Q(x) for x in axis) for axis in (
    ("9007199252014253/72057594037927936", "1/8"),
    ("4503529725714989/9007199254740992", "1/2"),
    ("140737488312723/1125899906842624", "2251799813685257/18014398509481984"),
    ("18446457756528788457/36893488147419103232", "1/2")))
CLIPS = ((Q(0), Q(1)),) * 4


def nested_rational_pair():
    first, second = nested_pair()
    weights = tuple((float(w),)*5 for w in (1, 2, 2, 2, 1))
    controls = []
    for i, (row_a, row_b) in enumerate(zip(first.controls, second.controls)):
        row = []
        for j, (a, b) in enumerate(zip(row_a, row_b)):
            z = Q(a[2])+(Q(b[2])-Q(a[2]))/Q(weights[i][j])
            assert Q(float(z)) == z
            row.append((a[0], a[1], float(z)))
        controls.append(tuple(row))
    return (NurbsSurface(4, 4, first.knots_u, first.knots_v, first.controls, weights),
            NurbsSurface(4, 4, second.knots_u, second.knots_v, tuple(controls), weights))


def test_captured_regular_cell_certifies_alternate_chart():
    first, second = nested_rational_pair()
    pair = _Pair(first, second, CLIPS).anchored(CELL)
    order = _atlas._free_order(pair, CELL, "filtered")
    assert order == [0, 2, 3, 1]
    assert _atlas._graph_axes(order, CELL, 40, 40) == tuple(order)
    certificates = {}
    for free in order:
        region = _atlas._graph_region(CELL, CLIPS, free)
        assert all(c <= a <= lo <= hi <= b <= d
                   for (lo, hi), (a, b), (c, d) in zip(CELL, region, CLIPS))
        certificate = certify_prepared(pair, CLIPS, free,
            [Interval(*region[k]) for k in range(4) if k != free], Interval(*region[free]),
            interval_mode="filtered", stats=WorkCounters())
        certificates[free] = certificate
        assert certificate.weights_positive and certificate.immersed
    assert not certificates[0].certified and not certificates[2].certified
    assert certificates[1].certified and certificates[3].certified
    for free in (1, 3):
        region = _atlas._graph_region(CELL, CLIPS, free)
        assert certificates[free].regular
        assert all(image.strictly_inside(Interval(*region[k]))
                   for image, k in zip(certificates[free].unknown_enclosure,
                                       (k for k in range(4) if k != free)))


def test_graph_axis_policy_and_exact_boundary_clipping():
    broad = ((Q(0), Q(1)),)*4
    order = [0, 2, 3, 1]
    assert _atlas._graph_axes(order, broad, 0, 40) == (0,)
    assert _atlas._graph_axes(order, broad, 40, 40) == tuple(order)
    assert _atlas._graph_axes(order, broad, 0, 40, True) == tuple(order)
    thin = ((Q(0), Q(1, 16)),)+broad[1:]
    assert _atlas._graph_axes(order, thin, 0, 40) == tuple(order)
    singleton = ((Q(0), Q(0)),)+broad[1:]
    assert _atlas._graph_axes(order, singleton, 40, 40) == (2, 3, 1)
    assert _atlas._graph_region(broad, CLIPS, 0) == CLIPS


def _production_fallback(monkeypatch, cancel=False):
    knots = (0., 0., 1., 1.)
    first = NurbsSurface(1, 1, knots, knots,
        tuple(tuple((x, y, 0.) for y in (-1., 1.)) for x in (0., 1.)))
    second = NurbsSurface(1, 1, knots, knots,
        tuple(tuple((x, 0., z) for z in (-1., 1.)) for x in (-1., 2.)))
    monkeypatch.setattr(_atlas, "_free_order", lambda *a: [1, 0, 2, 3])
    actual = _atlas.certify_prepared
    attempts = []
    def observe(*args, **kwargs):
        if not kwargs.get("for_root_filter", False):
            attempts.append(args[2])
        return actual(*args, **kwargs)
    monkeypatch.setattr(_atlas, "certify_prepared", observe)
    def tick():
        if cancel and attempts == [1]:
            raise RuntimeError("cancel before alternate proof")
    stats = WorkCounters()
    limits = SimpleNamespace(max_boxes_per_pair=128, max_subdivision_depth=0,
        max_newton_iterations=8, max_trace_segments=64, max_components=8, interval_mode="filtered")
    if cancel:
        with pytest.raises(RuntimeError, match="before alternate"):
            _atlas.run_atlas(SourceDefinition.capture(first), SourceDefinition.capture(second),
                             TolerancePolicy(), limits, stats, tick)
        assert attempts == [1]
    else:
        report = _atlas.run_atlas(SourceDefinition.capture(first), SourceDefinition.capture(second),
                                 TolerancePolicy(), limits, stats, tick)
        assert attempts[:2] == [1, 0]
        assert report.complete_local_cover and report.unresolved == 0 and report.covered == 1
        assert report.witnesses[0].free_coordinate == 0
        assert stats.certificate_attempts >= 3


def test_production_terminal_chart_fallback(monkeypatch):
    _production_fallback(monkeypatch)


def test_cancellation_before_production_alternate_chart(monkeypatch):
    _production_fallback(monkeypatch, cancel=True)
