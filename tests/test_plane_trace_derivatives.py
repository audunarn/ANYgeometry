"""PlaneSide uses true trace derivatives, not positive-area chart authority."""
from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from anygeometry import GeometryError, Plane
from anygeometry.nurbs import NurbsSurface
from anygeometry.planar_exact import ExactPlane
from anygeometry.plane_source_reference import PlaneSide
from anygeometry.regular_source_reference import SourceNode, source_node, source_side
from anygeometry.support_curves import PlaneNurbsCurve


def parabola():
    return NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.),
        tuple(tuple((u, v, z) for v in (0., 1.)) for u, z in ((0., 0.), (.5, 0.), (1., 1.))))


def authentic_section_side():
    surface = parabola()
    plane = Plane((0., .5, 0.), (1., 0., 0.), (0., 0., 1.))
    frame = ExactPlane.from_plane(plane)
    curve = PlaneNurbsCurve.from_plane(plane, surface)
    nodes = tuple(source_node(curve, t) for t in (Q(0), Q(1)))
    native = source_side(curve, (Q(0), Q(1)), *nodes)
    endpoints = tuple(SourceNode(('plane-test', i), tuple((p, p) for p in frame.uv(row[2])))
                      for i, row in enumerate(curve._exact_points((0, 1))))
    return PlaneSide(frame, native, *endpoints)


@pytest.mark.parametrize('t', [Q(0), Q(1, 2), Q(1)])
def test_authentic_plane_section_point_derivative_has_exact_literal_value(t):
    side = authentic_section_side()
    assert side.native.box(t, t)[1] == (Q(1, 2), Q(1, 2))
    # World section is (t, 1/2, t^2), in the literal plane's (x,z) basis.
    assert side.derivative(t, t) == ((1, 1), (2*t, 2*t))


def test_authentic_closed_trace_derivative_preserves_nonzero_interval():
    side = authentic_section_side()
    before = (side.native, side.start, side.end, side.curve.surface.controls)
    assert side.native.box(Q(1, 4), Q(3, 4))[1] == (Q(1, 2), Q(1, 2))
    assert side.derivative(Q(1, 4), Q(3, 4)) == ((1, 1), (Q(1, 2), Q(3, 2)))
    assert (side.native, side.start, side.end, side.curve.surface.controls) == before


def protocol_side(surface, box):
    # Internal adapter dispatch unit only; this is NOT a source/owner receipt.
    native = SimpleNamespace(curve=SimpleNamespace(surface=surface),
        box=lambda lo, hi: box, derivative=lambda lo, hi: ((Q(1), Q(1)), (Q(0), Q(0))))
    frame = ExactPlane.from_plane(Plane((0., 0., 0.), (1., 0., 0.), (0., 0., 1.)))
    return PlaneSide(frame, native, None, None)


def test_positive_area_adapter_keeps_original_derivative_service(monkeypatch):
    import anygeometry.plane_source_reference as adapter
    original = adapter.derivative_intervals
    calls = []
    def observe(surface, box):
        calls.append(box)
        return original(surface, box)
    monkeypatch.setattr(adapter, 'derivative_intervals', observe)
    monkeypatch.setattr(adapter, '_trace_derivative_intervals',
        lambda *args: pytest.fail('positive-area contract was replaced'), raising=False)
    box = ((Q(1, 4), Q(3, 4)), (Q(1, 4), Q(3, 4)))
    assert protocol_side(parabola(), box).derivative(0, 1) == ((1, 1), (Q(1, 2), Q(3, 2)))
    assert calls == [box]


def test_trace_adapter_hulls_both_one_sided_knot_derivatives():
    surface = NurbsSurface(1, 1, (0., 0., .5, 1., 1.), (0., 0., 1., 1.),
        tuple(((x, 0., 0.), (x, 1., 0.)) for x in (0., .5, 2.)))
    side = protocol_side(surface, ((Q(1, 2), Q(1, 2)), (Q(1, 3), Q(2, 3))))
    assert side.derivative(0, 1) == ((1, 3), (0, 0))


@pytest.mark.parametrize('box', [((-1, -1), (0, 1)), ((1, 0), (0, 0))])
def test_trace_adapter_preserves_domain_refusal(box):
    with pytest.raises(GeometryError):
        protocol_side(parabola(), box).derivative(0, 1)
