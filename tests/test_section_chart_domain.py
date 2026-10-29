"""Implicit section curves retain their declared support chart in material domains."""
import math
from fractions import Fraction as Q
import numpy as np
import pytest
from anygeometry import Plane
from anygeometry.nurbs import NurbsSurface
from anygeometry.nurbs import NurbsCurve, CurveOnSurface
from anygeometry.support_curves import PlaneNurbsCurve
from anygeometry.section_chart import SectionChart
from anygeometry.material_arrangement import MaterialDomain, ArrangementPath


def circle():
    a = (1., -1., 1.)
    support = NurbsSurface(2, 2, (0., 0., 0., 1., 1., 1.), (0., 0., 0., 1., 1., 1.),
        tuple(tuple((i-1., j-1., a[i]+a[j]) for j in range(3)) for i in range(3)))
    curve = PlaneNurbsCurve.from_plane(Plane((0., 0., .5), (1., 0., 0.), (0., 1., 0.)), support)
    return curve


def test_implicit_chart_level_events_and_material_contains():
    curve = circle()
    adapter = SectionChart(curve)
    events = adapter.level_roots(1, Q(1, 2))
    points = curve.chart(np.asarray(events))
    assert len(events) == 2
    assert np.allclose(points[:, 1], .5, atol=1e-12)
    assert np.allclose(sorted(points[:, 0]), [.5-math.sqrt(.125), .5+math.sqrt(.125)])
    loop = (ArrangementPath(curve),)
    domain = MaterialDomain(1, curve.surface, (loop,))
    assert domain.contains_loop(loop, (.5, .5), 1e-9)
    assert not domain.contains_loop(loop, (.01, .01), 1e-9)
    assert np.allclose(domain.uv(curve, .3), curve.chart(.3))
    assert abs(abs(domain.area_loop(loop, 1e-8))-math.pi/8) < 1e-8


def test_reversed_trimmed_section_events():
    curve = circle()
    adapter = SectionChart(curve)
    full = adapter.level_roots(1, .5)
    reversed_curve = curve.subcurve(1., 0.)
    assert np.allclose(SectionChart(reversed_curve).level_roots(1, .5), sorted(1-t for t in full))


def test_boundary_flags_and_certified_section_sampling():
    curve = circle()
    loop = (ArrangementPath(curve),)
    domain = MaterialDomain(1, curve.surface, (loop,))
    point = curve.chart(0.)
    assert domain.contains_loop(loop, point, 1e-7, boundary=True)
    assert not domain.contains_loop(loop, point, 1e-7, boundary=False)
    polyline, bound = SectionChart(curve).certified_polyline(.05)
    assert bound <= .05
    assert np.allclose(polyline[0], polyline[-1])


def test_certified_transverse_trim_crossings_keep_isolators():
    from anygeometry.section_events import section_segment_events
    section = circle()
    trim = CurveOnSurface(section.surface, NurbsCurve(1, (0., 0., 1., 1.),
        ((0., .4), (1., .4)), (1., 4.)))
    events = section_segment_events(section, trim)
    assert len(events) == 2
    for event in events:
        a, b = event.section_interval
        c, d = event.segment_interval
        assert isinstance(a, Q) and b-a <= Q(1, 2**48)
        point = section.chart(float((a+b)/2))
        other = trim.chart(float((c+d)/2))
        assert np.allclose(point, other, atol=1e-12)
        assert abs(point[1]-.4) < 1e-12


@pytest.mark.parametrize('slicing', (0, 1))
def test_ambient_derivatives_survive_zero_width_root_boxes(slicing):
    from anygeometry.branch_predicates import _Arc
    arc = object.__new__(_Arc)
    arc.u_box, arc.v_box, arc.slicing = (Q(2), Q(4)), (Q(5), Q(9)), slicing
    # f(u,v)=2u+3v. Its physical-knot derivatives remain (2,3)
    # even after restricting the root box to an exact u or v isocurve.
    coefficients = [[Q(19), Q(31)], [Q(23), Q(35)]]
    box = ((Q(2), Q(4)), (Q(7), Q(7))) if slicing == 0 else ((Q(5), Q(9)), (Q(3), Q(3)))
    assert arc.derivative_ranges(coefficients, *box) == ((Q(2), Q(2)), (Q(3), Q(3)))
    arc.g1 = [[Q(-1), Q(1)], [Q(-1), Q(1)]] if slicing == 0 else [[Q(-1), Q(-1)], [Q(1), Q(1)]]
    arc.g2 = [[Q(-1), Q(-1)], [Q(1), Q(1)]] if slicing == 0 else [[Q(-1), Q(1)], [Q(-1), Q(1)]]
    assert arc.monotone(*box)


def test_exact_isocurve_joint_exclusion_accounts_for_both_crossings(monkeypatch):
    import anygeometry.nurbs_section_faces as faces
    section = circle()
    vertical = PlaneNurbsCurve.from_plane(Plane((.2, 0., 0.), (0., 1., 0.), (0., 0., 1.)), section.surface)
    # The two intersections are regular; declaring only one joint must not
    # excuse the other crossing, even though the ambient root box is a line.
    radius = math.sqrt(.46)/2
    joints = [(Q(.6), Q(.5-radius)), (Q(.6), Q(.5+radius))]
    monkeypatch.setattr(faces, 'MAX_DEPTH', 16)
    assert faces.section_pair_clean(vertical, section, joints, Q(1, 100000)) is True
    assert faces.section_pair_clean(vertical, section, joints[:1], Q(1, 100000)) is None


@pytest.mark.parametrize('slicing', (0, 1))
def test_constant_isocurve_rectangle_refines_transverse_enclosure(slicing):
    from anygeometry.branch_predicates import _Arc
    from types import SimpleNamespace
    arc = object.__new__(_Arc)
    arc.u_box, arc.v_box, arc.x_box = (Q(0), Q(1)), (Q(0), Q(1)), (Q(0), Q(1))
    arc.slicing = slicing
    # G1=x-3/5 has G1_s identically zero. Its exact transverse
    # coordinate must not be limited by the old 24+level hull floor.
    arc.g1 = [[Q(-3, 5), Q(2, 5)], [Q(-3, 5), Q(2, 5)]] if slicing == 0 else [
        [Q(-3, 5), Q(-3, 5)], [Q(2, 5), Q(2, 5)]]
    arc.arc = SimpleNamespace(first=0, second=1)
    class Curve:
        @staticmethod
        def _tight_x(*args, **kwargs):
            raise AssertionError('a constant isocurve must use monotone endpoint brackets')
        @staticmethod
        def _ordered_root_rect(_arc, root):
            return (Q(root), Q(root)), (Q(3, 5), Q(3, 5))
    arc.curve = Curve()
    lo, hi = Q(1, 4), Q(1, 4)+Q(1, 2**32)
    s, x = arc.rectangle(lo, hi, 8, (Q(0), Q(1)))
    assert s == (lo, hi)
    assert x[0] <= Q(3, 5) <= x[1]
    assert x[1]-x[0] <= (hi-lo)/64


def test_endpoint_contact_is_enclosed_without_assuming_interior_crossing(monkeypatch):
    import anygeometry.nurbs_section_faces as faces
    support = circle().surface
    circle_quarter = PlaneNurbsCurve.from_plane(Plane((0., 0., .25), (1., 0., 0.), (0., 1., 0.)), support)
    vertical = PlaneNurbsCurve.from_plane(Plane((0., 0., 0.), (0., 1., 0.), (0., 0., 1.)), support)
    # The first end at v=1/4 lies exactly on z=1/4. Its zero
    # endpoint sign grants no proof; the full possible-root enclosure must
    # fit the declared joint ball despite a deliberately shallow depth cap.
    parameter = .25 if vertical.chart(0.)[1] < vertical.chart(1.)[1] else .75
    first = vertical.subcurve(parameter, .5)
    joint = (Q(1, 2), Q(1, 4))
    monkeypatch.setattr(faces, 'MAX_DEPTH', 12)
    assert faces.section_pair_clean(first, circle_quarter, [joint], Q(1, 10**10)) is True


def test_segment_isocurve_endpoint_uses_full_root_enclosure(monkeypatch):
    import anygeometry.nurbs_section_faces as faces
    support = circle().surface
    vertical = PlaneNurbsCurve.from_plane(Plane((.2, 0., 0.), (0., 1., 0.), (0., 0., 1.)), support)
    monkeypatch.setattr(faces, 'MAX_DEPTH', 12)
    assert faces.segment_section(vertical, (Q(0), Q(0)), (Q(1), Q(0)),
        [(Q(3, 5), Q(0))], Q(1, 10**10)) is True
    # An undeclared endpoint contact must not receive the joint exclusion.
    assert faces.segment_section(vertical, (Q(0), Q(0)), (Q(1), Q(0)),
        [], Q(1, 10**10)) is not True
