"""Exact interior events for authored rational carriers with cropped domains."""
from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from anygeometry import GeometryError, Plane
from anygeometry.nurbs import NurbsCurve, NurbsSurface, CurveOnSurface
from anygeometry.support_curves import PlaneNurbsCurve
from anygeometry.section_events import section_segment_events, _exact_affine_segment_event
from anygeometry.nurbs_section_faces import _Chart


def fixture(reverse=False):
    support = NurbsSurface(1, 1, (0, 0, 1, 1), (0, 0, 1, 1),
        (((-1.5, -1.5, 0), (-1.5, 2.5, 0)), ((2.5, -1.5, 0), (2.5, 2.5, 0))))
    controls, weights, domain = ((.0625, .375), (.6875, .375)), (1, 3), (3, 5)
    if reverse:
        controls, weights, domain = controls[::-1], weights[::-1], domain[::-1]
    trim = CurveOnSurface(support, NurbsCurve(1, (2, 2, 6, 6), controls, weights, domain=domain))
    section = PlaneNurbsCurve.from_plane(Plane((.5, 0, 0), (0, 1, 0), (0, 0, 1)), support)
    return section, trim


@pytest.mark.parametrize('reverse_trim', [False, True])
@pytest.mark.parametrize('reverse_section', [False, True])
def test_exact_interior_cropped_crossing_retains_both_native_domains(reverse_trim, reverse_section):
    section, trim = fixture(reverse_trim)
    if reverse_section:
        section = section.subcurve(1, 0)
    events = section_segment_events(section, trim)
    assert len(events) == 1
    event = events[0]
    assert event.chart_enclosure == ((Q(1, 2), Q(1, 2)), (Q(3, 8), Q(3, 8)))
    # Reversing controls, weights AND the active native domain preserves the
    # normalized geometric direction; the native knot itself is reversed.
    expected = Q(3, 8)
    assert event.segment_interval == (expected, expected)
    a, b = map(Q, trim.pcurve.domain)
    assert a+(b-a)*expected == (Q(17, 4) if reverse_trim else Q(15, 4))
    assert event.section_interval[0] == event.section_interval[1]
    assert 0 < event.section_interval[0] < 1
    # Independent public point enclosure checks the arc-to-section mapping,
    # including reversal; dyadic fixture stations convert to float exactly.
    enclosure = section.chart_enclosure(float(event.section_interval[0]))
    assert all(low <= value <= high for (low, high), value in zip(enclosure, (Q(1, 2), Q(3, 8))))


def test_affine_line_extension_pole_and_active_boundary_are_not_inverted(monkeypatch):
    import anygeometry.section_events as owner
    section, trim = fixture()
    def forbidden(*args):
        raise AssertionError('outside/endpoint active segment reached inverse')
    monkeypatch.setattr(owner, 'rational_segment_parameter', forbidden)
    for u, expected in ((Q(1), False), (Q(3, 8), None), (Q(5, 8), None)):
        view = SimpleNamespace(g1=((-u, -u), (1-u, 1-u)),
            g2=((-Q(3, 8), Q(5, 8)), (-Q(3, 8), Q(5, 8))),
            u_box=(Q(0), Q(1)), v_box=(Q(0), Q(1)), slicing=1,
            arc=SimpleNamespace(s0=Q(0), s1=Q(1)))
        context = SimpleNamespace(view=view, s_from=Q(0), s_to=Q(1), k=0, count=1, a=Q(0), b=Q(1))
        assert _exact_affine_segment_event(context, _Chart(section.surface), trim.pcurve, 0) is expected


def test_affine_event_cancellation_is_not_hidden():
    section, trim = fixture()
    with pytest.raises(GeometryError, match='cancelled'):
        section_segment_events(section, trim, cancellation_check=lambda: True)


def test_coincident_and_nonaffine_pairs_keep_the_conservative_classifier():
    section, trim = fixture()
    line = ((-Q(1, 2), -Q(1, 2)), (Q(1, 2), Q(1, 2)))
    for first in (line, ((-Q(1, 2), -Q(1, 2)), (Q(1, 2), Q(3, 4)))):
        context = SimpleNamespace(view=SimpleNamespace(g1=first, g2=line))
        assert _exact_affine_segment_event(context, _Chart(section.surface), trim.pcurve, 0) is None
