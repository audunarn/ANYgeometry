"""A finite segment's supporting-line extension is not a boundary contact."""
from fractions import Fraction as Q

import pytest

from anygeometry import GeometryError
from anygeometry.entities import OrientedEdge
from anygeometry.planar_exact import ExactPlane
from anygeometry.plane_section_plate import _contains
from anygeometry.support_curves import PlaneNurbsCurve
from test_plane_split import bowl


def mixed_half_circle():
    model, face, plane = bowl()
    full = PlaneNurbsCurve.from_plane(plane, model.faces[face].surface, 0)
    half = full.subcurve(0., .5)
    start, end = model.add_points((half.evaluate(0.), half.evaluate(1.)))
    curved = model.add_curve(start, end, half)
    straight = model.add_line(end, start)
    frame = ExactPlane.from_plane(plane)
    a = frame.uv(model.vertex_position(end))
    b = frame.uv(model.vertex_position(start))
    return model, (OrientedEdge(curved, True), OrientedEdge(straight, True)), frame, a, b


@pytest.mark.parametrize('reverse', (False, True))
def test_point_on_a_supporting_line_beyond_the_segment_is_outside(reverse):
    model, loop, frame, a, b = mixed_half_circle()
    # The point has segment parameter r=2 exactly, not r in [0,1].
    point = tuple(2*y-x for x, y in zip(a, b))
    centre = frame.uv((Q(0), Q(0), Q(1, 2)))
    assert sum((x-y)**2 for x, y in zip(point, centre)) > 1
    if reverse:
        loop = tuple(OrientedEdge(use.edge, not use.forward) for use in reversed(loop))
    assert _contains(model, loop, point, frame, lambda: None) is False


@pytest.mark.parametrize('fraction', (Q(0), Q(1, 2), Q(1)))
def test_true_segment_and_vertex_contacts_remain_uncertified(fraction):
    model, loop, frame, a, b = mixed_half_circle()
    point = tuple(x+fraction*(y-x) for x, y in zip(a, b))
    with pytest.raises(GeometryError, match='cycle nesting is not certified'):
        _contains(model, loop, point, frame, lambda: None)
