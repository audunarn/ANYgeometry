"""Exact ray-domain controls for mixed straight/curved cell nesting."""
from fractions import Fraction as Q

import pytest

from anygeometry.plane_section_plate import _UndecidedRay, _ray_segment_crossing


@pytest.mark.parametrize('slope', [Q(0), Q(1, 3), Q(-2, 5), Q(7, 11)])
def test_origin_on_segment_line_extension_is_not_a_boundary_contact(slope):
    # All four rays meet the supporting line at their origin, but that point
    # is outside the real segment; no ray choice should be discarded.
    assert not _ray_segment_crossing((Q(0), Q(0)), (Q(1), slope),
                                     (Q(0), Q(1)), (Q(0), Q(2)))


@pytest.mark.parametrize('a,b,expected', [
    ((-1, 0), (-1, 1), False),  # a segment endpoint, behind the ray
    ((1, 1), (1, 2), False),   # forward supporting-line hit outside segment
    ((1, -1), (1, 1), True),
    ((1, 1), (1, -1), True),   # segment orientation has no effect on parity
    ((0, 1), (1, 1), False),   # parallel, noncollinear
])
def test_nonparallel_domains_and_ordinary_parity_are_exact(a, b, expected):
    assert _ray_segment_crossing((Q(0), Q(0)), (Q(1), Q(0)),
        tuple(map(Q, a)), tuple(map(Q, b))) is expected


@pytest.mark.parametrize('a,b', [((1, 0), (1, 1)), ((1, -1), (1, 0)),
                                ((0, -1), (0, 1)), ((1, 0), (2, 0))])
def test_forward_vertex_origin_and_collinear_contacts_still_refuse(a, b):
    with pytest.raises(_UndecidedRay):
        _ray_segment_crossing((Q(0), Q(0)), (Q(1), Q(0)), tuple(map(Q, a)), tuple(map(Q, b)))


def test_a_strictly_positive_rational_crossing_is_not_rounded_to_origin_contact():
    tiny = Q(1, 2**100)
    assert _ray_segment_crossing((Q(0), Q(0)), (Q(1), Q(0)), (tiny, Q(-1)), (tiny, Q(1)))


def _mixed_rectangle():
    from anygeometry import GeometryModel, Plane
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.support_curves import PlaneNurbsCurve
    from anygeometry.entities import OrientedEdge
    from anygeometry.planar_exact import ExactPlane
    surface = NurbsSurface(1, 1, (0, 0, 1, 1), (0, 0, 1, 1),
        (((0, 0, 0), (0, 2, 0)), ((1, 0, 0), (1, 2, 0))))
    curve = PlaneNurbsCurve.from_plane(Plane((0, 1, 0), (1, 0, 0), (0, 0, 1)), surface)
    if curve.evaluate(0)[0] > curve.evaluate(1)[0]:
        curve = curve.subcurve(1, 0)
    model = GeometryModel()
    vertices = model.add_points(((0, 1, 0), (1, 1, 0), (1, 2, 0), (0, 2, 0)))
    curved = model.add_curve(vertices[0], vertices[1], curve)
    edges = (curved, *(model.add_line(vertices[i], vertices[(i+1)%4]) for i in (1, 2, 3)))
    frame = ExactPlane.from_plane(Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)))
    return model, tuple(OrientedEdge(edge, True) for edge in edges), frame


@pytest.mark.parametrize('point,expected', [((Q(0), Q(3)), False), ((Q(1, 2), Q(3, 2)), True)])
def test_actual_mixed_loop_parity_handles_extensions_and_material(point, expected):
    from anygeometry.plane_section_plate import _contains
    from anygeometry import to_dict
    model, loop, frame = _mixed_rectangle()
    before = to_dict(model)
    assert _contains(model, loop, point, frame, lambda: None) is expected
    assert to_dict(model) == before


def test_actual_mixed_loop_boundary_and_cancellation_are_not_admitted():
    from anygeometry import GeometryError
    from anygeometry.plane_section_plate import _contains
    model, loop, frame = _mixed_rectangle()
    with pytest.raises(GeometryError, match='nesting is not certified'):
        _contains(model, loop, (Q(0), Q(3, 2)), frame, lambda: None)
    error = GeometryError('cancel cycle containment')
    def cancel():
        raise error
    with pytest.raises(GeometryError) as caught:
        _contains(model, loop, (Q(1, 2), Q(3, 2)), frame, cancel)
    assert caught.value is error
