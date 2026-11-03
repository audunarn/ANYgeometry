"""Passing weighted trims retain their original section representative bits."""
from fractions import Fraction as Q

import pytest

from anygeometry import GeometryModel, Plane, apply_plane_split, plan_plane_split, to_dict
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface


def _ordinary_weighted_rectangle(reverse):
    # This event's old midpoint is different from the exact trim station, but
    # its existing joint proof succeeds. The repair must preserve those bits.
    shift = 3*2.**-34
    bounds = (-1.5-4*shift, 2.5-4*shift)
    support = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        tuple(tuple((x, y, 0.) for y in bounds) for x in bounds))
    uv = tuple((u+shift, v+shift) for u, v in ((.375, .375), (.625, .375), (.625, .625), (.375, .625)))
    model = GeometryModel()
    vertices = model.add_points(((0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)))
    edges = []
    for index, first in enumerate(uv):
        last = uv[(index+1)%4]
        controls = tuple(tuple(a+factor*(b-a) for a, b in zip(first, last)) for factor in (-1.25, 1.25))
        weights, domain = (1., 3.), (3., 5.)
        if reverse:
            controls, weights, domain = controls[::-1], weights[::-1], domain[::-1]
        pcurve = NurbsCurve(1, (2., 2., 6., 6.), controls, weights, domain)
        edges.append(model.add_curve(vertices[index], vertices[(index+1)%4], CurveOnSurface(support, pcurve)))
    face = model.add_face(edges, surface=support)
    return model, face, Q(shift)


@pytest.mark.parametrize('reverse', (False, True))
def test_ordinary_weighted_trim_retains_original_trace_station_bits(reverse):
    model, face, shift = _ordinary_weighted_rectangle(reverse)
    before = to_dict(model)
    plan = plan_plane_split(model, [face], Plane((.6, 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    assert to_dict(model) == before
    application = apply_plane_split(model, plan, policy='imprint')
    assert len(application.trace_edges) == 1
    trace = model.edges[application.trace_edges[0]].curve
    original = (Q(3, 8)+Q(1, 2**32), Q(5, 8)+Q(1, 2**32))
    canonical = (Q(3, 8)+shift, Q(5, 8)+shift)
    assert canonical != original
    assert tuple(map(Q, trace.domain)) == original
    assert model.validate_topology() == ()


def test_small_valid_affine_rectangle_is_excluded_from_station_repair():
    from anygeometry.nurbs_plane_arrangement import _affine_parent_joint_bound
    model = GeometryModel()
    support = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
        tuple(tuple((x, y, 0.) for y in (0., .125)) for x in (0., .125)))
    uv = ((0., 0.), (1., 0.), (1., 1.), (0., 1.))
    vertices = model.add_points(((0., 0., 0.), (.125, 0., 0.), (.125, .125, 0.), (0., .125, 0.)))
    edges = [model.add_curve(vertices[i], vertices[(i+1)%4], CurveOnSurface(support,
        NurbsCurve(1, (0., 0., 1., 1.), (first, uv[(i+1)%4])))) for i, first in enumerate(uv)]
    face = model.add_face(edges, surface=support)
    assert model.validate_topology() == ()
    assert _affine_parent_joint_bound(model, model.faces[face], support, model.edges) is None
