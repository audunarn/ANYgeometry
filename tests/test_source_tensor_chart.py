"""An owner tensor view derives each directed rim from stored definitions."""
from dataclasses import replace
from fractions import Fraction as Q

import pytest

from anygeometry import GeometryError, GeometryModel, source_tensor_chart, to_dict
from anygeometry.entities import OrientedEdge
from anygeometry.nurbs import NurbsCurve
from anygeometry.surfaces import CoonsSurface, RuledSurface


def sampled():
    model = GeometryModel()
    lo = ((0.,0.,0.),(.5,0.,.25),(1.,0.,0.))
    hi = ((0.,1.,0.),(.5,1.,.5),(1.,1.,.25))
    support = RuledSurface(lo, hi)
    a,b,c,d = model.add_points((lo[0],lo[-1],hi[-1],hi[0]))
    knots = (0.,0.,.5,1.,1.)
    bottom = model.add_curve(b,a,NurbsCurve(1,knots,lo[::-1]))
    right = model.add_line(b,c)
    top = model.add_curve(d,c,NurbsCurve(1,knots,hi))
    left = model.add_line(d,a)
    face = model.add_face_from_loop((OrientedEdge(bottom, False), OrientedEdge(right, True),
                                    OrientedEdge(top, False), OrientedEdge(left, True)),
                                   corners=(0,1,2,3),surface=support)
    return model,face


def test_sampled_source_tensor_preserves_kink_and_edge_reversal():
    g,f = sampled()
    before = to_dict(g)
    view = source_tensor_chart(g,f)
    t = Q(3,8)
    support = view.support
    assert support.knots_u == (0.,0.,.5,1.,1.)
    assert support.exact_point(t,Q(1,4)) == (t,Q(1,4),Q(15,64))
    bottom = g.faces[f].loop[0].edge
    assert view.edges[bottom].curve.exact_point(t) == g.edges[bottom].curve.exact_point(t)
    assert to_dict(g) == before
    g.add_point(4.,0.,0.)
    with pytest.raises(GeometryError,match="stale"):
        view.face_point(f, .3, .4)


def test_changed_sampled_midpoint_fails_the_whole_edge_identity():
    g,f = sampled()
    face = g.faces[f]
    lower = face.surface.first_boundary.copy()
    lower[1,2] += .03125
    g._put_entity('face', replace(face,surface=RuledSurface(lower,face.surface.second_boundary)))
    with pytest.raises(GeometryError,match='directed authored edge'):
        source_tensor_chart(g,f)


def test_source_tensor_rejects_hidden_evaluator():
    g,f = sampled()
    object.__setattr__(g.faces[f].surface, 'evaluate', lambda u,v: (0.,0.,0.))
    with pytest.raises(GeometryError,match='canonical authored evaluators'):
        source_tensor_chart(g,f)


def test_bezier_rim_equality_requires_all_directed_bernstein_coefficients():
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.source_tensor_chart import _matches_edge, _same_bezier_boundary
    profile = ((0.,0.,0.),(.5,0.,.125),(1.,0.,0.))
    top = tuple((x,1.,z) for x,_,z in profile)
    ku, kv = (0.,0.,0.,1.,1.,1.), (0.,0.,1.,1.)
    support = NurbsSurface(2,1,ku,kv,tuple(zip(profile,top)))
    curve = NurbsCurve(2,ku,top[::-1])
    assert _same_bezier_boundary(support, 2, curve, True)
    assert not _same_bezier_boundary(support, 2, curve, False)
    changed = NurbsCurve(2,ku,(top[-1], (.5,1.,.1), top[0]))
    assert not _same_bezier_boundary(support, 2, changed, True)
    with pytest.raises(GeometryError, match="directed authored edge"):
        _matches_edge(support, 2, changed, True)
