"""Exact structural knot anchors, never sampled/snapped pcurve roots."""
from fractions import Fraction as Q
import pytest
from anygeometry.nurbs import NurbsCurve,NurbsSurface,CurveOnSurface
from anygeometry.predicates import IntersectionQualificationPolicy
from anygeometry._local_curve_geometry import bound_curve_geometry
from anygeometry._local_curvature import LocalApproximationNotCertified

K=.313758476624643


def support():
    return NurbsSurface(3,1,(0.,0.,0.,0.,K,1.,1.,1.,1.),(0.,0.,1.,1.),
        tuple(((x,0.,0.),(x,1.,0.)) for x in (0.,.1,.4,.7,1.)))


def coverage(result):
    intervals=sorted(tuple(sorted(piece.parameter_range)) for piece in result.pieces)
    reach=min(result.parameter_range)
    for lo,hi in intervals:
        assert lo==reach and hi>lo
        reach=hi
    assert reach==max(result.parameter_range)
    assert all(piece.tangent_norm_squared_lower>0 for piece in result.pieces)


def test_affine_cos_crossing_has_exact_support_knot_anchor():
    curve=CurveOnSurface(support(),NurbsCurve(1,(0.,0.,1.,1.),((0.,.5),(1.,.5))))
    result=bound_curve_geometry(curve)
    assert Q(K) in result.anchors
    coverage(result)
    assert result.boxes>=2
    assert all(piece.curvature_upper==0 for piece in result.pieces)


@pytest.mark.parametrize('domain',[(2.5,3.5),(3.5,2.5)])
@pytest.mark.parametrize('reverse',[False,True])
def test_trimmed_nonunit_pcurve_domain_and_oriented_interval(domain,reverse):
    pc=NurbsCurve(1,(2.,2.,4.,4.),((0.,.5),(1.,.5)),(7.,7.),domain=domain)
    curve=CurveOnSurface(support(),pc)
    result=bound_curve_geometry(curve,1 if reverse else 0,0 if reverse else 1)
    alpha,beta=map(Q,domain)
    a=(alpha-2)/2;d=(beta-alpha)/2;root=(Q(K)-a)/d
    assert root in result.anchors and Q(0)<root<Q(1)
    assert result.anchors==tuple(sorted(result.anchors,reverse=reverse))
    coverage(result)


@pytest.mark.parametrize('u',[K,.5])
def test_constant_coordinate_on_or_off_knot_retains_one_sided_bounds(u):
    curve=CurveOnSurface(support(),NurbsCurve(1,(0.,0.,1.,1.),((u,0.),(u,1.))))
    result=bound_curve_geometry(curve)
    coverage(result)
    assert result.anchors==(Q(0),Q(1)) and all(piece.curvature_upper==0 for piece in result.pieces)


@pytest.mark.parametrize('kind',['nonlinear','unequal_weights'])
def test_nonaffine_pcurves_do_not_gain_an_affine_claim(kind):
    from anygeometry._local_curve_geometry import _affine_uv_map
    pc=(NurbsCurve(2,(0.,0.,0.,1.,1.,1.),((0.,.5),(.4,.5),(1.,.5))) if kind=='nonlinear'
        else NurbsCurve(1,(0.,0.,1.,1.),((0.,.5),(1.,.5)),(1.,2.)))
    assert _affine_uv_map(pc) is None
    with pytest.raises(LocalApproximationNotCertified):
        bound_curve_geometry(CurveOnSurface(support(),pc),policy=IntersectionQualificationPolicy(max_subdivision_depth=4))


def test_added_pieces_obey_original_box_cap():
    curve=CurveOnSurface(support(),NurbsCurve(1,(0.,0.,1.,1.),((0.,.5),(1.,.5))))
    with pytest.raises(LocalApproximationNotCertified,match='box budget'):
        bound_curve_geometry(curve,policy=IntersectionQualificationPolicy(max_boxes_per_pair=1))
