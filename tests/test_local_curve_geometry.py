"""Real exact curves; sampled checks below are tests, never proof producers."""
from dataclasses import replace
from fractions import Fraction as Q
import pytest
from anygeometry.arrangement_geometry import LinePath
from anygeometry.nurbs import NurbsCurve,NurbsSurface,CurveOnSurface
from anygeometry.support_curves import PlaneNurbsCurve
from anygeometry.predicates import IntersectionQualificationPolicy
from anygeometry.errors import GeometryError
from anygeometry._local_curve_geometry import bound_curve_geometry


def coverage(result):
    intervals=sorted(tuple(sorted(p.parameter_range)) for p in result.pieces)
    a,b=sorted(result.parameter_range)
    reach=a
    for lo,hi in intervals:
        assert lo==reach and hi>lo
        reach=hi
    assert reach==b
    assert all(p.tangent_norm_squared_lower>0 for p in result.pieces)


def flat():
    return NurbsSurface(1,1,(0.,0.,1.,1.),(0.,0.,1.,1.),
        (((0.,0.,0.),(0.,1.,0.)),((1.,0.,0.),(1.,1.,0.))))


@pytest.mark.parametrize('reverse',(False,True))
def test_line_exact_bounds_and_oriented_tangent(reverse):
    curve=LinePath((0.,0.,0.),(2.,0.,0.))
    result=bound_curve_geometry(curve,1 if reverse else 0,0 if reverse else 1)
    coverage(result)
    piece=result.pieces[0]
    assert piece.world_box==((0,2),(0,0),(0,0))
    assert piece.tangent[0]==((-2,-2) if reverse else (2,2))
    assert piece.curvature_upper==0 and piece.radius_lower is None


@pytest.mark.parametrize('cos',(False,True))
def test_parabola_exact_curvature_bound(cos):
    controls=((0.,0.),(.5,0.),(1.,1.))
    pcurve=NurbsCurve(2,(0.,0.,0.,1.,1.,1.),controls if cos else tuple((*p,0.) for p in controls))
    curve=CurveOnSurface(flat(),pcurve) if cos else pcurve
    result=bound_curve_geometry(curve)
    coverage(result)
    assert result.pieces[0].curvature_upper>=2  # exact curvature at t=0
    assert all(p.parameter_convention=='oriented descriptor coordinate r=sign*t' for p in result.pieces)


def test_rational_cropped_reversed_curve_keeps_entire_parameter_image():
    curve=NurbsCurve(2,(0.,0.,0.,1.,1.,1.),((0.,0.,0.),(.5,1.,0.),(1.,0.,0.)),
        weights=(1.,.5,2.),domain=(.875,.125))
    result=bound_curve_geometry(curve,Q(1,8),Q(7,8));coverage(result)
    for piece in result.pieces:
        for t in (*piece.parameter_range,sum(piece.parameter_range)/2):
            point=curve.exact_point(t)
            assert all(lo<=x<=hi for x,(lo,hi) in zip(point,piece.world_box))


def test_cos_pcurve_knot_anchors_preserve_one_sided_pieces():
    pc=NurbsCurve(1,(0.,0.,.25,1.,1.),((0.,0.),(.5,.5),(1.,.5)))
    result=bound_curve_geometry(CurveOnSurface(flat(),pc));coverage(result)
    assert Q(1,4) in result.anchors
    assert all(not min(p.parameter_range)<Q(1,4)<max(p.parameter_range) for p in result.pieces)
    assert all(p.curvature_upper==0 for p in result.pieces)


@pytest.mark.parametrize('reverse',(False,True))
def test_closed_plane_section_includes_seam_and_endpoint_isolators(reverse):
    from test_plane_split import bowl
    model,face,plane=bowl()
    curve=PlaneNurbsCurve.from_plane(plane,model.faces[face].surface)
    if reverse: curve=replace(curve,domain=(1.,0.))
    result=bound_curve_geometry(curve);coverage(result)
    assert result.anchors[0]==0 and result.anchors[-1]==1
    assert all('NOT descriptor derivative' in p.parameter_convention for p in result.pieces)
    assert all(p.curvature_upper*p.curvature_upper>=2 for p in result.pieces)
    # Flow norm bounds deliberately do not claim descriptor nonzero speed at
    # endpoints, where the native evaluator can use constant isolator clamps.


def test_supported_nn_uses_fresh_owner_whole_branch_bounds():
    from test_prepared_curve_bounds import sources
    from anygeometry.nurbs_surface_intersection import query_nurbs_surface_intersection,NurbsIntersectionCurve,IntersectionStatus
    from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve
    from anygeometry.tolerance import TolerancePolicy
    first,second=sources('scalar')
    query=query_nurbs_surface_intersection(first,second,tolerances=TolerancePolicy())
    assert query.status is IntersectionStatus.CERTIFIED
    curve=NurbsSurfaceIntersectionCurve(NurbsIntersectionCurve.from_result(query),domain=(Q(3,4),Q(1,4)))
    result=bound_curve_geometry(curve);coverage(result)
    assert all(p.parameter_convention=='oriented descriptor coordinate r=sign*t' for p in result.pieces)


def test_degenerate_unknown_and_budget_refuse():
    policy=IntersectionQualificationPolicy(max_boxes_per_pair=4,max_subdivision_depth=1)
    with pytest.raises(GeometryError):
        bound_curve_geometry(LinePath((0.,0.,0.),(0.,0.,0.)),policy=policy)
    with pytest.raises(GeometryError): bound_curve_geometry(object())
    with pytest.raises(GeometryError): bound_curve_geometry(LinePath((0.,0.,0.),(1.,0.,0.)),0,0)


def test_callback_mutation_and_exception_identity():
    curve=LinePath((0.,0.,0.),(1.,0.,0.))
    def mutate(phase):
        object.__setattr__(curve,'end',(2.,0.,0.))
        return False
    with pytest.raises(GeometryError,match='changed'): bound_curve_geometry(curve,cancellation_check=mutate)
    class Stop(BaseException): pass
    def stop(phase): raise Stop()
    with pytest.raises(Stop): bound_curve_geometry(curve,cancellation_check=stop)
    with pytest.raises(GeometryError,match='cancelled'):
        bound_curve_geometry(curve,cancellation_check=lambda phase:True)


def test_native_curve_cache_poison_refuses_before_callback_repair():
    curve=NurbsCurve(1,(0.,0.,1.,1.),((0.,0.,0.),(1.,0.,0.)))
    curve.evaluate(.5)
    original=curve._prepared
    object.__setattr__(curve,'_prepared',())
    called=[]
    def repair(phase):
        called.append(phase);object.__setattr__(curve,'_prepared',original)
        return False
    try:
        with pytest.raises(GeometryError):bound_curve_geometry(curve,cancellation_check=repair)
        assert not called
    finally:object.__setattr__(curve,'_prepared',original)


def test_exact_curve_curvature_scales_in_world_units():
    curve=NurbsCurve(2,(0.,0.,0.,1.,1.,1.),((0.,0.,0.),(.5,0.,0.),(1.,1.,0.)))
    scaled=replace(curve,controls=tuple(tuple(8*x for x in row) for row in curve.controls))
    a,b=bound_curve_geometry(curve),bound_curve_geometry(scaled)
    assert [p.curvature_upper for p in a.pieces]==[8*p.curvature_upper for p in b.pieces]


def test_unresolved_support_knot_crossing_refuses_boundedly():
    support=NurbsSurface(1,1,(0.,0.,.3,1.,1.),(0.,0.,1.,1.),
        (((0.,0.,0.),(0.,1.,0.)),((.3,0.,0.),(.3,1.,.1)),((1.,0.,0.),(1.,1.,.2))))
    pc=NurbsCurve(1,(0.,0.,1.,1.),((0.,.5),(1.,.5)))
    with pytest.raises(GeometryError,match='support-knot'):
        bound_curve_geometry(CurveOnSurface(support,pc),
            policy=IntersectionQualificationPolicy(max_boxes_per_pair=20,max_subdivision_depth=2))
