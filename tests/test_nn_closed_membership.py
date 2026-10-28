"""Analytic sign/clearance proofs and independent winding fallback parity."""
from fractions import Fraction as Q
import pytest
from anygeometry import GeometryError
from anygeometry.nurbs import NurbsSurface
from anygeometry.tolerance import TolerancePolicy
from anygeometry.nurbs_surface_intersection import query_nurbs_surface_intersection,NurbsIntersectionCurve,IntersectionStatus
from anygeometry.nurbs_intersection_curve import NurbsSurfaceIntersectionCurve,intersection_curve_operation
from anygeometry.intersection_chart import _closed_sign_contains,_contains_owner_loop_winding,contains_owner_loop


def loop(rational=False,reverse=False):
    knots=(0.,0.,0.,1.,1.,1.); square=(1.,-1.,1.); weights=(1.,2.,1.)
    surfaces=[]
    for second in (False,True):
        controls=tuple(tuple((float(i-1),float(j-1),
            (square[i]+square[j]-.25)/weights[i] if second else 0.) for j in range(3)) for i in range(3))
        if not rational:
            controls=tuple(tuple((float(i-1),float(j-1),
                square[i]+square[j]-.25 if second else 0.) for j in range(3)) for i in range(3))
        selected=(1.,0.) if reverse else (0.,1.)
        surfaces.append(NurbsSurface(2,2,knots,knots,controls,
            tuple((w,w,w) for w in weights) if rational else None,selected,selected))
    result=query_nurbs_surface_intersection(*surfaces,tolerances=TolerancePolicy())
    assert result.status is IntersectionStatus.CERTIFIED,result.reason
    curve=NurbsSurfaceIntersectionCurve(NurbsIntersectionCurve.from_result(result))
    return tuple(curve.subcurve(Q(i,3),Q(i+1,3)) for i in range(3)),surfaces,result


@pytest.mark.parametrize('rational',(False,True))
@pytest.mark.parametrize('reverse',(False,True))
def test_analytic_circle_sign_clearance_winding_rotations_and_orientation(rational,reverse):
    curves,surfaces,_=loop(rational,reverse)
    with intersection_curve_operation():
        for support in surfaces:
            for chain in (curves,curves[1:]+curves[:1],tuple(c.subcurve(1,0) for c in reversed(curves))):
                for point,answer in (((Q(1,2),Q(1,2)),True),((Q(7,8),Q(7,8)),False)):
                    assert _closed_sign_contains(chain,support,point,Q(1,100)) is answer
                    assert contains_owner_loop(chain,support,point,Q(1,100)) is answer
                    assert _contains_owner_loop_winding(chain,support,point,Q(1,100)) is answer


def test_clearance_contact_outside_domain_and_bad_partition_retain_winding(monkeypatch):
    import anygeometry.intersection_chart as chart
    curves,surfaces,_=loop()
    support=surfaces[0]
    with intersection_curve_operation():
        assert _closed_sign_contains(curves,support,(Q(3,4),Q(1,2)),Q(1,100)) is None
        assert _closed_sign_contains(curves,support,(Q(0),Q(0)),Q(1,100)) is None
        assert _closed_sign_contains(curves[:2],support,(Q(1,2),Q(1,2)),Q(0)) is None
        assert _closed_sign_contains((curves[1],curves[0],curves[2]),support,(Q(1,2),Q(1,2)),Q(0)) is None
        calls=[]
        def fallback(*args,**kwargs):
            calls.append(1)
            return 'original-winding-result'
        monkeypatch.setattr(chart,'_contains_owner_loop_winding',fallback)
        assert contains_owner_loop(curves,support,(Q(3,4),Q(1,2)),Q(1,100))=='original-winding-result'
        assert calls==[1]


def test_no_caller_result_or_generic_cache_can_issue_sign_authority():
    from anygeometry.nurbs_intersection_curve import operation_fact
    curves,surfaces,_=loop()
    with intersection_curve_operation():
        operation_fact(lambda:('closed-sign',curves),lambda:False)
        assert _closed_sign_contains(curves,surfaces[0],(Q(1,2),Q(1,2)),Q(0)) is True
        class Impostor:
            owner=curves[0].owner
        assert _closed_sign_contains((Impostor(),),surfaces[0],(Q(1,2),Q(1,2)),Q(0)) is None


def test_warmed_membership_rejects_forged_context_and_raw_knot_changes():
    curves,surfaces,_=loop()
    with intersection_curve_operation():
        assert _closed_sign_contains(curves,surfaces[0],(Q(1,2),Q(1,2)),Q(0)) is True
        shape=curves[0]._shape()
        original=shape._context
        class Forged:
            def _closed_sign_membership(self,*args):
                return False
        shape._context=Forged()
        try:
            with pytest.raises(GeometryError):
                _closed_sign_contains(curves,surfaces[0],(Q(1,2),Q(1,2)),Q(0))
        finally:
            shape._context=original
        knots=surfaces[0].knots_u
        object.__setattr__(surfaces[0],'knots_u',(0.,0.,0.,.9,.9,.9))
        try:
            with pytest.raises(GeometryError):
                _closed_sign_contains(curves,surfaces[0],(Q(1,2),Q(1,2)),Q(0))
        finally:
            object.__setattr__(surfaces[0],'knots_u',knots)


def test_warmed_membership_keeps_external_source_and_callback_integrity():
    curves,surfaces,_=loop(); support=surfaces[0]
    state={'mutate':False}
    original=support.controls
    def check():
        if state['mutate']:
            p=original[0][0]
            object.__setattr__(support,'controls',(((p[0],p[1],p[2]+.125),*original[0][1:]),*original[1:]))
        return False
    with intersection_curve_operation(check):
        assert _closed_sign_contains(curves,support,(Q(1,2),Q(1,2)),Q(0)) is True
        state['mutate']=True
        try:
            with pytest.raises(GeometryError):
                _closed_sign_contains(curves,support,(Q(1,2),Q(1,2)),Q(0))
        finally:
            object.__setattr__(support,'controls',original)
            state['mutate']=False
    failure=RuntimeError('membership callback sentinel')
    def fail():
        raise failure
    with intersection_curve_operation(fail):
        with pytest.raises(RuntimeError) as caught:
            _closed_sign_contains(curves,support,(Q(1,2),Q(1,2)),Q(0))
        assert caught.value is failure


def test_sign_hull_charges_resource_scope_and_does_not_swallow_exhaustion(monkeypatch):
    import anygeometry.plane_nurbs as work
    curves,surfaces,_=loop()
    with intersection_curve_operation():
        assert _closed_sign_contains(curves,surfaces[0],(Q(1,2),Q(1,2)),Q(0)) is True
        failure=GeometryError('membership box budget exhausted')
        def exhausted():
            raise failure
        monkeypatch.setattr(work,'charge_box',exhausted)
        with pytest.raises(GeometryError) as caught:
            contains_owner_loop(curves,surfaces[0],(Q(1,2),Q(1,2)),Q(0))
        assert caught.value is failure


def test_conservative_applicability_units_reject_c0_ordered_range_multiple_and_incomplete():
    """Guard-unit negatives: modified rows are NOT canonical proof receipts."""
    from dataclasses import replace
    from anygeometry.nurbs import _exact_insert
    from anygeometry.nurbs_surface_intersection import SourceDefinition
    from anygeometry.nurbs_surface_intersection._closed_membership import ClosedSignMembership
    from anygeometry.nurbs_surface_intersection._structural import derive_structural
    _,surfaces,result=loop()
    def forbidden_hull(*args):
        raise AssertionError('unsupported guard reached position certification')
    def rejects(row):
        proof=ClosedSignMembership(row,0,forbidden_hull,lambda:None)
        assert not proof.applicable
        assert proof.contains((Q(1,2),Q(1,2)),Q(0)) is None
    rejects(replace(result,cover_complete=False))
    rejects(replace(result,components=result.components*2))
    source=surfaces[0]
    values=list(SourceDefinition.capture(source).values)
    values[6]=tuple(reversed(values[6]))
    reversed_source=NurbsSurface(*values)
    rejects(replace(result,first_source=SourceDefinition.capture(reversed_source)))
    c0=[]
    for support in surfaces:
        flattened=[tuple(Q(x) for point in row for x in point) for row in support.controls]
        knots,rows=_exact_insert(2,list(map(Q,support.knots_u)),flattened,Q(1,2),2)
        controls=tuple(tuple(tuple(float(x) for x in row[3*j:3*j+3]) for j in range(3)) for row in rows)
        c0.append(NurbsSurface(2,2,tuple(map(float,knots)),support.knots_v,controls))
    definitions=tuple(SourceDefinition.capture(s) for s in c0)
    certificate,reason=derive_structural(*definitions)
    assert certificate is not None,reason
    rejects(replace(result,first_source=definitions[0],second_source=definitions[1],structural=certificate))


def test_nonregular_closed_zero_guard_does_not_treat_jordan_shape_as_sign_flip():
    """Guard unit explicitly rejects zero derivative, independently of status flags."""
    from dataclasses import replace
    from anygeometry.nurbs_surface_intersection._closed_membership import ClosedSignMembership
    _,_,result=loop()
    boxes=tuple(replace(box,coefficients=tuple(tuple(Q(0) for x in row) for row in box.coefficients))
                for box in result.boxes)
    def no_hull(*args):
        raise AssertionError('singular scalar regularity reached hull proof')
    assert not ClosedSignMembership(replace(result,boxes=boxes),0,no_hull,lambda:None).applicable


def test_real_squared_circle_tangent_query_does_not_issue_regular_curve():
    from math import comb
    from types import SimpleNamespace
    # F=9(x²+y²-1/4)²: exactly one Jordan zero set but NO sign flip.
    # Factor9 makes every degree4 Bernstein control exactly binary64.
    power={(2,0):Q(4),(1,0):Q(-4),(0,2):Q(4),(0,1):Q(-4),(0,0):Q(7,4)}
    squared={}
    for (a,b),c in power.items():
        for (d,e),f in power.items():
            squared[a+d,b+e]=squared.get((a+d,b+e),Q(0))+c*f
    controls=[]
    for i in range(5):
        row=[]
        for j in range(5):
            z=sum((c*Q(comb(i,a),comb(4,a))*Q(comb(j,b),comb(4,b))
                   for (a,b),c in squared.items() if a<=i and b<=j),Q(0))
            z*=9
            assert z.denominator & (z.denominator-1) == 0
            row.append((float(Q(i,2)-1),float(Q(j,2)-1),float(z)))
        controls.append(tuple(row))
    knots=(0.,)*5+(1.,)*5
    second=NurbsSurface(4,4,knots,knots,tuple(controls))
    assert second.exact_point(Q(3,4),Q(1,2))[2] == 0
    assert second.exact_point(Q(1,2),Q(1,2))[2] == Q(9,16)
    first=NurbsSurface(4,4,knots,knots,tuple(tuple((x,y,0.) for x,y,z in row) for row in controls))
    result=query_nurbs_surface_intersection(first,second,tolerances=TolerancePolicy(),
        qualification=SimpleNamespace(max_boxes_per_pair=128,max_subdivision_depth=8))
    assert result.status is not IntersectionStatus.CERTIFIED
