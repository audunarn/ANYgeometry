"""Exact restriction identities and invocation-local endpoint-cache guards.

Definitions only: root owns numerical execution and its ledger.
"""
from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from anygeometry.nurbs_surface_intersection import (query_nurbs_surface_intersection,evaluate_arcs,
    IntersectionStatus,NurbsSurfaceIntersectionIntegrityError,NurbsSurfaceIntersectionCancelled)
from anygeometry.nurbs_surface_intersection._bernstein import _blossom_axis,_blossom_axis_reference
from anygeometry.nurbs_surface_intersection._filtered import Bound,_axis,_lerp
from anygeometry.tolerance import TolerancePolicy
from test_generic import seam_planes


@pytest.mark.parametrize("degree",range(7))
def test_fast_exact_axis_equals_original_blossom_coefficientwise(degree):
    vectors=[tuple(Q((i+2)*(k+1)*(-1)**(i+k),i+k+2) for k in range(4)) for i in range(degree+1)]
    intervals=((Q(0),Q(1)),(Q(0),Q(1,3)),(Q(1,4),Q(1)),(Q(1,7),Q(6,7)),
               (Q(1,3),Q(1,3)),(Q(0),Q(0)),(Q(1),Q(1)),(Q(-1,3),Q(-1,3)),
               (Q(-1,5),Q(4,5)),(Q(1,5),Q(6,5)),(Q(3,4),Q(1,4)),(Q(1),Q(2)))
    for a,b in intervals:
        assert _blossom_axis(vectors,a,b)==_blossom_axis_reference(vectors,a,b)


def _old_float_axis(vectors,a,b):
    n=len(vectors)-1
    out=[]
    for k in range(n+1):
        work=list(vectors)
        for tau in [a]*(n-k)+[b]*k:
            work=[_lerp(work[i],work[i+1],tau) for i in range(len(work)-1)]
        out.append(work[0])
    return out


@pytest.mark.parametrize("degree",range(7))
def test_float_identical_point_shortcut_is_bit_identical_and_outward(degree):
    exact=[tuple(Q((-1)**(i+k)*(i+k+1),i+2) for k in range(4)) for i in range(degree+1)]
    vectors=[tuple(Bound.exact(x) for x in p) for p in exact]
    for tau in (Q(0),Q(1,3),Q(1),Q(-1,4)):
        a=Bound.exact(tau)
        b=Bound(a.lo,a.hi)
        fast,old=_axis(vectors,a,b),_old_float_axis(vectors,a,b)
        bits=lambda values:tuple(tuple((x.lo.hex(),x.hi.hex()) for x in p) for p in values)
        assert bits(fast)==bits(old)
        point=_blossom_axis_reference(exact,tau,tau)[0]
        assert all(Q(x.lo)<=value<=Q(x.hi) for x,value in zip(fast[0],point))


@pytest.mark.parametrize("degree",range(7))
def test_directed_two_split_axis_encloses_exact_blossom_coefficients(degree):
    exact=[tuple(Q((-1)**(i+k)*(i+2)*(k+1),i+k+3) for k in range(4)) for i in range(degree+1)]
    vectors=[tuple(Bound.exact(x) for x in p) for p in exact]
    intervals=((Q(0),Q(1)),(Q(0),Q(2,3)),(Q(1,5),Q(1)),(Q(1,7),Q(6,7)),
               (Q(1,3),Q(1,3)),(Q(3,4),Q(1,4)),(Q(-1,4),Q(5,4)),
               (Q(1)-Q(1,2**100),Q(1)),(Q(1,3),Q(1,3)+Q(1,2**100)))
    for a,b in intervals:
        bounds=_axis(vectors,Bound.exact(a),Bound.exact(b))
        reference=_blossom_axis_reference(exact,a,b)
        assert len(bounds)==len(reference)
        assert all(Q(x.lo)<=value<=Q(x.hi)
                   for p,q in zip(bounds,reference) for x,value in zip(p,q))


def test_factored_value_only_enclosure_matches_full_exact_enclosure():
    from anygeometry.nurbs_surface_intersection._certificate import _Pair
    from test_generic import rational_circle
    a,b=rational_circle()
    pair=_Pair(a,b,((Q(0),Q(1)),)*4)
    for cell in (((Q(1,7),Q(6,7)),)*4,((Q(1,3),Q(1,3)),)*4):
        anchored=pair.anchored(cell)
        assert anchored.enclose_values(cell)==anchored.enclose(cell)[:2]


def test_sign_monotonic_products_enclose_all_exact_corner_products():
    intervals=((-3,-1),(-3,0),(-3,2),(0,0),(0,2),(1,1),(2,3),(-1e-300,1e-300))
    for a,b in intervals:
        for c,d in intervals:
            left,right=Bound(float(a),float(b)),Bound(float(c),float(d))
            product=left*right
            corners=[Q(x)*Q(y) for x in (left.lo,left.hi) for y in (right.lo,right.hi)]
            assert Q(product.lo)<=min(corners)<=max(corners)<=Q(product.hi)


def test_prepared_float_enclosure_reuse_is_bounded_and_matches_exact_ranges():
    from anygeometry.nurbs_surface_intersection._certificate import _Pair
    from anygeometry.nurbs_surface_intersection._filtered import Pair
    from test_generic import rational_circle
    a,b=rational_circle()
    cell=((Q(1,4),Q(3,4)),)*4
    exact_pair=_Pair(a,b,((Q(0),Q(1)),)*4).anchored(cell)
    prepared=Pair(exact_pair)
    first=prepared.enclose(cell)
    assert prepared.enclose(cell) is first and prepared.cache_hits==1
    def flat(enc):
        return (*enc[0],*enc[1],*enc[2][0],*enc[2][1],*enc[3][0],*enc[3][1])
    assert all(Q(x.lo)<=y.lo<=y.hi<=Q(x.hi) for x,y in zip(flat(first),flat(exact_pair.enclose(cell))))
    for i in range(7):
        box=((Q(i,16),Q(i+2,16)),)+cell[1:]
        prepared.enclose(box)
        assert len(prepared.enclosure_cache)<=4
    again=prepared.enclose(cell)
    assert tuple((x.lo,x.hi) for x in flat(again))==tuple((x.lo,x.hi) for x in flat(first))


def test_nonfinite_filtered_arithmetic_escalates_to_exact_generic_proof():
    from anygeometry.nurbs import NurbsSurface
    from anygeometry.nurbs_surface_intersection._filtered import Unknown
    with pytest.raises(Unknown):
        Bound(1e308)*Bound(1e308)
    a,b=seam_planes()
    def weighted(surface):
        weights=tuple(tuple(1e300 for _ in row) for row in surface.controls)
        return NurbsSurface(surface.degree_u,surface.degree_v,surface.knots_u,surface.knots_v,
                            surface.controls,weights,surface.u_range,surface.v_range)
    a,b=weighted(a),weighted(b)
    results=[]
    for mode in ("filtered","exact"):
        limits=SimpleNamespace(max_boxes_per_pair=256,max_subdivision_depth=24,interval_mode=mode)
        results.append(query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy(),qualification=limits))
    filtered,exact=results
    assert filtered.status is exact.status is IntersectionStatus.CERTIFIED
    assert len(filtered.components)==len(exact.components)==1
    assert not filtered.components[0].closed and not exact.components[0].closed
    assert filtered.stats.float_attempts>0 and filtered.stats.exact_escalations>0


def test_batch_shared_endpoint_reuses_box_and_rewraps_request(monkeypatch):
    from anygeometry.nurbs_surface_intersection._atlas import _Surface
    a,b=seam_planes()
    limits=SimpleNamespace(max_boxes_per_pair=256,max_subdivision_depth=24)
    r=query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy(),qualification=limits)
    assert r.status is IntersectionStatus.CERTIFIED
    assert len(r.components[0].arcs)>=2
    assert r.components[0].arcs[0].second_root==r.components[0].arcs[1].first_root
    count=[0]
    original=_Surface.hull
    def counted(self,box):
        count[0]+=1
        return original(self,box)
    monkeypatch.setattr(_Surface,"hull",counted)
    def requests():
        yield 0,0,Q(1)
        after_first=count[0]
        yield 0,1,Q(0)
        assert count[0]==after_first  # second endpoint performs no new hull work
    first,second=evaluate_arcs(r,requests(),first=a,second=b,tolerances=TolerancePolicy(),qualification=limits)
    assert first.arc_index==0 and first.station==1
    assert second.arc_index==1 and second.station==0
    assert (first.native_box,first.world_lo,first.world_hi)==(second.native_box,second.world_lo,second.world_hi)


def test_batch_cached_endpoint_still_checks_live_source_at_next_request():
    a,b=seam_planes()
    limits=SimpleNamespace(max_boxes_per_pair=256,max_subdivision_depth=24)
    r=query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy(),qualification=limits)
    assert r.status is IntersectionStatus.CERTIFIED and len(r.components[0].arcs)>=2
    def requests():
        yield 0,0,Q(1)
        controls=list(a.controls)
        row=list(controls[0])
        x,y,z=row[0]
        row[0]=(x,y,z+1.)
        controls[0]=tuple(row)
        object.__setattr__(a,"controls",tuple(controls))
        yield 0,1,Q(0)
    with pytest.raises(NurbsSurfaceIntersectionIntegrityError):
        evaluate_arcs(r,requests(),first=a,second=b,tolerances=TolerancePolicy(),qualification=limits)


def test_batch_cached_shared_endpoint_still_polls_cancellation():
    a,b=seam_planes()
    limits=SimpleNamespace(max_boxes_per_pair=256,max_subdivision_depth=24)
    r=query_nurbs_surface_intersection(a,b,tolerances=TolerancePolicy(),qualification=limits)
    assert r.status is IntersectionStatus.CERTIFIED and len(r.components[0].arcs)>=2
    assert r.components[0].arcs[0].second_root==r.components[0].arcs[1].first_root
    armed=[False]
    def requests():
        yield 0,0,Q(1)
        # Canonical validation and the first endpoint already completed. The
        # next request targets that same cached root on its adjacent arc.
        armed[0]=True
        yield 0,1,Q(0)
    with pytest.raises(NurbsSurfaceIntersectionCancelled):
        evaluate_arcs(r,requests(),first=a,second=b,tolerances=TolerancePolicy(),qualification=limits,
                      cancellation_check=lambda:armed[0])
