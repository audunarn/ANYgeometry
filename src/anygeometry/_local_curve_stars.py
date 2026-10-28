"""Finite regular branch stars with explicit nonlocal competitor complements.

This private producer grants no contact or material-side exclusion authority.
"""
from dataclasses import dataclass, replace
from fractions import Fraction as Q
from ._local_curve_geometry import bound_curve_geometry
from ._local_curvature import LocalApproximationNotCertified
from .definition_binding import definition_checksum
from .nurbs import CurveOnSurface,NurbsCurve,_sqrt_down
from .arrangement_geometry import LinePath
from .regular_source_reference import source_node
from .predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY
from .plane_nurbs import charge_split

@dataclass(frozen=True)
class CurveStar:
    curve_index: int
    inner: tuple
    outer_domains: tuple
    world_boxes: tuple
    curvature_upper: Q
    clearance_lower: Q


def _anchor(curve,t):
    if type(curve) is LinePath:
        return ('world_point',*(Q(x) for x in (curve.start if t==0 else curve.end)))
    if type(curve) in (NurbsCurve,CurveOnSurface):
        return ('world_point',*(Q(x) for x in curve.exact_point(t)))
    return source_node(curve,Q(t)).identity


def _dot_lower(q,box):
    return sum((min(x*a,x*b) for x,(a,b) in zip(q,box)),Q(0))


def certify_curve_stars(loops,*,check=lambda:None,policy=None):
    from ._local_clearance_bvh import (CompetitorPiece,ClearanceBudget,
        build_clearance_bvh,query_clearance_bvh)
    policy=policy or DEFAULT_INTERSECTION_QUALIFICATION_POLICY
    curves=tuple(c for loop in loops for c in loop)
    if not curves:raise LocalApproximationNotCertified('empty curve star source')
    signatures=tuple(definition_checksum(c) for c in curves)
    used=0
    budget=None
    def guarded():
        check()
        if budget is not None and used+budget.counters.operations>policy.max_boxes_per_pair:
            raise LocalApproximationNotCertified('combined curve-star work budget exhausted')
        if tuple(definition_checksum(c) for c in curves)!=signatures:
            raise LocalApproximationNotCertified('curve star source changed')
    budget=ClearanceBudget(policy.max_boxes_per_pair,max_depth=policy.max_subdivision_depth,
        cancellation_check=lambda phase:(guarded() or False))
    def bounds(index,a,b):
        nonlocal used
        guarded();remaining=policy.max_boxes_per_pair-used-budget.counters.operations
        if remaining<1:raise LocalApproximationNotCertified('curve star box budget exhausted')
        result=bound_curve_geometry(curves[index],a,b,
            policy=replace(policy,max_boxes_per_pair=remaining),
            cancellation_check=lambda phase:(guarded() or False))
        used+=result.boxes
        return result.pieces
    offset=0;pending=[]
    for loop in loops:
        if not loop:raise LocalApproximationNotCertified('empty trim loop')
        for k,curve in enumerate(loop):
            guarded()
            if _anchor(curve,1)!=_anchor(loop[(k+1)%len(loop)],0):
                raise LocalApproximationNotCertified('trim corner or closed seam lacks exact owner anchor')
            for j in range(8):pending.append((offset,k,len(loop),Q(j,8),Q(j+1,8),0))
        offset+=len(loop)
    stars=[]
    while pending:
        offset,k,n,a,b,depth=pending.pop();width=b-a
        left,right=Q(k)+a-width/2,Q(k)+b+width/2
        domains=[]
        # At most three adjacent edge intervals; wrap only after exact seam proof.
        first=left.numerator//left.denominator;last=right.numerator//right.denominator
        for j in range(first,last+1):
            lo,hi=max(left,Q(j))-j,min(right,Q(j+1))-j
            if lo<hi:domains.append((offset+j%n,lo,hi))
        outer=[piece for index,lo,hi in domains for piece in bounds(index,lo,hi)]
        midpoints=[tuple((x+y)/2 for x,y in piece.tangent) for piece in outer]
        q=tuple(sum((v[axis]/sum(map(abs,v)) for v in midpoints if any(v)),Q(0)) for axis in range(3))
        if not any(q) or any(_dot_lower(q,piece.tangent)<=0 for piece in outer):
            if depth>=policy.max_subdivision_depth:
                raise LocalApproximationNotCertified('curved branch star projection unresolved')
            charge_split();mid=(a+b)/2
            pending.extend(((offset,k,n,a,mid,depth+1),(offset,k,n,mid,b,depth+1)));continue
        index=offset+k
        inners=[(*sorted(p.parameter_range),p.world_box) for p in bounds(index,a,b)]
        excluded={}
        for other,lo,hi in domains:excluded.setdefault(other,[]).append((lo,hi))
        competitors=[]
        for other in range(len(curves)):
            intervals=sorted(excluded.get(other,()))
            cursor=Q(0)
            for lo,hi in (*intervals,(Q(1),Q(1))):
                if cursor<lo:
                    for piece in bounds(other,cursor,lo):
                        u,v=sorted(piece.parameter_range);competitors.append((other,u,v,piece.world_box))
                cursor=max(cursor,hi)
        if not competitors:raise LocalApproximationNotCertified('empty curve competitor domain needs separate proof')
        for refinement in range(policy.max_subdivision_depth+1):
            guarded()
            tree=build_clearance_bvh(tuple(CompetitorPiece((other,lo,hi),box)
                for other,lo,hi,box in competitors),budget=budget)
            results=[query_clearance_bvh(tree,box,budget=budget) for lo,hi,box in inners]
            if all(result.status=='positive' for result in results):
                distance=_sqrt_down(min(result.minimum_squared for result in results));break
            if refinement==policy.max_subdivision_depth:
                raise LocalApproximationNotCertified('nonlocal curve clearance unresolved')
            keys={key for result in results if result.status=='zero' for key in result.limiting_keys}
            refined=[]
            for other,lo,hi,box in competitors:
                if (other,lo,hi) not in keys:refined.append((other,lo,hi,box));continue
                charge_split();mid=(lo+hi)/2
                for u,v in ((lo,mid),(mid,hi)):
                    for piece in bounds(other,u,v):
                        x,y=sorted(piece.parameter_range);refined.append((other,x,y,piece.world_box))
            competitors=refined
            refined=[]
            for lo,hi,box in inners:
                charge_split();mid=(lo+hi)/2
                for u,v in ((lo,mid),(mid,hi)):
                    for piece in bounds(index,u,v):
                        x,y=sorted(piece.parameter_range);refined.append((x,y,piece.world_box))
            inners=refined
        stars.append(CurveStar(index,(a,b),tuple(domains),tuple(box for lo,hi,box in inners),
            max(piece.curvature_upper for piece in outer),distance))
    guarded()
    return tuple(sorted(stars,key=lambda star:(star.curve_index,star.inner)))
