"""Exact piecewise world curve bounds, not contact/field/projection authority.

Tangents of explicit curves use oriented descriptor coordinate r=sign*t.
Implicit PlaneNurbs tangents use an oriented ODE flow in each arc's normalized
box: magnitude is NOT descriptor speed. Endpoints retain one-sided bounds;
no continuity or corner-star certificate is inferred from piece adjacency.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from contextlib import nullcontext

from .definition_binding import definition_checksum
from .arrangement_geometry import LinePath
from .nurbs import NurbsCurve, NurbsSurface, CurveOnSurface, _exact_refine, _sqrt_down, _sqrt_up
from .support_curves import PlaneNurbsCurve
from .nurbs_intersection_curve import NurbsSurfaceIntersectionCurve, intersection_curve_operation
from .nurbs_runtime_binding import validate_curve_runtime, validate_section_runtime
from .native_support_snapshots import _material_support_read_pin
from .predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY
from .plane_nurbs import charge_box, charge_split, depth_limit
from ._local_curvature import LocalApproximationNotCertified, _cross, _square_lower, _square_upper
from .nurbs_surface_intersection._bernstein import bezier_piece, grid_ranges, subbox
from .nurbs_surface_intersection._curve_bounds import jet
from .nurbs_surface_intersection._intervals import Interval as I


@dataclass(frozen=True, slots=True)
class CurveGeometryPiece:
    parameter_range: tuple
    world_box: tuple
    tangent: tuple
    acceleration: tuple
    curvature_upper: Q
    radius_lower: Q | None
    tangent_norm_squared_lower: Q
    parameter_convention: str
    chart_box: tuple = ()


@dataclass(frozen=True, slots=True)
class CurveGeometryBounds:
    definition_checksum: str
    parameter_range: tuple
    pieces: tuple
    anchors: tuple
    boxes: int
    subdivisions: int


class _Unresolved(Exception):
    pass


def _divide(a, b):
    if b.lo <= 0:
        raise LocalApproximationNotCertified('curve homogeneous weight is not positive')
    return a*I(1/b.hi, 1/b.lo)


def _rational(h, d, hh):
    x = tuple(_divide(v,h[-1]) for v in h[:-1])
    dx = tuple(tuple(_divide(row[k]-x[k]*row[-1],h[-1]) for k in range(len(x))) for row in d)
    dd = tuple(tuple(tuple(_divide(hh[i][j][k]-x[k]*hh[i][j][-1]
        -dx[i][k]*d[j][-1]-dx[j][k]*d[i][-1],h[-1]) for k in range(len(x)))
        for j in range(len(d))) for i in range(len(d)))
    return x,dx,dd


def _curve_jet(curve, lo, hi):
    a,b = map(Q,curve.domain)
    left,right = sorted((a+(b-a)*lo,a+(b-a)*hi))
    weights = curve.weights or (1,)*len(curve.controls)
    points = [tuple(Q(x)*Q(w) for x in row)+(Q(w),) for row,w in zip(curve.controls,weights)]
    _, controls, active = _exact_refine(curve.degree,list(map(Q,curve.knots)),points,left,right)
    if len(active) != curve.degree+1:
        raise LocalApproximationNotCertified('pcurve interval crosses an unresolved knot')
    coefficients = [controls[i] for i in active]
    if b<a: coefficients.reverse()
    def deriv(rows):
        n=len(rows)-1
        return [tuple(n*(y-x)/(hi-lo) for x,y in zip(p,q)) for p,q in zip(rows,rows[1:])] if n else [tuple(Q(0) for _ in rows[0])]
    def hull(rows):
        return tuple(I(min(p[k] for p in rows),max(p[k] for p in rows)) for k in range(len(rows[0])))
    first=deriv(coefficients);second=deriv(first)
    x,d,h = _rational(hull(coefficients),(hull(first),),((hull(second),),))
    return x,d[0],h[0][0]


def _surface_jet(surface, box):
    # Select a full knot span containing the whole requested normalized chart
    # interval. Constant coordinates at a knot retain one-sided bounds.
    for knots,degree in ((surface.knots_u,surface.degree_u),(surface.knots_v,surface.degree_v)):
        if any(knots.count(k)>degree for k in set(knots) if knots[0]<k<knots[-1]):
            raise LocalApproximationNotCertified('discontinuous support knots have no curve adapter')
    spans=[]; local=[]; factors=[]
    for bounds,knots,domain in zip(box,(surface.knots_u,surface.knots_v),(surface.u_range,surface.v_range)):
        a,b=map(Q,domain); lo,hi=sorted((a+(b-a)*bounds[0],a+(b-a)*bounds[1]))
        if not 0<=bounds[0]<=bounds[1]<=1:
            raise LocalApproximationNotCertified('curve leaves the support chart')
        distinct=sorted(set(map(Q,knots)))
        candidates=[(u,v) for u,v in zip(distinct,distinct[1:]) if u<v and u<=lo<=hi<=v]
        if not candidates: raise _Unresolved('support-knot crossing needs certified parameter anchors')
        u,v=candidates[0];spans.append((u,v));local.append(((lo-u)/(v-u),(hi-u)/(v-u)));factors.append((b-a)/(v-u))
    grid=bezier_piece(surface,surface.exact_net(),*spans)
    x,d,h=_rational(*jet(grid,tuple(local)))
    d=tuple(tuple(value.scale(factors[i]) for value in row) for i,row in enumerate(d))
    h=tuple(tuple(tuple(value.scale(factors[i]*factors[j]) for value in row)
        for j,row in enumerate(rows)) for i,rows in enumerate(h))
    return x,d,h


def _cos(curve, lo, hi):
    uv,v,a=_curve_jet(curve.pcurve,lo,hi)
    box=tuple((x.lo,x.hi) for x in uv)
    x,j,h=_surface_jet(curve.support,box)
    velocity=tuple(sum((j[i][k]*v[i] for i in range(2)),I(0)) for k in range(3))
    acceleration=tuple(sum((j[i][k]*a[i] for i in range(2)),I(0))+
        sum((h[i][l][k]*v[i]*v[l] for i in range(2) for l in range(2)),I(0)) for k in range(3))
    matrix=tuple(tuple(map(Q,row)) for row in curve.transform)
    def affine(values,position=False):
        return tuple(sum((values[j].scale(matrix[i][j]) for j in range(3)),I(matrix[i][3] if position else 0)) for i in range(3))
    return [(affine(x,True),affine(velocity),affine(acceleration),'oriented descriptor coordinate r=sign*t',box)]


def _plane(curve,lo,hi,sign):
    normal=_cross(tuple(Q(x) for x in curve.plane_record[1]),tuple(Q(x) for x in curve.plane_record[2]))
    rows=[]
    for arc,s_range,x_range in curve._arc_rects(lo,hi):
        knot_box=(s_range,x_range) if arc.slicing==0 else (x_range,s_range)
        spans=(tuple(map(Q,arc.box.u_range)),tuple(map(Q,arc.box.v_range)))
        grid=bezier_piece(curve.surface,curve.surface.exact_net(),*spans)
        local=tuple(((v[0]-p[0])/(p[1]-p[0]),(v[1]-p[0])/(p[1]-p[0])) for v,p in zip(knot_box,spans))
        x,j,h=_rational(*jet(grid,local))
        phi=tuple(sum((j[i][k].scale(normal[k]) for k in range(3)),I(0)) for i in range(2))
        hp=tuple(tuple(sum((h[i][l][k].scale(normal[k]) for k in range(3)),I(0)) for l in range(2)) for i in range(2))
        q=(phi[1].scale(-1),phi[0])
        direction=q[arc.slicing]
        if direction.lo<=0<=direction.hi: raise _Unresolved('implicit tangent orientation unresolved')
        traversal=(arc.s1-arc.s0)*(Q(curve.domain[1])-Q(curve.domain[0]))*sign
        if traversal==0: raise _Unresolved('collapsed implicit arc traversal')
        orient=1 if (direction.lo>0)==(traversal>0) else -1
        q=tuple(v.scale(orient) for v in q)
        # Dq*q = oriented_rot(Hphi*q); the two orientation factors cancel
        # in acceleration, while the tangent follows the requested traversal.
        hq=tuple(sum((hp[i][l]*q[l] for l in range(2)),I(0)) for i in range(2))
        dq=(hq[1].scale(-orient),hq[0].scale(orient))
        velocity=tuple(sum((j[i][k]*q[i] for i in range(2)),I(0)) for k in range(3))
        acceleration=tuple(sum((h[i][l][k]*q[i]*q[l] for i in range(2) for l in range(2)),I(0))+
            sum((j[i][k]*dq[i] for i in range(2)),I(0)) for k in range(3))
        rows.append((x,velocity,acceleration,'oriented implicit flow in normalized arc box; NOT descriptor derivative',knot_box))
    if not rows: raise LocalApproximationNotCertified('implicit interval has no complete arc enclosure')
    return rows


def bound_curve_geometry(curve, lower=0, upper=1, *, policy=None, cancellation_check=None):
    """Cover the entire oriented descriptor interval with exact one-sided bounds.

    Curvature is |T cross A|/|T|**3. No sampled derivative or world inverse
    authorizes a piece. Unknown families, unresolved speed or support-knot
    crossing refuse within the existing qualification/active Work limits.
    """
    lo,hi=map(Q,(lower,upper));sign=1 if hi>lo else -1
    if not 0<=lo<=1 or not 0<=hi<=1 or lo==hi:
        raise LocalApproximationNotCertified('curve interval must be nonempty within [0,1]')
    kind=type(curve)
    if kind not in (LinePath,NurbsCurve,CurveOnSurface,PlaneNurbsCurve,NurbsSurfaceIntersectionCurve):
        raise LocalApproximationNotCertified('curve family has no exact geometry adapter')
    if kind is LinePath and (len(curve.start)!=3 or len(curve.end)!=3):
        raise LocalApproximationNotCertified('world line must be three dimensional')
    policy=policy or DEFAULT_INTERSECTION_QUALIFICATION_POLICY
    signature=definition_checksum(curve)
    policy_signature=definition_checksum(policy)
    def check():
        if definition_checksum(policy)!=policy_signature:
            raise LocalApproximationNotCertified('curve bound policy changed')
        if definition_checksum(curve)!=signature:
            raise LocalApproximationNotCertified('curve definition changed')
        if cancellation_check is not None and cancellation_check('local curve geometry'):
            raise LocalApproximationNotCertified('local curve geometry cancelled')
        if definition_checksum(policy)!=policy_signature:
            raise LocalApproximationNotCertified('curve bound policy changed during callback')
        if definition_checksum(curve)!=signature:
            raise LocalApproximationNotCertified('curve definition changed during callback')
    def native():
        if kind is NurbsCurve: validate_curve_runtime(curve)
        elif kind is CurveOnSurface:
            if type(curve.support) is not NurbsSurface or type(curve.pcurve) is not NurbsCurve:
                raise LocalApproximationNotCertified('COS requires canonical literal support and pcurve')
            validate_curve_runtime(curve.pcurve)
            if _material_support_read_pin(curve.support) is None:
                raise LocalApproximationNotCertified('curve support native binding unknown')
        elif kind is PlaneNurbsCurve:
            validate_section_runtime(curve)
            if _material_support_read_pin(curve.surface) is None:
                raise LocalApproximationNotCertified('section support native binding unknown')
    native();check();native()
    context=intersection_curve_operation(check) if kind is NurbsSurfaceIntersectionCurve else nullcontext()
    with context:
        shape=curve._shape() if kind is NurbsSurfaceIntersectionCurve else None
        start,end=sorted((lo,hi));anchors={start,end}
        if kind in (NurbsCurve,CurveOnSurface):
            pc=curve if kind is NurbsCurve else curve.pcurve
            a,b=map(Q,pc.domain)
            if any(pc.knots.count(k)>pc.degree for k in set(pc.knots)
                   if min(a+(b-a)*start,a+(b-a)*end)<Q(k)<max(a+(b-a)*start,a+(b-a)*end)):
                raise LocalApproximationNotCertified('discontinuous curve knot requires separate branches')
            anchors.update((Q(k)-a)/(b-a) for k in pc.knots if start<(Q(k)-a)/(b-a)<end)
        elif kind is PlaneNurbsCurve:
            a,b=map(Q,curve.domain);n=len(curve._arcs())
            anchors.update((Q(i,n)-a)/(b-a) for i in range(n+1) if start<(Q(i,n)-a)/(b-a)<end)
        elif shape is not None: anchors.update(t for t in shape.breakpoints if start<t<end)
        ordered=sorted(anchors);queue=[(a,b,0) for a,b in zip(ordered,ordered[1:])]
        pieces=[];boxes=splits=0
        while queue:
            check();native();charge_box();boxes+=1
            if boxes>policy.max_boxes_per_pair: raise LocalApproximationNotCertified('curve box budget exhausted')
            a,b,depth=queue.pop()
            try:
                if kind is LinePath:
                    delta=tuple(Q(y)-Q(x) for x,y in zip(curve.start,curve.end))
                    positions=tuple(I(min(Q(x)+a*d,Q(x)+b*d),max(Q(x)+a*d,Q(x)+b*d)) for x,d in zip(curve.start,delta))
                    rows=[(positions,tuple(I(d) for d in delta),(I(0),)*3,'oriented descriptor coordinate r=sign*t',())]
                elif kind is NurbsCurve:
                    x,v,acc=_curve_jet(curve,a,b)
                    if len(x)!=3: raise LocalApproximationNotCertified('world curve must be three dimensional')
                    rows=[(x,v,acc,'oriented descriptor coordinate r=sign*t',())]
                elif kind is CurveOnSurface: rows=_cos(curve,a,b)
                elif kind is PlaneNurbsCurve: rows=_plane(curve,a,b,sign)
                else:
                    bounds=shape.enclose(a,b)
                    if not bounds: raise LocalApproximationNotCertified('NN bound has empty coverage')
                    intervals=sorted(row.parameter_range for row in bounds)
                    reach=a
                    for u,v in intervals:
                        if u>reach: raise LocalApproximationNotCertified('NN bound omitted a parameter interval')
                        reach=max(reach,v)
                    if reach<b: raise LocalApproximationNotCertified('NN bound omitted interval end')
                    rows=[(tuple(I(x,y) for x,y in zip(row.world_lo,row.world_hi)),row.world_derivative,
                        row.world_second_derivative,'oriented descriptor coordinate r=sign*t',row.first_chart) for row in bounds]
                pending=[]
                for x,v,acc,convention,chart in rows:
                    if kind is not PlaneNurbsCurve: v=tuple(t.scale(sign) for t in v)
                    speed2=sum((_square_lower(t) for t in v),Q(0))
                    if speed2<=0: raise _Unresolved('curve speed lower bound unresolved')
                    upper=_sqrt_up(sum((_square_upper(t) for t in _cross(v,acc)),Q(0)))/(speed2*_sqrt_down(speed2))
                    pending.append(CurveGeometryPiece((a,b) if sign>0 else (b,a),tuple((t.lo,t.hi) for t in x),
                        tuple((t.lo,t.hi) for t in v),tuple((t.lo,t.hi) for t in acc),upper,
                        None if upper==0 else 1/upper,speed2,convention,chart))
                # The owner can return overlapping endpoint/neighbor enclosures.
                # Their UNION covers this parameter interval; no individual box
                # is mislabeled as bounding the entire interval on its own.
                def union(attribute):
                    values=[getattr(item,attribute) for item in pending]
                    return tuple((min(row[k][0] for row in values),max(row[k][1] for row in values)) for k in range(3))
                maximum=max(item.curvature_upper for item in pending)
                pieces.append(CurveGeometryPiece((a,b) if sign>0 else (b,a),union('world_box'),union('tangent'),
                    union('acceleration'),maximum,None if maximum==0 else 1/maximum,
                    min(item.tangent_norm_squared_lower for item in pending),pending[0].parameter_convention,
                    tuple(item.chart_box for item in pending)))
            except _Unresolved as error:
                if depth>=min(policy.max_subdivision_depth,depth_limit(policy.max_subdivision_depth)):
                    raise LocalApproximationNotCertified(str(error)+' at subdivision limit') from error
                charge_split();splits+=1;mid=(a+b)/2
                queue.extend(((a,mid,depth+1),(mid,b,depth+1)))
        anchors.update(t for piece in pieces for t in piece.parameter_range)
        check();native()
        if shape is not None: shape._validate()
        if definition_checksum(curve)!=signature: raise LocalApproximationNotCertified('curve changed before result')
        return CurveGeometryBounds(signature,(lo,hi),tuple(sorted(pieces,key=lambda p:sign*p.parameter_range[0])),
            tuple(sorted(anchors,reverse=sign<0)),boxes,splits)
