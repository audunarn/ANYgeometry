"""Exact curvature bounds: private building block, NOT local-field admission.

Clearance, topology, native qualification and source lineage are separate gates.
No result from this module is an authenticated source-cell receipt.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from .errors import GeometryError
from .definition_binding import definition_checksum
from .nurbs import NurbsSurface, _sqrt_up
from .surfaces import Plane
from .predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY
from .plane_nurbs import charge_box, charge_split
from .nurbs_surface_intersection._bernstein import bezier_piece, derivative_grid, grid_ranges, subbox
from .nurbs_surface_intersection._intervals import Interval as I

class LocalApproximationNotCertified(GeometryError):
    """An unresolved local bound; never evidence of an actual error violation."""

@dataclass(frozen=True, slots=True)
class CurvatureTile:
    chart_box: tuple
    curvature_upper: Q
    radius_lower: Q | None
    sigma_min_squared_lower: Q
    hessian_norm_upper: Q

@dataclass(frozen=True, slots=True)
class SurfaceCurvatureBounds:
    definition_checksum: str
    chart_box: tuple
    tiles: tuple
    boxes: int
    subdivisions: int


def _divide(a, b):
    if b.lo <= 0:
        raise LocalApproximationNotCertified('rational weight is not strictly positive')
    return a*I(1/b.hi,1/b.lo)

def _cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])

def _square_upper(x): return max(x.lo*x.lo,x.hi*x.hi)
def _square_lower(x): return Q(0) if x.lo<=0<=x.hi else min(x.lo*x.lo,x.hi*x.hi)

def _jet(grid):
    h=grid_ranges(grid)
    first=tuple(derivative_grid(grid,k) for k in range(2))
    dh=tuple(grid_ranges(g) for g in first)
    ddh=tuple(tuple(grid_ranges(derivative_grid(g,k)) for k in range(2)) for g in first)
    x=tuple(_divide(h[k],h[3]) for k in range(3))
    dx=tuple(tuple(_divide(dh[i][k]-x[k]*dh[i][3],h[3]) for k in range(3)) for i in range(2))
    ddx=tuple(tuple(tuple(_divide(ddh[i][j][k]-x[k]*ddh[i][j][3]
        -dx[i][k]*dh[j][3]-dx[j][k]*dh[i][3],h[3]) for k in range(3))
        for j in range(2)) for i in range(2))
    return dx,ddx

def _planar(grid):
    points=tuple(tuple(h[k]/h[3] for k in range(3)) for row in grid for h in row)
    a=points[0]
    offsets=[tuple(p[k]-a[k] for k in range(3)) for p in points]
    for b in offsets:
        for c in offsets:
            n=(b[1]*c[2]-b[2]*c[1],b[2]*c[0]-b[0]*c[2],b[0]*c[1]-b[1]*c[0])
            if any(n):
                return all(sum(n[k]*v[k] for k in range(3))==0 for v in offsets)
    return False  # a collapsed net is not an admitted regular plane

def _bound(grid,box):
    if any(h[3]<=0 for row in grid for h in row):
        raise LocalApproximationNotCertified('rational control weights are not positive')
    dx,ddx=_jet(grid)
    cross=_cross(*dx)
    cross2=sum((_square_lower(x) for x in cross),Q(0))
    maximum2=sum((_square_upper(x) for v in dx for x in v),Q(0))
    if cross2<=0 or maximum2<=0:
        return None
    minimum2=cross2/maximum2
    hessian=_sqrt_up(sum((_square_upper(x) for row in ddx for v in row for x in v),Q(0)))
    curvature=Q(0) if _planar(grid) else hessian/minimum2
    return CurvatureTile(box,curvature,None if curvature==0 else 1/curvature,minimum2,hessian)

def _pieces(knots, chart_range, interval):
    a,b=map(Q,chart_range);scale=b-a
    if scale==0: raise LocalApproximationNotCertified('collapsed source chart')
    values={interval[0],interval[1]}
    values.update((Q(k)-a)/scale for k in knots if interval[0]<(Q(k)-a)/scale<interval[1])
    values=sorted(values)
    return tuple((values[i],values[i+1],tuple(sorted((a+scale*values[i],a+scale*values[i+1]))))
                 for i in range(len(values)-1))

def bound_surface_curvature(surface, chart_box=((0,1),(0,1)), *, policy=None, cancellation_check=None, _atlas=None):
    """Bound every knot piece, subdividing only unresolved regularity boxes.

    Uses the existing intersection policy's depth/box limits and charges active
    owner Work. No sampling, floor lift, feature exclusion, or proof admission.
    """
    policy=policy or DEFAULT_INTERSECTION_QUALIFICATION_POLICY
    box=tuple(tuple(map(Q,part)) for part in chart_box)
    if len(box)!=2 or any(len(p)!=2 or not 0<=p[0]<p[1]<=1 for p in box):
        raise LocalApproximationNotCertified('curvature requires a positive source chart box')
    if type(surface) not in (Plane,NurbsSurface):
        raise LocalApproximationNotCertified('surface has no exact curvature adapter')
    signature=definition_checksum(surface)
    def check():
        if cancellation_check is not None and cancellation_check('local curvature bounds'):
            raise LocalApproximationNotCertified('local curvature query cancelled')
        if definition_checksum(surface)!=signature:
            raise LocalApproximationNotCertified('curvature source changed')
    check()
    if type(surface) is Plane:
        dx=tuple(tuple(I(Q(float(x))) for x in v) for v in (surface.u_vector,surface.v_vector))
        cross2=sum((_square_lower(x) for x in _cross(*dx)),Q(0))
        maximum2=sum((_square_upper(x) for v in dx for x in v),Q(0))
        charge_box();check()
        if cross2<=0 or maximum2<=0: raise LocalApproximationNotCertified('plane is not regular')
        return SurfaceCurvatureBounds(signature,box,(CurvatureTile(box,Q(0),None,cross2/maximum2,Q(0)),),1,0)
    queue=[]
    if _atlas is not None:
        from ._local_piecewise_surface_star import restricted_span_pieces,validate_span_coverage
        if _atlas.source_definition!=signature:
            raise LocalApproximationNotCertified('curvature span atlas source mismatch')
        validate_span_coverage(_atlas)
        visits=0
        def take():
            nonlocal visits
            visits+=1
            if visits>policy.max_boxes_per_pair:raise LocalApproximationNotCertified('curvature box budget exhausted')
            check();charge_box()
        queue=[(piece.grid,piece.chart_box,0) for piece in restricted_span_pieces(_atlas,box,take=take)]
    else:
        up=_pieces(surface.knots_u,surface.u_range,box[0]);vp=_pieces(surface.knots_v,surface.v_range,box[1])
        if len(up)*len(vp)>policy.max_boxes_per_pair:
            raise LocalApproximationNotCertified('curvature extraction exceeds box budget')
        # Exact homogeneous literals; this building block grants no native authority.
        net=surface.exact_net()
        queue=[]
        for u0,u1,uk in up:
            for v0,v1,vk in vp:
                check();charge_box()
                grid=bezier_piece(surface,net,uk,vk)
                if surface.u_range[1]<surface.u_range[0]: grid=tuple(reversed(grid))
                if surface.v_range[1]<surface.v_range[0]: grid=tuple(tuple(reversed(row)) for row in grid)
                queue.append((grid,((u0,u1),(v0,v1)),0))
    tiles=[];boxes=len(queue);splits=0
    while queue:
        check();charge_box();boxes+=1
        if boxes>policy.max_boxes_per_pair: raise LocalApproximationNotCertified('curvature box budget exhausted')
        grid,tile,depth=queue.pop()
        result=_bound(grid,tile)
        if result is not None:
            tiles.append(result);continue
        if depth>=policy.max_subdivision_depth:
            raise LocalApproximationNotCertified('curvature regularity unresolved at depth limit')
        charge_split();splits+=1
        axis=max(range(2),key=lambda k:tile[k][1]-tile[k][0]);mid=sum(tile[axis])/2
        for lower,upper in ((Q(0),Q(1,2)),(Q(1,2),Q(1))):
            limits=[(Q(0),Q(1)),(Q(0),Q(1))];limits[axis]=(lower,upper)
            child=list(tile);child[axis]=(tile[axis][0],mid) if lower==0 else (mid,tile[axis][1])
            queue.append((subbox(grid,*limits[0],*limits[1]),tuple(child),depth+1))
    check()
    return SurfaceCurvatureBounds(signature,box,tuple(sorted(tiles,key=lambda t:t.chart_box)),boxes,splits)
