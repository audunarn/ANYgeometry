"""Exact inner/outer star primitives; caller must supply exhaustive competitors.

These are mathematical building blocks, not authenticated geometry fields.
Topology/contact-star/lineage coverage is NOT inferred from adjacency or samples.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from ._local_curvature import _jet, _sqrt_up, _square_lower, _square_upper, LocalApproximationNotCertified
from .nurbs_surface_intersection._bernstein import subbox, grid_ranges
from .nurbs_surface_intersection._intervals import Interval as I

@dataclass(frozen=True, slots=True)
class SurfaceStarBound:
    inner_box: tuple
    outer_box: tuple
    projection: tuple
    strong_monotonicity_lower: Q
    projection_norm_upper: Q
    parameter_buffer: Q | None
    inner_to_outer_boundary_lower: Q | None
    source_boundary_sides: tuple = ()


def _box(value):
    result=tuple(tuple(map(Q,v)) for v in value)
    if len(result)!=2 or any(len(v)!=2 or not 0<=v[0]<v[1]<=1 for v in result):
        raise LocalApproximationNotCertified('star needs positive unit-chart boxes')
    return result


def _star_boxes(inner_box,outer_box,source_boundary_sides):
    inner,outer=_box(inner_box),_box(outer_box)
    sides=tuple(sorted(set(source_boundary_sides)))
    if any(side not in ((0,0),(0,1),(1,0),(1,1)) for side in sides):
        raise LocalApproximationNotCertified('invalid owner source-boundary side declaration')
    buffers=[]
    for axis,(i,o) in enumerate(zip(inner,outer)):
        if not o[0]<=i[0]<i[1]<=o[1]:
            raise LocalApproximationNotCertified('inner star leaves outer star')
        for end in (0,1):
            if (axis,end) in sides:
                if o[end]!=end:
                    raise LocalApproximationNotCertified('declared source boundary is not a source-chart endpoint')
                continue  # owner establishes there is no same-chart source beyond this side
            gap=i[0]-o[0] if end==0 else o[1]-i[1]
            if gap<=0:
                raise LocalApproximationNotCertified('star requires a strict inner/outer buffer')
            buffers.append(gap)
    return inner,outer,sides,min(buffers) if buffers else None


def _projection_from_derivatives(dx):
    middle=tuple(tuple((x.lo+x.hi)/2 for x in v) for v in dx)
    a=sum(x*x for x in middle[0]);b=sum(x*y for x,y in zip(*middle));c=sum(x*x for x in middle[1])
    determinant=a*c-b*b
    if determinant<=0:raise LocalApproximationNotCertified('midpoint projection is singular')
    return (tuple((c*x-b*y)/determinant for x,y in zip(*middle)),
            tuple((a*y-b*x)/determinant for x,y in zip(*middle)))


def _projection_margin(dx,projection):
    projection=tuple(tuple(map(Q,row)) for row in projection)
    if len(projection)!=2 or any(len(row)!=3 for row in projection):
        raise LocalApproximationNotCertified('surface projection must be two by three')
    matrix=tuple(tuple(I.sum(I(projection[i][k])*dx[j][k] for k in range(3))
        for j in range(2)) for i in range(2))
    off=(matrix[0][1]+matrix[1][0])*I(Q(1,2))
    off_abs=max(abs(off.lo),abs(off.hi))
    mu=min(matrix[0][0].lo-off_abs,matrix[1][1].lo-off_abs)
    norm=_sqrt_up(sum((x*x for row in projection for x in row),Q(0)))
    return mu,norm


def certify_surface_star(grid, inner_box, outer_box, projection=None, *, source_boundary_sides=()):
    """Prove local injectivity over a whole convex outer Bezier chart box.

    A midpoint chooses P only. All authority comes from interval sym(PJ)>0.
    The buffer bound covers inner-to-outer-boundary points in this outer patch;
    it is NOT clearance to arbitrary distant/folded source portions.
    source_boundary_sides must be established by the owner from the FULL source
    domain, never inferred from a knot piece. Declared sides have no outside
    same-chart source; None means no remaining outside side, not infinite
    geometric clearance to other curves/sheets.
    """
    inner,outer,sides,buffer=_star_boxes(inner_box,outer_box,source_boundary_sides)
    restricted=subbox(grid,*outer[0],*outer[1])
    dx,_=_jet(restricted)
    # Convert derivatives of the restricted piece back to the original chart.
    dx=tuple(tuple(x*I(1/(outer[k][1]-outer[k][0])) for x in dx[k]) for k in range(2))
    if projection is None:projection=_projection_from_derivatives(dx)
    projection=tuple(tuple(map(Q,row)) for row in projection)
    mu,norm=_projection_margin(dx,projection)
    if mu<=0 or norm<=0:
        raise LocalApproximationNotCertified('outer star strong monotonicity unresolved')
    return SurfaceStarBound(inner,outer,projection,mu,norm,buffer,
        None if buffer is None else mu*buffer/norm,sides)


def outside_rectangles(outer_box):
    """Disjoint-interior cover of unit square outside outer; no remote omission."""
    (a,b),(c,d)=_box(outer_box)
    boxes=(((Q(0),a),(Q(0),Q(1))),((b,Q(1)),(Q(0),Q(1))),
           ((a,b),(Q(0),c)),((a,b),(d,Q(1))))
    return tuple(box for box in boxes if all(lo<hi for lo,hi in box))


def projected_hull(grid):
    if any(h[3]<=0 for row in grid for h in row):
        raise LocalApproximationNotCertified('star homogeneous weights not positive')
    points=[tuple(h[k]/h[3] for k in range(3)) for row in grid for h in row]
    return tuple(I(min(p[k] for p in points),max(p[k] for p in points)) for k in range(3))


def separation_squared(first, second):
    """Exact nonnegative AABB distance lower square. Zero is inconclusive."""
    return sum((max(Q(0),a.lo-b.hi,b.lo-a.hi)**2 for a,b in zip(first,second)),Q(0))


def curve_curvature_from_jet(first, second):
    """Upper |C' x C''|/|C'|^3 from certified world derivative intervals."""
    from ._local_curvature import _cross
    first,second=tuple(first),tuple(second)
    if len(first)!=3 or len(second)!=3 or any(type(x) is not I for x in first+second):
        raise LocalApproximationNotCertified('curve curvature needs exact world jet intervals')
    speed2=sum((_square_lower(x) for x in first),Q(0))
    if speed2<=0:
        raise LocalApproximationNotCertified('curve tangent regularity unresolved')
    from .nurbs import _sqrt_down
    cross=_sqrt_up(sum((_square_upper(x) for x in _cross(first,second)),Q(0)))
    upper=cross/(speed2*_sqrt_down(speed2))
    return upper,None if upper==0 else 1/upper
