"""Operation-local exact span operands and stitched C1 projection certificates.

Numerical building blocks, not issued model/material/contact authority. All
coordinates are the SAME normalized authored support chart, including reversals.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from math import comb
from .definition_binding import definition_checksum
from .nurbs import NurbsSurface
from .predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY
from .plane_nurbs import charge_box, charge_split, depth_limit
from ._local_curvature import _pieces, _jet, LocalApproximationNotCertified
from ._local_neighborhood import (_box, _star_boxes, _projection_from_derivatives,
    _projection_margin, SurfaceStarBound, projected_hull)
from .nurbs_surface_intersection._bernstein import bezier_piece, subbox, derivative_grid
from .nurbs_surface_intersection._intervals import Interval as I


@dataclass(frozen=True, slots=True)
class SurfaceSpan:
    chart_box: tuple
    grid: tuple


@dataclass(frozen=True, slots=True)
class SurfaceSpanAtlas:
    source_definition: str
    pieces: tuple


class SpanBudget:
    def __init__(self, policy=None, check=None):
        policy=policy or DEFAULT_INTERSECTION_QUALIFICATION_POLICY
        self.limit=policy.max_boxes_per_pair
        self.depth=depth_limit(policy.max_subdivision_depth)
        self.used=0; self.splits=0; self.check=check
    def take(self):
        self.used+=1
        if self.used>self.limit:raise LocalApproximationNotCertified('span box budget exhausted')
        charge_box()
        if self.check is not None and self.check():raise LocalApproximationNotCertified('span operation cancelled')
    def split(self):
        self.splits+=1;charge_split()


def surface_span_atlas(surface, *, budget):
    if type(surface) is not NurbsSurface:
        raise LocalApproximationNotCertified('span atlas needs canonical NURBS support')
    require_span_context(surface)
    signature=definition_checksum(surface)
    budget.take();net=surface.exact_net()
    up=_pieces(surface.knots_u,surface.u_range,(Q(0),Q(1)))
    vp=_pieces(surface.knots_v,surface.v_range,(Q(0),Q(1)))
    pieces=[]
    for a,b,uk in up:
        for c,d,vk in vp:
            budget.take();grid=bezier_piece(surface,net,uk,vk)
            if surface.u_range[1]<surface.u_range[0]:grid=tuple(reversed(grid))
            if surface.v_range[1]<surface.v_range[0]:grid=tuple(tuple(reversed(row)) for row in grid)
            if any(h[3]<=0 for row in grid for h in row):
                raise LocalApproximationNotCertified('span weights must be positive')
            pieces.append(SurfaceSpan(((a,b),(c,d)),grid))
    if definition_checksum(surface)!=signature:raise LocalApproximationNotCertified('span source changed')
    atlas=SurfaceSpanAtlas(signature,tuple(pieces))
    validate_span_coverage(atlas)
    require_span_context(surface)
    return atlas


def validate_span_coverage(atlas):
    if type(atlas) is not SurfaceSpanAtlas or not atlas.pieces:
        raise LocalApproximationNotCertified('missing span coverage')
    rows={}
    axes=[set(),set()]
    for piece in atlas.pieces:
        if type(piece) is not SurfaceSpan:raise LocalApproximationNotCertified('invalid span record')
        box=_box(piece.chart_box)
        if box in rows:raise LocalApproximationNotCertified('duplicate span coverage')
        rows[box]=piece
        for k in range(2):axes[k].update(box[k])
    axes=[sorted(a) for a in axes]
    if any(a[0]!=0 or a[-1]!=1 for a in axes):raise LocalApproximationNotCertified('incomplete span coverage')
    expected={((a,b),(c,d)) for a,b in zip(axes[0],axes[0][1:]) for c,d in zip(axes[1],axes[1][1:])}
    if set(rows)!=expected:raise LocalApproximationNotCertified('gap or overlap in span coverage')
    return rows,axes


def _edge(grid,axis,end):
    return grid[0 if end==0 else -1] if axis==0 else tuple(row[0 if end==0 else -1] for row in grid)


def _product(a,b):
    m,n=len(a)-1,len(b)-1
    return tuple(sum((Q(comb(m,i)*comb(n,k-i),comb(m+n,k))*a[i]*b[k-i]
        for i in range(max(0,k-n),min(m,k)+1)),Q(0)) for k in range(m+n+1))


def _c1_seam(first,second,axis):
    a,b=_edge(first.grid,axis,1),_edge(second.grid,axis,0)
    # Canonical extraction uses one common homogeneous spline and identical
    # transverse spans, so boundary controls must be literally identical.
    if a!=b:raise LocalApproximationNotCertified('span C0 continuity unresolved')
    da=_edge(derivative_grid(first.grid,axis),axis,1)
    db=_edge(derivative_grid(second.grid,axis),axis,0)
    wa=first.chart_box[axis][1]-first.chart_box[axis][0]
    wb=second.chart_box[axis][1]-second.chart_box[axis][0]
    difference=tuple(tuple(x/wa-y/wb for x,y in zip(u,v)) for u,v in zip(da,db))
    # Compare world rational normal derivatives as exact Bernstein polynomials:
    # (Dxyz_left-Dxyz_right)*W - Hxyz*(Dw_left-Dw_right) == 0.
    weight=tuple(h[3] for h in a);dw=tuple(h[3] for h in difference)
    for k in range(3):
        if _product(tuple(h[k] for h in difference),weight)!=_product(tuple(h[k] for h in a),dw):
            raise LocalApproximationNotCertified('span C1 continuity unresolved')


def certify_span_continuity(atlas, *, budget):
    rows,axes=validate_span_coverage(atlas)
    for axis in (0,1):
        other=1-axis
        for i in range(len(axes[axis])-2):
            left=tuple(axes[axis][i:i+2]);right=tuple(axes[axis][i+1:i+3])
            for a,b in zip(axes[other],axes[other][1:]):
                first=[None,None];second=[None,None]
                first[axis]=left;second[axis]=right;first[other]=second[other]=(a,b)
                budget.take();_c1_seam(rows[tuple(first)],rows[tuple(second)],axis)


def restricted_span_pieces(atlas, box, *, take):
    require_span_context()
    box=_box(box);result=[]
    for piece in atlas.pieces:
        overlap=tuple((max(a,c),min(b,d)) for (a,b),(c,d) in zip(box,piece.chart_box))
        if any(a>=b for a,b in overlap):continue
        take()
        local=tuple(((a-c)/(d-c),(b-c)/(d-c)) for (a,b),(c,d) in zip(overlap,piece.chart_box))
        result.append(SurfaceSpan(overlap,subbox(piece.grid,*local[0],*local[1])))
    if not result:raise LocalApproximationNotCertified('empty span restriction')
    return tuple(result)


def span_hull(atlas,box,*,take):
    pieces=restricted_span_pieces(atlas,box,take=take)
    hulls=tuple(projected_hull(piece.grid) for piece in pieces)
    return tuple(I(min(h[k].lo for h in hulls),max(h[k].hi for h in hulls)) for k in range(3))


def _global_derivatives(piece):
    dx,_=_jet(piece.grid)
    return tuple(tuple(x*I(1/(piece.chart_box[k][1]-piece.chart_box[k][0])) for x in dx[k]) for k in range(2))


def certify_piecewise_surface_star(atlas,inner_box,outer_box,*,budget,projection=None,source_boundary_sides=()):
    require_span_context()
    atlas_pin=definition_checksum(atlas)
    inner,outer,sides,buffer=_star_boxes(inner_box,outer_box,source_boundary_sides)
    certify_span_continuity(atlas,budget=budget)
    pieces=restricted_span_pieces(atlas,outer,take=budget.take)
    if projection is None:
        derivatives=tuple(_global_derivatives(p) for p in pieces)
        hull=tuple(tuple(I(min(d[k][j].lo for d in derivatives),max(d[k][j].hi for d in derivatives))
            for j in range(3)) for k in range(2))
        projection=_projection_from_derivatives(hull)
    queue=[(p,0) for p in reversed(pieces)];minimum=None;norm=None
    while queue:
        piece,level=queue.pop();budget.take()
        mu,norm=_projection_margin(_global_derivatives(piece),projection)
        if norm<=0:raise LocalApproximationNotCertified('piecewise strong monotonicity unresolved')
        if mu>0:
            minimum=mu if minimum is None else min(minimum,mu);continue
        if level>=budget.depth:raise LocalApproximationNotCertified('piecewise strong monotonicity unresolved at depth limit')
        budget.split();axis=max(range(2),key=lambda k:piece.chart_box[k][1]-piece.chart_box[k][0])
        mid=sum(piece.chart_box[axis])/2
        for lo,hi in ((Q(0),Q(1,2)),(Q(1,2),Q(1))):
            limits=[(Q(0),Q(1)),(Q(0),Q(1))];limits[axis]=(lo,hi)
            box=list(piece.chart_box);box[axis]=(box[axis][0],mid) if lo==0 else (mid,box[axis][1])
            queue.append((SurfaceSpan(tuple(box),subbox(piece.grid,*limits[0],*limits[1])),level+1))
    require_span_context()
    if definition_checksum(atlas)!=atlas_pin:raise LocalApproximationNotCertified('span atlas changed during certification')
    return SurfaceStarBound(inner,outer,tuple(tuple(map(Q,row)) for row in projection),minimum,norm,
        buffer,None if buffer is None else minimum*buffer/norm,sides)


# Finite effective algorithm context; this is not a model revision cache.
# Imported aliases AND their defining module bindings are pinned separately.
from . import _local_curvature as _curvature, _local_neighborhood as _neighborhood
from . import nurbs as _nurbs, _source_box_hull as _hull_context
from .nurbs_surface_intersection import _bernstein, _intervals
import sys as _sys
_THIS=_sys.modules[__name__]
_FUNCTION_BINDINGS=tuple((owner,name) for owner,names in (
    (_THIS,('surface_span_atlas','validate_span_coverage','_edge','_product','_c1_seam',
        'certify_span_continuity','restricted_span_pieces','span_hull','_global_derivatives',
        'certify_piecewise_surface_star','_pieces','_jet','_box','_star_boxes','charge_box','charge_split','depth_limit',
        '_projection_from_derivatives','_projection_margin','projected_hull',
        'bezier_piece','subbox','derivative_grid','comb')),
    (_curvature,('_jet','_divide','_pieces','_bound','_cross','_square_lower','_square_upper','_planar','derivative_grid','grid_ranges','subbox')),
    (_neighborhood,('_star_boxes','_projection_from_derivatives','_projection_margin','projected_hull','_sqrt_up')),
    (_bernstein,('bezier_piece','_exact_refine','subbox','_lerp','_split_axis','_blossom_axis','_blossom_axis_reference','derivative_grid','grid_ranges')),
    (_nurbs,('_exact_refine','_exact_insert','_exact_span','_sqrt_up')),
    (_nurbs.NurbsSurface,('exact_net',)),
    (SurfaceSpan,('__init__',)),(SurfaceSpanAtlas,('__init__',)),
    (SurfaceStarBound,('__init__',)),(SpanBudget,('__init__','take','split')),
    (_curvature.CurvatureTile,('__init__',)),(_curvature.SurfaceCurvatureBounds,('__init__',)),
    (I,('__init__','__add__','__sub__','__mul__','sum')))
    for name in names)

def span_runtime_signature():
    functions=tuple((getattr(owner,name),getattr(getattr(owner,name),'__code__',None),
        definition_checksum(getattr(getattr(owner,name),'__defaults__',None)),
        definition_checksum(getattr(getattr(owner,name),'__kwdefaults__',None)))
        for owner,name in _FUNCTION_BINDINGS)
    aliases=(_THIS.Q,_THIS.I,_THIS.NurbsSurface,_bernstein.Fraction,_intervals.Fraction,
        _curvature.I,_neighborhood.I,_curvature.Q,_neighborhood.Q,_bernstein.Interval,
        _intervals.Interval,_nurbs.Fraction,_nurbs.math.isqrt,
        _nurbs.NurbsSurface,SurfaceSpan,SurfaceSpanAtlas,SurfaceStarBound,
        _neighborhood.SurfaceStarBound,_curvature.CurvatureTile,_curvature.SurfaceCurvatureBounds,SpanBudget)
    return functions,aliases,definition_checksum(DEFAULT_INTERSECTION_QUALIFICATION_POLICY)

_SPAN_CONTEXT=span_runtime_signature()

def require_span_context(surface=None):
    if span_runtime_signature()!=_SPAN_CONTEXT or surface is not None and not _hull_context._context(surface):
        raise LocalApproximationNotCertified('span numerical implementation context unsupported or changed')
