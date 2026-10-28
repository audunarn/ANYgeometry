"""Complete bounded per-tile competitor partition, without contact authority.

Contact rows are conditional source plans. Only field allocation after actual
prepared realization may turn their limits into executable numerical budgets.
"""
from fractions import Fraction as Q
from heapq import heappush,heappop
from . import _source_box_hull as _hull_context
from . import _local_neighborhood as _neighborhood
from .nurbs_surface_intersection import _bernstein,_intervals
from .nurbs_surface_intersection._intervals import Interval as I
from .nurbs import NurbsSurface
from .surfaces import Plane
from itertools import product
from ._local_curvature import LocalApproximationNotCertified
from ._local_neighborhood import projected_hull,separation_squared
from .nurbs_surface_intersection._bernstein import bezier_piece
from .plane_nurbs import charge_box,charge_split,depth_limit
from .predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY
from ._local_contact_stars import query_local_contact_stars,query_local_contact_proposal,_ContactProposalRegistry


class SpatialBudget:
    def __init__(self,check):
        policy=DEFAULT_INTERSECTION_QUALIFICATION_POLICY
        self.limit=policy.max_boxes_per_pair
        self.depth=depth_limit(policy.max_subdivision_depth)
        self.used=0
        self.check=check
    def take(self,*unused):
        self.used+=1
        if self.used>self.limit: raise LocalApproximationNotCertified('local spatial clearance budget exhausted')
        charge_box();self.check()
        return False


def _native(support,box):
    return tuple(tuple(sorted(Q(a)+(Q(b)-Q(a))*t for t in pair))
                 for (a,b),pair in zip((support.u_range,support.v_range),box))


def _directions(grid):
    """An arbitrary exact candidate direction, NOT a planarity certificate.

    Positive rational Bezier weights place the entire patch in the convex
    hull of its dehomogenized controls. Projection onto ANY nonzero direction
    therefore gives a valid whole-patch interval, including nonplanar patches.
    """
    if any(h[3]<=0 for row in grid for h in row):
        raise LocalApproximationNotCertified('directional hull weights must be positive')
    points=tuple(tuple(tuple(h[k]/h[3] for k in range(3)) for h in row) for row in grid)
    a=points[0][0]
    u=tuple(points[-1][0][k]-a[k] for k in range(3))
    v=tuple(points[0][-1][k]-a[k] for k in range(3))
    q=(u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0])
    if not any(q):return ()
    values=tuple(sum(q[k]*p[k] for k in range(3)) for row in points for p in row)
    return ((q,min(values),max(values)),)


def _gap(inner,hull,directions):
    lower=separation_squared(inner,hull)
    for q,lo,hi in directions:
        norm=sum(x*x for x in q)
        if norm==0:continue
        a=sum(min(x*v.lo,x*v.hi) for x,v in zip(q,inner))
        b=sum(max(x*v.lo,x*v.hi) for x,v in zip(q,inner))
        gap=max(Q(0),a-hi,lo-b)
        lower=max(lower,gap*gap/norm)
    return lower


def _inflate_transport(hull,directions,slack):
    """Conservative source/proxy allowance on BOTH distance certificates."""
    if slack<0:raise LocalApproximationNotCertified('negative source transport allowance')
    if not slack:return hull,directions
    return (tuple(I(v.lo-slack,v.hi+slack) for v in hull),
        tuple((q,lo-sum(abs(x) for x in q)*slack,hi+sum(abs(x) for x in q)*slack)
              for q,lo,hi in directions))


def _memo_functions():
    return (_native,_directions,_inflate_transport,bezier_piece,projected_hull,I.__init__,_hull_context._context,
            _hull_context._canonical_literals,_hull_context._keywords)


_MEMO_FUNCTIONS=tuple((f,f.__code__,f.__defaults__,tuple(sorted((f.__kwdefaults__ or {}).items())))
                      for f in _memo_functions())
def _memo_bindings():
    return (_bernstein._exact_refine,_bernstein.Fraction,_neighborhood.Q,_neighborhood.I,
            I,_intervals.Fraction,Q,I.__getattribute__)


_MEMO_BINDINGS=_memo_bindings()


class _HullMemo:
    """One partition operation only; exact literal hulls, never gap/receipt truth."""
    def __init__(self,atlases=None,plane_covers=None,transport_slacks=None):
        self.rows={};self.sources={};self.atlases={} if atlases is None else dict(atlases)
        self.plane_covers={} if plane_covers is None else dict(plane_covers)
        self.transport_slacks={} if transport_slacks is None else dict(transport_slacks)
    def key(self,support,box):
        if id(support) in self.atlases:
            from ._local_piecewise_surface_star import require_span_context
            require_span_context(support)
        if any(a is not b for a,b in zip(_memo_bindings(),_MEMO_BINDINGS)):return None
        if type(box) is not tuple or any(type(part) is not tuple or any(type(x) is not Q for x in part) for part in box):return None
        for f,(original,code,defaults,keywords) in zip(_memo_functions(),_MEMO_FUNCTIONS):
            if (f is not original or f.__code__ is not code or f.__defaults__ is not defaults
                    or tuple(sorted((f.__kwdefaults__ or {}).items()))!=keywords):return None
        if not _hull_context._context(support) or not _hull_context._canonical_literals(support):return None
        self.sources[id(support)]=support  # Retain exact native identity for this operation.
        literal=(support.degree_u,support.degree_v,support.knots_u,support.knots_v,
                 support.controls,support.weights,support.u_range,support.v_range)
        return id(support),literal,box,self.transport_slacks.get(id(support),Q(0))


def _hull_data(support,box,full_hull,budget,memo=None):
    budget.take()  # Every visit, including memo hits, retains full caller guard/Work.
    cover=None if memo is None else memo.plane_covers.get(id(support))
    if type(support) is Plane and cover is not None:
        native=tuple(tuple(a+(b-a)*t for t in pair) for (a,b),pair in zip(cover.chart_box,box))
        points=tuple(cover.frame.world(u,v) for u,v in product(*native))
        slack=cover.transport_upper
        return tuple(I(min(p[k] for p in points)-slack,max(p[k] for p in points)+slack) for k in range(3)),()
    if type(support) is not NurbsSurface:
        return full_hull,()  # Actual full trimmed world hull, including nonunit Plane UV.
    key=None if memo is None else memo.key(support,box)
    if key is not None and key in memo.rows:
        bounds,directions=memo.rows[key]
        return tuple(I(a,b) for a,b in bounds),directions
    atlas=None if memo is None else memo.atlases.get(id(support))
    if atlas is not None:
        from ._local_piecewise_surface_star import restricted_span_pieces
        pieces=restricted_span_pieces(atlas,box,take=budget.take)
        hulls=tuple(projected_hull(piece.grid) for piece in pieces)
        result=tuple(I(min(h[k].lo for h in hulls),max(h[k].hi for h in hulls)) for k in range(3))
        # A direction is usable only with extrema over the COMPLETE restriction.
        points=tuple(tuple(h[k]/h[3] for k in range(3)) for piece in pieces for row in piece.grid for h in row)
        candidates=tuple(q for piece in pieces for q,_,_ in _directions(piece.grid))
        directions=tuple((q,min(sum(q[k]*point[k] for k in range(3)) for point in points),
                            max(sum(q[k]*point[k] for k in range(3)) for point in points)) for q in candidates)
    else:
        native=_native(support,box)
        grid=bezier_piece(support,support.exact_net(),*native)
        result=projected_hull(grid)
        directions=_directions(grid)
    result,directions=_inflate_transport(result,directions,Q(0) if memo is None else memo.transport_slacks.get(id(support),Q(0)))
    # No mutable Interval, support, proof or caller-owned object is cached.
    if key is not None and key==memo.key(support,box) and len(memo.rows)<4096:
        bounds=tuple((Q(x.lo.numerator,x.lo.denominator),Q(x.hi.numerator,x.hi.denominator)) for x in result)
        memo.rows[key]=(bounds,directions)
    return result,directions


def _hull(support,box,full_hull,budget,memo=None):
    return _hull_data(support,box,full_hull,budget,memo)[0]


def _split(box,cuts=()):
    axis=max(range(2),key=lambda k:box[k][1]-box[k][0])
    midpoint=sum(box[axis])/2
    eligible=tuple((abs(value-sum(box[k])/2),k,value) for k,value in cuts if box[k][0]<value<box[k][1])
    if eligible:
        _,axis,mid=min(eligible)
    else:mid=midpoint
    left=list(box);right=list(box)
    left[axis]=(box[axis][0],mid);right[axis]=(mid,box[axis][1])
    charge_split()
    return tuple(left),tuple(right)


def _ordinary(inner,support,full_hull,budget,depth,memo=None):
    """Complete retained cover, visiting smallest exact lower bounds first.

    A node may remain coarse only if its own bound cannot reduce the already
    accepted minimum. It stays explicitly in the cover with that valid bound.
    """
    queue=[];serial=0;leaves=[];best=None
    extent=max(v.hi-v.lo for v in inner)
    def push(box,level):
        nonlocal serial
        hull,directions=_hull_data(support,box,full_hull,budget,memo)
        gap=_gap(inner,hull,directions)
        coarse=max(v.hi-v.lo for v in hull)>extent
        serial+=1
        heappush(queue,(gap,serial,box,level,coarse))
    subdividable=type(support) is NurbsSurface or (memo is not None and id(support) in memo.plane_covers)
    push(((Q(0),Q(1)),(Q(0),Q(1))),0)
    while queue:
        gap,_,box,level,coarse=heappop(queue)
        if gap>0 and (best is not None and gap>=best or not coarse or level>=depth or not subdividable):
            leaves.append((box,gap));best=gap if best is None else min(best,gap)
            continue
        if level>=depth or not subdividable:return None
        for child in _split(box):push(child,level+1)
    return best,tuple(leaves)


def partition_surface_clearance(model,supports,hulls,curvatures,*,connect_intent,check,initial_work=0,contact_registry=None,numerical_boundary=None,span_atlases=None,plane_covers=None,sweep_unions=()):
    """Every source tile and every other source face must have a whole proof."""
    if contact_registry is None:
        contact_registry=_ContactProposalRegistry(model)
        check=contact_registry.wrap_callback(check)
    contact_guard=contact_registry.guard
    contact_runtime_check=contact_registry._runtime_check
    budget=SpatialBudget(check);budget.used=initial_work
    pairs={};views={};contacts=[]
    transport_slacks={id(supports[m.face_id]):Q(union.transport_upper) for union in sweep_unions for m in union.members}
    memo=_HullMemo(span_atlases,plane_covers,transport_slacks)
    # Broad-phase boxes must be at least as conservative as retained leaves.
    hulls={face:_inflate_transport(hull,(),transport_slacks.get(id(supports[face]),Q(0))+
        (Q(0) if plane_covers is None or id(supports[face]) not in plane_covers else plane_covers[id(supports[face])].transport_upper))[0]
        for face,hull in hulls.items()}
    from ._local_graph_clearance import GraphClearance
    from .definition_binding import definition_checksum
    # Missing proxy slack is not zero-offset authority. Admit only literal
    # original tensor supports, never an effective sweep/alternate chart.
    graph_supports={face:support for face,support in supports.items()
        if type(model.faces[face].surface) is NurbsSurface
        and (model.faces[face].parameterization is None or model.faces[face].parameterization is model.faces[face].surface)
        and definition_checksum(support)==definition_checksum(model.faces[face].surface)
        and transport_slacks.get(id(support),Q(0))==0}
    graph=GraphClearance(graph_supports,budget,transport_slacks)
    contact_registry.register_graph(graph)
    faces=tuple(supports)
    sweep=None
    if sweep_unions:
        from .local_sweep_plane_contact import _SweepCompetition
        sweep=_SweepCompetition(model,sweep_unions,
            tuple(face for face in faces if type(supports[face]) is Plane) if connect_intent else (),
            lambda face,box:_hull(supports[face],box,hulls[face],budget,memo),budget,numerical_boundary)
        contact_registry.register_sweep(sweep)
    # Only an explicit preparation request may trigger fresh pair qualification.
    if connect_intent:
        for i,face in enumerate(faces):
            for other in faces[i+1:]:
                budget.take()
                if sweep is not None and face in sweep.members and other in sweep.members and sweep.members[face][0] is sweep.members[other][0]:continue
                if separation_squared(hulls[face],hulls[other])>0: continue
                if type(supports[face]) is NurbsSurface and type(supports[other]) is NurbsSurface:
                    receipt=query_local_contact_stars(model,(face,other),connect_intent=True,
                        cancellation_check=budget.take,allow_disjoint=True,_numerical_boundary=numerical_boundary)
                    if receipt is not None:
                        views[frozenset((face,other))]=contact_registry.prepare(receipt)
                        pairs[frozenset((face,other))]=receipt
                        contacts.append(receipt)
    output={};relied=set()
    for face,_,curvature in curvatures:
        queue=[]
        for tile in reversed(curvature.tiles):
            seed=(tile.chart_box,)
            # Complete spatial seeds also for flat sheets with variable gaps.
            if len(faces)>1 and budget.depth>=2:
                seed=tuple(child for box in _split(tile.chart_box) for child in _split(box))
            queue.extend((box,tile.curvature_upper,tile.radius_lower,2 if len(seed)>1 else 0) for box in reversed(seed))
        accepted=[]
        # Exact atlas boundary cuts, converted from shared native knots. These
        # proposals partition the whole source box; none authorize exclusion.
        cuts=[]
        for pair,receipt in pairs.items():
            if face not in pair:continue
            for inner_box,_,_,_,_ in views[pair].rows:
                for axis,(a,b) in enumerate((supports[face].u_range,supports[face].v_range)):
                    cuts.extend((axis,(value-Q(a))/(Q(b)-Q(a))) for value in inner_box[axis])
        while queue:
            box,curvature_upper,radius_lower,level=queue.pop();budget.take()
            inner=_hull(supports[face],box,hulls[face],budget,memo)
            ledger=[];gaps=[];unresolved=False
            for other in faces:
                if other==face: continue
                receipt=pairs.get(frozenset((face,other)))
                proposal=None
                if receipt is not None:
                    proposal=views[frozenset((face,other))].query(face,_native(supports[face],box))
                if proposal is not None:
                    gaps.append(proposal.exterior_squared_lower)
                    ledger.append((other,'conditional_contact',proposal))
                    continue
                special=None if sweep is None or separation_squared(inner,hulls[other])>0 else sweep.query(face,box,other)
                if special is not None:
                    gaps.append(special.exterior_squared_lower)
                    ledger.append((other,'sweep_plane_conditional' if special.requires_realization else 'source_sweep_continuation',special))
                    continue
                graph_gap=graph.query(face,other,box)
                if graph_gap is not None:
                    gap,pieces=graph_gap;gaps.append(gap)
                    ledger.append((other,'implicit_graph_clearance',pieces))
                    continue
                ordinary=_ordinary(inner,supports[other],hulls[other],budget,min(budget.depth,level+2),memo)
                if ordinary is None: unresolved=True;break
                gap,pieces=ordinary;gaps.append(gap);ledger.append((other,'ordinary' if type(supports[face]) is NurbsSurface else 'whole_trim_world_hull',pieces))
            if unresolved:
                if level>=budget.depth: raise LocalApproximationNotCertified('whole-tile sheet/contact clearance unresolved')
                queue.extend((child,curvature_upper,radius_lower,level+1) for child in reversed(_split(box,cuts)))
                continue
            for other,kind,_ in ledger:
                if kind=='conditional_contact': relied.add(frozenset((face,other)))
            accepted.append((box,curvature_upper,radius_lower,min(gaps) if gaps else None,tuple(ledger)))
        output[face]=tuple(accepted)
    # Only relied-upon pairs impose realization obligations. All other sources
    # remain ordinary competitors and every accepted tile has a complete ledger.
    sweep_contacts=() if sweep is None else sweep.finish()
    for receipt in sweep_contacts:contact_registry.add_sweep_receipt(receipt)
    result=output,(*tuple(r for r in contacts if frozenset(r.face_ids) in relied),*sweep_contacts),budget.used
    contact_runtime_check();contact_guard()
    return result

