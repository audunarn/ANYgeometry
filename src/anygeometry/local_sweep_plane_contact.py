"""Finite source windows for plane/translation contact.

A window is conditional geometry evidence, NEVER prepared CONNECT authority.
The exact inverse plane and monotone sweep branches cover the full outer
window; remote profile portions remain explicit competitors.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import product
from weakref import WeakKeyDictionary,ref
from .errors import GeometryError
from .definition_binding import definition_checksum
from .surfaces import Plane
from .curves import Straight,LineRange
from .planar_exact import ExactPlane
from .plane_section_plate import plate_loops
from .prepared_material_source_transfer import _state
from ._local_effective_sweep import validate_effective_sweep_union
from ._local_curve_geometry import bound_curve_geometry
from .nurbs_surface_intersection._intervals import Interval as I
from .plane_nurbs import charge_box
from .predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY
from .support_curves import PlaneNurbsCurve
from .plane_nurbs import plane_nurbs_section
from .regular_source_reference import source_node
from .nurbs_runtime_binding import validate_section_runtime
from .prepared_material_source_transfer import _effective


class SweepPlaneContactNotCertified(GeometryError):pass


def _fail(message):raise SweepPlaneContactNotCertified(message)
def _dot(a,b):return sum((x*y for x,y in zip(a,b)),Q(0))
def _linear(row,box):return sum((x.scale(a) for a,x in zip(row,box)),I(0))
def _cross2(a,b):return a[0]*b[1]-a[1]*b[0]


@dataclass(frozen=True,slots=True)
class ConvexPlaneCover:
    face_id:int
    frame:object
    polygon:tuple
    halfspaces:tuple
    chart_box:tuple
    transport_upper:Q


@dataclass(frozen=True,eq=False)
class SweepPlaneWindow:
    source_model_id:object
    source_checksum:str
    union:object
    plane_cover:ConvexPlaneCover
    profile_interval:tuple
    projection:tuple
    inner_box:tuple
    outer_box:tuple
    projection_margin:Q
    sweep_containment_margin:Q
    plane_containment_margin:Q
    exterior_squared_lower:Q
    retained_profile_intervals:tuple
    profile_pieces:tuple
    trace_law:tuple
    coordinate_system:str='projected_world_window'
    requires_realization:bool=True


_WINDOWS=WeakKeyDictionary()


def _plane_cover(source,face):
    record=source.faces[face]
    if type(record.surface) is not Plane or record.parameterization not in (None,record.surface) or record.holes:
        _fail('contact plane requires unholed canonical planar material')
    frame,loops=plate_loops(source,record)
    polygon=loops[0]
    area=sum(_cross2(a,b) for a,b in zip(polygon,polygon[1:]+polygon[:1]))
    if not area:_fail('contact plane has degenerate material')
    sign=1 if area>0 else -1
    rows=[]
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        d=(b[0]-a[0],b[1]-a[1])
        if d==(0,0):_fail('contact plane has repeated boundary vertex')
        row=(-sign*d[1],sign*d[0]);offset=-_dot(row,a)
        if any(_dot(row,p)+offset<0 for p in polygon):_fail('contact plane material is not convex')
        rows.append((*row,offset))
    transport=Q(0)
    for use in record.loop:
        edge=source.edges[use.edge]
        if type(edge.curve) is LineRange:points=(edge.curve.exact_point(0),edge.curve.exact_point(1))
        elif type(edge.curve) is Straight:points=tuple(tuple(Q(float(x)) for x in source.vertex_position(v)) for v in (edge.start,edge.end))
        else:_fail('contact plane boundary family unsupported')
        for point in points:
            projected=frame.world(*frame.uv(point))
            transport=max(transport,sum(abs(x-y) for x,y in zip(point,projected)))
    return ConvexPlaneCover(face,frame,polygon,tuple(rows),
        tuple((min(p[k] for p in polygon),max(p[k] for p in polygon)) for k in range(2)),transport)


def _inverse_plane(cover,projection,point):
    frame=cover.frame;q,r=projection
    a,b=_dot(q,frame.u_vector),_dot(q,frame.v_vector)
    c,d=_dot(r,frame.u_vector),_dot(r,frame.v_vector)
    det=a*d-b*c
    if not det:_fail('plane projection lacks rank two')
    x,y=point[0]-_dot(q,frame.origin),point[1]-_dot(r,frame.origin)
    return ((d*x-b*y)/det,(-c*x+a*y)/det)


def _profile_ranges(union,interval):
    a,b=interval
    if not a<b or b-a>=1:_fail('finite profile window must be shorter than one traversal')
    if not union.closed and not 0<=a<b<=1:_fail('open profile window leaves material')
    out=[];cursor=a
    while cursor<b:
        period=cursor.numerator//cursor.denominator
        end=min(b,Q(period+1))
        out.append((cursor,end,cursor-period,end-period))
        cursor=end
    return tuple(out)


def _controlled_work(source,callback,boundary):
    """Only the established owner's budget tick may use detached operation scope."""
    if boundary is None:return False
    from ._detached_geometry_read import NumericalBoundary
    from ._local_spatial_clearance import SpatialBudget
    if type(boundary) is not NumericalBoundary or boundary.source is not source:
        _fail('unknown sweep numerical boundary')
    budget=getattr(callback,'__self__',None)
    check=getattr(budget,'check',None)
    controlled=(type(budget) is SpatialBudget and getattr(callback,'__func__',None) is SpatialBudget.take
        and getattr(check,'__self__',None) is boundary and getattr(check,'__func__',None) is NumericalBoundary.tick
        and boundary._detached)
    boundary.guard()
    return controlled


def _calculate_window(numeric,cover,profile_interval,projection,inner_box,outer_box,check,curve_bounds=None):
    """Pure exact window operands; no source receipt or exclusion is issued."""
    interval=tuple(map(Q,profile_interval));rows=tuple(tuple(map(Q,row)) for row in projection)
    inner=tuple(tuple(map(Q,p)) for p in inner_box);outer=tuple(tuple(map(Q,p)) for p in outer_box)
    if len(rows)!=2 or any(len(row)!=3 for row in rows) or len(inner)!=2 or len(outer)!=2:_fail('invalid projection window shape')
    if any(len(p)!=2 for p in (*inner,*outer)):_fail('invalid projection window shape')
    if any(not c<a<=b<d for (a,b),(c,d) in zip(inner,outer)):_fail('inner window needs strict outer buffer')
    q,r=rows;direction=numeric.translation
    if _dot(q,direction)!=0 or _dot(r,direction)!=1:_fail('projection does not preserve sweep coordinates')
    normal=cover.frame.normal;beta=_dot(normal,direction)
    if beta==0:_fail('parallel plane/sweep graph requires a different theorem')
    pieces=[];margins=[];hulls=[]
    for start,end,lo,hi in _profile_ranges(numeric,interval):
        bound=bound_curve_geometry(numeric.profile,lo,hi,cancellation_check=check) if curve_bounds is None else curve_bounds(lo,hi)
        for piece in bound.pieces:
            tangent=tuple(I(*p) for p in piece.tangent)
            margin=_linear(q,tangent).lo
            if margin<=0:_fail('piecewise profile projection is not strictly monotone')
            margins.append(margin)
            hulls.append(tuple(I(*p) for p in piece.world_box))
            pieces.append((start,end,piece))
    # Every range endpoint is an exact point of the common profile. Periodic
    # copies are permitted only after the union's exact shared seam binding.
    a,b=interval
    def point(t):
        k=t.numerator//t.denominator
        local=t-k
        if not numeric.closed:local=t
        return numeric.profile.exact_point(local)
    pa,pb=point(a),point(b)
    slack=numeric.transport_upper+cover.transport_upper
    projected_slack=tuple(sum(abs(x) for x in row)*slack for row in rows)
    if not _dot(q,pa)+projected_slack[0]<outer[0][0]<outer[0][1]<_dot(q,pb)-projected_slack[0]:
        _fail('outer window has no complete local profile inverse')
    h=I(min(_linear(r,box).lo for box in hulls),max(_linear(r,box).hi for box in hulls))
    sweep_margin=min(outer[1][0]-h.hi,1-(outer[1][1]-h.lo))-projected_slack[1]
    if sweep_margin<=0:_fail('outer window leaves swept material')
    # Exact inverse of the plane projection; convexity makes the four outer
    # corners sufficient for EVERY point, not a sampling-based containment.
    corners=tuple(_inverse_plane(cover,rows,p) for p in product(*outer))
    # Physical offset allowance is converted to UV with exact dual rows.
    uv_slack=tuple(sum(abs(x) for x in row)/cover.frame._scale*slack
                   for row in (cover.frame._dual_u,cover.frame._dual_v))
    plane_margin=min(min(x*u+y*v+c for u,v in corners)-abs(x)*uv_slack[0]-abs(y)*uv_slack[1]
                     for x,y,c in cover.halfspaces)
    if plane_margin<=0:_fail('outer window leaves true plane material')
    delta=min(min(a-c,d-b)-s for (a,b),(c,d),s in zip(inner,outer,projected_slack))
    if delta<=0:_fail('transport consumes the finite window buffer')
    norm=sum(x*x for row in rows for x in row)
    # Retain ALL remote same-profile intervals, including returning branches.
    covered=[]
    for _,_,lo,hi in _profile_ranges(numeric,interval):covered.append((lo,hi))
    covered.sort();remote=[];cursor=Q(0)
    for lo,hi in covered:
        if cursor<lo:remote.append((cursor,lo))
        cursor=max(cursor,hi)
    if cursor<1:remote.append((cursor,Q(1)))
    # Exact rational affine world trace law X -> X+D*(c-n.X)/beta.
    matrix=tuple(tuple(Q(i==j)-direction[i]*normal[j]/beta for j in range(3)) for i in range(3))
    offset=tuple(d*cover.frame.offset()/beta for d in direction)
    return (cover,interval,rows,inner,outer,min(margins),sweep_margin,plane_margin,
        delta*delta/norm,tuple(remote),tuple(pieces),(matrix,offset))


def certify_sweep_plane_window(source,union,plane_face,profile_interval,projection,inner_box,outer_box,*,cancellation_check=None,_numerical_boundary=None):
    """Prove one supplied finite window; no source-field exclusion is issued.

    Proposals may be chosen numerically, but all acceptance inequalities below
    use exact complete interval enclosures. No whole closed profile is assumed
    monotone, and no artificial seam is treated as an exterior boundary.
    """
    controlled=_controlled_work(source,cancellation_check,_numerical_boundary)
    validate_effective_sweep_union(source,union,cancellation_check=cancellation_check)
    initial=_state(source);runtime=_runtime();union_pin=definition_checksum(union)
    if runtime!=_INITIAL_RUNTIME:_fail('contact window runtime context unsupported')
    used=0
    cap=DEFAULT_INTERSECTION_QUALIFICATION_POLICY.max_boxes_per_pair
    def guard():
        if _state(source)!=initial or definition_checksum(union)!=union_pin or _runtime()!=runtime:_fail('contact source changed')
    def check(phase='sweep plane finite window'):
        nonlocal used
        used+=1
        if used>cap:_fail('contact window work budget exhausted')
        charge_box()
        if not controlled:guard()
        if cancellation_check is not None:
            cancelled=cancellation_check(phase)
            if not controlled:guard()
            if cancelled:_fail('contact window cancelled')
        return False
    check()
    numeric=_numeric_sweep(union)
    values=_calculate_window(numeric,_plane_cover(source,plane_face),profile_interval,
        projection,inner_box,outer_box,check)
    check()
    if controlled:_numerical_boundary.guard()
    validate_effective_sweep_union(source,union)
    guard()  # No callback after the final current source/native/runtime gate.
    result=SweepPlaneWindow(source.model_id,initial[1],union,*values)
    _WINDOWS[result]=(ref(source),initial,definition_checksum(result),runtime)
    return result


def _window_binding(source,result,state=None,runtime=None):
    entry=_WINDOWS.get(result) if type(result) is SweepPlaneWindow else None
    if entry is None or entry[0]() is not source:_fail('contact window not issued for source')
    if ((state if state is not None else _state(source))!=entry[1] or definition_checksum(result)!=entry[2]
            or (runtime if runtime is not None else _runtime())!=entry[3]):_fail('contact window stale or changed')


def validate_sweep_plane_window(source,result,*,cancellation_check=None):
    _window_binding(source,result)
    validate_effective_sweep_union(source,result.union,cancellation_check=cancellation_check)
    _window_binding(source,result)


@dataclass(frozen=True,eq=False)
class SweepPlaneSectionLaw:
    source_model_id:object
    source_checksum:str
    union:object
    plane_face:int
    # Original member, canonical implicit component, descriptor endpoint keys.
    components:tuple
    trace_law:tuple
    profile_station_margin:Q
    requires_realization:bool=True


_LAWS=WeakKeyDictionary()


def query_sweep_plane_section_law(source,union,plane_face,*,cancellation_check=None,_numerical_boundary=None):
    """Complete original-member section correspondence, not prepared ownership.

    The nonzero-beta law proves a unique graph over each complete profile
    interval. Actual canonical owner paths independently supply their implicit
    descriptor domains; descriptor t is NEVER identified with profile u.
    """
    controlled=_controlled_work(source,cancellation_check,_numerical_boundary)
    validate_effective_sweep_union(source,union,cancellation_check=cancellation_check)
    state=_state(source);pin=definition_checksum(union);runtime=_runtime()
    if runtime!=_INITIAL_RUNTIME:_fail('contact section runtime unsupported')
    count=0;cap=DEFAULT_INTERSECTION_QUALIFICATION_POLICY.max_boxes_per_pair
    def guard():
        if _state(source)!=state or definition_checksum(union)!=pin or _runtime()!=runtime:_fail('contact section source changed')
    def check():
        nonlocal count
        count+=1;charge_box()
        if count>cap:_fail('contact section work budget exhausted')
        if not controlled:guard()
        if cancellation_check is not None:
            cancelled=cancellation_check('sweep plane section correspondence')
            if not controlled:guard()
            if cancelled:_fail('contact section cancelled')
    check();cover=_plane_cover(source,plane_face)
    from ._detached_geometry_read import detach_numeric
    numeric=_numeric_sweep(union)
    plane=detach_numeric(source.faces[plane_face].surface)
    members=tuple((m.face_id,tuple(m.connectors),detach_numeric(_effective(source,m.face_id))) for m in union.members)
    seams=tuple(tuple(row) for row in union.seams)
    guard()
    normal=cover.frame.normal;offset=cover.frame.offset();d=numeric.translation;beta=_dot(normal,d)
    if not beta:_fail('parallel sweep/plane section law unsupported')
    # A positive rational profile lies in its control hull. This sufficient
    # global guard establishes one interior section graph for EVERY parameter.
    weights=numeric.profile.weights or (1,)*len(numeric.profile.controls)
    if any(Q(w)<=0 for w in weights):_fail('section profile needs positive weights')
    vcontrols=tuple((offset-_dot(normal,tuple(map(Q,p))))/beta for p in numeric.profile.controls)
    margin=min(min(vcontrols),1-max(vcontrols))
    if margin<=0:_fail('complete section graph is not certified inside sweep height')
    matrix=tuple(tuple(Q(i==j)-d[i]*normal[j]/beta for j in range(3)) for i in range(3))
    shift=tuple(x*offset/beta for x in d)
    components=[];seam_vertices={}
    for face_id,connectors,support in members:
        check()
        section=plane_nurbs_section(plane,support)
        if section.coincident or len(section.paths)!=1 or section.paths[0].closed:
            _fail('member section is not one complete open canonical graph')
        curve=PlaneNurbsCurve.from_plane(plane,support,0)
        validate_section_runtime(curve)
        nodes=tuple(source_node(curve,t) for t in (Q(0),Q(1)))
        ends=[]
        for node in nodes:
            if node.box[0] not in ((Q(0),Q(0)),(Q(1),Q(1))):
                _fail('canonical component endpoint is not a source profile endpoint')
            u=node.box[0][0];point=support.exact_point(u,Q(0))
            v=(offset-_dot(normal,point))/beta
            if not node.box[1][0]<=v<=node.box[1][1]:_fail('canonical root does not contain exact graph endpoint')
            edge=connectors[int(u)]
            key=('sweep_plane_connector_root',edge,point,d,normal,offset,v)
            previous=seam_vertices.get(edge)
            if previous is not None and previous!=key:_fail('source seam has inconsistent exact section root')
            seam_vertices[edge]=key;ends.append((key,node.identity))
        if nodes[0].box[0]==nodes[1].box[0]:_fail('canonical component does not span complete profile interval')
        components.append((face_id,curve,tuple(ends)))
        check()
    for edge,first,second,_ in seams:
        occurrences=[key for face,curve,ends in components for key,_ in ends if key[1]==edge]
        if len(occurrences)!=2 or occurrences[0]!=occurrences[1]:_fail('source section omits an actual shared seam root')
    check()
    if controlled:_numerical_boundary.guard()
    validate_effective_sweep_union(source,union)
    for _,curve,_ in components:validate_section_runtime(curve)
    guard()
    result=SweepPlaneSectionLaw(source.model_id,state[1],union,plane_face,tuple(components),(matrix,shift),margin)
    _LAWS[result]=(ref(source),state,definition_checksum(result),runtime)
    return result


def _law_binding(source,result,state=None,runtime=None):
    stored=_LAWS.get(result) if type(result) is SweepPlaneSectionLaw else None
    if stored is None or stored[0]() is not source:_fail('section law was not issued for source')
    if ((state if state is not None else _state(source))!=stored[1] or definition_checksum(result)!=stored[2]
            or (runtime if runtime is not None else _runtime())!=stored[3]):_fail('section law stale or changed')
    for _,curve,_ in result.components:validate_section_runtime(curve)


def validate_sweep_plane_section_law(source,result,*,cancellation_check=None):
    _law_binding(source,result)
    validate_effective_sweep_union(source,result.union,cancellation_check=cancellation_check)
    _law_binding(source,result)


@dataclass(frozen=True,slots=True)
class _NumericalSweep:
    profile:object
    translation:tuple
    closed:bool
    transport_upper:Q
    face_ids:tuple


@dataclass(frozen=True,slots=True)
class _NumericalMember:
    profile_interval:tuple


def _numeric_sweep(union):
    from ._detached_geometry_read import detach_numeric
    return _NumericalSweep(detach_numeric(union.profile),tuple(Q(x) for x in union.translation),
        bool(union.closed),Q(union.transport_upper),tuple(union.face_ids))


@dataclass(frozen=True,slots=True)
class SweepCompetitorPartition:
    source_face:int
    partner_face:int
    source_box:tuple
    # Full target chart cover: each leaf is retained or conditionally local.
    retained:tuple
    excluded:tuple
    exterior_squared_lower:Q
    window:object
    requires_realization:bool


@dataclass(frozen=True,slots=True)
class SweepContinuationWindow:
    profile_interval:tuple
    projection:tuple
    projection_margin:Q
    exterior_squared_lower:Q


@dataclass(frozen=True,eq=False)
class SweepPlaneContactStars:
    source_model_id:object
    source_checksum:str
    union:object
    plane_face:int
    law:SweepPlaneSectionLaw
    partitions:tuple
    intent:str='CONNECT'
    coordinate_system:str='projected_world_window'
    @property
    def face_ids(self):return (*self.union.face_ids,self.plane_face)


_STARS=WeakKeyDictionary()


def validate_sweep_plane_contact_stars(source,result,*,cancellation_check=None):
    entry=_STARS.get(result) if type(result) is SweepPlaneContactStars else None
    if entry is None or entry[0]() is not source:_fail('sweep contact aggregate not issued for source')
    def guard():
        if _state(source)!=entry[1] or definition_checksum(result)!=entry[2] or _runtime()!=entry[3]:_fail('sweep contact aggregate changed')
    guard();validate_sweep_plane_section_law(source,result.law,cancellation_check=cancellation_check)
    state=_state(source);runtime=_runtime()
    for row in result.partitions:_window_binding(source,row.window,state,runtime)
    guard()


def _in_profile(interval,outer,closed):
    shifts=(-1,0,1) if closed else (0,)
    return any(outer[0]<=interval[0]+k<=interval[1]+k<=outer[1] for k in shifts)


def _profile_interval(union,member,box):
    a,b=map(Q,union.profile.domain);u,v=member.profile_interval
    return tuple(sorted(((u+(v-u)*t-a)/(b-a)) for t in box[0]))


def _profile_point(union,t):
    if union.closed:t-=t.numerator//t.denominator
    return union.profile.exact_point(t)


def _profile_data(union,interval,check):
    pieces=[]
    for start,end,lo,hi in _profile_ranges(union,interval):
        result=bound_curve_geometry(union.profile,lo,hi,cancellation_check=check)
        pieces.extend(result.pieces)
    return tuple(pieces)


def _projection_candidate(union,interval):
    p,q=(_profile_point(union,t) for t in interval);d=union.translation
    chord=tuple(b-a for a,b in zip(p,q));dd=_dot(d,d)
    return tuple(x-dk*_dot(chord,d)/dd for x,dk in zip(chord,d))


class _SweepCompetition:
    """One field operation, complete target partitions; no receipt cache."""
    def __init__(self,source,unions,plane_faces,hull,budget,numerical_boundary=None):
        self.source=source;self.unions=tuple(unions);self.planes=tuple(plane_faces)
        self.numerical_boundary=numerical_boundary
        self.controlled=_controlled_work(source,budget.take,numerical_boundary)
        self.hull=hull;self.budget=budget;self.rows={};self.laws={};self.pending=[]
        self.window_pins={};self.curve_bounds={};self.curve_pins={};self.initial_state=_state(source)
        self.partition_pins=[]
        self.covers={face:_plane_cover(source,face) for face in plane_faces}
        self.numeric={id(u):_numeric_sweep(u) for u in unions}
        self.members={m.face_id:(u,_NumericalMember(tuple(Q(x) for x in m.profile_interval))) for u in unions for m in u.members}
        self.numeric_pin=definition_checksum((self.numeric,self.covers,self.members))
        self.runtime=_runtime()
        if self.runtime!=_INITIAL_RUNTIME:_fail('sweep competition runtime unsupported')
        self.pins=tuple((u,definition_checksum(u)) for u in unions)

    def guard(self):
        if _runtime()!=self.runtime:_fail('sweep competition runtime changed')
        if definition_checksum((self.numeric,self.covers,self.members))!=self.numeric_pin:_fail('detached sweep input changed')
        for union,pin in self.pins:
            if definition_checksum(union)!=pin:_fail('sweep competition union changed')
            validate_effective_sweep_union(self.source,union)
        state=_state(self.source);runtime=_runtime()
        for law in self.laws.values():_law_binding(self.source,law,state,runtime)
        if state!=self.initial_state:_fail('sweep operation source changed')
        for window in self.pending:
            if definition_checksum(window)!=self.window_pins[id(window)]:_fail('unpublished sweep window changed')
        for key,bound in self.curve_bounds.items():
            if definition_checksum(bound)!=self.curve_pins[key]:_fail('detached sweep curve operand changed')
        expected={}
        for row,pin,key in self.partition_pins:
            if definition_checksum(row)!=pin:_fail('completed sweep partition changed')
            if key is not None:expected.setdefault(key,[]).append(row)
        if (type(self.rows) is not dict or set(self.rows)!=set(expected) or any(
                type(self.rows[key]) is not list or tuple(map(id,self.rows[key]))!=tuple(map(id,rows))
                for key,rows in expected.items())):
            _fail('completed sweep partition inventory changed')

    def _continuation(self,face,box,other):
        owner,member=self.members[face];union=self.numeric[id(owner)]
        if other not in union.face_ids:return None
        inner=_profile_interval(union,member,box);width=inner[1]-inner[0]
        if width<=0:return None
        outer=(inner[0]-width,inner[1]+width)
        if not union.closed:outer=(max(Q(0),outer[0]),min(Q(1),outer[1]))
        if outer[1]-outer[0]>=1:return None
        q=_projection_candidate(union,outer)
        if not any(q):return None
        pieces=_profile_data(union,outer,self.budget.take)
        margin=min(_linear(q,tuple(I(*p) for p in row.tangent)).lo for row in pieces)
        if margin<=0:return None
        # Only true original domain ends may lack an outer parameter buffer.
        gaps=[]
        if inner[0]>outer[0]:gaps.append(_dot(q,_profile_point(union,inner[0]))-_dot(q,_profile_point(union,outer[0])))
        elif union.closed or outer[0]!=0:return None
        if inner[1]<outer[1]:gaps.append(_dot(q,_profile_point(union,outer[1]))-_dot(q,_profile_point(union,inner[1])))
        elif union.closed or outer[1]!=1:return None
        if not gaps:return None
        gap=min(gaps)-2*sum(abs(x) for x in q)*union.transport_upper
        if gap<=0:return None
        return SweepContinuationWindow(outer,(q,),margin,gap*gap/_dot(q,q))

    def _window(self,face,box,other,inner_hull):
        if face in self.members and other in self.planes:
            owner,member=self.members[face];plane=other;seed=_profile_interval(self.numeric[id(owner)],member,box)
        elif face in self.planes and other in self.members:
            owner,member=self.members[other];plane=face;seed=_profile_interval(self.numeric[id(owner)],member,((Q(0),Q(1)),(Q(0),Q(1))))
        else:return None
        key=(id(owner),plane)
        if key not in self.laws:self.laws[key]=query_sweep_plane_section_law(self.source,owner,plane,cancellation_check=self.budget.take,_numerical_boundary=self.numerical_boundary)
        union=self.numeric[id(owner)]
        cover=self.covers[plane];d=union.translation;dd=_dot(d,d)
        # Finite dyadic window proposals; all acceptance is whole-interval exact.
        for level in range(1,5):
            count=2**level;step=(seed[1]-seed[0])/count
            for index in range(count):
                self.budget.take()
                center=seed[0]+(Q(index)+Q(1,2))*step
                outer=(center-2*step,center+2*step)
                if not union.closed:outer=(max(Q(0),outer[0]),min(Q(1),outer[1]))
                if outer[1]-outer[0]>=1 or outer[0]>=outer[1]:continue
                if face in self.members and not _in_profile(seed,outer,union.closed):continue
                q=_projection_candidate(union,outer)
                if not any(q):continue
                pieces=_profile_data(union,outer,self.budget.take)
                if min(_linear(q,tuple(I(*p) for p in row.tangent)).lo for row in pieces)<=0:continue
                for axis in range(3):
                    for power in range(5):
                        self.budget.take()
                        r=tuple(x/dd+(Q(k==axis)-d[k]*d[axis]/dd)/2**power for k,x in enumerate(d))
                        projection=(q,r)
                        # Exact rank only: no condition-number admission threshold.
                        a,b=(_dot(q,v) for v in (cover.frame.u_vector,cover.frame.v_vector))
                        c,f=(_dot(r,v) for v in (cover.frame.u_vector,cover.frame.v_vector))
                        if a*f-b*c==0:continue
                        inner=tuple((v.lo,v.hi) for v in (_linear(row,inner_hull) for row in projection))
                        g0,g1=(_dot(q,_profile_point(union,t)) for t in outer)
                        h=[_linear(r,tuple(I(*p) for p in row.world_box)) for row in pieces]
                        slack=union.transport_upper+cover.transport_upper
                        sx,sy=(sum(abs(v) for v in row)*slack for row in projection)
                        margins=[inner[0][0]-g0-sx,g1-inner[0][1]-sx,
                            inner[1][0]-max(v.hi for v in h)-sy,1+min(v.lo for v in h)-inner[1][1]-sy]
                        if min(margins)<=0:continue
                        corners=tuple(_inverse_plane(cover,projection,p) for p in product(*inner))
                        origin=_inverse_plane(cover,projection,(Q(0),Q(0)))
                        units=tuple(tuple(a-b for a,b in zip(_inverse_plane(cover,projection,p),origin)) for p in ((Q(1),Q(0)),(Q(0),Q(1))))
                        uvslack=tuple(sum(abs(x) for x in row)/cover.frame._scale*slack for row in (cover.frame._dual_u,cover.frame._dual_v))
                        for x,y,z in cover.halfspaces:
                            value=min(x*u+y*v+z for u,v in corners)-abs(x)*uvslack[0]-abs(y)*uvslack[1]
                            rate=sum(abs(x*u+y*v) for u,v in units)
                            if rate:margins.append(value/rate)
                            elif value<=0:margins.append(Q(0))
                        padding=min(margins)/4
                        if padding<=max(sx,sy):continue
                        projected_outer=tuple((lo-padding,hi+padding) for lo,hi in inner)
                        return self._make_window(owner,plane,outer,projection,inner,projected_outer)
        return None

    def _make_window(self,owner,plane,outer,projection,inner,projected_outer):
        # Registered operation only: unpublished pure numeric proposals.
        if not self.controlled:
            result=certify_sweep_plane_window(self.source,owner,plane,outer,projection,inner,projected_outer,
                cancellation_check=self.budget.take)
            self.window_pins[id(result)]=definition_checksum(result);self.pending.append(result)
            return result
        union=self.numeric[id(owner)];cover=self.covers[plane]
        def curve_bounds(lo,hi):
            self.budget.take()
            key=(id(owner),lo,hi)
            if key not in self.curve_bounds:
                bound=bound_curve_geometry(union.profile,lo,hi,cancellation_check=self.budget.take)
                self.curve_bounds[key]=bound;self.curve_pins[key]=definition_checksum(bound)
            return self.curve_bounds[key]
        values=_calculate_window(union,cover,outer,projection,inner,projected_outer,self.budget.take,curve_bounds)
        self.budget.take()
        result=SweepPlaneWindow(self.source.model_id,self.initial_state[1],owner,*values)
        self.window_pins[id(result)]=definition_checksum(result)
        self.pending.append(result)
        return result

    def query(self,face,box,other):
        self.budget.take();inner=self.hull(face,box)
        same=(face in self.members and other in self.members and self.members[face][0] is self.members[other][0])
        window=self._continuation(face,box,other) if same else self._window(face,box,other,inner)
        if window is None:return None
        profile_interval=tuple(window.profile_interval)
        projection=tuple(tuple(Q(x) for x in row) for row in window.projection)
        outer_box=() if same else tuple(tuple(Q(x) for x in pair) for pair in window.outer_box)
        window_lower=Q(window.exterior_squared_lower)
        owner=self.members[face if face in self.members else other][0];union=self.numeric[id(owner)]
        from ._local_neighborhood import separation_squared
        from ._local_spatial_clearance import _split
        queue=[(((Q(0),Q(1)),(Q(0),Q(1))),0)];retained=[];excluded=[]
        while queue:
            target,depth=queue.pop();self.budget.take();hull=self.hull(other,target)
            gap=separation_squared(inner,hull)
            if gap>0:
                retained.append((target,gap));continue
            local=True
            if other in self.members:
                local=_in_profile(_profile_interval(union,self.members[other][1],target),profile_interval,union.closed)
            if local and not same:
                local=all(a<=v.lo<=v.hi<=b for (a,b),v in zip(outer_box,(_linear(row,hull) for row in projection)))
            if local:excluded.append(target);continue
            if depth>=self.budget.depth:return None
            queue.extend((child,depth+1) for child in _split(target))
        if not excluded:return None
        lower=min([window_lower]+[gap for _,gap in retained])
        row=SweepCompetitorPartition(face,other,box,tuple(retained),tuple(excluded),lower,window,not same)
        key=None if same else (id(owner),other if other in self.planes else face)
        # Pin every completed operand before return/next callback, including
        # continuation rows consumed directly by the field rather than stars.
        self.partition_pins.append((row,definition_checksum(row),key))
        if key is not None:self.rows.setdefault(key,[]).append(row)
        return row

    def finish(self):
        self.guard()
        state=_state(self.source);runtime=_runtime();prepared=[]
        for (uid,plane),rows in self.rows.items():
            union=next(u for u in self.unions if id(u)==uid);law=self.laws[(uid,plane)]
            result=SweepPlaneContactStars(self.source.model_id,state[1],union,plane,law,tuple(rows))
            prepared.append((result,definition_checksum(result)))
        # All source/native/runtime/operand checks precede ALL registry writes.
        # No callbacks or numerical work occur after this final guard.
        self.guard()
        for window in self.pending:
            _WINDOWS[window]=(ref(self.source),state,self.window_pins[id(window)],runtime)
        for result,pin in prepared:_STARS[result]=(ref(self.source),state,pin,runtime)
        return tuple(result for result,pin in prepared)


def _runtime_literal(value):
    try:return definition_checksum(value)
    except (TypeError,ValueError,OverflowError) as error:
        raise SweepPlaneContactNotCertified('unsupported sweep runtime defaults') from error


def _runtime():
    from . import _local_curve_geometry as jets,plane_section_plate as plates
    from ._detached_geometry_read import NumericalBoundary
    from ._local_spatial_clearance import SpatialBudget
    functions=(_calculate_window,_runtime_literal,_controlled_work,_window_binding,_law_binding,NumericalBoundary.tick,NumericalBoundary.guard,
        SpatialBudget.take,plate_loops,plates.plate_loops,ExactPlane.from_plane.__func__,ExactPlane.__init__,
        ExactPlane.__post_init__,ExactPlane.uv,ExactPlane.world,ExactPlane.offset,
        bound_curve_geometry,jets.bound_curve_geometry,jets._curve_jet,jets._rational,jets._divide,
        jets._exact_refine,ConvexPlaneCover.__init__,SweepPlaneWindow.__init__,
        _plane_cover,_inverse_plane,_profile_ranges,_linear,_dot,_cross2,I.__init__,I.scale,
        I.__add__,I.__sub__,I.__mul__,SweepPlaneSectionLaw.__init__,
        plane_nurbs_section,source_node,validate_section_runtime,PlaneNurbsCurve.__init__,
        PlaneNurbsCurve.__post_init__,PlaneNurbsCurve.from_plane.__func__,PlaneNurbsCurve._arcs,_effective,
        SweepCompetitorPartition.__init__,SweepContinuationWindow.__init__,SweepPlaneContactStars.__init__,SweepPlaneContactStars.face_ids.fget,
        _numeric_sweep,_NumericalSweep.__init__,_NumericalMember.__init__,
        _in_profile,_profile_interval,_profile_point,_profile_data,_projection_candidate,
        _SweepCompetition.__init__,_SweepCompetition.guard,_SweepCompetition._continuation,_SweepCompetition._window,
        _SweepCompetition._make_window,_SweepCompetition.query,_SweepCompetition.finish)
    return tuple((f,getattr(f,'__code__',None),_runtime_literal(getattr(f,'__defaults__',None)),
        _runtime_literal(getattr(f,'__kwdefaults__',None))) for f in functions)


_INITIAL_RUNTIME=_runtime()
