"""Issued pre-cell realization of finite source contact stars.

No cell/chord-dependent classifier is called. Actual material lineage and
current topology supply a separate, strictly pre-mesh authority. This never
replaces source-cell, station, error, quality, assignment or publication gates.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from weakref import WeakKeyDictionary, ref
from .definition_binding import definition_checksum
from .local_field_transport import validate_local_field_transport
from ._local_contact_stars import validate_local_contact_stars, _state, _fail
from .nurbs_intersection_curve import NurbsSurfaceIntersectionCurve, intersection_curve_operation
from .joint_edges import query_joint_edge
from .structural import AttachmentKind, AttachmentEvidence, ConnectionIntent, ParameterRange
from .plane_nurbs import charge_box
from .predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY


@dataclass(frozen=True, eq=False)
class LocalContactTransport:
    source_model_id: object
    prepared_model_id: object
    stars: object
    material_transport: object
    # component, canonical normalized interval, edge, actual four faces
    trace_fragments: tuple


@dataclass(frozen=True, slots=True)
class LocalContactExclusion:
    source_face: int
    partner_face: int
    source_native_box: tuple
    excluded_partner_native_box: tuple
    retained_partner_native_rectangles: tuple
    exterior_squared_lower: Q
    component: int
    coordinate_system: str = 'shared_native_knots'
    # All other faces/features remain competitors without exception.


_ISSUED = WeakKeyDictionary()


def _snapshot(source,prepared,stars,transport):
    # Complete current literal/native/runtime binding; no numerical or material
    # requalification. The transport checksum includes every region/footprint.
    return _state(source),_state(prepared),definition_checksum(stars),definition_checksum(transport)


def _inputs(source,prepared,stars,transport):
    validate_local_contact_stars(source,stars)
    validate_local_field_transport(source,prepared,transport)
    return _snapshot(source,prepared,stars,transport)


def _binding_frontier(source,prepared,stars,transport,policy,cancellation_check):
    """One query lifetime: fresh gates surround immutable binding checks.

    Internal owner ticks do not re-enter full material qualification. Every
    actual external callback still owes fresh qualification on BOTH sides;
    neither unknown callables nor passive signals receive an exemption here.
    """
    initial=_inputs(source,prepared,stars,transport)
    cap=policy.max_boxes_per_pair; signature=definition_checksum(policy); count=0
    if type(cap) is not int or cap<1: _fail('invalid contact realization budget')
    def guard(fresh=False):
        current=(_inputs if fresh else _snapshot)(source,prepared,stars,transport)
        if current!=initial or definition_checksum(policy)!=signature:
            _fail('contact realization inputs or policy changed')
    def check():
        nonlocal count
        count+=1
        if count>cap: _fail('contact realization box budget exhausted')
        charge_box()
        guard()
        if cancellation_check is not None:
            guard(fresh=True)
            cancelled=cancellation_check('local contact realization')
            guard(fresh=True)
            if cancelled: _fail('contact realization cancelled')
        return False
    def finish():
        # Independent fresh authority gate, after all callbacks and branch work.
        guard(fresh=True)
    return initial,check,finish


def _sweep_fragments(source,prepared,stars,transport,check):
    """Actual four-use CONNECT per original member, across the full union."""
    from .support_curves import PlaneNurbsCurve
    from .regular_source_reference import source_node
    from .nurbs_runtime_binding import validate_section_runtime
    parents={row.prepared_face.id:row.source_face.id for row in transport.footprints}
    if len(parents)!=len(transport.footprints) or not set(stars.face_ids)<=set(parents.values()):
        _fail('sweep contact transport omits original participant material')
    selected={f for f,p in parents.items() if p in stars.face_ids}
    rows=[];stations={};components=stars.law.components
    for edge in prepared.edges.values():
        curve=edge.curve
        if type(curve) is not PlaneNurbsCurve:continue
        incident=set(prepared.faces_using_edge(edge.id))
        if not incident.intersection(selected):continue
        matches=[i for i,(parent,owner,ends) in enumerate(components)
            if curve.plane_record==owner.plane_record and curve.path==owner.path
            and definition_checksum(curve.surface)==definition_checksum(owner.surface)]
        if not matches:
            incident_parents={parents[f] for f in incident if f in parents}
            if stars.plane_face in incident_parents and incident_parents.intersection(stars.union.face_ids):
                _fail('actual sweep/plane trace has a foreign canonical component definition')
            continue  # Traces involving other source participants stay separate.
        if len(matches)!=1:_fail('sweep trace has ambiguous original component')
        check();charge_box();ci=matches[0];member,owner,source_ends=components[ci]
        local_parents={member,stars.plane_face}
        counts={f:sum(use.edge==edge.id for loop in (prepared.faces[f].loop,*prepared.faces[f].holes) for use in loop) for f in incident}
        if (edge.id in source.edges or len(incident)!=4 or any(n!=1 for n in counts.values())
                or not incident<=selected or {parents[f] for f in incident}!=local_parents
                or any(sum(parents[f]==p for f in incident)!=2 for p in local_parents)):
            _fail('sweep CONNECT needs two actual children of each local original participant')
        joint=query_joint_edge(prepared,edge.id)
        if not joint.junctions:_fail('sweep trace has no actual CONNECT junction')
        sheets={h.id for h in joint.sheets}
        sheet_faces={sid:{prepared.face_uses[u].face_id for u in prepared.sheets[sid].face_use_ids} for sid in sheets}
        if not sheets or not incident<=set().union(*sheet_faces.values()) or any(not any(
                {parents[f] for f in incident.intersection(fs)}=={parent} for fs in sheet_faces.values()) for parent in local_parents):
            _fail('sweep CONNECT does not preserve separate original sheet participants')
        allowed={identifier for h in joint.junctions for identifier in prepared.junctions[h.id].attachment_ids
            for item in (prepared.attachments[identifier],)
            if item.kind is AttachmentKind.SHEET_ON_JOINT and item.source_kind=='sheet' and item.source_id in sheets
            and item.target_kind=='edge' and item.target_id==edge.id
            and item.connection_intent is ConnectionIntent.CONNECT and item.evidence is AttachmentEvidence.EXACT
            and item.target_parameters==(ParameterRange(0.,1.),) and item.max_residual<=item.tolerance_used}
        if any(not any(prepared.attachments[a].source_id==sid for a in allowed) for sid in sheets):
            _fail('sweep CONNECT lacks exact attachment evidence for every sheet')
        if prepared.members_using_edge(edge.id) or any(
                ((item.source_kind=='edge' and item.source_id==edge.id) or (item.target_kind=='edge' and item.target_id==edge.id))
                and item.id not in allowed for item in prepared.attachments.values()):
            _fail('sweep contact cannot discard other attachment ownership')
        validate_section_runtime(curve)
        endpoints=[]
        for t,parameter,vertex in zip((Q(0),Q(1)),tuple(map(Q,curve.domain)),(edge.start,edge.end)):
            node=source_node(curve,t)
            if parameter in (Q(0),Q(1)):
                expected=source_ends[int(parameter)]
                if node.identity!=expected[1]:_fail('prepared trace endpoint is not its authentic source root')
                key=expected[0]
            else:key=('sweep_section_station',ci,node.identity)
            if key in stations and stations[key]!=vertex:_fail('same source contact station has different prepared vertex owners')
            stations[key]=vertex;endpoints.append(key)
        rows.append((ci,tuple(curve.domain),edge.id,tuple(sorted(incident)),tuple(endpoints)))
        check()
    for ci in range(len(components)):
        intervals=sorted((min(row[1]),max(row[1])) for row in rows if row[0]==ci)
        if (not intervals or intervals[0][0]!=0 or intervals[-1][1]!=1 or any(a>=b for a,b in intervals)
                or any(a[1]!=b[0] for a,b in zip(intervals,intervals[1:]))):
            _fail('sweep CONNECT has missing or duplicate canonical component fragments')
    # Every source seam root must have one actual prepared owner across both
    # original members. Equality is the owner root key, never XYZ proximity.
    for edge,first,second,point in stars.union.seams:
        keys=[key for parent,owner,ends in components for key,_ in ends if key[1]==edge]
        if len(keys)!=2 or keys[0]!=keys[1] or keys[0] not in stations:
            _fail('sweep CONNECT does not realize the original shared seam station')
    return tuple(rows)


def _fragments(source,prepared,stars,transport,check):
    from .local_sweep_plane_contact import SweepPlaneContactStars
    if type(stars) is SweepPlaneContactStars:return _sweep_fragments(source,prepared,stars,transport,check)
    parents={row.prepared_face.id:row.source_face.id for row in transport.footprints}
    if len(parents)!=len(transport.footprints) or not set(stars.face_ids)<=set(parents.values()):
        _fail('contact realization omits required source material')
    selected={f for f,p in parents.items() if p in stars.face_ids}
    rows=[]
    stations={}
    with intersection_curve_operation(lambda:check()):
        for edge in prepared.edges.values():
            curve=edge.curve
            counts={face.id:sum(use.edge==edge.id for loop in (face.loop,*face.holes) for use in loop)
                    for face in prepared.faces.values()}
            incident={f for f,n in counts.items() if n}
            if not incident.intersection(selected): continue
            if type(curve) is not NurbsSurfaceIntersectionCurve: continue
            check(); charge_box()
            matches=[i for i,owner in enumerate(stars.components) if curve.owner==owner]
            if len(matches)!=1:
                _fail('selected contact trace has an unexpected canonical branch or native source')
            if (edge.id in source.edges or len(incident)!=4 or any(counts[f]!=1 for f in incident)
                    or set(prepared.faces_using_edge(edge.id))!=incident or not incident<=selected
                    or any(sum(parents[f]==p for f in incident)!=2 for p in stars.face_ids)):
                _fail('contact realization needs four actual uses and two original parents')
            joint=query_joint_edge(prepared,edge.id)
            if not joint.junctions: _fail('contact realization requires exact current CONNECT junctions')
            sheets={h.id for h in joint.sheets}
            sheet_faces={sid:{prepared.face_uses[use].face_id for use in prepared.sheets[sid].face_use_ids}
                         for sid in sheets}
            if (not incident<=set().union(*sheet_faces.values()) or any(not any(
                    {parents[f] for f in incident.intersection(fs)}=={parent}
                    for fs in sheet_faces.values()) for parent in stars.face_ids)):
                _fail('CONNECT sheets do not separately represent both original parents')
            allowed={identifier for h in joint.junctions for identifier in prepared.junctions[h.id].attachment_ids
                for item in (prepared.attachments[identifier],)
                if item.kind is AttachmentKind.SHEET_ON_JOINT and item.source_kind=='sheet'
                and item.source_id in sheets and item.target_kind=='edge' and item.target_id==edge.id
                and item.connection_intent is ConnectionIntent.CONNECT and item.evidence is AttachmentEvidence.EXACT
                and item.target_parameters==(ParameterRange(0.,1.),) and item.max_residual<=item.tolerance_used}
            if prepared.members_using_edge(edge.id) or any(
                    ((item.source_kind=='edge' and item.source_id==edge.id) or (item.target_kind=='edge' and item.target_id==edge.id))
                    and item.id not in allowed for item in prepared.attachments.values()):
                _fail('contact realization cannot reinterpret other attachment ownership')
            curve._shape()
            for parent in stars.face_ids:
                curve.for_support(source.faces[parent].surface).chart_box(0,1)
            # Real canonical endpoint identities, not coincident XYZ samples.
            endpoints=(curve.point_identity(0),curve.point_identity(1))
            for key,vertex in zip(endpoints,(edge.start,edge.end)):
                if key in stations and stations[key]!=vertex:
                    _fail('canonical contact station has different prepared vertex owners')
                stations[key]=vertex
            rows.append((matches[0],tuple(curve.domain),edge.id,tuple(sorted(incident)),endpoints))
            check()
    for ci in range(len(stars.components)):
        intervals=sorted((min(r[1]),max(r[1])) for r in rows if r[0]==ci)
        if not intervals or intervals[0][0]!=0 or intervals[-1][1]!=1:
            _fail('contact realization misses canonical component endpoints or seam')
        if any(a>=b for a,b in intervals) or any(left[1]!=right[0] for left,right in zip(intervals,intervals[1:])):
            _fail('contact realization has duplicate or missing branch fragments')
    return tuple(rows)


def query_local_contact_transport(source, prepared, stars, transport, *, policy=None, cancellation_check=None):
    policy=policy or DEFAULT_INTERSECTION_QUALIFICATION_POLICY
    initial,check,finish=_binding_frontier(source,prepared,stars,transport,policy,cancellation_check)
    check()
    fragments=_fragments(source,prepared,stars,transport,check)
    check()
    finish()
    result=LocalContactTransport(source.model_id,prepared.model_id,stars,transport,fragments)
    _ISSUED[result]=(ref(source),ref(prepared),initial,definition_checksum(result))
    return result


def validate_local_contact_transport(source,prepared,result,*,cancellation_check=None):
    stored=_ISSUED.get(result) if type(result) is LocalContactTransport else None
    if stored is None or stored[0]() is not source or stored[1]() is not prepared:
        _fail('contact transport was not issued for these models')
    def guard():
        if definition_checksum(result)!=stored[3] or _inputs(source,prepared,result.stars,result.material_transport)!=stored[2]:
            _fail('contact transport source, topology or receipt changed')
    guard()
    if cancellation_check is not None and cancellation_check('local contact transport binding'): _fail('contact transport cancelled')
    guard()


def query_local_contact_exclusion(source,prepared,result,prepared_face,child_box,*,cancellation_check=None):
    """Return one finite exclusion only for an ENTIRE child-chart rectangle.

    child_box uses effective-native normalized UV, exactly as the authenticated
    LocalFieldFootprint.child_domain. None means no authority; retain competitors.
    Returned bounds are in common native knots, never world or descriptor t.
    """
    validate_local_contact_transport(source,prepared,result,cancellation_check=cancellation_check)
    from .local_sweep_plane_contact import SweepPlaneContactStars
    if type(result.stars) is SweepPlaneContactStars:
        _fail('projected sweep exclusions are bound through complete field partitions, not shared native coordinates')
    rows=[r for r in result.material_transport.footprints if r.prepared_face.id==prepared_face]
    if len(rows)!=1: _fail('contact lookup needs one actual prepared footprint')
    row=rows[0]
    if row.source_coordinate_system!='authored_query_uv': _fail('unknown contact source chart convention')
    box=tuple(tuple(map(Q,x)) for x in child_box)
    if len(box)!=2 or any(len(x)!=2 or not 0<=x[0]<=x[1]<=1 for x in box): _fail('invalid whole footprint box')
    if row.source_face.id not in result.stars.face_ids: return None
    corners=tuple((u,v) for u in box[0] for v in box[1])
    authored=tuple(tuple(sum(matrix[k]*uv[k] for k in range(2))+matrix[2] for uv in corners)
                   for matrix in row.source_from_child)
    support=source.faces[row.source_face.id].surface
    native=tuple((min(Q(r[0])+(Q(r[1])-Q(r[0]))*v for v in values),max(Q(r[0])+(Q(r[1])-Q(r[0]))*v for v in values))
                 for r,values in zip((support.u_range,support.v_range),authored))
    index=result.stars.face_ids.index(row.source_face.id); partner=1-index
    for star in result.stars.stars:
        if all(b[0]<=v[0]<=v[1]<=b[1] for b,v in zip(star.inner_box,native)):
            answer=LocalContactExclusion(row.source_face.id,result.stars.face_ids[partner],native,
                star.outer_box,star.exterior_rectangles[partner],star.exterior_squared_lower,star.component)
            validate_local_contact_transport(source,prepared,result)
            return answer
    validate_local_contact_transport(source,prepared,result)
    return None




