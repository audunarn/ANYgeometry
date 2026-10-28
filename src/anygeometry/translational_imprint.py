"""Bounded full-ring plane imprint for declared axis-aligned NURBS sweeps.

This operation is not a general Coons intersection capability. Unsupported,
partial, tangent and uncertain cases refuse before publication.
"""
from dataclasses import dataclass, replace
from fractions import Fraction as Q
import numpy as np
from weakref import WeakKeyDictionary
from .curves import Straight
from .nurbs import NurbsCurve

_RECEIPTS = WeakKeyDictionary()
from .errors import GeometryError
from .entities import OrientedEdge
from .surfaces import Plane
from ._translational_pairs import _sweep


@dataclass(frozen=True)
class TranslationalRingImprint:
    source_revision: int
    working_revision: int
    plane_parent: int
    plane_children: tuple[int, int]
    wall_children: tuple[tuple[int, tuple[int, int]], ...]
    trace_edges: tuple[int, ...]


def _imprint_translational_ring_plane(model, wall_faces, plane_face):
    """Atomically partition a complete X/Y/Z sweep ring and a convex plate.

    The plate retains both inside and outside material. Shared trace topology
    is constructed once from declared connector identities, never proximity.
    """
    from .intersections import _fragment_with_edge_chain, _fragment_with_closed_chain
    from .sketch_cut import validate_sketch_cut
    walls=tuple(dict.fromkeys(map(int,wall_faces)))
    flat=model.faces[int(plane_face)]
    if set(model.faces) != set(walls)|{int(plane_face)} or model.members:
        raise GeometryError('ring imprint requires the complete face closure without extra members')
    if {u.edge for face in model.faces.values() for loop in (face.loop,*face.holes) for u in loop} != set(model.edges):
        raise GeometryError('ring imprint closure contains an unhandled edge')
    if (not walls or flat.holes
            or any(type(model.edges[u.edge].curve) is not Straight for u in flat.loop) or not isinstance(flat.surface,Plane)
            or any(item.edge in {u.edge for f in walls for u in model.faces[f].loop}
                   for item in flat.loop)):
        raise GeometryError('ring imprint requires an independent hole-free planar plate')
    plane=flat.surface
    vertices=np.asarray([model.vertex_position(model.oriented_start_vertex(u)) for u in flat.loop])
    normal=np.asarray(plane.normal); origin=np.asarray(plane.origin)
    scale=max(float(np.linalg.norm(np.ptp(vertices,axis=0))),1.)
    tolerance=model.tolerance.effective_surface_residual(scale)
    # Require a strictly convex authored polygon, with each inward edge halfspace.
    halfspaces=[]
    centre=vertices.mean(axis=0)
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        inward=np.cross(normal,b-a);length=float(np.linalg.norm(inward))
        if length==0.:raise GeometryError('degenerate plane boundary')
        inward/=length
        if float((centre-a)@inward)<0.:inward=-inward
        if np.any((vertices-a)@inward < -tolerance):
            raise GeometryError('ring imprint plane must be convex')
        halfspaces.append((a,inward))
    plans=[];connector_counts={};vector=None;axis=None
    for face in walls:
        sweep=_sweep(model,face)
        if sweep is None:raise GeometryError('face is not a qualified translational NURBS sweep')
        curve,exact_vector,connectors=sweep
        d=np.asarray(exact_vector,dtype=float)
        active=np.flatnonzero(d)
        if len(active)!=1:raise GeometryError('ring imprint currently requires an axis-aligned sweep')
        if vector is None:vector=d;axis=int(active[0])
        if not np.array_equal(d,vector):raise GeometryError('ring sweep vectors disagree')
        denominator=float(normal@d)
        if abs(denominator)<=model.tolerance.angular*float(np.linalg.norm(d)):
            raise GeometryError('ring imprint plane is tangent to the sweep')
        controls=np.asarray(curve.controls)
        if len(set(controls[:,axis])) != 1:
            raise GeometryError('ring directrix must lie in the transverse coordinate plane')
        heights=(float(normal@origin)-controls@normal)/denominator
        qnormal=tuple(Q(float(x)) for x in normal)
        qorigin=tuple(Q(float(x)) for x in origin)
        qd=tuple(Q(float(x)) for x in d)
        qden=sum(x*y for x,y in zip(qnormal,qd))
        qoffset=sum(x*y for x,y in zip(qnormal,qorigin))
        def inside_span(lo,hi,depth=0):
            hull=curve.exact_hull(lo,hi)
            values=[(qoffset-sum(x*y for x,y in zip(qnormal,p)))/qden for p in hull]
            if min(values)>0 and max(values)<1:return True
            if depth==12:return False
            middle=(lo+hi)/2
            return inside_span(lo,middle,depth+1) and inside_span(middle,hi,depth+1)
        if not inside_span(Q(0),Q(1)):
            raise GeometryError('plane does not cross the complete sweep span')
        made=controls.copy();made[:,axis]+=heights*d[axis]
        residual=max(abs(sum(n*(Q(float(x))-o) for n,x,o in zip(qnormal,p,qorigin))) for p in made)
        if residual>Q(float(tolerance)):
            raise GeometryError('plane section control reconstruction exceeds geometry tolerance')
        section=replace(curve,controls=tuple(map(tuple,made)))
        # Qualify the ACTUAL rounded controls, not just ideal affine heights.
        base=Q(float(controls[0,axis]));rate=Q(float(d[axis]))
        clearance=[None]
        def actual_span(lo,hi,depth=0):
            values=[(p[axis]-base)/rate for p in section.exact_hull(lo,hi)]
            lower=min(values);upper=max(values)
            if lower>0 and upper<1:
                gap=min(lower,1-upper)*abs(rate)
                clearance[0]=gap if clearance[0] is None else min(clearance[0],gap)
                return True
            if depth==12:return False
            middle=(lo+hi)/2
            return actual_span(lo,middle,depth+1) and actual_span(middle,hi,depth+1)
        if not actual_span(Q(0),Q(1)):
            raise GeometryError('rounded section does not stay within the complete sweep span')
        def inside_plate(lo,hi,depth=0):
            hull=section.exact_hull(lo,hi)
            okay=True
            for a,inward in halfspaces:
                qa=tuple(Q(float(x)) for x in a);qn=tuple(Q(float(x)) for x in inward)
                values=[sum(n*(x-y) for n,x,y in zip(qn,p,qa)) for p in hull]
                if min(values)<=Q(float(tolerance)):okay=False;break
            if okay:return True
            if depth==12:return False
            middle=(lo+hi)/2
            return inside_plate(lo,middle,depth+1) and inside_plate(middle,hi,depth+1)
        if not inside_plate(Q(0),Q(1)):
            raise GeometryError('complete trace is not strictly inside the planar plate')
        for edge in connectors:connector_counts[edge]=connector_counts.get(edge,0)+1
        plans.append((face,section,connectors,clearance[0],residual))
    if any(count!=2 for count in connector_counts.values()) or len(connector_counts)!=len(walls):
        raise GeometryError('wall faces do not form one declared closed sweep ring')
    revision=model.revision
    with model.transaction():
        split_vertices={}
        for edge_id in sorted(connector_counts):
            edge=model.edges[edge_id];a=model.vertex_position(edge.start);b=model.vertex_position(edge.end)
            denominator=float(normal@(b-a))
            parameter=float(normal@(origin-a))/denominator
            if not 0.<parameter<1.:raise GeometryError('section endpoint is not inside its declared connector')
            vertex,_=model.split_edge(edge_id,parameter)
            split_vertices[edge_id]=vertex
        traces=[];children=[]
        for face,curve,(first,last),clearance,residual in plans:
            start,end=split_vertices[first],split_vertices[last]
            endpoints=curve.evaluate([0.,1.])
            stored=np.asarray([model.vertex_position(start),model.vertex_position(end)])
            transverse=[i for i in range(3) if i!=axis]
            if not np.array_equal(endpoints[:,transverse],stored[:,transverse]):
                raise GeometryError('connector correction changes the exact transverse profile')
            correction=max(abs(Q(float(a))-Q(float(b))) for a,b in zip(endpoints[:,axis],stored[:,axis]))
            # Coons' bilinear endpoint correction is confined to the sweep
            # coordinate; four endpoint terms bound it conservatively.
            if (8*correction>=clearance or
                    residual+4*correction*abs(Q(float(normal[axis])))>Q(float(tolerance))):
                raise GeometryError('actual Coons endpoint correction cannot qualify the section')
            trace=model.add_curve(start,end,curve)
            children.append((face,_fragment_with_edge_chain(model,face,start,end,(trace,))))
            traces.append(trace)
        # Order by declared vertices. No spatial endpoint matching is used.
        remaining=set(traces);ring=[];current=model.edges[traces[0]].start;begin=current
        while remaining:
            matches=[edge for edge in remaining if model.edges[edge].start==current or model.edges[edge].end==current]
            if len(matches)!=1 and ring:raise GeometryError('section ring is disconnected or ambiguous')
            edge=traces[0] if not ring else matches[0]
            forward=model.edges[edge].start==current
            ring.append(OrientedEdge(edge,forward));remaining.remove(edge)
            current=model.edges[edge].end if forward else model.edges[edge].start
            if current==begin and remaining:raise GeometryError('section has more than one ring')
        if current!=begin:raise GeometryError('section ring is open')
        validate_sketch_cut(model,plane_face,(tuple(item.edge for item in ring),))
        flat_children=_fragment_with_closed_chain(model,plane_face,tuple(ring))
        # One explicit structural owner per physical authored plate, while
        # preserving any existing Sheet/engineering attachments remapped above.
        owned={model.face_uses[use].face_id for sheet in model.sheets.values() for use in sheet.face_use_ids}
        for group in [pair for _,pair in children]+[flat_children]:
            missing=tuple(face for face in group if face not in owned)
            if missing:model.add_sheet(missing)
        errors=model.validate_topology()
        if errors:raise GeometryError('ring imprint topology invalid: '+'; '.join(errors))
    return TranslationalRingImprint(revision,model.revision,int(plane_face),tuple(flat_children),
                                   tuple(children),tuple(traces))



def _checksum(model):
    from .serialization import to_dict
    return to_dict(model)['checksum']['value']


def query_translational_ring_preparation(model, *, face_ids=None):
    """Fresh complete-closure proof, issued only by this owner operation.

    Plain clones and serialized documents contain no authority. Raw edits are
    detected by the complete current content checksum as well as revision.
    """
    entry=_RECEIPTS.get(model)
    if entry is None:return None
    receipt,revision,checksum,faces=entry
    if model.revision!=revision or (face_ids is not None and set(map(int,face_ids))!=set(faces)):
        return None
    if set(model.faces)!=set(faces) or model.members or _checksum(model)!=checksum:return None
    return receipt


def clone_translational_ring_prepared_geometry(model):
    """Canonical detached clone, freshly rebound after exact content check."""
    receipt=query_translational_ring_preparation(model)
    if receipt is None:raise GeometryError('no current complete ring preparation receipt')
    made=model.clone(preserve_identity=True)
    checksum=_checksum(made)
    if checksum!=_RECEIPTS[model][2]:raise GeometryError('ring prepared clone changed content')
    _RECEIPTS[made]=(receipt,made.revision,checksum,tuple(sorted(made.faces)))
    return made


def imprint_translational_ring_plane(model, wall_faces, plane_face):
    """Stage, qualify and atomically publish the complete bounded ring imprint."""
    from .nurbs_audit import certify_simple,certify_only_shared_ends
    # Exact positive-weight transverse quotient proves the entire source ring
    # simple, not merely its displayed polygon or its endpoint arrangement.
    projections=[]
    for face in wall_faces:
        item=_sweep(model,int(face))
        if item is None:raise GeometryError('unqualified translational wall')
        curve,vector,_=item
        axes=[i for i,x in enumerate(vector) if x]
        if len(axes)!=1:raise GeometryError('ring imprint requires an axis-aligned sweep')
        axis=axes[0]
        projected=replace(curve,controls=tuple(tuple(x for i,x in enumerate(p) if i!=axis) for p in curve.controls))
        if certify_simple(projected) is not True:raise GeometryError('ring profile is not certified simple')
        projections.append(projected)
    for i,a in enumerate(projections):
        for b in projections[i+1:]:
            if not certify_only_shared_ends(a,b):raise GeometryError('ring profiles cross or cannot be separated')
    staged=model.clone(preserve_identity=True)
    receipt=_imprint_translational_ring_plane(staged,wall_faces,plane_face)
    checksum=_checksum(staged)
    _publish_qualified(model, staged, receipt, checksum)
    return receipt


def _publish_qualified(model, staged, receipt, checksum, *, register=None):
    """Publish only after validation, deferring hooks until the receipt is live.

    Compatibility restore changes counters and replaces some mappings, so its
    topology snapshot alone is not a rollback record. Preserve the actual live
    containers and scalar state as well, including allocator high-water marks.
    """
    if model._transaction_journal is not None or model._notifying_hooks:
        raise GeometryError('ring publication requires an idle model')
    state = dict(vars(model))
    containers = {name: value.copy() for name, value in state.items()
                  if isinstance(value, (dict, list, set))}
    previous = _RECEIPTS.get(model)
    hooks = model._change_hooks
    notifications = tuple(hooks)
    model._change_hooks = []
    try:
        model.restore_topology(staged.topology_snapshot())
        if _checksum(model) != checksum:
            raise GeometryError('ring publication changed qualified content')
        if register is None:
            _RECEIPTS[model] = (receipt, model.revision, checksum, tuple(sorted(model.faces)))
        else:
            register()
    except BaseException:
        # Restore container identities too: public read-only views refer to them.
        for name, contents in containers.items():
            original = state[name]
            original.clear()
            if isinstance(original, list):
                original.extend(contents)
            else:
                original.update(contents)
        vars(model).clear()
        vars(model).update(state)
        if previous is None:
            _RECEIPTS.pop(model, None)
        else:
            _RECEIPTS[model] = previous
        raise
    # Commit frontier: no observer can veto publication or cause a rollback
    # after another subscriber has observed the committed change.
    model._change_hooks = hooks
    model._notifying_hooks = True
    try:
        for hook in notifications if model.revision != state['_revision'] else ():
            try:
                hook(model._last_change_set)
            except BaseException:
                continue
    finally:
        model._notifying_hooks = False
