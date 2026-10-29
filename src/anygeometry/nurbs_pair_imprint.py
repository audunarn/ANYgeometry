"""Public two-NURBS owner query and atomic closed-branch fragmentation candidate."""
from dataclasses import fields,replace
from fractions import Fraction as Q

from .errors import GeometryError
from .nurbs import NurbsSurface
from .nurbs_surface_intersection import query_nurbs_surface_intersection,NurbsIntersectionCurve,IntersectionStatus
from .nurbs_intersection_curve import NurbsSurfaceIntersectionCurve,intersection_curve_operation
from .nurbs_intersection_faces import rectangle_loop,loop_area


def handles_pair(model,handles):
    return len(handles) == 2 and all(h.kind == 'face' and isinstance(model.faces[h.id].surface,NurbsSurface) for h in handles)


def _definitions(model,handles):
    if not handles_pair(model,handles) or handles[0] == handles[1]:
        raise GeometryError('two distinct authored NURBS faces are required')
    for h in handles:
        face = model.faces[h.id]
        if face.holes or face.parameterization is not None:
            raise GeometryError('pretrimmed or remapped NURBS operands require a certified arrangement event graph')
        rectangle_loop(model,face.loop,face.surface)
    return tuple(model.faces[h.id].surface for h in handles)


def _curves(model,handles,qualification,check=lambda:False):
    first,second = _definitions(model,handles)
    result = query_nurbs_surface_intersection(first,second,tolerances=model.tolerance,
        qualification=qualification,cancellation_check=check)
    if result.status is IntersectionStatus.DISJOINT:
        return ()
    if result.status is not IntersectionStatus.CERTIFIED:
        raise GeometryError('two-support intersection remains unclassified: '+result.reason)
    settings = tuple((f.name,getattr(model.tolerance,f.name)) for f in fields(model.tolerance))
    limits = tuple((f.name,getattr(qualification,f.name)) for f in fields(qualification))
    return tuple(NurbsSurfaceIntersectionCurve(NurbsIntersectionCurve.from_result(result,i),
        tolerance_values=settings,qualification_values=limits) for i in range(len(result.components)))


def query_pair(model,first,second,qualification):
    from .intersections import _qualified_result
    from .predicates import (IntersectionKind,IntersectionDimension,IntersectionComponent,
        IntersectionQuality,IntersectionCertificate,CertifiedCurveTrace)
    import numpy as np
    try:
        with intersection_curve_operation():
            curves = _curves(model,(first,second),qualification)
            components = []
            for curve in curves:
                stations = (0.,.25,.5,.75,1.) if curve.owner.closed else (0.,.5,1.)
                points = tuple(tuple(float(x) for x in p) for p in curve.evaluate(stations))
                a,b = (model.faces[h.id].surface for h in (first,second))
                uv_a = tuple(tuple(float(x) for x in p) for p in curve.for_support(a).chart(stations))
                uv_b = tuple(tuple(float(x) for x in p) for p in curve.for_support(b).chart(stations))
                residual = 0.
                for t,p in zip(stations,points):
                    low,high = curve.bounds(t,t)
                    residual = max(residual,float(np.nextafter(np.linalg.norm(np.maximum(abs(low-p),abs(high-p))),np.inf)))
                if residual > model.tolerance.length:
                    raise GeometryError('two-support witness enclosure exceeds the model length tolerance')
                cert = IntersectionCertificate('nurbs_two_support_owner_v1',model.tolerance.length,
                    max_residual=residual,max_enclosure_width=residual,complete=True)
                trace = CertifiedCurveTrace(points,uv_a,uv_b,cert,closed=curve.owner.closed)
                components.append(IntersectionComponent(points,IntersectionQuality.VERIFIED_APPROXIMATE,
                    first_parameter_path=uv_a,second_parameter_path=uv_b,analytic_curve=curve,
                    curve_traces=(trace,),max_residual=residual,certificate=cert,
                    first_subparent=first,second_subparent=second))
    except GeometryError as error:
        return _qualified_result(model,first,second,IntersectionKind.UNCLASSIFIED,diagnostics=(str(error),))
    return _qualified_result(model,first,second,IntersectionKind.CROSS if components else IntersectionKind.DISJOINT,
        components,dimension=IntersectionDimension.CURVE if components else IntersectionDimension.NONE)


def _fragment(model,parent,ring):
    from .entities import EntityRef,OrientedEdge
    face = model.faces[parent]
    sign = rectangle_loop(model,face.loop,face.surface)
    area = loop_area(model,ring,face.surface)
    if (sign > 0) != (area[0] > 0):
        ring = tuple(OrientedEdge(e.edge,not e.forward) for e in reversed(ring))
    reverse = tuple(OrientedEdge(e.edge,not e.forward) for e in reversed(ring))
    metadata,tags = dict(face.metadata),model.tags_for(face.ref)
    model._delete_entity('face',parent)
    outer = model.add_face_from_loop(face.loop,face.corners,surface=face.surface)
    model._put_entity('face',replace(model.faces[outer],holes=(reverse,),metadata=metadata,
                                    parameterization=face.parameterization))
    inner = model.add_face_from_loop(ring,surface=face.surface)
    model._put_entity('face',replace(model.faces[inner],metadata=metadata,
                                    parameterization=face.parameterization))
    for child in (outer,inner):
        model.tag(EntityRef('face',child),*tags)
    model.record_replacement(EntityRef('face',parent),(EntityRef('face',outer),EntityRef('face',inner)))
    return outer,inner


def _declaration_feature_extent(curve,check):
    """Origin-independent conservative feature scale, not a length integral."""
    from .plane_nurbs import charge_box
    from .nurbs_surface_intersection._intervals import float_up
    check()
    charge_box()  # actual certified world-enclosure query
    boxes = curve.enclose(Q(0),Q(1))
    if not boxes:
        raise GeometryError('joint declaration needs a certified source world enclosure')
    extent = sum((max(Q(box.world_hi[k]) for box in boxes)-
                  min(Q(box.world_lo[k]) for box in boxes) for k in range(3)),Q(0))
    check()
    return float_up(extent)


def candidate_pair(model,handles,policy,qualification,*,require_cells):
    from .predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY
    from .translational_imprint import _checksum
    from .serialization import _qualified_model_state
    from .native_support_snapshots import capture_native_supports
    from .prepared_face_preimages import _capture_application_preimages,_compose_application_preimages
    from .edge_subcurve_preimages import _capture_edge_subcurve_preimages,_finalize_edge_subcurve_preimages
    from .prepared_material_source_transfer import _begin_edge_partition,_finish_edge_partition
    from .entities import OrientedEdge
    qualification = DEFAULT_INTERSECTION_QUALIFICATION_POLICY if qualification is None else replace(qualification)
    _definitions(model,handles)
    if model.members or model.attachments or model.junctions:
        raise GeometryError('two-support structural attachment remapping is not yet qualified')
    from .plane_split import _qualify_sheet_ownership
    for h in handles:
        _qualify_sheet_ownership(model,h.id)
    revision,checksum,native = model.revision,_checksum(model),capture_native_supports(model)
    original_policy = tuple(getattr(policy,f.name) for f in fields(policy))
    examined = [0]
    def check(*_phase,_full=False):
        examined[0] += 1
        callback = policy.cancellation_check
        if callback is not None and callback():
            raise GeometryError('two-support planning cancelled')
        if callback is not None:
            # Encode complete live content after EVERY returning external
            # callback. The existing service also binds reverse incidence and
            # repeats qualification whenever either binding changed.
            _qualified_model_state(model,expected_checksum=checksum,
                changed_message='two-support planning source changed during callback')
        if (model.revision != revision or capture_native_supports(model) != native
                or (_full and _checksum(model) != checksum)
                or original_policy != tuple(getattr(policy,f.name) for f in fields(policy))):
            raise GeometryError('two-support planning source or policy changed during callback')
        if policy.max_predicates is not None and examined[0] > policy.max_predicates:
            raise GeometryError('two-support predicate budget exhausted')
        return False
    if policy.max_candidate_pairs is not None and policy.max_candidate_pairs < 1:
        raise GeometryError('two-support candidate pair budget exhausted')
    check(_full=True)
    face_preimages = _capture_application_preimages(model,allow_seed=True)
    edge_draft = _capture_edge_subcurve_preimages(model,allow_seed=True,cancellation_check=check)
    candidate = model.clone(preserve_identity=True)
    source_edges = _begin_edge_partition(candidate)
    with intersection_curve_operation(check):
        curves = _curves(model,handles,qualification,check)
        if len(curves) != 1 or not curves[0].owner.closed or any(s[0] != 'scalar' for s in curves[0].owner.segments):
            raise GeometryError('fragment publication currently requires one complete embedded scalar closed branch')
        curve = curves[0]
        stations = sorted({Q(0),Q(1,3),Q(2,3),Q(1)} |
            {Q(i,len(curve.owner.segments)) for i in range(len(curve.owner.segments)+1)})
        trace_edges,mappings = [],[]
        with candidate.transaction():
            vertices = [candidate.add_point(*map(float,point)) for point in curve.evaluate(stations[:-1])]
            vertices.append(vertices[0])
            for a,b,v,w in zip(stations,stations[1:],vertices,vertices[1:]):
                check()
                trace_edges.append(candidate.add_curve(v,w,curve.subcurve(a,b)))
            ring = tuple(OrientedEdge(e,True) for e in trace_edges)
            for handle in handles:
                mappings.append((handle.id,_fragment(candidate,handle.id,ring)))
            from .structural import ConnectionIntent
            if policy.intent is ConnectionIntent.CONNECT:
                from .joint_edges import declare_joint_edge
                for edge_id in trace_edges:
                    check()
                    declare_joint_edge(candidate,edge_id,
                        candidate.tolerance.effective_length(
                            _declaration_feature_extent(candidate.edges[edge_id].curve,check)))
        failures = (*candidate.validate_topology(),*candidate._validate_structural())
        if failures:
            raise GeometryError('; '.join(str(x) for x in failures))
    changes = candidate.last_change_set.replacements
    face_preimages = _compose_application_preimages(candidate,face_preimages,changes)
    edge_preimages = _finalize_edge_subcurve_preimages(candidate,edge_draft,cancellation_check=check)
    partition = _finish_edge_partition(source_edges,candidate)
    check(_full=True)
    return candidate,tuple(trace_edges),tuple(mappings),face_preimages,edge_preimages,native,qualification,partition,()
