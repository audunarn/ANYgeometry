"""Public batch/pair owner for certified closed bounded plane sections.

The candidate is built in one transaction, then geometry and provenance are
published together. Boundary-crossing and interior-ended trace graphs remain
typed refusals in this first connected production domain.
"""
from dataclasses import fields, replace
from weakref import ref

from .errors import GeometryError
from .nurbs import NurbsSurface
from .surfaces import Plane, CoonsSurface
from .structural import ConnectionIntent
from .definition_binding import definition_checksum
from .translational_imprint import _checksum, _publish_qualified
from .native_support_snapshots import capture_native_supports, NURBS_TENSOR, NURBS_TRANSLATION
from .plane_nurbs import CURRENT_WORK

_PLANS = {}
ALGORITHM = 'plane_nurbs_bounded_station_graph_v1'


def handles_bounded(model, handles):
    from .nurbs_pair_imprint import handles_pair
    if handles_pair(model,handles):
        return True
    if not handles or any(h.kind != 'face' for h in handles):
        return False
    from .nurbs_sweep_chart import has_sweep_chart
    from ._translational_pairs import _sweep
    return any(isinstance(model.faces[h.id].surface, Plane) for h in handles) and any(
        isinstance(model.faces[h.id].surface, NurbsSurface) or has_sweep_chart(model.faces[h.id])
        or (isinstance(model.faces[h.id].surface, CoonsSurface) and _sweep(model, h.id) is not None) for h in handles)


def _entry(plan):
    record = _PLANS.get(id(plan))
    return record[1] if record is not None and record[0]() is plan else None


def _signature(plan):
    from .batch_intersections import _plan_content
    policy = tuple((f.name, id(getattr(plan.policy, f.name)) if f.name == 'cancellation_check'
                    else getattr(plan.policy, f.name)) for f in fields(plan.policy))
    return (plan.model_id.hex, plan.revision, plan.source_checksum,
        tuple((h.model_id.hex, h.kind, h.id) for h in plan.operands), policy, _plan_content(plan), plan.content_checksum)


def _candidate(model, handles, policy, qualification, *, require_cells):
    from .nurbs_pair_imprint import handles_pair,candidate_pair
    if handles_pair(model,handles):
        return candidate_pair(model,handles,policy,qualification,require_cells=require_cells)
    from .predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY, IntersectionQualificationPolicy
    from .section_audit import Work
    from .plane_split import _stage_plane_split
    from .plane_section_plate import (plate_loops, trace_in_plate, certify_trace_embedding,
                                     fragment_plate, PlateBoundaryCrossingRequired)
    from .prepared_face_preimages import _capture_application_preimages, _compose_application_preimages
    from .edge_subcurve_preimages import _capture_edge_subcurve_preimages, _finalize_edge_subcurve_preimages
    qualification = DEFAULT_INTERSECTION_QUALIFICATION_POLICY if qualification is None else qualification
    if type(qualification) is not IntersectionQualificationPolicy:
        raise GeometryError('bounded sections require IntersectionQualificationPolicy')
    qualification = replace(qualification)
    source_revision, checksum, native = model.revision, _checksum(model), capture_native_supports(model)
    original_policy = tuple(getattr(policy, f.name) for f in fields(policy))
    if any(kind not in ('plane', NURBS_TENSOR, NURBS_TRANSLATION) for _, kind, _ in native):
        raise GeometryError('bounded section native support binding is unavailable')
    if model.members or model.attachments or model.junctions:
        raise GeometryError('bounded section structural reference remapping is not yet qualified')
    for handle in handles:
        # Sheet/FaceUse ownership transfers through the model's existing replacement; ambiguity refuses first
        from .plane_split import _qualify_sheet_ownership
        _qualify_sheet_ownership(model, handle.id)
    flat = [h.id for h in handles if isinstance(model.faces[h.id].surface, Plane)]
    parents = tuple(h.id for h in handles if h.id not in flat)
    if len(flat) != 1 or not parents:
        raise GeometryError('bounded section currently requires one authored plate and NURBS operands')
    examined = [0]
    def check(*_phase):
        examined[0] += 1
        if policy.cancellation_check is not None and policy.cancellation_check():
            raise GeometryError('bounded intersection planning cancelled')
        if (model.revision != source_revision or _checksum(model) != checksum or capture_native_supports(model) != native
                or tuple(getattr(policy, f.name) for f in fields(policy)) != original_policy):
            raise GeometryError('bounded intersection source or policy changed during callback')
        if policy.max_predicates is not None and examined[0] > policy.max_predicates:
            raise GeometryError('bounded intersection predicate budget exhausted')
        return False
    check()
    pairs = len(handles)*(len(handles)-1)//2
    if policy.max_candidate_pairs is not None and pairs > policy.max_candidate_pairs:
        raise GeometryError('bounded intersection candidate pair budget exhausted')
    face_preimages = _capture_application_preimages(model, allow_seed=True)
    edge_draft = _capture_edge_subcurve_preimages(model, allow_seed=True, cancellation_check=check)
    candidate = model.clone(preserve_identity=True)
    from .prepared_material_source_transfer import _begin_edge_partition, _finish_edge_partition
    source_edge_draft = _begin_edge_partition(candidate)
    work = CURRENT_WORK.get() or Work(qualification.max_boxes_per_pair, qualification.max_subdivision_depth)
    work.max_newton_iterations = qualification.max_newton_iterations
    token = CURRENT_WORK.set(work)
    try:
        frame, loops = plate_loops(candidate, candidate.faces[flat[0]])
        certify_trace_embedding(model, parents)
        auxiliary = ()
        with candidate.transaction():
            routed = None
            if len(parents) == 1:
                # plate-boundary events present: the connected shared-station route (clips in both domains);
                # otherwise None and the legacy all-inside/all-outside route below decides, unchanged
                from .bounded_cut_graph import stage_bounded, tag_auxiliary
                routed = stage_bounded(candidate, flat[0], parents[0], qualification, check,
                                       edge_draft, source_edge_draft)
            if routed is not None:
                mappings, traces, plate_children, auxiliary = routed
                traces = list(traces)
                # every created trace interval passed ``_Clip.keep`` (certified inside the plate with its end
                # events authenticated); dropped intervals create no trace, so this is derived, not assumed
                statuses = [True for _trace in traces]
                if traces:
                    mappings.append((flat[0], plate_children))
                    tag_auxiliary(candidate, auxiliary)
            else:
                mappings, traces, _positive, _negative = _stage_plane_split(candidate, parents,
                    candidate.faces[flat[0]].surface, qualification, edge_draft=edge_draft,
                    source_edge_draft=source_edge_draft)
                statuses = [trace_in_plate(candidate, edge, frame, loops, check) for edge in traces]
                if statuses and not all(statuses) and any(statuses):
                    raise PlateBoundaryCrossingRequired('partial bounded trace systems need a plate boundary-event graph')
                if traces and all(statuses) and require_cells:
                    children = fragment_plate(candidate, flat[0], traces, frame, check)
                    mappings.append((flat[0], children))
                else:
                    mappings.append((flat[0], (flat[0],)))
            if require_cells and traces and any(statuses) and policy.intent is ConnectionIntent.CONNECT:
                # Geometry and structural declarations form ONE logical commit.
                # The journal retains complete face replacements and source+1
                # revision, matching atomic publication's qualified checksum.
                from .joint_edges import declare_joint_edge
                for edge_id in sorted(set(traces)):
                    check()
                    declare_joint_edge(candidate,edge_id,
                        candidate.tolerance.effective_length(candidate.edge_length(edge_id)))
        if not traces or not any(statuses):
            auxiliary = ()
            candidate, traces = model.clone(preserve_identity=True), []
            mappings = [(h.id, (h.id,)) for h in handles]
            edge_draft = _capture_edge_subcurve_preimages(model, allow_seed=True, cancellation_check=check)
            source_edge_draft = _begin_edge_partition(candidate)
        edge_partition = _finish_edge_partition(source_edge_draft, candidate)
        if not require_cells:
            return candidate, tuple(traces), tuple(mappings), None, None, native, qualification, edge_partition, tuple(auxiliary)
        changes = candidate.last_change_set.replacements if candidate.revision != source_revision else ()
        face_preimages = _compose_application_preimages(candidate, face_preimages, changes)
        edge_preimages = _finalize_edge_subcurve_preimages(candidate, edge_draft, cancellation_check=check)
        check()
        return (candidate, tuple(traces), tuple(mappings), face_preimages, edge_preimages, native, qualification,
                edge_partition, tuple(auxiliary))
    finally:
        CURRENT_WORK.reset(token)


def plan_bounded(model, handles, policy, qualification=None, *, stored_policy=None):
    # ``policy`` is the caller's object (pinned in _candidate); ``stored_policy`` (default the same)
    # is only what the issued plan stores.
    from .batch_intersections import IntersectionPlan, _plan_content, _policy
    policy = _policy(policy)
    stored_policy = policy if stored_policy is None else stored_policy
    if policy.intent not in (ConnectionIntent.CONNECT, ConnectionIntent.IMPRINT):
        raise GeometryError('bounded section planning requires explicit CONNECT or IMPRINT')
    candidate, traces, mappings, faces, edges, native, qualification, edge_partition, auxiliary = _candidate(
        model, handles, policy, qualification, require_cells=True)
    plan = IntersectionPlan(model.model_id, model.revision, _checksum(model), tuple(handles), stored_policy, ())
    plan = replace(plan, content_checksum=_plan_content(plan))
    data = (ref(model), _signature(plan), candidate, _checksum(candidate), traces,
            mappings, faces, edges, native, capture_native_supports(candidate), qualification, edge_partition,
            auxiliary)
    key = id(plan)
    _PLANS[key] = (ref(plan, lambda weak: _PLANS.pop(key, None)), data)
    from .preparation_epochs import _epoch_plan_required, _issue_native_plan
    if _epoch_plan_required(model):
        _issue_native_plan(plan, native)
    return plan


def is_bounded_plan(plan):
    return _entry(plan) is not None


def apply_bounded(model, plan, policy, *, planning_mode=None):
    # ``policy`` is the caller's object itself; ``planning_mode`` is the separate mode override.
    from .batch_intersections import IntersectionApplication, _validate_policy
    from .transactions import ChangeSet
    from .preparation_epochs import (_seals, _permits, _consume_epoch, _epoch_plan_required, _native_plan_supports)
    from .prepared_face_preimages import _publish_application_preimages
    from .edge_subcurve_preimages import _publish_edge_subcurve_preimages
    row = _entry(plan)
    if row is None or row[0]() is not model or _signature(plan) != row[1]:
        raise GeometryError('bounded intersection plan was copied, changed or belongs to another model')
    _validate_policy(plan, policy, planning_mode=planning_mode)
    (candidate, checksum, traces, mappings, faces, edges, native, candidate_native, _qualification,
     edge_partition, auxiliary) = row[2:]
    if model.revision != plan.revision or _checksum(model) != plan.source_checksum or capture_native_supports(model) != native:
        raise GeometryError('bounded intersection plan is stale or source definitions changed')
    if _epoch_plan_required(model) and _native_plan_supports(plan) != native:
        raise GeometryError('bounded epoch native source binding changed')
    if policy.cancellation_check is not None and policy.cancellation_check():
        raise GeometryError('bounded intersection cancelled before publication')
    if (model.revision != plan.revision or _checksum(model) != plan.source_checksum or capture_native_supports(model) != native
            or _signature(plan) != row[1] or _checksum(candidate) != checksum
            or capture_native_supports(candidate) != candidate_native):
        raise GeometryError('bounded intersection operands changed before publication')
    coverage = tuple(sorted(candidate.faces)) if set(model.faces) == {h.id for h in plan.operands} else ()
    from .prepared_material_source_transfer import (_prepare_source_lineage,
        _publish_source_lineage, _restore_source_lineage)
    lineage = _prepare_source_lineage(model, candidate, mappings, edge_partition)
    old_seal, old_permit = _seals.get(model), _permits.get(model)
    before = model.revision
    from .prepared_material_source_transfer import _register_auxiliary, _restore_auxiliary, _AUXILIARY
    old_auxiliary = _AUXILIARY.get(model)
    def register():
        _publish_source_lineage(model, lineage)
        if auxiliary:
            _register_auxiliary(model, auxiliary)       # owner-captured; a persisted tag is never authority
        _publish_application_preimages(model, faces, coverage, checksum)
        _publish_edge_subcurve_preimages(model, edges, checksum)
        model._intersection_application_receipt = (plan, model.revision, checksum, traces)
        model._intersection_preparation_receipt = (plan, model.revision, checksum, coverage)
        if faces is not None and edges is not None:
            _consume_epoch(model)
    try:
        _publish_qualified(model, candidate, None, checksum, register=register)
    except BaseException:
        _restore_source_lineage(model, lineage)
        _restore_auxiliary(model, old_auxiliary)
        if old_seal is None: _seals.pop(model, None)
        else: _seals[model] = old_seal
        if old_permit is None: _permits.pop(model, None)
        else: _permits[model] = old_permit
        raise
    return IntersectionApplication(plan, model.last_change_set if model.revision != before else ChangeSet(before, before),
                                   tuple(model.handle('edge', edge) for edge in traces), model.revision == before)


def query_pair(model, first, second, qualification):
    from .predicates import DEFAULT_INTERSECTION_QUALIFICATION_POLICY, IntersectionQualificationPolicy
    from .section_audit import Work
    qualification = DEFAULT_INTERSECTION_QUALIFICATION_POLICY if qualification is None else qualification
    if type(qualification) is not IntersectionQualificationPolicy:
        raise GeometryError('bounded sections require IntersectionQualificationPolicy')
    work = Work(qualification.max_boxes_per_pair, qualification.max_subdivision_depth)
    work.max_newton_iterations = qualification.max_newton_iterations
    token = CURRENT_WORK.set(work)
    try:
        return _query_pair(model, first, second, qualification, work)
    finally:
        CURRENT_WORK.reset(token)


def _connected_trace_curves(staged, traces):
    """One native curve per declared connected component, never per arc edge."""
    from fractions import Fraction as Q
    from .plane_nurbs import plane_nurbs_section
    by_vertex = {}
    for identifier in traces:
        edge = staged.edges[identifier]
        for vertex in (edge.start, edge.end):
            by_vertex.setdefault(vertex, []).append(identifier)
    if any(len(uses) > 2 for uses in by_vertex.values()):
        raise GeometryError('query trace component has an unresolved branch station')
    remaining, components = set(traces), []
    while remaining:
        pending, found = [min(remaining)], set()
        while pending:
            identifier = pending.pop()
            if identifier in found:
                continue
            found.add(identifier)
            remaining.remove(identifier)
            edge = staged.edges[identifier]
            pending.extend(other for v in (edge.start, edge.end) for other in by_vertex[v] if other not in found)
        curves = [staged.edges[identifier].curve for identifier in sorted(found)]
        first = curves[0]
        if any((c.plane_record, c.surface, c.path) != (first.plane_record, first.surface, first.path) for c in curves):
            raise GeometryError('query component crosses unsupported native chart definitions')
        intervals = sorted(tuple(sorted(map(Q, c.domain))) for c in curves)
        if any(a[1] != b[0] for a, b in zip(intervals, intervals[1:])):
            raise GeometryError('wrapped or noncontiguous native query component needs a composite curve definition')
        vertices = {v for identifier in found for v in (staged.edges[identifier].start, staged.edges[identifier].end)}
        closed = all(len(by_vertex[v]) == 2 for v in vertices)
        if closed and (intervals[0][0] != 0 or intervals[-1][1] != 1
                       or not plane_nurbs_section(first.plane, first.surface).paths[first.path].closed):
            raise GeometryError('closed native query component does not cover its complete owner path')
        components.append((replace(first, domain=(float(intervals[0][0]), float(intervals[-1][1]))), closed))
    return tuple(components)


def _query_pair(model, first, second, qualification, work):
    from .nurbs_pair_imprint import handles_pair,query_pair as query_nurbs_pair
    if handles_pair(model,(first,second)):
        return query_nurbs_pair(model,first,second,qualification)
    from .batch_intersections import IntersectionBatchPolicy
    from .intersections import _qualified_result
    from .predicates import IntersectionKind, IntersectionDimension, IntersectionComponent, IntersectionQuality, IntersectionCertificate
    handles = tuple(sorted((first, second)))
    try:
        staged, traces, mappings, _faces, _edges, _native, _qualification, _partition, _auxiliary = _candidate(model, handles,
            IntersectionBatchPolicy(ConnectionIntent.IMPRINT), qualification, require_cells=False)
    except GeometryError as cause:
        return _qualified_result(model, first, second, IntersectionKind.UNCLASSIFIED, diagnostics=(str(cause),))
    components = []
    native_components = _connected_trace_curves(staged, traces)
    if len(native_components) > qualification.max_components:
        raise GeometryError('bounded query component budget exhausted')
    for native_curve, closed in native_components:
        # Detached public definitions, not candidate topology or evaluator
        # caches, form the immutable analytic record and its checksum content.
        curve = native_curve
        curve = replace(curve, surface=replace(curve.surface))
        from .nurbs_runtime_binding import validate_section_runtime
        from .predicates import CertifiedCurveTrace
        import numpy as np
        stations = (0., .25, .5, .75, 1.) if closed else (0., .5, 1.)
        witnesses = tuple(tuple(float(x) for x in curve.evaluate(t)) for t in stations)
        residual = 0.
        for t, point in zip(stations, witnesses):
            low, high = curve.bounds(t, t)
            residual = max(residual, float(np.linalg.norm(np.maximum(np.abs(np.asarray(low)-point), np.abs(np.asarray(high)-point)))))
        if residual > model.tolerance.length:
            raise GeometryError('bounded query witness residual exceeds owner tolerance')
        validate_section_runtime(curve)
        chart = tuple(tuple(float(x) for x in p) for p in curve.chart(np.asarray(stations)))
        plane = tuple(tuple(float(x) for x in curve.plane.local_uv(p)) for p in witnesses)
        first_plane = isinstance(model.faces[first.id].surface, Plane)
        from .nurbs_sweep_chart import authored_query_uv
        chart = authored_query_uv(model, second.id if first_plane else first.id, curve.surface, chart)
        first_uv, second_uv = (plane, chart) if first_plane else (chart, plane)
        # These residuals qualify published witnesses/parameter records only;
        # complete section topology comes from the exact owner graph above.
        for handle, parameters in ((first, first_uv), (second, second_uv)):
            for point, (u, v) in zip(witnesses, parameters):
                residual = max(residual, float(np.nextafter(np.linalg.norm(
                    np.asarray(model.face_support_point(handle.id, u, v))-point), np.inf)))
        if residual > model.tolerance.length:
            raise GeometryError('authored query parameter witness residual exceeds owner tolerance')
        certificate = IntersectionCertificate(ALGORITHM, model.tolerance.length,
            max_residual=residual, max_enclosure_width=residual, complete=True)
        trace = CertifiedCurveTrace(witnesses, first_uv, second_uv, certificate, closed=closed)
        components.append(IntersectionComponent(witnesses, IntersectionQuality.VERIFIED_APPROXIMATE,
            analytic_curve=curve, first_parameter_path=first_uv, second_parameter_path=second_uv,
            curve_traces=(trace,), max_residual=residual, first_subparent=first, second_subparent=second,
            certificate=certificate))
    # These freshly owned records have not escaped. Finalize their accounting
    # after ALL runtime requalification and witness bounds, under this SAME
    # work scope. No geometry evaluation follows this publication frontier.
    for component in components:
        certificate = replace(component.certificate, boxes_examined=work.boxes,
            subdivisions=work.subdivisions, trace_segments=len(traces))
        object.__setattr__(component, 'certificate', certificate)
        for trace in component.curve_traces:
            object.__setattr__(trace, 'certificate', certificate)
    result = _qualified_result(model, first, second, IntersectionKind.CROSS if components else IntersectionKind.DISJOINT,
        tuple(components), dimension=IntersectionDimension.CURVE if components else IntersectionDimension.NONE)
    return replace(result, certificate=replace(result.certificate, algorithm=ALGORITHM,
        boxes_examined=work.boxes, subdivisions=work.subdivisions, trace_segments=len(traces)))
