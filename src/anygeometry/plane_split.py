"""Owner-planned regular plane splits on declared NURBS charts.

No chart inversion or sampled intersection classification is used. Broader trim
arrangements deliberately refuse until their clipping is certified.
"""
from dataclasses import dataclass, replace
from fractions import Fraction as Q
from weakref import WeakKeyDictionary, ref

from .errors import GeometryError
from .entities import EntityRef, OrientedEdge
from .nurbs import CurveOnSurface, NurbsSurface, _IDENTITY
from .plane_nurbs import plane_nurbs_section
from .policies import MutationPolicy
from .support_curves import PlaneNurbsCurve
from .surfaces import Plane
from .identity import validate_local_id
from .translational_imprint import _checksum, _publish_qualified


@dataclass(frozen=True, eq=False)
class PlaneSplitPlan:
    source_revision: int
    parent_children: tuple
    trace_edges: tuple
    positive_faces: tuple
    negative_faces: tuple
    qualification: object = None


@dataclass(frozen=True)
class PlaneSplitApplication:
    change_set: object
    parent_children: tuple
    trace_edges: tuple
    positive_faces: tuple
    negative_faces: tuple


_PLANS = WeakKeyDictionary()


def _plane_sign(plane, surface, chart_point):
    a, b = [tuple(Q(float(x)) for x in v) for v in (plane.u_vector, plane.v_vector)]
    normal = (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    point = surface.exact_point(*chart_point)
    return sum(n*(x-Q(float(o))) for n, x, o in zip(normal, point, plane.origin))


def _classify_children(model, children, plane):
    from .nurbs_plane_arrangement import _inside
    positive, negative = [], []
    for child in children:
        face = model.faces[child]
        signs = set()
        for use in face.loop:
            curve = model.edges[use.edge].curve
            if isinstance(curve, CurveOnSurface):
                sign = _plane_sign(plane, face.surface, curve.pcurve.exact_point(Q(1, 2)))
                if sign:
                    signs.add(1 if sign > 0 else -1)
        if not signs:
            # Rational probes are merely candidate witnesses: exact plane sign
            # and certified trim containment establish acceptance of a witness.
            for level in range(1, 6):
                size = 2**level
                for u in range(1, size, 2):
                    for v in range(1, size, 2):
                        point = (Q(u, size), Q(v, size))
                        sign = _plane_sign(plane, face.surface, point)
                        if sign and _inside(model, (face.loop, *face.holes), point) is True:
                            signs.add(1 if sign > 0 else -1)
                            break
                    if signs:
                        break
                if signs:
                    break
        if len(signs) != 1:
            raise GeometryError('fragment plane-side sign is not certified')
        (positive if next(iter(signs)) > 0 else negative).append(child)
    return positive, negative


def _qualify_sheet_ownership(model, parent):
    """Whether ``parent`` is sheet-owned, refusing unsupported ownership BEFORE any staging.

    Sheet/FaceUse correspondence itself is the model's existing replacement transfer
    (``GeometryModel._replace_structural_face_ownership``: sheet/part identities, the first child keeps the
    old FaceUse, further children get new FaceUses with its orientation and metadata, the >= 3 coedge
    structural gate on every loop). Here only ambiguity the transfer would refuse is refused early: several
    non-shared-region owners and attachments targeting the face. Members, attachments and junctions stay
    refused by the caller.
    """
    if parent not in model.faces:
        return False                      # the staged transaction reports the invalid face ID
    owners = tuple(model.face_uses[i] for i in sorted(model._face_structural_uses.get(parent, ())))
    if not owners:
        return False
    if model._target_attachments.get(('face', parent)):
        raise GeometryError(f'plane split face {parent} is an attachment target that requires a parameter remap')
    shared = all(owner.metadata.get('anygeometry.shared_region') is True for owner in owners) and len(
        {owner.sheet_id for owner in owners}) == len(owners)
    if len(owners) > 1 and not shared:
        raise GeometryError(f'plane split face {parent} has multiple structural owners')
    return True


def plan_plane_split(model, face_ids, plane, *, qualification=None):
    from .predicates import IntersectionQualificationPolicy, DEFAULT_INTERSECTION_QUALIFICATION_POLICY
    from .section_audit import Work
    from .plane_nurbs import CURRENT_WORK
    policy = DEFAULT_INTERSECTION_QUALIFICATION_POLICY if qualification is None else qualification
    if not isinstance(policy, IntersectionQualificationPolicy):
        raise GeometryError('plane split qualification requires IntersectionQualificationPolicy')
    work = Work(policy.max_boxes_per_pair, policy.max_subdivision_depth)
    work.max_newton_iterations = policy.max_newton_iterations
    token = CURRENT_WORK.set(work)
    try:
        return _plan_plane_split(model, face_ids, plane, qualification=policy)
    finally:
        CURRENT_WORK.reset(token)


def _plan_plane_split(model, face_ids, plane, *, qualification):
    """Plan a retained-both-sides split without modifying the caller.

    Admits regular sections against own-support straight chart-segment trims,
    including holes. Previous implicit trims and unresolved contacts refuse.
    """
    from .nurbs_section_faces import chart_loop_area, validate_nurbs_section_face
    if not isinstance(plane, Plane):
        raise GeometryError('plane split requires a Plane support')
    parents = tuple(dict.fromkeys(validate_local_id(i, name='face ID') for i in face_ids))
    if not parents:
        raise GeometryError('plane split requires selected faces')
    if model.members or model.attachments or model.junctions:
        raise GeometryError('plane split structural reference remapping is not yet qualified')
    for parent in parents:
        _qualify_sheet_ownership(model, parent)
    from .native_support_snapshots import capture_native_supports
    source_native = capture_native_supports(model)
    if any(kind.startswith('unsupported') for identifier, kind, definition in source_native):
        raise GeometryError('plane split whole-document native support binding is unavailable')
    checksum = _checksum(model)
    staged = model.clone(preserve_identity=True)
    if capture_native_supports(staged) != source_native:
        raise GeometryError('plane split clone changed native source definitions')
    from .prepared_material_source_transfer import _begin_edge_partition, _finish_edge_partition
    edge_partition_draft = _begin_edge_partition(staged)
    with staged.transaction():
        mappings, traces, positive, negative = _stage_plane_split(staged, parents, plane, qualification,
            source_edge_draft=edge_partition_draft)
    edge_partition = _finish_edge_partition(edge_partition_draft, staged)
    plan = PlaneSplitPlan(model.revision, tuple(mappings), tuple(traces), tuple(positive), tuple(negative), qualification)
    if _checksum(model) != checksum or capture_native_supports(model) != source_native:
        raise GeometryError('plane split source definitions changed during planning')
    _PLANS[plan] = (ref(model), checksum, staged, _checksum(staged), tuple(vars(plan).items()),
                    source_native, capture_native_supports(staged), edge_partition)
    return plan


def _stage_plane_split(staged, parents, plane, qualification, *, edge_draft=None, source_edge_draft=None,
                       keep=None, extra_context=None):
    """One transaction's shared implementation for unbounded/bounded owners."""
    from .nurbs_sweep_chart import has_sweep_chart, install_sweep_chart, chart_model
    mappings, traces, positive, negative = [], [], [], []
    for parent in parents:
        if not isinstance(staged._require_face(parent).surface, NurbsSurface):
            install_sweep_chart(staged, parent)
    context = {'faces': dict(staged.faces), 'edges': dict(staged.edges), 'ledgers': {}, 'events': {},
               'edge_draft': edge_draft, 'source_edge_draft': source_edge_draft}
    context.update(extra_context or {})          # bounded owner: shared 'generated' set and 'station_log' list
    for parent in parents:
        owner = chart_model(staged, parent) if has_sweep_chart(staged.faces[parent]) else staged
        face = owner.faces[parent]
        if not isinstance(face.surface, NurbsSurface):
            raise GeometryError('plane split requires a declared NURBS chart')
        if face.parameterization is not None and face.parameterization != face.surface:
            raise GeometryError('plane split does not admit an unrelated face parameterization')
        section = plane_nurbs_section(plane, face.surface)
        if len(section.paths) > qualification.max_components or section.coincident:
            raise GeometryError('plane split component budget or coincident support refused')
        if section.paths:
            from .nurbs_plane_arrangement import fragment_nurbs_plane
            children, edges = fragment_nurbs_plane(owner, parent, plane, section, source_context=context,
                coalesce_sections=has_sweep_chart(staged.faces[parent]),
                # a sheet-owned parent's FaceUse loops need >= 3 coedges: closed trace loops keep at least
                # three certified arc partitions (legacy non-sheet semantics are unchanged: 0)
                min_closed_pieces=3 if staged._face_structural_uses.get(parent) else 0, keep=keep)
        else:
            children, edges = (parent,), ()
        pos, neg = _classify_children(owner, children, plane)
        mappings.append((parent, children))
        traces.extend(edges)
        if 'generated' in context:
            context['generated'].update(edges)
            from .prepared_material_source_transfer import _mark_generated_edges
            _mark_generated_edges(source_edge_draft, edges)
        if len(traces) > qualification.max_trace_segments:
            raise GeometryError('plane split trace qualification budget exhausted')
        positive.extend(pos)
        negative.extend(neg)
    return mappings, traces, positive, negative


def apply_plane_split(model, plan, *, policy):
    """Atomically apply a current owner-issued plan with explicit imprint policy."""
    entry = _PLANS.get(plan)
    if policy != MutationPolicy.IMPRINT or entry is None:
        raise GeometryError('plane split requires an owner-issued plan and imprint policy')
    source, checksum, staged, staged_checksum, fields, source_native, staged_native, edge_partition = entry
    from .native_support_snapshots import capture_native_supports
    if tuple(vars(plan).items()) != fields:
        raise GeometryError('plane split plan fields changed')
    if (source() is not model or model.revision != plan.source_revision or _checksum(model) != checksum
            or capture_native_supports(model) != source_native):
        raise GeometryError('plane split plan is stale or belongs to another model')
    if _checksum(staged) != staged_checksum or capture_native_supports(staged) != staged_native:
        raise GeometryError('plane split staged definition changed')
    certified = dict(fields)
    if not certified['trace_edges']:
        from .transactions import ChangeSet
        return PlaneSplitApplication(ChangeSet(model.revision, model.revision),
            certified['parent_children'], (), certified['positive_faces'], certified['negative_faces'])
    from .prepared_material_source_transfer import (_prepare_source_lineage,
        _publish_source_lineage, _restore_source_lineage)
    lineage = _prepare_source_lineage(model, staged, certified['parent_children'], edge_partition)
    try:
        _publish_qualified(model, staged, None, staged_checksum,
                           register=lambda: _publish_source_lineage(model, lineage))
    except BaseException:
        _restore_source_lineage(model, lineage)
        raise
    return PlaneSplitApplication(model._last_change_set, certified['parent_children'], certified['trace_edges'],
                                 certified['positive_faces'], certified['negative_faces'])
