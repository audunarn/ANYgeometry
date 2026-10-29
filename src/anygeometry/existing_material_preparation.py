"""Owner-issued complete SAME-chart no-recut preparation.

This strict route never connects geometric copies. All shared intervals and
endpoints must already have authenticated current world topology.
"""
from dataclasses import fields, replace
from itertools import combinations
from weakref import ref

from .errors import GeometryError
from .definition_binding import definition_checksum
from .nurbs import NurbsSurface
from .structural import ConnectionIntent
from .prepared_material_source_transfer import _state
from .plane_nurbs import CURRENT_WORK, charge_box

_PLANS = {}


def _policy_pin(policy):
    if type(policy.intent) is not ConnectionIntent or any(type(getattr(policy, name)) is not bool
        for name in ('face_connections', 'member_connections', 'member_face_connections')):
        raise GeometryError('existing material policy has noncanonical field types')
    for name in ('max_predicates', 'max_candidate_pairs'):
        value = getattr(policy, name)
        if value is not None and (type(value) is not int or value < 1):
            raise GeometryError('existing material policy budget is malformed')
    if policy.cancellation_check is not None and not callable(policy.cancellation_check):
        raise GeometryError('existing material policy cancellation_check is not callable')
    return tuple((f.name, type(getattr(policy, f.name)), id(getattr(policy, f.name))
                  if f.name == 'cancellation_check' else getattr(policy, f.name)) for f in fields(policy))


def _signature(plan):
    from .batch_intersections import _plan_content
    return (plan.model_id, plan.revision, plan.source_checksum, tuple(plan.operands),
            _policy_pin(plan.policy),
            _plan_content(plan), plan.content_checksum)


def _entry(plan):
    entry = _PLANS.get(id(plan))
    return entry if entry is not None and entry[0]() is plan else None


def is_existing_plan(plan):
    return _entry(plan) is not None


def eligible(model, handles):
    """Geometric family selection only; this grants no preparation authority."""
    from .nurbs_sweep_chart import has_sweep_chart, chart_model
    if len(handles) < 2 or any(h.kind != 'face' or h.id not in model.faces for h in handles):
        return False
    supports = []
    for h in handles:
        face = model.faces[h.id]
        if has_sweep_chart(face):
            support = chart_model(model, h.id).support
        else:
            support = face.surface
            if face.parameterization is not None and face.parameterization is not support:
                return False
        if type(support) is not NurbsSurface:
            return False
        supports.append(support)
    return len({definition_checksum(support) for support in supports}) == 1


def _ownership(model, handles, shared, intent):
    from .plane_split import _qualify_sheet_ownership
    from .joint_edges import query_joint_edge
    for h in handles:
        owned = _qualify_sheet_ownership(model, h.id)
        if intent is ConnectionIntent.CONNECT and not owned:
            raise GeometryError('existing material CONNECT requires current Sheet ownership')
    allowed_attachments, allowed_junctions = set(), set()
    for edge in shared:
        owners = tuple(model.sheets_using_edge(edge))
        if len(owners) > 1:
            joint = query_joint_edge(model, edge)
            if not joint.declared:
                raise GeometryError('existing material contact between Sheets has no declared joint')
            for handle in joint.junctions:
                allowed_junctions.add(handle.id)
                allowed_attachments.update(model.junctions[handle.id].attachment_ids)
    if model.members or set(model.attachments)-allowed_attachments or set(model.junctions)-allowed_junctions:
        raise GeometryError('existing material reference remapping is not qualified')


def try_plan_existing(model, handles, policy):
    """Return an issued plan, or None only outside this strict native family."""
    policy_pin = _policy_pin(policy)
    if not policy.face_connections:
        return None
    from .batch_intersections import IntersectionPlan, _plan_content
    from .same_support_material_relation import _query_material_pair
    from .section_audit import Work
    from .preparation_epochs import _epoch_plan_required, _issue_native_plan
    if model._transaction_journal is not None or model._notifying_hooks:
        return None    # publication (seals, permits, lineage) is not journaled: not offered inside a transaction
    token = CURRENT_WORK.set(Work()) if CURRENT_WORK.get() is None else None
    try:
        if not eligible(model, handles):
            return None
        initial = _state(model)
        count = [0]
        def predicate():
            count[0] += 1
            if policy.max_predicates is not None and count[0] > policy.max_predicates:
                raise GeometryError('existing material preparation predicate budget exhausted')
        def unchanged():
            actual = _policy_pin(policy)
            if _state(model) != initial or actual != policy_pin:
                raise GeometryError('existing material source or policy changed during planning')
        def check(*_phase):
            unchanged()
            if policy.cancellation_check is not None:
                cancelled = policy.cancellation_check()
                unchanged()
                if cancelled:
                    raise GeometryError('existing material preparation cancelled')
            return False
        check()
        ledger, shared = [], set()
        for number, (first, second) in enumerate(combinations(handles, 2), 1):
            charge_box(); predicate()
            if policy.max_candidate_pairs is not None and number > policy.max_candidate_pairs:
                raise GeometryError('existing material candidate pair budget exhausted')
            contacts = []
            if not _query_material_pair(model, first, second,
                cancellation_check=check if policy.cancellation_check is not None else None,
                contact_ledger=contacts, _tick_hook=predicate):
                raise GeometryError('existing material pair overlaps or is outside the proved strict arrangement')
            ledger.append((first.id, second.id, tuple(contacts)))
            shared.update(row[1] for row in contacts if row[0] == 'edge')
            check()
        _ownership(model, handles, shared, policy.intent)
        candidate = model.clone(preserve_identity=True)
        candidate_state = _state(candidate)
        check()
        if candidate_state != initial:
            raise GeometryError('existing material clone changed complete geometry or native supports')
        plan = IntersectionPlan(model.model_id, model.revision, initial[1], tuple(handles), policy, ())
        plan = replace(plan, content_checksum=_plan_content(plan))
        # Detached immutable tuples contain the complete pair/contact ledger;
        # no mutable self-carried digest can replace its external issuance pin.
        data = (ref(plan, lambda weak, key=id(plan): _PLANS.pop(key, None)), ref(model), _signature(plan),
                initial, candidate, candidate_state, tuple(ledger), tuple(sorted(shared)))
        _PLANS[id(plan)] = data
        if _epoch_plan_required(model):
            _issue_native_plan(plan, initial[2])
        return plan
    finally:
        if token is not None:
            CURRENT_WORK.reset(token)


def apply_existing(model, plan, policy):
    """No topology change; atomic publication of normal current owner proof."""
    from .batch_intersections import IntersectionApplication, _validate_policy
    from .translational_imprint import _publish_qualified
    from .transactions import ChangeSet
    from .prepared_face_preimages import (_capture_application_preimages, _compose_application_preimages,
                                          _publish_application_preimages)
    from .edge_subcurve_preimages import (_capture_edge_subcurve_preimages, _finalize_edge_subcurve_preimages,
                                          _publish_edge_subcurve_preimages)
    from .prepared_material_source_transfer import (_prepare_source_lineage, _publish_source_lineage,
                                                    _restore_source_lineage)
    from .preparation_epochs import (_seals, _permits, _consume_epoch, _epoch_plan_required, _native_plan_supports)
    entry = _entry(plan)
    if entry is None or entry[1]() is not model or _signature(plan) != entry[2]:
        raise GeometryError('existing material plan was not issued for this unchanged owner')
    _validate_policy(plan, policy)
    _, _, _, initial, candidate, candidate_state, ledger, shared = entry
    policy_pin = _policy_pin(policy)
    predicates = [0]
    def unchanged():
        current = _policy_pin(policy)
        if (current != policy_pin or _state(model) != initial or _state(candidate) != candidate_state
                or _signature(plan) != entry[2]):
            raise GeometryError('existing material application policy or operands changed')
    def check(*_phase):
        predicates[0] += 1
        if policy.max_predicates is not None and predicates[0] > policy.max_predicates:
            raise GeometryError('existing material application predicate budget exhausted')
        unchanged()
        if policy.cancellation_check is not None:
            cancelled = policy.cancellation_check()
            unchanged()
            if cancelled:
                raise GeometryError('existing material application cancelled')
    check()
    if _epoch_plan_required(model) and _native_plan_supports(plan) != initial[2]:
        raise GeometryError('existing material native epoch plan is unavailable')
    _ownership(model, plan.operands, shared, policy.intent)
    faces = _capture_application_preimages(model, allow_seed=True)
    draft = _capture_edge_subcurve_preimages(model, allow_seed=True, cancellation_check=check)
    faces = _compose_application_preimages(candidate, faces, ())
    edges = _finalize_edge_subcurve_preimages(candidate, draft, cancellation_check=check)
    lineage = _prepare_source_lineage(model, candidate, ())
    coverage = tuple(sorted(model.faces)) if set(model.faces) == {h.id for h in plan.operands} else ()
    old_seal, old_permit = _seals.get(model), _permits.get(model)
    before = model.revision
    check()
    def register():
        if _state(model) != candidate_state:
            raise GeometryError('existing material publication changed native geometry')
        _publish_source_lineage(model, lineage)
        _publish_application_preimages(model, faces, coverage, candidate_state[1])
        _publish_edge_subcurve_preimages(model, edges, candidate_state[1])
        model._intersection_application_receipt = (plan, model.revision, candidate_state[1], shared)
        model._intersection_preparation_receipt = (plan, model.revision, candidate_state[1], coverage)
        if faces is not None and edges is not None:
            _consume_epoch(model)
    try:
        _publish_qualified(model, candidate, None, candidate_state[1], register=register)
    except BaseException:
        _restore_source_lineage(model, lineage)
        if old_seal is None: _seals.pop(model, None)
        else: _seals[model] = old_seal
        if old_permit is None: _permits.pop(model, None)
        else: _permits[model] = old_permit
        raise
    return IntersectionApplication(plan, model.last_change_set if model.revision != before else ChangeSet(before, before),
                                   tuple(model.handle('edge', edge) for edge in shared), model.revision == before)
