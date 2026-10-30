"""Genuine public no-recut batches, topology contacts and atomic publication."""
from dataclasses import replace

import pytest

from anygeometry import (GeometryError, IntersectionBatchPolicy, ConnectionIntent,
    plan_intersections, apply_intersections, has_current_intersection_preparation,
    clone_prepared_geometry, query_prepared_face_preimages, to_dict, from_dict,
    query_material_surface_regions, query_prepared_material_boundary_chords,
    validate_prepared_material_curvilinear_cells, query_prepared_material_source_transfer,
    validate_prepared_material_source_transfer_binding, CurvilinearMaterialCell)
from anygeometry.plane_nurbs import CURRENT_WORK
from anygeometry.section_audit import Work
from anygeometry.batch_intersections import _plan_content
from anygeometry.existing_material_preparation import _PLANS
from anygeometry.preparation_epochs import _permits, _seals
from anygeometry.prepared_material_source_transfer import _LINKS
from test_same_support_material_relation import flat_cut, closed_cut, polygon, flat, rectangle
from anygeometry import GeometryModel


def handles(model):
    return tuple(model.handle('face', i) for i in sorted(model.faces))


def imprint(**kwargs):
    return IntersectionBatchPolicy(intent=ConnectionIntent.IMPRINT, **kwargs)


@pytest.mark.parametrize('family', ('flat_cut', 'closed_cut'))
@pytest.mark.parametrize('loaded', (False, True))
def test_public_complete_no_recut_plan_apply_and_repeated_apply(request, family, loaded):
    original, ids = request.getfixturevalue(family)
    model = from_dict(to_dict(original)) if loaded else original.clone()
    before = to_dict(model)
    incidences = {edge: tuple(sorted(model.faces_using_edge(edge))) for edge in model.edges}
    policy = imprint()  # member flags remain their public True defaults: zero members
    plan = plan_intersections(model, handles(model), policy=policy)
    assert to_dict(model) == before
    result = apply_intersections(model, plan, policy=policy)
    assert result.reused and result.plan is plan
    assert has_current_intersection_preparation(model)
    assert to_dict(model) == before
    assert {edge: tuple(sorted(model.faces_using_edge(edge))) for edge in model.edges} == incidences
    assert {h.id for h in result.joint_edges} == {edge for edge, faces in incidences.items() if len(faces) > 1}
    again = apply_intersections(model, plan, policy=policy)
    assert again.reused and has_current_intersection_preparation(model)
    assert to_dict(model) == before


def test_same_sheet_connect_preserves_owner_and_does_not_fabricate_joint(flat_cut):
    model = flat_cut[0].clone()
    sheet = model.add_sheet(tuple(model.faces), name='current authored sheet')
    before = to_dict(model)
    policy = IntersectionBatchPolicy()  # actual default CONNECT flags
    plan = plan_intersections(model, handles(model), policy=policy)
    result = apply_intersections(model, plan, policy=policy)
    assert result.reused and has_current_intersection_preparation(model)
    assert tuple(model.sheets) == (sheet,) and not model.junctions and not model.attachments
    assert to_dict(model) == before


@pytest.mark.parametrize('owners', ('none', 'different_sheets'))
def test_connect_ownership_is_not_silently_invented(flat_cut, owners):
    model = flat_cut[0].clone()
    if owners == 'different_sheets':
        for face in model.faces: model.add_sheet((face,))
    before = to_dict(model)
    with pytest.raises(GeometryError):
        plan_intersections(model, handles(model), policy=IntersectionBatchPolicy())
    assert to_dict(model) == before and not has_current_intersection_preparation(model)


def test_reuse_existing_issues_complete_proof_without_changes(flat_cut):
    model = flat_cut[0].clone(); before = to_dict(model)
    policy = IntersectionBatchPolicy(intent=ConnectionIntent.REUSE_EXISTING)
    plan = plan_intersections(model, handles(model), policy=policy)
    assert apply_intersections(model, plan, policy=policy).reused
    assert has_current_intersection_preparation(model) and to_dict(model) == before


@pytest.mark.parametrize('kind', ('duplicate_boundary', 'endpoint_only', 'overlap'))
def test_geometric_contacts_without_actual_shared_topology_refuse(kind):
    model = GeometryModel(); support = flat()
    first = rectangle(0., .5, 0., .5)
    second = {'duplicate_boundary': rectangle(.5, 1., 0., .5),
              'endpoint_only': rectangle(.5, 1., .5, 1.),
              'overlap': rectangle(.25, .75, 0., .5)}[kind]
    polygon(model, support, first); polygon(model, support, second)
    before = to_dict(model)
    with pytest.raises(GeometryError): plan_intersections(model, handles(model), policy=imprint())
    assert to_dict(model) == before and not has_current_intersection_preparation(model)


@pytest.mark.parametrize('kind', ('copy', 'forged_empty', 'content', 'operands', 'policy_string'))
def test_unissued_or_tampered_plan_cannot_launder_complete_native_preparation(flat_cut, kind):
    model = flat_cut[0].clone(); policy = imprint()
    plan = plan_intersections(model, handles(model), policy=policy)
    if kind == 'copy': plan = replace(plan)
    if kind == 'forged_empty':
        plan = replace(plan, arrangements=(), content_checksum='')
        plan = replace(plan, content_checksum=_plan_content(plan))
    if kind == 'content': object.__setattr__(plan, 'content_checksum', 'changed')
    if kind == 'operands': object.__setattr__(plan, 'operands', tuple(reversed(plan.operands)))
    if kind == 'policy_string': object.__setattr__(plan.policy, 'intent', 'imprint')
    with pytest.raises(GeometryError): apply_intersections(model, plan, policy=policy)
    assert not has_current_intersection_preparation(model)


def test_plan_ledger_is_detached_immutable_complete_pair_data(flat_cut):
    model = flat_cut[0].clone(); plan = plan_intersections(model, handles(model), policy=imprint())
    ledger = _PLANS[id(plan)][6]
    assert type(ledger) is tuple and len(ledger) == 1 and type(ledger[0][2]) is tuple
    with pytest.raises(TypeError): ledger[0] = ()


@pytest.mark.parametrize('kind', ('intent', 'flag', 'cap'))
def test_equal_value_policy_type_changes_in_callback_refuse(flat_cut, kind):
    model = flat_cut[0].clone()
    policy = imprint(max_predicates=10000)
    name, value = {'intent': ('intent', 'imprint'), 'flag': ('face_connections', 1),
                   'cap': ('max_predicates', 10000.)}[kind]
    old = getattr(policy, name)
    def mutate(): object.__setattr__(policy, name, value)
    object.__setattr__(policy, 'cancellation_check', mutate)
    try:
        with pytest.raises(GeometryError): plan_intersections(model, handles(model), policy=policy)
    finally: object.__setattr__(policy, name, old)
    assert not has_current_intersection_preparation(model)


def test_effective_apply_policy_is_independently_pinned_and_budgeted(flat_cut):
    model = flat_cut[0].clone(); plan = plan_intersections(model, handles(model), policy=imprint())
    limited = imprint(max_predicates=1)
    with pytest.raises(GeometryError, match='budget'): apply_intersections(model, plan, policy=limited)
    malformed = imprint(); object.__setattr__(malformed, 'max_predicates', 0)
    with pytest.raises(GeometryError): apply_intersections(model, plan, policy=malformed)
    actual = imprint()
    def mutate(): object.__setattr__(actual, 'face_connections', 1)
    object.__setattr__(actual, 'cancellation_check', mutate)
    with pytest.raises(GeometryError): apply_intersections(model, plan, policy=actual)
    assert not has_current_intersection_preparation(model)


def _planning_switch(monkeypatch, switch):
    """T2-1b planning-mode switch: off, explicit field, or the environment variable."""
    from anygeometry.batch_intersections import PLANNING_MODE_ENVIRONMENT
    monkeypatch.delenv(PLANNING_MODE_ENVIRONMENT, raising=False)
    if switch == 'environment':
        monkeypatch.setenv(PLANNING_MODE_ENVIRONMENT, 'topology_certified')
        return {}
    return {} if switch == 'off' else {'planning_mode': switch}


@pytest.mark.parametrize('switch', ('topology_certified', 'environment', 'exact'))
def test_effective_apply_policy_is_pinned_with_planning_mode_set(flat_cut, monkeypatch, switch):
    # T2-1b ON (field or environment) and explicit exact: the mode-agnostic existing-material
    # route still pins the caller's own policy object, and the state-E plan still applies.
    on = _planning_switch(monkeypatch, switch)
    model = flat_cut[0].clone(); plan = plan_intersections(model, handles(model), policy=imprint(**on))
    assert plan.policy.planning_mode is None and not hasattr(plan, 'route_checksum')
    actual = imprint(**on)
    def mutate(): object.__setattr__(actual, 'max_predicates', 10**9)    # canonical type, new value
    object.__setattr__(actual, 'cancellation_check', mutate)
    with pytest.raises(GeometryError, match='policy or operands changed'):
        apply_intersections(model, plan, policy=actual)
    assert not has_current_intersection_preparation(model)
    apply_intersections(model, plan, policy=imprint(**on))
    assert has_current_intersection_preparation(model)


@pytest.mark.parametrize('switch', ('topology_certified', 'environment', 'exact'))
def test_planning_policy_change_is_detected_with_planning_mode_set(flat_cut, monkeypatch, switch):
    on = _planning_switch(monkeypatch, switch)
    model = flat_cut[0].clone(); policy = imprint(**on)
    def mutate(): object.__setattr__(policy, 'max_predicates', 10**9)    # canonical type, new value
    object.__setattr__(policy, 'cancellation_check', mutate)
    with pytest.raises(GeometryError, match='policy changed during planning'):
        plan_intersections(model, handles(model), policy=policy)
    assert not has_current_intersection_preparation(model)


@pytest.mark.parametrize('switch', ('off', 'topology_certified', 'environment', 'exact'))
def test_noncanonical_apply_policy_is_not_laundered_by_a_copy(flat_cut, monkeypatch, switch):
    on = _planning_switch(monkeypatch, switch)
    model = flat_cut[0].clone(); plan = plan_intersections(model, handles(model), policy=imprint(**on))
    actual = imprint(**on); object.__setattr__(actual, 'intent', 'imprint')     # equal value, str type
    with pytest.raises(GeometryError, match='noncanonical'): apply_intersections(model, plan, policy=actual)
    assert not has_current_intersection_preparation(model)


@pytest.mark.parametrize('switch', ('off', 'topology_certified', 'environment', 'exact'))
def test_mode_agnostic_routes_receive_the_callers_policy_object(flat_cut, monkeypatch, switch):
    # Identity contract: both mode-agnostic routes are handed the caller's policy object itself
    # on planning and on apply; the mode-free copy is only the stored policy, the mode a keyword.
    import anygeometry.existing_material_preparation as emp
    import anygeometry.bounded_nurbs as bn
    on = _planning_switch(monkeypatch, switch)
    model = flat_cut[0].clone(); policy = imprint(**on); seen = []
    real_try, real_apply = emp.try_plan_existing, emp.apply_existing
    def spy_try(model_, handles_, given, **kw):
        seen.append(('try_plan_existing', given, kw)); return real_try(model_, handles_, given, **kw)
    def spy_apply(model_, plan_, given, **kw):
        seen.append(('apply_existing', given, kw)); return real_apply(model_, plan_, given, **kw)
    monkeypatch.setattr(emp, 'try_plan_existing', spy_try)
    monkeypatch.setattr(emp, 'apply_existing', spy_apply)
    plan = plan_intersections(model, handles(model), policy=policy)
    assert plan.policy.planning_mode is None and (plan.policy is policy) == (switch in ('off', 'environment'))
    apply_intersections(model, plan, policy=policy)
    # The bounded route, reached with spies only (no bounded geometry work runs).
    monkeypatch.setattr(emp, 'try_plan_existing', lambda *a, **kw: None)
    monkeypatch.setattr(bn, 'handles_bounded', lambda *a: True)
    monkeypatch.setattr(bn, 'plan_bounded',
                        lambda model_, handles_, given, **kw: seen.append(('plan_bounded', given, kw)) or plan)
    assert plan_intersections(model, handles(model), policy=policy) is plan
    monkeypatch.setattr(emp, 'is_existing_plan', lambda plan_: False)
    monkeypatch.setattr(emp, 'eligible', lambda *a: False)
    monkeypatch.setattr(bn, 'is_bounded_plan', lambda plan_: True)
    monkeypatch.setattr(bn, 'apply_bounded',
                        lambda model_, plan_, given, **kw: seen.append(('apply_bounded', given, kw)) or 'sentinel')
    assert apply_intersections(model, plan, policy=policy) == 'sentinel'
    assert [name for name, _, _ in seen] == ['try_plan_existing', 'apply_existing', 'plan_bounded', 'apply_bounded']
    assert all(given is policy for _, given, _ in seen)
    for name, _, kw in seen:
        if name.startswith('apply'):
            assert kw == {'planning_mode': 'exact'}
        else:
            assert set(kw) == {'stored_policy'} and kw['stored_policy'].planning_mode is None
            assert (kw['stored_policy'] is policy) == (switch in ('off', 'environment'))
            assert kw['stored_policy'] == replace(policy, planning_mode=None)


# Every field IntersectionBatchPolicy.__post_init__ checks, set after construction (object.__setattr__).
MALFORMED_POLICY_FIELDS = (
    ('max_predicates', 0, 'predicate budget must be a positive integer'),
    ('face_connections', 1, 'face_connections policy must be boolean'),
    ('max_candidate_pairs', 0, 'candidate pair budget must be a positive integer'),
    ('member_connections', 1, 'member_connections policy must be boolean'),
    ('member_face_connections', 1, 'member_face_connections policy must be boolean'),
    ('cancellation_check', 1, 'cancellation check must be callable'),
    ('intent', 'imprint', 'noncanonical'),
    ('planning_mode', 'bogus', 'planning_mode must be one of'),
)


def test_bounded_apply_refuses_a_malformed_caller_policy_in_place():
    # Review T2-1b R7 (Codex P2 on e0646dd): the real bounded route (plan_intersections ->
    # plan_bounded, apply_intersections -> apply_bounded). Without the copy that ran __post_init__,
    # apply_bounded validated nothing itself (_validate_policy compares four fields, 1 == True), so
    # a malformed caller policy reached publication. It is now validated in place, never copied.
    from test_plane_split import bowl
    from anygeometry.bounded_nurbs import is_bounded_plan
    model, face, _plane = bowl()
    plate = model.add_plate([model.add_point(x, y, .5) for x, y in ((-2., -2.), (2., -2.), (2., 2.), (-2., 2.))])
    plan = plan_intersections(model, (model.handle('face', face), model.handle('face', plate)), policy=imprint())
    assert is_bounded_plan(plan)
    before = to_dict(model)
    for name, value, message in MALFORMED_POLICY_FIELDS:
        actual = imprint(); object.__setattr__(actual, name, value)
        with pytest.raises(GeometryError, match=message):
            apply_intersections(model, plan, policy=actual)
        assert model.revision == plan.revision and to_dict(model) == before, name
    result = apply_intersections(model, plan, policy=imprint())       # the plan itself still applies
    assert not result.reused and model.revision != plan.revision


@pytest.mark.parametrize('name, value', [c[:2] for c in MALFORMED_POLICY_FIELDS[:7]],
                         ids=[c[0] for c in MALFORMED_POLICY_FIELDS[:7]])
def test_existing_apply_refuses_a_malformed_caller_policy_through_its_pin(flat_cut, name, value):
    # Class check for R7: the existing-material apply also lost the copy in e0646dd; its own
    # _policy_pin covers every such field (the planning_mode field was overwritten by that copy, so
    # it was never checked on this route and is not part of this contract).
    model = flat_cut[0].clone(); plan = plan_intersections(model, handles(model), policy=imprint())
    actual = imprint(); object.__setattr__(actual, name, value)
    with pytest.raises(GeometryError, match='noncanonical|malformed|not callable'):
        apply_intersections(model, plan, policy=actual)
    assert not has_current_intersection_preparation(model)


def test_shared_work_and_policy_pair_caps_are_not_renewed(flat_cut):
    model = flat_cut[0].clone()
    token = CURRENT_WORK.set(Work(0, 12))
    try:
        with pytest.raises(GeometryError): plan_intersections(model, handles(model), policy=imprint())
    finally: CURRENT_WORK.reset(token)
    with pytest.raises(GeometryError):
        plan_intersections(model, handles(model), policy=imprint(max_predicates=1))


def test_private_callback_exception_and_cancellation_propagate(flat_cut):
    from anygeometry.same_support_material_relation import _Unsupported
    model = flat_cut[0].clone(); original = _Unsupported('caller exception')
    calls = 0
    def fail():
        nonlocal calls
        calls += 1
        if calls == 2: raise original
        return False
    with pytest.raises(_Unsupported) as caught:
        plan_intersections(model, handles(model), policy=imprint(cancellation_check=fail))
    assert caught.value is original
    with pytest.raises(GeometryError, match='cancelled'):
        plan_intersections(model, handles(model), policy=imprint(cancellation_check=lambda: True))


@pytest.mark.parametrize('kind', ('revision', 'native'))
def test_changed_source_after_planning_never_publishes(flat_cut, kind):
    from anygeometry.nurbs import CurveOnSurface
    model = flat_cut[0].clone(); policy = imprint()
    plan = plan_intersections(model, handles(model), policy=policy)
    curve, old = None, None
    if kind == 'revision':
        model.add_point(10., 10., 0.)
    else:
        curve = next(edge.curve.pcurve for edge in model.edges.values() if type(edge.curve) is CurveOnSurface)
        old = curve._prepared
        object.__setattr__(curve, '_prepared', replace(curve, weights=(1., 2.))._prepared)
    try:
        with pytest.raises(GeometryError): apply_intersections(model, plan, policy=policy)
        assert not hasattr(model, '_intersection_preparation_receipt')
    finally:
        if curve is not None: object.__setattr__(curve, '_prepared', old)


@pytest.mark.parametrize('stage', ('checksum', 'face_receipt', 'edge_receipt', 'source_lineage', 'epoch_consume'))
@pytest.mark.parametrize('prior', (False, True))
def test_publication_failure_restores_entire_caller_and_prior_transient_authority(flat_cut, monkeypatch, stage, prior):
    source = from_dict(to_dict(flat_cut[0]))
    model = clone_prepared_geometry(source, new_preparation_epoch=True)
    policy = imprint()
    if prior:
        old = plan_intersections(model, handles(model), policy=policy)
        apply_intersections(model, old, policy=policy)
    plan = plan_intersections(model, handles(model), policy=policy)
    before, state = to_dict(model), dict(vars(model))
    seal, permit, links = _seals.get(model), _permits.get(model), _LINKS.get(model)
    if stage == 'checksum':
        import anygeometry.translational_imprint as target
        real = target._checksum
        monkeypatch.setattr(target, '_checksum', lambda actual: 'injected mismatch' if actual is model else real(actual))
    else:
        import importlib
        module, name = {'face_receipt': ('prepared_face_preimages', '_publish_application_preimages'),
                        'edge_receipt': ('edge_subcurve_preimages', '_publish_edge_subcurve_preimages'),
                        'source_lineage': ('prepared_material_source_transfer', '_publish_source_lineage'),
                        'epoch_consume': ('preparation_epochs', '_consume_epoch')}[stage]
        target = importlib.import_module('anygeometry.'+module); real = getattr(target, name)
        def fail(*args, **kwargs):
            real(*args, **kwargs)
            raise GeometryError('injected publication failure')
        monkeypatch.setattr(target, name, fail)
    with pytest.raises(GeometryError): apply_intersections(model, plan, policy=policy)
    assert to_dict(model) == before
    for name, value in state.items(): assert vars(model)[name] is value
    assert _seals.get(model) == seal and _permits.get(model) == permit and _LINKS.get(model) is links
    assert has_current_intersection_preparation(model) is prior


def test_loaded_source_epoch_keeps_current_identity_preimages_and_genuine_source_cell_route(flat_cut):
    source = from_dict(to_dict(flat_cut[0])); original = to_dict(source)
    model = clone_prepared_geometry(source, new_preparation_epoch=True)
    policy = imprint(); plan = plan_intersections(model, handles(model), policy=policy)
    apply_intersections(model, plan, policy=policy)
    binding = query_prepared_face_preimages(model)
    assert binding.authored_face_ids == tuple(sorted(source.faces))
    assert binding.face_descendants == tuple((i, (i,)) for i in sorted(source.faces))
    receipts = []
    for face in model.faces.values():
        selected = model.handle('face', face.id)
        regions = query_material_surface_regions(model, [selected])
        chords = query_prepared_material_boundary_chords(model, regions,
            tuple((model.handle('edge', use.edge), 0., 1.) for use in face.loop), chord_error=.01)
        by_edge = {row.edge.id: (index, row) for index, row in enumerate(chords.intervals)}
        uv, xyz, ids, intervals = [], [], [], []
        for side, use in enumerate(face.loop):
            index, chord = by_edge[use.edge]
            occurrence = next(row for row in chord.occurrences if row.face == selected)
            endpoint = 0 if use.forward else 1
            uv.append(occurrence.uv[endpoint]); xyz.append(chord.xyz[endpoint])
            ids.append(model.oriented_start_vertex(use)); intervals.append((side, index, use.forward))
        cell = CurvilinearMaterialCell(selected, tuple(ids), tuple(uv), tuple(xyz), tuple(intervals))
        receipts.append(validate_prepared_material_curvilinear_cells(model, regions, chords, (cell,), surface_error=.0025))
    transfer = query_prepared_material_source_transfer(source, model, tuple(receipts))
    validate_prepared_material_source_transfer_binding(source, model, transfer, cell_receipts=tuple(receipts))
    assert to_dict(source) == original


def test_baseexception_after_real_epoch_consumption_restores_all_authority(flat_cut, monkeypatch):
    import anygeometry.preparation_epochs as epochs
    class StopPublication(BaseException): pass
    failure = StopPublication('after real epoch consumption')
    source = from_dict(to_dict(flat_cut[0])); source_before = to_dict(source)
    model = clone_prepared_geometry(source, new_preparation_epoch=True)
    policy = imprint(); plan = plan_intersections(model, handles(model), policy=policy)
    before, state = to_dict(model), dict(vars(model))
    seal, permit, links = _seals.get(model), _permits.get(model), _LINKS.get(model)
    real = epochs._consume_epoch
    def fail(actual):
        real(actual)
        raise failure
    monkeypatch.setattr(epochs, '_consume_epoch', fail)
    with pytest.raises(StopPublication) as caught:
        apply_intersections(model, plan, policy=policy)
    assert caught.value is failure
    assert to_dict(model) == before and to_dict(source) == source_before
    assert set(vars(model)) == set(state)
    for name, value in state.items(): assert vars(model)[name] is value
    assert _seals.get(model) == seal and _permits.get(model) == permit and _LINKS.get(model) is links
    assert not has_current_intersection_preparation(model)


def test_planning_inside_an_open_transaction_uses_the_general_route():
    """Automation plans and applies every command inside model.transaction(). Publication of the
    strict same-chart route is not journaled (global seals/permits), so it must not be offered
    there; ordinary crossing plates must still plan through the general route."""
    model = GeometryModel()
    horizontal = model.add_plate(model.add_points(((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0))))
    vertical = model.add_plate(model.add_points(((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1))))
    faces = (model.handle("face", horizontal), model.handle("face", vertical))
    with model.transaction():
        plan = plan_intersections(model, faces, policy=imprint())
    assert id(plan) not in _PLANS
