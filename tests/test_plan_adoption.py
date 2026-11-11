"""Plan transport (design A): closed plan codec and the owner's verified adoption of a worker plan.

The worker plans only; the owner decodes, verifies against its OWN model/operands/policy and runs the
ordinary apply, which mints every receipt. These tests fail if a proof could be forged or a stale or
foreign plan adopted, and if the codec loses a record type, a bit or a sharing relation.
"""
import json
import hashlib
import inspect
import pickle
import re
from dataclasses import replace
from uuid import UUID

import numpy as np
import pytest

from anygeometry import GeometryModel, GeometryError, to_dict, from_dict, has_current_intersection_preparation
from anygeometry.batch_intersections import (plan_intersections, apply_intersections, IntersectionBatchPolicy,
                                             IntersectionPlan, TopologyCertifiedPlan, _plan_content)
from anygeometry.edge_subcurve_preimages import query_prepared_edge_subcurve_preimages
from anygeometry.structural import ConnectionIntent
from anygeometry import plan_adoption as pa
from anygeometry.plan_adoption import PlanAdoptionRefused, adopt_intersection_plan, encode_intersection_plan
from anygeometry import preparation_epochs as pe

RECEIPTS = ('_intersection_preparation_receipt', '_intersection_application_receipt',
            '_prepared_face_preimages_receipt', '_edge_subcurve_preimages_receipt')


def _plates():
    model = GeometryModel()
    faces = []
    for points in (((0, 0, 0), (5, 0, 0), (5, 5, 0), (0, 5, 0)),
                   ((0, 2.5, -2.5), (5, 2.5, -2.5), (5, 2.5, 2.5), (0, 2.5, 2.5)),
                   ((2.5, 0, -2.5), (2.5, 5, -2.5), (2.5, 5, 2.5), (2.5, 0, 2.5))):
        faces.append(model.add_plate(model.add_points(points)))
    return model, tuple(model.handle('face', f) for f in faces)


def _checksum(model):
    return to_dict(model)['checksum']['value']


def _worker(model, operands, policy, job='job-1'):
    """What the worker does: plan on its own copy (from the document), encode with its own policy."""
    child = from_dict(to_dict(model))
    plan = plan_intersections(child, [child.handle(h.kind, h.id) for h in operands], policy=policy)
    return plan, encode_intersection_plan(plan, job_id=job)


def _adopt(model, packet, operands, policy, job='job-1', checksum=None):
    return adopt_intersection_plan(model, packet, policy=policy, operands=operands, job_id=job,
                                   expected_source_checksum=_checksum(model) if checksum is None else checksum)


def _reseal(packet, edit):
    """Tamper with the plan body and recompute the envelope digest (a self-consistent packet)."""
    envelope = json.loads(packet)
    edit(envelope['plan'])
    envelope['plan_sha256'] = hashlib.sha256(pa._canonical(envelope['plan'])).hexdigest()
    return pa._canonical(envelope)


def _refused(reason, call):
    with pytest.raises(PlanAdoptionRefused) as info:
        call()
    assert info.value.reason == reason, str(info.value)
    assert isinstance(info.value, GeometryError)
    return info.value


def _receipt_state(model):
    return tuple((name, getattr(model, name, None)) for name in RECEIPTS)


def _registries():
    from anygeometry import existing_material_preparation as emp, bounded_nurbs as bn
    return (dict(pe._plan_bindings), dict(pe._seals), dict(pe._permits), dict(emp._PLANS), dict(bn._PLANS))


def _walk_records(value):
    seen, out = set(), []
    stack = [value]
    while stack:
        v = stack.pop()
        if id(v) in seen or type(v) in (int, float, str, bool, type(None), np.float64, UUID):
            continue
        seen.add(id(v))
        out.append(v)
        if type(v) in (tuple, list):
            stack.extend(v)
        elif hasattr(type(v), '__dataclass_fields__'):
            stack.extend(getattr(v, f) for f in type(v).__dataclass_fields__ if f != 'cancellation_check')
    return out


# ------------------------------------------------------------------------------------------- codec identity

@pytest.mark.parametrize('seam', [None, 'ring_split'])
def test_codec_round_trip_keeps_exact_type_bits_content_and_sharing(seam):
    model, faces = _plates()
    policy = IntersectionBatchPolicy(seam_policy=seam)
    plan = plan_intersections(model, faces, policy=policy)
    assert type(plan) is (IntersectionPlan if seam is None else TopologyCertifiedPlan)
    packet = encode_intersection_plan(plan, job_id='j')
    assert type(packet) is bytes
    classes, schema = pa._schema()
    decoded, job = pa._decode_packet(packet, classes, schema)
    assert job == 'j' and type(decoded) is type(plan) and decoded is not plan
    assert pa._same(decoded, plan)
    assert _plan_content(decoded) == plan.content_checksum == decoded.content_checksum
    assert encode_intersection_plan(decoded, job_id='j') == packet          # canonical, deterministic
    # Every record keeps its exact class; sharing is kept (same number of distinct objects).
    original, copy = _walk_records(plan), _walk_records(decoded)
    assert [type(x) for x in original] == [type(x) for x in copy]
    assert len(original) == len(copy)
    assert decoded.policy.cancellation_check is None


def test_every_registry_record_type_round_trips_exactly():
    from anygeometry.member_arrangements import MemberAxisArrangement, MemberPointContact
    from anygeometry.arrangement_geometry import LinePath
    from anygeometry.material_arrangement import ArrangementPoint
    from anygeometry.batch_intersections import FacePointContact
    classes, _ = pa._schema()
    names, _ = zip(*pa._RECORD_FIELDS)
    assert set(names) == set(classes)
    line = LinePath(np.array([0., 0, 0]), np.array([1., 0, 0]))
    values = (MemberAxisArrangement(3, (1, 2), line, (0.25, np.float64(0.5)), 1e-9),
              MemberPointContact(None, (0., 1., 2.), ((1, 0.5),), (4,), 1e-9),
              ArrangementPoint((0., 0., 0.), (1,)), FacePointContact((1, 2), (0., 0., 0.), 1e-9),
              np.int64(7), np.bool_(True), [1, 2], (True, None, 'x'))
    tree = pa._encode_tree(values, classes)
    back = pa._decode_tree(json.loads(pa._canonical(tree)), classes)
    assert pa._same(back, values) and [type(x) for x in back] == [type(x) for x in values]


def test_callables_never_cross_and_the_policy_is_rebuilt_from_plain_fields():
    model, faces = _plates()
    calls = []
    plain = plan_intersections(model, faces, policy=IntersectionBatchPolicy())
    closure = plan_intersections(model, faces, policy=IntersectionBatchPolicy(cancellation_check=lambda: calls.append(1)))
    assert closure.policy.cancellation_check is not None
    assert encode_intersection_plan(closure, job_id='j') == encode_intersection_plan(plain, job_id='j')
    classes, schema = pa._schema()
    decoded, _ = pa._decode_packet(encode_intersection_plan(closure, job_id='j'), classes, schema)
    assert decoded.policy.cancellation_check is None
    assert decoded.policy == replace(closure.policy, cancellation_check=None)


@pytest.mark.parametrize('bad', [b'', b'\x80\x04', b'not json', b'[]', pickle.dumps({'a': 1})])
def test_garbage_truncated_and_pickle_packets_refuse(bad):
    model, faces = _plates()
    _refused('packet', lambda: _adopt(model, bad, faces, IntersectionBatchPolicy()))


def test_truncated_noncanonical_unknown_and_foreign_schema_packets_refuse():
    model, faces = _plates()
    policy = IntersectionBatchPolicy()
    _, packet = _worker(model, faces, policy)
    _refused('packet', lambda: _adopt(model, packet[:-10], faces, policy))
    _refused('packet', lambda: _adopt(model, packet.replace(b',', b', ', 1), faces, policy))
    envelope = json.loads(packet)
    envelope['schema'] = '0' * 64
    _refused('codec', lambda: _adopt(model, pa._canonical(envelope), faces, policy))
    envelope = json.loads(packet)
    envelope['extra'] = 1
    _refused('packet', lambda: _adopt(model, pa._canonical(envelope), faces, policy))

    def unknown_record(tree):
        tree[3][5][2][0][2] = 'Pickler'                     # first arrangement record name
    _refused('packet', lambda: _adopt(model, _reseal(packet, unknown_record), faces, policy))

    def missing_field(tree):
        tree[3][5][2][0][3].pop()
    _refused('packet', lambda: _adopt(model, _reseal(packet, missing_field), faces, policy))

    def callable_tag(tree):
        tree[3][4][2][0] = ['callable', 'os.system']
    _refused('packet', lambda: _adopt(model, _reseal(packet, callable_tag), faces, policy))


def test_body_digest_is_checked_before_decode():
    model, faces = _plates()
    policy = IntersectionBatchPolicy()
    _, packet = _worker(model, faces, policy)
    text = packet.decode('ascii')
    match = re.search(r'\["f","([0-9a-f]{16})"\]', text)
    flipped = text[:match.start(1)] + format(int(match.group(1), 16) ^ 1, '016x') + text[match.end(1):]
    _refused('packet', lambda: _adopt(model, flipped.encode('ascii'), faces, policy))


# --------------------------------------------------------------------------- design section 5 forge/stale tests

def test_1_receipt_forged_onto_a_loaded_document_has_no_edge_provenance_and_the_route_never_mints():
    model, faces = _plates()
    policy = IntersectionBatchPolicy()
    plan = plan_intersections(model, faces, policy=policy)
    apply_intersections(model, plan, policy=policy)
    assert query_prepared_edge_subcurve_preimages(model)
    loaded = from_dict(to_dict(model))
    # Test double: what a route that minted from the child's document would do.
    object.__setattr__(loaded, '_intersection_preparation_receipt',
                       (plan, loaded.revision, _checksum(loaded), model._intersection_preparation_receipt[3]))
    with pytest.raises(GeometryError):
        query_prepared_edge_subcurve_preimages(loaded)
    # The adoption module has no receipt- or registry-writing path at all.
    source = inspect.getsource(pa)
    for name in RECEIPTS + ('_plan_bindings', '_seals', '_permits', '_PLANS', '_issue_native_plan', '_issue_epoch'):
        assert not re.search(re.escape(name) + r'\s*(\[[^\]]*\])?\s*=(?!=)', source), name
    assert 'setattr' not in source and '__dict__[' not in source


def test_2_adopt_alone_touches_no_receipt_or_registry_then_the_owner_apply_mints_on_the_adopted_plan():
    model, faces = _plates()
    serial = from_dict(to_dict(model))
    serial_plan = plan_intersections(serial, [serial.handle('face', h.id) for h in faces], policy=IntersectionBatchPolicy())
    apply_intersections(serial, serial_plan, policy=IntersectionBatchPolicy())
    policy = IntersectionBatchPolicy(cancellation_check=lambda: False)
    _, packet = _worker(model, faces, policy)
    before_doc, before_receipts, before_registries = to_dict(model), _receipt_state(model), _registries()
    plan = _adopt(model, packet, faces, policy)
    assert _receipt_state(model) == before_receipts and all(v is None for _, v in before_receipts)
    assert _registries() == before_registries
    assert to_dict(model) == before_doc and not has_current_intersection_preparation(model)
    apply_intersections(model, plan, policy=policy)
    assert model._intersection_preparation_receipt[0] is plan
    assert has_current_intersection_preparation(model)
    assert query_prepared_edge_subcurve_preimages(model) == query_prepared_edge_subcurve_preimages(serial)
    assert to_dict(model)['checksum'] == to_dict(serial)['checksum']           # bit-identical to serial


def test_2b_wrong_operands_refuse():
    model, faces = _plates()
    policy = IntersectionBatchPolicy()
    _, packet = _worker(model, faces, policy)
    _refused('operands', lambda: _adopt(model, packet, faces[:2], policy))
    _, smaller = _worker(model, faces[:2], policy)
    _refused('operands', lambda: _adopt(model, smaller, faces, policy))


def test_2c_journaled_model_refuses_before_decode():
    model, faces = _plates()
    policy = IntersectionBatchPolicy()
    _, packet = _worker(model, faces, policy)
    with model.transaction():
        _refused('journaled_model', lambda: _adopt(model, packet, faces, policy, checksum='0' * 64))


def test_2c_epoch_model_refuses_and_an_epoch_issued_plan_cannot_be_encoded():
    from anygeometry.preparation_epochs import _issue_epoch
    model, faces = _plates()
    policy = IntersectionBatchPolicy()
    _, packet = _worker(model, faces, policy)
    _issue_epoch(model)
    assert pe._epoch_plan_required(model)
    _refused('epoch_model', lambda: _adopt(model, packet, faces, policy))
    epoch_plan = plan_intersections(model, faces, policy=policy)
    assert id(epoch_plan) in pe._plan_bindings
    _refused('encode', lambda: encode_intersection_plan(epoch_plan, job_id='j'))


def test_2c_existing_material_eligible_operands_refuse_and_its_plan_cannot_be_encoded():
    from test_same_support_material_relation import flat, polygon, rectangle
    from anygeometry.plane_split import plan_plane_split, apply_plane_split
    from anygeometry.surfaces import Plane
    model = GeometryModel()
    face = polygon(model, flat(), rectangle(0., 1.))
    result = apply_plane_split(model, plan_plane_split(model, [face], Plane((.6, 0., 0.), (0., 1., 0.), (0., 0., 1.))),
                               policy='imprint')
    handles = tuple(model.handle('face', i) for i in sorted(dict(result.parent_children)[face]))
    policy = IntersectionBatchPolicy(intent=ConnectionIntent.IMPRINT)
    plan = plan_intersections(model, handles, policy=policy)
    _refused('encode', lambda: encode_intersection_plan(plan, job_id='j'))
    plates, faces = _plates()
    _, packet = _worker(plates, faces, IntersectionBatchPolicy())
    _refused('registry_route', lambda: _adopt(model, packet, handles, policy))


def test_2c_bounded_eligible_operands_refuse_and_its_plan_cannot_be_encoded():
    from test_bounded_nurbs import fixture
    model, wall, plate = fixture()
    handles = (model.handle('face', wall), model.handle('face', plate))
    policy = IntersectionBatchPolicy(intent=ConnectionIntent.IMPRINT)
    plan = plan_intersections(model, handles, policy=policy)
    _refused('encode', lambda: encode_intersection_plan(plan, job_id='j'))
    plates, faces = _plates()
    _, packet = _worker(plates, faces, IntersectionBatchPolicy())
    _refused('registry_route', lambda: _adopt(model, packet, handles, policy))


def test_2d_downcast_of_a_topology_certified_plan_refuses():
    model, faces = _plates()
    policy = IntersectionBatchPolicy(seam_policy='ring_split')
    plan, packet = _worker(model, faces, policy)
    assert type(plan) is TopologyCertifiedPlan

    def downcast(tree):
        tree[2] = 'IntersectionPlan'
        del tree[3][len(pa._PLAN_FIELDS):]
    _refused('route', lambda: _adopt(model, _reseal(packet, downcast), faces, policy))
    _refused('route', lambda: _adopt(model, packet, faces, IntersectionBatchPolicy()))
    _, exact = _worker(model, faces, IntersectionBatchPolicy())
    _refused('route', lambda: _adopt(model, exact, faces, policy))


def test_3_parent_edited_after_spawn_refuses_and_serial_gives_the_serial_result():
    model, faces = _plates()
    policy = IntersectionBatchPolicy()
    spawn_checksum = _checksum(model)
    _, packet = _worker(model, faces, policy)
    reference = from_dict(to_dict(model))
    model.add_point(9, 9, 9)
    reference.add_point(9, 9, 9)
    _refused('revision', lambda: _adopt(model, packet, faces, policy, checksum=spawn_checksum))
    plan = plan_intersections(model, faces, policy=policy)          # the serial fallback
    apply_intersections(model, plan, policy=policy)
    ref_plan = plan_intersections(reference, [reference.handle('face', h.id) for h in faces], policy=policy)
    apply_intersections(reference, ref_plan, policy=policy)
    assert to_dict(model)['checksum'] == to_dict(reference)['checksum']


def test_3b_same_revision_source_edit_and_stale_job_checksum_refuse():
    model, faces = _plates()
    policy = IntersectionBatchPolicy()
    _, packet = _worker(model, faces, policy)
    _refused('source', lambda: _adopt(model, packet, faces, policy, checksum='a' * 64))
    # A plan bound to another source at the same model id and revision: the source binding refuses.

    def stale_source(tree):
        tree[3][2] = ['s', 'b' * 64]
    _refused('source', lambda: _adopt(model, _reseal(packet, stale_source), faces, policy))


def test_4_flipped_coordinate_is_a_content_refusal():
    model, faces = _plates()
    policy = IntersectionBatchPolicy()
    _, packet = _worker(model, faces, policy)

    def flip(tree):
        text = json.dumps(tree)
        match = re.search(r'\["f", "([0-9a-f]{16})"\]', text)
        value = format(int(match.group(1), 16) ^ 1, '016x')
        tree[:] = json.loads(text[:match.start(1)] + value + text[match.end(1):])
    _refused('content', lambda: _adopt(model, _reseal(packet, flip), faces, policy))


def test_6_plan_from_another_model_or_revision_refuses():
    model, faces = _plates()
    policy = IntersectionBatchPolicy()
    other, other_faces = _plates()
    _, foreign = _worker(other, other_faces, policy)
    _refused('operands', lambda: _adopt(model, foreign, faces, policy))
    _, packet = _worker(model, faces, policy)

    def other_id(tree):
        tree[3][0] = ['u', str(other.model_id)]
    _refused('model', lambda: _adopt(model, _reseal(packet, other_id), faces, policy))

    def other_revision(tree):
        tree[3][1] = ['i', tree[3][1][1] + 1]
    _refused('revision', lambda: _adopt(model, _reseal(packet, other_revision), faces, policy))


def test_policy_and_job_mismatches_refuse():
    model, faces = _plates()
    _, packet = _worker(model, faces, IntersectionBatchPolicy(face_connections=False))
    _refused('policy', lambda: _adopt(model, packet, faces, IntersectionBatchPolicy()))
    _, budget = _worker(model, faces, IntersectionBatchPolicy(max_predicates=10**9))
    _refused('policy', lambda: _adopt(model, budget, faces, IntersectionBatchPolicy()))
    _, packet = _worker(model, faces, IntersectionBatchPolicy())
    _refused('job', lambda: _adopt(model, packet, faces, IntersectionBatchPolicy(), job='job-2'))
    _, unbound = _worker(model, faces, IntersectionBatchPolicy(), job=None)
    _refused('job', lambda: _adopt(model, unbound, faces, IntersectionBatchPolicy()))
    _refused('argument', lambda: adopt_intersection_plan(model, packet, policy=ConnectionIntent.CONNECT,
                                                         operands=faces, job_id='job-1',
                                                         expected_source_checksum=_checksum(model)))
    _refused('argument', lambda: _adopt(model, packet.decode('ascii'), faces, IntersectionBatchPolicy()))


def test_7_registry_and_foreign_plan_objects_cannot_be_encoded():
    _refused('encode', lambda: encode_intersection_plan(object(), job_id='j'))
    model, faces = _plates()
    plan = plan_intersections(model, faces, policy=IntersectionBatchPolicy())
    _refused('encode', lambda: encode_intersection_plan(replace(plan, arrangements=()), job_id='j'))
    _refused('argument', lambda: encode_intersection_plan(plan, job_id=''))


def test_8_codec_schema_drift_disables_the_codec(monkeypatch):
    model, faces = _plates()
    plan = plan_intersections(model, faces, policy=IntersectionBatchPolicy())
    monkeypatch.setattr(pa, '_RECORD_FIELDS', pa._RECORD_FIELDS[:-1] + (('PlaneRuledSupport', ('base',)),))
    _refused('codec', lambda: encode_intersection_plan(plan, job_id='j'))


def test_default_path_does_not_import_the_adoption_module():
    import anygeometry
    import anygeometry.batch_intersections as bi
    for module in (anygeometry, bi):
        assert 'plan_adoption' not in inspect.getsource(module)


@pytest.mark.parametrize('policy', [IntersectionBatchPolicy(planning_mode='topology_certified'),
                                    IntersectionBatchPolicy(seam_policy='ring_split'),
                                    IntersectionBatchPolicy(intent=ConnectionIntent.IMPRINT)],
                         ids=['topology_certified', 'ring_split', 'imprint'])
def test_route_plans_adopt_and_apply_bit_identically_to_serial(policy):
    model, faces = _plates()
    serial = from_dict(to_dict(model))
    serial_plan = plan_intersections(serial, [serial.handle('face', h.id) for h in faces], policy=policy)
    serial_result = apply_intersections(serial, serial_plan, policy=policy)
    _, packet = _worker(model, faces, policy)
    plan = _adopt(model, packet, faces, policy)
    assert type(plan) is type(serial_plan) and plan.content_checksum == serial_plan.content_checksum
    result = apply_intersections(model, plan, policy=policy)
    assert type(result) is type(serial_result)
    assert to_dict(model)['checksum'] == to_dict(serial)['checksum']
    assert has_current_intersection_preparation(model)
    assert model._intersection_preparation_receipt[0] is plan
