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
    decoded, job, _ = pa._decode_packet(packet, classes, schema)
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
    decoded, _, _ = pa._decode_packet(encode_intersection_plan(closure, job_id='j'), classes, schema)
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
    assert "differ from the producer's route" in str(_refused('route', lambda: _adopt(model, _reseal(packet, downcast), faces, policy)))
    _refused('route', lambda: _adopt(model, packet, faces, IntersectionBatchPolicy()))
    _, exact = _worker(model, faces, IntersectionBatchPolicy())
    assert "producer's planning mode / seam policy" in str(_refused('route', lambda: _adopt(model, exact, faces, policy)))


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


# ------------------------------------------------------------------------------------------ review round 1

def _renumber(tree):
    """Re-assign object ordinals in decode (pre-)order; placeholders may be any hashable value."""
    mapping, counter = {}, [0]

    def walk(p):
        if p[0] == 'r':
            return ['r', mapping[p[1]]]
        if p[0] in ('t', 'l', 'a', 'p', 'R'):
            q = list(p)
            mapping[p[1]] = q[1] = counter[0]
            counter[0] += 1
            slot = 3 if p[0] == 'R' else 2
            if p[0] != 'a':
                q[slot] = [walk(c) for c in p[slot]]
            return q
        return p
    return walk(tree)


def test_r1_shared_reference_expansion_is_bounded_before_any_constructor_or_hash():
    import time
    model, faces = _plates()
    policy = IntersectionBatchPolicy()
    _, packet = _worker(model, faces, policy)
    depth = 22                                       # 2**22 expanded nodes in a ~2 kB body

    def bomb(tree):
        level = ['t', 'b0', [['f', struct_hex(1.0)]]]
        for k in range(1, depth):
            level = ['t', 'b%d' % k, [level, ['r', 'b%d' % (k - 1)]]]
        tree[3][6] = ['t', 'top', [level]]                # axes: nothing else refers into it
        tree[:] = _renumber(tree)
    tampered = _reseal(packet, bomb)
    assert len(tampered) < len(packet) + 4096
    start = time.perf_counter()
    error = _refused('packet', lambda: _adopt(model, tampered, faces, policy))
    assert 'expanded' in str(error) and time.perf_counter() - start < 5.0


def struct_hex(value):
    import struct
    return struct.pack('!d', value).hex()


def test_r1_encoder_refuses_an_oversized_expansion_and_a_cyclic_plan(monkeypatch):
    model, faces = _plates()
    plan = plan_intersections(model, faces, policy=IntersectionBatchPolicy())
    looped = []
    looped.append(looped)
    error = _refused('encode', lambda: encode_intersection_plan(replace(plan, arrangements=looped), job_id='j'))
    assert 'cyclic' in str(error)
    monkeypatch.setattr(pa, '_MAX_EXPANDED', 50)
    assert 'expanded' in str(_refused('encode', lambda: encode_intersection_plan(plan, job_id='j')))


def test_r1_same_revision_owner_edit_after_spawn_refuses():
    model, faces = _plates()
    policy = IntersectionBatchPolicy()
    spawn_checksum = _checksum(model)
    _, packet = _worker(model, faces, policy)
    vertex = next(iter(model.vertices.values()))
    model._vertices[vertex.id] = replace(vertex, position=vertex.position + np.array([-1e-3, 0., 0.]))
    assert _checksum(model) != spawn_checksum
    revision = model.revision
    _refused('source', lambda: _adopt(model, packet, faces, policy, checksum=spawn_checksum))
    assert model.revision == revision


def _plan_with(model_faces=None):
    model, faces = _plates() if model_faces is None else model_faces
    return plan_intersections(model, faces, policy=IntersectionBatchPolicy())


def test_r1_unknown_or_altered_record_state_refuses_at_encode():
    from anygeometry.surfaces import Plane
    plan = _plan_with()
    support = next(a.support for a in plan.arrangements if type(a.support) is Plane)
    object.__setattr__(support, 'extra', 1)
    _refused('encode', lambda: encode_intersection_plan(plan, job_id='j'))
    plan = _plan_with()
    support = next(a.support for a in plan.arrangements if type(a.support) is Plane)
    object.__setattr__(support, 'normal', support.normal * -1.0)          # cached normal not the recomputation
    _refused('encode', lambda: encode_intersection_plan(plan, job_id='j'))
    plan = _plan_with()
    support = next(a.support for a in plan.arrangements if type(a.support) is Plane)
    object.__setattr__(support, 'u_vector', support.u_vector * 2.0)        # the constructor would normalise it
    _refused('encode', lambda: encode_intersection_plan(plan, job_id='j'))


def _cylinders():
    from anygeometry.generators import cylinder
    model = cylinder(1., 4., origin=(0, 0, -2), circumferential_segments=4)
    model.insert_model(cylinder(0.6, 4., origin=(-2, 0, 0), axis=(1, 0, 0), radial_direction=(0, 1, 0),
                                circumferential_segments=4))
    return model, tuple(model.handle('face', f) for f in sorted(model.faces))


def test_r1_altered_cylinder_derived_state_refuses_and_cylinder_plans_round_trip():
    from anygeometry.surfaces import Cylinder
    model, faces = _cylinders()
    policy = IntersectionBatchPolicy()
    plan, packet = _worker(model, faces, policy)
    adopted = _adopt(model, packet, faces, policy)
    assert pa._same(adopted, plan)
    support = next(a.support for a in plan.arrangements if type(a.support) is Cylinder)
    object.__setattr__(support, '_circumferential', support._circumferential * -1.0)
    _refused('encode', lambda: encode_intersection_plan(plan, job_id='j'))


def test_r1_every_registry_class_round_trips_with_its_exact_type():
    from anygeometry.member_arrangements import MemberAxisArrangement, MemberPointContact
    from anygeometry.arrangement_geometry import LinePath
    from anygeometry.material_arrangement import ArrangementPoint
    from anygeometry.branch_curves import BezierQuadricCurve
    from anygeometry.quadric_algebra import QuadricSupport
    from anygeometry.plane_ruled import PlaneRuledSupport
    from anygeometry.batch_intersections import FacePointContact
    classes, _ = pa._schema()
    ruled = PlaneRuledSupport(base=(56.464392319893776, 0.6794350112397475, -17.5),
                              directrix_vector=(-0.007083671712961483, -0.0007445238982637945, 0.9999746333178796),
                              ruling=(-0.9944966676474915, -0.10452581172735692, -0.007122690557092655),
                              t0=-7.424585283355499, length=36.13361611880462)
    quadric = QuadricSupport(kind='cylinder', origin=(50.0, 0.0, -24.5), axis=(0.0, 0.0, 1.0), radius=6.5, slope=0.0,
                             matrix=(), linear=(), constant=0.0, u_vector=(), v_vector=())
    curve = BezierQuadricCurve(first=ruled, second=quadric, start=0.2054758444032841, sweep=0.6605876794881089,
                               branch=-1, parameterization='linear')
    curve.parameters_of(curve.evaluate(0.5) if hasattr(curve, 'evaluate') else (0., 0., 0.))   # warm the answer memo
    line = LinePath(np.array([0., 0, 0]), np.array([1., 0, 0]))
    loose = (curve, MemberAxisArrangement(3, (1, 2), line, (0.25, np.float64(0.5)), 1e-9),
             MemberPointContact(None, (0., 1., 2.), ((1, 0.5),), (4,), 1e-9), ArrangementPoint((0., 0., 0.), (1,)),
             FacePointContact((1, 2), (0., 0., 0.), 1e-9),
             np.int64(7), np.bool_(True), [1, 2], (True, None, 'x'))
    back = pa._decode_tree(json.loads(pa._canonical(pa._encode_tree(loose, classes))), classes)
    assert pa._same(back, loose) and back[0]._inversions == {}
    seen = {type(x).__name__ for x in _walk_records(back)}
    for plan in (_plan_with(), _plan_with(_cylinders()),
                 plan_intersections(*_plates(), policy=IntersectionBatchPolicy(seam_policy='ring_split'))):
        decoded = pa._decode_tree(json.loads(pa._canonical(pa._encode_tree(plan, classes))), classes)
        assert pa._same(decoded, plan)
        seen |= {type(x).__name__ for x in _walk_records(decoded)}
    assert set(classes) <= seen, set(classes) - seen


# ------------------------------------------------------------------------------------------ review round 2

def _downcast(tree):
    tree[2] = 'IntersectionPlan'
    del tree[3][len(pa._PLAN_FIELDS):]


def test_r2_downcast_plus_baseline_owner_route_refuses():
    model, faces = _plates()
    _, packet = _worker(model, faces, IntersectionBatchPolicy(seam_policy='ring_split'))
    _refused('route', lambda: _adopt(model, _reseal(packet, _downcast), faces, IntersectionBatchPolicy()))


def test_r2_envelope_route_must_match_the_decoded_plan_and_the_owner():
    model, faces = _plates()
    _, packet = _worker(model, faces, IntersectionBatchPolicy())
    envelope = json.loads(packet)
    envelope['route'] = ['exact', 'ring_split']
    assert "differ from the producer's route" in str(_refused('route', lambda: _adopt(model, pa._canonical(envelope), faces, IntersectionBatchPolicy(seam_policy='ring_split'))))
    _refused('route', lambda: _adopt(model, pa._canonical(envelope), faces, IntersectionBatchPolicy()))


# ------------------------------------------------------------------------------------------ review round 3

@pytest.mark.parametrize('payload', ['string', 'array'])
def test_r3_shared_large_payload_amplification_is_bounded_by_payload_weight(payload):
    import time
    model, faces = _plates()
    policy = IntersectionBatchPolicy()
    _, packet = _worker(model, faces, policy)
    if payload == 'string':
        leaf = ['s', 'x' * (1 << 20)]                                   # 1 MiB, one node
    else:
        leaf = ['a', 'leaf', [1 << 17], struct_hex(1.0) * (1 << 17), False]   # 1 MiB of float64, one node

    def bomb(tree):
        level = ['t', 'b0', [leaf]]
        for k in range(1, 6):                                           # 32 MiB expanded, ~50 expanded nodes
            level = ['t', 'b%d' % k, [level, ['r', 'b%d' % (k - 1)]]]
        tree[3][6] = ['t', 'top', [level]]
        tree[:] = _renumber(tree)
    tampered = _reseal(packet, bomb)
    start = time.perf_counter()
    error = _refused('packet', lambda: _adopt(model, tampered, faces, policy))
    assert 'expanded' in str(error) and time.perf_counter() - start < 10.0
