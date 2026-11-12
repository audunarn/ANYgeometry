"""Exact trusted runtime transfer, integrity and independent mutability."""
from dataclasses import replace
import hashlib
import multiprocessing as mp
import pickle

import numpy as np
import pytest

from anygeometry.entities import EntityRef
from anygeometry.errors import GeometryError
from anygeometry.model import GeometryModel
from anygeometry.model_transport import capture_model_transfer, restore_model_transfer
from anygeometry.serialization import to_dict
from anygeometry.strict_audit import strict_audit
from anygeometry.structural import AttachmentKind, AttachmentTargetKind, ParameterRange


def fixture(kind):
    model = GeometryModel()
    if kind == 'curved':
        p = model.add_points(((0, 0, 0), (.5, .5, 0), (1, 0, 0)))
        model.add_arc(*p)
    else:
        p = model.add_points(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)))
        face = model.add_plate(p)
        if kind == 'structural':
            axis = model.add_points(((.2, 1, 0), (1.8, 1, 0)))
            edge = model.add_line(*axis)
            member = model.add_member([edge])
            model.add_attachment(member, AttachmentKind.MEMBER_ON_FACE, AttachmentTargetKind.FACE, face,
                                 ParameterRange(0, 1), (ParameterRange(.1, .9), ParameterRange.point(.5)))
        model.set_face_metadata(face, {'name': 'face', 'nested': {'values': (1, 2)}})
    model.add_to_group('points', [EntityRef('vertex', p[0])])
    model.set_document_settings(local_origin=(10, 20, 30), crs_metadata={'authority': 'test'})
    return model


@pytest.mark.parametrize('kind', ['healthy', 'curved', 'structural'])
def test_persisted_definition_and_entire_raw_audit_are_exact(kind):
    model = fixture(kind)
    original = to_dict(model)
    audit = strict_audit(model).to_dict()
    revision = model.revision
    transfer = capture_model_transfer(model)
    restored = restore_model_transfer(transfer)
    assert to_dict(restored) == original
    assert strict_audit(restored).to_dict() == audit
    assert restored.model_id == model.model_id and restored.revision == revision
    assert restored.id_state() == model.id_state()
    assert model.revision == revision and to_dict(model) == original


def test_transfer_is_immutable_detached_and_restorations_do_not_share_arrays():
    model = fixture('healthy')
    transfer = capture_model_transfer(model)
    first, second = restore_model_transfer(transfer), restore_model_transfer(transfer)
    first.move_point(1, -.1, 0)
    assert np.array_equal(model.vertex_position(1), [0, 0, 0])
    assert np.array_equal(second.vertex_position(1), [0, 0, 0])
    assert not np.shares_memory(first.vertex_position(2), second.vertex_position(2))
    assert not second.faces[1].surface.origin.flags.writeable
    assert not second.vertices[1].position.flags.writeable


def test_standard_pickle_roundtrip_and_version_hash_envelope_refusals():
    transfer = capture_model_transfer(fixture('healthy'))
    assert to_dict(restore_model_transfer(pickle.loads(pickle.dumps(transfer)))) == to_dict(restore_model_transfer(transfer))
    with pytest.raises(GeometryError, match='version'):
        restore_model_transfer(replace(transfer, version=999))
    with pytest.raises(GeometryError, match='checksum'):
        restore_model_transfer(replace(transfer, payload=transfer.payload + b'corrupt'))
    with pytest.raises(GeometryError, match='envelope'):
        restore_model_transfer(replace(transfer, revision=transfer.revision + 1))
    with pytest.raises(TypeError):
        restore_model_transfer(transfer.payload)


def test_bad_recomputed_payload_still_cannot_bypass_topology_validation():
    transfer = capture_model_transfer(fixture('healthy'))
    state = pickle.loads(transfer.payload)
    state['design']['topology']['vertices'].pop(1)
    data = pickle.dumps(state, protocol=5)
    with pytest.raises(GeometryError):
        restore_model_transfer(replace(transfer, payload=data, source_checksum=hashlib.sha256(data).hexdigest()))


def test_runtime_checksum_excludes_derived_surface_caches():
    model = fixture('healthy')
    first = capture_model_transfer(model)
    _ = model.faces[1].surface.normal
    second = capture_model_transfer(model)
    assert first.source_checksum == second.source_checksum


def _restore_in_child(connection, transfer):
    model = restore_model_transfer(transfer)
    connection.send((str(model.model_id), model.revision, to_dict(model)))
    connection.close()


def test_importable_standard_multiprocessing_restore():
    model = fixture('structural')
    context = mp.get_context('spawn')
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_restore_in_child, args=(writer, capture_model_transfer(model)))
    process.start()
    writer.close()
    try:
        assert reader.poll(10)
        identity, revision, document = reader.recv()
        assert identity == str(model.model_id) and revision == model.revision
        assert document == to_dict(model)
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        reader.close()
    assert not process.is_alive()


def test_inflight_capture_refuses_and_preparation_receipt_is_invalidated():
    model = fixture('healthy')
    with model.transaction():
        with pytest.raises(GeometryError, match='committed'):
            capture_model_transfer(model)
    model._intersection_preparation_receipt = object()
    restored = restore_model_transfer(capture_model_transfer(model))
    assert restored._anyfem_transport_requires_preparation
    assert not hasattr(restored, '_intersection_preparation_receipt')


def test_feature_intent_lineage_allocator_settings_and_extensions_are_exact():
    from anygeometry.features import FeatureOutputRef
    model = GeometryModel()
    history = model.features
    history.capture_baseline(model)
    first = history.append('geometry.point', parameters={'position': [0., 0., 0.]})
    second = history.append('geometry.point', parameters={'position': [1., 0., 0.]})
    history.append('geometry.line', inputs={'start': [FeatureOutputRef(first.feature_id, 'point', 'vertex')],
                                           'end': [FeatureOutputRef(second.feature_id, 'point', 'vertex')]})
    assert model.regenerate_features().success
    extra = model.add_point(10, 10)
    model.remove_vertex(extra)
    model.reserve_id_state({'vertex': 100, 'edge': 200, 'face': 300})
    transform = np.eye(4)
    transform[0, 0] = 2
    transform[1, 3] = 40
    model.set_document_settings(units='mm', local_origin=(3, 4, 5), coordinate_transform=transform,
                                crs_metadata={'epsg': 'test'})
    model._serialization_extensions = {'vendor:runtime': {'name': 'runtime-test', 'values': [1, 2]}}
    model.tag(EntityRef('vertex', 1), 'protected')
    before = to_dict(model)
    captured, restored_timing = {}, {}
    transfer = capture_model_transfer(model, timings=captured)
    restored = restore_model_transfer(transfer, timings=restored_timing)
    assert to_dict(restored) == before
    assert strict_audit(restored).to_dict() == strict_audit(model).to_dict()
    assert restored.replacement_history() == model.replacement_history()
    assert restored.id_state() == model.id_state()
    assert captured['total'] >= 0 and restored_timing['total'] >= 0
    restored._serialization_extensions['vendor:runtime']['values'].append(3)
    assert model._serialization_extensions['vendor:runtime']['values'] == [1, 2]
    assert restored.add_point(7, 8) == 100
    assert model.id_state()['vertex'] == 100


@pytest.mark.parametrize('shape', ['cylinder', 'cone', 'ruled', 'translated_plane'])
def test_real_surface_geometry_evaluation_and_raw_audit_are_exact(shape):
    from anygeometry.generators import cylinder, cone
    from anygeometry.surfaces import Plane, RuledSurface
    if shape in ('cylinder', 'cone'):
        kwargs = dict(circumferential_segments=4, origin=(2.3, -1.1, .4),
                      axis=(1., 2., 3.), radial_direction=(1., .1, 0.))
        model = cylinder(1.2, 2.4, **kwargs) if shape == 'cylinder' else cone(.8, 1.2, 2.4, **kwargs)
    else:
        support = (RuledSurface(np.array(((0., 0, 0), (2., 0, .2))),
                                np.array(((0., 1, 0), (2., 1, .2)))) if shape == 'ruled' else
                   Plane((23.25, -10.75, 4.125), (2., 0, .1), (.25, 3., .2)))
        model = GeometryModel()
        vertices = model.add_points(support.evaluate(u, v) for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)))
        model.add_face(model.add_polyline(vertices, close=True), surface=support)
    before = to_dict(model)
    restored = restore_model_transfer(capture_model_transfer(model))
    assert to_dict(restored) == before
    for identifier, face in model.faces.items():
        first, second = face.surface, restored.faces[identifier].surface
        for u, v in ((0., 0.), (.13, .29), (.41, .73), (1., 1.)):
            assert np.array_equal(first.evaluate(u, v), second.evaluate(u, v))
            assert np.array_equal(first.local_uv(first.evaluate(u, v)), second.local_uv(second.evaluate(u, v)))
        if hasattr(first, '_circumferential'):
            assert first._circumferential.tobytes() == second._circumferential.tobytes()
            assert first.axis.tobytes() == second.axis.tobytes()
            assert first.radial_direction.tobytes() == second.radial_direction.tobytes()
    assert strict_audit(restored).to_dict() == strict_audit(model).to_dict()


def test_runtime_committed_record_payload_omits_redundant_undo_maps():
    transfer = capture_model_transfer(fixture('healthy'))
    state = pickle.loads(transfer.payload)
    assert not {'vertex_state', 'edge_state', 'face_state'} & state['design']['topology'].keys()


def test_classification_receipt_is_bound_and_public_restore_still_validates(monkeypatch):
    from anygeometry.audit_parallel import _ParallelState
    from anygeometry.audit import AuditCollector, AuditContext, AuditScope
    from anygeometry.model_transport import _issue_classification_receipt, _restore_classification_transfer
    model = fixture('healthy')
    context = AuditContext(str(model.model_id), model.revision, AuditScope.FULL_MODEL)
    root = _ParallelState(model, context, AuditCollector(str(model.model_id), model.revision),
                          workers=2, min_candidates=0, cancelled=None, execution={})
    transfer = capture_model_transfer(model)
    with pytest.raises(GeometryError, match='completed root'):
        _issue_classification_receipt(transfer, root)
    root.audit_topology()
    receipt = _issue_classification_receipt(transfer, root)
    def fail(self):
        raise RuntimeError('full topology validator invoked')
    monkeypatch.setattr(GeometryModel, 'validate_topology', fail)
    assert _restore_classification_transfer(transfer, receipt).revision == model.revision
    with pytest.raises(GeometryError, match='validator'):
        restore_model_transfer(transfer)
    with pytest.raises(GeometryError, match='receipt'):
        _restore_classification_transfer(replace(transfer, source_checksum='wrong'), receipt)


def test_moved_vertices_and_changed_face_metadata_survive_compact_capture():
    model = fixture('healthy')
    model.move_point(2, 2.1, 0.)
    model.set_face_metadata(1, {'updated': True, 'nested': {'value': 17}})
    before = to_dict(model)
    restored = restore_model_transfer(capture_model_transfer(model))
    assert to_dict(restored) == before
    assert np.array_equal(restored.vertex_position(2), [2.1, 0, 0])
    assert strict_audit(restored).to_dict() == strict_audit(model).to_dict()
