"""Freshness reuses prior qualification but still binds live lookup semantics."""
from dataclasses import replace

import pytest

from anygeometry import (GeometryError, has_current_intersection_preparation,
    query_prepared_face_preimages, query_prepared_edge_subcurve_preimages)
from anygeometry.authored_boundary_correspondence import query_prepared_authored_boundary_correspondence


def prepared():
    from test_edge_subcurve_preimages_batch import crossing, prepare
    model = crossing()
    prepare(model)
    return model


def test_unchanged_prepared_queries_do_not_repeat_topology_qualification(monkeypatch):
    model = prepared()
    def forbidden():
        raise AssertionError('unchanged receipt requalified the entire model')
    monkeypatch.setattr(model, 'validate_topology', forbidden)
    assert has_current_intersection_preparation(model)
    query_prepared_face_preimages(model)
    query_prepared_edge_subcurve_preimages(model)
    query_prepared_authored_boundary_correspondence(model, 1)


def test_valid_raw_content_mismatch_requalifies_before_receipt_refusal(monkeypatch):
    model = prepared()
    identifier = min(model.faces)
    model._faces[identifier] = replace(model.faces[identifier], metadata={'raw': 'edit'})
    calls = []
    original = model.validate_topology
    def counted():
        calls.append(True)
        return original()
    monkeypatch.setattr(model, 'validate_topology', counted)
    assert not has_current_intersection_preparation(model)
    assert len(calls) == 1


@pytest.mark.parametrize('store', ('_vertices', '_edges', '_faces', '_sheets', '_face_uses', '_coedges'))
def test_same_revision_lookup_permutation_cannot_preserve_preparation_authority(store):
    model = prepared()
    revision = model.revision
    records = getattr(model, store)
    first, second = sorted(records)[:2]
    before = sorted((row.id, repr(row)) for row in records.values())
    records[first], records[second] = records[second], records[first]
    assert sorted((row.id, repr(row)) for row in records.values()) == before
    assert model.revision == revision
    with pytest.raises(GeometryError, match='fingerprint.*key'):
        has_current_intersection_preparation(model)
    with pytest.raises(GeometryError, match='fingerprint.*key'):
        query_prepared_edge_subcurve_preimages(model)


def test_late_callback_lookup_permutation_is_detected_even_without_revision_change():
    model = prepared()
    revision = model.revision
    calls = 0
    def callback(_):
        nonlocal calls
        calls += 1
        if calls == 2:
            first, second = sorted(model.vertices)[:2]
            model._vertices[first], model._vertices[second] = model._vertices[second], model._vertices[first]
        return False
    with pytest.raises(GeometryError, match='fingerprint.*key'):
        query_prepared_edge_subcurve_preimages(model, cancellation_check=callback)
    assert model.revision == revision


@pytest.mark.parametrize('edit', ('position', 'face_metadata'))
def test_persisted_value_edit_invalidates_proof_even_without_revision_change(edit):
    model = prepared()
    revision = model.revision
    if edit == 'position':
        identifier = min(model.vertices)
        point = model.vertices[identifier].position.copy()
        point[0] += .25
        model._vertices[identifier] = replace(model.vertices[identifier], position=point)
    else:
        identifier = min(model.faces)
        model._faces[identifier] = replace(model.faces[identifier], metadata={'raw': 'edit'})
    assert model.revision == revision
    assert not has_current_intersection_preparation(model)
    with pytest.raises(GeometryError):
        query_prepared_face_preimages(model)
    with pytest.raises(GeometryError):
        query_prepared_edge_subcurve_preimages(model)


@pytest.mark.parametrize('field,value', (
    ('feature_id', 1.25), ('kind_version', 1.25),
    ('state', 'not-a-feature-state'), ('diagnostic', {'invalid': 'diagnostic'}),
))
def test_feature_normalization_cannot_hide_invalid_same_revision_definitions(field, value):
    from anygeometry import EntityRef
    from anygeometry.features import FeatureRecord
    from test_edge_subcurve_preimages_batch import crossing, prepare
    model = crossing()
    point = model.add_point(3, 4, 5)
    record = FeatureRecord(feature_id=1, kind='vendor.point', name='Frozen point',
                           outputs={'point': EntityRef('vertex', point)})
    model.features.adopt_frozen(model, kind=record.kind, name=record.name,
        outputs=record.outputs,
        expected_checksum=model.features.materialization_checksum(record, model))
    prepare(model)
    assert has_current_intersection_preparation(model)
    revision = model.revision
    setattr(model.features._records[0], field, value)
    assert model.revision == revision
    with pytest.raises(GeometryError):
        has_current_intersection_preparation(model)
    with pytest.raises(GeometryError):
        query_prepared_edge_subcurve_preimages(model)
