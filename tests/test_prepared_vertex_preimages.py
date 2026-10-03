from dataclasses import FrozenInstanceError, replace

import pytest

from anygeometry import (
    EntityHandle, GeometryError, GeometryModel, Resolution, ResolutionStatus, to_dict,
    query_prepared_vertex_preimages, validate_prepared_vertex_preimages_binding,
)
from test_prepared_model_scope import fixture, prepare


def test_complete_ancestry_includes_isolated_source_vertices_and_generated_stations():
    model, (_faces, _sheet, _member, _attachments, isolated) = fixture()
    original = to_dict(model)
    prepare(model)
    before = to_dict(model)
    receipt = query_prepared_vertex_preimages(model)
    original_ids = {row['id'] for row in original['vertices']}
    reverse = dict(receipt.current_to_authored)
    assert set(reverse) == set(model.vertices)
    assert {row[0] for row in receipt.authored_resolutions} == original_ids
    assert reverse[isolated] == (isolated,)
    assert receipt.without_authored_vertex
    assert receipt.without_authored_vertex == tuple(key for key, value in receipt.current_to_authored if not value)
    for source, status, targets in receipt.authored_resolutions:
        resolved = model.resolve_handle(EntityHandle(model.model_id, 'vertex', source))
        assert status == str(resolved.status)
        assert targets == tuple(sorted(handle.id for handle in resolved.resolved))
        assert all(source in reverse[target] for target in targets)
    # No source-vertex ancestry is NOT absence of source-edge attachments.
    assert before['structural']['attachments']
    assert tuple(receipt.authored_positions) == tuple((row['id'], tuple(row['position'])) for row in sorted(original['vertices'], key=lambda x:x['id']))
    validate_prepared_vertex_preimages_binding(model, receipt)
    assert to_dict(model) == before
    with pytest.raises(FrozenInstanceError):
        receipt.without_authored_vertex = ()


@pytest.mark.parametrize('field', ['current_to_authored', 'authored_resolutions', 'without_authored_vertex', 'authored_positions', 'current_positions'])
def test_forged_receipt_refuses(field):
    model, _ = fixture(); prepare(model)
    receipt = query_prepared_vertex_preimages(model)
    assert getattr(receipt, field)
    with pytest.raises(GeometryError, match='binding changed'):
        validate_prepared_vertex_preimages_binding(model, replace(receipt, **{field: ()}))


def test_stale_wrong_model_and_unprepared_refuse():
    model, _ = fixture(); prepare(model)
    receipt = query_prepared_vertex_preimages(model)
    other, _ = fixture(); prepare(other)
    with pytest.raises(GeometryError):
        validate_prepared_vertex_preimages_binding(other, receipt)
    model.add_point(100,100,100)
    with pytest.raises(GeometryError):
        validate_prepared_vertex_preimages_binding(model, receipt)
    with pytest.raises(GeometryError):
        query_prepared_vertex_preimages(GeometryModel())


def test_cancellation_and_same_revision_edit_refuse():
    model, _ = fixture(); prepare(model)
    before = to_dict(model)
    with pytest.raises(GeometryError, match='cancelled'):
        query_prepared_vertex_preimages(model, cancellation_check=lambda phase: True)
    assert to_dict(model) == before
    receipt = query_prepared_vertex_preimages(model)
    vertex = next(iter(model.vertices.values()))
    try:
        model._vertices[vertex.id] = replace(vertex, position=(100.,100.,100.))
        with pytest.raises(GeometryError):
            validate_prepared_vertex_preimages_binding(model, receipt)
    finally:
        model._vertices[vertex.id] = vertex
    validate_prepared_vertex_preimages_binding(model, receipt)


def test_owner_weld_retains_every_original_vertex_preimage_without_traversal_claim():
    model = GeometryModel()
    first = model.add_points(((0,0,0),(2,0,0),(2,2,0),(0,2,0)))
    second = model.add_points(((0,0,0),(0,2,0),(0,2,2),(0,0,2)))
    model.add_plate(first); model.add_plate(second)
    prepare(model)
    receipt = query_prepared_vertex_preimages(model)
    reverse = dict(receipt.current_to_authored)
    for sources in ((first[0], second[0]), (first[3], second[1])):
        matches = [identifier for identifier, ancestors in reverse.items() if set(sources) <= set(ancestors)]
        assert len(matches) == 1
    validate_prepared_vertex_preimages_binding(model, receipt)


def test_transient_callback_edit_cannot_pollute_snapshot():
    model, _ = fixture(); prepare(model)
    expected = query_prepared_vertex_preimages(model)
    original = dict(model._replacement_history)
    count = 0
    def callback(phase):
        nonlocal count
        count += 1
        if count == 1:
            model._replacement_history.clear()
        else:
            model._replacement_history.update(original)
        return False
    try:
        with pytest.raises(GeometryError):
            query_prepared_vertex_preimages(model, cancellation_check=callback)
    finally:
        model._replacement_history.clear()
        model._replacement_history.update(original)
    assert query_prepared_vertex_preimages(model) == expected


@pytest.mark.parametrize('status', [ResolutionStatus.DELETED, ResolutionStatus.UNKNOWN])
def test_terminal_owner_resolution_remains_explicit_or_refuses(monkeypatch, status):
    model, _ = fixture(); prepare(model)
    resolve = model.resolve_handle
    source = next(iter(model.vertices))
    monkeypatch.setattr(model, 'resolve_handle', lambda handle:
        Resolution.terminal(handle, status) if handle.id == source else resolve(handle))
    if status is ResolutionStatus.UNKNOWN:
        with pytest.raises(GeometryError, match='unresolved original vertex'):
            query_prepared_vertex_preimages(model)
    else:
        receipt = query_prepared_vertex_preimages(model)
        assert next(row for row in receipt.authored_resolutions if row[0] == source) == (source, 'deleted', ())


def test_late_receipt_mutation_refuses():
    model, _ = fixture(); prepare(model)
    receipt = query_prepared_vertex_preimages(model)
    calls = 0
    def callback(phase):
        nonlocal calls
        calls += 1
        if calls == 3:
            object.__setattr__(receipt, 'current_to_authored', ())
        return False
    with pytest.raises(GeometryError, match='binding changed'):
        validate_prepared_vertex_preimages_binding(model, receipt, cancellation_check=callback)
    assert calls == 3


def test_callback_exception_is_preserved():
    model, _ = fixture(); prepare(model)
    before = to_dict(model)
    failure = RuntimeError('caller cancellation')
    def callback(phase):
        raise failure
    with pytest.raises(RuntimeError) as caught:
        query_prepared_vertex_preimages(model, cancellation_check=callback)
    assert caught.value is failure
    assert to_dict(model) == before
