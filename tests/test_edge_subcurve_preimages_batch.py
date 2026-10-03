"""Public batch provenance, without meshing or large-model qualification."""
from dataclasses import replace

import pytest

from anygeometry import (GeometryError, GeometryModel, apply_intersections,
    clone_prepared_geometry, from_dict, plan_intersections, set_prepared_face_corners,
    to_dict)
from anygeometry.edge_subcurve_preimages import (
    query_prepared_edge_subcurve_preimages as query,
    validate_prepared_edge_subcurve_preimages_binding as validate)


def crossing(*, extra_boundary_point=False):
    model = GeometryModel()
    first = ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0))
    if extra_boundary_point:
        first = (first[0], (0, -1, 0), *first[1:])
    for points in (first,
                   ((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1))):
        model.add_plate(model.add_points(points))
    return model


def prepare(model):
    plan = plan_intersections(model, tuple(model.faces), policy='connect')
    apply_intersections(model, plan, policy='connect')
    return plan


def test_batch_retains_binary_splits_without_qualifying_new_joint_edges():
    model = crossing()
    original = to_dict(model)
    roots = set(model.edges)
    plan = prepare(model)
    committed = to_dict(model)
    binding = query(model)
    assert binding.records and binding.unavailable_edge_ids
    assert set(binding.coverage) == set(model.edges)
    assert set(r.edge_id for r in binding.records).isdisjoint(binding.unavailable_edge_ids)
    assert set(r.edge_id for r in binding.records) | set(binding.unavailable_edge_ids) == set(model.edges)
    descendants = [r for r in binding.records if r.edge_id not in roots]
    assert descendants
    assert all(r.ancestor.definition.edge_id in roots for r in descendants)
    assert all(r.ancestor.source_checksum == original['checksum']['value'] for r in descendants)
    validate(model, binding)
    with pytest.raises(GeometryError, match='unavailable'):
        query(model, edge_ids=binding.unavailable_edge_ids)
    assert to_dict(model) == committed
    assert apply_intersections(model, plan, policy='connect').reused
    assert query(model) == binding


def test_qualified_clone_and_corner_reseal_retain_exact_ancestry():
    model = crossing(extra_boundary_point=True)
    prepare(model)
    old = query(model)
    cloned = clone_prepared_geometry(model)
    assert query(cloned) == old
    face = next(f.id for f in cloned.faces.values() if len(f.loop) >= 5)
    corners = next(c for c in ((0, 1, 2, 3), (0, 1, 2, 4)) if c != cloned.faces[face].corners)
    set_prepared_face_corners(cloned, {face: corners})
    new = query(cloned)
    assert new.records == old.records
    assert new.unavailable_edge_ids == old.unavailable_edge_ids
    assert new.revision == cloned.revision
    validate(cloned, new)
    assert query(model) == old


@pytest.mark.parametrize('copy', ('load', 'ordinary_clone'))
def test_ordinary_copy_has_no_transferred_proof(copy):
    model = crossing()
    prepare(model)
    binding = query(model)
    other = from_dict(to_dict(model)) if copy == 'load' else model.clone(preserve_identity=True)
    with pytest.raises(GeometryError):
        query(other)
    with pytest.raises(GeometryError):
        validate(other, binding)


def test_cubic_exterior_splits_carry_the_actual_local_interval():
    model = GeometryModel()
    points = model.add_points(((0, 0, 0), (1, 2, 0), (2, -1, 0), (3, 1, 0)))
    original_edge = model.add_spline(points[0], points[1:-1], points[-1])
    model.extrude((original_edge,), (.25, 0, 1.5))
    model.add_plate(model.add_points(((1.5, -3, -1), (1.5, 3, -1),
                                     (1.5, 3, 3), (1.5, -3, 3))))
    prepare(model)
    binding = query(model)
    descendants = [r for r in binding.records if r.ancestor.definition.edge_id == original_edge]
    assert len(descendants) == 2
    descendants.sort(key=lambda r: r.interval)
    assert descendants[0].interval[0] == (0, 1)
    assert descendants[0].interval[1] == descendants[1].interval[0]
    assert descendants[1].interval[1] == (1, 1)
    query(model, edge_ids=tuple(r.edge_id for r in descendants))
    validate(model, binding)


def test_mismatched_binding_is_not_accepted_by_the_owner():
    model = crossing()
    prepare(model)
    binding = query(model)
    with pytest.raises(GeometryError, match='definition binding changed'):
        validate(model, replace(binding, unavailable_edge_ids=()))


def test_last_policy_callback_cannot_publish_a_changed_candidate(monkeypatch):
    from anygeometry import IntersectionBatchPolicy
    import anygeometry.edge_subcurve_preimages as provenance
    model = crossing()
    original = to_dict(model)
    plan = plan_intersections(model, tuple(model.faces), policy='connect')
    finalize = provenance._finalize_edge_subcurve_preimages
    ready = []
    def seal(candidate, draft, **kwargs):
        result = finalize(candidate, draft, **kwargs)
        ready.append(candidate)
        return result
    monkeypatch.setattr(provenance, '_finalize_edge_subcurve_preimages', seal)
    def change_candidate():
        if ready:
            candidate = ready.pop()
            vertex = min(candidate.vertices)
            x, y, z = candidate.vertex_position(vertex)
            candidate.move_point(vertex, x+2.**-44, y, z)
        return False
    policy = IntersectionBatchPolicy(cancellation_check=change_candidate)
    with pytest.raises(GeometryError, match='candidate changed before commit'):
        apply_intersections(model, plan, policy=policy)
    assert to_dict(model) == original
    assert not hasattr(model, '_edge_subcurve_preimages_receipt')


def test_publication_revision_normalization_cannot_ignore_metadata_changes():
    from anygeometry import EntityRef
    from anygeometry.edge_subcurve_preimages import (
        _capture_edge_subcurve_preimages, _finalize_edge_subcurve_preimages,
        _publish_edge_subcurve_preimages)
    model = crossing()
    prepare(model)
    original_receipt = model._edge_subcurve_preimages_receipt
    draft = _capture_edge_subcurve_preimages(model)
    candidate = model.clone(preserve_identity=True)
    sealed = _finalize_edge_subcurve_preimages(candidate, draft)
    model.tag(EntityRef('edge', min(model.edges)), 'later-edit')
    with pytest.raises(GeometryError, match='does not match prepared candidate'):
        _publish_edge_subcurve_preimages(model, sealed, to_dict(model)['checksum']['value'])
    assert model._edge_subcurve_preimages_receipt is original_receipt
