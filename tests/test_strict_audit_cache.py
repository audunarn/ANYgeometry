"""Exact per-audit memoization: complete report parity and isolated lifetime."""
import pytest

from anygeometry.model import GeometryModel
from anygeometry.strict_audit import strict_audit, _StrictAuditState
from anygeometry.tolerance import feature_extent


def model_fixture(defective=False):
    model = GeometryModel()
    if defective:
        vertices = model.add_points(((0, 0, 0), (2, 0, 0), (1, -1, 0), (1, 1, 0), (1, 0, 0), (3, 0, 0)))
        for a, b in ((0, 1), (2, 3), (4, 5)):
            model.add_line(vertices[a], vertices[b])
    else:
        vertices = model.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)))
        model.add_plate(vertices)
    return model


@pytest.mark.parametrize('defective', [False, True])
def test_exact_complete_report_matches_uncached_measurements(monkeypatch, defective):
    model = model_fixture(defective)
    cached = strict_audit(model).to_dict()
    monkeypatch.setattr(_StrictAuditState, '_edge_length', lambda self, identifier: self.model.edge_length(identifier))
    monkeypatch.setattr(_StrictAuditState, '_face_extent', lambda self, identifier: feature_extent(self.face_points(identifier)))
    monkeypatch.setattr(_StrictAuditState, '_plane_local_uv', lambda self, plane, point: plane.local_uv(point))
    monkeypatch.setattr(_StrictAuditState, '_face_boundary_edges', lambda self, identifier:
        frozenset(item.edge for loop in (self.model.faces[identifier].loop,) + tuple(self.model.faces[identifier].holes) for item in loop))
    assert strict_audit(model).to_dict() == cached


def test_new_audit_after_edit_does_not_reuse_measurements():
    model = model_fixture()
    before = strict_audit(model)
    model.move_point(2, 2, 0)
    after = strict_audit(model)
    assert after.revision == model.revision > before.revision
    assert after.metrics.classification_complete


def test_cached_temporary_plane_is_retained_and_distinct_frames_are_correct():
    import gc
    import weakref
    import numpy as np
    from anygeometry.surfaces import Plane
    state = object.__new__(_StrictAuditState)
    state._plane_uv_cache = {}
    state._plane_uv_owners = {}
    first = Plane(np.array([0., 0, 0]), np.array([1., 0, 0]), np.array([0., 1, 0]))
    lifetime = weakref.ref(first)
    assert state._plane_local_uv(first, (2., 3, 0)) == (2., 3.)
    del first
    gc.collect()
    assert lifetime() is not None
    second = Plane(np.array([1., 1, 0]), np.array([1., 0, 0]), np.array([0., 1, 0]))
    assert state._plane_local_uv(second, (2., 3, 0)) == (1., 2.)
    assert len(state._plane_uv_owners) == 2
