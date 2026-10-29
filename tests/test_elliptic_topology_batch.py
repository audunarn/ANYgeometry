"""Elliptic topology patches retain scalar geometry and validation semantics."""
import numpy as np
import pytest

from anygeometry import CoonsSurface, GeometryModel, from_dict, to_dict
from anygeometry.exact_curves import EllipticArc
from test_topology_projection_batch import old_scalar_uv, assert_derivative_parity


def elliptic_patch(sweep=1.2):
    geometry = GeometryModel()
    first = EllipticArc((0., 0., 0.), (2., 0., 0.), (0., 1., .2), .15, sweep)
    second = EllipticArc((.3, -.2, 3.), (1.7, 0., .1), (0., 1.3, .2), .15, sweep)
    ends = [*first.evaluate([0., 1.]), *second.evaluate([0., 1.])]
    a, b, d, c = geometry.add_points(ends)
    bottom = geometry.add_curve(a, b, first)
    right = geometry.add_line(b, c)
    top = geometry.add_curve(d, c, second)
    left = geometry.add_line(d, a)
    face = geometry.add_face((bottom, right, top, left), corners=(0, 1, 2, 3), surface=CoonsSurface())
    return geometry, face


@pytest.mark.parametrize('sweep', [1.2, -1.2, .08])
def test_elliptic_point_derivative_and_inverse_scalar_parity(sweep):
    geometry, face = elliptic_patch(sweep)
    assert geometry.supports_topology_face_batch(face)
    uv = np.vstack((np.random.default_rng(42).uniform(0., 1., (32, 2)),
                    [[0., 0.], [1., 1.], [0., 1.], [1., 0.], [.5, .5]]))
    expected = np.asarray([geometry.face_point(face, *row) for row in uv])
    np.testing.assert_array_equal(geometry.evaluate_face_many(face, uv), expected)
    assert_derivative_parity(geometry, face, uv)
    targets = np.vstack((expected, expected[:4] + [.01, -.02, .03]))
    scalar = np.asarray([old_scalar_uv(geometry, face, point) for point in targets])
    np.testing.assert_array_equal(geometry.face_local_uv_many(face, targets), scalar)
    np.testing.assert_array_equal([geometry.face_local_uv(face, point) for point in targets], scalar)


def test_validation_batches_elliptic_boundary_and_preserves_round_trip(monkeypatch):
    geometry, face = elliptic_patch()
    original = geometry.face_local_uv_many
    calls = []
    def observed(identifier, points):
        calls.append((identifier, len(points)))
        return original(identifier, points)
    monkeypatch.setattr(geometry, 'face_local_uv_many', observed)
    assert not geometry.validate_topology()
    assert calls and all(identifier == face and count > 1 for identifier, count in calls)
    document = to_dict(geometry)
    assert to_dict(from_dict(document)) == document


def test_elliptic_batch_disabled_in_transaction_and_invalidated_after_edit():
    geometry, face = elliptic_patch()
    uv = np.array([[.25, .75], [.6, .3]])
    before = geometry.evaluate_face_many(face, uv)
    with geometry.transaction():
        assert not geometry.supports_topology_face_batch(face)
    geometry.add_point(10., 20., 30.)
    assert geometry.supports_topology_face_batch(face)
    np.testing.assert_array_equal(geometry.evaluate_face_many(face, uv), before)


def test_batch_inverse_failure_remains_a_validation_error(monkeypatch):
    geometry, face = elliptic_patch()
    def failed(*_args):
        raise np.linalg.LinAlgError('inverse failed')
    monkeypatch.setattr(geometry, 'face_local_uv_many', failed)
    assert geometry.validate_topology() == (
        f'face {face} has invalid Coons geometry: inverse failed',
    )


def test_transaction_validation_uses_scalar_equivalent_uncached_rows():
    geometry, face = elliptic_patch()
    uv = np.random.default_rng(17).uniform(0., 1., (16, 2))
    geometry._topology_corner_cache.clear()
    with geometry.transaction():
        assert not geometry.supports_topology_face_batch(face)
        points = np.asarray([geometry.face_point(face, *row) for row in uv])
        np.testing.assert_array_equal(geometry._topology_face_points_many(face, uv), points)
        expected = np.asarray([old_scalar_uv(geometry, face, point) for point in points])
        np.testing.assert_array_equal(geometry._validation_face_local_uv_many(face, points), expected)
        assert not geometry._topology_corner_cache


def test_transaction_local_rows_follow_edits_and_rollback():
    geometry = GeometryModel()
    vertices = geometry.add_points(((0., 0., 0.), (2., 0., 0.),
                                   (2., 2., .5), (0., 2., 0.)))
    face = geometry.add_face(geometry.add_polyline(vertices, close=True),
                             corners=(0, 1, 2, 3), surface=CoonsSurface())
    uv = np.array([[.25, .75], [.5, .5]])
    before = geometry.evaluate_face_many(face, uv)
    with pytest.raises(RuntimeError, match='rollback'):
        with geometry.transaction():
            geometry.move_point(vertices[0], 0., 0., .2)
            moved = geometry._topology_face_points_many(face, uv)
            assert not np.array_equal(moved, before)
            np.testing.assert_array_equal(moved, [geometry.face_point(face, *row) for row in uv])
            geometry.move_point(vertices[0], 0., 0., .4)
            moved_again = geometry._topology_face_points_many(face, uv)
            assert not np.array_equal(moved_again, moved)
            np.testing.assert_array_equal(moved_again, [geometry.face_point(face, *row) for row in uv])
            raise RuntimeError('rollback')
    np.testing.assert_array_equal(geometry.evaluate_face_many(face, uv), before)
