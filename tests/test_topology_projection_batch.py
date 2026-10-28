"""Topology-Coons batching retains the scalar centre-start inverse."""
import numpy as np
import pytest

from anygeometry import GeometryModel, GeometryError, SketchDefinition, SketchEntity, SketchWire
from anygeometry.sketch import plane_sketch_plane, materialize_sketch_on_plane
from anygeometry.nurbs import NurbsCurve


def old_scalar_uv(geometry, face, point):
    from anygeometry.surfaces import closest_uv
    class Reference:
        def evaluate(self, u, v):
            return geometry.face_point(face, u, v)
    return closest_uv(Reference(), point)


@pytest.fixture(scope='module')
def reported():
    geometry = GeometryModel()
    points = {'a': (1.840810486134475, -2.865807234095718),
              'b': (6.045388982873449, .06275490293640225),
              'c': (3.556111166396149, 3.807130778141757),
              'd': (-3.953558884993364, 4.12090529282377),
              'e': (-5.355085050573022, -1.2969346606856549),
              'f': (-.8994869420884367, -3.702539273247754)}
    definition = SketchDefinition(points, version=2,
        entities=(SketchEntity('curve', 'spline', tuple(points), closed=True),),
        wires=(SketchWire('profile', ('curve',), True),))
    outputs = materialize_sketch_on_plane(geometry,
        plane_sketch_plane((0.,0.,0.), (1.,0.,0.), (0.,1.,0.)), definition)
    edges = [ref.id for key, ref in outputs.items() if key.startswith('entity/')]
    faces = geometry.extrude(edges, (5.,0.,0.))
    return geometry, faces


def test_point_batch_is_scalar_bit_identical_on_reported_faces(reported):
    geometry, faces = reported
    random = np.random.default_rng(123).uniform(-.1, 1.1, (80, 2))
    parameters = np.vstack((random, [[0.,0.],[1.,1.],[0.,1.],[1.,0.],[.5,.5]]))
    for face in faces:
        expected = np.asarray([geometry.face_point(face, *uv) for uv in parameters])
        np.testing.assert_array_equal(geometry.evaluate_face_many(face, parameters), expected)
        assert_derivative_parity(geometry, face, parameters)


def assert_derivative_parity(geometry, face, parameters):
    scalar = [geometry._topology_face_point_and_derivatives(face, *uv) for uv in parameters]
    du = np.asarray([row[1] for row in scalar])
    dv = np.asarray([row[2] for row in scalar])
    batch_du, batch_dv = geometry.face_derivatives_many(face, parameters)
    np.testing.assert_array_equal(batch_du, du)
    np.testing.assert_array_equal(batch_dv, dv)
    normals = np.cross(du, dv)
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    np.testing.assert_array_equal(geometry.face_normal_many(face, parameters), normals)


def test_actual_cell_samples_inverse_and_normals_match_scalar(reported):
    geometry, faces = reported
    for face in faces:
        # Curved midsides of one ordinary physical Q8-shaped cell. The
        # interpolated points lie slightly off the exact surface, as in the
        # quality gate; this is not merely an on-surface round-trip test.
        uv_nodes = np.array([[.2,.3],[.3,.3],[.3,.4],[.2,.4],
                             [.25,.3],[.3,.35],[.25,.4],[.2,.35]])
        nodes = geometry.evaluate_face_many(face, uv_nodes)
        q = np.linspace(-1., 1., 5)
        x, y = np.meshgrid(q, q)
        x, y = x.ravel(), y.ravel()
        shapes = np.column_stack((
            -.25*(1-x)*(1-y)*(1+x+y), -.25*(1+x)*(1-y)*(1-x+y),
            -.25*(1+x)*(1+y)*(1-x-y), -.25*(1-x)*(1+y)*(1+x-y),
            .5*(1-x*x)*(1-y), .5*(1+x)*(1-y*y),
            .5*(1-x*x)*(1+y), .5*(1-x)*(1-y*y)))
        targets = shapes @ nodes
        targets = np.vstack((targets, geometry.evaluate_face_many(face, [[0.,0.],[1.,1.]]),
                             geometry.face_point(face, 0., .5) + [-.2, .4, -.1]))
        scalar = np.asarray([old_scalar_uv(geometry, face, point) for point in targets])
        batched = geometry.face_local_uv_many(face, targets)
        np.testing.assert_array_equal([geometry.face_local_uv(face, point) for point in targets], scalar)
        np.testing.assert_array_equal(batched, scalar)
        assert_derivative_parity(geometry, face, scalar)
        np.testing.assert_array_equal(geometry.face_normal_many(face, batched),
                                      geometry.face_normal_many(face, scalar))


def test_weighted_reversed_curve_inverse_and_clamping():
    geometry = GeometryModel()
    curve = NurbsCurve(2, (0.,0.,0.,1.,1.,1.),
        ((0.,0.,0.), (0.,1.,2.), (0.,3.,0.)), (1.,2.,1.), (1.,0.))
    ends = [geometry.add_point(*point) for point in curve.evaluate([0.,1.])]
    face, = geometry.extrude((geometry.add_curve(*ends, curve),), (-3.,0.,0.))
    points = geometry.evaluate_face_many(face, [[0.,0.],[1.,1.],[.2,.8],[.7,.1]])
    points = np.vstack((points, [[10.,10.,10.],[-10.,-10.,-10.]]))
    scalar = np.asarray([old_scalar_uv(geometry, face, point) for point in points])
    np.testing.assert_array_equal([geometry.face_local_uv(face, point) for point in points], scalar)
    np.testing.assert_array_equal(geometry.face_local_uv_many(face, points), scalar)
    assert_derivative_parity(geometry, face, scalar)
    assert np.all((scalar >= 0.) & (scalar <= 1.))


def test_unsupported_subclass_retains_scalar_override():
    class Custom(GeometryModel):
        def face_local_uv(self, face_id, point):
            return (.125, .75)
    geometry = Custom()
    vertices = geometry.add_points(((0.,0.,0.),(1.,0.,0.),(1.,1.,1.),(0.,1.,0.)))
    face = geometry.add_face(geometry.add_polyline(vertices, close=True), corners=(0,1,2,3))
    np.testing.assert_array_equal(geometry.face_local_uv_many(face, [[.1,.1,.1]]), [[.125,.75]])


def test_empty_invalid_and_mutation_binding(reported):
    geometry, faces = reported
    face = faces[0]
    revision = geometry.revision
    assert geometry.face_local_uv_many(face, np.empty((0,3))).shape == (0,2)
    with pytest.raises(GeometryError):
        geometry.face_local_uv_many(face, [[np.nan,0.,0.]])
    before = geometry.evaluate_face_many(face, [[.2,.3]])
    assert geometry.revision == revision
    # Cache binding is revision/face identity, not a persisted inverse proof.
    geometry.add_point(99.,99.,99.)
    np.testing.assert_array_equal(geometry.evaluate_face_many(face, [[.2,.3]]), before)


def test_other_curves_and_segmented_chains_keep_scalar_path(monkeypatch):
    geometry = GeometryModel()
    start, via, end = geometry.add_points(((0.,1.,0.),(0.,0.,1.),(0.,-1.,0.)))
    arc = geometry.add_arc(start, via, end)
    face, = geometry.extrude((arc,), (3.,1.,0.))
    assert not geometry._batch_topology_supported(geometry.faces[face])
    monkeypatch.setattr(geometry, '_topology_face_points_many', lambda *args, **kwargs: pytest.fail('unsupported curve batched'))
    point = geometry.face_point(face, .4, .6)
    geometry.evaluate_face_many(face, [[.4,.6]])
    geometry.face_derivatives_many(face, [[.4,.6]])
    np.testing.assert_array_equal(geometry.face_local_uv_many(face, [point]),
                                  [geometry.face_local_uv(face, point)])
    vertices = geometry.add_points(((10.,0.,0.),(10.5,0.,0.),(11.,0.,0.),(11.,.5,.2),
                                   (11.,1.,.4),(10.5,1.,.2),(10.,1.,0.),(10.,.5,0.)))
    segmented = geometry.add_face(geometry.add_polyline(vertices, close=True), corners=(0,2,4,6))
    assert not geometry._batch_topology_supported(geometry.faces[segmented])


def test_trim_aware_projection_matches_scalar_for_bounded_and_outside_points(reported):
    geometry, faces = reported
    for face in faces:
        points = geometry.evaluate_face_many(face, [[0.,0.],[1.,1.],[.2,.4],[.99,.01]])
        points = np.vstack((points, [[30.,30.,30.],[-40.,-40.,-40.]]))
        scalar = [geometry.project_to_face(face, point) for point in points]
        projected, uv, distances = geometry.project_to_face_many(face, points)
        np.testing.assert_array_equal(projected, [row[0] for row in scalar])
        np.testing.assert_array_equal(uv, [row[1] for row in scalar])
        np.testing.assert_array_equal(distances, [row[2] for row in scalar])
    with pytest.raises(GeometryError):
        geometry.project_to_face_many(face, [[0.,np.nan,0.]])


def test_coalescing_reduces_callback_count_without_changing_iterations():
    from anygeometry.surfaces import _closest_uv_many
    calls = [[], []]
    targets = np.array([[.2,.3,0.],[-1.,2.,0.],[0.,0.,0.],[1.,1.,0.]])
    def evaluator(index):
        def evaluate(uv):
            calls[index].append(len(uv))
            return np.column_stack((uv, np.zeros(len(uv))))
        return evaluate
    old = _closest_uv_many(None, targets, _evaluate=evaluator(0))
    new = _closest_uv_many(None, targets, _evaluate=evaluator(1), _batch_samples=True)
    np.testing.assert_array_equal(new, old)
    assert len(calls[0]) == 5*len(calls[1])
    assert sum(calls[0]) == sum(calls[1])
