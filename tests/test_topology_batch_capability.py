import pytest
from anygeometry import GeometryModel, GeometryError
from test_topology_projection_batch import reported


def test_capability_revision_transaction_and_unknown_face(reported):
    geometry, faces=reported
    face=faces[0]
    assert geometry.supports_topology_face_batch(face)
    with geometry.transaction():
        assert not geometry.supports_topology_face_batch(face)
    geometry.add_point(100.,100.,100.)
    assert geometry.supports_topology_face_batch(face)
    with pytest.raises(GeometryError):
        geometry.supports_topology_face_batch(999999)


def test_custom_model_not_advertised():
    class Custom(GeometryModel):
        pass
    geometry=Custom()
    points=geometry.add_points(((0.,0.,0.),(1.,0.,0.),(1.,1.,0.),(0.,1.,0.)))
    face=geometry.add_face(geometry.add_polyline(points,close=True),corners=(0,1,2,3))
    assert not geometry.supports_topology_face_batch(face)
