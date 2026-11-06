"""A cold Plane must satisfy the same declared-field contract as a warm one."""
import numpy as np
import pytest

from anygeometry import GeometryModel, Plane, to_dict
from anygeometry._planning_executor import freeze_model
from anygeometry._planning_qualification import QualificationDecline


@pytest.mark.parametrize('field', ['origin', 'u_vector', 'v_vector'])
@pytest.mark.parametrize('warm', [False, True])
def test_writable_plane_fields_decline_before_capture_without_mutation(field, warm):
    model = GeometryModel()
    face = model.add_plate(model.add_points(((0., 0., 0.), (2., 0., 0.),
                                            (2., 2., 0.), (0., 2., 0.))))
    plane = model.faces[face].surface
    assert type(plane) is Plane
    if warm:
        plane.normal
    getattr(plane, field).flags.writeable = True
    before = to_dict(model)
    arrays = tuple((name, id(value), value.tobytes(), value.flags.writeable)
                   for name, value in vars(plane).items() if type(value) is np.ndarray)
    with pytest.raises(QualificationDecline, match='unqualified Plane declared field'):
        freeze_model(model)
    assert to_dict(model) == before
    assert tuple((name, id(value), value.tobytes(), value.flags.writeable)
                 for name, value in vars(plane).items() if type(value) is np.ndarray) == arrays


def test_canonical_plane_cold_and_warm_have_identical_transport():
    model = GeometryModel()
    face = model.add_plate(model.add_points(((0., 0., 0.), (2., 0., 0.),
                                            (2., 2., 0.), (0., 2., 0.))))
    cold = freeze_model(model)
    model.faces[face].surface.normal
    assert freeze_model(model) == cold
