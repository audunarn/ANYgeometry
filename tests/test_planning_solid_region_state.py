"""Planning qualification vs the solid-region model field: schema registered, non-empty regions decline."""
import pytest

from anygeometry import GeometryModel
import anygeometry._planning_qualification as q
from anygeometry._planning_model_state import declarative_state


def _model():
    return GeometryModel(model_id='00000000-0000-0000-0000-000000000009')


def test_ordinary_model_without_solid_regions_keeps_the_planning_state_schema():
    model = _model()
    assert 'solid_regions' in model.__dict__
    assert '_solid_regions' in declarative_state(model)
    q.check_types((model,), {'types': q.input_type_names()})


def test_model_with_solid_regions_is_declined_for_qualified_planning_not_accepted():
    model = _model()
    model.add_solid_box((0, 0, 0), (1, 1, 1))
    with pytest.raises(q.QualificationDecline, match='solid region'):
        declarative_state(model)
    with pytest.raises(q.QualificationDecline, match='solid region'):
        q.check_types((model,), {'types': q.input_type_names()})
