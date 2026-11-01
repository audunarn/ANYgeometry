"""Explicit task-scoped size/walk limits, preserving all existing defaults."""
import pytest
from anygeometry import _planning_executor as a
from anygeometry._planning_qualification import QualificationDecline
from anygeometry._planning_canaries import _fixture
from anygeometry.serialization import from_dict,to_dict


def test_transport_limit_is_explicit_and_configurable():
    with pytest.raises(QualificationDecline,match='limit=10'):
        a.pack({'x':'z'*40},'model',limit=10)
    packet=a.pack({'x':'z'*40},'model',limit=100)
    assert a.unpack(packet,'model',limit=100)=={'x':'z'*40}
    with pytest.raises(QualificationDecline,match='limit=10'):a.unpack(packet,'model',limit=10)


def test_model_walk_limit_is_explicit_and_configurable():
    model,_=_fixture();model=from_dict(to_dict(model))
    with pytest.raises(QualificationDecline,match='limit=10'):
        a.freeze_model(model,walk_limit=10)
    packet=a.freeze_model(model,walk_limit=100000)
    assert to_dict(a.thaw_model(packet,walk_limit=100000))==to_dict(model)


def test_planner_records_requested_limits(tmp_path):
    with a.OwnedPlanner(tmp_path/'p',payload_limit=6000000,walk_limit=200000) as planner:
        assert a.read_packet(tmp_path/'p/limits.json','limits')==dict(payload_limit=6000000,walk_limit=200000)
