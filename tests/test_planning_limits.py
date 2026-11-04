"""Explicit task-scoped size/walk limits, preserving all existing defaults."""
import pytest
from _planning_contract_helpers import require_qualified_planning_lane
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


def test_planner_records_requested_limits(tmp_path,request):
    from anygeometry import _planning_process as process
    if process.os.name!='nt':
        require_qualified_planning_lane('owned planner requires Windows Job containment')
        with pytest.raises(QualificationDecline,match='^owned planner requires Windows Job containment$'):
            a.OwnedPlanner(tmp_path/'p',payload_limit=6000000,walk_limit=200000)
        assert a.read_packet(tmp_path/'p/limits.json','limits')==dict(payload_limit=6000000,walk_limit=200000)
        request.node.user_properties.extend((('execution_scope','initial-Windows-Job-refusal'),
            ('postcapture_qualification',False),('original_test_body_executed',False)))
        return
    request.node.user_properties.append(('original_test_body_executed',True))
    with a.OwnedPlanner(tmp_path/'p',payload_limit=6000000,walk_limit=200000) as planner:
        assert a.read_packet(tmp_path/'p/limits.json','limits')==dict(payload_limit=6000000,walk_limit=200000)


def test_requested_limits_component_with_owned_fake_tree(tmp_path,monkeypatch,request):
    from anygeometry import _planning_process as process
    events=[]
    class Tree:
        def __init__(self,directory):events.append('create')
        def spawn(self,role,directory,timeout):events.append(('spawn',role));return object()
        def close(self):events.append('close')
    monkeypatch.setattr(process,'OwnedTree',Tree)
    with a.OwnedPlanner(tmp_path/'component',payload_limit=6000000,walk_limit=200000):
        assert a.read_packet(tmp_path/'component/limits.json','limits')==dict(payload_limit=6000000,walk_limit=200000)
    assert events==['create',('spawn','planner'),'close']
    request.node.user_properties.extend((('execution_scope','owned-fake-tree-limit-recording-unit'),
                                       ('native_process_qualification',False)))
