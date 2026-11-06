"""Model read views retain exact private-store ownership before serialization."""
import types
import pytest
from _planning_numpy_profile_contract import numpy_profile_test
from _planning_contract_helpers import capture_with_dispatcher_boundary
from anygeometry._planning_executor import freeze_model
from anygeometry._planning_model_state import production_snapshot
from anygeometry._planning_manifest import _RetainedChecks
from anygeometry._planning_qualification import QualificationDecline

VIEWS=('vertices','edges','faces','parts','sheets','face_uses','coedges','members',
       'member_edge_uses','attachments','junctions')


def model_fixture():
    from anygeometry._planning_canaries import _fixture
    from anygeometry.serialization import from_dict,to_dict
    model,_=_fixture()
    return from_dict(to_dict(model))


@pytest.mark.parametrize('snapshot',[freeze_model,production_snapshot])
@pytest.mark.parametrize('name',VIEWS)
def test_equal_but_unowned_public_backing_refuses(name,snapshot):
    model=model_fixture();state=object.__getattribute__(model,'__dict__')
    original=state[name];private=state['_'+name]
    replacement=types.MappingProxyType(dict(private))
    state[name]=replacement
    with pytest.raises(QualificationDecline):snapshot(model)
    assert state[name] is replacement and state['_'+name] is private and original is not replacement


@pytest.mark.parametrize('snapshot',[freeze_model,production_snapshot])
@pytest.mark.parametrize('change',['plain_view','private_rebind','other_owner','construction_subclass','private_subclass','state_subclass'])
def test_model_mapping_shape_refuses_without_hooks(snapshot,change):
    model=model_fixture();state=object.__getattribute__(model,'__dict__');events=[]
    class Foreign(dict):
        def items(self):events.append('items');raise AssertionError('foreign items')
        def keys(self):events.append('keys');raise AssertionError('foreign keys')
        def __getitem__(self,key):events.append('getitem');raise AssertionError('foreign getitem')
        def __iter__(self):events.append('iter');raise AssertionError('foreign iter')
        def __len__(self):events.append('len');raise AssertionError('foreign len')
    if change=='plain_view':state['faces']=dict(state['_faces'])
    elif change=='private_rebind':state['_faces']=dict(state['_faces'])
    elif change=='other_owner':state['parts']=types.MappingProxyType(state['_sheets'])
    elif change=='construction_subclass':state['_construction_vertices']=Foreign()
    elif change=='private_subclass':
        state['_faces']=Foreign(state['_faces']);state['faces']=types.MappingProxyType(state['_faces'])
    else:
        state=Foreign(state);object.__setattr__(model,'__dict__',state)
    with pytest.raises(QualificationDecline):snapshot(model)
    assert not events and object.__getattribute__(model,'__dict__') is state


@pytest.mark.parametrize('proxy',[False,True])
def test_retained_dict_subclass_refuses_before_items(proxy):
    events=[]
    class Foreign(dict):
        def items(self):events.append('items');raise AssertionError('foreign items')
    value=Foreign(x=1)
    if proxy:value=types.MappingProxyType(value)
    with pytest.raises(QualificationDecline):_RetainedChecks().mapping('foreign',value)
    assert events==[]


@numpy_profile_test
def test_infrastructure_mapping_refuses_before_hooks(monkeypatch):
    from anygeometry import _planning_code as code
    events=[]
    class Foreign:
        def __getitem__(self,key):events.append('getitem');raise AssertionError('foreign getitem')
        def items(self):events.append('items');raise AssertionError('foreign items')
    value=types.MappingProxyType(Foreign())
    # Actual owned infrastructure function defaults are walked by the pin.
    # Keep its source/code/registered owner intact to reach the mapping guard.
    monkeypatch.setattr(code.stage_cost_vector,'__defaults__',(value,))
    with pytest.raises(QualificationDecline,match='retained mapping backing'):
        code.InfrastructurePin('host')
    assert events==[]


@pytest.mark.parametrize('full',[False,True])
def test_rebound_proxy_is_terminal_without_reading_new_mapping(full,request):
    from anygeometry._planning_manifest import capture_callable_manifest
    from anygeometry._planning_executor import PlannerSourceChanged
    module=types.ModuleType('ppq_mapping_rebind')
    module.VALUE=types.MappingProxyType({'x':1})
    exec('def root(): return VALUE',vars(module))
    captured=capture_with_dispatcher_boundary((module.root,),request=request)
    if captured is None:return # Exact initial refusal is not postcapture coverage.
    events=[]
    class Foreign:
        def __getitem__(self,key):events.append('getitem');raise AssertionError('foreign getitem')
        def items(self):events.append('items');raise AssertionError('foreign items')
    module.VALUE=types.MappingProxyType(Foreign())
    with pytest.raises(PlannerSourceChanged):captured.verify(full=full)
    assert events==[]


@pytest.mark.parametrize('snapshot',[freeze_model,production_snapshot])
def test_unowned_model_view_after_snapshot_is_not_read(snapshot):
    model=model_fixture();freeze_model(model);events=[]
    class Foreign:
        def __getitem__(self,key):events.append('getitem');raise AssertionError('foreign getitem')
        def items(self):events.append('items');raise AssertionError('foreign items')
    replacement=types.MappingProxyType(Foreign());model.faces=replacement
    with pytest.raises(QualificationDecline):snapshot(model)
    assert model.faces is replacement and events==[]


@pytest.mark.parametrize('warm',[False,True])
def test_valid_model_views_and_cache_bindings_survive_snapshots(warm):
    from anygeometry.serialization import to_dict
    model=model_fixture()
    if warm:
        for face in model.faces.values():face.surface.normal
        for edge in model.edges:model.edge_length(edge)
    state=object.__getattribute__(model,'__dict__')
    before={name:id(value) for name,value in state.items()}
    document=to_dict(model)
    frozen=freeze_model(model)
    assert production_snapshot(model)==frozen
    assert {name:id(value) for name,value in state.items()}==before
    assert to_dict(model)==document
    # A new native proxy over the same owned store is a valid initial relation.
    state['faces']=types.MappingProxyType(state['_faces'])
    replacement=state['faces']
    assert freeze_model(model)==frozen and production_snapshot(model)==frozen
    assert state['faces'] is replacement
