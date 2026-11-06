"""Opt-in caller construction preserves source authority and constructor state."""
import json
import pytest
from anygeometry import query_trimmed_surface_charts
from anygeometry._planning_qualification import QualificationDecline


def array_state(model):
    import numpy as np
    return {(face_id,name):(value.dtype.str,value.shape,value.tobytes(),bool(value.flags.writeable),bool(value.flags.owndata),
                            None if value.base is None else (type(value.base).__name__,value.base.tobytes()))
        for face_id,face in model.faces.items() for name,value in vars(face.surface).items() if type(value) is np.ndarray}


@pytest.mark.parametrize('kind',['plane','cylinder'])
@pytest.mark.parametrize('preserve',[False,True])
def test_clone_adapter_preserves_document_constructor_and_source(kind,preserve):
    from anygeometry._planning_model_state import clone_for_planning
    from anygeometry._planning_executor import freeze_model,unpack
    from anygeometry.serialization import to_dict
    if kind=='plane':
        from anygeometry._planning_canaries import _fixture
        source,_=_fixture()
    else:
        from test_planning_constructor_state import model,support
        source=model();support(source)._circumferential.flags.writeable=False
    source._serialization_extensions={'test:retained':{'values':[1,-0.,True]}}
    query_trimmed_surface_charts(source,tuple(source.faces))
    source._arc_cache['retained sentinel']='keep'
    cache_id=id(source._arc_cache);state=array_state(source);original=freeze_model(source)
    made=clone_for_planning(source,preserve_identity=preserve)
    assert (made.model_id==source.model_id) is preserve and made.revision==source.revision
    query_trimmed_surface_charts(made,tuple(made.faces))
    rebuilt=unpack(freeze_model(made),'model');before=unpack(original,'model')
    assert rebuilt.get('constructor_state',[])==before.get('constructor_state',[])
    left=to_dict(source);right=to_dict(made)
    for name in ('model_id','checksum'):left.pop(name);right.pop(name)
    assert json.dumps(left,sort_keys=True)==json.dumps(right,sort_keys=True)
    assert freeze_model(source)==original and array_state(source)==state
    assert id(source._arc_cache)==cache_id and source._arc_cache['retained sentinel']=='keep'


@pytest.mark.parametrize('state',['extra','hook','journal','features','extension_alias'])
def test_invalid_original_state_cannot_be_erased_by_clone(monkeypatch,state):
    from anygeometry._planning_model_state import clone_for_planning
    from anygeometry._planning_canaries import _fixture
    from anygeometry import GeometryModel
    source,_=_fixture()
    if state=='extra':source.foreign_authority=True
    elif state=='hook':source._change_hooks.append(lambda *args:None)
    elif state=='journal':source._transaction_journal={}
    elif state=='features':source.features._next_id=2
    else:
        shared=[];source._serialization_extensions={'shared':shared};source._arc_cache['shared']=shared
    monkeypatch.setattr(GeometryModel,'clone',lambda *args,**kwargs:pytest.fail('invalid original reached clone'))
    with pytest.raises(QualificationDecline):clone_for_planning(source,preserve_identity=True)


def test_initial_decline_scope_is_nested_restored_and_serial(monkeypatch):
    from types import SimpleNamespace
    from anygeometry import _planning_controller as controller
    from anygeometry.runtime_diagnostics import collect_runtime_records
    monkeypatch.setattr(controller,'_storage',lambda:pytest.fail('decline reached capture'))
    policy=SimpleNamespace(cancellation_check=None,max_predicates=None,max_candidate_pairs=None)
    assert controller._INITIAL_DECLINE.get() is None
    with controller._initial_decline('outer'):
        with pytest.raises(RuntimeError):
            with controller._initial_decline('inner'):
                with collect_runtime_records() as records:
                    assert controller.run(None,lambda:'serial',lambda _:pytest.fail('dispatch'),policy=policy,mode='exact')=='serial'
                assert records['parallel_planning'][-1]['reason']=='inner'
                raise RuntimeError('restore')
        assert controller._INITIAL_DECLINE.get()=='outer'
    assert controller._INITIAL_DECLINE.get() is None


def test_invalid_extension_namespace_is_not_normalized():
    from anygeometry._planning_model_state import clone_for_planning
    from anygeometry._planning_executor import freeze_model
    from anygeometry._planning_canaries import _fixture
    from anygeometry.errors import GeometryError
    source,_=_fixture();source._serialization_extensions={'retained':{'values':[1,-0.,True]}}
    for operation in (freeze_model,clone_for_planning):
        with pytest.raises(GeometryError,match='extension keys'):operation(source)


@pytest.mark.parametrize('mutation',['valid','invalid','intermediate_decline','roundtrip_decline'])
def test_source_mutation_during_clone_is_terminal_before_eligible_decline(monkeypatch,mutation):
    from anygeometry import GeometryModel
    from anygeometry._planning_model_state import clone_for_planning
    from anygeometry import _planning_executor as executor
    from anygeometry._planning_canaries import _fixture
    source,_=_fixture();original=GeometryModel.clone
    def clone(self,**kwargs):
        made=original(self,**kwargs)
        if mutation=='invalid':self.unknown_during_clone=True
        else:self._units='cm'
        if mutation=='intermediate_decline':made._units='mm'
        return made
    monkeypatch.setattr(GeometryModel,'clone',clone)
    if mutation=='roundtrip_decline':
        def refuse(*args,**kwargs):raise QualificationDecline('injected roundtrip decline')
        monkeypatch.setattr(executor,'thaw_model',refuse)
    with pytest.raises(executor.PlannerSourceChanged,match='source changed'):
        clone_for_planning(source,preserve_identity=True)
