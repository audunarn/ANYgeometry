"""R7 connected admission regressions; wrapper authority is never unwrapped away."""
import types
import pytest
from anygeometry import _planning_manifest as manifest
from anygeometry._planning_qualification import QualificationDecline
from anygeometry.model import GeometryModel


def test_actual_transactional_host_callers_are_admitted():
    manifest.verify_operation_registry('host')


@pytest.mark.parametrize('change',['closure','wrapped','code','factory','body'])
def test_transactional_wrapper_relationship_mutations_refuse(monkeypatch,change):
    from anygeometry import model
    original=GeometryModel.add_arc
    body=vars(original)['__wrapped__']
    if change=='closure':
        cells=tuple(types.CellType(lambda *args:None) for cell in original.__closure__)
        forged=types.FunctionType(original.__code__,original.__globals__,original.__name__,original.__defaults__,cells)
        forged.__dict__.update(vars(original));forged.__qualname__=original.__qualname__
        monkeypatch.setattr(GeometryModel,'add_arc',forged)
    elif change=='wrapped':monkeypatch.setitem(vars(original),'__wrapped__',lambda *args:None)
    elif change=='code':monkeypatch.setattr(original,'__code__',original.__code__.replace(co_filename='<forged-wrapper>'))
    elif change=='body':monkeypatch.setattr(body,'__code__',body.__code__.replace(co_filename='<forged-body>'))
    else:monkeypatch.setattr(model,'_transactional',lambda function:function)
    with pytest.raises(QualificationDecline):manifest.verify_operation_registry('host')


def test_transaction_wrapper_selects_context_manager_model_edges():
    methods=manifest.operation_methods(GeometryModel,'host')
    assert {'transaction','_enter_transaction','_exit_transaction','_commit_transaction','_rollback_transaction'}<=methods


def test_transaction_context_forwarding_mutation_refuses(monkeypatch):
    from anygeometry.transactions import TopologyTransaction
    monkeypatch.setattr(TopologyTransaction,'__enter__',lambda self:self)
    with pytest.raises(QualificationDecline):manifest.operation_methods(GeometryModel,'host')


def test_persistence_registry_constructs_but_does_not_execute_stored_executors():
    from anygeometry.features import builtin_feature_registry,FeatureRegistry,FeatureHistory
    instructions=list(manifest._registry_factory_instructions(builtin_feature_registry))
    names={op.argval for op in instructions if op.opname in ('LOAD_GLOBAL','IMPORT_NAME')}
    assert {'FeatureRegistry','_builtin_topology_role'}<=names
    assert not {'_before','_one','_created_outputs'}&names
    assert builtin_feature_registry().has('geometry.arc')
    assert 'execute' not in manifest.operation_methods(FeatureRegistry,'host')
    assert 'recompute' not in manifest.operation_methods(FeatureHistory,'constructor')


@pytest.mark.parametrize('change',['factory','executor','query'])
def test_persistence_registry_definition_and_query_mutations_refuse(monkeypatch,change):
    from anygeometry import features
    factory=features.builtin_feature_registry
    if change=='query':
        monkeypatch.setattr(features.FeatureRegistry,'has',lambda self,kind:self.execute(None,None,None))
        with pytest.raises(QualificationDecline):manifest.operation_methods(features.FeatureRegistry,'host')
    else:
        code=factory.__code__
        if change=='executor':
            constants=tuple(value.replace(co_filename='<changed-executor>') if type(value) is types.CodeType and value.co_name=='arc' else value for value in code.co_consts)
            changed=code.replace(co_consts=constants)
        else:changed=code.replace(co_filename='<changed-factory>')
        monkeypatch.setattr(factory,'__code__',changed)
        with pytest.raises(QualificationDecline):list(manifest._registry_factory_instructions(factory))


def test_registered_quadrature_definition_keeps_flags_and_rejects_byte_drift():
    from anygeometry import section_faces
    from anygeometry._planning_executor import PlannerSourceChanged
    first,second=section_faces._GAUSS
    flags=first.flags.writeable,second.flags.writeable
    captured=manifest.capture_callable_manifest((first,second))
    original=first[0]
    try:
        first[0]=original+1
        with pytest.raises(PlannerSourceChanged):captured.verify()
    finally:first[0]=original
    assert (first.flags.writeable,second.flags.writeable)==flags


def test_definition_to_execution_upgrade_keeps_callable_default_closure():
    from anygeometry._planning_executor import PlannerSourceChanged
    state=[3]
    def callback():return state[0]
    def root(function=callback):return function()
    root.__annotations__={'return':callback}
    captured=manifest.capture_callable_manifest((root,))
    assert captured.graph['definition_expansions']>0
    assert captured.graph['execution_expansions']>0
    captured.verify()
    state[0]=7
    with pytest.raises(PlannerSourceChanged):captured.verify()


def test_definition_upgrade_retains_distinct_alias_bindings():
    from anygeometry._planning_executor import PlannerSourceChanged
    state=[1]
    def callback():return state[0]
    def root(function=callback):return function()
    root.__annotations__={'a':state,'b':state,'callback':callback}
    captured=manifest.capture_callable_manifest((root,))
    root.__annotations__['b']=[1]
    with pytest.raises(PlannerSourceChanged):captured.verify()
