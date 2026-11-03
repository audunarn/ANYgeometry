"""R6 role separation keeps definitions, typed operation edges and the bound."""
import pytest
from _planning_contract_helpers import capture_with_dispatcher_boundary
from anygeometry import _planning_manifest as manifest
from anygeometry._planning_qualification import QualificationDecline
from anygeometry.model import GeometryModel


def test_common_model_operations_do_not_execute_authoring_methods():
    methods=manifest.operation_methods(GeometryModel,'common')
    assert {'vertex_position','oriented_start_vertex','arc_frame','tags_for'}<=methods
    assert not {'save','move_point','transaction','add_vertex','split_edge'} & methods


def test_common_roots_exclude_host_orchestration_and_document_authority():
    roles=manifest.operation_roots()
    common={function.__name__ for function in roles['common']}
    assert 'stage_task' in common
    assert not {'_plan_intersections','to_dict','from_dict'} & common
    assert '_plan_intersections' in {f.__name__ for f in roles['host']}
    assert {'from_dict','to_dict'}<={f.__name__ for f in roles['constructor']}


def test_nonexecuting_annotation_data_still_consumes_the_bound():
    def root():return None
    root.__annotations__={'return':list(range(80001))}
    with pytest.raises(QualificationDecline,match='bound|truncat'):
        manifest.capture_callable_manifest((root,))


def test_worker_root_memo_and_collectors_cannot_use_host_policy():
    from anygeometry.analytic_roots import _ROOT_MEMO
    from anygeometry.runtime_diagnostics import _COLLECTORS
    for context,value in ((_ROOT_MEMO,{}),(_COLLECTORS,({},))):
        token=context.set(value)
        try:
            manifest.context_state(context,'registered',role='host')
            with pytest.raises(QualificationDecline):
                manifest.context_state(context,'registered',role='worker')
        finally:context.reset(token)


def test_shared_finite_definition_is_bound_once_and_mutation_refuses(request):
    from anygeometry._planning_executor import PlannerSourceChanged
    shared=[1,2,3]
    def root():return shared
    root.__annotations__={'first':shared,'second':shared}
    captured=capture_with_dispatcher_boundary((root,root),request=request)
    if captured is None:return # Exact initial refusal is not postcapture coverage.
    assert captured.nodes<100
    assert captured.graph['unique_objects']<captured.nodes
    captured.verify()
    shared.append(4)
    with pytest.raises(PlannerSourceChanged):captured.verify()


def test_operation_registry_rejects_live_dynamic_model_call(monkeypatch):
    original=GeometryModel.vertex_position
    def changed(self,key):return getattr(self,'_require_vertex')(key).position
    monkeypatch.setattr(GeometryModel,'vertex_position',changed)
    with pytest.raises(QualificationDecline,match='operation caller'):
        manifest.verify_operation_registry()
    monkeypatch.setattr(GeometryModel,'vertex_position',original)


def test_shared_role_graph_counts_only_new_expansions_and_still_rechecks(request):
    from anygeometry._planning_executor import PlannerSourceChanged
    shared_data=[1,2,3]
    def first():return shared_data
    def second():return shared_data
    graph={};budget=[0]
    common=capture_with_dispatcher_boundary((first,),shared=graph,budget=budget,request=request)
    if common is None:return # Exact initial refusal is not postcapture coverage.
    extra=capture_with_dispatcher_boundary((second,),shared=graph,budget=budget,request=request)
    if extra is None:return # Exact initial refusal is not postcapture coverage.
    assert budget[0]==common.nodes+extra.nodes
    assert extra.nodes<common.nodes
    check={};common.verify(shared=check);extra.verify(shared=check)
    shared_data[0]=9
    with pytest.raises(PlannerSourceChanged):common.verify(shared={})


def test_cancellation_authority_keeps_selected_event_and_allows_unrelated_signal():
    from anygeometry.cancellation import CancellationSignal
    from anygeometry._planning_executor import PlannerCancelled
    signal=CancellationSignal();authority=manifest.CancellationAuthority(signal)
    other=CancellationSignal();other.cancel()
    authority.verify()
    with pytest.raises(QualificationDecline,match='not empty'):
        manifest.cancellation_registry(worker=True)
    signal.cancel()
    with pytest.raises(PlannerCancelled):authority.verify()


def test_cancellation_authority_rejects_forged_signal():
    from anygeometry.cancellation import CancellationSignal
    with pytest.raises(QualificationDecline,match='not issued'):
        manifest.CancellationAuthority(object.__new__(CancellationSignal))


@pytest.mark.parametrize('change',['container','event','wrongtype'])
def test_cancellation_authority_rejects_substitution(monkeypatch,change):
    import threading,weakref
    from anygeometry import cancellation
    signal=cancellation.CancellationSignal();authority=manifest.CancellationAuthority(signal)
    if change=='container':monkeypatch.setattr(cancellation,'_EVENTS',weakref.WeakKeyDictionary())
    else:monkeypatch.setitem(cancellation._EVENTS,signal,threading.Event() if change=='event' else object())
    with pytest.raises(QualificationDecline):authority.verify()


@pytest.mark.parametrize('change',['namespace','function','module','names'])
def test_reference_namespace_rejects_rebinding(monkeypatch,change):
    from anygeometry import backend
    owner=manifest.reference_namespace()
    assert owner is not None
    if change=='namespace':monkeypatch.setattr(backend,'_RUNTIME_PYTHON_GLOBALS',dict(vars(owner)))
    elif change=='function':monkeypatch.setattr(owner,'find_spans',lambda *args:None)
    elif change=='module':monkeypatch.setattr(owner,'np',object())
    else:monkeypatch.setattr(backend,'_RUNTIME_REFERENCE_NAMES',('unknown',))
    with pytest.raises(QualificationDecline,match='reference namespace'):
        manifest.reference_namespace()


def test_filtered_history_follows_transitive_methods_and_rejects_changed_body(monkeypatch):
    from anygeometry.features import FeatureHistory
    selected=manifest.operation_methods(FeatureHistory,'common')
    assert {'__init__','_bind_owner','validate','dependencies_of','_get_record'}<=selected
    assert 'recompute' not in selected
    def changed(self):return getattr(self,'unregistered_operation')()
    monkeypatch.setattr(FeatureHistory,'validate',changed)
    with pytest.raises(QualificationDecline,match='filtered method'):
        manifest.operation_methods(FeatureHistory,'common')


@pytest.mark.parametrize('value',[None,(),{},[],set()])
def test_unknown_explicit_context_binding_refuses_even_when_empty(value):
    from contextvars import ContextVar
    variable=ContextVar('unregistered-planning-test')
    manifest.context_state(variable,'unknown')
    token=variable.set(value)
    try:
        with pytest.raises(QualificationDecline,match='active callable context'):
            manifest.context_state(variable,'unknown')
    finally:variable.reset(token)


def test_operation_registry_rejects_source_change(monkeypatch):
    from pathlib import Path
    original=Path.read_bytes
    monkeypatch.setattr(Path,'read_bytes',lambda path:original(path)+b'\n')
    with pytest.raises(QualificationDecline,match='operation caller source'):
        manifest.verify_operation_registry()


@pytest.mark.parametrize('role',['host','constructor'])
def test_feature_history_geometry_argument_edges_are_selected(role):
    methods=manifest.operation_methods(GeometryModel,role)
    edges=manifest._ROLE_MODEL_CALLER_EDGES[role]
    actual={method for caller,receiver,method in edges if caller.startswith('anygeometry.features.FeatureHistory.')}
    assert {'resolve_ref','group'}<=actual<=methods
