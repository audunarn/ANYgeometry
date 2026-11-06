"""Production-controller refusals preserve original serial behavior."""
from _planning_runtime_observation import capture_witness,retain_observation
from types import SimpleNamespace
from pathlib import Path
import pytest
from anygeometry import _planning_controller as controller
from anygeometry.runtime_diagnostics import collect_runtime_records
from _planning_numpy_profile_contract import numpy_profile_test


def _finalization_fixture(monkeypatch,tmp_path):
    """Exercise real controller finalization, with no native/model execution."""
    from anygeometry import _planning_manifest,_planning_code,_planning_model_state,serialization
    work=tmp_path/'call'/'work';work.mkdir(parents=True)
    packet={'sha256':'model'};plan=SimpleNamespace(content_checksum='plan')
    manifest=SimpleNamespace(callables=SimpleNamespace(digest='common',_reference_tokens=[]),
        roles={name:SimpleNamespace(digest=name,_reference_tokens=[]) for name in ('constructor','host')},
        wire={'sha256':'manifest'},parsing_release={},verify=lambda:None)
    lease=SimpleNamespace(tree_limit_bytes=100,check=lambda:None,report=lambda:{},observe_phase=lambda *args:None)
    # Fake models must reach the admitted parallel internals; the actual cost
    # gate is covered separately in test_r11_planning.py.
    monkeypatch.setattr(controller,'_estimate_cost',lambda model,operands=None:(10**6,'test admit'))
    monkeypatch.setattr(controller,'_storage',lambda:(work.parent,work))
    monkeypatch.setattr(controller,'memory_profile',lambda:SimpleNamespace(begin=lambda:lease))
    monkeypatch.setattr(controller,'freeze_model',lambda model:packet)
    monkeypatch.setattr(serialization,'to_dict',lambda model:{})
    monkeypatch.setattr(_planning_model_state,'production_snapshot',lambda model:packet)
    monkeypatch.setattr(_planning_manifest,'prepare_interfaces',lambda:None)
    monkeypatch.setattr(_planning_manifest,'capture_loaded_manifest',lambda **kw:manifest)
    monkeypatch.setattr(_planning_code,'InfrastructurePin',lambda role:SimpleNamespace(digest='infra',nodes=1,native={},parsing_release={},verify=lambda:None))
    monkeypatch.setattr(controller,'source_files',lambda:{})
    monkeypatch.setattr(controller,'verify_sources',lambda source:None)
    class Bridge:
        pending=[];attempts=[];retired=True;_retirement_error=None
        def __init__(self,*args,**kwargs):pass
        def finish(self):pass
        def close(self):pass
        def verify_pending(self):pass
        def check_clock(self):pass
    monkeypatch.setattr(controller,'_ProductionBridge',Bridge)
    policy=SimpleNamespace(cancellation_check=None,max_predicates=None,max_candidate_pairs=None)
    return work,plan,policy


@pytest.mark.parametrize('name',['PlannerDeadline','PlannerCancelled','PlannerSourceChanged','PlannerCleanupError','PlannerMemoryExceeded'])
def test_production_sample_terminal_preserves_original_after_retirement(monkeypatch,tmp_path,name):
    import json
    from anygeometry import _planning_host as host,_planning_executor as executor
    from anygeometry._planning_values import encode_value
    from anygeometry.errors import GeometryError
    work,plan,policy=_finalization_fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(controller,'_ProductionBridge',host._SessionBridge)
    error=getattr(executor,name)('original production sample terminal');seen=[];armed=[False]
    def parallel(bridge):
        seen.append(bridge);original=bridge.poll
        def poll():
            if armed[0]:raise error
            original()
        bridge.poll=poll
        def kernel(task):
            assert bridge.retired is True and bridge.session is None
            armed[0]=True
            try:
                try:bridge.check_deadline()
                except GeometryError:pytest.fail('kernel retry swallowed terminal sample error')
            finally:armed[0]=False
        candidate=dict(sampling_sha256='c'*64,task_layout=((0,0,'pair'),),
            results=[json.dumps(encode_value(None,codec='v2'))])
        bridge.pending.append(('pair',({'kind':'pair','cost':1.},),kernel,candidate))
        return plan
    with collect_runtime_records() as records,pytest.raises(type(error)) as caught:
        host.qualified_plan(None,None,lambda:pytest.fail('terminal sample replayed'),parallel,policy=policy,mode='exact')
    assert caught.value is error and seen[0]._inside_numerical_call is False
    receipt=records['parallel_planning'][-1]
    assert receipt['disposition']=='terminal' and receipt['retired'] is True and receipt['stages']==[]
    assert str(error) in receipt['reason'] and work.is_dir()


@pytest.mark.parametrize('fault',['remove','receipt_replay','receipt_cancel'])
def test_finalization_fault_always_records_terminal_original_error(monkeypatch,tmp_path,fault):
    import json
    from anygeometry._planning_executor import PlannerCancelled
    work,plan,policy=_finalization_fixture(monkeypatch,tmp_path)
    failure=PlannerCancelled('replay cancelled') if fault=='receipt_cancel' else OSError('original finalization failure')
    if fault=='remove':
        def remove(path):raise failure
        monkeypatch.setattr(controller.shutil,'rmtree',remove)
    else:
        write=Path.write_text
        def fail_receipt(path,*args,**kwargs):
            if path.name=='receipt.json':raise OSError('receipt unavailable')
            return write(path,*args,**kwargs)
        monkeypatch.setattr(Path,'write_text',fail_receipt)
    def serial():raise failure
    with collect_runtime_records() as records:
        with pytest.raises(type(failure)) as caught:
            controller.run(None,serial,lambda bridge:plan,policy=policy,mode='exact')
    assert caught.value is failure
    receipt=records['parallel_planning'][-1]
    assert receipt['disposition']=='terminal' and str(failure) in receipt['reason']
    assert receipt['retired'] is True and work.is_dir()
    if fault=='remove':assert json.loads((work.parent/'receipt.json').read_text())['disposition']=='terminal'
    else:assert receipt['receipt_error']=='receipt unavailable'


def test_cleanup_failure_does_not_replace_primary_exception(monkeypatch,tmp_path):
    from anygeometry._planning_executor import PlannerCleanupError
    work,plan,policy=_finalization_fixture(monkeypatch,tmp_path)
    primary=RuntimeError('primary callback failure');cleanup=PlannerCleanupError('owned cleanup failure')
    def close(self):
        self.retired=False;self._retirement_error=cleanup
        raise cleanup
    monkeypatch.setattr(controller._ProductionBridge,'close',close)
    def parallel(bridge):raise primary
    with collect_runtime_records() as records,pytest.raises(RuntimeError) as caught:
        controller.run(None,lambda:pytest.fail('replay'),parallel,policy=policy,mode='exact')
    assert caught.value is primary
    receipt=records['parallel_planning'][-1]
    assert receipt['disposition']=='terminal' and receipt['retired'] is False
    assert 'primary callback failure' in receipt['reason'] and 'owned cleanup failure' in receipt['cleanup_error']
    assert work.is_dir()


def test_receipt_failure_successful_replay_retains_scratch(monkeypatch,tmp_path):
    work,plan,policy=_finalization_fixture(monkeypatch,tmp_path)
    original=Path.write_text
    def write(path,*args,**kwargs):
        if path.name=='receipt.json':raise OSError('receipt unavailable')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'write_text',write)
    calls=[];serial_plan=object()
    with collect_runtime_records() as records:
        result=controller.run(None,lambda:calls.append('serial') or serial_plan,lambda bridge:plan,policy=policy,mode='exact')
    assert result is serial_plan and calls==['serial'] and work.is_dir()
    assert records['parallel_planning'][-1]['disposition']=='serial-receipt-failure'


def test_receipt_counts_every_poll_memory_check_and_host_phase(monkeypatch,tmp_path):
    work,plan,policy=_finalization_fixture(monkeypatch,tmp_path)
    from anygeometry import _planning_host as host
    monkeypatch.setattr(controller,'_ProductionBridge',host._SessionBridge)
    lease=controller.memory_profile().begin();checks=[]
    lease.check=lambda:checks.append('fresh')
    def parallel(bridge):
        # Real active checks: an explicit full boundary and a wait iteration,
        # not a fabricated record. Ordinary polls stay below the 2048/25ms gate.
        bridge.poll();bridge.poll.boundary();bridge.poll.wait()
        assert len(checks)==2
        return plan
    with collect_runtime_records() as records:
        assert controller.run(None,lambda:None,parallel,policy=policy,mode='exact') is plan
    costs=records['parallel_planning'][-1]['costs']
    assert costs['boundary_totals']['memory_check_seconds']['count']==len(checks)>=2
    assert set(costs['host_phases'])=={'initial_snapshot','orchestration','finish','retirement','samples_and_second_decode','final_boundary'}
    assert all(value>=0 for value in costs['host_phases'].values())


@pytest.mark.parametrize('change',['identity','presence','value'])
def test_postcapture_initial_decline_context_mutation_is_terminal(monkeypatch,tmp_path,change):
    import contextvars
    from anygeometry._planning_executor import PlannerSourceChanged
    work,plan,policy=_finalization_fixture(monkeypatch,tmp_path)
    variable=controller._INITIAL_DECLINE;tokens=[]
    def parallel(bridge):
        if change=='identity':monkeypatch.setattr(controller,'_INITIAL_DECLINE',contextvars.ContextVar(variable.name,default=None))
        else:tokens.append(variable.set(None if change=='presence' else 'changed after capture'))
        return plan
    try:
        with collect_runtime_records() as records,pytest.raises(PlannerSourceChanged):
            controller.run(None,lambda:pytest.fail('terminal context drift replayed'),parallel,policy=policy,mode='exact')
        assert records['parallel_planning'][-1]['disposition']=='terminal'
    finally:
        for token in reversed(tokens):variable.reset(token)


@pytest.mark.parametrize('fallback',[False,True])
def test_production_diagnostics_publish_only_final_path_once(monkeypatch,tmp_path,fallback):
    from anygeometry import runtime_diagnostics as diagnostics
    from anygeometry._planning_qualification import QualificationDecline
    work,plan,policy=_finalization_fixture(monkeypatch,tmp_path)
    def emit(name,amount):
        diagnostics._increment('batch_face_pairs_candidates',amount)
        diagnostics._record('test_path',name)
    def parallel(bridge):
        emit('candidate',3)
        bridge.attempts=[dict(stage='pair',received=3,total=3,received_weight=7.,total_weight=7.)]
        return plan
    def verify(self):
        emit('sample',19)
        if fallback:raise QualificationDecline('sample mismatch')
    monkeypatch.setattr(controller._ProductionBridge,'verify_pending',verify)
    def serial():emit('serial',5);return 'serial'
    with diagnostics.collect_runtime_diagnostics() as outer,collect_runtime_records() as records:
        with diagnostics.collect_runtime_diagnostics() as inner:
            result=controller.run(None,serial,parallel,policy=policy,mode='exact')
    expected=5 if fallback else 3
    assert outer['batch_face_pairs_candidates']==inner['batch_face_pairs_candidates']==expected
    assert records['test_path']==['serial' if fallback else 'candidate']
    receipt=records['parallel_planning'][-1]
    assert receipt['attempted_stages'][0]['received']==3
    if fallback:assert result=='serial' and receipt['stages']==[] and work.is_dir()
    else:assert result is plan and not work.exists()


@pytest.mark.parametrize('action',['model','code','exception','cancel'])
@numpy_profile_test
def test_real_public_callback_after_capture_is_terminal(monkeypatch,tmp_path,action,request):
    from anygeometry import batch_intersections as bi,_planning_values
    from anygeometry._planning_canaries import _fixture
    from anygeometry._planning_executor import PlannerCancelled,PlannerSourceChanged
    model,_=_fixture();calls=[];witness=[];original=RuntimeError('original callback exception')
    # Private admission only: keep the real public model/code/exception/cancel path.
    monkeypatch.setattr(controller,'_estimate_cost',lambda m,o=None:(10**6,'test admit'))
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    monkeypatch.setenv('ANYGEOMETRY_PLANNING_SCRATCH_ROOT',str(tmp_path/'production'))
    def callback():
        calls.append(None)
        phase=capture_witness();witness.append(phase)
        if not phase['captured']:return False
        if action=='model':model.unknown_callback_authority=True
        elif action=='code':monkeypatch.setattr(_planning_values.encode_value,'__code__',_planning_values.encode_value.__code__.replace())
        elif action=='exception':raise original
        else:return True
        return False
    expected=RuntimeError if action=='exception' else PlannerCancelled if action=='cancel' else PlannerSourceChanged
    with collect_runtime_records() as records,retain_observation(request,records,
            label='callback-'+action,witness=witness),pytest.raises(expected) as caught:
        bi.plan_intersections(model,tuple(model.faces),policy=bi.IntersectionBatchPolicy(cancellation_check=callback),_parallel_control=None)
    assert any(phase['captured'] for phase in witness),witness
    if action=='exception':assert caught.value is original
    receipt=records['parallel_planning'][-1]
    first=next(index for index,phase in enumerate(witness) if phase['captured'])
    if action in ('exception','cancel'):
        assert len(calls)==first+1          # aborts immediately at the trigger poll
    else:
        # Mutation is first applied at the first captured callback and is not
        # accepted: it is caught at the next explicit verification boundary
        # (ordinary polls no longer verify), after a bounded number of callbacks.
        assert first+1<len(calls)<=200
        assert receipt['stages']==[]
    assert receipt['costs']['callback_calls']==len(calls) and receipt['manifest_sha256']
    assert receipt['disposition']=='terminal' and receipt['stages']==[] and receipt['retired'] is True


@pytest.mark.parametrize('failure',['missing_environment','mkdir','inventory','free_space'])
def test_initial_storage_unavailable_replays_original_serial(monkeypatch,tmp_path,failure):
    root=tmp_path/'storage'
    monkeypatch.setattr(controller,'_estimate_cost',lambda m,o=None:(10**6,'test admit'))
    monkeypatch.delenv('ANYGEOMETRY_PLANNING_SCRATCH_ROOT',raising=False)
    monkeypatch.delenv('LOCALAPPDATA',raising=False)
    def denied(*args,**kwargs):raise OSError('storage unavailable')
    if failure!='missing_environment':
        monkeypatch.setenv('ANYGEOMETRY_PLANNING_SCRATCH_ROOT',str(root))
        if failure=='mkdir':monkeypatch.setattr(Path,'mkdir',denied)
        elif failure=='inventory':monkeypatch.setattr(Path,'rglob',denied)
        else:monkeypatch.setattr(controller.shutil,'disk_usage',denied)
    policy=SimpleNamespace(cancellation_check=None,max_predicates=None,max_candidate_pairs=None)
    calls=[];result=object()
    def serial():calls.append('serial');return result
    with collect_runtime_records() as records:
        actual=controller.run(None,serial,lambda bridge:pytest.fail('parallel dispatch after storage refusal'),
            policy=policy,mode='exact')
    assert actual is result and calls==['serial']
    receipt=records['parallel_planning'][-1]
    assert receipt['disposition']=='serial' and receipt['retired'] is True
    assert 'storage' in receipt['reason']


@pytest.mark.parametrize('warm',[False,True])
def test_consolidated_snapshot_matches_full_constructor_packet(warm):
    from test_planning_constructor_state import model
    from anygeometry._planning_executor import freeze_model
    from anygeometry._planning_model_state import production_snapshot
    from anygeometry import query_trimmed_surface_charts
    geometry=model()
    if warm:query_trimmed_surface_charts(geometry,tuple(geometry.faces))
    expected=freeze_model(geometry)
    assert production_snapshot(geometry)==expected
    assert freeze_model(geometry)==expected


@pytest.mark.parametrize('mutation',['extra','extension_alias','derived_alias','weak_alias','gc_view','derived_bits'])
def test_consolidated_snapshot_preserves_constructor_refusals(mutation):
    import numpy as np,weakref
    from test_planning_constructor_state import model,support
    from anygeometry._planning_executor import freeze_model
    from anygeometry._planning_model_state import production_snapshot
    from anygeometry._planning_qualification import QualificationDecline
    geometry=model();freeze_model(geometry);surface=support(geometry);retained=None
    if mutation=='extra':geometry.unknown_authority=True
    elif mutation=='extension_alias':
        shared=[];geometry._serialization_extensions={'shared':shared};geometry._arc_cache['alias']=shared
    elif mutation=='derived_alias':retained=[surface._circumferential]
    elif mutation=='weak_alias':retained=weakref.ref(surface._circumferential)
    elif mutation=='gc_view':
        class View(np.ndarray):pass
        retained=surface._circumferential.view(View)
    else:surface._circumferential[0]+=0.125
    for snapshot in (production_snapshot,freeze_model):
        with pytest.raises(QualificationDecline):snapshot(geometry)


@pytest.mark.parametrize('mutation',['declared_bits','derived_flags','extension_presence'])
def test_consolidated_snapshot_detects_exact_admitted_changes(mutation):
    from test_planning_constructor_state import model,support
    from anygeometry._planning_executor import freeze_model
    from anygeometry._planning_model_state import production_snapshot
    geometry=model();before=freeze_model(geometry);surface=support(geometry)
    if mutation=='declared_bits':
        import numpy as np
        changed=surface.origin.copy()
        index=next((index for index,value in enumerate(changed) if value!=0.),None)
        if index is None:changed[0]=-0.
        else:changed[index]=np.nextafter(changed[index],float('inf'))
        changed.flags.writeable=False
        object.__setattr__(surface,'origin',changed)
    elif mutation=='derived_flags':surface._circumferential.flags.writeable=False
    elif '_serialization_extensions' in vars(geometry):del geometry._serialization_extensions
    else:geometry._serialization_extensions={}
    assert production_snapshot(geometry)==freeze_model(geometry)
    assert production_snapshot(geometry)!=before


def test_consolidated_snapshot_preserves_invalid_topology_refusal():
    from test_planning_constructor_state import model,support
    from anygeometry._planning_executor import freeze_model
    from anygeometry._planning_model_state import production_snapshot
    from anygeometry.errors import GeometryError
    geometry=model();surface=support(geometry)
    changed=surface.origin.copy();changed[0]+=0.125;changed.flags.writeable=False
    object.__setattr__(surface,'origin',changed)
    for snapshot in (production_snapshot,freeze_model):
        with pytest.raises(GeometryError,match='cannot serialize invalid topology'):snapshot(geometry)
