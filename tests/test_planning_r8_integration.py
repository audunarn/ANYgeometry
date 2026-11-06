"""R8 full admission and production integration regressions."""
from _planning_runtime_observation import retain_observation
import dataclasses
import sys
import pytest
from _planning_legacy_contract import capture_with_legacy_dataclass_boundary
from _planning_platform_contract import windows_owned_test
from _planning_numpy_profile_contract import numpy_profile_test
from _planning_contract_helpers import capture_with_runtime_boundary, capture_with_dispatcher_boundary
from anygeometry import _planning_manifest as manifest
from anygeometry._planning_executor import PlannerSourceChanged
from anygeometry.tolerance import TolerancePolicy


# Python 3.14 stores the native type annotation cache separately. Earlier
# runtimes materialize the same annotation-free class metadata in __annotations__.
_CLASS_ANNOTATIONS = '__annotations_cache__' if sys.version_info >= (3, 14) else '__annotations__'


def test_dataclass_field_metadata_value_and_aliases_remain_bound(monkeypatch,request):
    field=dataclasses.fields(TolerancePolicy)[0]
    shared=[1]
    monkeypatch.setattr(field,'metadata',{'first':shared,'second':shared})
    captured=capture_with_legacy_dataclass_boundary((TolerancePolicy,),request=request)
    if captured is None:return # Exact initial refusal is not postcapture coverage.
    captured.verify()
    shared[0]=2
    with pytest.raises(PlannerSourceChanged):captured.verify()


def test_dataclass_field_flags_remain_bound(monkeypatch,request):
    field=dataclasses.fields(TolerancePolicy)[0]
    captured=capture_with_legacy_dataclass_boundary((TolerancePolicy,),request=request)
    if captured is None:return # Exact initial refusal is not postcapture coverage.
    monkeypatch.setattr(field,'repr',not field.repr)
    with pytest.raises(PlannerSourceChanged):captured.verify()


def test_registered_default_factory_definition_remains_bound(monkeypatch,request):
    from anygeometry._canonical_curve_runtime import _Bank
    field=next(field for field in dataclasses.fields(_Bank) if field.name=='declarations')
    captured=capture_with_legacy_dataclass_boundary((_Bank,),request=request)
    if captured is None:return # Exact initial refusal is not postcapture coverage.
    monkeypatch.setattr(field,'default_factory',list)
    with pytest.raises(PlannerSourceChanged):captured.verify()


def test_direct_slot_edges_equal_existing_diagnostic_metadata():
    from anygeometry import _planning_ledger as ledger
    values=(*dataclasses.fields(TolerancePolicy),vars(TolerancePolicy)['__dataclass_params__'])
    for value in values:
        before=ledger._metadata_parts(value)
        after=tuple((name,part) for name,descriptor,part in manifest._dataclass_slot_values(value))
        assert len(before)==len(after)
        assert all(a==b and first is second for (a,first),(b,second) in zip(before,after))


@pytest.mark.parametrize('change',['schema','descriptor'])
def test_dataclass_slot_schema_or_descriptor_substitution_refuses(monkeypatch,change):
    from anygeometry._planning_qualification import QualificationDecline
    field=dataclasses.fields(TolerancePolicy)[0]
    if change=='schema':monkeypatch.setattr(dataclasses.Field,'__slots__',dataclasses.Field.__slots__+('unknown',))
    else:monkeypatch.setattr(dataclasses.Field,'default',property(lambda self:pytest.fail('descriptor hook executed')))
    with pytest.raises(QualificationDecline):list(manifest._dataclass_slot_values(field))


def test_interface_preparation_only_materializes_exact_empty_class_metadata(monkeypatch):
    from anygeometry import nurbs
    from anygeometry._planning_canaries import _fixture
    model,_=_fixture();cache=model._arc_cache;cache[991]=object()
    contents=dict(cache)
    monkeypatch.delattr(nurbs._PreparedCurveState,_CLASS_ANNOTATIONS,raising=False)
    manifest.prepare_interfaces()
    metadata=vars(nurbs._PreparedCurveState)[_CLASS_ANNOTATIONS]
    assert type(metadata) is dict and not metadata
    manifest.prepare_interfaces()
    assert vars(nurbs._PreparedCurveState)[_CLASS_ANNOTATIONS] is metadata
    assert model._arc_cache is cache and cache==contents


@pytest.mark.parametrize('change',['nonempty','producer','class','source'])
def test_interface_preparation_rejects_annotation_authority_changes(monkeypatch,change):
    from anygeometry import nurbs
    from anygeometry._planning_qualification import QualificationDecline
    from pathlib import Path
    cls=nurbs._PreparedCurveState
    if change=='nonempty':monkeypatch.setattr(cls,_CLASS_ANNOTATIONS,{'field':int},raising=False)
    elif change=='producer':monkeypatch.setattr(cls,'__annotate_func__',lambda *args:pytest.fail('annotation producer executed'),raising=False)
    elif change=='class':monkeypatch.setattr(nurbs,'_PreparedCurveState',type('Other',(),{}))
    else:
        original=Path.read_bytes
        monkeypatch.setattr(Path,'read_bytes',lambda path:original(path).replace(b'class _PreparedCurveState:',b'class ChangedPreparedCurveState:'))
    with pytest.raises(QualificationDecline):manifest.prepare_interfaces()


def test_prepared_annotation_cache_replacement_is_terminal(monkeypatch,request):
    from anygeometry import nurbs
    manifest.prepare_interfaces()
    captured=capture_with_runtime_boundary((nurbs._PreparedCurveState,),request=request)
    if captured is None:return # Exact initial refusal is not postcapture coverage.
    monkeypatch.setattr(nurbs._PreparedCurveState,_CLASS_ANNOTATIONS,{})
    with pytest.raises(PlannerSourceChanged):captured.verify()

def _production_ledger_context_body(monkeypatch):
    from anygeometry import _planning_ledger as ledger,analytic_roots
    from anygeometry._planning_canaries import _fixture
    from anygeometry._planning_executor import freeze_model,thaw_model
    from anygeometry._planning_qualification import QualificationDecline
    model,_=_fixture();model=thaw_model(freeze_model(model))
    monkeypatch.setattr(ledger,'runtime_infrastructure',lambda:pytest.fail('legacy external-internals gate used'))
    owned=ledger.package_worker_ledger(model)
    assert owned.manifest.roles.keys()=={'constructor'}
    owned.prepare(owned_worker=True)
    token=analytic_roots._ROOT_MEMO.set({})
    try:
        with pytest.raises(QualificationDecline):owned.reset_for_batch(owned_worker=True)
    finally:analytic_roots._ROOT_MEMO.reset(token)
    owned.validate()


@numpy_profile_test
def test_production_ledger_has_strict_contexts_and_no_external_anchor(tmp_path,request):
    _run_worker_cache_child(tmp_path,request,'production_context')


@numpy_profile_test
def test_package_numpy_context_retains_getter_identity_without_anchor(monkeypatch):
    import types,numpy as np
    from anygeometry import _planning_ledger as ledger
    from anygeometry._planning_numpy_context import capture_numpy_context
    monkeypatch.setattr(ledger,'runtime_infrastructure',lambda:pytest.fail('external-internals anchor used'))
    captured=capture_numpy_context(package_only=True)
    captured.verify()
    original=np.geterr
    replacement=types.FunctionType(original.__code__,original.__globals__,original.__name__,
        original.__defaults__,original.__closure__)
    replacement.__module__=original.__module__
    monkeypatch.setattr(np,'geterr',replacement)
    monkeypatch.setattr(__import__('sys').modules[original.__module__],'geterr',replacement)
    with pytest.raises(PlannerSourceChanged):captured.verify()

@pytest.mark.parametrize('warm',[False,True])
@numpy_profile_test
@windows_owned_test
def test_public_eight_worker_canary_matches_serial(monkeypatch,tmp_path,warm,request):
    import os,json,psutil
    from pathlib import Path
    memory=psutil.virtual_memory()
    evidence_directory=Path(os.environ.get('TASK_EVIDENCE_DIRECTORY',tmp_path))
    evidence_directory.mkdir(parents=True,exist_ok=True)
    (evidence_directory/f'load-before-canary-{warm}.json').write_text(json.dumps(dict(
        cpu_percent=psutil.cpu_percent(interval=.1),available_bytes=memory.available,total_bytes=memory.total,
        logical_cpus=os.cpu_count(),python_process_count=None)))
    from anygeometry import batch_intersections as bi,query_trimmed_surface_charts
    from anygeometry._planning_canaries import _fixture
    from anygeometry.serialization import to_dict,from_dict
    from anygeometry.runtime_diagnostics import collect_runtime_records,collect_runtime_diagnostics
    model,_=_fixture();document=to_dict(model)
    source=from_dict(document);reference=from_dict(document);faces=tuple(source.faces)
    if warm:query_trimmed_surface_charts(source,faces)
    monkeypatch.setenv('ANYGEOMETRY_PLANNING_SCRATCH_ROOT',str(tmp_path/'production'))
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    with collect_runtime_records() as records,retain_observation(request,records,
            label='warm8-warm' if warm else 'warm8-cold'),collect_runtime_diagnostics() as parallel_counts:
        # Prospective R11 transition: this tiny canary is routed serial by the DEFAULT cost
        # selector (covered, unforced, by the separate public-selector tests). The required 8W
        # numerical/ownership coverage is kept by forcing ONLY the dispatch seam of the parallel
        # side; every other guard and all assertions below are unchanged. Not a default-selector claim.
        from anygeometry import _planning_controller as controller
        with collect_runtime_diagnostics() as nested_counts,controller._forced_route('CI required positive 8W canary'):
            plan=bi.plan_intersections(source,faces,policy='connect',_parallel_control=None)
    assert parallel_counts==nested_counts
    receipt=records['parallel_planning'][-1]
    assert receipt['forced_selector']=='CI required positive 8W canary',receipt
    assert receipt['disposition']=='parallel',receipt
    assert receipt['workers']==receipt['startup_concurrency']==8
    assert receipt['retired'] is True and receipt['ci_verified'] is False
    assert [stage['stage'] for stage in receipt['stages']]==['pair','face']
    assert all(stage['accepted']==stage['total'] and stage['total']>0 for stage in receipt['stages'])
    from anygeometry._planning_values import value_codec_contract
    codec=value_codec_contract()
    resource=receipt['stages'][0]['infrastructure']['resource']
    for stage in receipt['stages']:
        assert stage['infrastructure']['resource']==resource
        assert stage['infrastructure']['value_codec']==codec
        assert stage['infrastructure']['planner_reference']!=stage['infrastructure']['worker_reference']
    for stage in receipt['costs']['infrastructure']:
        assert len(stage['workers'])==8
        assert all(worker['value_codec']==codec for worker in stage['workers'])
        assert all(worker['resource_guard']==dict(controller_loaded=False,native_adapter_loaded=False,
            native_core_loaded=False,binary_sha256=resource['binary_sha256']) for worker in stage['workers'])
    from anygeometry._planning_code import exact_numerical_record
    import json
    parallel_record=exact_numerical_record(plan)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','0')
    with collect_runtime_diagnostics() as serial_counts:
        serial=bi.plan_intersections(reference,faces,policy='connect')
    assert parallel_record==exact_numerical_record(serial)
    assert plan.content_checksum==serial.content_checksum==receipt['plan_checksum']
    with collect_runtime_diagnostics() as parallel_apply:
        bi.apply_intersections(source,plan,policy='connect')
    with collect_runtime_diagnostics() as serial_apply:
        bi.apply_intersections(reference,serial,policy='connect')
    prefixes=('batch_face_pairs_','intersection_pair_','t2b_','planning_mode:',
        'ring_split_','arrangement_snap_','arrangement_seam_')
    def outcomes(data):return {key:value for key,value in data.items() if key.startswith(prefixes) and not key.endswith('_time_ns')}
    (evidence_directory/f'diagnostics-canary-{warm}.json').write_text(json.dumps(dict(
        parallel=parallel_counts,serial=serial_counts,parallel_apply=parallel_apply,serial_apply=serial_apply,
        classification='named deterministic outcomes compared; timing/work/cache counters retained separately')))
    assert outcomes(parallel_counts)==outcomes(serial_counts)
    assert outcomes(parallel_apply)==outcomes(serial_apply)
    assert json.dumps(to_dict(source),sort_keys=True,separators=(',',':'),allow_nan=False)==json.dumps(to_dict(reference),sort_keys=True,separators=(',',':'),allow_nan=False)

def test_slot_metadata_native_schema_inheritance_and_mangling():
    class _Base:__slots__='__secret'
    class Child(_Base):__slots__=('value','__weakref__')
    assert manifest._slot_names(Child)==['value','_Base__secret']


@pytest.mark.parametrize('change',['delete','replace','corrupt'])
def test_slot_metadata_recheck_is_binding_sensitive(monkeypatch,change,request):
    from anygeometry import nurbs
    cls=nurbs._PreparedCurveState
    manifest.prepare_interfaces()
    captured=capture_with_runtime_boundary((cls,),request=request)
    if captured is None:return # Exact initial refusal is not postcapture coverage.
    original=vars(cls)['__slotnames__'];expected=list(original)
    captured.verify();assert vars(cls)['__slotnames__'] is original
    if change=='delete':monkeypatch.delattr(cls,'__slotnames__')
    elif change=='replace':monkeypatch.setattr(cls,'__slotnames__',list(expected))
    else:monkeypatch.setattr(cls,'__slotnames__',expected+['foreign'])
    with pytest.raises(PlannerSourceChanged):captured.verify()


def test_cached_immutable_code_rejects_equivalent_live_code_replacement(monkeypatch,request):
    from anygeometry import _planning_code
    captured=capture_with_runtime_boundary((_planning_code.passive_data,),request=request)
    if captured is None:return # Exact initial refusal is not postcapture coverage.
    captured.verify()
    monkeypatch.setattr(_planning_code.passive_data,'__code__',_planning_code.passive_data.__code__.replace())
    with pytest.raises(PlannerSourceChanged):captured.verify()


def test_exact_numerical_evidence_distinguishes_signed_zero_and_array_types():
    import numpy as np
    from anygeometry._planning_code import exact_numerical_record as record
    assert record(0.)!=record(-0.)
    assert record(np.array([0.]))!=record(np.array([-0.]))
    assert record(np.array([1.],dtype='float32'))!=record(np.array([1.],dtype='float64'))
    assert record((1,))!=record([1])

@pytest.mark.parametrize('change',['global','default','closure','array'])
def test_retained_and_full_verifiers_agree_on_protected_mutation(monkeypatch,change,request):
    from anygeometry import _planning_code,runtime_diagnostics,surface_capabilities
    if change in ('global','default'):
        captured=capture_with_runtime_boundary((_planning_code.passive_data,),request=request)
        if captured is None:return # Exact initial refusal is not postcapture coverage.
        if change=='global':monkeypatch.setattr(_planning_code,'_encode',lambda value:b'changed')
        else:monkeypatch.setitem(_planning_code.passive_data.__kwdefaults__,'key',True)
    elif change=='closure':
        function=runtime_diagnostics.collect_runtime_records
        captured=capture_with_runtime_boundary((function,),request=request)
        if captured is None:return # Exact initial refusal is not postcapture coverage.
        monkeypatch.setattr(function.__closure__[0],'cell_contents',lambda:None)
    else:
        value=surface_capabilities._SEEDS
        captured=capture_with_dispatcher_boundary((value,),request=request)
        if captured is None:return # Exact initial refusal is not postcapture coverage.
        before=value.copy();value.flat[0]+=1.
    try:
        with pytest.raises(PlannerSourceChanged):captured.verify()
        with pytest.raises(PlannerSourceChanged):captured.verify(full=True)
    finally:
        if change=='array':value[:]=before

@pytest.mark.parametrize('change',['controller','bridge','ctypes','native_image'])
@numpy_profile_test
@windows_owned_test
def test_infrastructure_loaded_code_and_imported_interfaces_are_terminal(monkeypatch,tmp_path,change):
    import ctypes
    from pathlib import Path
    from anygeometry._planning_code import InfrastructurePin
    from anygeometry import _planning_controller as controller,_planning_host as host,_planning_native as native
    captured=InfrastructurePin('host');captured.verify()
    if change=='controller':monkeypatch.setattr(controller.run,'__code__',controller.run.__code__.replace())
    elif change=='bridge':monkeypatch.setattr(host._SessionBridge.run,'__code__',host._SessionBridge.run.__code__.replace())
    elif change=='ctypes':monkeypatch.setattr(ctypes,'WinDLL',lambda *a,**k:pytest.fail('changed ctypes interface invoked'))
    else:
        image=Path(native.load().__file__).resolve();original=Path.read_bytes
        monkeypatch.setattr(Path,'read_bytes',lambda path:b'changed image' if path.resolve()==image else original(path))
    with pytest.raises(PlannerSourceChanged):captured.verify()


def test_retained_annotation_equal_copy_and_type_parameters_are_terminal(monkeypatch,request):
    from anygeometry import _planning_code
    function=_planning_code.passive_data
    captured=capture_with_runtime_boundary((function,),request=request)
    if captured is None:return # Exact initial refusal is not postcapture coverage.
    monkeypatch.setattr(function,'__annotations__',dict(function.__annotations__))
    with pytest.raises(PlannerSourceChanged):captured.verify()
    with pytest.raises(PlannerSourceChanged):captured.verify(full=True)

@pytest.mark.parametrize('phase',['initial','retained'])
@numpy_profile_test
@windows_owned_test
def test_infrastructure_generated_dataclass_code_skew_refuses(monkeypatch,tmp_path,phase):
    from anygeometry._planning_code import InfrastructurePin
    from anygeometry._planning_memory import MemoryBudget
    from anygeometry._planning_qualification import QualificationDecline
    captured=InfrastructurePin('host') if phase=='retained' else None
    function=MemoryBudget.__init__
    monkeypatch.setattr(function,'__code__',function.__code__.replace(co_consts=(*function.__code__.co_consts,'forged')))
    if captured is None:
        with pytest.raises(QualificationDecline):InfrastructurePin('host')
    else:
        with pytest.raises(PlannerSourceChanged):captured.verify()

def _worker_cache_alias_body(kind):
    import numpy as np
    from types import SimpleNamespace
    from anygeometry._planning_ledger import PackageWorkerLedger
    from anygeometry._planning_qualification import QualificationDecline
    protected=() if kind=='empty_tuple' else np.zeros(2) if kind=='array' else {}
    cache={'entry':(protected,) if kind=='wrapped_dict' else protected}
    ledger=PackageWorkerLedger.__new__(PackageWorkerLedger)
    ledger.role=None;ledger.resource_pin=None;ledger._resource_binding=None;ledger._resource_token="null"
    ledger._contexts=();ledger._references=(protected,)
    ledger.store_bindings={'fixture':(None,'fixture',cache,dict)}
    ledger.resets={'fixture':cache}
    ledger.manifest=SimpleNamespace(callables=SimpleNamespace(_identities=(('protected',id(protected)),)),roles={})
    if kind=='empty_tuple':ledger._check_live_state()
    else:
        with pytest.raises(QualificationDecline,match='aliases protected definition'):ledger._check_live_state()


_WORKER_CACHE_CHILD=r'''
import json,sys
from pathlib import Path
sys.path[:0]=[sys.argv[2],sys.argv[3]]
import anygeometry,numpy
import test_planning_r8_integration as contract
from anygeometry import cancellation,_planning_ledger
from anygeometry._planning_manifest import cancellation_registry
from anygeometry._planning_qualification import QualificationDecline
source=Path(sys.argv[2]).resolve()
assert Path(anygeometry.__file__).resolve().is_relative_to(source)
assert Path(_planning_ledger.__file__).resolve().is_relative_to(source)
assert Path(contract.__file__).resolve()==Path(sys.argv[3]).resolve()/'test_planning_r8_integration.py'
owner,lock,registry,rows=cancellation_registry(worker=True)
assert not rows
contract._worker_cache_alias_body('empty_tuple')
report=dict(scope='clean-child-worker-cache-alias-unit',scenario=sys.argv[1],
    executable=sys.executable,version=sys.version,numpy=numpy.__version__,
    anygeometry_origin=anygeometry.__file__,ledger_origin=_planning_ledger.__file__,
    contract_origin=contract.__file__,worker_positive_baseline_passed=True)
if sys.argv[1]=='signal':
    signal=cancellation.CancellationSignal()
    issued=cancellation_registry()[3]
    assert len(issued)==1 and issued[0][0] is signal
    try:contract._worker_cache_alias_body('empty_tuple')
    except QualificationDecline as error:
        assert str(error)=='worker cancellation authority is not empty'
        report['reason']=str(error)
    else:raise AssertionError('worker accepted live cancellation authority')
elif sys.argv[1]=='production_context':
    with contract.pytest.MonkeyPatch.context() as patch:
        contract._production_ledger_context_body(patch)
    assert cancellation_registry(worker=True)[3]==()
else:
    contract._worker_cache_alias_body(sys.argv[1])
    assert cancellation_registry(worker=True)[3]==()
assert cancellation._EVENTS is registry and cancellation._LOCK is lock
report['original_contract_body_executed']=True
print(json.dumps(report))
'''


def _run_worker_cache_child(tmp_path,request,scenario):
    """Worker-only unit preconditions in a fresh child; never reset host authority."""
    import json
    from pathlib import Path
    import subprocess
    from anygeometry import cancellation
    from anygeometry._planning_qualification import QualificationDecline
    root=Path(__file__).resolve().parents[1]
    # A genuine live host token makes isolation discriminating even in a small run.
    signal=cancellation.CancellationSignal()
    owner,lock,registry,rows=manifest.cancellation_registry()
    event=next(event for token,event in rows if token is signal)
    with pytest.raises(QualificationDecline,match='^worker cancellation authority is not empty$'):
        manifest.cancellation_registry(worker=True)
    command=[sys.executable,'-B','-c',_WORKER_CACHE_CHILD,scenario,str(root/'src'),str(root/'tests')]
    try:
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=15)
    except subprocess.TimeoutExpired as error:
        for name,value in (('stdout',error.stdout),('stderr',error.stderr)):
            text=value.decode('utf-8','replace') if isinstance(value,bytes) else value or ''
            (tmp_path/(name+'.txt')).write_text(text,encoding='utf-8')
        raise
    finally:
        current_owner,current_lock,current_registry,current_rows=manifest.cancellation_registry()
        assert current_owner is owner and current_lock is lock and current_registry is registry
        assert next(value for token,value in current_rows if token is signal) is event
        assert current_rows==rows
    (tmp_path/'stdout.txt').write_text(result.stdout,encoding='utf-8')
    (tmp_path/'stderr.txt').write_text(result.stderr,encoding='utf-8')
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(result.stdout)
    assert report['scenario']==scenario and report['worker_positive_baseline_passed'] is True
    assert report['original_contract_body_executed'] is True
    (tmp_path/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    request.node.user_properties.extend((('execution_scope',report['scope']),
        ('worker_positive_baseline_passed',True),('host_cancellation_authority_preserved',True),
        ('original_test_body_executed',True),
        ('parallel_qualified',False),('child_numpy',report['numpy']),
        ('child_anygeometry',report['anygeometry_origin'])))
    return report


@pytest.mark.parametrize('kind',['empty_tuple','dict','array','wrapped_dict'])
def test_worker_cache_alias_retains_mutable_protection(kind,tmp_path,request):
    _run_worker_cache_child(tmp_path,request,kind)


def test_worker_cache_child_refuses_actual_cancellation_authority(tmp_path,request):
    report=_run_worker_cache_child(tmp_path,request,'signal')
    assert report['reason']=='worker cancellation authority is not empty'


def test_released_source_trees_keep_compiled_identity_and_refuse_drift_before_parse(monkeypatch):
    import ast
    from pathlib import Path
    from anygeometry import _planning_code
    from anygeometry._planning_qualification import QualificationDecline
    evidence=manifest._ImmutableEvidence();filename=_planning_code.__file__
    first=evidence.source(_planning_code,filename);codes=first[3]
    ids=tuple(map(id,codes.values()))
    released=evidence.release_parsing()
    assert released['source_trees']==1 and released['ast_node_visits']>0
    second=evidence.source(_planning_code,filename)
    assert second[3] is codes and tuple(map(id,codes.values()))==ids
    evidence.release_parsing();evidence.begin()
    original=Path.read_bytes;target=Path(filename).resolve()
    monkeypatch.setattr(Path,'read_bytes',lambda path:original(path)+b'\n' if path.resolve()==target else original(path))
    monkeypatch.setattr(ast,'parse',lambda *a,**k:pytest.fail('changed source reparsed'))
    with pytest.raises(QualificationDecline,match='immutable source evidence changed'):
        evidence.source(_planning_code,filename)
    assert tuple(map(id,codes.values()))==ids


def test_capture_instruction_cache_uses_exact_code_and_releases_epoch():
    from anygeometry._planning_code import exact_numerical_record
    evidence=manifest._ImmutableEvidence(cache_traversal=True)
    code=exact_numerical_record.__code__
    first=evidence.instructions(code)
    assert first==tuple(manifest._instructions(code)) and evidence.instructions(code) is first
    changed=code.replace()
    assert evidence.instructions(changed) is not first
    evidence.release_parsing()
    assert not evidence.instruction_cache and not evidence.path_cache
    assert evidence.instructions(code)==first and evidence.instructions(code) is not first


def test_capture_path_cache_rechecks_resolution_before_acceptance(tmp_path,monkeypatch):
    from pathlib import Path
    from anygeometry._planning_qualification import QualificationDecline
    first=tmp_path/'first.py';second=tmp_path/'second.py'
    first.write_text('same');second.write_text('same')
    key=str(first);original=Path.resolve;target=[first];calls=[]
    def resolve(path,*args,**kwargs):
        if str(path)==key:calls.append(1);return original(target[0])
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'resolve',resolve)
    evidence=manifest._ImmutableEvidence(cache_traversal=True)
    assert evidence.resolve(key)==first and evidence.resolve(key)==first and len(calls)==1
    target[0]=second
    with pytest.raises(QualificationDecline,match='resolution changed'):evidence.verify_paths()
    evidence.release_parsing();evidence.begin()
    assert evidence.resolve(key)==second


@numpy_profile_test
def test_cached_and_uncached_complete_role_graphs_are_identical():
    from anygeometry._planning_canaries import _fixture
    from anygeometry._planning_executor import freeze_model
    model,_=_fixture();freeze_model(model)
    uncached=manifest.LoadedManifest(model=model,include_host=False,_cache_traversal=False)
    cached=manifest.LoadedManifest(model=model,include_host=False,_cache_traversal=True)
    assert cached.nodes==uncached.nodes and cached.wire==uncached.wire
    assert cached.callables._reference_tokens==uncached.callables._reference_tokens
    assert cached.roles['constructor']._reference_tokens==uncached.roles['constructor']._reference_tokens
    assert not cached.callables._immutable_evidence.path_cache
    assert not cached.callables._immutable_evidence.instruction_cache
    cached.verify();cached.verify(full=True);uncached.verify()


@pytest.mark.parametrize('full',[False,True])
def test_capture_path_retarget_after_capture_is_terminal(tmp_path,monkeypatch,full,request):
    from pathlib import Path
    from anygeometry import _planning_code
    captured=capture_with_runtime_boundary((_planning_code.passive_data,),request=request)
    if captured is None:return # Exact initial refusal is not postcapture coverage.
    origin=Path(_planning_code.__file__);other=tmp_path/origin.name
    other.write_bytes(origin.read_bytes())
    resolve=Path.resolve
    def retarget(path,*args,**kwargs):
        return resolve(other) if path==origin else resolve(path,*args,**kwargs)
    monkeypatch.setattr(Path,'resolve',retarget)
    with pytest.raises(PlannerSourceChanged):captured.verify(full=full)


def test_instruction_memo_has_one_assembly_budget_and_releases_it():
    # The per-request memo is a STORAGE-only performance cap (named _INSTRUCTION_CACHE_SLOTS), not the 80k
    # callable/authority or 100k value bound. Past it instructions are still returned in full, just not retained.
    cap=manifest._INSTRUCTION_CACHE_SLOTS
    def total(evidence):return sum(len(ops) for code,ops in evidence.instruction_cache.values())
    evidence=manifest._ImmutableEvidence(cache_traversal=True)
    mid=compile('\n'.join('value='+str(i) for i in range(3000)),'<mid-evidence>','exec')
    assert 4096<len(tuple(manifest._instructions(mid)))<cap          # the former cap would have refused this one
    assert evidence.instructions(mid)==tuple(manifest._instructions(mid))
    assert any(code is mid for code,ops in evidence.instruction_cache.values())
    # genuinely over-capacity code object: its ACTUAL instruction count exceeds the named cap
    large=compile('\n'.join('value='+str(i) for i in range(cap//2+1000)),'<large-evidence>','exec')
    expected=tuple(manifest._instructions(large))
    assert len(expected)>cap
    assert evidence.instructions(large)==expected                     # complete result, nothing omitted
    assert total(evidence)<=cap
    assert not any(code is large for code,ops in evidence.instruction_cache.values())   # oversized object not retained
    assert evidence.instruction_slots==total(evidence)
    sizes=[]
    for i in range(1200):                                             # many small bodies exhaust the combined budget
        code=compile('\n'.join('value='+str(j+i) for j in range(40)),'<small-evidence>','exec')
        ops=tuple(manifest._instructions(code));sizes.append(len(ops))
        assert evidence.instructions(code)==ops
        assert total(evidence)<=cap
    assert sum(sizes)+len(tuple(manifest._instructions(mid)))>cap     # the budget really was exhausted
    assert evidence.instruction_slots==total(evidence)<=cap and cap-evidence.instruction_slots<max(sizes)+1
    assert len(evidence.instruction_cache)<1200+1                      # some small bodies were not retained
    evidence.release_parsing()
    assert evidence.instruction_slots==0 and not evidence.instruction_cache
