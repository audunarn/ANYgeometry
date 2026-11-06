"""Public opt-in controller. CI oracles remain exclusive to StageControl."""
import contextvars
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
from collections import deque

from ._planning_memory import MemoryBudget,observe_memory
from ._planning_qualification import QualificationDecline
from ._planning_executor import (freeze_model,source_files,verify_sources,
    PlannerCancelled,PlannerDeadline,PlannerSourceChanged)
from ._planning_host import _SessionBridge,_CallbackFailed
from .runtime_diagnostics import _planning_transaction,_publish_planning,_record


_INITIAL_DECLINE=contextvars.ContextVar('anygeometry_planning_initial_decline',default=None)


@contextmanager
def _initial_decline(reason):
    """Private caller preflight refusal; no policy, environment or callback mutation."""
    if type(reason) is not str or not 0<len(reason)<=1024:
        raise QualificationDecline('initial decline requires bounded immutable reason')
    token=_INITIAL_DECLINE.set(reason)
    try:yield
    finally:_INITIAL_DECLINE.reset(token)


_FORCE_ROUTE=contextvars.ContextVar('anygeometry_planning_forced_route',default=None)


@contextmanager
def _forced_route(reason):
    """PRIVATE diagnostic selector (R11 profiling): skips only the cost gate.

    Every other guard (unsupported-policy exit, storage, memory lease, H/P/W
    manifests, 100 s allowance, retirement) is unchanged. The receipt records
    ``forced_selector``; a forced run never satisfies default-selector acceptance.
    """
    if type(reason) is not str or not 0<len(reason)<=1024:
        raise QualificationDecline('forced route requires bounded immutable reason')
    token=_FORCE_ROUTE.set(reason)
    try:yield
    finally:_FORCE_ROUTE.reset(token)


def memory_profile():
    observed=observe_memory()
    return MemoryBudget(observed['host_peak_bytes']+512*2**20,4*2**30,8*2**30,(1,2))


class _ProductionControl:
    """Fixed production resources; no diagnostic CI oracle or private selector."""
    production=True
    workers=8
    startup_concurrency=8
    timeout=100
    reuse_workers=True
    def __init__(self,directory,lease,manifest,infrastructure):
        self.directory=directory;self.lease=lease
        self.reports=[]
        self.infrastructure=infrastructure
        self.limits=dict(payload_limit=4_000_000,walk_limit=100000)
        from ._planning_values import value_codec_contract
        self.numerical_manifest=dict(value_codec=value_codec_contract(),common=manifest.callables.digest,
            constructor=manifest.roles['constructor'].digest,host=manifest.roles['host'].digest,
            infrastructure_host=infrastructure.digest,resource=infrastructure.native)
    def tree_options(self):return dict(tree_memory_limit=self.lease.tree_limit_bytes)


def _prune_success(root):
    candidates=[];now=time.time()
    for folder in root.iterdir():
        if folder.is_symlink() or not folder.is_dir() or len(folder.name)!=32 or any(c not in '0123456789abcdef' for c in folder.name):continue
        if folder.resolve().parent!=root.resolve():continue
        path=folder/'receipt.json'
        if not path.is_file() or path.is_symlink() or set(p.name for p in folder.iterdir())!={'receipt.json'}:continue
        try:record=json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError):continue
        if record.get('schema')=='R5-production-v1' and record.get('disposition')=='parallel' and record.get('retired') is True:
            candidates.append((path.stat().st_mtime,folder,path))
    candidates.sort(key=lambda item:item[0],reverse=True)
    for index,(stamp,folder,path) in enumerate(candidates):
        if index>=1000 or now-stamp>30*86400:
            path.unlink();folder.rmdir()


def _storage():
    try:
        override=os.environ.get('ANYGEOMETRY_PLANNING_SCRATCH_ROOT')
        root=Path(override) if override else Path(os.environ['LOCALAPPDATA'])/'ANYgeometry'/'planning'
        root.mkdir(parents=True,exist_ok=True)
        if not override:_prune_success(root)
        # Failed/active evidence is never a pruning target.
        if sum(p.stat().st_size for p in root.rglob('*') if p.is_file())>=4*2**30:
            raise QualificationDecline('planning scratch highwater')
        if shutil.disk_usage(root).free<512*2**20:raise QualificationDecline('planning scratch space')
        folder=root/uuid.uuid4().hex;folder.mkdir()
        work=folder/'work';work.mkdir()
        return folder,work
    except (OSError,KeyError) as error:
        raise QualificationDecline('planning storage unavailable: '+type(error).__name__) from error


_MEMORY_POLL_INTERVAL=2048
_MEMORY_SECONDS=.025
# Provisional routing threshold in estimated face count (parent calibrates with
# subset timings). 97 and 240 faces stay serial; 561 faces may parallelize.
_ROUTING_THRESHOLD=400


def _estimate_cost(model,operands=None):
    """Passive structure estimate: (units, reason). units None => serial.

    Reads only the exact private dict of the exact model type; no property,
    kernel, thaw or mapping subclass is invoked. With ``operands`` (the public
    caller's normalized handles) the requested face scope is counted rather
    than the whole container model; non-face or foreign handles are unknown.
    """
    try:
        from .model import GeometryModel
        if type(model) is not GeometryModel:return None,'unknown model type'
        state=object.__getattribute__(model,'__dict__')
        faces=state.get('_faces') if type(state) is dict else None
        if type(faces) is not dict:return None,'unsafe face storage'
        if operands is not None:
            from .identity import EntityHandle
            if type(operands) is not tuple:return None,'unsafe operand scope'
            seen=set()
            # Property-free: the private field of the exact instance dict, never model.model_id.
            model_id=state.get('_model_id')
            if model_id is None:return None,'unknown model identity'
            for handle in operands:
                if (type(handle) is not EntityHandle or handle.kind!='face' or
                        handle.model_id!=model_id or type(handle.id) is not int):
                    return None,'unknown operand scope'
                seen.add(handle.id)
            return len(seen),'requested faces=%d of %d'%(len(seen),len(faces))
        return len(faces),'estimated faces=%d'%len(faces)
    except Exception as error:
        return None,'estimate failed: '+type(error).__name__


class _ProductionAllowanceExpired(BaseException):
    """Only the internal production attempt expired; caller limits still apply."""


class _ProductionBridge(_SessionBridge):
    def check_clock(self):
        if time.monotonic()>=self.deadline:
            raise _ProductionAllowanceExpired('internal production allowance consumed')


def run(model,serial,parallel,*,policy,mode,eligible=True,preparation=None,serial_replay=None,route_operands=None):
    # Unsupported routes never enter the parallel transaction or adapt caller
    # callbacks. Preserve the original serial error and work-budget contracts.
    if not eligible or mode!='exact' or policy.max_predicates is not None or policy.max_candidate_pairs is not None:
        return serial()
    forced=_FORCE_ROUTE.get()
    if forced is not None and (type(forced) is not str or not 0<len(forced)<=1024):
        raise QualificationDecline('forced route reason type')
    units,route_reason=(_ROUTING_THRESHOLD,'forced') if forced is not None else (
        _estimate_cost(model) if route_operands is None else _estimate_cost(model,route_operands))
    if type(units) is not int or units<_ROUTING_THRESHOLD:
        # Cheap automatic routing: before storage/freeze/manifest/child setup.
        from .runtime_diagnostics import _record as record_route
        record_route('parallel_planning',dict(schema='R5-production-v1',disposition='serial',
            ci_verified=False,stages=[],costs={},retired=True,
            reason='routed serial: '+route_reason+' threshold=%d'%_ROUTING_THRESHOLD))
        return serial()
    from ._planning_manifest import capture_loaded_manifest,prepare_interfaces
    from .serialization import to_dict
    from ._planning_model_state import production_snapshot
    started=time.perf_counter();bridge=None;lease=None;manifest=None;source=None;packet=None;parallel_active=True;infrastructure=None
    decline_variable=_INITIAL_DECLINE;decline_state=None
    folder=None;work=None;receipt=dict(schema='R5-production-v1',ci_verified=False,workers=8,
        startup_concurrency=8,disposition='terminal',stages=[],costs={})
    if forced is not None:receipt['forced_selector']=forced
    def model_bytes():return json.dumps(to_dict(model),sort_keys=True,separators=(',',':'),allow_nan=False)
    from . import runtime_diagnostics as diagnostics
    from .cancellation import is_cancellation_signal
    collectors=(diagnostics._COLLECTORS.get(),diagnostics._RECORD_COLLECTORS.get())
    verification_times=deque(maxlen=64);verification_components=deque(maxlen=64);callback_calls=0
    verification_totals={}
    def measure(name,value):
        row=verification_totals.setdefault(name,dict(count=0,sum_seconds=0.,min_seconds=value,max_seconds=value))
        row['count']+=1;row['sum_seconds']+=value
        row['min_seconds']=min(row['min_seconds'],value);row['max_seconds']=max(row['max_seconds'],value)
    @contextmanager
    def phase(name):
        stamp=time.perf_counter()
        try:yield
        finally:receipt['costs'].setdefault('host_phases',{})[name]=time.perf_counter()-stamp
    @contextmanager
    def transaction():
        nonlocal collectors
        previous=collectors
        with _planning_transaction() as data:
            collectors=(diagnostics._COLLECTORS.get(),diagnostics._RECORD_COLLECTORS.get())
            try:yield data
            finally:
                check_collectors()
                collectors=previous
    def check_collectors():
        if diagnostics._COLLECTORS.get() is not collectors[0] or diagnostics._RECORD_COLLECTORS.get() is not collectors[1]:
            raise PlannerSourceChanged('host diagnostic collector scope changed')
    memory_polls=[0];memory_stamp=[time.monotonic()];child_memory_active=True
    def poll(*,full=False,wait=False):
        nonlocal callback_calls
        explicit_boundary=full
        check_collectors()
        if source is not None:
            if _INITIAL_DECLINE is not decline_variable:
                raise PlannerSourceChanged('initial decline context identity changed')
            if (decline_variable in contextvars.copy_context(),decline_variable.get())!=decline_state:
                raise PlannerSourceChanged('initial decline context changed after capture')
        if policy.cancellation_check is not None:
            callback_calls+=1
            try:cancelled=bool(policy.cancellation_check())
            except BaseException as error:raise _CallbackFailed(error) from error
            if cancelled:raise PlannerCancelled('intersection planning cancelled')
            # Callback presence is cancellation only; it never promotes verification.
        check_collectors()
        if source is not None and full:
            stamp=time.perf_counter();components={}
            verify_sources(source)
            components['source_seconds']=time.perf_counter()-stamp;step=time.perf_counter()
            infrastructure.verify()
            components['infrastructure_seconds']=time.perf_counter()-step;step=time.perf_counter()
            manifest.verify()
            components['manifest_seconds']=time.perf_counter()-step;step=time.perf_counter()
            try:current=production_snapshot(model)
            except QualificationDecline as error:raise PlannerSourceChanged('host declarative state changed') from error
            components['constructor_seconds']=time.perf_counter()-step;step=time.perf_counter()
            if current!=packet:raise PlannerSourceChanged('host model changed')
            components['document_seconds']=time.perf_counter()-step
            verification_times.append(time.perf_counter()-stamp)
            verification_components.append(components)
            measure('total',verification_times[-1])
            for name,value in components.items():measure(name,value)
            if (work is not None and os.environ.get('ANYGEOMETRY_PLANNING_SCRATCH_ROOT') and
                    (explicit_boundary or verification_totals['total']['count']%16==0)):
                path=work/'boundary-costs.json';temporary=path.with_suffix('.tmp')
                temporary.write_text(json.dumps(dict(callback_calls=callback_calls,totals=verification_totals,
                    recent_limit=64,recent_seconds=list(verification_times),recent_components=list(verification_components),
                    scope='last completed checkpoint; terminal receipt has complete totals')),encoding='utf-8')
                temporary.replace(path)
        if lease is not None and child_memory_active:
            memory_polls[0]+=1;now=time.monotonic()
            if explicit_boundary or wait or memory_polls[0]>=_MEMORY_POLL_INTERVAL or now-memory_stamp[0]>=_MEMORY_SECONDS:
                memory_polls[0]=0;memory_stamp[0]=now
                memory_started=time.perf_counter()
                try:lease.check()
                finally:measure('memory_check_seconds',time.perf_counter()-memory_started)
        if bridge is not None and parallel_active:bridge.check_clock()
    poll.boundary=lambda:poll(full=True)
    # File/pipe wait iterations: memory check every iteration, no full verification.
    poll.wait=lambda:poll(wait=True)
    failure=None;published=None;result=None
    try:
        if type(decline_variable) is not contextvars.ContextVar:
            raise QualificationDecline('initial decline context type')
        decline_state=(decline_variable in contextvars.copy_context(),decline_variable.get())
        if decline_state[1] is not None:
            if type(decline_state[1]) is not str or not 0<len(decline_state[1])<=1024:
                raise QualificationDecline('initial decline context reason type')
            raise QualificationDecline(decline_state[1])
        if not eligible or mode!='exact' or policy.max_predicates is not None or policy.max_candidate_pairs is not None:
            raise QualificationDecline('unsupported production policy')
        poll()
        with phase('initial_snapshot'):
            folder,work=_storage()
            lease=memory_profile().begin()
            packet=freeze_model(model);before=model_bytes()
        capture_started=time.perf_counter()
        from ._planning_code import InfrastructurePin
        prepare_interfaces()
        infrastructure=InfrastructurePin('host')
        manifest=capture_loaded_manifest(model=model,cancellation=policy.cancellation_check if is_cancellation_signal(policy.cancellation_check) else None)
        lease.observe_phase('capture')
        receipt['costs']['capture_seconds']=time.perf_counter()-capture_started
        source=source_files()
        receipt.update(manifest_sha256=manifest.wire['sha256'],model_sha256=packet['sha256'])
        if os.environ.get('ANYGEOMETRY_PLANNING_SCRATCH_ROOT'):
            (work/'manifest-tokens.json').write_text(json.dumps(dict(common=manifest.callables._reference_tokens,
                constructor=manifest.roles['constructor']._reference_tokens)),encoding='utf-8')
        receipt['infrastructure']=dict(host=infrastructure.digest,host_nodes=infrastructure.nodes)
        receipt['costs']['released_parser_scaffolding']=dict(numerical=manifest.parsing_release,infrastructure=infrastructure.parsing_release)
        control=_ProductionControl(work,lease,manifest,infrastructure)
        bridge=_ProductionBridge(packet,control,poll=poll)
        with transaction() as provisional:
            try:
                bridge._inside_numerical_call=True
                with phase('orchestration'):result=parallel(bridge)
                bridge._inside_numerical_call=False
                with phase('finish'):bridge.finish()
            except BaseException:
                # Retirement is still mandatory, but cannot replace the original
                # callback/numerical failure. close retains its poisoned owner.
                try:
                    with phase('retirement'):bridge.close()
                except BaseException:pass
                raise
            else:
                with phase('retirement'):bridge.close()
                if bridge.retired:child_memory_active=False  # verified: no child tree to watch
        with transaction() as verification:
            with phase('samples_and_second_decode'):bridge.verify_pending()
        with phase('final_boundary'):
            poll(full=True);bridge.check_clock()
        for kind,tasks,serial_task,report in bridge.pending:
            from ._planning_code import stage_cost_vector
            vector={name:value for name,value in stage_cost_vector(kind,tasks).items() if name!='rows'}
            if [attempt['cost_vector'] for attempt in bridge.attempts if attempt['stage']==kind]!=[vector]:
                raise PlannerSourceChanged('stage cost vector changed after dispatch')
            receipt['stages'].append(dict(stage=kind,accepted=len(tasks),total=len(tasks),
                accepted_weight=sum(float(t.get('cost',1)) for t in tasks),
                total_weight=sum(float(t.get('cost',1)) for t in tasks),
                sampling=report.get('sampling'),startup=report.get('startup'),
                infrastructure=report.get('infrastructure'),
                request_sha256=report['request_sha256'],cost_vector=vector,transport_bytes=report.get('transport_bytes')))
        receipt['disposition']='parallel';receipt['plan_checksum']=result.content_checksum;published=provisional
        receipt['costs']['verification']=verification
        receipt['costs']['infrastructure']=[dict(stage=kind,host=report.get('host_timing'),
            workers=[{key:worker.get(key) for key in ('startup_seconds','assignment_seconds','task_seconds','task_timing','boundary_seconds','resource_guard','value_codec')}
                for worker in report.get('worker_reports',[])]) for kind,tasks,serial_task,report in bridge.pending]
    except (QualificationDecline,_ProductionAllowanceExpired) as error:
        failure=error;receipt['reason']=str(error);receipt['disposition']='serial'
        receipt['costs']['abandoned']=locals().get('provisional',{})
        if bridge is not None:bridge.close()  # sticky retirement errors propagate; no replay
        parallel_active=False;child_memory_active=False  # verified retirement: no child tree remains
        poll(full=True)
        with transaction() as replay:
            result=serial() if serial_replay is None else serial_replay(poll)
        poll(full=True);published=replay
    except BaseException as error:
        failure=error
        original=error.error if type(error) is _CallbackFailed else error
        receipt['reason']=type(original).__name__+': '+str(original)
        raise
    finally:
        primary=sys.exc_info()[1];final_error=None
        def terminal_error(error,field=None):
            nonlocal final_error
            if final_error is None:final_error=error
            original=primary if primary is not None else final_error
            original=original.error if type(original) is _CallbackFailed else original
            receipt['disposition']='terminal'
            receipt['stages']=[]
            receipt['reason']=type(original).__name__+': '+str(original)
            if field is not None:receipt[field]=type(error).__name__+': '+str(error)
        def refresh():
            receipt['elapsed_seconds']=time.perf_counter()-started
            receipt['costs'].update(manifest_recheck_seconds=list(verification_times),
                boundary_components=list(verification_components),boundary_recent_limit=64,
                boundary_totals=verification_totals,callback_calls=callback_calls)
            if lease is not None:receipt['memory']=lease.report()
            receipt['retired']=bridge is None or bridge.retired
            receipt['attempted_stages']=[] if bridge is None else list(bridge.attempts)
        def persist():
            path=folder/'receipt.json';receipt['receipt_path']=str(path)
            try:path.write_text(json.dumps(receipt,sort_keys=True,allow_nan=False),encoding='utf-8')
            except OSError as error:
                receipt['receipt_error']=str(error)
                return False
            return True
        if primary is not None:terminal_error(primary)
        try:
            if bridge is not None:
                cleanup_error=bridge._retirement_error
                if cleanup_error is None:
                    try:bridge.close()
                    except BaseException as error:cleanup_error=error
                if cleanup_error is not None:terminal_error(cleanup_error,'cleanup_error')
            if folder is not None:
                receipt['retirement']={}
                for name in ('cleanup.json','workers-cleanup.json'):
                    path=folder/'work'/'session'/name
                    if path.is_file():receipt['retirement'][name]=json.loads(path.read_text(encoding='utf-8'))
            refresh()
            written=folder is not None and persist()
            if primary is None and final_error is None and receipt['disposition']=='parallel':
                if written and receipt['retired']:
                    try:shutil.rmtree(folder/'work')
                    except BaseException as error:
                        terminal_error(error,'scratch_cleanup_error')
                        refresh();persist()
                elif folder is not None:
                    # Retain scratch on every failed write/replay. The parallel
                    # deadline is no longer active; caller replay budget remains.
                    parallel_active=False
                    if receipt['retired']:child_memory_active=False
                    try:
                        poll(full=True)
                        with transaction() as replay:
                            result=serial() if serial_replay is None else serial_replay(poll)
                        poll(full=True);published=replay
                        receipt['disposition']='serial-receipt-failure'
                        receipt['stages']=[]
                    except BaseException as error:terminal_error(error,'replay_error')
                    refresh();persist()
        except BaseException as error:
            terminal_error(error,'finalization_error')
            if folder is not None:
                try:refresh();persist()
                except BaseException as evidence_error:
                    receipt['receipt_error']=type(evidence_error).__name__+': '+str(evidence_error)
        finally:
            # Even a terminal receipt I/O failure must reach caller diagnostics.
            try:refresh()
            except BaseException as error:terminal_error(error,'accounting_error')
            _record('parallel_planning',receipt)
        if primary is None and final_error is not None:raise final_error
    if published is not None:_publish_planning(published)
    return result
