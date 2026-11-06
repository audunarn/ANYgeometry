"""PP5 host stage bridge; private controls, mandatory real plan/apply CI.

Without explicit controls the opt-in request declines to the original serial
caller. PP4 can supply admission here; PP3 owns future budgets/callback adapters.
"""
import json
import math
import time
import types
from pathlib import Path
import psutil
from ._planning_executor import (OwnedPlanner,PlannerDeadline,PlannerSourceChanged,
    PlannerCancelled,PlannerCleanupError,PlannerMemoryExceeded,freeze_model,pack,verify_sources)
from ._planning_qualification import QualificationDecline,check_ci_equality
from ._planning_admission import stratified_samples
from ._planning_values import encode_value,decode_value,comparison_packet,decode_stage_values
from .serialization import to_dict
from .runtime_diagnostics import _increment
from .cancellation import is_cancellation_signal
from .errors import GeometryError


def _overall_deadline(value):
    if value is not None and (type(value) not in (int,float) or
            type(value) is float and not math.isfinite(value)):
        raise GeometryError('overall deadline must be a finite absolute monotonic time or None')
    return value


class StageControl:
    def __init__(self,directory,*,expected_ci,observed_ci,workers=8,timeout=100,admission=None,payload_limit=4_000_000,walk_limit=100000,reuse_workers=False,tree_memory_limit=None,memory_budget=None,startup_concurrency=1,overall_deadline=None):
        self.directory=Path(directory);self.expected_ci=expected_ci;self.observed_ci=observed_ci
        self.workers=workers;self.timeout=timeout;self.admission=admission
        self.overall_deadline=_overall_deadline(overall_deadline)
        self.reuse_workers=reuse_workers
        from ._planning_session import _startup_concurrency
        self.startup_concurrency=_startup_concurrency(startup_concurrency,workers)
        self.tree_memory_limit=tree_memory_limit
        self.memory_budget=memory_budget;self._memory_lease=None;self.memory_report=None
        self.limits=dict(payload_limit=payload_limit,walk_limit=walk_limit)
        self.reports=[];self.mode='not-run';self.decline_reason=None;self.ci_verified=False
        self.package_stages=False

    def tree_options(self):
        if self._memory_lease is not None:
            return dict(tree_memory_limit=self._memory_lease.tree_limit_bytes)
        if self.tree_memory_limit is None:return {}
        from ._planning_process import validate_memory_limit
        validate_memory_limit(self.tree_memory_limit)
        return dict(tree_memory_limit=self.tree_memory_limit)


class _CallbackFailed(BaseException):
    """Keep caller exceptions outside infrastructure-refusal handlers until cleanup."""
    def __init__(self,error):
        self.error=error


def _package_values(control):
    return getattr(control,'production',False) is True or getattr(control,'package_stages',False) is True


def _stage_descriptor(preparation,model,policy):
    """Recognize the existing public entry's closed descriptor without calling it."""
    from . import batch_intersections as batch
    from .identity import EntityHandle
    from .structural import ConnectionIntent
    from ._planning_manifest import _ImmutableEvidence,_MODEL_CALLER_SOURCE_PINS
    from ._planning_session import decode_preparation
    if type(batch) is not types.ModuleType:
        raise QualificationDecline('public stage descriptor module changed')
    entry=vars(batch).get('plan_intersections')
    if type(entry) is not types.FunctionType or entry.__globals__ is not vars(batch):
        raise QualificationDecline('public stage descriptor owner changed')
    if (type(preparation) is not types.FunctionType or preparation.__globals__ is not vars(batch) or
            preparation.__defaults__ is not None or preparation.__kwdefaults__ is not None or
            not any(preparation.__code__ is code for code in entry.__code__.co_consts) or
            preparation.__code__.co_freevars!=('effective','handles','seam')):
        raise QualificationDecline('unregistered public stage descriptor')
    evidence=_ImmutableEvidence()
    try:
        digest,_,_,codes=evidence.source(batch,entry.__code__.co_filename)
        if (digest!=_MODEL_CALLER_SOURCE_PINS[batch.__name__] or
                evidence.digest(entry.__code__)!=evidence.digest(codes['plan_intersections'])):
            raise QualificationDecline('public stage descriptor source/code changed')
        values=dict(zip(preparation.__code__.co_freevars,(cell.cell_contents for cell in preparation.__closure__)))
        handles=values['handles']
        if (values['effective'] is not policy or type(policy) is not batch.IntersectionBatchPolicy or
                type(policy.intent) is not ConnectionIntent or type(handles) is not tuple or
                any(type(h) is not EntityHandle or h.model_id!=model.model_id for h in handles)):
            raise QualificationDecline('public stage descriptor closure changed')
        descriptor=dict(operands=[[h.kind,h.id] for h in handles],seam=values['seam'],
            policy={name:(policy.intent.value if name=='intent' else getattr(policy,name)) for name in
                ('intent','face_connections','member_connections','member_face_connections','planning_mode','seam_policy')})
        decode_preparation(descriptor)
        evidence.verify_paths()
        return descriptor
    finally:evidence.release_parsing()


def _admit_stage_package(model,control,policy,packet):
    """Verification API admission; fixed production resource policy is separate."""
    from ._planning_code import InfrastructurePin
    from ._planning_manifest import prepare_interfaces,capture_loaded_manifest
    from ._planning_model_state import production_snapshot
    from ._planning_values import value_codec_contract
    from ._planning_executor import source_files
    try:current=production_snapshot(model,**control.limits)
    except QualificationDecline as error:raise PlannerSourceChanged('stage model changed during admission') from error
    if current!=packet:raise PlannerSourceChanged('stage model changed during admission')
    prepare_interfaces()
    infrastructure=InfrastructurePin('host')
    manifest=capture_loaded_manifest(model=model,cancellation=policy.cancellation_check)
    sources=source_files()
    control.numerical_manifest=dict(value_codec=value_codec_contract(),common=manifest.callables.digest,
        constructor=manifest.roles['constructor'].digest,host=manifest.roles['host'].digest,
        infrastructure_host=infrastructure.digest,resource=infrastructure.native)
    control.package_stages=True
    binding=pack(control.numerical_manifest,'stage-package-admission')
    def verify():
        verify_sources(sources)
        infrastructure.verify();manifest.verify()
        if control.package_stages is not True or pack(control.numerical_manifest,'stage-package-admission')!=binding:
            raise PlannerSourceChanged('stage package capability changed')
        try:current=production_snapshot(model,**control.limits)
        except QualificationDecline as error:raise PlannerSourceChanged('stage model state changed') from error
        if current!=packet:raise PlannerSourceChanged('stage model state changed')
    return verify


class Bridge:
    def __init__(self,packet,control,poll=lambda:None):
        self.packet=packet;self.control=control;self.deadline=time.monotonic()+control.timeout
        self.poll=poll
        self.tree_options=control.tree_options()

    def check_clock(self):
        if time.monotonic()>=self.deadline:raise PlannerDeadline('stage planning deadline consumed')

    def check_deadline(self):
        try:
            self.check_clock()
            self.poll()
            self.check_clock()
        except (PlannerCancelled,PlannerDeadline,PlannerSourceChanged,PlannerCleanupError,PlannerMemoryExceeded) as error:
            # Kernel numerical retry handlers catch GeometryError. Resource
            # failure must cross them intact, then unwrap at qualified_plan.
            if isinstance(error,PlannerMemoryExceeded) or getattr(self,'_inside_numerical_call',False):
                raise _CallbackFailed(error) from error
            raise

    def run(self,kind,tasks,serial_task):
        if not tasks:return ()
        self.check_deadline();planner=None
        try:
            try:
                planner=OwnedPlanner(self.control.directory/kind,workers=self.control.workers,
                    timeout=min(110,self.deadline-time.monotonic()),**self.control.limits,**self.tree_options)
                candidate=planner.submit_stage(self.packet,tasks,poll=self.check_deadline)
            except (OSError,psutil.AccessDenied) as exc:
                raise QualificationDecline('stage infrastructure '+type(exc).__name__+': '+str(exc)) from exc
        finally:
            if planner is not None:
                try:planner.close()
                except Exception as exc:
                    raise PlannerCleanupError('stage planner cleanup failed') from exc
        self.check_deadline()
        return self.verify_candidate(kind,tasks,serial_task,candidate)

    def verify_candidate(self,kind,tasks,serial_task,candidate):
        seed=candidate['sampling_sha256'] if self.control.reuse_workers or _package_values(self.control) else candidate['request_sha256']
        sample=stratified_samples(candidate['task_layout'],seed)
        for index in sample['indices']:
            self.check_deadline()
            # Deferred samples re-enter the original numerical retry handlers
            # after the parallel call has ended. Resource failures must cross
            # those handlers intact, just as during provisional construction.
            had_flag=hasattr(self,'_inside_numerical_call')
            previous=getattr(self,'_inside_numerical_call',False)
            self._inside_numerical_call=True
            try:sample_value=serial_task(tasks[index])
            finally:
                if had_flag:self._inside_numerical_call=previous
                else:del self._inside_numerical_call
            try:reference=json.dumps(encode_value(sample_value,limit=self.control.limits['walk_limit']),sort_keys=True,separators=(',',':'),allow_nan=False)
            finally:del sample_value
            if comparison_packet(json.loads(reference),limit=self.control.limits['walk_limit'])!=comparison_packet(json.loads(candidate['results'][index]),limit=self.control.limits['walk_limit'],codec='v2' if _package_values(self.control) else None,payload_limit=self.control.limits['payload_limit']):
                self._record_mismatch(kind,index,reference,candidate['results'][index])
                raise QualificationDecline('stage sample mismatch '+kind+':'+str(index))
            self.check_deadline()
        candidate.update(stage=kind,sampling=sample)
        self.control.reports.append(candidate)
        if _package_values(self.control):
            return decode_stage_values(candidate['results'],limit=self.control.limits['walk_limit'],payload_limit=self.control.limits['payload_limit'],codec='v2')
        return tuple(decode_value(json.loads(value),limit=self.control.limits['walk_limit']) for value in candidate['results'])

    def _record_mismatch(self,kind,index,reference,candidate):
        self.control.directory.mkdir(parents=True,exist_ok=True)
        (self.control.directory/('sample-mismatch-'+kind+'.json')).write_text(json.dumps(dict(index=index,reference=json.loads(reference),candidate=json.loads(candidate)),indent=2),encoding='utf-8')


class _SessionBridge(Bridge):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.session=None;self.pending=[];self.retired=True;self._retirement_error=None;self.attempts=[]

    def _session_directory(self,kind):return self.control.directory/'session'

    def check_boundary(self):
        try:
            self.check_clock()
            getattr(self.poll,'boundary',self.poll)()
            self.check_clock()
        except (PlannerCancelled,PlannerDeadline,PlannerSourceChanged,PlannerCleanupError,PlannerMemoryExceeded) as error:
            if isinstance(error,PlannerMemoryExceeded) or getattr(self,'_inside_numerical_call',False):
                raise _CallbackFailed(error) from error
            raise

    def run(self,kind,tasks,serial_task):
        if not tasks:return ()
        attempt=None
        if getattr(self.control,'production',False):
            from ._planning_code import stage_cost_vector
            vector=stage_cost_vector(kind,tasks)
            attempt=dict(stage=kind,total=len(tasks),total_weight=sum(float(task.get('cost',1)) for task in tasks),
                received=0,received_weight=0.,scope='validated provisional replies; not published acceptance',
                cost_vector={name:value for name,value in vector.items() if name!='rows'})
            self.attempts.append(attempt)
        self.check_boundary()
        from ._planning_session import OwnedSession
        try:
            if _package_values(self.control):
                from ._planning_values import verify_value_codec_contract
                marker=getattr(self.control,'numerical_manifest',None)
                if type(marker) is not dict or set(marker)!={'common','constructor','host','infrastructure_host','resource','value_codec'}:
                    raise QualificationDecline('package stage admission marker missing')
                verify_value_codec_contract(marker['value_codec'])
            if self.session is None:
                self.retired=False
                self.session=OwnedSession(self._session_directory(kind),workers=self.control.workers,
                    timeout=min(110,self.deadline-time.monotonic()),startup_concurrency=self.control.startup_concurrency,
                    numerical_manifest=getattr(self.control,'numerical_manifest',None),
                    **self.control.limits,**self.tree_options)
            transported=tasks
            if _package_values(self.control):
                from ._planning_values import transport_tasks_v2,verify_transport_tasks_v2
                transported=transport_tasks_v2(tasks,limit=self.control.limits['walk_limit'],payload_limit=self.control.limits['payload_limit'])
                verify_transport_tasks_v2(tasks,transported,limit=self.control.limits['walk_limit'],payload_limit=self.control.limits['payload_limit'])
            candidate=self.session.submit_stage(kind,self.packet,transported,poll=self.check_deadline)
        except (OSError,psutil.AccessDenied) as exc:
            raise QualificationDecline('session infrastructure '+type(exc).__name__+': '+str(exc)) from exc
        if attempt is not None:
            attempt.update(received=len(candidate['results']),received_weight=attempt['total_weight'],
                request_sha256=candidate['request_sha256'],startup=candidate.get('startup'),
                transport_bytes=candidate.get('transport_bytes'))
            self.control.lease.observe_phase(kind+'-returned',self.session.tree.memory_snapshot())
        self.check_boundary()
        if getattr(self.control,'production',False):
            from .runtime_diagnostics import _publish_planning
            diagnostics=[row for worker in candidate['worker_reports'] for row in worker.get('diagnostics',[])]
            for ordinal,data in sorted(diagnostics,key=lambda row:row[0]):_publish_planning(data)
        self.pending.append((kind,tuple(tasks),serial_task,candidate))
        # These values remain inside the tentative plan transaction, not public.
        if _package_values(self.control):
            return decode_stage_values(candidate['results'],limit=self.control.limits['walk_limit'],payload_limit=self.control.limits['payload_limit'],codec='v2')
        return tuple(decode_value(json.loads(value),limit=self.control.limits['walk_limit']) for value in candidate['results'])

    def finish(self):
        self.check_boundary()
        if self.session is not None:self.session.finish(poll=self.check_deadline)
        self.check_boundary()

    def close(self):
        if self._retirement_error is not None:raise self._retirement_error
        if self.session is not None:
            owned=self.session
            try:
                if getattr(self.control,'production',False):
                    self.control.lease.observe_phase('before-retirement',owned.tree.memory_snapshot())
            except BaseException as error:
                # A failed accounting observation cannot prevent containment.
                self._retirement_error=error
                try:owned.close()
                except BaseException as cleanup:
                    self._retirement_error=PlannerCleanupError('session planner cleanup failed')
                    raise self._retirement_error from cleanup
                self.session=None;self.retired=True
                raise error
            try:owned.close()
            except BaseException as exc:
                self._retirement_error=PlannerCleanupError('session planner cleanup failed')
                raise self._retirement_error from exc
            self.session=None;self.retired=True
            if getattr(self.control,'production',False):
                try:self.control.lease.observe_phase('retired',owned.tree._memory_observation)
                except BaseException as error:
                    self._retirement_error=error
                    raise

    def verify_pending(self):
        if self.session is not None:raise PlannerCleanupError('session must retire before verification')
        for kind,tasks,serial_task,candidate in self.pending:
            self.check_boundary()
            self.verify_candidate(kind,tasks,serial_task,candidate)


class _PackageStageBridge(_SessionBridge):
    """Fresh package session per stage, retired before reference sampling."""
    def _session_directory(self,kind):return self.control.directory/kind

    def run(self,kind,tasks,serial_task):
        if not tasks:return ()
        try:
            values=super().run(kind,tasks,serial_task)
            self.finish()
        finally:self.close()
        self.verify_pending()
        self.pending.clear()
        return values


class _TranscriptBridge(Bridge):
    """Replay only after retirement; the caller constructs every task independently."""
    def __init__(self,packet,control,entries,*,sources,poll=lambda:None):
        super().__init__(packet,control,poll)
        if type(entries) is not list or [e.get('kind') for e in entries] not in ([],['pair'],['face'],['pair','face']):
            raise QualificationDecline('transcript stage order')
        self.entries=entries;self.sources=sources;self.position=0

    def run(self,kind,tasks,serial_task):
        if not tasks:return ()
        self.check_deadline();verify_sources(self.sources)
        if self.position>=len(self.entries):raise QualificationDecline('transcript missing stage')
        entry=self.entries[self.position]
        from ._planning_session import validate_entry
        validate_entry(entry,kind,tasks,payload_limit=self.control.limits['payload_limit'])
        def canonical(value):return pack(value,'transcript',limit=self.control.limits['payload_limit'])
        from ._planning_session import sampling_digest
        from ._planning_stages import dispatch
        _,layout=dispatch(tasks,self.control.workers)
        seed=sampling_digest(self.packet,tasks,workers=self.control.workers,batch_size=None,
                             sources=self.sources,payload_limit=self.control.limits['payload_limit'])
        candidate=entry['candidate']
        if (type(candidate) is not dict or canonical(candidate.get('task_layout'))!=canonical(layout) or
                candidate.get('sampling_sha256')!=seed or type(candidate.get('results')) is not list or
                len(candidate['results'])!=len(tasks) or any(type(v) is not str for v in candidate['results'])):
            raise QualificationDecline('transcript layout/seed/result coverage')
        if canonical(candidate.get('preparation_samples'))!=canonical(stratified_samples(layout,seed)):
            raise QualificationDecline('transcript preparation sample coverage')
        # All strings, including unsampled ones, cross the same JSON boundary
        # before reference execution. The unchanged value decoder still checks
        # the closed codec, canonical reconstruction and walk bound below.
        def reject_constant(value):raise ValueError('nonfinite JSON constant')
        try:
            for value in candidate['results']:json.loads(value,parse_constant=reject_constant)
        except (ValueError,RecursionError) as exc:
            raise QualificationDecline('transcript result JSON refused') from exc
        values=self.verify_candidate(kind,tasks,serial_task,candidate)
        self.position+=1
        return values

    def finish(self):
        if self.position!=len(self.entries):raise QualificationDecline('transcript unused stages')

    def _record_mismatch(self,kind,index,reference,candidate):
        try:super()._record_mismatch(kind,index,reference,candidate)
        except OSError as exc:
            raise QualificationDecline('stage sample mismatch '+kind+':'+str(index)+'; diagnostic write failed') from exc


class _PreparedTranscript(BaseException):
    def __init__(self,entries,sources):self.entries=entries;self.sources=sources


class _PreparingBridge(Bridge):
    """Lazy probe: never return provisional values or execute host references."""
    def __init__(self,packet,control,descriptor,*,poll):
        super().__init__(packet,control,poll)
        from ._planning_session import decode_preparation
        decode_preparation(descriptor)
        self.descriptor=descriptor

    def run(self,kind,tasks,serial_task):
        if not tasks:return ()
        self.check_deadline()
        from ._planning_session import OwnedSession
        owned=None
        try:
            try:
                owned=OwnedSession(self.control.directory/'session',workers=self.control.workers,
                    timeout=min(110,self.deadline-time.monotonic()),startup_concurrency=self.control.startup_concurrency,
                    numerical_manifest=getattr(self.control,'numerical_manifest',None),
                    **self.control.limits,**self.tree_options)
                entries=owned.prepare_plan(self.packet,self.descriptor,poll=self.check_deadline)
                sources=owned.sources
            except (OSError,psutil.AccessDenied) as exc:
                raise QualificationDecline('preparation infrastructure '+type(exc).__name__+': '+str(exc)) from exc
        finally:
            if owned is not None:
                try:owned.close()
                except Exception as exc:raise PlannerCleanupError('preparation cleanup failed') from exc
        self.check_deadline()
        raise _PreparedTranscript(entries,sources)


def qualified_plan(model,control,serial,parallel,*,policy,mode,eligible=True,preparation=None,serial_replay=None):
    if control is None:
        from ._planning_controller import run
        try:
            return run(model,serial,parallel,policy=policy,mode=mode,eligible=eligible,preparation=preparation,serial_replay=serial_replay)
        except _CallbackFailed as failure:
            raise failure.error
    owned_control=type(control) is StageControl
    if owned_control:
        if control.mode!='not-run':raise GeometryError('stage control already used')
        control.mode='running'
    try:
        return _qualified_plan(model,control,serial,parallel,policy=policy,mode=mode,eligible=eligible,preparation=preparation,serial_replay=serial_replay)
    except _CallbackFailed as failure:
        raise failure.error
    finally:
        if owned_control and control.mode=='running':control.mode='terminal'
        if owned_control and control._memory_lease is not None:
            control.memory_report=control._memory_lease.report()
            control.memory_report['outcome']=control.mode


def _qualified_plan(model,control,serial,parallel,*,policy,mode,eligible=True,preparation=None,serial_replay=None):
    if type(control) is not StageControl:
        _increment('parallel_planning_decline:mandatory_ci');return serial()
    # Capture once: admission/callbacks cannot renew the caller's budget by
    # changing the control. An absent deadline never creates a serial allowance.
    overall_deadline=_overall_deadline(control.overall_deadline)
    def check_overall():
        if overall_deadline is not None and time.monotonic()>=overall_deadline:
            raise PlannerDeadline('caller planning deadline consumed')
    bridge=None;before=None;lease=None;poll=check_overall;package_verify=None
    def check_resources():
        # The last memory observation must cover allocations made by source
        # serialization, without executing another caller cancellation callback.
        if lease is not None:lease.check()
        check_overall()
    def invoke_parallel(active):
        # Preserve direct bridge-call exception types. Only a numerical callback
        # needs the carrier that crosses its GeometryError retry handlers.
        previous=getattr(active,'_inside_numerical_call',False)
        active._inside_numerical_call=True
        try:return parallel(active)
        finally:active._inside_numerical_call=previous
    def check_cancel():
        try:
            cancelled=policy.cancellation_check is not None and bool(policy.cancellation_check())
        except BaseException as exc:
            raise _CallbackFailed(exc) from exc
        if cancelled:raise PlannerCancelled('intersection planning cancelled')
    def check_host():
        check_overall()
        check_cancel()
        check_overall()
    def source_bytes():return json.dumps(to_dict(model),sort_keys=True,separators=(',',':'),allow_nan=False)
    try:
        check_overall()
        if not eligible or mode!='exact' or policy.max_predicates is not None or policy.max_candidate_pairs is not None:
            raise QualificationDecline('PP3/special-route adapter unavailable')
        if type(control.reuse_workers) is not bool:raise QualificationDecline('session selection must be bool')
        if control.reuse_workers and preparation is not None:
            from .runtime_diagnostics import _COLLECTORS,_RECORD_COLLECTORS
            if _COLLECTORS.get() or _RECORD_COLLECTORS.get():
                raise QualificationDecline('session replay cannot duplicate active diagnostic collectors')
        if policy.cancellation_check is not None and not is_cancellation_signal(policy.cancellation_check):
            raise QualificationDecline('parallel cancellation requires an owner-issued passive signal')
        poll=check_host
        check_ci_equality(control.expected_ci,control.expected_ci)
        if not callable(control.observed_ci):raise QualificationDecline('mandatory applied-model CI callback missing')
        if type(control.timeout) not in (int,float) or not 0<control.timeout<=110:raise QualificationDecline('stage timeout must be in (0,110]')
        if control.memory_budget is not None:
            from ._planning_memory import MemoryBudget
            if type(control.memory_budget) is not MemoryBudget:raise QualificationDecline('unknown host memory budget')
            if control.tree_memory_limit is not None:
                control.tree_options()
                if control.tree_memory_limit!=control.memory_budget.tree_limit_bytes:
                    raise QualificationDecline('conflicting tree memory caps')
            check_host()
            lease=control.memory_budget.begin();control._memory_lease=lease
            def poll():
                check_host()
                check_resources()
        poll()
        packet=freeze_model(model,**control.limits)
        poll();before=source_bytes()
        poll()
        if control.admission is not None and control.admission(model,policy) is not True:
            raise QualificationDecline('PP4 host/memory admission refused')
        if source_bytes()!=before:raise PlannerSourceChanged('source changed during admission')
        package_selected=(not control.reuse_workers and preparation is not None or
                          control.reuse_workers and preparation is None)
        if package_selected:
            if preparation is not None:_stage_descriptor(preparation,model,policy)
            package_verify=_admit_stage_package(model,control,policy,packet)
            ordinary_poll=poll
            def package_boundary():
                ordinary_poll();package_verify();check_resources()
            poll.boundary=package_boundary
        bridge=(_PreparingBridge(packet,control,preparation() if callable(preparation) else preparation,poll=poll) if control.reuse_workers and preparation is not None
                else (_SessionBridge if control.reuse_workers else _PackageStageBridge if package_selected else Bridge)(packet,control,poll=poll))
        bridge.check_deadline()
        if isinstance(bridge,_PreparingBridge):
            try:
                candidate=invoke_parallel(bridge)
            except _PreparedTranscript as prepared:
                bridge.check_deadline()
                if source_bytes()!=before:raise PlannerSourceChanged('source changed during preparation')
                replay=_TranscriptBridge(packet,control,prepared.entries,sources=prepared.sources,poll=poll)
                replay.deadline=bridge.deadline;bridge=replay
                candidate=invoke_parallel(bridge)
                bridge.finish()
        elif control.reuse_workers:
            try:
                try:
                    candidate=invoke_parallel(bridge)
                except (PlannerCancelled,PlannerDeadline,PlannerCleanupError,PlannerSourceChanged,PlannerMemoryExceeded,QualificationDecline):
                    raise
                except Exception as exc:
                    # Unverified provisional values cannot dictate public error
                    # attribution. Original serial replay follows retirement and
                    # unchanged-source/cancellation/deadline checks below.
                    raise QualificationDecline('provisional session plan failed: '+type(exc).__name__+': '+str(exc)) from exc
                bridge.finish()
            finally:
                bridge.close()
            bridge.verify_pending()
        else:
            candidate=invoke_parallel(bridge)
        bridge.check_deadline()
        if package_verify is not None:package_verify()
        check_ci_equality(control.expected_ci,control.observed_ci(candidate))
        bridge.check_deadline()
        if package_verify is not None:package_verify()
        if source_bytes()!=before:raise PlannerSourceChanged('source changed during stage planning/CI')
        # No caller callback after the last source snapshot: it could mutate
        # the source and otherwise publish an unchecked candidate.
        check_resources();bridge.check_clock()
        control.mode='parallel';control.ci_verified=True
        return candidate
    except QualificationDecline as exc:
        # Cancellation wins over a concurrent admission refusal. The callback may
        # mutate the model, so source identity is checked AFTER the final poll.
        # A raised parallel timeout remains terminal. Merely crossing that
        # clock while retiring an eligible refusal must not invent a timeout.
        poll()
        if package_verify is not None:package_verify()
        if before is not None and source_bytes()!=before:
            raise PlannerSourceChanged('source changed before serial fallback') from exc
        check_resources()
        control.mode='serial';control.decline_reason=type(exc).__name__+': '+str(exc)
        _increment('parallel_planning_decline:'+str(exc))
        direct_serial=bridge is None and control._memory_lease is None
        check_replay=poll
        def replay_poll():
            # GeometryError wrappers in the original kernel must not turn an
            # executor deadline/cancellation into a face-pair numerical error.
            try:check_replay()
            except _CallbackFailed:raise
            except BaseException as error:raise _CallbackFailed(error) from error
        try:
            # Unsupported routes keep their original numerical caller, without
            # installing an unqualified replay adapter.
            result=serial() if direct_serial or serial_replay is None else serial_replay(replay_poll)
            check_replay()
            if before is not None and source_bytes()!=before:raise PlannerSourceChanged('source changed during serial fallback')
            check_resources()
            return result
        except (_CallbackFailed,PlannerCancelled,PlannerDeadline,PlannerCleanupError,PlannerSourceChanged,PlannerMemoryExceeded):
            control.mode='terminal'
            raise
