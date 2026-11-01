"""PP5 host stage bridge; private controls, mandatory real plan/apply CI.

Without explicit controls the opt-in request declines to the original serial
caller. PP4 can supply admission here; PP3 owns future budgets/callback adapters.
"""
import json
import time
from pathlib import Path
import psutil
from ._planning_executor import (OwnedPlanner,PlannerDeadline,PlannerSourceChanged,
    PlannerCancelled,PlannerCleanupError,freeze_model,pack,verify_sources)
from ._planning_qualification import QualificationDecline,check_ci_equality
from ._planning_admission import stratified_samples
from ._planning_values import encode_value,decode_value
from .serialization import to_dict
from .runtime_diagnostics import _increment
from .cancellation import is_cancellation_signal
from .errors import GeometryError


class StageControl:
    def __init__(self,directory,*,expected_ci,observed_ci,workers=8,timeout=100,admission=None,payload_limit=4_000_000,walk_limit=100000,reuse_workers=False,tree_memory_limit=None):
        self.directory=Path(directory);self.expected_ci=expected_ci;self.observed_ci=observed_ci
        self.workers=workers;self.timeout=timeout;self.admission=admission
        self.reuse_workers=reuse_workers
        self.tree_memory_limit=tree_memory_limit
        self.limits=dict(payload_limit=payload_limit,walk_limit=walk_limit)
        self.reports=[];self.mode='not-run';self.decline_reason=None;self.ci_verified=False

    def tree_options(self):
        if self.tree_memory_limit is None:return {}
        from ._planning_process import validate_memory_limit
        validate_memory_limit(self.tree_memory_limit)
        return dict(tree_memory_limit=self.tree_memory_limit)


class _CallbackFailed(BaseException):
    """Keep caller exceptions outside infrastructure-refusal handlers until cleanup."""
    def __init__(self,error):
        self.error=error


class Bridge:
    def __init__(self,packet,control,poll=lambda:None):
        self.packet=packet;self.control=control;self.deadline=time.monotonic()+control.timeout
        self.poll=poll
        self.tree_options=control.tree_options()

    def check_deadline(self):
        if time.monotonic()>=self.deadline:raise PlannerDeadline('stage planning deadline consumed')
        self.poll()
        if time.monotonic()>=self.deadline:raise PlannerDeadline('stage planning deadline consumed')

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
        seed=candidate['sampling_sha256'] if self.control.reuse_workers else candidate['request_sha256']
        sample=stratified_samples(candidate['task_layout'],seed)
        for index in sample['indices']:
            self.check_deadline()
            reference=json.dumps(encode_value(serial_task(tasks[index]),limit=self.control.limits['walk_limit']),sort_keys=True,separators=(',',':'),allow_nan=False)
            if reference!=candidate['results'][index]:
                self._record_mismatch(kind,index,reference,candidate['results'][index])
                raise QualificationDecline('stage sample mismatch '+kind+':'+str(index))
            self.check_deadline()
        candidate.update(stage=kind,sampling=sample)
        self.control.reports.append(candidate)
        return tuple(decode_value(json.loads(value),limit=self.control.limits['walk_limit']) for value in candidate['results'])

    def _record_mismatch(self,kind,index,reference,candidate):
        self.control.directory.mkdir(parents=True,exist_ok=True)
        (self.control.directory/('sample-mismatch-'+kind+'.json')).write_text(json.dumps(dict(index=index,reference=json.loads(reference),candidate=json.loads(candidate)),indent=2),encoding='utf-8')


class _SessionBridge(Bridge):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.session=None;self.pending=[]

    def run(self,kind,tasks,serial_task):
        if not tasks:return ()
        self.check_deadline()
        from ._planning_session import OwnedSession
        try:
            if self.session is None:
                self.session=OwnedSession(self.control.directory/'session',workers=self.control.workers,
                    timeout=min(110,self.deadline-time.monotonic()),**self.control.limits,**self.tree_options)
            candidate=self.session.submit_stage(kind,self.packet,tasks,poll=self.check_deadline)
        except (OSError,psutil.AccessDenied) as exc:
            raise QualificationDecline('session infrastructure '+type(exc).__name__+': '+str(exc)) from exc
        self.check_deadline()
        self.pending.append((kind,tuple(tasks),serial_task,candidate))
        # These values remain inside the tentative plan transaction, not public.
        return tuple(decode_value(json.loads(value),limit=self.control.limits['walk_limit']) for value in candidate['results'])

    def finish(self):
        if self.session is not None:self.session.finish(poll=self.check_deadline)

    def close(self):
        if self.session is not None:
            owned=self.session;self.session=None
            try:owned.close()
            except Exception as exc:raise PlannerCleanupError('session planner cleanup failed') from exc

    def verify_pending(self):
        if self.session is not None:raise PlannerCleanupError('session must retire before verification')
        for kind,tasks,serial_task,candidate in self.pending:
            self.check_deadline()
            self.verify_candidate(kind,tasks,serial_task,candidate)


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
        if type(entry) is not dict or set(entry)!={'kind','tasks','candidate'} or entry['kind']!=kind:
            raise QualificationDecline('transcript stage binding')
        def canonical(value):return pack(value,'transcript',limit=self.control.limits['payload_limit'])
        if canonical(list(tasks))!=canonical(entry['tasks']):raise QualificationDecline('transcript task bytes differ')
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
                    timeout=min(110,self.deadline-time.monotonic()),**self.control.limits,**self.tree_options)
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


def _qualified_plan(model,control,serial,parallel,*,policy,mode,eligible=True,preparation=None,serial_replay=None):
    if type(control) is not StageControl:
        _increment('parallel_planning_decline:mandatory_ci');return serial()
    bridge=None;before=None;poll=lambda:None
    def check_cancel():
        try:
            cancelled=policy.cancellation_check is not None and bool(policy.cancellation_check())
        except BaseException as exc:
            raise _CallbackFailed(exc) from exc
        if cancelled:raise PlannerCancelled('intersection planning cancelled')
    def source_bytes():return json.dumps(to_dict(model),sort_keys=True,separators=(',',':'),allow_nan=False)
    try:
        if not eligible or mode!='exact' or policy.max_predicates is not None or policy.max_candidate_pairs is not None:
            raise QualificationDecline('PP3/special-route adapter unavailable')
        if type(control.reuse_workers) is not bool:raise QualificationDecline('session selection must be bool')
        if control.reuse_workers and preparation is not None:
            from .runtime_diagnostics import _COLLECTORS,_RECORD_COLLECTORS
            if _COLLECTORS.get() or _RECORD_COLLECTORS.get():
                raise QualificationDecline('session replay cannot duplicate active diagnostic collectors')
        if policy.cancellation_check is not None and not is_cancellation_signal(policy.cancellation_check):
            raise QualificationDecline('parallel cancellation requires an owner-issued passive signal')
        poll=check_cancel
        check_ci_equality(control.expected_ci,control.expected_ci)
        if not callable(control.observed_ci):raise QualificationDecline('mandatory applied-model CI callback missing')
        if type(control.timeout) not in (int,float) or not 0<control.timeout<=110:raise QualificationDecline('stage timeout must be in (0,110]')
        packet=freeze_model(model,**control.limits);before=source_bytes()
        poll()
        if control.admission is not None and control.admission(model,policy) is not True:
            raise QualificationDecline('PP4 host/memory admission refused')
        if source_bytes()!=before:raise PlannerSourceChanged('source changed during admission')
        bridge=(_PreparingBridge(packet,control,preparation() if callable(preparation) else preparation,poll=poll) if control.reuse_workers and preparation is not None
                else (_SessionBridge if control.reuse_workers else Bridge)(packet,control,poll=poll))
        bridge.check_deadline()
        if isinstance(bridge,_PreparingBridge):
            try:
                candidate=parallel(bridge)
            except _PreparedTranscript as prepared:
                bridge.check_deadline()
                if source_bytes()!=before:raise PlannerSourceChanged('source changed during preparation')
                replay=_TranscriptBridge(packet,control,prepared.entries,sources=prepared.sources,poll=poll)
                replay.deadline=bridge.deadline;bridge=replay
                candidate=parallel(bridge)
                bridge.finish()
        elif control.reuse_workers:
            try:
                try:
                    candidate=parallel(bridge)
                except (PlannerCancelled,PlannerDeadline,PlannerCleanupError,PlannerSourceChanged,QualificationDecline):
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
            candidate=parallel(bridge)
        bridge.check_deadline()
        check_ci_equality(control.expected_ci,control.observed_ci(candidate))
        bridge.check_deadline()
        if source_bytes()!=before:raise PlannerSourceChanged('source changed during stage planning/CI')
        control.mode='parallel';control.ci_verified=True
        return candidate
    except QualificationDecline as exc:
        # Cancellation wins over a concurrent admission refusal. The callback may
        # mutate the model, so source identity is checked AFTER the final poll.
        if bridge is not None:bridge.check_deadline()
        else:poll()
        if before is not None and source_bytes()!=before:
            raise PlannerSourceChanged('source changed before serial fallback') from exc
        control.mode='serial';control.decline_reason=type(exc).__name__+': '+str(exc)
        _increment('parallel_planning_decline:'+str(exc))
        if bridge is None:return serial()
        def replay_poll():
            # GeometryError wrappers in the original kernel must not turn an
            # executor deadline/cancellation into a face-pair numerical error.
            try:bridge.check_deadline()
            except _CallbackFailed:raise
            except BaseException as error:raise _CallbackFailed(error) from error
        try:
            result=serial() if serial_replay is None else serial_replay(replay_poll)
            bridge.check_deadline()
            if source_bytes()!=before:raise PlannerSourceChanged('source changed during serial fallback')
            bridge.check_deadline()
            return result
        except (_CallbackFailed,PlannerCancelled,PlannerDeadline,PlannerCleanupError,PlannerSourceChanged):
            control.mode='terminal'
            raise
