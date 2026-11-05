"""Private, finite per-plan transport. Stage replies are never public plans.

One original runtime baseline, a new closed gate for each phase, and an explicit
finish transaction. No process or poisoned gate is reused by another plan.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from threadpoolctl import threadpool_info
from ._planning_executor import (OwnedPlanner, PlannerCancelled, PlannerDeadline,
    PlannerCleanupError, pack, unpack, read_packet, write_packet, check_result_status,
    source_files, verify_sources, verify_threads, reduce_results, thaw_model,
    assignment_size, _private_packet_io, _atomic_private_text)
from ._planning_qualification import QualificationDecline
from . import _planning_qualification as q
from . import _planning_ledger as ledger_api
from ._planning_batch import BatchWorkerGate
from ._planning_stages import WorkerModel, stage_task, dispatch
from ._planning_process import OwnedTree


def _digest(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


_INITIALIZATION_PHASES=frozenset(('child_prepare','setup_and_initial_thaw','infrastructure',
    'model_context','package_ledger','environment_closure','worker_creation',
    'gate_pair','gate_face','finish','retirement','child_process_total',
    'finish_received','finish_receipts',*(f'finish_wait_{index}' for index in range(8))))


def _read_initialization(folder,pid):
    def read():
        with (folder/'initialization-phases.json').open('rb') as stream:return stream.read(8193)
    raw=_private_packet_io(read)
    if len(raw)>8192:raise QualificationDecline('initialization evidence size')
    try:row=json.loads(raw)
    except (ValueError,UnicodeError) as exc:raise QualificationDecline('initialization evidence JSON') from exc
    if (type(row) is not dict or set(row) not in ({'pid','phases'},{'pid','phases','events'}) or type(row['pid']) is not int or row['pid']!=pid or
            type(row['phases']) is not dict or not set(row['phases'])<=_INITIALIZATION_PHASES or
            any(type(value) is not float or not math.isfinite(value) or value<0 for value in row['phases'].values())):
        raise QualificationDecline('initialization evidence binding/schema')
    if 'events' in row:
        events=row['events']
        if type(events) is not dict or not set(events)<=_INITIALIZATION_PHASES:
            raise QualificationDecline('initialization event schema')
        for name,event in events.items():
            if (type(event) is not dict or set(event)!={'started','ended'} or
                    type(event['started']) is not float or not math.isfinite(event['started']) or event['started']<0):
                raise QualificationDecline('initialization event clock')
            end=event['ended']
            if end is None:
                if name in row['phases']:raise QualificationDecline('initialization pending event completed')
            elif (type(end) is not float or not math.isfinite(end) or end<event['started'] or
                    row['phases'].get(name)!=end-event['started']):
                raise QualificationDecline('initialization event duration')
    return row


def _record_initialization(folder,name,started,*,pending=False):
    """Bounded process perf-counter observations, never additive across workers.

    A pending event proves only entry into a wait/cleanup phase. Completion is
    recorded only after the existing operation returns successfully.
    """
    if name not in _INITIALIZATION_PHASES:raise QualificationDecline('initialization phase name')
    if type(pending) is not bool or type(started) is not float or not math.isfinite(started) or started<0:
        raise QualificationDecline('initialization phase clock')
    path=folder/'initialization-phases.json'
    row=_read_initialization(folder,os.getpid()) if path.exists() else dict(pid=os.getpid(),phases={})
    if name in row['phases']:raise QualificationDecline('initialization phase repeated')
    events=row.setdefault('events',{})
    previous=events.get(name)
    if previous is not None and (pending or previous!={'started':started,'ended':None}):
        raise QualificationDecline('initialization event repeated/mismatched')
    ended=time.perf_counter();duration=ended-started
    if type(ended) is not float or not math.isfinite(ended) or duration<0:
        raise QualificationDecline('initialization phase clock')
    events[name]=dict(started=started,ended=None if pending else ended)
    if not pending:row['phases'][name]=duration
    encoded=json.dumps(row,allow_nan=False)
    if len(encoded.encode('utf-8'))>8192:raise QualificationDecline('initialization evidence size')
    _atomic_private_text(path,encoded,temporary=path.with_suffix('.tmp'))


def _retire_session_tree(tree,folder,*,production,prior_error=None):
    """Tracing cannot skip containment or replace a pending primary failure."""
    trace_error=None
    try:
        stamp=time.perf_counter()
        if production:_record_initialization(folder,'retirement',stamp,pending=True)
    except BaseException as exc:trace_error=exc
    try:
        tree.close()
        (folder/'cleanup.json').replace(folder/'workers-cleanup.json')
    except Exception as exc:raise PlannerCleanupError('session worker cleanup failed') from exc
    if production and trace_error is None:
        try:_record_initialization(folder,'retirement',stamp)
        except BaseException as exc:trace_error=exc
    if trace_error is not None and prior_error is None:
        raise PlannerCleanupError('session retirement evidence failed after containment') from trace_error


def _read_published_packet(path,kind,deadline,*,poll=None,cancel_folder=None,**kwargs):
    """Retry only temporary open denial of an already-published owned packet.

    No source/image/storage reads use this adapter. Exact packet validation is
    repeated unchanged. A retained sidecar records every permission episode;
    callback and deadline errors retain their original terminal precedence.
    """
    first=None;event=None
    def save():
        target=path.with_name('packet-read-'+str(os.getpid())+'-'+path.name)
        temporary=target.with_suffix(target.suffix+'.tmp')
        temporary.write_text(json.dumps(event,allow_nan=False),encoding='utf-8');temporary.replace(target)
    def retry_poll():
        if cancel_folder is not None and (cancel_folder/'cancel.flag').exists():
            raise PlannerCancelled('owned session cancelled during packet read')
        if poll is not None:poll()
    def denied(error,attempt,started):
        nonlocal first,event
        if first is None:first=error
        event=dict(packet=path.name,kind=kind,pid=os.getpid(),attempts=attempt,status='retrying',
            errno=error.errno,winerror=getattr(error,'winerror',None),
            first_errno=first.errno,first_winerror=getattr(first,'winerror',None),
            elapsed_seconds=time.monotonic()-started,maximum_retry_seconds=.05)
        save()
    def recovered(attempt,started):
        event.update(attempts=attempt,status='recovered',elapsed_seconds=time.monotonic()-started);save()
    def io(operation):
        return _private_packet_io(operation,deadline=deadline,poll=retry_poll,
            on_denial=denied,on_recovery=recovered)
    try:
        return read_packet(path,kind,_io=io,**kwargs)
    except BaseException as error:
        if event is not None:
            event.update(status=type(error).__name__)
            try:save()
            except OSError:pass  # Preserve the original terminal exception; prior episode record remains.
        raise



def _packet_read_evidence(folder):
    rows=[]
    for path in folder.rglob('packet-read-*.json'):
        if len(rows)>=128:raise QualificationDecline('packet retry evidence count')
        def read():
            with path.open('rb') as stream:return stream.read(8193)
        raw=_private_packet_io(read)
        if len(raw)>8192:raise QualificationDecline('packet retry evidence size')
        row=json.loads(raw)
        if (type(row) is not dict or type(row.get('attempts')) is not int or not 1<=row['attempts']<=8 or
                type(row.get('status')) is not str or len(row['status'])>200 or row.get('maximum_retry_seconds')!=.05):
            raise QualificationDecline('packet retry evidence schema')
        rows.append(dict(path=str(path.relative_to(folder)),event=row))
    return sorted(rows,key=lambda row:row['path'])


def _packet_bytes(path,kind,limit):
    """Actual bounded envelope/payload bytes, not an inferred compression ratio."""
    def read():
        size=path.stat().st_size
        if size>2*limit+100000:raise QualificationDecline('transport observation envelope bound')
        return size,path.read_bytes()
    size,raw=_private_packet_io(read)
    if len(raw)!=size or len(raw)>2*limit+100000:
        raise QualificationDecline('transport observation changed envelope')
    packet=json.loads(raw)
    unpack(packet,kind,limit=limit)
    return dict(payload_utf8_bytes=len(packet['payload'].encode('utf-8')),envelope_bytes=len(raw))


def task_binding(tasks, *, payload_limit=4_000_000):
    return dict(task_count=len(tasks),tasks_sha256=pack(list(tasks),'tasks',limit=payload_limit)['sha256'])


def validate_entry(entry,kind,tasks,*,payload_limit=4_000_000):
    """Bind independently reconstructed tasks; legacy diagnostics stay readable."""
    if type(entry) is not dict or entry.get('kind')!=kind:
        raise QualificationDecline('transcript stage binding')
    fields=set(entry)
    if fields=={'kind','tasks','candidate'}:
        if pack(entry['tasks'],'tasks',limit=payload_limit)!=pack(list(tasks),'tasks',limit=payload_limit):
            raise QualificationDecline('transcript task bytes differ')
    elif fields=={'kind','task_count','tasks_sha256','candidate'}:
        expected=task_binding(tasks,payload_limit=payload_limit)
        if (type(entry['task_count']) is not int or not _digest(entry['tasks_sha256']) or
                any(entry[key]!=value for key,value in expected.items())):
            raise QualificationDecline('transcript task digest/count binding')
    else:raise QualificationDecline('transcript entry schema')


def _validate_rows(rows,assigned):
    if (type(rows) is not list or any(type(row) is not list or len(row)!=2 or
            type(row[0]) is not int or type(row[1]) is not str for row in rows) or
            [row[0] for row in rows]!=list(assigned)):
        raise QualificationDecline('prepared worker assignment coverage')


def compact_worker_report(report,assigned,*,payload_limit=4_000_000):
    """Validate original ordered rows before dropping the nested result echo."""
    if type(report) is not dict or 'result_count' in report or 'results_sha256' in report:
        raise QualificationDecline('mixed worker result schema')
    rows=report.get('rows');_validate_rows(rows,assigned)
    result={key:value for key,value in report.items() if key!='rows'}
    result.update(result_count=len(rows),results_sha256=pack(rows,'worker-results',limit=payload_limit)['sha256'])
    return result


def _startup_concurrency(value,workers):
    if type(value) is not int or type(workers) is not int or not 1<=value<=workers:
        raise QualificationDecline('invalid startup concurrency')
    return value


def _verify_finish_workers(reports,stages,workers,model_sha):
    if type(reports) is not list or len(reports)!=workers:
        raise QualificationDecline('session finish worker coverage')
    for i,report in enumerate(reports):
        if (type(report) is not dict or report.get('status')!='passed' or
                type(report.get('worker')) is not int or report['worker']!=i or report.get('stages')!=list(stages) or
                type(report.get('gate_closures')) is not int or report['gate_closures']!=len(stages) or
                report.get('model_sha256')!=model_sha or type(report.get('model_decode_count')) is not int or
                report['model_decode_count']!=1 or type(report.get('pid')) is not int or report['pid']<=0):
            raise QualificationDecline('session finish worker gate/model/PID binding')
        verify_threads(report.get('blas_threads',[]))


def validate_command(body, *, session, sequence, previous, seen, worker=None):
    fields={'session','sequence','previous','stage','tasks'}
    if worker is not None:fields|={'worker','parent_request'}
    if type(body) is not dict or set(body)!=fields:
        raise QualificationDecline('session command schema')
    if (not _digest(body['session']) or body['session']!=session or
            type(body['sequence']) is not int or body['sequence']!=sequence or
            not _digest(body['previous']) or body['previous']!=previous or
            type(seen) is not tuple or seen not in ((),('pair',),('face',),('pair','face')) or
            sequence!=len(seen) or type(body['stage']) is not str or type(body['tasks']) is not list):
        raise QualificationDecline('session command binding/order')
    kind=body['stage']
    if (kind=='pair' and seen or kind=='face' and seen not in ((),('pair',)) or
            kind=='finish' and (not seen or body['tasks']) or kind not in ('pair','face','finish')):
        raise QualificationDecline('session phase repeated/reordered/unstarted')
    if worker is not None:
        if (type(body['worker']) is not int or body['worker']!=worker or
                not _digest(body['parent_request'])):
            raise QualificationDecline('worker session command binding')
        ordinals=[]
        tasks=[]
        for row in body['tasks']:
            if (type(row) is not list or len(row)!=2 or type(row[0]) is not int or row[0]<0):
                raise QualificationDecline('worker session ordinal schema')
            ordinals.append(row[0]);tasks.append(row[1])
        if len(set(ordinals))!=len(ordinals):raise QualificationDecline('duplicate worker ordinal')
    else:tasks=body['tasks']
    if any(type(t) is not dict or t.get('kind')!=kind for t in tasks):
        raise QualificationDecline('session task/stage mismatch')
    return kind


def sampling_digest(packet,tasks,*,workers,batch_size,sources,payload_limit=4_000_000):
    """Exactly the legacy request, independent of session transport chaining."""
    return pack(dict(model=packet,tasks=tasks,workers=workers,batch_size=batch_size,
                     sources=sources,stage=True),'request',limit=payload_limit)['sha256']


def _path(folder,kind,sequence):
    return Path(folder)/(kind+'-'+str(sequence)+'.json')


def _arm_memory_failure_reply(folder,binding,sequence,publication,*,payload_limit):
    """Prewrite one failure-only result after the current command is validated."""
    if type(publication) is not list or len(publication)!=1:
        raise QualificationDecline('worker memory publication context')
    publication[0]=None
    names=('session_sha256','worker','worker_setup_sha256','request_sha256')
    if (type(binding) is not dict or set(binding)!=set(names) or
            type(sequence) is not int or not 0<=sequence<=2 or
            type(binding['worker']) is not int or binding['worker']<0 or
            any(type(binding[name]) is not str or len(binding[name])!=64 or
                any(c not in '0123456789abcdef' for c in binding[name])
                for name in (names[0],names[2],names[3]))):
        raise QualificationDecline('worker memory failure binding')
    pending=str(folder/'memory-error.pending.json');destination=str(folder/'result.json')
    report=dict(binding,status='declined',reason='MemoryError: ',sequence=sequence,
                memory_failure='armed-command-v1')
    write_packet(Path(pending),pack(report,'worker-result',limit=payload_limit))
    publication[0]=(pending,destination,binding['session_sha256'],binding['worker'],
                    binding['worker_setup_sha256'],binding['request_sha256'],sequence)


def _check_worker_memory_failure(report,session,request,worker,setup,sequence):
    """Check the armed failure binding before the generic status propagator."""
    if type(report) is dict and report.get('status')=='declined' and report.get('reason')=='MemoryError: ':
        if (report.get('memory_failure')!='armed-command-v1' or
                type(report.get('sequence')) is not int or report['sequence']!=sequence or
                type(report.get('worker')) is not int or report['worker']!=worker or
                report.get('session_sha256')!=session or report.get('request_sha256')!=request or
                report.get('worker_setup_sha256')!=setup):
            raise QualificationDecline('worker memory failure binding')


def _wait_file(path,deadline,*,poll=None,child=None,folder=None,result_kind=None,memory_binding=None):
    while not path.exists():
        if poll is not None:poll()
        if time.monotonic()>=deadline:raise PlannerDeadline('session deadline consumed')
        if child is not None and child.poll() is not None:
            if path.exists():break
            if folder is not None and (folder/'result.json').exists():
                report=read_packet(folder/'result.json',result_kind)
                if memory_binding is not None:_check_worker_memory_failure(report,*memory_binding)
                check_result_status(report)
            raise QualificationDecline('session child ended before reply')
        time.sleep(.01)
    if poll is not None:poll()
    if time.monotonic()>=deadline:raise PlannerDeadline('session deadline consumed')


def _reply_binding(report,session,request):
    check_result_status(report)
    if report.get('session_sha256')!=session or report.get('request_sha256')!=request:
        raise QualificationDecline('session reply request binding')


class OwnedSession(OwnedPlanner):
    """One fresh owned tree, at most pair/face, one final exit handshake."""
    def __init__(self,directory,*,startup_concurrency=1,numerical_manifest=None,**kwargs):
        if numerical_manifest is not None:
            from ._planning_values import verify_value_codec_contract
            verify_value_codec_contract(numerical_manifest.get('value_codec'))
        self.numerical_manifest=numerical_manifest
        self.startup_concurrency=_startup_concurrency(startup_concurrency,kwargs.get('workers',8))
        if startup_concurrency>1 and kwargs.get('tree_memory_limit') is None:
            raise QualificationDecline('concurrent startup requires explicit tree memory cap')
        super().__init__(directory,_role='session-planner',_package_admission=numerical_manifest,**kwargs)
        self.session=None;self.previous=None;self.stages=();self.sources=None
        self.packet=None;self.failed=False;self.finished=False;self._stage_receipts=[];self._stage_reports=[]
        self._infrastructure=None
        self._transport_setup=None

    def _verify_infrastructure(self,report):
        check_result_status(report)
        if self.numerical_manifest is None:return
        value=report.get('infrastructure')
        if (type(value) is not dict or set(value)!={'host','planner','planner_nodes','resource','planner_reference','worker_reference','value_codec'} or
                value['host']!=self.numerical_manifest['infrastructure_host'] or
                type(value['planner']) is not str or len(value['planner'])!=64 or
                any(c not in '0123456789abcdef' for c in value['planner']) or
                type(value['planner_nodes']) is not int or not 0<value['planner_nodes']<=80000 or
                value['resource']!=self.numerical_manifest['resource'] or
                value['value_codec']!=self.numerical_manifest['value_codec'] or
                not _digest(value['planner_reference']) or not _digest(value['worker_reference'])):
            raise QualificationDecline('planner infrastructure request binding')
        if self._infrastructure is None:self._infrastructure=dict(value)
        elif value!=self._infrastructure:raise QualificationDecline('planner infrastructure changed between stages')

    def _poll(self,poll):
        if time.monotonic()>=self.deadline:raise PlannerDeadline('session deadline consumed')
        if (self.directory/'cancel.flag').exists():raise PlannerCancelled('owned session cancelled')
        if poll is not None:poll()
        if time.monotonic()>=self.deadline:raise PlannerDeadline('session deadline consumed')

    def _command(self,kind,tasks):
        body=dict(session=self.session,sequence=len(self.stages),previous=self.previous,stage=kind,tasks=list(tasks))
        validate_command(body,session=self.session,sequence=len(self.stages),previous=self.previous,seen=self.stages)
        return body,pack(body,'session-command',limit=self.limits['payload_limit'])

    def submit_stage(self,kind,packet,tasks,*,poll=None):
        if self.closed or self.failed or self.finished:raise QualificationDecline('session is terminal')
        try:
            self._poll(poll)
            if self.session is None:
                self.packet=dict(packet);self.sources=source_files()
                setup=dict(model=self.packet,sources=self.sources,workers=self.workers,
                           batch_size=self.batch_size,limits=self.limits)
                if self.startup_concurrency!=1:setup['startup_concurrency']=self.startup_concurrency
                if self.numerical_manifest is not None:setup['numerical_manifest']=self.numerical_manifest
                initial=pack(setup,'session',limit=self.limits['payload_limit'])
                self.session=self.previous=initial['sha256']
                write_packet(self.directory/'session.json',initial)
                if self.numerical_manifest is not None:
                    self._transport_setup=_packet_bytes(self.directory/'session.json','session',self.limits['payload_limit'])
            if packet!=self.packet:raise QualificationDecline('session model changed')
            seed=sampling_digest(self.packet,tasks,workers=self.workers,batch_size=self.batch_size,
                                 sources=self.sources,payload_limit=self.limits['payload_limit'])
            body,command=self._command(kind,tasks)
            sequence=len(self.stages)
            write_packet(_path(self.directory,'command',sequence),command)
            path=_path(self.directory,'reply',sequence)
            _wait_file(path,self.deadline,poll=lambda:self._poll(poll),child=self.child,
                       folder=self.directory,result_kind='result')
            report=(_read_published_packet(path,'session-reply',self.deadline,poll=lambda:self._poll(poll),limit=self.limits['payload_limit'])
                if self.numerical_manifest is not None else read_packet(path,'session-reply',limit=self.limits['payload_limit']))
            self._verify_infrastructure(report)
            _reply_binding(report,self.session,command['sha256'])
            if report.get('stage')!=kind:raise QualificationDecline('session reply stage binding')
            verify_sources(self.sources)
            parts,_=dispatch(body['tasks'],self.workers)
            _validate_rows(report.get('rows'),[ordinal for part in parts for ordinal in part])
            report['results']=reduce_results(report.pop('rows'),len(body['tasks']))
            # Provisional stages still require full compact accounting. Final
            # PID/gate binding is repeated against independent finish receipts.
            setup=dict(workers=self.workers,model=self.packet,startup_concurrency=self.startup_concurrency)
            if self.numerical_manifest is not None:
                setup['resource_identity']=self.numerical_manifest['resource']
                setup['value_codec']=self.numerical_manifest['value_codec']
            verify_prepared_stage(dict(report,results=list(report['results'])),body['tasks'],setup,
                report.get('worker_reports'),payload_limit=self.limits['payload_limit'])
            if self.numerical_manifest is not None:
                diagnostics=[len(json.dumps(worker.get('diagnostics',[]),sort_keys=True,separators=(',',':'),
                    allow_nan=False).encode('utf-8')) for worker in report['worker_reports']]
                report['transport_bytes']=dict(payload_limit_bytes=self.limits['payload_limit'],session_setup=self._transport_setup,
                    request=_packet_bytes(_path(self.directory,'command',sequence),'session-command',self.limits['payload_limit']),
                    reply=_packet_bytes(path,'session-reply',self.limits['payload_limit']),
                    worker_diagnostics_canonical_json_bytes=diagnostics,
                    diagnostics_canonical_json_total_bytes=sum(diagnostics),
                    result_value_utf8_total_bytes=sum(len(value.encode('utf-8')) for value in report['results']),
                    scope='actual H/P envelopes; diagnostics and result values are contained components, not additive wire totals')
            report['sampling_sha256']=seed
            report['host_timing']=dict(start_to_return_seconds=time.perf_counter()-self.started)
            if self.numerical_manifest is not None:
                report['host_timing']['initialization']=dict(planner=_read_initialization(self.directory,self.child.pid),
                    workers=[_read_initialization(self.directory/('worker-'+str(index)),worker['pid'])
                        for index,worker in enumerate(report['worker_reports'])],
                    scope='inclusive per-process phases; cross-process overlap; not additive critical-path time')
                report['host_timing']['packet_read_retries']=_packet_read_evidence(self.directory)
            self.stages+=kind,
            self.previous=command['sha256'];self._stage_receipts.append(command['sha256'])
            self._stage_reports.append(report)
            return report
        except BaseException:
            self.failed=True
            raise

    def prepare_plan(self,packet,specification,*,poll=None):
        if self.closed or self.failed or self.finished or self.session is not None:raise QualificationDecline('session is terminal')
        try:
            self._poll(poll);decode_preparation(specification)
            self.packet=dict(packet);self.sources=source_files()
            setup=dict(model=self.packet,sources=self.sources,workers=self.workers,batch_size=self.batch_size,limits=self.limits)
            if self.startup_concurrency!=1:setup['startup_concurrency']=self.startup_concurrency
            initial=pack(setup,'session',limit=self.limits['payload_limit']);self.session=initial['sha256']
            plan=pack(specification,'plan-preparation',limit=self.limits['payload_limit'])
            # Plan precedes the session latch: P never mistakes this for interactive mode.
            write_packet(self.directory/'plan.json',plan);write_packet(self.directory/'session.json',initial)
            path=self.directory/'result.json'
            _wait_file(path,self.deadline,poll=lambda:self._poll(poll),child=self.child)
            self.tree.wait(self.child,self.deadline,poll=lambda:self._poll(poll))
            result=read_packet(path,'result',limit=self.limits['payload_limit']);check_result_status(result)
            if result.get('preparation_sha256')!=plan['sha256'] or result.get('session_sha256')!=self.session:
                raise QualificationDecline('preparation result binding')
            entries=result.get('prepared_entries');stages=result.get('stages')
            if type(entries) is not list or stages not in (['pair'],['face'],['pair','face']) or len(entries)!=len(stages):
                raise QualificationDecline('preparation stage coverage')
            seen=();previous=self.session;requests=[];stage_tasks=[]
            for i,kind in enumerate([*stages,'finish']):
                body,request=read_packet(_path(self.directory,'command',i),'session-command',with_digest=True,limit=self.limits['payload_limit'])
                if validate_command(body,session=self.session,sequence=i,previous=previous,seen=seen)!=kind:
                    raise QualificationDecline('preparation command order')
                if kind!='finish':
                    entry=entries[i]
                    validate_entry(entry,kind,body['tasks'],payload_limit=self.limits['payload_limit'])
                    stage_tasks.append(body['tasks'])
                    candidate=entry['candidate'];_reply_binding(candidate,self.session,request)
                    if 'preparation_samples' not in candidate:raise QualificationDecline('preparation samples missing')
                    seen+=kind,;requests.append(request)
                previous=request
            _reply_binding(result,self.session,previous)
            if result.get('stage_requests')!=requests:raise QualificationDecline('preparation request coverage')
            reports=result.get('worker_reports')
            _verify_finish_workers(reports,stages,self.workers,packet['sha256'])
            for entry,tasks in zip(entries,stage_tasks):
                self._poll(poll)
                verify_prepared_stage(entry['candidate'],tasks,setup,reports,payload_limit=self.limits['payload_limit'])
            verify_sources(self.sources);self.finished=True
            return entries
        except BaseException:
            self.failed=True;raise

    def finish(self,*,poll=None):
        if self.closed or self.failed or self.finished:raise QualificationDecline('session is terminal')
        try:
            self._poll(poll)
            _,command=self._command('finish',[])
            write_packet(_path(self.directory,'command',len(self.stages)),command)
            path=self.directory/'result.json'
            _wait_file(path,self.deadline,poll=lambda:self._poll(poll),child=self.child)
            self.tree.wait(self.child,self.deadline,poll=lambda:self._poll(poll))
            result=(_read_published_packet(path,'result',self.deadline,poll=lambda:self._poll(poll),limit=self.limits['payload_limit'])
                if self.numerical_manifest is not None else read_packet(path,'result',limit=self.limits['payload_limit']))
            self._verify_infrastructure(result)
            _reply_binding(result,self.session,command['sha256'])
            if result.get('stage_requests')!=self._stage_receipts or result.get('stages')!=list(self.stages):
                raise QualificationDecline('session finish stage coverage')
            reports=result.get('worker_reports')
            _verify_finish_workers(reports,self.stages,self.workers,self.packet['sha256'])
            if self.numerical_manifest is not None:
                for worker in reports:_verify_worker_resources(worker,self.numerical_manifest['resource'],self.numerical_manifest['value_codec'])
            if len(self._stage_reports)!=len(self.stages):
                raise QualificationDecline('session retained stage coverage')
            previous=self.session
            setup=dict(workers=self.workers,model=self.packet,startup_concurrency=self.startup_concurrency)
            if self.numerical_manifest is not None:
                setup['resource_identity']=self.numerical_manifest['resource']
                setup['value_codec']=self.numerical_manifest['value_codec']
            for i,(kind,report) in enumerate(zip(self.stages,self._stage_reports)):
                self._poll(poll)
                body,request=read_packet(_path(self.directory,'command',i),'session-command',
                    with_digest=True,limit=self.limits['payload_limit'])
                if (validate_command(body,session=self.session,sequence=i,previous=previous,seen=self.stages[:i])!=kind or
                        request!=self._stage_receipts[i]):
                    raise QualificationDecline('session retained command binding')
                _reply_binding(report,self.session,request)
                verify_prepared_stage(dict(report,results=list(report['results'])),body['tasks'],setup,reports,
                    payload_limit=self.limits['payload_limit'])
                previous=request
            verify_sources(self.sources)
            if self.numerical_manifest is not None:
                terminal=dict(planner=_read_initialization(self.directory,self.child.pid),
                    workers=[_read_initialization(self.directory/('worker-'+str(index)),worker['pid'])
                        for index,worker in enumerate(reports)])
                retries=_packet_read_evidence(self.directory)
                for report in self._stage_reports:
                    report['host_timing']['terminal_phases']=terminal
                    report['host_timing']['packet_read_retries']=retries
            self.finished=True
        except BaseException:
            self.failed=True
            raise


def _setup(folder,limits,deadline,*,decode=True):
    path=folder/'session.json';_wait_file(path,deadline)
    setup,digest=read_packet(path,'session',with_digest=True,limit=limits['payload_limit'])
    fields={'model','sources','workers','batch_size','limits'}
    if (type(setup) is not dict or set(setup)-{'startup_concurrency','numerical_manifest'}!=fields or
            setup['limits']!=limits or type(setup['workers']) is not int or setup['workers'] not in (1,2,8,16)):
        raise QualificationDecline('session setup schema')
    _startup_concurrency(setup.get('startup_concurrency',1),setup['workers'])
    assignment_size(setup['batch_size'],0)
    verify_sources(setup['sources'])
    if 'numerical_manifest' in setup:
        from ._planning_values import verify_value_codec_contract
        verify_value_codec_contract(setup['numerical_manifest'].get('value_codec'))
    if decode:thaw_model(setup['model'],**limits)
    return setup,digest


def session_planner(folder,deadline,canaries,goldens,binding,limits,generated):
    stamp=time.perf_counter()
    setup,session=_setup(folder,limits,deadline)
    production='numerical_manifest' in setup
    if production:_record_initialization(folder,'setup_and_initial_thaw',stamp)
    binding['session_sha256']=session
    infrastructure=None;infrastructure_record=None
    if 'numerical_manifest' in setup:
        from ._planning_code import InfrastructurePin
        stamp=time.perf_counter()
        infrastructure=InfrastructurePin('planner')
        _record_initialization(folder,'infrastructure',stamp)
        if infrastructure.native!=setup['numerical_manifest']['resource']:
            raise QualificationDecline('H/P resource identity mismatch')
        infrastructure_record=dict(host=setup['numerical_manifest']['infrastructure_host'],
            planner=infrastructure.digest,planner_nodes=infrastructure.nodes,resource=infrastructure.native,
            value_codec=setup['numerical_manifest']['value_codec'])
        if read_packet(folder/'admission.json','package-admission')!=setup['numerical_manifest']:
            raise QualificationDecline('planner admission marker mismatch')
        stamp=time.perf_counter();model=thaw_model(setup['model'],**limits)
        _record_initialization(folder,'model_context',stamp);stamp=time.perf_counter()
        ledger=ledger_api.package_worker_ledger(model,resource_pin=q.resource_file_pin(infrastructure.native),role='planner')
        _record_initialization(folder,'package_ledger',stamp)
        if ledger.live_code_pins()!={name:setup['numerical_manifest'].get(name) for name in ('common','constructor')}:
            (folder/'manifest-mismatch.json').write_text(json.dumps(dict(expected=setup['numerical_manifest'],
                actual=ledger.live_code_pins(),common=ledger.manifest.callables._reference_tokens,
                constructor=ledger.manifest.roles['constructor']._reference_tokens)),encoding='utf-8')
            raise QualificationDecline('H/P common/constructor manifest mismatch')
    else:ledger=ledger_api.loaded_geometry_ledger(trusted_image=True)
    stamp=time.perf_counter();ledger.prepare(owned_worker=True)
    if generated != q.generated_runtime_modules():
        raise QualificationDecline('generated module selection changed after preparation')
    expected=q.capture_environment(generated_modules=generated,declined_stores=ledger.resets,trusted_image=ledger.trusted_image,
        numerical_manifest=ledger.live_code_pins() if 'numerical_manifest' in setup else None,
        resource_pin=getattr(ledger,'resource_pin',None))
    expected['live_code_pins']=ledger.live_code_pins()
    write_packet(folder/'reference.json',pack(expected,'reference'))
    worker_expected=q.worker_environment_reference(expected) if production else expected
    if production:
        write_packet(folder/'worker-reference.json',pack(worker_expected,'worker-reference'))
        infrastructure_record.update(planner_reference=pack(expected,'reference')['sha256'],
            worker_reference=pack(worker_expected,'worker-reference')['sha256'])
    if production:_record_initialization(folder,'environment_closure',stamp)
    tree=OwnedTree(folder);workers=[];seen=();previous=session;stage_requests=[]
    try:
        stamp=time.perf_counter()
        for index in range(setup['workers']):
            location=folder/('worker-'+str(index));location.mkdir()
            initial=pack(dict(session_sha256=session,worker=index,model=setup['model'],sources=setup['sources'],
                              expected=worker_expected,workers=setup['workers'],batch_size=setup['batch_size']),
                         'session-worker',limit=limits['payload_limit'])
            write_packet(location/'limits.json',pack(limits,'limits'))
            if 'numerical_manifest' in setup:
                write_packet(location/'admission.json',pack(setup['numerical_manifest'],'package-admission'))
            write_packet(location/'session.json',initial)
            child=tree.spawn('session-worker',location,max(.1,deadline-time.monotonic()-5))
            workers.append(dict(child=child,folder=location,setup=initial['sha256'],previous=initial['sha256']))
        if production:_record_initialization(folder,'worker_creation',stamp)
        def execute(body,request):
            phase_started=time.perf_counter()
            nonlocal seen,previous
            binding['request_sha256']=request
            kind=validate_command(body,session=session,sequence=len(seen),previous=previous,seen=seen)
            if production and kind=='finish':_record_initialization(folder,'finish_received',phase_started)
            verify_sources(setup['sources'])
            if infrastructure is not None:infrastructure.verify()
            parts,layout=([],[]) if kind=='finish' else dispatch(body['tasks'],setup['workers'])
            sent=[];ready_count=0;max_inflight=0
            concurrency=_startup_concurrency(setup.get('startup_concurrency',1),setup['workers'])
            width=len(workers) if kind=='finish' else concurrency
            for first in range(0,len(workers),width):
                batch=list(enumerate(workers))[first:first+width]
                # The default batch has one worker. Explicitly admitted batches
                # publish every command before waiting for any startup reply.
                for index,item in batch:
                    assigned=[] if index>=len(parts) else [[i,body['tasks'][i]] for i in parts[index]]
                    command=dict(session=session,sequence=len(seen),previous=item['previous'],stage=kind,
                                 tasks=assigned,worker=index,parent_request=request)
                    packet=pack(command,'session-worker-command',limit=limits['payload_limit'])
                    write_packet(_path(item['folder'],'command',len(seen)),packet)
                    sent.append(packet['sha256'])
                if kind=='finish':continue
                max_inflight=max(max_inflight,len(batch))
                for index,item in batch:
                    admitted=_path(item['folder'],'admitted',len(seen))
                    _wait_file(admitted,deadline-3,child=item['child'],folder=item['folder'],result_kind='worker-result',
                        memory_binding=(session,sent[index],index,item['setup'],len(seen)))
                    ready=(_read_published_packet(admitted,'admitted',deadline-3,cancel_folder=folder,limit=limits['payload_limit'])
                        if production else read_packet(admitted,'admitted',limit=limits['payload_limit']))
                    if (type(ready) is not dict or set(ready)!={'stage','sequence'} or
                            type(ready['stage']) is not str or ready['stage']!=kind or
                            type(ready['sequence']) is not int or ready['sequence']!=len(seen)):
                        raise QualificationDecline('session worker admission binding')
                    ready_count+=1
            reports=[]
            for index,item in enumerate(workers):
                if kind=='finish':
                    wait_started=time.perf_counter()
                    if production:_record_initialization(folder,f'finish_wait_{index}',wait_started,pending=True)
                    tree.wait(item['child'],deadline-3)
                    if production:_record_initialization(folder,f'finish_wait_{index}',wait_started)
                    report=(_read_published_packet(item['folder']/'result.json','worker-result',deadline-3,cancel_folder=folder,limit=limits['payload_limit'])
                        if production else read_packet(item['folder']/'result.json','worker-result',limit=limits['payload_limit']))
                else:
                    response=_path(item['folder'],'reply',len(seen))
                    _wait_file(response,deadline-3,child=item['child'],folder=item['folder'],result_kind='worker-result',
                        memory_binding=(session,sent[index],index,item['setup'],len(seen)))
                    report=(_read_published_packet(response,'session-worker-reply',deadline-3,cancel_folder=folder,limit=limits['payload_limit'])
                        if production else read_packet(response,'session-worker-reply',limit=limits['payload_limit']))
                _check_worker_memory_failure(report,session,sent[index],index,item['setup'],len(seen))
                _reply_binding(report,session,sent[index])
                if (report.get('worker_setup_sha256')!=item['setup'] or type(report.get('worker')) is not int or
                        report['worker']!=index or report.get('model_sha256')!=setup['model']['sha256']):
                    raise QualificationDecline('session worker/model binding')
                verify_threads(report.get('blas_threads',[]))
                if production:_verify_worker_resources(report,setup['numerical_manifest']['resource'],setup['numerical_manifest']['value_codec'])
                if (type(report.get('pid')) is not int or report['pid']!=item['child'].pid or
                        type(report.get('model_decode_count')) is not int or report['model_decode_count']!=1):
                    raise QualificationDecline('worker PID/decode binding')
                if kind=='finish':
                    if report.get('stages')!=list(seen) or type(report.get('gate_closures')) is not int or report['gate_closures']!=len(seen):
                        raise QualificationDecline('worker finish gate coverage')
                elif report.get('gate_closed') is not True or report.get('gate_poisoned') is not False:
                    raise QualificationDecline('session stage gate not closed')
                reports.append(report);item['previous']=sent[index]
            if infrastructure is not None:infrastructure.verify()
            if kind=='finish':
                if production:
                    _record_initialization(folder,'finish_receipts',phase_started)
                    _record_initialization(folder,'finish',phase_started)
                return dict(worker_reports=reports,request_sha256=request,infrastructure=infrastructure_record)
            rows=[row for report in reports for row in report['rows']]
            reduce_results(rows,len(body['tasks']))
            reports=[compact_worker_report(report,parts[i] if i<len(parts) else [],
                        payload_limit=limits['payload_limit']) for i,report in enumerate(reports)]
            verify_sources(setup['sources'])
            report=dict(status='passed',session_sha256=session,request_sha256=request,stage=kind,rows=rows,
                        infrastructure=infrastructure_record,
                        worker_reports=reports,requested_workers=setup['workers'],actual_workers=len(parts),
                        task_layout=layout,model_sha256=setup['model']['sha256'],
                        startup=dict(configured_concurrency=concurrency,max_inflight_startups=max_inflight,
                                     ready_workers=ready_count,active_workers=len(parts)))
            write_packet(_path(folder,'reply',len(seen)),pack(report,'session-reply',limit=limits['payload_limit']))
            seen+=kind,;previous=request;stage_requests.append(request)
            return report

        if (folder/'plan.json').exists():
            specification,plan_sha=read_packet(folder/'plan.json','plan-preparation',with_digest=True,limit=limits['payload_limit'])
            operands,policy,seam=decode_preparation(specification)
            from .batch_intersections import _plan_intersections,root_isolation_memo
            from ._planning_values import encode_value,decode_value
            from ._planning_admission import stratified_samples
            model=thaw_model(setup['model'],**limits)
            entries=[]
            def command(kind,tasks):
                body=dict(session=session,sequence=len(seen),previous=previous,stage=kind,tasks=list(tasks))
                packet=pack(body,'session-command',limit=limits['payload_limit'])
                write_packet(_path(folder,'command',len(seen)),packet)
                return execute(body,packet['sha256'])
            class Preparation:
                def check_deadline(self):
                    if time.monotonic()>=deadline-3:raise PlannerDeadline('private preparation deadline consumed')
                def run(self,kind,tasks,serial_task):
                    if not tasks:return ()
                    self.check_deadline()
                    report=command(kind,tasks)
                    values=reduce_results(report.pop('rows'),len(tasks))
                    seed=sampling_digest(setup['model'],tasks,workers=setup['workers'],batch_size=setup['batch_size'],
                                         sources=setup['sources'],payload_limit=limits['payload_limit'])
                    sample=stratified_samples(report['task_layout'],seed)
                    # Untrusted preparation only: reproduce legacy state transitions.
                    # H must independently rebuild the task bytes and rerun ALL samples
                    # after retirement; no value computed here is an accepted oracle.
                    for index in sample['indices']:
                        self.check_deadline()
                        json.dumps(encode_value(serial_task(tasks[index]),limit=limits['walk_limit']),
                                   sort_keys=True,separators=(',',':'),allow_nan=False)
                        self.check_deadline()
                    report.update(results=list(values),sampling_sha256=seed,preparation_samples=sample)
                    entries.append(dict(kind=kind,**task_binding(tasks,payload_limit=limits['payload_limit']),candidate=report))
                    pack(entries,'transcript',limit=limits['payload_limit'])
                    return tuple(decode_value(json.loads(value),limit=limits['walk_limit']) for value in values)
            with root_isolation_memo():
                _plan_intersections(model,operands,policy=policy,_planning_mode='exact',_seam_policy=seam,_stages=Preparation())
            final=command('finish',[])
            reports=final['worker_reports'];request=final['request_sha256']
        else:
            entries=None;plan_sha=None
            while True:
                path=_path(folder,'command',len(seen));_wait_file(path,deadline)
                body,request=read_packet(path,'session-command',with_digest=True,limit=limits['payload_limit'])
                report=execute(body,request)
                if body['stage']=='finish':
                    reports=report['worker_reports'];break
    finally:
        _retire_session_tree(tree,folder,production=production,prior_error=sys.exc_info()[1])
    verify_sources(setup['sources'])
    if infrastructure is not None:infrastructure.verify()
    return dict(status='passed',session_sha256=session,request_sha256=request,stages=list(seen),
                infrastructure=infrastructure_record,
                stage_requests=stage_requests,worker_reports=reports,prepared_entries=entries,preparation_sha256=plan_sha)


def session_worker(folder,deadline,canaries,goldens,binding,limits,generated,*,emergency=None):
    if emergency is not None and (type(emergency) is not list or len(emergency)!=1):
        raise QualificationDecline('worker memory publication context')
    if emergency is not None:emergency[0]=None
    stamp=time.perf_counter()
    setup,setup_sha=read_packet(folder/'session.json','session-worker',with_digest=True,limit=limits['payload_limit'])
    if type(setup) is not dict or set(setup)!={'session_sha256','worker','model','sources','expected','workers','batch_size'}:
        raise QualificationDecline('worker session setup schema')
    root,session=_setup(folder.parent,limits,deadline,decode=False)
    production='numerical_manifest' in root
    if production:_record_initialization(folder,'setup_and_initial_thaw',stamp)
    index=setup['worker']
    if (setup['session_sha256']!=session or type(index) is not int or not 0<=index<root['workers'] or
            setup['workers']!=root['workers'] or setup['batch_size']!=root['batch_size'] or
            setup['sources']!=root['sources'] or setup['model']!=root['model']):
        raise QualificationDecline('worker session setup binding')
    binding.update(session_sha256=session,worker=index,worker_setup_sha256=setup_sha)
    stamp=time.perf_counter();context=WorkerModel(setup['model'],value_codec='v2' if production else None,**limits)
    if production:_record_initialization(folder,'model_context',stamp)
    if 'numerical_manifest' in root:
        if read_packet(folder/'admission.json','package-admission')!=root['numerical_manifest']:
            raise QualificationDecline('worker admission marker mismatch')
        resource=q.resource_file_pin(root['numerical_manifest']['resource'])
        if setup['expected'].get('resource_pin')!=resource:raise QualificationDecline('worker resource reference mismatch')
        stamp=time.perf_counter();ledger=ledger_api.package_worker_ledger(context.model,resource_pin=resource,role='worker')
        _record_initialization(folder,'package_ledger',stamp)
        if ledger.live_code_pins()!={name:root['numerical_manifest'].get(name) for name in ('common','constructor')}:
            raise QualificationDecline('H/W common/constructor manifest mismatch')
    else:ledger=ledger_api.loaded_geometry_ledger(trusted_image=True)
    expected=setup['expected']
    if generated != q.generated_runtime_modules(reference=expected):
        raise QualificationDecline('generated module selection changed after preparation')
    seen=();previous=setup_sha;parent_previous=session;gate=None
    while True:
        if emergency is not None:emergency[0]=None
        path=_path(folder,'command',len(seen));_wait_file(path,deadline)
        body,request=read_packet(path,'session-worker-command',with_digest=True,limit=limits['payload_limit'])
        binding['request_sha256']=request
        kind=validate_command(body,session=session,sequence=len(seen),previous=previous,seen=seen,worker=index)
        parent,parent_request=read_packet(_path(folder.parent,'command',len(seen)),'session-command',
                                           with_digest=True,limit=limits['payload_limit'])
        validate_command(parent,session=session,sequence=len(seen),previous=parent_previous,seen=seen)
        if body['parent_request']!=parent_request or parent['stage']!=kind:
            raise QualificationDecline('worker parent-command binding')
        parts,_=([],[]) if kind=='finish' else dispatch(parent['tasks'],setup['workers'])
        assigned=[] if index>=len(parts) else [[i,parent['tasks'][i]] for i in parts[index]]
        if pack(body['tasks'],'tasks',limit=limits['payload_limit'])!=pack(assigned,'tasks',limit=limits['payload_limit']):
            raise QualificationDecline('worker assignment differs from parent dispatch')
        verify_sources(setup['sources'])
        if emergency is not None:
            _arm_memory_failure_reply(folder,binding,len(seen),emergency,payload_limit=limits['payload_limit'])
        if kind=='finish':
            stamp=time.perf_counter()
            if gate is None or not gate.closed or gate.poisoned:raise QualificationDecline('unfinished session gate')
            gate._boundary();context.verify();verify_threads(threadpool_info())
            if production:_record_initialization(folder,'finish',stamp)
            result=dict(status='passed',**binding,model_sha256=context.verify(),stages=list(seen),gate_closures=len(seen),
                        blas_threads=threadpool_info(),pid=os.getpid(),model_decode_count=1)
            if production:
                result['resource_guard']=_worker_resource_observation(resource['identity'])
                result['value_codec']=root['numerical_manifest']['value_codec']
            return result
        stamp=time.perf_counter();gate=BatchWorkerGate(expected,stores=ledger.resets,owned_worker=True,ledger=ledger,
            generated_modules=generated,batch_size=assignment_size(setup['batch_size'],len(assigned)),coalesce_boundaries=True)
        if production:_record_initialization(folder,'gate_'+kind,stamp)
        started=time.perf_counter();gate.start(canaries,goldens);startup=time.perf_counter()-started
        write_packet(_path(folder,'admitted',len(seen)),pack(dict(stage=kind,sequence=len(seen)),'admitted'))
        tasks=[]
        for ordinal,task in assigned:
            payload=json.dumps(dict(args=[task],kwargs={}),sort_keys=True,separators=(',',':')).encode()
            tasks.append((stage_task,payload,hashlib.sha256(payload).hexdigest()))
        started=time.perf_counter();values=gate.run_tasks(tasks,context=context);elapsed=time.perf_counter()-started
        if time.monotonic()>=deadline:raise PlannerDeadline('session worker deadline consumed')
        verify_threads(threadpool_info());model_sha=context.verify()
        report=dict(status='passed',**binding,rows=[[row[0],value] for row,value in zip(assigned,values)],
                    model_sha256=model_sha,model_decode_count=1,pid=os.getpid(),task_count=len(assigned),
                    batch_size=gate.batch_size,assignment_seconds=elapsed,startup_seconds=startup,
                    task_seconds=gate.task_seconds,boundary_seconds=gate.boundary_seconds,reset_count=len(ledger.costs),
                    full_environment_checks=len(gate.verification_seconds),blas_threads=threadpool_info(),
                    gate_closed=gate.closed,gate_poisoned=gate.poisoned,
                    diagnostics=[[row[0],item] for row,item in zip(assigned,gate.task_diagnostics)])
        if production:
            report['resource_guard']=_worker_resource_observation(resource['identity'])
            report['value_codec']=root['numerical_manifest']['value_codec']
        write_packet(_path(folder,'reply',len(seen)),pack(report,'session-worker-reply',limit=limits['payload_limit']))
        seen+=kind,;previous=request;parent_previous=parent_request


def _worker_resource_observation(identity):
    q.verify_worker_resource_absence()
    return dict(controller_loaded=False,native_adapter_loaded=False,native_core_loaded=False,binary_sha256=identity['binary_sha256'])


def _verify_worker_resources(report,identity,codec=None):
    if codec is not None:
        from ._planning_values import verify_value_codec_contract
        verify_value_codec_contract(codec)
        verify_value_codec_contract(report.get('value_codec'))
        if report['value_codec']!=codec:raise QualificationDecline('worker value codec binding')
    row=report.get('resource_guard')
    if (type(row) is not dict or set(row)!={'controller_loaded','native_adapter_loaded','native_core_loaded','binary_sha256'} or
            any(row[name] is not False for name in ('controller_loaded','native_adapter_loaded','native_core_loaded')) or
            row['binary_sha256']!=identity['binary_sha256']):
        raise QualificationDecline('worker resource role binding')


def decode_preparation(value):
    from .batch_intersections import IntersectionBatchPolicy,SEAM_POLICIES
    fields={'intent','face_connections','member_connections','member_face_connections','planning_mode','seam_policy'}
    if type(value) is not dict or set(value)!={'operands','policy','seam'} or type(value['policy']) is not dict or set(value['policy'])!=fields:
        raise QualificationDecline('preparation descriptor schema')
    policy=value['policy'];operands=value['operands']
    if (type(operands) is not list or any(type(v) is not list or len(v)!=2 or type(v[0]) is not str or
            v[0] not in ('face','member') or type(v[1]) is not int or v[1]<=0 for v in operands) or
            type(policy['intent']) is not str or any(type(policy[k]) is not bool for k in ('face_connections','member_connections','member_face_connections')) or
            policy['planning_mode'] not in (None,'exact') or type(value['seam']) is not str or value['seam'] not in SEAM_POLICIES):
        raise QualificationDecline('preparation descriptor values')
    return tuple(tuple(v) for v in operands),IntersectionBatchPolicy(**policy),value['seam']


def verify_prepared_stage(report,tasks,setup,final,*,payload_limit=4_000_000):
    """Independent host accounting; no worker-supplied coverage is trusted."""
    parts,layout=dispatch(tasks,setup['workers'])
    def canonical(v):return pack(v,'stage-accounting',limit=payload_limit)
    if (type(report) is not dict or report.get('status')!='passed' or
            type(report.get('requested_workers')) is not int or report['requested_workers']!=setup['workers'] or
            type(report.get('actual_workers')) is not int or report['actual_workers']!=len(parts) or
            report.get('model_sha256')!=setup['model']['sha256'] or canonical(report.get('task_layout'))!=canonical(layout) or
            type(report.get('worker_reports')) is not list or len(report['worker_reports'])!=setup['workers']):
        raise QualificationDecline('prepared stage coverage/model')
    results=report.get('results')
    if type(results) is not list or len(results)!=len(tasks) or any(type(value) is not str for value in results):
        raise QualificationDecline('prepared result coverage')
    formats=[type(worker) is dict and ('result_count' in worker or 'results_sha256' in worker)
             for worker in report['worker_reports']]
    if any(formats) and (not all(formats) or 'rows' in report or 'startup' not in report):
        raise QualificationDecline('mixed prepared result schema')
    if 'startup' in report:
        startup=report['startup'];configured=_startup_concurrency(setup.get('startup_concurrency',1),setup['workers'])
        expected=dict(configured_concurrency=configured,max_inflight_startups=min(configured,setup['workers']),
                      ready_workers=setup['workers'],active_workers=len(parts))
        if (type(startup) is not dict or set(startup)!=set(expected) or
                any(type(startup[key]) is not int or startup[key]!=value for key,value in expected.items())):
            raise QualificationDecline('prepared startup coverage')
    rows=[]
    for i,worker in enumerate(report['worker_reports']):
        if 'resource_identity' in setup:_verify_worker_resources(worker,setup['resource_identity'],setup.get('value_codec'))
        assigned=parts[i] if i<len(parts) else []
        if (type(worker) is not dict or worker.get('status')!='passed' or type(worker.get('worker')) is not int or worker['worker']!=i or
                type(worker.get('pid')) is not int or worker['pid']!=final[i]['pid'] or
                type(worker.get('model_decode_count')) is not int or worker['model_decode_count']!=1 or
                worker.get('model_sha256')!=setup['model']['sha256'] or
                type(worker.get('task_count')) is not int or worker['task_count']!=len(assigned) or
                worker.get('gate_closed') is not True or worker.get('gate_poisoned') is not False or
                type(worker.get('full_environment_checks')) is not int or worker['full_environment_checks']!=1 or
                type(worker.get('boundary_seconds')) is not list or len(worker['boundary_seconds'])!=(4 if assigned else 3)):
            raise QualificationDecline('prepared worker gate/model/PID')
        verify_threads(worker.get('blas_threads',[]))
        diagnostics=worker.get('diagnostics')
        if diagnostics is not None:
            if (type(diagnostics) is not list or len(diagnostics)!=len(assigned) or
                    any(type(row) is not list or len(row)!=2 or row[0]!=ordinal or
                        type(row[1]) is not dict or set(row[1])!={'counters','records'} or
                        type(row[1]['counters']) is not dict or type(row[1]['records']) is not dict or
                        any(type(k) is not str or type(v) is not int or v<0 for k,v in row[1]['counters'].items()) or
                        any(type(k) is not str or type(v) is not list for k,v in row[1]['records'].items())
                        for row,ordinal in zip(diagnostics,assigned))):
                raise QualificationDecline('task diagnostic envelope binding/schema')
            pack(diagnostics,'task-diagnostics',limit=payload_limit)
        compact='result_count' in worker or 'results_sha256' in worker
        if compact:
            allowed={'status','session_sha256','request_sha256','worker','worker_setup_sha256',
                'model_sha256','model_decode_count','pid','task_count','batch_size','assignment_seconds',
                'startup_seconds','task_seconds','boundary_seconds','reset_count','full_environment_checks',
                'blas_threads','gate_closed','gate_poisoned','result_count','results_sha256','diagnostics'}
            if 'resource_identity' in setup:allowed.add('resource_guard')
            if 'value_codec' in setup:allowed.add('value_codec')
            values=[[ordinal,results[ordinal]] for ordinal in assigned]
            if (set(worker)-allowed or type(worker.get('result_count')) is not int or
                    worker['result_count']!=len(assigned) or not _digest(worker.get('results_sha256')) or
                    worker['results_sha256']!=pack(values,'worker-results',limit=payload_limit)['sha256']):
                raise QualificationDecline('prepared worker result digest/count binding')
        else:
            values=worker.get('rows');_validate_rows(values,assigned)
        rows.extend(values)
    if canonical(list(reduce_results(rows,len(tasks))))!=canonical(report.get('results')):
        raise QualificationDecline('prepared result rows differ')
