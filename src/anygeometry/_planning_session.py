"""Private, finite per-plan transport. Stage replies are never public plans.

One original runtime baseline, a new closed gate for each phase, and an explicit
finish transaction. No process or poisoned gate is reused by another plan.
"""
import hashlib
import json
import os
from pathlib import Path
import time

from threadpoolctl import threadpool_info
from ._planning_executor import (OwnedPlanner, PlannerCancelled, PlannerDeadline,
    PlannerCleanupError, pack, read_packet, write_packet, check_result_status,
    source_files, verify_sources, verify_threads, reduce_results, thaw_model,
    assignment_size)
from ._planning_qualification import QualificationDecline
from . import _planning_qualification as q
from . import _planning_ledger as ledger_api
from ._planning_batch import BatchWorkerGate
from ._planning_stages import WorkerModel, stage_task, dispatch
from ._planning_process import OwnedTree


def _digest(value):
    return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)


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


def _wait_file(path,deadline,*,poll=None,child=None,folder=None,result_kind=None):
    while not path.exists():
        if poll is not None:poll()
        if time.monotonic()>=deadline:raise PlannerDeadline('session deadline consumed')
        if child is not None and child.poll() is not None:
            if path.exists():break
            if folder is not None and (folder/'result.json').exists():
                check_result_status(read_packet(folder/'result.json',result_kind))
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
    def __init__(self,directory,**kwargs):
        super().__init__(directory,_role='session-planner',**kwargs)
        self.session=None;self.previous=None;self.stages=();self.sources=None
        self.packet=None;self.failed=False;self.finished=False;self._stage_receipts=[]

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
                initial=pack(setup,'session',limit=self.limits['payload_limit'])
                self.session=self.previous=initial['sha256']
                write_packet(self.directory/'session.json',initial)
            if packet!=self.packet:raise QualificationDecline('session model changed')
            seed=sampling_digest(self.packet,tasks,workers=self.workers,batch_size=self.batch_size,
                                 sources=self.sources,payload_limit=self.limits['payload_limit'])
            body,command=self._command(kind,tasks)
            sequence=len(self.stages)
            write_packet(_path(self.directory,'command',sequence),command)
            path=_path(self.directory,'reply',sequence)
            _wait_file(path,self.deadline,poll=lambda:self._poll(poll),child=self.child,
                       folder=self.directory,result_kind='result')
            report=read_packet(path,'session-reply',limit=self.limits['payload_limit'])
            _reply_binding(report,self.session,command['sha256'])
            if report.get('stage')!=kind:raise QualificationDecline('session reply stage binding')
            verify_sources(self.sources)
            report['results']=reduce_results(report.pop('rows'),len(body['tasks']))
            report['sampling_sha256']=seed
            report['host_timing']=dict(start_to_return_seconds=time.perf_counter()-self.started)
            self.stages+=kind,
            self.previous=command['sha256'];self._stage_receipts.append(command['sha256'])
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
            seen=();previous=self.session;requests=[]
            for i,kind in enumerate([*stages,'finish']):
                body,request=read_packet(_path(self.directory,'command',i),'session-command',with_digest=True,limit=self.limits['payload_limit'])
                if validate_command(body,session=self.session,sequence=i,previous=previous,seen=seen)!=kind:
                    raise QualificationDecline('preparation command order')
                if kind!='finish':
                    entry=entries[i]
                    if type(entry) is not dict or set(entry)!={'kind','tasks','candidate'} or entry['kind']!=kind:
                        raise QualificationDecline('preparation entry schema')
                    if pack(entry['tasks'],'tasks',limit=self.limits['payload_limit'])!=pack(body['tasks'],'tasks',limit=self.limits['payload_limit']):
                        raise QualificationDecline('preparation task binding')
                    candidate=entry['candidate'];_reply_binding(candidate,self.session,request)
                    if 'preparation_samples' not in candidate:raise QualificationDecline('preparation samples missing')
                    seen+=kind,;requests.append(request)
                previous=request
            _reply_binding(result,self.session,previous)
            if result.get('stage_requests')!=requests:raise QualificationDecline('preparation request coverage')
            reports=result.get('worker_reports')
            if type(reports) is not list or len(reports)!=self.workers:raise QualificationDecline('preparation worker coverage')
            for i,report in enumerate(reports):
                if (type(report.get('worker')) is not int or report['worker']!=i or report.get('stages')!=stages or
                        type(report.get('gate_closures')) is not int or report['gate_closures']!=len(stages) or
                        report.get('model_sha256')!=packet['sha256'] or type(report.get('model_decode_count')) is not int or
                        report['model_decode_count']!=1 or type(report.get('pid')) is not int or report['pid']<=0):
                    raise QualificationDecline('preparation finish worker binding')
                verify_threads(report.get('blas_threads',[]))
            for entry in entries:
                self._poll(poll)
                verify_prepared_stage(entry['candidate'],entry['tasks'],setup,reports,payload_limit=self.limits['payload_limit'])
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
            result=read_packet(path,'result',limit=self.limits['payload_limit'])
            _reply_binding(result,self.session,command['sha256'])
            if result.get('stage_requests')!=self._stage_receipts or result.get('stages')!=list(self.stages):
                raise QualificationDecline('session finish stage coverage')
            verify_sources(self.sources)
            self.finished=True
        except BaseException:
            self.failed=True
            raise


def _setup(folder,limits,deadline,*,decode=True):
    path=folder/'session.json';_wait_file(path,deadline)
    setup,digest=read_packet(path,'session',with_digest=True,limit=limits['payload_limit'])
    if (type(setup) is not dict or set(setup)!={'model','sources','workers','batch_size','limits'} or
            setup['limits']!=limits or type(setup['workers']) is not int or setup['workers'] not in (1,2,8,16)):
        raise QualificationDecline('session setup schema')
    assignment_size(setup['batch_size'],0)
    verify_sources(setup['sources'])
    if decode:thaw_model(setup['model'],**limits)
    return setup,digest


def session_planner(folder,deadline,canaries,goldens,binding,limits,generated):
    setup,session=_setup(folder,limits,deadline)
    binding['session_sha256']=session
    ledger=ledger_api.loaded_geometry_ledger(trusted_image=True)
    ledger.prepare(owned_worker=True)
    expected=q.capture_environment(generated_modules=generated,declined_stores=ledger.resets,trusted_image=True)
    expected['live_code_pins']=ledger.live_code_pins()
    write_packet(folder/'reference.json',pack(expected,'reference'))
    tree=OwnedTree(folder);workers=[];seen=();previous=session;stage_requests=[]
    try:
        for index in range(setup['workers']):
            location=folder/('worker-'+str(index));location.mkdir()
            initial=pack(dict(session_sha256=session,worker=index,model=setup['model'],sources=setup['sources'],
                              expected=expected,workers=setup['workers'],batch_size=setup['batch_size']),
                         'session-worker',limit=limits['payload_limit'])
            write_packet(location/'limits.json',pack(limits,'limits'))
            write_packet(location/'session.json',initial)
            child=tree.spawn('session-worker',location,max(.1,deadline-time.monotonic()-5))
            workers.append(dict(child=child,folder=location,setup=initial['sha256'],previous=initial['sha256']))
        def execute(body,request):
            nonlocal seen,previous
            binding['request_sha256']=request
            kind=validate_command(body,session=session,sequence=len(seen),previous=previous,seen=seen)
            verify_sources(setup['sources'])
            parts,layout=([],[]) if kind=='finish' else dispatch(body['tasks'],setup['workers'])
            sent=[]
            for index,item in enumerate(workers):
                assigned=[] if index>=len(parts) else [[i,body['tasks'][i]] for i in parts[index]]
                command=dict(session=session,sequence=len(seen),previous=item['previous'],stage=kind,
                             tasks=assigned,worker=index,parent_request=request)
                packet=pack(command,'session-worker-command',limit=limits['payload_limit'])
                write_packet(_path(item['folder'],'command',len(seen)),packet)
                sent.append(packet['sha256'])
            reports=[]
            for index,item in enumerate(workers):
                if kind=='finish':
                    tree.wait(item['child'],deadline-3)
                    report=read_packet(item['folder']/'result.json','worker-result',limit=limits['payload_limit'])
                else:
                    response=_path(item['folder'],'reply',len(seen))
                    _wait_file(response,deadline-3,child=item['child'],folder=item['folder'],result_kind='worker-result')
                    report=read_packet(response,'session-worker-reply',limit=limits['payload_limit'])
                _reply_binding(report,session,sent[index])
                if (report.get('worker_setup_sha256')!=item['setup'] or type(report.get('worker')) is not int or
                        report['worker']!=index or report.get('model_sha256')!=setup['model']['sha256']):
                    raise QualificationDecline('session worker/model binding')
                verify_threads(report.get('blas_threads',[]))
                if (type(report.get('pid')) is not int or report['pid']!=item['child'].pid or
                        type(report.get('model_decode_count')) is not int or report['model_decode_count']!=1):
                    raise QualificationDecline('worker PID/decode binding')
                if kind=='finish':
                    if report.get('stages')!=list(seen) or type(report.get('gate_closures')) is not int or report['gate_closures']!=len(seen):
                        raise QualificationDecline('worker finish gate coverage')
                elif report.get('gate_closed') is not True or report.get('gate_poisoned') is not False:
                    raise QualificationDecline('session stage gate not closed')
                reports.append(report);item['previous']=sent[index]
            if kind=='finish':return dict(worker_reports=reports,request_sha256=request)
            rows=[row for report in reports for row in report['rows']]
            reduce_results(rows,len(body['tasks']))
            verify_sources(setup['sources'])
            report=dict(status='passed',session_sha256=session,request_sha256=request,stage=kind,rows=rows,
                        worker_reports=reports,requested_workers=setup['workers'],actual_workers=len(parts),
                        task_layout=layout,model_sha256=setup['model']['sha256'])
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
                    entries.append(dict(kind=kind,tasks=list(tasks),candidate=report))
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
        try:tree.close()
        except Exception as exc:raise PlannerCleanupError('session worker cleanup failed') from exc
        (folder/'cleanup.json').replace(folder/'workers-cleanup.json')
    verify_sources(setup['sources'])
    return dict(status='passed',session_sha256=session,request_sha256=request,stages=list(seen),
                stage_requests=stage_requests,worker_reports=reports,prepared_entries=entries,preparation_sha256=plan_sha)


def session_worker(folder,deadline,canaries,goldens,binding,limits,generated):
    setup,setup_sha=read_packet(folder/'session.json','session-worker',with_digest=True,limit=limits['payload_limit'])
    if type(setup) is not dict or set(setup)!={'session_sha256','worker','model','sources','expected','workers','batch_size'}:
        raise QualificationDecline('worker session setup schema')
    root,session=_setup(folder.parent,limits,deadline,decode=False)
    index=setup['worker']
    if (setup['session_sha256']!=session or type(index) is not int or not 0<=index<root['workers'] or
            setup['workers']!=root['workers'] or setup['batch_size']!=root['batch_size'] or
            setup['sources']!=root['sources'] or setup['model']!=root['model']):
        raise QualificationDecline('worker session setup binding')
    binding.update(session_sha256=session,worker=index,worker_setup_sha256=setup_sha)
    context=WorkerModel(setup['model'],**limits)
    ledger=ledger_api.loaded_geometry_ledger(trusted_image=True)
    expected=setup['expected']
    seen=();previous=setup_sha;parent_previous=session;gate=None
    while True:
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
        if kind=='finish':
            if gate is None or not gate.closed or gate.poisoned:raise QualificationDecline('unfinished session gate')
            gate._boundary();context.verify();verify_threads(threadpool_info())
            return dict(status='passed',**binding,model_sha256=context.verify(),stages=list(seen),gate_closures=len(seen),
                        blas_threads=threadpool_info(),pid=os.getpid(),model_decode_count=1)
        gate=BatchWorkerGate(expected,stores=ledger.resets,owned_worker=True,ledger=ledger,
            generated_modules=generated,batch_size=assignment_size(setup['batch_size'],len(assigned)),coalesce_boundaries=True)
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
                    gate_closed=gate.closed,gate_poisoned=gate.poisoned)
        write_packet(_path(folder,'reply',len(seen)),pack(report,'session-worker-reply',limit=limits['payload_limit']))
        seen+=kind,;previous=request;parent_previous=parent_request


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
    rows=[]
    for i,worker in enumerate(report['worker_reports']):
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
        values=worker.get('rows')
        if (type(values) is not list or any(type(row) is not list or len(row)!=2 or type(row[0]) is not int or type(row[1]) is not str for row in values) or
                [row[0] for row in values]!=assigned):
            raise QualificationDecline('prepared worker assignment coverage')
        rows.extend(values)
    if canonical(list(reduce_results(rows,len(tasks))))!=canonical(report.get('results')):
        raise QualificationDecline('prepared result rows differ')
