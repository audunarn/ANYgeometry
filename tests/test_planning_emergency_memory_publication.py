"""Failure-publication components only; no actual memory/capacity qualification."""
from contextlib import nullcontext
import json
from pathlib import Path
import pytest
from anygeometry import _planning_child as child,_planning_session as session
from anygeometry._planning_executor import pack,read_packet,PlannerSourceChanged
from anygeometry._planning_qualification import QualificationDecline

def binding():return dict(session_sha256='a'*64,worker=0,worker_setup_sha256='b'*64,request_sha256='c'*64)

def test_armed_packet_is_current_bound_and_same_protocol(tmp_path):
    state=[None];bound=binding()
    session._arm_memory_failure_reply(tmp_path,bound,2,state,payload_limit=4_000_000)
    assert child._publish_memory_failure(MemoryError(),state,bound) is True
    report=read_packet(tmp_path/'result.json','worker-result')
    assert report==dict(bound,status='declined',reason='MemoryError: ',sequence=2,memory_failure='armed-command-v1')
    assert state==[None]

@pytest.mark.parametrize('fault',['unarmed','stale','unknown_primary','message'])
def test_emergency_never_reclassifies_unknown_or_stale(tmp_path,fault):
    state=[None];bound=binding()
    if fault!='unarmed':session._arm_memory_failure_reply(tmp_path,bound,0,state,payload_limit=4_000_000)
    if fault=='stale':bound['request_sha256']='d'*64
    error=ValueError('primary') if fault=='unknown_primary' else MemoryError('detail') if fault=='message' else MemoryError()
    assert child._publish_memory_failure(error,state,bound) is False
    assert not (tmp_path/'result.json').exists()

def test_replace_failure_remains_failure(tmp_path,monkeypatch):
    state=[None];bound=binding();session._arm_memory_failure_reply(tmp_path,bound,0,state,payload_limit=4_000_000)
    original=OSError('replace failed')
    def fail(*args):raise original
    monkeypatch.setattr(child.os,'replace',fail)
    with pytest.raises(OSError) as caught:child._publish_memory_failure(MemoryError(),state,bound)
    assert caught.value is original and not (tmp_path/'result.json').exists()

@pytest.mark.parametrize('fault',['valid','sequence','bool','session','request','worker','setup','marker','missing'])
def test_parent_resource_boundary_requires_current_binding(fault):
    report=dict(binding(),status='declined',reason='MemoryError: ',sequence=2,memory_failure='armed-command-v1')
    if fault=='sequence':report['sequence']=1
    elif fault=='bool':report['sequence']=True
    elif fault=='session':report['session_sha256']='d'*64
    elif fault=='request':report['request_sha256']='d'*64
    elif fault=='worker':report['worker']=1
    elif fault=='setup':report['worker_setup_sha256']='d'*64
    elif fault=='marker':report['memory_failure']='unknown'
    elif fault=='missing':report.pop('sequence')
    if fault=='valid':session._check_worker_memory_failure(report,'a'*64,'c'*64,0,'b'*64,2)
    else:
        with pytest.raises(QualificationDecline,match='memory failure binding'):
            session._check_worker_memory_failure(report,'a'*64,'c'*64,0,'b'*64,2)

def test_nonresource_status_keeps_original_terminal_order():
    report=dict(status='source_changed',reason='original source')
    session._check_worker_memory_failure(report,'a'*64,'c'*64,0,'b'*64,2)
    with pytest.raises(PlannerSourceChanged,match='original source'):session._reply_binding(report,'a'*64,'c'*64)

def test_arming_packet_size_failure_does_not_leave_authority(tmp_path):
    state=[None]
    with pytest.raises(QualificationDecline):session._arm_memory_failure_reply(tmp_path,binding(),0,state,payload_limit=1)
    assert state==[None]


def fake_main(monkeypatch,tmp_path,primary):
    monkeypatch.setattr(child.sys,'argv',['child','session-worker',str(tmp_path),'5'])
    monkeypatch.setattr(child,'threadpool_limits',lambda **kw:nullcontext())
    monkeypatch.setattr(child,'prepare',lambda **kw:((),()))
    monkeypatch.setattr(child.q,'generated_runtime_modules',lambda **kw:())
    def worker(folder,deadline,canaries,goldens,bound,limits,generated,emergency=None):
        bound.update(binding())
        if emergency is not None:session._arm_memory_failure_reply(folder,bound,2,emergency,payload_limit=limits['payload_limit'])
        def no_allocation(*args,**kwargs):raise MemoryError()
        monkeypatch.setattr(child,'write_packet',no_allocation)
        raise primary
    monkeypatch.setattr(session,'session_worker',worker)

def test_actual_main_publishes_primary_memoryerror_without_json_textio(monkeypatch,tmp_path):
    fake_main(monkeypatch,tmp_path,MemoryError())
    child.main()
    report=read_packet(tmp_path/'result.json','worker-result')
    assert report['reason']=='MemoryError: ' and report['sequence']==2
    assert report['memory_failure']=='armed-command-v1'

def test_other_primary_secondary_memoryerror_never_uses_emergency(monkeypatch,tmp_path):
    fake_main(monkeypatch,tmp_path,PlannerSourceChanged('original source'))
    with pytest.raises(MemoryError):child.main()
    assert not (tmp_path/'result.json').exists()

def test_next_command_read_disarms_prior_capsule(monkeypatch,tmp_path):
    from types import SimpleNamespace
    root=dict(workers=2,batch_size=1,sources={},model={})
    setup=dict(root,session_sha256='a'*64,worker=0,expected={})
    monkeypatch.setattr(session,'read_packet',lambda *a,**kw:(setup,'b'*64))
    monkeypatch.setattr(session,'_setup',lambda *a,**kw:(root,'a'*64))
    monkeypatch.setattr(session,'WorkerModel',lambda *a,**kw:SimpleNamespace())
    monkeypatch.setattr(session.ledger_api,'loaded_geometry_ledger',lambda **kw:SimpleNamespace())
    monkeypatch.setattr(session.q,'generated_runtime_modules',lambda **kw:())
    state=[('stale',)]
    class Stop(Exception):pass
    def wait(*args,**kwargs):
        assert state==[None]
        raise Stop()
    monkeypatch.setattr(session,'_wait_file',wait)
    with pytest.raises(Stop):session.session_worker(tmp_path,999,(),(),{},dict(payload_limit=4_000_000),(),emergency=state)

@pytest.mark.parametrize('stale',[False,True])
def test_actual_dead_worker_wait_validates_resource_binding_first(tmp_path,stale):
    from types import SimpleNamespace
    import time
    bound=binding();state=[None]
    session._arm_memory_failure_reply(tmp_path,bound,2,state,payload_limit=4_000_000)
    assert child._publish_memory_failure(MemoryError(),state,bound)
    request='d'*64 if stale else 'c'*64
    with pytest.raises(QualificationDecline) as caught:
        session._wait_file(tmp_path/'reply-2.json',time.monotonic()+5,
            child=SimpleNamespace(poll=lambda:0),folder=tmp_path,result_kind='worker-result',
            memory_binding=('a'*64,request,0,'b'*64,2))
    assert str(caught.value)==('worker memory failure binding' if stale else 'MemoryError: ')


def test_generic_exit_one_is_never_resource_success(tmp_path):
    from types import SimpleNamespace
    import time
    with pytest.raises(QualificationDecline,match='session child ended before reply'):
        session._wait_file(tmp_path/'reply-2.json',time.monotonic()+5,
            child=SimpleNamespace(poll=lambda:1),folder=tmp_path,result_kind='worker-result',
            memory_binding=('a'*64,'c'*64,0,'b'*64,2))

@pytest.mark.parametrize('sequence',[True,-1,3])
def test_invalid_arming_sequence_never_creates_capsule(tmp_path,sequence):
    state=[None]
    with pytest.raises(QualificationDecline,match='memory failure binding'):
        session._arm_memory_failure_reply(tmp_path,binding(),sequence,state,payload_limit=4_000_000)
    assert state==[None] and not (tmp_path/'memory-error.pending.json').exists()

@pytest.mark.parametrize('failure',['command','source','gate'])
def test_actual_worker_arms_only_after_command_dispatch_and_source_checks(tmp_path,monkeypatch,failure):
    from types import SimpleNamespace
    import time
    folder=tmp_path/'worker-0';folder.mkdir()
    root=dict(workers=2,batch_size=1,sources={},model={})
    setup=dict(root,session_sha256='a'*64,worker=0,expected={})
    initial=pack(setup,'session-worker');session.write_packet(folder/'session.json',initial)
    tasks=[dict(kind='pair',cost=1.)]
    parent=pack(dict(session='a'*64,sequence=0,previous='a'*64,stage='pair',tasks=tasks),'session-command')
    session.write_packet(tmp_path/'command-0.json',parent)
    command=dict(session='a'*64,sequence=0,previous=initial['sha256'],stage='pair',
                 tasks=[[0,tasks[0]]],worker=0,parent_request=parent['sha256'])
    if failure=='command':command['sequence']=True
    session.write_packet(folder/'command-0.json',pack(command,'session-worker-command'))
    monkeypatch.setattr(session,'_setup',lambda *a,**kw:(root,'a'*64))
    monkeypatch.setattr(session,'WorkerModel',lambda *a,**kw:SimpleNamespace())
    monkeypatch.setattr(session.ledger_api,'loaded_geometry_ledger',lambda **kw:SimpleNamespace(resets={}))
    monkeypatch.setattr(session.q,'generated_runtime_modules',lambda **kw:())
    def sources(*args):
        if failure=='source':raise MemoryError()
    monkeypatch.setattr(session,'verify_sources',sources)
    def gate(*args,**kw):raise MemoryError()
    monkeypatch.setattr(session,'BatchWorkerGate',gate)
    state=[None];bound={}
    with pytest.raises(QualificationDecline if failure=='command' else MemoryError):
        session.session_worker(folder,time.monotonic()+5,(),(),bound,
            dict(payload_limit=4_000_000,walk_limit=100000),(),emergency=state)
    assert (state[0] is not None)==(failure=='gate')
    if failure=='gate':
        assert child._publish_memory_failure(MemoryError(),state,bound)
        report=read_packet(folder/'result.json','worker-result')
        assert report['sequence']==0 and report['request_sha256']==bound['request_sha256']


def test_successful_main_packet_has_no_emergency_fields(monkeypatch,tmp_path):
    monkeypatch.setattr(child.sys,'argv',['child','session-worker',str(tmp_path),'5'])
    monkeypatch.setattr(child,'threadpool_limits',lambda **kw:nullcontext())
    monkeypatch.setattr(child,'prepare',lambda **kw:((),()))
    monkeypatch.setattr(child.q,'generated_runtime_modules',lambda **kw:())
    monkeypatch.setattr(session,'session_worker',lambda *a,**kw:dict(original='same'))
    child.main()
    report=read_packet(tmp_path/'result.json','worker-result')
    assert type(report.pop('process_seconds')) is float
    assert report==dict(original='same',status='passed')

@pytest.mark.parametrize('role',['planner','worker','session-planner'])
def test_other_roles_keep_legacy_primary_memoryerror_packet(monkeypatch,tmp_path,role):
    monkeypatch.setattr(child.sys,'argv',['child',role,str(tmp_path),'5'])
    monkeypatch.setattr(child,'threadpool_limits',lambda **kw:nullcontext())
    monkeypatch.setattr(child,'prepare',lambda **kw:((),()))
    monkeypatch.setattr(child.q,'generated_runtime_modules',lambda **kw:())
    def fail(*args,**kwargs):raise MemoryError()
    monkeypatch.setattr(session if role=='session-planner' else child,
                        role.replace('-','_'),fail)
    child.main()
    report=read_packet(tmp_path/'result.json','result' if role in ('planner','session-planner') else 'worker-result')
    assert report['status']=='declined' and report['reason']=='MemoryError: '
    assert type(report['process_seconds']) is float
    assert 'memory_failure' not in report and 'sequence' not in report


def test_unarmed_session_worker_primary_memoryerror_remains_failure(monkeypatch,tmp_path):
    monkeypatch.setattr(child.sys,'argv',['child','session-worker',str(tmp_path),'5'])
    monkeypatch.setattr(child,'threadpool_limits',lambda **kw:nullcontext())
    monkeypatch.setattr(child,'prepare',lambda **kw:((),()))
    monkeypatch.setattr(child.q,'generated_runtime_modules',lambda **kw:())
    def fail(*args,**kwargs):raise MemoryError()
    monkeypatch.setattr(session,'session_worker',fail)
    with pytest.raises(MemoryError):child.main()
    assert not (tmp_path/'result.json').exists()

@pytest.mark.parametrize('sequence',[True,-1,3])
def test_malformed_capsule_sequence_never_publishes(tmp_path,sequence):
    state=[None];bound=binding()
    session._arm_memory_failure_reply(tmp_path,bound,2,state,payload_limit=4_000_000)
    state[0]=state[0][:6]+(sequence,)
    assert child._publish_memory_failure(MemoryError(),state,bound) is False
    assert not (tmp_path/'result.json').exists()
