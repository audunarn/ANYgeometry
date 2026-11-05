"""Failure-only setup publication; pure fault injection, no geometry/capacity claim."""
import time
from types import SimpleNamespace as NS
import pytest
from anygeometry import _planning_session as s,_planning_child as c
from anygeometry._planning_executor import read_packet,pack
from anygeometry._planning_qualification import QualificationDecline

def binding():return dict(session_sha256='a'*64,worker=0,worker_setup_sha256='b'*64)

def arm(folder,bound=None):
    state=[None];s._arm_setup_memory_failure_reply(folder,binding() if bound is None else bound,state,payload_limit=4_000_000)
    return state

def report():return dict(binding(),status='declined',reason='MemoryError: ',memory_failure='armed-setup-v1')

def test_setup_capsule_has_no_command_claim_and_consumes_once(tmp_path):
    state=arm(tmp_path);assert c._publish_memory_failure(MemoryError(),state,binding())
    actual=read_packet(tmp_path/'result.json','worker-result');assert actual==report()
    assert 'request_sha256' not in actual and 'sequence' not in actual
    assert state==[None] and not c._publish_memory_failure(MemoryError(),state,binding())

@pytest.mark.parametrize('fault',['session','worker','setup','request','sequence','marker','boolworker','status','reason'])
def test_parent_setup_schema_is_exact(fault):
    data=report()
    key={'session':'session_sha256','worker':'worker','setup':'worker_setup_sha256','request':'request_sha256','sequence':'sequence','marker':'memory_failure','boolworker':'worker','status':'status','reason':'reason'}[fault]
    data[key]={'session':'d'*64,'worker':1,'setup':'d'*64,'request':'c'*64,'sequence':0,'marker':'unknown','boolworker':False,'status':'passed','reason':'ValueError: original'}[fault]
    with pytest.raises(QualificationDecline,match='memory failure binding'):
        s._check_worker_memory_failure(data,'a'*64,'c'*64,0,'b'*64,0,admitted=False)

@pytest.mark.parametrize('sequence,admitted',[(0,True),(1,False),(2,False),(True,False)])
def test_setup_failure_is_only_pre_first_admission(sequence,admitted):
    with pytest.raises(QualificationDecline,match='memory failure binding'):
        s._check_worker_memory_failure(report(),'a'*64,'c'*64,0,'b'*64,sequence,admitted=admitted)


def test_parent_accepts_only_pre_admission_setup_binding():
    s._check_worker_memory_failure(report(),'a'*64,'c'*64,0,'b'*64,0,admitted=False)

@pytest.mark.parametrize('fault',['stale','nonmemory','detailed','rename'])
def test_setup_publisher_keeps_other_or_failed_publication_terminal(tmp_path,monkeypatch,fault):
    state=arm(tmp_path);bound=binding()
    if fault=='stale':bound['worker_setup_sha256']='d'*64
    error=ValueError('original') if fault=='nonmemory' else MemoryError('detail') if fault=='detailed' else MemoryError()
    if fault=='rename':
        original=OSError('rename failed')
        def fail(*a):raise original
        monkeypatch.setattr(c.os,'replace',fail)
        with pytest.raises(OSError) as caught:c._publish_memory_failure(error,state,bound)
        assert caught.value is original
    else:assert c._publish_memory_failure(error,state,bound) is False
    assert not (tmp_path/'result.json').exists()


def test_real_worker_model_allocation_failure_has_setup_capsule(tmp_path,monkeypatch):
    root=dict(workers=2,batch_size=1,sources={},model={});setup=dict(root,session_sha256='a'*64,worker=0,expected={})
    monkeypatch.setattr(s,'read_packet',lambda *a,**kw:(setup,'b'*64))
    monkeypatch.setattr(s,'_setup',lambda *a,**kw:(root,'a'*64))
    primary=MemoryError()
    def allocate(*a,**kw):raise primary
    monkeypatch.setattr(s,'WorkerModel',allocate)
    state=[None];bound={}
    with pytest.raises(MemoryError) as caught:s.session_worker(tmp_path,time.monotonic()+5,(),(),bound,dict(payload_limit=4_000_000,walk_limit=100000),(),emergency=state)
    assert caught.value is primary and c._publish_memory_failure(caught.value,state,bound)
    assert read_packet(tmp_path/'result.json','worker-result')==report()


def test_failed_prewrite_is_not_armed(tmp_path,monkeypatch):
    state=[None];original=MemoryError()
    def fail(*a,**kw):raise original
    monkeypatch.setattr(s,'write_packet',fail)
    with pytest.raises(MemoryError) as caught:s._arm_setup_memory_failure_reply(tmp_path,binding(),state,payload_limit=4_000_000)
    assert caught.value is original and state==[None]


@pytest.mark.parametrize('admitted,sequence,expected',[(False,0,'MemoryError: '),(True,0,'worker memory failure binding'),(False,1,'worker memory failure binding')])
def test_actual_wait_requires_setup_phase_before_resource_status(tmp_path,admitted,sequence,expected):
    state=arm(tmp_path);assert c._publish_memory_failure(MemoryError(),state,binding())
    with pytest.raises(QualificationDecline) as caught:
        s._wait_file(tmp_path/'admitted-0.json',time.monotonic()+5,child=NS(poll=lambda:0),folder=tmp_path,result_kind='worker-result',memory_binding=('a'*64,'c'*64,0,'b'*64,sequence,admitted))
    assert str(caught.value)==expected


def test_setup_capsule_rejects_later_request_binding(tmp_path):
    state=arm(tmp_path);bound=dict(binding(),request_sha256='c'*64)
    assert c._publish_memory_failure(MemoryError(),state,bound) is False
    assert not (tmp_path/'result.json').exists()


def test_setup_prewrite_retains_original_packet_bound(tmp_path):
    state=[None]
    with pytest.raises(QualificationDecline):s._arm_setup_memory_failure_reply(tmp_path,binding(),state,payload_limit=1)
    assert state==[None]

@pytest.mark.parametrize('resource',[False,True])
def test_actual_main_primary_vs_secondary_oom_with_setup_capsule(tmp_path,monkeypatch,resource):
    from contextlib import nullcontext
    monkeypatch.setattr(c.sys,'argv',['child','session-worker',str(tmp_path),'5'])
    monkeypatch.setattr(c,'threadpool_limits',lambda **kw:nullcontext())
    monkeypatch.setattr(c,'prepare',lambda **kw:((),()))
    monkeypatch.setattr(c.q,'generated_runtime_modules',lambda **kw:())
    def worker(folder,deadline,canaries,goldens,bound,limits,generated,emergency=None):
        bound.update(binding());s._arm_setup_memory_failure_reply(folder,bound,emergency,payload_limit=limits['payload_limit'])
        def allocation_failure(*a,**kw):raise MemoryError()
        monkeypatch.setattr(c,'write_packet',allocation_failure)
        if resource:raise MemoryError()
        raise ValueError('original')
    monkeypatch.setattr(s,'session_worker',worker)
    if resource:
        c.main();assert read_packet(tmp_path/'result.json','worker-result')==report()
    else:
        with pytest.raises(MemoryError):c.main()
        assert not (tmp_path/'result.json').exists()


def test_mixed_setup_marker_never_masks_original_terminal_status():
    from anygeometry._planning_executor import PlannerSourceChanged
    data=dict(report(),status='source_changed',reason='original source')
    s._check_worker_memory_failure(data,'a'*64,'c'*64,0,'b'*64,0,admitted=False)
    with pytest.raises(PlannerSourceChanged,match='original source'):s._reply_binding(data,'a'*64,'c'*64)
