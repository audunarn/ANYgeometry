"""PP5 A3-A5 regressions: terminal deadlines and checked transport boundaries."""
import hashlib,json,subprocess,time
from pathlib import Path
import pytest
import psutil
from anygeometry import _planning_executor as a
from anygeometry._planning_qualification import QualificationDecline

@pytest.mark.parametrize('literal',['1e400','-1e400'])
def test_overflow_json_number_refuses(literal,tmp_path):
    text='{"x":'+literal+'}'
    p=a.pack({},'model');p.update(payload=text,sha256=hashlib.sha256(text.encode()).hexdigest())
    with pytest.raises(QualificationDecline,match='nonfinite'):
        a.unpack(p,'model')


def test_packet_binding_is_from_one_verified_read(tmp_path,monkeypatch):
    path=tmp_path/'packet.json';packet=a.pack({'x':1},'model');a.write_packet(path,packet)
    original=Path.read_text;reads=[]
    def read(self,*args,**kwargs):
        reads.append(str(self));return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    body,digest=a.read_packet(path,'model',with_digest=True)
    assert body=={'x':1} and digest==packet['sha256'] and len(reads)==1


def test_owned_wait_deadline_is_terminal():
    from anygeometry._planning_process import OwnedTree
    class Child:
        def wait(self,timeout):raise subprocess.TimeoutExpired('owned-test',timeout)
    with pytest.raises(a.PlannerDeadline):
        OwnedTree.wait(object(),Child(),time.monotonic())

@pytest.mark.parametrize('error',[OSError('spawn'),FileExistsError('directory'),psutil.AccessDenied()])
def test_infrastructure_failure_has_typed_serial_reason(error,tmp_path,monkeypatch):
    events=[]
    class Planner:
        def __init__(self,*args,**kwargs):pass
        def submit(self,*args):raise error
        def close(self):events.append('close')
    monkeypatch.setattr(a,'OwnedPlanner',Planner)
    result=a.execute(None,('pair',),lambda _:events.append('serial') or 'value',directory=tmp_path/'p',opt_in=True)
    assert events==['close','serial'] and result['mode']=='serial'
    assert type(error).__name__ in result['decline_reason']


def test_cleanup_oserror_cannot_grant_serial_replay(tmp_path,monkeypatch):
    class Planner:
        def __init__(self,*args,**kwargs):pass
        def submit(self,*args):raise QualificationDecline('candidate')
        def close(self):raise OSError('cleanup')
    monkeypatch.setattr(a,'OwnedPlanner',Planner)
    with pytest.raises(OSError,match='cleanup'):
        a.execute(None,('pair',),lambda _:pytest.fail('cleanup replay'),directory=tmp_path/'p',opt_in=True)

@pytest.mark.parametrize('status,exception',[('deadline',a.PlannerDeadline),('cancelled',a.PlannerCancelled)])
def test_terminal_result_status_not_replayed(status,exception):
    with pytest.raises(exception):a.check_result_status({'status':status,'reason':'consumed'})

@pytest.mark.parametrize('result',[None,[],1,{'status':'declined','reason':'refused'}])
def test_bad_result_refuses_typed(result):
    with pytest.raises(QualificationDecline):a.check_result_status(result)


def test_poll_exit_race_rechecks_result(tmp_path,monkeypatch):
    class Tree:
        def wait(self,*args):pass
    class Child:
        def poll(self):
            request=a.read_packet(tmp_path/'request.json','request')
            original=json.loads((tmp_path/'request.json').read_text())
            a.write_packet(tmp_path/'result.json',a.pack(dict(status='passed',request_sha256=original['sha256'],rows=[[0,'done']]),'result'))
            return 0
    planner=object.__new__(a.OwnedPlanner)
    planner.closed=False;planner.used=False;planner.directory=tmp_path;planner.workers=1;planner.batch_size=None
    planner.started=time.perf_counter();planner.deadline=time.monotonic()+5;planner.child=Child();planner.tree=Tree()
    monkeypatch.setattr(a,'freeze_model',lambda _,**limits:a.pack({},'model'))
    monkeypatch.setattr(a,'source_files',lambda :{})
    assert planner.submit(None,('pair',))['results']==('done',)

def test_constructor_cleanup_failure_cannot_grant_replay(tmp_path,monkeypatch):
    from anygeometry import _planning_process as process
    class Tree:
        def __init__(self,*args):pass
        def spawn(self,*args):raise OSError('spawn')
        def close(self):raise OSError('cleanup')
    monkeypatch.setattr(process,'OwnedTree',Tree)
    with pytest.raises(a.PlannerCleanupError,match='cleanup'):
        a.execute(None,('pair',),lambda _:pytest.fail('constructor cleanup replay'),directory=tmp_path/'p',opt_in=True)

def test_task_json_float_guard():
    from anygeometry._planning_qualification import decode_task_payload
    payload=b'{"args":[0.5],"kwargs":{}}'
    assert decode_task_payload(payload,hashlib.sha256(payload).hexdigest())==((0.5,),{})
    payload=b'{"args":[1e400],"kwargs":{}}'
    with pytest.raises(QualificationDecline):decode_task_payload(payload,hashlib.sha256(payload).hexdigest())

@pytest.mark.parametrize('rows',[None,1,{}])
def test_nonsequence_rows_refuse_typed(rows):
    with pytest.raises(QualificationDecline):a.reduce_results(rows,1)
