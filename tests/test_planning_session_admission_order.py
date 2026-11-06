"""Exercise planner dispatch order: stage admission cannot overlap across workers."""
from types import SimpleNamespace
import json,time
import pytest
from anygeometry import _planning_session as s

class RepliesReached(Exception):pass


def run_dispatch(tmp_path,monkeypatch,admission='valid',model='real'):
    events=[];limits={'payload_limit':4_000_000,'walk_limit':100000};session='a'*64
    # The legacy planner decodes the admitted model up front (production thaw_model, unchanged): the fake setup
    # must therefore carry a REAL frozen model packet. 'sha-only' is the old (invalid) shape, kept to prove refusal.
    if model=='real':
        from anygeometry._planning_canaries import _fixture
        from anygeometry._planning_executor import freeze_model
        packet=freeze_model(_fixture()[0])
    else:packet={'sha256':'b'*64}
    setup={'workers':2,'batch_size':1,'model':packet,'sources':{}}
    monkeypatch.setattr(s,'_setup',lambda *a,**k:(setup,session))
    ledger=SimpleNamespace(resets={},trusted_image=True,prepare=lambda **k:None,live_code_pins=lambda:{})
    def loaded_ledger(*,trusted_image):
        assert trusted_image is True
        return ledger
    monkeypatch.setattr(s.ledger_api,'loaded_geometry_ledger',loaded_ledger)
    monkeypatch.setattr(s.q,'capture_environment',lambda **k:{})
    # This dispatch-order unit already declares a fake empty environment.
    # Its generated selection must match that context, not the loaded host.
    declared_generated=()
    def select_generated(reference=None):
        assert reference is None  # P capture; no W reference is used by this unit.
        return declared_generated
    monkeypatch.setattr(s.q,'generated_runtime_modules',select_generated)
    monkeypatch.setattr(s,'verify_sources',lambda *a:None)
    monkeypatch.setattr(s,'dispatch',lambda *a:([[0],[1]],[[0],[1]]))
    class Tree:
        def __init__(self,folder):self.n=0
        def spawn(self,*a):self.n+=1;return SimpleNamespace(pid=self.n)
        def close(self):events.append('close');(tmp_path/'cleanup.json').write_text('{}')
    monkeypatch.setattr(s,'OwnedTree',Tree)
    write=s.write_packet
    def send(path,packet):
        write(path,packet)
        if path.name=='command-0.json' and path.parent.name.startswith('worker-'):
            events.append('send:'+path.parent.name)
    monkeypatch.setattr(s,'write_packet',send)
    command={'session':session,'sequence':0,'previous':session,'stage':'pair','tasks':[{'kind':'pair'},{'kind':'pair'}]}
    write(tmp_path/'command-0.json',s.pack(command,'session-command'))
    def wait(path,*a,**k):
        if path.parent==tmp_path:return
        if path.name=='admitted-0.json':
            events.append('admitted:'+path.parent.name)
            if admission=='deadline':raise s.PlannerDeadline('admission deadline')
            if admission=='child_cleanup':raise s.PlannerCleanupError('admission child cleanup failed')
            body={'stage':'pair','sequence':0} if admission=='valid' else admission
            write(path,s.pack(body,'admitted'));return
        raise RepliesReached()
    monkeypatch.setattr(s,'_wait_file',wait)
    return events,lambda:s.session_planner(tmp_path,time.monotonic()+30,{}, {},{},limits,declared_generated)


def test_legacy_planner_decodes_the_model_before_any_worker_dispatch(tmp_path,monkeypatch):
    events,run=run_dispatch(tmp_path,monkeypatch,model='sha-only')
    with pytest.raises(s.QualificationDecline):run()          # production thaw_model refuses the invalid packet
    assert events==[]                                          # before any tree/worker/admission activity


def test_next_worker_stage_waits_for_previous_admission(tmp_path,monkeypatch):
    events,run=run_dispatch(tmp_path,monkeypatch)
    with pytest.raises(RepliesReached):run()
    assert events==['send:worker-0','admitted:worker-0','send:worker-1','admitted:worker-1','close']


@pytest.mark.parametrize('body',[
    {'stage':'face','sequence':0},{'stage':'pair','sequence':1},
    {'stage':'pair','sequence':False},{'stage':'pair','sequence':0,'extra':1},
    {'stage':'pair'},
])
def test_invalid_admission_refuses_before_next_worker(tmp_path,monkeypatch,body):
    events,run=run_dispatch(tmp_path,monkeypatch,body)
    with pytest.raises(s.QualificationDecline,match='admission'):run()
    assert events==['send:worker-0','admitted:worker-0','close']


@pytest.mark.parametrize('fault,error',[('deadline',s.PlannerDeadline),('child_cleanup',s.PlannerCleanupError)])
def test_admission_terminal_failure_never_dispatches_next_worker(tmp_path,monkeypatch,fault,error):
    events,run=run_dispatch(tmp_path,monkeypatch,fault)
    with pytest.raises(error):run()
    assert events==['send:worker-0','admitted:worker-0','close']
