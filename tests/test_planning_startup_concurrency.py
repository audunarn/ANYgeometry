"""Admitted startup batches are explicit; the legacy default stays serialized."""
import pytest
from anygeometry import _planning_session as session
from anygeometry import _planning_host as host
from anygeometry._planning_qualification import QualificationDecline
from test_planning_session_admission_order import run_dispatch, RepliesReached


def test_admitted_batch_publishes_all_commands_before_admission_waits(tmp_path,monkeypatch):
    events,run=run_dispatch(tmp_path,monkeypatch)
    old=session._setup
    def setup(*args,**kwargs):
        value,digest=old(*args,**kwargs)
        return dict(value,startup_concurrency=2),digest
    monkeypatch.setattr(session,'_setup',setup)
    with pytest.raises(RepliesReached):run()
    assert events==['send:worker-0','send:worker-1','admitted:worker-0','admitted:worker-1','close']


def test_larger_worker_set_respects_admitted_batch_width(tmp_path,monkeypatch):
    events,run=run_dispatch(tmp_path,monkeypatch)
    old=session._setup
    def setup(*args,**kwargs):
        value,digest=old(*args,**kwargs)
        return dict(value,workers=8,startup_concurrency=3),digest
    monkeypatch.setattr(session,'_setup',setup)
    with pytest.raises(RepliesReached):run()
    expected=[]
    for first,last in ((0,3),(3,6),(6,8)):
        expected.extend('send:worker-'+str(i) for i in range(first,last))
        expected.extend('admitted:worker-'+str(i) for i in range(first,last))
    assert events==expected+['close']


@pytest.mark.parametrize('value',[True,0,-1,3,1.5])
def test_startup_concurrency_refuses_invalid_values_before_tree_start(tmp_path,monkeypatch,value):
    monkeypatch.setattr(session.OwnedPlanner,'__init__',lambda *a,**k:pytest.fail('tree started'))
    with pytest.raises(QualificationDecline):
        session.OwnedSession(tmp_path,workers=2,startup_concurrency=value,tree_memory_limit=2**30)


def test_concurrent_startup_requires_explicit_memory_cap_before_tree_start(tmp_path,monkeypatch):
    monkeypatch.setattr(session.OwnedPlanner,'__init__',lambda *a,**k:pytest.fail('tree started'))
    with pytest.raises(QualificationDecline,match='memory'):
        session.OwnedSession(tmp_path,workers=2,startup_concurrency=2)


def test_stage_control_defaults_and_explicit_option(tmp_path):
    args=dict(expected_ci=('p','a'),observed_ci=lambda _:('p','a'),workers=2)
    assert host.StageControl(tmp_path,**args).startup_concurrency==1
    assert host.StageControl(tmp_path,**args,startup_concurrency=2).startup_concurrency==2
