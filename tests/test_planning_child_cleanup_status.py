"""Injected child-main classification; this does not claim native cleanup."""
import contextlib
import json
import sys

import pytest

from anygeometry import _planning_executor as executor


@pytest.mark.parametrize('failure',['close','receipt'])
@pytest.mark.parametrize('role',['planner','session-planner'])
def test_planner_cleanup_failure_is_terminal_on_wire(tmp_path,monkeypatch,failure,role):
    from anygeometry import _planning_child as child
    events=[];packets=[]
    request=dict(model={},tasks=[],sources={},workers=1,batch_size=None,stage=False)
    (tmp_path/'request.json').touch()
    class Ledger:
        resets={}
        trusted_image=True
        def prepare(self,**k):pass
        def live_code_pins(self):return {}
    class Tree:
        def __init__(self,folder):events.append('tree')
        def close(self):
            events.append('close')
            if failure=='close':raise OSError('owned worker close failed')
            (tmp_path/'cleanup.json').write_text('{}')
    monkeypatch.setattr(child,'OwnedTree',Tree)
    def prepare(*,production=False):
        assert production is False
        return {},{}
    monkeypatch.setattr(child,'prepare',prepare)
    monkeypatch.setattr(child,'load',lambda:{})
    monkeypatch.setattr(child,'threadpool_limits',lambda **k:contextlib.nullcontext())
    monkeypatch.setattr(child,'verify_sources',lambda *a:None)
    monkeypatch.setattr(child,'thaw_model',lambda *a,**k:None)
    monkeypatch.setattr(child,'dispatch',lambda *a:([],[]))
    monkeypatch.setattr(child.ledger_api,'loaded_geometry_ledger',lambda **k:Ledger())
    monkeypatch.setattr(child.q,'capture_environment',lambda **k:{})
    monkeypatch.setattr(child,'read_packet',lambda *a,**k:(request,'request-sha'))
    monkeypatch.setattr(child,'write_packet',lambda path,packet:packets.append((path,packet)))
    if role=='session-planner':
        from anygeometry import _planning_session as session
        monkeypatch.setattr(session,'OwnedTree',Tree)
        # session_planner calls _setup(...,decode=False) and, on this legacy path, then really decodes the admitted
        # model with session.thaw_model (unpatched): the faked setup must carry a REAL frozen canary packet.
        from anygeometry._planning_canaries import _fixture
        packet=executor.freeze_model(_fixture()[0])
        monkeypatch.setattr(session,'_setup',lambda *a,**k: (dict(workers=0,sources={},model=packet), 'session-sha'))
        def refuse_command(*a,**k):
            raise child.q.QualificationDecline('injected session-command refusal')
        monkeypatch.setattr(session,'_wait_file',refuse_command)
    monkeypatch.setattr(sys,'argv',[child.__file__,role,str(tmp_path),'10'])
    if failure=='receipt':
        replace=child.Path.replace
        def fail_replace(self,target):
            if self==tmp_path/'cleanup.json':raise OSError('owned cleanup receipt failed')
            return replace(self,target)
        monkeypatch.setattr(child.Path,'replace',fail_replace)
    child.main()
    assert events==['tree','close'] and packets[-1][0]==tmp_path/'result.json'
    result=executor.unpack(packets[-1][1],'result')
    assert result['status']=='cleanup_error'
    with pytest.raises(executor.PlannerCleanupError):executor.check_result_status(result)
