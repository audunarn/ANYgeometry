"""Public preparation transaction sequencing and terminal ownership contracts."""
import pytest
from anygeometry import _planning_host as host
from anygeometry import _planning_session as session
from anygeometry import _planning_executor as executor
from anygeometry.batch_intersections import IntersectionBatchPolicy
from anygeometry._planning_canaries import _fixture
from test_planning_transcript import prepared


def fixture(tmp_path,monkeypatch,*,error=None,cleanup_error=None,mutate=False,tamper=None,serial_error=None):
    _,control,sources,entries=prepared(tmp_path);model,_=_fixture();events=[]
    if tamper is not None:tamper(entries)
    class Owned:
        def __init__(self,*a,**k):events.append('start');self.sources=sources
        def prepare_plan(self,packet,descriptor,*,poll):
            events.append('prepare');poll()
            if mutate:model.add_points(((100.,100.,100.),))
            if error is not None:raise error
            return entries
        def close(self):
            events.append('retired')
            if cleanup_error is not None:raise cleanup_error
    monkeypatch.setattr(session,'OwnedSession',Owned)
    spec=dict(operands=[],seam='baseline',policy=dict(intent='connect',face_connections=True,
        member_connections=True,member_face_connections=True,planning_mode=None,seam_policy=None))
    def parallel(bridge):
        events.append('assembly')
        def sample(kind):
            def run(task):
                assert 'retired' in events
                events.append('sample-'+kind);return kind
            return run
        bridge.run('pair',entries[0]['tasks'],sample('pair'))
        bridge.run('face',entries[1]['tasks'],sample('face'))
        return 'candidate'
    def observed(plan):events.append('ci');return ('p','a')
    control.observed_ci=observed
    def serial():
        events.append('serial')
        if serial_error is not None:raise serial_error
        return 'serial'
    def call():
        return host.qualified_plan(model,control,serial,parallel,
            policy=IntersectionBatchPolicy(),mode='exact',preparation=spec)
    return call,control,events


def test_probe_returns_no_result_and_replay_is_after_cleanup(tmp_path,monkeypatch):
    call,control,events=fixture(tmp_path,monkeypatch)
    assert call()=='candidate'
    assert events==['assembly','start','prepare','retired','assembly','sample-pair','sample-face','ci']
    assert control.mode=='parallel' and control.ci_verified


@pytest.mark.parametrize('error',[executor.PlannerCancelled('cancel'),executor.PlannerDeadline('deadline'),
    executor.PlannerSourceChanged('source'),KeyboardInterrupt()])
def test_preparation_terminal_errors_retire_without_replay(tmp_path,monkeypatch,error):
    call,control,events=fixture(tmp_path,monkeypatch,error=error)
    with pytest.raises(type(error)) as caught:call()
    assert caught.value is error
    assert events==['assembly','start','prepare','retired'] and control.mode=='terminal'


def test_preparation_source_change_prevents_host_reconstruction(tmp_path,monkeypatch):
    call,control,events=fixture(tmp_path,monkeypatch,mutate=True)
    with pytest.raises(executor.PlannerSourceChanged):call()
    assert events==['assembly','start','prepare','retired'] and control.mode=='terminal'


def test_preparation_cleanup_failure_is_terminal(tmp_path,monkeypatch):
    error=OSError('owned cleanup failed')
    call,control,events=fixture(tmp_path,monkeypatch,cleanup_error=error)
    with pytest.raises(executor.PlannerCleanupError) as caught:call()
    assert caught.value.__cause__ is error
    assert events==['assembly','start','prepare','retired'] and control.mode=='terminal'


@pytest.mark.parametrize('failure',['json','artifact'])
def test_invalid_candidate_replays_original_error_after_retirement(tmp_path,monkeypatch,failure):
    import json
    from pathlib import Path
    from anygeometry._planning_values import encode_value
    from anygeometry.errors import GeometryError
    def tamper(entries):
        entries[0]['candidate']['results'][0]=('invalid JSON' if failure=='json' else
            json.dumps(encode_value('corrupt'),separators=(',',':')))
    if failure=='artifact':
        original=Path.write_text
        def fail_artifact(path,*args,**kwargs):
            if path.name=='sample-mismatch-pair.json':raise OSError('artifact failed')
            return original(path,*args,**kwargs)
        monkeypatch.setattr(Path,'write_text',fail_artifact)
    original_error=GeometryError('original serial attribution')
    call,control,events=fixture(tmp_path,monkeypatch,tamper=tamper,serial_error=original_error)
    with pytest.raises(GeometryError) as caught:call()
    assert caught.value is original_error
    assert events==['assembly','start','prepare','retired','assembly']+(['sample-pair'] if failure=='artifact' else [])+['serial']
    assert control.mode=='serial' and not control.ci_verified
