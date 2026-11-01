"""Session host transaction tests; injected sessions do not prove real cleanup."""
import importlib
import json

import pytest

from anygeometry import _planning_host as host
from anygeometry import _planning_executor as executor
from anygeometry import batch_intersections as bi
from anygeometry._planning_canaries import _fixture
from anygeometry._planning_values import encode_value
from anygeometry.serialization import from_dict, to_dict


def setup(tmp_path, monkeypatch, *, corrupt=False):
    module=importlib.import_module('anygeometry._planning_session')
    events=[]
    class Fake:
        def __init__(self,*args,**kw):events.append('start')
        def submit_stage(self,kind,packet,tasks,*,poll):
            events.append(kind);poll()
            value='corrupt' if corrupt and kind=='pair' else kind
            return dict(results=(json.dumps(encode_value(value),sort_keys=True,separators=(',',':')),),
                        task_layout=((0,0,kind),),request_sha256='b'*64,sampling_sha256='c'*64)
        def finish(self,*,poll):events.append('finish');poll()
        def close(self):events.append('retired')
    monkeypatch.setattr(module,'OwnedSession',Fake)
    model,_=_fixture();model=from_dict(to_dict(model))
    def observed(_):events.append('ci');return ('plan','apply')
    control=host.StageControl(tmp_path/'session',expected_ci=('plan','apply'),observed_ci=observed,
                              workers=2,reuse_workers=True)
    def serial_task(kind):
        def run(task):events.append('sample-'+kind);return kind
        return run
    def parallel(bridge):
        pair=bridge.run('pair',[dict(kind='pair',cost=1.)],serial_task('pair'))
        assert pair==(('corrupt',) if corrupt else ('pair',))
        assert not any(e.startswith('sample-') for e in events)
        face=bridge.run('face',[dict(kind='face',cost=1.)],serial_task('face'))
        assert face==('face',)
        return 'candidate'
    return model,control,events,parallel


def test_both_stages_retire_before_all_samples_and_ci(tmp_path,monkeypatch):
    model,control,events,parallel=setup(tmp_path,monkeypatch)
    result=host.qualified_plan(model,control,lambda:pytest.fail('replayed'),parallel,
                               policy=bi.IntersectionBatchPolicy(),mode='exact')
    assert result=='candidate' and control.ci_verified
    assert events==['start','pair','face','finish','retired','sample-pair','sample-face','ci']
    assert [r['stage'] for r in control.reports]==['pair','face']
    assert all(r['sampling']['plan_digest']=='c'*64 for r in control.reports)


def test_bad_pair_stops_deferred_face_sampling_and_replays_after_cleanup(tmp_path,monkeypatch):
    model,control,events,parallel=setup(tmp_path,monkeypatch,corrupt=True)
    result=host.qualified_plan(model,control,lambda:events.append('serial') or 'original',parallel,
                               policy=bi.IntersectionBatchPolicy(),mode='exact')
    assert result=='original'
    assert events==['start','pair','face','finish','retired','sample-pair','serial']
    assert control.mode=='serial' and not control.ci_verified


def test_provisional_construction_error_replays_original_after_retirement(tmp_path,monkeypatch):
    model,control,events,parallel=setup(tmp_path,monkeypatch)
    provisional=ValueError('invalid provisional value')
    original=bi.GeometryError('original serial attribution')
    def fail(bridge):parallel(bridge);raise provisional
    def serial():events.append('serial');raise original
    with pytest.raises(bi.GeometryError) as caught:
        host.qualified_plan(model,control,serial,fail,policy=bi.IntersectionBatchPolicy(),mode='exact')
    assert caught.value is original
    assert events==['start','pair','face','retired','serial']
    assert original.__context__.__cause__ is provisional


@pytest.mark.parametrize('error', [executor.PlannerCancelled('cancelled'),executor.PlannerDeadline('deadline'),
    executor.PlannerCleanupError('cleanup'),executor.PlannerSourceChanged('source'),KeyboardInterrupt()])
def test_terminal_error_never_becomes_provisional_replay(tmp_path,monkeypatch,error):
    model,control,events,parallel=setup(tmp_path,monkeypatch)
    def fail(bridge):parallel(bridge);raise error
    with pytest.raises(type(error)) as caught:
        host.qualified_plan(model,control,lambda:pytest.fail('terminal replay'),fail,
                            policy=bi.IntersectionBatchPolicy(),mode='exact')
    assert caught.value is error and events==['start','pair','face','retired']
    assert control.mode=='terminal'


@pytest.mark.parametrize('field',['max_predicates','max_candidate_pairs'])
def test_finite_budget_never_constructs_session_or_samples(tmp_path,monkeypatch,field):
    model,control,events,parallel=setup(tmp_path,monkeypatch)
    result=host.qualified_plan(model,control,lambda:'serial',parallel,
                              policy=bi.IntersectionBatchPolicy(**{field:1}),mode='exact')
    assert result=='serial' and events==[]
