"""Closed finite session protocol and legacy sampling identity, before implementation."""
import copy
import importlib

import pytest

from anygeometry._planning_executor import pack
from anygeometry._planning_qualification import QualificationDecline


def api():
    return importlib.import_module('anygeometry._planning_session')


def command(stage='pair', sequence=0, previous='a'*64):
    return dict(session='a'*64, sequence=sequence, previous=previous, stage=stage,
                tasks=[] if stage=='finish' else [dict(kind=stage, cost=1.)])


def check(body, *, sequence=0, previous='a'*64, seen=()):
    return api().validate_command(body, session='a'*64, sequence=sequence,
                                  previous=previous, seen=seen)


def test_stage_control_reuse_is_explicit_and_default_off(tmp_path):
    from anygeometry._planning_host import StageControl
    default=StageControl(tmp_path/'unused', expected_ci=('p','a'), observed_ci=lambda _:('p','a'))
    assert default.reuse_workers is False
    selected=StageControl(tmp_path/'selected', expected_ci=('p','a'), observed_ci=lambda _:('p','a'), reuse_workers=True)
    assert selected.reuse_workers is True


@pytest.mark.parametrize('stage,seen', [('pair',()),('face',()),('face',('pair',)),
                                      ('finish',('pair',)),('finish',('face',)),
                                      ('finish',('pair','face'))])
def test_only_finite_forward_stage_sequences_are_admitted(stage, seen):
    body=command(stage, len(seen));before=copy.deepcopy(body)
    assert check(body, sequence=len(seen), seen=seen)==stage
    assert body==before


@pytest.mark.parametrize('field,value', [('session','b'*64),('sequence',True),
    ('sequence',1),('previous','b'*64),('stage','other'),('tasks',{}),('extra',None)])
def test_malformed_foreign_and_stale_command_cannot_advance(field, value):
    body=command();body[field]=value
    with pytest.raises(QualificationDecline):check(body)


@pytest.mark.parametrize('stage,seen', [('pair',('pair',)),('pair',('face',)),
                                      ('face',('pair','face')),('finish',())])
def test_repeated_reordered_or_unstarted_finish_refuses(stage, seen):
    with pytest.raises(QualificationDecline):
        check(command(stage,len(seen)), sequence=len(seen), seen=seen)


def test_finish_cannot_smuggle_work():
    body=command('finish',1);body['tasks']=[dict(kind='pair',cost=1.)]
    with pytest.raises(QualificationDecline):check(body,sequence=1,seen=('pair',))


def test_stage_kind_mismatch_refuses_before_dispatch():
    body=command();body['tasks']=[dict(kind='face',cost=1.)]
    with pytest.raises(QualificationDecline):check(body)


def test_sampling_seed_is_exact_legacy_stage_request_digest():
    packet=pack({'cold':'fixture'}, 'model');tasks=[dict(kind='pair',cost=1.)]
    sources={'pinned.py':'b'*64}
    expected=pack(dict(model=packet,tasks=tasks,workers=2,batch_size=None,sources=sources,stage=True),'request')['sha256']
    assert api().sampling_digest(packet,tasks,workers=2,batch_size=None,sources=sources)==expected


def test_sampling_seed_changes_with_order_and_source():
    packet=pack({},'model');tasks=[dict(kind='pair',cost=1.),dict(kind='pair',cost=2.)]
    args=dict(workers=2,batch_size=None,sources={'pinned.py':'b'*64})
    initial=api().sampling_digest(packet,tasks,**args)
    assert api().sampling_digest(packet,list(reversed(tasks)),**args)!=initial
    args['sources']={'pinned.py':'c'*64}
    assert api().sampling_digest(packet,tasks,**args)!=initial


def test_child_cleanup_failure_status_is_terminal():
    from anygeometry._planning_executor import check_result_status, PlannerCleanupError
    with pytest.raises(PlannerCleanupError):
        check_result_status(dict(status='cleanup_error',reason='owned cleanup failed'))
def test_packet_byte_observation_reports_actual_bounded_envelope(tmp_path):
    from anygeometry._planning_session import _packet_bytes
    from anygeometry._planning_executor import pack,write_packet
    from anygeometry._planning_qualification import QualificationDecline
    path=tmp_path/'packet.json';packet=pack({'value':'\u00e9'},'bytes')
    write_packet(path,packet)
    observed=_packet_bytes(path,'bytes',1000)
    assert observed==dict(payload_utf8_bytes=len(packet['payload'].encode('utf-8')),envelope_bytes=path.stat().st_size)
    with pytest.raises(QualificationDecline):_packet_bytes(path,'bytes',1)


def test_initialization_phase_evidence_is_bounded_owned_and_accumulative(tmp_path,monkeypatch):
    import os
    module=api();monkeypatch.setattr(module.time,'perf_counter',lambda:12.)
    module._record_initialization(tmp_path,'child_prepare',10.)
    module._record_initialization(tmp_path,'model_context',11.)
    row=module._read_initialization(tmp_path,os.getpid())
    assert row['phases']=={'child_prepare':2.,'model_context':1.}
    assert not (tmp_path/'initialization-phases.tmp').exists()
    with pytest.raises(QualificationDecline):module._read_initialization(tmp_path,os.getpid()+1)


@pytest.mark.parametrize('bad',[{'unknown':1.},{'child_prepare':-1.},{'child_prepare':float('nan')},{'child_prepare':True}])
def test_initialization_phase_evidence_refuses_invalid_rows(tmp_path,bad):
    import json,os
    (tmp_path/'initialization-phases.json').write_text(json.dumps({'pid':os.getpid(),'phases':bad}))
    with pytest.raises(QualificationDecline):api()._read_initialization(tmp_path,os.getpid())


def test_initialization_phase_evidence_refuses_oversized_file(tmp_path):
    import os
    (tmp_path/'initialization-phases.json').write_text(' '*8193)
    with pytest.raises(QualificationDecline):api()._read_initialization(tmp_path,os.getpid())


def test_finish_wait_sidecar_retains_pending_then_completed_clock(tmp_path,monkeypatch):
    import os
    module=api();clock=[12.]
    monkeypatch.setattr(module.time,'perf_counter',lambda:clock[0])
    module._record_initialization(tmp_path,'finish_wait_0',10.,pending=True)
    row=module._read_initialization(tmp_path,os.getpid())
    assert 'finish_wait_0' not in row['phases']
    assert row['events']['finish_wait_0']=={'started':10.,'ended':None}
    clock[0]=15.
    module._record_initialization(tmp_path,'finish_wait_0',10.)
    row=module._read_initialization(tmp_path,os.getpid())
    assert row['phases']['finish_wait_0']==5.
    assert row['events']['finish_wait_0']=={'started':10.,'ended':15.}
    with pytest.raises(QualificationDecline):module._record_initialization(tmp_path,'finish_wait_0',10.,pending=True)


@pytest.mark.parametrize('event,phases',[
    ({'started':True,'ended':None},{}),
    ({'started':10.,'ended':9.},{}),
    ({'started':10.,'ended':float('inf')},{}),
    ({'started':10.,'ended':None},{'finish_wait_0':0.}),
    ({'started':10.,'ended':12.},{'finish_wait_0':1.}),
    ({'started':10.,'ended':12.,'unknown':0},{'finish_wait_0':2.}),
])
def test_finish_event_schema_refuses_inconsistent_or_unknown_state(tmp_path,event,phases):
    import json,os
    row=dict(pid=os.getpid(),phases=phases,events={'finish_wait_0':event})
    (tmp_path/'initialization-phases.json').write_text(json.dumps(row))
    with pytest.raises(QualificationDecline):api()._read_initialization(tmp_path,os.getpid())


@pytest.mark.parametrize('failure',['start','end','close','prior'])
def test_finish_trace_never_skips_containment_or_masks_close(tmp_path,monkeypatch,failure):
    module=api();events=[];trace_error=OSError('original trace failure');close_error=RuntimeError('original close failure')
    def trace(*args,**kwargs):
        events.append('start' if kwargs.get('pending') else 'end')
        if failure in ('start','close','prior') and kwargs.get('pending') or failure=='end' and not kwargs.get('pending'):raise trace_error
    class Tree:
        def close(self):
            events.append('close')
            if failure=='close':raise close_error
            (tmp_path/'cleanup.json').write_text('{}')
    monkeypatch.setattr(module,'_record_initialization',trace)
    if failure=='prior':
        module._retire_session_tree(Tree(),tmp_path,production=True,prior_error=RuntimeError('pending primary'))
    else:
        from anygeometry._planning_executor import PlannerCleanupError
        with pytest.raises(PlannerCleanupError) as caught:module._retire_session_tree(Tree(),tmp_path,production=True)
        assert caught.value.__cause__ is (close_error if failure=='close' else trace_error)
    assert events.count('close')==1
    if failure!='close':assert (tmp_path/'workers-cleanup.json').exists()


@pytest.mark.parametrize('scenario',['transient','permanent','deadline','callback','cancel','corrupt'])
def test_published_packet_permission_retry_is_narrow_and_bounded(tmp_path,monkeypatch,scenario):
    from anygeometry._planning_executor import PlannerDeadline,PlannerCancelled
    import json
    module=api();path=tmp_path/'reply-1.json';path.write_text('{}')
    clock=[0.];calls=[];polls=[];original=PermissionError(13,'original access denial',str(path))
    callback_error=RuntimeError('original callback')
    def read(*args,**kwargs):
        calls.append(1)
        if scenario=='cancel':(tmp_path/'cancel.flag').write_text('cancel')
        if scenario=='corrupt':raise QualificationDecline('checksum mismatch')
        if len(calls)==1 or scenario in ('permanent','deadline','callback'):raise original
        return {'validated':True}
    def poll():
        polls.append(1)
        if scenario=='callback' and len(polls)==2:raise callback_error
    monkeypatch.setattr(module,'read_packet',read)
    monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(module.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    if scenario=='transient':assert module._read_published_packet(path,'reply',1.,poll=poll)=={'validated':True}
    else:
        expected=PlannerCancelled if scenario=='cancel' else PlannerDeadline if scenario=='deadline' else RuntimeError if scenario=='callback' else QualificationDecline if scenario=='corrupt' else PermissionError
        with pytest.raises(expected) as caught:module._read_published_packet(path,'reply',.005 if scenario=='deadline' else 1.,poll=poll,cancel_folder=tmp_path)
        if scenario=='callback':assert caught.value is callback_error
        if scenario=='permanent':assert caught.value is original
    assert clock[0]<=.25 and len(calls)<=32
    if scenario=='corrupt':assert len(calls)==1 and not list(tmp_path.glob('packet-read-*.json'))
    else:
        rows=[json.loads(file.read_text()) for file in tmp_path.glob('packet-read-*.json')]
        assert len(rows)==1 and rows[0]['errno']==13 and rows[0]['winerror'] is None
        assert rows[0]['attempts']==len(calls)
        if scenario=='transient':assert rows[0]['status']=='recovered'

