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
