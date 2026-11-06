"""Real sessions: exact serial/apply controls and version-bound sample streams.

The old v1 non-session request digest aliased its sampling seed. Package session
commands have a separate chain digest; face v2 bytes intentionally change the
sampling seed while preserving canonical task data and the same sampling rule.
"""
from _planning_numpy_profile_contract import numpy_profile_test
from _planning_platform_contract import windows_owned_test
import json

from anygeometry import batch_intersections as bi
from anygeometry._planning_host import StageControl
from anygeometry._planning_canaries import _fixture
from anygeometry.serialization import from_dict, to_dict


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _assert_sample_strata(layout,samples):
    indices=samples['indices']
    assert all(type(index) is int and 0<=index<len(layout) for index in indices)
    assert len(indices)==len(set(indices))
    assert set(samples['workers'])=={row[0] for row in layout}
    for worker in samples['workers']:
        # Layout carries worker-local positions; samples carry global ordinals.
        rows=sorted((position,ordinal,kind) for ordinal,(owner,position,kind) in enumerate(layout) if owner==worker)
        chosen=[row for row in rows if row[1] in samples['indices']]
        assert rows[-1][1] in samples['indices']
        assert len(chosen)>=min(len(rows),max(2,(len(rows)+19)//20))
        assert {row[2] for row in chosen}=={row[2] for row in rows}
        for kind in {row[2] for row in rows}:
            assert next(row[1] for row in reversed(rows) if row[2]==kind) in indices


def _stage_wire_evidence(control,report,index):
    """Independently bind the actual command, codec version and sample strata."""
    from anygeometry._planning_executor import read_packet
    from anygeometry._planning_session import sampling_digest
    from anygeometry._planning_stages import dispatch
    from anygeometry._planning_admission import stratified_samples
    from anygeometry._planning_values import expand_value_v2
    folder=control.directory/('session' if control.reuse_workers else report['stage'])
    sequence=index if control.reuse_workers else 0
    setup,session=read_packet(folder/'session.json','session',with_digest=True)
    command,digest=read_packet(folder/f'command-{sequence}.json','session-command',with_digest=True)
    assert command['stage']==report['stage'] and command['session']==session==report['session_sha256']
    assert report['request_sha256']==digest
    tasks=command['tasks']
    seed=sampling_digest(setup['model'],tasks,workers=setup['workers'],batch_size=setup['batch_size'],sources=setup['sources'])
    assert report['sampling_sha256']==seed
    layout=dispatch(tasks,setup['workers'])[1]
    assert _canonical(report['task_layout'])==_canonical(layout)
    samples=stratified_samples(layout,seed)
    assert report['sampling']==samples
    _assert_sample_strata(layout,samples)
    expanded=tasks if control.reuse_workers else [
        {key:expand_value_v2(value) if key in ('inputs','support') else value for key,value in task.items()}
        for task in tasks]
    return dict(wire=_canonical(tasks),canonical=_canonical(expanded),tasks=expanded,seed=seed,samples=samples)


def _compare_stage_streams(old,new,first,second,index):
    left=_stage_wire_evidence(old,first,index);right=_stage_wire_evidence(new,second,index)
    assert left['canonical']==right['canonical']
    assert left['samples']['workers']==right['samples']['workers']
    assert len(left['samples']['indices'])==len(right['samples']['indices'])
    if left['wire']==right['wire']:
        assert left['seed']==right['seed']
        assert left['samples']['indices']==right['samples']['indices']
    else:
        assert first['stage']==second['stage']=='face'
        assert left['seed']!=right['seed']
    return left,right


@windows_owned_test
@numpy_profile_test
def test_public_session_reuses_workers_with_same_samples_and_plan_apply(tmp_path,monkeypatch):
    model,_=_fixture();document=to_dict(model)
    reference=from_dict(document)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','0')
    serial=bi.plan_intersections(reference,tuple(reference.faces),policy='connect')
    bi.apply_intersections(reference,serial,policy='connect')
    expected=(serial.content_checksum,to_dict(reference)['checksum']['value'])
    def observed(plan):
        candidate=from_dict(document)
        bi.apply_intersections(candidate,plan,policy='connect')
        return plan.content_checksum,to_dict(candidate)['checksum']['value']
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    controls=[]
    for reuse in (False,True):
        source=from_dict(document)
        control=StageControl(tmp_path/('session' if reuse else 'legacy'),expected_ci=expected,
                             observed_ci=observed,workers=2,reuse_workers=reuse)
        plan=bi.plan_intersections(source,tuple(source.faces),policy='connect',_parallel_control=control)
        assert control.mode=='parallel',control.decline_reason
        assert control.ci_verified and plan.content_checksum==serial.content_checksum
        assert to_dict(source)==document
        controls.append(control)
    old,new=controls
    assert [r['stage'] for r in new.reports]==['pair','face']
    for index,(first,second) in enumerate(zip(old.reports,new.reports)):
        assert first['sampling']['indices']==second['sampling']['indices']
        _compare_stage_streams(old,new,first,second,index)
        assert first['task_layout']==second['task_layout']
        assert all(r['full_environment_checks']==1 and len(r['boundary_seconds'])==4
                   and r['gate_closed'] and not r['gate_poisoned'] for r in second['worker_reports'])
    pair,face=new.reports
    assert [r['pid'] for r in pair['worker_reports']]==[r['pid'] for r in face['worker_reports']]
    assert all(r['model_decode_count']==1 for report in new.reports for r in report['worker_reports'])
    folder=new.directory/'session'
    for name in ('cleanup.json','workers-cleanup.json'):
        cleanup=json.loads((folder/name).read_text())
        assert cleanup['verified'] and cleanup['active_processes']==0
    from anygeometry._planning_executor import read_packet
    finish=read_packet(folder/'result.json','result')
    assert finish['status']=='passed' and finish['stages']==['pair','face']
    assert all(r['gate_closures']==2 for r in finish['worker_reports'])
