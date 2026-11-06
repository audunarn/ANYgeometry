"""Compact transcript bindings retain independently checked ordered coverage."""
import copy
import pytest
from anygeometry import _planning_session as session
from anygeometry import _planning_host as host
from anygeometry._planning_executor import pack
from anygeometry._planning_qualification import QualificationDecline
from test_planning_transcript_reports import fixture
from test_planning_transcript import prepared


def compact_fixture():
    report,tasks,setup,final=fixture()
    for worker in report['worker_reports']:
        rows=worker.pop('rows')
        worker.update(result_count=len(rows),results_sha256=pack(rows,'worker-results')['sha256'])
    report['startup']=dict(configured_concurrency=1,max_inflight_startups=1,ready_workers=2,active_workers=1)
    return report,tasks,setup,final


def test_compact_worker_results_retain_empty_worker_and_gate_accounting():
    session.verify_prepared_stage(*compact_fixture())


@pytest.mark.parametrize('fault',['digest','digest_type','count','count_bool','missing_digest',
    'mixed_rows','missing_result','duplicate_result','wrong_value','pid','gate','extra'])
def test_compact_worker_result_tampering_refuses(fault):
    report,tasks,setup,final=compact_fixture();worker=report['worker_reports'][0]
    if fault=='digest':worker['results_sha256']='f'*64
    elif fault=='digest_type':worker['results_sha256']=False
    elif fault=='count':worker['result_count']=2
    elif fault=='count_bool':worker['result_count']=True
    elif fault=='missing_digest':del worker['results_sha256']
    elif fault=='mixed_rows':worker['rows']=[[0,'value']]
    elif fault=='missing_result':report['results']=[]
    elif fault=='duplicate_result':report['results']*=2
    elif fault=='wrong_value':report['results'][0]='forged'
    elif fault=='pid':worker['pid']+=1
    elif fault=='gate':worker['gate_closed']=False
    else:worker['unbound_extra']=0
    with pytest.raises(QualificationDecline):session.verify_prepared_stage(report,tasks,setup,final)


@pytest.mark.parametrize('rows',[[[1,'a'],[0,'b']],[[0,'a'],[0,'b']],[[0,'a']],[[False,'a'],[1,'b']]])
def test_compaction_refuses_missing_duplicate_reordered_or_bool_ordinals(rows):
    with pytest.raises(QualificationDecline):
        session.compact_worker_report({'rows':rows},[0,1])


def test_compaction_removes_rows_without_mutating_original_and_keeps_packet_bound():
    rows=[[0,'x'*1000],[1,'y'*1000]];original={'rows':rows,'task_count':2}
    compact=session.compact_worker_report(original,[0,1])
    assert 'rows' not in compact and original['rows']==rows
    assert compact['result_count']==2
    assert compact['results_sha256']==pack(rows,'worker-results')['sha256']
    assert len(str(compact))<300
    with pytest.raises(QualificationDecline):
        session.compact_worker_report(original,[0,1],payload_limit=100)


def compact_entries(entries):
    result=copy.deepcopy(entries)
    for entry in result:
        tasks=entry.pop('tasks')
        entry.update(task_count=len(tasks),tasks_sha256=pack(tasks,'tasks')['sha256'])
    return result


def test_compact_transcript_rebuilds_tasks_and_runs_normal_replay(tmp_path):
    packet,control,sources,entries=prepared(tmp_path)
    bridge=host._TranscriptBridge(packet,control,compact_entries(entries),sources=sources)
    for entry in entries:
        assert bridge.run(entry['kind'],entry['tasks'],lambda task:task['kind'])==(entry['kind'],)
    bridge.finish()


@pytest.mark.parametrize('fault',['digest','upper_digest','count','bool_count','mixed_tasks','extra','reorder'])
def test_compact_task_binding_refuses_before_reference_execution(tmp_path,fault):
    packet,control,sources,entries=prepared(tmp_path);compact=compact_entries(entries)
    entry=compact[0]
    if fault=='digest':entry['tasks_sha256']='f'*64
    elif fault=='upper_digest':entry['tasks_sha256']=entry['tasks_sha256'].upper()
    elif fault=='count':entry['task_count']+=1
    elif fault=='bool_count':entry['task_count']=True
    elif fault=='mixed_tasks':entry['tasks']=entries[0]['tasks']
    elif fault=='extra':entry['extra']=0
    else:compact.reverse()
    with pytest.raises(QualificationDecline):
        bridge=host._TranscriptBridge(packet,control,compact,sources=sources)
        bridge.run('pair',entries[0]['tasks'],lambda _:pytest.fail('unbound task sampled'))


def test_task_digest_binds_order_not_just_membership():
    tasks=[dict(kind='pair',ordinal=i) for i in range(2)]
    entry=dict(kind='pair',candidate={},task_count=2,tasks_sha256=pack(tasks,'tasks')['sha256'])
    with pytest.raises(QualificationDecline):session.validate_entry(entry,'pair',list(reversed(tasks)))


@pytest.mark.parametrize('key',['configured_concurrency','max_inflight_startups','ready_workers','active_workers'])
@pytest.mark.parametrize('value',[False,0,3])
def test_startup_accounting_tamper_refuses(key,value):
    report,tasks,setup,final=compact_fixture()
    report['startup']=dict(configured_concurrency=1,max_inflight_startups=1,ready_workers=2,active_workers=1)
    report['startup'][key]=value
    with pytest.raises(QualificationDecline):session.verify_prepared_stage(report,tasks,setup,final)


@pytest.mark.parametrize('fault',['mixed_workers','top_rows','missing_startup'])
def test_compact_report_cannot_mix_schemas_or_omit_startup(fault):
    report,tasks,setup,final=compact_fixture()
    if fault=='mixed_workers':
        worker=report['worker_reports'][1]
        del worker['result_count'],worker['results_sha256']
        worker['rows']=[]
    elif fault=='top_rows':report['rows']=[[0,'value']]
    else:del report['startup']
    with pytest.raises(QualificationDecline):session.verify_prepared_stage(report,tasks,setup,final)
