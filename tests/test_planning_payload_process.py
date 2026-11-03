"""Real compact preparation, admitted startup, retirement and transcript replay."""
import json
from anygeometry import batch_intersections as bi
from anygeometry._planning_canaries import _fixture
from anygeometry._planning_host import StageControl
from anygeometry._planning_executor import read_packet
from anygeometry.serialization import from_dict,to_dict


def test_real_compact_preparation_with_memory_bounded_startup(tmp_path,monkeypatch):
    model,_=_fixture();document=to_dict(model)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','0')
    reference=from_dict(document)
    serial=bi.plan_intersections(reference,tuple(reference.faces),policy='connect')
    bi.apply_intersections(reference,serial,policy='connect')
    expected=(serial.content_checksum,to_dict(reference)['checksum']['value'])
    def observed(plan):
        target=from_dict(document);bi.apply_intersections(target,plan,policy='connect')
        return plan.content_checksum,to_dict(target)['checksum']['value']
    control=StageControl(tmp_path/'compact',expected_ci=expected,observed_ci=observed,
        workers=2,reuse_workers=True,startup_concurrency=2,tree_memory_limit=2**31,timeout=85)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    source=from_dict(document)
    result=bi.plan_intersections(source,tuple(source.faces),policy='connect',_parallel_control=control)
    assert control.mode=='parallel',control.decline_reason
    assert control.ci_verified and result.content_checksum==serial.content_checksum
    assert to_dict(source)==document
    folder=control.directory/'session'
    final=read_packet(folder/'result.json','result')
    assert [entry['kind'] for entry in final['prepared_entries']]==['pair','face']
    for entry in final['prepared_entries']:
        assert set(entry)=={'kind','task_count','tasks_sha256','candidate'}
        candidate=entry['candidate']
        assert 'rows' not in candidate
        assert candidate['startup']==dict(configured_concurrency=2,max_inflight_startups=2,
            ready_workers=2,active_workers=candidate['actual_workers'])
        for worker in candidate['worker_reports']:
            assert 'rows' not in worker
            assert worker['result_count']==worker['task_count']
            assert len(worker['results_sha256'])==64
    for name in ('cleanup.json','workers-cleanup.json'):
        cleanup=json.loads((folder/name).read_text())
        assert cleanup['verified'] and cleanup['active_processes']==0
