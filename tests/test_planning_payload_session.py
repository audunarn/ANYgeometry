"""Ordinary sessions authenticate compact receipts before host acceptance."""
import copy
import time
from types import SimpleNamespace
import pytest
from anygeometry import _planning_session as session
from anygeometry._planning_executor import pack
from anygeometry._planning_stages import dispatch
from anygeometry._planning_qualification import QualificationDecline
from test_planning_payload_compaction import compact_fixture


def ordinary(tmp_path,monkeypatch):
    report,tasks,setup,final=compact_fixture()
    tasks=tasks+[dict(kind='pair',cost=1.)]
    parts,layout=dispatch(tasks,2)
    report.update(task_layout=layout,actual_workers=2,results=['first','second'])
    report['startup']['active_workers']=2
    report['rows']=[[i,report['results'][i]] for part in parts for i in part]
    for i,worker in enumerate(report['worker_reports']):
        rows=[[j,report['results'][j]] for j in parts[i]]
        worker.update(task_count=len(rows),result_count=len(rows),
            results_sha256=pack(rows,'worker-results')['sha256'],boundary_seconds=[0.]*4)
        final[i].update(status='passed',worker=i,stages=['pair'],gate_closures=1,
            model_sha256=setup['model']['sha256'],model_decode_count=1,blas_threads=[dict(num_threads=1)])
    del report['results']
    owned=session.OwnedSession.__new__(session.OwnedSession)
    owned.numerical_manifest=None  # OwnedSession constructor default for this ordinary route.
    owned.closed=owned.failed=owned.finished=False
    owned.session=owned.previous='b'*64;owned.stages=();owned.sources={}
    owned.packet=setup['model'];owned.workers=2;owned.batch_size=None;owned.startup_concurrency=1
    owned.directory=tmp_path;owned.limits=dict(payload_limit=4_000_000,walk_limit=100000)
    owned.started=time.perf_counter();owned.deadline=time.monotonic()+20
    owned._stage_receipts=[];owned._stage_reports=[];owned.child=None
    owned._poll=lambda poll:None;owned.tree=SimpleNamespace(wait=lambda *a,**k:None)
    monkeypatch.setattr(session,'_wait_file',lambda *a,**k:None)
    monkeypatch.setattr(session,'verify_sources',lambda *a:None)
    read=session.read_packet
    def reply(path,kind,**kwargs):
        if path.name.startswith('reply'):
            _,digest=read(tmp_path/'command-0.json','session-command',with_digest=True)
            report.update(session_sha256=owned.session,request_sha256=digest,stage='pair')
            return report
        if path.name=='result.json':
            _,digest=read(tmp_path/'command-1.json','session-command',with_digest=True)
            return dict(status='passed',session_sha256=owned.session,request_sha256=digest,
                stages=['pair'],stage_requests=owned._stage_receipts,worker_reports=final)
        return read(path,kind,**kwargs)
    monkeypatch.setattr(session,'read_packet',reply)
    return owned,report,tasks,final


@pytest.mark.parametrize('fault',['digest','count','missing_worker','reordered_rows','missing_row','duplicate_row'])
def test_ordinary_submit_refuses_compact_receipt_tampering(tmp_path,monkeypatch,fault):
    owned,report,tasks,final=ordinary(tmp_path,monkeypatch)
    if fault=='digest':report['worker_reports'][0]['results_sha256']='f'*64
    elif fault=='count':report['worker_reports'][0]['result_count']=0
    elif fault=='missing_worker':report['worker_reports'].pop()
    elif fault=='reordered_rows':report['rows'].reverse()
    elif fault=='missing_row':report['rows'].pop()
    else:report['rows'].append(copy.deepcopy(report['rows'][0]))
    with pytest.raises(QualificationDecline):owned.submit_stage('pair',owned.packet,tasks)
    assert owned.failed


@pytest.mark.parametrize('fault',['pid','gate','digest','missing_worker'])
def test_ordinary_finish_rebinds_retained_receipts_to_final_workers(tmp_path,monkeypatch,fault):
    owned,report,tasks,final=ordinary(tmp_path,monkeypatch)
    owned.submit_stage('pair',owned.packet,tasks)
    if fault=='pid':final[0]['pid']+=1
    elif fault=='gate':final[0]['gate_closures']=0
    elif fault=='digest':report['worker_reports'][0]['results_sha256']='f'*64
    else:final.pop()
    with pytest.raises(QualificationDecline):owned.finish()
    assert owned.failed and not owned.finished


def test_ordinary_compact_receipts_finish_with_matching_gate_and_pid(tmp_path,monkeypatch):
    owned,report,tasks,final=ordinary(tmp_path,monkeypatch)
    assert owned.submit_stage('pair',owned.packet,tasks)['results']==('first','second')
    owned.finish()
    assert owned.finished and not owned.failed
