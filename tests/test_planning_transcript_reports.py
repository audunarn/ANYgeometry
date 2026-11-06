"""Host independently validates prepared stage reports before replay."""
import copy
import pytest
from anygeometry import _planning_session as session
from anygeometry._planning_stages import dispatch
from anygeometry._planning_qualification import QualificationDecline


def fixture():
    tasks=[dict(kind='pair',cost=1.)];parts,layout=dispatch(tasks,2)
    setup=dict(workers=2,model=dict(sha256='a'*64))
    final=[dict(pid=100+i) for i in range(2)]
    rows=[[0,'value']]
    workers=[]
    for i in range(2):
        assigned=rows if i==0 else []
        workers.append(dict(status='passed',worker=i,pid=100+i,model_decode_count=1,model_sha256='a'*64,
            task_count=len(assigned),rows=assigned,gate_closed=True,gate_poisoned=False,
            full_environment_checks=1,boundary_seconds=[0.]*(4 if assigned else 3),blas_threads=[dict(num_threads=1)]))
    report=dict(status='passed',model_sha256='a'*64,requested_workers=2,actual_workers=1,
                task_layout=layout,results=['value'],worker_reports=workers)
    return report,tasks,setup,final


def test_prepared_report_and_empty_worker_are_checked():
    session.verify_prepared_stage(*fixture())


@pytest.mark.parametrize('change',['pid','decode','model','count','closed','poisoned','environment','boundaries',
    'threads','ordinal','value','worker_bool','missing_worker','requested_bool','actual','layout'])
def test_bad_prepared_worker_report_refuses(change):
    report,tasks,setup,final=fixture();w=report['worker_reports'][0]
    if change=='pid':w['pid']=200
    elif change=='decode':w['model_decode_count']=True
    elif change=='model':w['model_sha256']='b'*64
    elif change=='count':w['task_count']=0
    elif change=='closed':w['gate_closed']=False
    elif change=='poisoned':w['gate_poisoned']=True
    elif change=='environment':w['full_environment_checks']=0
    elif change=='boundaries':w['boundary_seconds']=[]
    elif change=='threads':w['blas_threads']=[dict(num_threads=2)]
    elif change=='ordinal':w['rows'][0][0]=1
    elif change=='value':w['rows'][0][1]='forged'
    elif change=='worker_bool':w['worker']=False
    elif change=='missing_worker':report['worker_reports'].pop()
    elif change=='requested_bool':report['requested_workers']=True
    elif change=='actual':report['actual_workers']=2
    else:report['task_layout']=[[1,0,'pair']]
    with pytest.raises(QualificationDecline):session.verify_prepared_stage(report,tasks,setup,final)
