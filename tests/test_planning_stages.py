"""PP5 assignment-owned model retention and deterministic dispatch regressions."""
import importlib
import pytest
from anygeometry import _planning_executor as a
from anygeometry._planning_qualification import QualificationDecline
from anygeometry._planning_tasks import model_task
from anygeometry._planning_canaries import _fixture
from anygeometry.serialization import from_dict,to_dict
from anygeometry.definition_binding import definition_checksum

def api():return importlib.import_module('anygeometry._planning_stages')

def test_worker_decodes_once_and_retains_cold_source(monkeypatch):
    module=api();model,_=_fixture();packet=a.freeze_model(from_dict(to_dict(model)))
    count=[];old=module.thaw_model
    def thaw(value,**limits):count.append(1);return old(value,**limits)
    monkeypatch.setattr(module,'thaw_model',thaw)
    context=module.WorkerModel(packet)
    faces=list(context.model.faces);tol=context.model.tolerance.effective_length(5.)
    task=dict(kind='pair',first=faces[0],second=faces[1],tolerance=tol,face_tolerances=[tol,tol])
    first=module.stage_task(context,task);second=module.stage_task(context,task)
    assert first==second and count==[1]
    assert context.verify()==packet['sha256']


def test_changed_worker_model_is_refused():
    module=api();model,_=_fixture();packet=a.freeze_model(from_dict(to_dict(model)))
    context=module.WorkerModel(packet);context.model.extra='mutated'
    with pytest.raises(QualificationDecline):context.verify()

@pytest.mark.parametrize('workers',[1,2,8,16])
def test_largest_first_balances_and_covers_classes(workers):
    tasks=[dict(kind=kind,cost=float(50-i)) for i in range(40) for kind in ('pair','face')]
    assignments,layout=api().dispatch(tasks,workers)
    assert sorted(i for part in assignments for i in part)==list(range(len(tasks)))
    assert len(layout)==len(tasks)
    for worker,part in enumerate(assignments):
        assert {tasks[i]['kind'] for i in part}=={'pair','face'}
        assert all(layout[i]==[worker,j,tasks[i]['kind']] for j,i in enumerate(part))
        assert [tasks[i]['cost'] for i in part]==sorted([tasks[i]['cost'] for i in part],reverse=True)
    assert assignments==api().dispatch(tasks,workers)[0]

def test_batch_gate_refuses_unqualified_context(monkeypatch):
    from test_planning_batch import setup,tasks
    gate,module,counts,ledger=setup(monkeypatch,2)
    with pytest.raises(QualificationDecline):gate.run_tasks(tasks(module,1),context=object())
    assert gate.poisoned and gate.pending_count==0

def test_face_worker_matches_serial_domain_bounds_setup():
    import json
    from anygeometry.material_arrangement import MaterialDomain,arrange_material
    from anygeometry.batch_intersections import _domain_bounds
    from anygeometry._planning_values import encode_value
    module=api();model,_=_fixture();context=module.WorkerModel(a.freeze_model(from_dict(to_dict(model))))
    face=next(iter(context.model.faces));domain=MaterialDomain.from_model(context.model,face);_domain_bounds(domain)
    tol=context.model.tolerance.effective_length(5.);area=context.model.tolerance.effective_area(5.)
    expected=encode_value(arrange_material(domain,(),points=(),tolerance=tol,area_tolerance=area,cancellation_check=lambda:False))
    task=dict(kind='face',face=face,inputs=encode_value(((),())),tolerance=tol,area_tolerance=area,seam='baseline')
    assert json.loads(module.stage_task(context,task))==expected

def test_face_worker_preserves_checked_support_derived_state(monkeypatch):
    import json
    from dataclasses import replace
    from anygeometry.material_arrangement import MaterialArrangement
    from anygeometry._planning_values import encode_value
    module=api();model,_=_fixture();context=module.WorkerModel(a.freeze_model(from_dict(to_dict(model))))
    face=next(iter(context.model.faces));support=replace(context.model.faces[face].surface);_=support.normal
    def arrange(domain,*args,**kwargs):return MaterialArrangement(face,(),(),0.,support=domain.support)
    monkeypatch.setattr(module,'arrange_material',arrange)
    task=dict(kind='face',face=face,inputs=encode_value(((),())),support=encode_value(support),tolerance=1e-9,area_tolerance=1e-12,seam='baseline')
    assert json.loads(module.stage_task(context,task))==encode_value(MaterialArrangement(face,(),(),0.,support=support))
