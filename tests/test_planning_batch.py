"""PP1h risk-accepted qualification, not a purity proof: batch publication gates."""
import hashlib
import importlib
import sys
import types
import pytest
from anygeometry import _planning_ledger as l
from anygeometry import _planning_purity as p
from anygeometry import _planning_qualification as q

PAYLOAD=b'{"args":[],"kwargs":{}}'
DIGEST=hashlib.sha256(PAYLOAD).hexdigest()

def setup(monkeypatch,k,mutate=0):
    api=importlib.import_module('anygeometry._planning_batch')
    m=types.ModuleType('anygeometry.pp1h_fixture');m.__file__=__file__
    exec("__all__=[]\n_MEMO={}\n_COUNT=0\n_MUTATE=0\ndef canary(): return 'ok'\ndef task():\n    global _COUNT\n    assert not _MEMO\n    _MEMO['warm']=1\n    _COUNT+=1\n    if _COUNT==_MUTATE: __all__.append('changed')\n    return str(_COUNT)\n" ,m.__dict__)
    m._MUTATE=mutate;monkeypatch.setitem(sys.modules,m.__name__,m)
    objects={};report=p.inventory((m,),store_objects=objects)
    ledger=l.build_ledger(report,objects,modules=(m,))
    pin=dict(schema='PP1b-v1',modules={m.__name__:q.pin_module(m)},files={},
        python={},versions={},cpu={},settings={},blas=[dict(num_threads=1)],
        types=q.input_type_names(),reset_registry={key:q._type_name(type(value)) for key,value in ledger.resets.items()})
    counts={'environment':0,'boundary':0}
    def observe(): counts['environment']+=1;return pin
    def boundary(expected,registered): counts['boundary']+=1
    monkeypatch.setattr(api,'verify_boundary',boundary)
    gate=api.BatchWorkerGate(pin,stores=ledger.resets,owned_worker=True,
        ledger=ledger,observe=observe,batch_size=k)
    gate.start({'pair':m.canary,'face':m.canary},{'pair':'ok','face':'ok'})
    return gate,m,counts,ledger

def tasks(m,n): return tuple((m.task,PAYLOAD,DIGEST) for _ in range(n))

@pytest.mark.parametrize('k,n,mutation', [(4,4,2),(4,3,3)])
def test_mid_or_final_partial_failure_discards_results(monkeypatch,k,n,mutation):
    gate,m,counts,ledger=setup(monkeypatch,k,mutation)
    published=[]
    with pytest.raises(q.QualificationDecline,match='protected'):
        published.extend(gate.run_tasks(tasks(m,n)))
    assert published==[] and gate.poisoned
    assert gate.discarded_batch_tasks==n and gate.discarded_run_tasks==n
    assert gate.pending_count==0


def test_failed_later_batch_discards_the_whole_candidate_run(monkeypatch):
    gate,m,counts,ledger=setup(monkeypatch,2,3)
    with pytest.raises(q.QualificationDecline): gate.run_tasks(tasks(m,4))
    assert gate.discarded_batch_tasks==2 and gate.discarded_run_tasks==4
    assert gate.pending_count==0


def test_per_task_path_has_no_census_or_hash_between_boundaries(monkeypatch):
    gate,m,counts,ledger=setup(monkeypatch,3)
    before=dict(counts);censuses=len(ledger.census_costs);resets=len(ledger.costs)
    observed=[];execute=gate._execute
    hashes={'files':0,'stores':0};file_hash=q._hash;store_hash=l.content_digest
    phase={'boundary':False};boundary_hashes=[];file_paths=[]
    boundary=gate._boundary
    def counted_boundary():
        assert not phase['boundary']
        phase['boundary']=True
        initial=hashes['files']
        try:
            return boundary()
        finally:
            boundary_hashes.append(hashes['files']-initial)
            phase['boundary']=False
    def counted_file(*args,**kwargs):
        assert phase['boundary'], 'file hashing escaped the batch boundary'
        hashes['files']+=1
        file_paths.append(str(args[0]))
        return file_hash(*args,**kwargs)
    def counted_store(*args,**kwargs):
        hashes['stores']+=1
        return store_hash(*args,**kwargs)
    monkeypatch.setattr(q,'_hash',counted_file)
    monkeypatch.setattr(l,'content_digest',counted_store)
    monkeypatch.setattr(gate,'_boundary',counted_boundary)
    store_observed=[]
    def record(*args):
        observed.append((counts['boundary']-before['boundary'],len(ledger.census_costs)-censuses))
        store_observed.append(hashes['stores'])
        initial=dict(hashes)
        try:
            return execute(*args)
        finally:
            assert hashes==initial, 'hashing occurred during reset/payload/task execution'
    monkeypatch.setattr(gate,'_execute',record)
    assert gate.run_tasks(tasks(m,5))==('1','2','3','4','5')
    assert observed==[(1,1),(1,1),(1,1),(3,3),(3,3)]
    assert counts['boundary']-before['boundary']==4
    # The retained diagnostic ledger pins its anchor/JSON/runtime files at
    # boundaries. Only hashing between those boundaries is forbidden.
    assert len(boundary_hashes)==4 and all(count>0 for count in boundary_hashes)
    assert sum(boundary_hashes)==hashes['files']==len(file_paths)
    assert store_observed==[1,1,1,3,3] and hashes['stores']==4
    assert counts['environment']==1
    assert len(ledger.costs)-resets==5


def test_k_one_retains_before_after_each_task_cadence(monkeypatch):
    gate,m,counts,ledger=setup(monkeypatch,1)
    before=counts['boundary'];censuses=len(ledger.census_costs)
    assert gate.run_tasks(tasks(m,3))==('1','2','3')
    assert counts['boundary']-before==6
    assert len(ledger.census_costs)-censuses==6


def test_end_boundary_runs_for_empty_assignment(monkeypatch):
    gate,m,counts,ledger=setup(monkeypatch,4)
    before=counts['boundary']
    assert gate.run_tasks(())==()
    assert counts['boundary']-before==1


def test_no_second_assignment_or_unchecked_single_task(monkeypatch):
    gate,m,counts,ledger=setup(monkeypatch,4)
    with pytest.raises(q.QualificationDecline): gate.run(m.task)
    assert gate.run_tasks(tasks(m,1))==('1',)
    with pytest.raises(q.QualificationDecline): gate.run_tasks(tasks(m,1))


def test_cancellation_clears_staged_results(monkeypatch):
    gate,m,counts,ledger=setup(monkeypatch,4)
    execute=gate._execute
    def cancel(*args):
        if m._COUNT==1: raise KeyboardInterrupt()
        return execute(*args)
    monkeypatch.setattr(gate,'_execute',cancel)
    with pytest.raises(KeyboardInterrupt): gate.run_tasks(tasks(m,3))
    assert gate.poisoned and gate.pending_count==0


@pytest.mark.parametrize('k',[0,-1,True,1.5])
def test_invalid_batch_size_refuses(monkeypatch,k):
    with pytest.raises(q.QualificationDecline,match='batch size'): setup(monkeypatch,k)


@pytest.mark.parametrize('change',['file','module','threads'])
def test_real_boundary_verifier_refuses_changes(monkeypatch,tmp_path,change):
    b=importlib.import_module('anygeometry._planning_batch')
    source=tmp_path/'uncertified_fixture.py';source.write_text('value=1')
    module=types.ModuleType('fixture');module.__file__=str(source)
    registered={'fixture':module};loaded=dict(registered)
    pin=q.pin_module(module,runtime={})
    expected=dict(modules={'fixture':pin},files={str(source):pin['sha256']},settings={},blas=[])
    monkeypatch.setattr(b,'sys',types.SimpleNamespace(modules=loaded))
    monkeypatch.setattr(q,'runtime_files',lambda:{})
    monkeypatch.setattr(b,'threadpool_info',lambda:[])
    b.verify_boundary(expected,registered)
    if change=='file': source.write_text('value=2')
    elif change=='module': loaded['extra']=types.ModuleType('extra')
    else: monkeypatch.setattr(b,'threadpool_info',lambda:[dict(filepath=str(source),num_threads=2)])
    with pytest.raises(q.QualificationDecline): b.verify_boundary(expected,registered)


def test_batch_failure_reaches_whole_run_serial_fallback(monkeypatch):
    from anygeometry._planning_admission import run_qualified
    gate,m,counts,ledger=setup(monkeypatch,2,3)
    stops=[]
    result=run_qualified(tuple(range(4)),start_workers=lambda:None,stop_workers=lambda:stops.append(1),
        parallel_tasks=lambda _:gate.run_tasks(tasks(m,4)),serial_task=str,opt_in=True)
    assert result['mode']=='serial' and result['results']==('0','1','2','3')
    assert 'protected' in result['decline_reason'] and stops==[1]
    assert gate.discarded_run_tasks==4 and gate.pending_count==0


def test_k_one_matches_legacy_results_and_resets(monkeypatch):
    gate,m,counts,ledger=setup(monkeypatch,1)
    actual=gate.run_tasks(tasks(m,3))
    m._COUNT=0
    legacy=q.WorkerGate(gate.expected,stores=ledger.resets,owned_worker=True,
        ledger=ledger,observe=lambda:gate.expected)
    legacy.start({'pair':m.canary,'face':m.canary},{'pair':'ok','face':'ok'})
    assert actual==tuple(legacy.run(m.task) for _ in range(3))==('1','2','3')
