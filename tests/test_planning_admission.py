"""PP1g risk-accepted qualification, not a purity proof."""
from copy import deepcopy
import importlib
import pytest
from anygeometry import _planning_ledger as l
from anygeometry import _planning_qualification as q

STORES = (
 'anygeometry.backend._RUNTIME_PYTHON_GLOBALS.__builtins__',
 'anygeometry.backend._RUNTIME_TRUSTED_NUMPY[12][1]._finfo_cache',
 'anygeometry.backend._RUNTIME_POINT_EDGES[3][0]',
 'anygeometry.backend._RUNTIME_POINT_EDGES[33][0]',
 'anygeometry.backend._RUNTIME_POINT_EDGES[52][0]',
 'anygeometry.backend._RUNTIME_POINT_EDGES[61][0]',
 'anygeometry.exact_curves._BranchConstants.__annotate_func__.cell[0]',
)

def api(): return importlib.import_module('anygeometry._planning_admission')

def pin():
    return dict(schema='PP1b-v1', versions={'numpy':'fixture','scipy':'fixture'},
        python={'version':'fixture','executable':'python.exe'}, cpu={}, settings={}, types=[], reset_registry={},
        blas=[dict(num_threads=1, filepath='blas.dll', sha256='d'*64)],
        modules={'numpy._core.fixture':dict(kind='native',path='numpy.pyd',sha256='a'*64),
                 'scipy.fixture':dict(kind='native',path='scipy.pyd',sha256='b'*64),
                 'sys':dict(kind='builtin',runtime_files={'python.dll':'c'*64})},
        files={'numpy.pyd':'a'*64,'scipy.pyd':'b'*64,'python.dll':'c'*64,
               'python.exe':'e'*64,'blas.dll':'d'*64})

def invoke(**changes):
    arguments=dict(tasks=tuple(range(40)), start_workers=lambda:None,
        stop_workers=lambda:None, parallel_tasks=lambda tasks:tuple(str(i) for i in tasks),
        serial_task=str, opt_in=True)
    arguments.update(changes)
    return api().run_qualified(**arguments)

def test_trusted_image_lists_exact_stores_and_limit():
    record=q.trusted_runtime_record(pin())
    assert tuple(item['path'] for item in record['stores'])==STORES
    assert all('not purity-proven' in item['basis'] for item in record['stores'])
    assert 'RISK-ACCEPTED QUALIFICATION' in record['qualification']
    assert record['binary_hashes']['numpy.pyd']=='a'*64


def test_trusted_ledger_is_explicit_and_has_no_seven_declines():
    strict=l.loaded_geometry_ledger()
    accepted=l.loaded_geometry_ledger(trusted_image=True)
    assert all(strict.rows[path]['action']=='decline' for path in STORES)
    assert all(accepted.rows[path]['action']=='trusted-runtime-image' for path in STORES)
    assert accepted.resets.keys()==strict.resets.keys()


@pytest.mark.parametrize('change', ['numpy_binary','extra_module'])
def test_runtime_mismatch_declines_entire_run_to_serial(change):
    expected=pin(); observed=deepcopy(expected)
    if change=='numpy_binary': observed['files']['numpy.pyd']='f'*64
    else: observed['modules']['extra']=dict(kind='native',path='extra.pyd',sha256='f'*64)
    calls=[]
    def start(): q.verify_environment(expected,observed)
    result=invoke(start_workers=start,parallel_tasks=lambda tasks:calls.append('must not run'))
    assert result['mode']=='serial' and result['results']==tuple(map(str,range(40)))
    assert calls==[] and result['decline_reason']


def test_canary_mismatch_declines_to_serial_and_cleans_up():
    from test_planning_review import bound_canary
    import sys
    reference=pin();reference['modules'][bound_canary.__module__]=q.pin_module(sys.modules[bound_canary.__module__])
    worker=q.WorkerGate(reference,stores={},owned_worker=True,observe=lambda:reference)
    stops=[]
    def start(): worker.start({'pair':bound_canary,'face':bound_canary},{'pair':'wrong','face':'wrong'})
    result=invoke(start_workers=start,stop_workers=lambda:stops.append(1))
    assert result['mode']=='serial' and 'canary' in result['decline_reason']
    assert worker.poisoned and stops==[1]


def test_sample_mismatch_discards_all_parallel_results():
    def wrong(tasks): return tuple('WRONG' if i==20 else str(i) for i in tasks)
    result=invoke(parallel_tasks=wrong)
    assert result['mode']=='serial' and result['results']==tuple(map(str,range(40)))
    assert 'sample' in result['decline_reason']


def test_opt_in_defaults_off_without_starting_pool(monkeypatch):
    monkeypatch.delenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING',raising=False)
    def forbidden(): pytest.fail('default-off launched a worker')
    result=invoke(opt_in=None,start_workers=forbidden)
    assert result['mode']=='serial' and result['samples']==0


@pytest.mark.parametrize('n,expected', [(0,()),(1,(0,)),(2,(0,1)),(40,(0,20)),(100,(0,20,40,60,80))])
def test_sampling_is_five_percent_with_minimum_two(n,expected):
    assert api().sample_indices(n)==expected


@pytest.mark.parametrize('which', [0,1])
def test_ci_hook_requires_both_plan_and_apply_equality(which):
    actual=['plan','apply'];actual[which]='wrong'
    result=invoke(expected_ci=('plan','apply'),observed_ci=lambda:tuple(actual))
    assert result['mode']=='serial' and 'CI' in result['decline_reason']


def test_matching_run_stays_parallel_and_counts_samples():
    result=invoke(expected_ci=('plan','apply'),observed_ci=lambda:('plan','apply'))
    assert result['mode']=='parallel' and result['samples']==2
    assert result['results']==tuple(map(str,range(40)))


def test_cleanup_failure_does_not_start_serial_fallback():
    calls=[]
    def stop(): raise q.QualificationDecline('owned cleanup failed')
    with pytest.raises(q.QualificationDecline,match='cleanup'):
        invoke(stop_workers=stop,serial_task=lambda task:calls.append(task))
    assert calls==[]


def test_serial_reference_failure_is_not_retried_as_fallback():
    calls=[]
    def serial(task):
        calls.append(task)
        raise q.QualificationDecline('serial reference refused')
    with pytest.raises(q.QualificationDecline,match='serial reference'):
        invoke(serial_task=serial)
    assert calls==[0]


def test_cancellation_retires_workers_without_serial_retry():
    stops=[];calls=[]
    def cancel(tasks): raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        invoke(parallel_tasks=cancel,stop_workers=lambda:stops.append(1),serial_task=lambda t:calls.append(t))
    assert stops==[1] and calls==[]


@pytest.mark.parametrize('value,mode', [('1','parallel'),('0','serial'),('true','serial')])
def test_effective_environment_opt_in(monkeypatch,value,mode):
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING',value)
    starts=[]
    result=invoke(opt_in=None,start_workers=lambda:starts.append(1))
    assert result['mode']==mode
    assert starts==([1] if mode=='parallel' else [])


def test_trusted_record_cannot_omit_owner_named_stores():
    reference=pin()
    reference['trusted_runtime_image']=q.trusted_runtime_record(reference)
    reference['trusted_runtime_image']['stores'].pop()
    with pytest.raises(q.QualificationDecline,match='trusted runtime'):
        q.verify_environment(reference,deepcopy(reference))
