"""PP1i review regressions, preserving every earlier acceptance test."""
import hashlib
import importlib
import sys
import types
import pytest
from anygeometry import _planning_admission as a
from anygeometry import _planning_ledger as l
from anygeometry import _planning_qualification as q
from test_planning_batch import setup, tasks
from test_planning_admission import invoke, pin


@pytest.mark.parametrize('workers', [2, 4, 5, 8, 10])
def test_sampler_covers_workers_classes_and_last(workers):
    layout = tuple((i % workers, i // workers, 'pair' if i % 2 else 'face') for i in range(600))
    plan = a.stratified_samples(layout, 'a'*64)
    indices = plan['indices']
    assert {layout[i][0] for i in indices} == set(range(workers))
    assert {layout[i][2] for i in indices} == {'pair', 'face'}
    for worker in range(workers):
        assigned = [i for i, row in enumerate(layout) if row[0] == worker]
        assert assigned[-1] in indices
        assert sum(i in indices for i in assigned) >= 2
        for kind in {layout[i][2] for i in assigned}:
            assert any(i in indices and layout[i][2] == kind for i in assigned)
    assert plan == a.stratified_samples(layout, 'a'*64)
    assert plan['offsets']


def test_sampler_uses_actual_positions_not_parent_order():
    layout = ((0, 3, 'pair'), (0, 0, 'face'), (0, 2, 'face'), (0, 1, 'pair'))
    plan = a.stratified_samples(layout, '1'*64)
    assert 0 in plan['indices'] and 2 in plan['indices']


@pytest.mark.parametrize('layout,digest', [(((0, 0, 'pair'), (0, 0, 'face')), 'a'*64),
    (((0, 2, 'pair'),), 'a'*64), (((True, 0, 'pair'),), 'a'*64),
    (((0, 0, 'unknown'),), 'a'*64), (((0, 0, 'pair'),), '')])
def test_bad_sampling_metadata_refuses(layout, digest):
    with pytest.raises(q.QualificationDecline): a.stratified_samples(layout, digest)


def test_sampler_integration_catches_last_worker_task():
    layout = tuple((i % 4, i // 4, 'pair' if i % 2 else 'face') for i in range(40))
    result = invoke(task_layout=layout, plan_digest='b'*64,
        parallel_tasks=lambda items: tuple('wrong' if i == 39 else str(i) for i in items))
    assert result['mode'] == 'serial' and 'sample' in result['decline_reason']
    assert 39 in result['sampling']['indices']


def test_mandatory_ci_does_not_start_without_both_arguments():
    result = invoke(require_ci=True, start_workers=lambda: pytest.fail('unqualified CI started'))
    assert result['mode'] == 'serial' and 'CI' in result['decline_reason']


def test_mandatory_ci_and_stratified_success():
    layout = tuple((i % 2, i // 2, 'pair' if i % 2 else 'face') for i in range(40))
    result = invoke(require_ci=True, task_layout=layout, plan_digest='c'*64,
                    expected_ci=('plan', 'apply'), observed_ci=lambda: ('plan', 'apply'))
    assert result['mode'] == 'parallel' and result['ci_verified']
    assert set(result['sampling']['workers']) == {0, 1}


def test_own_trusted_annotation_dict_mutation_is_checked():
    from anygeometry import exact_curves
    # Unit scope: the genuine annotation owner, not all unrelated loaded hosts.
    ledger = l.loaded_geometry_ledger(trusted_image=True, modules=(exact_curves,))
    ledger.validate()
    store = ledger.objects[l._TRUSTED_IMAGE_STORES[-1]]
    assert ledger.rows[l._TRUSTED_IMAGE_STORES[-1]]['action'] == 'trusted-runtime-image'
    identity = l.trusted_store_identity(l._TRUSTED_IMAGE_STORES[-1], store)
    assert identity['content_check'] == 'structural' and 'source' in identity['basis']
    from anygeometry.exact_curves import _BranchConstants
    if sys.version_info < (3, 14):
        assert store is vars(_BranchConstants)['__annotations__']
    else:
        assert store is _BranchConstants.__annotate_func__.__closure__[0].cell_contents
    store['pp1i_extra'] = int
    try:
        with pytest.raises(q.QualificationDecline): ledger.validate()
    finally:
        del store['pp1i_extra']


def test_trusted_namespace_cannot_be_arbitrary_dict():
    path = l._TRUSTED_IMAGE_STORES[2]
    with pytest.raises(q.QualificationDecline):
        l.trusted_store_identity(path, {'__name__': 'numpy._core.fromnumeric'})


def test_trusted_record_identifies_namespaces_and_source_basis():
    record = q.trusted_runtime_record(pin())
    stores = record['stores']
    assert stores[0]['namespace'] == 'builtins'
    assert stores[2]['namespace'] == 'numpy._core.fromnumeric'
    assert 'source' in stores[-1]['basis'] and 'structural' in stores[-1]['basis']


@pytest.mark.parametrize('mutation', ['rebind', 'new-store'])
def test_mid_chunk_live_mutation_discards_assignment(monkeypatch, mutation):
    gate, module, _, ledger = setup(monkeypatch, 4)
    execute = gate._execute
    def mutate(*args):
        # Keep the fixture's unrelated warm-cache assertion out of this census
        # test. The new root remains rebound throughout the chunk.
        if mutation == 'rebind' and module._COUNT:
            module._MEMO.clear()
        result = execute(*args)
        if module._COUNT == 1:
            if mutation == 'rebind': module._MEMO = {'new': 1}
            else: module.NEW_STORE = {'new': 1}
        return result
    monkeypatch.setattr(gate, '_execute', mutate)
    with pytest.raises(q.QualificationDecline): gate.run_tasks(tasks(module, 3))
    assert gate.closed and gate.poisoned and gate.pending_count == 0


def test_task_error_preserves_exception_and_no_serial_replay(monkeypatch):
    gate, module, _, ledger = setup(monkeypatch, 4)
    error = ValueError('genuine task error')
    def bad(*args): raise error
    monkeypatch.setattr(gate, '_execute', bad)
    with pytest.raises(ValueError) as caught:
        invoke(parallel_tasks=lambda _:gate.run_tasks(tasks(module, 3)),
               serial_task=lambda _:pytest.fail('task error replayed'))
    assert caught.value is error and gate.pending_count == 0 and gate.poisoned


@pytest.mark.parametrize('change', ['settings', 'threads-only'])
def test_real_boundary_settings_and_threads_only(monkeypatch, tmp_path, change):
    b = importlib.import_module('anygeometry._planning_batch')
    source = tmp_path/'fixture.py'; source.write_text('x=1')
    module = types.ModuleType('fixture'); module.__file__ = str(source)
    registered = {'fixture': module}
    module_pin = q.pin_module(module, runtime={})
    pool = dict(filepath=str(source), num_threads=1)
    expected = dict(modules={'fixture': module_pin}, files={str(source): module_pin['sha256']},
        settings={'PP1I_FIXTURE': None}, blas=[dict(pool, sha256=module_pin['sha256'])])
    monkeypatch.delenv('PP1I_FIXTURE', raising=False)
    monkeypatch.setattr(b, 'sys', types.SimpleNamespace(modules=registered))
    monkeypatch.setattr(q, 'runtime_files', lambda: {})
    monkeypatch.setattr(b, 'threadpool_info', lambda: [dict(pool)])
    b.verify_boundary(expected, registered)
    if change == 'settings': monkeypatch.setenv('PP1I_FIXTURE', 'changed')
    else: pool['num_threads'] = 2
    with pytest.raises(q.QualificationDecline): b.verify_boundary(expected, registered)


def test_opt_in_selector_is_not_a_numerical_setting():
    assert 'ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING' not in q.NUMERICAL_SETTINGS


def test_opt_in_absent_baseline_then_enabled_capture(monkeypatch):
    # Capture the real settings path with only image discovery isolated from
    # pytest's intentionally mutable loaded modules. Verify_environment is real.
    fake = types.ModuleType('fixture')
    monkeypatch.setattr(q, 'sys', types.SimpleNamespace(modules={'fixture':fake},
        executable=sys.executable, version=sys.version, implementation=sys.implementation, platform=sys.platform))
    monkeypatch.setattr(q, 'pin_module', lambda *args, **kwargs:dict(kind='builtin', runtime_files={}))
    monkeypatch.setattr(q, 'runtime_files', lambda:{})
    monkeypatch.setattr(q, '_hash', lambda path:'a'*64)
    import threadpoolctl
    monkeypatch.setattr(threadpoolctl,'threadpool_info',lambda:[dict(filepath='fixture.dll',num_threads=1)])
    monkeypatch.delenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING',raising=False)
    before=q.capture_environment()
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','1')
    after=q.capture_environment()
    q.verify_environment(before,after)


def test_optional_coalescing_retains_final_check(monkeypatch):
    gate, module, counts, ledger = setup(monkeypatch, 2)
    gate.coalesce_boundaries = True
    before = counts['boundary']
    assert gate.run_tasks(tasks(module, 5)) == ('1','2','3','4','5')
    assert counts['boundary'] - before == 4  # one start plus three post checks
