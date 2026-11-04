"""Small real host+owned-tree admission, raw identity, and cleanup checks."""
from _planning_numpy_profile_contract import numpy_profile_test
from _planning_platform_contract import windows_owned_test, observe_plan_calls, assert_unsupported512_terminal
import json
import sys

import pytest

from anygeometry import batch_intersections as bi
from anygeometry._planning_canaries import _fixture
from anygeometry._planning_host import StageControl
from anygeometry._planning_memory import MemoryBudget, observe_memory
from anygeometry.serialization import from_dict, to_dict





@pytest.mark.parametrize('reuse', [False, True])
@windows_owned_test
@numpy_profile_test
def test_real_host_and_tree_budget_preserves_small_fixture(tmp_path, monkeypatch, reuse, request):
    original, _ = _fixture()
    doc = to_dict(original)
    reference = from_dict(doc)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING','0')
    serial = bi.plan_intersections(reference, tuple(reference.faces), policy='connect')
    bi.apply_intersections(reference, serial, policy='connect')
    expected = (serial.content_checksum, to_dict(reference)['checksum']['value'])

    def observed(plan):
        value = from_dict(doc)
        bi.apply_intersections(value, plan, policy='connect')
        return plan.content_checksum, to_dict(value)['checksum']['value']

    native = observe_memory()
    assert type(native['host_peak_bytes']) is int
    assert 0 < native['host_private_bytes'] <= native['host_peak_bytes']
    # Fixture-only declared reserve; neither a measured full-model forecast nor
    # an aggregate allocation. P/W retains the already checked 512 MiB cap.
    host_estimate = native['host_peak_bytes'] + 256 * 1024**2
    tree_cap = 512 * 1024**2
    aggregate = (5 * (host_estimate + tree_cap) + 3) // 4
    budget = MemoryBudget(host_estimate, tree_cap, aggregate, (1, 1))
    source = from_dict(doc)
    control = StageControl(tmp_path/'geometry', expected_ci=expected, observed_ci=observed,
                           workers=2, timeout=80, reuse_workers=reuse, memory_budget=budget)
    monkeypatch.setenv('ANYGEOMETRY_EXPERIMENTAL_PARALLEL_PLANNING', '1')
    from anygeometry._planning_executor import PlannerCleanupError
    with observe_plan_calls(enabled=True) as calls:
        try:
            plan = bi.plan_intersections(source, tuple(source.faces), policy='connect', _parallel_control=control)
        except PlannerCleanupError as error:
            assert_unsupported512_terminal(error, reuse=reuse, control=control, source=source,
                                           document=doc, calls=calls, request=request)
            return
    assert control.mode == 'parallel' and control.ci_verified
    assert plan.content_checksum == expected[0] and to_dict(source) == doc
    assert observed(plan) == expected
    report = control.memory_report
    assert report['outcome'] == 'parallel' and report['scope'] == 'planning-and-CI'
    assert report['required_with_headroom_bytes'] == aggregate == report['effective_limit_bytes']
    assert report['host_peak_estimate_bytes'] == host_estimate
    assert 0 < report['observed_host_peak_bytes'] <= host_estimate
    assert report['samples'] > 2 and report['terminal_reason'] is None
    receipts = []
    for name in (['session'] if reuse else ['pair', 'face']):
        receipt = json.loads((control.directory/name/'cleanup.json').read_text())
        assert receipt['verified'] and receipt['active_processes'] == 0
        assert receipt['memory']['limit_bytes'] == tree_cap
        assert 0 < receipt['memory']['peak_job_bytes'] <= tree_cap
        receipts.append(receipt)
    assert [item['stage'] for item in control.reports] == ['pair', 'face']
    assert all(len(item['worker_reports']) == 2 for item in control.reports)
    (tmp_path/'host-memory-evidence.json').write_text(json.dumps(dict(
        admission=report, retirement=receipts, expected_ci=expected,
        actual_ci=observed(plan), input_unchanged=to_dict(source) == doc), indent=2))

    request.node.user_properties[:]=[(key,value) for key,value in request.node.user_properties if key!='execution_scope']
    request.node.user_properties.extend((('execution_scope','unsupported512-single-fit-observation'),
        ('resource_configuration','unsupported512-safe-fit-observation'),('reuse_workers',reuse),
        ('parallel_qualified',False),('memory_capacity_qualified',False)))
