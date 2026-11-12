"""Exact serial/parallel audit parity and bounded subprocess retirement."""
import multiprocessing as mp
import time

import pytest

from anygeometry.audit_parallel import strict_audit_parallel
from anygeometry.model import GeometryModel
from anygeometry.strict_audit import strict_audit


def fixture(kind):
    model = GeometryModel()
    if kind == 'healthy':
        model.add_plate(model.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))))
    elif kind == 'defective':
        p = model.add_points(((0, 0, 0), (2, 0, 0), (1, -1, 0), (1, 1, 0), (1, 0, 0), (3, 0, 0)))
        for a, b in ((0, 1), (2, 3), (4, 5)):
            model.add_line(p[a], p[b])
    else:
        p = model.add_points(((0, 0, 0), (.5, 1, 0), (1, 0, 0), (0, .5, 0), (1, .5, 0)))
        model.add_spline(p[0], [p[1]], p[2])
        model.add_line(p[3], p[4])
    return model


@pytest.mark.parametrize('kind', ['healthy', 'defective', 'curved'])
def test_complete_raw_report_is_identical(kind):
    model = fixture(kind)
    execution = {}
    before = {child.pid for child in mp.active_children()}
    parallel = strict_audit_parallel(model, workers=2, min_candidates=0, execution=execution)
    assert parallel.to_dict() == strict_audit(model).to_dict()
    assert execution['route'] == 'process_pool' and execution['effective_workers'] == 2
    assert execution['children_retired']
    assert {child.pid for child in mp.active_children()} == before


def test_small_route_is_honestly_serial():
    execution = {}
    strict_audit_parallel(fixture('healthy'), execution=execution)
    assert execution['route'] == 'serial' and execution['effective_workers'] == 1


def test_cancellation_retires_spawned_children():
    model = fixture('healthy')
    calls = []
    def cancelled():
        calls.append(None)
        return len(calls) >= 3
    before = {child.pid for child in mp.active_children()}
    execution = {}
    result = strict_audit_parallel(model, workers=2, min_candidates=0, cancelled=cancelled, execution=execution)
    assert not result.completed and not result.clean
    assert execution['status'] == 'cancelled' and execution['children_retired']
    assert {child.pid for child in mp.active_children()} == before


def test_codec_failure_is_explicit_non_success(monkeypatch):
    import anygeometry.audit_parallel as parallel
    def fail(model):
        raise RuntimeError('injected codec failure')
    monkeypatch.setattr(parallel, 'capture_model_transfer', fail)
    execution = {}
    report = parallel.strict_audit_parallel(fixture('healthy'), workers=2, min_candidates=0, execution=execution)
    assert not report.completed and not report.clean
    assert execution['effective_workers'] == 0
    assert any(issue.code.value == 'check_failed' for issue in report.issues)

@pytest.mark.parametrize('invalid', [False, True])
def test_structural_attachment_and_junction_report_parity(invalid):
    from anygeometry.structural import AttachmentKind, AttachmentTargetKind, ParameterRange, JunctionKind, JunctionMemberUse
    model = GeometryModel()
    face = model.add_plate(model.add_points(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0))))
    vertices = model.add_points(((.2, 1, 0), (1.8, 1, 0), (.2, 1, 1), (1.8, 1, 1)))
    edge = model.add_line(vertices[0], vertices[1])
    other = model.add_line(vertices[2], vertices[3])
    member, second = model.add_member([edge]), model.add_member([other])
    model.add_attachment(member, AttachmentKind.MEMBER_THROUGH_FACE if invalid else AttachmentKind.MEMBER_ON_FACE,
        AttachmentTargetKind.FACE, face,
        ParameterRange.point(.5) if invalid else ParameterRange(0, 1),
        (ParameterRange.point(.5) if invalid else ParameterRange(.1, .9), ParameterRange.point(.5)))
    model.add_junction(JunctionKind.CROSSING,
        (JunctionMemberUse(member, ParameterRange.point(.5)), JunctionMemberUse(second, ParameterRange.point(.5))))
    execution = {}
    parallel = strict_audit_parallel(model, workers=2, min_candidates=0, execution=execution)
    assert parallel.to_dict() == strict_audit(model).to_dict()
    assert execution['children_retired']


def test_nonmanifold_report_parity():
    model = GeometryModel()
    first, second, upper, lower, transverse = model.add_points(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)))
    common = model.add_line(first, second)
    for third in (upper, lower, transverse):
        model.add_face((common, model.add_line(second, third), model.add_line(third, first)))
    assert strict_audit_parallel(model, workers=2, min_candidates=0).to_dict() == strict_audit(model).to_dict()


def _daemon_probe(connection):
    execution = {}
    report = strict_audit_parallel(fixture('healthy'), workers=2, min_candidates=0, execution=execution)
    connection.send((report.completed, execution))
    connection.close()


def _failed_batch(connection, *args):
    connection.send(('error', 'InjectedClassifierFailure', 'expected classifier failure'))
    connection.close()


def test_daemonic_parent_refuses_instead_of_claiming_parallel_work():
    context = mp.get_context('spawn')
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_daemon_probe, args=(writer,), daemon=True)
    process.start()
    writer.close()
    try:
        assert reader.poll(10)
        completed, execution = reader.recv()
        assert not completed
        assert execution['status'] == 'refused_daemonic_parent'
        assert execution['effective_workers'] == 0
    finally:
        process.join(timeout=5)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        reader.close()
    assert not process.is_alive()


def test_child_classifier_failure_is_non_success_and_retires_children(monkeypatch):
    import anygeometry.audit_parallel as parallel
    monkeypatch.setattr(parallel, '_classify_batch', _failed_batch)
    execution = {}
    before = {child.pid for child in mp.active_children()}
    report = parallel.strict_audit_parallel(fixture('healthy'), workers=2, min_candidates=0, execution=execution)
    assert not report.completed and not report.clean
    assert execution['status'] == 'failed' and execution['children_retired']
    assert 'InjectedClassifierFailure' in execution['failure']
    assert {child.pid for child in mp.active_children()} == before


def test_bounded_curved_unclassified_report_is_exactly_preserved():
    from anygeometry.predicates import IntersectionQualificationPolicy
    model = GeometryModel()
    p = model.add_points(((-1, 0, 0), (0, 1, 0), (1, 0, 0), (-1, .5, 0), (0, -.5, 0), (1, .5, 0)))
    model.add_spline(p[0], [p[1]], p[2])
    model.add_spline(p[3], [p[4]], p[5])
    policy = IntersectionQualificationPolicy(max_boxes_per_pair=1, max_subdivision_depth=1)
    serial = strict_audit(model, qualification=policy)
    assert not serial.clean and serial.metrics.unclassified_count > 0
    assert strict_audit_parallel(model, workers=2, min_candidates=0, qualification=policy).to_dict() == serial.to_dict()


def test_python311_cpu_count_fallback_keeps_small_route_serial(monkeypatch):
    import os
    monkeypatch.delattr(os, 'process_cpu_count', raising=False)
    execution = {}
    report = strict_audit_parallel(fixture('healthy'), execution=execution)
    assert report.clean
    assert execution['route'] == 'serial' and execution['effective_workers'] == 1


def test_root_topology_audit_is_once_and_reuse_is_explicit(monkeypatch):
    from anygeometry.audit_parallel import _ParallelState
    original = _ParallelState.audit_topology
    calls = []
    def audit(self):
        calls.append((str(self.model.model_id), self.model.revision))
        return original(self)
    monkeypatch.setattr(_ParallelState, 'audit_topology', audit)
    model = fixture('healthy')
    execution = {}
    report = strict_audit_parallel(model, workers=2, min_candidates=0, execution=execution)
    assert report.to_dict() == strict_audit(model).to_dict()
    assert calls == [(str(model.model_id), model.revision)]
    assert execution['validation_reused'] == 'root_topology_and_structural_audit'


def test_reported_child_blas_settings_are_observed_or_explicitly_unknown():
    execution = {}
    report = strict_audit_parallel(fixture('healthy'), workers=2, min_candidates=0, execution=execution)
    assert report.clean and execution['blas_threads_requested'] == 1
    for child in execution['child_threadpools']:
        assert child['status'] in ('available', 'unknown')
        for library in child['libraries']:
            assert '/' not in library['library'] and '\\' not in library['library']
    if execution['blas_threads'] != 'unknown':
        assert execution['blas_threads'] >= 1
        assert all(child['status'] == 'available' for child in execution['child_threadpools'])


@pytest.mark.parametrize('workers', [0, 9, True, 1.5])
def test_worker_bound_is_explicitly_one_to_eight(workers):
    with pytest.raises(ValueError, match=r'\[1, 8\]'):
        strict_audit_parallel(fixture('healthy'), workers=workers)


def test_eight_worker_resource_route_preserves_exact_serial_report():
    import os
    model = fixture('healthy')
    execution = {}
    report = strict_audit_parallel(model, workers=8, min_candidates=0, execution=execution)
    assert report.to_dict() == strict_audit(model).to_dict()
    assert execution['requested_workers'] == 8
    assert execution['effective_workers'] == min(8, getattr(os, 'process_cpu_count', os.cpu_count)() or 1)
    assert execution['children_retired']
