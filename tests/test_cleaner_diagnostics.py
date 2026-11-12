"""Full diagnostic coverage, provenance, deterministic export and cancellation."""
from copy import deepcopy
import json

import pytest

from anygeometry.cleaner import CleanerOptions, propose_repair
from anygeometry.cleaner_diagnostics import diagnose_geometry
from anygeometry.errors import GeometryError
from anygeometry.model import GeometryModel
from anygeometry.strict_audit import strict_audit


def square():
    model = GeometryModel()
    vertices = model.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)))
    model.add_plate(vertices)
    return model


def test_complete_clean_diagnosis_is_deterministic_and_lossless_nonmutating():
    model = square()
    before, revision = model.design_snapshot(), model.revision
    report = diagnose_geometry(model)
    second = diagnose_geometry(model)
    assert report.complete and report.clean and report.audit_clean
    assert report.audit.to_dict() == strict_audit(model).to_dict()
    first_dict, second_dict = report.to_dict(), second.to_dict()
    first_dict.pop('timings_seconds')
    second_dict.pop('timings_seconds')
    assert first_dict == second_dict
    assert json.loads(json.dumps(report.to_dict(), allow_nan=False))['complete']
    assert '## Checks' in report.to_markdown()
    assert '## Source, options and counts' in report.to_markdown()
    assert '"short_edge"' in report.to_markdown()
    assert '## Owner audit provenance' in report.to_markdown()
    assert report.audit.checksum in report.to_markdown()
    assert model.revision == revision
    assert model.design_snapshot()['topology']['ids'] == before['topology']['ids']


def test_complete_is_distinct_from_defect_free_and_owner_details_preserved():
    model = GeometryModel()
    points = model.add_points(((0, 0, 0), (2, 0, 0), (1, -1, 0), (1, 1, 0)))
    model.add_line(points[0], points[1])
    model.add_line(points[2], points[3])
    report = diagnose_geometry(model)
    assert report.complete and not report.clean
    assert 'edge_crossing' in {item.kind for item in report.findings}
    raw = report.to_dict()['owner_audit']
    owned = [item.owner_issue for item in report.findings if item.owner_issue is not None]
    assert sorted(json.dumps(item, sort_keys=True) for item in owned) == sorted(json.dumps(item, sort_keys=True) for item in raw['issues'])
    assert all(json.dumps(item, sort_keys=True) in report.to_markdown() for item in raw['issues'])


def test_full_diagnosis_keeps_findings_beyond_fast_review_cap():
    model = GeometryModel()
    model.add_points((i * 1e-6, 0, 0) for i in range(5))
    report = diagnose_geometry(model, CleanerOptions(max_vertex_pairs=1))
    assert len([item for item in report.findings if item.kind == 'near_vertices']) == 10
    assert report.quality.complete
    assert report.counts['findings'] == len(report.to_dict()['findings'])


def test_diagnostic_quality_is_not_repair_authority():
    model = GeometryModel()
    model.add_point(0, 0)
    report = diagnose_geometry(model)
    with pytest.raises(GeometryError, match='forged'):
        propose_repair(model, report.quality)


def test_point_edge_and_planar_face_near_gaps_are_measured():
    model = square()
    edge_point = model.add_point(.5, -2e-6)
    face_point = model.add_point(.5, .5, 3e-6)
    report = diagnose_geometry(model)
    assert report.complete
    gaps = [item for item in report.findings if item.kind.startswith('near_point')]
    assert any(item.kind == 'near_point_edge' and item.refs[0].id == edge_point for item in gaps)
    assert any(item.kind == 'near_point_face' and item.refs[0].id == face_point for item in gaps)
    assert all(0 < item.value <= item.tolerance for item in gaps)


def test_unsupported_curves_and_coordinate_frame_remain_incomplete():
    model = GeometryModel()
    points = model.add_points(((0, 0, 0), (.5, .5, 0), (1, 0, 0)))
    model.add_arc(*points)
    report = diagnose_geometry(model)
    assert not report.complete and not report.clean
    assert any(check.status == 'unsupported' for check in report.checks)
    model.set_document_settings(units='mm')
    report = diagnose_geometry(model)
    assert not report.complete and report.quality is None
    assert report.audit is not None
    assert 'unsupported_coordinates' in {item.kind for item in report.findings}


def test_cancelled_report_has_every_declared_check_and_partial_results():
    report = diagnose_geometry(square(), cancelled=lambda: True)
    assert not report.complete and not report.clean
    assert any(check.status == 'cancelled' for check in report.checks)
    assert {'quality', 'strict_audit', 'near_gaps'} <= {check.name for check in report.checks}


def test_cancellation_after_audit_retains_raw_provenance():
    state = {'cancel': False}
    def progress(phase, fraction):
        if phase == 'strict_audit':
            state['cancel'] = True
    report = diagnose_geometry(square(), progress=progress, cancelled=lambda: state['cancel'])
    assert report.audit is not None and not report.complete
    assert any(check.status == 'cancelled' for check in report.checks)


def test_audit_exception_is_structured_failure(monkeypatch):
    import anygeometry.cleaner_diagnostics as diagnostics
    def fail(model):
        raise RuntimeError('injected audit failure')
    monkeypatch.setattr(diagnostics, 'strict_audit', fail)
    report = diagnostics.diagnose_geometry(square())
    assert not report.complete and not report.clean
    assert any('injected audit failure' in finding.message for finding in report.findings)


def test_source_change_during_diagnosis_is_explicitly_stale():
    model = square()
    def progress(phase, fraction):
        if phase == 'finished':
            model.add_point(10, 10)
    report = diagnose_geometry(model, progress=progress)
    assert not report.complete
    assert any(check.name == 'source_consistency' and check.status == 'failed' for check in report.checks)


def test_interior_skew_segment_gap_is_detected_without_nearby_endpoints():
    model = GeometryModel()
    p = model.add_points(((-1, 0, 0), (1, 0, 0), (0, -1, 2e-6), (0, 1, 2e-6)))
    model.add_line(p[0], p[1])
    model.add_line(p[2], p[3])
    report = diagnose_geometry(model)
    assert report.complete
    gaps = [item for item in report.findings if item.kind == 'near_edges']
    assert len(gaps) == 1 and gaps[0].value == pytest.approx(2e-6)


def test_planar_region_gap_without_contained_vertex_projections():
    model = GeometryModel()
    for points in (((-2, -.1, 0), (2, -.1, 0), (2, .1, 0), (-2, .1, 0)),
                   ((-.1, -2, 2e-6), (.1, -2, 2e-6), (.1, 2, 2e-6), (-.1, 2, 2e-6))):
        model.add_plate(model.add_points(points))
    report = diagnose_geometry(model)
    assert report.complete
    gaps = [item for item in report.findings if item.kind == 'near_edges']
    assert gaps and any(any(ref.kind == 'face' for ref in item.refs) for item in gaps)
    assert min(item.value for item in gaps) == pytest.approx(2e-6)


def test_pending_feature_intent_cannot_be_complete_or_clean():
    model = GeometryModel()
    model.features.capture_baseline(model)
    model.features.append('geometry.point', name='pending point', parameters={'position': (1, 2, 0)})
    revision = model.revision
    report = diagnose_geometry(model)
    assert not report.complete and not report.clean
    assert any(item.kind == 'unresolved_feature' for item in report.findings)
    assert model.revision == revision


def test_quality_failure_does_not_skip_owner_audit_or_near_gaps(monkeypatch):
    import anygeometry.cleaner_diagnostics as diagnostics
    model = GeometryModel()
    points = model.add_points(((0, 0, 0), (2, 0, 0), (1, -1, 0), (1, 1, 0), (.5, 2e-6, 0)))
    model.add_line(points[0], points[1])
    model.add_line(points[2], points[3])
    def fail(*args):
        raise RuntimeError('injected quality failure')
    monkeypatch.setattr(diagnostics, 'review_geometry', fail)
    report = diagnostics.diagnose_geometry(model)
    assert not report.complete
    assert report.audit is not None
    assert any(item.kind == 'edge_crossing' for item in report.findings)
    assert any(item.kind == 'near_point_edge' for item in report.findings)
    assert any(check.name == 'quality' and check.status == 'failed' for check in report.checks)


def test_audit_failure_does_not_skip_supported_near_gaps(monkeypatch):
    import anygeometry.cleaner_diagnostics as diagnostics
    model = square()
    model.add_point(.5, -.000002)
    def fail(*args):
        raise RuntimeError('injected independent audit failure')
    monkeypatch.setattr(diagnostics, 'strict_audit', fail)
    report = diagnostics.diagnose_geometry(model)
    assert not report.complete
    assert any(item.kind == 'near_point_edge' for item in report.findings)
    assert any(check.name == 'near_gaps' and check.status == 'completed' for check in report.checks)


def test_disconnected_regions_require_review_and_are_never_clean():
    model = GeometryModel()
    p = model.add_points(((0, 0, 0), (1, 0, 0), (3, 0, 0), (4, 0, 0)))
    model.add_line(p[0], p[1])
    model.add_line(p[2], p[3])
    report = diagnose_geometry(model)
    regions = [item for item in report.findings if item.kind == 'disconnected_region']
    assert regions and all(item.severity == 'warning' for item in regions)
    assert report.complete and not report.clean


def _force_small_overlap(monkeypatch):
    import anygeometry.cleaner_diagnostics as diagnostics
    import anygeometry.audit_parallel as parallel
    original = parallel.strict_audit_parallel
    def forced(model, **kwargs):
        kwargs['workers'] = 2
        kwargs['min_candidates'] = 0
        return original(model, **kwargs)
    monkeypatch.setattr(diagnostics, '_parallel_candidate_model', lambda model: True)
    monkeypatch.setattr(parallel, 'strict_audit_parallel', forced)


def test_overlapped_full_diagnosis_matches_sequential_all_findings(monkeypatch):
    model = square()
    model.add_point(.5, -.000002)
    sequential = diagnose_geometry(model).to_dict()
    _force_small_overlap(monkeypatch)
    parallel = diagnose_geometry(model).to_dict()
    assert parallel['source']['audit_execution']['near_gaps_overlapped']
    assert parallel['source']['audit_execution']['effective_workers'] == 2
    for document in (sequential, parallel):
        document.pop('timings_seconds')
        document['source'].pop('audit_execution')
    assert parallel == sequential


def test_overlapped_gap_failure_keeps_strict_report_and_retires_thread(monkeypatch):
    import threading
    import anygeometry.cleaner_diagnostics as diagnostics
    _force_small_overlap(monkeypatch)
    def fail(*args):
        raise TimeoutError('injected gap timeout failure')
    monkeypatch.setattr(diagnostics, '_gap_findings', fail)
    report = diagnose_geometry(square())
    assert not report.complete and report.audit is not None
    assert any(check.name == 'near_gaps' and check.status == 'failed' for check in report.checks)
    assert not any(thread.name.startswith('geometry-near-gaps') for thread in threading.enumerate())


def test_overlapped_gap_cancellation_joins_background_thread(monkeypatch):
    import threading
    import time
    import anygeometry.cleaner_diagnostics as diagnostics
    _force_small_overlap(monkeypatch)
    began = threading.Event()
    ended = threading.Event()
    def wait_gap(model, threshold, checkpoint):
        began.set()
        try:
            while True:
                checkpoint()
                time.sleep(.005)
        finally:
            ended.set()
    monkeypatch.setattr(diagnostics, '_gap_findings', wait_gap)
    report = diagnose_geometry(square(), cancelled=began.is_set)
    assert not report.complete and not report.clean
    assert ended.is_set()
    assert any(check.status == 'cancelled' for check in report.checks)
    assert not any(thread.name.startswith('geometry-near-gaps') for thread in threading.enumerate())
