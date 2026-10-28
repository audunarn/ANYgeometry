"""Same-invocation filtered facts reused only by the exact proof escalation.

Definitions only; numerical execution and measured claims belong to root.
"""
from dataclasses import replace
from types import SimpleNamespace
import json
import time

import pytest

from anygeometry.nurbs_surface_intersection import _certificate as engine, _filtered
from anygeometry.nurbs_surface_intersection import query_nurbs_surface_intersection, IntersectionStatus
from anygeometry.nurbs_surface_intersection._atlas import _graph_region
from anygeometry.nurbs_surface_intersection._intervals import Interval
from anygeometry.nurbs_surface_intersection._source import SourceDefinition, WorkCounters
from anygeometry.tolerance import TolerancePolicy
from test_chart_fallback import CELL, CLIPS, nested_rational_pair


def _case():
    first, second = nested_rational_pair()
    pair = engine._Pair(first, second, CLIPS).anchored(CELL)
    free = 0
    region = _graph_region(CELL, CLIPS, free)
    y = [Interval(*region[k]) for k in range(4) if k != free]
    phi = Interval(*region[free])
    return pair, free, region, y, phi


def _independent_same_c_image(pair, free, region, y, phi, matrix):
    conditioned = pair.preconditioned(matrix)
    center = [x.mid for x in y]
    cf = conditioned.section_value(free, center, phi)
    _, jac = conditioned.value_and_jacobian(conditioned.enclose(region))
    columns = [k for k in range(4) if k != free]
    image = []
    for i in range(3):
        value = Interval(center[i])-cf[i]
        for j, k in enumerate(columns):
            value += (Interval(int(i == j))-jac[i][k])*(y[j]-Interval(center[j]))
        image.append(value)
    return tuple(image)


def test_completed_filtered_handoff_keeps_exact_same_c_proof(monkeypatch):
    pair, free, region, y, phi = _case()
    completed = _filtered.certificate(pair, CLIPS, free, y, phi)
    assert not completed.certified and completed.weights_positive and completed.immersed
    before = repr(completed)
    expected = _independent_same_c_image(pair, free, region, y, phi, completed.preconditioner)
    def redundant_original_preparation(*args):
        pytest.fail("repeated original exact preparation after completed proof")
    monkeypatch.setattr(pair, "enclose", redundant_original_preparation)
    stats = WorkCounters()
    result = engine.certify_prepared(pair, CLIPS, free, y, phi, stats=stats)
    assert result.preconditioner == completed.preconditioner
    assert tuple((x.lo, x.hi) for x in result.unknown_enclosure) == tuple((x.lo, x.hi) for x in expected)
    assert result.certified == all(x.strictly_inside(bound) for x, bound in zip(expected, y))
    assert result.weights_positive and result.immersed
    assert stats.exact_preparations_reused == stats.exact_escalations == 1
    assert repr(completed) == before


@pytest.mark.parametrize("failure", ("unknown", "nonfinite", "missing_c", "missing_immersion", "missing_weights", "missing_image"))
def test_incomplete_filtered_proof_keeps_full_exact_preparation(monkeypatch, failure):
    pair, free, region, y, phi = _case()
    completed = _filtered.certificate(pair, CLIPS, free, y, phi)
    original = pair.enclose
    calls = []
    def observe(box):
        calls.append(tuple(box))
        return original(box)
    monkeypatch.setattr(pair, "enclose", observe)
    def incomplete(*args, **kwargs):
        if failure == "unknown":
            raise _filtered.Unknown("unknown arithmetic")
        if failure == "nonfinite":
            raise OverflowError("nonfinite arithmetic")
        if failure == "missing_c":
            return replace(completed, preconditioner=())
        if failure == "missing_immersion":
            return replace(completed, immersed=False)
        if failure == "missing_image":
            return replace(completed, unknown_enclosure=())
        return replace(completed, weights_positive=False)
    monkeypatch.setattr(_filtered, "certificate", incomplete)
    stats = WorkCounters()
    result = engine.certify_prepared(pair, CLIPS, free, y, phi, stats=stats)
    assert len(calls) >= 2
    assert calls[0] == region
    assert stats.exact_preparations_reused == 0 and stats.exact_escalations == 1
    assert result.unknown_enclosure


def test_exact_only_never_uses_filtered_handoff(monkeypatch):
    pair, free, region, y, phi = _case()
    def forbidden(*args, **kwargs):
        pytest.fail("filtered call in exact-only mode")
    monkeypatch.setattr(_filtered, "certificate", forbidden)
    stats = WorkCounters()
    result = engine.certify_prepared(pair, CLIPS, free, y, phi, interval_mode="exact", stats=stats)
    assert result.unknown_enclosure and stats.float_attempts == stats.exact_preparations_reused == 0


def test_ambiguous_root_filter_keeps_full_exact_route(monkeypatch):
    pair, free, region, y, phi = _case()
    completed = _filtered.certificate(pair, CLIPS, free, y, phi)
    monkeypatch.setattr(_filtered, "certificate", lambda *a, **kw: replace(completed, unknown_enclosure=()))
    original = pair.enclose
    calls = []
    def observe(box):
        calls.append(tuple(box))
        return original(box)
    monkeypatch.setattr(pair, "enclose", observe)
    stats = WorkCounters()
    result = engine.certify_prepared(pair, CLIPS, free, y, phi, for_root_filter=True, stats=stats)
    assert len(calls) >= 2 and result.unknown_enclosure
    assert stats.exact_escalations == 1 and stats.exact_preparations_reused == 0


def test_fixed_work_full_preparation_vs_reuse(monkeypatch):
    """A/B definition, no timing threshold: root records actual fixed-work costs."""
    first, second = nested_rational_pair()
    before = SourceDefinition.capture(first), SourceDefinition.capture(second)
    limits = SimpleNamespace(max_boxes_per_pair=128, max_subdivision_depth=40)
    actual = engine._certify_exact
    reports = []
    for reuse in (False, True):
        def prepare(*args, **kwargs):
            if not reuse:
                kwargs["_reuse_c"] = None
            return actual(*args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(engine, "_certify_exact", prepare)
            began = time.perf_counter()
            result = query_nurbs_surface_intersection(first, second, tolerances=TolerancePolicy(),
                                                     qualification=limits)
            elapsed = time.perf_counter()-began
        assert result.status is IntersectionStatus.UNCLASSIFIED
        assert result.stats.boxes_visited == 128 and result.atlas.frontier_remaining > 0
        assert before[0].matches(first) and before[1].matches(second)
        reports.append(dict(reuse_enabled=reuse, elapsed=elapsed,
            attempts=result.stats.certificate_attempts, certificates=result.stats.certificates,
            unresolved=result.stats.unresolved_cells, exact_seconds=result.stats.exact_seconds,
            escalations=result.stats.exact_escalations,
            preparation_handoffs=result.stats.exact_preparations_reused))
    print(json.dumps(reports, sort_keys=True))
