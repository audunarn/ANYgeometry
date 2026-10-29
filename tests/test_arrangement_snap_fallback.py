"""CAP-1: refusal-only snap fallback of the float cell route on a Plane support.

Reproducer (``data/cap1082_wall449_arrangement.json``): the AIP-Gen2 tripod cap face 1082 (a Plane with a
rotated, degenerate chart) with the one cylinder-wall trace of face 449. The float cell route builds cell loops
whose straight pieces end at one merged vertex but at different stations (per-path parameter merge); the open
loops' Green areas miss the conservation tolerance and the arrangement refuses. The fallback runs only on that
refusal: it closes the loops at their merged vertices (every straight end moves by at most the vertex-merge
tolerance) and the unchanged conservation check and tolerance decide; otherwise the original error is raised.
"""
from fractions import Fraction
import json
import math
import pathlib

import numpy as np
import pytest

from anygeometry import EllipticArc, GeometryError
from anygeometry import material_arrangement as ma
from anygeometry.arrangement_geometry import LinePath
from anygeometry.exact_predicates import exact_vector
from anygeometry.material_arrangement import ArrangementPath, MaterialDomain, arrange_material
from anygeometry import runtime_diagnostics as rd
from anygeometry.runtime_diagnostics import collect_runtime_diagnostics
from anygeometry.surfaces import Plane

DATA = pathlib.Path(__file__).resolve().parent / "data" / "cap1082_wall449_arrangement.json"
CONSERVATION = "arrangement failed material conservation"


def _fixture():
    record = json.loads(DATA.read_text())
    support = Plane(*(tuple(record["support"][key]) for key in ("origin", "u_vector", "v_vector")))
    boundary = tuple(ArrangementPath(LinePath(tuple(item["start"]), tuple(item["end"])), source_edge=item["source_edge"])
                     for item in record["boundary"])
    traces = tuple(ArrangementPath(EllipticArc(tuple(item["center"]), tuple(item["u_vector"]), tuple(item["v_vector"]),
                                               item["start_angle"], item["sweep_angle"]), owners=tuple(item["owners"]))
                   for item in record["traces"])
    domain = MaterialDomain(record["face_id"], support, (boundary,))
    return domain, traces, record["tolerance"], record["area_tolerance"]


def _arrange(**extra):
    domain, traces, tolerance, area_tolerance = _fixture()
    return arrange_material(domain, traces, tolerance=tolerance, area_tolerance=area_tolerance, **extra)


def _loop_gaps(arrangement):
    """Largest distance between consecutive piece ends of every cell loop (float evaluation, as the area sees it)."""
    worst = 0.
    for cell in arrangement.cells:
        for loop in (cell.outer, *cell.holes):
            ends = []
            for index, forward in loop:
                curve = arrangement.paths[index].curve
                a, b = np.asarray(curve.evaluate(0.)), np.asarray(curve.evaluate(1.))
                ends.append((a, b) if forward else (b, a))
            for k, (_start, end) in enumerate(ends):
                worst = max(worst, float(np.linalg.norm(ends[(k + 1) % len(ends)][0] - end)))
    return worst


def test_reproducer_is_accepted_by_the_snap_fallback():
    with collect_runtime_diagnostics() as counters:
        arrangement = _arrange()
    domain, _traces, tolerance, area_tolerance = _fixture()
    # The unchanged conservation check, recomputed independently on the returned cells.
    original = abs(domain.area_loop(domain.boundaries[0], area_tolerance / domain.area_jacobian * .05))
    assert abs(original - arrangement.area) <= area_tolerance / domain.area_jacobian
    assert arrangement.native_area_tolerance == area_tolerance / domain.area_jacobian
    assert len(arrangement.cells) >= 2
    assert _loop_gaps(arrangement) <= 1e-13          # the loops are closed (gaps were up to 8.7e-12 m)
    assert counters["arrangement_snap_fallback_attempts"] == 1
    assert counters["arrangement_snap_fallbacks"] == 1
    assert counters["arrangement_snap_moved_edges"] >= 1


def test_diagnostics_record_the_fired_fallback():
    with collect_runtime_diagnostics() as counters, rd.collect_runtime_records() as collected:
        _arrange()
    assert all(type(value) is int for value in counters.values())   # the counter contract stays integer
    records = collected["arrangement_snap_fallback"]
    assert len(records) == 1
    record = records[0]
    assert record["face_id"] == 1082
    assert record["moved_edges"] == counters["arrangement_snap_moved_edges"] >= 1
    assert record["snapped_vertices"] >= 1
    assert 0. < record["max_move"] <= 1e-9
    assert record["refused_ratio"] > 1. >= record["accepted_ratio"]


def test_accepted_arrangement_never_attempts_the_fallback():
    # The same cap without the trace: the float route is not reached, and an ordinary Plane arrangement with an
    # arc that the unchanged check accepts never attempts the fallback either.
    domain, _traces, tolerance, area_tolerance = _fixture()
    with collect_runtime_diagnostics() as counters, rd.collect_runtime_records() as records:
        arrange_material(domain, (), tolerance=tolerance, area_tolerance=area_tolerance)
    boundary = tuple(ArrangementPath(LinePath(a, b)) for a, b in (
        ((0., 0., 0.), (4., 0., 0.)), ((4., 0., 0.), (4., 4., 0.)), ((4., 4., 0.), (0., 4., 0.)), ((0., 4., 0.), (0., 0., 0.))))
    square = MaterialDomain(7, Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)), (boundary,))
    arc = ArrangementPath(EllipticArc((2., 2., 0.), (1., 0., 0.), (0., 1., 0.), 0., math.pi), owners=(7, 8))
    line = ArrangementPath(LinePath((0., 2., 0.), (4., 2., 0.)), owners=(7, 9))
    with collect_runtime_diagnostics() as more, rd.collect_runtime_records() as more_records:
        arranged = arrange_material(square, (arc, line), tolerance=1e-9)
    assert len(arranged.cells) == 3
    for collected in (counters, more):
        assert collected.get("arrangement_snap_fallback_attempts", 0) == 0
    assert records == {} and more_records == {}


def test_move_beyond_the_bound_refuses_with_the_original_error(monkeypatch):
    # Test seam: the snap sees a merge tolerance 1000x smaller, so the measured moves (~1e-11 m) exceed it.
    real = ma._snap_open_vertices
    monkeypatch.setattr(ma, "_snap_open_vertices",
                        lambda *args, tolerance: real(*args, tolerance=tolerance * 1e-3))
    with collect_runtime_diagnostics() as counters:
        with pytest.raises(GeometryError) as caught:
            _arrange()
    assert type(caught.value) is GeometryError and str(caught.value) == CONSERVATION
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert counters["arrangement_snap_fallback_attempts"] == 1
    assert counters.get("arrangement_snap_fallbacks", 0) == 0
    assert counters["arrangement_snap_fallback_refusals"] == 1


def test_snap_bound_is_exact_and_inclusive():
    tol = 1e-9
    a = (Fraction(0), Fraction(0), Fraction(0))
    far = (Fraction(1, 10 ** 6), Fraction(0), Fraction(0))
    ends = (Fraction(1), Fraction(0), Fraction(0))
    # One straight edge 0->1 whose start station differs from the representative of vertex 0 by exactly tol,
    # then by tol plus one part in 1e30.
    for offset, accepted in ((Fraction(tol), True), (Fraction(tol) * (1 + Fraction(1, 10 ** 30)), False)):
        start = (offset, Fraction(0), Fraction(0))
        segment = ma.stations.Segment(start, ends)
        edges = [ArrangementPath(ma.stations.to_line(segment))]
        other = ma.stations.Segment(a, far)
        edges.append(ArrangementPath(ma.stations.to_line(other)))
        result = ma._snap_open_vertices(edges, {0: segment, 1: other}, [(0, 1), (0, 2)], [a, ends, far],
                                        tolerance=tol)
        assert (result is not None) == accepted
        if accepted:
            new_edges, new_segments, moved, max_move2, _snapped, _open = result
            assert moved == {0} and new_segments[0][0] == a and max_move2 == Fraction(tol) ** 2
            assert new_edges[0].curve.start == (0., 0., 0.)


def test_explicit_predicate_budget_keeps_the_original_refusal():
    with collect_runtime_diagnostics() as counters:
        with pytest.raises(GeometryError, match=CONSERVATION):
            _arrange(max_predicates=10 ** 9)
    assert counters.get("arrangement_snap_fallback_attempts", 0) == 0


def test_cancellation_during_the_fallback_is_not_swallowed(monkeypatch):
    calls = {"n": 0, "at_fallback": None}
    real = ma._snap_open_vertices

    def spy(*args, tolerance):
        calls["at_fallback"] = calls["n"]
        return real(*args, tolerance=tolerance)

    monkeypatch.setattr(ma, "_snap_open_vertices", spy)

    def cancel():
        calls["n"] += 1
        return calls["at_fallback"] is not None
    with pytest.raises(GeometryError, match="intersection arrangement cancelled"):
        _arrange(cancellation_check=cancel)
