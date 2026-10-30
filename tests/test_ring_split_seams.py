"""RING-SPLIT RS-1: the ring_split seam algorithm on Plane faces (production brief section 2).

The synthetic cap is the tripod heave-plate layout in miniature: a square plate, a circle of arcs (r = 6) and N
radial webs from the circle to free inner ends at r = 5. ring_split chords the free ends into a ring and splits
the component (circle + webs + ring), which reaches no boundary, with 2 outward seams: N + 3 cells, no holes,
the rim arcs unsplit. Every test fails on the RS-0 commit 18b3d5e (ring_split declined every face there).
"""
import hashlib
import math
import pickle

import numpy as np
import pytest

from anygeometry import EllipticArc, GeometryError
from anygeometry import material_arrangement as ma
from anygeometry.arrangement_geometry import LinePath
from anygeometry.material_arrangement import ArrangementPath, ArrangementPoint, MaterialDomain, arrange_material
from anygeometry.runtime_diagnostics import collect_runtime_diagnostics, collect_runtime_records
from anygeometry.surfaces import Plane

N = 12
TOL, AREA_TOL = 1e-9, 1e-10


def rotate(point, angle):
    c, s = math.cos(angle), math.sin(angle)
    return (c * point[0] - s * point[1], s * point[0] + c * point[1], point[2])


def square(half, angle=0.0, start_edge=1):
    corners = [rotate(p, angle) for p in ((-half, -half, 0.), (half, -half, 0.), (half, half, 0.), (-half, half, 0.))]
    return tuple(ArrangementPath(LinePath(a, b), source_edge=start_edge + k)
                 for k, (a, b) in enumerate(zip(corners, corners[1:] + corners[:1])))


def plane(angle=0.0):
    return Plane((0., 0., 0.), rotate((1., 0., 0.), angle), rotate((0., 1., 0.), angle))


def cap(angle=0.0, count=N, skip=(), frame=None):
    """Square plate +-10, arcs on r = 6, radial webs r 5 -> 6 (indices in ``skip`` left out). The layout is
    rotated by ``angle``; the support's authored frame by ``frame`` (default: the same angle)."""
    domain = MaterialDomain(7, plane(angle if frame is None else frame), (square(10., angle),))
    traces = []
    step = 2 * math.pi / count
    for k in range(count):
        a = k * step + angle
        if k not in skip:
            traces.append(ArrangementPath(LinePath((5 * math.cos(a), 5 * math.sin(a), 0.),
                                                   (6 * math.cos(a), 6 * math.sin(a), 0.)), owners=(10 + k,)))
        traces.append(ArrangementPath(EllipticArc((0., 0., 0.), (6., 0., 0.), (0., 6., 0.), a, step), owners=(2,)))
    return domain, tuple(traces)


def arrange(domain, traces, policy, **kw):
    return arrange_material(domain, traces, tolerance=TOL, area_tolerance=AREA_TOL, seam_policy=policy, **kw)


def outcome(domain, traces, policy, **kw):
    with collect_runtime_diagnostics() as counters:
        try:
            result = ("ok", pickle.dumps(arrange(domain, traces, policy, **kw)))
        except GeometryError as error:
            result = ("error", str(error))
    return result, {k: v for k, v in counters.items() if k.startswith("ring_split") and v}


def summary(arrangement):
    arcs = [p for p in arrangement.paths if type(p.curve) is EllipticArc]
    seams = [p for p in arrangement.paths if p.decomposition]
    return len(arrangement.cells), sum(1 for c in arrangement.cells if c.holes), len(arcs), len(seams)


def length(curve):
    if type(curve) is EllipticArc:
        return abs(curve.sweep_angle) * float(np.linalg.norm(curve.u_vector))
    return float(np.linalg.norm(np.asarray(curve.evaluate(1.)) - np.asarray(curve.evaluate(0.))))


def signature(arrangement):
    """Rotation-free layout signature: cell count, holes, sorted (kind, decomposition, length) of every edge."""
    return (len(arrangement.cells), sum(1 for c in arrangement.cells if c.holes),
            tuple(sorted((type(p.curve).__name__, p.decomposition, round(length(p.curve), 6))
                         for p in arrangement.paths)))


def digest(arrangement):
    return hashlib.sha256(pickle.dumps(arrangement)).hexdigest()


# ----------------------------------------------------------------------------------- the cap layout

def test_ring_split_keeps_the_rim_whole_and_every_cell_simply_connected():
    domain, traces = cap()
    with collect_runtime_diagnostics() as counters, collect_runtime_records() as records:
        result = arrange(domain, traces, "ring_split")
    assert summary(result) == (N + 3, 0, N, N + 2)          # N slots + inner polygon + 2 outer halves
    assert counters["ring_split_applied"] == 1 and counters["ring_split_accepted"] == 1
    assert counters["ring_split_chords"] == N and counters["ring_split_extensions"] == 2
    (item,) = records["ring_split"]
    assert (item["free_ends"], item["chords"], item["split_components"]) == (N, N, 1)
    baseline = arrange(domain, traces, "baseline")
    assert summary(baseline)[2] > N                          # main's full-width seams cut the rim arcs
    assert pickle.dumps(baseline) == pickle.dumps(arrange(domain, traces, None))


def test_digest_is_identical_twice():
    domain, traces = cap()
    assert digest(arrange(domain, traces, "ring_split")) == digest(arrange(domain, traces, "ring_split"))
    again = cap()
    assert digest(arrange(*again, "ring_split")) == digest(arrange(domain, traces, "ring_split"))


@pytest.mark.parametrize("angle", [2 * math.pi / 3, 0.6435011087932844, 1.0, math.radians(120 + 1e-7)])
def test_rotation_gives_the_same_layout_signature(angle):
    # The layout turns, the authored frame does not (as between caps 1081/1082/1083): main's seams follow the
    # frame and change, ring_split uses no frame and keeps its signature.
    reference = signature(arrange(*cap(), "ring_split"))
    assert signature(arrange(*cap(angle, frame=0.0), "ring_split")) == reference
    assert signature(arrange(*cap(angle, frame=0.0), "baseline")) != signature(arrange(*cap(), "baseline"))


def test_input_order_does_not_change_the_layout():
    domain, traces = cap()
    reference = signature(arrange(domain, traces, "ring_split"))
    assert signature(arrange(domain, tuple(reversed(traces)), "ring_split")) == reference


# ----------------------------------------------------------------------------------- declines (bit-equal to baseline)

def _declines_to_baseline(domain, traces, reason, points=()):
    rs, counters = outcome(domain, traces, "ring_split", points=points)
    base, _ = outcome(domain, traces, "baseline", points=points)
    assert counters.get(f"ring_split_declined:{reason}") == 1, counters
    assert "ring_split_applied" not in counters
    assert rs == base                                        # the declined face is bit-identical to baseline
    return rs


def test_same_trace_rule_never_chords_one_arc_into_a_lens():
    # {1083, 809} --single in miniature: one arc trace whose two ends are free; they may not be chorded.
    domain = MaterialDomain(7, plane(), (square(10.),))
    arc = (ArrangementPath(EllipticArc((0., -3., 0.), (4., 0., 0.), (0., 4., 0.), math.radians(60), math.radians(60)),
                           owners=(809,)),)
    # the policy declines (no chord, no layout change): the face is main's, bit for bit
    _declines_to_baseline(domain, arc, "end_without_chord")


def _points(*xy):
    return tuple(ArrangementPoint((x, y, 0.)) for x, y in xy)


def test_exact_tie_between_ranks_two_and_three_declines():
    domain = MaterialDomain(7, plane(), (square(10.),))
    points = _points((0., 0.), (1., 0.), (-1., 0.), (0., 1.), (0., -1.))    # centre: four candidates at 1
    _declines_to_baseline(domain, (), "tie", points=points)
    # the same layout without the tie passes the ranking (and declines later: points are never split sources)
    perturbed = _points((0., 0.), (1., 0.), (-1.25, 0.), (0., 1.5), (0., -1.75))
    _rs, counters = outcome(domain, (), "ring_split", points=perturbed)
    assert "ring_split_declined:tie" not in counters and counters.get("ring_split_declined:split_candidates") == 1


def test_crossing_chords_decline(monkeypatch):
    # A bowtie whose sides are made invisible: each end's only visible candidate is a diagonal; they cross.
    domain = MaterialDomain(7, plane(), (square(10.),))
    points = _points((0., 0.), (2., 1.), (0., 1.), (2., 0.))
    diagonals = {((0., 0.), (2., 1.)), ((0., 1.), (2., 0.))}
    clear = ma._RingSplitGeometry.clear

    def only_diagonals(self, a, b, *args, **kwargs):
        ends = tuple(sorted((tuple(float(x) for x in a[:2]), tuple(float(x) for x in b[:2]))))
        return ends in diagonals and clear(self, a, b, *args, **kwargs)
    monkeypatch.setattr(ma._RingSplitGeometry, "clear", only_diagonals)
    _declines_to_baseline(domain, (), "chord_crossing", points=points)
    assert not ma._seams_meet_only_at_shared_ends(
        ma.stations.Segment(*(ma.exact_vector(p) for p in ((0., 0., 0.), (2., 1., 0.)))),
        ma.stations.Segment(*(ma.exact_vector(p) for p in ((0., 1., 0.), (2., 0., 0.)))), set(), TOL)


def test_seams_meet_only_at_shared_ends_is_exact():
    seg = lambda a, b: ma.stations.Segment(ma.exact_vector(a), ma.exact_vector(b))
    first = seg((0., 0., 0.), (1., 0., 0.))
    assert ma._seams_meet_only_at_shared_ends(first, seg((1., 0., 0.), (2., 1., 0.)), {0}, TOL)     # shared end
    assert not ma._seams_meet_only_at_shared_ends(first, seg((1., 0., 0.), (2., 1., 0.)), set(), TOL)
    assert not ma._seams_meet_only_at_shared_ends(first, seg((.5, 0., 0.), (.5, 1., 0.)), set(), TOL)  # T
    assert not ma._seams_meet_only_at_shared_ends(first, seg((.5, .5e-9, 0.), (.5, 1., 0.)), set(), TOL)  # graze
    assert ma._seams_meet_only_at_shared_ends(first, seg((.5, 2e-9, 0.), (.5, 1., 0.)), set(), TOL)


def test_domain_with_an_inner_loop_declines():
    domain, traces = cap()
    hole = tuple(reversed(square(.5, start_edge=20)))
    holed = MaterialDomain(7, domain.support, (domain.boundaries[0], hole))
    _declines_to_baseline(holed, traces, "domain_holes")


def test_uncertifiable_curved_end_declines():
    # An arc from the boundary corner (-10, -10) ends at E = (0, 0) heading -x; the only other free end F (a
    # straight trace end) lies on the arc's tangent at E, so chord E-F leaves E tangent to the arc: not certified.
    domain = MaterialDomain(7, plane(), (square(10.),))
    traces = (ArrangementPath(EllipticArc((0., -10., 0.), (10., 0., 0.), (0., 10., 0.), math.pi / 2, math.pi / 2),
                              owners=(1,)),
              ArrangementPath(LinePath((-10., 0., 0.), (-5., 0., 0.)), owners=(2,)))
    _declines_to_baseline(domain, traces, "end_without_chord")


def test_curved_end_with_a_transversal_chord_is_certified():
    # The same arc, the other end now off the tangent: the chord leaves E at 90 degrees and is certified.
    domain = MaterialDomain(7, plane(), (square(10.),))
    traces = (ArrangementPath(EllipticArc((0., -10., 0.), (10., 0., 0.), (0., 10., 0.), math.pi / 2, math.pi / 2),
                              owners=(1,)),
              ArrangementPath(LinePath((0., 10., 0.), (0., 5., 0.)), owners=(2,)))
    rs, counters = outcome(domain, traces, "ring_split")
    assert "ring_split_declined:end_without_chord" not in counters, counters
    assert counters.get("ring_split_applied") == 1 and rs[0] == "ok"


def test_point_ends_take_chords_but_are_never_split_sources():
    domain, traces = cap(skip=(0,))
    point = (ArrangementPoint((5., 0., 0.)),)                 # the missing web's inner end, as a point contact
    with collect_runtime_records() as records, collect_runtime_diagnostics() as counters:
        result = arrange(domain, traces, "ring_split", points=point)
    assert counters["ring_split_applied"] == 1
    (item,) = records["ring_split"]
    assert item["free_ends"] == N and item["chords"] == N and item["split_components"] == 1
    cells, holes, arcs, seams = summary(result)
    assert holes == 0 and arcs == N and seams == N + 2
    on_plate_edge = lambda q: abs(abs(q[0]) - 10.) <= 1e-9 or abs(abs(q[1]) - 10.) <= 1e-9
    extensions = [p for p in result.paths if p.decomposition
                  and any(on_plate_edge(p.curve.evaluate(t)) for t in (0., 1.))]
    assert len(extensions) == 2
    for p in extensions:                                      # extension sources: web vertices on the rim only
        source = next(q for q in (p.curve.evaluate(0.), p.curve.evaluate(1.)) if not on_plate_edge(q))
        assert abs(np.linalg.norm(source) - 6.) <= 1e-9


def test_components_follow_interior_crossings():
    # A trace from the plate boundary crosses a rim arc in its interior and ends on a web interior (a T): the ring
    # component then reaches the boundary through crossings only, so no split seams are added.
    domain, traces = cap()
    bridge = ArrangementPath(LinePath((10., .4, 0.), (5.5, 0., 0.)), owners=(99,))
    with collect_runtime_diagnostics() as counters:
        result = arrange(domain, (*traces, bridge), "ring_split")
    assert counters["ring_split_applied"] == 1
    assert counters.get("ring_split_extensions", 0) == 0, counters
    assert summary(result)[1] == 0                            # still no holed cell


# ----------------------------------------------------------------------------------- fallback, budget, cancellation

def test_a_downstream_refusal_of_the_ring_split_layout_reruns_baseline(monkeypatch):
    domain, traces = cap()
    base, _ = outcome(domain, traces, "baseline")
    once = ma._arrange_material_once

    def refuse_ring_split(*args, **kwargs):
        result = once(*args, **kwargs)
        if kwargs.get("seam_policy") == "ring_split":
            raise GeometryError("arrangement contains an unresolved bridge")
        return result
    monkeypatch.setattr(ma, "_arrange_material_once", refuse_ring_split)
    rs, counters = outcome(domain, traces, "ring_split")
    assert counters["ring_split_refused"] == 1 and rs == base


def test_a_holed_cell_postcondition_reruns_baseline(monkeypatch):
    domain, traces = cap()
    base, _ = outcome(domain, traces, "baseline")
    monkeypatch.setattr(ma._RingSplitGeometry, "meets", lambda self, first, second: True)   # "reaches a boundary"
    rs, counters = outcome(domain, traces, "ring_split")
    assert counters.get("ring_split_declined:postcondition_holed_cell") == 1, counters
    assert rs == base


def test_budget_keeps_baseline():
    domain, traces = cap()
    with_budget, counters = outcome(domain, traces, "ring_split", max_predicates=10 ** 9)
    assert not counters and with_budget == outcome(domain, traces, "baseline", max_predicates=10 ** 9)[0]


def _polls_inside(domain, traces):
    inside, polls = [False], [0]
    original = ma._ring_split_seams

    def counted(*args, **kwargs):
        inside[0] = True
        try:
            return original(*args, **kwargs)
        finally:
            inside[0] = False

    def poll():
        if inside[0]:
            polls[0] += 1
        return False
    return counted, poll, polls, inside


def test_cancellation_anywhere_in_the_selection_raises(monkeypatch):
    domain, traces = cap()
    counted, poll, polls, inside = _polls_inside(domain, traces)
    monkeypatch.setattr(ma, "_ring_split_seams", counted)
    arrange(domain, traces, "ring_split", cancellation_check=poll)
    total = polls[0]
    assert total > 100
    for target in sorted({1, 2, 3, *range(1, total + 1, max(1, total // 40)), total}):
        state = {"n": 0}

        def cancel():
            if inside[0]:
                state["n"] += 1
                return state["n"] >= target
            return False
        with pytest.raises(GeometryError, match="^intersection arrangement cancelled$"):
            arrange(domain, traces, "ring_split", cancellation_check=cancel)


def test_cancellation_while_the_ring_split_layout_is_refused_is_a_cancellation(monkeypatch):
    domain, traces = cap()
    once = ma._arrange_material_once
    state = {"cancel": False}

    def refuse_and_cancel(*args, **kwargs):
        result = once(*args, **kwargs)
        if kwargs.get("seam_policy") == "ring_split":
            state["cancel"] = True
            raise GeometryError("arrangement failed material conservation")
        return result
    monkeypatch.setattr(ma, "_arrange_material_once", refuse_and_cancel)
    with pytest.raises(GeometryError, match="^intersection arrangement cancelled$"):
        arrange(domain, traces, "ring_split", cancellation_check=lambda: state["cancel"])


def test_seams_are_exact_clipped_decomposition_segments():
    domain, traces = cap()
    result = arrange(domain, traces, "ring_split")
    seams = [p for p in result.paths if p.decomposition]
    assert len(seams) == N + 2 and all(type(p.curve) is LinePath for p in seams)
    ends = {tuple(np.round(t.curve.evaluate(0.), 12)) for t in traces if type(t.curve) is LinePath}
    chords = [p for p in seams if {tuple(np.round(p.curve.evaluate(s), 12)) for s in (0., 1.)} <= ends]
    assert len(chords) == N                                  # chords join existing free ends exactly


# ----------------------------------------------------------------------------------- Codex RS R1 findings (fix round 1)

def _geometry(tol, ends=()):
    domain = MaterialDomain(7, plane(), (square(10. if tol < 1e-3 else 5e6),))
    paths = list(domain.boundaries[0])
    segments = {k: ma.stations.segment_of(p.curve) for k, p in enumerate(paths)}
    return domain, paths, segments


def test_r1_p2_shared_end_root_check_is_not_skipped_by_the_prefilter():
    # Codex R1 finding 1: a small incident arc crosses the seam at x = 3 tol; arc_far's box test (the box starts
    # 4 tol from the shared end) must not skip the shared-end certified contact test.
    tol = 1e-6
    domain, paths, segments = _geometry(tol)
    arc = ArrangementPath(EllipticArc((1.5 * tol, 0., 0.), (1.5 * tol, 0., 0.), (0., 1.5 * tol, 0.), math.pi,
                                      -1.5 * math.pi), owners=(1,))
    paths.append(arc)
    index = len(paths) - 1
    geometry = ma._RingSplitGeometry(domain, paths, segments, len(paths), [], tol, lambda: None)
    a, b = ma.exact_vector((0., 0., 0.)), ma.exact_vector((30 * tol, 0., 0.))
    assert not geometry.arc_clear(index, a, b, True, False)
    assert not geometry.clear(a, b, {index}, set())


def test_r1_p2_arc_prefilter_bounds_the_angular_phase_error():
    # Codex R1 finding 2: an admitted but unnormalised phase (2**40 rad) makes the float extremum miss the arc's
    # top by ~4e-4 m; the prefilter must decline (certified test decides: the arc crosses the seam).
    tol = 1e-9
    domain, paths, segments = _geometry(tol)
    arc = ArrangementPath(EllipticArc((0., -999999.9996237048, 0.), (1e6, 0., 0.), (0., 1e6, 0.), float(2 ** 40),
                                      2 * math.pi), owners=(1,))
    paths.append(arc)
    geometry = ma._RingSplitGeometry(domain, paths, segments, len(paths), [], tol, lambda: None)
    a, b = ma.exact_vector((-2e6, 0., 0.)), ma.exact_vector((2e6, 0., 0.))
    assert not geometry.arc_clear(len(paths) - 1, a, b, False, False)
