"""CH3a2: loops of pieces, planar faces with rational arcs, the extended admission checks 1-6 and snapshot v2
(design CERTIFIED_HYDROSTATICS_CH3_DESIGN.md sections 2.3, 3.3, 3.4 and 9, CH3a row, second half; the narrowing of
the CH3a2 spec: B1 and Q08 only, everything else a typed refusal)."""
from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import replace
from fractions import Fraction as F

import pytest

import ch3_solids as solids
from anygeometry import Plane
from anygeometry.certified_hydrostatics import (CertifiedHydrostaticsPolicy, CertifiedHydrostaticsRefusal,
                                                HalfSpaceQuery, RefusalCategory, WorkBudget, check_preparation,
                                                prepare_certified_hydrostatics, query_certified_hydrostatics)
from anygeometry.certified_hydrostatics import _ch3_curved_admit as ca
from anygeometry.certified_hydrostatics._ch3_loops import check_loop, is_v_monotone, loop_sigma, region_contains
from anygeometry.certified_hydrostatics._identity import canonical_json
from anygeometry.certified_hydrostatics._snapshot import collect_snapshot
from anygeometry.certified_hydrostatics._work import Meter
from test_certified_hydrostatics_ch3_admission import (POLICY, admit, build_face, kern, planes_fa, planes_fb,
                                                       planes_fc, refusal_of, surface_fa, surface_fb, surface_fc)
from test_certified_hydrostatics_admission import U_SHAPE, extrusion, tetrahedron, unit_box

C = RefusalCategory
_CACHE: dict = {}


def half_hull():
    from conftest import b1_half_hull_shared
    return b1_half_hull_shared()


def full_hull():
    """The full B1 hull with outward loops (the library builds it inward, see the finding below)."""
    if "full" not in _CACHE:
        _CACHE["full"] = solids.outward_full_hull()
    return _CACHE["full"]


def solid(name):
    if name not in _CACHE:
        _CACHE[name] = {"fa": lambda: solids.solid_fa(build_face, surface_fa, planes_fa),
                        "fb": lambda: solids.solid_fb(build_face, surface_fb),
                        "fc": lambda: solids.solid_fc(build_face, surface_fc, planes_fc)}[name]()
    return _CACHE[name]


def prepared(name):
    if ("p", name) not in _CACHE:
        if name == "half":
            model, faces = half_hull()
            ids = list(faces.values())
        elif name == "full":
            model, faces = full_hull()
            ids = list(faces.values())
        else:
            model, ids = solid(name)
        _CACHE[("p", name)] = (model, ids, prepare_certified_hydrostatics(model, ids))
    return _CACHE[("p", name)]


def refused(model, ids, **kwargs):
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        prepare_certified_hydrostatics(model, ids, **kwargs)
    return info.value


# ------------------------------------------------------------------ planar behaviour is unchanged (digests, proofs)

PLANAR_PINS = {          # measured on the parent ec980a9 and on this branch: equal (artifact identity excluded)
    "box": ("7343d12e2bd729e4694d", ["5aa96032b06fd3c9c0ff", "c53977bf39af857c1b24"]),
    "tetra": ("2b3a0f473658a87b0609", ["f0363ceb036d01ed7663", "3605dcbac8e0a0796b3c"]),
    "scaled_box": ("b02178fd73119a04fe2b", ["e1bc28459ebab8bdbc9d", "bffdce11d070b6f63452"]),
    "prism": ("d22c504713fd0ba54b9f", ["30153c9543092ed046f8", "f2dcbb672bbb44a5f4cd"]),
    "u_prism": ("f8d6bd15c4af6236c8c5", ["acb6204d72dd920ffc15", "b5563cdfd511f70111f1"]),
    "pentagon": ("ade4b179eca84c5e54b5", ["42d1e90633c880fd6d22", "d55a922f7443c9804bc5"]),
}


def _planar_cases():
    return {"box": unit_box(), "tetra": tetrahedron(), "scaled_box": unit_box(2.0),
            "prism": extrusion([(0., 0.), (2., 0.), (0., 2.)], 1.5), "u_prism": extrusion(U_SHAPE, 2.0),
            "pentagon": extrusion([(0., 0.), (4., 0.), (5., 2.), (2., 4.), (-1., 2.)], 3.0)}


@pytest.mark.parametrize("name", sorted(PLANAR_PINS))
def test_planar_preparations_keep_the_convention_1_digest_and_proof_bytes(name):
    """The snapshot digest and the proof document (artifact identity, model identity and work counts excluded:
    they depend on the source files, the model id and the artifact size) equal the parent's, byte for byte."""
    model, faces = _planar_cases()[name]
    p = prepare_certified_hydrostatics(model, faces)
    assert p.snapshot.curved is None and p.policy.convention_version == "1"
    assert "snapshot_version" not in p.snapshot.canonical()
    assert p.snapshot_digest[:20] == PLANAR_PINS[name][0]
    hashes = []
    for q in (HalfSpaceQuery(normal=(0., 0., 1.), offset=0.5), HalfSpaceQuery(normal=(1., 2., 3.), offset=2.25)):
        doc = dict(query_certified_hydrostatics(p, q).proof_document())
        doc.pop("artifact")
        doc["model"] = {k: v for k, v in doc["model"].items() if k in ("face_ids", "snapshot_digest")}
        doc["result"] = {k: v for k, v in doc["result"].items() if k != "work"}
        hashes.append(hashlib.sha256(canonical_json(doc)).hexdigest()[:20])
    assert hashes == PLANAR_PINS[name][1]


def test_the_new_categories_exist():
    assert C.EMBEDDING_UNDECIDED.value == "EMBEDDING_UNDECIDED"
    assert C.CURVED_SECTION_DEGENERACY.value == "CURVED_SECTION_DEGENERACY"


# ------------------------------------------------------------------ both B1 builds and the fixtures prepare

@pytest.mark.parametrize("name", ["half", "full", "fa", "fb", "fc"])
def test_b1_and_the_fixture_solids_prepare_with_convention_2_and_snapshot_v2(name):
    model, ids, p = prepared(name)
    assert p.policy.convention_version == "2" and p.snapshot.curved is not None
    document = p.snapshot.canonical()
    assert document["snapshot_version"] == 2
    curved = [f for f in document["faces"] if isinstance(f["support"], dict)]
    assert [f["id"] for f in curved] == sorted(p.snapshot.curved.faces)
    for f in curved:
        assert f["support"]["kind"] == "ruled" and f["support"]["sigma_chart"] in (1, -1)
    sections = [e[4] for e in document["edges"] if len(e) == 5 and e[4] is not None]
    assert len(sections) == len(p.snapshot.curved.arcs) and all(s["kind"] == "section" for s in sections)
    assert sum(1 for v in document["vertices"] if v[2] is not None) == sum(
        len(c.corners) for c in p.snapshot.curved.faces.values())
    assert p.work["operations"] > 0 and check_preparation(model, p)["operations"] > 0


def test_the_snapshot_v2_document_records_the_corner_identity_and_is_deterministic():
    model, ids, p = prepared("half")
    again = prepare_certified_hydrostatics(model, ids)
    assert again.snapshot_digest == p.snapshot_digest
    document = p.snapshot.canonical()
    corner = next(v[2] for v in document["vertices"] if v[2] is not None)["corner"]
    assert set(corner) == {"poly", "ordinal", "hint"} and len(corner["hint"]) == 3
    # a planar subset has no curved record and the planar document
    assert "snapshot_version" not in collect_snapshot(*unit_box(), Meter(None)).canonical()


def test_finding_the_library_full_hull_is_built_with_inward_loops():
    """``build_b1_hull`` loops are anticlockwise about the inward normals: the exact signed volume is negative
    (-2 x the half hull's, which is +29.6157 = 59.2412 / 2) and admission refuses INVERTED_ORIENTATION."""
    from anygeometry.b1_example import build_b1_hull
    model, faces = build_b1_hull()
    error = refused(model, list(faces.values()))
    assert error.category is C.INVERTED_ORIENTATION and error.phase == "prepare"


def test_the_signed_volume_is_the_known_one():
    """Check 6 integrates the whole shell: the B1 half hull has V(z <= 3) = 59.241180737020 / 2 (design 1)."""
    model, ids, p = prepared("half")
    from fractions import Fraction
    from anygeometry.certified_hydrostatics._ch3_poly import Kernel
    k = Kernel(Meter(None), POLICY.kernel_limits())
    pts = ca._points(k, p.snapshot)
    views = ca._build_views(k, p.snapshot, pts)
    v = ca._volume(k, p.snapshot, views, pts, 1024, Fraction(1, 2 ** 20))
    assert float(v.lo) <= 59.241180737020 / 2 <= float(v.hi) and float(v.hi - v.lo) < 0.02
    m, i, fa = prepared("fa")
    k = Kernel(Meter(None), POLICY.kernel_limits())
    pts = ca._points(k, fa.snapshot)
    v = ca._volume(k, fa.snapshot, ca._build_views(k, fa.snapshot, pts), pts, 1024, Fraction(1, 2 ** 20))
    assert float(v.lo) <= 53 / 1536 <= float(v.hi) or float(v.lo) <= 0.034505208333 <= float(v.hi)


def test_cold_preparation_of_the_full_hull_is_recorded():
    """Handover: cold preparation of the full hull must be <= 120 s (measured here, recorded in the report)."""
    model, faces = full_hull()
    start = time.perf_counter()
    p = prepare_certified_hydrostatics(model, list(faces.values()))
    seconds = time.perf_counter() - start
    assert seconds < 120
    assert p.work["max_bits_seen"] <= 4096
    print(f"full hull preparation: {seconds:.2f} s, {p.work['operations']} units, {p.work['max_bits_seen']} bits")


def test_a_curved_prepared_model_answers_a_query_with_a_typed_refusal():
    model, ids, p = prepared("half")
    error = refusal_of(lambda: query_certified_hydrostatics(p, HalfSpaceQuery(normal=(0., 0., 1.), offset=1.5)))
    assert error.category is C.UNSUPPORTED_SURFACE and error.sub_reason == "curved_query" and error.phase == "query"


def test_check_preparation_detects_a_changed_curved_model():
    model, faces = solids.outward_full_hull()
    ids = list(faces.values())
    p = prepare_certified_hydrostatics(model, ids)
    assert check_preparation(model, p)["operations"] > 0
    model.add_point(9., 9., 9.)
    error = refused_check(model, p)
    assert error.category is C.STALE_PREPARATION


def refused_check(model, p):
    with pytest.raises(CertifiedHydrostaticsRefusal) as info:
        check_preparation(model, p)
    return info.value


# ------------------------------------------------------------------ loops of pieces on a curved face (3.3)

SQUARE_ISH = [(.2, .25), (.75, .27), (.78, .75), (.22, .7)]
C_SHAPE = [(.12, .1), (.9, .12), (.92, .3), (.3, .32), (.28, .7), (.9, .72), (.88, .9), (.1, .88)]
CONTACT = [(F(1, 8), F(1, 8)), (F(7, 8), F(1, 8)), (F(5, 8), F(5, 8)), (F(1, 2), F(1, 8)), (F(1, 4), F(5, 8))]
BOWTIE = [(.1, .1), (.9, .4), (.88, .1), (.12, .4)]
HAIRPIN = [(.1, .1), (.7, .1), (.8, .5), (.6, .1), (.3, .1), (.2, .5)]


def polygon(points):
    points = [(float(u), float(v)) for u, v in points]
    model, face, vertices, edges = solids.polygon_face(build_face, surface_fa, points)
    return model, face


def polygon_admitted(points):
    model, face = polygon(points)
    return admit(model, face)


def loop_error(points):
    cf = polygon_admitted(points)
    return refusal_of(lambda: check_loop(kern(), cf))


def test_sigma_chart_is_plus_one_anticlockwise_and_minus_one_clockwise():
    anticlockwise = polygon_admitted(SQUARE_ISH)
    clockwise = polygon_admitted(SQUARE_ISH[::-1])
    assert loop_sigma(kern(), anticlockwise) == 1 and loop_sigma(kern(), clockwise) == -1
    # the sign of -oint v du for the design's quadrilateral (chart area 0.2): a certified Enclosure sign
    for cf in (anticlockwise, clockwise):
        check_loop(kern(), cf)


def test_sigma_chart_of_the_b1_bows_and_the_solids_is_recorded_in_the_snapshot():
    sigma = {name: prepared(name)[2].snapshot.curved.sigma for name in ("half", "full", "fa", "fb", "fc")}
    assert list(sigma["half"].values()) == [-1]                      # the bow's loop is reversed in the library
    assert sorted(sigma["full"].values()) == [-1, 1]                 # the mirrored bow runs the other way
    assert all(len(v) == 1 and list(v.values())[0] in (1, -1) for k, v in sigma.items() if k in ("fa", "fb", "fc"))
    # the whole shell reversed: the loops (sigma) are decided first, the orientation check refuses afterwards
    model, ids, p = prepared("fc")
    solids.flip_all(model, ids)
    try:
        assert refused(model, ids).category is C.INVERTED_ORIENTATION
    finally:
        solids.flip_all(model, ids)


@pytest.mark.parametrize("name", ["half", "full", "fa", "fb", "fc"])
def test_v_monotone_exact_evidence_for_the_b1_and_fixture_faces(name):
    """Every vertical line meets the region in at most one interval: exactly two pieces cover each elementary
    u-interval between the sorted corners (exact algebraic comparisons)."""
    model, ids, p = prepared(name)
    k = kern()
    for fid, cf in p.snapshot.curved.faces.items():
        assert is_v_monotone(k, cf), (name, fid)
        # the same fact from the pieces: the sorted distinct corner roots and the cover counts
        from anygeometry.certified_hydrostatics._ch3_algebraic import compare
        roots = []
        for corner in cf.corners.values():
            if not any(compare(k, corner.number, r) == 0 for r in roots):
                roots.append(corner.number)
        assert len(roots) >= 2


def test_a_non_monotone_loop_is_simple_and_the_general_tests_admit_it():
    cf = polygon_admitted(C_SHAPE)
    k = kern()
    check_loop(k, cf)                                  # the four tests pass: a simple loop
    assert not is_v_monotone(k, cf)                    # u = 0.6 meets the region in two intervals
    assert region_contains(k, cf, F(3, 5), F(1, 5)) and region_contains(k, cf, F(3, 5), F(4, 5))
    assert not region_contains(k, cf, F(3, 5), F(1, 2))
    assert loop_sigma(k, cf) in (1, -1)
    # the same fact through preparation of the face alone would stop at the shell (one face is not closed)
    model, face = polygon(C_SHAPE)
    assert refused(model, [face]).category is C.NOT_CLOSED


def test_vertical_ray_parity_on_the_b1_bow():
    model, ids, p = prepared("half")
    cf = next(iter(p.snapshot.curved.faces.values()))
    k = kern()
    # design section 1: the deck graph v = 3/4 between the deck corners; points on the two sides of it
    assert not region_contains(k, cf, F(17, 100), F(3, 4)) and not region_contains(k, cf, F(43, 100), F(3, 4))
    assert not region_contains(k, cf, F(3, 10), F(3, 4) + F(1, 10 ** 6))
    assert region_contains(k, cf, F(3, 10), F(3, 4) - F(1, 10 ** 6))
    assert region_contains(k, cf, F(3, 10), F(1, 2)) and not region_contains(k, cf, F(3, 10), F(1, 20))
    assert refusal_of(lambda: region_contains(k, cf, F(3, 10), F(3, 4))).sub_reason == "on_boundary"
    # a corner abscissa is a breakpoint (F-A has the rational corners u = 1/8, 3/8, 5/8, 7/8)
    fa = next(iter(prepared("fa")[2].snapshot.curved.faces.values()))
    assert refusal_of(lambda: region_contains(k, fa, F(1, 8), F(1, 2))).sub_reason == "breakpoint"
    assert region_contains(k, fa, F(1, 4), F(5, 16)) and not region_contains(k, fa, F(1, 4), F(3, 4))


@pytest.mark.parametrize("points,sub", [(CONTACT, "endpoint_contact"), (BOWTIE, "crossing"),
                                        (HAIRPIN, "coincident_pieces")])
def test_each_simplicity_test_refuses_its_minimal_loop(points, sub):
    error = loop_error(points)
    assert error.category is C.INVALID_LOOP and error.sub_reason == sub


def test_the_endpoint_contact_loop_fails_test_ii_only():
    """Design 3.3: (1/8,1/8), (7/8,1/8), (5/8,5/8), (1/2,1/8), (1/4,5/8) has area 3/16, no crossing, one
    endpoint-interior contact. With test (ii) disabled the other three tests pass it."""
    from anygeometry.certified_hydrostatics import _ch3_loops as loops
    cf = polygon_admitted(CONTACT)
    original = loops.sign_at
    loops.sign_at = lambda k, g, x: 1                  # test (ii)/(iii) see no zero: every contact is lost
    try:
        check_loop(kern(), cf)
    finally:
        loops.sign_at = original
    assert refusal_of(lambda: check_loop(kern(), cf)).sub_reason == "endpoint_contact"


def test_a_refused_loop_stops_preparation_with_its_sub_reason():
    for points, sub in ((CONTACT, "endpoint_contact"), (BOWTIE, "crossing"), (HAIRPIN, "coincident_pieces")):
        model, face = polygon(points)
        error = refused(model, [face])
        assert error.category is C.INVALID_LOOP and error.sub_reason == sub and error.phase == "prepare"


# ------------------------------------------------------------------ planar loops (segments, rational arcs)

from anygeometry.certified_hydrostatics._ch3_exact import Pt, boxes_disjoint, sign  # noqa: E402


def rat(vertex, x, y, z=0):
    return Pt(vertex, rat=(F(x), F(y), F(z)))


def seg_view(points, view_id=1, plane=((0, 0, 1), 0), edge_base=100, vertex_base=0):
    """A planar view from rational points (vertex ids 1..n): closed loop of segments in the plane ``n . x = d``."""
    pts = [rat(i + 1 + vertex_base, *p) for i, p in enumerate(points)]
    n = len(pts)
    pieces = [ca._Piece("seg", edge_base + i, pts[i], pts[(i + 1) % n]) for i in range(n)]
    normal, d = plane
    drop = [abs(c) for c in normal].index(max(abs(c) for c in normal))
    return ca._View(view_id, False, [p.vertex for p in pts], pieces, frozenset(p.vertex for p in pts),
                    frozenset(p.edge for p in pieces), n=tuple(F(c) for c in normal), d=F(d), origin=(F(0),) * 3,
                    axes=tuple(a for a in range(3) if a != drop))


def planar_loop_error(view):
    return refusal_of(lambda: ca._check_planar_loop(kern(), view))


def test_planar_segment_loops_exact_signs():
    ca._check_planar_loop(kern(), seg_view([(0, 0), (4, 0), (4, 4), (0, 4)]))
    ca._check_planar_loop(kern(), seg_view([(0, 0), (1, 0), (2, 0), (2, 2), (0, 2)]))   # collinear continuation
    crossing = planar_loop_error(seg_view([(0, 0), (4, 4), (4, 0), (0, 4)]))
    assert crossing.category is C.INVALID_LOOP and crossing.sub_reason == "crossing"
    touching = planar_loop_error(seg_view([(0, 0), (4, 0), (4, 4), (2, 0), (0, 4)]))
    assert touching.category is C.INVALID_LOOP and touching.sub_reason == "touch"
    spike = planar_loop_error(seg_view([(0, 0), (2, 0), (1, 0), (1, 1)]))
    assert spike.category is C.INVALID_LOOP and spike.sub_reason == "spike"


def test_planar_loops_of_the_b1_faces_have_algebraic_vertices_and_exact_signs():
    model, ids, p = prepared("half")
    k = kern()
    pts = ca._points(k, p.snapshot)
    views = ca._build_views(k, p.snapshot, pts)
    planar = [v for v in views if not v.curved]
    assert len(planar) == 5 and sum(1 for v in planar for piece in v.pieces if piece.kind == "arc") == 4
    assert sum(1 for pt in pts.values() if pt.rat is None) == 4        # the four bow corners are algebraic
    for v in planar:
        ca._check_planar_loop(k, v)


def _deck_view():
    model, ids, p = prepared("half")
    k = kern()
    pts = ca._points(k, p.snapshot)
    views = ca._build_views(k, p.snapshot, pts)
    deck = next(v for v in views if not v.curved and abs(float(v.d)) > 0 and v.n[2] != 0 and
                any(piece.kind == "arc" for piece in v.pieces) and float(v.d) / float(v.n[2]) == 3.0)
    return k, deck


def test_an_arc_that_crosses_the_line_of_a_non_adjacent_segment_is_refused():
    """A segment on x = 2.5 (rational line) meets the deck arc inside: arc_line_contact."""
    k, deck = _deck_view()
    arc, a_seg, _b_seg, c_seg = deck.pieces
    extra = ca._Piece("seg", 900, rat(900, F(5, 2), 0, 3), rat(901, F(5, 2), 10, 3))
    view = replace(deck, pieces=[arc, a_seg, extra, c_seg])
    error = refusal_of(lambda: ca._check_planar_loop(k, view))
    assert error.category is C.UNDECIDED_PREDICATE and error.sub_reason == "arc_line_contact"
    # a rational line that the arc does not meet (x = 7) is fine when its box misses the arc
    far = ca._Piece("seg", 901, rat(902, 7, 0, 3), rat(903, 7, 10, 3))
    ca._check_planar_loop(k, replace(deck, pieces=[arc, a_seg, far, c_seg]))


def test_two_adjacent_arcs_and_two_overlapping_arc_boxes_are_refused():
    model, ids, p = prepared("half")
    k = kern()
    pts = ca._points(k, p.snapshot)
    bow = next(v for v in ca._build_views(k, p.snapshot, pts) if v.curved)
    arc0, arc1, arc2 = bow.pieces[0], bow.pieces[1], bow.pieces[2]
    filler = ca._Piece("seg", 950, rat(950, 0, 0, 0), rat(951, 1, 0, 0))
    view = replace(bow, curved=False, pieces=[arc0, arc1, filler], n=(F(0), F(0), F(1)), d=F(0), axes=(0, 1))
    error = refusal_of(lambda: ca._check_planar_loop(k, view))
    assert error.category is C.UNDECIDED_PREDICATE and error.sub_reason == "adjacent_arcs"
    twin = replace(arc0)
    separate = replace(bow, curved=False, pieces=[arc0, filler, replace(arc0), ca._Piece("seg", 951, rat(952, 5, 5, 5), rat(953, 6, 5, 5))],
                       n=(F(0), F(0), F(1)), d=F(0), axes=(0, 1))
    error = refusal_of(lambda: ca._check_planar_loop(k, separate))
    assert error.category is C.EMBEDDING_UNDECIDED and error.sub_reason == "arc_arc_boxes_overlap"
    # the aft and fore arcs of the bow are x = 1.5 and x = 3.5: separated boxes pass (segments far away)
    far1 = ca._Piece("seg", 960, rat(960, 100, 100, 0), rat(961, 101, 100, 0))
    far2 = ca._Piece("seg", 961, rat(962, 100, 200, 0), rat(963, 101, 200, 0))
    fine = replace(bow, curved=False, pieces=[arc0, far1, bow.pieces[2], far2],
                   n=(F(0), F(0), F(1)), d=F(0), axes=(0, 1))
    ca._check_planar_loop(k, fine)


def test_a_multi_root_zero_is_undecided_not_guessed():
    """The x coordinates of the two aft corners are both exactly 3/2 (different roots): equality is not provable by
    intervals, and the predicate refuses at the isolation depth instead of guessing."""
    model, ids, p = prepared("half")
    k = kern()
    pts = ca._points(k, p.snapshot)
    algebraic = [pt for pt in pts.values() if pt.rat is None]
    from anygeometry.certified_hydrostatics._ch3_poly import Kernel, KernelLimits
    from anygeometry.certified_hydrostatics._ch3_algebraic import algebraic_equal
    low = Kernel(Meter(None), KernelLimits(64, 16, 1024, 8))
    pair = next((a, b) for a in algebraic for b in algebraic
                if a is not b and not algebraic_equal(low, a.root, b.root) and
                sign(low, (a,), lambda alg: alg.sub(alg.coord(a, 0), alg.const(F(3, 2)))) == 0 and
                sign(low, (b,), lambda alg: alg.sub(alg.coord(b, 0), alg.const(F(3, 2)))) == 0)
    error = refusal_of(lambda: sign(low, pair, lambda alg: alg.sub(alg.coord(pair[0], 0), alg.coord(pair[1], 0))))
    assert error.category is C.UNDECIDED_PREDICATE and error.sub_reason == "multi_root_zero"
    # single-root zero tests are exact: the same corner against the rational 3/2
    assert sign(k, (pair[0],), lambda alg: alg.sub(alg.coord(pair[0], 0), alg.const(F(3, 2)))) == 0
    # two points of one root (a mirrored pair) compare exactly through Q(u*)
    assert all(sign(k, pair[:1], lambda alg: alg.sub(alg.coord(pair[0], 0), alg.coord(pair[0], 0))) == 0 for _ in range(1))


# ------------------------------------------------------------------ check 5: embedding of planar faces

def rect_view(view_id, plane, corners, edges):
    """A planar view of the loop ``corners`` ((vertex id, (x, y, z)) pairs) with the given edge ids."""
    pts = [rat(vid, *xyz) for vid, xyz in corners]
    n = len(pts)
    pieces = [ca._Piece("seg", edges[i], pts[i], pts[(i + 1) % n]) for i in range(n)]
    normal, d = plane
    drop = [abs(c) for c in normal].index(max(abs(c) for c in normal))
    return ca._View(view_id, False, [p.vertex for p in pts], pieces, frozenset(p.vertex for p in pts),
                    frozenset(edges), n=tuple(F(c) for c in normal), d=F(d), origin=(F(0),) * 3,
                    axes=tuple(a for a in range(3) if a != drop))


SQUARE_A = [(1, (0, 0, 0)), (2, (2, 0, 0)), (3, (2, 2, 0)), (4, (0, 2, 0))]


def embedding_error(*views):
    return refusal_of(lambda: ca._check_embedding(kern(), list(views)))


def test_planar_faces_sharing_exactly_their_declared_edge_embed():
    a = rect_view(1, ((0, 0, 1), 0), SQUARE_A, [11, 12, 13, 14])
    b = rect_view(2, ((0, 1, 0), 0), [(1, (0, 0, 0)), (2, (2, 0, 0)), (5, (2, 0, 2)), (6, (0, 0, 2))], [11, 21, 22, 23])
    ca._check_embedding(kern(), [a, b])
    # parallel faces on different planes never meet
    c = rect_view(3, ((0, 0, 1), 5), [(20, (0, 0, 5)), (21, (2, 0, 5)), (22, (2, 2, 5)), (23, (0, 2, 5))],
                  [41, 42, 43, 44])
    ca._check_embedding(kern(), [a, c])


def test_planar_faces_meeting_beyond_their_shared_cell_are_self_intersecting():
    a = rect_view(1, ((0, 0, 1), 0), SQUARE_A, [11, 12, 13, 14])
    longer = rect_view(2, ((0, 1, 0), 0), [(1, (0, 0, 0)), (7, (3, 0, 0)), (8, (3, 0, 2)), (6, (0, 0, 2))],
                       [31, 32, 33, 34])
    error = embedding_error(a, longer)
    assert error.category is C.SELF_INTERSECTING


def test_planar_faces_that_straddle_or_are_coplanar_are_embedding_undecided():
    a = rect_view(1, ((0, 0, 1), 0), SQUARE_A, [11, 12, 13, 14])
    straddling = rect_view(2, ((0, 1, 0), 0), [(9, (0, 0, -1)), (10, (2, 0, -1)), (11, (2, 0, 1)), (12, (0, 0, 1))],
                           [51, 52, 53, 54])
    error = embedding_error(a, straddling)
    assert error.category is C.EMBEDDING_UNDECIDED and error.sub_reason == "straddle"
    coplanar = rect_view(3, ((0, 0, 1), 0), [(30, (1, 1, 0)), (31, (3, 1, 0)), (32, (3, 3, 0)), (33, (1, 3, 0))],
                         [61, 62, 63, 64])
    error = embedding_error(a, coplanar)
    assert error.category is C.EMBEDDING_UNDECIDED and error.sub_reason == "coplanar"


def curved_views(name):
    model, ids, p = prepared(name)
    k = kern()
    pts = ca._points(k, p.snapshot)
    return k, ca._build_views(k, p.snapshot, pts)


def test_a_planar_face_whose_plane_cuts_the_bow_is_embedding_undecided():
    k, views = curved_views("half")
    bow = next(v for v in views if v.curved)
    wall = seg_view([(F(5, 2), 0, F(1, 2)), (F(5, 2), 10, F(1, 2)), (F(5, 2), 10, 3), (F(5, 2), 0, 3)], view_id=99,
                    plane=((1, 0, 0), F(5, 2)), edge_base=700, vertex_base=1000)
    wall.box = ca._view_box(k, wall)
    error = refusal_of(lambda: ca._curved_planar(k, bow, wall))
    assert error.category is C.EMBEDDING_UNDECIDED and error.sub_reason == "arc_meets_plane"


def test_a_ruling_in_the_plane_and_unshared_contact_are_embedding_undecided():
    k, views = curved_views("fa")
    face = next(v for v in views if v.curved)
    # y = x - 1/4 is tangent to y = u^2 along the ruling u = 1/2 (the face spans u in [1/8, 7/8])
    tangent = seg_view([(F(2, 5), F(3, 20), 0), (F(3, 5), F(7, 20), 0), (F(3, 5), F(7, 20), 1), (F(2, 5), F(3, 20), 1)],
                       view_id=98, plane=((-1, 1, 0), F(-1, 4)), edge_base=710, vertex_base=1100)
    error = refusal_of(lambda: ca._ruling_check(k, face, tangent))
    assert error.category is C.EMBEDDING_UNDECIDED and error.sub_reason == "ruling_in_plane"
    # a plane z = 1/4 face over the curved face's bottom trim, without sharing it: the trim, not a corner, meets it
    overlap = seg_view([(F(1, 5), 0, F(1, 4)), (F(3, 10), 0, F(1, 4)), (F(3, 10), F(1, 5), F(1, 4)),
                        (F(1, 5), F(1, 5), F(1, 4))], view_id=97, plane=((0, 0, 1), F(1, 4)), edge_base=720,
                       vertex_base=1200)
    overlap.box = ca._view_box(k, overlap)
    error = refusal_of(lambda: ca._curved_planar(k, face, overlap))
    assert error.category is C.EMBEDDING_UNDECIDED and error.sub_reason == "trim_contact"
    # the same plane with a face around the corner (3/8, 9/64, 1/4): corner_contact
    around = seg_view([(F(3, 10), 0, F(1, 4)), (F(1, 2), 0, F(1, 4)), (F(1, 2), F(1, 5), F(1, 4)),
                       (F(3, 10), F(1, 5), F(1, 4))], view_id=95, plane=((0, 0, 1), F(1, 4)), edge_base=740,
                      vertex_base=1400)
    around.box = ca._view_box(k, around)
    error = refusal_of(lambda: ca._curved_planar(k, face, around))
    assert error.category is C.EMBEDDING_UNDECIDED and error.sub_reason == "corner_contact"
    apart = seg_view([(F(3, 5), F(1, 2), F(1, 4)), (F(9, 10), F(1, 2), F(1, 4)), (F(9, 10), F(9, 10), F(1, 4)),
                      (F(3, 5), F(9, 10), F(1, 4))], view_id=96, plane=((0, 0, 1), F(1, 4)), edge_base=730,
                     vertex_base=1300)
    apart.box = ca._view_box(k, apart)
    ca._curved_planar(k, face, apart)                                  # the exact boxes are disjoint: no contact


def test_a_plane_through_two_opposite_corners_that_splits_the_loop_is_a_curved_straddle():
    """2x - z = 1/2 passes through the corners (3/8, 9/64, 1/4) and (5/8, 25/64, 3/4) of F-A with two arcs on each
    side and no interior crossing: the face lies on both sides of the plane."""
    k, views = curved_views("fa")
    face = next(v for v in views if v.curved)
    plane = seg_view([(F(2, 5), F(1, 2), F(3, 10)), (F(3, 5), F(1, 2), F(7, 10)), (F(3, 5), F(9, 10), F(7, 10)),
                      (F(2, 5), F(9, 10), F(3, 10))], view_id=94, plane=((2, 0, -1), F(1, 2)), edge_base=750,
                     vertex_base=1500)
    plane.box = ca._view_box(k, plane)
    error = refusal_of(lambda: ca._curved_planar(k, face, plane))
    assert error.category is C.EMBEDDING_UNDECIDED and error.sub_reason == "straddle"


def test_a_segment_between_algebraic_points_without_a_rational_line_is_refused():
    k, deck = _deck_view()
    arc, a_seg, _b_seg, c_seg = deck.pieces
    chord = ca._Piece("seg", 905, arc.b, arc.a)               # the chord between the two algebraic arc ends
    error = refusal_of(lambda: ca._check_planar_loop(k, replace(deck, pieces=[arc, a_seg, chord, c_seg])))
    assert error.category is C.UNDECIDED_PREDICATE and error.sub_reason == "irrational_segment_line"


def test_an_arc_off_the_plane_of_its_face_is_not_exactly_planar_although_its_vertices_are_in_it():
    k, views = curved_views("fa")
    arc_view = next(v for v in views if not v.curved and any(p.kind == "arc" for p in v.pieces))
    arc = next(p for p in arc_view.pieces if p.kind == "arc")
    a, b = arc.a, arc.b
    apex = rat(3000, F(1, 5), F(1, 20), F(9, 10))
    ab = tuple(b.rat[i] - a.rat[i] for i in range(3))
    ax = tuple(apex.rat[i] - a.rat[i] for i in range(3))
    normal = (ab[1] * ax[2] - ab[2] * ax[1], ab[2] * ax[0] - ab[0] * ax[2], ab[0] * ax[1] - ab[1] * ax[0])
    d = sum(n * c for n, c in zip(normal, a.rat))
    pieces = [arc, ca._Piece("seg", 3001, b, apex), ca._Piece("seg", 3002, apex, a)]
    view = replace(arc_view, pieces=pieces, n=normal, d=d, vids=[a.vertex, b.vertex, 3000])
    error = refusal_of(lambda: ca._check_supports(k, [view]))
    assert error.category is C.NOT_EXACTLY_PLANAR and "arc edge" in error.detail


def test_two_curved_faces_with_overlapping_boxes_are_embedding_undecided():
    k, views = curved_views("half")
    bow = next(v for v in views if v.curved)
    twin = replace(bow, id=77)
    error = refusal_of(lambda: ca._check_embedding(k, [bow, twin]))
    assert error.category is C.EMBEDDING_UNDECIDED and error.sub_reason == "curved_curved"


def test_overlapping_bows_in_a_hull_are_refused():
    from anygeometry.b1_example import b1_bow_surface
    from anygeometry.nurbs import NurbsSurface
    s = b1_bow_surface()
    shifted = NurbsSurface(s.degree_u, s.degree_v, s.knots_u, s.knots_v,
                           tuple(tuple((x, y + .5, z) for x, y, z in row) for row in s.controls), s.weights,
                           s.u_range, s.v_range)
    model, faces = solids.hull_with_port(shifted)
    error = refused(model, list(faces.values()))
    assert error.category is C.EMBEDDING_UNDECIDED


def test_a_shell_missing_a_face_is_not_closed_and_a_hole_is_invalid():
    model, faces = half_hull()
    ids = [v for k_, v in faces.items() if k_ != "centre"]
    assert refused(model, ids).category is C.NOT_CLOSED
    model, ids, p = prepared("fc")
    bottom = ids[-1]
    face = model.faces[bottom]
    model._put_entity("face", replace(face, holes=(solids._reverse(face.loop),)))
    try:
        error = refused(model, ids)
        assert error.category is C.INVALID_HOLES and error.sub_reason == "holes"
    finally:
        model._put_entity("face", face)
    assert prepare_certified_hydrostatics(model, ids).snapshot_digest == p.snapshot_digest


def test_a_planar_face_off_its_support_plane_is_not_exactly_planar():
    """The centre face has straight edges only: only the exact vertex-in-plane test can refuse it; the deck face
    also has an arc, which the arc test refuses with the same category."""
    model, faces = half_hull()
    ids = list(faces.values())
    for key, plane in (("centre", Plane((0., 1e-7, 0.), (1., 0., 0.), (0., 0., 1.))),
                       ("deck", Plane((0., 0., 3.0000001), (1., 0., 0.), (0., 1., 0.)))):
        face = model.faces[faces[key]]
        model._put_entity("face", replace(face, surface=plane))
        try:
            error = refused(model, ids)
            assert error.category is C.NOT_EXACTLY_PLANAR and "support plane" in error.detail or key == "deck"
            assert error.category is C.NOT_EXACTLY_PLANAR
        finally:
            model._put_entity("face", face)


def test_a_zero_length_planar_edge_is_an_invalid_loop():
    view = seg_view([(0, 0), (0, 0), (1, 1), (0, 2)])
    error = refusal_of(lambda: ca._check_supports(kern(), [view]))
    assert error.category is C.INVALID_LOOP and "zero length" in error.detail


# ------------------------------------------------------------------ unshared and doubly shared curved edges

def test_an_arc_edge_not_shared_with_a_curved_face_is_refused(monkeypatch):
    from anygeometry.certified_hydrostatics import _ch3_collect as collect
    model, ids, p = prepared("fa")
    real = collect.collect_curved

    def dropping(*args, **kwargs):
        data = real(*args, **kwargs)
        arcs = dict(data.arcs)
        arcs.pop(next(iter(arcs)))
        return replace(data, arcs=arcs)

    monkeypatch.setattr(collect, "collect_curved", dropping)
    error = refused(model, ids)
    assert error.category is C.UNSUPPORTED_TRIM and "not shared with a curved face" in error.detail


def test_a_section_arc_shared_by_two_curved_faces_is_refused():
    from anygeometry.entities import Face
    model, face = polygon(SQUARE_ISH)
    original = model.faces[face]
    twin = model._next_id["face"]
    model._next_id["face"] = twin + 1
    model._put_entity("face", Face(twin, solids._reverse(original.loop), surface=original.surface))
    error = refused(model, [face, twin])
    assert error.category is C.UNSUPPORTED_TRIM and error.sub_reason == "curved_edge_shared"


# ------------------------------------------------------------------ metering of the new loops

def _ops():
    bow = next(iter(prepared("half")[2].snapshot.curved.faces.values()))
    model, ids, p = prepared("fc")
    fc = next(iter(p.snapshot.curved.faces.values()))
    from anygeometry.certified_hydrostatics._ch3_collect import collect_curved
    return {"loop_b1": lambda k: check_loop(k, copy.deepcopy(bow)),
            "sigma_b1": lambda k: loop_sigma(k, copy.deepcopy(bow)),
            "monotone_b1": lambda k: is_v_monotone(k, copy.deepcopy(bow)),
            "loop_fc": lambda k: (check_loop(k, fc), loop_sigma(k, fc)),
            "collect_fc": lambda k: collect_curved(model, list(p.snapshot.curved.faces), k.meter, POLICY),
            "admit_fc": lambda k: ca.admit_curved(p.snapshot, k.meter, POLICY)}


def _units(op):
    budget = WorkBudget(None, 4096)
    op(kern(budget=budget))
    return budget.counts()


def test_hard_coded_unit_counts_of_the_loop_checks():
    ops = _ops()
    expected = {
        "loop_b1": (445, {"loop_pair": 28, "poly_coefficient": 347, "sign_count": 32, "root_refine": 10,
                          "bernstein_coefficient": 28}),
        "sigma_b1": (1725, {"sigma_round": 1, "loop_pair": 4, "poly_coefficient": 1168, "piece_split": 10,
                            "interval_op": 206, "bernstein_coefficient": 336}),
        "monotone_b1": (17, {"loop_pair": 17}),
        "loop_fc": (235, {"loop_pair": 32, "poly_coefficient": 190, "sign_count": 8, "root_refine": 2,
                          "sigma_round": 1, "interval_op": 2}),
        "collect_fc": (1621, {"face_read": 1, "admission_element": 129, "poly_coefficient": 1160,
                              "bernstein_coefficient": 219, "sign_count": 48, "root_refine": 29, "loop_pair": 32,
                              "sigma_round": 1, "interval_op": 2}),
        "admit_fc": (17477, {"admission_element": 742, "poly_coefficient": 4252, "loop_pair": 74, "plane_test": 136,
                             "predicate_round": 329, "interval_op": 4392, "bernstein_coefficient": 6464,
                             "sign_count": 304, "root_refine": 96, "edge_use": 24, "link_corner": 24, "bbox_pair": 15,
                             "interval_pair": 72, "volume_round": 1, "volume_piece": 552})}
    for name, (total, units) in expected.items():
        counts = _units(ops[name])
        assert counts["operations"] == total and counts["units"] == units, name


@pytest.mark.parametrize("name", ["loop_b1", "sigma_b1", "monotone_b1", "loop_fc", "collect_fc", "admit_fc"])
def test_cancellation_deadline_and_budget_stop_every_new_loop_at_its_checkpoint(name):
    op = _ops()[name]
    total = _units(op)["operations"]
    for stop in sorted({0, 1, total // 3, total // 2, total - 1}):
        budget = WorkBudget(None, 4096)
        error = refusal_of(lambda b=budget, s=stop: op(kern(budget=b, cancel=lambda: b.operations >= s)))
        assert error.category is C.CANCELLED and budget.operations == stop
        holder = []
        budget = WorkBudget(None, 4096, deadline=5.0, clock=lambda s=stop: 10.0 if holder[0].operations >= s else 0.0)
        holder.append(budget)
        error = refusal_of(lambda b=budget: op(kern(budget=b)))
        assert error.category is C.DEADLINE_EXCEEDED and budget.operations == stop
        budget = WorkBudget(stop, 4096)
        error = refusal_of(lambda b=budget: op(kern(budget=b)))
        assert error.category is C.BUDGET_EXHAUSTED and budget.operations == stop
    budget = WorkBudget(total, 4096)
    op(kern(budget=budget))
    assert budget.operations == total
    # every unit kind the loops reserve is a checkpoint: cancel at the first unit of each kind
    for kind in _units(op)["units"]:
        budget = WorkBudget(None, 4096)
        error = refusal_of(lambda b=budget, kd=kind: op(kern(budget=b, cancel=lambda: b.counts()["units"].get(kd, 0) >= 1)))
        assert error.category is C.CANCELLED


def test_the_bit_cap_refuses_the_curved_admission_loops():
    ops = _ops()
    caps = {"loop_b1": 300, "sigma_b1": 1000, "loop_fc": 8, "collect_fc": 40, "admit_fc": 100}
    for name, cap in caps.items():
        error = refusal_of(lambda n=name, c=cap: ops[n](kern(max_bits=c)))
        assert error.category is C.BIT_LENGTH_EXCEEDED
        ops[name](kern(max_bits=4096))                                  # and the roomy cap passes


def test_preparation_of_a_curved_model_is_a_budget_and_deadline_checkpoint():
    model, ids, p = prepared("fc")
    total = p.work["operations"]
    for limit in (0, 1, total // 2, total - 1):
        error = refused(model, ids, budget=WorkBudget(limit, None))
        assert error.category is C.BUDGET_EXHAUSTED and error.work["operations"] == limit
    prepare_certified_hydrostatics(model, ids, budget=WorkBudget(total, None))
    error = refused(model, ids, cancellation_check=lambda: True)
    assert error.category is C.CANCELLED
