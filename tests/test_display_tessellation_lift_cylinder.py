"""N3b: face display meshes on cylinders (SPEC-N3 section 4, slice N3b; SPEC-N3b).

T8-T11 and the acceptance cases: every accepted mesh is checked against independent oracles: the
exact (Fraction) chart combinatorics of the N3a tests and a 50-digit mpmath paired deviation and
footprint orientation against the EXACT orthonormal frame about the support's axis (the frame
the module's own float frame approximates).
"""
from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import pytest

import display_tessellation_corpus as corpus
import test_display_tessellation_lift as N3a
from anygeometry import _slab_triangulation as slab
from anygeometry import display_tessellation as dt
from anygeometry.entities import OrientedEdge
from anygeometry.exact_curves import EllipticArc
from anygeometry.model import GeometryModel
from anygeometry.surfaces import Cylinder, surface_normal

_mesh, _refused = N3a._mesh, N3a._refused


# ----------------------------------------------------------------------------- oracle
def _mp_frame(support):
    """The exact orthonormal frame about the support's (normalised) axis that the module's float
    frame approximates: ``e1 = (a x b) / |a x b|`` with ``b`` the basis vector of the axis's
    smallest component, ``e2 = a x e1``."""
    axis = np.asarray(support.axis, dtype=float)
    pick = int(np.argmin(np.abs(axis / math.hypot(*axis))))      # the same CHOICE as the module
    a = [mp.mpf(float(x)) for x in axis]
    na = mp.sqrt(sum(x * x for x in a))
    a = [x / na for x in a]
    b = [mp.mpf(0)] * 3
    b[pick] = mp.mpf(1)
    c = _mp_cross(a, b)
    nc = mp.sqrt(sum(x * x for x in c))
    e1 = [x / nc for x in c]
    return [mp.mpf(float(x)) for x in support.origin], e1, _mp_cross(a, e1), a


def _mp_cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _face_tolerance(model, face, tol):
    if tol.sag is not None:
        return tol.sag
    f = model.faces[face]
    a = max(dt.display_rounding_allowance(model, u.edge, tol).angle for loop in (f.loop, *f.holes) for u in loop)
    return f.surface.radius * (tol.angle + a) ** 2 / 8 + model.tolerance.surface_residual


def _grid(n):
    return [(i / n, j / n, (n - i - j) / n) for i in range(n + 1) for j in range(n + 1 - i)]


#: 33 barycentric points: the 28-point lattice of step 1/6 and five interior points
GRID_33 = _grid(6) + [(1 / 3, 1 / 3, 1 / 3), (.7, .2, .1), (.1, .7, .2), (.2, .1, .7), (.45, .45, .1)]


def _check_cylinder(model, face, tol, mesh, grid=GRID_33, every=1):
    """Independent oracles of every published claim of an accepted cylinder face mesh."""
    support = model.faces[face].surface
    rings = N3a._n1_rings(model, face, tol)
    N3a._check_combinatorics(mesh, rings)
    assert mesh.face_tolerance == _face_tolerance(model, face, tol)
    assert 0.0 < mesh.deviation_bound <= mesh.face_tolerance
    assert mesh.trim_sag_bound >= 0.0
    r = mp.mpf(float(support.radius))
    with mp.workdps(50):
        o, e1, e2, a = _mp_frame(support)

        def s_star(theta, h):
            ct, st = mp.cos(theta), mp.sin(theta)
            return [oo + r * (ct * x + st * y) + h * z for oo, x, y, z in zip(o, e1, e2, a)]
        pts = [[mp.mpf(float(x)) for x in p] for p in mesh.points]
        chart = [[mp.mpf(float(x)) for x in c] for c in mesh.chart]
        bound = mp.mpf(mesh.deviation_bound)
        worst = mp.mpf(0)
        folded = 0
        for t in mesh.triangles.tolist()[::every]:
            # paired deviation over the barycentric grid
            for w in grid:
                x = [sum(mp.mpf(wi) * pts[i][d] for wi, i in zip(w, t)) for d in range(3)]
                th = sum(mp.mpf(wi) * chart[i][0] for wi, i in zip(w, t))
                h = sum(mp.mpf(wi) * chart[i][1] for wi, i in zip(w, t))
                worst = max(worst, mp.sqrt(sum((xx - ss) ** 2 for xx, ss in zip(x, s_star(th, h)))))
            # orientation over the footprint: sense x outward radial, at 9 azimuths
            p, q, s = (pts[i] for i in t)
            n = _mp_cross([y - x for x, y in zip(p, q)], [y - x for x, y in zip(p, s)])
            ths = [chart[i][0] for i in t]
            lo, hi = min(ths), max(ths)
            if not all(mesh.sense * sum(nn * (mp.cos(tt) * x + mp.sin(tt) * y) for nn, x, y in zip(n, e1, e2)) > 0
                       for tt in (lo + (hi - lo) * j / 8 for j in range(9))):
                folded += 1
                width = mp.sqrt(sum(x * x for x in n)) / max(
                    mp.sqrt(sum((y - x) ** 2 for x, y in zip(u, v))) for u, v in ((p, q), (q, s), (s, p)))
                assert width <= mesh.deviation_bound, "a folded triangle is not rounding-thin"
        assert worst <= bound, (float(worst), mesh.deviation_bound)
    assert folded <= mesh.degenerate_count
    return mesh


# ----------------------------------------------------------------------------- fixtures
def _frame(axis):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    e1 = np.cross(axis, (1., 0., 0.) if abs(axis[0]) < .9 else (0., 1., 0.))
    e1 /= np.linalg.norm(e1)
    return axis, e1, np.cross(axis, e1)


def quarter_cylinder(origin=(0., 0., 0.), axis=(0., 0., 1.), radius=2.0, height=3.0, start=.2, sweep=math.pi / 2):
    """A quarter-cylinder face: two exact circle arcs (h = 0, h = height) and two rulings."""
    g = GeometryModel()
    o = np.asarray(origin, dtype=float)
    a, e1, e2 = _frame(axis)

    def at(t, h):
        return o + radius * (math.cos(t) * e1 + math.sin(t) * e2) + h * a
    t0, t1 = start, start + sweep
    p = [g.add_point(*at(t, h)) for t, h in ((t0, 0.), (t1, 0.), (t1, height), (t0, height))]
    low = g.add_curve(p[0], p[1], EllipticArc(tuple(o), tuple(radius * e1), tuple(radius * e2), t0, sweep))
    top = tuple(o + height * a)
    high = g.add_curve(p[3], p[2], EllipticArc(top, tuple(radius * e1), tuple(radius * e2), t0, sweep))
    support = Cylinder(tuple(o - a), tuple(a), tuple(e1), radius, height + 2.0)
    loop = (OrientedEdge(low, True), OrientedEdge(g.add_line(p[1], p[2]), True), OrientedEdge(high, False),
            OrientedEdge(g.add_line(p[3], p[0]), True))
    return g, g.add_face_from_loop(loop, (0, 1, 2, 3), surface=support)


def window_face(origin, axis, radius, height=None, start=.3, sweep=2.2, hole=(.35, .75, .42, .58)):
    """A cylinder patch (circle arcs, rulings) with a window hole: the hole's rulings lie inside
    the outer arcs' chords, so N2 cuts those chords on the hole's ruling azimuths (fuzz finding:
    the two ends of a ruling round to azimuths ulps apart)."""
    g = GeometryModel()
    o = np.asarray(origin, dtype=float)
    a, e1, e2 = _frame(axis)
    height = 2.0 * radius if height is None else height

    def loop(ta, tb, h0, h1, forward):
        def at(t, h):
            return o + radius * (math.cos(t) * e1 + math.sin(t) * e2) + h * a
        p = [g.add_point(*at(t, h)) for t, h in ((ta, h0), (tb, h0), (tb, h1), (ta, h1))]
        low = g.add_curve(p[0], p[1], EllipticArc(tuple(o + h0 * a), tuple(radius * e1), tuple(radius * e2), ta, tb - ta))
        high = g.add_curve(p[3], p[2], EllipticArc(tuple(o + h1 * a), tuple(radius * e1), tuple(radius * e2), ta, tb - ta))
        uses = [OrientedEdge(low, True), OrientedEdge(g.add_line(p[1], p[2]), True), OrientedEdge(high, False),
                OrientedEdge(g.add_line(p[3], p[0]), True)]
        return tuple(uses if forward else [OrientedEdge(u.edge, not u.forward) for u in reversed(uses)])
    outer = loop(start, start + sweep, 0.0, height, True)
    support = Cylinder(tuple(o - a), tuple(a), tuple(e1), radius, height + 2.0)
    face = g.add_face_from_loop(outer, (0, 1, 2, 3), surface=support)
    f0, f1, g0, g1 = hole
    N3a._force(g, face, holes=(loop(start + f0 * sweep, start + f1 * sweep, g0 * height, g1 * height, False),))
    return g, face


def _r8_sweep_face():
    """codex-n2r8 finding 2: a thin (2e-9) triangle on a 2e-4-sweep cylinder, two vertices 4e-9
    off the support; its normal is positive at the chart centroid and negative elsewhere."""
    g = GeometryModel()
    s = Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 1., 0., 2e-4)
    uv = np.array([(0., .5), (1., 0.), (0., .5 + 2e-9)])
    exact = np.array([s.evaluate(*p) for p in uv])
    xyz = exact.copy()
    xyz[[0, 2], 0] += [-4e-9, 4e-9]
    a, b, c = [g.add_point(*p) for p in xyz]
    edges = [g.add_line(a, b), g.add_line(b, c), g.add_line(c, a)]
    face = g.add_face(edges, surface=s)
    return g, face, s, uv, exact, xyz, edges


# ----------------------------------------------------------------------------- T8 (Q3)
@pytest.mark.parametrize("kind", ["axis_z", "tilted"])
def test_t8_q3_quarter_cylinder_vertices_orientation_and_paired_deviation(kind):
    """Ring vertices are N1's samples bit for bit, every triangle faces outward per sigma, and the
    mpmath paired error over a 33-point barycentric grid per triangle is <= deviation_bound <=
    T_face.  Naive: the support evaluated at a vertex's own chart point does not return it."""
    if kind == "axis_z":
        g, face = quarter_cylinder()
    else:
        g, face = quarter_cylinder(origin=(5., -3., 7.), axis=(1., 2., 2.), radius=.7, height=1.3, start=-2.9)
    for step in corpus.STEPS:
        tol = dt.DisplayTolerance(step)
        mesh = _check_cylinder(g, face, tol, _mesh(g, face, tol))
        assert mesh.sense == 1                                # the loop runs CCW in (theta, h): outward
        k = mesh.ring_vertex_count
        assert mesh.points[:k].tobytes() == np.vstack(N3a._n1_rings(g, face, tol)).tobytes()
        o, e1, e2 = dt._axis_frame(g.faces[face].surface)
        axis = np.asarray(g.faces[face].surface.axis)
        thin = 0
        for t in mesh.triangles:
            a, b, c = mesh.points[t]
            n = np.cross(b - a, c - a)
            longest = max(np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c))
            if np.linalg.norm(n) <= mesh.deviation_bound * longest:
                thin += 1                                      # rounding-thin (ulp-wide slabs at the rulings)
                continue
            radial = (a + b + c) / 3 - o
            radial -= (radial @ axis) * axis
            assert n @ radial > 0                              # outward
        assert mesh.degenerate_count <= thin
        r = g.faces[face].surface.radius
        rebuilt = o[None] + r * (np.cos(mesh.chart[:k, :1]) * e1 + np.sin(mesh.chart[:k, :1]) * e2) \
            + mesh.chart[:k, 1:] * axis
        if kind == "tilted":                                 # naive: evaluating the chart moves them
            assert not np.array_equal(rebuilt, mesh.points[:k])


# ----------------------------------------------------------------------------- T9 (Q7)
def test_t9_q7_a_sign_change_over_the_footprint_is_refused():
    """r8 finding 2: the naive centroid test passes, the normal is negative elsewhere on the
    footprint, the paired bound is met: N3 refuses TRIM_INVALID (the triangle's 3D width 2e-9 is
    far above the rounding-thin exemption)."""
    g, face, s, uv, exact, xyz, edges = _r8_sweep_face()
    tol = dt.DisplayTolerance(.1)
    ring = np.vstack([dt.edge_display_samples(g, e, tol).points[:-1] for e in edges])
    np.testing.assert_array_equal(ring, xyz)
    np.testing.assert_array_equal(slab.triangulate_rings([uv]).triangles, [[0, 1, 2]])
    offset = np.linalg.norm(ring - exact, axis=1).max()
    assert offset + s.sweep_angle ** 2 / 8 < g.tolerance.surface_residual
    n = np.cross(ring[1] - ring[0], ring[2] - ring[0])
    assert n @ surface_normal(s, *uv.mean(axis=0)) > 0               # naive centroid: accepted
    assert n @ surface_normal(s, *(np.array([.05, .9, .05]) @ uv)) < 0
    text = _refused("TRIM_INVALID", lambda: _mesh(g, face, tol))
    assert "folds" in text


# ----------------------------------------------------------------------------- T10 (winding)
def test_t10_a_loop_winding_the_axis_is_refused():
    """A full band between two whole circles (no seam edge): both loops wind the axis."""
    g = GeometryModel()
    support = Cylinder((0., 0., -1.), (0., 0., 1.), (1., 0., 0.), 1., 4.)
    loops = []
    for z, forward in ((0., True), (2., False)):
        p = [g.add_point(math.cos(t), math.sin(t), z) for t in np.arange(6) * math.pi / 3]
        arcs = [g.add_arc(p[0], p[1], p[2]), g.add_arc(p[2], p[3], p[4]), g.add_arc(p[4], p[5], p[0])]
        loop = [OrientedEdge(e, True) for e in arcs]
        loops.append(tuple(loop) if forward else tuple(OrientedEdge(u.edge, False) for u in reversed(loop)))
    face = g.add_face([u.edge for u in loops[0]], surface=support)
    N3a._force(g, face, loop=loops[0], holes=(loops[1],))
    _refused("TRIM_WINDS_SUPPORT", lambda: _mesh(g, face, dt.DisplayTolerance(.3)))
    N3a._force(g, face, loop=loops[0], holes=())                   # one winding loop alone
    _refused("TRIM_WINDS_SUPPORT", lambda: _mesh(g, face, dt.DisplayTolerance(.3)))


# ----------------------------------------------------------------------------- T11 + acceptance
@pytest.mark.parametrize("step", corpus.STEPS, ids=["fine", "interaction"])
def test_t11_tripod_skins_are_all_accepted_and_certified(step):
    """All 18 tripod skins (half tubes, mitre ellipses, rulings) accepted at the ANYfem steps;
    the certificates hold.  N3.0 measured ulp-wide slabs (rounding-thin triangles) on 12 skins;
    the lift joins azimuths within rounding, so none remain (counted, not required)."""
    case = corpus.tripod_skins()
    tol = dt.DisplayTolerance(step)
    exempt = []
    for label, face in case.faces.items():
        mesh = _check_cylinder(case.model, face, tol, _mesh(case.model, face, tol), grid=_grid(3))
        exempt.append(mesh.degenerate_count)
    assert len(exempt) == 18


@pytest.mark.parametrize("step", corpus.STEPS, ids=["fine", "interaction"])
def test_generated_cylinder_400_faces_are_all_accepted(step):
    case = corpus.generated_cylinder(400)
    tol = dt.DisplayTolerance(step)
    for i, face in enumerate(case.faces.values()):
        mesh = _mesh(case.model, face, tol)
        if i % 40 == 0:
            _check_cylinder(case.model, face, tol, mesh, grid=_grid(3))
        assert mesh.degenerate_count == 0 and mesh.deviation_bound <= mesh.face_tolerance


@pytest.mark.parametrize("origin, axis, radius", [((31., -17., 44.), (1., 2., 2.), 1.9),
                                                  ((5., -3., 7.), (2., -1., 3.), .7),
                                                  ((-40., 12., 9.), (0., 1., 1.), .25)])
def test_a_window_hole_beside_long_chords_is_accepted(origin, axis, radius, monkeypatch):
    """Fuzz finding (N3b): the two ends of a hole's ruling round to azimuths a few ulps apart;
    the ulp-wide slab between them puts the cut of a long outer chord (at the chart azimuth, but
    ON the 3D chord, tangentially off it by ~ dT^3) in a sliver triangle that truly folds (3D
    width ~1e-3, far above rounding).  The lift joins chart azimuths within their rounding
    (``_SNAP``): accepted and certified.  Without the join the fold is refused."""
    from anygeometry import _display_lift as lift
    g, face = window_face(origin, axis, radius)
    tol = dt.DisplayTolerance(.51)
    _check_cylinder(g, face, tol, _mesh(g, face, tol))
    monkeypatch.setattr(lift, "_SNAP", 0.0)
    assert "folds" in _refused("TRIM_INVALID", lambda: _mesh(g, face, tol))


@pytest.mark.parametrize("angle", [.1, .3])
def test_refinement_bisects_slabs_whose_triangles_exceed_t_face(angle, monkeypatch):
    """The window's hole arcs are sampled out of phase with the outer arcs: N2's strip triangles
    join cuts near the middle of full-step chords (variance near T_face's own dT^2 / 4) across
    wide slabs, and their paired bound exceeds T_face (refused without refinement).  Bisecting
    those slabs brings every triangle under T_face (oracle-checked)."""
    from anygeometry import _display_lift as lift
    g, face = window_face((0., 0., 0.), (0., 0., 1.), 1.0)
    tol = dt.DisplayTolerance(angle)
    refined = _check_cylinder(g, face, tol, _mesh(g, face, tol), grid=_grid(4))
    plain = slab.triangulate_rings([c.copy() for c in _charts(g, face, tol)])
    assert len(refined.triangles) > len(plain.triangles)
    monkeypatch.setattr(lift, "_REFINE_ROUNDS", 0)
    assert "deviation bound" in _refused("TRIM_INVALID", lambda: _mesh(g, face, tol))


def _charts(g, face, tol):
    from anygeometry import _display_lift as lift
    f = g.faces[face]
    rings = [lift._ring(g, face, loop, tol) for loop in (f.loop, *f.holes)]
    with np.errstate(all="ignore"):
        return lift._CylinderChart(f.surface).charts(face, rings)


# ----------------------------------------------------------------------------- Q5 / Q4 rules
def test_q5_face_tolerance_default_and_sag():
    g, face = quarter_cylinder()
    tol = dt.DisplayTolerance(.2)
    mesh = _mesh(g, face, tol)
    assert mesh.face_tolerance == _face_tolerance(g, face, tol)
    assert mesh.face_tolerance > 2.0 * .2 ** 2 / 8
    tol = dt.DisplayTolerance(.2, sag=1e-3)
    mesh = _check_cylinder(g, face, tol, _mesh(g, face, tol))
    assert mesh.face_tolerance == 1e-3


def test_q4_a_fixed_n1_chord_over_the_face_tolerance_is_edge_unsupported():
    """The r8 fixture's skew line is one fixed N1 chord (a line has no sag to refine) whose paired
    deviation r dT^2 / 8 ~ 5e-9 exceeds T_face = tol.sag = 1e-12."""
    g, face, *_ = _r8_sweep_face()
    text = _refused("EDGE_UNSUPPORTED", lambda: _mesh(g, face, dt.DisplayTolerance(.1, sag=1e-12)))
    assert "face tolerance" in text


# ----------------------------------------------------------------------------- E2 / E3 (N3b decisions)
def _pinch_face(nudge):
    """A quarter cylinder with a triangular hole whose vertex is an interior N1 sample of the
    outer bottom arc moved ``nudge`` ulps in x: a distinct 3D point at the same height whose
    azimuth is within rounding of the sample's."""
    g, face = quarter_cylinder()
    tol = dt.DisplayTolerance(.3)
    s = dt.edge_display_samples(g, g.faces[face].loop[0].edge, tol)
    p = s.points[len(s.points) // 2].copy()
    theta = math.atan2(p[1], p[0])
    for _ in range(nudge):
        p[0] = np.nextafter(p[0], np.inf)
    q = [g.add_point(*p), g.add_point(2 * math.cos(theta - .1), 2 * math.sin(theta - .1), 1.),
         g.add_point(2 * math.cos(theta + .1), 2 * math.sin(theta + .1), 1.)]
    edges = [g.add_line(a, b) for a, b in zip(q, q[1:] + q[:1])]
    N3a._force(g, face, holes=(tuple(OrientedEdge(e, True) for e in edges),))
    return g, face, tol, s.points[len(s.points) // 2], p


def test_e2_joined_azimuths_keep_the_ring_vertices_bit_for_bit():
    """E2: the join changes chart azimuths only (the window fixture's ruling ends are joined);
    every ring vertex is still its N1 sample bit for bit."""
    from anygeometry import _display_lift as lift
    g, face = window_face((31., -17., 44.), (1., 2., 2.), 1.9)
    tol = dt.DisplayTolerance(.51)
    joined = np.vstack(_charts(g, face, tol))
    saved = lift._SNAP
    try:
        lift._SNAP = 0.0
        raw = np.vstack(_charts(g, face, tol))
    finally:
        lift._SNAP = saved
    assert not np.array_equal(joined, raw) and np.array_equal(joined[:, 1], raw[:, 1])
    mesh = _check_cylinder(g, face, tol, _mesh(g, face, tol))
    k = mesh.ring_vertex_count
    assert mesh.points[:k].tobytes() == np.vstack(N3a._n1_rings(g, face, tol)).tobytes()
    np.testing.assert_array_equal(mesh.chart[:k], joined)


@pytest.mark.parametrize("nudge", [1, 4])
def test_e2_a_join_that_would_merge_two_n1_samples_is_refused(nudge):
    """E2: two distinct N1 samples (a hole vertex ulps from an outer sample, same height) whose
    azimuths are joined would become one chart point: TRIM_INVALID with the lift's own text."""
    g, face, tol, sample, moved = _pinch_face(nudge)
    assert not np.array_equal(sample, moved)
    text = _refused("TRIM_INVALID", lambda: _mesh(g, face, tol))
    assert "two ring points coincide in the chart" in text and text.startswith("TRIM_INVALID: face 1:")


def test_e3_refinement_stops_after_eight_bisection_rounds(monkeypatch):
    """E3: failing slabs are bisected at most 8 rounds, then TRIM_INVALID (deviation).  With
    ``tol.sag`` exactly the worst boundary-chord bound the boundary passes, but the triangles at
    a cut near a chord's middle carry that chord's variance plus rounding terms, which no
    bisection removes: 8 rounds, then refused.  A sag 1e-9 relative larger is accepted."""
    from anygeometry import _display_lift as lift
    g, face = window_face((0., 0., 0.), (0., 0., 1.), 1.0)
    f = g.faces[face]
    tol = dt.DisplayTolerance(.3)
    rings = [lift._ring(g, face, loop, tol) for loop in (f.loop, *f.holes)]
    with np.errstate(all="ignore"):
        cyl = lift._CylinderChart(f.surface)
        charts = cyl.charts(face, rings)
        points, chart = np.vstack([r.points for r in rings]), np.vstack(charts)
        deltas = cyl.offsets(points, chart)
        following = lift._ring_structure(rings)[1]
        span = np.abs(chart[following, 0] - chart[:, 0])
        eps = lift._EPS
        worst = float((np.maximum(deltas, deltas[following]) * (1 + 8 * eps) + cyl.r * span ** 2 / 8 * (1 + 8 * eps)
                       + cyl.rounding(points, chart)).max())
    rounds = []
    original = lift._bisection
    monkeypatch.setattr(lift, "_bisection", lambda *a, **k: rounds.append(1) or original(*a, **k))
    text = _refused("TRIM_INVALID", lambda: _mesh(g, face, dt.DisplayTolerance(.3, sag=worst)))
    assert "deviation bound" in text and len(rounds) == lift._REFINE_ROUNDS == 8
    rounds.clear()
    _check_cylinder(g, face, dt.DisplayTolerance(.3, sag=worst * (1 + 1e-9)),
                    _mesh(g, face, dt.DisplayTolerance(.3, sag=worst * (1 + 1e-9))), grid=_grid(3))
    assert 1 <= len(rounds) < 8


def test_the_lift_frame_and_cross_product_are_n1_s_bit_for_bit():
    """N3c speed work: the lift's own frame (explicit cross products) is N1's ``_axis_frame`` and
    its ``_cross`` is ``np.cross``, bit for bit, on random and axis-aligned supports."""
    from anygeometry import _display_lift as lift
    rng = np.random.default_rng(2615)
    axes = [rng.normal(size=3) * 10.0 ** rng.uniform(-3, 3) for _ in range(300)]
    axes += [np.array(v, dtype=float) for v in ((0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 0, -1), (1, 1, 0), (1e-9, 0, 1))]
    for axis in axes:
        radial = np.cross(axis, rng.normal(size=3))
        support = Cylinder(tuple(rng.uniform(-50, 50, 3)), tuple(axis), tuple(radial), 1.0, 1.0)
        for mine, theirs in zip(lift._axis_frame(support), dt._axis_frame(support)):
            assert mine.tobytes() == theirs.tobytes()
    u, v = rng.normal(size=(500, 3)) * 1e3, rng.normal(size=(500, 3))
    assert lift._cross(u, v).tobytes() == np.cross(u, v).tobytes()
