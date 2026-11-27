"""N3a: face display meshes on planes (SPEC-N3 section 4, slice N3a; owner decisions 2026-11-26).

Each of T1-T7 first reproduces the defect a naive lift of the N2 chart mesh has (the ``_naive_*``
helpers and the asserts before the lift), then checks the certified result of
:func:`anygeometry.display_tessellation.face_display_mesh` against independent oracles: exact
(Fraction) chart combinatorics and orientation, and a 60-digit Decimal paired deviation.
"""
from __future__ import annotations

import math
from dataclasses import replace
from decimal import Decimal, localcontext
from fractions import Fraction as F

import numpy as np
import pytest

import display_tessellation_corpus as corpus
from anygeometry import _slab_triangulation as slab
from anygeometry import display_tessellation as dt
from anygeometry.entities import OrientedEdge
from anygeometry.errors import GeometryError
from anygeometry.model import GeometryModel
from anygeometry.surfaces import Cylinder, Plane

XY = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))


# ----------------------------------------------------------------------------- helpers
def _mesh(model, face, tol, **kwargs):
    """The lift under a caller error state that raises on every floating-point event (the result
    must not depend on it, as for N1 and N2)."""
    with np.errstate(all="raise"):
        if kwargs:
            from anygeometry import _display_lift
            return _display_lift._lift(model, face, tol, **kwargs)
        return dt.face_display_mesh(model, face, tol)


def _refused(code, call):
    with pytest.raises(dt.DisplayTrimRefused) as info:
        call()
    assert info.value.code == code, str(info.value)
    return str(info.value)


def _planar(rings, plane=XY, tolerance=None):
    """One planar face: ``rings[0]`` the outer loop, the rest holes; each ring a list of points
    (x, y[, z]) joined by lines in the given order.  The loops are set explicitly so the ring order
    (and hence N2's indices) is the fixture's own.  ``tolerance``: the model's TolerancePolicy."""
    g = GeometryModel() if tolerance is None else GeometryModel(tolerance=tolerance)
    loops = []
    for ring in rings:
        ids = [g.add_point(*p) for p in ring]
        edges = [g.add_line(a, b) for a, b in zip(ids, ids[1:] + ids[:1])]
        loops.append(tuple(OrientedEdge(e, True) for e in edges))
    face = g.add_face([u.edge for u in loops[0]], surface=plane)
    _force(g, face, loop=loops[0], holes=tuple(loops[1:]))
    return g, face


def _force(g, face, **changes):
    """Replace face fields WITHOUT the model's commit validation (outside a transaction): the
    lift must refuse invalid trims itself (touching holes, slits, broken loops)."""
    g._put_entity("face", replace(g.faces[face], **changes))


def _n1_rings(model, face, tol):
    """The face's rings from N1 alone (the reference for the bit-identity checks)."""
    rings = []
    for loop in (model.faces[face].loop, *model.faces[face].holes):
        parts = []
        for use in loop:
            s = dt.edge_display_samples(model, use.edge, tol)
            parts.append((s.points if use.forward else s.points[::-1])[:-1])
        rings.append(np.vstack(parts))
    return rings


def _fp(p):
    return tuple(F(float(x)) for x in p)


def _orient(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _decimal_frame(plane):
    """The exact orthonormal frame (to 60 digits) of the plane through ``plane``'s float vectors."""
    o, u, v = ([Decimal(float(x)) for x in vec] for vec in (plane.origin, plane.u_vector, plane.v_vector))
    nu = sum(x * x for x in u).sqrt()
    e1 = [x / nu for x in u]
    dot = sum(a * b for a, b in zip(v, e1))
    w = [a - dot * b for a, b in zip(v, e1)]
    nw = sum(x * x for x in w).sqrt()
    return o, e1, [x / nw for x in w]


def _check_certified(model, face, tol, mesh, grid_triangles=8):
    """Independent oracles of every published claim of an accepted planar face mesh."""
    plane = model.faces[face].surface
    rings = _n1_rings(model, face, tol)
    k = sum(len(r) for r in rings)
    assert mesh.face_id == face and mesh.ring_vertex_count == k
    # face tolerance rule (Q5), the bounds' order
    expected = tol.sag if tol.sag is not None else model.tolerance.surface_residual
    assert mesh.face_tolerance == expected
    assert 0.0 < mesh.deviation_bound <= mesh.face_tolerance
    assert mesh.trim_sag_bound >= 0.0 and mesh.sense in (1, -1)
    _check_combinatorics(mesh, rings)
    tris = mesh.triangles.tolist()
    # 3D orientation: every lifted triangle has an exactly positive normal against sense x (u x v),
    # or an exactly zero one (exempt, counted)
    n = _cross(_fp(plane.u_vector), _fp(plane.v_vector))
    pts = [_fp(p) for p in mesh.points]
    folded = 0
    for t in tris:
        a, b, c = (pts[i] for i in t)
        nt = _cross([y - x for x, y in zip(a, b)], [y - x for x, y in zip(a, c)])
        if not mesh.sense * sum(x * y for x, y in zip(nt, n)) > 0:
            # the module's exact exemption rule on a plane: only an exactly zero normal (Codex N3
            # r2 L2: never just a width within deviation_bound)
            folded += 1
            assert nt == (0, 0, 0), "a folded triangle with a nonzero exact normal (exemption rule)"
    assert folded <= mesh.degenerate_count
    # paired deviation (Decimal, 60 digits): the plane map is affine, so the vertex errors bound
    # the triangle; checked at the vertices and at a barycentric grid of the first triangles
    with localcontext() as ctx:
        ctx.prec = 60
        o, e1, e2 = _decimal_frame(plane)

        def error(x, c):
            s = [oo + Decimal(float(c[0])) * a + Decimal(float(c[1])) * b for oo, a, b in zip(o, e1, e2)]
            return sum((Decimal(float(xx)) - ss) ** 2 for xx, ss in zip(x, s)).sqrt()

        bound = Decimal(mesh.deviation_bound)
        assert all(error(x, c) <= bound for x, c in zip(mesh.points, mesh.chart))
        grid = [(F(i, 4), F(j, 4), F(4 - i - j, 4)) for i in range(5) for j in range(5 - i)]
        for t in tris[:grid_triangles]:
            for w in grid:
                x = [sum(Decimal(wi.numerator) / Decimal(wi.denominator) * Decimal(float(mesh.points[i][d]))
                         for wi, i in zip(w, t)) for d in range(3)]
                cc = [sum(Decimal(wi.numerator) / Decimal(wi.denominator) * Decimal(float(mesh.chart[i][d]))
                          for wi, i in zip(w, t)) for d in range(2)]
                s = [oo + cc[0] * a + cc[1] * b for oo, a, b in zip(o, e1, e2)]
                assert sum((xx - ss) ** 2 for xx, ss in zip(x, s)).sqrt() <= bound
    return mesh


def _check_combinatorics(mesh, rings):
    """Q2 (i)-(iv), exactly: an injective chart; every chart triangle CCW about sigma; every directed
    edge used once; the edges used by one triangle are exactly the chain of every ring (its vertices,
    then the cuts on each chord in order), with a chain edge that occurs twice (a zero-width spike
    where two cuts on two chords merged into one vertex, owner decision D1) cancelling."""
    k = sum(len(r) for r in rings)
    assert mesh.ring_vertex_count == k and mesh.sense in (1, -1)
    for name in ("points", "triangles", "chart", "source"):
        array = getattr(mesh, name)
        assert type(array) is np.ndarray and not array.flags.writeable, name
    assert mesh.points.dtype == np.float64 and mesh.triangles.dtype == np.int64 and mesh.source.dtype == np.int64
    assert mesh.points.shape == (len(mesh.chart), 3) and mesh.source.shape == (len(mesh.chart), 2)
    # ring vertices are N1's samples bit for bit (T7)
    assert mesh.points[:k].tobytes() == np.vstack(rings).tobytes()
    # sources: ring vertex j names the chord starting at it; a cut its chord
    ring_sources = {tuple(s) for s in mesh.source[:k].tolist()}
    assert all(tuple(s) in ring_sources for s in mesh.source[k:].tolist())
    q = [_fp(p) for p in mesh.chart]
    assert len(set(q)) == len(q)                                    # Q2 (i): injective chart
    tris = mesh.triangles.tolist()
    # every triangle is CCW about sigma x support normal: in the chart, sense x orient > 0, exactly
    assert all(mesh.sense * _orient(*(q[i] for i in t)) > 0 for t in tris)
    directed = [(t[i], t[(i + 1) % 3]) for t in tris for i in range(3)]
    assert len(set(directed)) == len(directed)
    present = set(directed)
    boundary = {e for e in directed if (e[1], e[0]) not in present}
    used_once = {frozenset(e) for e in boundary}
    assert len(used_once) == len(boundary)
    by_source = {}
    for j in range(k, len(q)):
        by_source.setdefault(tuple(mesh.source[j].tolist()), []).append(j)
    cut_x = mesh.chart[k:, 0]
    counts = {}
    start = 0
    for ring in rings:
        m = len(ring)
        for i in range(m):
            a, b = start + i, start + (i + 1) % m
            on = {j for j in by_source.get(tuple(mesh.source[a].tolist()), ()) if _on_segment(mesh, j, a, b)}
            lo, hi = sorted((mesh.chart[a, 0], mesh.chart[b, 0]))
            # a merged cut keeps one chord's source; on its other chord it is N2's cut there: its
            # ordinate is the chord's exact ordinate at its x, correctly rounded
            # (and, in 3D, it lies within the reported trim bound of this chord: exact rationals)
            on |= {int(j) + k for j in np.flatnonzero((cut_x > lo) & (cut_x < hi))
                   if _rounds_onto(mesh.chart, int(j) + k, a, b)
                   and _on_chord(mesh.points[a], mesh.points[b], mesh.points[int(j) + k], mesh.trim_sag_bound)}
            xa, xb = q[a][0], q[b][0]
            on = sorted(on, key=lambda j: q[j][0] if xa < xb else -q[j][0])
            nodes = [a, *on, b]
            for pair in zip(nodes, nodes[1:]):
                key = frozenset(pair)
                counts[key] = counts.get(key, 0) + 1
        start += m
    assert max(counts.values()) <= 2
    assert used_once == {e for e, n in counts.items() if n % 2}
    # a chain edge on two chords cancelled (D1): the trim interval it stands for is not on the mesh
    # boundary, so the lifted interval must be within the reported trim bound of it (H1)
    cancelled = sum(math.dist(*mesh.points[list(e)]) for e, n in counts.items() if n == 2)
    assert cancelled <= mesh.trim_sag_bound, (cancelled, mesh.trim_sag_bound)


def _rounds_onto(chart, j, a, b):
    """``chart[j]``'s ordinate is the correctly rounded ordinate of the chart chord ``a``-``b`` at
    ``chart[j]``'s abscissa (exact rationals)."""
    (xa, ya), (xb, yb), (x, y) = ([F(float(v)) for v in chart[i]] for i in (a, b, j))
    exact = ya + (x - xa) * (yb - ya) / (xb - xa)
    yf = float(chart[j][1])
    below, above = F(math.nextafter(yf, -math.inf)), F(math.nextafter(yf, math.inf))
    return (below + y) / 2 <= exact <= (above + y) / 2


def _on_segment(mesh, j, a, b):
    """Cut point ``j`` lies on the 3D N1 chord ``a``-``b`` to the rounding of ``A + f (B - A)``."""
    return _on_chord(mesh.points[a], mesh.points[b], mesh.points[j])


def _on_chord(pa, pb, pj, allowance=0.0):
    """``pj`` within ``allowance`` plus the rounding of a float ``A + f (B - A)`` (``f`` in [0, 1])
    of the segment ``pa``-``pb``, decided exactly (rationals).  The rounding: the difference, the
    product and the sum each round by half an ulp of their result, at most ``2 eps (|A_i| + |B_i|)``
    per coordinate; 4 eps is used.  (The test was ``1e-12 max |X|``: at 1.25e6 it took a point on
    a chord 8e-9 away for one on this chord, Codex N3 r1 H1.)"""
    eps = F(2) ** -52
    A, B, P = ([F(float(x)) for x in p] for p in (pa, pb, pj))
    d = [y - x for x, y in zip(A, B)]
    w = [y - x for x, y in zip(A, P)]
    dd = sum(x * x for x in d)
    f = min(max(sum(x * y for x, y in zip(w, d)) / dd, F(0)), F(1)) if dd else F(0)
    distance2 = sum((x - f * y) ** 2 for x, y in zip(w, d))
    rounding2 = sum((4 * eps * (abs(x) + abs(y))) ** 2 for x, y in zip(A, B))
    limit = F(allowance) + _sqrt_up(rounding2)
    return distance2 <= limit * limit


def _sqrt_up(x):
    """A rational upper bound of the square root of a non-negative rational."""
    if x == 0:
        return F(0)
    scale = 2 ** 80
    n = x.numerator * scale * scale // x.denominator + 1
    return F(math.isqrt(n) + 1, scale)


def _naive_fraction_lift(result):
    """Each cut point rebuilt at ``cut_fraction`` on its segment (exact, then rounded): the lift N2
    does NOT certify."""
    uv = [_fp(p) for p in result.points]
    n = len(result.points) - len(result.cut_ends)
    lifted = uv[:n]
    for (a, b), f in zip(result.cut_ends.tolist(), result.cut_fraction.tolist()):
        lifted.append(tuple(F(float(x + F(f) * (y - x))) for x, y in zip(uv[a], uv[b])))
    return uv, lifted


# ----------------------------------------------------------------------------- T1 (Q1)
def test_t1_q1_cut_points_on_the_chord_cannot_invert_an_accepted_triangle():
    """r3: rebuilding cut points from cut_fraction inverts [7, 4, 10]; N3 places cut points on the
    3D N1 chord and certifies every lifted triangle (or refuses TRIM_INVALID)."""
    outer = [(-1., -1.), (1., -1.), (1., 3.), (-1., 3.)]
    hole = [(-.21, 1.), (.75, .75), (.75, 1.25)]
    extra = np.array([math.nextafter(-.21, -math.inf)])
    result = slab.triangulate_rings([np.array(outer), np.array(hole)], extra)
    uv, lifted = _naive_fraction_lift(result)
    inverted = [t for t in result.triangles.tolist() if _orient(*(lifted[i] for i in t)) < 0]
    assert any(t[k:] + t[:k] == [7, 4, 10] for t in inverted for k in range(3)), inverted
    g, face = _planar([outer, hole])
    tol = dt.DisplayTolerance(.5)
    try:
        mesh = _mesh(g, face, tol, _extra_x=extra)
    except dt.DisplayTrimRefused as error:
        assert error.code == "TRIM_INVALID", str(error)
        return
    _check_certified(g, face, tol, mesh)
    assert mesh.chart.tobytes() == result.points.tobytes()           # the r3 mesh itself was lifted


# ----------------------------------------------------------------------------- T2 (Q2)
def test_t2_q2_coincident_chart_vertices_are_refused():
    """r4: an outer cut coincides with a hole tip; the fraction lift separates them and opens a gap
    while every triangle stays positive.  N3 refuses (index -> chart map not injective)."""
    yb = .1 * 3
    tip = float(F(yb) / 3)
    outer = [(0., 0.), (3., yb), (3., 4.), (0., 4.)]
    hole = [(.5, .5), (1., tip), (2., 1.), (1., 2.)]
    m = slab.triangulate_rings([np.array(outer), np.array(hole)])
    q, r = _naive_fraction_lift(m)
    assert q[5] == q[9] and r[5] != r[9]
    assert all(_orient(*(r[i] for i in t)) > 0 for t in m.triangles.tolist())

    def covered(p, pts):
        return any(min(_orient(pts[a], pts[b], p), _orient(pts[b], pts[c], p), _orient(pts[c], pts[a], p)) >= 0
                   for a, b, c in m.triangles.tolist())
    p = F(3, 2), (r[5][1] + r[9][1] + 2) / 4
    assert covered(p, q) and not covered(p, r)                      # the naive lift has a gap
    g, face = _planar([outer, hole])
    text = _refused("TRIM_INVALID", lambda: _mesh(g, face, dt.DisplayTolerance(.5)))
    assert "coincide in the chart" in text


# ----------------------------------------------------------------------------- T3 (Q5)
def test_t3_q5_vertex_offsets_count_toward_the_face_tolerance():
    """r7: a planar triangle 4e-9 off its plane; N1 has no sag allowance on any edge.  T_face is
    surface_residual (no tol.sag): accepted with deviation_bound >= 4e-9; with sag=1e-9 the
    boundary chords cannot meet T_face: EDGE_UNSUPPORTED."""
    uv = [(0., 0.), (1., 0.), (0., 1.)]
    g = GeometryModel()
    vertices = [g.add_point(u, v, 4e-9) for u, v in uv]
    edges = [g.add_line(a, b) for a, b in zip(vertices, vertices[1:] + vertices[:1])]
    face = g.add_face(edges, surface=XY)
    tol = dt.DisplayTolerance(.1)
    assert all(dt.display_rounding_allowance(g, e, tol).sag is None for e in edges)
    xyz = np.array([dt.edge_display_samples(g, e, tol).points[0] for e in edges])
    error = max(np.linalg.norm(x - XY.evaluate(*XY.local_uv(x))) for x in xyz)
    assert g.tolerance.length < error <= g.tolerance.surface_residual   # naive: no face rule decides
    mesh = _check_certified(g, face, tol, _mesh(g, face, tol))
    assert mesh.face_tolerance == g.tolerance.surface_residual
    assert 4e-9 <= mesh.deviation_bound <= g.tolerance.surface_residual
    text = _refused("EDGE_UNSUPPORTED", lambda: _mesh(g, face, dt.DisplayTolerance(.1, sag=1e-9)))
    assert "face tolerance" in text


# ----------------------------------------------------------------------------- T4 (Q6)
def _r8_arc_face():
    g = GeometryModel()
    plane = Plane((0, 0, 0), (1, 0, 0), (-1, 1, 0))
    a, b, c, d = [g.add_point(*xy) for xy in [(1., 0.), (2 ** -.5, 2 ** -.5), (0., 1.), (.5, .5)]]
    arc = g.add_arc(a, b, c)
    edges = [arc, g.add_line(c, d), g.add_line(d, a)]
    return g, g.add_face(edges, surface=plane), arc, edges


def test_t4_q6_the_analytic_trim_distance_is_certified_and_reported():
    """r8 M1: the chart mesh keeps every N1 chord, its parametric error is tiny, yet a point of the
    analytic arc is 0.07612 from the mesh.  N3 reports trim_sag_bound >= that gap (not enforced
    without tol.sag); with tol.sag N1 enforces it and N3's bound is <= sag + N1's allowance."""
    g, face, arc, edges = _r8_arc_face()
    plane = g.faces[face].surface
    tol = dt.DisplayTolerance(.9)
    samples = [dt.edge_display_samples(g, e, tol) for e in edges]
    xyz = np.vstack([s.points[:-1] for s in samples])
    uv = np.column_stack((xyz[:, 0] + xyz[:, 1], xyz[:, 1]))
    np.testing.assert_array_equal(slab.triangulate_rings([uv]).points, uv)
    naive_error = max(np.linalg.norm(x - plane.evaluate(*p)) for x, p in zip(xyz, uv))
    qpt = g.evaluate_edge_many(arc, [np.mean(samples[0].parameters[:2])])[0]

    def distance(a, b):
        d = b - a
        return np.linalg.norm(qpt - a - np.clip((qpt - a) @ d / (d @ d), 0, 1) * d)
    gap = min(distance(a, b) for a, b in zip(xyz, np.roll(xyz, -1, axis=0)))
    assert naive_error <= g.tolerance.surface_residual < gap         # the naive certificate misses it
    assert gap >= .07612
    mesh = _check_certified(g, face, tol, _mesh(g, face, tol))
    assert mesh.trim_sag_bound >= gap and mesh.deviation_bound < 1e-12
    for sag in (.05, .01, 1e-4):
        tol = dt.DisplayTolerance(.9, sag=sag)
        mesh = _check_certified(g, face, tol, _mesh(g, face, tol))
        allowance = max(dt.display_rounding_allowance(g, e, tol).sag for e in edges)
        assert 0.0 < mesh.trim_sag_bound <= sag + allowance


def test_t4b_a_circular_hole_on_a_plane_reports_its_chord_sag():
    """Q6 owner decision: without tol.sag a planar face with a circular hole is accepted, its
    trim_sag_bound about r (1 - cos(angle / 2))."""
    g, face = _planar([[(-3., -3.), (3., -3.), (3., 3.), (-3., 3.)]])
    c, s = math.cos(math.pi / 3), math.sin(math.pi / 3)
    p = [g.add_point(*xy) for xy in ((1., 0.), (c, s), (-.5, s), (-1., 0.), (-.5, -s), (c, -s))]
    hole = [g.add_arc(p[0], p[1], p[2]), g.add_arc(p[2], p[3], p[4]), g.add_arc(p[4], p[5], p[0])]
    corpus._set_holes(g, face, [[OrientedEdge(e, True) for e in hole]])
    for angle in (.1, .5, 1.2):
        tol = dt.DisplayTolerance(angle)
        mesh = _check_certified(g, face, tol, _mesh(g, face, tol))
        expected = 1.0 - math.cos(angle / 2)
        assert .5 * expected <= mesh.trim_sag_bound <= expected * (1 + 1e-9), (angle, mesh.trim_sag_bound)


# ----------------------------------------------------------------------------- T5 (Q8)
def test_t5_q8_a_t_vertex_lies_on_the_shared_n1_chord():
    """r8 note: the triangle (0,0),(2,0),(1,1) gets a boundary cut at (1,0) absent from the N1
    samples of its edge.  N3 keeps it ON the 3D N1 chord (T-vertex, owner decision), the
    neighbour's samples are unchanged, and ``source`` names the chord."""
    g = GeometryModel()
    a, b, c, d = [g.add_point(*xy) for xy in [(0., 0.), (2., 0.), (1., 1.), (1., -1.)]]
    shared = g.add_line(a, b)
    face = g.add_face([shared, g.add_line(b, c), g.add_line(c, a)], surface=XY)
    below = g.add_face([shared, g.add_line(b, d), g.add_line(d, a)], surface=XY)
    tol = dt.DisplayTolerance(.3)
    before = dt.edge_display_samples(g, shared, tol).points.copy()
    rings = _n1_rings(g, face, tol)
    m = slab.triangulate_rings([r[:, :2].copy() for r in rings])
    assert len(m.points) > len(rings[0])
    assert not any(np.array_equal(np.append(p, 0.), s) for p in m.points[3:] for s in before)  # a T-vertex
    mesh = _check_certified(g, face, tol, _mesh(g, face, tol))
    k = mesh.ring_vertex_count
    cut = [j for j in range(k, len(mesh.points)) if np.array_equal(mesh.chart[j], (1., 0.))]
    assert len(cut) == 1
    j = cut[0]
    np.testing.assert_array_equal(mesh.points[j], (1., 0., 0.))       # on the chord, exactly here
    assert tuple(mesh.source[j]) == (shared, 0)
    np.testing.assert_array_equal(dt.edge_display_samples(g, shared, tol).points, before)
    _check_certified(g, below, tol, _mesh(g, below, tol))            # the neighbour lifts too


# ----------------------------------------------------------------------------- T6 (Q9)
def test_t6_q9_a_collapsed_trim_interval_is_refused():
    """r5: a hole edge and an outer edge round onto one segment: the chart mesh covers nothing of
    it and its cut points equal the hole's vertices.  N3 refuses TRIM_INVALID."""
    outer = [(0., np.nextafter(1., 0.)), (4., 1.), (4., 3.), (0., 3.)]
    hole = [(2.5, 1.), (3.5, 1.), (3.5, 2.), (2.5, 2.)]
    m = slab.triangulate_rings([np.array(outer), np.array(hole)])
    assert np.array_equal(m.points[8:10], np.array(hole[:2]))          # bitwise-equal distinct indices
    q = [_fp(p) for p in m.points]
    pt = (F(3), F(1))
    assert not any(min(_orient(q[a], q[b], pt), _orient(q[b], q[c], pt), _orient(q[c], q[a], pt)) >= 0
                   for a, b, c in m.triangles.tolist())                 # the interval is not covered
    g, face = _planar([outer, hole])
    _refused("TRIM_INVALID", lambda: _mesh(g, face, dt.DisplayTolerance(.5)))


# ----------------------------------------------------------------------------- T7
def _corpus_plane_faces():
    found = []
    for name, builder in corpus.ALL.items():
        if name == "generated_cylinder_400":
            continue
        case = builder()
        for label, face in case.faces.items():
            if isinstance(case.model.faces[face].surface, Plane):
                found.append((f"{name}/{label}", case.model, face))
    return found


def test_t7_ring_vertices_are_the_n1_samples_bit_for_bit():
    """Every corpus plane face (and the fixtures here): points[:k] are N1's samples bit for bit.
    Naive: evaluating the support at the vertex's own chart point does not give them back."""
    faces = _corpus_plane_faces()
    assert faces
    g, face, _, _ = _r8_arc_face()
    faces.append(("r8_arc", g, face))
    differs = 0
    accepted = 0
    for label, model, face in faces:
        support = model.faces[face].surface
        for step in corpus.STEPS:
            tol = dt.DisplayTolerance(step)
            rings = np.vstack(_n1_rings(model, face, tol))
            differs += sum(not np.array_equal(support.evaluate(*support.local_uv(p)), p) for p in rings)
            mesh = _mesh(model, face, tol)
            _check_certified(model, face, tol, mesh)
            accepted += 1
    assert differs > 0 and accepted == 2 * len(faces)


# ----------------------------------------------------------------------------- refusals and contract
def test_refusal_a_face_with_an_explicit_parameterization():
    g, face = _planar([[(0., 0.), (1., 0.), (0., 1.)]])
    g.set_face_parameterization(face, XY)
    _refused("SUPPORT_UNSUPPORTED", lambda: _mesh(g, face, dt.DisplayTolerance(.3)))


def test_refusal_an_extruded_support():
    case = corpus.whole_box_extrusion()
    for face in case.faces.values():
        _refused("SUPPORT_UNSUPPORTED", lambda: _mesh(case.model, face, dt.DisplayTolerance(.3)))


def test_refusal_a_spline_edge_via_n1():
    g = GeometryModel()
    ids = [g.add_point(*p) for p in ((0., 0., 0.), (1 / 3, .1, 0.), (2 / 3, 0., 0.), (1., 0., 0.), (.5, 1., 0.))]
    spline = g.add_spline(ids[0], ids[1:3], ids[3])
    face = g.add_face([spline, g.add_line(ids[3], ids[4]), g.add_line(ids[4], ids[0])], surface=XY)
    _refused("EDGE_UNSUPPORTED", lambda: _mesh(g, face, dt.DisplayTolerance(.3)))


def test_refusal_a_seam_edge_used_twice():
    g, face = _planar([[(0., 0.), (2., 0.), (2., 2.), (0., 2.)]])
    loop = g.faces[face].loop
    m = g.add_point(1., 1.)
    start = g.edges[loop[0].edge].start
    slit = g.add_line(start, m)
    _force(g, face, loop=(*loop, OrientedEdge(slit, True), OrientedEdge(slit, False)))
    _refused("TRIM_WINDS_SUPPORT", lambda: _mesh(g, face, dt.DisplayTolerance(.3)))


def test_refusal_coedges_that_do_not_meet():
    g, face = _planar([[(0., 0.), (2., 0.), (2., 2.), (0., 2.)]])
    loop = g.faces[face].loop
    _force(g, face, loop=(loop[0], loop[2], loop[1], loop[3]))
    text = _refused("TRIM_INVALID", lambda: _mesh(g, face, dt.DisplayTolerance(.3)))
    assert "do not meet" in text


def test_input_contract_and_public_surface():
    g, face = _planar([[(0., 0.), (1., 0.), (0., 1.)]])
    tol = dt.DisplayTolerance(.3)
    for bad in (True, 1.0, "1", None, 2 ** 63):
        with pytest.raises(GeometryError) as info:
            dt.face_display_mesh(g, bad, tol)
        assert not isinstance(info.value, dt.DisplayTrimRefused)
    with pytest.raises(GeometryError):
        dt.face_display_mesh(g, face + 1000, tol)
    with pytest.raises(GeometryError):
        dt.face_display_mesh(object(), face, tol)
    with pytest.raises(GeometryError):
        dt.face_display_mesh(g, face, .3)
    assert type(_mesh(g, np.int64(face), tol)) is dt.FaceDisplayMesh
    assert dt.REFUSAL_CODES == {"SUPPORT_UNSUPPORTED", "TRIM_INVALID", "TRIM_WINDS_SUPPORT", "EDGE_UNSUPPORTED",
                                "OUT_OF_DOMAIN"}
    assert "face_display_mesh" in dt.__all__ and "FaceDisplayMesh" in dt.__all__
    import anygeometry
    assert "face_display_mesh" not in getattr(anygeometry, "__all__", ())


# ----------------------------------------------------------------------------- findings and fuzz
def test_two_cuts_rounding_onto_one_point_beside_a_tip_are_merged():
    """Finding (N3a bench, aligned holes): a slab line an ulp beside a hole's left tip cuts both
    tip edges; both cut ordinates round to the tip's, so two distinct cut points share one chart
    point.  Owner decision D1 (SPEC-N3b): bitwise-equal CUT points are merged into one mesh vertex
    before the certificate, which then holds on the merged mesh (oracle-checked)."""
    outer = [(-3., -3.), (3., -3.), (3., 7.), (-3., 7.)]
    tip = [(0., 4.5), (1., 3.5), (1., 5.5)]
    near = [(1e-17, 0.), (.5, -.5), (.5, .5)]
    m = slab.triangulate_rings([np.array(outer), np.array(tip), np.array(near)])
    q = [_fp(p) for p in m.points]
    assert len(set(q)) == len(q) - 1                                # N2 is valid, yet not injective
    g, face = _planar([outer, tip, near])
    tol = dt.DisplayTolerance(.5)
    mesh = _check_certified(g, face, tol, _mesh(g, face, tol))
    assert len(mesh.points) == len(m.points) - 1
    first = sorted({x: i for i, x in reversed(list(enumerate(q)))}.values())    # each chart point's first index
    assert mesh.chart.tobytes() == m.points[first].tobytes()


def test_a_cut_on_a_ring_vertex_stays_refused():
    """D1 merges cut points only: a cut bitwise equal to a ring vertex (an N1 sample) stays
    TRIM_INVALID (the T2 fixture, and a hole vertex on another hole's chord)."""
    yb = .1 * 3
    outer = [(0., 0.), (3., yb), (3., 4.), (0., 4.)]
    hole = [(.5, .5), (1., float(F(yb) / 3)), (2., 1.), (1., 2.)]
    g, face = _planar([outer, hole])
    assert "coincide in the chart" in _refused("TRIM_INVALID", lambda: _mesh(g, face, dt.DisplayTolerance(.5)))


def big_polygon(aligned=False):
    """400-vertex outer polygon, 20 holes (circles and hexagons) on a 5 x 4 grid (the N3a bench
    face); unless ``aligned``, each hole centre is jittered (fixed seed) so no two holes share a
    vertex x."""
    jitter = np.random.default_rng(7).uniform(-.3, .3, (20, 2))
    g = GeometryModel()
    n = 400
    t = 2 * math.pi * np.arange(n) / n
    r = 10 + .5 * np.sin(7 * t)
    ids = [g.add_point(x, y) for x, y in zip(r * np.cos(t), r * np.sin(t))]
    outer = [g.add_line(a, b) for a, b in zip(ids, ids[1:] + ids[:1])]
    face = g.add_face(outer, surface=XY)
    holes = []
    for h in range(20):
        cx, cy = -6 + 3 * (h % 5), -4.5 + 3 * (h // 5)
        if not aligned:
            cx, cy = cx + jitter[h, 0], cy + jitter[h, 1]
        p = [g.add_point(cx + .8 * math.cos(a), cy + .8 * math.sin(a)) for a in np.arange(6) * math.pi / 3]
        if h % 2:                                    # circle hole, three arcs
            edges = [g.add_arc(p[0], p[1], p[2]), g.add_arc(p[2], p[3], p[4]), g.add_arc(p[4], p[5], p[0])]
        else:                                        # hexagon hole
            edges = [g.add_line(a, b) for a, b in zip(p, p[1:] + p[:1])]
        holes.append(tuple(OrientedEdge(e, True) for e in edges))
    _force(g, face, holes=tuple(holes))
    return g, face


def test_aligned_400_vertex_polygon_with_20_holes_is_accepted():
    """D1: the grid-aligned 400-vertex / 20-hole polygon (N3a bench) was refused for merged cut
    points; it is accepted and certified."""
    g, face = big_polygon(aligned=True)
    tol = dt.DisplayTolerance(corpus.INTERACTION_STEP)
    _check_certified(g, face, tol, _mesh(g, face, tol), grid_triangles=8)


def test_small_fuzz_planar_faces_with_round_holes():
    """40 random planar faces (skew planes, star outer, circle / ellipse / polygon holes): every
    accepted mesh passes the oracles; refusals are typed.  The 1200-face run is evidence
    n3a-fuzz.txt (n3a_fuzz.py)."""
    from anygeometry.exact_curves import EllipticArc
    rng = np.random.default_rng(26)
    accepted = 0
    for case in range(40):
        g = GeometryModel()
        o = rng.uniform(-50, 50, 3)
        u = rng.normal(size=3)
        v = np.cross(u, rng.normal(size=3)) + .4 * u
        plane = Plane(tuple(o), tuple(u), tuple(v))

        def at(x, y):
            return o + x * u + y * v
        m = int(rng.integers(5, 10))
        angles = 2 * math.pi * (np.arange(m) + rng.uniform(-.3, .3, m)) / m
        ids = [g.add_point(*at(r * math.cos(a), r * math.sin(a))) for a, r in zip(angles, rng.uniform(2, 4, m))]
        face = g.add_face([g.add_line(a, b) for a, b in zip(ids, ids[1:] + ids[:1])], surface=plane)
        holes = []
        for h in range(int(rng.integers(1, 3))):
            cx, cy = .55 * math.cos(2 * math.pi * h / 3), .55 * math.sin(2 * math.pi * h / 3)
            c3 = at(cx, cy)
            U, V = .2 * u, .1 * (v - (v @ u) / (u @ u) * u)
            s0 = float(rng.uniform(-3, 3))
            ends = [c3 + math.cos(s) * U + math.sin(s) * V for s in (s0, s0 + math.pi)]
            p0, p1 = g.add_point(*ends[0]), g.add_point(*ends[1])
            holes.append((OrientedEdge(g.add_curve(p0, p1, EllipticArc(tuple(c3), tuple(U), tuple(V), s0, math.pi)), True),
                          OrientedEdge(g.add_curve(p1, p0, EllipticArc(tuple(c3), tuple(U), tuple(V), s0 + math.pi,
                                                                       math.pi)), True)))
        _force(g, face, holes=tuple(holes))
        tol = dt.DisplayTolerance(float(rng.uniform(.1, 1.2)))
        try:
            mesh = _mesh(g, face, tol)
        except dt.DisplayTrimRefused:
            continue
        _check_certified(g, face, tol, mesh)
        accepted += 1
    assert accepted >= 30
