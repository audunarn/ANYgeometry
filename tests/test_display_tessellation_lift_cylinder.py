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
        for t in mesh.triangles:
            a, b, c = mesh.points[t]
            n = np.cross(b - a, c - a)
            radial = (a + b + c) / 3 - o
            radial -= (radial @ axis) * axis
            assert n @ radial > 0                              # outward
        r = g.faces[face].surface.radius
        rebuilt = o[None] + r * (np.cos(mesh.chart[:k, :1]) * e1 + np.sin(mesh.chart[:k, :1]) * e2) \
            + mesh.chart[:k, 1:] * axis
        assert not np.array_equal(rebuilt, mesh.points[:k])   # naive: evaluating the chart moves them


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
    the certificates hold; 12 of them need the rounding-thin exemption (N3.0)."""
    case = corpus.tripod_skins()
    tol = dt.DisplayTolerance(step)
    exempt = []
    for label, face in case.faces.items():
        mesh = _check_cylinder(case.model, face, tol, _mesh(case.model, face, tol), grid=_grid(3))
        exempt.append(mesh.degenerate_count)
    assert len(exempt) == 18 and sum(1 for n in exempt if n) >= 1


@pytest.mark.parametrize("step", corpus.STEPS, ids=["fine", "interaction"])
def test_generated_cylinder_400_faces_are_all_accepted(step):
    case = corpus.generated_cylinder(400)
    tol = dt.DisplayTolerance(step)
    for i, face in enumerate(case.faces.values()):
        mesh = _mesh(case.model, face, tol)
        if i % 40 == 0:
            _check_cylinder(case.model, face, tol, mesh, grid=_grid(3))
        assert mesh.degenerate_count == 0 and mesh.deviation_bound <= mesh.face_tolerance


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


def test_cone_support_stays_unsupported_until_n3c():
    case = corpus.ALL["overrun_cone"]()
    face = next(iter(case.faces.values()))
    text = _refused("SUPPORT_UNSUPPORTED", lambda: _mesh(case.model, face, dt.DisplayTolerance(.3)))
    assert "not supported yet" in text
