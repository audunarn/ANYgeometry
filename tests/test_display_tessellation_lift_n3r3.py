"""N3 Codex round 3 (codex-n3r3.md; SPEC-N3-R3-FIXES.md, owner decisions C1-C3): every reproduction
as a regression test.

H1 / C1  identifications are exact: vertices are identified only when their lifted 3D points are
         bitwise equal, boundary edges cancel only when both lifted endpoints are bitwise equal
         (L = 0); no allowance (the r3 ill-conditioned plane had A_face 15646 and a 0.125 gap).
M1 / C3  no orientation margin: a triangle is kept only if its orientation is certifiably positive
         over its footprint (an enclosure), a zero exact normal is DROPPED and counted, anything
         else is a fold.
M3 / C2  translation invariance is not a requirement: an oblique cylinder strip may be refused
         after an axial translation, but an accepted mesh is certified.
L1       the curved orientation oracles implement exactly the production rule (no 64 eps, no
         width limit): the M1 counterexample fails the oracle.
L2       the ring cap is checked before the ring list is copied (peak memory O(cap), not O(n)).
"""
from __future__ import annotations

import math
import tracemalloc
from dataclasses import replace
from fractions import Fraction as F
from types import SimpleNamespace

import mpmath as mp
import numpy as np
import pytest

import test_display_tessellation_lift as T
import test_display_tessellation_lift_cylinder as C
import test_display_tessellation_lift_n3r1 as R1
from anygeometry import _display_lift as lift
from anygeometry import display_tessellation as dt
from anygeometry.entities import OrientedEdge
from anygeometry.model import GeometryModel
from anygeometry.surfaces import Cylinder, Plane

_mesh, _refused, _status = T._mesh, T._refused, R1._status


# ----------------------------------------------------------------------------- H1 / C1
def _codex_h1_r3():
    """Codex N3 r3 H1: the r1 shared-trim fixture on an ill-conditioned plane (sine 1e-12) with
    the y coordinates scaled by 2**20, plus a neighbour face across the shared bottom edge."""
    u, scale = math.ulp(1.), 2 ** 20
    outer = [(0, 1), (8, 1 + 4 * u), (8, 4), (3.25, 4.1), (3, 4.2), (0, 4)]
    hole = [(1, 1 + u), (5, 1 + 3 * u), (5, 2), (1, 2)]
    s = Plane((0, 0, 0), (1, 0, 0), (1, 1.001e-12, 0))
    g, f = T._planar([[(x, y * scale) for x, y in r] for r in (outer, hole)], s)
    shared = g.faces[f].loop[0].edge
    e, d = g.edges[shared], g.add_point(4, 0)
    neighbour = g.add_face_from_loop([OrientedEdge(shared, False), OrientedEdge(g.add_line(e.start, d), True),
                                      OrientedEdge(g.add_line(d, e.end), True)], surface=s)
    assert not g.validate_topology()
    return g, f, neighbour, np.array([3.125, (1 + 3.125 * u / 2) * scale, 0.])


def _gap(mesh, p):
    edges = {(a, b) for tri in mesh.triangles.tolist() for a, b in zip(tri, tri[1:] + tri[:1])}

    def distance(a, b):
        d = b - a
        return np.linalg.norm(p - a - np.clip((p - a) @ d / (d @ d), 0, 1) * d)
    return min(distance(mesh.points[a], mesh.points[b]) for a, b in edges if (b, a) not in edges)


def test_h1_a_shared_trim_gap_on_an_ill_conditioned_plane_is_refused():
    """Codex N3 r3 H1: the d + L gate used A_face, inflated to 15646 by the plane's basis; the
    face was accepted with its boundary 0.125 from the shared interval's midpoint (the neighbour
    within 1.1e-10).  C1: the merged cut points are not bitwise equal in 3D -> TRIM_INVALID."""
    g, f, neighbour, p = _codex_h1_r3()
    tol = dt.DisplayTolerance(.1, sag=1e-6)
    text = _refused("TRIM_INVALID", lambda: _mesh(g, f, tol))
    assert "bitwise" in text and "merged cut points" in text
    code, other = _status(lambda: _mesh(g, neighbour, tol))
    if other is not None:
        assert _gap(other, p) <= other.trim_sag_bound + 1e-9


@pytest.mark.parametrize("sag", [1e-6, .5001])
def test_h1_the_round_1_and_2_fixtures_stay_refused_with_the_exact_rule(sag):
    """The r1 / r2 H1 fixture (8e-9 between two chords ulps apart in the chart): refused, now by
    the exact identification rule."""
    g, f, z, u = R1._codex_h1_face()
    text = _refused("TRIM_INVALID", lambda: _mesh(g, f, dt.DisplayTolerance(.1, sag=sag)))
    assert "bitwise" in text


def test_c1_the_trim_bound_carries_no_identification_term():
    """C1: trim_sag_bound no longer carries d + L; with no cut points it is the N1 chords' bound
    alone, with cut points that plus their rounding (no identification can move the boundary)."""
    g, f = T._planar([[(0, 0), (3, 0), (3, 3)]])
    tol = dt.DisplayTolerance(.5)
    plain = _mesh(g, f, tol)
    assert plain.trim_sag_bound == max(
        lift._trim_sag(f, g, s, tol, c) for r in [lift._ring(g, f, g.faces[f].loop, tol)] for s, c in r.edges)
    assert not hasattr(lift, "_identification") and not hasattr(lift, "_check_identification")


# ----------------------------------------------------------------------------- M1 / C3
def _codex_m1_r3():
    c = lift._CylinderChart(Cylinder((0, 0, 0), (1, 1, 0), (0, 0, 1), 1., 1.))
    x = 1 - 2. ** -11
    y, w = x + 2. ** -46, 2. ** -60
    p = np.array([(0., 0., 0.), (0., 0., w), (y, -x, 0.)])
    return c, p, x, y, w


def test_m1_a_fold_just_beyond_the_old_margin_is_folded():
    """Codex N3 r3 M1: exact footprint minimum -32.0156 eps |n|, returned exempt (the rounded dot
    compared with the 32 eps threshold).  C3: no margin; the sign is decided by an enclosure: the
    triangle is not certifiably positive -> folded, not zero."""
    c, p, x, y, w = _codex_m1_r3()
    assert (F(y) - F(x)) ** 2 > 2 * (32 * F(2) ** -52) ** 2 * (F(x) ** 2 + F(y) ** 2)
    zero, folded = lift._thin_curved(p, np.array([[0, 1, 2]]), [0.], [1e-16], c.e1, c.e2, c.a, 0., 1)
    assert folded[0] and not zero[0]


def _cos_sin_check(theta):
    cl, ch, sl, sh = lift._cos_sin(theta)
    with mp.workdps(80):
        t = mp.mpf(F(theta).numerator) / mp.mpf(F(theta).denominator)
        assert mp.mpf(cl.numerator) / cl.denominator <= mp.cos(t) <= mp.mpf(ch.numerator) / ch.denominator
        assert mp.mpf(sl.numerator) / sl.denominator <= mp.sin(t) <= mp.mpf(sh.numerator) / sh.denominator
    assert ch - cl < F(1, 2 ** 100) and sh - sl < F(1, 2 ** 100)


@pytest.mark.parametrize("theta", [0., 1e-300, -2.5e-17, .5, -1., math.pi, -3 * math.pi, 9.75, -12.25])
def test_m1_the_cosine_and_sine_enclosures_are_rigorous(theta):
    """The exact fallback encloses cos and sin of the footprint ends rigorously (fixed-point Taylor
    series in integers with a counted truncation error; no libm accuracy assumption)."""
    _cos_sin_check(theta)


def test_m1_positive_zero_and_folded_decisions():
    """The three outcomes of the exact fallback: certifiably positive (kept), an exactly zero normal
    (dropped), a definite fold; on a cylinder (k = 0) and a cone (k = .5)."""
    e1, e2, a = np.eye(3)
    good = np.array([(1., 0., 0.), (1., 0., 1.), (1., 1., 0.)])          # n = (-1, 0, 0) . sense -1
    zero = np.array([(1., 0., 0.), (2., 0., 0.), (3., 0., 0.)])
    for k in (0., .5):
        z, f = lift._thin_curved(good, np.array([[0, 1, 2]]), [0.], [.1], e1, e2, a, k, -1)
        assert not z[0] and not f[0]
        z, f = lift._thin_curved(good, np.array([[0, 1, 2]]), [0.], [.1], e1, e2, a, k, 1)
        assert not z[0] and f[0]
        z, f = lift._thin_curved(zero, np.array([[0, 1, 2]]), [0.], [.1], e1, e2, a, k, 1)
        assert z[0] and not f[0]


def _zero_area_face():
    """A planar triangle with two forced slab lines an ulp apart (the fuzz "extra" mode): two cut
    points on one chord are distinct in the chart but bitwise equal in 3D, so two lifted
    triangles have an exactly zero normal."""
    g, f = T._planar([[(0, 0), (3, 0), (3, 3)]])
    x0 = .42500000000000004
    return g, f, np.array([x0, math.nextafter(x0, 9.)])


def _zero_normal(points, t):
    a, b, c = (T._fp(points[i]) for i in t)
    return T._cross([y - x for x, y in zip(a, b)], [y - x for x, y in zip(a, c)]) == (0, 0, 0)


def test_c3_zero_area_triangles_are_dropped_and_counted():
    """C3: a lifted triangle whose exact normal is zero is dropped from the published mesh and
    counted in degenerate_count; every published triangle has a certifiably positive orientation
    (the oracle restores the dropped ones to check the chart covering)."""
    g, f, extra = _zero_area_face()
    tol = dt.DisplayTolerance(.5)
    mesh = T._check_certified(g, f, tol, _mesh(g, f, tol, _extra_x=extra))
    assert mesh.degenerate_count == 2
    assert not any(_zero_normal(mesh.points, t) for t in mesh.triangles.tolist())


def test_c3_a_zero_area_triangle_not_covered_by_its_neighbours_is_refused():
    """The coverage condition of a drop: a zero-area triangle's point set (its longest edge) must
    lie in kept triangles (two of its edges, or a longest one, shared with kept triangles); else
    TRIM_INVALID (fail closed)."""
    points = np.array([(0., 0., 0.), (1., 0., 0.), (2., 0., 0.), (1., 1., 0.)])
    triangles = np.array([[0, 1, 2], [2, 1, 3]])                 # [0, 1, 2] zero, one edge shared
    with pytest.raises(dt.DisplayTrimRefused, match="zero-area"):
        lift._drop_zero(7, points, triangles, np.array([0]))
    triangles = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3]])      # two edges shared: dropped
    kept = lift._drop_zero(7, points, triangles, np.array([0]))
    assert kept.tolist() == [[0, 1, 3], [1, 2, 3]]
    points = np.array([(0., 0., 0.), (2., 0., 0.), (1., 0., 0.), (1., 1., 0.)])
    triangles = np.array([[0, 1, 2], [1, 0, 3]])                 # one shared edge, the longest
    assert lift._drop_zero(7, points, triangles, np.array([0])).tolist() == [[1, 0, 3]]
    # a strip of two zero-area triangles along one chord (an ulp slab, fuzz finding): the second is
    # covered through the first (fixpoint), whose longest edge is shared with a kept triangle
    p, q = (0., 0., 0.), (1., 0., 0.)
    points = np.array([p, p, q, q, (0., 1., 0.)])
    triangles = np.array([[0, 1, 3], [0, 3, 2], [3, 1, 4]])
    assert lift._drop_zero(7, points, triangles, np.array([0, 1])).tolist() == [[3, 1, 4]]
    triangles = np.array([[0, 1, 3], [0, 3, 2], [4, 1, 3]])     # nothing kept beside them: refused
    with pytest.raises(dt.DisplayTrimRefused, match="zero-area"):
        lift._drop_zero(7, points, triangles, np.array([0, 1]))


# ----------------------------------------------------------------------------- M3 / C2
def _oblique_strip(z):
    g, o = GeometryModel(), (z, z, 0.)
    s = Cylinder(o, (1, 1, 0), (0, 0, 1), math.sqrt(2), 1.)
    v = [g.add_point(z + h + 1, z + h - 1, y) for y, h in [(0., 0.), (1e-8, 0.), (1e-8, .5), (0., .5)]]
    edges = [g.add_line(a, b) for a, b in zip(v, v[1:] + v[:1])]
    f = g.add_face(edges, surface=s)
    assert not g.validate_topology()
    return g, f


@pytest.mark.parametrize("z", [0., 2e6])
def test_m3_an_oblique_strip_is_refused_or_certified_at_any_placement(z):
    """Codex N3 r3 M3: accepted at the origin, refused (E2 collapses two ring points) after an
    axial translation along (1, 1, 0).  C2: a placement-dependent refusal is allowed (the
    consumer's old route); an accepted mesh must be certified."""
    g, f = _oblique_strip(z)
    tol = dt.DisplayTolerance(.1)
    code, mesh = _status(lambda: _mesh(g, f, tol))
    if mesh is None:
        assert code == "TRIM_INVALID"
    else:
        C._check_cylinder(g, f, tol, mesh)


# ----------------------------------------------------------------------------- L1
def test_l1_the_curved_oracle_rejects_the_m1_counterexample():
    """Codex N3 r3 L1: the shared oracle helper allowed 64 eps and a deviation_bound width limit,
    so the M1 counterexample passed it.  The oracle follows the production rule exactly: a kept
    triangle with a non-positive footprint minimum fails."""
    c, p, x, y, w = _codex_m1_r3()
    with mp.workdps(50):
        n = (mp.mpf(w) * x, mp.mpf(w) * y, mp.mpf(0))
        low = (n[0] - n[1]) / mp.sqrt(2)
        corners = [[mp.mpf(float(v)) for v in row] for row in p]
        with pytest.raises(AssertionError, match="exemption rule"):
            C._check_exempt(n, low, 0, corners, SimpleNamespace(deviation_bound=1e-15))
        with pytest.raises(AssertionError, match="exemption rule"):
            C._check_exempt((mp.mpf(0),) * 3, mp.mpf(0), 0, corners, SimpleNamespace(deviation_bound=1e-15))


# ----------------------------------------------------------------------------- L2
def test_l2_the_ring_cap_is_checked_before_the_ring_list_is_copied():
    """Codex N3 r3 L2 (r2 L3): 100000 holes were expanded into a tuple (peak > 8 bytes per ring)
    before the ring cap refused.  The cap is checked first: peak memory O(cap), not O(n)."""
    n = 100_000
    g, f = T._planar([[(0, 0), (2, 0), (0, 2)]])
    face = g.faces[f]
    g._put_entity("face", replace(face, holes=(face.loop,) * n))
    tracemalloc.start()
    try:
        with pytest.raises(dt.DisplayTrimRefused) as info:
            dt.face_display_mesh(g, f, dt.DisplayTolerance(.1))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert info.value.code == "OUT_OF_DOMAIN" and "MAX_SLAB_RINGS" in str(info.value)
    assert peak < n, peak
