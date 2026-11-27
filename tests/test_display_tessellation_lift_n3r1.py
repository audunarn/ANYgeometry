"""N3 Codex round 1 (codex-n3r1.md; SPEC-N3-R1-FIXES.md): every reproduction as a regression test.

H1  a D1 merge and the boundary cancellation it allows must hold in 3D (the lifted boundary covers
    every lifted trim interval); the oracle's on-chord test is exact.
M1  the rounding-thin exemption never admits a fold the stored floats decide (translation; since
    owner decision C2, r3, a translated face is refused or certified).
M2  the rounding of the cut points N3 creates is counted in ``trim_sag_bound``.
M3  an exactly planar face is accepted at the origin, refused or certified elsewhere (C2); a real
    offset is still refused.
M4  the N2 ring-point budget is enforced while the rings are sampled.
L1  the curved orientation oracles take the exact minimum over the footprint.
"""
from __future__ import annotations

import json
import math
from fractions import Fraction as F

import mpmath as mp
import numpy as np
import pytest

import display_tessellation_corpus as corpus
import test_display_tessellation_lift as T
import test_display_tessellation_lift_cone as K
import test_display_tessellation_lift_cylinder as C
from anygeometry import _slab_triangulation as slab
from anygeometry import display_tessellation as dt
from anygeometry.entities import OrientedEdge
from anygeometry.exact_curves import EllipticArc
from anygeometry.model import GeometryModel
from anygeometry.surfaces import Cylinder, Plane
from anygeometry.tolerance import TolerancePolicy

_mesh, _refused = T._mesh, T._refused


def _status(call):
    try:
        mesh = call()
    except dt.DisplayTrimRefused as error:
        return error.code, None
    return "accepted", mesh


# ----------------------------------------------------------------------------- H1
def _codex_h1_face():
    z, u = 1.25e6, math.ulp(1.)
    outer = [(0, 1), (8, 1 + 4 * u), (8, 4), (3.25, 4.1), (3, 4.2), (0, 4)]
    hole = [(1, 1 + u), (5, 1 + 3 * u), (5, 2), (1, 2)]
    rings = [[(x, y, z + dz) for x, y in ring] for ring, dz in [(outer, 4e-9), (hole, -4e-9)]]
    g, f = T._planar(rings, Plane((0, 0, z), (1, 0, 0), (0, 1, 0)))
    return g, f, z, u


def _boundary_distance(mesh, q):
    edges = {(a, b) for tri in mesh.triangles.tolist() for a, b in zip(tri, tri[1:] + tri[:1])}
    best = math.inf
    for a, b in edges:
        if (b, a) in edges:
            continue
        pa, pb = mesh.points[a], mesh.points[b]
        d = pb - pa
        best = min(best, float(np.linalg.norm(q - pa - np.clip((q - pa) @ d / (d @ d), 0, 1) * d)))
    return best


def test_h1_a_merge_cannot_erase_a_lifted_trim_interval():
    """Codex H1: the outer and the hole bottom chords are ulps apart in the chart but 8e-9 apart in
    3D; the cuts at x = 3 and 3.25 on both merge (D1) and the chain between them cancels, so the
    returned boundary omitted the interval x = 3..3.25 (0.125 from its midpoint).  Refused, or
    accepted only with a boundary covering it (the tightened oracle checks that)."""
    g, f, z, u = _codex_h1_face()
    tol = dt.DisplayTolerance(.1, sag=1e-6)
    code, mesh = _status(lambda: _mesh(g, f, tol))
    if mesh is None:
        assert code == "TRIM_INVALID"
        return
    T._check_certified(g, f, tol, mesh)
    q = np.array([3.125, 1 + (3.125 / 2) * u, z + 4e-9])
    assert _boundary_distance(mesh, q) <= mesh.trim_sag_bound


def test_h1_the_oracle_on_chord_test_is_exact():
    """The oracle's on-chord test was ``1e-12 max|X|`` (1.25e-6 here, above the 8e-9 between the
    two chords): a point on the other chord now fails it, the chord's own rounded cut passes."""
    pa, pb = np.array([0., 1., 1.25e6 + 4e-9]), np.array([8., 1., 1.25e6 + 4e-9])
    other = np.array([3., 1., 1.25e6 - 4e-9])
    assert not T._on_chord(pa, pb, other)
    f = 3. / 8.
    assert T._on_chord(pa, pb, pa + f * (pb - pa))


# ----------------------------------------------------------------------------- M1
def _codex_m1_face(z0):
    g = GeometryModel()
    s = Cylinder((0, 0, z0), (0, 0, 1), (1, 0, 0), 1., 1., 0., 2e-4)
    p = np.array([s.evaluate(u, v) for u, v in [(0, .5), (1, 0), (0, .5 + 2e-9)]])
    p[[0, 2], 0] += [-4e-9, 4e-9]
    a, b, c = [g.add_point(*x) for x in p]
    f = g.add_face([g.add_line(a, b), g.add_line(b, c), g.add_line(c, a)], surface=s)
    assert not g.validate_topology()
    return g, f


@pytest.mark.parametrize("z0", [0., 1e3, 1e6])
def test_m1_a_translated_thin_fold_is_refused(z0):
    """Codex M1: at 1e6 along the axis the r8 sweep triangle (3D width 8e-9) fell under the
    world-scaled rounding-thin threshold and was exempt although its normal dot changes sign over
    the footprint (+3.96e-13 / -4.04e-13).  Refused at every placement, as at the origin."""
    g, f = _codex_m1_face(z0)
    assert "folds" in _refused("TRIM_INVALID", lambda: _mesh(g, f, dt.DisplayTolerance(.1)))


def _shifted_tripod(shift):
    """The tripod skins with every coordinate moved by ``shift`` (vertices, arc centres, support
    origins)."""
    doc = json.loads((corpus.DATA / "display_tripod_skins.json").read_text())
    s = np.asarray(shift, dtype=float)
    g = GeometryModel()
    vertex = {k: g.add_point(*(np.asarray(v, dtype=float) + s)) for k, v in doc["vertices"].items()}
    edge = {}
    for k, e in doc["edges"].items():
        a, b = vertex[str(e["start"])], vertex[str(e["end"])]
        c = e["curve"]
        if c["kind"] == "line":
            edge[k] = g.add_line(a, b)
        else:
            edge[k] = g.add_curve(a, b, EllipticArc(tuple(np.asarray(c["center"], dtype=float) + s), tuple(c["u"]),
                                                    tuple(c["v"]), c["start"], c["sweep"]))
    faces = []
    for f in doc["faces"][:6]:
        c = f["cylinder"]
        support = Cylinder(tuple(np.asarray(c["origin"], dtype=float) + s), c["axis"], c["radial"], c["radius"],
                           c["height"], c["start"], c["sweep"])
        loop = tuple(OrientedEdge(edge[str(e)], forward) for e, forward in f["loop"])
        faces.append(g.add_face_from_loop(loop, f["corners"], surface=support))
    return g, faces


@pytest.mark.parametrize("shift", [(0., 0., 0.), (1e3, -2e3, 5e2), (1e5, 1e5, -1e5), (1e6, 1e6, -1e6)])
def test_m1_translated_tripod_and_cylinder_faces_are_refused_or_certified(shift):
    """Six tripod skins, a quarter cylinder, a cylinder window face and a cone frustum patch,
    translated.  Owner decision C2 (r3): translation invariance is not a requirement, only
    fail-open matters, so a translated face may be refused (the consumer's old route), and every
    accepted translated mesh passes the oracles; at the origin all are accepted.  (At 1e6 N1 itself
    refuses a tripod edge: the tripod is checked up to 1e5.)"""
    tol = dt.DisplayTolerance(corpus.INTERACTION_STEP)
    origin = not any(shift)

    def check(g, face, tol, oracle):
        code, mesh = _status(lambda: _mesh(g, face, tol))
        assert mesh is not None or not origin, code
        if mesh is not None:
            oracle(g, face, tol, mesh, grid=C._grid(2))
    if max(map(abs, shift)) <= 1e5:
        g, faces = _shifted_tripod(shift)
        for face in faces:
            check(g, face, tol, C._check_cylinder)
    o = np.asarray(shift)
    g, face = C.quarter_cylinder(origin=tuple(o))
    check(g, face, tol, C._check_cylinder)
    g, face = C.window_face(tuple(o + (31., -17., 44.)), (1., 2., 2.), 1.9)
    check(g, face, dt.DisplayTolerance(.51), C._check_cylinder)
    g, face = K.frustum_patch(r0=.5, origin=tuple(o))
    check(g, face, tol, K._check_cone)


# ----------------------------------------------------------------------------- M2
def test_m2_the_cut_rounding_is_counted_in_the_trim_bound():
    """Codex M2: the cut N3 makes by interpolation is 1.8990e-16 from its analytic line (exact
    rationals), above the reported 1.6937e-16 (N1's chord rounding only)."""
    p = [(-.6473297866068923, .7627848533596154), (.4777699609488105, -.6141109675736424),
         (.4121738379585351, .75)]
    g, f = T._planar([p])
    tol = dt.DisplayTolerance(.1)
    m = T._check_certified(g, f, tol, _mesh(g, f, tol))
    assert len(m.points) > m.ring_vertex_count
    a, b, q = [tuple(map(F, x)) for x in [*p[:2], m.points[-1, :2]]]
    dx, dy = b[0] - a[0], b[1] - a[1]
    cross = dx * (q[1] - a[1]) - dy * (q[0] - a[0])
    assert cross ** 2 <= F(m.trim_sag_bound) ** 2 * (dx ** 2 + dy ** 2)


# ----------------------------------------------------------------------------- M3
@pytest.mark.parametrize("z", [0., 2e6, -3e7])
def test_m3_an_exactly_planar_face_is_refused_or_certified_at_any_origin(z):
    """Codex M3: the triangle on z = 2e6 was EDGE_UNSUPPORTED because the world-scaled margin alone
    (1.42e-8) exceeded T_face = 1e-8; r1 made the bound use local magnitudes (kept: measured,
    accepted at every z here).  Owner decision C2 (r3): translation invariance is not a requirement;
    accepted at the origin, and wherever accepted it is certified."""
    g, f = T._planar([[(0, 0, z), (2, 0, z), (1, 1, z)]], Plane((0, 0, z), (1, 0, 0), (0, 1, 0)))
    tol = dt.DisplayTolerance(.1)
    code, mesh = _status(lambda: _mesh(g, f, tol))
    assert mesh is not None or z != 0., code
    if mesh is not None:
        T._check_certified(g, f, tol, mesh)


def test_m3_a_real_offset_at_a_far_origin_is_still_refused():
    """2e-8 off the plane at z = 2e6 is refused at T_face = 1e-8 (the local bound decides; the
    model's own residual is loosened so the face can be built)."""
    z = 2e6
    g, f = T._planar([[(0, 0, z + 2e-8), (2, 0, z), (1, 1, z)]], Plane((0, 0, z), (1, 0, 0), (0, 1, 0)), _LOOSE)
    _refused("EDGE_UNSUPPORTED", lambda: _mesh(g, f, dt.DisplayTolerance(.1, sag=1e-8)))


def test_m3_an_offset_of_exactly_t_face_is_decided_exactly():
    """A vertex exactly T_face (the float 1e-8) off the plane: the float bounds cannot decide, the
    exact squared-distance decision accepts it (bound = T_face); one ulp more is refused."""
    t = 1e-8
    tol = dt.DisplayTolerance(.1, sag=t)
    g, f = T._planar([[(0, 0, t), (2, 0, 0), (1, 1, 0)]], tolerance=_LOOSE)
    mesh = T._check_certified(g, f, tol, _mesh(g, f, tol))
    assert mesh.deviation_bound == t
    g, f = T._planar([[(0, 0, math.nextafter(t, 1.)), (2, 0, 0), (1, 1, 0)]], tolerance=_LOOSE)
    _refused("EDGE_UNSUPPORTED", lambda: _mesh(g, f, tol))


_LOOSE = TolerancePolicy(surface_residual=1e-7)


def test_m3_the_exact_plane_decision_matches_a_60_digit_distance():
    """The exact fallback on skew planes far from the origin: every proven ``D <= limit`` has a
    returned bound at or above the 60-digit distance (and at most the limit); every refusal has a
    distance above the limit or within 1e-30 relative of it (undecided: fail closed)."""
    from anygeometry import _display_lift as lift
    rng = np.random.default_rng(31)
    decided = 0
    for case in range(60):
        o = rng.uniform(-1e6, 1e6, 3)
        u, v = rng.normal(size=3), rng.normal(size=3)
        plane = lift._PlaneChart(1, Plane(tuple(o), tuple(u), tuple(v)))
        x, y = rng.uniform(-3, 3, 2)
        offset = float(rng.uniform(.5e-8, 1.5e-8))
        point = o + x * plane.e1 + y * plane.e2 + offset * plane.n
        chart = plane.chart(point[None])[0]
        limit = 1e-8
        got = plane.exact_bound(point, chart, limit)
        with mp.workdps(60):
            uu = [mp.mpf(float(t)) for t in u]
            vv = [mp.mpf(float(t)) for t in v]
            nu = mp.sqrt(sum(t * t for t in uu))
            e1 = [t / nu for t in uu]
            dot = sum(a * b for a, b in zip(vv, e1))
            w2 = [a - dot * b for a, b in zip(vv, e1)]
            nw = mp.sqrt(sum(t * t for t in w2))
            e2 = [t / nw for t in w2]
            s = [mp.mpf(float(oo)) + mp.mpf(float(chart[0])) * a + mp.mpf(float(chart[1])) * b
                 for oo, a, b in zip(o, e1, e2)]
            distance = mp.sqrt(sum((mp.mpf(float(p)) - q) ** 2 for p, q in zip(point, s)))
            if got is None:
                assert distance > mp.mpf(limit) * (1 - mp.mpf(10) ** -30)
            else:
                decided += 1
                assert distance <= mp.mpf(got) and got <= limit
    assert 10 < decided < 50


# ----------------------------------------------------------------------------- M4
def test_m4_the_ring_point_budget_is_enforced_while_sampling(monkeypatch):
    """Codex M4: with the N2 cap patched to 4, an eight-edge octagon made all eight N1 sampling
    calls before N2 refused.  Now the running total refuses as soon as it crosses the cap (one
    point per line coedge: the fifth call), with N2's fixed text."""
    monkeypatch.setattr(slab, "MAX_SLAB_RING_POINTS", 4)
    ring = [(math.cos(t), math.sin(t)) for t in np.arange(8) * math.pi / 4]
    g, f = T._planar([ring])
    calls = []
    original = dt._edge_display_samples_and_allowance

    def counting(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(dt, "_edge_display_samples_and_allowance", counting)
    text = _refused("OUT_OF_DOMAIN", lambda: _mesh(g, f, dt.DisplayTolerance(.1)))
    assert "ring points (MAX_SLAB_RING_POINTS = 4)" in text and text.startswith("OUT_OF_DOMAIN: trim too complex")
    assert len(calls) <= 4 + 1


# ----------------------------------------------------------------------------- L1
def test_l1_the_orientation_oracle_finds_an_interior_minimum():
    """Codex L1: ``0.9999 - cos(theta - 0.0625)`` is positive at the nine samples j/8 on [0, 1]
    that the cone oracle took, negative at 0.0625.  The oracle's exact footprint minimum (ends
    and the sinusoid's critical point) finds it."""
    with mp.workdps(50):
        p, q, c0 = -mp.cos(mp.mpf(.0625)), -mp.sin(mp.mpf(.0625)), mp.mpf(.9999)
        assert all(c0 + p * mp.cos(mp.mpf(j) / 8) + q * mp.sin(mp.mpf(j) / 8) > 0 for j in range(9))
        assert C._footprint_minimum(p, q, c0, mp.mpf(0), mp.mpf(1)) < 0
        assert C._footprint_minimum(p, q, c0 - 1, mp.mpf(0), mp.mpf(1)) < C._footprint_minimum(p, q, c0, 0, 1)
        # a footprint not reaching the critical point: the ends decide
        assert C._footprint_minimum(p, q, c0, mp.mpf(.5), mp.mpf(1)) > 0
