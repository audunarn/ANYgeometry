"""N3 Codex round 5 (codex-n3r5.md; SPEC-N3-R5-FIXES.md): the reproduction as a regression test.

H1  class "magnitude domain, upper side" (round 2 M1/M2 fixed only the lower side): a cone whose
    ``height |axis|`` overflowed computed the slope ``k = 0`` (exactly about 0.0556), so the lift
    published a folded triangle with an understated ``deviation_bound``.  The lift's domain is now
    symmetric: a support parameter or ring coordinate above ``2^200`` in magnitude is
    ``OUT_OF_DOMAIN`` (fixed text); inside it every derived float quantity is finite and a nonzero
    one never underflows (docs, "Codex round 5": the audit table).  The same face shape with
    in-domain parameters is certified by the independent cone oracle (exact slope).
"""
from __future__ import annotations

import math
import sys

import mpmath as mp
import numpy as np
import pytest

import test_display_tessellation_lift as T
import test_display_tessellation_lift_cone as K
import test_display_tessellation_lift_cylinder as C
from anygeometry import _display_lift as lift
from anygeometry import display_tessellation as dt
from anygeometry.model import GeometryModel
from anygeometry.surfaces import Cone, Cylinder, Plane
from anygeometry.tolerance import TolerancePolicy

_mesh, _refused = T._mesh, T._refused
_TOP = 2.0 ** 200


def _codex_face(radius_end=1e307, height=sys.float_info.max):
    """Codex N3 r5 H1: a small patch near radius 1 of a cone of enormous height and end radius."""
    s = Cone((0, 0, 0), (1, 1, 1), (0, 1, 0), 1., radius_end, height)
    g = GeometryModel(tolerance=TolerancePolicy(surface_residual=.01))
    p = [g.add_point(*(r * s.evaluate(t / s.sweep_angle, 0) + h * s.axis))
         for t, h, r in [(0., .001, 1.), (.005, .001, 1.), (.005, .00101, .99968), (0., .00101, .99968)]]
    f = g.add_face([g.add_line(a, b) for a, b in zip(p, p[1:] + p[:1])], surface=s)
    assert not g.validate_topology()
    return g, f


_CODEX_TOL = dt.DisplayTolerance(.02, sag=.00035)


def test_h1_the_codex_cone_is_out_of_domain():
    """Codex N3 r5 H1: on 75c52fc ``height |axis|`` overflowed, ``k = 0``, and the face was accepted
    with deviation_bound 3.23e-4 for a paired deviation of 3.76e-4 and a folded first triangle."""
    g, f = _codex_face()
    text = _refused("OUT_OF_DOMAIN", lambda: _mesh(g, f, _CODEX_TOL))
    assert "above 2**200" in text


def test_h1_the_codex_mesh_fails_the_exact_slope_oracle_had_it_been_published(monkeypatch):
    """The oracle side of the reproduction: with the upper guard and the derived-quantity check both
    bypassed, the lift publishes the old mesh (k = 0), whose deviation claim is false against the
    exact slope; so each of the two checks alone refuses the face."""
    monkeypatch.setattr(lift, "_MAX_MAGNITUDE", float("inf"))
    monkeypatch.setattr(lift, "_check_derived", lambda *values, nonzero=(): None)
    g, f = _codex_face()
    assert lift._ConeChart(g.faces[f].surface).k == 0.0
    mesh = _mesh(g, f, _CODEX_TOL)
    s = g.faces[f].surface
    with mp.workdps(60):
        _, e1, e2, axis = C._mp_frame(s)
        k = (mp.mpf(s.radius_end) - 1) / (mp.mpf(s.height) * mp.sqrt(sum(mp.mpf(float(v)) ** 2 for v in s.axis)))
        theta, h = map(mp.mpf, mesh.chart[2])
        radial = [mp.cos(theta) * x + mp.sin(theta) * y for x, y in zip(e1, e2)]
        ref = [(1 + k * h) * x + h * z for x, z in zip(radial, axis)]
        deviation = mp.sqrt(sum((mp.mpf(x) - y) ** 2 for x, y in zip(mesh.points[2], ref)))
    assert mesh.deviation_bound < mesh.face_tolerance < deviation


def test_h1_the_derived_slope_check_alone_refuses_the_codex_cone(monkeypatch):
    """Defence in depth: with only the upper guard bypassed, the slope check refuses (k = 0 for a
    cone with r1 != r0)."""
    monkeypatch.setattr(lift, "_MAX_MAGNITUDE", float("inf"))
    g, f = _codex_face()
    text = _refused("OUT_OF_DOMAIN", lambda: _mesh(g, f, _CODEX_TOL))
    assert "derived quantity" in text


@pytest.mark.parametrize("name, make", [
    ("cone height", lambda: Cone((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 1.5, 2.0 ** 201)),
    ("cone end radius", lambda: Cone((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 2.0 ** 201, 1.)),
    ("cone start radius", lambda: Cone((0, 0, 0), (0, 0, 1), (1, 0, 0), 2.0 ** 201, 1., 1.)),
    ("negative cone height", lambda: Cone((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 1.5, -2.0 ** 201)),
    ("cylinder radius", lambda: Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 2.0 ** 201, 1.)),
])
def test_h1_support_parameters_above_2_200_are_out_of_domain(name, make):
    """The upper guard: any support parameter above 2^200 in magnitude, before any arithmetic on it."""
    support = make()
    ring = np.array([[1., 0., .25], [0., 1., .25], [-1., 0., .25]])
    with pytest.raises(dt.DisplayDomainRefused, match=r"above 2\*\*200"):
        lift._check_magnitudes(7, support, [lift._Ring(ring, None, None, 0.)])


def test_h1_the_upper_guard_is_inclusive_at_2_200():
    """``2^200`` itself is inside the domain (the guard refuses strictly above it)."""
    support = Cone((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 1.5, _TOP)
    ring = np.array([[1., 0., .25], [0., 1., .25], [-1., 0., .25]])
    lift._check_magnitudes(7, support, [lift._Ring(ring, None, None, 0.)])
    with pytest.raises(dt.DisplayDomainRefused):
        lift._check_magnitudes(7, Cone((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 1.5, np.nextafter(_TOP, np.inf)),
                               [lift._Ring(ring, None, None, 0.)])


@pytest.mark.parametrize("radius_end, height", [
    (5.0e58, 8.98846567431158e59),          # the Codex slope (about 0.0556) inside the domain
    (_TOP * 0.0556, _TOP),                   # at the guard
    (1.0 + 2.0 ** -30, 2.0 ** 150),          # a nearly cylindrical cone: k = 2^-180, nonzero
])
def test_h1_the_codex_shape_inside_the_domain_is_certified(radius_end, height):
    """The same face with in-domain parameters: the slope is computed (finite, nonzero) and an
    accepted mesh passes the independent cone oracle (exact slope: deviation and orientation over
    every footprint, and no 3D overlap); a refusal is typed and is not the magnitude guard."""
    g, f = _codex_face(radius_end, height)
    chart = lift._ConeChart(g.faces[f].surface)
    assert np.isfinite(chart.k) and chart.k != 0.0
    try:
        mesh = _mesh(g, f, _CODEX_TOL)
    except dt.DisplayTrimRefused as error:
        assert "2**200" not in str(error)
        return
    K._check_cone(g, f, _CODEX_TOL, mesh)


def test_h1_an_accepted_cone_triangle_satisfies_q7_and_the_projection_jacobian():
    """Both orientation certificates hold on an accepted cone mesh, each against its own reference:
    Q7 the support normal ``c(theta) - k a`` (the same slope ``k`` as the deviation bound, the one
    :class:`_ConeChart` computes), the round-4 projection the radial ``c(theta)`` (P = (azimuth,
    axial) does not depend on ``k``).  Checked exactly at every vertex azimuth."""
    g, f = K.frustum_patch(r0=.3, r1=1.2, height=1., h=(.1, .9), sweep=1.)
    tol = dt.DisplayTolerance(.2)
    mesh = K._check_cone(g, f, tol, _mesh(g, f, tol))
    s = g.faces[f].surface
    chart = lift._ConeChart(s)
    with mp.workdps(50):
        o, e1, e2, a, r0, k = K._mp_cone(s)
        assert abs(k - mp.mpf(chart.k)) <= 8 * mp.mpf(2) ** -52 * abs(k)
        pts = [[mp.mpf(float(x)) for x in p] for p in mesh.points]
        for t in mesh.triangles.tolist():
            p, q, r = (pts[i] for i in t)
            n = C._mp_cross([y - x for x, y in zip(p, q)], [y - x for x, y in zip(p, r)])
            for i in t:
                theta = mp.mpf(float(mesh.chart[i][0]))
                c = [mp.cos(theta) * x + mp.sin(theta) * y for x, y in zip(e1, e2)]
                assert mesh.sense * sum(nn * (cc - k * z) for nn, cc, z in zip(n, c, a)) > 0       # Q7
                assert mesh.sense * sum(nn * cc for nn, cc in zip(n, c)) > 0                       # P


# ----------------------------------------------------------------------------- P1: the scalar path decides as the array path
def _outcome(call):
    try:
        call()
    except dt.DisplayTrimRefused as error:
        return error.code, str(error)
    return None


def _both(plane_or_chart, points, theta=None):
    if theta is None:
        big, small = lift._PlaneProjection(plane_or_chart, points), lift._SmallPlaneProjection(plane_or_chart, points)
    else:
        big = lift._CurvedProjection(7, plane_or_chart, points, theta)
        small = lift._SmallCurvedProjection(7, plane_or_chart, points, theta)
    big.normals = small.normals = None
    return big, small


def _random_chain(rng, n):
    order = rng.permutation(n).tolist()
    if n >= 6 and rng.random() < .5:
        cut = int(rng.integers(3, n - 2))
        loops = (order[:cut], order[cut:])
    else:
        loops = (order,)
    tails = np.array([v for loop in loops for v in loop], dtype=np.int64)
    heads = np.array([v for loop in loops for v in loop[1:] + loop[:1]], dtype=np.int64)
    return tails, heads


def test_p1_the_scalar_boundary_check_decides_as_the_array_one_on_a_plane():
    """Random small chains on a plane (grid-snapped points: collinear pieces, touching vertices,
    repeated points, crossings; one or two loops, either orientation): the scalar path's outcome
    (accepted, or the refusal code and text) equals the array path's on every one."""
    rng = np.random.default_rng(2615)
    plane = lift._PlaneChart(7, Plane((.5, -.25, 3.), (1., .5, 0.), (-.25, 1., .5)))
    outcomes = {}
    for case in range(1500):
        n = int(rng.integers(3, 11))
        uv = rng.integers(0, 5, size=(n, 2)) / 2.0 if case % 2 else rng.random((n, 2)) * 4.0
        points = np.array([plane.o + a * plane.e1 + b * plane.e2 for a, b in uv])
        big, small = _both(plane, points)
        chain = _random_chain(rng, n)
        want = _outcome(lambda: lift._check_boundary(7, big, *chain))
        got = _outcome(lambda: lift._check_boundary_small(7, small, *chain))
        assert got == want, (case, got, want)
        outcomes[want and want[1].split("face 7: ")[-1][:30]] = outcomes.get(want and want[1].split("face 7: ")[-1][:30], 0) + 1
    assert len(outcomes) >= 3, outcomes                    # accepted, not simple, not nested all seen


def test_p1_the_scalar_checks_decide_as_the_array_ones_on_a_cylinder():
    """Random small chains and triangles on a cylinder patch (azimuths within a quarter turn): the
    scalar path's boundary and triangle outcomes equal the array path's."""
    rng = np.random.default_rng(2616)
    support = Cylinder((.3, -.2, 1.), (0., .6, .8), (1., 0., 0.), 2., 1.)
    chart = lift._CylinderChart(support)
    seen = set()
    for case in range(800):
        n = int(rng.integers(3, 10))
        th = rng.integers(0, 6, size=n) * .1 if case % 2 else rng.random(n) * .6
        h = rng.integers(0, 4, size=n) * .25 if case % 3 else rng.random(n)
        points = np.array([chart.o + 2. * (math.cos(t) * chart.e1 + math.sin(t) * chart.e2) + z * chart.a
                           for t, z in zip(th, h)])
        big, small = _both(chart, points, th.astype(float))
        chain = _random_chain(rng, n)
        want = _outcome(lambda: lift._check_boundary(7, big, *chain))
        got = _outcome(lambda: lift._check_boundary_small(7, small, *chain))
        assert got == want, (case, got, want)
        seen.add(want and want[1].split("face 7: ")[-1][:30])
        tris = np.array([rng.choice(n, 3, replace=False) for _ in range(4)], dtype=np.int64)
        for sense in (1, -1):
            want = _outcome(lambda: lift._projection_triangles(7, big, points, tris, sense))
            got = _outcome(lambda: lift._projection_triangles_small(7, small, points, tris, sense))
            assert got == want, (case, sense, got, want)
    assert len(seen) >= 3, seen


def test_p1_the_scalar_triangle_check_decides_as_the_array_one_on_a_plane():
    rng = np.random.default_rng(2617)
    plane = lift._PlaneChart(7, Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    for case in range(400):
        n = 8
        uv = rng.integers(0, 3, size=(n, 2)) / 2.0 if case % 2 else rng.random((n, 2))
        points = np.array([(a, b, 0.) for a, b in uv])
        big, small = _both(plane, points)
        tris = np.array([rng.choice(n, 3, replace=False) for _ in range(5)], dtype=np.int64)
        want = _outcome(lambda: lift._projection_triangles(7, big, points, tris, 1))
        got = _outcome(lambda: lift._projection_triangles_small(7, small, points, tris, 1))
        assert got == want, (case, got, want)


def test_p1_small_faces_take_the_scalar_path_and_large_ones_the_array_path():
    support = Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 1.)
    chart = lift._CylinderChart(support)
    for n, kind in ((lift._SMALL_POINTS, lift._SmallCurvedProjection), (lift._SMALL_POINTS + 1, lift._CurvedProjection)):
        th = np.linspace(0., .5, n)
        points = np.array([chart.o + math.cos(t) * chart.e1 + math.sin(t) * chart.e2 for t in th])
        assert type(lift._projection(chart, points, th, None, 7)) is kind


def test_p1_the_plane_separating_axis_filter_is_sound():
    """``_beyond`` (the r5 float filter for non-adjacent plane boundary pairs) never claims two
    segments apart that meet exactly (random grid segments, many collinear and touching)."""
    rng = np.random.default_rng(2618)
    plane = lift._PlaneChart(7, Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    n = 400
    points = np.column_stack((rng.integers(0, 4, size=(n, 2)) / 4.0, np.zeros(n)))
    proj = lift._PlaneProjection(plane, points)
    small = lift._SmallPlaneProjection(plane, points)
    # four distinct indices, both segments of positive length (the exact meet's precondition: a
    # zero-length chain edge is refused as an adjacent pair first)
    quads = [q for q in rng.integers(0, n, size=(3000, 4)).tolist()
             if len(set(q)) == 4 and tuple(points[q[0]]) != tuple(points[q[1]]) and tuple(points[q[2]]) != tuple(points[q[3]])]
    p, q, r, s = np.array(quads, dtype=np.int64).T
    apart = proj.apart(p, q, r, s)
    for flag, (a, b, c, d) in zip(apart.tolist(), quads):
        if flag:
            assert not proj.meet(a, b, c, d), (a, b, c, d)
            assert small.apart_scalar(a, b, c, d)
    assert int(apart.sum()) > 100
