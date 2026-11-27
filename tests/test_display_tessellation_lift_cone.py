"""N3c: face display meshes on cones (SPEC-N3 section 4, slice N3c; SPEC-N3c).

T12-T13, the corpus cone faces and the frustum acceptance cases: every accepted mesh is checked
against independent oracles: the exact (Fraction) chart combinatorics of the N3a tests and a
50-digit mpmath paired deviation, sampled on a barycentric grid (``S*(theta, h) = o + rho(h) c(theta) + h a``, ``rho = r0 + k h``,
``k = (r1 - r0) / (height |axis|)``) and the exact footprint minimum of the orientation (``c(theta) - k a``) against the EXACT
orthonormal frame about the support's axis.
"""
from __future__ import annotations

import math

import mpmath as mp
import numpy as np
import pytest

import display_tessellation_corpus as corpus
import test_display_tessellation_lift as N3a
import test_display_tessellation_lift_cylinder as C
from anygeometry import _slab_triangulation as slab
from anygeometry import display_tessellation as dt
from anygeometry.entities import OrientedEdge
from anygeometry.exact_curves import EllipticArc
from anygeometry.model import GeometryModel
from anygeometry.surfaces import Cone
from anygeometry.tolerance import TolerancePolicy

_mesh, _refused = N3a._mesh, N3a._refused


# ----------------------------------------------------------------------------- oracle
def _mp_cone(support):
    o, e1, e2, a = C._mp_frame(support)
    axis_length = mp.sqrt(sum(mp.mpf(float(x)) ** 2 for x in support.axis))
    r0, r1 = mp.mpf(float(support.radius_start)), mp.mpf(float(support.radius_end))
    k = (r1 - r0) / (mp.mpf(float(support.height)) * axis_length)
    return o, e1, e2, a, r0, k


def _face_tolerance(model, face, tol):
    if tol.sag is not None:
        return tol.sag
    f = model.faces[face]
    a = max(dt.display_rounding_allowance(model, u.edge, tol).angle for loop in (f.loop, *f.holes) for u in loop)
    r_max = max(f.surface.radius_start, f.surface.radius_end)
    return r_max * (tol.angle + a) ** 2 / 8 + model.tolerance.surface_residual


def _check_cone(model, face, tol, mesh, grid=C.GRID_33, every=1):
    """Independent oracles of every published claim of an accepted cone face mesh.  The paired
    deviation is SAMPLED (a barycentric grid per triangle, a lower estimate of the maximum); the
    orientation is exact over the footprint (:func:`C._footprint_minimum`)."""
    support = model.faces[face].surface
    rings = N3a._n1_rings(model, face, tol)
    N3a._check_combinatorics(mesh, rings)
    k_ = mesh.ring_vertex_count
    assert mesh.points[:k_].tobytes() == np.vstack(rings).tobytes()
    assert mesh.face_tolerance == _face_tolerance(model, face, tol)
    assert 0.0 < mesh.deviation_bound <= mesh.face_tolerance
    assert mesh.trim_sag_bound >= 0.0
    with mp.workdps(50):
        o, e1, e2, a, r0, k = _mp_cone(support)

        def s_star(theta, h):
            ct, st, rho = mp.cos(theta), mp.sin(theta), r0 + k * h
            return [oo + rho * (ct * x + st * y) + h * z for oo, x, y, z in zip(o, e1, e2, a)]
        pts = [[mp.mpf(float(x)) for x in p] for p in mesh.points]
        chart = [[mp.mpf(float(x)) for x in c] for c in mesh.chart]
        bound = mp.mpf(mesh.deviation_bound)
        worst = mp.mpf(0)
        for t in mesh.triangles.tolist()[::every]:
            for w in grid:
                x = [sum(mp.mpf(wi) * pts[i][d] for wi, i in zip(w, t)) for d in range(3)]
                th = sum(mp.mpf(wi) * chart[i][0] for wi, i in zip(w, t))
                h = sum(mp.mpf(wi) * chart[i][1] for wi, i in zip(w, t))
                worst = max(worst, mp.sqrt(sum((xx - ss) ** 2 for xx, ss in zip(x, s_star(th, h)))))
            # orientation over the footprint: sense x (c(theta) - k a), its exact minimum (the
            # ends and the sinusoid's critical point, at 50 digits)
            p, q, s = (pts[i] for i in t)
            n = C._mp_cross([y - x for x, y in zip(p, q)], [y - x for x, y in zip(p, s)])
            na = sum(nn * z for nn, z in zip(n, a))
            ths = [chart[i][0] for i in t]
            lo, hi = min(ths), max(ths)
            pc = mesh.sense * sum(nn * x for nn, x in zip(n, e1))
            qc = mesh.sense * sum(nn * y for nn, y in zip(n, e2))
            minimum = C._footprint_minimum(pc, qc, -mesh.sense * k * na, lo, hi)
            if not minimum > 0:
                C._check_exempt(n, minimum, k, (p, q, s), mesh)
        assert worst <= bound, (float(worst), mesh.deviation_bound)
    return mesh


# ----------------------------------------------------------------------------- fixtures
def frustum_patch(r0=.01, r1=1., height=1., h=(0., 1.), start=0., sweep=math.pi / 2, origin=(0., 0., 0.),
                  axis=(0., 0., 1.), holes=()):
    """A cone patch: circle arcs at heights ``h`` and two rulings; ``holes`` (fractions of the
    sweep and height) are windows of the same kind, opposite orientation."""
    g = GeometryModel()
    o = np.asarray(origin, dtype=float)
    a, e1, e2 = C._frame(axis)
    support = Cone(tuple(o), tuple(a), tuple(e1), r0, r1, height)
    k = (r1 - r0) / height

    def loop(ta, tb, h0, h1, forward):
        def at(t, z):
            rho = r0 + k * z
            return o + rho * (math.cos(t) * e1 + math.sin(t) * e2) + z * a
        p = [g.add_point(*at(t, z)) for t, z in ((ta, h0), (tb, h0), (tb, h1), (ta, h1))]
        arcs = []
        for z, (u, v) in ((h0, (p[0], p[1])), (h1, (p[3], p[2]))):
            rho = r0 + k * z
            arcs.append(g.add_curve(u, v, EllipticArc(tuple(o + z * a), tuple(rho * e1), tuple(rho * e2), ta, tb - ta)))
        uses = [OrientedEdge(arcs[0], True), OrientedEdge(g.add_line(p[1], p[2]), True), OrientedEdge(arcs[1], False),
                OrientedEdge(g.add_line(p[3], p[0]), True)]
        return tuple(uses if forward else [OrientedEdge(u.edge, not u.forward) for u in reversed(uses)])
    outer = loop(start, start + sweep, h[0], h[1], True)
    face = g.add_face_from_loop(outer, (0, 1, 2, 3), surface=support)
    if holes:
        made = []
        for f0, f1, g0, g1 in holes:
            made.append(loop(start + f0 * sweep, start + f1 * sweep, h[0] + g0 * (h[1] - h[0]),
                             h[0] + g1 * (h[1] - h[0]), False))
        N3a._force(g, face, holes=tuple(made))
    return g, face


def skew_chord_face(theta1=.04357):
    """On the r6 frustum: a straight (non-ruling) line from uv (0, 0) to (theta1, h = 1), the top
    circle back to theta 0 and the ruling down.  N1 draws the line as one fixed chord (a line is
    exact, no sag to refine); its paired midpoint error on the cone is ~ |k| dT dH / 4 = 0.010784
    for theta1 = 0.04357 (the cone edge N1 accepts at sag 0.01068, codex-n2r7)."""
    g = GeometryModel(tolerance=TolerancePolicy(surface_residual=.05))     # the skew line validates
    support = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), .01, 1., 1., 0., math.pi / 2)
    p0 = g.add_point(.01, 0., 0.)
    p1 = g.add_point(math.cos(theta1), math.sin(theta1), 1.)
    p2 = g.add_point(1., 0., 1.)
    top = g.add_curve(p2, p1, EllipticArc((0., 0., 1.), (1., 0., 0.), (0., 1., 0.), 0., theta1))
    skew, ruling = g.add_line(p0, p1), g.add_line(p2, p0)
    loop = (OrientedEdge(skew, True), OrientedEdge(top, False), OrientedEdge(ruling, True))
    face = g.add_face([skew, top, ruling], surface=support)
    N3a._force(g, face, loop=loop)
    return g, face, support, skew


# ----------------------------------------------------------------------------- T12 (Q4)
def test_t12_q4_the_cone_bound_holds_where_the_radial_metric_does_not():
    """r6 frustum, unit uv square: on the naive two-triangle mesh the paired (parametric) error
    0.3800 is within B_cone = rho_max dT^2 / 8 + |k| dT dH / 4 = 0.697 (the mixed angle-height
    term is what makes it valid) while the radial distance 0.2929 is not an upper bound.  N3
    refines the real face (N1 rings) to <= T_face, oracle-checked."""
    cone = Cone((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), .01, 1., 1., 0., math.pi / 2)
    uv = np.array([(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
    naive = slab.triangulate_rings([uv])
    xyz = np.array([cone.evaluate(*p) for p in naive.points])
    with mp.workdps(30):
        worst = mp.mpf(0)
        for t in naive.triangles.tolist():
            for w in C._grid(12):
                x = [sum(mp.mpf(wi) * mp.mpf(float(xyz[i][d])) for wi, i in zip(w, t)) for d in range(3)]
                u = sum(mp.mpf(wi) * mp.mpf(float(naive.points[i][0])) for wi, i in zip(w, t))
                v = sum(mp.mpf(wi) * mp.mpf(float(naive.points[i][1])) for wi, i in zip(w, t))
                theta, rho = u * mp.pi / 2, mp.mpf(.01) + v * mp.mpf(.99)
                s = [rho * mp.cos(theta), rho * mp.sin(theta), v]
                worst = max(worst, mp.sqrt(sum((a - b) ** 2 for a, b in zip(x, s))))
    radial = 1 - 2 ** -.5
    dT, dH, k, rho_max = math.pi / 2, 1.0, .99, 1.0
    b_cone = rho_max * dT ** 2 / 8 + abs(k) * dT * dH / 4
    assert radial < float(worst) <= b_cone and abs(float(worst) - .3800) < 1e-4
    assert abs(b_cone - .697) < 1e-3
    assert rho_max * dT ** 2 / 8 < float(worst)                 # without the mixed term it would fail
    g, face = frustum_patch()
    for step in (.3, corpus.INTERACTION_STEP, .1):
        tol = dt.DisplayTolerance(step)
        mesh = _check_cone(g, face, tol, _mesh(g, face, tol), grid=C._grid(3), every=3)
        plain = slab.triangulate_rings([np.vstack(_cone_charts(g, face, tol))])
        assert len(mesh.triangles) > len(plain.triangles)        # refined: the mixed term needs narrow slabs
        assert mesh.sense == 1


def _cone_charts(g, face, tol):
    from anygeometry import _display_lift as lift
    f = g.faces[face]
    rings = [lift._ring(g, face, loop, tol) for loop in (f.loop, *f.holes)]
    with np.errstate(all="ignore"):
        return lift._ConeChart(f.surface).charts(face, rings)


def test_t12_q4_a_fixed_cone_chord_over_the_face_tolerance_is_edge_unsupported():
    """The skew line N1 accepts at sag 0.01068 is one fixed chord whose true paired midpoint error
    (0.010784, mpmath) exceeds T_face = 0.01068: refinement inside cannot repair a boundary chord,
    so the face is EDGE_UNSUPPORTED.  At a larger sag it is accepted and certified."""
    g, face, support, skew = skew_chord_face()
    tol = dt.DisplayTolerance(.1, sag=.01068)
    samples = dt.edge_display_samples(g, skew, tol)               # N1 accepts the edge: one chord
    assert len(samples.points) == 2
    with mp.workdps(30):
        a, b = ([mp.mpf(float(x)) for x in p] for p in samples.points)
        mid = [(x + y) / 2 for x, y in zip(a, b)]
        theta = mp.atan2(b[1], b[0]) / 2
        rho = mp.mpf(.01) + mp.mpf(.99) * mp.mpf(.5)
        paired = mp.sqrt(sum((x - y) ** 2 for x, y in zip(mid, [rho * mp.cos(theta), rho * mp.sin(theta), .5])))
    assert .01068 < float(paired) < .0108
    text = _refused("EDGE_UNSUPPORTED", lambda: _mesh(g, face, tol))
    assert "face tolerance" in text
    tol = dt.DisplayTolerance(.1, sag=.012)
    _check_cone(g, face, tol, _mesh(g, face, tol))


# ----------------------------------------------------------------------------- T13 (domain)
@pytest.mark.parametrize("name", ["cone_apex_30", "cone_apex_220", "far_apex_cone", "cone_section",
                                  "cone_section_steep", "cone_section_far"])
def test_t13_a_trim_reaching_the_apex_is_out_of_domain(name):
    case = corpus.ALL[name]()
    for face in case.faces.values():
        for step in corpus.STEPS:
            text = _refused("OUT_OF_DOMAIN", lambda: _mesh(case.model, face, dt.DisplayTolerance(step)))
            assert "apex" in text


@pytest.mark.parametrize("h, sweep, step", [((500., 1000.), .5, .3), ((500., 1000.), .5, .05),
                                            ((0., 1e-3), .5, .3), ((990., 1000.), 2.5, .2)])
def test_t13_a_very_large_frustum_is_accepted_or_too_complex(h, sweep, step):
    """Frustum radii 3e-6 / 5e8 over a height of 1e3: accepted (certified) or OUT_OF_DOMAIN
    (N1's radius floor, or 'too complex' refinement), never a raw error.  One more typed outcome
    (N3c finding): at radius 5e8 the rounding allowance A_face (~9e-6) exceeds the Q5 default's
    slack (surface_residual), so an N1 chord at exactly the full angle step on the largest radius
    (sweep / angle an integer: .5 / .05) is EDGE_UNSUPPORTED by rounding alone."""
    try:
        g, face = frustum_patch(3e-6, 5e8, 1e3, h=h, sweep=sweep)
    except Exception as error:                                     # the model refuses the patch itself
        pytest.skip(f"model refused the fixture: {error}")
    tol = dt.DisplayTolerance(step)
    try:
        mesh = _mesh(g, face, tol)
    except dt.DisplayTrimRefused as error:
        assert error.code in ("OUT_OF_DOMAIN", "EDGE_UNSUPPORTED"), str(error)
        if error.code == "EDGE_UNSUPPORTED":
            words = str(error).split()
            found, limit = float(words[words.index("to") + 1].rstrip(",")), float(words[-1])
            assert "boundary chord" in str(error) and found - limit < 1e-4          # rounding level
        return
    _check_cone(g, face, tol, mesh, grid=C._grid(2), every=max(1, len(mesh.triangles) // 40))


# ----------------------------------------------------------------------------- corpus + acceptance
CORPUS_CONE = {
    "overrun_cone": "accepted",
    "overrun_cone_seam": "accepted",
    "cone_comb_r1": "accepted",
    "cone_comb_r3": "accepted",
    "holed_cone_comb": "accepted",
    "lens_01": "accepted",
    "lens_0002": "accepted",
    "lens_with_neighbour": "accepted",
    "cone_apex_30": "OUT_OF_DOMAIN",
    "cone_apex_220": "OUT_OF_DOMAIN",
    "far_apex_cone": "OUT_OF_DOMAIN",
    "cone_section": "OUT_OF_DOMAIN",
    "cone_section_steep": "OUT_OF_DOMAIN",
    "cone_section_far": "OUT_OF_DOMAIN",
}


def test_corpus_cone_faces_are_all_listed():
    found = set()
    for name, make in corpus.ALL.items():
        if name in ("generated_cylinder_400", "tripod_skins"):      # large, cylinders only (N3b)
            continue
        case = make()
        if any(isinstance(case.model.faces[f].surface, Cone) for f in case.faces.values()):
            found.add(name)
    assert found == set(CORPUS_CONE)


@pytest.mark.parametrize("name", sorted(CORPUS_CONE))
@pytest.mark.parametrize("step", corpus.STEPS, ids=["fine", "interaction"])
def test_corpus_cone_faces_accepted_or_refused_with_their_code(name, step):
    case = corpus.ALL[name]()
    tol = dt.DisplayTolerance(step)
    for face in case.faces.values():
        if not isinstance(case.model.faces[face].surface, Cone):
            continue
        expected = CORPUS_CONE[name]
        if expected == "accepted":
            _check_cone(case.model, face, tol, _mesh(case.model, face, tol), grid=C._grid(3),
                        every=1 if name != "holed_cone_comb" else 2)
        else:
            _refused(expected, lambda: _mesh(case.model, face, tol))


@pytest.mark.parametrize("origin, axis, radii, height", [((0., 0., 0.), (0., 0., 1.), (1., 2.), 3.),
                                                         ((5., -3., 7.), (1., 2., 2.), (.7, .2), 1.3),
                                                         ((-40., 12., 9.), (0., 1., 1.), (.25, .25), .5),
                                                         ((0., 0., 0.), (0., 0., 1.), (2., .5), -1.5)])
def test_cone_patches_with_a_window_are_accepted(origin, axis, radii, height):
    """Tilted, offset, narrowing, equal-radius (k = 0) and negative-height cones with a window
    hole sampled out of phase: accepted and certified at both corpus steps."""
    span = (0., abs(height)) if height > 0 else (height, 0.)
    margin = .1 * (span[1] - span[0])
    g, face = frustum_patch(radii[0], radii[1], height, h=(span[0] + margin, span[1] - margin),
                            start=.3, sweep=2.2, origin=origin, axis=axis, holes=((.35, .75, .42, .58),))
    for step in corpus.STEPS:
        tol = dt.DisplayTolerance(step)
        _check_cone(g, face, tol, _mesh(g, face, tol), grid=C._grid(3), every=2)


def test_q5_cone_face_tolerance_uses_the_larger_radius():
    g, face = frustum_patch(2., .5, 3., h=(.5, 2.5), sweep=1.)
    tol = dt.DisplayTolerance(.2)
    mesh = _mesh(g, face, tol)
    assert mesh.face_tolerance == _face_tolerance(g, face, tol)
    assert mesh.face_tolerance > 2.0 * .2 ** 2 / 8
    tol = dt.DisplayTolerance(.2, sag=1e-3)
    assert _check_cone(g, face, tol, _mesh(g, face, tol)).face_tolerance == 1e-3
