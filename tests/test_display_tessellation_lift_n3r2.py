"""N3 Codex round 2 (codex-n3r2.md; SPEC-N3-R2-FIXES.md): every reproduction as a regression test.

H1  an identification (D1 merge, chain cancellation) is allowed only within the rounding allowance
    A_face, never merely within T_face or the sag (Q8: no boundary gap beyond rounding); since
    owner decision C1 (r3) only between bitwise-equal lifted points.
H2  on a cylinder or cone the chart -> 3D map must be injective: the unwrapped azimuth extent of the
    rings must be certifiably below 2 pi (a zero-winding loop over more than a turn is refused).
M1  the exact thin-orientation decision is scale-invariant (no absolute floor, no underflow).
M2  no exact -> binary64 conversion of a bound rounds down (and the 1e-200 plane is refused
    OUT_OF_DOMAIN by the coordinate-magnitude domain).
M3  the E2 azimuth join uses support-frame magnitudes (since owner decision C2, r3, translation
    invariance is not a requirement: a translated face is refused or certified).
L1  a planar vertex exactly T_face off the plane is accepted far from the origin (exact squared
    comparison, no inverse-square-root enclosure).
L2  the orientation oracles admit an exempt triangle only under the module's exact exemption rule.
L3  the ring and coedge budgets are checked before the seam scan.
"""
from __future__ import annotations

import math
import pathlib
import subprocess
import types
from unittest.mock import patch

import numpy as np
import pytest

import display_tessellation_corpus as corpus
import test_display_tessellation_lift as T
import test_display_tessellation_lift_cone as K
import test_display_tessellation_lift_cylinder as C
import test_display_tessellation_lift_n3r1 as R1
from anygeometry import _display_lift as lift
from anygeometry import _slab_triangulation as slab
from anygeometry import display_tessellation as dt
from anygeometry.entities import OrientedEdge
from anygeometry.model import GeometryModel
from anygeometry.surfaces import Cone, Cylinder, Plane
from anygeometry.tolerance import TolerancePolicy

_mesh, _refused, _status = T._mesh, T._refused, R1._status
ROOT = pathlib.Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- H1
@pytest.mark.parametrize("sag", [1e-6, .5001, 2.])
def test_h1_an_identification_beyond_rounding_is_refused_at_any_sag(sag):
    """Codex N3 r2 H1: with sag = .5001 the cancelled interval x = 3..3.25 (0.125 from the returned
    boundary at its midpoint) was accepted because its loss was within the sag.  An identification
    is allowed only within the rounding allowance A_face: refused at every sag."""
    g, f, z, u = R1._codex_h1_face()
    text = _refused("TRIM_INVALID", lambda: _mesh(g, f, dt.DisplayTolerance(.1, sag=sag)))
    assert "bitwise" in text and "merged cut points" in text            # C1 (r3): only bitwise-equal points


def test_h1_rounding_level_identifications_are_refused_since_c1():
    """The D1 cases whose merges were within rounding (the tip case, the aligned polygon) were
    accepted in r2; owner decision C1 (r3): identifications only of bitwise-equal lifted points, so
    both are refused ``TRIM_INVALID`` (their merged 3D cuts differ by rounding)."""
    outer = [(-3., -3.), (3., -3.), (3., 7.), (-3., 7.)]
    tip = [(0., 4.5), (1., 3.5), (1., 5.5)]
    near = [(1e-17, 0.), (.5, -.5), (.5, .5)]
    g, face = T._planar([outer, tip, near])
    assert "bitwise" in _refused("TRIM_INVALID", lambda: _mesh(g, face, dt.DisplayTolerance(.5)))
    g, face = T.big_polygon(aligned=True)
    tol = dt.DisplayTolerance(corpus.INTERACTION_STEP)
    assert "bitwise" in _refused("TRIM_INVALID", lambda: _mesh(g, face, tol))


# ----------------------------------------------------------------------------- H2
def _zero_winding(support, radius):
    """Codex N3 r2 H2: a loop over 9/8 of a turn and back (net winding 0), forced as the face's
    trim; its chart is a valid strip, its 3D image covers one part of the support twice."""
    g = GeometryModel(tolerance=TolerancePolicy(surface_residual=.1))

    def xyz(i, top):
        a = math.tau * (i % 8) / 8
        h = i / 16 + top
        rho = radius(h)
        return rho * math.cos(a), rho * math.sin(a), h

    v = [g.add_point(*xyz(i, 0)) for i in range(10)]
    v += [g.add_point(*xyz(i, 1)) for i in reversed(range(10))]
    e = [g.add_line(a, b) for a, b in zip(v, v[1:] + v[:1])]
    f = g.add_face([*e[:2], g.add_line(v[2], v[0])])
    T._force(g, f, loop=tuple(OrientedEdge(i, True) for i in e), surface=support)
    return g, f, xyz


def _double_hits(mesh, q):
    hits = 0
    for tri in mesh.points[mesh.triangles]:
        d = tri[1:] - tri[0]
        A = d[:, 1:].T
        if abs(np.linalg.det(A)) > 1e-14:
            w = np.linalg.solve(A, (q - tri[0])[1:])
            hits += bool(min(*w, 1 - w.sum()) > 1e-4 and np.linalg.norm(tri[0] + w @ d - q) < 1e-12)
    return hits


def test_h2_a_zero_winding_loop_over_more_than_a_turn_is_refused_on_a_cylinder():
    """Codex N3 r2 H2: 18 triangles, 0 exemptions, one 3D point in two triangle interiors."""
    g, f, xyz = _zero_winding(Cylinder((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 3.), lambda h: 1.)
    text = _refused("TRIM_WINDS_SUPPORT", lambda: _mesh(g, f, dt.DisplayTolerance(1.)))
    assert "2 pi" in text


def test_h2_a_zero_winding_loop_over_more_than_a_turn_is_refused_on_a_cone():
    """The same class on a cone (rho = 1 + h / 3)."""
    g, f, xyz = _zero_winding(Cone((0, 0, 0), (0, 0, 1), (1, 0, 0), 1., 2., 3.), lambda h: 1. + h / 3.)
    _refused("TRIM_WINDS_SUPPORT", lambda: _mesh(g, f, dt.DisplayTolerance(1.)))


@pytest.mark.parametrize("sweep", [math.pi, 1.8 * math.pi, 1.9 * math.pi])
def test_h2_a_face_below_a_full_turn_is_accepted(sweep):
    """Extents certifiably below 2 pi are unchanged (accepted, oracle-checked)."""
    tol = dt.DisplayTolerance(corpus.INTERACTION_STEP)
    g, face = C.quarter_cylinder(sweep=sweep)
    C._check_cylinder(g, face, tol, _mesh(g, face, tol), grid=C._grid(2))
    g, face = K.frustum_patch(r0=.5, sweep=sweep)
    K._check_cone(g, face, tol, _mesh(g, face, tol), grid=C._grid(2))


# ----------------------------------------------------------------------------- M1
@pytest.mark.parametrize("scale", [1., 2. ** -500, 2. ** -1000])
def test_m1_the_thin_orientation_is_decided_at_every_scale(scale):
    """Codex N3 r2 M1: for a normal of magnitude 1e-304 the squared norm underflowed and the
    absolute 2**-1000 floor exempted a definite fold.  The decision is the same at every scale."""
    fold = np.array([(0., 0., 0.), (0., 1., 0.), (0., 0., -1.)]) * scale
    good = np.array([(0., 0., 0.), (0., 0., -1.), (0., 1., 0.)]) * scale
    for p, folded_ in ((fold, True), (good, False)):
        exempt, folded = lift._thin_curved(p, np.array([[0, 1, 2]]), [0.], [.01], *np.eye(3), 0., 1)
        assert not exempt[0] and bool(folded[0]) is folded_
        exempt, folded = lift._thin_curved(p, np.array([[0, 1, 2]]), [0.], [.01], *np.eye(3), .5, 1)
        assert not exempt[0] and bool(folded[0]) is folded_


def test_m1_codex_reproduction():
    p = np.array([(0., 0., 0.), (0., 1., 0.), (0., 0., -1e-304)])
    exempt, folded = lift._thin_curved(p, np.array([[0, 1, 2]]), [0.], [.01], *np.eye(3), 0., 1)
    assert not exempt[0] and folded[0]


# ----------------------------------------------------------------------------- M2
def test_m2_a_plane_at_1e_200_is_refused_out_of_domain_or_bounded():
    """Codex N3 r2 M2: the exact enclosure of the squared distance (1e-400) rounded to 0 in binary64
    and the face was published with deviation_bound 1e-323 for an exact deviation of 1e-200."""
    z = 1e-200
    g, f = T._planar([[(0, 0, z), (2, 0, z), (1, 1, z)]], tolerance=TolerancePolicy(surface_residual=1e-199))
    assert not g.validate_topology()
    code, mesh = _status(lambda: _mesh(g, f, dt.DisplayTolerance(.1)))
    if mesh is not None:
        assert mesh.deviation_bound >= z
    else:
        assert code == "OUT_OF_DOMAIN"


def test_m2_the_exact_plane_bound_is_rounded_up():
    """The exact plane decision returns a binary64 bound at or above the exact distance even when
    the distance squared is far below the normal range (1e-170: squared 1e-340)."""
    plane = lift._PlaneChart(1, Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    for z in (1e-170, 3e-160, 1e-30):
        got = plane.exact_bound(np.array([.25, .5, z]), np.array([.25, .5]), 1.)
        assert got is not None and got >= z and got <= math.nextafter(math.nextafter(z, 1.), 1.) * (1 + 1e-15)


def test_m2_tiny_nonzero_coordinates_are_out_of_domain():
    """The stated domain: every nonzero coordinate magnitude of the ring points and the support
    is at least 2**-200 (below it, products of differences leave the normal range)."""
    for tiny in (1e-70, 1e-200):
        g, f = T._planar([[(0, 0), (2, tiny), (1, 1)]])
        _refused("OUT_OF_DOMAIN", lambda: _mesh(g, f, dt.DisplayTolerance(.1)))
    g, f = T._planar([[(0, 0), (2, 1e-60), (1, 1)]])
    T._check_certified(g, f, dt.DisplayTolerance(.1), _mesh(g, f, dt.DisplayTolerance(.1)))


# ----------------------------------------------------------------------------- M3
def _strip(z):
    g = GeometryModel()
    s = Cylinder((0, 0, z), (0, 0, 1), (1, 0, 0), 1., 1.)
    v = [g.add_point(1., y, z + h) for y, h in [(0., 0.), (1e-8, 0.), (1e-8, 1.), (0., 1.)]]
    f = g.add_face([g.add_line(a, b) for a, b in zip(v, v[1:] + v[:1])], surface=s)
    assert not g.validate_topology()
    return g, f


def test_m3_the_azimuth_join_after_an_axial_translation_is_refused_or_certified():
    """Codex N3 r2 M3: the 1e-8 strip was accepted at z = 0 and refused at z = 2e6 ("two ring points
    coincide"): the E2 threshold used world reach (r2: support-frame magnitudes, kept).  Owner
    decision C2 (r3): translation invariance is not a requirement; accepted at the origin, and an
    accepted translated mesh is certified."""
    for z in (0., 2e6):
        g, f = _strip(z)
        code, mesh = _status(lambda: _mesh(g, f, dt.DisplayTolerance(.1)))
        assert mesh is not None or z != 0., code
        if mesh is not None:
            C._check_cylinder(g, f, dt.DisplayTolerance(.1), mesh)


def _translated_cases(axial, lateral):
    """Cylinder and cone faces whose model AND support move by ``axial`` along the axis plus
    ``lateral`` across it (the axis is z)."""
    o = (lateral, -.5 * lateral, axial)
    yield "quarter", C.quarter_cylinder(origin=o), dt.DisplayTolerance(corpus.INTERACTION_STEP), C._check_cylinder
    yield "window", C.window_face((o[0] + 3., o[1] - 1., o[2] + 2.), (0., 0., 1.), 1.9), dt.DisplayTolerance(.51), \
        C._check_cylinder
    yield "frustum", K.frustum_patch(r0=.5, origin=o), dt.DisplayTolerance(corpus.INTERACTION_STEP), K._check_cone
    yield "frustum_holes", K.frustum_patch(r0=.3, origin=o, holes=((.3, .6, .3, .7),)), \
        dt.DisplayTolerance(corpus.INTERACTION_STEP), K._check_cone


def test_m3_axial_and_lateral_translations_are_refused_or_certified():
    """Model and support translated along the axis (to 2e6) and across it (to 1e5) on cylinder and
    cone faces, and the tripod skins at shifts of 1e3 and 3e4.  Owner decision C2 (r3):
    translation invariance is not a requirement (a placement-dependent refusal takes the consumer's
    old route), so only "accepted -> the oracles hold" is asserted; at the origin all are accepted.
    Measured (docs, known limit): the tripod skins moved by 1e3 are all accepted, by 1e4 6 to 12 of
    the 18 are refused (N1 samples 3e-13 apart, folds), by 1e5 N1 itself refuses 11."""
    for axial, lateral in ((0., 0.), (2e6, 0.), (0., 1e5), (-3e5, 4e4)):
        for name, (g, face), tol, check in _translated_cases(axial, lateral):
            code, mesh = _status(lambda: _mesh(g, face, tol))
            assert mesh is not None or (axial, lateral) != (0., 0.), (name, code)
            if mesh is not None:
                check(g, face, tol, mesh, grid=C._grid(2))
    for shift in ((1e3, 0., 0.), (0., 0., 1e3), (0., 3e4, 0.), (3e4, 0., 0.)):
        g, faces = R1._shifted_tripod(shift)
        for face in faces:
            tol = dt.DisplayTolerance(corpus.INTERACTION_STEP)
            code, mesh = _status(lambda: _mesh(g, face, tol))
            if mesh is not None:
                C._check_cylinder(g, face, tol, mesh, grid=C._grid(2))


# ----------------------------------------------------------------------------- L1
def test_l1_a_vertex_exactly_t_face_off_a_far_plane_is_accepted():
    """Codex N3 r2 L1: at z = 2e6 a vertex one ulp (= T_face) off the plane was refused
    EDGE_UNSUPPORTED: the inverse-square-root enclosure never decides equality.  The squared
    comparison is exact: accepted, deviation_bound <= T_face."""
    z = 2e6
    d = math.ulp(z)
    g, f = T._planar([[(0, 0, z + d), (2, 0, z + d), (1, 1, z + d)]], Plane((0, 0, z), (1, 0, 0), (0, 1, 0)),
                     TolerancePolicy(surface_residual=d))
    mesh = T._check_certified(g, f, dt.DisplayTolerance(.1), _mesh(g, f, dt.DisplayTolerance(.1)))
    assert mesh.deviation_bound == d


def test_l1_the_exact_plane_decision_at_equality_on_a_skew_plane():
    """Equality and one ulp either side, on a plane whose frame is not axis-aligned (its exact
    frame is rational, so the oracle is exact): the decision is exact either way."""
    plane = lift._PlaneChart(1, Plane((0., 0., 0.), (3., 4., 0.), (0., 0., 1.)))
    # e1* = (.6, .8, 0), e2* = (0, 0, 1), n* = (.8, -.6, 0): the point c + t n* exactly
    point = np.array([.6 + .8 * .5, .8 - .6 * .5, .25])
    chart = plane.chart(point[None])[0]
    from fractions import Fraction as F
    w = [F(float(x)) for x in point]
    exact = abs(w[0] * F(4, 5) - w[1] * F(3, 5))                # |w . n*| (rational here)
    cx, cy = (F(float(c)) for c in chart)
    d2 = (cx - (w[0] * F(3, 5) + w[1] * F(4, 5))) ** 2 + (cy - w[2]) ** 2 + exact ** 2
    limit = math.sqrt(float(d2))
    for t in (limit, math.nextafter(limit, 1.), math.nextafter(limit, 0.)):
        got = plane.exact_bound(point, chart, t)
        assert (got is not None) == (F(t) ** 2 >= d2)
        if got is not None:
            assert F(got) ** 2 >= d2 and got <= t


# ----------------------------------------------------------------------------- L2
def _module_at(sha, name):
    old = types.ModuleType(f"anygeometry.{name}")
    old.__package__ = "anygeometry"
    exec(compile(subprocess.check_output(["git", "show", f"{sha}:src/anygeometry/_display_lift.py"], cwd=ROOT),
                 f"{sha}:_display_lift.py", "exec"), old.__dict__)
    return old


def test_l2_the_cylinder_oracle_catches_the_round_1_fold():
    """Codex N3 r2 L2: the oracle let ee0bc2c's exempt fold pass because its width fit within
    deviation_bound.  An exempt triangle passes only under the module's exact rule (a zero exact
    normal, or a footprint minimum within the stated margin of its normal's magnitude); since C3
    (r3) there is no exemption: the old module's published exempt triangle fails the oracle."""
    old = _module_at("ee0bc2c", "_display_lift_ee0bc2c")
    g, f = R1._codex_m1_face(1e6)
    tol = dt.DisplayTolerance(.1)
    m = old._lift(g, f, tol)
    assert m.degenerate_count == 1
    with pytest.raises(AssertionError, match="exemption rule"):
        C._check_cylinder(g, f, tol, m)


# ----------------------------------------------------------------------------- L3
def test_l3_budgets_before_the_seam_scan():
    """Codex N3 r2 L3: an over-budget face ran the full seam scan first.  The ring count and the
    coedge count (each coedge gives at least one ring point) are checked before it."""
    g, f = T._planar([[(0, 0), (8, 0), (8, 8), (0, 8)], [(1, 1), (2, 1), (2, 2), (1, 2)]])
    with patch.object(lift._slab, "MAX_SLAB_RINGS", 1), patch.object(lift, "_check_loops",
                                                                      wraps=lift._check_loops) as scan:
        assert _refused("OUT_OF_DOMAIN", lambda: dt.face_display_mesh(g, f, dt.DisplayTolerance(.1)))
        assert scan.call_count == 0
    with patch.object(lift._slab, "MAX_SLAB_RING_POINTS", 7), patch.object(lift, "_check_loops",
                                                                            wraps=lift._check_loops) as scan:
        text = _refused("OUT_OF_DOMAIN", lambda: dt.face_display_mesh(g, f, dt.DisplayTolerance(.1)))
        assert "ring points (MAX_SLAB_RING_POINTS = 7)" in text and scan.call_count == 0
