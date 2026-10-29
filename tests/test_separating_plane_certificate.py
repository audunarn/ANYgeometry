"""T2-1a: exact separating-plane certificate (ADR-T2 v2 section 4, tests of section 8.6 / 4.2c).

Safety tests are paired with a test double of the unsound rule they guard: the check must
pass for the module and fail for the double (ADR 8.6). Soundness assertions hold for every
outcome; capability assertions are made only for fixed inputs of small proof cost.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
import math
import pathlib
import random
import types

import mpmath
import numpy as np
import pytest

from anygeometry import separating_plane_certificate as sp
from anygeometry.arrangement_geometry import LinePath
from anygeometry.exact_curves import EllipticArc

mpmath.mp.prec = 320
TOL = 1e-9
R_MAX = Fraction(sp.contract_r_max(TOL))


@dataclass(frozen=True)
class _RawArc:          # exact (Fraction) or unchecked stored inputs, read by the module like EllipticArc
    center: tuple
    u_vector: tuple
    v_vector: tuple
    start_angle: object
    sweep_angle: object


@dataclass(frozen=True)
class _RawLine:
    start: tuple
    end: tuple


def mp(x):
    x = Fraction(x)
    return mpmath.mpf(x.numerator)/x.denominator


# ----------------------------------------------------------------------------------- builders

def plate(fid, pts, vids, eids, support=None, curves=None, positions=None, edges=None):
    """Planar polygon face; curves[i] overrides the path from pts[i] to pts[i+1];
    edges[i] = (model edge curve, forward) names the shared edge record."""
    n = len(pts)
    uses = []
    for i in range(n):
        c = curves[i] if curves and curves[i] is not None else LinePath(tuple(pts[i]), tuple(pts[(i + 1) % n]))
        edge_curve, forward = (edges or {}).get(i, (None, True))
        uses.append(sp.BoundaryUse(c, eids[i], vids[i], vids[(i + 1) % n], forward, edge_curve))
    if support is None:
        o = pts[0]
        u = tuple(pts[1][j] - o[j] for j in range(3))
        v = tuple(pts[-1][j] - o[j] for j in range(3))
        support = sp.PlaneSupport(tuple(o), u, v)
    positions = positions or {}
    verts = tuple((vid, tuple(positions.get(vid, p))) for vid, p in zip(vids, pts))
    return sp.FaceRecord(fid, support, (tuple(uses),), verts)


def L_plates(angle=math.pi/2, *, vertex_shift=0.0, curve_end_shift=0.0):
    """Two plates sharing edge 10 (V1 -> V2 along x); plate B hinged at angle ``angle``."""
    c, s = math.cos(angle), math.sin(angle)
    V1, V2 = (0., 0., 0.), (1., 0., 0.)
    shared = LinePath(V1, V2)
    # The stored vertex 1 moves along the shared edge's line: it stays in both supports, so only the
    # incidence defect delta_V (not delta_S) measures the displacement.
    A = plate(1, [V1, V2, (1., 1., 0.), (0., 1., 0.)], [1, 2, 3, 4], [10, 11, 12, 13],
              positions={1: (-vertex_shift, 0., 0.)},
              curves=[shared, None, None, LinePath((0., 1., 0.), (0., curve_end_shift, 0.))],
              edges={0: (shared, True)})
    B = plate(2, [V2, V1, (0., c, s), (1., c, s)], [2, 1, 5, 6], [10, 14, 15, 16],
              positions={1: (-vertex_shift, 0., 0.)},
              curves=[LinePath(V2, V1), LinePath((0., curve_end_shift*c, curve_end_shift*s), (0., c, s)), None, None],
              edges={0: (shared, False)})
    return A, B


def cylinder_strip(fid, theta0, theta1, z0, z1, vids, eids, *, R=1.0, support_radius=None, top_tilt=0.0,
                   top_offset=0.0, edges=None):
    """Strip on the cylinder x^2 + y^2 = R^2 (axis z), loop: bottom arc, generator, top arc back, generator.
    The generator at theta0 is stored downwards (d -> a); edges[i] = (model edge curve, forward)."""
    support = sp.CylinderSupport((0., 0., 0.), (0., 0., 1.), support_radius or R, (1., 0., 0.))
    bottom = EllipticArc((0., 0., z0), (R, 0., 0.), (0., R, 0.), theta0, theta1 - theta0)
    top = EllipticArc((top_offset, 0., z1), (R, 0., top_tilt), (0., R, 0.), theta1, theta0 - theta1)
    p = lambda t, z: (R*math.cos(t), R*math.sin(t), z)
    a, b, cc, d = p(theta0, z0), p(theta1, z0), p(theta1, z1), p(theta0, z1)
    curves = (bottom, LinePath(b, cc), top, LinePath(d, a))
    ends = ((0, 1), (1, 2), (2, 3), (3, 0))
    uses = []
    for i, curve in enumerate(curves):
        edge_curve, forward = (edges or {}).get(i, (None, True))
        uses.append(sp.BoundaryUse(curve, eids[i], vids[ends[i][0]], vids[ends[i][1]], forward, edge_curve))
    uses = tuple(uses)
    verts = tuple(zip(vids, (a, b, cc, d)))
    return sp.FaceRecord(fid, support, (uses,), verts)


# ----------------------------------------------------------------------------------- numerics

def test_pi_and_sincos_enclose_high_precision_references():
    lo, hi = sp.PI_ENCLOSURE.lo, sp.PI_ENCLOSURE.hi
    with mpmath.workprec(900):
        assert mp(lo) < +mpmath.pi < mp(hi) and hi - lo < Fraction(1, 1 << 600)
    rng = random.Random(7)
    xs = [0.0, 1.0, -1.0, 1.5, 20*math.pi + 4*math.pi/3, 1e6, -1e6 + .3, 1e15, 2.0**80, 2.0**200 + 2.0**148,
          math.pi/2, -math.pi, 7*math.pi/4]
    xs += [rng.uniform(-1e4, 1e4) for _ in range(200)]
    for x in xs:
        (sl, sh), (cl, ch) = sp.certified_sincos(Fraction(x))
        X = mp(x)
        assert mp(sl) <= mpmath.sin(X) <= mp(sh), x
        assert mp(cl) <= mpmath.cos(X) <= mp(ch), x
        assert sh - sl < Fraction(1, 1 << 100) and ch - cl < Fraction(1, 1 << 100)


def test_sincos_agrees_with_existing_proof_arithmetic():
    from anygeometry.cylinder_charts import _Proof
    proof = _Proof(sp.SPCertificatePolicy(), None)
    for x in (Fraction(0), Fraction(1, 3), Fraction(-5, 2), Fraction(20)):
        s, c = proof.sincos(x)
        (sl, sh), (cl, ch) = sp.certified_sincos(x)
        assert max(sl, s.lo) <= min(sh, s.hi) and max(cl, c.lo) <= min(ch, c.hi)


def test_trig_budget_exhaustion_at_huge_argument_is_a_refusal():
    with pytest.raises(sp.SPRefusal):
        sp.certified_sincos(Fraction(2.0**500))
    with pytest.raises(sp.SPRefusal):
        sp.first_harmonic_range((0., 1., 0.), 2.0**500, 1.0)


# --------------------------------------------------------------- 4.2c: first harmonic, degree two

def _reference_extrema(a, b, c, s0, sweep):
    """High-precision extrema of a + b cos + c sin on the STORED interval (exact float inputs)."""
    a, b, c = mp(a), mp(b), mp(c)
    lo_, hi_ = sorted((mp(s0), mp(s0) + mp(sweep)))
    values = [a + b*mpmath.cos(t) + c*mpmath.sin(t) for t in (lo_, hi_)]
    psi = mpmath.atan2(c, b)
    amp = mpmath.sqrt(b*b + c*c)
    k0 = int(mpmath.floor((lo_ - psi)/mpmath.pi)) - 1
    for k in range(k0, k0 + 6):
        t = psi + k*mpmath.pi
        if lo_ <= t <= hi_:
            values.append(a + (amp if k % 2 == 0 else -amp))
    return min(values), max(values)


def _proto_j_window_range(a, b, c, s0, sweep):
    """Test double: the v1 prototype's float routine (critical points only for j = -4..4)."""
    va, vb = a + b*math.cos(s0) + c*math.sin(s0), a + b*math.cos(s0 + sweep) + c*math.sin(s0 + sweep)
    lo_, hi_ = min(s0, s0 + sweep), max(s0, s0 + sweep)
    out = [va, vb]
    phi0 = math.atan2(c, b)
    for j in range(-4, 5):
        for phi, sign in ((phi0 + j*math.tau, 1), (phi0 + math.pi + j*math.tau, -1)):
            if lo_ < phi < hi_:
                out.append(a + sign*math.hypot(b, c))
    return min(out), max(out)


def _codex_arc_check(range_routine):
    """Codex arc: stored start fl(20 pi + 4 pi/3), sweep 4 pi/3, v = cos: the interior +1 must be enclosed
    and the 'negative side' classification refused."""
    s0, sweep = 20*math.pi + 4*math.pi/3, 4*math.pi/3
    lo, hi = range_routine((0., 1., 0.), s0, sweep)
    ref_lo, ref_hi = _reference_extrema(0., 1., 0., s0, sweep)
    return mp(lo) <= ref_lo and ref_hi <= mp(hi) and hi > 0


def test_codex_arc_direct_extrema_and_double():
    assert _codex_arc_check(sp.first_harmonic_range)
    assert not _codex_arc_check(lambda coeff, s0, sw: _proto_j_window_range(*coeff, s0, sw))
    # Endpoint-only enclosure (another unsound double) misses the interior maximum as well.
    assert not _codex_arc_check(lambda coeff, s0, sw: tuple(sorted(
        (math.cos(s0), math.cos(s0 + sw)))))


@pytest.mark.parametrize("k", [0, 1, -1, 10, -10, 1000, -1000, 10**6, -10**6])
def test_direct_extrema_at_large_offsets(k):
    s0 = 2*math.pi*k + 0.3
    sweep = 1.25
    psi = 2*math.pi*k + 0.9                 # interior maximum about 0.6 rad after the start
    b, c, a = math.cos(psi), math.sin(psi), -0.25
    try:
        lo, hi = sp.first_harmonic_range((a, b, c), s0, sweep)
    except sp.SPRefusal:
        return                              # UNDECIDED is acceptable; a wrong enclosure is not
    ref_lo, ref_hi = _reference_extrema(a, b, c, s0, sweep)
    assert mp(lo) <= ref_lo and ref_hi <= mp(hi)
    assert hi - lo < 4                      # an enclosure, not a vacuous one


@pytest.mark.parametrize("k", [1, -1, 10, 1000, -10**6])
def test_rounded_shift_with_phase_accounting(k):
    s0, sweep, coeff = 0.7, 2.0, (0.1, 0.6, -0.8)
    shifted = s0 + 2*math.pi*k              # a DIFFERENT stored curve: phase delta enclosed below
    outcomes = []
    for start in (s0, shifted):
        try:
            outcomes.append(sp.first_harmonic_range(coeff, start, sweep))
        except sp.SPRefusal:
            outcomes.append(None)
    for start, out in zip((s0, shifted), outcomes):
        if out is not None:
            ref_lo, ref_hi = _reference_extrema(*coeff, start, sweep)
            assert mp(out[0]) <= ref_lo and ref_hi <= mp(out[1])
    if None in outcomes:
        return
    delta = abs(mp(shifted) - mp(s0) - 2*mpmath.pi*k)
    amp = mpmath.sqrt(mp(coeff[1])**2 + mp(coeff[2])**2)
    widths = sum(mp(o[1]) - mp(o[0]) for o in outcomes)
    assert abs(mp(outcomes[0][0]) - mp(outcomes[1][0])) <= amp*delta + widths
    assert abs(mp(outcomes[0][1]) - mp(outcomes[1][1])) <= amp*delta + widths


@pytest.mark.parametrize("sweep", [-1.0, -4.0, 2*math.pi, -2*math.pi, 2*math.pi + 16*2.0**-52, 6.0])
def test_negative_sweeps_and_full_turns(sweep):
    coeff = (0.2, -0.3, 0.9)
    lo, hi = sp.first_harmonic_range(coeff, 1.1, sweep)
    ref_lo, ref_hi = _reference_extrema(*coeff, 1.1, sweep)
    assert mp(lo) <= ref_lo and ref_hi <= mp(hi)


def test_randomized_first_harmonic_ranges_enclose_references():
    """Soundness on many inputs, including ends placed on (rounded) critical points."""
    rng = random.Random(11)
    for i in range(300):
        a, b, c = (rng.uniform(-2, 2) for _ in range(3))
        psi = math.atan2(c, b) + rng.choice((0, math.pi, -math.pi, 2*math.pi))
        start = rng.choice((psi, psi - rng.uniform(0, 3), rng.uniform(-50, 50)))
        sweep = rng.choice((1, -1))*rng.choice((rng.uniform(1e-6, 7.0), abs(psi - start) or 0.5, 2*math.pi))
        lo, hi = sp.first_harmonic_range((a, b, c), start, sweep)
        ref_lo, ref_hi = _reference_extrema(a, b, c, start, sweep)
        assert mp(lo) <= ref_lo and ref_hi <= mp(hi), i
        assert hi - lo <= 2*math.hypot(b, c) + 1e-9


def test_first_harmonic_routine_rejects_degree_two_input():
    with pytest.raises(ValueError):
        sp.first_harmonic_range((0, 1, 0, 1, 0), 1.7, 1.55)


def _endpoint_sign_degree_two_double(f, df, a, b):
    """Test double: the 4.2a endpoint-derivative rule misapplied to degree two."""
    if df(a) < 0 and df(b) < 0:
        return f(b), f(a)                   # 'monotone decreasing'
    return min(f(a), f(b)), max(f(a), f(b))


def test_degree_two_cos_plus_cos2_enclosed_and_double():
    lo, hi = sp.degree_two_bound((0, 1, 0, 1, 0))
    assert lo <= Fraction(-9, 8) and hi >= 0               # true range [-1.125, 0] on [1.7, 3.25]
    f = lambda t: math.cos(t) + math.cos(2*t)
    df = lambda t: -math.sin(t) - 2*math.sin(2*t)
    assert df(1.7) < 0 and df(3.25) < 0
    dlo, dhi = _endpoint_sign_degree_two_double(f, df, 1.7, 3.25)
    assert not (dlo <= -1.125 and dhi >= 0)


def _face(record):
    return sp._Face(record, sp.SPCertificatePolicy(), None)


def test_coaxial_arc_on_cylinder_has_small_certified_defect():
    f = _face(cylinder_strip(1, 0.1, 0.9, 0.0, 1.0, [1, 2, 3, 4], [10, 11, 12, 13], R=1.7))
    arcs = [u for u in f.uses if u.geom.kind == "arc"]
    assert all(u.dS < Fraction(1, 10**14) for u in arcs)


def test_non_coaxial_arc_bound_exceeds_budget_and_pair_is_undecided():
    # A planar tilted section stays on the cylinder (delta_S = 0); an off-axis centre does not.
    flat = _face(cylinder_strip(1, 0.1, 0.9, 0.0, 1.0, [1, 2, 3, 4], [10, 11, 12, 13], top_tilt=1e-3))
    assert all(u.dS < Fraction(1, 10**14) for u in flat.uses if u.geom.kind == 'arc')
    tilted = cylinder_strip(1, 0.1, 0.9, 0.0, 1.0, [1, 2, 3, 4], [10, 11, 12, 13], top_offset=1e-6)
    f = _face(tilted)
    top = [u for u in f.uses if u.geom.kind == "arc"][1]
    assert top.dS > R_MAX                                 # the 4.2b bound exceeds the budget
    other = cylinder_strip(2, 0.9, 1.5, 0.0, 1.0, [2, 5, 6, 3], [14, 15, 16, 11])
    assert not sp.certify_face_pair(tilted, other, TOL).certified


# ------------------------------------------------------------------- closure / incidence defects

def _loop_defect_double(monkeypatch):
    """Test double: exact incidence assumed (curve-end-to-vertex defects ignored)."""
    monkeypatch.setattr(sp._Ar, "norm_hi", lambda self, lows, highs, centre=None: Fraction(0))


@pytest.mark.parametrize("delta", [3e-14, 2e-13])
def test_vertex_displaced_by_delta_raises_r_by_at_least_delta(delta, monkeypatch):
    exact = sp.certify_face_pair(*L_plates(), TOL)
    assert exact.certified and exact.r_pair == 0.0
    moved = sp.certify_face_pair(*L_plates(vertex_shift=delta), TOL)
    assert (not moved.certified) or moved.r_pair >= delta
    if delta < 1e-13:
        assert moved.certified                             # capability: small defects still certify
    _loop_defect_double(monkeypatch)
    double = sp.certify_face_pair(*L_plates(vertex_shift=delta), TOL)
    assert double.certified and double.r_pair < delta      # the double would fail the check


def test_large_vertex_defect_refuses_beyond_r_max():
    res = sp.certify_face_pair(*L_plates(vertex_shift=1e-10), TOL)
    assert not res.certified


def test_curve_end_missing_its_vertex_uses_connectors():
    res = sp.certify_face_pair(*L_plates(curve_end_shift=5e-14), TOL)
    assert res.certified and res.r_pair >= 5e-14


def test_full_turn_face_is_undecided():
    support = sp.CylinderSupport((0., 0., 0.), (0., 0., 1.), 1.0, (1., 0., 0.))
    bottom = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0.0, 2*math.pi)
    top = EllipticArc((0., 0., 1.), (1., 0., 0.), (0., 1., 0.), 0.0, -2*math.pi)
    annulus = sp.FaceRecord(1, support, ((sp.BoundaryUse(bottom, 10, 1, 1),), (sp.BoundaryUse(top, 11, 2, 2),)),
                            ((1, (1., 0., 0.)), (2, (1., 0., 1.))))
    with pytest.raises(sp.SPRefusal, match="cylinder_loop_winds"):
        _face(annulus)
    cap = plate(2, [(1., 0., 0.), (2., 0., 0.), (2., 1., 0.)], [1, 3, 4], [20, 21, 22])
    res = sp.certify_face_pair(annulus, cap, TOL)
    assert not res.certified and "cylinder_loop_winds" in res.reason


# --------------------------------------------------------------- step 6(iii) separation, re-entry

RHO = Fraction(2.5e-13)


def _reentry_arc(sweep):
    return _RawArc((0, 1, 0), (0, -1, 0), (1, 0, 0), 0, sweep)


def _segment_through(target_xy, length=Fraction(5, 10**6)):
    """Segment from the origin, direction (-1, s) scaled to |x| = length, passing at a chosen point."""
    x, y = target_xy
    return _RawLine((0, 0, 0), (-length, -length*y/x, 0))


def _distance_point_segment(p, a, b):
    ap = [mp(p[i]) - mp(a[i]) for i in range(3)]
    ab = [mp(b[i]) - mp(a[i]) for i in range(3)]
    t = max(0, min(1, sum(x*y for x, y in zip(ap, ab))/sum(x*x for x in ab)))
    return mpmath.sqrt(sum((ap[i] - t*ab[i])**2 for i in range(3)))


def _undecided(fn, *args):
    try:
        fn(*args)
    except sp.SPRefusal:
        return True
    return False


def _ball_restricted_s1_double(c1, c2, V, rho1, rho2, R0=1e-5):
    """Test double: r1's rule, tangents enclosed only on c ∩ B(V, R0), no (S3); accepts the re-entry case."""
    def tangents(curve):
        out = []
        if isinstance(curve, _RawArc):
            sweep = float(curve.sweep_angle)
            for t in [sweep*i/20000 for i in range(20001)]:
                p = [float(curve.center[j]) + float(curve.u_vector[j])*math.cos(t) +
                     float(curve.v_vector[j])*math.sin(t) for j in range(3)]
                if math.dist(p, V) <= R0:
                    out.append([-float(curve.u_vector[j])*math.sin(t) + float(curve.v_vector[j])*math.cos(t)
                                for j in range(3)])
        else:
            out.append([float(curve.end[j]) - float(curve.start[j]) for j in range(3)])
        return out
    t1, t2 = tangents(c1), tangents(c2)
    worst = min(sum(a*b for a, b in zip(x, y))/(math.hypot(*x)*math.hypot(*y)) for x in t1 for y in t2)
    if worst > -0.5:      # the double still refuses when the end tangents are not well separated
        raise sp.SPRefusal("double:angle")
    return min(rho1, rho2) + (rho1 + rho2)


def _reentry_math_case():
    """Exact rational stored input within 2^-250 of 2 pi - 1e-6 (the mathematical case)."""
    value = 2*mpmath.pi - mpmath.mpf(10)**-6
    return Fraction(mpmath.nstr(value, 90))


def test_reentry_regression_mathematical_case_and_double():
    sweep = _reentry_math_case()
    end = (mpmath.sin(mp(sweep)), 1 - mpmath.cos(mp(sweep)), 0)
    seg = _RawLine((0, 0, 0), (Fraction(-5, 10**6), 0, 0))
    assert _distance_point_segment(end, seg.start, seg.end) < 2*mp(RHO)   # precondition: returns within rho12
    assert _undecided(sp.vertex_separation_bound, _reentry_arc(sweep), seg, (0, 0, 0), RHO, RHO)
    assert not _undecided(_ball_restricted_s1_double, _reentry_arc(sweep), seg, (0., 0., 0.), RHO, RHO)


def test_reentry_module_double_without_s3_accepts(monkeypatch):
    sweep = _reentry_math_case()
    seg = _RawLine((0, 0, 0), (Fraction(-5, 10**6), 0, 0))
    monkeypatch.setattr(sp, "_s3_rest_apart", lambda *args: True)
    assert not _undecided(sp.vertex_separation_bound, _reentry_arc(sweep), seg, (0, 0, 0), RHO, RHO)


def _binary64_return_point():
    sweep = 2*math.pi - 1e-6
    assert sweep == 6.283184307179586
    return sweep, (mpmath.sin(mp(sweep)), 1 - mpmath.cos(mp(sweep)))


@pytest.mark.parametrize("factor", [Fraction(0), Fraction(1, 10)])
def test_reentry_binary64_margin_variants_undecided(factor):
    sweep, (ex, ey) = _binary64_return_point()
    rho12 = 2*RHO
    # Line through the origin passing at distance factor*rho12 from the returning end (exact rationals).
    target_y = ey - mp(factor*rho12)
    slope = Fraction(mpmath.nstr(target_y/ex, 80))
    seg = _RawLine((0, 0, 0), (Fraction(-5, 10**6), Fraction(-5, 10**6)*slope, 0))
    d = _distance_point_segment((ex, ey, 0), seg.start, seg.end)
    assert d <= mp(factor*rho12)*mpmath.mpf(1.01) + mpmath.mpf(10)**-40
    assert _undecided(sp.vertex_separation_bound, _reentry_arc(sweep), seg, (0, 0, 0), RHO, RHO)


def test_reentry_binary64_far_variant_may_certify_with_formula_bound():
    sweep, (ex, ey) = _binary64_return_point()
    rho12 = 2*RHO
    target_y = ey + mp(10*rho12)
    slope = Fraction(mpmath.nstr(target_y/ex, 80))
    seg = _RawLine((0, 0, 0), (Fraction(-5, 10**6), Fraction(-5, 10**6)*slope, 0))
    assert _distance_point_segment((ex, ey, 0), seg.start, seg.end) > 9*mp(rho12)
    try:
        r = sp.vertex_separation_bound(_reentry_arc(sweep), seg, (0, 0, 0), RHO, RHO)
    except sp.SPRefusal:
        return
    # theta_V is capped at pi/2 here (the tangents are nearly opposite): r_V = rho + rho12 + O(delta_V).
    assert RHO + rho12 <= r <= RHO + rho12 + Fraction(1, 10**20)


def test_return_inside_small_ball_is_undecided():
    # A tiny circle (radius 1e-6) that leaves V along +x and returns within rho12 of c2.
    r, eta = Fraction(1, 10**6), mpmath.mpf(10)**-3
    sweep = Fraction(mpmath.nstr(2*mpmath.pi - eta, 90))
    arc = _RawArc((0, r, 0), (0, -r, 0), (r, 0, 0), 0, sweep)
    seg = _RawLine((0, 0, 0), (Fraction(-5, 10**6), 0, 0))
    end = (mp(r)*mpmath.sin(mp(sweep)), mp(r)*(1 - mpmath.cos(mp(sweep))), 0)
    assert _distance_point_segment(end, seg.start, seg.end) < 2*mp(RHO)
    assert _undecided(sp.vertex_separation_bound, arc, seg, (0, 0, 0), RHO, RHO)


# --------------------------------------------------------------------------- end-local sign (b)

def _fn(pieces, sgn=1, speed=Fraction(1)):
    f = sp._Fn()
    f.pieces, f.sgn, f.speed_hi = pieces, sgn, speed
    f.lo, f.hi = min(p[0] for p in pieces), max(p[1] for p in pieces)
    return f


def _r1_end_local_double(ctx, f, eps_c, s, smin):
    """Test double: r1's rule, |dv/ds| >= s_c only, and no nonempty rest between two final pieces."""
    N = len(f.pieces)
    S = [i for i, p in enumerate(f.pieces) if sp._flip(p[0], p[1], s)[1] >= -eps_c]
    if not S:
        return None
    absolute = [min(abs(p[2]), abs(p[3]))/f.speed_hi if p[2]*p[3] > 0 else 0 for p in f.pieces]
    for k0 in range(N + 1):
        if all(a >= smin for a in absolute):
            return {"start": min(absolute), "end": min(absolute)}
    return None


def test_end_local_requires_sign_towards_the_end():
    eps = Fraction(1, 10**14)
    # v rises from -1 to 0 (interior maximum in the band) and falls back to -1: two pieces,
    # both decreasing towards their own ends.
    two = _fn([(Fraction(-1), Fraction(0), Fraction(1), Fraction(2)),
               (Fraction(-1), Fraction(0), Fraction(-2), Fraction(-1))])
    assert sp._end_local(None, two, eps, 1, sp.S_MIN) is None          # class (a)
    assert _r1_end_local_double(None, two, eps, 1, sp.S_MIN) is not None
    # A single final piece increasing towards E covering a line: class (b).
    line = _fn([(Fraction(-1), Fraction(0), Fraction(1), Fraction(1))])
    out = sp._end_local(None, line, eps, 1, sp.S_MIN)
    assert out == {"start": None, "end": Fraction(1)}


def test_end_local_two_pieces_need_a_nonempty_rest():
    eps = Fraction(1, 10**14)
    pieces = [(Fraction(-1), Fraction(0), Fraction(-2), Fraction(-1)),      # rises towards the start
              (Fraction(-1), Fraction(0), Fraction(1), Fraction(2))]        # rises towards the end
    assert sp._end_local(None, _fn(pieces), eps, 1, sp.S_MIN) is None
    rest = [pieces[0], (Fraction(-3), Fraction(-2), Fraction(-1), Fraction(1)), pieces[1]]
    out = sp._end_local(None, _fn(rest), eps, 1, sp.S_MIN)
    assert out == {"start": Fraction(1), "end": Fraction(1)}


# ----------------------------------------------------------------------------- degenerate supports

def test_plane_face_in_P_with_cylinder_partner_reaches_the_branch():
    strip = cylinder_strip(1, 0.2, 0.8, 0.0, 1.0, [1, 2, 3, 4], [10, 11, 12, 13])
    a = (math.cos(0.2), math.sin(0.2), 0.)
    b = (math.cos(0.8), math.sin(0.8), 0.)
    arc = strip.loops[0][0].curve
    cap = plate(2, [b, a, (0., 0., 0.)], [2, 1, 9], [10, 20, 21],
                curves=[EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0.8, -0.6), None, None],
                support=sp.PlaneSupport((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)), edges={0: (arc, False)})
    res = sp.certify_face_pair(strip, cap, TOL)
    assert res.certified, res.reason
    assert res.evidence["case"] == "plane_support_branch" and res.evidence["plane"][0] == "support"


def test_two_plane_faces_in_P_are_undecided():
    A = plate(1, [(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)], [1, 2, 3, 4], [10, 11, 12, 13])
    B = plate(2, [(1., 0., 0.), (0., 0., 0.), (0., -1., 0.), (1., -1., 0.)], [2, 1, 5, 6], [10, 14, 15, 16],
              edges={0: (A.loops[0][0].curve, False)})
    res = sp.certify_face_pair(A, B, TOL)
    assert not res.certified and res.reason == "coplanar_supports"


def test_plane_face_parallel_to_P_at_offset_is_undecided():
    A = plate(1, [(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)], [1, 2, 3, 4], [10, 11, 12, 13])
    B = plate(2, [(1., 0., 0.), (0., 0., 0.), (0., -1., 0.), (1., -1., 0.)], [2, 1, 5, 6], [10, 14, 15, 16],
              support=sp.PlaneSupport((0., 0., 1e-15), (1., 0., 0.), (0., -1., 0.)),
              edges={0: (A.loops[0][0].curve, False)})
    res = sp.certify_face_pair(A, B, TOL)
    assert not res.certified
    assert any("plane_parallel_to_P" in a for a in res.evidence.get("attempts", ()))


def test_plate_tangent_to_cylinder_along_shared_generator():
    """P = the plate's support is tangent along the generator: step 4 must refuse it. The pair may still
    be certified through the axial plane of the shared generator (a separating plane of ADR 4.3 step 1),
    and then the bound must hold for the exact geometry (the faces meet only on the generator)."""
    strip = cylinder_strip(1, 0.0, 0.5, 0.0, 1.0, [1, 2, 3, 4], [10, 11, 12, 13])
    plate_rec = plate(2, [(1., 0., 1.), (1., 0., 0.), (1., -1., 0.), (1., -1., 1.)], [4, 1, 5, 6],
                      [13, 20, 21, 22], support=sp.PlaneSupport((1., 0., 0.), (0., -1., 0.), (0., 0., 1.)),
                      edges={0: (strip.loops[0][3].curve, True)})
    certifier = sp.SPCertifier()
    F1, F2 = certifier._face(strip, None), certifier._face(plate_rec, None)
    ctx = sp._PairCtx()
    ctx.ar, ctx.policy, ctx.fns, ctx.F1, ctx.F2 = sp._Ar(certifier.policy, None), certifier.policy, {}, F1, F2
    ctx.SE, ctx.SV = {13}, {1, 4}
    ctx.SVE, ctx.TV, ctx.r_max, ctx.threshold_r = {1, 4}, set(), R_MAX, None
    n, k = sp._normalise(F2.nF, F2.kF)
    for sign in (1, -1):
        with pytest.raises(sp.SPRefusal, match="gradient|sides|branch"):
            sp._attempt(ctx, tuple(sign*x for x in n), sign*k, ("support", 2))
    res = certifier.certify(strip, plate_rec, TOL)
    if res.certified:
        assert res.evidence["plane"][0] == "axial"


# -------------------------------------------------------------------------------- numeric window

def _window_double(monkeypatch):
    original = sp._exact

    def no_window(value, what):
        if isinstance(value, float) and math.isfinite(value):
            return Fraction(value)
        return original(value, what)
    monkeypatch.setattr(sp, "_exact", no_window)


@pytest.mark.parametrize("bad", [5e-324, 2.0**-1000, 2.0**1000])
def test_numeric_window_stored_coefficients(bad, monkeypatch):
    A, B = L_plates()
    pts = [(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (bad, 1., 0.)]
    A = plate(1, pts, [1, 2, 3, 4], [10, 11, 12, 13],
              support=sp.PlaneSupport((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    res = sp.certify_face_pair(A, B, TOL)
    assert not res.certified and "numeric_window" in res.reason
    if bad < 1:                       # the double (no window) certifies the tiny-coefficient case
        _window_double(monkeypatch)
        assert sp.certify_face_pair(A, B, TOL).certified


def test_numeric_window_tolerance_and_nonfinite_angle():
    A, B = L_plates()
    res = sp.certify_face_pair(A, B, 1e-306)            # tol itself below 2^-960 (tol/1000 subnormal)
    assert not res.certified and "numeric_window:tolerance" in res.reason
    res = sp.certify_face_pair(A, B, 1e-288)            # tol in the window, tol/1000 below it
    assert not res.certified and "numeric_window:r_max" in res.reason
    strip = cylinder_strip(1, 0.2, 0.8, 0.0, 1.0, [1, 2, 3, 4], [10, 11, 12, 13])
    loop = list(strip.loops[0])
    loop[0] = sp.BoundaryUse(_RawArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), float("nan"), 0.6),
                             10, 1, 2)
    broken = sp.FaceRecord(1, strip.support, (tuple(loop),), strip.vertices)
    other = cylinder_strip(2, 0.8, 1.4, 0.0, 1.0, [2, 5, 6, 3], [14, 15, 16, 11])
    res = sp.certify_face_pair(broken, other, TOL)
    assert not res.certified and "nonfinite" in res.reason


def test_delta_eval_is_reported_and_window_guarded():
    res = sp.certify_face_pair(*_mitre(), TOL)
    assert res.certified and 0 < res.evidence["delta_eval_max"] < 1e-13
    # Inputs inside the window whose difference (an intermediate of evaluate) is below it.
    s = 2.0**-940
    e = s + 2.0**-992
    assert e != s
    with pytest.raises(sp.SPRefusal, match="numeric_window:delta_eval"):
        sp._delta_eval(LinePath((s, 1., 0.), (e, 1., 0.)), sp.SPCertificatePolicy())


def test_r_max_is_exact_tol_over_1000_rounded_down():
    rng = random.Random(3)
    differs = 0
    for _ in range(2000):
        tol = rng.uniform(1e-12, 1e-3)
        r = sp.contract_r_max(tol)
        exact = Fraction(tol)/1000
        assert Fraction(r) <= exact < Fraction(math.nextafter(r, math.inf))
        differs += r != 1e-3*tol            # the float product (a double of the rule) differs
    assert differs > 0


# -------------------------------------------------------------- 8.6 named cases and outcomes

def test_t_joint_is_refused():
    A = plate(1, [(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.), (0., .5, 0.)], [1, 2, 3, 4, 7],
              [10, 11, 12, 13, 17])
    B = plate(2, [(0., .5, 0.), (.5, .5, 0.), (.5, .5, 1.), (0., .5, 1.)], [7, 8, 9, 6], [20, 21, 22, 23])
    assert not sp.certify_face_pair(A, B, TOL).certified


def test_fillet_grazing_a_plate_along_an_interior_generator_is_refused():
    """A cylinder face (axis y, radius 0.3) tangent to plate z = 0 along a whole generator; the faces
    share only the vertex where that generator meets the plate's boundary: the true intersection is the
    generator (length 1), so the pair must not certify."""
    R, cx, cz = 0.3, 0.5, 0.3
    p = lambda th, y: (cx + R*math.sin(th), y, cz - R*math.cos(th))
    arc = lambda y, th0, sw: EllipticArc((cx, y, cz), (0., 0., -R), (R, 0., 0.), th0, sw)
    support = sp.CylinderSupport((cx, 0., cz), (0., 1., 0.), R, (0., 0., -1.))
    loop = (sp.BoundaryUse(arc(0., -0.5, 0.5), 30, 11, 12), sp.BoundaryUse(arc(0., 0.0, 0.5), 31, 12, 13),
            sp.BoundaryUse(LinePath(p(.5, 0.), p(.5, 1.)), 32, 13, 14),
            sp.BoundaryUse(arc(1., 0.5, -1.0), 33, 14, 15),
            sp.BoundaryUse(LinePath(p(-.5, 1.), p(-.5, 0.)), 34, 15, 11))
    fillet = sp.FaceRecord(1, support, (loop,), ((11, p(-.5, 0.)), (12, p(0., 0.)), (13, p(.5, 0.)),
                                                 (14, p(.5, 1.)), (15, p(-.5, 1.))))
    A = plate(2, [(0., 0., 0.), p(0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)], [1, 12, 2, 3, 4],
              [40, 41, 42, 43, 44], support=sp.PlaneSupport((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))
    assert not sp.certify_face_pair(fillet, A, TOL).certified


def test_coincident_cylinder_strips_need_opposite_traversal():
    a = cylinder_strip(1, 0.0, 0.5, 0.0, 1.0, [1, 2, 3, 4], [10, 11, 12, 13])
    generator = a.loops[0][1].curve
    b = cylinder_strip(2, 0.5, 1.0, 0.0, 1.0, [2, 5, 6, 3], [14, 15, 16, 11], edges={3: (generator, False)})
    res = sp.certify_face_pair(a, b, TOL)
    assert res.certified, res.reason
    assert res.evidence["coincident_support"] and res.evidence["shadow_required"] == "_coplanar_traces"
    same = cylinder_strip(2, 0.5, 1.0, 0.0, 1.0, [2, 5, 6, 3], [14, 15, 16, 11], edges={3: (generator, True)})
    res = sp.certify_face_pair(a, same, TOL)
    assert not res.certified and "coincident_support_ineligible" in res.reason


def _mitre(beta=0.35, half=0.3):
    t = math.tan(beta)
    n = (-math.sin(beta), 0., math.cos(beta))

    def reflect(p):
        d = p[0]*n[0] + p[1]*n[1] + p[2]*n[2]
        return tuple(p[i] - 2*d*n[i] for i in range(3))

    def rvec(v):
        d = v[0]*n[0] + v[1]*n[1] + v[2]*n[2]
        return tuple(v[i] - 2*d*n[i] for i in range(3))
    ell = EllipticArc((0., 0., 0.), (1., 0., t), (0., 1., 0.), -half, 2*half)
    bottom = EllipticArc((0., 0., -1.), (1., 0., 0.), (0., 1., 0.), half, -2*half)
    P = lambda th: (math.cos(th), math.sin(th), t*math.cos(th))
    Q = lambda th: (math.cos(th), math.sin(th), -1.)
    s1 = sp.CylinderSupport((0., 0., 0.), (0., 0., 1.), 1.0, (1., 0., 0.))
    f1 = sp.FaceRecord(1, s1, ((sp.BoundaryUse(ell, 10, 1, 2), sp.BoundaryUse(LinePath(P(half), Q(half)), 11, 2, 3),
                                sp.BoundaryUse(bottom, 12, 3, 4), sp.BoundaryUse(LinePath(Q(-half), P(-half)), 13, 4, 1)),),
                       ((1, P(-half)), (2, P(half)), (3, Q(half)), (4, Q(-half))))
    a2 = rvec((0., 0., 1.))
    s2 = sp.CylinderSupport((0., 0., 0.), a2, 1.0, rvec((1., 0., 0.)))
    bottom2 = EllipticArc(reflect((0., 0., -1.)), rvec((1., 0., 0.)), rvec((0., 1., 0.)), -half, 2*half)
    f2 = sp.FaceRecord(2, s2, ((sp.BoundaryUse(EllipticArc((0., 0., 0.), (1., 0., t), (0., 1., 0.), half, -2*half),
                                               10, 2, 1, forward=False, edge_curve=ell),
                                sp.BoundaryUse(LinePath(P(-half), reflect(Q(-half))), 21, 1, 5),
                                sp.BoundaryUse(bottom2, 22, 5, 6),
                                sp.BoundaryUse(LinePath(reflect(Q(half)), P(half)), 23, 6, 2)),),
                       ((1, P(-half)), (2, P(half)), (5, reflect(Q(-half))), (6, reflect(Q(half)))))
    return f1, f2


def test_equal_radius_mitre_through_the_tangent_point_is_plane_separated():
    f1, f2 = _mitre()
    res = sp.certify_face_pair(f1, f2, TOL)
    assert res.certified, (res.reason, res.evidence.get("attempts"))
    assert res.evidence["plane"][0] == "arc"


def test_defected_supports_certify_with_per_face_membership():
    A, B = L_plates()
    B = plate(2, [(1., 0., 0.), (0., 0., 0.), (0., 0., 1.), (1., 0., 1.)], [2, 1, 5, 6], [10, 14, 15, 16],
              curves=[LinePath((1., 0., 0.), (0., 0., 0.)), None, None, None],
              support=sp.PlaneSupport((0., 1e-14, 0.), (1., 0., 0.), (0., 0., 1.)),
              edges={0: (A.loops[0][0].curve, False)})
    res = sp.certify_face_pair(A, B, TOL)
    assert res.certified
    membership = res.evidence["output_membership"]
    assert membership[1] == 0.0 and membership[2] >= 1e-14


def test_cancellation_propagates_and_budget_gives_undecided():
    A, B = L_plates()

    class Stop(Exception):
        pass

    def check():
        raise Stop
    with pytest.raises(Stop):
        sp.certify_face_pair(A, B, TOL, check=check)
    tiny = sp.SPCertificatePolicy(max_interval_operations=20)
    res = sp.certify_face_pair(*_mitre(), TOL, policy=tiny)
    assert not res.certified and "budget" in res.reason


def _exact_plates(offset):
    """Codex R1 P1-a: perpendicular rectangles meeting along y = 1 + offset (exact rational stored
    input); the output edge record is the line y = 1. True distance from G_A ∩ G_B to T_out ~ offset."""
    y = 1 + offset
    shared = LinePath((0, 1, 0), (1, 1, 0))
    p = lambda *c: tuple(Fraction(x) for x in c)
    support_a = sp.PlaneSupport(p(0, 0, 0), p(1, 0, 0), p(0, 1, 0))
    support_b = sp.PlaneSupport(p(0, y, 0), p(1, 0, 0), p(0, 0, 1))
    A = sp.FaceRecord(1, support_a, ((
        sp.BoundaryUse(LinePath(p(0, 0, 0), p(1, 0, 0)), 11, 1, 2),
        sp.BoundaryUse(LinePath(p(1, 0, 0), p(1, y, 0)), 12, 2, 3),
        sp.BoundaryUse(LinePath(p(1, y, 0), p(0, y, 0)), 10, 3, 4, False, shared),
        sp.BoundaryUse(LinePath(p(0, y, 0), p(0, 0, 0)), 13, 4, 1)),),
        ((1, p(0, 0, 0)), (2, p(1, 0, 0)), (3, p(1, y, 0)), (4, p(0, y, 0))))
    B = sp.FaceRecord(2, support_b, ((
        sp.BoundaryUse(LinePath(p(0, y, 0), p(1, y, 0)), 10, 4, 3, True, shared),
        sp.BoundaryUse(LinePath(p(1, y, 0), p(1, y, 1)), 14, 3, 5),
        sp.BoundaryUse(LinePath(p(1, y, 1), p(0, y, 1)), 15, 5, 6),
        sp.BoundaryUse(LinePath(p(0, y, 1), p(0, y, 0)), 16, 6, 4)),),
        ((3, p(1, y, 0)), (4, p(0, y, 0)), (5, p(1, y, 1)), (6, p(0, y, 1))))
    return A, B


def test_representative_comparison_is_exact():
    """A displacement below float resolution (2^-54) must not vanish in the record comparison."""
    offset = Fraction(1, 1 << 54)
    shared = LinePath((0, 1, 0), (1, 1, 0))
    assert not sp._same_record(LinePath((Fraction(0), 1 + offset, 0), (1, 1 + offset, 0)), shared)
    A, B = _exact_plates(offset)
    res = sp.certify_face_pair(A, B, 1e-15)
    if res.certified:
        assert Fraction(res.r_pair) >= offset            # true distance to the output edge is the offset


class _Sentinel(sp.SPRefusal):
    pass


def test_callback_refusal_propagates_unchanged_and_is_not_cached():
    """Codex R1 P1-b: an exception raised by the callback - even an SPRefusal - is cancellation."""
    A, B = _mitre()
    sentinel = _Sentinel("cancel-original")

    def always():
        raise sentinel
    with pytest.raises(_Sentinel) as info:
        sp.certify_face_pair(A, B, TOL, check=always)
    assert info.value is sentinel
    certifier = sp.SPCertifier()
    calls = []

    def first_only():
        calls.append(1)
        if len(calls) == 1:
            raise sentinel
    with pytest.raises(_Sentinel) as info:
        certifier.certify(A, B, TOL, check=first_only)
    assert info.value is sentinel
    later = certifier.certify(A, B, TOL, check=first_only)
    assert later.certified and later.r_pair == sp.certify_face_pair(A, B, TOL).r_pair


def test_delta_eval_window_covers_sums_of_arcs():
    """Codex R1 P2: C + U cos(phi) can leave the N3 window even when every product is inside it."""
    s = 2.0**-940
    arc = _RawArc((-s, 0., 0.), (s + 2.0**-992, 0., 0.), (0., 1., 0.), 0.0, 1.0)   # EllipticArc's float check underflows
    with pytest.raises(sp.SPRefusal, match="numeric_window:delta_eval"):
        sp._delta_eval(arc, sp.SPCertificatePolicy())
    ordinary = EllipticArc((0.5, 0., 0.), (1., 0., 0.), (0., 1., 0.), 0.0, 1.0)
    assert sp._delta_eval(ordinary, sp.SPCertificatePolicy()) < Fraction(1, 10**13)


# ------------------------------------------- numeric types (Codex R2: exact conversion class)

def _typed_plates(y, y_out):
    """Codex R2 P1: perpendicular rectangles meeting along (x, y, 0), 0 <= x <= 1, with y stored in
    an arbitrary numeric type; the output edge record lies at y_out. True distance |y - y_out|."""
    shared = LinePath((0, y_out, 0), (1, y_out, 0))
    support_a = sp.PlaneSupport((0, 0, 0), (1, 0, 0), (0, 1, 0))
    support_b = sp.PlaneSupport((0, y, 0), (1, 0, 0), (0, 0, 1))
    A = sp.FaceRecord(1, support_a, ((
        sp.BoundaryUse(LinePath((0, 0, 0), (1, 0, 0)), 11, 1, 2),
        sp.BoundaryUse(LinePath((1, 0, 0), (1, y, 0)), 12, 2, 3),
        sp.BoundaryUse(LinePath((1, y, 0), (0, y, 0)), 10, 3, 4, False, shared),
        sp.BoundaryUse(LinePath((0, y, 0), (0, 0, 0)), 13, 4, 1)),),
        ((1, (0, 0, 0)), (2, (1, 0, 0)), (3, (1, y, 0)), (4, (0, y, 0))))
    B = sp.FaceRecord(2, support_b, ((
        sp.BoundaryUse(LinePath((0, y, 0), (1, y, 0)), 10, 4, 3, True, shared),
        sp.BoundaryUse(LinePath((1, y, 0), (1, y, 1)), 14, 3, 5),
        sp.BoundaryUse(LinePath((1, y, 1), (0, y, 1)), 15, 5, 6),
        sp.BoundaryUse(LinePath((0, y, 1), (0, y, 0)), 16, 6, 4)),),
        ((3, (1, y, 0)), (4, (0, y, 0)), (5, (1, y, 1)), (6, (0, y, 1))))
    return A, B


def _old_rational(value, what):
    """Test double of the 3372265 reading rule: int/Fraction exact, anything else through float()."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise sp.SPRefusal(f"numeric_window:type:{what}")
    if isinstance(value, int):
        return Fraction(value)
    number = float(value)
    if not math.isfinite(number):
        raise sp.SPRefusal(f"numeric_window:nonfinite:{what}")
    return Fraction(number)


def test_numpy_integer_coordinate_is_read_exactly(monkeypatch):
    """Codex R2 P1: numpy.int64(2^53 + 1) rounded through float to 2^53 made the stored path and the
    output edge record at y = 2^53 look identical; CERTIFIED(0.0) although the true distance is 1."""
    y = np.int64(2**53 + 1)
    res = sp.certify_face_pair(*_typed_plates(y, 2**53), TOL)
    assert not res.certified or Fraction(res.r_pair) >= 1
    assert not res.certified                                  # 1 is far above r_max
    assert not sp._same_record(LinePath((0, y, 0), (1, y, 0)), LinePath((0, 2**53, 0), (1, 2**53, 0)))
    monkeypatch.setattr(sp, "_rational", _old_rational)       # the double certifies the false bound
    bad = sp.certify_face_pair(*_typed_plates(y, 2**53), TOL)
    assert bad.certified and Fraction(bad.r_pair) < 1


def _huge_edge_plates(big):
    V1, V2 = (0, 0, 0), (big, 0, 0)
    shared = LinePath(V1, V2)
    A = plate(1, [V1, V2, (big, 1, 0), (0, 1, 0)], [1, 2, 3, 4], [10, 11, 12, 13],
              edges={0: (shared, True)})
    B = plate(2, [V2, V1, (0, 0, 1), (big, 0, 1)], [2, 1, 5, 6], [10, 14, 15, 16],
              curves=[LinePath(V2, V1), None, None, None], edges={0: (shared, False)})
    return A, B


def test_out_of_range_rational_coordinate_is_undecided(monkeypatch):
    """Codex R2 P2: an exact rational beyond the float range was converted by float() before any
    range check, so OverflowError escaped certify_face_pair instead of UNDECIDED."""
    A, B = _huge_edge_plates(Fraction(2)**1100)
    res = sp.certify_face_pair(A, B, TOL)
    assert not res.certified and "numeric_window" in res.reason
    # Double: no range check before conversion (exact inputs unwindowed, raw float()).
    monkeypatch.setattr(sp, "_input_in_window", lambda made: True)
    monkeypatch.setattr(sp, "_to_float", lambda value, what: float(value))
    with pytest.raises(OverflowError):
        sp.certify_face_pair(A, B, TOL)


_EXACT_KINDS = ("int", "np.int64", "np.uint64", "np.int32", "Fraction", "Decimal", "float",
                "np.float64", "np.float32")
# np.longdouble is refused explicitly: its precision is platform dependent and a test built from
# Python floats cannot exercise values beyond binary64 (Codex R3 P3).
_REFUSED_KINDS = ("bool", "np.bool_", "str", "complex", "ndarray0d", "None", "np.longdouble")
_MAGNITUDES = {"one": Fraction(1), "tenth": Fraction(1, 10), "2^53+1": Fraction(2**53 + 1),
               "2^1100": Fraction(2)**1100, "2^-1100": Fraction(1, 2**1100), "nan": "nan", "inf": "inf",
               "-inf": "-inf"}


def _make_typed(kind, m):
    """(typed value, exact value or None) or None when the type cannot hold the magnitude."""
    finite = isinstance(m, Fraction)
    integral = finite and m.denominator == 1
    small = finite and abs(m) < 2**1000
    if kind in ("int", "np.int64", "np.uint64", "np.int32"):
        lo, hi = {"int": (-math.inf, math.inf), "np.int64": (-2**63, 2**63), "np.uint64": (0, 2**64),
                  "np.int32": (-2**31, 2**31)}[kind]
        if not integral or not lo <= m < hi:
            return None
        value = int(m) if kind == "int" else getattr(np, kind[3:])(int(m))
        return value, Fraction(int(m))
    if kind == "Fraction":
        return (m, m) if finite else None
    if kind == "Decimal":
        if not finite:
            return Decimal({"nan": "NaN", "inf": "Infinity", "-inf": "-Infinity"}[m]), None
        if integral:
            return Decimal(int(m)), m
        return (Decimal("0.1"), m) if m == Fraction(1, 10) else None
    if kind in ("float", "np.float64", "np.float32"):
        if finite and not small:
            return None
        number = float(m)
        value = {"float": float, "np.float64": np.float64, "np.float32": np.float32}[kind](number)
        if not finite:
            return value, None
        return value, Fraction(*value.as_integer_ratio())
    if not small or m != 1:                    # refused kinds: one representative magnitude
        return None
    value = {"bool": True, "np.bool_": np.True_, "str": "1", "complex": complex(1.0),
             "ndarray0d": np.array(1.0), "None": None, "np.longdouble": np.longdouble(1)}[kind]
    return value, None


_GRID = [(k, label) for k in _EXACT_KINDS + _REFUSED_KINDS for label, m in _MAGNITUDES.items()
         if _make_typed(k, m) is not None]


def _plain(res):
    return (res.certified, res.r_pair if res.certified else res.reason)


def _grid_failure(kind, label):
    """None when the module reads the typed coordinate as its exact value or refuses it with the
    typed reason; else a description. Never lets an exception through."""
    m = _MAGNITUDES[label]
    value, exact = _make_typed(kind, m)
    basis = exact if exact is not None else (m if isinstance(m, Fraction) else Fraction(1))
    if abs(basis) < 2**1000:
        y_out = Fraction(float(basis))                       # the binary64 image a float reading sees
        y_out = int(y_out) if y_out.denominator == 1 else y_out
    else:
        y_out = basis
    try:
        res = sp.certify_face_pair(*_typed_plates(value, y_out), TOL)
    except Exception as error:                 # noqa: BLE001 - the finding class under test
        return f"escaped {type(error).__name__}: {error}"
    if res.certified and exact is not None and Fraction(res.r_pair) < abs(exact - y_out):
        return f"unsound r_pair {res.r_pair} < {abs(exact - y_out)}"
    if kind in _REFUSED_KINDS:
        return None if (not res.certified and "numeric_window:type" in res.reason) else f"accepted: {_plain(res)}"
    if exact is None:
        return None if (not res.certified and "nonfinite" in res.reason) else f"nonfinite accepted: {_plain(res)}"
    reference = sp.certify_face_pair(*_typed_plates(exact, y_out), TOL)
    return None if _plain(res) == _plain(reference) else f"differs from exact: {_plain(res)} vs {_plain(reference)}"


@pytest.mark.parametrize("kind,label", _GRID)
def test_numeric_type_grid_reads_exact_values_or_refuses(kind, label):
    assert _grid_failure(kind, label) is None


def test_numeric_type_grid_catches_the_float_reading_double(monkeypatch):
    monkeypatch.setattr(sp, "_rational", _old_rational)
    failures = {case: _grid_failure(*case) for case in _GRID}
    assert failures[("np.int64", "2^53+1")].startswith("unsound")
    assert failures[("Decimal", "2^53+1")].startswith("unsound")
    assert failures[("str", "one")] is not None and failures[("np.bool_", "one")] is not None


_TOLERANCES = [1e-9, np.float64(1e-9), np.float32(1e-9), Fraction(1, 10**9), Decimal("1e-9"), 1, np.int64(1),
               10**400, Fraction(10)**400, Decimal(10)**400, True, np.True_, "1e-9", np.array(1e-9),
               float("nan"), Decimal("NaN"), float("inf"), None]


@pytest.mark.parametrize("tol", _TOLERANCES, ids=[f"{type(t).__name__}-{i}" for i, t in enumerate(_TOLERANCES)])
def test_tolerance_of_every_numeric_type_is_exact_or_refused(tol):
    A, B = L_plates()
    try:
        res = sp.certify_face_pair(A, B, tol)
    except Exception as error:                 # noqa: BLE001
        pytest.fail(f"escaped {type(error).__name__}: {error}")
    exactly = isinstance(tol, (int, np.integer, Fraction, Decimal, float, np.floating)) \
        and not isinstance(tol, (bool, np.bool_))
    if not exactly:
        assert not res.certified and "numeric_window:type:tolerance" in res.reason
        return
    try:
        ratio = Fraction(*tol.as_integer_ratio()) if hasattr(tol, "as_integer_ratio") else Fraction(int(tol))
    except (OverflowError, ValueError):
        assert not res.certified and "nonfinite" in res.reason
        return
    reference = sp.certify_face_pair(A, B, ratio)
    assert _plain(res) == _plain(reference)
    if ratio > 2**960:
        assert not res.certified and "numeric_window" in res.reason


@pytest.mark.parametrize("kind,label", [c for c in _GRID if c[1] in ("one", "tenth", "2^53+1", "2^1100", "nan")])
def test_public_helpers_read_exact_values_or_refuse(kind, label):
    value, exact = _make_typed(kind, _MAGNITUDES[label])
    calls = (lambda v: sp.degree_two_bound((v, 0, 0, 0, 0)),
             lambda v: sp.first_harmonic_range((v, 1, 0), 0, 1),
             lambda v: sp.certified_sincos(v))

    def outcome(fn, v):
        try:
            return fn(v)
        except sp.SPRefusal as refusal:
            return ("refused", str(refusal))
    for fn in calls:
        try:
            got = outcome(fn, value)
        except Exception as error:             # noqa: BLE001
            pytest.fail(f"escaped {type(error).__name__}: {error}")
        if exact is None:
            assert got[0] == "refused" and ("type" in got[1] or "nonfinite" in got[1])
        else:
            assert got == outcome(fn, exact)


def test_policy_fields_are_validated_exactly():
    assert sp.SPCertificatePolicy(s_min=np.float64(0.02)).s_min == Fraction(0.02)
    assert type(sp.SPCertificatePolicy(end_refinement=np.int64(3)).end_refinement) is int
    for bad in ({"end_refinement": 2.5}, {"trig_argument_bits": True}, {"max_distance_boxes": -1},
                {"s_min": "0.02"}, {"s_min": float("nan")}, {"s_min": 0}):
        with pytest.raises(ValueError):
            sp.SPCertificatePolicy(**bad)


# ------------------------------------- ADR addendum A1-A5: departures with unsound doubles

def _swapped_one_end_range(vp, vq, dp, dq, crit_lo, crit_hi):
    """Double of A1: with exactly one certified end sign, add the critical value of the WRONG kind."""
    lo, hi = min(vp.lo, vq.lo), max(vp.hi, vq.hi)
    sp_, sq_ = sp._sign(dp), sp._sign(dq)
    if sp_ and sp_ == sq_:
        return lo, hi
    if (sp_ == 0) != (sq_ == 0):                 # exactly one end certified: swapped kind
        if sp_ == 1 or sq_ == -1:
            return min(lo, crit_lo), hi
        return lo, max(hi, crit_hi)
    return sp._piece_range(vp, vq, dp, dq, crit_lo, crit_hi)


@pytest.mark.parametrize("sign,p,q,which", [(1, Fraction(-1), Fraction(1, 2), "q"),     # max at 0, sq = -1
                                            (1, Fraction(-1, 2), Fraction(1), "p"),     # max at 0, sp = +1
                                            (-1, Fraction(-1), Fraction(1, 2), "q"),    # min at 0, sq = +1
                                            (-1, Fraction(-1, 2), Fraction(1), "p")])   # min at 0, sp = -1
def test_one_certified_end_sign_keeps_the_compatible_critical_value(sign, p, q, which):
    """Addendum A1: f = sign*cos on [p, q] has its extremum sign*1 at 0. One end sign is certified, the
    other end's derivative enclosure [-1, 1] is valid but undecided; the range must contain f(0)."""
    def enc(x):
        (sl, sh), (cl, ch) = sp.certified_sincos(x)
        value = sp._I(cl, ch) if sign > 0 else sp._I(-ch, -cl)
        deriv = sp._I(-sh, -sl) if sign > 0 else sp._I(sl, sh)          # f' = -sign*sin
        return value, deriv
    (vp, dp), (vq, dq) = enc(p), enc(q)
    if which == "q":
        dp = sp._I(Fraction(-1), Fraction(1))
    else:
        dq = sp._I(Fraction(-1), Fraction(1))
    assert sp._sign(dp if which == "p" else dq) != 0
    extremum = Fraction(sign)

    def contains(rng):
        return rng[0] <= extremum <= rng[1]
    assert contains(sp._piece_range(vp, vq, dp, dq, Fraction(-1), Fraction(1)))
    assert not contains(_swapped_one_end_range(vp, vq, dp, dq, Fraction(-1), Fraction(1)))


def _fin_pair(t):
    """Addendum A2: a radial fin y = t z (stored at y = 0) on the strip's generator at theta = 0."""
    strip = cylinder_strip(1, 0.0, 0.5, 0.0, 1.0, [1, 2, 3, 4], [10, 11, 12, 13])
    fin = plate(2, [(1., 0., 1.), (1., 0., 0.), (2., 0., 0.), (2., 0., 1.)], [4, 1, 5, 6], [13, 20, 21, 22],
                support=sp.PlaneSupport((1., 0., 0.), (1., 0., 0.), (0., t, 1.)),
                edges={0: (strip.loops[0][3].curve, True)})
    return strip, fin


def test_circumferential_family_with_axial_component_certifies():
    """Addendum A2 (i): n.a = t != 0 below s_min uses the circumferential family. The stored faces meet only
    on the shared generator (exact distance 0 between the stored faces); the bound covers the 1e-15 tilt."""
    strip, fin = _fin_pair(1e-15)
    F = sp._Face(fin, sp.SPCertificatePolicy(), None)
    assert sp._dot(F.nF, (0, 0, 1)) != 0
    res = sp.certify_face_pair(strip, fin, TOL)
    assert res.certified and res.r_pair >= 1e-15
    assert res.evidence["case"] == "plane_support_branch" and res.evidence["family"][1] == "circumferential"


def _flat_circumferential_double(monkeypatch):
    """Double of A2: circumferential rate |n x a|/|a| with the angular dependence dropped."""
    original = sp._gradient

    def flat(ctx, face, n, k, s, n2):
        if face.kind == "cyl":
            na = sp._dot(n, face.a)
            if Fraction(na*na, 1)/face.a2 < ctx.policy.s_min**2:
                m = sp._cross(n, face.a)
                return ctx.ar.sqrt(sp._dot(m, m)/face.a2).lo, "circumferential"
        return original(ctx, face, n, k, s, n2)
    monkeypatch.setattr(sp, "_gradient", flat)


def _fin_gradient(theta1, t):
    """(g, family, true minimum rate) of the strip theta in [0, theta1] under the fin's plane (n.a = t)."""
    strip, fin = _fin_pair(t)
    strip = cylinder_strip(1, 0.0, theta1, 0.0, 1.0, [1, 2, 3, 4], [10, 11, 12, 13])
    certifier = sp.SPCertifier()
    F1, F2 = certifier._face(strip, None), certifier._face(fin, None)
    ctx = sp._PairCtx()
    ctx.ar, ctx.policy, ctx.fns, ctx.F1, ctx.F2 = sp._Ar(certifier.policy, None), certifier.policy, {}, F1, F2
    n, k = sp._normalise(F2.nF, F2.kF)
    g, family = sp._gradient(ctx, F1, n, k, 1, sp._dot(n, n))
    m = sp._cross(n, F1.a)                       # rate at angle theta: (cos, sin, 0).(n x a)/|a|
    true_min = min(abs(mpmath.cos(th)*mp(m[0]) + mpmath.sin(th)*mp(m[1]))
                   for th in mpmath.linspace(0, mp(Fraction(theta1)), 400))/mpmath.sqrt(mp(F1.a2))
    return g, family, true_min


def test_circumferential_rate_is_below_the_true_minimum_and_double(monkeypatch):
    """Addendum A2 (ii): the circumferential g_F must not exceed the true rate anywhere on the face's
    angles (here cos(theta1)|n x a|/|a| at theta1 = 1.2). The flat-rate double |n x a|/|a| exceeds it.
    (The addendum's proposal, the tangent-plate attempts of T:532, was measured NOT to discriminate:
    under the double both attempts still refuse, at 'sides:band_above_r_max' and 'branch:contact_not_shared'.)"""
    for t in (0.0, 1e-15):
        g, family, true_min = _fin_gradient(1.2, t)
        assert family == "circumferential" and 0 < mp(g) <= true_min
    _flat_circumferential_double(monkeypatch)
    g, family, true_min = _fin_gradient(1.2, 1e-15)
    assert mp(g) > true_min


def _sin_theta_reference(t1, t2, c1, c2):
    v1, v2 = [[mp(x) for x in t] for t in (t1, t2)]
    cos12 = mpmath.fsum(a*b for a, b in zip(v1, v2))/mpmath.sqrt(mpmath.fsum(a*a for a in v1)*mpmath.fsum(b*b for b in v2))
    x = mpmath.acos(max(-1, min(1, cos12))) - mpmath.acos(mp(c1)) - mpmath.acos(mp(c2))
    return x, mpmath.sin(min(mpmath.pi/2, x))


def _sin_theta_double(abs_sin=False, zero_half_angles=False):
    def double(ar, cone1, cone2):
        t1, t2 = cone1[2], cone2[2]
        den = ar.mul(ar.sqrt(sp._dot(t1, t1)), ar.sqrt(sp._dot(t2, t2)))
        cos12 = ar.div(sp._dot(t1, t2), den)
        cr = sp._cross(t1, t2)
        sin12 = ar.div(ar.sqrt(sp._dot(cr, cr)), den)
        c1, c2 = (Fraction(1), Fraction(1)) if zero_half_angles else (cone1[3], cone2[3])
        s1, s2 = ar.sqrt(1 - c1*c1), ar.sqrt(1 - c2*c2)
        cos_s = ar.sub(c1*c2, ar.mul(s1, s2))
        sin_s = ar.add(ar.mul(s1, c2), ar.mul(s2, c1))
        cos_x = ar.add(ar.mul(cos12, cos_s), ar.mul(sin12, sin_s))
        sin_x = ar.sub(ar.mul(sin12, cos_s), ar.mul(cos12, sin_s))
        if cos_s.lo > 0 and cos_x.hi <= 0:
            return Fraction(1)
        low = max(sin_x.lo, -sin_x.hi) if abs_sin else sin_x.lo
        return min(Fraction(1), low)
    return double


def _sin_theta_violations(fn, seed=11, count=600):
    rng = random.Random(seed)
    ar = sp._Ar(sp.SPCertificatePolicy(), None)
    bad = positive = 0
    for _ in range(count):
        t1 = tuple(Fraction(rng.randint(-9, 9)) for _ in range(3))
        t2 = tuple(Fraction(rng.randint(-9, 9)) for _ in range(3))
        if t1 == (0, 0, 0) or t2 == (0, 0, 0):
            continue
        c1, c2 = (Fraction(rng.randint(1, 1000), 1000) for _ in range(2))
        got = fn(ar, (None, None, t1, c1, None), (None, None, t2, c2, None))
        if got <= 0:
            continue
        positive += 1
        x, ref = _sin_theta_reference(t1, t2, c1, c2)
        bad += not (x > 0 and ref >= mp(got))
    return bad, positive


def test_sin_theta_lower_bound_against_reference_and_doubles():
    """Addendum A3: theta_V with alpha = arccos(c_alpha) exactly; result > 0 must imply theta_V > 0 and
    sin(theta_V) >= result (320-bit reference). Doubles: |sin x| (accepts overlapping cones) and alpha = 0."""
    bad, positive = _sin_theta_violations(sp._sin_theta_lower)
    assert bad == 0 and positive > 50
    assert _sin_theta_violations(_sin_theta_double(abs_sin=True))[0] > 0
    assert _sin_theta_violations(_sin_theta_double(zero_half_angles=True))[0] > 0


def _lambda_strip():
    """Exaggerated scale (addendum A5): stored strip at radius 1 on a support of radius 1.25, and the
    stored vertex 3 moved 0.1 along the tangent: delta_V = 0.1 = 0.08 R, delta_S = 0.2 R (both projected
    distances grow by R/r = 1.25 under radial projection)."""
    rec = cylinder_strip(1, 0.1, 0.9, 0.0, 1.0, [1, 2, 3, 4], [10, 11, 12, 13], R=1.0, support_radius=1.25)
    verts = []
    for vid, p in rec.vertices:
        if vid == 3:
            p = (p[0] - 0.1*math.sin(0.9), p[1] + 0.1*math.cos(0.9), p[2])
        verts.append((vid, p))
    return sp.FaceRecord(rec.face_id, rec.support, rec.loops, tuple(verts))


def _connector_excess(face, lam):
    """max over connector samples of d(sample, projected curve end) - lam*delta_V (mpmath, 120 samples)."""
    R = mp(face.R)
    worst = -mpmath.inf
    for u, nxt, vid, d in face.loop_vertices:
        V = face.vertices[vid]
        for box in (u.ends[1], nxt.ends[0]):
            p = tuple((box[0][i] + box[1][i])/2 for i in range(3))
            tp, tv = (mpmath.atan2(mp(x[1]), mp(x[0])) for x in (p, V))
            zp, zv = mp(p[2]), mp(V[2])
            start = mpmath.matrix([R*mpmath.cos(tp), R*mpmath.sin(tp), zp])
            for j in range(121):
                s = mpmath.mpf(j)/120
                th, z = tp + s*(tv - tp), zp + s*(zv - zp)
                q = mpmath.matrix([R*mpmath.cos(th), R*mpmath.sin(th), z])
                worst = max(worst, mpmath.norm(q - start) - mp(lam)*mp(d))
    return worst


def test_lambda_covers_true_connectors_on_a_cylinder():
    """Addendum A5: every point of the true (chart-straight) connector lies within lambda*delta_V of its
    projected curve end. The double lambda = 1 violates it at this scale."""
    face = sp._Face(_lambda_strip(), sp.SPCertificatePolicy(), None)
    assert max(d for *_, d in face.loop_vertices) > Fraction(9, 100)
    assert _connector_excess(face, face.lam) <= 0
    assert _connector_excess(face, Fraction(1)) > 0


def test_box_turn_error_checks_its_width_hypothesis():
    """Addendum A5 / O3: the box-to-angle bound needs |d_perp| < |r|; now checked (2 hw < R/2). Box with the
    true point (0.5, 0, 0) (|r| = R/2) and midpoint (-0.1, 0, 0): the true angle is pi; the unguarded bound
    pi*hw/R = 0.6 pi is below it."""
    box = ((Fraction(-7, 10), Fraction(0), Fraction(0)), (Fraction(1, 2), Fraction(0), Fraction(0)))
    with pytest.raises(sp.SPRefusal, match="cylinder_turn_unqualified:box_width"):
        sp._box_turn_error(box, Fraction(1))
    hw = Fraction(6, 10)
    unguarded = sp.PI_ENCLOSURE.hi*hw/1                   # the double (no width check)
    assert unguarded < sp.PI_ENCLOSURE.lo                  # below the true angle pi: unsound
    tiny = ((Fraction(1), Fraction(0), Fraction(0)), (Fraction(1) + Fraction(1, 2**100), Fraction(0), Fraction(0)))
    mid, err = sp._box_turn_error(tiny, Fraction(1))
    assert err == sp.PI_ENCLOSURE.hi*Fraction(1, 2**101)    # ordinary boxes: the bound is unchanged


def test_full_turn_refusal_catches_the_no_net_turn_double(monkeypatch):
    """Addendum A5: coded double for T:321 - without the net-turn check the winding annulus is accepted."""
    support = sp.CylinderSupport((0., 0., 0.), (0., 0., 1.), 1.0, (1., 0., 0.))
    bottom = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0.0, 2*math.pi)
    top = EllipticArc((0., 0., 1.), (1., 0., 0.), (0., 1., 0.), 0.0, -2*math.pi)
    annulus = sp.FaceRecord(1, support, ((sp.BoundaryUse(bottom, 10, 1, 1),), (sp.BoundaryUse(top, 11, 2, 2),)),
                            ((1, (1., 0., 0.)), (2, (1., 0., 1.))))
    with pytest.raises(sp.SPRefusal, match="cylinder_loop_winds"):
        _face(annulus)
    monkeypatch.setattr(sp._Face, "_check_net_turn", lambda self, ar: None)
    _face(annulus)                                          # the double accepts the winding loop


# ------------------------------- Codex R3: N3 exempts exact rationals; the window is a float boundary

BIG, TINY = Fraction(2)**1000, Fraction(1, 2**1000)


def _origin_plates(origin):
    """Codex R3 repro: plate A's support origin moved along its own u axis (the plane is unchanged)."""
    A, B = L_plates()
    return replace(A, support=sp.PlaneSupport(origin, A.support.u_vector, A.support.v_vector)), B


def _tall_plates(h):
    """Plate A (z = 0, 0 <= y <= h) and plate B (y = 0, 0 <= z <= 1) share the edge 0 <= x <= 1 and meet
    nowhere else, for every h > 0: the exact distance of G_A n G_B to T_out is 0."""
    V1, V2 = (0, 0, 0), (1, 0, 0)
    shared = LinePath(V1, V2)
    A = plate(1, [V1, V2, (1, h, 0), (0, h, 0)], [1, 2, 3, 4], [10, 11, 12, 13], edges={0: (shared, True)})
    B = plate(2, [V2, V1, (0, 0, 1), (1, 0, 1)], [2, 1, 5, 6], [10, 14, 15, 16],
              curves=[LinePath(V2, V1), None, None, None], edges={0: (shared, False)})
    return A, B


def _tilted_fin_pair(tilt):
    """The fin pair of A2 with the strip's (unshared) top arc tilted by ``tilt`` along the axis; the arc
    stays on the cylinder (radius 1) exactly, so only incidence defects of size ~tilt change."""
    strip, fin = _fin_pair(0.0)
    loop = list(strip.loops[0])
    top = loop[2].curve
    loop[2] = replace(loop[2], curve=_RawArc(tuple(top.center), (1., 0., tilt), (0., 1., 0.),
                                             float(top.start_angle), float(top.sweep_angle)))
    return replace(strip, loops=(tuple(loop),)), fin


# Expected outcomes are derived without the module's reader: the exact geometry is unchanged or the
# exact distance is 0 (so any certified bound is sound), and these inputs certified on the R2 module
# (3372265, which applied the window to float-held values only); a shared-edge endpoint is a float
# input of the delta_eval forward-error model, so outside the window it must refuse.
_EXACT_RATIONAL_CASES = {
    "origin_tiny": (lambda: _origin_plates((TINY, 0, 0)), "certified"),
    "origin_big": (lambda: _origin_plates((BIG, 0, 0)), "certified"),
    "far_corner_big": (lambda: _tall_plates(BIG), "certified"),
    # |n_A| = 2^-1000 is below the 2^-80 grid of the outward square root: an exact-path capability
    # limit (the same refusal on R2), not a window refusal.
    "far_corner_tiny": (lambda: _tall_plates(TINY), "face:degenerate_plane_support"),
    "arc_tilt_tiny": (lambda: _tilted_fin_pair(TINY), "certified"),
    "shared_edge_endpoint_big": (lambda: _huge_edge_plates(BIG), "numeric_window:delta_eval"),
}


def _r3_outcome(name):
    build, expected = _EXACT_RATIONAL_CASES[name]
    res = sp.certify_face_pair(*build(), TOL)
    if expected == "certified":
        return res.certified and res.r_pair >= 0           # exact distance 0: any certified bound is sound
    return not res.certified and expected in res.reason


@pytest.mark.parametrize("name", list(_EXACT_RATIONAL_CASES))
def test_exact_rationals_outside_the_float_window_are_exact_inputs(name, monkeypatch):
    """Codex R3 P2: N3 exempts exact rational arithmetic; the window applies where a float is held or
    produced. Double: the b20fbeb rule (window on every input) refuses the certified cases."""
    assert _r3_outcome(name)
    if _EXACT_RATIONAL_CASES[name][1] == "certified":
        monkeypatch.setattr(sp, "_float_held", lambda value: True)
        assert not _r3_outcome(name)


def test_degree_two_bound_of_a_tiny_exact_rational():
    """Codex R3 repro: a purely rational routine; a0 = 2^-1000 and S = 0 give [a0, a0] exactly."""
    assert sp.degree_two_bound((TINY, 0, 0, 0, 0)) == (TINY, TINY)
    assert sp.degree_two_bound((BIG, 1, 0, 0, 0)) == (BIG - 1, BIG + 1)


_FLOAT_HELD_CASES = {
    "origin": lambda: _origin_plates((2.0**-1000, 0., 0.)),
    "arc_tilt": lambda: _tilted_fin_pair(2.0**-1000),
    "far_corner": lambda: _tall_plates(2.0**1000),
}


@pytest.mark.parametrize("name", list(_FLOAT_HELD_CASES))
def test_float_held_values_outside_the_window_still_refuse(name, monkeypatch):
    """N3 for stored binary64 values is unchanged; double: no window at float inputs certifies."""
    res = sp.certify_face_pair(*_FLOAT_HELD_CASES[name](), TOL)
    assert not res.certified and "numeric_window" in res.reason
    if name != "far_corner":
        monkeypatch.setattr(sp, "_float_held", lambda value: False)
        assert sp.certify_face_pair(*_FLOAT_HELD_CASES[name](), TOL).certified


def test_lower_bound_above_the_window_refuses():
    """Codex R3 P3: a float lower bound above 2^960 is a refusal (N3), not a clamped value."""
    with pytest.raises(sp.SPRefusal, match="numeric_window"):
        sp._down(Fraction(2)**1100)
    assert sp._down(Fraction(3, 2)) == 1.5


def test_malformed_records_are_undecided():
    """Codex R3 P3: structurally malformed records give UNDECIDED, never TypeError/ValueError/AttributeError."""
    A, B = L_plates()
    broken = [replace(A, loops=None), replace(A, vertices=((1,),)), replace(A, vertices=None),
              replace(A, loops=((object(),),)), replace(A, loops=(None,)),
              types.SimpleNamespace(support=A.support, loops=A.loops, vertices=A.vertices), None]
    for record in broken:
        for pair in ((record, B), (B, record)):
            try:
                res = sp.certify_face_pair(*pair, TOL)
            except Exception as error:             # noqa: BLE001
                pytest.fail(f"escaped {type(error).__name__}: {error}")
            assert not res.certified and "record:malformed" in res.reason


def test_longdouble_is_refused_explicitly():
    with pytest.raises(sp.SPRefusal, match="numeric_window:type"):
        sp.degree_two_bound((np.longdouble(1), 0, 0, 0, 0))


def test_module_is_not_wired_into_production():
    src = pathlib.Path(sp.__file__).resolve().parent
    users = [p.name for p in src.rglob("*.py") if p.name != "separating_plane_certificate.py"
             and "separating_plane_certificate" in p.read_text(encoding="utf-8", errors="replace")]
    assert users == []


# ------------------------------------------------------------ randomized soundness (exact oracle)

def _proj(p, n, k):
    t = (sum(a*b for a, b in zip(n, p)) - k)/sum(a*a for a in n)
    return tuple(p[i] - t*n[i] for i in range(3))


def _cross(a, b):
    return (a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0])


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _polygon(record):
    """Gamma_F of a plate: projected stored curve ends and connectors through the stored vertices (exact)."""
    s = record.support
    o, u, v = (tuple(map(Fraction, x)) for x in (s.origin, s.u_vector, s.v_vector))
    n = _cross(u, v)
    k = _dot(n, o)
    pos = {vid: tuple(map(Fraction, p)) for vid, p in record.vertices}
    pts = []
    for use in record.loops[0]:
        pts.append(_proj(tuple(map(Fraction, use.curve.start)), n, k))
        pts.append(_proj(tuple(map(Fraction, use.curve.end)), n, k))
        pts.append(_proj(pos[use.end_vertex], n, k))
    return pts, n, k


def _member(pts, n, x):
    """x (in the plane) on Gamma_F or with nonzero winding (exact, 2-D projection)."""
    drop = max(range(3), key=lambda i: abs(n[i]))
    keep = [i for i in range(3) if i != drop]
    P = [(p[keep[0]], p[keep[1]]) for p in pts]
    X = (x[keep[0]], x[keep[1]])
    wind = 0
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        cr = (b[0] - a[0])*(X[1] - a[1]) - (b[1] - a[1])*(X[0] - a[0])
        if cr == 0 and min(a[0], b[0]) <= X[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= X[1] <= max(a[1], b[1]):
            return True
        if a[1] <= X[1] < b[1] and cr > 0:
            wind += 1
        elif b[1] <= X[1] < a[1] and cr < 0:
            wind -= 1
    return wind != 0


def _line_params(pts, n, P0, d):
    drop = max(range(3), key=lambda i: abs(n[i]))
    i0, i1 = [i for i in range(3) if i != drop]
    out = set()
    for j in range(len(pts)):
        a, b = pts[j], pts[(j + 1) % len(pts)]
        e = (b[i0] - a[i0], b[i1] - a[i1])
        det = d[i0]*(-e[1]) - d[i1]*(-e[0])
        r = (a[i0] - P0[i0], a[i1] - P0[i1])
        if det != 0:
            t = (r[0]*(-e[1]) - r[1]*(-e[0]))/det
            s_ = (d[i0]*r[1] - d[i1]*r[0])/det
            if 0 <= s_ <= 1:
                out.add(t)
        else:
            for q in (a, b):
                dd = d[i0]*d[i0] + d[i1]*d[i1]
                out.add(((q[i0] - P0[i0])*d[i0] + (q[i1] - P0[i1])*d[i1])/dd)
    return out


def _seg_dist2(x, a, b):
    ab = tuple(b[i] - a[i] for i in range(3))
    ax = tuple(x[i] - a[i] for i in range(3))
    t = max(Fraction(0), min(Fraction(1), _dot(ax, ab)/_dot(ab, ab)))
    return sum((ax[i] - t*ab[i])**2 for i in range(3))


def _exact_excess(A, B, tout_edges, tout_points):
    """Max squared distance from G_A ∩ G_B to T_out (exact; regions are planar polygons)."""
    pa, na, ka = _polygon(A)
    pb, nb, kb = _polygon(B)
    d = _cross(na, nb)
    dd = _dot(d, d)
    if dd == 0:
        return None
    P0 = tuple(((ka*_dot(nb, nb) - kb*_dot(na, nb))*na[i] + (kb*_dot(na, na) - ka*_dot(na, nb))*nb[i])/dd
               for i in range(3))
    ts = sorted(_line_params(pa, na, P0, d) | _line_params(pb, nb, P0, d))
    X = lambda t: tuple(P0[i] + t*d[i] for i in range(3))
    inside = lambda t: _member(pa, na, X(t)) and _member(pb, nb, X(t))
    worst = Fraction(0)
    probes = list(ts) + [(ts[i] + ts[i + 1])/2 for i in range(len(ts) - 1)]
    for t in probes:
        if inside(t):
            x = X(t)
            dist = min([_seg_dist2(x, a, b) for a, b in tout_edges] +
                       [sum((x[i] - p[i])**2 for i in range(3)) for p in tout_points] or [Fraction(10**9)])
            worst = max(worst, dist)
    return worst


def _rotation(rng):
    q = [rng.gauss(0, 1) for _ in range(4)]
    s = math.sqrt(sum(x*x for x in q))
    w, x, y, z = (c/s for c in q)
    return [[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]


def _random_pair(rng, kind):
    M = _rotation(rng)
    off = [rng.uniform(-5, 5) for _ in range(3)]
    tr = lambda p: tuple(sum(M[i][j]*p[j] for j in range(3)) + off[i] for i in range(3))
    noise = lambda s: tuple(rng.uniform(-s, s) for _ in range(3))
    jitter = lambda p, s: tuple(p[i] + e for i, e in enumerate(noise(s)))
    Lx, w1, w2 = rng.uniform(.5, 2), rng.uniform(.3, 1.5), rng.uniform(.3, 1.5)
    alpha = rng.uniform(.15, math.pi - .15)
    c, s = math.cos(alpha), math.sin(alpha)
    dv, dc, dsup = 10.0**rng.uniform(-16, -13.5), 10.0**rng.uniform(-16, -13.5), 10.0**rng.uniform(-16, -14)
    V1, V2 = tr((0., 0., 0.)), tr((Lx, 0., 0.))
    A3, A4 = tr((Lx, w1, 0.)), tr((0., w1, 0.))
    if kind == "edge":
        B3, B4 = tr((0., w2*c, w2*s)), tr((Lx, w2*c, w2*s))
    else:   # crossing: B shares V1 only and cuts through A's interior
        B3, B4 = tr((Lx*.6, w1*.5, -w2)), tr((Lx*.6, w1*.5, w2))
    pos = {1: jitter(V1, dv), 2: jitter(V2, dv), 3: jitter(A3, dv), 4: jitter(A4, dv), 5: jitter(B3, dv),
           6: jitter(B4, dv)}
    shared = LinePath(pos[1], pos[2])

    def face(fid, order, eids, shared_index=None, reverse_shared=False):
        uses = []
        for i in range(len(order)):
            a, b = order[i], order[(i + 1) % len(order)]
            if i == shared_index:
                curve = LinePath(shared.end, shared.start) if reverse_shared else shared
                uses.append(sp.BoundaryUse(curve, eids[i], a, b, forward=not reverse_shared, edge_curve=shared))
            else:
                uses.append(sp.BoundaryUse(LinePath(jitter(pos[a], dc), jitter(pos[b], dc)), eids[i], a, b))
        o = jitter(pos[order[0]], dsup)
        u = tuple(pos[order[1]][i] - pos[order[0]][i] + e for i, e in enumerate(noise(dsup)))
        v = tuple(pos[order[-1]][i] - pos[order[0]][i] + e for i, e in enumerate(noise(dsup)))
        return sp.FaceRecord(fid, sp.PlaneSupport(o, u, v), (tuple(uses),),
                             tuple((vid, pos[vid]) for vid in order))
    A = face(1, [1, 2, 3, 4], [10, 11, 12, 13], shared_index=0)
    if kind == "edge":
        B = face(2, [2, 1, 5, 6], [10, 14, 15, 16], shared_index=0, reverse_shared=True)
        tout = ([(tuple(map(Fraction, shared.start)), tuple(map(Fraction, shared.end)))], [])
    else:
        B = face(2, [1, 5, 6], [20, 21, 22])
        tout = ([], [tuple(map(Fraction, pos[1]))])
    return A, B, tout


def test_randomized_certified_bounds_never_below_exact_distance():
    rng = random.Random(20261029)
    certifier = sp.SPCertifier()
    certified = {"edge": 0, "cross": 0}
    total = {"edge": 0, "cross": 0}
    for i in range(160):
        kind = "edge" if i % 4 else "cross"
        A, B, (edges, points) = _random_pair(rng, kind)
        res = certifier.certify(A, B, TOL)
        total[kind] += 1
        excess = _exact_excess(A, B, edges, points)
        if res.certified:
            certified[kind] += 1
            assert excess is not None
            assert Fraction(res.r_pair)**2 >= excess, (i, kind, res.r_pair, float(excess)**.5)
    assert certified["cross"] == 0
    assert certified["edge"] >= total["edge"]//2          # capability sanity on small defects
