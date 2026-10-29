"""T2-1a: exact separating-plane certificate (ADR-T2 v2 section 4, tests of section 8.6 / 4.2c).

Safety tests are paired with a test double of the unsound rule they guard: the check must
pass for the module and fail for the double (ADR 8.6). Soundness assertions hold for every
outcome; capability assertions are made only for fixed inputs of small proof cost.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import pathlib
import random

import mpmath
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
