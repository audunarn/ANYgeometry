"""Plane/Cylinder support intersection as exact polynomial-chart branches of the STORED plane.

The counterpart of :func:`anygeometry.branch_supports.bezier_quadric_support` for a plane first support
(:class:`~anygeometry.plane_ruled.PlaneRuledSupport`). Nothing is dropped, merged or classified by a heuristic
slack: every decision is either exact (the events are exact polynomial roots), carries an interval/derived world
bound that is checked against a quarter of the face tolerance, or is a typed :class:`PlaneRuledSupportError`.

Completeness (P1, P4)
---------------------
* The chart window contains, with a margin of at least 100 tolerances, the exact rational projection of the whole
  finite patch (:func:`~anygeometry.plane_ruled.build_plane_ruled_support`), so every point of plane-meet-patch has
  chart coordinates in the window; a kept piece may not reach the window ends (checked).
* Membership in the patch of a point of a branch changes only at (a) folds / double contacts of the discriminant,
  (b) crossings of the four boundary planes of the patch (exact roots of the resultants). Every such root is kept
  unless the branch point is provably outside the patch by more than ``max(100 tol, 1e-6)`` (retaining an extra
  event is always safe; dropping one is not). Events closer than the chart tolerance are CLUSTERED, and a cluster is
  accepted only when the interval-arithmetic world extent of the whole cluster (Bernstein ranges of ``B`` and ``D``
  including the square root, so a near-fold ``sqrt`` growth is covered) is at most a quarter tolerance; it then
  becomes a recorded point contact. Otherwise the pair is refused.
* Pieces between events are classified by their midpoint margin to the NATIVE finite bounds (axial extent and
  angular sweep, in metres). Margins beyond +-tol/4 decide; inside that band the piece must be proven COINCIDENT
  with one boundary plane (the interval distance along the whole piece is within tol/4) - it is then retained as
  boundary of the closed patch - or the pair is refused as ambiguous.

Evaluation error (P3)
---------------------
:func:`curve_error_bound` derives, per kept curve, a world bound on the distance of every evaluated point to the
exact curve (hence to BOTH exact supports): correctly rounded Bernstein coefficients plus Horner evaluation error
of ``B`` and ``D`` (and of the fold-chart ``h`` form), the square-root sensitivity
``min(sqrt(dD), dD / sqrt(D_lo - dD))`` (finite even at ``D = 0``), the cancellation-safe root formula, the rounding
of the chart parameter, the fold anchor (``D'`` times its rounding over ``4 A |grad Q|``), the final vector
operations and the rounding of the rational chart origin. A curve whose bound exceeds a quarter tolerance is
refused. These are derived first-order bounds with a documented safety factor of two, not a machine-checked proof.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

from .analytic_supports import SupportIntersection
from .branch_algebra import get_poly_plan
from .branch_curves import BezierQuadricCurve
from .branch_supports import _branch_values, _in_range, _points, _TOLERANCE
from .errors import GeometryError
from .plane_ruled import PlaneRuledSupportError, build_plane_ruled_support
from .quadric_algebra import QuadricSupport, _fraction_cross
from .quadric_supports import _boundary_planes, _merge, _poll
from .runtime_diagnostics import _increment

_EPS = float(np.finfo(float).eps)
_TAU = 2.0 * math.pi


def _refuse(message, reason, face_pair, **details):
    _increment("plane_chart_refusals")
    raise PlaneRuledSupportError(message, reason=reason, face_pair=face_pair, **details)


# ---------------------------------------------------------------------------
# Native finite bounds


def patch_margins(cylinder, point):
    """``(axial, angular)`` signed distances in metres from ``point`` to the patch's native bounds (negative = outside)."""
    rel = np.asarray(point, dtype=float) - np.asarray(cylinder.origin)
    axis = np.asarray(cylinder.axis)
    z = float(rel @ axis)
    height = float(cylinder.height)
    lo, hi = (0.0, height) if height > 0 else (height, 0.0)
    axial = min(z - lo, hi - z)
    sweep = float(cylinder.sweep_angle)
    if abs(abs(sweep) - _TAU) <= 1e-12:
        return axial, math.inf
    radial = rel - z * axis
    angle = math.atan2(float(radial @ cylinder.circumferential_direction), float(radial @ cylinder.radial_direction))
    delta = ((angle - float(cylinder.start_angle)) * math.copysign(1.0, sweep)) % _TAU
    span = abs(sweep)
    radius = float(cylinder.radius)
    if delta <= span:
        return axial, radius * min(delta, span - delta)
    return axial, -radius * min(delta - span, _TAU - delta)


# ---------------------------------------------------------------------------
# Interval helpers on the exact-coefficient Bernstein forms


def _val_err(form):
    """Absolute error of a ``BernsteinForm`` value: rounded coefficients (eps/2 each, convex weights) plus the Horner
    recurrence (``ratio``, ``m`` multiply-adds, the power scale): ``(4 m + 8) eps max|coefficient|``."""
    return (4 * form.degree + 8) * _EPS * max(form.bound, 1e-300)


def _s_interval(fs, branch, lo, hi):
    """Outward interval of ``s = (-B +- sqrt(D)) / (2 A)`` of one branch over ``t`` in ``[lo, hi]``."""
    two_a = 2.0 * fs.a0
    b = fs.B_forms[0].range(lo, hi)
    d = fs.D_forms[0].range(lo, hi)
    r_lo, r_hi = math.sqrt(max(d[0], 0.0)), math.sqrt(max(d[1], 0.0))
    if branch > 0:
        return (-b[1] + r_lo) / two_a, (-b[0] + r_hi) / two_a
    return (-b[1] - r_hi) / two_a, (-b[0] - r_lo) / two_a


def _padded(poly, size):
    return [Fraction(v) for v in poly] + [Fraction(0)] * (size - len(poly))


def coincidence_bound(support, plan, boundary, lo, hi, *, mid_world_distance=None):
    """Metre bound of the distance of a branch to a boundary plane over the whole chart piece ``[lo, hi]``, or ``inf``.

    With ``d = lin(t) + sigma k sqrt(D)`` the signed (unnormalized) plane distance of branch ``sigma`` (``lin`` and
    ``k`` exact rationals, ``lin`` linear because ``B`` is), the conjugate product ``d_sigma d_-sigma = lin^2 - k^2 D
    = p(t)`` is an EXACT quadratic. Since ``d_sigma - d_-sigma = 2 sigma k sqrt(D)``, ``|d_-sigma| >= 2K - |d_sigma|``
    with ``K = |k| sqrt(D_lo)``, hence ``f (2K - f) <= p_max`` for ``f = |d_sigma|`` and ``f`` lies on the small side
    ``f <= p_max / (K + sqrt(K^2 - p_max))`` as soon as it does at one point (continuity; the caller checks the
    middle). Both branches are covered (the bound does not depend on ``sigma``). ``p_max`` is the exact maximum of
    ``|p|`` over the piece. ``inf`` when ``K^2 <= p_max`` (the piece is not close to the plane, or ``D`` is not
    bounded below).
    """
    fs = plan.floats()
    normal = tuple(Fraction(c) for c in boundary.axis)
    anchor = tuple(Fraction(c) for c in boundary.origin)
    origin = support.origin
    w1 = tuple(Fraction(c) for c in support.directrix_vector)
    w2 = tuple(Fraction(c) for c in support.ruling)
    length = Fraction(support.length)
    d0 = sum(normal[i] * (origin[i] - anchor[i]) for i in range(3))
    c1 = sum(normal[i] * w1[i] for i in range(3)) * length
    c2 = sum(normal[i] * w2[i] for i in range(3))
    big_a = Fraction(plan.A[0])
    b0, b1 = _padded(plan.B, 2)[:2]
    d_poly = _padded(plan.disc, 3)[:3]
    k = c2 / (2 * big_a)
    l0, l1 = d0 - k * b0, c1 - k * b1
    p = (l0 * l0 - k * k * d_poly[0], 2 * l0 * l1 - k * k * d_poly[1], l1 * l1 - k * k * d_poly[2])
    stations = [Fraction(lo), Fraction(hi)]
    if p[2] != 0:
        vertex = -p[1] / (2 * p[2])
        if stations[0] < vertex < stations[1]:
            stations.append(vertex)
    p_max = max(abs(p[0] + p[1] * t + p[2] * t * t) for t in stations)
    d_range = fs.D_forms[0].range(lo, hi)
    norm = Fraction(math.sqrt(float(sum(c * c for c in normal)))) * Fraction(1.0 - 1e-12)
    if norm <= 0:
        return math.inf
    # direct bound |d| <= max|lin| (exact, lin is linear) + |k| sqrt(D_max); exact when k == 0 (ruling parallel to the plane)
    lin_max = max(abs(l0 + l1 * Fraction(lo)), abs(l0 + l1 * Fraction(hi)))
    direct = (lin_max + abs(k) * Fraction(math.sqrt(max(d_range[1], 0.0))) * Fraction(1.0 + 1e-12)) / norm
    best = 2.0 * float(direct) + 8 * _EPS
    if d_range[0] > 0.0 and mid_world_distance is not None:
        big_k = abs(k) * Fraction(math.sqrt(d_range[0])) * Fraction(1.0 - 1e-12)
        # branch selection: the conjugate-product inequality bounds BOTH d_sigma and d_-sigma; the selected branch is
        # on the small side only if its (upper-bounded) middle distance is below K (the small and large roots of
        # f(2K - f) = p_max are separated by K)
        selected = Fraction(float(mid_world_distance)) * norm < big_k
        if selected and big_k * big_k > p_max:
            small = p_max / (big_k + Fraction(math.sqrt(float(big_k * big_k - p_max))))
            best = min(best, 2.0 * float(small / norm) + 8 * _EPS)
    return best


def cluster_extent_metres(support, fs, lo, hi):
    """Interval bound of the world extent of ALL branches over the chart interval ``[lo, hi]`` (covers sqrt growth)."""
    b = fs.B_forms[0].range(lo, hi)
    d = fs.D_forms[0].range(lo, hi)
    r_lo, r_hi = math.sqrt(max(d[0], 0.0)), math.sqrt(max(d[1], 0.0))
    # ONE branch s = (-B +- sqrt(D)) / 2A moves by at most (width of B + width of sqrt(D)) / 2A over the cluster; the
    # width of sqrt(D) is sqrt(D_hi) - sqrt(D_lo) <= sqrt(D_hi - D_lo) (a near-fold sqrt growth is covered)
    s_width = ((b[1] - b[0]) + (r_hi - r_lo)) / (2.0 * fs.a0)
    w1 = math.sqrt(sum(c * c for c in support.directrix_vector))
    w2 = math.sqrt(sum(c * c for c in support.ruling))
    return support.length * w1 * (hi - lo) + w2 * s_width


def _sqrt_down(x):
    return math.sqrt(max(float(x), 0.0)) * (1.0 - 4e-16)


def _sqrt_up(x):
    return math.sqrt(max(float(x), 0.0)) * (1.0 + 4e-16)


def fold_distance_bound(plan, support, cylinder, anchors, z_interval):
    """Nonlinear world bound ``T`` on the distance (in the stored plane) from the shifted fold curve to the exact curve.

    A fold anchor that is only a double makes the evaluated chart the exact branch of ``Q = -D(anchor)/(4A)`` (a
    constant offset, exactly ``E = max_anchor |D(anchor)| / (4 A)``, evaluated in rationals at the anchor double).
    ``Q`` is the polynomial of ``QuadricSupport.exact_relative``: ``Q(v) = |v|^2 - (a.v)^2/alpha - r^2`` with
    ``alpha = a.a`` (the stored axis is divided by its exact norm), so with ``z' = a.v``

        ||grad Q||^2 / 4 = r^2 + Q,        ||P_plane grad Q||^2 / 4 = r^2 + Q - (k - c z'/alpha)^2 / N

    (``n = w1 x w2`` the raw stored-plane normal, ``N = n.n``, ``c = n.a``, ``k = n.(o_plane - o_cyl)``; the plane
    equation fixes ``n.v = k`` for every point of the stored plane). The Hessian of ``Q`` is ``2 (I - a a^T/alpha)``,
    norm 2 (``K = 1``, global). The start points are the SHIFTED curve itself, so ``z'`` ranges over the exact
    interval ``a.(o' - o_c) + L (a.w1) t + (a.w2) s`` for ``t`` in the chart interval ``z_interval`` (a ``(t_lo,
    t_hi)`` pair) and ``s = (-B +- sqrt(D(t) + |D(anchor)|)) / 2A`` (outward-padded Bernstein ranges, combined in exact
    rationals). With ``Q >= -E`` the in-plane gradient at those points satisfies
    ``g^2 / 4 >= max(r^2 - E - H^2, (r^2 - E) c^2 / (N alpha))`` where ``H^2`` is the larger endpoint value of the
    convex ``(k - c z'/alpha)^2 / N`` over that interval (the second term uses ``|n.rho| <= |n_perp|``). Along the
    in-plane gradient line ``q(tau) = Q(y + tau u)`` has ``q' >= g`` and ``|q''| <= 2``, so if ``g^2 > 4 E`` a zero
    lies within ``T = 2E/g`` (``T <= g/2`` and ``E - g T + T^2 <= 0`` exactly; intermediate value theorem). ``g`` is
    rounded DOWN and ``T`` UP by bounded exact-rational post-checks; ``inf`` when ``g^2 <= 4E`` or a rounding is not
    certifiable (positive underflow). No tube around the curve is assumed.
    """
    if not anchors:
        return 0.0
    big_a = Fraction(plan.A[0])
    d_poly = [Fraction(v) for v in plan.disc]
    e_q = Fraction(0)
    for anchor in anchors:
        x = Fraction(float(anchor))
        e_q = max(e_q, abs(sum(c * x ** i for i, c in enumerate(d_poly))) / (4 * big_a))
    if e_q == 0:
        return 0.0
    w1, w2 = (tuple(Fraction(c) for c in v) for v in (support.directrix_vector, support.ruling))
    n = _fraction_cross(w1, w2)
    big_n = sum(c * c for c in n)
    a = tuple(Fraction(c) for c in cylinder.axis)
    alpha = sum(c * c for c in a)
    o_p, o_c = support.origin, tuple(Fraction(c) for c in cylinder.origin)
    k = sum(n[i] * (o_p[i] - o_c[i]) for i in range(3))
    c = sum(n[i] * a[i] for i in range(3))
    # z' interval of the SHIFTED curve itself: z' = a.(o' - o_c) + L (a.w1) t + (a.w2) s over the chart t interval and
    # the s interval s = (-B +- sqrt(Delta(t) - Delta(anchor))) / 2A with Delta(t) - Delta(anchor) <= D(t) + |D(anchor)|
    # (all from the outward-padded Bernstein ranges, combined in exact rationals).
    fs = plan.floats()
    t_lo, t_hi = (float(v) for v in z_interval)
    b_iv = fs.B_forms[0].range(t_lo, t_hi)
    d_hi = fs.D_forms[0].range(t_lo, t_hi)[1]
    d_anchor = max(abs(sum(c0 * Fraction(float(x)) ** i for i, c0 in enumerate(d_poly))) for x in anchors)
    radicand = Fraction(max(d_hi, 0.0)) + d_anchor
    r_float = _sqrt_up_certified(radicand)
    if not math.isfinite(r_float):
        return math.inf                                  # positive underflow / overflow: not certifiable
    r_hi = Fraction(r_float)
    two_a = 2 * big_a
    s_lo = (-Fraction(b_iv[1]) - r_hi) / two_a
    s_hi = (-Fraction(b_iv[0]) + r_hi) / two_a
    a_w1 = sum(a[i] * w1[i] for i in range(3)) * Fraction(support.length)
    a_w2 = sum(a[i] * w2[i] for i in range(3))
    z0 = sum(a[i] * (o_p[i] - o_c[i]) for i in range(3))
    t_terms = sorted((a_w1 * Fraction(t_lo), a_w1 * Fraction(t_hi)))
    s_terms = sorted((a_w2 * s_lo, a_w2 * s_hi))
    z_lo, z_hi = z0 + t_terms[0] + s_terms[0], z0 + t_terms[1] + s_terms[1]
    r2 = Fraction(cylinder.radius) ** 2
    h2 = max((k - c * z / alpha) ** 2 for z in (z_lo, z_hi)) / big_n          # convex: endpoints
    # normalized exact Q: ||P_plane grad Q||^2 / 4 = r^2 + Q - (k - c z'/alpha)^2/N and, because rho is perpendicular to
    # the axis, (k - c z'/alpha)^2/N <= (r^2 + Q)(1 - c^2/(N alpha)); with Q >= -E this gives the two lower bounds
    g2 = 4 * max(r2 - e_q - h2, (r2 - e_q) * c * c / (big_n * alpha))
    if g2 <= 4 * e_q:
        return math.inf                                 # Hessian norm 2 (K = 1): needs g^2 > 4 E
    # certified float g: Fraction(g)^2 <= g2 (rounded DOWN by exact post-check), then T = 2E/g satisfies, exactly,
    # T <= g/2 and E - g T + T^2 = E - 2E + 4E^2/g^2 <= 0 iff g^2 >= 4E, and is rounded UP by exact post-check
    g = _sqrt_down_certified(g2)
    if not (g > 0.0 and Fraction(g) ** 2 > 4 * e_q):
        return math.inf
    return _div_up_certified(2 * e_q, g)


def curve_error_bound(plan, support, curve, cylinder, *, with_ends=True, z_pad=1e-6):
    """Derived world bound (metres) on the distance of any evaluated point of ``curve`` to the exact curve.

    Returns ``(bound, details)``; ``math.inf`` when the regime is not covered (``D`` not bounded away from the
    rounding of its own evaluation in a linear chart where its derivative is needed).
    """
    fs = plan.floats()
    a = fs.a0
    t_a, t_b = (float(v) for v in curve._chart(np.asarray((0.0, 1.0)))[0])
    lo, hi = min(t_a, t_b), max(t_a, t_b)
    b_form, d_form = fs.B_forms[0], fs.D_forms[0]
    b_iv, d_iv = b_form.range(lo, hi), d_form.range(lo, hi)
    b_max = max(abs(b_iv[0]), abs(b_iv[1]))
    d_max = max(d_iv[1], 0.0)
    b1_max = max(abs(v) for v in fs.B_forms[1].range(lo, hi))
    d1_max = max(abs(v) for v in fs.D_forms[1].range(0.0, 1.0))
    d_b = _val_err(b_form) + _EPS * b1_max                       # coefficient/Horner error + rounding of t
    d_d = _val_err(d_form)
    mode = curve.parameterization
    fold_anchors = ()
    if mode == "linear":
        d_lo = d_iv[0]
        d_root = math.sqrt(d_d)
        if d_lo > 2.0 * d_d:
            d_root = min(d_root, d_d / math.sqrt(d_lo - d_d))
        root_max = math.sqrt(d_max)
    else:
        start, end, sweep = curve.start, curve.start + curve.sweep, curve.sweep
        anchors = {"left_square": (start,), "right_square": (end,), "both_sine": (start, end)}[mode]
        fold_anchors = anchors
        d_root, root_max = 0.0, 0.0
        for anchor in anchors:
            extent = sweep if anchor == start else -sweep
            form = fs.h_form(float(anchor), float(extent))
            # branch_curves._evaluate_scalar / _chart: in both_sine the LEFT anchor serves tau <= 1/2 (f = sin(pi tau/2),
            # f^2 <= 1/2) and the RIGHT anchor tau > 1/2 (f = cos(pi tau/2), f^2 < 1/2), so h is only evaluated for
            # u = f^2 in [0, 1/2]; the one-sided square charts use the full [0, 1]
            # Include the rounded sin/cos square at the half-chart switch.
            h_iv = form.range(0.0, 0.5 + 4 * _EPS if mode == "both_sine" else 1.0)
            radicand = sorted((extent * h_iv[0], extent * h_iv[1]))
            d_r = abs(extent) * _val_err(form)
            piece_root = math.sqrt(d_r)
            if radicand[0] > 2.0 * d_r:
                piece_root = min(piece_root, d_r / math.sqrt(radicand[0] - d_r))
            d_root = max(d_root, piece_root)
            root_max = max(root_max, math.sqrt(max(radicand[1], 0.0)))
    s_max = (b_max + root_max) / (2.0 * a)
    s_error = ((d_b + d_root) / (2.0 * a)
               + 32.0 * _EPS * (b_max + root_max) / (2.0 * a)      # cancellation-safe root formula (c / q switch)
               + _EPS * s_max)                                       # rounding of A
    w2 = np.asarray(support.ruling)
    w1 = np.asarray(support.directrix_vector)
    origin = np.asarray(support.float_origin)
    vector_error = (8.0 * math.sqrt(3.0) * _EPS * (float(np.max(np.abs(origin))) + support.length * float(np.max(np.abs(w1)))
                                                  + s_max * float(np.max(np.abs(w2))))
                    + math.sqrt(3.0) * (4 + 8) * _EPS * support.length * float(np.max(np.abs(w1))))
    fold_error = 0.0
    if mode != "linear":
        # The shifted fold curve (anchor only a double) has the exact constant offset Q = -D(anchor)/(4A); its distance
        # in the stored plane to the exact curve is the NONLINEAR bound T of fold_distance_bound (exact rational
        # E, in-plane gradient lower bound on the shifted tube, Hessian norm 2), not a first-order Q / |grad Q|.
        fold_error = fold_distance_bound(plan, support, cylinder, fold_anchors, (lo, hi))
        if not math.isfinite(fold_error):
            return math.inf, {"reason": "fold_distance_unproved", "mode": mode}
    end_error = 0.0
    if with_ends:
        # Chart-end events are exact roots rounded to the nearest double: |dt| <= delta (4 eps, stated). The world
        # displacement over that t-interval is bounded in the t coordinate (NOT through the public parameter, whose
        # sweep and fold regularization would be mixed in): delta (L |w1| + |w2| B'_max / 2A) plus the sqrt step
        # |w2| min(sqrt(dD), dD / sqrt(D_lo - dD)) / 2A with dD = D'_max delta + the evaluation error of D.
        delta = 4.0 * _EPS
        for end, t_event in ((0, t_a), (1, t_b)):
            if curve._is_fold_end(end):
                continue
            lo_e, hi_e = max(0.0, t_event - delta), min(1.0, t_event + delta)
            d_local = d_form.range(lo_e, hi_e)
            d1_local = max(abs(v) for v in fs.D_forms[1].range(lo_e, hi_e))
            b1_local = max(abs(v) for v in fs.B_forms[1].range(lo_e, hi_e))
            step_d = d1_local * 2.0 * delta + d_d
            step_root = math.sqrt(step_d)
            if d_local[0] > 2.0 * step_d:
                step_root = min(step_root, step_d / math.sqrt(d_local[0] - step_d))
            end_error = max(end_error, delta * (support.length * float(np.linalg.norm(w1))
                                                + float(np.linalg.norm(w2)) * b1_local / (2.0 * a))
                            + float(np.linalg.norm(w2)) * step_root / (2.0 * a))
    bound = float(np.linalg.norm(w2)) * s_error + vector_error + fold_error + end_error
    return bound, {"s_error": s_error, "vector_error": vector_error, "fold_error": fold_error,
                   "end_error": end_error, "mode": mode}


# ---------------------------------------------------------------------------


def plane_ruled_cylinder_support(plane, cylinder, *, tolerance=1e-10, cancellation_check=None, face_pair=None):
    """All branches of ``plane`` cut with the finite ``cylinder`` as :class:`BezierQuadricCurve` charts."""
    support, s_lo, s_hi = build_plane_ruled_support(plane, cylinder, tolerance=tolerance, face_pair=face_pair)
    quarter = .25 * float(tolerance)
    retention = max(100.0 * float(tolerance), 1e-6)
    plan = get_poly_plan(support, cylinder)
    if plan.coincident:
        _refuse("a non-parallel plane cannot coincide with a cylinder", "coincident", face_pair)
    if plan.linear or not plan.floats().a0 > 0.0:
        _refuse("a non-parallel plane has a positive quadratic ruling equation against a cylinder", "linear_branch",
                face_pair)
    _poll(cancellation_check)
    try:
        roots = plan.discriminant_roots()
    except GeometryError as error:
        _refuse(f"the plane touches the cylinder along a whole curve: {error}", "double_branch", face_pair)
    if any(m > 2 for _t, m in roots):
        _refuse("higher-order contact of plane and cylinder", "cusp", face_pair)
    folds = [(t, m) for t, m in roots if m == 1]
    nodes = [(t, m) for t, m in roots if m == 2]
    fold_ts = _in_range(folds, 0.0, 1.0)
    node_ts = _in_range(nodes, 0.0, 1.0)
    structural = [0.0, 1.0, *fold_ts, *node_ts]
    fs = plan.floats()
    boundaries = _boundary_planes(cylinder)
    if abs(abs(float(cylinder.sweep_angle)) - _TAU) <= 1e-12:
        # FULL sweep: angular membership is identically true (patch_margins returns inf under the same predicate), so
        # the two seam planes (start y=0 and the stored end plane with sin(2 pi) ~ -2.45e-16) carry no boundary of the
        # patch; their roots would only add genuinely distinct nearby events. Keep the two height planes. Partial
        # sweeps keep all four planes. Face boundaries are applied later by the material-domain clipping.
        boundaries = boundaries[:2]
    candidate_events = []                         # (t, boundary index)
    for index, boundary in enumerate(boundaries):
        _poll(cancellation_check)
        boundary_roots = plan.resultant_roots(boundary)
        if boundary_roots is None:
            continue          # the whole branch lies in this plane: decided exactly by the coincidence test below
        candidate_events.extend((t, index) for t in _in_range(boundary_roots, 0.0, 1.0))
    # Global evaluation error of the middle-point classification (linear model over the whole chart: the fold charts
    # are only more accurate). Classification beyond +-(quarter + error) is certified; inside, it must be proven.
    class_error = 0.0
    for branch in (-1, 1):
        whole = BezierQuadricCurve._make(support, plan.second, 0.0, 1.0, branch, "linear")
        bound, details = curve_error_bound(plan, support, whole, cylinder, with_ends=False)
        if not math.isfinite(bound):
            _refuse("the evaluation error of the plane chart is not bounded", "unproved_evaluation", face_pair,
                    branch=branch, **details)
        class_error = max(class_error, bound)
    z_pad = 4.0 * (quarter + class_error) + 1e-12
    event_polys = {None: _ptrim(plan.disc)}

    def polys_for(tags):
        result = {}
        for tag in tags:
            if tag is None:
                result[None] = event_polys[None]
            elif isinstance(tag, int):
                if tag not in event_polys:
                    event_polys[tag] = _boundary_resultant(plan, boundaries[tag])
                result[tag] = event_polys[tag]
        return result

    def point_status(point, error):
        """True/False certified membership of the TRUE finite patch at the actual tolerance, else typed refusal."""
        return certify_point_membership(cylinder, point, error, tolerance, face_pair=face_pair)

    def local_error(lo_t, hi_t, branch):
        lo_t, hi_t = max(0.0, lo_t), min(1.0, hi_t)
        if hi_t <= lo_t:
            lo_t, hi_t = max(0.0, lo_t - 1e-12), min(1.0, lo_t + 1e-12)
        pseudo = BezierQuadricCurve._make(support, plan.second, lo_t, hi_t - lo_t, branch, "linear")
        return curve_error_bound(plan, support, pseudo, cylinder, with_ends=False, z_pad=z_pad)[0]

    def fold_point_error():
        """Error of the fold point (anchor, -B/2A): no sqrt is evaluated; the anchor itself is within 4 eps of the root."""
        delta = 4.0 * _EPS
        b_form = fs.B_forms[0]
        b_all = b_form.range(0.0, 1.0)
        b1 = max(abs(v) for v in fs.B_forms[1].range(0.0, 1.0))
        w1n, w2n = float(np.linalg.norm(support.directrix_vector)), float(np.linalg.norm(support.ruling))
        s_reach = max(abs(b_all[0]), abs(b_all[1])) / (2.0 * fs.a0)
        origin_n = float(np.max(np.abs(support.float_origin)))
        vector = (8.0 * math.sqrt(3.0) * _EPS * (origin_n + support.length * w1n + s_reach * w2n)
                  + math.sqrt(3.0) * 12.0 * _EPS * support.length * w1n)
        return (w2n * ((_val_err(b_form) + b1 * (delta + _EPS)) / (2.0 * fs.a0) + _EPS * s_reach)
                + support.length * w1n * delta + vector)

    curves, contacts = [], []
    for branch in (-1, 1):
        events = [(t, None if (t in fold_ts or t in node_ts) else "window") for t in structural]
        if candidate_events:
            ts = np.array([t for t, _i in candidate_events])
            s = _branch_values(plan, ts, branch)
            pts = _points(plan, ts, s)
            if not (np.all(np.isfinite(s)) and np.all(np.isfinite(pts))):
                _refuse("a boundary event of the branch is not finite", "nonfinite_event", face_pair, branch=branch)
            for (t, index), point in zip(candidate_events, pts):
                if min(patch_margins(cylinder, point)) >= -retention:      # event over-retention is always safe
                    events.append((t, index))
        ordered = sorted(events, key=lambda item: item[0])
        reps, group = [], [ordered[0]]
        for item in ordered[1:] + [(math.inf, "end")]:
            if item[0] - group[-1][0] <= _TOLERANCE:
                group.append(item)
                continue
            group_t = [g[0] for g in group]
            fold_in = next((f for f in (*fold_ts, *node_ts) if any(abs(f - g) <= _TOLERANCE for g in group_t)), None)
            rep = fold_in if fold_in is not None else group_t[0]
            if len(group) > 1:      # equal float witnesses are NOT identity either: distinct roots may round together
                # Distinct witnesses are the SAME event only by exact root identity (Sturm counts on one isolating
                # interval + exact gcd, for EVERY compared root); never by 1e-12 proximity or equal witnesses.
                resolved = polys_for({tag for _t, tag in group if tag is None or isinstance(tag, int)})
                exact = events_identical(resolved, group)
                if not exact:
                    # OUTWARD root brackets of every candidate (verified by Sturm), never the bare float witnesses:
                    # distinct exact roots may share one float and a zero-width witness span proves nothing
                    interval = group_interval(resolved, group)
                    if interval is None:
                        _refuse("clustered chart events cannot be bracketed by their defining polynomials",
                                "event_cluster_unresolved", face_pair, branch=branch,
                                chart=(group_t[0], group_t[-1]), tags=[str(tag) for _t, tag in group])
                    extent = cluster_extent_metres(support, fs, interval[0], interval[1])
                    if not extent <= quarter:
                        tags = [tag for _t, tag in group]
                        counts = {}
                        try:                      # diagnostics only: distinct roots of each polynomial in the group interval
                            lo_i, hi_i = Fraction(group_t[0]) - Fraction(_TOLERANCE), Fraction(group_t[-1]) + Fraction(_TOLERANCE)
                            for tag, poly_ in polys_for({x for x in tags if x is None or isinstance(x, int)}).items():
                                counts[str(tag)] = sturm_count(poly_, lo_i, hi_i)
                        except Exception as error:      # noqa: BLE001 - never mask the refusal
                            counts["error"] = repr(error)
                        _refuse("clustered chart events span more than a quarter of the face tolerance",
                                "event_cluster", face_pair, branch=branch, chart=(group_t[0], group_t[-1]),
                                metres=extent, quarter_tolerance=quarter, tags=[str(x) for x in tags],
                                witnesses=group_t, roots_in_interval=counts)
                    point = _points(plan, [rep], _branch_values(plan, [rep], branch))[0]
                    if point_status(point, local_error(interval[0], interval[1], branch)):
                        contacts.append(tuple(float(v) for v in point))        # recorded, not erased
            reps.append(rep)
            group = [item]
        for e0, e1 in zip(reps, reps[1:]):
            _poll(cancellation_check)
            mid = .5 * (e0 + e1)
            d_mid, d_err = fs.discriminant_scalar(float(mid)), (4 * fs.D_forms[0].degree + 8) * _EPS * fs.D_forms[0].bound
            if d_mid < -d_err:
                continue                          # sign of D is constant between its roots: no real branch here
            if d_mid <= d_err:
                _refuse("the discriminant is not resolved at the middle of a chart piece", "discriminant_ambiguous",
                        face_pair, branch=branch, chart=(e0, e1), value=d_mid, error=d_err)
            left = any(abs(e0 - f) <= _TOLERANCE for f in fold_ts)
            right = any(abs(e1 - f) <= _TOLERANCE for f in fold_ts)
            mode = "both_sine" if left and right else "left_square" if left else "right_square" if right else "linear"
            curve = BezierQuadricCurve._make(support, plan.second, e0, e1 - e0, branch, mode)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                ends = curve.evaluate(np.array([0., .5, 1.]))
            if not np.all(np.isfinite(ends)):
                _refuse("a branch chart evaluates to a non-finite point", "nonfinite_piece", face_pair,
                        branch=branch, chart=(e0, e1))
            keep = _classify_piece(plan, support, fs, curve, ends[1], e0, e1, branch, cylinder, boundaries, quarter,
                                   class_error, face_pair)
            if not keep:
                continue
            if abs(e0) <= _TOLERANCE or abs(e1 - 1.0) <= _TOLERANCE:
                _refuse("a piece inside the patch reaches the end of the chart window", "window_end", face_pair,
                        branch=branch, chart=(e0, e1))
            s_values = [float(support.ruling_coordinate(p)) for p in ends]
            if min(s_values) < s_lo or max(s_values) > s_hi:
                _refuse("a piece inside the patch leaves the ruling window", "window_ruling", face_pair,
                        branch=branch, chart=(e0, e1))
            if np.ptp(ends, axis=0).max() <= tolerance:
                piece_error = curve_error_bound(plan, support, curve, cylinder, with_ends=False, z_pad=z_pad)[0]
                if point_status(ends[1], piece_error):
                    contacts.append(tuple(float(v) for v in ends[1]))          # a sub-tolerance piece is a point
                continue
            curves.append(curve)
    points = []
    for t in (*fold_ts, *node_ts):
        a, b, _c = fs.abc_scalar(float(t))
        point = _points(plan, [t], [-b / (2 * a)])[0]
        if not np.all(np.isfinite(point)):
            _refuse("a fold point is not finite", "nonfinite_fold", face_pair, t=float(t))
        if point_status(point, fold_point_error()):
            points.append(tuple(float(v) for v in point))
    points.extend(contacts)
    kept = []
    for point in points:
        if not any(np.linalg.norm(np.asarray(point) - curve.evaluate(u)) <= tolerance
                   for curve in curves for u in (0., 1.)) and not any(
                np.linalg.norm(np.asarray(point) - np.asarray(other)) <= tolerance for other in kept):
            kept.append(point)
    if plan.common_roots():
        _refuse("a whole ruling of the plane lies on the cylinder", "common_ruling", face_pair)
    _validate_curves(plan, support, curves, cylinder, quarter, face_pair, z_pad)
    _increment("plane_chart_curves", len(curves))
    _increment("plane_chart_points", len(kept))
    return SupportIntersection(curves=tuple(curves), points=tuple(kept))


def _boundary_resultant(plan, boundary):
    """Exact ascending Fraction coefficients of the resultant of the branch with a boundary plane (zero poly -> [])."""
    from .branch_algebra import _resultant, _trim, build_abc
    a3, b3, c3 = build_abc(plan.first, QuadricSupport.from_surface(boundary))
    poly = _ptrim(_trim(_resultant(plan.A, plan.B, plan.C, a3, b3, c3)))
    return poly


# -- exact univariate arithmetic over the rationals (ascending coefficients), for root identity

def _ptrim(poly):
    poly = [Fraction(c) for c in poly]
    while poly and poly[-1] == 0:
        poly.pop()
    return poly


def _peval(poly, x):
    value = Fraction(0)
    for c in reversed(poly):
        value = value * x + c
    return value


def _pderiv(poly):
    return [i * c for i, c in enumerate(poly)][1:]


def _prem(a, b):
    a = _ptrim(a)
    b = _ptrim(b)
    while len(a) >= len(b) and a:
        factor = a[-1] / b[-1]
        shift = len(a) - len(b)
        for i, c in enumerate(b):
            a[i + shift] -= factor * c
        a = _ptrim(a[:-1] if a[-1] == 0 else a)
    return a


def _pgcd(a, b):
    a, b = _ptrim(a), _ptrim(b)
    while b:
        a, b = b, _prem(a, b)
    return a


def _pquot(a, b):
    a, b = _ptrim(a), _ptrim(b)
    quotient = [Fraction(0)] * max(len(a) - len(b) + 1, 1)
    while len(a) >= len(b) and a:
        factor = a[-1] / b[-1]
        shift = len(a) - len(b)
        quotient[shift] = factor
        for i, c in enumerate(b):
            a[i + shift] -= factor * c
        a = _ptrim(a[:-1] if a[-1] == 0 else a)
    return quotient


def sturm_count(poly, lo, hi):
    """Number of DISTINCT real roots of ``poly`` in ``(lo, hi]`` (exact Sturm chain of the squarefree part)."""
    poly = _ptrim(poly)
    if len(poly) < 2:
        return 0
    common = _pgcd(poly, _pderiv(poly))
    if len(common) > 1:
        poly = _ptrim(_pquot(poly, common))
    chain = [poly, _ptrim(_pderiv(poly))]
    while len(chain[-1]) > 1:
        remainder = _prem(chain[-2], chain[-1])
        if not remainder:
            break
        chain.append([-c for c in remainder])

    def variations(x):
        signs = [v for v in (_peval(q, x) for q in chain) if v != 0]
        return sum(1 for u, w in zip(signs, signs[1:]) if (u > 0) != (w > 0))

    return variations(Fraction(lo)) - variations(Fraction(hi))


def events_identical(polys, events):
    """Exact identity of grouped chart events: do they all denote ONE real root of their defining polynomials?

    ``polys`` maps an event tag to the exact rational polynomial the event is a root of (the discriminant for a fold,
    a resultant for a boundary plane); ``events`` are ``(float witness, tag)``. Never float proximity or equal
    witnesses: (1) every witness is verified to have a true root of ITS polynomial within 4 eps (Sturm count >= 1 on
    ``[t - 4 eps, t + 4 eps]``); (2) on the common isolating interval ``I`` every involved polynomial has exactly
    ONE distinct real root (so a distinct nearby root of any of them forces ``False``); (3) the exact gcd of all of
    them has a root in ``I`` - hence the unique roots coincide and every witness is that root.
    """
    if not events or any(tag not in polys for _t, tag in events):
        return False
    pad = Fraction(_TOLERANCE)             # the grouping scale; soundness does not depend on it (checks below)
    for t, tag in events:
        x = Fraction(float(t))
        if sturm_count(polys[tag], x - pad, x + pad) < 1:
            return False
    values = [Fraction(float(t)) for t, _tag in events]
    lo, hi = min(values) - pad, max(values) + pad
    common = None
    for tag in {tag for _t, tag in events}:
        if sturm_count(polys[tag], lo, hi) != 1:
            return False
        common = _ptrim(polys[tag]) if common is None else _pgcd(common, polys[tag])
    return len(common) > 1 and sturm_count(common, lo, hi) == 1


def group_interval(polys, events):
    """Outward float interval containing the true root of EVERY grouped event, or ``None`` when unverifiable.

    Each witness is verified (Sturm count >= 1) to have a true root of its own polynomial within 1e-12; the interval
    is the union of those brackets (window ends are exact values), rounded outward.
    """
    pad = Fraction(_TOLERANCE)
    lows, highs = [], []
    for t, tag in events:
        x = Fraction(float(t))
        if tag == "window":
            lows.append(x)
            highs.append(x)
            continue
        if tag not in polys or sturm_count(polys[tag], x - pad, x + pad) < 1:
            return None
        lows.append(x - pad)
        highs.append(x + pad)
    lo, hi = float(min(lows)), float(max(highs))
    if Fraction(lo) > min(lows):
        lo = math.nextafter(lo, -math.inf)
    if Fraction(hi) < max(highs):
        hi = math.nextafter(hi, math.inf)
    return lo, hi


def _sqrt_up_certified(x):
    """Float ``r`` with ``Fraction(r)**2 >= x`` (bounded post-check), ``inf`` on underflow/overflow/failure."""
    x = Fraction(x)
    if x == 0:
        return 0.0
    f = float(x)
    if not math.isfinite(f) or f < 1e-290:
        return math.inf
    r = math.sqrt(f)
    for _ in range(4):
        if Fraction(r) ** 2 >= x:
            return r
        r = math.nextafter(r, math.inf)
    return math.inf


def _sqrt_down_certified(x):
    """Float ``g`` with ``Fraction(g)**2 <= x`` (bounded post-check), ``0.0`` when it cannot be certified."""
    x = Fraction(x)
    f = float(x)
    if x <= 0 or not math.isfinite(f) or f < 1e-290:
        return 0.0
    g = math.sqrt(f)
    for _ in range(4):
        if Fraction(g) ** 2 <= x:
            return g
        g = math.nextafter(g, 0.0)
    return 0.0


def _div_up_certified(numerator, denominator):
    """Float ``>= numerator / denominator`` exactly (``numerator`` a Fraction, ``denominator`` a positive float);
    a positive quotient never underflows to zero; ``inf`` when it cannot be certified."""
    exact = Fraction(numerator) / Fraction(denominator)
    bound = float(exact)
    if not math.isfinite(bound):
        return math.inf
    for _ in range(4):
        if Fraction(bound) >= exact:
            return bound
        bound = math.nextafter(bound, math.inf)
    return math.inf


def certify_point_membership(cylinder, point, error, tolerance, *, face_pair=None):
    """Certified membership of an emitted point in the true finite patch at the ACTUAL tolerance.

    ``error`` bounds the Euclidean distance from ``point`` to the exact curve (hence both support residuals). The
    margins are metres to the native bounds: axial changes by at most ``error``; the angular arc changes by at most
    ``r asin(error / r)``; plus the rounding of the margin computation itself. Accept (True) iff the margin minus its
    error is >= -tolerance, reject (False) iff the margin plus its error is < -tolerance, otherwise refuse.
    """
    radius = float(cylinder.radius)
    if not math.isfinite(error) or error > .25 * float(tolerance):
        _refuse("a point's position error exceeds a quarter of the face tolerance", "point_unproved", face_pair,
                error=float(error), quarter_tolerance=.25 * float(tolerance))
    p = np.asarray(point, dtype=float)
    origin = np.asarray(cylinder.origin, dtype=float)
    # Arithmetic model of patch_margins (u = eps = 2^-53 rounding unit, gamma_n = n u / (1 - n u) ~ n u):
    #  rel = p - o:   |d rel_i| <= u |rel_i|;   z = rel . a (3 products, 2 sums): |d z| <= gamma_5 (|rel| . |a|) with
    #  |a| ~ 1  =>  |d z| <= 5 u ||rel||_1 + (u from the first subtraction); the radial vector rel - z a adds
    #  gamma_2-level terms: each component error <= 8 u (||p||_1 + ||o||_1 + ||rel||_1) =: e_vec.
    rel1 = float(np.sum(np.abs(p - origin)))
    e_vec = 8.0 * _EPS * (float(np.sum(np.abs(p))) + float(np.sum(np.abs(origin))) + rel1)
    axial, angular = patch_margins(cylinder, p)
    # hi - z and z - lo each add one rounding of a quantity of size <= |height| + rel1
    e_axial = (error + e_vec) + _EPS * (abs(float(cylinder.height)) + rel1)
    e_angular = 0.0
    if math.isfinite(angular):
        rho_min = radius - error - e_vec            # the point lies at least rho_min from the axis
        if not rho_min > 0.0:
            _refuse("the position error is not small against the cylinder radius", "point_unproved", face_pair,
                    error=float(error), radius=radius)
        # a displacement of at most (error + e_vec) changes the azimuth by at most asin((error + e_vec) / rho_min);
        # atan2 is accurate to 2 ulp of pi (2 u pi); (angle - start) mod 2 pi adds rounding of a quantity of size
        # <= 3 pi + |start| (two operations: 2 u (3 pi + |start|)); the arc margin multiplies by r (one u relative).
        d_theta = (math.asin(min(1.0, (error + e_vec) / rho_min)) + 2.0 * _EPS * math.pi
                   + 2.0 * _EPS * (3.0 * math.pi + abs(float(cylinder.start_angle))))
        e_angular = radius * d_theta * (1.0 + _EPS)
    lower = min(axial - e_axial, angular - e_angular)
    upper = min(axial + e_axial, angular + e_angular)
    if lower >= -tolerance:
        return True
    if upper < -tolerance:
        return False
    _refuse("a point contact is within its error of the patch boundary at the tolerance", "point_ambiguous",
            face_pair, margin=min(axial, angular), error=float(error), tolerance=float(tolerance))


def _classify_piece(plan, support, fs, curve, mid_point, e0, e1, branch, cylinder, boundaries, quarter, class_error,
                    face_pair):
    """Certified classification: keep beyond ``+(quarter+error)``, discard beyond ``-(quarter+error)`` of the native
    bounds (the midpoint is evaluated with at most ``class_error``); inside that band the piece must be PROVEN to lie
    in a boundary plane (closed patch) or the pair is refused."""
    band = quarter + class_error
    axial, angular = patch_margins(cylinder, mid_point)
    margin = min(axial, angular)
    if margin >= band:
        return True
    if margin <= -band:
        return False
    lo, hi = min(e0, e1), max(e0, e1)
    coincident = []
    for index, boundary in enumerate(boundaries):
        normal = np.asarray(boundary.axis, dtype=float)
        at_mid = abs(float(normal @ (np.asarray(mid_point) - np.asarray(boundary.origin)))) / float(np.linalg.norm(normal))
        if at_mid > quarter:
            continue                      # the continuity argument needs the small side at the middle
        if coincidence_bound(support, plan, boundary, lo, hi, mid_world_distance=at_mid + class_error) <= quarter:
            coincident.append(index)
    if not coincident:
        _refuse("a piece lies within the tolerance band of the patch boundary without being a boundary trace",
                "boundary_ambiguous", face_pair, branch=branch, chart=(e0, e1), axial_margin=axial,
                angular_margin=angular, band=band)
    height_boundary = any(index < 2 for index in coincident)
    angular_boundary = any(index >= 2 for index in coincident)
    # the other constraint must be decided; a corner (both in the band) is ambiguous
    if (height_boundary and not angular_boundary and angular < band) or (
            angular_boundary and not height_boundary and axial < band):
        _refuse("a boundary-coincident piece also touches the other patch bound within tolerance", "boundary_corner",
                face_pair, branch=branch, chart=(e0, e1), axial_margin=axial, angular_margin=angular, band=band)
    _increment("plane_chart_boundary_coincident")
    return True


def _validate_curves(plan, support, curves, cylinder, quarter, face_pair, z_pad=1e-6):
    """Whole-curve derived bound (not a sample certificate): refuse any curve whose bound exceeds a quarter tolerance."""
    for curve in curves:
        lo, hi = curve.bounds()
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            _refuse("the interval enclosure of a branch chart is unbounded", "unbounded_enclosure", face_pair,
                    chart=(curve.start, curve.start + curve.sweep))
        bound, details = curve_error_bound(plan, support, curve, cylinder, z_pad=z_pad)
        if not bound <= quarter:
            _refuse("the derived evaluation bound of a branch chart exceeds a quarter of the face tolerance",
                    "unproved_evaluation", face_pair, chart=(curve.start, curve.start + curve.sweep),
                    metres=bound, quarter_tolerance=quarter, **details)
