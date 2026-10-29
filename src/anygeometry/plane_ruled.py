"""A stored Plane as the first (ruled) support of a polynomial-chart Plane/Cylinder intersection.

``Plane`` is exactly ``origin + u*u_vector + v*v_vector``. For a plane that is only *nearly* parallel to a
cylinder axis the elliptic-arc form divides by a rounding-level axial coefficient. Here the intersection is
instead the ``BezierQuadricCurve`` branch of the ruled surface ``c(t) + s*d`` with

* ``c(t) = base + (t0 + length*t) * directrix_vector`` (a degree-1 directrix, ``t`` in ``[0, 1]``),
* ``d = ruling`` (the other stored plane vector),

so the first support is the *original* stored plane, not a rounded re-derivation of it. The exact origin of the
chart ``base + t0*directrix_vector`` is a rational point; the polynomial algebra receives it unrounded (the
``origin`` property) and only the float evaluator rounds it once, to the nearest double (an absolute error of at
most half an ulp per coordinate, see :meth:`PlaneRuledSupport.float_origin`).

This module also owns the exact (``Fraction``) route classification, the exact finite window of the chart, the
typed refusal, and the a-priori error bound that decides whether the ``EllipticArc`` route is still admissible.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from .errors import GeometryError
from .quadric_algebra import _fraction_cross, _fraction_vec

_EPS = float(np.finfo(float).eps)

# The ruling must keep at least this fraction of its length perpendicular to the cylinder axis, so that the
# leading coefficient A = |d|^2 - (d.axis)^2 of the ruling quadratic is >= MIN_PERPENDICULAR**2 |d|^2.
MIN_PERPENDICULAR = 1e-3

PLANE_RULED_CAPABILITY = "plane_ruled_support/1"


class PlaneRuledSupportError(GeometryError):
    """A typed refusal of the plane-chart route; ``reason`` is a stable key and ``details`` the numbers."""

    def __init__(self, message, *, reason, **details):
        super().__init__(message)
        self.reason = str(reason)
        self.details = dict(details)

    def to_diagnostic(self):
        return {"error": "plane_ruled_support", "reason": self.reason, "message": str(self), **self.details}


def _vec(value, name):
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise GeometryError(f"{name} must be a finite 3-vector") from error
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise GeometryError(f"{name} must be a finite 3-vector")
    return result


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@dataclass(frozen=True, slots=True)
class PlaneRuledSupport:
    """The ruled surface ``c(t) + s d`` of a stored plane (hashable cache key; see the module docstring)."""
    base: tuple
    directrix_vector: tuple
    ruling: tuple
    t0: float
    length: float

    def __post_init__(self):
        base = _vec(self.base, "plane base")
        w1 = _vec(self.directrix_vector, "plane directrix vector")
        w2 = _vec(self.ruling, "plane ruling")
        t0, length = float(self.t0), float(self.length)
        if not (math.isfinite(t0) and math.isfinite(length)) or length <= 0.0:
            raise GeometryError("a plane ruled support needs a finite window with positive length")
        normal = _fraction_cross(_fraction_vec(w1), _fraction_vec(w2))
        if not any(normal):
            raise GeometryError("the directrix vector and the ruling of a plane support must be independent")
        for name, value in (("base", base), ("directrix_vector", w1), ("ruling", w2), ("t0", t0), ("length", length)):
            object.__setattr__(self, name, value)

    @property
    def degree(self):
        return 1

    @property
    def origin(self):
        """The exact rational chart origin ``base + t0 * directrix_vector`` (a tuple of Fractions)."""
        t0 = Fraction(self.t0)
        return tuple(Fraction(b) + t0 * Fraction(w) for b, w in zip(self.base, self.directrix_vector))

    @property
    def float_origin(self):
        """The nearest doubles of :attr:`origin` (absolute error <= half an ulp of each coordinate)."""
        return tuple(float(value) for value in self.origin)

    @property
    def direction(self):
        return self.ruling

    def relative_controls(self):
        """Exact control points relative to :attr:`origin`: ``[0, length * directrix_vector]``."""
        length = Fraction(self.length)
        return ((Fraction(0),) * 3, tuple(length * Fraction(w) for w in self.directrix_vector))

    def exact_power(self):
        """``a_0 .. a_1`` of ``c(t) - origin`` as Fraction vectors (``a_0 = 0``, relative to ``origin``)."""
        return [list(row) for row in self.relative_controls()]

    def quadric(self):
        raise GeometryError("a plane ruled support is a plane, not a degree-two quadric ruled support")

    # -- direct plane coordinates (no least squares, no Bezier directrix)

    def _dual(self):
        w1, w2 = _fraction_vec(self.directrix_vector), _fraction_vec(self.ruling)
        n = _fraction_cross(w1, w2)
        nn = sum(c * c for c in n)
        return (tuple(float(c / nn) for c in _fraction_cross(w2, n)),
                tuple(float(c / nn) for c in _fraction_cross(n, w1)))

    def chart_parameter(self, point):
        """The unclipped chart parameter ``t`` of a point of the plane (inverts the stored vectors directly)."""
        lam1, _lam2 = self._dual()
        origin = self.float_origin
        offset = [float(point[i]) - origin[i] for i in range(3)]
        return _dot(lam1, offset) / self.length

    def ruling_coordinate(self, point):
        """The ruling coordinate ``s`` of a point of the plane (measured from :attr:`origin`)."""
        _lam1, lam2 = self._dual()
        origin = self.float_origin
        offset = [float(point[i]) - origin[i] for i in range(3)]
        return _dot(lam2, offset)


# ---------------------------------------------------------------------------
# Exact classification


def exact_axial_determinant(plane, cylinder):
    """``det(u_vector, v_vector, axis)`` of the STORED doubles, exactly (zero iff the plane is parallel to the axis)."""
    u, v, a = (_fraction_vec(x) for x in (plane.u_vector, plane.v_vector, cylinder.axis))
    cross = _fraction_cross(u, v)
    return cross[0] * a[0] + cross[1] * a[1] + cross[2] * a[2]


def exact_parallel(plane, cylinder):
    return exact_axial_determinant(plane, cylinder) == 0


def choose_ruling(plane, cylinder):
    """``(directrix_vector, ruling, perpendicular_fraction)``: the stored vector least aligned with the axis rules."""
    axis = np.asarray(cylinder.axis, dtype=float)
    best = None
    for name, own, other in (("u", plane.u_vector, plane.v_vector), ("v", plane.v_vector, plane.u_vector)):
        vector = np.asarray(own, dtype=float)
        length = float(np.linalg.norm(vector))
        perpendicular = float(np.linalg.norm(vector - float(vector @ axis) * axis))
        fraction = perpendicular / length if length > 0.0 else 0.0
        key = (fraction, name == "u")                        # deterministic tie-break: the u vector rules
        if best is None or key > best[0]:
            best = (key, tuple(float(x) for x in other), tuple(float(x) for x in vector))
    return best[1], best[2], best[0][0]


def _down(value):
    result = float(value)
    return result if Fraction(result) <= value else math.nextafter(result, -math.inf)


def _up(value):
    result = float(value)
    return result if Fraction(result) >= value else math.nextafter(result, math.inf)


def window_margin(cylinder, tolerance):
    """Metres by which the window exceeds the exact projection of the patch (never below 100 tolerances)."""
    return max(1e-2 * (float(cylinder.radius) + abs(float(cylinder.height))), 100.0 * float(tolerance), 1e-6)


def build_plane_ruled_support(plane, cylinder, *, tolerance, face_pair=None):
    """``(support, s_lo, s_hi)``: the chart support whose window holds the projection of the whole finite patch.

    The rational window is ``[t_lo, t_hi] x [s_lo, s_hi]`` in the coordinates ``point = base + t w1 + s w2`` of
    the stored plane, computed exactly: every in-plane point ``p`` of the patch has ``t = lam1.(p - base)`` and
    ``s = lam2.(p - base)`` with the dual vectors ``lam1 = (w2 x n)/|n|^2``, ``lam2 = (n x w1)/|n|^2``, and for
    ``p = o + z a + r (cos e1 + sin e2)``, ``z`` between ``0`` and ``h``, the functional ``lam.p`` lies within
    ``lam.(o - base) + [min(0, h lam.a), max(0, h lam.a)] +- r (|lam.e1| + |lam.e2|)`` (exact rational, no
    square root). The window is then widened by :func:`window_margin` and rounded OUTWARD to doubles.
    """
    w1, w2, fraction = choose_ruling(plane, cylinder)
    if fraction < MIN_PERPENDICULAR:
        raise PlaneRuledSupportError(
            "the plane's stored vectors are all within 1e-3 radian of the cylinder axis: the plane chart is ill-conditioned",
            reason="ruling_conditioning", face_pair=face_pair, perpendicular_fraction=fraction,
            minimum=MIN_PERPENDICULAR)
    f1, f2 = _fraction_vec(w1), _fraction_vec(w2)
    n = _fraction_cross(f1, f2)
    nn = sum(c * c for c in n)
    lam = (tuple(c / nn for c in _fraction_cross(f2, n)), tuple(c / nn for c in _fraction_cross(n, f1)))
    base = _fraction_vec(plane.origin)
    origin = _fraction_vec(cylinder.origin)
    axis = _fraction_vec(cylinder.axis)
    e1, e2 = _fraction_vec(cylinder.radial_direction), _fraction_vec(cylinder.circumferential_direction)
    radius, height = Fraction(cylinder.radius), Fraction(cylinder.height)
    margin = Fraction(window_margin(cylinder, tolerance))
    relative = tuple(o - b for o, b in zip(origin, base))
    spans = []
    for functional in lam:
        centre = sum(functional[i] * relative[i] for i in range(3))
        along = height * sum(functional[i] * axis[i] for i in range(3))
        radial = radius * (abs(sum(functional[i] * e1[i] for i in range(3)))
                           + abs(sum(functional[i] * e2[i] for i in range(3))))
        pad = margin * sum(abs(c) for c in functional)
        spans.append((centre + min(Fraction(0), along) - radial - pad, centre + max(Fraction(0), along) + radial + pad))
    (t_lo, t_hi), (s_lo, s_hi) = spans
    t0 = _down(t_lo)
    length = float(t_hi - Fraction(t0))
    while Fraction(t0) + Fraction(length) < t_hi:
        length = math.nextafter(length, math.inf)
    if not (math.isfinite(t0) and math.isfinite(length) and length > 0.0):
        raise PlaneRuledSupportError("the plane chart window is not finite", reason="window", face_pair=face_pair)
    support = PlaneRuledSupport(tuple(plane.origin), w1, w2, t0, length)
    return support, _down(s_lo), _up(s_hi)


def support_residuals(point, plane, cylinder):
    """``(plane_distance, cylinder_distance)`` of a float point to the STORED supports, evaluated in exact rationals."""
    p, o = _fraction_vec(point), _fraction_vec(plane.origin)
    n = _fraction_cross(_fraction_vec(plane.u_vector), _fraction_vec(plane.v_vector))
    nn = sum(c * c for c in n)
    plane_distance = abs(sum((p[i] - o[i]) * n[i] for i in range(3))) / Fraction(math.sqrt(float(nn)))
    w = [p[i] - Fraction(cylinder.origin[i]) for i in range(3)]
    a = _fraction_vec(cylinder.axis)
    along = sum(w[i] * a[i] for i in range(3)) / sum(c * c for c in a)
    perp = [w[i] - along * a[i] for i in range(3)]
    radius = Fraction(cylinder.radius)
    g = abs(sum(c * c for c in perp) - radius * radius)
    # | |perp| - r | = |g| / (|perp| + r) <= |g| / r
    return float(plane_distance), float(g / radius)


def qualify_plane_cylinder_result(result, plane, cylinder, tolerance, face_pair):
    """Refuse (typed, with the pair, residual and tolerance) any returned curve/segment/point off either support.

    Run per pair BEFORE the result is accumulated. Curve stations are the ends and the middle; the whole-curve
    guarantee is the producing route's own bound (:func:`elliptic_route_bound` or the plane-chart
    ``curve_error_bound``), this is the independent exact-support gate.
    """
    stations = [tuple(point) for point in result.points]
    for segment in result.segments:
        stations.extend(tuple(end) for end in segment)
    for curve in result.curves:
        stations.extend(tuple(float(v) for v in curve.evaluate(u)) for u in (0.0, 0.5, 1.0))
    worst = (0.0, 0.0)
    for point in stations:
        plane_distance, cylinder_distance = support_residuals(point, plane, cylinder)
        worst = (max(worst[0], plane_distance), max(worst[1], cylinder_distance))
        if plane_distance > tolerance or cylinder_distance > tolerance:
            raise PlaneRuledSupportError(
                "an intersection result lies off one of its stored supports",
                reason="support_residual", face_pair=face_pair, plane_residual=plane_distance,
                cylinder_residual=cylinder_distance, tolerance=float(tolerance), point=tuple(point))
    return worst


# ---------------------------------------------------------------------------
# Admission of the EllipticArc route (a-priori, over the whole closed curve)


def elliptic_route_bound(plane, cylinder, c0, u_vec, v_vec, *, limit):
    """Metre-level bound for ``EllipticArc(c0, u_vec, v_vec)`` against the STORED plane and cylinder supports.

    ``bound = floating + residual`` with

    * ``floating = 16 eps (S + R)`` with ``S = max_i(|o_i| + |c0_i| + |u_i| + |v_i|)`` (rounding of the
      evaluation ``c0 + cos u + sin v``: at most seven operations of relative error eps/2 plus the cos/sin error,
      per coordinate) and ``R = |u| + |v|`` the speed bound, multiplying the event-angle error (<= 4 eps, the
      isolator's angular tolerance; a grazing event is not covered, see the plane-chart route for that);
    * ``residual``: the exact supremum over EVERY angle (not samples) of the distance bound to both supports of
      the float vectors read as exact rationals: ``(|N.(c0-o_p)| + |N.u| + |N.v|)/|N|`` for the plane (``N`` the
      exact cross product of the stored plane vectors) and ``(|g0|+|g1|+|g2|+|g3|+|g4|)/r`` for the cylinder,
      where ``g`` is the exact trigonometric polynomial ``|P_perp|^2 - r^2`` (``| |P_perp| - r | = |g|/(|P_perp|+r)``).

    The exact part is only computed when ``floating <= limit``; otherwise ``floating`` already refuses.
    """
    vectors = [np.asarray(item, dtype=float) for item in (c0, u_vec, v_vec)]
    if not all(np.all(np.isfinite(item)) for item in vectors):
        return math.inf
    c0, u_vec, v_vec = vectors
    with np.errstate(all="ignore"):
        # per-coordinate sum of the magnitudes of every term of c0 + cos u + sin v (infinity norm) ...
        scale = float(np.max(np.abs(np.asarray(cylinder.origin)) + np.abs(c0) + np.abs(u_vec) + np.abs(v_vec)))
        # ... and the Euclidean speed bound of the curve, for the event-angle error
        reach = float(np.linalg.norm(u_vec) + np.linalg.norm(v_vec))
    floating = 16.0 * _EPS * (scale + reach)
    if not math.isfinite(floating):
        return math.inf
    if floating > limit:
        return floating
    fc0, fu, fv = (_fraction_vec(item) for item in (c0, u_vec, v_vec))
    fo = _fraction_vec(plane.origin)
    n = _fraction_cross(_fraction_vec(plane.u_vector), _fraction_vec(plane.v_vector))
    nn = sum(c * c for c in n)
    norm_low = Fraction(math.sqrt(float(nn))) * Fraction(1.0 - 1e-12)
    if norm_low <= 0:
        return math.inf

    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    shift = tuple(c - o for c, o in zip(fc0, fo))
    plane_residual = (abs(dot(n, shift)) + abs(dot(n, fu)) + abs(dot(n, fv))) / norm_low
    axis = _fraction_vec(cylinder.axis)
    aa = dot(axis, axis)

    def m_dot(a, b):                                       # a.(I - axis axis^T / |axis|^2).b
        return dot(a, b) - dot(a, axis) * dot(b, axis) / aa

    radius = Fraction(cylinder.radius)
    a0 = tuple(c - o for c, o in zip(fc0, _fraction_vec(cylinder.origin)))
    uu, vv, uv = m_dot(fu, fu), m_dot(fv, fv), m_dot(fu, fv)
    g = (m_dot(a0, a0) - radius * radius + (uu + vv) / 2, 2 * m_dot(a0, fu), 2 * m_dot(a0, fv), (uu - vv) / 2, uv)
    cylinder_residual = sum(abs(value) for value in g) / radius
    return floating + float(plane_residual) + float(cylinder_residual) + 8 * _EPS * float(plane_residual + cylinder_residual)
