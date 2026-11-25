"""Display samples of edges: lines, circles and ellipses on planes, cylinders and cones.

A viewer draws a face as triangles.  Neighbouring faces cannot crack apart when the polyline drawn
along an edge depends only on the edge and the supports of the faces that use it
(:func:`edge_display_samples`): every face, drawing route and caller then draws the same
polyline there.

Scope (owner decision 2026-11-25, N1): only straight lines (``Straight``, ``LineRange``), circles
(``Arc``, circular ``EllipticArc``) and ellipses (``EllipticArc``), on faces supported by a
``Plane``, ``Cylinder`` or ``Cone``.  Each is divided in closed form, so every bound below holds by
construction (to floating-point rounding), not by measurement:

* the tangent turns by at most ``tol.angle`` over every interval;
* the azimuth about the axis of every incident cylinder or cone (the turning of its normal, an
  upper bound for a cone) changes by at most ``tol.angle`` over every interval;
* the chord leaves the curve by at most the sag allowance (:class:`DisplayTolerance`).

Everything else is refused, typed (:class:`DisplayTrimRefused`): an edge used by a face on any other
support or with an explicit parameterization (``SUPPORT_UNSUPPORTED``: the caller keeps its own
route), any other curve, or a division that cannot be made exact (``EDGE_UNSUPPORTED``), and input
outside the declared numeric domain (``OUT_OF_DOMAIN``, :class:`DisplayDomainRefused`).  See
``docs/DISPLAY_TESSELLATION.md``.

This module only reads the model through public calls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .curves import Arc, Straight
from .errors import GeometryError
from .exact_curves import EllipticArc
from .line_ranges import LineRange
from .surfaces import Cone, Cylinder, Plane

__all__ = [
    "DisplayTolerance",
    "DisplayTrimRefused",
    "DisplayDomainRefused",
    "EdgeSamples",
    "REFUSAL_CODES",
    "MAX_DISPLAY_COORDINATE",
    "MAX_DISPLAY_ANGLE",
    "MIN_DISPLAY_RADIUS_RATIO",
    "edge_display_samples",
]

#: Typed refusal codes.  ``SUPPORT_UNSUPPORTED``: an incident face's support has no display route
#: here (or the face has an explicit parameterization): the caller keeps its own; ``TRIM_INVALID``
#: and ``TRIM_WINDS_SUPPORT``: reserved for face trims; ``EDGE_UNSUPPORTED``: the curve is not a
#: line, circle or ellipse, or cannot be divided exactly (degenerate, non-monotone support azimuth,
#: unresolvable sag, more than 4096 intervals); ``OUT_OF_DOMAIN``: the input lies outside the
#: declared numeric domain below (:class:`DisplayDomainRefused`).
REFUSAL_CODES = frozenset({"SUPPORT_UNSUPPORTED", "TRIM_INVALID", "TRIM_WINDS_SUPPORT", "EDGE_UNSUPPORTED",
                           "OUT_OF_DOMAIN"})

# Declared numeric domain (owner decision 2026-11-25): the bounds hold inside it (to rounding) and
# the refusal is typed outside it.
#: Largest absolute coordinate of any vertex, curve definition point, support origin (and plane
#: vector) or emitted point (model units).  Its float spacing is 1.2e-7.
MAX_DISPLAY_COORDINATE = 1.0e9
#: Largest absolute stored angle (start, start + sweep) of an ellipse arc, cylinder or cone
#: (radians).  Its float spacing is 1.2e-10 rad, four orders below the smallest display angle.
MAX_DISPLAY_ANGLE = 1.0e6
#: Smallest distance from a support axis, as a fraction of max(1, |coordinate|): every incident
#: cylinder radius and every cone point used for an azimuth (a line's ends and closest approach,
#: a lower bound of an arc's) except a cone's exact apex.  An azimuth computed from coordinates of
#: size |p| at distance r from the axis is rounded by about eps |p| / r: at the bound 2.2e-7 rad,
#: below the smallest display angle (1e-6).
MIN_DISPLAY_RADIUS_RATIO = 1.0e-9

_MAX_INTERVALS = 4096      # per edge
_SLACK = 1.0e-9            # relative slack so exact uniform divisions are not split again
_MIN_ANGLE = 1.0e-6        # smallest display angle (radians)
_CIRCLE_RTOL = 1.0e-12     # |u| = |v| and u.v = 0 to this relative tolerance: a circle
_DEGENERATE = 1.0e-12      # |det| <= this * |M|^2: a projected or in-plane frame is degenerate
_EPS = float(np.finfo(float).eps)
_TWO_PI = 2.0 * math.pi


class DisplayTrimRefused(GeometryError):
    """A typed refusal: ``code`` is one of :data:`REFUSAL_CODES`."""

    def __init__(self, code: str, message: str):
        if code not in REFUSAL_CODES:
            raise ValueError(f"unknown display refusal code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code


class DisplayDomainRefused(DisplayTrimRefused):
    """The input lies outside the declared numeric domain (code ``OUT_OF_DOMAIN``)."""

    def __init__(self, message: str):
        super().__init__("OUT_OF_DOMAIN", message)


@dataclass(frozen=True)
class DisplayTolerance:
    """How finely to draw: ``angle`` is the largest turning per segment (radians, in [1e-6, pi/2]).

    ``sag`` optionally caps the distance of a chord from its curve.  When omitted, the allowance is
    the smallest over the edge's incident cylinders and cones of ``r_max (1 - cos(angle / 2))``
    (``r_max`` the cylinder radius, the cone's larger radius); with neither, chords are bounded by
    turning alone.
    """

    angle: float
    sag: float | None = None

    def __post_init__(self):
        angle = float(self.angle)
        if not (math.isfinite(angle) and _MIN_ANGLE <= angle <= math.pi / 2):
            raise GeometryError("display angle must be finite and in [1e-6, pi/2]")
        object.__setattr__(self, "angle", angle)
        if self.sag is not None:
            sag = float(self.sag)
            if not (math.isfinite(sag) and sag > 0.0):
                raise GeometryError("display sag must be finite and positive")
            object.__setattr__(self, "sag", sag)


def _frozen(array: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(array)
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class EdgeSamples:
    """Edge parameters (strictly increasing, from exactly 0 to exactly 1) and their points.

    The first and last points are the edge's start and end vertex positions, so edges meeting at
    a vertex meet exactly; interior points are ``model.evaluate_edge_many(edge, parameters)``.
    """

    edge_id: int
    parameters: np.ndarray
    points: np.ndarray


def _unsupported(edge_id, message):
    return DisplayTrimRefused("EDGE_UNSUPPORTED", f"edge {edge_id}: {message}")


def _count(edge_id, span, step) -> int:
    """``ceil(span / step)`` uniform pieces (an exact multiple is not split again)."""
    value = abs(span) / step
    if not math.isfinite(value) or value > _MAX_INTERVALS:
        raise _unsupported(edge_id, f"needs more than {_MAX_INTERVALS} intervals")
    return max(1, math.ceil(value - _SLACK))


# ---------------------------------------------------------------------------------------------
# angle of an affine ellipse seen from the origin of its plane, with its closed-form inverse
# ---------------------------------------------------------------------------------------------
class _AngleMap:
    """The direction angle Phi(theta) of ``p(theta) = c + M w(theta + phase)`` in a plane, where
    ``w(x) = (cos x, sin x)`` and ``M`` is an invertible 2x2 matrix, as a continuous lift over
    ``[theta_a, theta_b]``, and its exact inverse.

    With ``J = diag(1, -1)`` when ``det M < 0`` (so ``M w(x) = (M J) w(-x)``), write ``x = s (theta +
    phase)`` (``s = +-1``), ``N = M J^k`` with ``det N > 0`` and ``z = N^-1 c``: ``p = N (z + w(x))``.

    * ``chi(x) = arg(z + w(x))`` lifts continuously as ``x + atan2(cross(w, z + w), 1 + z.w)`` while
      ``g(x) = 1 + z.w > 0`` (the angle between ``w`` and ``z + w`` stays in (-pi/2, pi/2)), and as
      ``x + pi + atan2(-cross, -g)`` while ``g < 0``.  ``dchi/dx = g / |z + w|^2``, so ``chi`` is
      monotone exactly when ``g`` keeps its sign; ``g`` reaching 0 (or ``z + w = 0``: the curve
      reaching the origin) is refused.
    * ``Phi = arg(N w(chi))`` lifts as ``chi + alpha + atan2(cross(e, S e), e.S e)`` with the polar
      decomposition ``N = R(alpha) S`` (``S`` symmetric positive definite: the angle between ``e``
      and ``S e`` stays in (-pi/2, pi/2)); orientation-preserving, so monotone in ``chi``.

    Inverse: ``chi`` from ``N^-1 (cos Phi, sin Phi)``, lifted to within pi/2 of ``Phi - alpha``;
    then ``z + w`` on that ray, ``w = lambda e - z`` with ``|w| = 1``: ``lambda = z.e + sign(g)
    sqrt((z.e)^2 - |z|^2 + 1)``, the root on the monotone branch (``g = r (r + sign(g) z.e)`` with
    ``r`` the square root); ``x`` lifted to within pi/2 of ``chi`` (``g > 0``) or of ``chi - pi``
    (``g < 0``).
    """

    def __init__(self, edge_id, matrix, offset, phase, theta_a, theta_b):
        m = np.array(matrix, dtype=float).reshape(2, 2)
        det = float(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0])
        frobenius = float((m * m).sum())
        if not (math.isfinite(det) and abs(det) > _DEGENERATE * frobenius):
            raise _unsupported(edge_id, "the curve's frame (or its projection along a support axis) is degenerate")
        self.flip = -1.0 if det < 0 else 1.0
        if det < 0:
            m[:, 1] = -m[:, 1]
            det = -det
        self.m, self.phase = m, float(phase)
        self.inverse = np.array(((m[1, 1], -m[0, 1]), (-m[1, 0], m[0, 0]))) / det
        self.alpha = math.atan2(m[1, 0] - m[0, 1], m[0, 0] + m[1, 1])
        ca, sa = math.cos(self.alpha), math.sin(self.alpha)
        self.s = np.array(((ca * m[0, 0] + sa * m[1, 0], ca * m[0, 1] + sa * m[1, 1]),
                           (-sa * m[0, 0] + ca * m[1, 0], -sa * m[0, 1] + ca * m[1, 1])))
        self.z = self.inverse @ np.asarray(offset, dtype=float)
        # sign of g = 1 + z.w(x) over the x interval: its extremes are at the ends or where
        # x - arg z is a multiple of pi
        xa, xb = sorted((self._x(theta_a), self._x(theta_b)))
        radius = float(math.hypot(*self.z))
        middle = math.atan2(self.z[1], self.z[0])
        candidates = [xa, xb]
        k = math.ceil((xa - middle) / math.pi)
        while middle + k * math.pi < xb and len(candidates) < 8:
            candidates.append(middle + k * math.pi)
            k += 1
        g = [1.0 + radius * math.cos(x - middle) for x in candidates]
        self.g_min, g_max = min(g), max(g)
        if self.g_min > 0.0:
            self.branch = 1.0
        elif g_max < 0.0:
            self.branch = -1.0
        else:
            raise _unsupported(edge_id, "the azimuth about a support axis is not monotone along the edge "
                                        "(its projection reaches or turns back round the axis)")
        self.z_radius = radius
        # |p| >= sigma_min(N) |z + w| and |z + w|^2 = |z|^2 - 1 + 2 g (smallest at g_min): a lower
        # bound of the curve's closest approach to the origin over the interval
        sigma_max = math.sqrt(0.5 * (frobenius + math.sqrt(max(0.0, frobenius * frobenius - 4.0 * det * det))))
        self.closest = det / sigma_max * math.sqrt(max(0.0, radius * radius - 1.0 + 2.0 * self.g_min))

    def _x(self, theta):
        return self.flip * (theta + self.phase)

    def angle(self, theta) -> float:
        x = self._x(theta)
        w0, w1 = math.cos(x), math.sin(x)
        q0, q1 = self.z[0] + w0, self.z[1] + w1
        b = self.branch
        chi = x + (0.0 if b > 0 else math.pi) + math.atan2(b * (w0 * q1 - w1 * q0), b * (w0 * q0 + w1 * q1))
        e0, e1 = math.cos(chi), math.sin(chi)
        s0, s1 = self.s[0, 0] * e0 + self.s[0, 1] * e1, self.s[1, 0] * e0 + self.s[1, 1] * e1
        return chi + self.alpha + math.atan2(e0 * s1 - e1 * s0, e0 * s0 + e1 * s1)

    def solve(self, phi) -> float:
        d0, d1 = math.cos(phi), math.sin(phi)
        x0 = self.inverse[0, 0] * d0 + self.inverse[0, 1] * d1
        x1 = self.inverse[1, 0] * d0 + self.inverse[1, 1] * d1
        chi = math.atan2(x1, x0)
        chi += _TWO_PI * round((phi - self.alpha - chi) / _TWO_PI)
        e0, e1 = math.cos(chi), math.sin(chi)
        ze = self.z[0] * e0 + self.z[1] * e1
        root = math.sqrt(max(0.0, ze * ze - self.z_radius * self.z_radius + 1.0))
        lam = ze + self.branch * root
        x = math.atan2(lam * e1 - self.z[1], lam * e0 - self.z[0])
        reference = chi if self.branch > 0 else chi - math.pi
        x += _TWO_PI * round((reference - x) / _TWO_PI)
        return self.flip * x - self.phase


def _angle_division(amap, theta_a, theta_b, n):
    """The ``n - 1`` interior thetas splitting ``amap``'s angle over [theta_a, theta_b] equally."""
    pa, pb = amap.angle(theta_a), amap.angle(theta_b)
    return [amap.solve(pa + (pb - pa) * k / n) for k in range(1, n)]


# ---------------------------------------------------------------------------------------------
# domain
# ---------------------------------------------------------------------------------------------
def _check_angle(name, *values):
    for value in values:
        if not (math.isfinite(value) and abs(value) <= MAX_DISPLAY_ANGLE):
            raise DisplayDomainRefused(f"{name} angle {value!r} is outside |angle| <= {MAX_DISPLAY_ANGLE:g} rad")


def _check_points(name, points):
    points = np.asarray(points, dtype=float)
    if points.size and not (np.all(np.isfinite(points)) and float(np.abs(points).max()) <= MAX_DISPLAY_COORDINATE):
        raise DisplayDomainRefused(f"{name} lies outside |coordinate| <= {MAX_DISPLAY_COORDINATE:g}")


def _check_support_domain(support):
    if isinstance(support, Plane):
        _check_points("a support plane", np.vstack((support.origin, support.u_vector, support.v_vector)))
    else:
        _check_points("a support origin", support.origin)
        _check_angle("a support", support.start_angle, support.start_angle + support.sweep_angle)


def _check_radius(name, distance, *points):
    scale = max([1.0] + [float(np.abs(np.asarray(p, dtype=float)).max()) for p in points])
    if not distance >= MIN_DISPLAY_RADIUS_RATIO * scale:
        raise DisplayDomainRefused(f"{name} is {distance!r} from a support axis, below "
                                   f"{MIN_DISPLAY_RADIUS_RATIO:g} x {scale:g}")


def _incident_supports(model, edge_id: int) -> tuple:
    found = []
    for face_id in model.faces_using_edge(edge_id):
        face = model.faces[face_id]
        support = face.surface
        if face.parameterization is not None:
            raise DisplayTrimRefused("SUPPORT_UNSUPPORTED",
                                     f"edge {edge_id} is used by face {face_id}, which has an explicit parameterization")
        if not isinstance(support, (Plane, Cylinder, Cone)):
            raise DisplayTrimRefused("SUPPORT_UNSUPPORTED", f"edge {edge_id} is used by face {face_id} on a "
                                                            f"{type(support).__name__} support")
        if not any(support is s for s in found):
            found.append(support)
    return tuple(found)


def _axis_frame(support):
    origin = np.asarray(support.origin, dtype=float)
    return origin, np.asarray(support.radial_direction, dtype=float), np.asarray(support.circumferential_direction,
                                                                                 dtype=float)


def _apex(support):
    """A cone's apex point, only when one of its radii is exactly 0."""
    if not isinstance(support, Cone) or (support.radius_start != 0.0 and support.radius_end != 0.0):
        return None
    v = 0.0 if support.radius_start == 0.0 else 1.0
    return np.asarray(support.origin, dtype=float) + v * support.height * np.asarray(support.axis, dtype=float)


# ---------------------------------------------------------------------------------------------
# divisions
# ---------------------------------------------------------------------------------------------
def _line_azimuth_division(model, edge_id, support, a, b, start, end, step):
    """Edge parameters dividing a line's azimuth about ``support``'s axis into equal steps.

    The projection ``A' + t D'`` of a line onto the plane normal to the axis has a monotone
    azimuth unless it passes through the axis point (cross(A' + t D', D') = cross(A', D') is
    constant); it subtends less than pi, so the wrapped end difference is its span.  A target
    azimuth ``phi`` is reached at ``t = -cross(A', e) / cross(D', e)``, ``e = (cos phi, sin phi)``.
    A line through a cone's exact apex is a generator: constant azimuth, no division.
    """
    apex = _apex(support)
    if apex is not None and min(float(np.linalg.norm(start - apex)),
                                float(np.linalg.norm(end - apex))) <= model.tolerance.length:
        return []
    origin, x_axis, y_axis = _axis_frame(support)
    a3, d3 = a - origin, b - a
    a2 = np.array((a3 @ x_axis, a3 @ y_axis))
    d2 = np.array((d3 @ x_axis, d3 @ y_axis))
    length2 = float(d2 @ d2)
    closest = min(1.0, max(0.0, -float(a2 @ d2) / length2)) if length2 > 0.0 else 0.0
    for t in sorted({0.0, 1.0, closest}):
        _check_radius(f"edge {edge_id}'s point at t = {t:.17g}", float(math.hypot(*(a2 + t * d2))),
                      a + t * d3, origin)
    pa = math.atan2(a2[1], a2[0])
    b2 = a2 + d2
    span = (math.atan2(b2[1], b2[0]) - pa + math.pi) % _TWO_PI - math.pi
    n = _count(edge_id, span, step)
    values = []
    for k in range(1, n):
        phi = pa + span * k / n
        e0, e1 = math.cos(phi), math.sin(phi)
        denominator = d2[0] * e1 - d2[1] * e0
        if denominator == 0.0:
            raise _unsupported(edge_id, "a target azimuth runs along the line's projection")
        values.append(-(a2[0] * e1 - a2[1] * e0) / denominator)
    return values


class _Conic:
    """``c + cos(theta) u + sin(theta) v``, ``theta = start + sweep t`` (a circle or an ellipse)."""

    def __init__(self, edge_id, centre, u, v, start, sweep):
        self.edge_id = edge_id
        self.c, self.u, self.v = (np.asarray(x, dtype=float) for x in (centre, u, v))
        self.start, self.sweep = float(start), float(sweep)
        nu, nv = float(np.linalg.norm(self.u)), float(np.linalg.norm(self.v))
        if not (nu > 0.0 and nv > 0.0):
            raise _unsupported(edge_id, "an ellipse axis vector is zero")
        e1 = self.u / nu
        rest = self.v - float(self.v @ e1) * e1
        nr = float(np.linalg.norm(rest))
        if not nr > _DEGENERATE * nv:
            raise _unsupported(edge_id, "the ellipse's axis vectors are parallel")
        e2 = rest / nr
        # in-plane matrix of (u, v): upper triangular with a positive diagonal (det > 0)
        self.plane = np.array(((nu, float(self.v @ e1)), (0.0, float(self.v @ e2))))
        self.circle = (abs(nu - nv) <= _CIRCLE_RTOL * max(nu, nv)
                       and abs(float(self.u @ self.v)) <= _CIRCLE_RTOL * nu * nv)
        frobenius = float((self.plane * self.plane).sum())
        det = float(self.plane[0, 0] * self.plane[1, 1])
        major = math.sqrt(0.5 * (frobenius + math.sqrt(max(0.0, frobenius * frobenius - 4.0 * det * det))))
        minor = det / major
        self.rho_max = major * major / minor          # largest radius of curvature, a^2 / b
        self.radius = max(nu, nv)                     # a circle's radius (conservative to 1e-12)

    @property
    def theta_end(self):
        return self.start + self.sweep

    def to_parameters(self, thetas):
        return [(theta - self.start) / self.sweep for theta in thetas]

    def tangent_division(self, step, allowance, residual):
        """Edge parameters of the tangent division, refined for the sag allowance.

        Circle: uniform in theta (the tangent turns uniformly).  A chord over a theta step
        ``delta`` leaves the circle by exactly ``rho (1 - cos(delta / 2)) = 2 rho sin^2(delta / 4)``
        and sub-intervals sag less, so ``delta <= 4 asin(sqrt(sag / (2 rho)))`` bounds it.

        Ellipse: uniform in the tangent angle ``psi`` (closed-form inverse, :class:`_AngleMap`).
        Proof of the sag bound: over an interval of tangent turning ``h <= pi/2`` and length ``L``
        the chord direction is one of the tangent directions (mean value theorem), so every tangent
        is within ``h`` of it; the distance of a point at arc length ``s`` from the chord's line is
        ``|int sin(psi - psi_c)| <= min(s, L - s) sin h <= (L / 2) sin h`` (integrated from the
        nearer end), and its foot lies on the segment because the projection onto the chord grows
        at ``cos(psi - psi_c) >= cos h >= 0``.  ``L = int rho dpsi <= rho_max h`` with ``rho_max =
        a^2 / b`` the largest radius of curvature (principal semi-axes ``a >= b``).  So the sag is at
        most ``(rho_max h / 2) sin h``, which grows with ``h``: halving the step (doubling ``n``)
        until it holds is sufficient, and sub-intervals of the union keep it.
        """
        if self.circle:
            span = abs(self.sweep)
            n = _count(self.edge_id, span, step)
            if allowance is not None and allowance < 2.0 * self.radius:
                largest = 4.0 * math.asin(math.sqrt(allowance / (2.0 * self.radius)))
                if not largest > 0.0:
                    raise _unsupported(self.edge_id, "the sag allowance is below the circle's float resolution")
                n = max(n, _count(self.edge_id, span, largest))
            if n == 1 and 2.0 * self.radius * math.sin(span / 4) ** 2 > residual:
                n = 2                                   # a curved edge is never one chord
            return np.linspace(0.0, 1.0, n + 1)[1:-1].tolist()
        amap = _AngleMap(self.edge_id, self.plane, (0.0, 0.0), math.pi / 2, self.start, self.theta_end)
        total = abs(amap.angle(self.theta_end) - amap.angle(self.start))
        n = _count(self.edge_id, total, step)

        def bound(h):
            return 0.5 * self.rho_max * h * math.sin(h)

        if allowance is not None:
            while bound(total / n) > allowance:
                n *= 2
                if n > _MAX_INTERVALS:
                    raise _unsupported(self.edge_id, f"needs more than {_MAX_INTERVALS} intervals for its sag")
        if n == 1 and bound(total) > residual:
            n = 2                                       # a curved edge is never one chord
        return self.to_parameters(_angle_division(amap, self.start, self.theta_end, n))

    def azimuth_division(self, support, step, scale_points):
        """Edge parameters dividing the azimuth about ``support``'s axis into equal steps: the
        projection ``c' + cos(theta) u' + sin(theta) v'`` (``c'`` relative to the axis point) in
        the support's (radial, circumferential) frame, through :class:`_AngleMap` (the exact root of
        ``A cos(theta) + B sin(theta) = C`` on the monotone branch)."""
        origin, x_axis, y_axis = _axis_frame(support)
        c3 = self.c - origin
        projected = np.array(((self.u @ x_axis, self.v @ x_axis), (self.u @ y_axis, self.v @ y_axis)))
        amap = _AngleMap(self.edge_id, projected, (c3 @ x_axis, c3 @ y_axis), 0.0, self.start, self.theta_end)
        if isinstance(support, Cone):
            _check_radius(f"edge {self.edge_id} (closest approach, lower bound)", amap.closest,
                          origin, *scale_points)
        span = amap.angle(self.theta_end) - amap.angle(self.start)
        n = _count(self.edge_id, span, step)
        return self.to_parameters(_angle_division(amap, self.start, self.theta_end, n))


def _sag_allowance(supports, tol: DisplayTolerance):
    if tol.sag is not None:
        return tol.sag
    radii = []
    for support in supports:
        if isinstance(support, Cylinder):
            radii.append(support.radius)
        elif isinstance(support, Cone):
            radii.append(max(support.radius_start, support.radius_end))
    if not radii:
        return None
    return min(radii) * 2.0 * math.sin(tol.angle / 4) ** 2          # r (1 - cos(angle / 2))


def _union(edge_id, first, others, gap):
    """Sorted union of division parameters in (0, 1) with the ends; a value of a later division
    within ``gap`` of one already kept (the same division point reached by two divisions, up to
    rounding) is not repeated."""
    kept = np.unique(np.clip(np.asarray([0.0, 1.0, *first], dtype=float), 0.0, 1.0))
    for values in others:
        for value in values:
            value = min(1.0, max(0.0, float(value)))
            i = int(np.searchsorted(kept, value))
            if (i < len(kept) and kept[i] - value <= gap) or (i > 0 and value - kept[i - 1] <= gap):
                continue
            kept = np.insert(kept, i, value)
        if len(kept) - 1 > _MAX_INTERVALS:
            raise _unsupported(edge_id, f"needs more than {_MAX_INTERVALS} intervals")
    if len(kept) - 1 > _MAX_INTERVALS:
        raise _unsupported(edge_id, f"needs more than {_MAX_INTERVALS} intervals")
    return kept


# ---------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------
def edge_display_samples(model, edge_id: int, tol: DisplayTolerance) -> EdgeSamples:
    """Sample one edge for display: the same result for every face using the edge.

    The samples depend on the edge and on the supports of the faces using it
    (``model.faces_using_edge``), which must be planes, cylinders or cones without an explicit
    face parameterization (else ``SUPPORT_UNSUPPORTED``).  The edge must be a line
    (``Straight``, ``LineRange``), a circle (``Arc``, circular ``EllipticArc``) or an ellipse
    (``EllipticArc``; else ``EDGE_UNSUPPORTED``).  The parameters are the sorted union of

    * the tangent division: none for a line; uniform in theta for a circle; uniform in tangent
      angle for an ellipse (exact closed-form inverse), ``ceil(turning / angle)`` steps, refined
      for the sag allowance (:meth:`_Conic.tangent_division`);
    * for each incident cylinder or cone, the division of the azimuth about its axis into
      ``ceil(span / angle)`` equal steps, solved in closed form (none along a cylinder ruling or
      a generator through a cone's exact apex).

    Every interval of the union lies inside one interval of each division, so the tangent turns,
    and every incident cylinder's or cone's azimuth changes, by at most ``tol.angle`` over it,
    and its chord leaves the curve by at most the sag allowance (:class:`DisplayTolerance`).  A
    curved edge whose single chord could leave it by more than ``model.tolerance.surface_residual``
    gets at least two intervals.  The first and last points are the edge's vertex positions.
    Inputs are checked against the declared numeric domain first (``OUT_OF_DOMAIN``).  Typed
    refusal (``EDGE_UNSUPPORTED``) for a degenerate curve, an azimuth that is not monotone along
    the edge, an unresolvable sag (at most 4 eps of the coordinates) or more than 4096 intervals.
    """
    if not isinstance(tol, DisplayTolerance):
        raise GeometryError("tol must be a DisplayTolerance")
    edge_id = int(edge_id)
    edge = model.edges[edge_id]
    supports = _incident_supports(model, edge_id)
    curve = edge.curve
    start = np.asarray(model.vertex_position(edge.start), dtype=float)
    end = np.asarray(model.vertex_position(edge.end), dtype=float)

    # --- declared domain, before any division
    _check_points(f"edge {edge_id}'s vertices", np.vstack((start, end)))
    conic = None
    line = None
    if isinstance(curve, Straight):
        line = (start, end)
    elif isinstance(curve, LineRange):
        _check_points(f"edge {edge_id}'s line carrier", np.vstack((curve.a, curve.b)))
        carrier_a, carrier_b = np.asarray(curve.a, dtype=float), np.asarray(curve.b, dtype=float)
        lo, hi = (float(x) for x in curve.domain)
        line = (carrier_a + lo * (carrier_b - carrier_a), carrier_a + hi * (carrier_b - carrier_a))
    elif isinstance(curve, EllipticArc):
        centre = np.asarray(curve.center, dtype=float)
        _check_points(f"edge {edge_id}'s ellipse", np.vstack((centre, centre + curve.u_vector,
                                                              centre + curve.v_vector)))
        _check_angle(f"edge {edge_id}'s ellipse", curve.start_angle, curve.start_angle + curve.sweep_angle)
    elif isinstance(curve, Arc):
        _check_points(f"edge {edge_id}'s arc via vertex", model.vertex_position(curve.via_vertex))
    else:
        raise _unsupported(edge_id, f"a {type(curve).__name__} is not a line, circle or ellipse")
    for support in supports:
        _check_support_domain(support)
    if isinstance(curve, Arc):
        try:
            frame = model.arc_frame(edge_id)
        except GeometryError as error:
            raise _unsupported(edge_id, str(error)) from error
        _check_points(f"edge {edge_id}'s arc centre", frame.center)
        conic = _Conic(edge_id, frame.center, frame.radius * frame.e1, frame.radius * frame.e2, 0.0, frame.sweep)
    elif isinstance(curve, EllipticArc):
        conic = _Conic(edge_id, curve.center, curve.u_vector, curve.v_vector, curve.start_angle, curve.sweep_angle)
    definition = [start, end] if conic is None else [start, end, conic.c, conic.c + conic.u, conic.c + conic.v]
    if line is not None:
        definition += list(line)
    for support in supports:
        if isinstance(support, Cylinder):
            _check_radius("a cylinder radius", support.radius, support.origin, *definition)

    # --- divisions
    step = tol.angle
    if line is not None:
        divisions = [_line_azimuth_division(model, edge_id, s, line[0], line[1], start, end, step)
                     for s in supports if isinstance(s, (Cylinder, Cone))]
        parameters = _union(edge_id, [], divisions, 64.0 * _EPS)
    else:
        allowance = _sag_allowance(supports, tol)
        resolution = 4.0 * _EPS * max(1.0, float(np.abs(np.vstack((start, end))).max()))
        if allowance is not None and allowance <= resolution:
            raise _unsupported(edge_id, f"the sag allowance {allowance!r} is below its coordinates' resolution")
        tangent = conic.tangent_division(step, allowance, model.tolerance.surface_residual)
        divisions = [conic.azimuth_division(s, step, definition)
                     for s in supports if isinstance(s, (Cylinder, Cone))]
        # parameters closer than 64 ulps of the stored angle evaluate to the same theta (to rounding)
        gap = 64.0 * _EPS * max(1.0, abs(conic.start), abs(conic.theta_end)) / abs(conic.sweep)
        parameters = _union(edge_id, tangent, divisions, gap)
    return _edge_result(model, edge_id, parameters, start, end)


def _edge_result(model, edge_id, parameters, start, end) -> EdgeSamples:
    try:
        points = np.array(model.evaluate_edge_many(edge_id, parameters), dtype=float)
    except GeometryError as error:
        raise _unsupported(edge_id, str(error)) from error
    _check_points(f"edge {edge_id}'s emitted points", points)
    if np.any(np.all(points[1:] == points[:-1], axis=1)):     # before the ends become the vertices
        raise _unsupported(edge_id, "distinct parameters evaluate to one point (lost precision)")
    points[0], points[-1] = start, end
    if np.any(np.all(points[1:] == points[:-1], axis=1)):
        raise _unsupported(edge_id, "distinct parameters evaluate to one point (lost precision)")
    return EdgeSamples(edge_id, _frozen(parameters), _frozen(points))
