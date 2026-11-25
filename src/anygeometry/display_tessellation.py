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
    "MAX_DISPLAY_ELLIPSE_ASPECT",
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
#: cylinder radius and every point used for an azimuth (a line's ends, except one on the axis to
#: the frame's rounding, 8 eps max(1, |coordinate|), and its closest approach; a lower bound of a
#: cylinder or cone arc's, whose scale includes max(1, |stored angle|) sigma_max of its projected
#: frame: the evaluator rounds start + sweep t).  A line with both ends on the axis is refused.
#: An azimuth computed from coordinates of size |p| at distance r from the axis is rounded by
#: about eps |p| / r: at the bound 2.2e-7 rad, below the smallest display angle (1e-6).
MIN_DISPLAY_RADIUS_RATIO = 1.0e-9
#: Largest ``a / b x max(1, |start|, |start + sweep|)`` of a non-circular ellipse arc (``a``, ``b``
#: its principal semi-axes, the singular values of ``[u v]``).  The tangent direction moves up to
#: ``a / b`` times faster than the arc angle, and the evaluator rounds the absolute angle
#: ``start + sweep t`` by eps |theta|: the tangent moves by at most ``1e6 x eps = 2.2e-10`` rad,
#: four orders below the smallest display angle.  The same bound applies to the arc's projection
#: across every incident cylinder's or cone's axis (the 2x2 projected frame ``M'``, which can be
#: far thinner than the curve): the azimuth is resolved through ``M'^-1`` and back, so its
#: rounding is at most ``aspect(M') x eps`` (and the stored angle's, as for the tangent).
MAX_DISPLAY_ELLIPSE_ASPECT = 1.0e6

_MAX_INTERVALS = 4096      # per edge
_STEP_ROUNDING = 4.0 * float(np.finfo(float).eps)   # relative rounding of a computed span
_MIN_ANGLE = 1.0e-6        # smallest display angle (radians)
_EPS = float(np.finfo(float).eps)
_CIRCLE_RTOL = 8.0 * _EPS  # |u| = |v| and u.v = 0 to a few eps relative: a circle (else an ellipse)
_ON_AXIS = 8.0 * _EPS      # a projection within 8 eps max(1, |coordinates|) of an axis is on it
_DEGENERATE = 1.0e-12      # |det| <= this * |M|^2: a projected or in-plane frame is degenerate
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
    """``ceil(span / step)`` uniform pieces, checked against the cap before anything is built.

    The quotient is rounded, so an exact multiple can come out a few ulps above an integer; one
    piece fewer is used only when its emitted step ``span / (n - 1)`` is verified to be at most
    ``step`` to the span's own rounding (``4 eps`` relative: the span is a difference of computed
    angles), never by a fixed slack."""
    span = abs(span)
    value = span / step
    if not math.isfinite(value) or value > _MAX_INTERVALS:
        raise _unsupported(edge_id, f"needs more than {_MAX_INTERVALS} intervals")
    n = max(1, math.ceil(value))
    if n > 1 and span / (n - 1) <= step * (1.0 + _STEP_ROUNDING):
        n -= 1
    return n


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
      monotone exactly when ``g`` keeps its sign; ``g`` reaching 0 inside the interval (or
      ``z + w = 0``: the curve reaching the origin) is refused, ``g = 0`` at an end (a stationary
      end of a monotone azimuth) is not.
    * ``Phi = arg(N w(chi))`` lifts as ``chi + alpha + atan2(cross(e, S e), e.S e)`` with the polar
      decomposition ``N = R(alpha) S`` (``S`` symmetric positive definite: the angle between ``e``
      and ``S e`` stays in (-pi/2, pi/2)); orientation-preserving, so monotone in ``chi``.

    Inverse: ``chi`` from ``N^-1 (cos Phi, sin Phi)``, lifted to within pi/2 of ``Phi - alpha``;
    then ``z + w`` on that ray, ``w = lambda e - z`` with ``|w| = 1``: with ``z = (z.e) e + h e_perp``
    (``h = cross(e, z)``), ``w = sign(g) sqrt((1 - h)(1 + h)) e - h e_perp``, the root on the
    monotone branch (``g = r (r + sign(g) z.e)`` with ``r`` the square root), formed directly so
    nothing of size ``|z|`` cancels; ``x`` lifted to within pi/2 of ``chi`` (``g > 0``) or of
    ``chi - pi`` (``g < 0``).  ``sigma_max`` and ``aspect`` (``sigma_max / sigma_min``) come from
    two sums of squares and ``det``, without cancellation.
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
        # x - arg z is a multiple of pi (strictly inside).  g may vanish at an end (the curve is
        # tangent there to a ray from the axis: the azimuth is stationary at the end and still
        # monotone over the interval), never inside.
        xa, xb = sorted((self._x(theta_a), self._x(theta_b)))
        radius = float(math.hypot(*self.z))
        middle = math.atan2(self.z[1], self.z[0])
        inside = []
        k = math.ceil((xa - middle) / math.pi)
        while middle + k * math.pi < xb and len(inside) < 8:
            if middle + k * math.pi > xa:
                inside.append((k, middle + k * math.pi))
            k += 1
        g_ends = [1.0 + radius * math.cos(x - middle) for x in (xa, xb)]
        g_inside = [1.0 + radius * math.cos(x - middle) for _k, x in inside]
        # An end value within rounding of 0 (|g| <= 16 eps (1 + |z|)) of the wrong sign counts as
        # 0 when the turnback it could hide is below eps rad: g < 0 over at most delta / |g'|
        # next to the end, turning the azimuth back by about delta^2 / (2 q^3), q = |z + w| there.
        g_bound = 16.0 * _EPS * (1.0 + radius)
        for i, x in enumerate((xa, xb)):
            q = math.hypot(self.z[0] + math.cos(x), self.z[1] + math.sin(x))
            if 0.0 < abs(g_ends[i]) <= g_bound and g_ends[i] * g_ends[i] <= 2.0 * _EPS * q ** 3:
                g_ends[i] = 0.0
        if min(g_ends) >= 0.0 and all(g > 0.0 for g in g_inside) and max(g_ends + g_inside) > 0.0:
            self.branch = 1.0
        elif max(g_ends) <= 0.0 and all(g < 0.0 for g in g_inside) and min(g_ends + g_inside) < 0.0:
            self.branch = -1.0
        else:
            raise _unsupported(edge_id, "the azimuth about a support axis is not monotone along the edge "
                                        "(its projection reaches or turns back round the axis)")
        self.z_radius = radius
        # Singular values without cancellation: with det > 0, sigma_max + sigma_min and
        # sigma_max - sigma_min are the square roots of two sums of squares; sigma_min = det /
        # sigma_max (never the difference of two close numbers).
        self.sigma_max = 0.5 * (math.hypot(m[0, 0] + m[1, 1], m[1, 0] - m[0, 1])
                                + math.hypot(m[0, 0] - m[1, 1], m[0, 1] + m[1, 0]))
        self.aspect = self.sigma_max * self.sigma_max / det
        # Closest approach of the curve to the origin over the interval, bounded below by
        # sigma_min(N) min |z + w(x)|.  |z + w(x)| is the distance of the unit circle point w(x)
        # from -z: smallest at x = arg z + pi (odd k) where it is exactly | |z| - 1 |, else at an
        # end; each is evaluated directly (no expansion of |z + w|^2 that would cancel).
        distances = [math.hypot(self.z[0] + math.cos(x), self.z[1] + math.sin(x)) for x in (xa, xb)]
        distances += [abs(radius - 1.0) for k, _x in inside if k % 2]
        self.closest = det / self.sigma_max * min(distances)

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
        # z = (z.e) e + h e_perp (e_perp = (-e1, e0)): the ray point is w = lambda e - z
        # = sign(g) sqrt(1 - h^2) e - h e_perp, formed directly (no (z.e)^2 - |z|^2 + 1, no
        # lambda e - z: both cancel terms of size |z|^2 or |z| when the projection is thin)
        h = -self.z[0] * e1 + self.z[1] * e0
        root = self.branch * math.sqrt(max(0.0, (1.0 - h) * (1.0 + h)))
        x = math.atan2(root * e1 - h * e0, root * e0 + h * e1)
        reference = chi if self.branch > 0 else chi - math.pi
        x += _TWO_PI * round((reference - x) / _TWO_PI)
        return self.flip * x - self.phase


def _angle_division(amap, theta_a, theta_b, n):
    """The ``n - 1`` interior thetas splitting ``amap``'s angle over [theta_a, theta_b] equally."""
    pa, pb = amap.angle(theta_a), amap.angle(theta_b)
    return [amap.solve(pa + (pb - pa) * k / n) for k in range(1, n)]


@dataclass(frozen=True)
class _Plan:
    """A division's interval count (checked against the cap) and how to build it: every count
    of an edge is known and checked before any division is materialised."""

    count: int
    build: object


@dataclass(frozen=True)
class _Division:
    """Edge parameters dividing a monotone angle (``angle(t)``, radians, small: local to the
    edge) into equal steps.  ``angle`` lets the union verify that a near-duplicate cut it drops
    is reached, to rounding, by the cut it keeps."""

    values: list
    angle: object = None


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


# ---------------------------------------------------------------------------------------------
# divisions
# ---------------------------------------------------------------------------------------------
def _line_azimuth_division(edge_id, support, a, b, step) -> "_Plan":
    """The plan (count first, then the edge parameters) dividing a line's azimuth about
    ``support``'s axis into equal steps.

    ``a``, ``b`` are the line's evaluated ends (the exact evaluator's).  The projection ``A'`` of
    ``a`` relative to the axis point and the projection ``D'`` of the exact difference ``b - a``
    (never of ``b`` separately: a ruling's difference projects to exactly 0 where its separately
    rounded ends do not) onto the plane normal to the axis, with ``B' = A' + D'``, decide:

    * an end on the axis (its projection within ``8 eps max(1, |end|, |axis point|)`` of it, the
      rounding of the frame: a cone's exact apex): the projection is a ray from the axis point,
      constant azimuth, no division; both ends on the axis: no azimuth anywhere, ``OUT_OF_DOMAIN``;
    * ``A'``, ``D'`` collinear (``|A' x D'| <= 8 eps |A'| |D'|``, including ``D' = 0``) with
      ``A'``, ``B'`` on the same side: constant azimuth (a cylinder ruling, a cone generator), no
      division; on opposite sides: the line crosses the axis, ``EDGE_UNSUPPORTED``;
    * otherwise ``A' + t D'`` does not pass through the axis point, its azimuth is monotone
      (``cross(A' + t D', D') = cross(A', D')`` is constant) and subtends less than pi, so the
      wrapped end difference is its span.  A target azimuth ``phi`` is reached at
      ``t = -cross(A', e) / cross(D', e)``, ``e = (cos phi, sin phi)``.

    Each end off the axis, and the closest approach ``|cross(A', D')| / |D'|`` when it lies
    between the ends, is checked against the radius bound (``OUT_OF_DOMAIN``).  There is no
    proximity shortcut beyond the frame's rounding: a line from near an apex is divided like any
    other.
    """
    origin, x_axis, y_axis = _axis_frame(support)
    a3, d3 = a - origin, b - a
    a2 = np.array((float(a3 @ x_axis), float(a3 @ y_axis)))
    d2 = np.array((float(d3 @ x_axis), float(d3 @ y_axis)))
    b2 = a2 + d2
    ra, rb = math.hypot(*a2), math.hypot(*b2)
    on_axis = [r <= _ON_AXIS * max(1.0, float(np.abs(p).max()), float(np.abs(origin).max()))
               for r, p in ((ra, a), (rb, b))]
    if all(on_axis):
        raise DisplayDomainRefused(f"edge {edge_id} lies on a support axis: it has no azimuth")
    for name, distance, point, on in (("start", ra, a, on_axis[0]), ("end", rb, b, on_axis[1])):
        if not on:
            _check_radius(f"edge {edge_id}'s {name}", distance, point, origin)
    if any(on_axis):
        return _Plan(1, lambda: _Division([]))
    length = math.hypot(*d2)
    cross = float(a2[0] * d2[1] - a2[1] * d2[0])
    if not length > 0.0 or abs(cross) <= 8.0 * _EPS * ra * length:
        if float(a2 @ b2) > 0.0:
            return _Plan(1, lambda: _Division([]))
        raise _unsupported(edge_id, "the line's projection passes through a support axis")
    closest = -float(a2 @ d2) / length / length
    if 0.0 < closest < 1.0:
        _check_radius(f"edge {edge_id}'s closest approach (t = {closest:.17g})", abs(cross) / length,
                      a + closest * d3, origin)
    pa = math.atan2(a2[1], a2[0])
    span = (math.atan2(b2[1], b2[0]) - pa + math.pi) % _TWO_PI - math.pi
    n = _count(edge_id, span, step)

    def build():
        values = []
        for k in range(1, n):
            phi = pa + span * k / n
            e0, e1 = math.cos(phi), math.sin(phi)
            denominator = d2[0] * e1 - d2[1] * e0
            if denominator == 0.0:
                raise _unsupported(edge_id, "a target azimuth runs along the line's projection")
            values.append(-(a2[0] * e1 - a2[1] * e0) / denominator)

        def angle(t):
            p = a2 + t * d2
            return (math.atan2(p[1], p[0]) - pa + math.pi) % _TWO_PI - math.pi

        return _Division(values, angle)

    return _Plan(n, build)


class _Conic:
    """``c + cos(theta) u + sin(theta) v``, ``theta = start + sweep t`` (a circle or an ellipse).

    Every division works in the arc's local angle ``s = theta - start = sweep t``: the frame is
    rotated once, ``u' = cos(start) u + sin(start) v``, ``v' = -sin(start) u + cos(start) v``, so
    ``c + cos(s) u' + sin(s) v'`` is exactly the curve at local angle ``s``, and no absolute
    angle (up to 1e6 rad) is formed and rounded before the start is subtracted.
    """

    def __init__(self, edge_id, centre, u, v, start, sweep):
        self.edge_id = edge_id
        self.c, self.u, self.v = (np.asarray(x, dtype=float) for x in (centre, u, v))
        self.start, self.sweep = float(start), float(sweep)
        if not (math.isfinite(self.sweep) and self.sweep != 0.0):
            raise _unsupported(edge_id, "the arc's sweep is zero")
        nu, nv = float(np.linalg.norm(self.u)), float(np.linalg.norm(self.v))
        if not (nu > 0.0 and nv > 0.0):
            raise _unsupported(edge_id, "an ellipse axis vector is zero")
        e1 = self.u / nu
        rest = self.v - float(self.v @ e1) * e1
        nr = float(np.linalg.norm(rest))
        if not nr > _DEGENERATE * nv:
            raise _unsupported(edge_id, "the ellipse's axis vectors are parallel")
        self.circle = (abs(nu - nv) <= _CIRCLE_RTOL * max(nu, nv)
                       and abs(float(self.u @ self.v)) <= _CIRCLE_RTOL * nu * nv)
        self.area = float(np.linalg.norm(np.cross(self.u, self.v)))    # |u x v|
        self.radius = max(nu, nv)                     # a circle's radius (conservative to 8 eps)
        singular = np.linalg.svd(np.column_stack((self.u, self.v)), compute_uv=False)
        self.aspect = float(singular[0] / singular[1]) if singular[1] > 0.0 else math.inf
        cs, sn = math.cos(self.start), math.sin(self.start)
        self.lu = cs * self.u + sn * self.v           # the local frame at s = 0
        self.lv = -sn * self.u + cs * self.v
        nu, nv = float(np.linalg.norm(self.lu)), float(np.linalg.norm(self.lv))
        e1 = self.lu / nu
        rest = self.lv - float(self.lv @ e1) * e1
        e2 = rest / float(np.linalg.norm(rest))
        # in-plane matrix of (u', v'): upper triangular with a positive diagonal (det > 0)
        self.plane = np.array(((nu, float(self.lv @ e1)), (0.0, float(self.lv @ e2))))

    def to_parameters(self, local):
        return [s / self.sweep for s in local]

    def tangent_division(self, step, allowance, residual, offset=0.0) -> _Plan:
        """The plan of the tangent division (the sag is refined afterwards, :meth:`sag`).

        Circle: uniform in theta (the tangent turns uniformly).  A chord over a theta step
        ``delta`` leaves the circle by exactly ``rho (1 - cos(delta / 2)) = 2 rho sin^2(delta / 4)``
        and sub-intervals sag less, so ``delta <= 4 asin(sqrt(sag / (2 rho)))`` bounds it.

        Ellipse: uniform in the tangent angle ``psi`` (closed-form inverse, :class:`_AngleMap`).

        A single interval is split in two when its DRAWN chord (from the vertices, up to
        ``offset`` from the curve's ends) could leave the curve by more than ``residual``: the
        exact sag plus the offset bounds that distance (:meth:`refine_sag`).
        """
        if self.circle:
            span = abs(self.sweep)
            n = _count(self.edge_id, span, step)
            if allowance is not None and allowance < 2.0 * self.radius:
                largest = 4.0 * math.asin(math.sqrt(allowance / (2.0 * self.radius)))
                if not largest > 0.0:
                    raise _unsupported(self.edge_id, "the sag allowance is below the circle's float resolution")
                n = max(n, _count(self.edge_id, span, largest))
            if n == 1 and 2.0 * self.radius * math.sin(span / 4) ** 2 + offset > residual:
                n = 2                                   # a curved edge is never one chord
            return _Plan(n, lambda: np.linspace(0.0, 1.0, n + 1)[1:-1].tolist())
        amap = _AngleMap(self.edge_id, self.plane, (0.0, 0.0), math.pi / 2, 0.0, self.sweep)
        total = abs(amap.angle(self.sweep) - amap.angle(0.0))
        n = _count(self.edge_id, total, step)
        if n == 1 and self.sag(0.0, 1.0) + offset > residual:
            n = 2                                       # a curved edge is never one chord
        return _Plan(n, lambda: self.to_parameters(_angle_division(amap, 0.0, self.sweep, n)))

    def sag(self, t0, t1) -> float:
        """The exact largest distance of the curve over edge parameters [t0, t1] from its chord.

        Proof: ``P(theta) = c + M w(theta)`` is an affine image of the unit circle.  The signed
        distance of ``P(theta)`` from the chord's line is an affine function of ``P`` vanishing on
        the line, so it is a fixed multiple of the circle point's signed distance from the
        preimage chord: both are largest at the same theta, the circle arc's parametric midpoint
        ``m`` (where the circle's tangent is parallel to its chord), for ``|delta| <= pi``.  On an
        emitted interval the tangent turns by at most ``angle <= pi/2`` (so ``|delta| < pi``: affine
        maps keep parallel tangents) and the projection onto the chord is monotone, so every foot
        lies on the segment: the distance from the segment is the distance from the line.  With
        sum-to-product forms (no cancellation of large coordinates), ``cross(P(m) - P0, P1 - P0)
        = 4 sin^2(delta / 4) sin(delta / 2) (u x v)`` and ``|P1 - P0| = 2 |sin(delta / 2)| |T(m)|``,
        ``T(m) = -sin(m) u + cos(m) v``: the sag is ``2 sin^2(delta / 4) |u x v| / |T(m)|``.
        """
        delta = self.sweep * (t1 - t0)
        m = self.sweep * (0.5 * (t0 + t1))           # local angle: no absolute angle is rounded
        tangent = float(np.linalg.norm(-math.sin(m) * self.lu + math.cos(m) * self.lv))
        if not tangent > 0.0:
            raise _unsupported(self.edge_id, "the curve's tangent vanishes")
        return 2.0 * math.sin(delta / 4) ** 2 * self.area / tangent

    def refine_sag(self, parameters, allowance, offsets):
        """Bisect (in theta, deterministically) every interval whose sag bound exceeds the
        allowance until none does; sub-intervals keep the tangent and azimuth bounds.

        The first and last drawn chords end at the vertex positions, which may lie up to
        ``offsets`` (start, end) from the curve's ends: such a chord is the analytic chord with
        its end moved, every point of it by at most that offset, so its distance from the curve
        is at most the exact sag plus the offset.  A circle's intervals already meet the
        allowance (its uniform count), so only an end interval with a vertex offset is checked.

        The computed sag is raised by its own rounding before the comparison (fail closed):
        ``|u x v|`` and ``|T(m)|`` of a skew frame cancel to ``eps |u| |v|`` and ``eps (|u| + |v|)``,
        a relative error of at most ``eps x aspect`` each (``|u| |v| <= sigma_max^2``, ``|T| >=
        sigma_min``), so ``(1 + 8 eps x aspect)`` bounds the exact sag."""
        inflate = 1.0 + 8.0 * _EPS * (1.0 if self.circle else self.aspect)
        kept = [float(parameters[0])]
        pending = [(float(a), float(b)) for a, b in zip(parameters[:-1], parameters[1:])]
        pending.reverse()
        while pending:
            a, b = pending.pop()
            offset = max(offsets[0] if a == 0.0 else 0.0, offsets[1] if b == 1.0 else 0.0)
            if (self.circle and offset == 0.0) or self.sag(a, b) * inflate + offset <= allowance:
                kept.append(b)
                continue
            middle = a + 0.5 * (b - a)
            if not a < middle < b:
                raise _unsupported(self.edge_id, "the sag allowance cannot be met at its float resolution")
            if len(kept) + len(pending) + 1 > _MAX_INTERVALS:
                raise _unsupported(self.edge_id, f"needs more than {_MAX_INTERVALS} intervals for its sag")
            pending.append((middle, b))
            pending.append((a, middle))
        return np.asarray(kept)

    def azimuth_division(self, support, step, scale_points) -> _Plan:
        """The plan dividing the azimuth about ``support``'s axis into equal steps: the
        projection ``c' + cos(theta) u' + sin(theta) v'`` (``c'`` relative to the axis point) in
        the support's (radial, circumferential) frame, through :class:`_AngleMap` (the exact root of
        ``A cos(theta) + B sin(theta) = C`` on the monotone branch).

        One domain rule for cylinders and cones, on the PROJECTED conic ``M'`` (which can be far
        thinner than the curve): its aspect x ``max(1, |stored angle|)`` is at most
        :data:`MAX_DISPLAY_ELLIPSE_ASPECT`, and its closest approach to the axis (a lower bound)
        is at least the radius ratio of a scale including ``max(1, |stored angle|) sigma_max(M')``
        (the evaluator rounds ``start + sweep t`` by eps |theta|, moving the projected point by up
        to that times eps)."""
        origin, x_axis, y_axis = _axis_frame(support)
        c3 = self.c - origin
        projected = np.array(((self.lu @ x_axis, self.lv @ x_axis), (self.lu @ y_axis, self.lv @ y_axis)))
        amap = _AngleMap(self.edge_id, projected, (c3 @ x_axis, c3 @ y_axis), 0.0, 0.0, self.sweep)
        stored = max(1.0, abs(self.start), abs(self.start + self.sweep))
        weight = amap.aspect * stored
        if not weight <= MAX_DISPLAY_ELLIPSE_ASPECT:
            raise DisplayDomainRefused(f"edge {self.edge_id}'s projection across a support axis has the aspect "
                                       f"{amap.aspect!r}: aspect x max(1, |stored angle|) = {weight!r} is above "
                                       f"{MAX_DISPLAY_ELLIPSE_ASPECT:g}")
        _check_radius(f"edge {self.edge_id} (closest approach, lower bound)", amap.closest,
                      origin, *scale_points, (stored * amap.sigma_max, 0.0, 0.0))
        start = amap.angle(0.0)
        span = amap.angle(self.sweep) - start
        n = _count(self.edge_id, span, step)

        def build():
            values = self.to_parameters(_angle_division(amap, 0.0, self.sweep, n))
            return _Division(values, lambda t: amap.angle(self.sweep * t) - start)

        return _Plan(n, build)


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
    """Sorted union of division parameters in (0, 1) with the ends (one sort, one pass).

    Every value of ``first`` (the tangent division) is kept.  A value ``v`` of a later
    :class:`_Division` within ``gap`` of a kept value ``k`` (the same division point reached by
    two divisions) is dropped only when its own division's angle at ``k`` equals its angle at
    ``v`` to rounding (``64 eps max(1, |angle|)``, the angles being local to the edge): every
    interval of the union then still lies inside one interval of that division, up to that
    rounding.  Otherwise it is kept (and a cut too close to evaluate apart is refused later)."""
    base = np.unique(np.clip(np.asarray([0.0, 1.0, *first], dtype=float), 0.0, 1.0))
    values = [np.clip(np.asarray(d.values, dtype=float), 0.0, 1.0) for d in others]
    owners = [np.full(len(v), i) for i, v in enumerate(values)]
    if len(base) - 1 > _MAX_INTERVALS:
        raise _unsupported(edge_id, f"needs more than {_MAX_INTERVALS} intervals")
    if not sum(len(v) for v in values):
        return base
    values, owners = np.concatenate(values), np.concatenate(owners)
    order = np.lexsort((owners, values))
    values, owners = values[order], owners[order]
    above = np.searchsorted(base, values)
    accepted = []
    for value, owner, i in zip(values.tolist(), owners.tolist(), above.tolist()):
        near = [k for k in (base[i] if i < len(base) else None, base[i - 1] if i > 0 else None,
                            accepted[-1] if accepted else None)
                if k is not None and abs(float(k) - value) <= gap]
        if near:
            angle = others[owner].angle
            target = angle(value)
            if any(abs(angle(float(k)) - target) <= 64.0 * _EPS * max(1.0, abs(target)) for k in near):
                continue
        accepted.append(value)
        if len(base) - 1 + len(accepted) > _MAX_INTERVALS:
            raise _unsupported(edge_id, f"needs more than {_MAX_INTERVALS} intervals")
    kept = np.unique(np.concatenate((base, np.asarray(accepted, dtype=float))))
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
      angle for an ellipse (exact closed-form inverse, in the arc's local angle),
      ``ceil(turning / angle)`` steps; a circle's count is raised for the sag allowance
      (:meth:`_Conic.tangent_division`);
    * for each incident cylinder or cone, the division of the azimuth about its axis into
      ``ceil(span / angle)`` equal steps, solved in closed form (none for a line whose ends are
      collinear with the axis point on one side: a ruling or a generator).

    Every count is computed and checked against the cap before any division is built.

    then every ellipse interval whose exact sag (:meth:`_Conic.sag`) exceeds the allowance is
    bisected in theta until none does, and so is every end interval whose sag plus its vertex's
    offset from the curve does (an offset above the allowance is refused).  Every interval lies inside one interval of each division,
    so the tangent turns,
    and every incident cylinder's or cone's azimuth changes, by at most ``tol.angle`` over it,
    and its chord leaves the curve by at most the sag allowance (:class:`DisplayTolerance`).  A
    curved edge whose single chord could leave it by more than ``model.tolerance.surface_residual``
    (its exact sag plus its vertex offsets) gets at least two intervals.  The first and last points
    are the edge's vertex positions.
    Inputs are checked against the declared numeric domain first (``OUT_OF_DOMAIN``).  Typed
    refusal (``EDGE_UNSUPPORTED``) for a degenerate curve, an azimuth that is not monotone along
    the edge, an unresolvable sag (at most 4 eps of the coordinates) or more than 4096 intervals.
    An edge id not in the model is a plain :class:`GeometryError`; no raw arithmetic error escapes
    a division (typed ``EDGE_UNSUPPORTED`` at worst).
    """
    if not isinstance(tol, DisplayTolerance):
        raise GeometryError("tol must be a DisplayTolerance")
    edge_id = int(edge_id)
    if edge_id not in model.edges:
        raise GeometryError(f"edge {edge_id} is not in the model")
    edge = model.edges[edge_id]
    supports = _incident_supports(model, edge_id)
    curve = edge.curve
    start = np.asarray(model.vertex_position(edge.start), dtype=float)
    end = np.asarray(model.vertex_position(edge.end), dtype=float)

    # --- declared domain, before any division
    _check_points(f"edge {edge_id}'s vertices", np.vstack((start, end)))
    conic = None
    line = None
    if isinstance(curve, (Straight, LineRange)):
        if isinstance(curve, LineRange):
            _check_points(f"edge {edge_id}'s line carrier", np.vstack((curve.a, curve.b)))
        # the line's geometry is its evaluated ends (the exact evaluator), never the carrier's
        # float fields: a retained range can be far narrower than its carrier
        line = tuple(_evaluate(model, edge_id, [0.0, 1.0]))
        _check_points(f"edge {edge_id}'s line ends", np.vstack(line))
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
    if conic is not None and not conic.circle:
        # the evaluator rounds the absolute angle start + sweep t (by eps |theta|), which moves the
        # tangent by up to aspect x that: the bound holds for aspect x max(1, |stored angle|)
        weight = conic.aspect * max(1.0, abs(conic.start), abs(conic.start + conic.sweep))
        if not weight <= MAX_DISPLAY_ELLIPSE_ASPECT:
            raise DisplayDomainRefused(f"edge {edge_id}'s ellipse has the aspect {conic.aspect!r} at stored angles up "
                                       f"to {max(abs(conic.start), abs(conic.start + conic.sweep))!r}: "
                                       f"aspect x max(1, |angle|) = {weight!r} is above {MAX_DISPLAY_ELLIPSE_ASPECT:g}")
    definition = [start, end] if conic is None else [start, end, conic.c, conic.c + conic.u, conic.c + conic.v]
    if line is not None:
        definition += list(line)
    for support in supports:
        if isinstance(support, Cylinder):
            _check_radius("a cylinder radius", support.radius, support.origin, *definition)

    # --- divisions: every count first (checked against the cap), then the divisions
    try:
        parameters = _parameters(model, edge_id, tol, supports, start, end, line, conic, definition)
    except DisplayTrimRefused:
        raise
    except (ArithmeticError, ValueError) as error:      # no raw arithmetic error escapes
        raise _unsupported(edge_id, f"{type(error).__name__} in the closed-form division: {error}") from error
    return _edge_result(model, edge_id, parameters, start, end)


def _parameters(model, edge_id, tol, supports, start, end, line, conic, definition):
    step = tol.angle
    allowance = _sag_allowance(supports, tol)
    # the drawn ends are the vertices: their offsets from the curve's ends enter the sag bound
    curve_ends = line if line is not None else _evaluate(model, edge_id, [0.0, 1.0])
    offsets = (float(np.linalg.norm(start - curve_ends[0])), float(np.linalg.norm(end - curve_ends[1])))
    # an offset at the coordinates' resolution (4 eps x max(1, |vertex coordinate|), the floor any
    # sag allowance must exceed) is rounding: the vertex is the curve's end to that resolution
    resolution = 4.0 * _EPS * max(1.0, float(np.abs(np.vstack((start, end))).max()))
    offsets = tuple(0.0 if offset <= resolution else offset for offset in offsets)
    if allowance is not None and max(offsets) > allowance:
        raise _unsupported(edge_id, f"a vertex lies farther from its curve ({max(offsets)!r}) than the sag "
                                    f"allowance {allowance!r}")
    if line is not None:
        plans = [_line_azimuth_division(edge_id, s, line[0], line[1], step)
                 for s in supports if isinstance(s, (Cylinder, Cone))]
        return _union(edge_id, [], [plan.build() for plan in plans], 64.0 * _EPS)
    if allowance is not None and allowance <= resolution:
        raise _unsupported(edge_id, f"the sag allowance {allowance!r} is below its coordinates' resolution")
    tangent = conic.tangent_division(step, allowance, model.tolerance.surface_residual, max(offsets))
    plans = [conic.azimuth_division(s, step, definition) for s in supports if isinstance(s, (Cylinder, Cone))]
    # parameters closer than 64 ulps of the stored angle evaluate to the same theta (to rounding)
    gap = 64.0 * _EPS * max(1.0, abs(conic.start), abs(conic.start + conic.sweep)) / abs(conic.sweep)
    parameters = _union(edge_id, tangent.build(), [plan.build() for plan in plans], gap)
    if allowance is not None:
        parameters = conic.refine_sag(parameters, allowance, offsets)
    return parameters


def _evaluate(model, edge_id, parameters) -> np.ndarray:
    try:
        return np.array(model.evaluate_edge_many(edge_id, parameters), dtype=float)
    except GeometryError as error:
        raise _unsupported(edge_id, str(error)) from error


def _edge_result(model, edge_id, parameters, start, end) -> EdgeSamples:
    points = _evaluate(model, edge_id, parameters)
    _check_points(f"edge {edge_id}'s emitted points", points)
    if np.any(np.all(points[1:] == points[:-1], axis=1)):     # before the ends become the vertices
        raise _unsupported(edge_id, "distinct parameters evaluate to one point (lost precision)")
    points[0], points[-1] = start, end
    if np.any(np.all(points[1:] == points[:-1], axis=1)):
        raise _unsupported(edge_id, "distinct parameters evaluate to one point (lost precision)")
    return EdgeSamples(edge_id, _frozen(parameters), _frozen(points))
