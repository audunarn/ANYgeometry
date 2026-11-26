"""Display samples of edges: lines, circles and ellipses on planes, cylinders and cones.

A viewer draws a face as triangles.  Neighbouring faces cannot crack apart when the polyline drawn
along an edge depends only on the edge and the supports of the faces that use it
(:func:`edge_display_samples`): every face, drawing route and caller then draws the same
polyline there.

Scope (owner decision 2026-11-25, N1): only straight lines (``Straight``, and a ``LineRange`` whose
carrier is the edge's segment: exact domain [0, 1], owner decision N1 round 4), circles
(``Arc``, circular ``EllipticArc``) and ellipses (``EllipticArc``), on faces supported by a
``Plane``, ``Cylinder`` or ``Cone``.  Each is divided in closed form, so every bound below holds by
construction (to the edge's documented rounding allowance, :func:`display_rounding_allowance`, at
most 2 % of the angle and sag), not by measurement:

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
from fractions import Fraction

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
    "DisplayRoundingAllowance",
    "EdgeSamples",
    "REFUSAL_CODES",
    "MAX_DISPLAY_COORDINATE",
    "MAX_DISPLAY_ANGLE",
    "MIN_DISPLAY_RADIUS_RATIO",
    "MAX_DISPLAY_ELLIPSE_ASPECT",
    "MIN_DISPLAY_SWEEP",
    "edge_display_samples",
    "display_rounding_allowance",
    "FaceDisplayMesh",
    "face_display_mesh",
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
#: cylinder radius and every point used for an azimuth (a line's ends by their own projections,
#: except one on the axis to the frame's rounding, 8 eps max(1, |coordinate|), at a cone's exact
#: apex: a zero radius and within the model's length tolerance of the apex point, N1 round 4,
#: and its interior closest approach, at the scale of the edge's definition points; a lower bound
#: of a cylinder or cone arc's, at the scale of its definition points and max(1, |stored angle|)
#: sigma_max of its projected frame (the evaluator rounds start + sweep t), not the support
#: origin, and separately above 64 eps |c - o|, the rounding of its centre's offset from the axis
#: point, N1 round 5).  A line with both ends on the axis, or
#: an end on the axis of a cylinder or frustum, is refused.
#: An azimuth computed from coordinates of size |p| at distance r from the axis is rounded by
#: about eps |p| / r: at the bound 2.2e-7 rad, below the smallest display angle (1e-6).
MIN_DISPLAY_RADIUS_RATIO = 1.0e-9
#: Largest ``a / b x max(1, |start|, |start + sweep|)`` of a non-circular ellipse arc (``a``, ``b``
#: its principal semi-axes, the singular values of ``[u v]``).  The tangent direction moves up to
#: ``a / b`` times faster than the arc angle, and the evaluator rounds the absolute angle
#: ``start + sweep t`` by eps |theta|: the tangent moves by at most ``1e4 x eps = 2.2e-12`` rad,
#: six orders below the smallest display angle.  The same bound applies to the arc's projection
#: across every incident cylinder's or cone's axis (the 2x2 projected frame ``M'``, which can be
#: far thinner than the curve; a circle seen along the axis has aspect 1, so its stored angles
#: are bounded by 1e4 there): the azimuth is resolved through ``M'^-1`` and back, so its
#: rounding is at most ``aspect(M') x eps`` (and the stored angle's, as for the tangent).
#: Owner decision (N1 round 6, the recommended option): lowered from 1e6, where conic rounding
#: amplified by the aspect was measured beyond the published bound (rounds 3 and 6, at products
#: 9.9e5 and 6.8e5, within a factor ~1e2 of the old bound; exact boundary cases at 1e6 were
#: refused by rounding in rounds 5 and 6).  The bound is inclusive: the computed
#: product is compared with ``1e4 x (1 + 4 eps)`` (:func:`_aspect_admitted`).
MAX_DISPLAY_ELLIPSE_ASPECT = 1.0e4
#: Smallest ``|sweep|`` of an ellipse arc or a circular arc (radians, inclusive), checked before
#: any division (``OUT_OF_DOMAIN`` below it).  Owner decision (N1 round 8, the recommended
#: option): both round-8 MEDIUM findings needed sweeps of 1e-16 and 5e-11 rad, where the stored
#: angles' own float spacing is a sizeable share of the sweep: the tangent inverse rounded a
#: forced cut onto an end, and the azimuth branch test saw both ends inside its snapping band.
#: With ``|stored angle| <= 1e4`` (every non-circular conic, aspect >= 1, and every conic with a
#: cylinder or cone support, by :data:`MAX_DISPLAY_ELLIPSE_ASPECT`) the float spacing of the
#: stored angles is at most 1.8e-12 rad, four orders below the minimum sweep, so the tangent
#: inverse and the azimuth branch see a resolved interval; a lone circle's stored angles (up to
#: :data:`MAX_DISPLAY_ANGLE`, spacing 1.2e-10) still lie two orders below it.
MIN_DISPLAY_SWEEP = 1.0e-8

_MAX_INTERVALS = 4096      # per edge
_STEP_ROUNDING = 4.0 * float(np.finfo(float).eps)   # relative rounding of a computed span
_MIN_ANGLE = 1.0e-6        # smallest display angle (radians)
_EPS = float(np.finfo(float).eps)
_CIRCLE_RTOL = 8.0 * _EPS  # |u| = |v| and u.v = 0 to a few eps relative: a circle (else an ellipse)
_ON_AXIS = 8.0 * _EPS      # a projection within 8 eps max(1, |coordinates|) of an axis is on it
_DEGENERATE = 1.0e-12      # |det| <= this * |M|^2: a projected or in-plane frame is degenerate
_TWO_PI = 2.0 * math.pi
#: N1 round 10 (owner decision, corrected): a sanity check for self-inconsistent supports only.
#: A cylinder's or cone's STORED frame is admitted when ``|axis . radial|``, ``|axis .
#: circumferential|`` and ``||radial| - 1|`` (absolute, on unit vectors) are at most this, else
#: OUT_OF_DOMAIN.  Ordinary supports are off by rounding (measured up to 1.8e-11, 55x below, over
#: 20,000 random seeds); no bound uses the stored frame except through the support's own normal,
#: which it tilts by at most about 2e-9 rad here (docs: domain table)
_SUPPORT_FRAME_TOL = 1.0e-9
#: the module's own azimuth frame (:func:`_axis_frame`) about the stored unit axis is within this
#: of an exactly orthonormal frame about that axis, per unit vector (e1 one normalisation, about
#: eps; e2 = axis x e1 that plus the cross product's rounding, about 2.5 eps): every projection
#: onto it carries ``_FRAME_ERROR |vector|`` beyond its dot product's own rounding
_FRAME_ERROR = 4.0 * _EPS
#: relative slack of the inclusive aspect bound: an aspect computed from exactly orthogonal axes
#: at the bound (a ratio of two rounded norms, or an SVD quotient) rounds by a few eps (N1 round 6)
_ASPECT_SLACK = 4.0 * _EPS
#: The rounding allowance may be at most this share of the requested angle and sag allowance;
#: a finer request is refused (the declared domain, tightened in N1 round 3).
_ALLOWANCE_SHARE = 0.02


class DisplayTrimRefused(GeometryError):
    """A typed refusal: ``code`` is one of :data:`REFUSAL_CODES`."""

    def __init__(self, code: str, message: str):
        # owner decision N2 r5 (1): exact types only and fixed refusal text (nothing of the caller's
        # formatted, hashed or compared before its type is known)
        if not (type(code) is str and code in REFUSAL_CODES):     # GeometryError is a ValueError
            raise GeometryError("unknown display refusal code (not one of REFUSAL_CODES)")
        if type(message) is not str:
            raise GeometryError("a display refusal message must be a str")
        super().__init__(code + ": " + message)
        self.code = code


class DisplayDomainRefused(DisplayTrimRefused):
    """The input lies outside the declared numeric domain (code ``OUT_OF_DOMAIN``)."""

    def __init__(self, message: str):
        super().__init__("OUT_OF_DOMAIN", message)


#: owner decision N2 r5 (1): the exact concrete types accepted (``type(x) is``, never an ABC, never
#: a subclass: ``bool`` is not ``int`` here, ``np.timedelta64`` is not an integer)
_REAL_TYPES = (float, int, np.float32, np.float64)
_ID_TYPES = (int, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)
#: an int beyond this magnitude is refused before ``float()`` (which would overflow near 2**1024)
_INT_FLOAT_BOUND = 2 ** 1023
#: edge ids are integers with ``|id| < 2**63`` (checked arithmetically on a Python int)
_ID_BOUND = 2 ** 63


def _exact_type_in(value, types) -> bool:
    kind = type(value)
    return any(kind is t for t in types)


def _real_argument(value, message: str) -> float:
    """A float, int (not bool), np.float32 or np.float64 (exact types) as a float, else
    :class:`GeometryError` with fixed text (owner decision N2 round 5 (1): no ABC, no attribute or
    string conversion of a caller object; an int is compared with a bound before ``float()``)."""
    if not _exact_type_in(value, _REAL_TYPES):
        raise GeometryError(f"{message} (not a float, int, np.float32 or np.float64)")
    if type(value) is int and not -_INT_FLOAT_BOUND < value < _INT_FLOAT_BOUND:
        raise GeometryError(f"{message} (an int out of float range)")
    return float(value)


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
        angle = _real_argument(self.angle, "display angle must be finite and in [1e-6, pi/2]")
        if not (math.isfinite(angle) and _MIN_ANGLE <= angle <= math.pi / 2):
            raise GeometryError("display angle must be finite and in [1e-6, pi/2]")
        object.__setattr__(self, "angle", angle)
        if self.sag is not None:
            sag = _real_argument(self.sag, "display sag must be finite and positive")
            if not (math.isfinite(sag) and sag > 0.0):
                raise GeometryError("display sag must be finite and positive")
            object.__setattr__(self, "sag", sag)


@dataclass(frozen=True)
class DisplayRoundingAllowance:
    """What "holds to rounding" means for one edge (:func:`display_rounding_allowance`).

    ``angle``: absolute radians: every emitted interval's tangent turns, and every incident
    cylinder's or cone's azimuth changes, by at most ``tol.angle + angle``.  ``sag``: absolute
    model units: every drawn chord lies within the sag allowance ``+ sag`` of the curve (``None``
    when the edge has no sag allowance).  ``axis_floor``: a summary only,
    ``MIN_DISPLAY_RADIUS_RATIO x scale`` with ``scale`` over the edge's definition points and
    EVERY incident support's origin; it is not a measurement threshold for any one support (it can
    exceed a small support's every distance from its axis, N1 rounds 5 and 6): callers measuring
    support turning use ``axis_floors``, one per support.
    ``axis_floors`` (N1 round 5): each incident cylinder's or cone's OWN floor, in the order of
    ``model.faces_using_edge`` (first use): the azimuth about that support is defined, and
    bounded, at points at least that far from its axis (a line: ``MIN_DISPLAY_RADIUS_RATIO`` x the
    scale of its definition points and that support's origin; a conic: the larger of its two
    closest-approach terms, :data:`MIN_DISPLAY_RADIUS_RATIO`).  A floor shared across supports can
    exceed a small support's every distance.
    :func:`edge_display_samples` refuses a tolerance for which ``angle`` exceeds 2 % of
    ``tol.angle`` (``OUT_OF_DOMAIN``) or ``sag`` 2 % of the sag allowance (``EDGE_UNSUPPORTED``).
    """

    angle: float
    sag: float | None
    axis_floor: float
    axis_floors: tuple = ()


#: NumPy's default error state, under which every MODEL call (positions, frames, evaluation) runs
#: (owner decision N2 round 3 (4)): the module's own arithmetic runs under a local
#: ``np.errstate(all="ignore")`` with its explicit finiteness and domain guards deciding, so
#: neither depends on the caller's ``np.seterr``; the model keeps the state its results were
#: measured under (its own fast paths check for exactly this state).
_MODEL_ERRSTATE = {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}


def _model_call(call, *args):
    with np.errstate(call=None, **_MODEL_ERRSTATE):
        return call(*args)


def _norm(vector) -> float:
    return math.hypot(*(float(x) for x in vector))


def _scale(points) -> float:
    """max(1, largest absolute coordinate of ``points``)."""
    return max(1.0, float(np.abs(np.vstack([np.asarray(p, dtype=float) for p in points])).max()))


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


def _id_text(value) -> str:
    """An int formatted into a message only if |v| < 10**18 (checked arithmetically, never by
    ``str``), else fixed text: a model-produced id may have thousands of digits."""
    if -10 ** 18 < value < 10 ** 18:
        return str(int(value))
    return "<id out of display range>"


def _unsupported(edge_id, message):
    return DisplayTrimRefused("EDGE_UNSUPPORTED", f"edge {_id_text(edge_id)}: {message}")


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
      monotone exactly when ``g`` keeps its sign.  The sign is decided only with margin (N1
      round 9, owner decision): ``g`` at the ends, at every interior extreme and at the
      interval's midpoint must exceed :attr:`g_bound` (a rigorous bound on the rounding of the
      computed ``g``, :meth:`_g_rounding`) in magnitude, else the azimuth is near-stationary and
      the edge is ``OUT_OF_DOMAIN`` (no branch is inferred from a value within its bound, and
      nothing is snapped to 0); a sign change with margin is ``EDGE_UNSUPPORTED``.
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

    def __init__(self, edge_id, matrix, offset, phase, theta_a, theta_b, matrix_error=0.0, offset_error=0.0):
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
        # Singular values without cancellation: with det > 0, sigma_max + sigma_min and
        # sigma_max - sigma_min are the square roots of two sums of squares; sigma_min = det /
        # sigma_max (never the difference of two close numbers).
        self.sigma_max = 0.5 * (math.hypot(m[0, 0] + m[1, 1], m[1, 0] - m[0, 1])
                                + math.hypot(m[0, 0] - m[1, 1], m[0, 1] + m[1, 0]))
        self.aspect = self.sigma_max * self.sigma_max / det
        # sign of g = 1 + z.w(x) over the x interval: its extremes are at the ends or where
        # x - arg z is a multiple of pi (strictly inside).  N1 round 9: the sign is decided only
        # where |g| exceeds its rounding bound at every end, interior extreme and the midpoint;
        # within it the azimuth is near-stationary and the branch is not decidable (refused
        # OUT_OF_DOMAIN, never snapped).
        xa, xb = sorted((self._x(theta_a), self._x(theta_b)))
        radius = float(math.hypot(*self.z))
        middle = math.atan2(self.z[1], self.z[0])
        inside = []
        k = math.ceil((xa - middle) / math.pi)
        while middle + k * math.pi < xb and len(inside) < 8:
            if middle + k * math.pi > xa:
                inside.append((k, middle + k * math.pi))
            k += 1
        checks = [xa, xb] + [x for _k, x in inside] + [0.5 * (xa + xb)]
        values = [1.0 + radius * math.cos(x - middle) for x in checks]
        self.g_bound = self._g_rounding(det, float(math.hypot(*offset)), radius,
                                        max(abs(xa), abs(xb)), float(matrix_error), float(offset_error))
        if not all(abs(g) > self.g_bound for g in values):
            raise DisplayDomainRefused(f"edge {_id_text(edge_id)}: azimuth near-stationary: branch not decidable (|g| = "
                                       f"{min(abs(g) for g in values)!r} along the edge is within its rounding "
                                       f"bound {self.g_bound!r})")
        if all(g > 0.0 for g in values):
            self.branch = 1.0
        elif all(g < 0.0 for g in values):
            self.branch = -1.0
        else:
            raise _unsupported(edge_id, "the azimuth about a support axis is not monotone along the edge "
                                        "(its projection reaches or turns back round the axis)")
        self.z_radius = radius
        # Closest approach of the curve to the origin over the interval, bounded below by
        # sigma_min(N) min |z + w(x)|.  |z + w(x)| is the distance of the unit circle point w(x)
        # from -z: smallest at x = arg z + pi (odd k) where it is exactly | |z| - 1 |, else at an
        # end; each is evaluated directly (no expansion of |z + w|^2 that would cancel).
        distances = [math.hypot(self.z[0] + math.cos(x), self.z[1] + math.sin(x)) for x in (xa, xb)]
        distances += [abs(radius - 1.0) for k, _x in inside if k % 2]
        self.closest = det / self.sigma_max * min(distances)

    def _g_rounding(self, det, offset_norm, radius, x_max, matrix_error, offset_error) -> float:
        """A rigorous bound on ``|computed g - g|`` at any point of the interval, ``g`` the exact
        ``1 + z.w`` of the exact curve (N1 round 9; first order, doubled for the second-order
        terms).  Three parts, with ``s = sigma_min(N) = det / sigma_max``:

        * inputs: ``g = cross(p, p') / det N`` (``p = c' + N w``, ``p' = N w'``), so absolute
          errors ``E_N`` of the frame and ``E_c`` of the offset (``matrix_error``,
          ``offset_error``) move ``g`` by at most ``(E_c + E_N) / s + (|c'| + sigma_max) E_N /
          (sigma_max s)``;
        * ``z = N^-1 c'`` (adjugate / det): the computed ``det`` is off by a relative
          ``eps |N|_F^2 / det <= 2 eps aspect`` and the product by ``2 eps`` per term, so ``|z|``
          (at most ``|c'| / s``) moves by ``8 eps (aspect + 2) |c'| / s``;
        * the evaluation ``1 + |z| cos(x - arg z)``: ``|z|`` (hypot) and ``arg z`` (atan2) to an
          ulp, ``x - arg z`` to ``eps (|x| + pi)``, cos to an ulp, the product and sum to an ulp
          each: ``8 eps (1 + |z| (4 + |x|))``.
        """
        s = det / self.sigma_max
        inputs = (offset_error + matrix_error) / s + (offset_norm + self.sigma_max) * matrix_error / (self.sigma_max * s)
        inverse = 8.0 * _EPS * (self.sigma_max * self.sigma_max / det + 2.0) * offset_norm / s
        evaluation = 8.0 * _EPS * (1.0 + radius * (4.0 + x_max))
        return 2.0 * (inputs + inverse + evaluation)

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
    rounding: float = 0.0      # absolute azimuth rounding of this division (radians)
    floor: float = 0.0         # this support's own axis floor (DisplayRoundingAllowance.axis_floors)


@dataclass(frozen=True)
class _Division:
    """Edge parameters dividing a monotone angle (``angle(t)``, radians, small: local to the
    edge) into equal steps.  ``angle`` lets the union measure, in this division's own angle, how
    far a near-duplicate cut it drops is from the cut it keeps, and verify every merged interval
    (N1 round 7).  ``protected``: the cuts are mandatory (the minimum-two-interval split) and are
    never merged away.  ``substituted``: cuts placed otherwise (verified after the union)."""

    values: list
    angle: object = None
    protected: bool = False
    #: cuts not from the equal-angle construction (a forced split's midpoint, N1 round 8): the
    #: union's "by construction" argument does not cover them, so their intervals are verified
    substituted: tuple = ()


# ---------------------------------------------------------------------------------------------
# domain
# ---------------------------------------------------------------------------------------------
def _check_angle(name, *values):
    for value in values:
        if not (math.isfinite(value) and abs(value) <= MAX_DISPLAY_ANGLE):
            raise DisplayDomainRefused(f"{name} angle {float(value)!r} is outside |angle| <= "
                                       f"{MAX_DISPLAY_ANGLE:g} rad")


def _check_sweep(name, sweep):
    """``|sweep| >= MIN_DISPLAY_SWEEP`` (inclusive; NaN is refused), before any division."""
    if not abs(sweep) >= MIN_DISPLAY_SWEEP:
        raise DisplayDomainRefused(f"{name} sweep {sweep!r} is outside |sweep| >= {MIN_DISPLAY_SWEEP:g} rad")


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
        _check_support_frame(support)


def _check_support_frame(support):
    """N1 round 10 (owner decision, the recommended option): a cylinder or cone whose STORED frame
    is not orthonormal to :data:`_SUPPORT_FRAME_TOL` (any of the lengths ``||axis| - 1|``,
    ``||radial| - 1|``, ``||circumferential| - 1|`` or the pairwise dot products ``|axis .
    radial|``, ``|axis . circumferential|``, ``|radial . circumferential|`` above 1e-9, absolute;
    NaN refused; N1 round 11 added the axis and circumferential lengths and the radial .
    circumferential product: a seed whose squared norm is subnormal stored an axis 1.125 long) is
    ``OUT_OF_DOMAIN``: its own evaluate() (built on that frame) then disagrees with its declared
    axis geometry.  A sanity check for self-inconsistent supports only: a stored frame's ordinary
    rounding never enters a bound (the azimuth uses the module's own frame).  (A
    radial seed nearly along the axis, orthogonalised by the support, came out 2.2e-5 off
    orthogonal and moved the azimuth's ``g`` by 1.1e-5, round 10.)  The azimuth itself never
    uses the stored radial or circumferential direction (:func:`_axis_frame`)."""
    axis = np.asarray(support.axis, dtype=float)
    radial = np.asarray(support.radial_direction, dtype=float)
    circumferential = np.asarray(support.circumferential_direction, dtype=float)
    defects = {"||axis| - 1|": abs(_norm(axis) - 1.0),
               "||radial| - 1|": abs(_norm(radial) - 1.0),
               "||circumferential| - 1|": abs(_norm(circumferential) - 1.0),
               "|axis . radial|": abs(float(axis @ radial)),
               "|axis . circumferential|": abs(float(axis @ circumferential)),
               "|radial . circumferential|": abs(float(radial @ circumferential))}
    if not all(d <= _SUPPORT_FRAME_TOL for d in defects.values()):
        found = ", ".join(f"{name} = {value!r}" for name, value in defects.items())
        raise DisplayDomainRefused(f"a {_support_name(support)} support frame not orthonormal ({found}; each must "
                                   f"be at most {_SUPPORT_FRAME_TOL!r}): its evaluate() disagrees with its declared "
                                   f"axis geometry")


def _aspect_admitted(weight) -> bool:
    """``aspect x max(1, |stored angle|) <= MAX_DISPLAY_ELLIPSE_ASPECT``, inclusive, for a COMPUTED
    product: compared with the bound raised by :data:`_ASPECT_SLACK` (4 eps relative), the
    rounding of an aspect computed from orthogonal axes exactly at the bound (N1 round 6: the
    SVD quotient of ``u = (1e6, 1e6, 0)``, ``v = (-1, 1, 0)`` came out ``1e6 (1 + 2.2e-16)``).
    NaN is not admitted."""
    return weight <= MAX_DISPLAY_ELLIPSE_ASPECT * (1.0 + _ASPECT_SLACK)


def _support_name(support) -> str:
    """The module's own name for a cylinder or cone support (never the caller class's name)."""
    return "Cylinder" if isinstance(support, Cylinder) else "Cone"


def _check_radius(name, distance, *points):
    distance = float(distance)
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
                                     f"edge {_id_text(edge_id)} is used by face {_id_text(face_id)}, which has an explicit parameterization")
        if not isinstance(support, (Plane, Cylinder, Cone)):
            raise DisplayTrimRefused("SUPPORT_UNSUPPORTED", f"edge {_id_text(edge_id)} is used by face {_id_text(face_id)} on a "
                                                            "support that is not a plane, cylinder or cone")
        if not any(support is s for s in found):
            found.append(support)
    return tuple(found)


def _axis_frame(support):
    """The axis point and the module's OWN orthonormal frame ``(e1, e2)`` about ``support``'s unit
    axis (N1 round 10, owner decision): ``e1`` the normalised cross product of the axis with the
    basis vector least aligned with it (exact before the normalisation: its components are axis
    components or 0; its norm is at least sqrt(2/3)), ``e2 = axis x e1``.  Never the support's
    stored radial / circumferential directions: the azimuth's turning and every branch decision
    are independent of the frame's phase, and a stored frame off orthonormal cannot enter them
    (round 10: a radial 2.2e-5 off orthogonal moved ``g`` by 1.1e-5 against a bound of 1.2e-13).
    The stored axis is normalised here first (N1 round 11: it need not be unit, a seed whose
    squared norm is subnormal stored it 0.9999999995 long, which entered ``e2`` directly and
    turned an interval 2e-11 rad beyond the bound); within :data:`_FRAME_ERROR` of an exact
    orthonormal frame about the axis direction."""
    origin = np.asarray(support.origin, dtype=float)
    axis = np.asarray(support.axis, dtype=float)
    axis = axis / _norm(axis)
    basis = np.zeros(3)
    basis[int(np.argmin(np.abs(axis)))] = 1.0
    cross = np.cross(axis, basis)
    e1 = cross / _norm(cross)
    return origin, e1, np.cross(axis, e1)


# ---------------------------------------------------------------------------------------------
# divisions
# ---------------------------------------------------------------------------------------------
def _at_apex(support, point, length) -> bool:
    """Only a cone with an EXACTLY zero radius has a point on its axis, its apex (``origin`` for
    ``radius_start == 0``, ``origin + height axis`` for ``radius_end == 0``); ``point`` is at it
    when within the model's length tolerance plus the apex's own float rounding (``8 eps
    max(1, |apex coordinate|)``), the axial position included.  A frustum (both radii positive,
    however small) has no apex (N1 round 4).  The apex uses the STORED axis, not normalised (N1
    round 11 audit): it is the support's own evaluated point (``evaluate`` forms ``origin + v
    height axis`` with the stored axis), so its length is part of where the apex is."""
    if not isinstance(support, Cone):
        return False
    origin = np.asarray(support.origin, dtype=float)
    apexes = []
    if support.radius_start == 0.0:
        apexes.append(origin)
    if support.radius_end == 0.0:
        apexes.append(origin + support.height * np.asarray(support.axis, dtype=float))
    point = np.asarray(point, dtype=float)
    return any(_norm(point - apex) <= length + _ON_AXIS * max(1.0, float(np.abs(apex).max())) for apex in apexes)


def _line_azimuth_division(edge_id, support, a, b, step, scale_points, apex_length=0.0) -> "_Plan":
    """The plan (count first, then the edge parameters) dividing a line's azimuth about
    ``support``'s axis into equal steps.

    ``a``, ``b`` are the line's evaluated ends (the exact evaluator's).  Every decision uses the
    quantity that is accurate for it, never a reconstruction such as ``A' + P(b - a)`` (which
    cancels an end's own small radius):

    * each end's radius: its OWN projection ``A' = P(a - o)``, ``B' = P(b - o)`` relative to the
      axis point ``o`` (rounded by ``(4 eps + _FRAME_ERROR) |p - o|``, the module frame of
      :func:`_axis_frame` included, N1 round 10);
    * the direction: the projection ``D' = P(b - a)`` of the exact difference (a ruling's is
      exactly 0 where its separately rounded ends are not);
    * the segment's interior closest approach, from each end's estimate ``t* = -(A'.D') / |D'|^2``
      and ``1 - (B'.D') / |D'|^2``, distance ``|P' x D'| / |D'|``, counted when either estimate
      lies inside the segment beyond its ends' rounding: checked BEFORE any end shortcut (the
      conservative branch: a near-axis end cannot hide an interior crossing).

    Then: an end on the axis (projected radius ``<= 8 eps max(1, |end|, |axis point|)``, the
    frame's rounding) is only possible at a cone's exact apex, within the model's ``apex_length``
    tolerance of it (:func:`_at_apex`; elsewhere ``OUT_OF_DOMAIN``): the projection is a ray, constant
    azimuth; both ends on the axis: ``OUT_OF_DOMAIN``.  Both ``A' x D'`` and ``B' x D'`` within
    ``8 eps |P'| |D'|`` (``D' = 0`` included; when the two disagree the line is divided):
    collinear, constant azimuth when ``A'``, ``B'`` lie on one side, else (or when collinear with
    an interior closest approach) the line crosses the axis, ``EDGE_UNSUPPORTED``.  Otherwise
    ``A' + t D'`` misses the axis point, its azimuth is monotone (``cross(A' + t D', D')`` is
    constant) and subtends less than pi, so the wrapped difference of the ends' own azimuths is
    its span.  A target azimuth ``phi`` is reached at ``t = -cross(A', e) / cross(D', e)``,
    ``e = (cos phi, sin phi)``.

    Every off-axis end and the interior closest approach are checked against the radius bound at
    the scale of ``scale_points`` (the edge's definition points: an evaluated point
    ``a + t (b - a)`` is rounded at the scale of the line's ends) and the axis point.

    The plan's ``rounding`` (:func:`display_rounding_allowance`): with ``G = err(A') + err(B') +
    err(D') + 8 eps max(|A'|, |B'|) + 2 X`` (how far the line's true position across the axis can
    be from the computed one; ``X = eps max |coordinate|`` of the evaluated ends, the absolute
    rounding of each end, N1 round 4), ``2 G / r`` over the smallest distance ``r`` used (each cut, or the
    collinearity of a constant azimuth, is off by at most ``G / r``), plus ``4 E / r`` when
    interior samples are drawn (``E = 8 eps (|a| + |b| + |o|)``: an evaluated sample's rounding);
    a ray from an apex: ``2 (rho + err + X) / axis_floor`` (``rho`` the apex end's projected radius:
    the chord's azimuth leaves the ray's by at most that over the distance, wherever an azimuth
    is defined).
    """
    origin, x_axis, y_axis = _axis_frame(support)
    a3, b3, d3 = a - origin, b - origin, b - a
    a2 = np.array((float(a3 @ x_axis), float(a3 @ y_axis)))
    b2 = np.array((float(b3 @ x_axis), float(b3 @ y_axis)))
    d2 = np.array((float(d3 @ x_axis), float(d3 @ y_axis)))
    ra, rb = math.hypot(*a2), math.hypot(*b2)
    length = math.hypot(*d2)
    # each projection: the subtraction and the dot product (4 eps) plus the module frame's own
    # deviation from an exact orthonormal frame (_FRAME_ERROR, N1 round 10)
    projection = 4.0 * _EPS + _FRAME_ERROR
    err_a, err_b, err_d = projection * _norm(a3), projection * _norm(b3), projection * _norm(d3)
    ends = _line_end_rounding(a, b)
    rounding = [_ON_AXIS * max(1.0, float(np.abs(p).max()), float(np.abs(origin).max())) for p in (a, b)]
    on_axis = [ra <= rounding[0], rb <= rounding[1]]
    points = [*scale_points, origin]
    scale = _scale(points)
    if all(on_axis):
        raise DisplayDomainRefused(f"edge {_id_text(edge_id)} lies on a support axis: it has no azimuth")
    if any(on and not _at_apex(support, p, apex_length) for on, p in zip(on_axis, (a, b))):
        raise DisplayDomainRefused(f"edge {_id_text(edge_id)} ends on a support axis away from an exact cone apex (a "
                                   f"cylinder, a frustum, or farther than the model length from the apex): "
                                   f"it has no azimuth there")
    for name, distance, on in (("start", ra, on_axis[0]), ("end", rb, on_axis[1])):
        if not on:
            _check_radius(f"edge {_id_text(edge_id)}'s {name}", distance, *points)
    cross_a = float(a2[0] * d2[1] - a2[1] * d2[0])
    cross_b = float(b2[0] * d2[1] - b2[1] * d2[0])
    collinear = not length > 0.0 or (abs(cross_a) <= 8.0 * _EPS * ra * length
                                     and abs(cross_b) <= 8.0 * _EPS * rb * length)
    closest = None
    if length > 0.0:
        found = []
        for t, c, err in ((-float(a2 @ d2) / length / length, cross_a, err_a),
                          (1.0 - float(b2 @ d2) / length / length, cross_b, err_b)):
            if t * length > rounding[0] + err and (1.0 - t) * length > rounding[1] + err:
                found.append((abs(c) / length, t))
        if found:
            if collinear:
                raise _unsupported(edge_id, "the line's projection passes through a support axis")
            closest, t_star = min(found)
            _check_radius(f"edge {_id_text(edge_id)}'s closest approach (t = {t_star:.17g})", closest, a + t_star * d3, *points)
    if any(on_axis):
        rho = (ra + err_a if on_axis[0] else rb + err_b) + ends
        return _Plan(1, lambda: _Division([]), 2.0 * rho / (MIN_DISPLAY_RADIUS_RATIO * scale),
                     MIN_DISPLAY_RADIUS_RATIO * scale)
    spread = err_a + err_b + err_d + 8.0 * _EPS * max(ra, rb) + 2.0 * ends
    if collinear:
        if float(a2 @ b2) > 0.0:
            return _Plan(1, lambda: _Division([]), 2.0 * spread / min(ra, rb), MIN_DISPLAY_RADIUS_RATIO * scale)
        raise _unsupported(edge_id, "the line's projection passes through a support axis")
    nearest = min(ra, rb, math.inf if closest is None else closest)
    pa = math.atan2(a2[1], a2[0])
    span = (math.atan2(b2[1], b2[0]) - pa + math.pi) % _TWO_PI - math.pi
    n = _count(edge_id, span, step)
    sample = 8.0 * _EPS * (_norm(a) + _norm(b) + _norm(origin))
    rounding_term = 2.0 * spread / nearest + (4.0 * sample / nearest if n > 1 else 0.0)

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

    return _Plan(n, build, rounding_term, MIN_DISPLAY_RADIUS_RATIO * scale)


def _line_end_rounding(a, b) -> float:
    """``X = eps max |coordinate|`` of a line's evaluated ends: each end is the true end rounded to
    the nearest float per coordinate (at most ``eps/2`` relative each), so it lies within ``X`` of
    it; the line terms of :func:`display_rounding_allowance` carry it (N1 round 4)."""
    return _EPS * max(float(np.abs(np.asarray(a, dtype=float)).max()), float(np.abs(np.asarray(b, dtype=float)).max()))


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
        self.circle = (abs(nu - nv) <= _CIRCLE_RTOL * max(nu, nv)
                       and abs(float(self.u @ self.v)) <= _CIRCLE_RTOL * nu * nv)
        singular = np.linalg.svd(np.column_stack((self.u, self.v)), compute_uv=False)
        self.aspect = float(singular[0] / singular[1]) if singular[1] > 0.0 else math.inf
        self.theta_star = max(1.0, abs(self.start), abs(self.start + self.sweep))
        # the declared aspect bound before the parallel-axes test (N1 round 5): a nondegenerate
        # ellipse (sigma_min above 64 eps sigma_max, its float resolution) beyond the bound is
        # OUT_OF_DOMAIN; only axes parallel to that resolution are EDGE_UNSUPPORTED
        weight = self.aspect * self.theta_star
        if not self.circle and not _aspect_admitted(weight) and singular[1] > 64.0 * _EPS * singular[0]:
            raise DisplayDomainRefused(f"edge {_id_text(edge_id)}'s ellipse has the aspect {self.aspect!r} at stored angles up "
                                       f"to {max(abs(self.start), abs(self.start + self.sweep))!r}: "
                                       f"aspect x max(1, |angle|) = {weight!r} is above {MAX_DISPLAY_ELLIPSE_ASPECT:g}")
        e1 = self.u / nu
        rest = self.v - float(self.v @ e1) * e1
        nr = float(np.linalg.norm(rest))
        if not nr > _DEGENERATE * nv:
            raise _unsupported(edge_id, "the ellipse's axis vectors are parallel")
        self.area = float(np.linalg.norm(np.cross(self.u, self.v)))    # |u x v|
        self.radius = max(nu, nv)                     # a circle's radius (conservative to 8 eps)
        # rounding of an evaluated point c + cos(theta) u + sin(theta) v (theta = start + sweep t,
        # rounded by 2 eps max(1, |stored angle|), moving the point by sigma_max times that): the
        # E of display_rounding_allowance
        self.point_rounding = 8.0 * _EPS * (_norm(self.c) + nu + nv
                                            + float(singular[0]) * self.theta_star)
        cs, sn = math.cos(self.start), math.sin(self.start)
        self.lu = cs * self.u + sn * self.v           # the local frame at s = 0
        self.lv = -sn * self.u + cs * self.v
        self._lu, self._lv = self.lu.tolist(), self.lv.tolist()     # floats for the sag's hot loop
        nu, nv = float(np.linalg.norm(self.lu)), float(np.linalg.norm(self.lv))
        e1 = self.lu / nu
        rest = self.lv - float(self.lv @ e1) * e1
        # in-plane matrix of (u', v'): upper triangular with a positive diagonal (det > 0).  The
        # small diagonal is the RESIDUAL's norm |v' - (v'.e1) e1| (absolute error eps |v'|, a
        # relative eps x aspect), never v'.e2 with e2 = rest / |rest|: that dot product adds
        # (v'.e1)(e1.rest) / |rest|, where e1.rest is the rounding eps |v'| of the residual's
        # orthogonality, a relative error eps x aspect^2 (N1 round 6: at aspect 6.8e5 an interval
        # turned 1.0e-5 rad beyond the bound)
        self.plane = np.array(((nu, float(self.lv @ e1)), (0.0, float(np.linalg.norm(rest)))))

    def to_parameters(self, local):
        return [s / self.sweep for s in local]

    def tangent_division(self, step, allowance, residual, offset=0.0) -> _Plan:
        """The plan of the tangent division (the sag is refined afterwards, :meth:`sag`).

        Circle: uniform in theta (the tangent turns uniformly).  A chord over a theta step
        ``delta`` leaves the circle by exactly ``rho (1 - cos(delta / 2)) = 2 rho sin^2(delta / 4)``
        and sub-intervals sag less, so ``delta <= 4 asin(sqrt(sag / (2 rho)))`` bounds it.

        Ellipse: uniform in the tangent angle ``psi`` (closed-form inverse, :class:`_AngleMap`).

        A single interval is split in two when its DRAWN chord could leave the curve by more than
        ``residual``: ``offset`` is the larger distance of a vertex from its curve end, measured
        (never zeroed as rounding, N1 round 4), and :meth:`drawn_sag_bound` bounds the drawn
        chord's distance from the evaluated curve by the exact sag (raised by its rounding), that
        offset and the evaluator's rounding ``3 E`` (conservative: two intervals when in doubt).
        """
        if self.circle:
            span = abs(self.sweep)
            n = _count(self.edge_id, span, step)
            if allowance is not None and allowance < 2.0 * self.radius:
                largest = 4.0 * math.asin(math.sqrt(allowance / (2.0 * self.radius)))
                if not largest > 0.0:
                    raise _unsupported(self.edge_id, "the sag allowance is below the circle's float resolution")
                n = max(n, _count(self.edge_id, span, largest))
            forced = n == 1 and self.drawn_sag_bound(offset) > residual
            if forced:
                n = 2                                   # a curved edge is never one chord
            return _Plan(n, lambda: _Division(np.linspace(0.0, 1.0, n + 1)[1:-1].tolist(),
                                              lambda t: self.sweep * t, forced))
        amap = _AngleMap(self.edge_id, self.plane, (0.0, 0.0), math.pi / 2, 0.0, self.sweep)
        total = abs(amap.angle(self.sweep) - amap.angle(0.0))
        n = _count(self.edge_id, total, step)
        forced = n == 1 and self.drawn_sag_bound(offset) > residual
        if forced:
            n = 2                                       # a curved edge is never one chord
        start = amap.angle(0.0)

        def build():
            values = self.to_parameters(_angle_division(amap, 0.0, self.sweep, n))
            substituted = ()
            if forced and not all(0.0 < value < 1.0 for value in values):
                # N1 round 8: the inverse rounded the mandatory cut onto an end; the parametric
                # midpoint is exactly representable and distinct, and is verified like any cut
                # (:func:`_build`)
                values, substituted = [0.5], (0.5,)
            return _Division(values, lambda t: amap.angle(self.sweep * t) - start, forced, substituted)

        return _Plan(n, build)

    def drawn_sag_bound(self, offset) -> float:
        """An upper bound of the distance of the EVALUATED curve over [0, 1] from the chord
        DRAWN between the vertices (each within ``offset`` of its curve end): the exact sag of the
        analytic chord (raised by its rounding, :meth:`refine_sag`, and ``64 eps``), plus ``offset``
        (a point of the drawn chord is the analytic chord's point moved by at most that), plus
        ``3 E`` (the evaluator's rounding of the curve point and of both chord ends: the stored
        angle ``start + sweep t`` is rounded by up to ``2 eps theta*``)."""
        inflate = 1.0 + 64.0 * _EPS + 8.0 * _EPS * (1.0 if self.circle else self.aspect)
        exact = (2.0 * self.radius * math.sin(abs(self.sweep) / 4) ** 2 if self.circle else self.sag(0.0, 1.0))
        return exact * inflate + offset + 3.0 * self.point_rounding

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
        sn, cs = -math.sin(m), math.cos(m)
        t0, t1, t2 = (sn * a + cs * b for a, b in zip(self._lu, self._lv))
        tangent = math.sqrt(t0 * t0 + t1 * t1 + t2 * t2)
        if not tangent > 0.0:
            raise _unsupported(self.edge_id, "the curve's tangent vanishes")
        return 2.0 * math.sin(delta / 4) ** 2 * self.area / tangent

    def sag_exceeds(self, a, b, allowance, offsets) -> bool:
        """Whether the chord over [a, b] can leave the curve by more than ``allowance``: the
        exact sag :meth:`sag` (at the parametric midpoint; for a circle it is ``2 rho
        sin^2(delta / 4)``) raised by its rounding, plus the offset of a vertex at an end of the
        edge."""
        inflate = 1.0 + 8.0 * _EPS * (1.0 if self.circle else self.aspect)
        offset = max(offsets[0] if a == 0.0 else 0.0, offsets[1] if b == 1.0 else 0.0)
        return not self.sag(a, b) * inflate + offset <= allowance

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
        kept = [float(parameters[0])]
        pending = [(float(a), float(b)) for a, b in zip(parameters[:-1], parameters[1:])]
        pending.reverse()
        while pending:
            a, b = pending.pop()
            offset = max(offsets[0] if a == 0.0 else 0.0, offsets[1] if b == 1.0 else 0.0)
            if (self.circle and offset == 0.0) or not self.sag_exceeds(a, b, allowance, offsets):
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
        :data:`MAX_DISPLAY_ELLIPSE_ASPECT` (a projected frame that is a similarity to 8 eps
        relative, a circle seen along its axis, has aspect exactly 1: N1 round 5), and its closest
        approach to the axis (a lower bound) is at least both

        * the radius ratio of the conic's own scale: its definition points (vertices, ``c``,
          ``c + u``, ``c + v``) and ``max(1, |stored angle|) sigma_max(M')`` (the evaluator rounds
          ``start + sweep t`` by eps |theta|, moving the projected point by up to that times eps),
          not the support origin (N1 round 5: the documented scale), and
        * ``64 eps |c - o|``, the frame's own rounding: the centre's offset from the support's
          axis point ``o`` is a float subtraction, rounded by eps times its result.

        That floor is the plan's ``floor`` (:attr:`DisplayRoundingAllowance.axis_floors`)."""
        origin, x_axis, y_axis = _axis_frame(support)
        c3 = self.c - origin
        projected = np.array(((self.lu @ x_axis, self.lv @ x_axis), (self.lu @ y_axis, self.lv @ y_axis)))
        # the projection's absolute rounding (N1 round 9, the branch bound): u', v' from one
        # rotation (2 eps (|u'| + |v'|)) and each dot product with a unit frame vector (3 eps, its
        # unit length an eps more); c - o rounded by eps |c - o|, then projected; plus (N1 round 10)
        # the module frame's deviation from an exact orthonormal frame about the axis (_FRAME_ERROR)
        matrix_error = (8.0 * _EPS + _FRAME_ERROR) * (_norm(self.lu) + _norm(self.lv))
        offset_error = (8.0 * _EPS + _FRAME_ERROR) * _norm(c3)
        amap = _AngleMap(self.edge_id, projected, (c3 @ x_axis, c3 @ y_axis), 0.0, 0.0, self.sweep,
                         matrix_error, offset_error)
        stored = max(1.0, abs(self.start), abs(self.start + self.sweep))
        n0, n1 = math.hypot(*projected[:, 0]), math.hypot(*projected[:, 1])
        similar = (abs(n0 - n1) <= _CIRCLE_RTOL * max(n0, n1)
                   and abs(float(projected[:, 0] @ projected[:, 1])) <= _CIRCLE_RTOL * n0 * n1)
        aspect = 1.0 if similar else amap.aspect
        weight = aspect * stored
        if not _aspect_admitted(weight):
            raise DisplayDomainRefused(f"edge {_id_text(self.edge_id)}'s projection across a support axis has the aspect "
                                       f"{aspect!r}: aspect x max(1, |stored angle|) = {weight!r} is above "
                                       f"{MAX_DISPLAY_ELLIPSE_ASPECT:g}")
        floor = max(MIN_DISPLAY_RADIUS_RATIO * _scale([*scale_points, (stored * amap.sigma_max, 0.0, 0.0)]),
                    64.0 * _EPS * _norm(c3))
        if not amap.closest >= floor:
            raise DisplayDomainRefused(f"edge {_id_text(self.edge_id)} (closest approach, lower bound) is {amap.closest!r} from "
                                       f"a support axis, below its floor {floor!r} (the larger of "
                                       f"{MIN_DISPLAY_RADIUS_RATIO:g} x its scale and 64 eps x its centre's offset "
                                       f"{_norm(c3)!r} from the axis point)")
        start = amap.angle(0.0)
        span = amap.angle(self.sweep) - start
        n = _count(self.edge_id, span, step)
        # display_rounding_allowance: the projected inverse and forward map stretch an angle's
        # rounding by aspect(M') x max(1, |stored angle|); every evaluated point (and its
        # projection relative to the axis point) is rounded by E, an azimuth error E / r
        sample = self.point_rounding + 8.0 * _EPS * _norm(origin)
        rounding = 32.0 * _EPS * aspect * stored + 8.0 * sample / amap.closest

        def build():
            values = self.to_parameters(_angle_division(amap, 0.0, self.sweep, n))
            return _Division(values, lambda t: amap.angle(self.sweep * t) - start)

        return _Plan(n, build, rounding, floor)


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


def _union(edge_id, divisions, headrooms, verify):
    """Sorted union of the divisions' parameters in (0, 1) with the ends: merge, then verify with
    exact forward maps; restore on failure (N1 round 7).

    *Merge* (one sort, one pass; N1 round 6: a fixed 64-ulp gap had kept a sliver interval
    1.04e-14 long).  Two adjacent cuts ``k`` (kept so far) and ``v`` of different divisions are
    merged when the cut to be dropped lies, in its OWN division's angle, within
    ``max(headroom_D, 64 eps max(1, |angle|))`` of the cut kept.  The cut of the stricter division
    (the smaller headroom) is kept, the earlier cut on ties; a kept cut that already absorbed
    others is replaced only by a cut within every absorbed cut's own threshold, and no two cuts of
    one division are merged.  The ends and every cut of a ``protected`` division (the
    minimum-two-interval split) are mandatory: never dropped, never replaced.  Exact duplicates
    merge.  The rule is a proposal only: each dropped cut is checked alone, so two moved ends of
    one interval can add (round 7).

    *Verify*.  Every final interval with a dropped cut at either end is checked by ``verify(a,
    b)`` (:func:`_build`: every division's angle span over it from its own forward map, and a
    conic's exact sag, each against the published bound).  An interval that fails gets the dropped
    cuts of both its ends back (the original cuts, never a new one; a merge group is a contiguous
    run of the sorted cuts, so this restores every cut that lay inside it) and its pieces are
    verified again.  An interval without a dropped cut is one of the no-merge union, which meets
    every bound by construction, so the loop ends, at worst at "no merge at all"."""
    values = [np.clip(np.asarray(d.values, dtype=float), 0.0, 1.0) for d in divisions]
    owners = [np.full(len(v), i) for i, v in enumerate(values)]
    total = sum(len(v) for v in values)
    if not total:
        return np.array([0.0, 1.0])
    values, owners = np.concatenate(values), np.concatenate(owners)
    order = np.lexsort((owners, values))
    end = -1                                    # owner of the ends: strictest, never dropped
    fixed = {end}
    # the minimum-two-interval split is mandatory: the result is never one chord (verified below)
    two = any(d.protected for d in divisions)

    def room(owner):
        return -1.0 if owner in fixed else headrooms[owner]

    def close(owner, a, b):
        if owner in fixed:
            return False
        angle = divisions[owner].angle
        target = angle(b)
        return abs(angle(a) - target) <= max(headrooms[owner], 64.0 * _EPS * max(1.0, abs(target)))

    # each kept cut with the cuts merged into it (value, owner): a cut replacing it must be close
    # to every one of them in its own division's angle
    kept = [(0.0, end, [(0.0, end)])]
    for value, owner in zip(values[order].tolist(), owners[order].tolist()):
        cut, cut_owner, group = kept[-1]
        if value == cut and owner == cut_owner:
            continue
        if all(o != owner for _v, o in group):
            if room(owner) >= room(cut_owner):        # the new cut's division is looser (or a tie)
                if close(owner, cut, value):
                    group.append((value, owner))
                    continue
            elif all(close(o, value, v) for v, o in group):
                group.append((value, owner))
                kept[-1] = (value, owner, group)
                continue
        kept.append((value, owner, [(value, owner)]))
        if len(kept) > _MAX_INTERVALS:
            raise _unsupported(edge_id, f"needs more than {_MAX_INTERVALS} intervals")
    last = [(1.0, end)]
    cut, cut_owner, group = kept[-1]
    if cut_owner not in fixed and (cut >= 1.0 or all(close(o, 1.0, v) for v, o in group)):
        kept.pop()
        last += group
    kept.append((1.0, end, last))
    # nodes [value, dropped values, alternatives], sorted (a merge group is a contiguous run of the
    # sorted cuts): a mandatory cut at exactly 0 or 1 is that end, an exact duplicate one node.
    # ``alternatives``: the group's other ORIGINAL cuts that may stand for it instead (none for an
    # end, which is never dropped)
    nodes = []
    for cut, owner, group in kept:
        dropped = [v for v, _o in group if v != cut]
        if nodes and cut <= nodes[-1][0]:
            nodes[-1][1].extend(x for x in dropped + [cut] if x != nodes[-1][0])
            nodes[-1][2] = []
            continue
        nodes.append([cut, dropped, [] if owner in fixed else sorted(set(dropped))])
    nodes[0][0], nodes[-1][0] = 0.0, 1.0
    nodes[0][2], nodes[-1][2] = [], []
    while True:
        bad = {0, 1} if two and len(nodes) == 2 and (nodes[0][1] or nodes[1][1]) else set()
        for i in range(len(nodes) - 1):
            if (nodes[i][1] or nodes[i + 1][1]) and not verify(nodes[i][0], nodes[i + 1][0]):
                bad.update((i, i + 1))
        if not bad:
            break
        restored = []
        for i, (cut, dropped, alternatives) in enumerate(nodes):
            if i not in bad or not dropped:
                restored.append([cut, dropped, alternatives])
                continue
            members = sorted({cut, *dropped})
            if alternatives:
                # first another original cut of the group stands for it (verified again) ...
                other = alternatives[0]
                restored.append([other, [x for x in members if x != other], alternatives[1:]])
                continue
            # ... then every cut of the group is restored
            if i == 0:
                members = [0.0] + [x for x in members if x > 0.0]
            if i == len(nodes) - 1:
                members = [x for x in members if x < 1.0] + [1.0]
            restored.extend([x, [], []] for x in members)
        nodes = []
        for cut, dropped, alternatives in restored:
            if nodes and cut == nodes[-1][0]:
                nodes[-1][1].extend(x for x in dropped if x != nodes[-1][0])
                continue
            nodes.append([cut, dropped, alternatives])
    result = np.asarray([node[0] for node in nodes])
    if len(result) - 1 > _MAX_INTERVALS:
        raise _unsupported(edge_id, f"needs more than {_MAX_INTERVALS} intervals")
    return result


# ---------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------
def edge_display_samples(model, edge_id: int, tol: DisplayTolerance) -> EdgeSamples:
    """Sample one edge for display: the same result for every face using the edge.

    The samples depend on the edge and on the supports of the faces using it
    (``model.faces_using_edge``), which must be planes, cylinders or cones without an explicit
    face parameterization (else ``SUPPORT_UNSUPPORTED``).  The edge must be a line
    (``Straight``, ``LineRange`` with exact domain [0, 1]: its whole carrier), a circle (``Arc``,
    circular ``EllipticArc``) or an ellipse (``EllipticArc``; else, and for a ``LineRange`` drawn
    over part of its carrier, ``EDGE_UNSUPPORTED``).  The parameters are the sorted union of

    * the tangent division: none for a line; uniform in theta for a circle; uniform in tangent
      angle for an ellipse (exact closed-form inverse, in the arc's local angle),
      ``ceil(turning / angle)`` steps; a circle's count is raised for the sag allowance
      (:meth:`_Conic.tangent_division`);
    * for each incident cylinder or cone, the division of the azimuth about its axis into
      ``ceil(span / angle)`` equal steps, solved in closed form (none for a line whose ends are
      collinear with the axis point on one side: a ruling or a generator).

    Every count is computed first, and their sum (``1 + sum(n - 1)``, the most intervals the
    union can have) is checked against the 4096 cap before any division is built.

    then every ellipse interval whose exact sag (:meth:`_Conic.sag`) exceeds the allowance is
    bisected in theta until none does, and so is every end interval whose sag plus its vertex's
    offset from the curve does (an offset above the allowance is refused).  Every interval of
    the no-merge union lies inside one interval of each division, and a merged interval is kept
    only when verified against every bound (:func:`_union`: merge, then verify with exact forward
    maps; restore on failure), so the tangent turns, and every incident cylinder's or
    cone's azimuth changes, by at most ``tol.angle`` over it, and its chord leaves the curve by at
    most the sag allowance (:class:`DisplayTolerance`), each to the edge's documented rounding
    allowance (:func:`display_rounding_allowance`), which is at most 2 % of the angle and of the
    sag allowance (a finer request is refused: ``OUT_OF_DOMAIN`` for the angle,
    ``EDGE_UNSUPPORTED`` for the sag).  A curved edge whose single chord could leave it by more
    than ``model.tolerance.surface_residual`` (decided on the DRAWN chord: its exact sag, its
    measured vertex offsets and the evaluator's rounding) gets at least two intervals.  The first and last points are the edge's vertex positions.
    Inputs are checked against the declared numeric domain first (``OUT_OF_DOMAIN``).  Typed
    refusal (``EDGE_UNSUPPORTED``) for a degenerate curve, an azimuth that is not monotone along
    the edge, an unresolvable sag or more than 4096 intervals.  An edge id not in the model is a
    plain :class:`GeometryError`; no raw arithmetic error escapes a division (typed
    ``EDGE_UNSUPPORTED`` at worst).  Independent of the caller's NumPy error state (N2 round 3):
    the module's arithmetic runs under ``np.errstate(all="ignore")``, model calls under NumPy's
    default state, and the explicit guards decide.
    """
    with np.errstate(all="ignore"):
        return _edge_display_samples(model, edge_id, tol)


def _edge_display_samples(model, edge_id, tol) -> EdgeSamples:
    setup = _setup(model, edge_id, tol)

    def run():
        analysis = _analyse(model, setup, tol)
        allowance = analysis.allowance
        if not allowance.angle <= _ALLOWANCE_SHARE * tol.angle:
            raise DisplayDomainRefused(f"edge {_id_text(setup.edge_id)}: its rounding allowance {allowance.angle!r} rad is "
                                       f"above 2 % of the display angle {tol.angle!r} (the angle is finer than "
                                       f"the edge resolves at its scale and distance from the axis)")
        if allowance.sag is not None and not allowance.sag <= _ALLOWANCE_SHARE * analysis.sag:
            raise _unsupported(setup.edge_id, f"its sag rounding allowance {allowance.sag!r} is above 2 % of the "
                                              f"sag allowance {analysis.sag!r} (unresolvable at its scale)")
        return _build(setup, analysis)

    parameters = _guarded(setup.edge_id, run)
    return _edge_result(model, setup.edge_id, parameters, setup.start, setup.end)


def display_rounding_allowance(model, edge_id: int, tol: DisplayTolerance) -> DisplayRoundingAllowance:
    """The documented rounding allowance of one edge at ``tol``: inside the declared domain,
    every interval :func:`edge_display_samples` emits turns (tangent, and every incident
    cylinder's or cone's azimuth) by at most ``tol.angle + allowance.angle`` and its drawn chord
    lies within ``sag + allowance.sag`` of the curve.

    With ``eps`` the float epsilon, ``theta* = max(1, |start|, |start + sweep|)`` (a conic's
    stored angles; 1 for a line), ``E`` the rounding of an evaluated point (a conic:
    ``8 eps (|c| + |u| + |v| + sigma_max theta*)``, the evaluator rounding ``start + sweep t`` by
    ``2 eps theta*``; a line: ``8 eps (|a| + |b|)``) and, per incident cylinder or cone, ``E_o = E +
    8 eps |o|`` (the projection relative to its axis point ``o``) and ``r`` the smallest distance
    from its axis that the division uses (a conic: the closest-approach lower bound; a line: its
    ends' and interior closest approach):

    * ``angle = 64 eps x tol.angle + max(T, Z_support...)``; the tangent term ``T = 32 eps W``,
      ``W = aspect x theta*`` (1 for a circle's aspect; ``8 eps`` for a line); the azimuth term of a
      conic ``Z = 32 eps aspect(M') theta* + 8 E_o / r`` (``M'`` its projected frame), of a line
      ``2 G / r (+ 4 E_o / r with interior cuts)`` (:func:`_line_azimuth_division`; ``G`` includes
      ``2 X``, ``X = eps max |coordinate|`` of the evaluated ends, N1 round 4);
    * ``sag = 64 eps x sag + 3 E`` for a conic (both chord ends and the measured point move by
      ``E``), ``64 eps x sag + 8 eps max(1, |b - a|) + X (+ E with interior cuts)`` for a line, plus
      any vertex offset treated as 0 at the coordinates' resolution.

    Same refusals as :func:`edge_display_samples` before its divisions are built (except the 2 %
    rule itself), and likewise independent of the caller's NumPy error state.
    """
    with np.errstate(all="ignore"):
        setup = _setup(model, edge_id, tol)
        return _guarded(setup.edge_id, lambda: _analyse(model, setup, tol).allowance)


def _guarded(edge_id, call):
    try:
        return call()
    except DisplayTrimRefused:
        raise
    except (ArithmeticError, ValueError) as error:      # no raw arithmetic error escapes
        # fixed text (owner decision N2 r5 (1)); the error itself stays the refusal's __cause__
        raise _unsupported(edge_id, "an arithmetic error in the closed-form division") from error


@dataclass(frozen=True)
class _Setup:
    edge_id: int
    supports: tuple
    start: np.ndarray
    end: np.ndarray
    line: object
    conic: object
    definition: list


@dataclass(frozen=True)
class _Analysis:
    plans: list               # the incident cylinders' and cones' azimuth plans
    tangent: object           # a conic's tangent plan (None for a line)
    sag: object               # the sag allowance (None without one)
    terms: tuple              # the rounding terms: the tangent's, then each plan's (radians)
    offsets: tuple            # vertex offsets from the curve's ends (0 at the coordinates' resolution)
    allowance: DisplayRoundingAllowance
    step: float = 0.0         # tol.angle


def _setup(model, edge_id, tol) -> _Setup:
    """Every input check against the declared domain, before any division."""
    from .model import GeometryModel
    # owner decision N2 r5 (1): the argument's own type, never ``isinstance`` (which looks up a
    # foreign ``__class__``) and never a caller's name in the message
    if not issubclass(type(model), GeometryModel):
        raise GeometryError("model must be a GeometryModel")
    if not issubclass(type(tol), DisplayTolerance):
        raise GeometryError("tol must be a DisplayTolerance")
    # an int (not bool) or a NumPy integer scalar of an exact type, |id| < 2**63
    if not _exact_type_in(edge_id, _ID_TYPES):
        raise GeometryError("edge id must be an int or a NumPy integer scalar")
    edge_id = int(edge_id)
    if not -_ID_BOUND < edge_id < _ID_BOUND:
        raise GeometryError("edge id is out of range (|id| < 2**63)")
    if edge_id not in model.edges:
        raise GeometryError(f"edge {_id_text(edge_id)} is not in the model")
    edge = model.edges[edge_id]
    supports = _model_call(_incident_supports, model, edge_id)
    curve = edge.curve
    start = np.asarray(_model_call(model.vertex_position, edge.start), dtype=float)
    end = np.asarray(_model_call(model.vertex_position, edge.end), dtype=float)

    _check_points(f"edge {_id_text(edge_id)}'s vertices", np.vstack((start, end)))
    conic = None
    line = None
    if isinstance(curve, (Straight, LineRange)):
        if isinstance(curve, LineRange):
            _check_points(f"edge {_id_text(edge_id)}'s line carrier", np.vstack((curve.a, curve.b)))
        # the line's geometry is its evaluated ends (the exact evaluator), never the carrier's
        # float fields: a retained range can be far narrower than its carrier
        line = tuple(_evaluate(model, edge_id, [0.0, 1.0]))
        _check_points(f"edge {_id_text(edge_id)}'s line ends", np.vstack(line))
        if isinstance(curve, LineRange):
            # owner decision (N1 round 4, recommended option): a LineRange is supported only when
            # its carrier IS the edge's segment: exact domain [0, 1] and both carrier endpoints the
            # evaluated ends within the model's length tolerance.  A range drawn over part of its
            # carrier is refused: its float ends need not lie on the exact carrier to the bounds.
            length = float(model.tolerance.length)
            if (tuple(curve.domain) != (Fraction(0), Fraction(1))
                    or _norm(np.asarray(curve.a, dtype=float) - line[0]) > length
                    or _norm(np.asarray(curve.b, dtype=float) - line[1]) > length):
                raise _unsupported(edge_id, "a line range drawn over part of its carrier is not supported "
                                            "(only a LineRange whose exact domain is [0, 1])")
            # N1 round 5 (recommended option): the carrier endpoints must also be the edge's
            # TOPOLOGY vertices within the model's length tolerance, not only its evaluated ends
            if (_norm(np.asarray(curve.a, dtype=float) - start) > length
                    or _norm(np.asarray(curve.b, dtype=float) - end) > length):
                raise _unsupported(edge_id, "the line range's carrier endpoints are not the edge's vertices "
                                            "within the model's length tolerance")
    elif isinstance(curve, EllipticArc):
        centre = np.asarray(curve.center, dtype=float)
        _check_points(f"edge {_id_text(edge_id)}'s ellipse", np.vstack((centre, centre + curve.u_vector,
                                                              centre + curve.v_vector)))
        _check_angle(f"edge {_id_text(edge_id)}'s ellipse", curve.start_angle, curve.start_angle + curve.sweep_angle)
        _check_sweep(f"edge {_id_text(edge_id)}'s ellipse", float(curve.sweep_angle))
    elif isinstance(curve, Arc):
        _check_points(f"edge {_id_text(edge_id)}'s arc via vertex", _model_call(model.vertex_position, curve.via_vertex))
    else:
        raise _unsupported(edge_id, "the curve is not a line, circle or ellipse")
    for support in supports:
        _check_support_domain(support)
    if isinstance(curve, Arc):
        try:
            frame = _model_call(model.arc_frame, edge_id)
        except GeometryError as error:
            raise _unsupported(edge_id, "the model has no arc frame for it") from error
        _check_points(f"edge {_id_text(edge_id)}'s arc centre", frame.center)
        _check_sweep(f"edge {_id_text(edge_id)}'s arc", float(frame.sweep))
        conic = _Conic(edge_id, frame.center, frame.radius * frame.e1, frame.radius * frame.e2, 0.0, frame.sweep)
    elif isinstance(curve, EllipticArc):
        conic = _Conic(edge_id, curve.center, curve.u_vector, curve.v_vector, curve.start_angle, curve.sweep_angle)
    if conic is not None and not conic.circle:
        # the evaluator rounds the absolute angle start + sweep t (by eps |theta|), which moves the
        # tangent by up to aspect x that: the bound holds for aspect x max(1, |stored angle|)
        weight = conic.aspect * max(1.0, abs(conic.start), abs(conic.start + conic.sweep))
        if not _aspect_admitted(weight):
            raise DisplayDomainRefused(f"edge {_id_text(edge_id)}'s ellipse has the aspect {conic.aspect!r} at stored angles up "
                                       f"to {max(abs(conic.start), abs(conic.start + conic.sweep))!r}: "
                                       f"aspect x max(1, |angle|) = {weight!r} is above {MAX_DISPLAY_ELLIPSE_ASPECT:g}")
    definition = [start, end] if conic is None else [start, end, conic.c, conic.c + conic.u, conic.c + conic.v]
    if line is not None:
        definition += list(line)
    for support in supports:
        if isinstance(support, Cylinder):
            _check_radius("a cylinder radius", support.radius, support.origin, *definition)
    return _Setup(edge_id, supports, start, end, line, conic, definition)


def _analyse(model, setup, tol) -> _Analysis:
    """Every division's count (each checked against the cap, then their sum) and the edge's
    rounding allowance; nothing is built."""
    edge_id, line, conic = setup.edge_id, setup.line, setup.conic
    step = tol.angle
    sag = _sag_allowance(setup.supports, tol)
    # the drawn ends are the vertices: their offsets from the curve's ends enter the sag bound
    curve_ends = line if line is not None else _evaluate(model, edge_id, [0.0, 1.0])
    raw = (float(np.linalg.norm(setup.start - curve_ends[0])), float(np.linalg.norm(setup.end - curve_ends[1])))
    # an offset at the coordinates' resolution (4 eps x max(1, |vertex coordinate|), the floor any
    # sag allowance must exceed) is rounding: the vertex is the curve's end to that resolution
    resolution = 4.0 * _EPS * max(1.0, float(np.abs(np.vstack((setup.start, setup.end))).max()))
    offsets = tuple(0.0 if offset <= resolution else offset for offset in raw)
    hidden = max([0.0] + [offset for offset in raw if offset <= resolution])
    if sag is not None and max(offsets) > sag:
        raise _unsupported(edge_id, f"a vertex lies farther from its curve ({max(offsets)!r}) than the sag "
                                    f"allowance {sag!r}")
    round_supports = [s for s in setup.supports if isinstance(s, (Cylinder, Cone))]
    if line is not None:
        plans = [_line_azimuth_division(edge_id, s, line[0], line[1], step, setup.definition,
                                        float(model.tolerance.length)) for s in round_supports]
        tangent = None
        tangent_rounding = 8.0 * _EPS
        sample = 8.0 * _EPS * (_norm(line[0]) + _norm(line[1]))
        sag_rounding = (8.0 * _EPS * max(1.0, _norm(line[1] - line[0])) + _line_end_rounding(*line)
                        + (sample if any(plan.count > 1 for plan in plans) else 0.0))
    else:
        if sag is not None and sag <= resolution:
            raise _unsupported(edge_id, f"the sag allowance {sag!r} is below its coordinates' resolution")
        # the minimum-two rule measures the DRAWN chord: the vertex offsets as measured (raw)
        tangent = conic.tangent_division(step, sag, model.tolerance.surface_residual, max(raw))
        plans = [conic.azimuth_division(s, step, setup.definition) for s in round_supports]
        tangent_rounding = 32.0 * _EPS * (1.0 if conic.circle else conic.aspect) * conic.theta_star
        sag_rounding = 3.0 * conic.point_rounding
    counts = [plan.count for plan in plans] + ([] if tangent is None else [tangent.count])
    total = 1 + sum(max(1, count) - 1 for count in counts)
    if total > _MAX_INTERVALS:
        raise _unsupported(edge_id, f"its divisions need up to {_id_text(total)} intervals together, more than {_MAX_INTERVALS}")
    scale = _scale([*setup.definition, *(s.origin for s in round_supports)])
    allowance = DisplayRoundingAllowance(
        angle=64.0 * _EPS * step + max([tangent_rounding] + [plan.rounding for plan in plans]),
        sag=None if sag is None else 64.0 * _EPS * sag + sag_rounding + hidden,
        axis_floor=MIN_DISPLAY_RADIUS_RATIO * scale,
        axis_floors=tuple(plan.floor for plan in plans))
    return _Analysis(plans, tangent, sag, tuple([tangent_rounding] + [plan.rounding for plan in plans]), offsets,
                     allowance, step)


def _build(setup, analysis):
    """The divisions (their counts already checked), their union (merge, then verify) and the
    sag refinement."""
    # each division's headroom: the share of the published allowance max(T, Z...) its own
    # rounding term leaves unused (the merge proposes moves of its interval ends within it)
    terms = analysis.terms if setup.line is None else analysis.terms[1:]
    top = max(analysis.terms)
    headrooms = [top - term for term in terms]
    conic = setup.conic
    divisions = ([plan.build() for plan in analysis.plans] if setup.line is not None
                 else [analysis.tangent.build(), *(plan.build() for plan in analysis.plans)])
    # the published angle bound; a division's computed span over an interval is accepted only when
    # it meets that bound with the division's own rounding term (the documented error of its
    # computed angles) added: the true span then meets it too
    limit = analysis.step + analysis.allowance.angle
    checks = [(division.angle, term) for division, term in zip(divisions, terms) if division.angle is not None]

    def verify(a, b):
        for angle, term in checks:
            if not abs(angle(b) - angle(a)) + term <= limit:
                return False
        # the sag against the published sag + allowance.sag less the evaluator's 3 E (and the
        # hidden offsets) it carries for the drawn points: sag (1 + 64 eps); the exact sag is
        # raised by its own rounding inside sag_exceeds
        # (a circle only, without the vertex offsets: an ellipse's every final interval, and a
        # circle's end interval with a vertex offset, is checked, and bisected, by refine_sag on
        # its exact sag; a circle's other intervals are not, its uniform count meeting the sag)
        return (conic is None or analysis.sag is None or not conic.circle
                or not conic.sag_exceeds(a, b, analysis.sag * (1.0 + 64.0 * _EPS), (0.0, 0.0)))

    parameters = _union(setup.edge_id, divisions, headrooms, verify)
    # a substituted cut (a forced split's midpoint) is verified like a merged one; it is never
    # moved, so a failure is refused (fail closed)
    cuts = parameters.tolist()
    for division in divisions:
        for value in division.substituted:
            if value in cuts:
                i = cuts.index(value)
                if not (verify(cuts[i - 1], value) and verify(value, cuts[i + 1])):
                    raise _unsupported(setup.edge_id, "the forced split's midpoint does not meet the bounds")
    if conic is not None and analysis.sag is not None:
        parameters = conic.refine_sag(parameters, analysis.sag, analysis.offsets)
    return parameters


def _evaluate(model, edge_id, parameters) -> np.ndarray:
    try:
        return np.array(_model_call(model.evaluate_edge_many, edge_id, parameters), dtype=float)
    except GeometryError as error:
        raise _unsupported(edge_id, "the model cannot evaluate it") from error


def _edge_result(model, edge_id, parameters, start, end) -> EdgeSamples:
    points = _evaluate(model, edge_id, parameters)
    _check_points(f"edge {_id_text(edge_id)}'s emitted points", points)
    if np.any(np.all(points[1:] == points[:-1], axis=1)):     # before the ends become the vertices
        raise _unsupported(edge_id, "distinct parameters evaluate to one point (lost precision)")
    points[0], points[-1] = start, end
    if np.any(np.all(points[1:] == points[:-1], axis=1)):
        raise _unsupported(edge_id, "distinct parameters evaluate to one point (lost precision)")
    return EdgeSamples(edge_id, _frozen(parameters), _frozen(points))


# ---------------------------------------------------------------------------------------------
# face lift (N3)
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class FaceDisplayMesh:
    """A certified display mesh of one face (:func:`face_display_mesh`).

    ``points`` (n, 3) float64: first the ring vertices, which are the N1 :class:`EdgeSamples`
    points bit for bit (outer loop, then each hole, each coedge's samples in loop direction without
    its last point), then the cut points, each ON the 3D N1 chord it subdivides (a T-vertex against
    the neighbour, owner decision Q8).  ``triangles`` (m, 3) int64, counter-clockwise about the face
    normal ``sense x`` the support normal.  ``chart`` (n, 2): the 2D points of the chart mesh (plane
    frame coordinates).  ``ring_vertex_count``: ``points[:k]`` are ring vertices.  ``source`` (n, 2)
    int64: ``(edge id, interval index)`` of the N1 chord each point lies on (a ring vertex: the chord
    starting at it in loop direction).  ``sense``: +1/-1, the sign of the outer ring's exact chart
    area.  ``face_tolerance``: T_face.  ``deviation_bound``: the certified largest paired deviation
    of the mesh from its support, ``<= face_tolerance``.  ``trim_sag_bound``: the certified largest
    distance of the drawn boundary chords from the analytic trim (reported always, enforced only
    when ``tol.sag`` is given).  ``degenerate_count``: triangles exempt from the orientation test as
    rounding-thin.
    """

    face_id: int
    points: np.ndarray
    triangles: np.ndarray
    chart: np.ndarray
    ring_vertex_count: int
    source: np.ndarray
    sense: int
    face_tolerance: float
    deviation_bound: float
    trim_sag_bound: float
    degenerate_count: int


def face_display_mesh(model, face_id: int, tol: DisplayTolerance) -> FaceDisplayMesh:
    """A certified triangle mesh of one face for display (N3; N3a: faces on a ``Plane``).

    The face's rings are the N1 samples of its coedges (:func:`edge_display_samples`); their chart
    polygon is triangulated exactly (N2) and lifted by index.  The result is returned only when
    every certificate holds: the chart map is injective, every mesh edge is shared by two oppositely
    oriented triangles or lies on one ring chord, the boundary chains exactly to every ring, every
    lifted triangle (except rounding-thin ones, counted) is positively oriented about the face
    normal, and the paired deviation from the support is at most T_face (``tol.sag`` if given, else
    the model's ``surface_residual`` on a plane); otherwise a typed :class:`DisplayTrimRefused`,
    never a partial mesh.  Faces on a cylinder or cone, with an explicit parameterization, or on
    any other support: ``SUPPORT_UNSUPPORTED``.  See ``docs/DISPLAY_TESSELLATION.md``.
    """
    from ._display_lift import _lift
    return _lift(model, face_id, tol)
