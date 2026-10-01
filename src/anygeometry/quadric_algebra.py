"""Exact algebra for rulings of a Cylinder or Cone against any quadric.

A ruled support is ``S(t, s) = P(t) + s*D(t)`` with the angle ``t`` about its
axis. Against a quadric ``x.M.x + 2*l.x + c = 0`` each ruling meets the surface
where ``A(t)*s**2 + B(t)*s + C(t) = 0`` with trigonometric coefficients, so the
intersection is ``s = (-B +- sqrt(B*B - 4*A*C)) / (2*A)`` over ``t``.

Every event of that branch (folds, double contacts, poles, finite-patch
boundaries, other surfaces) is a root of a trigonometric polynomial. Those are
homogenized in ``x = tan(t/2)`` as integer polynomials, built exactly from the
stored doubles, and isolated with rational Sturm counts. The exact system is
built in the frame of the ruled support's own origin from rational arithmetic;
a Cone is reduced about its exact rational apex, which halves the degree of
every elimination without any rounding of a distant apex.

A :class:`Plan` is the immutable-key, lazily filled cache of everything that
depends on a support *pair* but not on a patch, so the facets of a shell share
one exact solve.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from functools import lru_cache
import math
from math import gcd

import numpy as np

from .analytic_roots import (_derivative, _exact_quotient, _integer_gcd, _integer_row, _integer_value, _sturm, _variations,
                             isolate_real_roots)
from .errors import GeometryError
from .extrusions import BezierDirectrix, EllipseDirectrix
from .surfaces import Cone, Cylinder, ExtrudedSurface, Plane

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Support descriptors (hashable cache keys; patch ranges are deliberately absent)


def _vec(value, name):
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise GeometryError(f"{name} must be a finite 3-vector") from error
    if len(result) != 3 or not all(math.isfinite(item) for item in result):
        raise GeometryError(f"{name} must be a finite 3-vector")
    return result


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _fraction_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _fraction_vec(vector):
    return tuple(Fraction(item) for item in vector)


@dataclass(frozen=True, slots=True)
class RuledSupport:
    """A Cylinder (``slope == 0``) or Cone ruled by lines at angle ``t`` about ``axis``.

    The ring ``origin + radius*u(t)`` lies in the plane through ``origin``
    perpendicular to ``axis``; ``u(t) = cos(t)*radial_direction + sin(t)*(axis x
    radial_direction)``. Rulings are ``axis + slope*u(t)``. The axial coordinate
    of a point is therefore the ruling parameter ``s`` itself.
    """
    origin: tuple
    axis: tuple
    radial_direction: tuple
    radius: float
    slope: float = 0.0

    def __post_init__(self):
        object.__setattr__(self, "origin", _vec(self.origin, "support origin"))
        object.__setattr__(self, "axis", _vec(self.axis, "support axis"))
        object.__setattr__(self, "radial_direction", _vec(self.radial_direction, "support radial direction"))
        radius, slope = float(self.radius), float(self.slope)
        if not math.isfinite(radius) or not math.isfinite(slope) or radius < 0:
            raise GeometryError("a ruled support needs a finite non-negative ring radius and finite slope")
        if radius == 0 and slope == 0:
            raise GeometryError("a ruled support needs a ring radius or a slope")
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "slope", slope)

    @property
    def circumferential_direction(self):
        return _cross(self.axis, self.radial_direction)

    @property
    def is_cone(self):
        return self.slope != 0.0

    @classmethod
    def from_surface(cls, surface):
        if isinstance(surface, Cylinder):
            return cls(tuple(surface.origin), tuple(surface.axis), tuple(surface.radial_direction),
                       float(surface.radius), 0.0)
        if isinstance(surface, Cone):
            height = float(surface.height)
            slope = (float(surface.radius_end) - float(surface.radius_start)) / height
            if slope == 0.0:
                raise GeometryError("a cone with equal radii is a Cylinder")
            return cls(tuple(surface.origin), tuple(surface.axis), tuple(surface.radial_direction),
                       float(surface.radius_start), slope)
        raise GeometryError("ruled supports are Cylinder or Cone")

    # -- float rows relative to ``origin``: P(t) = P0 + P1 cos t + P2 sin t, D likewise
    def float_rows(self):
        e1, e2 = self.radial_direction, self.circumferential_direction
        zero = (0.0, 0.0, 0.0)
        p = (zero, tuple(self.radius * c for c in e1), tuple(self.radius * c for c in e2))
        d = (self.axis, tuple(self.slope * c for c in e1), tuple(self.slope * c for c in e2))
        return p, d

    # -- exact rows (Fractions), reduced about the apex for a cone
    def exact_rows(self):
        e1, e2 = self.radial_direction, self.circumferential_direction
        zero = (Fraction(0),) * 3
        slope = Fraction(self.slope)
        d = (_fraction_vec(self.axis), tuple(slope * Fraction(c) for c in e1), tuple(slope * Fraction(c) for c in e2))
        if self.is_cone:
            reach = Fraction(self.radius) / slope           # ring station minus apex station
            apex = tuple(-reach * Fraction(c) for c in self.axis)
            return (apex, zero, zero), d, reach
        radius = Fraction(self.radius)
        p = (zero, tuple(radius * Fraction(c) for c in e1), tuple(radius * Fraction(c) for c in e2))
        return p, d, Fraction(0)

    def exact_station(self, axial):
        """Exact ruling parameter of the axial coordinate ``axial`` (cone: measured from the apex)."""
        _p, _d, reach = self.exact_rows()
        return Fraction(axial) + reach

    def angle_of(self, point):
        """Azimuth of ``point`` about the axis, in ``(-pi, pi]``."""
        rel = np.asarray(point, dtype=float) - self.origin
        axial = rel @ self.axis
        radial = rel - axial[..., None] * self.axis if rel.ndim > 1 else rel - axial * np.asarray(self.axis)
        return np.arctan2(radial @ np.asarray(self.circumferential_direction), radial @ np.asarray(self.radial_direction))

    def axial(self, point):
        return (np.asarray(point, dtype=float) - self.origin) @ np.asarray(self.axis)

    def quadric(self):
        """The ruled surface itself as a world-space quadric support."""
        return QuadricSupport("cone" if self.is_cone else "cylinder", self.origin, self.axis, self.radius, self.slope)


def _inverse3(rows):
    """Exact inverse of a 3x3 matrix of Fractions (rows)."""
    (a, b, c), (d, e, f), (g, h, i) = rows
    cofactors = ((e * i - f * h, c * h - b * i, b * f - c * e),
                 (f * g - d * i, a * i - c * g, c * d - a * f),
                 (d * h - e * g, b * g - a * h, a * e - b * d))
    determinant = a * cofactors[0][0] + b * cofactors[1][0] + c * cofactors[2][0]
    if determinant == 0:
        raise GeometryError("the base ellipse and the ruling direction must be independent")
    return tuple(tuple(value / determinant for value in row) for row in cofactors)


@dataclass(frozen=True, slots=True)
class EllipticRuledSupport:
    """An elliptic cylinder: rulings of direction ``axis`` through the base ellipse ``origin + u cos t + v sin t``.

    The counterpart of :class:`RuledSupport` (``slope == 0``) for an oblique or elliptic base. The base lies in
    the plane through ``origin`` with normal ``u x v``; the ruling parameter ``s`` of a point is its height
    above that plane measured along ``axis`` (a unit vector), and ``t`` is the ellipse parameter of the
    point's projection along ``axis`` into that plane.
    """
    origin: tuple
    u_vector: tuple
    v_vector: tuple
    axis: tuple

    slope = 0.0
    is_cone = False

    def __post_init__(self):
        origin = _vec(self.origin, "support origin")
        u, v = _vec(self.u_vector, "ellipse u vector"), _vec(self.v_vector, "ellipse v vector")
        axis = _vec(self.axis, "support axis")
        length = math.sqrt(sum(c * c for c in axis))
        if length <= 0.0:
            raise GeometryError("the ruling direction must be non-zero")
        if abs(length - 1.0) > 4 * float(np.finfo(float).eps):              # a unit axis stays as it is
            axis = tuple(c / length for c in axis)
        normal = _cross(u, v)
        scale = math.sqrt(sum(c * c for c in normal))
        if scale <= 0.0 or abs(sum(n * a for n, a in zip(normal, axis))) <= 1e-12 * scale:
            raise GeometryError("the ruling direction must leave the base plane of the ellipse")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "u_vector", u)
        object.__setattr__(self, "v_vector", v)
        object.__setattr__(self, "axis", axis)

    @property
    def radius(self):
        return max(math.sqrt(sum(c * c for c in self.u_vector)), math.sqrt(sum(c * c for c in self.v_vector)))

    @property
    def normal(self):
        value = np.cross(self.u_vector, self.v_vector)
        return value / np.linalg.norm(value)

    @classmethod
    def from_surface(cls, surface):
        directrix = getattr(surface, "directrix", None)
        if isinstance(surface, ExtrudedSurface) and isinstance(directrix, EllipseDirectrix):
            return cls(directrix.center, directrix.u_vector, directrix.v_vector, surface.vector)
        raise GeometryError("an elliptic ruled support is an extruded ellipse")

    def float_rows(self):
        zero = (0.0, 0.0, 0.0)
        return (zero, self.u_vector, self.v_vector), (self.axis, zero, zero)

    def exact_rows(self):
        zero = (Fraction(0),) * 3
        return (zero, _fraction_vec(self.u_vector), _fraction_vec(self.v_vector)), (_fraction_vec(self.axis), zero, zero), Fraction(0)

    def exact_station(self, axial):
        return Fraction(axial)

    def axial(self, point):
        point = np.asarray(point, dtype=float)
        normal = self.normal
        return (point - self.origin) @ normal / float(normal @ np.asarray(self.axis))

    def angle_of(self, point):
        """The ellipse parameter ``t`` in ``(-pi, pi]`` of the projection of ``point`` along the ruling."""
        point = np.asarray(point, dtype=float)
        single = point.ndim == 1
        point = np.atleast_2d(point)
        rel = point - self.origin
        in_plane = rel - self.axial(point)[:, None] * np.asarray(self.axis)
        solution = np.linalg.lstsq(np.column_stack((self.u_vector, self.v_vector)), in_plane.T, rcond=None)[0]
        angle = np.arctan2(solution[1], solution[0])
        return angle[0] if single else angle

    def quadric(self):
        """The implicit elliptic cylinder, exact in the stored doubles (see :class:`QuadricSupport`)."""
        return QuadricSupport("elliptic", self.origin, self.axis, u_vector=self.u_vector, v_vector=self.v_vector)


def ruled_support(surface):
    """The ruled support of a Cylinder, Cone or extruded ellipse (an already built support passes through)."""
    if isinstance(surface, (RuledSupport, EllipticRuledSupport)):
        return surface
    if isinstance(surface, ExtrudedSurface):
        return EllipticRuledSupport.from_surface(surface)
    return RuledSupport.from_surface(surface)


@dataclass(frozen=True, slots=True)
class QuadricSupport:
    """An implicit quadric ``x.M.x + 2*l.x + c = 0`` kept in its natural exact form.

    ``kind`` is ``plane`` (``origin``, unit ``axis``=normal), ``cylinder`` (``origin``,
    ``axis``, ``radius``), ``cone`` (``origin``, ``axis``, ``radius`` at ``origin``,
    ``slope``), ``elliptic`` (the cylinder of rulings ``axis`` through the ellipse ``origin + u_vector cos t +
    v_vector sin t``), ``parabolic`` (rulings ``axis`` through the quadratic Bezier curve with control points
    ``origin``, ``u_vector``, ``v_vector``) or ``general`` (``matrix`` row-major, ``linear``, ``constant``).
    Coefficients are rational functions of the stored doubles.
    """
    kind: str
    origin: tuple = (0.0, 0.0, 0.0)
    axis: tuple = (0.0, 0.0, 1.0)
    radius: float = 0.0
    slope: float = 0.0
    matrix: tuple = ()
    linear: tuple = ()
    constant: float = 0.0
    u_vector: tuple = ()
    v_vector: tuple = ()

    def __post_init__(self):
        if self.kind not in ("plane", "cylinder", "cone", "elliptic", "parabolic", "general"):
            raise GeometryError("quadric kind must be plane, cylinder, cone, elliptic, parabolic or general")
        if self.kind == "parabolic":
            origin, p1, p2 = _vec(self.origin, "parabola start"), _vec(self.u_vector, "parabola control"), _vec(
                self.v_vector, "parabola end")
            axis = _vec(self.axis, "parabola ruling")
            first = tuple(b - a for a, b in zip(origin, p1))
            second = tuple(c - 2 * b + a for a, b, c in zip(origin, p1, p2))
            normal = _cross(first, second)
            scale = math.sqrt(sum(c * c for c in normal))
            if scale <= 1e-14 * math.sqrt(sum(c * c for c in first)) * math.sqrt(sum(c * c for c in second)):
                raise GeometryError("the control points of a parabola must not be collinear")
            if abs(sum(n * a for n, a in zip(normal, axis))) <= 1e-12 * scale * math.sqrt(sum(c * c for c in axis)):
                raise GeometryError("the ruling direction must leave the plane of the parabola")
            object.__setattr__(self, "u_vector", p1)
            object.__setattr__(self, "v_vector", p2)
        if self.kind == "elliptic":
            EllipticRuledSupport(self.origin, self.u_vector, self.v_vector, self.axis)       # validates the three vectors
            object.__setattr__(self, "u_vector", _vec(self.u_vector, "ellipse u vector"))
            object.__setattr__(self, "v_vector", _vec(self.v_vector, "ellipse v vector"))
        object.__setattr__(self, "origin", _vec(self.origin, "quadric origin"))
        object.__setattr__(self, "axis", _vec(self.axis, "quadric axis"))
        object.__setattr__(self, "radius", float(self.radius))
        object.__setattr__(self, "slope", float(self.slope))
        if self.kind == "general":
            matrix = tuple(float(item) for item in self.matrix)
            if len(matrix) != 9 or not all(math.isfinite(item) for item in matrix):
                raise GeometryError("a general quadric needs a finite 3x3 matrix")
            object.__setattr__(self, "matrix", matrix)
            object.__setattr__(self, "linear", _vec(self.linear, "quadric linear term"))
            object.__setattr__(self, "constant", float(self.constant))

    @classmethod
    def from_surface(cls, surface):
        if isinstance(surface, cls):
            return surface
        if isinstance(surface, Plane):
            return cls("plane", tuple(surface.origin), tuple(surface.normal))
        if isinstance(surface, Cylinder):
            return cls("cylinder", tuple(surface.origin), tuple(surface.axis), float(surface.radius))
        if isinstance(surface, Cone):
            slope = (float(surface.radius_end) - float(surface.radius_start)) / float(surface.height)
            if slope == 0.0:
                return cls("cylinder", tuple(surface.origin), tuple(surface.axis), float(surface.radius_start))
            return cls("cone", tuple(surface.origin), tuple(surface.axis), float(surface.radius_start), slope)
        if isinstance(surface, ExtrudedSurface) and isinstance(surface.directrix, EllipseDirectrix):
            return EllipticRuledSupport.from_surface(surface).quadric()
        if (isinstance(surface, ExtrudedSurface) and isinstance(surface.directrix, BezierDirectrix)
                and surface.directrix.degree == 2):
            p0, p1, p2 = surface.directrix.controls
            return cls("parabolic", p0, surface.vector, u_vector=p1, v_vector=p2)
        raise GeometryError("quadric supports are Plane, Cylinder, Cone, an extruded ellipse or a quadratic Bezier "
                            "extrusion")

    @classmethod
    def sphere(cls, center, radius):
        c = _vec(center, "sphere center")
        return cls("general", matrix=(1., 0., 0., 0., 1., 0., 0., 0., 1.), linear=tuple(-v for v in c),
                   constant=sum(v * v for v in c) - float(radius) ** 2)

    # -- exact coefficients in the frame of ``frame_origin`` (Fractions): (M rows, l, c)
    def exact_relative(self, frame_origin):
        d = tuple(Fraction(f) - Fraction(o) for f, o in zip(frame_origin, self.origin))   # frame minus own origin
        a = _fraction_vec(self.axis)
        zero3 = (Fraction(0),) * 3
        if self.kind == "general":
            m = tuple(tuple(Fraction(self.matrix[3 * i + j]) for j in range(3)) for i in range(3))
            lin = _fraction_vec(self.linear)
            f = _fraction_vec(frame_origin)
            md = tuple(sum(m[i][j] * f[j] for j in range(3)) for i in range(3))
            l_rel = tuple(lin[i] + md[i] for i in range(3))
            c_rel = (sum(f[i] * md[i] for i in range(3)) + 2 * sum(lin[i] * f[i] for i in range(3))
                     + Fraction(self.constant))
            return m, l_rel, c_rel
        eye = tuple(tuple(Fraction(int(i == j)) for j in range(3)) for i in range(3))
        if self.kind == "plane":
            m = tuple((zero3,) * 3)
            l_own = tuple(c / 2 for c in a)                                   # Q(w) = a.w
            c_own = Fraction(0)
        elif self.kind == "parabolic":
            # X(t) = P0 + 2 t e + t^2 f; the coordinates of y = X - P0 projected along the ruling d onto the plane of
            # (e, f) are (a, b) with y' = a e + b f, and the surface is a^2 = 4 b. Both are linear in y.
            p0, p1, p2 = (_fraction_vec(v) for v in (self.origin, self.u_vector, self.v_vector))
            e = tuple(p1[i] - p0[i] for i in range(3))
            f = tuple(p2[i] - 2 * p1[i] + p0[i] for i in range(3))
            n = _fraction_cross(e, f)
            nn = sum(c * c for c in n)
            nd = sum(n[i] * a[i] for i in range(3))
            if nn == 0 or nd == 0:
                raise GeometryError("the parabola and its ruling direction must be independent")

            def functional(g):                         # y -> (P y) . g with P = I - d n^T / (n . d)
                dg = sum(a[i] * g[i] for i in range(3))
                return tuple(g[i] - n[i] * dg / nd for i in range(3))

            la = functional(tuple(c / nn for c in _fraction_cross(f, n)))
            lb = functional(tuple(c / nn for c in _fraction_cross(n, e)))
            m = tuple(tuple(la[i] * la[j] for j in range(3)) for i in range(3))
            l_own = tuple(-2 * c for c in lb)                                  # Q(y) = a^2 - 4 b = y.M.y + 2 l.y
            c_own = Fraction(0)
        elif self.kind == "elliptic":
            basis = [[Fraction(self.u_vector[i]), Fraction(self.v_vector[i]), a[i]] for i in range(3)]
            inverse = _inverse3(basis)                       # row k gives the k-th coordinate (x, y, s) of w = p - origin
            first, second = inverse[0], inverse[1]
            m = tuple(tuple(first[i] * first[j] + second[i] * second[j] for j in range(3)) for i in range(3))
            l_own = zero3                                                     # Q(w) = w.M.w - 1
            c_own = Fraction(-1)
        else:
            k = Fraction(self.slope) if self.kind == "cone" else Fraction(0)
            r = Fraction(self.radius)
            # a cylinder's axis is divided by its exact norm: its null direction is the stored axis itself, so rulings
            # parallel to it (an exactly equal direction) are parallel in the algebra too
            norm2 = sum(c * c for c in a) if self.kind == "cylinder" else Fraction(1)
            m = tuple(tuple(eye[i][j] - (1 + k * k) * a[i] * a[j] / norm2 for j in range(3)) for i in range(3))
            l_own = tuple(-r * k * c for c in a)                              # Q(w) = w.M.w - 2 r k a.w - r^2
            c_own = -r * r
        md = tuple(sum(m[i][j] * d[j] for j in range(3)) for i in range(3))
        l_rel = tuple(md[i] + l_own[i] for i in range(3))
        c_rel = sum(d[i] * md[i] for i in range(3)) + 2 * sum(l_own[i] * d[i] for i in range(3)) + c_own
        return m, l_rel, c_rel

    def general_float(self):
        """``(M, l, c)`` as floats in absolute coordinates (correctly rounded from the exact form)."""
        m, l, c = self.exact_relative((0, 0, 0))
        return (np.array([[float(v) for v in row] for row in m]), np.array([float(v) for v in l]), float(c))

    def pulled_back(self, transform):
        """The same surface expressed in the reference frame of ``x_world = T x_ref`` (a ``general`` quadric)."""
        matrix = np.asarray(transform, dtype=float)
        a, t = matrix[:3, :3], matrix[:3, 3]
        m, l, c = self.general_float()
        mt = m @ t
        return QuadricSupport("general", matrix=tuple(float(v) for v in (a.T @ m @ a).ravel()),
                              linear=tuple(float(v) for v in a.T @ (mt + l)),
                              constant=float(t @ mt + 2 * l @ t + c))

    def value(self, points):
        """Implicit value ``Q(x)`` in floating point (verification and residuals)."""
        x = np.asarray(points, dtype=float)
        w = x - self.origin
        if self.kind == "plane":
            return w @ np.asarray(self.axis)
        if self.kind in ("elliptic", "parabolic"):
            m, l, c = _own_form(self)
            return np.einsum("...i,ij,...j->...", w, m, w) + 2 * w @ l + c
        if self.kind == "general":
            m = np.asarray(self.matrix).reshape(3, 3)
            return np.einsum("...i,ij,...j->...", x, m, x) + 2 * x @ np.asarray(self.linear) + self.constant
        axial = w @ np.asarray(self.axis)
        radial = w - axial[..., None] * np.asarray(self.axis)
        radius = self.radius + (self.slope * axial if self.kind == "cone" else 0.0)
        return np.einsum("...i,...i->...", radial, radial) - radius * radius

    def gradient_norm(self, points):
        x = np.asarray(points, dtype=float)
        w = x - self.origin
        a = np.asarray(self.axis)
        if self.kind == "plane":
            return np.full(x.shape[:-1], float(np.linalg.norm(a)))
        if self.kind in ("elliptic", "parabolic"):
            m, l, _c = _own_form(self)
            return 2 * np.linalg.norm(w @ m + l, axis=-1)
        if self.kind == "general":
            m = np.asarray(self.matrix).reshape(3, 3)
            return 2 * np.linalg.norm(x @ m + np.asarray(self.linear), axis=-1)
        axial = w @ a
        radial = w - axial[..., None] * a
        k = self.slope if self.kind == "cone" else 0.0
        radius = self.radius + k * axial
        return 2 * np.linalg.norm(radial - (k * radius)[..., None] * a, axis=-1)


@lru_cache(maxsize=256)
def _own_form(support):
    """``(M, l, c)`` of ``w.M.w + 2 l.w + c`` in the frame of the support's own origin, correctly rounded."""
    m, l, c = support.exact_relative(support.origin)
    return np.array([[float(v) for v in row] for row in m]), np.array([float(v) for v in l]), float(c)


# ---------------------------------------------------------------------------
# Homogenized integer polynomials in x = tan(t/2)


def _pmul(p, q):
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                if b:
                    out[i + j] += a * b
    return out


def _padd(p, q, sign=1):
    size = max(len(p), len(q))
    return [(p[i] if i < len(p) else 0) + sign * (q[i] if i < len(q) else 0) for i in range(size)]


def _trim_ints(p):
    p = list(p)
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    return p


_ONE_PLUS_X2 = [[1]]
for _k in range(1, 12):
    _ONE_PLUS_X2.append(_pmul(_ONE_PLUS_X2[-1], [1, 0, 1]))


class HP:
    """``T(t) * (1 + x^2)^n`` for a trigonometric polynomial ``T`` of degree <= ``n``.

    ``coeffs`` ascend in ``x = tan(t/2)``; the formal degree is ``2*n``. A product
    adds the ``n``; a sum lifts the smaller operand by powers of ``1 + x^2``.
    """
    __slots__ = ("c", "n")

    def __init__(self, coeffs, n):
        self.c = _trim_ints(coeffs)
        self.n = n

    @staticmethod
    def trig1(c0, a1, b1):
        """``c0 + a1 cos t + b1 sin t`` (integers) at ``n = 1``, or a constant at ``n = 0``."""
        if a1 == 0 and b1 == 0:
            return HP([c0], 0)
        return HP([c0 + a1, 2 * b1, c0 - a1], 1)

    def lift(self, n):
        if n == self.n:
            return self
        return HP(_pmul(self.c, _ONE_PLUS_X2[n - self.n]), n)

    def __mul__(self, other):
        if isinstance(other, int):
            return HP([other * v for v in self.c], self.n)
        return HP(_pmul(self.c, other.c), self.n + other.n)

    __rmul__ = __mul__

    def __add__(self, other):
        n = max(self.n, other.n)
        return HP(_padd(self.lift(n).c, other.lift(n).c), n)

    def __sub__(self, other):
        n = max(self.n, other.n)
        return HP(_padd(self.lift(n).c, other.lift(n).c, -1), n)

    def is_zero(self):
        return all(v == 0 for v in self.c)

    def value_at(self, t):
        x = math.tan(t / 2)
        return sum(float(v) * x ** i for i, v in enumerate(self.c)) / (1 + x * x) ** self.n


def _common_integers(rows):
    """Scale an iterable of Fractions by their common denominator: ``(ints, L)``."""
    values = list(rows)
    lcm = 1
    for value in values:
        d = value.denominator
        lcm = lcm * d // gcd(lcm, d)
    return [int(v * lcm) for v in values], lcm


def _dot(u, v):
    result = u[0] * v[0]
    for i in range(1, len(u)):
        result = result + u[i] * v[i]
    return result


def build_hp(first, second):
    """Exact ``(A, B, C)`` as :class:`HP` for ``Q(P + s D) = A s^2 + B s + C``.

    Built in the frame of ``first.origin`` from rational arithmetic. All three share
    one positive scale factor, which no root depends on.
    """
    p_rows, d_rows, _reach = first.exact_rows()
    m, l, c = QuadricSupport.from_surface(second).exact_relative(first.origin)
    # Common denominator over every rational that enters the three coefficients.
    flat = [v for row in (*p_rows, *d_rows) for v in row] + [v for row in m for v in row] + list(l) + [c]
    ints, scale = _common_integers(flat)
    it = iter(ints)
    p_int = [[next(it) for _ in range(3)] for _ in range(3)]
    d_int = [[next(it) for _ in range(3)] for _ in range(3)]
    m_int = [[next(it) for _ in range(3)] for _ in range(3)]
    l_int = [next(it) for _ in range(3)]
    c_int = next(it)

    def vector(rows):
        return [HP.trig1(rows[0][i], rows[1][i], rows[2][i]) for i in range(3)]

    P, D = vector(p_int), vector(d_int)
    MD = [sum_hp([D[j] * m_int[i][j] for j in range(3)]) for i in range(3)]
    MP = [sum_hp([P[j] * m_int[i][j] for j in range(3)]) for i in range(3)]
    A = sum_hp([D[i] * MD[i] for i in range(3)])
    B = (sum_hp([P[i] * MD[i] for i in range(3)])
         + sum_hp([D[i] * (l_int[i] * scale) for i in range(3)])) * 2
    C = (sum_hp([P[i] * MP[i] for i in range(3)])
         + sum_hp([P[i] * (l_int[i] * scale) for i in range(3)]) * 2
         + HP([c_int * scale * scale], 0))
    return A, B, C


def sum_hp(items):
    total = items[0]
    for item in items[1:]:
        total = total + item
    return total


def discriminant_hp(A, B, C):
    return B * B - (A * C) * 4


def resultant_hp(A2, B2, C2, A3, B3, C3):
    """Common-root condition of two quadratics in ``s`` (zero means a shared ruling point exists).

    Handles the linear cases (a plane has ``A == 0``) exactly; a zero polynomial in
    both cases is returned as such.
    """
    if A2.is_zero() and A3.is_zero():
        return B2 * C3 - B3 * C2
    if A3.is_zero():
        return A2 * (C3 * C3) - B2 * (B3 * C3) + C2 * (B3 * B3)
    if A2.is_zero():
        return A3 * (C2 * C2) - B3 * (B2 * C2) + C3 * (B2 * B2)
    m = A2 * C3 - A3 * C2
    ab = A2 * B3 - A3 * B2
    bc = B2 * C3 - B3 * C2
    return m * m - ab * bc


# ---------------------------------------------------------------------------
# Roots of an HP over the full circle, with exact multiplicity


def _classes(poly):
    """Yun decomposition: ``classes[k-1]`` has exactly the roots of multiplicity >= ``k``."""
    classes = []
    a = _integer_row(poly)
    while len(a) > 1:
        g = _integer_gcd(a, _derivative(a))
        classes.append(_exact_quotient(a, g) if len(g) > 1 else a)
        a = g
    return classes


def _chart_roots(poly):
    """``[(x, multiplicity)]`` for the roots of integer polynomial ``poly`` in ``[-1, 1]``."""
    if len(poly) < 2:
        return []
    classes = _classes(poly)
    sequences = [None, *(_sturm(cls) for cls in classes[1:])]          # the first class needs no multiplicity test
    tolerance = 4 * np.finfo(float).eps
    out = []
    for root in isolate_real_roots(poly, tolerance=tolerance, interval=(-1, 1)):
        multiplicity = 1
        for k, (cls, sequence) in enumerate(zip(classes[1:], sequences[1:]), start=2):
            if root.lower == root.upper:
                hit = _integer_value(cls, root.lower) == 0
            else:
                hit = _variations(sequence, root.lower) - _variations(sequence, root.upper) >= 1
            if hit:
                multiplicity = k
        out.append((root.witness, multiplicity))
    return out


def hp_roots(hp):
    """Roots of ``hp`` over the whole circle as ``[(angle in [0, 2 pi), multiplicity)]``.

    ``None`` when ``hp`` is identically zero. The point ``t = pi`` (``x`` infinite)
    is a root of order ``2*n - degree``. Roots with ``|x| > 1`` are isolated in
    ``y = 1/x`` so their precision stays relative.
    """
    if hp.is_zero():
        return None
    poly = hp.c
    result = _finite_circle_roots(poly)
    deficit = 2 * hp.n + 1 - len(poly)
    if deficit > 0:
        result.append((math.pi, deficit))
    result.sort()
    return result


def _finite_circle_roots(poly):
    """Roots of the integer polynomial ``poly`` in ``x = tan(t/2)`` as ``[(angle in [0, 2 pi), multiplicity)]``."""
    found = {}
    for x, multiplicity in _chart_roots(poly):
        found[round(x, 15)] = (2 * math.atan(x), multiplicity)
    for y, multiplicity in _chart_roots(_trim_ints(list(reversed(poly)))):
        if y == 0:
            continue
        angle = 2 * math.atan2(1.0, y)
        if abs(y) == 1.0 and round(1.0 / y, 15) in found:
            continue
        found[round(1.0 / y, 15)] = (angle, multiplicity)
    return [(angle % TWO_PI, multiplicity) for angle, multiplicity in found.values()]


def _poly_gcd(polynomials):
    """Integer-coefficient gcd of ascending polynomials (zero ones divide nothing new); ``None`` if all are zero."""
    current = None
    for poly in polynomials:
        q = _trim_ints(list(poly))
        if all(v == 0 for v in q):
            continue
        current = _integer_row(q) if current is None else _integer_gcd(current, q)
    return None if current is None else list(current)


# ---------------------------------------------------------------------------
# The cached plan for one (ruled support, quadric) pair


class Plan:
    """Everything that depends on a support pair but not on a patch (lazy, shared by all facets)."""

    __slots__ = ("first", "second", "hp", "linear", "_disc", "_poles", "_bound", "_resultant", "_float", "_common")

    def __init__(self, first, second):
        self.first, self.second = first, second
        self.hp = build_hp(first, second)
        self.linear = self.hp[0].is_zero()
        self._disc = self._poles = self._common = None
        self._bound = {}
        self._resultant = {}
        self._float = None

    # -- exact events (full circle)
    def discriminant_roots(self):
        """``[(angle, multiplicity)]`` of ``B^2 - 4AC`` (empty for a linear branch)."""
        if self._disc is None:
            if self.linear:
                self._disc = []
            else:
                roots = hp_roots(discriminant_hp(*self.hp))
                self._disc = [] if roots is None else roots
        return self._disc

    def common_roots(self):
        """Angles where ``A``, ``B`` and ``C`` all vanish: the whole ruling lies on the second support.

        Such a generator is a component of the intersection by itself (a line), and the branches end on it
        with limits rather than values. ``[(angle, 1)]``, empty when the three are coprime.
        """
        if self._common is None:
            n = max(h.n for h in self.hp)
            lifted = [h.lift(n).c for h in self.hp]
            roots = []
            divisor = _poly_gcd(lifted)
            if divisor is not None and len(divisor) > 1:
                roots.extend((angle, 1) for angle, _m in _finite_circle_roots(divisor))
            if n > 0 and all(len(c) <= 2 * n for c in lifted):
                roots.append((math.pi, 1))                 # x = infinity: every leading coefficient vanishes
            self._common = sorted(roots)
        return self._common

    def pole_roots(self):
        """Roots of ``A`` (quadratic) or of ``B`` (linear): angles where a root leaves to infinity."""
        if self._poles is None:
            roots = hp_roots(self.hp[1] if self.linear else self.hp[0])
            self._poles = [] if roots is None else roots
        return self._poles

    def bound_roots(self, axial):
        """Angles where the branch crosses axial station ``axial`` (a float): roots of ``A s0^2 + B s0 + C``."""
        key = float(axial)
        if key not in self._bound:
            s0 = self.first.exact_station(key)
            n = max(h.n for h in self.hp)
            A, B, C = (h.lift(n) for h in self.hp)
            num, den = s0.numerator, s0.denominator
            poly = _padd(_padd([v * num * num for v in A.c], [v * num * den for v in B.c]),
                         [v * den * den for v in C.c])
            roots = hp_roots(HP(poly, n))
            self._bound[key] = [] if roots is None else roots
        return self._bound[key]

    def floats(self):
        """The shared :class:`FloatSystem` (built on first use)."""
        if self._float is None:
            self._float = FloatSystem(self)
        return self._float

    def resultant_roots(self, other):
        """Angles where the branch can meet ``other`` (resultant of the two ruling quadratics), or ``None``
        when the resultant vanishes identically (a shared curve or a degenerate pair)."""
        other = QuadricSupport.from_surface(other)
        if other not in self._resultant:
            A3, B3, C3 = build_hp(self.first, other)
            self._resultant[other] = hp_roots(resultant_hp(*self.hp, A3, B3, C3))
        return self._resultant[other]


@lru_cache(maxsize=512)
def _cached_plan(first, second):
    return Plan(first, _aligned(first, second))


_PARALLEL_ROUNDING = 64 * float(np.finfo(float).eps)


def _aligned(first, second):
    """``second`` with its rulings made exactly parallel to ``first``'s when they agree up to rounding.

    Two cylinder-like supports whose unit directions differ by a few ulp are parallel in every modelling
    sense (their axes come from one vector normalized twice), yet exact arithmetic on the stored doubles
    would find them crossing far outside any patch. Replacing the second direction by the first changes the
    support by less than ``64 eps`` of its length and lets the plan see the generators they share.
    """
    if first.is_cone or second.kind not in ("cylinder", "elliptic", "parabolic"):
        return second
    a, b = np.asarray(first.axis, dtype=float), np.asarray(second.axis, dtype=float)
    size = float(np.linalg.norm(np.cross(a, b))) / float(np.linalg.norm(a) * np.linalg.norm(b))
    if size <= _PARALLEL_ROUNDING and tuple(second.axis) != tuple(first.axis):     # parallel to rounding, not yet equal
        return replace(second, axis=first.axis)
    return second


def get_plan(first, second):
    """The shared :class:`Plan` for a support pair; ``second`` may be a surface or a quadric."""
    return _cached_plan(first, QuadricSupport.from_surface(second))


def circle_angles(roots, lower, upper, tolerance=1e-12):
    """Every ``angle + 2 pi k`` of ``roots`` inside ``[lower, upper]`` (clipped), as ``[(angle, multiplicity)]``."""
    result = []
    for angle, multiplicity in roots:
        first = math.ceil((lower - angle - tolerance) / TWO_PI)
        last = math.floor((upper - angle + tolerance) / TWO_PI)
        for turn in range(first, last + 1):
            result.append((min(upper, max(lower, angle + turn * TWO_PI)), multiplicity))
    result.sort()
    return result


# ---------------------------------------------------------------------------
# Floating-point trigonometric system (evaluation, derivatives, difference form)


def tp_mul(p, q):
    """Product of trigonometric polynomials ``(c0, a1, b1, ..., an, bn)`` (floats)."""
    n, m = (len(p) - 1) // 2, (len(q) - 1) // 2
    cosine = [0.0] * (n + m + 1)
    sine = [0.0] * (n + m + 1)

    def harmonics(values, count):
        return [(values[0], 0.0)] + [(values[2 * k - 1], values[2 * k]) for k in range(1, count + 1)]

    for j, (aj, bj) in enumerate(harmonics(p, n)):
        for k, (ak, bk) in enumerate(harmonics(q, m)):
            if aj and ak:
                cosine[j + k] += aj * ak / 2
                cosine[abs(j - k)] += aj * ak / 2
            if bj and bk:
                cosine[abs(j - k)] += bj * bk / 2
                cosine[j + k] -= bj * bk / 2
            if bj and ak:
                sine[j + k] += bj * ak / 2
                if j != k:
                    sine[abs(j - k)] += (1 if j > k else -1) * bj * ak / 2
            if aj and bk:
                sine[j + k] += aj * bk / 2
                if j != k:
                    sine[abs(j - k)] += (1 if k > j else -1) * aj * bk / 2
    out = [cosine[0]]
    for k in range(1, n + m + 1):
        out.extend((cosine[k], sine[k]))
    return tuple(out)


def tp_add(*items):
    size = max(len(item) for item in items)
    return tuple(sum((item[i] if i < len(item) else 0.0) for item in items) for i in range(size))


def tp_scale(p, factor):
    return tuple(factor * value for value in p)


def tp_derivative(p):
    out = [0.0]
    for k in range(1, (len(p) - 1) // 2 + 1):
        out.extend((k * p[2 * k], -k * p[2 * k - 1]))
    return tuple(out)


def tp_eval(p, angle):
    """Evaluate at a float or an array of angles."""
    if isinstance(angle, float):
        value = p[0]
        for k in range(1, (len(p) - 1) // 2 + 1):
            value += p[2 * k - 1] * math.cos(k * angle) + p[2 * k] * math.sin(k * angle)
        return value
    angle = np.asarray(angle, dtype=float)
    value = np.full_like(angle, p[0])
    for k in range(1, (len(p) - 1) // 2 + 1):
        value = value + p[2 * k - 1] * np.cos(k * angle) + p[2 * k] * np.sin(k * angle)
    return value


class FloatSystem:
    """Correctly rounded trigonometric coefficients of a plan, with derivatives, for evaluation.

    ``abc[i]`` holds the ``i``-th angular derivative (``0..3``) of ``A``, ``B`` and ``C``;
    ``disc[i]`` those of the discriminant (``0..3``). Rows are relative to the support origin.
    """
    __slots__ = ("p_rows", "d_rows", "abc", "disc", "linear", "origin", "scale")

    def __init__(self, plan):
        first = plan.first
        self.origin = np.asarray(first.origin)
        self.p_rows, self.d_rows = first.float_rows()
        m, l, c = plan.second.exact_relative(first.origin)
        m = [[float(v) for v in row] for row in m]
        l = [float(v) for v in l]
        c = float(c)
        P = [(self.p_rows[0][i], self.p_rows[1][i], self.p_rows[2][i]) for i in range(3)]
        D = [(self.d_rows[0][i], self.d_rows[1][i], self.d_rows[2][i]) for i in range(3)]
        MD = [tp_add(*(tp_scale(D[j], m[i][j]) for j in range(3))) for i in range(3)]
        MP = [tp_add(*(tp_scale(P[j], m[i][j]) for j in range(3))) for i in range(3)]
        A = tp_add(*(tp_mul(D[i], MD[i]) for i in range(3)))
        B = tp_scale(tp_add(*(tp_mul(P[i], MD[i]) for i in range(3)),
                            *(tp_scale(D[i], l[i]) for i in range(3))), 2.0)
        C = tp_add(*(tp_mul(P[i], MP[i]) for i in range(3)),
                   tp_scale(tp_add(*(tp_scale(P[i], l[i]) for i in range(3))), 2.0), (c,))
        self.linear = plan.linear
        disc = tp_add(tp_mul(B, B), tp_scale(tp_mul(A, C), -4.0))
        derivatives = []
        for poly in (A, B, C):
            rows = [poly]
            for _ in range(3):
                rows.append(tp_derivative(rows[-1]))
            derivatives.append(rows)
        def padded(poly, size):
            return tuple(poly) + (0.0,) * (size - len(poly))

        self.abc = tuple(tuple(padded(poly, 5) for poly in order) for order in zip(*derivatives))
        rows = [disc]
        for _ in range(3):
            rows.append(tp_derivative(rows[-1]))
        self.disc = tuple(padded(row, 9) for row in rows)
        self.scale = max([abs(v) for poly in (A, B, C) for v in poly] + [1e-300])

    def abc_scalar(self, theta, order=0):
        """``(A, B, C)`` (or an angular derivative) at one float angle, without numpy."""
        a, b, c = self.abc[order]
        c1, s1 = math.cos(theta), math.sin(theta)
        c2, s2 = c1 * c1 - s1 * s1, 2.0 * s1 * c1
        return (a[0] + a[1] * c1 + a[2] * s1 + a[3] * c2 + a[4] * s2,
                b[0] + b[1] * c1 + b[2] * s1 + b[3] * c2 + b[4] * s2,
                c[0] + c[1] * c1 + c[2] * s1 + c[3] * c2 + c[4] * s2)

    def discriminant_scalar(self, theta, order=0):
        d = self.disc[order]
        value = d[0]
        for k in range(1, 5):
            value += d[2 * k - 1] * math.cos(k * theta) + d[2 * k] * math.sin(k * theta)
        return value

    def difference_scalar(self, anchor, delta):
        d = self.disc[0]
        total = 0.0
        for k in range(1, 5):
            half = .5 * k * delta
            mid = k * anchor + half
            sinc = 1.0 if half == 0.0 else math.sin(half) / half
            total += k * sinc * (-d[2 * k - 1] * math.sin(mid) + d[2 * k] * math.cos(mid))
        return total

    def at(self, theta, order=0):
        """``(A, B, C)`` of the ``order``-th angular derivative at ``theta``."""
        a, b, c = self.abc[order]
        return tp_eval(a, theta), tp_eval(b, theta), tp_eval(c, theta)

    def discriminant(self, theta, order=0):
        return tp_eval(self.disc[order], theta)

    def difference(self, anchor, delta):
        """``(Delta(anchor + delta) - Delta(anchor)) / delta`` by trigonometric difference identities."""
        coefficients = self.disc[0]
        anchor = np.asarray(anchor, dtype=float)
        delta = np.asarray(delta, dtype=float)
        total = np.zeros(np.broadcast(anchor, delta).shape)
        for k in range(1, (len(coefficients) - 1) // 2 + 1):
            half = .5 * k * delta
            mid = k * anchor + half
            total = total + k * np.sinc(half / math.pi) * (-coefficients[2 * k - 1] * np.sin(mid)
                                                           + coefficients[2 * k] * np.cos(mid))
        return total
