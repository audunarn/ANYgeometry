"""PROTOTYPE - exact branches of a ruled quadric with an implicit quadric.

Not wired into any engine path, not exported, no serialization. It exists to
test one claim: the exact machinery that already handles cylinder/cylinder
(graph of a quadratic root over the first support's angle, Sturm-isolated
events, resultant elimination against other quadrics) carries over to a Cone
as the parametrized support and to any quadric (Plane, Cylinder, Cone) as the
other support.

A *ruled support* is ``S(t, s) = P(t) + s*D(t)`` with ``P`` and ``D`` degree-one
trigonometric vector polynomials in the angle ``t``. A Cylinder has constant
``D`` (its axis); a Cone has constant ``P`` (its apex). A *quadric* is
``x.M.x + 2*l.x + c = 0``. Along a ruling the quadric equation is
``A(t)*s**2 + B(t)*s + C(t) = 0`` with trigonometric coefficients, so every
branch is ``s = (-B +- sqrt(B*B - 4*A*C)) / (2*A)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import math

import numpy as np

from .analytic_roots import isolate_real_roots, _derivative, _division, _integer_value, _sturm, _variations
from .errors import GeometryError
from .surfaces import Cone, Cylinder, Plane

# --------------------------------------------------------------------------
# Trigonometric polynomials: (c0, a1, b1, ..., an, bn) = c0 + sum a_k cos kt + b_k sin kt.
# Written once for float and Fraction coefficients.


def tp_mul(p, q):
    n, m = (len(p) - 1) // 2, (len(q) - 1) // 2
    cosine = [0] * (n + m + 1)
    sine = [0] * (n + m + 1)

    def harmonics(values, count):
        return [(values[0], 0)] + [(values[2 * k - 1], values[2 * k]) for k in range(1, count + 1)]

    for j, (aj, bj) in enumerate(harmonics(p, n)):
        for k, (ak, bk) in enumerate(harmonics(q, m)):
            if aj != 0 and ak != 0:                     # cos j * cos k
                cosine[j + k] += aj * ak / 2
                cosine[abs(j - k)] += aj * ak / 2
            if bj != 0 and bk != 0:                     # sin j * sin k
                cosine[abs(j - k)] += bj * bk / 2
                cosine[j + k] -= bj * bk / 2
            if bj != 0 and ak != 0:                     # sin j * cos k
                sine[j + k] += bj * ak / 2
                if j != k:
                    sine[abs(j - k)] += (1 if j > k else -1) * bj * ak / 2
            if aj != 0 and bk != 0:                     # cos j * sin k
                sine[j + k] += aj * bk / 2
                if j != k:
                    sine[abs(j - k)] += (1 if k > j else -1) * aj * bk / 2
    out = [cosine[0]]
    for k in range(1, n + m + 1):
        out.extend((cosine[k], sine[k]))
    return tuple(out)


def tp_add(*items):
    size = max(len(item) for item in items)
    return tuple(sum((item[i] if i < len(item) else 0) for item in items) for i in range(size))


def tp_scale(p, factor):
    return tuple(factor * value for value in p)


def tp_eval(p, angle):
    angle = np.asarray(angle, dtype=float)
    value = np.full_like(angle, float(p[0]))
    for k in range(1, (len(p) - 1) // 2 + 1):
        value = value + float(p[2 * k - 1]) * np.cos(k * angle) + float(p[2 * k]) * np.sin(k * angle)
    return value


def tp_derivative(p):
    out = [0]
    for k in range(1, (len(p) - 1) // 2 + 1):
        out.extend((k * p[2 * k], -k * p[2 * k - 1]))
    return tuple(out)


def tp_is_zero(p):
    return not any(value != 0 for value in p)


def _tangent_half_polynomial(p):
    """``p(t)*(1+x*x)**n`` as an ascending polynomial in ``x = tan(t/2)``."""
    n = (len(p) - 1) // 2

    def complex_power(k):                   # (1 + i x)**(2k) -> (real poly, imag poly)
        real, imag = [Fraction(1)], [Fraction(0)]
        for _ in range(2 * k):
            new_real = [Fraction(0)] * (len(real) + 1)
            new_imag = [Fraction(0)] * (len(real) + 1)
            for i, (r, m) in enumerate(zip(real, imag)):
                new_real[i] += r
                new_imag[i] += m
                new_real[i + 1] -= m            # i*x * (r + i m) = -m x + i r x
                new_imag[i + 1] += r
            real, imag = new_real, new_imag
        return real, imag

    def times_one_plus_square(poly, power):
        for _ in range(power):
            out = [Fraction(0)] * (len(poly) + 2)
            for i, value in enumerate(poly):
                out[i] += value
                out[i + 2] += value
            poly = out
        return poly

    total = [Fraction(0)] * (2 * n + 1)
    constant = times_one_plus_square([Fraction(p[0])], n)
    for i, value in enumerate(constant):
        total[i] += value
    for k in range(1, n + 1):
        real, imag = complex_power(k)
        for coefficient, part in ((Fraction(p[2 * k - 1]), real), (Fraction(p[2 * k]), imag)):
            if coefficient == 0:
                continue
            scaled = times_one_plus_square([coefficient * value for value in part], n - k)
            for i, value in enumerate(scaled):
                total[i] += value
    while len(total) > 1 and total[-1] == 0:
        total.pop()
    return tuple(total)


def tp_roots(p, *, start=0.0, sweep=math.tau, tolerance=1e-12, cancellation_check=None):
    """Every angle in ``[start, start+sweep]`` where the exact polynomial vanishes."""
    if tp_is_zero(p):
        return ()
    n = (len(p) - 1) // 2
    poly = _tangent_half_polynomial(tuple(Fraction(v) for v in p))
    angles = []
    if len(poly) > 1:
        roots = isolate_real_roots(poly, tolerance=tolerance / 4, cancellation_check=cancellation_check)
        angles = [2 * math.atan(root.witness) for root in roots]
    if len(poly) < 2 * n + 1:                   # vanishing leading coefficient: the point t = pi
        angles.append(math.pi)
    lower, upper = sorted((start, start + sweep))
    result = []
    for angle in angles:
        first = math.ceil((lower - angle - tolerance) / math.tau)
        last = math.floor((upper - angle + tolerance) / math.tau)
        for turn in range(first, last + 1):
            result.append(min(upper, max(lower, angle + turn * math.tau)))
    result.sort(reverse=sweep < 0)
    return tuple(result)


def _gcd(a, b):
    zero = (Fraction(0),)
    while b != zero and b != (0,):
        a, b = b, _division(a, b)[1]
    return a


def _multiplicity_classes(poly):
    """Yun decomposition: classes[k-1] has exactly the roots of multiplicity >= k."""
    classes = []
    a = tuple(poly)
    while len(a) > 1:
        g = _gcd(a, _derivative(a))
        classes.append(_division(a, g)[0] if len(g) > 1 else a)
        a = g
    return classes


def tp_roots_multiplicity(p, *, start=0.0, sweep=math.tau, tolerance=1e-12, cancellation_check=None):
    """``[(angle, multiplicity)]`` for every root in ``[start, start+sweep]``.

    Multiplicity is exact: a Sturm count of each multiplicity class inside the
    root's isolating interval, not an estimate from derivatives.
    """
    if tp_is_zero(p):
        return []
    n = (len(p) - 1) // 2
    poly = _tangent_half_polynomial(tuple(Fraction(v) for v in p))
    found = []
    if len(poly) > 1:
        classes = _multiplicity_classes(poly)
        sequences = [_sturm(c) for c in classes]
        for root in isolate_real_roots(poly, tolerance=tolerance / 4, cancellation_check=cancellation_check):
            multiplicity = 1
            for k, (cls, sequence) in enumerate(zip(classes[1:], sequences[1:]), start=2):
                if root.lower == root.upper:               # an exact dyadic root is a point interval
                    hit = _integer_value(tuple(Fraction(v) for v in cls), root.lower) == 0
                else:
                    hit = _variations(sequence, root.lower) - _variations(sequence, root.upper) >= 1
                if hit:
                    multiplicity = k
            found.append((2 * math.atan(root.witness), multiplicity))
    if len(poly) < 2 * n + 1:                   # root at t = pi of order 2n - degree
        found.append((math.pi, 2 * n + 1 - len(poly)))
    lower, upper = sorted((start, start + sweep))
    result = []
    for angle, multiplicity in found:
        first = math.ceil((lower - angle - tolerance) / math.tau)
        last = math.floor((upper - angle + tolerance) / math.tau)
        for turn in range(first, last + 1):
            result.append((min(upper, max(lower, angle + turn * math.tau)), multiplicity))
    result.sort(reverse=sweep < 0)
    return result


# --------------------------------------------------------------------------
# Supports


def _frac(value):
    return Fraction(float(value))


@dataclass(frozen=True, slots=True)
class Quadric:
    """``x.M.x + 2*l.x + c = 0`` for finite doubles, exactly as supplied."""
    matrix: tuple
    linear: tuple
    constant: float

    @classmethod
    def from_surface(cls, surface):
        if isinstance(surface, cls):
            return surface
        if isinstance(surface, Plane):
            n = np.asarray(surface.normal, dtype=float)
            return cls(((0., 0., 0.),) * 3, tuple(float(v) for v in n / 2), float(-(n @ surface.origin)))
        if isinstance(surface, Cylinder):
            a = np.asarray(surface.axis)
            m = np.eye(3) - np.outer(a, a)
            o = np.asarray(surface.origin)
            return cls(tuple(map(tuple, m)), tuple(float(v) for v in -(m @ o)),
                       float(o @ m @ o - surface.radius ** 2))
        if isinstance(surface, Cone):
            apex, a, k = cone_apex_axis_slope(surface)
            m = np.eye(3) - (1.0 + k * k) * np.outer(a, a)
            return cls(tuple(map(tuple, m)), tuple(float(v) for v in -(m @ apex)),
                       float(apex @ m @ apex))
        raise GeometryError("quadric supports are Plane, Cylinder or Cone")

    @classmethod
    def sphere(cls, center, radius):
        c = np.asarray(center, dtype=float)
        return cls(tuple(map(tuple, np.eye(3))), tuple(float(v) for v in -c), float(c @ c - radius ** 2))

    def value(self, points):
        x = np.asarray(points, dtype=float)
        m = np.asarray(self.matrix)
        return np.einsum("...i,ij,...j->...", x, m, x) + 2 * x @ np.asarray(self.linear) + self.constant

    def gradient_norm(self, points):
        x = np.asarray(points, dtype=float)
        return 2 * np.linalg.norm(x @ np.asarray(self.matrix) + np.asarray(self.linear), axis=-1)


def cone_apex_axis_slope(cone):
    r0, r1, h = float(cone.radius_start), float(cone.radius_end), float(cone.height)
    k = (r1 - r0) / h
    if k == 0:
        raise GeometryError("a cylindrical Cone is a Cylinder")
    z_apex = -r0 / k
    return np.asarray(cone.origin) + z_apex * np.asarray(cone.axis), np.asarray(cone.axis), k


@dataclass(frozen=True, slots=True)
class RuledSupport:
    """``S(t, s) = P(t) + s*D(t)`` over a native angular and ``s`` rectangle."""
    p: tuple                    # (p0, p1, p2): P = p0 + p1 cos t + p2 sin t
    d: tuple                    # (d0, d1, d2)
    start_angle: float
    sweep_angle: float
    s_range: tuple
    surface: object

    @classmethod
    def from_surface(cls, surface):
        e1 = np.asarray(surface.radial_direction)
        e2 = np.asarray(surface.circumferential_direction)
        axis = np.asarray(surface.axis)
        zero = (0., 0., 0.)
        if isinstance(surface, Cylinder):
            p = (tuple(surface.origin), tuple(surface.radius * e1), tuple(surface.radius * e2))
            d = (tuple(axis), zero, zero)
            z0, z1 = 0.0, float(surface.height)
        elif isinstance(surface, Cone):
            apex, _a, k = cone_apex_axis_slope(surface)
            p = (tuple(apex), zero, zero)
            d = (tuple(axis), tuple(k * e1), tuple(k * e2))
            z_apex = float(np.dot(apex - surface.origin, axis))
            z0, z1 = -z_apex, float(surface.height) - z_apex
        else:
            raise GeometryError("ruled supports are Cylinder or Cone")
        return cls(tuple(tuple(float(v) for v in row) for row in p),
                   tuple(tuple(float(v) for v in row) for row in d),
                   float(surface.start_angle), float(surface.sweep_angle),
                   (min(z0, z1), max(z0, z1)), surface)

    def vector(self, rows, angle):
        angle = np.asarray(angle, dtype=float)
        r0, r1, r2 = (np.asarray(row, dtype=float) for row in rows)
        return r0 + np.cos(angle)[..., None] * r1 + np.sin(angle)[..., None] * r2

    def point(self, angle, s):
        return self.vector(self.p, angle) + np.asarray(s, dtype=float)[..., None] * self.vector(self.d, angle)


def _vector_polys(rows):
    return [(_frac(rows[0][i]), _frac(rows[1][i]), _frac(rows[2][i])) for i in range(3)]


@lru_cache(maxsize=512)
def _float_coefficients(p, d, matrix, linear, constant):
    return _coefficients(p, d, matrix, linear, constant, float)


def coefficients(first, second, *, exact=True):
    """Trigonometric ``(A, B, C)`` of ``Q(P + s D) = A s^2 + B s + C``."""
    if not exact:
        return _float_coefficients(first.p, first.d, second.matrix, second.linear, second.constant)
    return _coefficients(first.p, first.d, second.matrix, second.linear, second.constant, _frac)


def _coefficients(p, d, matrix, linear, constant, cast):
    P = [tuple(cast(v) for v in comp) for comp in zip(*p)]
    D = [tuple(cast(v) for v in comp) for comp in zip(*d)]
    M = [[cast(v) for v in row] for row in matrix]
    l = [cast(v) for v in linear]
    c = cast(constant)

    def dot(u, v):
        return tp_add(*(tp_mul(u[i], v[i]) for i in range(3)))

    def matvec(vec):
        return [tp_add(*(tp_scale(vec[j], M[i][j]) for j in range(3))) for i in range(3)]

    MD, MP = matvec(D), matvec(P)
    A = dot(D, MD)
    B = tp_scale(tp_add(dot(P, MD), tp_add(*(tp_scale(D[i], l[i]) for i in range(3)))), cast(2))
    C = tp_add(dot(P, MP), tp_scale(tp_add(*(tp_scale(P[i], l[i]) for i in range(3))), cast(2)), (c,))
    return A, B, C


def discriminant(A, B, C):
    return tp_add(tp_mul(B, B), tp_scale(tp_mul(A, C), -4))


@lru_cache(maxsize=512)
def _float_discriminant(p, d, matrix, linear, constant):
    return discriminant(*_float_coefficients(p, d, matrix, linear, constant))


def _strip(p):
    p = list(p)
    while len(p) > 1 and p[-1] == 0 and p[-2] == 0:
        p.pop(); p.pop()
    return tuple(p)


def _inside(surface, point, tolerance):
    if isinstance(surface, Quadric):
        return True
    u, v = surface.local_uv(point)
    if isinstance(surface, Plane):
        lu = float(np.linalg.norm(surface.u_vector))
        lv = float(np.linalg.norm(surface.v_vector))
        return -tolerance / lu <= u <= 1 + tolerance / lu and -tolerance / lv <= v <= 1 + tolerance / lv
    radius = surface.radius if isinstance(surface, Cylinder) else max(surface.radius_start, surface.radius_end)
    angular = tolerance / max(radius * abs(surface.sweep_angle), tolerance)
    axial = tolerance / max(abs(surface.height), tolerance)
    return -angular <= u <= 1 + angular and -axial <= v <= 1 + axial


@dataclass(frozen=True, slots=True)
class QuadricBranch:
    """One real branch of ``first`` cut with ``second`` over a certified angular chart."""
    first: RuledSupport
    second: Quadric
    start_angle: float
    sweep_angle: float
    branch: int
    left_fold: bool = False
    right_fold: bool = False
    linear: bool = False            # second is a plane: B s + C = 0, a single rational branch

    def _abc(self, angle):
        A, B, C = coefficients(self.first, self.second, exact=False)
        return tp_eval(A, angle), tp_eval(B, angle), tp_eval(C, angle)

    def _angle(self, parameters):
        t = np.asarray(parameters, dtype=float)
        if self.left_fold and self.right_fold:
            delta = np.where(t <= .5, self.sweep_angle * np.sin(.5 * np.pi * t) ** 2,
                             -self.sweep_angle * np.cos(.5 * np.pi * t) ** 2)
            anchor = np.where(t <= .5, self.start_angle, self.start_angle + self.sweep_angle)
            return anchor + delta, anchor, delta
        if self.left_fold:
            delta = self.sweep_angle * t * t
            return self.start_angle + delta, np.full_like(t, self.start_angle), delta
        if self.right_fold:
            delta = -self.sweep_angle * (1 - t) ** 2
            end = self.start_angle + self.sweep_angle
            return end + delta, np.full_like(t, end), delta
        return self.start_angle + self.sweep_angle * t, np.full_like(t, np.nan), None

    def evaluate(self, parameters):
        t = np.asarray(parameters, dtype=float)
        angle, anchor, delta = self._angle(t)
        a, b, c = self._abc(angle)
        if self.linear:
            with np.errstate(divide="ignore", invalid="ignore"):
                return self.first.point(angle, -c / b)
        disc = b * b - 4 * a * c
        if delta is not None:
            # Chart end at a discriminant root: evaluate D(anchor + delta) - D(anchor)
            # with trigonometric difference identities, so the two branches meet
            # exactly at the endpoint instead of 1e-8 apart.
            Dtp = _float_discriminant(self.first.p, self.first.d, self.second.matrix, self.second.linear, self.second.constant)
            diff = np.zeros_like(angle)
            for k in range(1, (len(Dtp) - 1) // 2 + 1):
                half = .5 * k * delta
                mid = k * anchor + half
                diff = diff + 2 * np.sin(half) * (-float(Dtp[2 * k - 1]) * np.sin(mid)
                                                  + float(Dtp[2 * k]) * np.cos(mid))
            disc = diff
        root = np.sqrt(np.maximum(disc, 0))
        sign_b = np.where(b >= 0, 1.0, -1.0)
        q = -.5 * (b + sign_b * root)
        with np.errstate(divide="ignore", invalid="ignore"):
            big = q / a                          # root with the larger magnitude
            small = np.where(q != 0, c / q, 0.0)  # the other, free of cancellation
        # (-b + branch*root)/(2a): for b >= 0 the "+" root is c/q, the "-" root is q/a.
        plus = np.where(sign_b > 0, small, big)
        minus = np.where(sign_b > 0, big, small)
        s = plus if self.branch > 0 else minus
        return self.first.point(angle, s)


# --------------------------------------------------------------------------
# Events and charts


def _angle_tolerance(first, tolerance=1e-10):
    """Merging width for events, as ``analytic_supports`` uses for cylinders."""
    surface = first.surface
    radius = surface.radius if isinstance(surface, Cylinder) else max(surface.radius_start, surface.radius_end)
    return tolerance / max(radius, 1.0)


def branch_events(first, second, *, extra_planes=(), tolerance=1e-10, cancellation_check=None):
    """Every angle where the chart structure changes, from exact polynomial roots."""
    A, B, C = coefficients(first, second)
    start, sweep = first.start_angle, first.sweep_angle
    tol = min(_angle_tolerance(first, tolerance), 4 * np.finfo(float).eps)
    folds = nodes = ()
    if tp_is_zero(A):                  # plane: one rational branch, poles where B vanishes
        poles = tp_roots(_strip(B), start=start, sweep=sweep, tolerance=tol, cancellation_check=cancellation_check)
    else:
        roots = tp_roots_multiplicity(_strip(discriminant(A, B, C)), start=start, sweep=sweep, tolerance=tol,
                                      cancellation_check=cancellation_check)
        if any(m > 2 for _a, m in roots):
            raise GeometryError("higher-order contact of the supports; an explicit chart is required")
        folds = tuple(a for a, m in roots if m == 1)           # square-root folds need a regularized chart end
        nodes = tuple(a for a, m in roots if m == 2)           # double contacts are regular in the linear chart
        poles = tp_roots(_strip(A), start=start, sweep=sweep, tolerance=tol, cancellation_check=cancellation_check)
    bounds = []
    for s0 in first.s_range:
        s0 = Fraction(float(s0))
        eq = tp_add(tp_scale(A, s0 * s0), tp_scale(B, s0), C)
        bounds.extend(tp_roots(_strip(eq), start=start, sweep=sweep, tolerance=tol, cancellation_check=cancellation_check))
    planes = []
    for plane in extra_planes:                     # other support's rings and seam planes
        a3, b3, c3 = coefficients(first, Quadric.from_surface(plane))
        planes.extend(plane_branch_events(A, B, C, a3, b3, c3, start, sweep, tol, cancellation_check))
    return dict(folds=folds, nodes=nodes, poles=poles, bounds=tuple(bounds), planes=tuple(planes))


def plane_branch_events(A2, B2, C2, A3, B3, C3, start, sweep, tol, cancellation_check=None):
    """Angles where two equations in s share a root (their resultant vanishes)."""
    if tp_is_zero(A2) and tp_is_zero(A3):             # two linear equations
        mixed = tp_add(tp_mul(B2, C3), tp_scale(tp_mul(B3, C2), -1))
        return tp_roots(_strip(mixed), start=start, sweep=sweep, tolerance=tol, cancellation_check=cancellation_check)
    if tp_is_zero(A3):                                # quadratic against linear: A2 C3^2 - B2 B3 C3 + C2 B3^2
        mixed = tp_add(tp_mul(A2, tp_mul(C3, C3)), tp_scale(tp_mul(B2, tp_mul(B3, C3)), -1),
                       tp_mul(C2, tp_mul(B3, B3)))
        return tp_roots(_strip(mixed), start=start, sweep=sweep, tolerance=tol, cancellation_check=cancellation_check)
    m = tp_add(tp_mul(A2, C3), tp_scale(tp_mul(A3, C2), -1))
    ab = tp_add(tp_mul(A2, B3), tp_scale(tp_mul(A3, B2), -1))
    bc = tp_add(tp_mul(B2, C3), tp_scale(tp_mul(B3, C2), -1))
    resultant = tp_add(tp_mul(m, m), tp_scale(tp_mul(ab, bc), -1))
    return tp_roots(_strip(resultant), start=start, sweep=sweep, tolerance=tol, cancellation_check=cancellation_check)


def _boundary_planes(surface):
    """Planes carrying the finite boundary of a support's native rectangle."""
    if isinstance(surface, Plane):
        o, u, v = (np.asarray(x) for x in (surface.origin, surface.u_vector, surface.v_vector))
        n = np.cross(u, v)
        return [Plane(o, u, n), Plane(o + v, u, n), Plane(o, v, n), Plane(o + u, v, n)]
    planes = []
    for owner_z in (0.0, float(surface.height)):
        origin = np.asarray(surface.origin) + owner_z * np.asarray(surface.axis)
        planes.append(Plane(origin, np.asarray(surface.radial_direction), np.asarray(surface.circumferential_direction)))
    for angle in (surface.start_angle, surface.start_angle + surface.sweep_angle):
        radial = math.cos(angle) * np.asarray(surface.radial_direction) + math.sin(angle) * np.asarray(surface.circumferential_direction)
        planes.append(Plane(np.asarray(surface.origin), radial, np.asarray(surface.axis)))
    return planes


def support_intersection(first_surface, second_surface, *, tolerance=1e-10, cancellation_check=None, mid_split=False):
    """All branch charts of two Cylinder/Cone/Plane supports inside both native rectangles.

    ``first_surface`` (Cylinder or Cone) supplies the angle. Exact events cut the
    angular range; each interval is classified by its midpoint, never by sampling
    for existence. Returns ``(charts, info)``; ``info['points']`` holds isolated
    contact points that no chart reaches.
    """
    first = RuledSupport.from_surface(first_surface)
    # A bare Quadric (for example a sphere) is an unbounded second support: no finite boundary of its own.
    second = second_surface if isinstance(second_surface, Quadric) else Quadric.from_surface(second_surface)
    A0, B0, _C0 = coefficients(first, second)
    linear = tp_is_zero(A0)
    if linear and tp_is_zero(B0):
        raise GeometryError("rulings are parallel to the plane for every angle; the intersection is not a curve branch")
    events_by_kind = branch_events(first, second,
                                   extra_planes=() if isinstance(second_surface, Quadric) else _boundary_planes(second_surface),
                                   tolerance=tolerance, cancellation_check=cancellation_check)
    folds, nodes = events_by_kind['folds'], events_by_kind['nodes']
    angle_tolerance = _angle_tolerance(first, tolerance)
    events = [first.start_angle, first.start_angle + first.sweep_angle, *folds, *nodes, *events_by_kind['poles'],
              *events_by_kind['bounds'], *events_by_kind['planes']]
    if mid_split or abs(abs(first.sweep_angle) - math.tau) <= angle_tolerance:
        events.append(first.start_angle + .5 * first.sweep_angle)
    events = sorted(events, reverse=first.sweep_angle < 0)
    merged = []
    for value in events:
        if not merged or abs(value - merged[-1]) > angle_tolerance:
            merged.append(value)
    A, B, C = coefficients(first, second, exact=False)
    apex = np.asarray(first.p[0])
    apex_on_second = (isinstance(first_surface, Cone)
                      and abs(float(second.value(apex))) <= tolerance * max(float(second.gradient_norm(apex)), tolerance))
    if apex_on_second and first.s_range[0] <= 0.0 <= first.s_range[1]:
        # The branch s = 0 is the apex itself and the other branch crosses it where B = 0, so the exact
        # structure depends on rounding of the apex position. Never guess: ask for an explicit decision.
        raise GeometryError("cone apex lies on the other support within tolerance; the contact needs an explicit decision")
    curves = []
    for lo, hi in zip(merged, merged[1:]):
        mid = .5 * (lo + hi)
        disc = float(tp_eval(B, mid) ** 2 - 4 * tp_eval(A, mid) * tp_eval(C, mid))
        if not linear and disc <= 0:
            continue
        left = any(abs(lo - f) <= angle_tolerance for f in folds)
        right = any(abs(hi - f) <= angle_tolerance for f in folds)
        for branch in ((1,) if linear else (-1, 1)):
            curve = QuadricBranch(first, second, lo, hi - lo, branch, left, right, linear)
            point = curve.evaluate(.5)
            if apex_on_second and np.linalg.norm(point - apex) <= tolerance and np.linalg.norm(
                    curve.evaluate(0.) - apex) <= tolerance:
                continue                     # the s = 0 root is the apex itself, not a curve
            if _inside(first_surface, point, tolerance) and _inside(second_surface, point, tolerance):
                curves.append(curve)
    points = []
    for angle in (*folds, *nodes):
        a, b = float(tp_eval(A, angle)), float(tp_eval(B, angle))
        if a == 0:
            continue
        point = first.point(angle, -b / (2 * a))
        if (_inside(first_surface, point, tolerance) and _inside(second_surface, point, tolerance)
                and not any(np.linalg.norm(point - curve.evaluate(t)) <= tolerance for curve in curves for t in (0., 1.))):
            points.append(tuple(point))
    return curves, dict(events_by_kind, events=merged, points=points)


def parameter_for_angle(curve, angle):
    """Invert the chart's angular reparametrization (squares or sines at fold ends)."""
    w = (angle - curve.start_angle) / curve.sweep_angle
    w = min(1.0, max(0.0, w))
    if curve.left_fold and curve.right_fold:
        if w <= .5:
            return 2 / math.pi * math.asin(math.sqrt(w))
        return 1 - 2 / math.pi * math.asin(math.sqrt(1 - w))
    if curve.left_fold:
        return math.sqrt(w)
    if curve.right_fold:
        return 1 - math.sqrt(1 - w)
    return w


def branch_roots(curve, other, *, tolerance=1e-10, cancellation_check=None):
    """Chart parameters where the branch lies on another quadric.

    The two quadratics in ``s`` share a root only where their resultant vanishes;
    that trigonometric polynomial is Sturm-isolated exactly, and each candidate is
    qualified by the implicit residual of the branch point. ``None`` means a common
    factor: the whole chart lies on ``other``.
    """
    first = curve.first
    A2, B2, C2 = coefficients(first, curve.second)
    A3, B3, C3 = coefficients(first, other)
    lo, hi = sorted((curve.start_angle, curve.start_angle + curve.sweep_angle))
    tol = 4 * np.finfo(float).eps
    if tp_is_zero(A2) and tp_is_zero(A3):
        resultant = tp_add(tp_mul(B2, C3), tp_scale(tp_mul(B3, C2), -1))
    elif tp_is_zero(A3):
        resultant = tp_add(tp_mul(A2, tp_mul(C3, C3)), tp_scale(tp_mul(B2, tp_mul(B3, C3)), -1),
                           tp_mul(C2, tp_mul(B3, B3)))
    else:
        m = tp_add(tp_mul(A2, C3), tp_scale(tp_mul(A3, C2), -1))
        ab = tp_add(tp_mul(A2, B3), tp_scale(tp_mul(A3, B2), -1))
        bc = tp_add(tp_mul(B2, C3), tp_scale(tp_mul(B3, C2), -1))
        resultant = tp_add(tp_mul(m, m), tp_scale(tp_mul(ab, bc), -1))

    def distance(point):
        return abs(float(other.value(point))) / max(float(other.gradient_norm(point)), tolerance)

    if tp_is_zero(resultant):
        return None if distance(curve.evaluate(.5)) <= tolerance else tuple(
            t for t in (0., 1.) if distance(curve.evaluate(t)) <= tolerance)
    angles = tp_roots(_strip(resultant), start=lo, sweep=hi - lo, tolerance=tol, cancellation_check=cancellation_check)
    parameters = []
    for angle in angles:
        t = parameter_for_angle(curve, angle)
        if distance(curve.evaluate(t)) <= tolerance and not any(abs(t - old) <= 64 * np.finfo(float).eps for old in parameters):
            parameters.append(t)
    return tuple(sorted(parameters))
