"""Immutable analytic curves used by surface intersections.

These definitions carry geometry, never a display polyline. Edge endpoint
vertices must agree with the definition; moving an endpoint alone is rejected
by topology validation. Whole-model transformations transform the definition.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from typing import NamedTuple
import math
import heapq
import numpy as np
from ._fixed_sums import weighted_sum

from .errors import GeometryError
from .surfaces import Cylinder
from ._cross3 import _cross3, _gauss_legendre
from .analytic_roots import trigonometric_roots


def _vector(value, name):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"{name} must be a finite 3-vector") from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise GeometryError(f"{name} must be a finite 3-vector")
    return tuple(float(item) for item in result)


def _parameters(value):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise GeometryError("curve parameters must be finite and in [0, 1]") from exc
    if not np.all(np.isfinite(result)) or np.any((result < 0) | (result > 1)):
        raise GeometryError("curve parameters must be finite and in [0, 1]")
    return result


def _angles(start, sweep):
    try:
        start, sweep = float(start), float(sweep)
    except (TypeError, ValueError) as exc:
        raise GeometryError("curve angles must be finite") from exc
    if not math.isfinite(start) or not math.isfinite(sweep) or sweep == 0:
        raise GeometryError("curve angles must be finite with a nonzero sweep")
    if abs(sweep) > math.tau + 16 * np.finfo(float).eps:
        raise GeometryError("one curve interval cannot exceed a full turn")
    return start, sweep


def _affine(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise GeometryError("curve transformation must be a finite 4x4 matrix")
    if not np.array_equal(matrix[3], (0, 0, 0, 1)) or np.linalg.det(matrix[:3, :3]) == 0:
        raise GeometryError("curve transformation must be nonsingular and affine")
    return matrix


def analytic_curve_length(curve, tolerance):
    """Adaptive Gaussian integration of the analytic derivative.

    Piecewise curves (NURBS) are integrated span by span: Gauss rules converge
    on each smooth span, not across the derivative's kinks at interior knots.
    """
    breaks = getattr(curve, "breakpoints", None)
    pieces = (0., *breaks(), 1.) if breaks is not None else (0., 1.)
    share = tolerance / (len(pieces) - 1)
    total = 0.
    for a, b in zip(pieces, pieces[1:]):
        previous = None
        count = 16
        while True:
            nodes, weights = _gauss_legendre(count)
            value = .5*(b-a)*weighted_sum(weights, np.linalg.norm(curve.derivative(a+.5*(b-a)*(nodes+1)), axis=-1))
            if previous is not None and abs(value-previous) <= share:
                total += value
                break
            if count >= 1024:
                raise GeometryError("analytic curve length did not converge to the requested tolerance")
            previous, count = value, count*2
    return total


def project_analytic_curve(curve, point, tolerance):
    """Global bounded projection; samples are upper bounds, never exclusions."""
    target = np.asarray(_vector(point, "projection point"))
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise GeometryError("projection tolerance must be finite and positive")
    if isinstance(curve, EllipticArc):
        return _project_ellipse(curve, target, tolerance)
    seeds = np.asarray((0., .5, 1.))
    points = curve.evaluate(seeds)
    distances = np.linalg.norm(points-target, axis=-1)
    index = int(np.argmin(distances))
    best, parameter, made = float(distances[index]), float(seeds[index]), points[index]

    def result():
        # Preserve exact branch stations after distance-tolerance subdivision.
        # A closer feasible point retains the global distance certificate already
        # established below; polishing cannot replace it with a worse candidate.
        if isinstance(curve, BezierQuadricCurve):
            candidate = curve._polish_point(target, parameter)
            point = curve.evaluate(candidate)
            residual = float(np.linalg.norm(point-target))
            if residual < best:
                return point.copy(), candidate, residual
        if hull_bound is not None or type(curve).__name__ == "CurveOnSurface":
            # NURBS / curves on NURBS: Newton on C'.(C - p), accepted only if closer
            current, point, residual = parameter, made, best
            for _ in range(8):
                values = curve.derivatives(current, 2)
                offset = values[0]-target
                slope = float(values[1] @ values[1] + offset @ values[2])
                if not math.isfinite(slope) or slope <= 0:
                    break
                candidate = float(np.clip(current-float(values[1] @ offset)/slope, 0., 1.))
                candidate_point = curve.evaluate(candidate)
                candidate_residual = float(np.linalg.norm(candidate_point-target))
                if not candidate_residual < residual:
                    break
                current, point, residual = candidate, candidate_point, candidate_residual
            return point.copy(), current, residual
        return made.copy(), parameter, best

    hull_bound = getattr(curve, "distance_lower_bound", None)

    def bound(a, b):
        if hull_bound is not None:                    # NURBS: control-hull bound (quadratic convergence)
            return hull_bound(a, b, target)
        lo, hi = curve.bounds(a, b)
        return float(np.linalg.norm(np.maximum(np.maximum(lo-target, target-hi), 0)))

    queue = [(bound(0., 1.), 0., 1.)]
    while queue:
        distance, a, b = heapq.heappop(queue)
        if best-distance <= tolerance:
            return result()
        mid = .5*(a+b)
        if mid == a or mid == b:
            raise GeometryError("analytic projection cannot resolve the requested tolerance")
        point = curve.evaluate(mid)
        residual = float(np.linalg.norm(point-target))
        if residual < best:
            best, parameter, made = residual, mid, point
        for start, end in ((a, mid), (mid, b)):
            lower_bound = bound(start, end)
            if lower_bound < best-tolerance:
                heapq.heappush(queue, (lower_bound, start, end))
    return result()


def _project_ellipse(curve, target, tolerance):
    """Global minimum from endpoints and every stationary angular root.

    The squared-distance derivative has two harmonics. Exact rational dot
    products preserve an identically zero derivative (for example projection
    from a circle's axis), rather than forcing box refinement of every point
    of a constant-distance arc. Sturm isolation supplies all stationary roots.
    """
    offset = tuple(Fraction(c)-Fraction(float(p))
                   for c, p in zip(curve.center, target))
    u = tuple(map(Fraction, curve.u_vector))
    v = tuple(map(Fraction, curve.v_vector))
    def dot(left, right):
        return sum(a*b for a, b in zip(left, right))
    coefficients = (Fraction(0), dot(offset, v), -dot(offset, u),
                    dot(u, v), (dot(v, v)-dot(u, u))/2)
    if not any(coefficients):
        parameters = np.asarray((0.,))
    else:
        angular_tolerance = min(1e-12, tolerance/(8*(
            np.linalg.norm(curve.u_vector)+np.linalg.norm(curve.v_vector))))
        angles = trigonometric_roots(coefficients, start=curve.start_angle,
                                    sweep=curve.sweep_angle,
                                    tolerance=angular_tolerance)
        parameters = np.asarray((0., 1., *(
            (angle-curve.start_angle)/curve.sweep_angle for angle in angles)))
        parameters = np.clip(parameters, 0., 1.)
    points = curve.evaluate(parameters)
    distances = np.linalg.norm(points-target, axis=-1)
    index = int(np.argmin(distances))
    return points[index].copy(), float(parameters[index]), float(distances[index])


@dataclass(frozen=True, slots=True)
class EllipticArc:
    """Exact affine ellipse: center + u*cos(angle) + v*sin(angle).

    ``u_vector`` and ``v_vector`` are independent, not necessarily orthogonal.
    Parameters run from zero to one over the signed angular interval.
    """
    center: tuple[float, float, float]
    u_vector: tuple[float, float, float]
    v_vector: tuple[float, float, float]
    start_angle: float = 0.0
    sweep_angle: float = math.tau

    def __post_init__(self):
        for name in ("center", "u_vector", "v_vector"):
            object.__setattr__(self, name, _vector(getattr(self, name), name))
        u, v = np.asarray(self.u_vector), np.asarray(self.v_vector)
        if np.linalg.norm(_cross3(u, v)) == 0:
            raise GeometryError("ellipse basis vectors must be independent")
        start, sweep = _angles(self.start_angle, self.sweep_angle)
        object.__setattr__(self, "start_angle", start)
        object.__setattr__(self, "sweep_angle", sweep)

    def evaluate(self, parameters):
        angle = self.start_angle + self.sweep_angle * _parameters(parameters)
        return (np.asarray(self.center) + np.cos(angle)[..., None] * self.u_vector
                + np.sin(angle)[..., None] * self.v_vector)

    def derivative(self, parameters):
        angle = self.start_angle + self.sweep_angle * _parameters(parameters)
        return self.sweep_angle * (-np.sin(angle)[..., None] * self.u_vector
                                  + np.cos(angle)[..., None] * self.v_vector)

    def bounds(self, lower=0.0, upper=1.0):
        values = _parameters((lower, upper))
        if values[0] > values[1]:
            raise GeometryError("curve bound interval is reversed")
        a, b = sorted(self.start_angle + self.sweep_angle * values)
        angles = [a, b]
        for u, v in zip(self.u_vector, self.v_vector):
            phase = math.atan2(v, u)
            angles.extend(phase + k * math.pi for k in range(
                math.ceil((a-phase)/math.pi), math.floor((b-phase)/math.pi)+1))
        angle = np.asarray(angles)
        points = (self.center + np.cos(angle)[:, None] * self.u_vector
                  + np.sin(angle)[:, None] * self.v_vector)
        margin = 16*np.finfo(float).eps*(np.abs(self.center)+np.abs(self.u_vector)+np.abs(self.v_vector))
        return (np.nextafter(points.min(axis=0)-margin, -np.inf),
                np.nextafter(points.max(axis=0)+margin, np.inf))

    def subcurve(self, lower, upper):
        lower, upper = _parameters((lower, upper))
        if lower == upper:
            raise GeometryError("curve interval must have positive length")
        return replace(self, start_angle=self.start_angle + lower*self.sweep_angle,
                       sweep_angle=(upper-lower)*self.sweep_angle)

    def transformed(self, matrix):
        matrix = _affine(matrix)
        return replace(self, center=matrix[:3, :3] @ self.center + matrix[:3, 3],
                       u_vector=matrix[:3, :3] @ self.u_vector,
                       v_vector=matrix[:3, :3] @ self.v_vector)


@dataclass(frozen=True, slots=True)
class _CylinderSupport:
    origin: tuple[float, float, float]
    axis: tuple[float, float, float]
    radial_direction: tuple[float, float, float]
    radius: float
    height: float
    start_angle: float
    sweep_angle: float

    @classmethod
    def from_surface(cls, surface):
        if isinstance(surface, cls):
            return surface
        if not isinstance(surface, Cylinder):
            raise GeometryError("intersection curve supports must be Cylinders")
        return cls(tuple(surface.origin), tuple(surface.axis),
                   tuple(surface.radial_direction), surface.radius, surface.height,
                   surface.start_angle, surface.sweep_angle)

    def surface(self):
        return Cylinder(**self.to_dict())

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class _BranchConstants(NamedTuple):
    """Angle-independent float terms of one branch chart (read-only arrays)."""
    axis: np.ndarray
    other_axis: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    w: np.ndarray
    a: float
    cross: np.ndarray
    n0: float
    nc: float
    ns: float
    magnitude: float
    remainder: float


# Keyed by the identities of the two immutable support descriptors. Each entry
# holds both objects, so their ids cannot be reused while the entry exists.
_BRANCH_CONSTANTS: dict = {}
_BRANCH_CONSTANTS_LIMIT = 4096


def _branch_constants(curve):
    """Terms of ``_coefficients``/``_root_at_angles``/``_root`` that do not depend on the angle.

    Computed once per support pair with exactly the expressions those methods
    evaluated on every call, so every value is bit-identical; subcurves share
    their support objects and therefore the entry. No tolerance or decision is
    involved, and nothing is computed that the uncached methods did not compute.
    """
    first, second = curve.first, curve.second
    key = (id(first), id(second))
    entry = _BRANCH_CONSTANTS.get(key)
    if entry is not None and entry[0] is first and entry[1] is second:
        return entry[2]
    axis = np.asarray(first.axis)
    other_axis = np.asarray(second.axis)
    e1 = np.asarray(first.radial_direction)
    e2 = np.cross(axis, e1)
    w = axis-float(axis @ other_axis)*other_axis
    a = float(w @ w)
    cross = np.cross(first.axis, second.axis)
    n0 = float((np.asarray(first.origin)-second.origin) @ cross)
    nc = float(first.radius*np.asarray(first.radial_direction) @ cross)
    ns = float(first.radius*np.cross(first.axis, first.radial_direction) @ cross)
    magnitude = math.hypot(nc, ns)
    remainder = math.fsum((float(cross @ cross)*second.radius**2, -magnitude**2, -n0*n0))
    for array in (axis, other_axis, e1, e2, w, cross):
        array.flags.writeable = False
    constants = _BranchConstants(axis, other_axis, e1, e2, w, a, cross, n0, nc, ns, magnitude, remainder)
    if len(_BRANCH_CONSTANTS) >= _BRANCH_CONSTANTS_LIMIT:
        _BRANCH_CONSTANTS.clear()
    _BRANCH_CONSTANTS[key] = (first, second, constants)
    return constants


class _DoubleNode(NamedTuple):
    """One exact double zero of a support pair's chart discriminant (a tangent contact node)."""
    sign: int            # fold sign s: the node solves n(theta) = s |a1 x a2| r2
    tau: int             # n0 - s |a1 x a2| r2 = -tau m exactly, so cos(theta* - phi) = tau there
    angle: float         # float node angle theta* (phi or phi + pi, phi = atan2(ns, nc))


# Exact node certificates per support pair, keyed like ``_BRANCH_CONSTANTS`` (same lifetime rule).
_DOUBLE_NODES: dict = {}


def _signum(value) -> int:
    return (value > 0)-(value < 0)


def _sign_of_sum_with_root(rational: Fraction, factor: Fraction, radicand: Fraction) -> int:
    """Exact sign of ``rational + factor*sqrt(radicand)`` (radicand >= 0)."""
    if factor == 0 or radicand == 0:
        return _signum(rational)
    root_sign = _signum(factor)
    if _signum(rational) in (0, root_sign):
        return _signum(rational) or root_sign
    left, right = rational*rational, factor*factor*radicand
    return _signum(rational) if left > right else root_sign if left < right else 0


def _fold_doubles(o1, a1, e1, e2, r1, o2, a2, r2):
    """Fold signs ``s`` whose fold equation is an exact double, with ``tau``, over the given rationals.

    ``n = n0 + nc cos + ns sin`` with ``c = a1 x a2``, ``n0 = (o1 - o2) . c``, ``nc = e1 . c``,
    ``ns = e2 . c`` (``e1``, ``e2`` already scaled by ``r1`` when ``r1`` is None). The fold
    ``n = s |c| r2`` is a double root exactly when ``nc^2 + ns^2 = (n0 - s |c| r2)^2``; |c| is
    irrational in general, so the equality is decided by comparing squares.
    """
    cross = lambda u, v: [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    dot = lambda u, v: sum(x*y for x, y in zip(u, v))
    c = cross(a1, a2)
    scale = 1 if r1 is None else r1
    n0 = dot([x-y for x, y in zip(o1, o2)], c)
    nc, ns = scale*dot(e1, c), scale*dot(e2, c)
    m2, c2 = nc*nc+ns*ns, dot(c, c)
    result = {}
    if m2 == 0 or c2 == 0:
        return result
    for sign in (-1, 1):
        # nc^2 + ns^2 - (n0 - s|c| r2)^2 = (m2 - n0^2 - c2 r2^2) + 2 s n0 r2 |c|
        if _sign_of_sum_with_root(m2-n0*n0-c2*r2*r2, 2*sign*n0*r2, c2) == 0:
            # n0 - s|c| r2 = -tau m: tau is minus its exact sign (nonzero since m > 0).
            result[sign] = -_sign_of_sum_with_root(n0, -sign*r2, c2)
    return result


def _half_ulp(value):
    """Half an ulp of a float as an exact rational: ``|fl(t) - t| <= _half_ulp(fl(t))`` under round to nearest, in the
    normal and the subnormal range (``np.spacing(0)`` is the smallest subnormal)."""
    return Fraction(float(np.spacing(abs(float(value)))))/2


#: Unit roundoff and the classical dot-product constant gamma_3 = 3u / (1 - 3u) (any summation order, FMA or not).
_UNIT = Fraction(1, 2**53)
_GAMMA3 = 3*_UNIT/(1-3*_UNIT)
#: Absolute underflow allowance of a 3-term dot product: 3 products and 2 sums, each at most half the smallest
#: subnormal (2^-1075) beyond the relative gamma_3 bound.
_DOT_UNDERFLOW = Fraction(5, 2**1075)
#: Classification floor |a1 x a2|^2 >= 2^-60 (axes at least about 1e-9 rad apart). Nearer-parallel pairs are never
#: classified (main's path, with its own near-parallel handling); the contact point's axial solve divides by it.
_FOLD_CROSS_FLOOR = 2.**-60


def _fold_coefficient_bound(first, second, sign):
    """Rigorous upper bound ``B`` (an exact rational) of ``|D(fl coefficients) - D(exact coefficients)|`` for the
    fold ``sign``, or None when an input is not finite.

    ``D = nc^2 + ns^2 - c0^2`` with ``c0 = n0 - s |a1 x a2| r2`` is the discriminant of the fold equation
    ``c0 + nc cos + ns sin = 0``. The float coefficients are those ``analytic_supports.cylinder_cylinder_support``
    forms (``cross = a1 x a2``, ``magnitude = |cross|``, ``n0 = (o1 - o2) . cross``, ``nc = r1 (e1 . cross)``,
    ``ns = r1 ((a1 x e1) . cross)``); the exact coefficients are those of the stored doubles (``e2 = a1 x e1``
    exactly). A forward running-error bound over the kernel's own float operations: every product, difference and
    square root contributes half an ulp of its float result unless it is exact (checked exactly), a 3-term dot
    product ``gamma_3`` times its absolute terms plus an underflow allowance (any summation order, FMA or not), the
    norm (numpy: ``sqrt(c . c)``) ``gamma_3`` relative plus half an ulp plus an underflow allowance. All error
    propagation (first and second order) is exact rational arithmetic, so the bound itself never rounds or
    underflows.
    """
    Q = Fraction
    h = _half_ulp
    a1, a2 = np.asarray(first.axis, dtype=float), np.asarray(second.axis, dtype=float)
    e1 = np.asarray(first.radial_direction, dtype=float)
    o1, o2 = np.asarray(first.origin, dtype=float), np.asarray(second.origin, dtype=float)
    r1, r2 = float(first.radius), float(second.radius)
    if not all(math.isfinite(float(value)) for value in (*a1, *a2, *e1, *o1, *o2, r1, r2)):
        return None

    def product(x, y):
        z = float(x)*float(y)
        return z, (Q(0) if Q(z) == Q(float(x))*Q(float(y)) else h(z))

    def difference(x, y):
        z = float(x)-float(y)
        return z, (Q(0) if Q(z) == Q(float(x))-Q(float(y)) else h(z))

    def cross(u, v):
        values, errors = [], []
        for j, k in ((1, 2), (2, 0), (0, 1)):
            (p, ep_), (q, eq_) = product(u[j], v[k]), product(u[k], v[j])
            z, ez = difference(p, q)
            values.append(z)
            errors.append(ep_+eq_+ez)
        return np.asarray(values), errors

    def dot(x, ex, y, ey):
        # |fl(x . y) - (x_exact . y_exact)| for |x - x_exact| <= ex, |y - y_exact| <= ey componentwise.
        total = _DOT_UNDERFLOW
        for xi, exi, yi, eyi in zip(x, ex, y, ey):
            ax, ay = abs(Q(float(xi))), abs(Q(float(yi)))
            total += _GAMMA3*ax*ay+ax*eyi+ay*exi+exi*eyi
        return total

    c, ec = cross(a1, a2)
    e2, ee2 = cross(a1, e1)
    magnitude = float(np.linalg.norm(c))
    # fl(sqrt(fl(c . c))): |fl(c.c) - |c_fl|^2| <= gamma_3 |c_fl|^2 + underflow, so the root is within gamma_3 / 2
    # relative (second order included in gamma_3) plus sqrt(underflow) < 2^-536, and the rounded root within half
    # an ulp more; then | |c_fl| - |c_exact| | <= ||c_fl - c_exact||_1.
    em = _GAMMA3*(Q(magnitude)+h(magnitude))+h(magnitude)+Q(1, 2**536)+sum(ec)
    pairs = [difference(x, y) for x, y in zip(o1, o2)]
    d = [value for value, _ in pairs]
    ed = [error for _, error in pairs]
    n0 = float(np.asarray(d) @ c)
    en0 = dot(d, ed, c, ec)
    t_c, t_s = float(e1 @ c), float(e2 @ c)
    et_c, et_s = dot(e1, [Q(0)]*3, c, ec), dot(e2, ee2, c, ec)
    (nc, rc), (ns, rs) = product(r1, t_c), product(r1, t_s)
    enc, ens = abs(Q(r1))*et_c+rc, abs(Q(r1))*et_s+rs
    reach, er = product(magnitude, r2)
    ep = abs(Q(r2))*em+er
    c0, e_sub = difference(n0, sign*reach)
    ec0 = en0+ep+e_sub
    a = lambda value: abs(Q(float(value)))
    return (2*a(nc)+enc)*enc+(2*a(ns)+ens)*ens+(2*a(c0)+ec0)*ec0


def _fold_undecidable(first, second):
    """``{fold sign: tau}`` of the folds whose exact discriminant (stored doubles) is nonzero but no larger than
    :func:`_fold_coefficient_bound`: the kernel's float fold coefficients cannot decide whether the fold is two
    simple folds, a double contact or no contact (the F550 class). ``tau`` is minus the exact sign of
    ``n0 - s |c| r2``, as for an exact double. Exact doubles (:func:`_fold_doubles`) are not included.

    Classified only inside the K3 magnitude range (L = max(|origin coordinates|, r1, r2) <= 2^64, min(r1, r2) >=
    2^-64, L <= 2^64 min(r1, r2); no overflow) and for |a1 x a2|^2 >= ``_FOLD_CROSS_FLOOR``; outside, the result is
    empty (main's path).
    """
    lengths = [abs(float(value)) for value in (*np.asarray(first.origin, dtype=float),
                                               *np.asarray(second.origin, dtype=float), first.radius, second.radius)]
    smallest = min(abs(float(first.radius)), abs(float(second.radius)))
    if not (all(math.isfinite(value) for value in lengths) and smallest >= 2.**-64
            and max(lengths) <= 2.**64 and max(lengths) <= 2.**64*smallest):
        return {}
    q = lambda vector: [Fraction(float(value)) for value in np.asarray(vector, dtype=float)]
    cross = lambda u, v: [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    dot = lambda u, v: sum(x*y for x, y in zip(u, v))
    a1, a2 = np.asarray(first.axis, dtype=float), np.asarray(second.axis, dtype=float)
    c = np.cross(a1, a2)
    if not float(c @ c) >= _FOLD_CROSS_FLOOR:
        return {}
    magnitude = float(np.linalg.norm(c))
    e1 = np.asarray(first.radial_direction, dtype=float)
    r1, r2 = float(first.radius), float(second.radius)
    n0 = float((np.asarray(first.origin, dtype=float)-np.asarray(second.origin, dtype=float)) @ c)
    nc, ns = r1*float(e1 @ c), r1*float(np.cross(a1, e1) @ c)
    result = {}
    exact = None
    for sign in (-1, 1):
        bound = _fold_coefficient_bound(first, second, sign)
        if bound is None:
            continue
        c0 = n0-sign*magnitude*r2
        # Rigorous prefilter in exact arithmetic on the float coefficients: |D(fl) - D(exact)| <= bound, so
        # |D(fl coefficients)| > 2 bound implies |D(exact)| > bound (decided) without forming the exact D.
        floated = Fraction(nc)**2+Fraction(ns)**2-Fraction(c0)**2
        if abs(floated) > 2*bound:
            continue
        if exact is None:
            A1, E1 = q(a1), q(e1)
            C = cross(A1, q(a2))
            E2 = cross(A1, E1)
            N0 = dot([x-y for x, y in zip(q(first.origin), q(second.origin))], C)
            R1, R2 = Fraction(r1), Fraction(r2)
            NC, NS, C2 = R1*dot(E1, C), R1*dot(E2, C), dot(C, C)
            exact = (N0, NC*NC+NS*NS-N0*N0-C2*R2*R2, C2, R2)
        N0, rational, C2, R2 = exact
        if C2 == 0:
            return {}
        factor = 2*sign*N0*R2
        # exact D = rational + factor sqrt(C2); undecidable iff 0 < |D| <= bound
        if _sign_of_sum_with_root(rational, factor, C2) == 0:
            continue
        if (_sign_of_sum_with_root(rational-bound, factor, C2) <= 0
                <= _sign_of_sum_with_root(rational+bound, factor, C2)):
            tau = -_sign_of_sum_with_root(N0, -sign*R2, C2)
            if tau:
                result[sign] = tau
    return result


def _trim(values):
    values = list(values)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return tuple(values)


#: Largest separation (in the half-angle chart t = tan(theta / 2), or 1 / t) the elimination chart's
#: discriminant roots may have from a certified node, and the window searched around it (about 0.25 rad:
#: every root in it must lie in the small ball, and at least one must).
_NODE_CHART_AGREEMENT = Fraction(1, 2**44)
_NODE_CHART_WINDOW = Fraction(1, 8)


def _elimination_agrees(curve, angles, others=()):
    """True when the rational elimination chart (``cylinder_curve_events._quadratic``, identity transform)
    has a root of its discriminant ``b^2 - 4 a c`` within ``_NODE_CHART_AGREEMENT`` of each node and no
    other root in the window ``_NODE_CHART_WINDOW`` around it (exact isolation).

    That chart rationalizes float-projected vectors, so its node can differ from the certified one by
    rounding; this bounds the difference rigorously. An empty window (the chart's fold missing or moved
    away) is a disagreement, not agreement."""
    from .analytic_roots import isolate_real_roots
    from .cylinder_curve_events import _add, _multiply, _quadratic, _scale
    identity = ((1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.))
    chart = type("_Chart", (), {"first": curve.first, "transform": identity})   # no branch construction here
    a, b, c = _quadratic(chart, curve.second, identity)
    discriminant = _trim(_add(_multiply(b, b), _scale(c, -4*a)))
    if not any(discriminant):
        return False
    discriminant = tuple(discriminant)+(Fraction(0),)*(5-len(discriminant))   # degree-4 projective chart
    for angle in angles:
        half = math.remainder(angle, math.tau)/2
        if abs(half) <= math.pi/4:
            polynomial, centre = discriminant, Fraction(math.tan(half))
        else:
            polynomial, centre = tuple(reversed(discriminant)), Fraction(1/math.tan(half))
        polynomial = _trim(polynomial)
        if len(polynomial) == 1:
            return False                      # no root at all: the chart has no fold at this node
        window = _NODE_CHART_WINDOW
        for other in others:
            # The other fold's own (simple) roots are expected elsewhere: keep them out of the window.
            other_half = math.remainder(other, math.tau)/2
            if abs(half) <= math.pi/4:
                if abs(other_half) >= math.pi/2-1e-12:
                    continue
                coordinate = math.tan(other_half)
            else:
                if other_half == 0:
                    continue
                coordinate = 1/math.tan(other_half)
            distance = abs(Fraction(coordinate)-centre)
            window = min(window, distance/2)
        if window <= 4*_NODE_CHART_AGREEMENT:
            return False
        lower, upper = centre-window, centre+window
        roots = isolate_real_roots(polynomial, tolerance=2.**-60, interval=(lower, upper))
        if not roots:
            return False
        for root in roots:
            if root.lower < centre-_NODE_CHART_AGREEMENT or root.upper > centre+_NODE_CHART_AGREEMENT:
                return False
    return True


def _double_nodes(curve):
    """Certified exact double zeros of the chart discriminant of ``curve``'s support pair (tangent nodes).

    The kernel's discriminant is ``D(theta) = 4 (|c|^2 r2^2 - n(theta)^2)`` (see ``_root_at_angles``).
    A fold sign is a node only when its fold equation is an exact double root in every model the
    kernel evaluates it in: (1) the exact values of the stored doubles, (2) the rounded chart vectors
    ``_branch_constants`` uses (``fl(r1 e1)``, ``fl(r1 (a1 x e1))``, ``fl(a1 x a2)``), and (3) the rational
    elimination chart of ``cylinder_curve_events._quadratic`` places its discriminant roots within
    ``_NODE_CHART_AGREEMENT`` of the node (:func:`_elimination_agrees`). Any disagreement (an O(eps) model
    difference can split a node by O(sqrt(eps))) certifies no node and keeps the original path. No tolerance is involved: a near miss (the F550 kind) is not a node.
    Returns a tuple of :class:`_DoubleNode` (empty for every ordinary pair).
    """
    first, second = curve.first, curve.second
    key = (id(first), id(second))
    entry = _DOUBLE_NODES.get(key)
    if entry is not None and entry[0] is first and entry[1] is second:
        return entry[2]
    if not _angles_admitted(first.start_angle, first.sweep_angle):
        # Decided on the first support's own angular range, so the support's events and every chart cut
        # from it agree; such pairs keep the original expressions everywhere (not cached: no work to save).
        return ()
    if not _node_conditioned(curve):
        # An ill-conditioned axial solve (nearly parallel axes) or extreme magnitudes: main's path, unchanged.
        return ()
    q = lambda vector: [Fraction(float(value)) for value in np.asarray(vector, dtype=float)]
    cross = lambda u, v: [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
    a1 = q(first.axis)
    exact = _fold_doubles(q(first.origin), a1, q(first.radial_direction), cross(a1, q(first.radial_direction)),
                          Fraction(float(first.radius)), q(second.origin), q(second.axis),
                          Fraction(float(second.radius)))
    nodes = []
    if exact:
        constants = _branch_constants(curve)
        r1 = float(first.radius)
        rounded = _fold_doubles(q(first.origin), a1, q(r1*np.asarray(first.radial_direction)),
                                q(r1*np.asarray(constants.e2)), None, q(second.origin), q(second.axis),
                                Fraction(float(second.radius)))
        signs = [sign for sign in (-1, 1) if sign in exact and rounded.get(sign) == exact[sign]]
        phi = math.atan2(constants.ns, constants.nc)
        candidates = [_DoubleNode(sign, exact[sign], phi if exact[sign] > 0 else phi+math.pi) for sign in signs]
        others = []
        for sign in (-1, 1):
            if sign not in exact:
                # Predicted simple roots of the other fold (float witnesses; only used to size the window).
                reach = math.sqrt(float(constants.cross @ constants.cross))*float(second.radius)
                others.extend(trigonometric_roots((constants.n0-sign*reach, constants.nc, constants.ns, 0., 0.),
                                                  start=-math.pi, sweep=2*math.pi,
                                                  tolerance=4*np.finfo(float).eps))
        if candidates and len(signs) == len(exact) and _elimination_agrees(
                curve, [node.angle for node in candidates], others):
            nodes = candidates
    nodes = tuple(nodes)
    if len(_DOUBLE_NODES) >= _BRANCH_CONSTANTS_LIMIT:
        _DOUBLE_NODES.clear()
    _DOUBLE_NODES[key] = (first, second, nodes)
    return nodes


#: 2 pi to 100 digits, for exact argument reduction of node angle differences outside the float range below.
_TWO_PI = Fraction(Decimal("6.283185307179586476925286766559005768394338798750211641949889184615632812572417997256069650684234136"))
#: Within this angle range the float difference ``angle - theta* - 2 pi k`` (k <= 4) errs by a few ulp of 2 pi,
#: inside every slack used below; beyond it the reduction is done exactly.
_FLOAT_REDUCTION_LIMIT = 8*math.pi


#: Node treatment is admitted only for supports (and charts) whose angles stay within this many radians
#: (2^10 turns): image enumeration stays small and the 100-digit period reduces exactly there.
_NODE_ADMISSION_LIMIT = 2**10*2*math.pi


def _angles_admitted(start, sweep):
    start, sweep = float(start), float(sweep)
    return (math.isfinite(start) and math.isfinite(sweep)
            and max(abs(start), abs(start+sweep)) <= _NODE_ADMISSION_LIMIT)


#: Node treatment needs a well-conditioned axial solve. evaluate() solves z = (-b + root)/(2a) with
#: a = |w|^2 = sin^2 of the angle between the axes; its rounding grows like eps/a (measured: 0.1 to 0.4 eps/a
#: for sin from 0.84 down to 1e-7). A node is certified to _NODE_CHART_AGREEMENT = 2^-44, so admit only pairs
#: with eps/a <= 2^-44, i.e. a >= 2^-8 (axes at least about 3.6 degrees apart). Pairs below keep main's path.
_NODE_CONDITION_LIMIT = 2.0**-8
#: Magnitude range of node treatment, deliberately conservative (real models sit far inside it). With
#: L = max(|origin coordinates|, r1, r2) <= 2^64, the smaller radius s >= 2^-64 and L <= 2^64 s, every term of
#: the axial solve, the jets and the bounds (at most a fourth power of these lengths, such as the ellipse's
#: cross-product norm, b^2 and 4ac, times constants up to about 2^10) has a length-dependent factor within
#: 2^+-266. This bounds only the length-dependent coefficients: a value that is also small through its angle
#: (a discriminant near its zero, for instance) can still underflow. Pairs outside the range decline node
#: treatment and keep main's path.
_NODE_MAGNITUDE_LIMIT = 2.0**64


def _node_conditioned(curve):
    """Whether ``curve``'s support pair is inside the node treatment's conditioning and magnitude range."""
    first, second = curve.first, curve.second
    lengths = [abs(float(value)) for value in (*np.asarray(first.origin, dtype=float),
                                               *np.asarray(second.origin, dtype=float),
                                               first.radius, second.radius)]
    if not all(math.isfinite(value) for value in lengths):
        return False
    largest, smallest = max(lengths), min(abs(float(first.radius)), abs(float(second.radius)))
    if not (smallest >= 1/_NODE_MAGNITUDE_LIMIT and largest <= _NODE_MAGNITUDE_LIMIT
            and largest <= _NODE_MAGNITUDE_LIMIT*smallest):
        return False
    return _branch_constants(curve).a >= _NODE_CONDITION_LIMIT


def _node_psi(angle, node_angle):
    """``angle - node_angle`` reduced to about ``[-pi, pi]`` with an ulp-level error (the radial coordinates'
    ``cos``/``sin`` reduce their arguments exactly, so the node factor must too). Each element is reduced by
    the same rule whatever array it comes in (float within ``_FLOAT_REDUCTION_LIMIT``, exact beyond)."""
    angle = np.asarray(angle, dtype=float)
    psi = angle-node_angle
    psi = psi-2*math.pi*np.round(psi/(2*math.pi))
    far = ~(np.abs(angle) <= _FLOAT_REDUCTION_LIMIT) if abs(node_angle) <= _FLOAT_REDUCTION_LIMIT else (
        np.ones(angle.shape, dtype=bool))
    if np.any(far):
        psi = np.array(psi, dtype=float, order="C", copy=True)     # C order: reshape(-1) is a view
        flat, values, mask = psi.reshape(-1), np.ravel(angle), np.ravel(far)
        assert np.shares_memory(flat, psi)
        centre = Fraction(float(node_angle))
        for index in np.flatnonzero(mask):
            value = float(values[index])
            if not math.isfinite(value):
                flat[index] = math.nan
                continue
            difference = Fraction(value)-centre
            turns = round(difference/_TWO_PI)
            flat[index] = float(difference-turns*_TWO_PI)
    return psi


def _node_treatment(curve):
    """The certified nodes a chart's evaluation, bounds, jets and constructor checks use (one decision for all
    of them): :func:`_double_nodes`, except for a square chart anchored at a node (outside the square-chart
    composition; it keeps the original expressions everywhere). Ordinary pairs: always empty."""
    nodes = _double_nodes(curve)
    if not nodes or not _angles_admitted(curve.start_angle, curve.sweep_angle):
        return ()
    if curve.parameterization == "linear":
        return nodes
    anchors = {"left_square": (curve.start_angle,), "right_square": (curve.start_angle+curve.sweep_angle,),
               "both_sine": (curve.start_angle, curve.start_angle+curve.sweep_angle)}[curve.parameterization]
    envelope = 64*np.finfo(float).eps*max(1., abs(curve.start_angle), abs(curve.start_angle+curve.sweep_angle))
    for node in nodes:
        for anchor in anchors:
            if abs(float(_node_psi(anchor, node.angle))) <= envelope:
                return ()
    return nodes


def _node_angle_images(angle, start, sweep):
    """The images ``angle + 2 pi k`` in the closed angular interval ``start .. start + sweep``,
    widened by the endpoint envelope the branch charts use (64 eps of the angles)."""
    if not _angles_admitted(start, sweep):
        return []                     # never enumerated: such charts carry no node treatment
    lower, upper = sorted((start, start+sweep))
    envelope = 64*np.finfo(float).eps*max(1., abs(start), abs(start+sweep))
    first = math.ceil((lower-envelope-angle)/math.tau)
    last = math.floor((upper+envelope-angle)/math.tau)
    images = []
    for k in range(first-1, last+2):
        # exact image (2 pi to 100 digits), rounded once; then the envelope test on the rounded value
        image = float(Fraction(float(angle))+k*_TWO_PI) if k else float(angle)
        if lower-envelope <= image <= upper+envelope:
            # An image within the envelope outside the chart is the chart's own end (clamped): it never moves it.
            images.append(min(upper, max(lower, image)))
    return images


def _node_half(curve, node, angle):
    """``(sin(psi/2), cos(psi/2), sin psi, cos psi)`` with ``psi = angle - theta*`` for a certified node.

    ``sin psi = tau sine`` and ``cos psi = tau cosine`` with the chart's rotated ``sine``/``cosine`` of
    ``_root_at_angles`` (``theta* = phi`` or ``phi + pi``), so no rounded node angle enters next to the node:
    there (``cos psi >= 0``) ``cos(psi/2) = sqrt((1 + cos psi) / 2)`` and ``|sin(psi/2)| = |sin psi| / (2 cos(psi/2))``,
    signed by the chart's witness ``psi`` (0 exactly at the witness).
    Away from it the exactly reduced ``psi`` (:func:`_node_psi`) is used.
    """
    constants = _branch_constants(curve)
    nc, ns, m = constants.nc, constants.ns, constants.magnitude
    angle = np.asarray(angle, dtype=float)
    cosine = (nc*np.cos(angle)+ns*np.sin(angle))/m
    sine = (nc*np.sin(angle)-ns*np.cos(angle))/m
    cos_psi, sin_psi = node.tau*cosine, node.tau*sine
    near = cos_psi >= 0
    psi = _node_psi(angle, node.angle)
    c_near = np.sqrt(np.maximum(.5*(1+cos_psi), .5))
    # Magnitude from the rotated sine (no rounded node angle); side from the chart's node witness psi (the
    # chart ends at the float witness, so an angle that rounds onto it is the node: exactly 0).
    s = np.where(near, np.copysign(np.abs(sin_psi)/(2*c_near), psi), np.sin(.5*psi))
    s = np.where(psi == 0, 0., s)
    c = np.where(near, c_near, np.cos(.5*psi))
    return s, c, sin_psi, cos_psi


def _node_discriminant(curve, nodes, angle):
    """``D(theta)`` of an exact-double support pair without cancellation at its nodes.

    ``D = -4 F(+1) F(-1)`` with ``F(s) = n - s |c| r2`` exactly. For a node's fold sign,
    ``F(s) = -tau m (1 - tau cos(theta - phi)) = -2 tau m sin^2(psi / 2)``, ``psi = theta - theta*``
    (cos(theta* - phi) = tau): a product of accurate factors, so D has a small relative error and its
    square root an absolute error O(eps * scale) up to the node, where ``b^2 - 4ac`` gives
    O(sqrt(eps) * scale). The other fold factor is formed directly.
    """
    constants = _branch_constants(curve)
    n0, nc, ns, m = constants.n0, constants.nc, constants.ns, constants.magnitude
    reach = math.sqrt(float(constants.cross @ constants.cross))*curve.second.radius
    by_sign = {node.sign: node for node in nodes}
    factors = []
    for sign in (1, -1):
        node = by_sign.get(sign)
        if node is None:
            factors.append((n0-sign*reach)+nc*np.cos(angle)+ns*np.sin(angle))
            continue
        half = _node_half(curve, node, angle)[0]
        factors.append(-2*node.tau*m*half*half)
    return -4*factors[0]*factors[1]


def _node_axial_jets(curve, nodes, angle, db, ddb, a):
    """First and second angle derivatives of the axial root on a certified node pair, without the 0/0
    quotients ``-(db z + dc) / root`` (and its second-order analogue) next to a node.

    With the node factor ``F = -2 tau m s^2`` (``s, c = sin, cos(psi / 2)``) and the other fold factor ``G``,
    ``D = -4 F G = 8 tau m s^2 G``, so ``sqrt(D) = |s| H`` with ``H = sqrt(8 tau m G)``. Then
    ``(sqrt D)' = sgn(s) (c / 2) H + |s| H'`` and ``(sqrt D)'' = -|s| H / 4 + sgn(s) c H' + |s| H''`` with
    ``H' = 4 tau m G' / H`` and ``H'' = (4 tau m G'' - H'^2) / H``; ``z = (-b + branch sqrt(D)) / (2 a)``.
    Used only where the node factor is the smaller one (``|F| <= |G|``), ``s != 0`` and ``tau G > 0``;
    elsewhere (``usable`` False) the caller keeps its own expressions, which are accurate there.
    Returns ``(usable, dz, ddz)`` per angle.
    """
    constants = _branch_constants(curve)
    n0, nc, ns, m = constants.n0, constants.nc, constants.ns, constants.magnitude
    reach = math.sqrt(float(constants.cross @ constants.cross))*curve.second.radius
    angle = np.asarray(angle, dtype=float)
    by_sign = {node.sign: node for node in nodes}

    halves = {sign: _node_half(curve, node, angle) for sign, node in by_sign.items()}
    usable_all = np.zeros(angle.shape, dtype=bool)
    dz_all = np.zeros(angle.shape)
    ddz_all = np.zeros(angle.shape)
    for sign, node in by_sign.items():
        s, c, _sin_psi, _cos_psi = halves[sign]
        node_factor = 2*m*s*s
        other = -sign
        if other in by_sign:
            half, _c, other_sin, other_cos = halves[other]
            g = -2*by_sign[other].tau*m*half*half
            dg = -by_sign[other].tau*m*other_sin
            ddg = -by_sign[other].tau*m*other_cos
        else:
            g = (n0-other*reach)+nc*np.cos(angle)+ns*np.sin(angle)
            dg = -nc*np.sin(angle)+ns*np.cos(angle)
            ddg = -nc*np.cos(angle)-ns*np.sin(angle)
        scale = 8*node.tau*m*g
        usable = (node_factor <= np.abs(g)) & (s != 0) & (scale > 0) & ~usable_all
        h = np.sqrt(np.where(usable, scale, 1.))
        dh = 4*node.tau*m*dg/h
        ddh = (4*node.tau*m*ddg-dh*dh)/h
        sign_s, abs_s = np.sign(s), np.abs(s)
        droot = sign_s*.5*c*h+abs_s*dh
        ddroot = -.25*abs_s*h+sign_s*c*dh+abs_s*ddh
        dz_all = np.where(usable, (-db+curve.branch*droot)/(2*a), dz_all)
        ddz_all = np.where(usable, (-ddb+curve.branch*ddroot)/(2*a), ddz_all)
        usable_all = usable_all | usable
    return usable_all, dz_all, ddz_all


def _node_discriminant_upper(curve, nodes, start, sweep):
    """Outward upper bound of ``D`` over the angles ``start .. start + sweep`` of an exact-double pair.

    ``|D| = 4 |F(+1)| |F(-1)|`` (see :func:`_node_discriminant`). A node factor is bounded by
    ``2 m sin^2(X / 2) <= m X^2 / 2`` with ``X`` the largest ``|psi|`` over the interval plus ``slack``,
    which covers the float node angle (atan2 of the rounded ``nc``, ``ns``: their rounding is bounded by
    ``8 eps`` times the absolute terms of their dot products, divided by ``m``) and the rounding of
    ``psi``. A non-node factor is bounded by ``|n0 - s |c| r2| + m``. A relative ``64 eps`` and the
    evaluation route's rounding are added on top. Returns ``inf`` when no useful bound applies.
    """
    constants = _branch_constants(curve)
    n0, nc, ns, m = constants.n0, constants.nc, constants.ns, constants.magnitude
    if not (m > 0 and math.isfinite(start) and math.isfinite(sweep)):
        return math.inf
    eps = np.finfo(float).eps
    cross = np.asarray(constants.cross)
    c_abs = float(np.sum(np.abs(cross)))
    r1 = abs(float(curve.first.radius))
    e1, e2 = np.abs(np.asarray(constants.e1)), np.abs(np.asarray(constants.e2))
    term = r1*c_abs*(float(np.max(e1))+float(np.max(e2)))*3
    reach = math.sqrt(float(cross @ cross))*curve.second.radius
    by_sign = {node.sign: node for node in nodes}
    bound = 4.
    for sign in (1, -1):
        node = by_sign.get(sign)
        if node is None:
            bound *= (abs(n0-sign*reach)+m)*(1+64*eps)+64*eps*(abs(n0)+reach+m)
            continue
        slack = 64*eps*term/m+64*eps*(1+abs(node.angle)+abs(start)+abs(sweep))
        psi = float(_node_psi(start, node.angle))
        reach_psi = max(abs(psi), abs(psi+sweep))+slack
        # |F| = 2 m sin^2(psi / 2) <= min(2 m, m psi^2 / 2) over the interval.
        bound *= min(2., .5*reach_psi*reach_psi)*m*(1+64*eps)
    return bound*(1+64*eps)


@dataclass(frozen=True, slots=True)
class CylinderIntersectionCurve:
    """One analytic quadratic-root branch of two cylinder supports.

    The first support supplies angle; the axial coordinate is the selected
    exact root of the second cylinder equation. Parallel supports are handled
    as generators/coincident regions, not by this nonparallel branch type.
    ``parameterization`` regularizes a simple discriminant root at an endpoint.
    An affine image is retained exactly through ``transform``.
    """
    first: Cylinder | _CylinderSupport
    second: Cylinder | _CylinderSupport
    start_angle: float
    sweep_angle: float
    branch: int
    parameterization: str = "linear"
    transform: tuple[tuple[float, ...], ...] = (
        (1., 0., 0., 0.), (0., 1., 0., 0.),
        (0., 0., 1., 0.), (0., 0., 0., 1.))

    def __post_init__(self):
        object.__setattr__(self, "first", _CylinderSupport.from_surface(self.first))
        object.__setattr__(self, "second", _CylinderSupport.from_surface(self.second))
        start, sweep = _angles(self.start_angle, self.sweep_angle)
        object.__setattr__(self, "start_angle", start)
        object.__setattr__(self, "sweep_angle", sweep)
        if type(self.branch) is not int or self.branch not in (-1, 1):
            raise GeometryError("intersection branch must be -1 or 1")
        if self.parameterization not in ("linear", "left_square", "right_square", "both_sine"):
            raise GeometryError("invalid intersection branch parameterization")
        transform = _affine(self.transform)
        object.__setattr__(self, "transform", tuple(tuple(row) for row in transform))
        if np.linalg.norm(np.cross(self.first.axis, self.second.axis)) == 0:
            raise GeometryError("parallel cylinders require generator or region intersections")
        # A public branch chart must stay real over its entire interval.
        # Inspect every stationary value of its degree-two trigonometric
        # discriminant, not a set of witness samples.
        first, second = self.first, self.second
        project = np.eye(3)-np.outer(second.axis, second.axis)
        q0 = project @ (np.asarray(first.origin)-second.origin)
        qc = project @ (first.radius*np.asarray(first.radial_direction))
        qs = project @ (first.radius*np.cross(first.axis, first.radial_direction))
        w = project @ np.asarray(first.axis)
        a = float(w @ w)
        b0, bc, bs = 2*float(q0 @ w), 2*float(qc @ w), 2*float(qs @ w)
        cross=np.cross(first.axis,second.axis)
        magnitude=float(np.linalg.norm(cross))
        n0=float((np.asarray(first.origin)-second.origin) @ cross)
        nc=first.radius*float(np.asarray(first.radial_direction) @ cross)
        ns=first.radius*float(np.cross(first.axis,first.radial_direction) @ cross)
        c0=4*magnitude*magnitude*second.radius**2-4*n0*n0-2*(nc*nc+ns*ns)
        c1,c2,c3,c4=-8*n0*nc,-8*n0*ns,-2*(nc*nc-ns*ns),-4*nc*ns
        transitions=[]
        nodes = {node.sign: node for node in _node_treatment(self)}
        for sign in (-1,1):
            if sign in nodes:
                # A certified double fold: its event is the node angle itself, not float roots of a
                # double root (which can be off by sqrt(eps), split in two or vanish).
                transitions.extend(_node_angle_images(nodes[sign].angle, start, sweep))
                continue
            transitions.extend(trigonometric_roots((n0-sign*magnitude*second.radius,nc,ns,0.,0.),
                start=start,sweep=sweep,tolerance=4*np.finfo(float).eps))
        stationary = ()
        if any(value != 0 for value in (c1, c2, c3, c4)):
            stationary = trigonometric_roots((0., c2, -c1, 2*c4, -2*c3),
                start=start, sweep=sweep, tolerance=4*np.finfo(float).eps)
        self._root_at_angles(np.asarray((start, start+sweep, *stationary)))
        endpoint_envelope=64*np.finfo(float).eps*max(1.,abs(start),abs(start+sweep))
        if any(min(abs(angle-start),abs(angle-start-sweep))>endpoint_envelope
               for angle in transitions):
            raise GeometryError("intersection branch chart must split at every discriminant transition")


    def _angle(self, parameters):
        t = _parameters(parameters)
        if self.parameterization == "left_square":
            value, derivative = t*t, 2*t
        elif self.parameterization == "right_square":
            value, derivative = 1-(1-t)**2, 2*(1-t)
        elif self.parameterization == "both_sine":
            value = np.sin(.5*np.pi*t)**2
            derivative = .5*np.pi*np.sin(np.pi*t)
        else:
            value, derivative = t, np.ones_like(t)
        return self.start_angle+self.sweep_angle*value, self.sweep_angle*derivative

    def _coefficients(self, angle):
        first, second = self.first, self.second
        constants = _branch_constants(self)
        other_axis, e1, e2, w, a = (constants.other_axis, constants.e1, constants.e2,
                                    constants.w, constants.a)
        base = (first.origin + first.radius*(np.cos(angle)[..., None]*e1
                                             + np.sin(angle)[..., None]*e2))
        base_derivative = first.radius*(-np.sin(angle)[..., None]*e1
                                       + np.cos(angle)[..., None]*e2)
        offset = base-second.origin
        q = offset-(offset @ other_axis)[..., None]*other_axis
        dq = base_derivative-(base_derivative @ other_axis)[..., None]*other_axis
        b = 2*(q @ w)
        c = np.sum(q*q, axis=-1)-second.radius**2
        db, dc = 2*(dq @ w), 2*np.sum(q*dq, axis=-1)
        return base, base_derivative, a, b, c, db, dc

    def _root_at_angles(self, angle, *, discriminant=None):
        base, dbase, a, b, c, db, dc = self._coefficients(angle)
        # Evaluate the geometric discriminant before subtracting translated
        # axial quadratics. At a double transition b*b and 4*a*c can be large
        # while their difference is the square of a tiny sine. That cancellation
        # corrupts both the curve and its area integral near a branch endpoint.
        constants = _branch_constants(self)
        n0, nc, ns, magnitude = constants.n0, constants.nc, constants.ns, constants.magnitude
        # Rotate sine/cosine algebraically. Subtracting a phase near pi would
        # lose the tiny angle and turn an exact transition into a false root.
        if magnitude:
            cosine=(nc*np.cos(angle)+ns*np.sin(angle))/magnitude
            sine=(nc*np.sin(angle)-ns*np.cos(angle))/magnitude
        else:
            cosine,sine=np.zeros_like(angle),np.zeros_like(angle)
        # This identity preserves the exact equal-radius double-root case
        # without subtracting nearly equal cosine squares.
        remainder = constants.remainder
        disc = 4*(remainder+magnitude**2*sine*sine-2*n0*magnitude*cosine)
        if discriminant is not None:
            disc=discriminant
        else:
            nodes = _node_treatment(self)
            if nodes:
                disc = _node_discriminant(self, nodes, angle)
        roundoff = 64*np.finfo(float).eps*(b*b+4*a*np.abs(c)+self.second.radius**2*a)
        if np.any(disc < -roundoff):
            raise GeometryError("intersection branch leaves the real cylinder intersection")
        root = np.sqrt(np.maximum(disc, 0))
        # The old c/q alternate reintroduced the same cancelled discriminant
        # through c near symmetric double roots (one branch collapsed to zero).
        # Use the certified geometric separation for both named branches.
        z = (-b+self.branch*root)/(2*a)
        return base, dbase, z, a, b, root, db, dc

    def _root(self, parameters):
        t = _parameters(parameters)
        angle, angle_derivative = self._angle(t)
        discriminant=None
        if self.parameterization != 'linear':
            # Evaluate D(theta)-D(anchor) with trig difference identities.
            # The certified endpoint represents D(anchor)=0; its float angular
            # witness has an arithmetic residual. Subtracting that residual
            # before the square root prevents error amplification as t -> 0.
            # The original support residual is still bounded by the existing
            # endpoint arithmetic envelope, independently of the parameter.
            left = (np.ones_like(t,dtype=bool) if self.parameterization=='left_square' else
                    np.zeros_like(t,dtype=bool) if self.parameterization=='right_square' else t<=.5)
            anchor=np.where(left,self.start_angle,self.start_angle+self.sweep_angle)
            if self.parameterization=='left_square':
                delta=self.sweep_angle*t*t
            elif self.parameterization=='right_square':
                delta=-self.sweep_angle*(1-t)*(1-t)
            else:
                delta=np.where(left,self.sweep_angle*np.sin(.5*np.pi*t)**2,
                               -self.sweep_angle*np.cos(.5*np.pi*t)**2)
            _base,_dbase,_z,a,b,endpoint_root,_db,_dc=self._root_at_angles(anchor)
            envelope=64*np.finfo(float).eps*(b*b+a*self.second.radius**2)
            if np.any(endpoint_root*endpoint_root>envelope):
                raise GeometryError("regular endpoint is not a discriminant transition")
            constants=_branch_constants(self)
            n0,nc,ns=constants.n0,constants.nc,constants.ns
            middle=anchor+.5*delta
            discriminant=-16*np.sin(.5*delta)*(-nc*np.sin(middle)+ns*np.cos(middle))*(
                n0+(nc*np.cos(middle)+ns*np.sin(middle))*np.cos(.5*delta))
            nodes = _node_treatment(self)
            if nodes:
                # D = -4 (n - n_a)(n + n_a), n_a = s_a |c| r2 the anchor's simple fold. Its partner
                # n + n_a = F(-s_a) vanishes at a certified double node of the other fold: take that
                # factor from the node form, keep the anchored difference for the simple fold.
                by_sign = {node.sign: node for node in nodes}
                anchor_sign = np.where(n0+nc*np.cos(anchor)+ns*np.sin(anchor) >= 0, 1, -1)
                partner = np.full_like(np.asarray(angle, dtype=float), np.nan)
                for sign, node in by_sign.items():
                    half = _node_half(self, node, angle)[0]
                    partner = np.where(anchor_sign == -sign, -2*node.tau*constants.magnitude*half*half, partner)
                anchored = 2*np.sin(.5*delta)*(-nc*np.sin(middle)+ns*np.cos(middle))
                discriminant = np.where(np.isnan(partner), discriminant, -4*anchored*partner)
        base, dbase, z, a, b, root, db, dc = self._root_at_angles(angle,discriminant=discriminant)
        regular = ((t == 0) & (self.parameterization in ("left_square", "both_sine"))) | (
                   (t == 1) & (self.parameterization in ("right_square", "both_sine")))
        # A double transition at an angular seam can have sin(pi) roundoff
        # rather than a floating zero. Qualify the endpoint using the stable
        # geometric discriminant and its first two derivatives; interior
        # parameters and ordinary near-transition branches are not snapped.
        constants=_branch_constants(self)
        n0,nc,ns=constants.n0,constants.nc,constants.ns
        n=n0+nc*np.cos(angle)+ns*np.sin(angle)
        dn=-nc*np.sin(angle)+ns*np.cos(angle)
        ddn=-nc*np.cos(angle)-ns*np.sin(angle)
        envelope=64*np.finfo(float).eps*(b*b+a*self.second.radius**2)
        double=((t==0)|(t==1)) & (root*root<=envelope) & (
            np.abs(-8*n*dn)<=2*envelope) & (-8*(dn*dn+n*ddn)>0)
        regular=regular|double
        if np.any(regular):
            # Certified discriminant events are represented by a floating
            # angular witness. Avoid sqrt(roundoff) separating the two
            # branches at their shared endpoint. The implicit residual is
            # bounded by the same arithmetic envelope used above.
            envelope = 64*np.finfo(float).eps*(b*b+a*self.second.radius**2)
            if np.any(regular & (root*root > envelope)):
                raise GeometryError("regular endpoint is not a discriminant transition")
            z = np.where(regular, -b/(2*a), z)
            root = np.where(regular, 0., root)
        return base, dbase, z, a, b, root, db, dc, angle_derivative

    def evaluate(self, parameters):
        base, _d, z, *_rest = self._root(parameters)
        points = base+z[..., None]*self.first.axis
        matrix = np.asarray(self.transform)
        return points @ matrix[:3, :3].T+matrix[:3, 3]

    def derivative(self, parameters):
        t = _parameters(parameters)
        _base, dbase, z, a, b, root, db, dc, da = self._root(t)
        denominator = self.branch*root
        dz = np.divide(-(db*z+dc)*da, denominator,
                       out=np.zeros_like(z), where=denominator != 0)
        nodes = _node_treatment(self)
        if nodes:
            usable, node_dz, _node_ddz = _node_axial_jets(self, nodes, self._angle(t)[0], db, 0.*db, a)
            dz = np.where(usable, node_dz*da, dz)
        singular = denominator == 0
        if np.any(singular):
            ddisc = 2*b*db-4*a*dc
            left = t == 0
            right = t == 1
            left_scale = self.sweep_angle*(np.pi**2/4 if self.parameterization == "both_sine" else 1)
            limit = self.branch*np.sqrt(np.maximum(ddisc*left_scale, 0))/a
            end_limit = -self.branch*np.sqrt(np.maximum(-ddisc*left_scale, 0))/a
            regular_left = left & (self.parameterization in ("left_square", "both_sine"))
            regular_right = right & (self.parameterization in ("right_square", "both_sine"))
            angle, _ = self._angle(t)
            first, second = self.first, self.second
            axis = np.asarray(second.axis)
            project = np.eye(3)-np.outer(axis, axis)
            radial = first.radius*(np.cos(angle)[..., None]*first.radial_direction
                       + np.sin(angle)[..., None]*np.cross(first.axis, first.radial_direction))
            q = (_base-second.origin) @ project.T
            dq, ddq = dbase @ project.T, -radial @ project.T
            w = project @ np.asarray(first.axis)
            ddb = 2*(ddq @ w)
            ddc = 2*(np.sum(dq*dq, axis=-1)+np.sum(q*ddq, axis=-1))
            d2disc = 2*(db*db+b*ddb)-4*a*ddc
            envelope = 128*np.finfo(float).eps*(b*b+a*second.radius**2)
            # On a certified node pair an interior parameter can round onto the node angle itself (the
            # node factorisation then gives root 0 exactly): take the one-sided limit from the piece's side.
            nodal = singular & ~(left | right) & bool(_node_treatment(self))
            left_side, right_side = left | (nodal & (t < .5)), right | (nodal & (t >= .5))
            double = (np.abs(ddisc) <= envelope) & (d2disc > 0) & (left_side | right_side)
            double_limit = -db*da/(2*a)+self.branch*np.where(left_side, 1., -1.)*np.abs(da)*np.sqrt(np.maximum(.5*d2disc, 0))/(2*a)
            if np.any(singular & ~(regular_left | regular_right | double)):
                raise GeometryError("intersection branch needs a regular endpoint chart")
            dz = np.where(singular & regular_left, limit, dz)
            dz = np.where(singular & regular_right, end_limit, dz)
            dz = np.where(singular & double & ~(regular_left | regular_right), double_limit, dz)
        result = dbase*da[..., None]+dz[..., None]*self.first.axis
        return result @ np.asarray(self.transform)[:3, :3].T

    def second_derivative(self, parameters):
        """Analytic second derivative, including one-sided endpoint jets."""
        t = _parameters(parameters)
        base, dbase, z, a, b, root, db, dc, da = self._root(t)
        angle, _ = self._angle(t)
        if self.parameterization == "left_square":
            dda = np.full_like(t, 2*self.sweep_angle)
        elif self.parameterization == "right_square":
            dda = np.full_like(t, -2*self.sweep_angle)
        elif self.parameterization == "both_sine":
            dda = .5*np.pi**2*self.sweep_angle*np.cos(np.pi*t)
        else:
            dda = np.zeros_like(t)
        first, second = self.first, self.second
        project = np.eye(3)-np.outer(second.axis, second.axis)
        radial = first.radius*(np.cos(angle)[..., None]*first.radial_direction
                   + np.sin(angle)[..., None]*np.cross(first.axis, first.radial_direction))
        d2base = -radial
        q, dq, ddq = (base-second.origin) @ project.T, dbase @ project.T, d2base @ project.T
        w = project @ np.asarray(first.axis)
        ddb = 2*(ddq @ w)
        ddc = 2*(np.sum(dq*dq, axis=-1)+np.sum(q*ddq, axis=-1))
        denominator = self.branch*root
        dz = np.divide(-(db*z+dc), denominator, out=np.zeros_like(z), where=denominator != 0)
        ddz = np.divide(-(2*a*dz*dz+2*db*dz+ddb*z+ddc), denominator,
                        out=np.zeros_like(z), where=denominator != 0)
        nodes = _node_treatment(self)
        if nodes:
            usable, node_dz, node_ddz = _node_axial_jets(self, nodes, angle, db, ddb, a)
            dz, ddz = np.where(usable, node_dz, dz), np.where(usable, node_ddz, ddz)
        axial_second = ddz*da*da+dz*dda
        singular = denominator == 0
        regular = ((t == 0) & (self.parameterization in ("left_square", "both_sine"))) | (
                   (t == 1) & (self.parameterization in ("right_square", "both_sine")))
        axial_second = np.where(singular & regular, -db*dda/(2*a), axial_second)
        ddisc = 2*b*db-4*a*dc
        d2disc = 2*(db*db+b*ddb)-4*a*ddc
        d3c = 2*(3*np.sum(dq*ddq, axis=-1)-np.sum(q*dq, axis=-1))
        d3disc = 6*db*ddb-2*b*db-4*a*d3c
        envelope = 128*np.finfo(float).eps*(b*b+a*second.radius**2)
        # As in ``derivative``: an interior parameter that rounds onto a certified node takes the one-sided jet.
        nodal = singular & (t != 0) & (t != 1) & bool(_node_treatment(self))
        left_side = (t == 0) | (nodal & (t < .5))
        double = (np.abs(ddisc) <= envelope) & (d2disc > 0) & ((t == 0) | (t == 1) | nodal)
        side = np.where(left_side, 1., -1.)*np.sign(self.sweep_angle)
        double_second = (-ddb/(2*a)+np.divide(
            self.branch*side*np.sqrt(np.maximum(.5*d2disc, 0))*d3disc,
            6*a*d2disc, out=np.zeros_like(z), where=d2disc != 0))*da*da
        axial_second = np.where(singular & double & ~regular, double_second, axial_second)
        if np.any(singular & ~(regular | double)):
            raise GeometryError("intersection endpoint needs a regular second-derivative chart")
        result = d2base*da[..., None]**2+dbase*dda[..., None]+axial_second[..., None]*first.axis
        return result @ np.asarray(self.transform)[:3, :3].T

    def subcurve(self, lower, upper):
        lower, upper = _parameters((lower, upper))
        angles, _ = self._angle(np.asarray((lower, upper)))
        if lower == upper:
            raise GeometryError("curve interval must have positive length")
        low, high = min(lower, upper), max(lower, upper)
        left = low == 0 and self.parameterization in ("left_square", "both_sine")
        right = high == 1 and self.parameterization in ("right_square", "both_sine")
        mode = "both_sine" if left and right else "left_square" if left else "right_square" if right else "linear"
        if upper < lower:
            mode = {"left_square": "right_square", "right_square": "left_square"}.get(mode, mode)
        return replace(self, start_angle=float(angles[0]),
                       sweep_angle=float(angles[1]-angles[0]), parameterization=mode)

    def transformed(self, matrix):
        matrix = _affine(matrix) @ np.asarray(self.transform)
        return replace(self, transform=tuple(tuple(row) for row in matrix))

    def bounds(self, lower=0.0, upper=1.0):
        """Outward interval enclosure of the exact analytic quadratic branch over ``lower .. upper``.

        The contract is the exact branch, not evaluate()'s rounded output. On node-treated charts the
        enclosure also covers evaluate()'s rounding (the node gate keeps the axial solve well conditioned),
        and a limit that is not finite is returned as the corresponding infinity."""
        interval = _parameters((lower, upper))
        if lower > upper:
            raise GeometryError("curve bound interval is reversed")
        first, second = self.first, self.second
        angles, _ = self._angle(interval)
        e1 = np.asarray(first.radial_direction)
        e2 = np.cross(first.axis, e1)
        sweep = float(angles[1]-angles[0])
        nodes = _node_treatment(self)
        if sweep == 0:
            if not nodes:
                point = self.evaluate(lower)
                return np.nextafter(point, -np.inf), np.nextafter(point, np.inf)
            # Node charts: the float angles coincide, but a square chart still moves through its sub-ulp
            # angle offset and evaluation rounding is amplified next to a node. Enclose the exact branch
            # over the neighbouring float angles instead of padding an evaluated point.
            angle = float(angles[0])
            low_angle, high_angle = np.nextafter(angle, -np.inf), np.nextafter(angle, np.inf)
            angles = np.asarray((low_angle, high_angle))
            sweep = float(high_angle-low_angle)
        base_lo, base_hi = EllipticArc(first.origin, first.radius*e1,
                                     first.radius*e2, float(angles[0]), sweep).bounds()
        other_axis = np.asarray(second.axis)
        projection = np.eye(3)-np.outer(other_axis, other_axis)
        if nodes:
            # Halve before adding so that a finite arc near the float limit keeps a finite midpoint.
            center, half = .5*base_lo+.5*base_hi-second.origin, .5*base_hi-.5*base_lo
        else:
            center, half = .5*(base_lo+base_hi)-second.origin, .5*(base_hi-base_lo)
        qcenter, qhalf = projection @ center, np.abs(projection) @ half
        qlo, qhi = qcenter-qhalf, qcenter+qhalf
        w = projection @ np.asarray(first.axis)
        a = float(w @ w)
        bcenter, bhalf = 2*float(w @ qcenter), 2*float(np.abs(w) @ qhalf)
        blo, bhi = bcenter-bhalf, bcenter+bhalf
        clo = float(np.maximum(np.maximum(qlo, -qhi), 0) @ np.maximum(np.maximum(qlo, -qhi), 0))-second.radius**2
        chi = float(np.maximum(qlo*qlo, qhi*qhi).sum())-second.radius**2
        bsqlo = 0 if blo <= 0 <= bhi else min(blo*blo, bhi*bhi)
        bsqhi = max(blo*blo, bhi*bhi)
        rounding = 128*np.finfo(float).eps*(bsqhi+4*a*max(abs(clo), abs(chi))+a*second.radius**2)
        dlo = max(bsqlo-4*a*chi-rounding, 0)
        dhi = max(bsqhi-4*a*clo+rounding, 0)
        if nodes:
            # Both are outward enclosures of D over the piece; their intersection is one too.
            dhi = min(dhi, _node_discriminant_upper(self, nodes, float(angles[0]), sweep))
        rlo, rhi = math.sqrt(dlo), math.sqrt(dhi)
        zlo, zhi = -bhi/(2*a), -blo/(2*a)
        if self.branch == 1:
            zlo, zhi = zlo+rlo/(2*a), zhi+rhi/(2*a)
        else:
            zlo, zhi = zlo-rhi/(2*a), zhi-rlo/(2*a)
        eps16 = 16*np.finfo(float).eps
        if nodes:
            # Node charts are enclosed down to a single ulp of angle, so the enclosure must also cover the
            # rounding of evaluate() itself (scalar and batched alike): z = (-b + root)/(2a) cancels next to
            # a node, so its error is bounded by the pre-cancellation magnitudes of b (and of the offset and
            # projection that form it) and of the root, divided by 2a. Epsilon-first scaling keeps each term
            # finite wherever its operand is.
            offset = np.abs(center)+half
            projected = np.abs(qcenter)+qhalf
            bround = 2*float(np.abs(w) @ (eps16*offset+eps16*projected))
            zpad = (eps16*max(abs(blo), abs(bhi))+bround+eps16*rhi)/(2*a)+eps16*max(abs(zlo), abs(zhi))
            zlo, zhi = zlo-zpad, zhi+zpad
        axis = np.asarray(first.axis)
        lower = base_lo+np.minimum(axis*zlo, axis*zhi)
        upper = base_hi+np.maximum(axis*zlo, axis*zhi)
        if nodes:
            # base + z*axis as evaluate() rounds it.
            point_pad = (eps16*np.maximum(np.abs(base_lo), np.abs(base_hi))
                         + eps16*np.abs(axis)*max(abs(zlo), abs(zhi)))
            lower, upper = lower-point_pad, upper+point_pad
        matrix = np.asarray(self.transform)
        if nodes:
            center, half = .5*lower+.5*upper, .5*upper-.5*lower
        else:
            center, half = .5*(lower+upper), .5*(upper-lower)
        if nodes:
            # The transform of the evaluated point rounds on its pre-cancellation magnitudes, which a
            # transform cancelling a large coordinate would otherwise lose.
            transform_pad = np.abs(matrix[:3, :3]) @ (eps16*np.abs(center)+eps16*half) + eps16*np.abs(matrix[:3, 3])
        center = matrix[:3, :3] @ center+matrix[:3, 3]
        half = np.abs(matrix[:3, :3]) @ half
        if nodes:
            half = half+transform_pad+eps16*np.abs(center)
            # The contract is an enclosure of the exact branch: a limit that overflowed (or became NaN from
            # inf - inf) is replaced by the infinite limit, which still encloses it.
            low, high = np.nextafter(center-half, -np.inf), np.nextafter(center+half, np.inf)
            return np.where(np.isfinite(low), low, -np.inf), np.where(np.isfinite(high), high, np.inf)
        return np.nextafter(center-half, -np.inf), np.nextafter(center+half, np.inf)


from .quadric_curves import QuadricIntersectionCurve  # noqa: E402 (needs the definitions above only in spirit)
from .branch_curves import BezierQuadricCurve  # noqa: E402

from .nurbs import CurveOnSurface, NurbsCurve  # noqa: E402
from .support_curves import PlaneNurbsCurve  # noqa: E402
from .nurbs_intersection_curve import NurbsSurfaceIntersectionCurve  # noqa: E402
from .line_ranges import LineRange  # noqa: E402

EXACT_CURVES = (EllipticArc, CylinderIntersectionCurve, QuadricIntersectionCurve, BezierQuadricCurve, NurbsCurve,
                CurveOnSurface, PlaneNurbsCurve, NurbsSurfaceIntersectionCurve, LineRange)
