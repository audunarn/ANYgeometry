"""Immutable analytic curves used by surface intersections.

These definitions carry geometry, never a display polyline. Edge endpoint
vertices must agree with the definition; moving an endpoint alone is rejected
by topology validation. Whole-model transformations transform the definition.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import math
import heapq
import numpy as np

from .errors import GeometryError
from .surfaces import Cylinder
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
    """Adaptive Gaussian integration of the analytic derivative."""
    previous = None
    count = 16
    while True:
        nodes, weights = np.polynomial.legendre.leggauss(count)
        value = .5*float(weights @ np.linalg.norm(curve.derivative(.5*(nodes+1)), axis=-1))
        if previous is not None and abs(value-previous) <= tolerance:
            return value
        if count >= 1024:
            raise GeometryError("analytic curve length did not converge to the requested tolerance")
        previous, count = value, count*2


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

    def bound(a, b):
        lo, hi = curve.bounds(a, b)
        return float(np.linalg.norm(np.maximum(np.maximum(lo-target, target-hi), 0)))

    queue = [(bound(0., 1.), 0., 1.)]
    while queue:
        distance, a, b = heapq.heappop(queue)
        if best-distance <= tolerance:
            return made.copy(), parameter, best
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
    return made.copy(), parameter, best


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
        if np.linalg.norm(np.cross(u, v)) == 0:
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
        for sign in (-1,1):
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
        axis = np.asarray(first.axis)
        other_axis = np.asarray(second.axis)
        e1 = np.asarray(first.radial_direction)
        e2 = np.cross(axis, e1)
        base = (first.origin + first.radius*(np.cos(angle)[..., None]*e1
                                             + np.sin(angle)[..., None]*e2))
        base_derivative = first.radius*(-np.sin(angle)[..., None]*e1
                                       + np.cos(angle)[..., None]*e2)
        offset = base-second.origin
        q = offset-(offset @ other_axis)[..., None]*other_axis
        dq = base_derivative-(base_derivative @ other_axis)[..., None]*other_axis
        w = axis-float(axis @ other_axis)*other_axis
        a = float(w @ w)
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
        cross = np.cross(self.first.axis,self.second.axis)
        n0 = float((np.asarray(self.first.origin)-self.second.origin) @ cross)
        nc = float(self.first.radius*np.asarray(self.first.radial_direction) @ cross)
        ns = float(self.first.radius*np.cross(self.first.axis,self.first.radial_direction) @ cross)
        magnitude = math.hypot(nc,ns)
        # Rotate sine/cosine algebraically. Subtracting a phase near pi would
        # lose the tiny angle and turn an exact transition into a false root.
        if magnitude:
            cosine=(nc*np.cos(angle)+ns*np.sin(angle))/magnitude
            sine=(nc*np.sin(angle)-ns*np.cos(angle))/magnitude
        else:
            cosine,sine=np.zeros_like(angle),np.zeros_like(angle)
        # This identity preserves the exact equal-radius double-root case
        # without subtracting nearly equal cosine squares.
        remainder = math.fsum((float(cross @ cross)*self.second.radius**2,-magnitude**2,-n0*n0))
        disc = 4*(remainder+magnitude**2*sine*sine-2*n0*magnitude*cosine)
        if discriminant is not None:
            disc=discriminant
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
            cross=np.cross(self.first.axis,self.second.axis)
            n0=float((np.asarray(self.first.origin)-self.second.origin) @ cross)
            nc=float(self.first.radius*np.asarray(self.first.radial_direction) @ cross)
            ns=float(self.first.radius*np.cross(self.first.axis,self.first.radial_direction) @ cross)
            middle=anchor+.5*delta
            discriminant=-16*np.sin(.5*delta)*(-nc*np.sin(middle)+ns*np.cos(middle))*(
                n0+(nc*np.cos(middle)+ns*np.sin(middle))*np.cos(.5*delta))
        base, dbase, z, a, b, root, db, dc = self._root_at_angles(angle,discriminant=discriminant)
        regular = ((t == 0) & (self.parameterization in ("left_square", "both_sine"))) | (
                   (t == 1) & (self.parameterization in ("right_square", "both_sine")))
        # A double transition at an angular seam can have sin(pi) roundoff
        # rather than a floating zero. Qualify the endpoint using the stable
        # geometric discriminant and its first two derivatives; interior
        # parameters and ordinary near-transition branches are not snapped.
        cross=np.cross(self.first.axis,self.second.axis)
        n0=float((np.asarray(self.first.origin)-self.second.origin) @ cross)
        nc=float(self.first.radius*np.asarray(self.first.radial_direction) @ cross)
        ns=float(self.first.radius*np.cross(self.first.axis,self.first.radial_direction) @ cross)
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
            double = (np.abs(ddisc) <= envelope) & (d2disc > 0) & (left | right)
            double_limit = -db*da/(2*a)+self.branch*np.where(left, 1., -1.)*np.abs(da)*np.sqrt(np.maximum(.5*d2disc, 0))/(2*a)
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
        double = (np.abs(ddisc) <= envelope) & (d2disc > 0) & ((t == 0) | (t == 1))
        side = np.where(t == 0, 1., -1.)*np.sign(self.sweep_angle)
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
        """Outward interval enclosure of the analytic quadratic branch."""
        interval = _parameters((lower, upper))
        if lower > upper:
            raise GeometryError("curve bound interval is reversed")
        first, second = self.first, self.second
        angles, _ = self._angle(interval)
        e1 = np.asarray(first.radial_direction)
        e2 = np.cross(first.axis, e1)
        sweep = float(angles[1]-angles[0])
        if sweep == 0:
            point = self.evaluate(lower)
            return np.nextafter(point, -np.inf), np.nextafter(point, np.inf)
        base_lo, base_hi = EllipticArc(first.origin, first.radius*e1,
                                     first.radius*e2, float(angles[0]), sweep).bounds()
        other_axis = np.asarray(second.axis)
        projection = np.eye(3)-np.outer(other_axis, other_axis)
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
        rlo, rhi = math.sqrt(dlo), math.sqrt(dhi)
        zlo, zhi = -bhi/(2*a), -blo/(2*a)
        if self.branch == 1:
            zlo, zhi = zlo+rlo/(2*a), zhi+rhi/(2*a)
        else:
            zlo, zhi = zlo-rhi/(2*a), zhi-rlo/(2*a)
        axis = np.asarray(first.axis)
        lower = base_lo+np.minimum(axis*zlo, axis*zhi)
        upper = base_hi+np.maximum(axis*zlo, axis*zhi)
        matrix = np.asarray(self.transform)
        center, half = .5*(lower+upper), .5*(upper-lower)
        center = matrix[:3, :3] @ center+matrix[:3, 3]
        half = np.abs(matrix[:3, :3]) @ half
        return np.nextafter(center-half, -np.inf), np.nextafter(center+half, np.inf)


EXACT_CURVES = (EllipticArc, CylinderIntersectionCurve)
