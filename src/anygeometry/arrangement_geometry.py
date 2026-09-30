"""Immutable curve paths and analytic predicates for material arrangements.

This module has no mutable topology. It is shared by planning and surface-chart
queries; discretization samples are never used for trim or junction predicates.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import numpy as np

from .analytic_roots import isolate_real_roots, trigonometric_roots
from .curves import Arc, Straight, Spline
from .errors import GeometryError
from .exact_curves import EXACT_CURVES, EllipticArc, CylinderIntersectionCurve
from .surfaces import Cylinder, Plane


@dataclass(frozen=True, slots=True)
class LinePath:
    start: tuple
    end: tuple

    def evaluate(self, parameters):
        t = np.asarray(parameters, dtype=float)
        return np.asarray(self.start)+t[..., None]*(np.asarray(self.end)-self.start)

    def derivative(self, parameters):
        return np.broadcast_to(np.asarray(self.end)-self.start, np.shape(parameters)+(3,)).copy()

    def subcurve(self, lower, upper):
        return LinePath(tuple(self.evaluate(lower)), tuple(self.evaluate(upper)))

    def bounds(self, lower=0., upper=1.):
        points = self.evaluate((lower, upper))
        return np.nextafter(points.min(axis=0), -np.inf), np.nextafter(points.max(axis=0), np.inf)


@dataclass(frozen=True, slots=True)
class BezierPath:
    controls: tuple

    def evaluate(self, parameters):
        t = np.asarray(parameters, dtype=float)
        work = np.broadcast_to(self.controls, t.shape+np.shape(self.controls)).copy()
        for size in range(len(self.controls)-1, 0, -1):
            work[..., :size, :] = ((1-t[..., None, None])*work[..., :size, :]
                                   + t[..., None, None]*work[..., 1:size+1, :])
        return work[..., 0, :].copy()

    def derivative(self, parameters):
        if len(self.controls) == 1:
            return np.zeros(np.shape(parameters)+(3,))
        return BezierPath(tuple(map(tuple, (len(self.controls)-1)*np.diff(self.controls, axis=0)))).evaluate(parameters)

    def _split(self, t):
        work = np.asarray(self.controls).copy()
        left, right = [work[0].copy()], [work[-1].copy()]
        for size in range(len(work)-1, 0, -1):
            work[:size] = (1-t)*work[:size]+t*work[1:size+1]
            left.append(work[0].copy()); right.append(work[size-1].copy())
        return BezierPath(tuple(map(tuple, left))), BezierPath(tuple(map(tuple, right[::-1])))

    def subcurve(self, lower, upper):
        if upper < lower:
            return BezierPath(self.subcurve(upper, lower).controls[::-1])
        left = self if upper == 1 else self._split(upper)[0]
        return left if lower == 0 else left._split(lower/upper)[1]

    def bounds(self, lower=0., upper=1.):
        controls = np.asarray(self.subcurve(lower, upper).controls)
        return np.nextafter(controls.min(axis=0), -np.inf), np.nextafter(controls.max(axis=0), np.inf)

    def transformed(self,matrix):
        matrix=np.asarray(matrix,dtype=float)
        points=np.asarray(self.controls) @ matrix[:3,:3].T+matrix[:3,3]
        return BezierPath(tuple(tuple(point) for point in points))


def freeze_edge(model, edge_id):
    edge = model.edges[edge_id]
    if isinstance(edge.curve, Straight):
        return LinePath(tuple(model.vertex_position(edge.start)), tuple(model.vertex_position(edge.end)))
    if isinstance(edge.curve, Arc):
        frame = model.arc_frame(edge_id)
        return EllipticArc(frame.center, frame.radius*frame.e1, frame.radius*frame.e2, 0., frame.sweep)
    if isinstance(edge.curve, Spline):
        return BezierPath(tuple(map(tuple, model._spline_points(edge))))
    if isinstance(edge.curve, EXACT_CURVES):
        return edge.curve
    raise GeometryError("arrangement curve definition is unsupported")


def _angle_parameter(curve, angle):
    fraction = (angle-curve.start_angle)/curve.sweep_angle
    if -128*np.finfo(float).eps <= fraction <= 1+128*np.finfo(float).eps:
        fraction=min(1.,max(0.,fraction))
    if isinstance(curve, CylinderIntersectionCurve):
        if curve.parameterization == "left_square":
            return math.sqrt(max(0., fraction))
        if curve.parameterization == "right_square":
            return 1-math.sqrt(max(0., 1-fraction))
        if curve.parameterization == "both_sine":
            return 2/math.pi*math.asin(math.sqrt(min(1., max(0., fraction))))
    return fraction


def plane_roots(curve, normal, offset, *, tolerance=1e-12, cancellation_check=None):
    """Every curve/plane root; None denotes coincidence with the whole curve."""
    normal = np.asarray(normal)
    if isinstance(curve, LinePath):
        start = float(normal @ curve.start-offset)
        delta = float(normal @ (np.asarray(curve.end)-curve.start))
        if delta == 0:
            return None if abs(start) <= tolerance else ()
        root = -start/delta
        return (min(1., max(0., root)),) if -tolerance <= root <= 1+tolerance else ()
    if isinstance(curve, EllipticArc):
        coefficients = (float(normal @ curve.center-offset), float(normal @ curve.u_vector),
                        float(normal @ curve.v_vector), 0., 0.)
        if sum(abs(value) for value in coefficients) <= tolerance:
            return None
        angles = trigonometric_roots(coefficients, start=curve.start_angle,
                                      sweep=curve.sweep_angle, tolerance=4*np.finfo(float).eps,
                                      cancellation_check=cancellation_check)
        return tuple(_angle_parameter(curve, angle) for angle in angles)
    if isinstance(curve, CylinderIntersectionCurve):
        from .quadric_curve_events import quadric_roots
        return quadric_roots(curve,np.zeros((3,3)),.5*normal,-offset,
                             tolerance=tolerance,cancellation_check=cancellation_check)
    if isinstance(curve, BezierPath):
        # Bernstein-to-power conversion of the scalar plane constraint.
        values = np.asarray(curve.controls) @ normal-offset
        if np.max(np.abs(values)) <= tolerance:
            return None
        degree = len(values)-1
        coefficients = [sum((-1)**(k-i)*math.comb(degree, i)*math.comb(degree-i, k-i)*float(values[i])
                            for i in range(k+1)) for k in range(degree+1)]
        return tuple(min(1., max(0., root.witness)) for root in isolate_real_roots(
            coefficients, tolerance=4*np.finfo(float).eps, cancellation_check=cancellation_check)
            if -tolerance <= root.witness <= 1+tolerance)
    raise GeometryError("arrangement plane predicate is unsupported")


def point_parameters(curve, point, *, tolerance=1e-10):
    """All analytic parameters, retaining exact chart endpoint identities.

    Inverting a regularized branch at its transition can amplify an angular
    rounding error. The immutable endpoint definition is an independent exact
    candidate and must not be lost (or misclassified as an interior station).
    """
    point=np.asarray(point,dtype=float)
    endpoints=tuple(t for t in (0.,1.) if np.linalg.norm(curve.evaluate(t)-point)<=tolerance)
    roots=_point_parameters(curve,point,tolerance=tolerance)
    values=[]
    for parameter in (*endpoints,*roots):
        snapped=next((t for t in endpoints if abs(parameter-t)<=128*np.finfo(float).eps),parameter)
        if snapped not in values:
            values.append(snapped)
    return tuple(values)


def _point_parameters(curve, point, *, tolerance=1e-10):
    """All parameters for an exact point on a line or ellipse."""
    point = np.asarray(point)
    if isinstance(curve, LinePath):
        direction = np.asarray(curve.end)-curve.start
        length2 = float(direction @ direction)
        if length2 == 0:
            raise GeometryError("degenerate arrangement line")
        parameter = float((point-curve.start) @ direction)/length2
        if -tolerance <= parameter <= 1+tolerance and np.linalg.norm(curve.evaluate(parameter)-point) <= tolerance:
            return (min(1., max(0., parameter)),)
        return ()
    if isinstance(curve, EllipticArc):
        matrix = np.column_stack((curve.u_vector, curve.v_vector))
        coefficients = np.linalg.lstsq(matrix, point-curve.center, rcond=None)[0]
        if abs(float(coefficients @ coefficients)-1) > tolerance/max(np.linalg.norm(matrix), tolerance):
            return ()
        if np.linalg.norm(matrix @ coefficients+curve.center-point) > tolerance:
            return ()
        angle = math.atan2(coefficients[1], coefficients[0])
        lower, upper = sorted((curve.start_angle, curve.start_angle+curve.sweep_angle))
        roots = []
        for turn in range(math.ceil((lower-angle-tolerance)/math.tau), math.floor((upper-angle+tolerance)/math.tau)+1):
            parameter = (angle+turn*math.tau-curve.start_angle)/curve.sweep_angle
            roots.append(min(1., max(0., parameter)))
        return tuple(roots)
    if isinstance(curve, CylinderIntersectionCurve):
        matrix = np.asarray(curve.transform)
        original = np.linalg.solve(matrix[:3, :3], point-matrix[:3, 3])
        offset = original-curve.first.origin
        e1 = np.asarray(curve.first.radial_direction)
        e2 = np.cross(curve.first.axis, e1)
        angle = math.atan2(float(offset @ e2), float(offset @ e1))
        lower, upper = sorted((curve.start_angle, curve.start_angle+curve.sweep_angle))
        roots = []
        for turn in range(math.ceil((lower-angle-tolerance)/math.tau), math.floor((upper-angle+tolerance)/math.tau)+1):
            parameter = min(1., max(0., _angle_parameter(curve, angle+turn*math.tau)))
            if np.linalg.norm(curve.evaluate(parameter)-point) <= tolerance:
                roots.append(parameter)
        return tuple(roots)
    if isinstance(curve, BezierPath):
        controls = np.asarray(curve.controls)
        coordinate = int(np.argmax(np.ptp(controls, axis=0)))
        normal = np.eye(3)[coordinate]
        roots = plane_roots(curve, normal, float(point[coordinate]), tolerance=tolerance)
        return tuple(t for t in (() if roots is None else roots)
                     if np.linalg.norm(curve.evaluate(t)-point) <= tolerance)
    raise GeometryError("arrangement point predicate is unsupported")


def line_curve_junctions(line, curve, *, tolerance=1e-10,cancellation_check=None):
    """All isolated crossings and overlap endpoints of a line and a curve."""
    direction = np.asarray(line.end)-line.start
    if np.linalg.norm(direction) <= tolerance:
        raise GeometryError("degenerate arrangement line")
    # Two independent planes contain the infinite line. Any isolated common
    # point must be a root of at least one non-coincident plane constraint.
    coordinate = np.eye(3)[int(np.argmin(np.abs(direction)))]
    normal = np.cross(direction, coordinate)
    normal /= np.linalg.norm(normal)
    first_roots = plane_roots(curve, normal, float(normal @ line.start), tolerance=tolerance,
                              cancellation_check=cancellation_check)
    other_normal = np.cross(direction, normal)
    other_normal /= np.linalg.norm(other_normal)
    second_roots = plane_roots(curve, other_normal, float(other_normal @ line.start), tolerance=tolerance,
                               cancellation_check=cancellation_check)
    if first_roots is None and second_roots is None:
        roots = (0., 1., *point_parameters(curve, line.start, tolerance=tolerance),
                 *point_parameters(curve, line.end, tolerance=tolerance))
    else:
        # Either plane can be tangent to a curved path. Enumerate both root
        # sets, then qualify the full line residual; a tangent scalar root's
        # roundoff must not hide a transverse root of the other constraint.
        roots = (*(first_roots or ()), *(second_roots or ()))
    result = []
    for parameter in roots:
        point = curve.evaluate(parameter)
        for line_parameter in point_parameters(line, point, tolerance=tolerance):
            result.append((line_parameter, parameter))
    return tuple(result)


def ellipse_curve_junctions(ellipse, curve, *, tolerance=1e-10,cancellation_check=None):
    """Analytic ellipse intersections with coplanar lines and ellipses."""
    if isinstance(curve, LinePath):
        return tuple((second, first) for first, second in line_curve_junctions(curve, ellipse, tolerance=tolerance,
                                                                           cancellation_check=cancellation_check))
    normal = np.cross(ellipse.u_vector, ellipse.v_vector)
    normal /= np.linalg.norm(normal)
    roots = plane_roots(curve, normal, float(normal @ ellipse.center), tolerance=tolerance,
                        cancellation_check=cancellation_check)
    if roots is None:
        if isinstance(curve,CylinderIntersectionCurve):
            from .quadric_curve_events import quadric_roots
            if ellipse_on_branch_supports(ellipse,curve,tolerance):
                roots=None
            else:
                inverse=np.linalg.pinv(np.column_stack((ellipse.u_vector,ellipse.v_vector)))
                matrix=inverse.T @ inverse
                center=np.asarray(ellipse.center)
                roots=quadric_roots(curve,matrix,-matrix @ center,float(center @ matrix @ center-1),
                                    tolerance=tolerance/max(np.linalg.norm(inverse),tolerance),
                                    cancellation_check=cancellation_check)
            if roots is None:
                roots=(0.,1.,*point_parameters(curve,ellipse.evaluate(0.),tolerance=tolerance),
                       *point_parameters(curve,ellipse.evaluate(1.),tolerance=tolerance))
        elif isinstance(curve, BezierPath):
            inverse = np.linalg.pinv(np.column_stack((ellipse.u_vector, ellipse.v_vector)))
            controls = (np.asarray(curve.controls)-ellipse.center) @ inverse.T
            degree = len(controls)-1
            power = [[sum(Fraction(float(controls[i, coordinate]))*((-1)**(k-i)
                      * math.comb(degree, i)*math.comb(degree-i, k-i)) for i in range(k+1))
                      for k in range(degree+1)] for coordinate in range(2)]
            polynomial = [Fraction(0)]*(2*degree+1)
            polynomial[0] = Fraction(-1)
            for values in power:
                for i, first in enumerate(values):
                    for j, second in enumerate(values):
                        polynomial[i+j] += first*second
            roots = tuple(root.witness for root in isolate_real_roots(polynomial,
                          tolerance=4*np.finfo(float).eps,cancellation_check=cancellation_check) if 0 <= root.witness <= 1)
        elif not isinstance(curve, EllipticArc):
            raise GeometryError("coincident-plane curve/ellipse predicate is not implemented")
        else:
            inverse = np.linalg.pinv(np.column_stack((ellipse.u_vector, ellipse.v_vector)))
            q0 = inverse @ (np.asarray(curve.center)-ellipse.center)
            qc, qs = inverse @ curve.u_vector, inverse @ curve.v_vector
            coefficients = (float(q0 @ q0+.5*(qc @ qc+qs @ qs)-1),
                            float(2*q0 @ qc), float(2*q0 @ qs),
                            float(.5*(qc @ qc-qs @ qs)), float(qc @ qs))
            if sum(abs(value) for value in coefficients) <= tolerance:
                roots = (0., 1., *point_parameters(curve, ellipse.evaluate(0.), tolerance=tolerance),
                         *point_parameters(curve, ellipse.evaluate(1.), tolerance=tolerance))
            else:
                angles = trigonometric_roots(coefficients, start=curve.start_angle,
                                              sweep=curve.sweep_angle, tolerance=4*np.finfo(float).eps,
                                              cancellation_check=cancellation_check)
                roots = tuple(_angle_parameter(curve, angle) for angle in angles)
    result = []
    for parameter in roots:
        point = curve.evaluate(parameter)
        for ellipse_parameter in point_parameters(ellipse, point, tolerance=tolerance):
            result.append((ellipse_parameter, parameter))
    return tuple(result)


def ellipse_on_branch_supports(ellipse,curve,tolerance):
    """Whole-curve implicit coefficient certificates on both supports.

    This avoids losing an exact common ellipse through floating pseudoinverse
    cancellation. Branch selection remains a separate chart-point inversion.
    """
    from .member_arrangements import _support_roots
    original=ellipse.transformed(np.linalg.inv(np.asarray(curve.transform)))
    return all(_support_roots(original,parent.surface(),tolerance,lambda:None) is None
               for parent in (curve.first,curve.second))


def curve_junctions(first, second, *, tolerance=1e-10,cancellation_check=None):
    # Named interval endpoints are independent qualified candidates. A root
    # of a clipped conic can round just beyond its new angular chart; retaining
    # its full world-space endpoint avoids an artificial detached port.
    candidates=[]
    for t in (0.,1.):
        if cancellation_check is not None and cancellation_check():
            raise GeometryError('curve intersection predicate cancelled')
        candidates.extend((t,s) for s in point_parameters(second,first.evaluate(t),tolerance=tolerance))
        candidates.extend((s,t) for s in point_parameters(first,second.evaluate(t),tolerance=tolerance))
    candidates.extend(_curve_junctions(first,second,tolerance=tolerance,
                                      cancellation_check=cancellation_check))
    return tuple(sorted(dict.fromkeys(candidates)))


def _curve_junctions(first, second, *, tolerance=1e-10,cancellation_check=None):
    if cancellation_check is not None and cancellation_check():
        raise GeometryError('curve intersection predicate cancelled')
    if isinstance(first, LinePath):
        return line_curve_junctions(first, second, tolerance=tolerance,cancellation_check=cancellation_check)
    if isinstance(second, LinePath):
        return tuple((b, a) for a, b in line_curve_junctions(second, first, tolerance=tolerance,
                                                         cancellation_check=cancellation_check))
    if isinstance(first, EllipticArc):
        return ellipse_curve_junctions(first, second, tolerance=tolerance,cancellation_check=cancellation_check)
    if isinstance(second, EllipticArc):
        return tuple((b, a) for a, b in ellipse_curve_junctions(second, first, tolerance=tolerance,
                                                            cancellation_check=cancellation_check))
    if isinstance(first, CylinderIntersectionCurve) and isinstance(second, CylinderIntersectionCurve):
        from .cylinder_curve_events import branch_junctions
        return branch_junctions(first, second, tolerance=tolerance,cancellation_check=cancellation_check)
    if isinstance(first,BezierPath) and isinstance(second,BezierPath):
        from .bezier_intersections import bezier_junctions
        return bezier_junctions(first,second,tolerance=tolerance,cancellation_check=cancellation_check)
    if isinstance(first,BezierPath) and isinstance(second,CylinderIntersectionCurve):
        from .bezier_intersections import bezier_branch_junctions
        return bezier_branch_junctions(first,second,tolerance=tolerance,cancellation_check=cancellation_check)
    if isinstance(second,BezierPath) and isinstance(first,CylinderIntersectionCurve):
        from .bezier_intersections import bezier_branch_junctions
        return tuple((b,a) for a,b in bezier_branch_junctions(second,first,tolerance=tolerance,
                                                           cancellation_check=cancellation_check))
    raise GeometryError("general analytic curve-pair arrangement predicate is not implemented")
