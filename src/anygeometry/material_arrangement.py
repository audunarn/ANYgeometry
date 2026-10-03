"""Read-only half-edge arrangements of exact material-domain curves.

Original domains and all physical traces are inputs together. Construction
seams connect interior-ended traces without extending the physical joint.
Topology is returned only after every junction and material cycle is resolved.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
import heapq
import math
import numpy as np

from .arrangement_geometry import (LinePath, BezierPath, curve_junctions,
                                   plane_roots, point_parameters, freeze_edge)
from .exact_curves import EllipticArc, CylinderIntersectionCurve
from .branch_curves import BezierQuadricCurve
from .quadric_curves import QuadricIntersectionCurve
from .errors import GeometryError
from .extruded_chart import area_density, chart_accelerations, chart_rates, density_bound, world_metric
from .extruded_supports import extruded_support
from .extrusions import EllipseDirectrix
from .surfaces import Cone, Cylinder, ExtrudedSurface, Plane, _angle_on_sweep

_EPS = float(np.finfo(float).eps)


@dataclass(frozen=True, slots=True)
class ArrangementPath:
    curve: object
    source_edge: int | None = None
    owners: tuple[int, ...] = ()
    decomposition: bool = False
    member_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ArrangementPoint:
    position: tuple[float, float, float]
    member_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ArrangementCell:
    outer: tuple[tuple[int, bool], ...]
    holes: tuple[tuple[tuple[int, bool], ...], ...] = ()


@dataclass(frozen=True, slots=True)
class MaterialArrangement:
    face_id: int
    paths: tuple[ArrangementPath, ...]
    cells: tuple[ArrangementCell, ...]
    area: float
    world_tolerance: float = 1e-9
    native_area_tolerance: float = 1e-12
    support: object = None
    orientation: int = 1                # -1 when the stored face loop runs clockwise in the support chart


@dataclass(frozen=True, slots=True)
class MaterialDomain:
    face_id: int
    support: Plane | Cylinder | Cone | ExtrudedSurface
    boundaries: tuple[tuple[ArrangementPath, ...], ...]

    @property
    def area_jacobian(self):
        if isinstance(self.support, Plane):
            return float(np.linalg.norm(np.cross(self.support.u_vector, self.support.v_vector)))
        if isinstance(self.support, ExtrudedSurface):
            return density_bound(self.support)            # an upper bound: the area element varies along the directrix
        if isinstance(self.support, Cone):
            # Upper bound: the cone's area element grows with its radius.
            slant = math.hypot(self.support.height, self.support.radius_end-self.support.radius_start)
            return abs(max(self.support.radius_start, self.support.radius_end)*self.support.sweep_angle*slant)
        return abs(self.support.radius*self.support.sweep_angle*self.support.height)

    def world_delta(self, delta, v=0., u=0.):
        if isinstance(self.support, Plane):
            return delta[0]*self.support.u_vector+delta[1]*self.support.v_vector
        if isinstance(self.support, ExtrudedSurface):
            return world_metric(self.support, delta, u)   # the two chart directions are not orthogonal: a 3-vector
        if isinstance(self.support, Cone):
            radius = (1-v)*self.support.radius_start+v*self.support.radius_end
            slant = math.hypot(self.support.height, self.support.radius_end-self.support.radius_start)
            return np.asarray((delta[0]*radius*self.support.sweep_angle, delta[1]*slant))
        return np.asarray((delta[0]*self.support.radius*self.support.sweep_angle,
                           delta[1]*self.support.height))

    def world_area_loop(self, loop, tolerance):
        """Signed physical area of a closed loop (positive counterclockwise in the chart).

        A Cylinder or Plane has a constant area element, so this is the native area times
        :attr:`area_jacobian`. A Cone's element ``|sweep|*slant*r(v)`` grows with the
        radius, so ``-integral(F(v) du)`` with ``F`` the antiderivative of ``r`` is used.
        """
        if isinstance(self.support, ExtrudedSurface):
            return self._extruded_world_area(loop, tolerance)
        if not isinstance(self.support, Cone):
            return self.area_loop(loop, tolerance/self.area_jacobian)*self.area_jacobian
        support = self.support
        slant = math.hypot(support.height, support.radius_end-support.radius_start)
        delta_radius = support.radius_end-support.radius_start

        def antiderivative(v):
            return support.radius_start*v+.5*delta_radius*v*v

        scale = abs(support.sweep_angle)*slant

        def integrand(curve, parameters):
            world = curve.evaluate(parameters)
            direction = curve.derivative(parameters)
            offset = world-support.origin
            v = offset @ support.axis/support.height                    # the axial fraction, as Cone.local_uv
            x, y = offset @ support.radial_direction, offset @ support.circumferential_direction
            dx, dy = direction @ support.radial_direction, direction @ support.circumferential_direction
            return -scale*antiderivative(v)*(x*dy-y*dx)/(x*x+y*y)/support.sweep_angle

        return _integrate_loop(loop, lambda curve: 0., integrand, tolerance)

    def _extruded_world_area(self, loop, tolerance):
        """``-integral(v |c'(t) x d| du)``: the area element of an extruded surface depends on ``u`` only."""
        support = self.support

        def integrand(curve, parameters):
            world = curve.evaluate(parameters)
            uv = support.local_uv_many(world)
            rates = chart_rates(support, world, curve.derivative(parameters))
            return -uv[:, 1]*area_density(support, uv[:, 0])*rates[:, 0]

        return _integrate_loop(loop, lambda curve: 0., integrand, tolerance)

    def original_world_area(self, tolerance):
        """Physical area of the stored face: outer loop minus its holes (never a native-area estimate)."""
        return (abs(self.world_area_loop(self.boundaries[0], tolerance))
                - sum(abs(self.world_area_loop(loop, tolerance)) for loop in self.boundaries[1:]))

    def material_world_area(self, arrangement):
        """Physical area of an arrangement's material cells."""
        if not isinstance(self.support, (Cone, ExtrudedSurface)):
            return arrangement.area*self.area_jacobian
        tolerance = arrangement.native_area_tolerance*self.area_jacobian*.05
        total = 0.
        for cell in arrangement.cells:
            def loop(chain):
                return tuple(ArrangementPath(arrangement.paths[e].curve if forward
                                             else arrangement.paths[e].curve.subcurve(1., 0.)) for e, forward in chain)
            total += abs(self.world_area_loop(loop(cell.outer), tolerance))
            total -= sum(abs(self.world_area_loop(loop(hole), tolerance)) for hole in cell.holes)
        return total

    def left_probe(self, point, tangent, tolerance):
        normal = np.asarray((-tangent[1], tangent[0]))/np.linalg.norm(tangent)
        return point+normal*(16*tolerance/np.linalg.norm(self.world_delta(normal, float(point[1]), float(point[0]))))

    @classmethod
    def from_model(cls, model, face_id):
        from .intersections import _qualified_face_plane
        from .entities import EntityRef
        face = model.faces[face_id]
        support = face.surface
        extruded = None if isinstance(support, (Cylinder, Cone)) else extruded_support(model, face_id)
        if extruded is not None:
            support = extruded
        elif not isinstance(support, (Cylinder, Cone)):
            support = _qualified_face_plane(model, face_id)
            if not isinstance(face.surface, Plane):
                # Prefer the authored boundary frame to arbitrary SVD axes
                # in a symmetric plate. Decomposition seams then follow the
                # existing plate directions and preserve its orientation.
                vertices=[model.vertex_position(model.oriented_start_vertex(use)) for use in face.loop]
                origin=vertices[0]
                first=next(point-origin for point in vertices[1:]
                           if np.linalg.norm(point-origin)>model.tolerance.length)
                for point in reversed(vertices[1:]):
                    second=point-origin
                    cross=np.cross(first,second)
                    if np.linalg.norm(cross)>model.tolerance.area:
                        # An authored quadrilateral's two adjacent directions
                        # provide the full affine frame, including scale/shear.
                        support=Plane(origin,first,second)
                        break
        boundaries = []
        for loop in (face.loop, *face.holes):
            paths = []
            for use in loop:
                curve = freeze_edge(model, use.edge)
                if not use.forward:
                    curve = curve.subcurve(1., 0.)
                decomposition = "intersection_decomposition_seam" in model.tags_for(
                    EntityRef("edge", use.edge))
                paths.append(ArrangementPath(curve, use.edge, decomposition=decomposition))
            boundaries.append(tuple(paths))
        return cls(face_id, support, tuple(boundaries))

    def at_apex(self, point):
        """Whether ``point`` is the apex of a Cone support: there the angular chart collapses to a point."""
        support = self.support
        if not isinstance(support, Cone):
            return False
        offset = np.asarray(point)-support.origin
        radial = offset-float(offset @ support.axis)*np.asarray(support.axis)
        return float(np.linalg.norm(radial)) <= 1e-9*max(support.radius_start, support.radius_end)

    def _extruded_uv(self, curve, parameter):
        support = self.support
        uv = np.asarray(support.local_uv(curve.evaluate(parameter)))
        directrix = support.directrix
        if isinstance(directrix, EllipseDirectrix) and abs(abs(directrix.sweep_angle)-math.tau) <= 1e-12:
            # A full-turn profile closes on itself: pick the endpoint's side by the curve's interior chart.
            middle = support.local_uv(curve.evaluate(.5))[0]
            period = 1./abs(support.u_range[1]-support.u_range[0])
            uv[0] += round((middle-uv[0])/period)*period
        return uv

    def uv(self, curve, parameter):
        if isinstance(self.support, ExtrudedSurface):
            return self._extruded_uv(curve, parameter)
        point = curve.evaluate(parameter)
        uv = np.asarray(self.support.local_uv(point))
        if self.at_apex(point):
            # The azimuth of the apex is the direction the curve reaches it along (the limit of the chart).
            direction = np.asarray(curve.derivative(parameter))*(-1. if parameter >= .5 else 1.)
            support = self.support
            angle = math.atan2(float(direction @ support.circumferential_direction),
                               float(direction @ support.radial_direction))
            uv[0] = _angle_on_sweep(angle, support.start_angle, support.sweep_angle)/support.sweep_angle
        if isinstance(self.support, (Cylinder, Cone)):
            # Native seam events split curves before arrangement. Select the
            # endpoint's side using the interior chart, not a display path.
            middle = np.asarray(self.support.local_uv(curve.evaluate(.5)))
            period = math.tau/self.support.sweep_angle
            if abs(abs(self.support.sweep_angle)-math.tau) <= 1e-12:
                uv[0] += round((middle[0]-uv[0])/period)*period
        return uv

    def tangent(self, curve, parameter):
        derivative = curve.derivative(parameter)
        if isinstance(self.support, Plane):
            return np.linalg.lstsq(np.column_stack((self.support.u_vector, self.support.v_vector)), derivative, rcond=None)[0]
        if isinstance(self.support, ExtrudedSurface):
            return chart_rates(self.support, curve.evaluate(parameter), derivative)[0]
        point = curve.evaluate(parameter)-self.support.origin
        x, y = float(point @ self.support.radial_direction), float(point @ self.support.circumferential_direction)
        dx, dy = float(derivative @ self.support.radial_direction), float(derivative @ self.support.circumferential_direction)
        if self.at_apex(point+self.support.origin):
            # 0/0: the azimuth rate at the apex is (x' y'' - y' x'') / (2 (x'^2 + y'^2)) by l'Hopital.
            second = _second_derivative(curve, parameter)
            if second is None or dx*dx+dy*dy == 0:
                rate = 0.
            else:
                ddx, ddy = float(second @ self.support.radial_direction), float(second @ self.support.circumferential_direction)
                rate = (dx*ddy-dy*ddx)/(2*(dx*dx+dy*dy))
            return np.asarray((rate/self.support.sweep_angle, float(derivative @ self.support.axis)/self.support.height))
        return np.asarray(((x*dy-y*dx)/(x*x+y*y)/self.support.sweep_angle,
                           float(derivative @ self.support.axis)/self.support.height))

    def curvature(self, curve, parameter, forward):
        if self.at_apex(curve.evaluate(parameter)):
            return None                  # the chart is singular here; edges at an apex are ordered by azimuth
        second = _second_derivative(curve, parameter)
        if second is None:
            return None
        if isinstance(self.support, ExtrudedSurface):
            acceleration = chart_accelerations(self.support, curve.evaluate(parameter), curve.derivative(parameter), second)[0]
        elif isinstance(self.support, Plane):
            acceleration = np.linalg.lstsq(np.column_stack((self.support.u_vector, self.support.v_vector)), second, rcond=None)[0]
        else:
            point = curve.evaluate(parameter)-self.support.origin
            first = curve.derivative(parameter)
            e1, e2 = self.support.radial_direction, self.support.circumferential_direction
            x, y, dx, dy, ddx, ddy = (float(point @ e1), float(point @ e2),
                                     float(first @ e1), float(first @ e2),
                                     float(second @ e1), float(second @ e2))
            r2 = x*x+y*y
            acceleration = np.asarray((((x*ddy-y*ddx)/r2-2*(x*dx+y*dy)*(x*dy-y*dx)/r2**2)/self.support.sweep_angle,
                                       float(second @ self.support.axis)/self.support.height))
        tangent = self.tangent(curve, parameter)*(1 if forward else -1)
        return float(tangent[0]*acceleration[1]-tangent[1]*acceleration[0])/float(np.linalg.norm(tangent))**3

    def horizontal_roots(self, curve, value, tolerance):
        if isinstance(self.support, ExtrudedSurface):
            # The extrusion coordinate is linear in position: a level of v is a plane parallel to the profile.
            support = self.support
            normal = support.profile_normal
            offset = float(normal @ support.profile_origin)+(support.v_range[0]+value*(
                support.v_range[1]-support.v_range[0]))*support.profile_rate
        elif isinstance(self.support, Plane):
            inverse = np.linalg.pinv(np.column_stack((self.support.u_vector, self.support.v_vector)))
            normal = inverse[1]
            offset = value+float(normal @ self.support.origin)
        else:
            normal = self.support.axis
            offset = float(normal @ self.support.origin)+value*self.support.height
        scalar_tolerance=tolerance*np.linalg.norm(normal)
        roots=plane_roots(curve,normal,offset,tolerance=scalar_tolerance)
        if roots is None:
            return None
        endpoints=tuple(t for t in (0.,1.) if abs(float(normal @ curve.evaluate(t))-offset)<=scalar_tolerance)
        return tuple(dict.fromkeys(next((t for t in endpoints
            if abs(root-t)<=128*np.finfo(float).eps),root) for root in roots))

    def contains_loop(self, loop, uv, tolerance, *, boundary=True):
        winding = 0
        for index, path in enumerate(loop):
            curve = path.curve
            roots = self.horizontal_roots(curve, float(uv[1]), tolerance)
            if roots is None:
                # A horizontal curve can contain the query but contributes no
                # ray crossing. Exact point inversion handles its boundary.
                world = self.support.evaluate(*uv)
                if point_parameters(curve, world, tolerance=tolerance):
                    return boundary
                continue
            roots = sorted(set(min(1., max(0., float(t))) for t in roots))
            for position, t in enumerate(roots):
                point_uv = self.uv(curve, t)
                if np.linalg.norm(self.world_delta(point_uv-uv, float(uv[1]), float(uv[0]))) <= tolerance:
                    return boundary
                if point_uv[0] <= uv[0] or t >= 1-8*np.finfo(float).eps:
                    continue
                if t <= 8*np.finfo(float).eps:
                    previous = loop[index-1].curve
                    previous_roots = self.horizontal_roots(previous, float(uv[1]), tolerance)
                    for distance in range(2, len(loop)+1):
                        if previous_roots is not None:
                            break
                        # A ray lying on one or several consecutive boundary
                        # edges must inspect the preceding non-horizontal
                        # branch. Zero alone is not a crossing sign.
                        previous = loop[index-distance].curve
                        previous_roots = self.horizontal_roots(previous, float(uv[1]), tolerance)
                    previous_lower = max((r for r in (previous_roots or ()) if r < 1-8*np.finfo(float).eps), default=0.)
                    before = self.uv(previous, .5*(previous_lower+1))[1]-uv[1]
                else:
                    previous_root = roots[position-1] if position else 0.
                    before = self.uv(curve, .5*(previous_root+t))[1]-uv[1]
                next_root = roots[position+1] if position+1 < len(roots) else 1.
                after = self.uv(curve, .5*(t+next_root))[1]-uv[1]
                if before <= 0 < after:
                    winding += 1
                elif after <= 0 < before:
                    winding -= 1
        return winding != 0

    def contains(self, curve, parameter, tolerance, *, boundary=True):
        uv = self.uv(curve, parameter)
        return (self.contains_loop(self.boundaries[0], uv, tolerance, boundary=boundary)
                and not any(self.contains_loop(loop, uv, tolerance, boundary=not boundary)
                            for loop in self.boundaries[1:]))

    def area_loop(self, loop, tolerance):
        if isinstance(self.support,Plane):
            from .planar_area import planar_loop_area
            analytic=planar_loop_area(self.support,loop,tolerance)
            if analytic is not None:
                return analytic
        support = self.support
        if isinstance(support, ExtrudedSurface):
            # Integrate in a common translated frame. Evaluating a Bezier boundary
            # at a large world origin and then inverting it loses chart precision;
            # split and unsplit boundaries can consequently disagree in area.
            matrix = np.eye(4)
            matrix[:3, 3] = -support.profile_origin
            support = support.transformed(matrix)
            local_domain = replace(self, support=support)
            def translated(curve):
                if isinstance(curve, LinePath):
                    return LinePath(tuple(np.asarray(curve.start)+matrix[:3, 3]),
                                    tuple(np.asarray(curve.end)+matrix[:3, 3]))
                return curve.transformed(matrix)
            loop = tuple(replace(path, curve=translated(path.curve)) for path in loop)
            constant_v = {}
            for path in loop:
                curve = path.curve
                controls = (curve.start, curve.end) if isinstance(curve, LinePath) else (
                    curve.controls if isinstance(curve, BezierPath) else None)
                if controls is not None:
                    s = support.extrusion_coordinate(controls)
                    normal = tuple(Fraction(float(n)) for n in support.profile_normal)
                    first = tuple(Fraction(float(c)) for c in controls[0])
                    exact_height = np.all(s == s[0]) and all(
                        sum((Fraction(float(c))-a)*n for c, a, n in zip(point, first, normal)) == 0
                        for point in controls[1:])
                    if exact_height:
                        constant_v[id(curve)] = (s[0]-support.v_range[0])/(support.v_range[1]-support.v_range[0])
            def integrand(curve, parameters):
                if id(curve) in constant_v:
                    return np.zeros_like(parameters)
                world = curve.evaluate(parameters)
                return -support.local_uv_many(world)[:, 1]*chart_rates(support, world, curve.derivative(parameters))[:, 0]

            def boundary(curve):
                first, last = local_domain.uv(curve, 0.), local_domain.uv(curve, 1.)
                value = .5*(last[0]*last[1]-first[0]*first[1])
                if id(curve) in constant_v:
                    # Polynomial control heights certify constant v; integrate
                    # -v du by its antiderivative, without recovering noisy rates.
                    value -= constant_v[id(curve)]*(last[0]-first[0])
                return value
        elif isinstance(support, (Cylinder, Cone)):
            def integrand(curve, parameters):
                world = curve.evaluate(parameters)
                direction = curve.derivative(parameters)
                offset = world-support.origin
                if isinstance(support, Cone):
                    v = offset @ support.axis/support.height              # the axial fraction, as Cone.local_uv
                else:
                    v = np.asarray([support.local_uv(point)[1] for point in world])
                x, y = offset @ support.radial_direction, offset @ support.circumferential_direction
                dx, dy = direction @ support.radial_direction, direction @ support.circumferential_direction
                # Green's theorem after integration by parts:
                # .5*integral(u dv-v du) = .5*[uv] - integral(v du).
                # In the native cylinder chart axial v can approach a
                # quadratic branch transition. Its value is well conditioned
                # while computing dv divides by a vanishing discriminant.
                # Avoid that unnecessary division in the area integral.
                return -v*((x*dy-y*dx)/(x*x+y*y)/support.sweep_angle)

            def boundary(curve):
                first, last = self.uv(curve, 0.), self.uv(curve, 1.)
                return .5*(last[0]*last[1]-first[0]*first[1])
        else:
            def integrand(curve, parameters):
                world = curve.evaluate(parameters)
                direction = curve.derivative(parameters)
                positions = np.asarray([support.local_uv(point) for point in world])
                derivatives = np.linalg.lstsq(np.column_stack((support.u_vector, support.v_vector)),
                                              direction.T, rcond=None)[0].T
                return .5*(positions[:, 0]*derivatives[:, 1]-positions[:, 1]*derivatives[:, 0])

            def boundary(curve):
                return 0.
        return _integrate_loop(loop, boundary, integrand, tolerance)


def _integrate_loop(loop, boundary, integrand, tolerance):
    """Green-theorem loop area from Gauss rules of doubling size; adaptive when those do not settle.

    ``integrand(curve, parameters)`` returns the samples of one path's integrand over the
    unit parameter interval and ``boundary(curve)`` its closed-form boundary term. Two
    successive rules cannot agree below the rounding noise of their own sums, so that
    floor bounds the requested tolerance.
    """
    previous = None
    for count in (16, 32, 64, 128, 256, 512):
        nodes, weights = np.polynomial.legendre.leggauss(count)
        value = magnitude = 0.
        for path in loop:
            samples = integrand(path.curve, .5*(nodes+1))
            edge = boundary(path.curve)
            value += edge+.5*float(weights @ samples)                    # [-1, 1] rule on [0, 1]
            magnitude += abs(edge)+.5*float(weights @ np.abs(samples))
        if previous is not None and abs(value-previous) <= max(tolerance, 64*_EPS*magnitude):
            return value
        previous = value
    # A branch point just outside a path (a fold of the exact curve) converges slowly
    # in a global rule; bisection toward it resolves the same integral locally.
    value = 0.
    for path in loop:
        value += boundary(path.curve)+_adaptive_gauss(
            lambda parameters, curve=path.curve: integrand(curve, parameters), tolerance/len(loop))
    return value


def _adaptive_gauss(function, tolerance):
    """Integral of ``function`` over ``[0, 1]``: a 24-point Gauss rule, always bisecting the worst interval.

    An interval's error is the disagreement between its own rule and its two halves'.
    Refinement stops once the total error is below ``tolerance`` or below the noise floor
    of the evaluation itself (near a fold the exact curves' derivative carries a few
    hundred ulp of cancellation noise that no refinement removes); if the work budget
    ends first the integral is refused.
    """
    nodes, weights = np.polynomial.legendre.leggauss(24)

    def rule(a, b):
        samples = function(.5*(a+b)+.5*(b-a)*nodes)
        return .5*(b-a)*float(weights @ samples), .5*(b-a)*float(weights @ np.abs(samples))

    def refine(a, b, whole):
        middle = .5*(a+b)
        left, left_mass = rule(a, middle)
        right, right_mass = rule(middle, b)
        return (abs(left+right-whole), a, b, middle, left, right, left_mass+right_mass)

    first = refine(0., 1., rule(0., 1.)[0])
    heap = [(-first[0], first)]
    settled = []                                    # intervals too small to refine: their error is accepted
    for _ in range(400):
        error = sum(item[0] for _key, item in heap)
        mass = sum(item[6] for _key, item in heap)+sum(item[6] for item in settled)
        if error <= max(tolerance, 2.**-40*mass):
            break
        _key, item = heapq.heappop(heap)
        _error, a, b, middle, left, right, _size = item
        if b-a < 1e-13:
            settled.append(item)
            continue
        for lo, hi, value in ((a, middle, left), (middle, b, right)):
            child = refine(lo, hi, value)
            heapq.heappush(heap, (-child[0], child))
    leaves = [item for _key, item in heap]+settled
    error = sum(item[0] for item in leaves)
    if error > max(tolerance, 2.**-36*sum(item[6] for item in leaves)):
        raise GeometryError("material area integral did not resolve to the requested tolerance")
    return sum(item[4]+item[5] for item in leaves)


def _second_derivative(curve, parameter):
    """Second derivative of an arrangement curve in world coordinates.

    ``None`` for an unknown family or where the exact curve has no finite second derivative (an end at a
    cone's apex): the caller then has no curvature to break a tangent tie.
    """
    with np.errstate(all="ignore"):
        second = _second_derivative_raw(curve, parameter)
    if second is not None and not np.all(np.isfinite(second)):
        return None
    return second


def _second_derivative_raw(curve, parameter):
    if isinstance(curve, LinePath):
        return np.zeros(3)
    if isinstance(curve, EllipticArc):
        angle = curve.start_angle+parameter*curve.sweep_angle
        return -curve.sweep_angle**2*(math.cos(angle)*np.asarray(curve.u_vector)
                                       + math.sin(angle)*np.asarray(curve.v_vector))
    if isinstance(curve, BezierPath):
        controls = np.asarray(curve.controls)
        return (np.zeros(3) if len(controls) <= 2 else BezierPath(tuple(map(tuple,
            (len(controls)-1)*(len(controls)-2)*np.diff(controls, n=2, axis=0)))).evaluate(parameter))
    if isinstance(curve, (CylinderIntersectionCurve, QuadricIntersectionCurve, BezierQuadricCurve)):
        return curve.second_derivative(parameter)
    return None


def _clip(domain, curve, tolerance, check):
    return tuple(curve.subcurve(a,b) for a,b in _clip_intervals(domain,curve,tolerance,check))


def _clip_intervals(domain,curve,tolerance,check,*,isolated_contacts=None):
    """Keep exact source parameter intervals alongside clipped definitions."""
    parameters = [0., 1.]
    for loop in domain.boundaries:
        for boundary in loop:
            check()
            parameters.extend(a for a, _b in curve_junctions(curve, boundary.curve, tolerance=tolerance,
                              cancellation_check=lambda:(check() or False)))
    parameters = _parameters(parameters, tolerance, curve)
    intervals=tuple((a,b) for a, b in zip(parameters, parameters[1:])
                    if domain.contains(curve, .5*(a+b), tolerance))
    if isolated_contacts is not None:
        for parameter in parameters:
            check()
            if not any(a<=parameter<=b for a,b in intervals) and domain.contains(curve,parameter,tolerance):
                isolated_contacts.append(tuple(curve.evaluate(parameter)))
    return intervals


def _parameters(values, tolerance, curve=None):
    values = sorted(values)
    result = []
    # Merge only numerical duplicates. Material-space tolerance is applied to
    # canonical vertices, never as an artificial parameter-feature count cap.
    for value in values:
        duplicate = bool(result and value-result[-1] <= 32*np.finfo(float).eps)
        if result and not duplicate and curve is not None:
            lo, hi = curve.subcurve(result[-1],value).bounds()
            duplicate = np.linalg.norm(hi-lo) <= tolerance
        if not result or not duplicate:
            result.append(value)
        elif value == 1.:
            result[-1] = value
    return result


def arrange_material(domain, traces, *, tolerance, cancellation_check=None,
                     max_predicates=None, area_tolerance=None, points=()):
    """Return the complete arrangement, or raise without returning partial cells.

    ``max_predicates`` is an explicit optional resource budget; default planning
    has no model/intersection-count limit. The caller owns atomic application.
    """
    native_area_tolerance = (tolerance if area_tolerance is None else area_tolerance/domain.area_jacobian)
    examined = 0
    def check():
        nonlocal examined
        examined += 1
        if cancellation_check is not None and cancellation_check():
            raise GeometryError("intersection arrangement cancelled")
        if max_predicates is not None and examined > max_predicates:
            raise GeometryError("intersection arrangement predicate budget exhausted")

    boundaries = [path for loop in domain.boundaries for path in loop]
    paths = list(boundaries)
    for trace in traces:
        for curve in _clip(domain, trace.curve, tolerance, check):
            paths.append(replace(trace, curve=curve))

    # Interior-ended traces need material decomposition seams. Their physical
    # directions must not inherit scale/shear from an affine support chart.
    endpoints = [np.asarray(point.position) for point in points
                 if not any(point_parameters(boundary.curve, point.position, tolerance=tolerance)
                            for boundary in boundaries)]
    for path in paths[len(boundaries):]:
        for t in (0., 1.):
            point = path.curve.evaluate(t)
            if any(point_parameters(boundary.curve, point, tolerance=tolerance) for boundary in boundaries):
                continue
            rays = []
            for other in paths[len(boundaries):]:
                for parameter in point_parameters(other.curve, point, tolerance=tolerance):
                    tangent = other.curve.derivative(parameter)
                    length = np.linalg.norm(tangent)
                    if length == 0:
                        raise GeometryError("interior junction needs a regular chart")
                    tangent = tangent/length
                    for sign in ((-1,) if parameter == 1 else (1,) if parameter == 0 else (-1, 1)):
                        ray = sign*tangent
                        if not any(np.linalg.norm(ray-old) <= 64*np.finfo(float).eps for old in rays):
                            rays.append(ray)
            if len(rays) > 1:
                continue
            if not any(np.linalg.norm(point-old) <= tolerance for old in endpoints):
                endpoints.append(point)
    if isinstance(domain.support, Plane):
        boxes=[path.curve.bounds() for path in boundaries]
        lower=np.min([box[0] for box in boxes],axis=0)
        upper=np.max([box[1] for box in boxes],axis=0)
        extent=max(float(np.linalg.norm(upper-lower)),1.)
        normal=np.cross(domain.support.u_vector,domain.support.v_vector)
        normal/=np.linalg.norm(normal)
        authored=np.asarray(domain.support.u_vector)/np.linalg.norm(domain.support.u_vector)
        for point in sorted(endpoints, key=lambda value: tuple(value)):
            # Preserve the authored first direction and its physical
            # perpendicular. An affine chart's sheared second direction must
            # not introduce artificial acute wedges into the material cells.
            for direction in (authored,np.cross(normal,authored)):
                seam = LinePath(tuple(point-extent*direction),tuple(point+extent*direction))
                paths.extend(ArrangementPath(curve, decomposition=True)
                             for curve in _clip(domain, seam, tolerance, check))
    elif endpoints:
        support = domain.support
        for point in sorted(endpoints, key=lambda value: tuple(value)):
            u, _v = support.local_uv(point)
            seam = LinePath(tuple(support.evaluate(u, -1)), tuple(support.evaluate(u, 2)))
            paths.extend(ArrangementPath(curve, decomposition=True)
                         for curve in _clip(domain, seam, tolerance, check))

    splits = [[0., 1.] for _ in paths]
    for index, path in enumerate(paths):
        for point in points:
            splits[index].extend(point_parameters(path.curve, point.position, tolerance=tolerance))
    for first in range(len(paths)):
        for second in range(first+1, len(paths)):
            check()
            lo1, hi1 = paths[first].curve.bounds()
            lo2, hi2 = paths[second].curve.bounds()
            if np.any(hi1 < lo2-tolerance) or np.any(hi2 < lo1-tolerance):
                continue
            for a, b in curve_junctions(paths[first].curve, paths[second].curve, tolerance=tolerance,
                                       cancellation_check=lambda:(check() or False)):
                splits[first].append(a); splits[second].append(b)

    vertices, edges, incidence = [], [], []
    def vertex(point):
        for index, candidate in enumerate(vertices):
            if np.linalg.norm(point-candidate) <= tolerance:
                return index
        vertices.append(point.copy()); incidence.append([])
        return len(vertices)-1

    endpoints_by_edge = []
    for path, parameters in zip(paths, splits):
        parameters = _parameters(parameters, tolerance, path.curve)
        for a, b in zip(parameters, parameters[1:]):
            curve = path.curve.subcurve(a, b)
            start, end = vertex(curve.evaluate(0.)), vertex(curve.evaluate(1.))
            if start == end:
                raise GeometryError("unresolved closed or degenerate arrangement edge")
            duplicate = None
            for index in incidence[start]:
                old_start, old_end = endpoints_by_edge[index]
                if {start, end} != {old_start, old_end}:
                    continue
                old = edges[index].curve
                if isinstance(curve, LinePath) and isinstance(old, LinePath):
                    # Affine segments with the same canonical endpoints have
                    # the same complete image; no scalar-plane root test is
                    # needed (its nearly-zero normal can amplify roundoff).
                    duplicate = index; break
                if isinstance(curve,BezierPath) and isinstance(old,BezierPath):
                    first_controls=np.asarray(curve.controls)
                    second_controls=np.asarray(old.controls)
                    if first_controls.shape==second_controls.shape and any(
                            np.max(np.linalg.norm(first_controls-values,axis=1))<=tolerance
                            for values in (second_controls,second_controls[::-1])):
                        # Bernstein weights are nonnegative and sum to one:
                        # the control-difference hull bounds the whole image.
                        duplicate=index;break
                # Candidate coincidence is established by the analytic
                # predicate returning both interval endpoints, not midpoints.
                hits = curve_junctions(curve, old, tolerance=tolerance,cancellation_check=lambda:(check() or False))
                if any(abs(x) <= 1e-12 for x, _ in hits) and any(abs(x-1) <= 1e-12 for x, _ in hits):
                    if isinstance(curve,(CylinderIntersectionCurve,QuadricIntersectionCurve,BezierQuadricCurve)) or isinstance(
                            old,(CylinderIntersectionCurve,QuadricIntersectionCurve,BezierQuadricCurve)):
                        from .batch_intersections import _coincident
                        if _coincident(curve,old,tolerance):
                            duplicate=index; break
                    if isinstance(curve, EllipticArc) and isinstance(old, EllipticArc):
                        # Two isolated crossings are not coincident arcs.
                        n = np.cross(old.u_vector, old.v_vector)
                        inv = np.linalg.pinv(np.column_stack((old.u_vector, old.v_vector)))
                        q0 = inv @ (np.asarray(curve.center)-old.center)
                        qc, qs = inv @ curve.u_vector, inv @ curve.v_vector
                        if (abs(float(n @ (np.asarray(curve.center)-old.center))) <= tolerance
                                and np.linalg.norm(q0) <= tolerance
                                and abs(float(qc @ qc)-1) <= tolerance
                                and abs(float(qs @ qs)-1) <= tolerance
                                and abs(float(qc @ qs)) <= tolerance
                                and np.linalg.norm(curve.evaluate(.5)-old.evaluate(.5)) <= tolerance):
                            duplicate = index; break
            if duplicate is not None:
                old = edges[duplicate]
                edges[duplicate] = ArrangementPath(old.curve, old.source_edge or path.source_edge,
                    tuple(sorted(set((*old.owners, *path.owners)))), old.decomposition and path.decomposition,
                    tuple(sorted(set((*old.member_ids, *path.member_ids)))))
                continue
            index = len(edges)
            edges.append(replace(path, curve=curve))
            endpoints_by_edge.append((start, end))
            incidence[start].append(index); incidence[end].append(index)

    outgoing = [[] for _ in vertices]
    for index, ((start, end), path) in enumerate(zip(endpoints_by_edge, edges)):
        for vertex_id, forward in ((start, True), (end, False)):
            end_parameter = 0. if forward else 1.
            if domain.at_apex(path.curve.evaluate(end_parameter)):
                # Every edge leaves the collapsed apex edge upward; the plane embedding around it is the
                # order of the azimuths (counterclockwise in the chart: decreasing u).
                u = float(domain.uv(path.curve, end_parameter)[0])
                outgoing[vertex_id].append((.5*math.pi+.25*math.pi*(1-2*u), None, index, forward))
                continue
            tangent = domain.tangent(path.curve, end_parameter)*(1 if forward else -1)
            if np.linalg.norm(tangent) == 0:
                raise GeometryError("arrangement endpoint needs a regular curve chart")
            angle = math.atan2(tangent[1], tangent[0]) % math.tau
            if min(angle, math.tau-angle) < 64*np.finfo(float).eps:
                angle = 0.
            curvature = domain.curvature(path.curve, end_parameter, forward)
            outgoing[vertex_id].append((angle, curvature, index, forward))
    for values in outgoing:
        values.sort(key=lambda item: item[0])
        groups = []
        for value in values:
            if not groups or value[0]-groups[-1][0][0] > 64*np.finfo(float).eps:
                groups.append([])
            groups[-1].append(value)
        values.clear()
        for group in groups:
            if len(group) > 1 and any(value[1] is None for value in group):
                raise GeometryError("tangent junction needs a certified higher-order chart")
            values.extend(sorted(group, key=lambda item: (0. if item[1] is None else item[1], item[2:])))

    seen, cycles = set(), []
    for index in range(len(edges)):
        for forward in (True, False):
            start = (index, forward)
            if start in seen:
                continue
            chain, current = [], start
            while current not in seen:
                check()
                seen.add(current); chain.append(current)
                edge_id, direction = current
                vertex_id = endpoints_by_edge[edge_id][1 if direction else 0]
                values = outgoing[vertex_id]
                position = next(i for i, (_angle, _curvature, candidate, orientation) in enumerate(values)
                                if candidate == edge_id and orientation != direction)
                _angle, _curvature, candidate, orientation = values[(position-1) % len(values)]
                current = (candidate, orientation)
            if current != start:
                raise GeometryError("arrangement half-edge cycle is inconsistent")
            if len(chain) < 2 or len({edge_id for edge_id, _ in chain}) != len(chain):
                # Bridges indicate an unresolved interior endpoint; returning
                # an accepted face with a retraced boundary is prohibited.
                if len(chain) > 2:
                    raise GeometryError("arrangement contains an unresolved bridge")
                continue
            loop = tuple(ArrangementPath(edges[e].curve if direction else edges[e].curve.subcurve(1., 0.))
                         for e, direction in chain)
            area = domain.area_loop(loop, native_area_tolerance*.05)
            if abs(area) <= tolerance*tolerance:
                continue
            curve = loop[0].curve
            middle = domain.uv(curve, .5)
            tangent = domain.tangent(curve, .5)
            probe = domain.left_probe(middle, tangent, 2*tolerance)
            material = (domain.contains_loop(domain.boundaries[0], probe, tolerance, boundary=False)
                        and not any(domain.contains_loop(hole, probe, tolerance, boundary=True)
                                    for hole in domain.boundaries[1:]))
            if material:
                cycles.append((tuple(chain), loop, area))

    positive = [item for item in cycles if item[2] > 0]
    holes = [[] for _ in positive]
    for chain, loop, area in cycles:
        if area >= 0:
            continue
        curve = loop[0].curve
        point = domain.left_probe(domain.uv(curve,.5), domain.tangent(curve,.5), 2*tolerance)
        containers = [i for i, (_chain, outer, _area) in enumerate(positive)
                      if domain.contains_loop(outer, point, tolerance, boundary=False)]
        if not containers:
            raise GeometryError("arrangement hole has no material owner")
        owner = min(containers, key=lambda i: positive[i][2])
        holes[owner].append(chain)
    cells = tuple(ArrangementCell(chain, tuple(holes[index]))
                  for index, (chain, _loop, _area) in enumerate(positive))
    outer_area = domain.area_loop(domain.boundaries[0], native_area_tolerance*.05)
    original = abs(outer_area)-sum(
        abs(domain.area_loop(loop, native_area_tolerance*.05)) for loop in domain.boundaries[1:])
    retained = sum(item[2] for item in cycles)
    if not cells or abs(original-retained) > native_area_tolerance:
        raise GeometryError("arrangement failed material conservation")
    return MaterialArrangement(domain.face_id, tuple(edges), cells, retained, tolerance,
                               native_area_tolerance, domain.support, -1 if outer_area < 0 else 1)
