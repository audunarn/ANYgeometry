"""Read-only half-edge arrangements of exact material-domain curves.

Original domains and all physical traces are inputs together. Construction
seams connect interior-ended traces without extending the physical joint.
Topology is returned only after every junction and material cycle is resolved.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
import functools
import heapq
import itertools
import math
from contextlib import nullcontext
import numpy as np

from .arrangement_geometry import (LinePath, BezierPath, curve_junctions,
                                   plane_roots, point_parameters, freeze_edge, _junction_bounds_policy)
from .exact_curves import EllipticArc, CylinderIntersectionCurve
from .branch_curves import BezierQuadricCurve
from .quadric_curves import QuadricIntersectionCurve
from .errors import GeometryError
from .exact_predicates import correctly_rounded_sqrt, exact_dot, exact_vector
from .planar_exact import ExactPlane
from . import planar_stations as stations
from ._cross3 import _gauss_legendre
from .extruded_chart import area_density, chart_accelerations, chart_rates, density_bound, world_metric
from .extruded_supports import extruded_support
from .extrusions import EllipseDirectrix
from .surfaces import Cone, Cylinder, ExtrudedSurface, Plane, _angle_on_sweep
from .nurbs import NurbsSurface
from . import nurbs_domain

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
    certified: bool = False             # cells decided on a closed planar embedding (T-b/D8 E2/E3)


@dataclass(frozen=True, slots=True)
class MaterialDomain:
    face_id: int
    support: Plane | Cylinder | Cone | ExtrudedSurface | NurbsSurface
    boundaries: tuple[tuple[ArrangementPath, ...], ...]
    # Instance-lifetime reuse of the support's exact chart (see _exact_chart).
    _exact_chart_cache: tuple | None = field(default=None, init=False, hash=False,
                                             compare=False, repr=False, metadata={"definition": False})
    _exact_boundary_cache: tuple | None = field(default=None, init=False, hash=False,
                                                compare=False, repr=False, metadata={"definition": False})

    def _prepared_exact_boundary(self, loop):
        """Lazy exact line data, retained only for this immutable domain's edges."""
        if (type(self) is not MaterialDomain or type(self.support) is not Plane
                or any(vector.flags.writeable for vector in (
                    self.support.origin, self.support.u_vector, self.support.v_vector))
                or type(self.boundaries) is not tuple
                or not any(loop is item for item in self.boundaries)
                or type(loop) is not tuple):
            return None
        # LinePath is frozen, but callers can construct it with mutable payloads.
        # Such records continue to use the uncached exact predicates.
        if not all(type(path) is ArrangementPath and type(path.curve) is LinePath
                   and type(path.curve.start) is tuple and type(path.curve.end) is tuple
                   and len(path.curve.start) == 3 and len(path.curve.end) == 3
                   and all(type(value) in (float, int, np.float64, Fraction)
                           for value in (*path.curve.start, *path.curve.end))
                   for path in loop):
            return None
        cache = self._exact_boundary_cache
        if cache is None or cache[0] is not self.support or cache[1] is not self.boundaries:
            cache = (self.support, self.boundaries, {})
            object.__setattr__(self, "_exact_boundary_cache", cache)
        return cache[2].setdefault(id(loop), {})

    def _exact_chart(self):
        """The support's exact chart, reused across this domain's queries.

        ``ExactPlane.from_plane`` is a pure function of the immutable support
        record, so the first query's chart serves every later query of the
        same domain. The cache is identity-checked: a domain copied with a
        different support (``replace``) never reuses the stale chart, and a
        failed construction is never cached.
        """
        # Only the exact immutable owner record carries the reuse guarantee.
        if type(self.support) is not Plane:
            return ExactPlane.from_plane(self.support)
        cache = self._exact_chart_cache
        if cache is not None and cache[0] is self.support:
            return cache[1]
        chart = ExactPlane.from_plane(self.support)
        object.__setattr__(self, '_exact_chart_cache', (self.support, chart))
        return chart

    @property
    def area_jacobian(self):
        if isinstance(self.support, NurbsSurface):        # upper bound sigma_max^2 (A1e-lite)
            return nurbs_domain.area_jacobian(self)
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
        if isinstance(self.support, NurbsSurface):
            return nurbs_domain.world_delta(self, delta, v, u)
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
        if isinstance(self.support, NurbsSurface):        # oint G dv, G = int J du (A1e-lite)
            return nurbs_domain.world_area_loop(self, loop, tolerance)
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
        if not isinstance(self.support, (Cone, ExtrudedSurface, NurbsSurface)):
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
        from .nurbs_sweep_chart import has_sweep_chart, chart_model
        if has_sweep_chart(face):
            model = chart_model(model, face_id)
            face = model.faces[face_id]
        else:
            from .nurbs_sweep_chart import source_sweep_chart_model
            source_view = source_sweep_chart_model(model, face_id)
            if source_view is not None:
                model = source_view
                face = model.faces[face_id]
        support = face.surface
        if isinstance(support, NurbsSurface):              # the chart of the face's own support (A3b)
            extruded = None
        else:
            extruded = None if isinstance(support, (Cylinder, Cone)) else extruded_support(model, face_id)
        if extruded is not None:
            support = extruded
        elif not isinstance(support, (Cylinder, Cone, NurbsSurface)):
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
        if isinstance(self.support, NurbsSurface):        # the pcurve's chart point; never an inversion
            return nurbs_domain.uv(self, curve, parameter)
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
        if isinstance(self.support, NurbsSurface):
            return nurbs_domain.tangent(self, curve, parameter)
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
        if isinstance(self.support, NurbsSurface):
            return nurbs_domain.curvature(self, curve, parameter, forward)
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
        if isinstance(self.support, NurbsSurface):        # pcurve level roots (float route)
            return nurbs_domain.horizontal_roots(self, curve, value, tolerance)
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

    def _exact_planar(self, loops) -> bool:
        """Whether the exact planar route applies: a Plane support and straight loops only."""
        return isinstance(self.support, Plane) and all(
            type(path.curve) is LinePath for loop in loops for path in loop)

    def _exact_curved_planar(self, loops) -> bool:
        """Whether the exact line-and-arc route applies: a Plane support, lines and elliptic arcs (R1c)."""
        return isinstance(self.support, Plane) and all(
            type(path.curve) in (LinePath, EllipticArc) for loop in loops for path in loop)

    def _exact_station_uv(self, curve, parameter):
        """Exact chart point of a line/Bezier station, or None (other curves stay on the float route)."""
        from .exact_predicates import Station
        position = Station(curve, parameter).exact_position() if type(curve) in (LinePath, BezierPath) else None
        if position is None:
            return None
        return self._exact_chart().uv(position)

    def contains_loop(self, loop, uv, tolerance, *, boundary=True):
        from .support_curves import PlaneNurbsCurve
        if isinstance(self.support, Plane) and any(isinstance(path.curve, PlaneNurbsCurve) for paths in self.boundaries for path in paths):
            if all(isinstance(path.curve, LinePath) for path in loop):
                return _contains_loop_exact(self._exact_chart(), loop, tuple(map(Fraction, uv)), tolerance,
                                            boundary=boundary, prepared=self._prepared_exact_boundary(loop))
            from .plane_source_reference import contains_section_loop
            return contains_section_loop(self.support, loop, uv, tolerance, boundary=boundary)
        # Exact route (robust-predicate R1b): only for an exact (rational) query
        # point on an all-straight Plane domain. A float query point is a
        # rounded chart output, so it keeps the float route unchanged.
        if len(uv) == 2 and all(type(v) is Fraction for v in uv):
            if self._exact_planar(self.boundaries) and self._exact_planar((loop,)):
                return _contains_loop_exact(self._exact_chart(), loop, tuple(uv), tolerance,
                                            boundary=boundary,
                                            prepared=self._prepared_exact_boundary(loop))
            if self._exact_curved_planar(self.boundaries) and self._exact_curved_planar((loop,)):
                # Lines and elliptic arcs (robust-predicate R1c, E4): certified replica.
                from .planar_arcs import contains_loop_exact_curved
                return contains_loop_exact_curved(self._exact_chart(), loop, tuple(uv),
                                                  tolerance, boundary=boundary)
        if isinstance(self.support, NurbsSurface):        # exact level sets on the pcurves (A1e-lite)
            return nurbs_domain.contains_loop(self, loop, uv, tolerance, boundary=boundary)
        winding = 0
        for index, path in enumerate(loop):
            curve = path.curve
            roots = self.horizontal_roots(curve, float(uv[1]), tolerance)
            if roots is None:
                # A horizontal curve can contain the query but contributes no
                # ray crossing. Exact point inversion handles its boundary.
                if isinstance(self.support, NurbsSurface):
                    if nurbs_domain.on_horizontal(self, curve, uv, tolerance):
                        return boundary
                    continue
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
        uv = None
        if self._exact_planar(self.boundaries) or self._exact_curved_planar(self.boundaries):
            uv = self._exact_station_uv(curve, parameter)          # exact station (robust-predicate R1b/R1c)
        if uv is None:
            uv = self.uv(curve, parameter)
        return (self.contains_loop(self.boundaries[0], uv, tolerance, boundary=boundary)
                and not any(self.contains_loop(loop, uv, tolerance, boundary=not boundary)
                            for loop in self.boundaries[1:]))

    def area_loop(self, loop, tolerance):
        if isinstance(self.support, NurbsSurface):        # chart area on the pcurves (A1e-lite)
            return nurbs_domain.area_loop(self, loop, tolerance)
        if isinstance(self.support,Plane):
            from .polynomial_green import polynomial_planar_loop_area
            exact = polynomial_planar_loop_area(self.support,loop,tolerance)
            if exact is not None:
                return exact
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
        nodes, weights = _gauss_legendre(count)
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
    nodes, weights = _gauss_legendre(24)

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


def _contains_point_exact(domain, point, tolerance):
    """Material containment of an exact world point on an all-straight Plane domain (R1b)."""
    uv = domain._exact_chart().uv(point)
    return (domain.contains_loop(domain.boundaries[0], uv, tolerance)
            and not any(domain.contains_loop(loop, uv, tolerance, boundary=False) for loop in domain.boundaries[1:]))


def _clip_line_exact(domain, curve, tolerance, check, segment=None, *, deferred=True):
    """Exact stations of ``_clip`` for a straight curve on a Plane support (R1b step 2a).

    The same rule as ``_clip_intervals``: split at every junction with the
    domain boundary, merge numerical duplicates, keep the pieces whose
    midpoint is material. Junctions with straight boundaries and the
    midpoint test (on all-straight domains) are exact; junctions with curved
    boundaries keep the float predicate (their parameters enter as exact
    values of the returned doubles).
    """
    segment = stations.segment_of(curve) if segment is None else segment
    parameters = [Fraction(0), Fraction(1)]
    for loop in domain.boundaries:
        for boundary in loop:
            check()
            if type(boundary.curve) is LinePath:
                parameters.extend(a for a, _b in stations.junctions(
                    segment, stations.segment_of(boundary.curve), tolerance))
            else:
                parameters.extend(Fraction(float(a)) for a, _b in curve_junctions(
                    curve, boundary.curve, tolerance=tolerance, cancellation_check=lambda: (check() or False)))
    parameters = stations.merge_parameters(parameters, tolerance, segment, deferred=deferred)
    exact_domain = domain._exact_planar(domain.boundaries)
    pieces = []
    for a, b in zip(parameters, parameters[1:]):
        middle = (a + b) / 2
        inside = (_contains_point_exact(domain, stations.at(segment, middle), tolerance) if exact_domain
                  else domain.contains(curve, float(middle), tolerance))
        if inside:
            pieces.append(stations.sub(segment, a, b))
    return pieces


def _same_ray(first, second) -> bool:
    """Main's ray dedupe ``norm(unit_a - unit_b) <= 64 eps``; exactly for two exact directions.

    ``|u_a - u_b|^2 = 2 - 2 cos``, so the test is ``a . b >= k |a| |b|`` with
    ``k = 1 - (64 eps)^2 / 2``, decided through squares. Otherwise (a curved
    piece's tangent) main's float rule on the float units.
    """
    (exact_a, unit_a), (exact_b, unit_b) = first, second
    if exact_a is not None and exact_b is not None:
        dot = exact_dot(exact_a, exact_b)
        k = 1 - (64 * _EPS_EXACT) ** 2 / 2
        return dot >= 0 and dot * dot >= k * k * exact_dot(exact_a, exact_a) * exact_dot(exact_b, exact_b)
    return bool(np.linalg.norm(unit_a - unit_b) <= 64 * _EPS)


def _add_exact_planar_seams(domain, boundaries, paths, segments, points, tolerance, check, *, deferred=True):
    """Decomposition seams on a Plane support through exact stations (R1b step 2b, D6/D7).

    Main's rules: an interior trace end (on no boundary) with at most one ray
    gets two seams, along the authored ``u`` direction and its in-plane
    perpendicular, unless an earlier end lies within ``tol``. Straight pieces
    decide on their exact stations: boundary membership and incidence through
    the exact point parameters, rays as exact directions (dedupe through
    squares), end dedupe by exact distance; a seam is the exact segment through
    the station with exact rational directions ``u`` and ``(u x v) x u``. Points
    of curved pieces keep main's float rules.
    """
    tol2 = Fraction(tolerance) ** 2

    def on_boundary(exact_point, float_point):
        for index, boundary in enumerate(boundaries):
            if exact_point is not None and index in segments:
                hit = stations.point_parameters(segments[index], exact_point, tolerance)
            else:
                hit = point_parameters(boundary.curve, float_point, tolerance=tolerance)
            if hit:
                return True
        return False

    def near(first, second):
        if first[0] is not None and second[0] is not None:
            return stations.distance2(first[0], second[0]) <= tol2
        return bool(np.linalg.norm(first[1] - second[1]) <= tolerance)

    endpoints = []                                    # (exact station or None, float position)
    for point in points:
        item = (exact_vector(point.position), np.asarray(point.position))
        if not on_boundary(*item):
            endpoints.append(item)
    traces = range(len(boundaries), len(paths))
    for index in traces:
        for t in (0, 1):
            if index in segments:
                station = segments[index][t]
                item = (station, np.asarray(stations.rounded(station)))
            else:
                item = (None, paths[index].curve.evaluate(float(t)))
            if on_boundary(*item):
                continue
            rays = []
            for other in traces:
                exact_other = item[0] is not None and other in segments
                parameters = (stations.point_parameters(segments[other], item[0], tolerance) if exact_other
                              else point_parameters(paths[other].curve, item[1], tolerance=tolerance))
                for parameter in parameters:
                    if exact_other:
                        start, end = segments[other]
                        direction = tuple(b - a for a, b in zip(start, end))
                        if not any(direction):
                            raise GeometryError("interior junction needs a regular chart")
                        # Exact non-zero direction; its float unit (for comparisons with curved
                        # pieces only) from the direction scaled to unit max-norm, so no underflow.
                        largest = max(abs(value) for value in direction)
                        tangent = np.asarray(stations.rounded(tuple(value / largest for value in direction)))
                        unit = tangent/np.linalg.norm(tangent)
                    else:
                        direction, tangent = None, paths[other].curve.derivative(parameter)
                        length = np.linalg.norm(tangent)
                        if length == 0:
                            raise GeometryError("interior junction needs a regular chart")
                        unit = tangent/length
                    for sign in ((-1,) if parameter == 1 else (1,) if parameter == 0 else (-1, 1)):
                        ray = (None if direction is None else tuple(sign*value for value in direction), sign*unit)
                        if not any(_same_ray(ray, old) for old in rays):
                            rays.append(ray)
            if len(rays) > 1:
                continue
            if not any(near(item, old) for old in endpoints):
                endpoints.append(item)

    support = domain.support
    boxes = [path.curve.bounds() for path in boundaries]
    lower = np.min([box[0] for box in boxes], axis=0)
    upper = np.max([box[1] for box in boxes], axis=0)
    extent = max(float(np.linalg.norm(upper-lower)), 1.)
    normal = np.cross(support.u_vector, support.v_vector)
    normal /= np.linalg.norm(normal)
    authored = np.asarray(support.u_vector)/np.linalg.norm(support.u_vector)
    # Main's seam: through the end point along the authored direction and its
    # physical perpendicular, as float unit vectors scaled by ``extent``. The
    # same parameterization is kept (clipping's 32-eps parameter windows depend
    # on it); the seam passes exactly through the station.
    offsets = [exact_vector(extent*direction) for direction in (authored, np.cross(normal, authored))]
    for station, position in sorted(endpoints, key=lambda item: tuple(item[1])):   # main's (stable) order
        point = station if station is not None else exact_vector(position)
        for offset in offsets:
            seam = stations.Segment(tuple(p - d for p, d in zip(point, offset)),
                                    tuple(p + d for p, d in zip(point, offset)), derived=True)
            for piece in _clip_line_exact(domain, stations.to_line(seam), tolerance, check, segment=seam,
                                          deferred=deferred):
                segments[len(paths)] = piece
                paths.append(ArrangementPath(stations.to_line(piece), decomposition=True))


def _exact_outgoing_direction(chart, edge_segments, item):
    start, end = edge_segments[item[2]]
    vector = tuple(b - a for a, b in zip(start, end)) if item[3] else tuple(a - b for a, b in zip(start, end))
    return chart.uv_direction(vector)


def _compare_outgoing(chart, edge_segments, first, second) -> int:
    """Counterclockwise order of two straight outgoing edges from the chart's +u axis, exactly.

    Directions with ``v > 0`` or (``v == 0`` and ``u > 0``) come first (angles
    in ``[0, pi)``), then by the sign of the cross product; exactly equal
    directions fall back to main's tie key ``(curvature, index, forward)``.
    """
    (ua, va), (ub, vb) = (_exact_outgoing_direction(chart, edge_segments, item) for item in (first, second))
    half_a = 0 if va > 0 or (va == 0 and ua > 0) else 1
    half_b = 0 if vb > 0 or (vb == 0 and ub > 0) else 1
    if half_a != half_b:
        return half_a - half_b
    cross = ua * vb - va * ub
    if cross != 0:
        return -1 if cross > 0 else 1
    key_a, key_b = ((0. if item[1] is None else item[1], item[2:]) for item in (first, second))
    return (key_a > key_b) - (key_a < key_b)


def _exact_loop_area(chart, pieces) -> Fraction:
    """Main's ``planar_loop_area`` for straight pieces, on exact stations.

    The sum over the pieces of ``cross(p(start), p(end)) / 2`` with ``p`` the
    chart coordinates relative to the first piece's start -- main's formula
    (it does not assume that consecutive pieces share their stations exactly;
    tolerance-merged vertices keep their own stations).
    """
    anchor = pieces[0][0]
    total = Fraction(0)
    for start, end in pieces:
        a = chart.uv_direction(tuple(x - y for x, y in zip(start, anchor)))
        b = chart.uv_direction(tuple(x - y for x, y in zip(end, anchor)))
        total += a[0] * b[1] - a[1] * b[0]
    return total / 2


def _exact_planar_cells(domain, chains, edges, edge_segments, tolerance, native_area_tolerance, check, *,
                        boundary_uses=None):
    """Main's cell phase on exact stations for an all-straight planar arrangement (R1b step 2d, D3/D5).

    An exact replica of main's rules, so that no decision depends on the
    rounding of evaluations or charts: cycle areas by main's formula on the
    pieces' exact stations, the ``|area| <= tol^2`` drop, main's probe
    (``32 tol`` physical along the chart left normal at the first piece's
    midpoint; the physical length is the correctly rounded norm of the exact
    world normal), main's tolerance-based material test and hole ownership
    (exact containment on the exact loops) and the conservation check, all
    in exact arithmetic. The probe is *not* certified to lie in its cell (main
    does not certify it either; a certificate needs an exactly closed planar
    embedding, which tolerance-merged vertices do not provide -- open item T-b).
    """
    chart = ExactPlane.from_plane(domain.support)
    u_vector, v_vector = exact_vector(domain.support.u_vector), exact_vector(domain.support.v_vector)

    def pieces(chain):
        return [edge_segments[e] if forward else edge_segments[e][::-1] for e, forward in chain]

    def probe_of(chain):
        start, end = pieces(chain)[0]
        middle = chart.uv(tuple((a + b) / 2 for a, b in zip(start, end)))
        du, dv = chart.uv_direction(tuple(b - a for a, b in zip(start, end)))
        normal = (-dv, du)
        world = tuple(normal[0] * a + normal[1] * b for a, b in zip(u_vector, v_vector))
        length = Fraction(correctly_rounded_sqrt(exact_dot(world, world)))
        if length == 0:
            raise GeometryError("arrangement endpoint needs a regular curve chart")
        scale = 32 * Fraction(tolerance) / length                 # left_probe(middle, tangent, 2 tol)
        return (middle[0] + scale * normal[0], middle[1] + scale * normal[1])

    def straight_loop(chain):
        return tuple(ArrangementPath(LinePath(*piece)) for piece in pieces(chain))

    tol4 = Fraction(tolerance) ** 4
    cycles = []
    for chain in chains:
        check()
        area = _exact_loop_area(chart, pieces(chain))
        if area * area <= tol4:                                  # |area| <= tol^2
            continue
        probe = probe_of(chain)
        material = (domain.contains_loop(domain.boundaries[0], probe, tolerance, boundary=False)
                    and not any(domain.contains_loop(hole, probe, tolerance, boundary=True)
                                for hole in domain.boundaries[1:]))
        if material:
            cycles.append((chain, area, probe))
    positive = [item for item in cycles if item[1] > 0]
    holes = [[] for _ in positive]
    for chain, area, probe in cycles:
        if area >= 0:
            continue
        containers = [i for i, (outer, _area, _probe) in enumerate(positive)
                      if _contains_loop_exact(chart, straight_loop(outer), probe, tolerance, boundary=False)]
        if not containers:
            raise GeometryError("arrangement hole has no material owner")
        owner = min(containers, key=lambda i: positive[i][1])
        holes[owner].append(chain)
    cells = tuple(ArrangementCell(chain, tuple(holes[index])) for index, (chain, _area, _probe) in enumerate(positive))

    def boundary_area(loop):
        return _exact_loop_area(chart, [stations.segment_of(path.curve) for path in loop])
    outer_area = boundary_area(domain.boundaries[0])
    original = abs(outer_area) - sum(abs(boundary_area(loop)) for loop in domain.boundaries[1:])
    if boundary_uses is not None:
        # E2 (S9): the cells partition the re-anchored domain, so conservation compares with the
        # re-anchored loops' exact areas; their deformation from the stored loops is bounded by
        # 2 tol * perimeter (owner Q-b), refused otherwise.
        loop_of, first = [], 0
        for loop in domain.boundaries:
            loop_of.append(range(first, first + len(loop)))
            first += len(loop)
        anchored = []
        for paths_range in loop_of:
            pieces = [edge_segments[index] if same else edge_segments[index][::-1]
                      for path_index, index, same in boundary_uses if path_index in paths_range]
            anchored.append(_exact_loop_area(chart, pieces) if pieces else Fraction(0))
        perimeter = sum(math.sqrt(float(stations.segment_of(path.curve).length2()))
                        for loop in domain.boundaries for path in loop)
        stored_areas = [outer_area] + [boundary_area(loop) for loop in domain.boundaries[1:]]
        if any(abs(float(a - b)) > 2 * tolerance * perimeter * (1 + 1e-9) + 1e-300
               for a, b in zip(anchored, stored_areas)):
            raise GeometryError("arrangement deformation exceeds the tolerance contract")
        original = abs(anchored[0]) - sum(abs(value) for value in anchored[1:])
    retained = sum((item[1] for item in cycles), Fraction(0))
    if not cells or abs(original - retained) > Fraction(native_area_tolerance):
        raise GeometryError("arrangement failed material conservation")
    return cells, float(retained), outer_area

def _segments_meet(p, q, r, t) -> bool:
    """Closed 2-D segments ``[p, q]`` and ``[r, t]`` share a point (exact)."""
    def orient(a, b, c):
        value = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return (value > 0) - (value < 0)

    def on(a, b, c):                                   # c on [a, b], given collinear
        return min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= c[1] <= max(a[1], b[1])
    o1, o2, o3, o4 = orient(p, q, r), orient(p, q, t), orient(r, t, p), orient(r, t, q)
    if o1 != o2 and o3 != o4 and 0 not in (o1, o2, o3, o4):
        return True
    return ((o1 == 0 and on(p, q, r)) or (o2 == 0 and on(p, q, t))
            or (o3 == 0 and on(r, t, p)) or (o4 == 0 and on(r, t, q)))


def _edges_conflict(first, second, first_vertices, second_vertices) -> bool:
    """Whether two re-anchored straight edges (chart points) violate a planar embedding.

    Edges may meet only at shared vertices. Sharing one vertex, they conflict
    when collinear and pointing the same way (overlap, or one containing the
    other's far end); sharing none, when their closed segments meet at all
    (proper crossing, T-junction, collinear overlap).
    """
    shared = set(first_vertices) & set(second_vertices)
    if len(shared) == 2:
        return True                                    # duplicates are merged before; a second copy conflicts
    if len(shared) == 1:
        (vertex,) = shared
        p = first[first_vertices.index(vertex)]
        q = first[1 - first_vertices.index(vertex)]
        r = second[1 - second_vertices.index(vertex)]
        cross = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        dot = (q[0] - p[0]) * (r[0] - p[0]) + (q[1] - p[1]) * (r[1] - p[1])
        return cross == 0 and dot > 0
    return _segments_meet(first[0], first[1], second[0], second[1])


def _close_embedding(domain, vertices, lookup, stored_of, edges, edge_segments, endpoints_by_edge, check):
    """Representatives, re-anchoring and the conservative closure check (T-b/D8 slice E2: S5, S6, S9).

    Every vertex gets one exact position: the stored face vertex in its
    cluster when there is one (owner Q-a), else its first station; two stored
    vertices in one cluster refuse. Every straight edge becomes the segment
    between its vertices' positions (stored coordinates = their correct
    rounding). Distinct vertices must keep distinct exact and stored
    positions. The re-anchored edges must form a planar embedding: any
    crossing, T-junction or overlap refuses ("arrangement not closable within
    tolerance") -- the conservative policy (owner Q-g). Inputs must lie on the
    support within tol (S9).
    """
    chart = ExactPlane.from_plane(domain.support)
    normal = chart.normal
    bound2 = Fraction(lookup.tolerance) ** 2 * exact_dot(normal, normal)
    offset = exact_dot(normal, chart.origin)
    representatives = []
    for index in range(len(vertices)):
        stored = stored_of.get(index, {})
        loops_seen = [corner[0] for corner in stored]
        if len(set(loops_seen)) != len(loops_seen):
            # two corners of one loop in one cluster would collapse its edge (main: start == end)
            raise GeometryError("two stored face vertices within one tolerance cluster")
        # corners of different loops may touch (touching holes): the earliest loop's corner
        point = stored[min(stored)] if stored else lookup.exact[index]
        residual = exact_dot(normal, point) - offset
        if residual * residual > bound2:
            raise GeometryError("material domain input off its support")
        representatives.append(point)
    exact_seen, rounded_seen = {}, {}
    for index, point in enumerate(representatives):
        rounded = stations.rounded(point)
        if point in exact_seen or rounded in rounded_seen:
            raise GeometryError("arrangement not closable within tolerance")
        exact_seen[point] = rounded_seen[rounded] = index
        vertices[index] = np.asarray(rounded)
    for index, (start, end) in enumerate(endpoints_by_edge):
        if index in edge_segments:
            segment = stations.Segment(representatives[start], representatives[end], derived=True)
            edge_segments[index] = segment
            edges[index] = replace(edges[index], curve=stations.to_line(segment))
    uv = [chart.uv(point) for point in representatives]
    straight = sorted(edge_segments)
    if len(straight) < 2:
        return
    ends = np.array([[[float(c) for c in uv[endpoints_by_edge[index][k]]] for k in (0, 1)] for index in straight])
    lo, hi = ends.min(axis=1), ends.max(axis=1)
    pad = 4 * _EPS * np.maximum(np.abs(lo), np.abs(hi)) + 1e-300
    lo, hi = lo - pad, hi + pad
    for i, first in enumerate(straight):
        check()
        candidates = np.flatnonzero(np.all((lo[i + 1:] <= hi[i]) & (hi[i + 1:] >= lo[i]), axis=1)) + i + 1
        first_vertices = endpoints_by_edge[first]
        first_points = (uv[first_vertices[0]], uv[first_vertices[1]])
        for j in candidates:
            second = straight[j]
            second_vertices = endpoints_by_edge[second]
            if _edges_conflict(first_points, (uv[second_vertices[0]], uv[second_vertices[1]]),
                               first_vertices, second_vertices):
                raise GeometryError("arrangement not closable within tolerance")


def _shoelace(points) -> Fraction:
    """Exact signed area of a closed polygon of exact chart points."""
    return sum((a[0] * b[1] - a[1] * b[0] for a, b in zip(points, points[1:] + points[:1])), Fraction(0)) / 2


def _certified_planar_cells(domain, chains, edges, edge_segments, endpoints_by_edge, boundary_uses,
                            tolerance, native_area_tolerance, check):
    """Cells of a closed planar embedding, decided combinatorially (T-b/D8 slice E3: S8, S9).

    The embedding is closed and planar (E2), so faces are exact regions:
    * validity (typed "invalid material domain"): every loop simple on its
      stored stations and after re-anchoring (no vertex visited twice),
      non-zero area, no coincident loop occurrences (owner Q-c), and open-edge
      containment (hole-edge midpoints strictly inside the outer loop and
      outside other holes; outer-edge midpoints strictly outside every hole);
    * hole ownership: the open horizontal segment leaving a negative cycle's
      leftmost vertex toward -U lies in the face on its left; the nearest
      hit (an edge interior: the half-edge with the vertex on its left; a
      vertex: the cycle of the wedge containing +U there) names that face;
    * material by parity over the faces: the unbounded face is not material,
      crossing an edge flips the state iff its boundary multiplicity is odd;
    * exact areas, main's |area| <= tol^2 sliver drop (owner Q-d), the exact
      partition identity and main's conservation check.
    Main's probe is not used.
    """
    from .exact_predicates import Outcome, point_in_polygon
    from .planar_exact import region_is_simple
    chart = ExactPlane.from_plane(domain.support)
    vertex_count = 1 + max(max(pair) for pair in endpoints_by_edge) if endpoints_by_edge else 0
    vertex_uv = [None] * vertex_count
    for index, (start, end) in enumerate(endpoints_by_edge):
        a, b = edge_segments[index]
        vertex_uv[start], vertex_uv[end] = chart.uv(a), chart.uv(b)

    def invalid():
        return GeometryError("invalid material domain")

    # -- validity
    multiplicity = {}
    for _path, index, _same in boundary_uses:
        multiplicity[index] = multiplicity.get(index, 0) + 1
    if any(count > 1 for count in multiplicity.values()):
        raise invalid()
    loops_uv, first = [], 0
    for loop in domain.boundaries:
        stored = [chart.uv(stations.segment_of(path.curve)[0]) for path in loop]
        if not region_is_simple([stored]):
            raise invalid()
        uses = [(index, same) for path, index, same in boundary_uses if first <= path < first + len(loop)]
        sequence = [endpoints_by_edge[index][0 if same else 1] for index, same in uses]
        if len(set(sequence)) != len(sequence) or len(sequence) < 3:
            raise invalid()
        polygon = [vertex_uv[v] for v in sequence]
        if _shoelace(polygon) == 0:
            raise invalid()
        loops_uv.append((polygon, uses))
        first += len(loop)

    def midpoint(index):
        a, b = (vertex_uv[v] for v in endpoints_by_edge[index])
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    outer_polygon, outer_uses = loops_uv[0]
    # S9 (ii): the re-anchored loops deform the stored ones by at most 2 tol x perimeter (owner Q-b).
    perimeter = sum(math.sqrt(float(stations.segment_of(path.curve).length2()))
                    for loop in domain.boundaries for path in loop)
    bound = Fraction(2 * tolerance * perimeter * (1 + 1e-9))
    jacobian2 = exact_dot(chart.normal, chart.normal)          # world area = chart area * |u x v|
    for (polygon, _uses), loop in zip(loops_uv, domain.boundaries):
        stored_area = _exact_loop_area(chart, [stations.segment_of(path.curve) for path in loop])
        difference = abs(_shoelace(polygon)) - abs(stored_area)
        if difference * difference * jacobian2 > bound * bound:
            raise GeometryError("arrangement deformation exceeds the tolerance contract")
    for k, (polygon, uses) in enumerate(loops_uv[1:], start=1):
        for index, _same in uses:
            point = midpoint(index)
            if point_in_polygon(point, outer_polygon) is not Outcome.ABOVE:
                raise invalid()
            if any(point_in_polygon(point, other) is not Outcome.BELOW
                   for j, (other, _uses) in enumerate(loops_uv[1:], start=1) if j != k):
                raise invalid()
    for index, _same in outer_uses:
        point = midpoint(index)
        if any(point_in_polygon(point, hole) is not Outcome.BELOW for hole, _uses in loops_uv[1:]):
            raise invalid()

    # -- cycles and the half-edge -> cycle map
    cycle_of = {}
    areas, polygons = [], []
    for c, chain in enumerate(chains):
        for item in chain:
            cycle_of[item] = c
        polygon = [vertex_uv[endpoints_by_edge[e][0 if forward else 1]] for e, forward in chain]
        polygons.append(polygon)
        areas.append(_shoelace(polygon))

    outgoing = {}
    for index, (start, end) in enumerate(endpoints_by_edge):
        outgoing.setdefault(start, []).append((index, True))
        outgoing.setdefault(end, []).append((index, False))

    def direction(half):
        index, forward = half
        a, b = (vertex_uv[v] for v in endpoints_by_edge[index])
        return (b[0] - a[0], b[1] - a[1]) if forward else (a[0] - b[0], a[1] - b[1])

    def angle_key(half):
        du, dv = direction(half)
        return 0 if dv > 0 or (dv == 0 and du > 0) else 1

    def ccw(first, second):
        (ua, va), (ub, vb) = direction(first), direction(second)
        ha, hb = angle_key(first), angle_key(second)
        if ha != hb:
            return ha - hb
        cross = ua * vb - va * ub
        return -1 if cross > 0 else 1 if cross < 0 else 0

    def wedge_cycle(w):
        # The face in the wedge at w containing +U: left of the outgoing half-edge pointing exactly
        # +U if there is one (the wedge above it), else left of the last outgoing half-edge.
        halves = sorted(outgoing.get(w, ()), key=functools.cmp_to_key(ccw))
        if not halves:
            raise GeometryError("arrangement ownership ray met an isolated vertex")
        du, dv = direction(halves[0])
        chosen = halves[0] if (dv == 0 and du > 0) else halves[-1]
        return cycle_of[chosen]

    owner_cache = {}

    def owner(c):
        """The positive cycle bounding the face on the left of negative cycle ``c`` (None: unbounded)."""
        if c in owner_cache:
            return owner_cache[c]
        chain = chains[c]
        start_vertices = [endpoints_by_edge[e][0 if forward else 1] for e, forward in chain]
        v = min(start_vertices, key=lambda w: (vertex_uv[w][0], vertex_uv[w][1]))
        uv_v, vv = vertex_uv[v], vertex_uv[v][1]
        best = None                                   # (U of the hit, kind, payload)
        for w in range(vertex_count):
            if w == v or vertex_uv[w] is None:
                continue
            if vertex_uv[w][1] == vv and vertex_uv[w][0] < uv_v[0] and (best is None or vertex_uv[w][0] > best[0]):
                best = (vertex_uv[w][0], "vertex", w)
        for index, (a_id, b_id) in enumerate(endpoints_by_edge):
            check()
            if v in (a_id, b_id):
                continue
            a, b = vertex_uv[a_id], vertex_uv[b_id]
            if not (min(a[1], b[1]) < vv < max(a[1], b[1])):
                continue
            u = a[0] + (vv - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if u < uv_v[0] and (best is None or u > best[0]):
                best = (u, "edge", index)
        if best is None:
            result = None
        else:
            _u, kind, payload = best
            if kind == "vertex":
                hit = wedge_cycle(payload)
            else:
                a, b = (vertex_uv[x] for x in endpoints_by_edge[payload])
                left = (b[0] - a[0]) * (uv_v[1] - a[1]) - (b[1] - a[1]) * (uv_v[0] - a[0])
                hit = cycle_of[(payload, left > 0)]
            result = hit if areas[hit] > 0 else owner(hit)
        owner_cache[c] = result
        return result

    # -- faces and parity
    face_of_cycle = {c: (c if areas[c] > 0 else owner(c)) for c in range(len(chains))}
    material = {None: False}
    pending = [None]
    neighbours = {}
    for index in range(len(edges)):
        f1, f2 = face_of_cycle[cycle_of[(index, True)]], face_of_cycle[cycle_of[(index, False)]]
        flip = multiplicity.get(index, 0) % 2 == 1
        if f1 == f2 and flip:
            raise invalid()
        neighbours.setdefault(f1, []).append((f2, flip))
        neighbours.setdefault(f2, []).append((f1, flip))
    while pending:
        face = pending.pop()
        for other, flip in neighbours.get(face, ()):
            state = material[face] != flip
            if other in material:
                if material[other] != state:
                    raise invalid()
                continue
            material[other] = state
            pending.append(other)

    # -- cells, areas, conservation
    tol4 = Fraction(tolerance) ** 4
    positive = [c for c in range(len(chains)) if areas[c] > 0 and material.get(c, False)
                and areas[c] * areas[c] > tol4]
    index_of = {c: k for k, c in enumerate(positive)}
    holes = [[] for _ in positive]
    retained = sum((areas[c] for c in positive), Fraction(0))
    for c in range(len(chains)):
        if areas[c] < 0 and areas[c] * areas[c] > tol4:
            face = face_of_cycle[c]
            if face is not None and face in index_of:
                holes[index_of[face]].append(chains[c])
                retained += areas[c]
    cells = tuple(ArrangementCell(chains[c], tuple(holes[index_of[c]])) for c in positive)
    outer_area = _shoelace(outer_polygon)
    original = abs(outer_area) - sum(abs(_shoelace(polygon)) for polygon, _uses in loops_uv[1:])
    partition = sum((areas[c] for c in range(len(chains))
                     if material.get(face_of_cycle[c], False) and (areas[c] > 0 or face_of_cycle[c] is not None)),
                    Fraction(0))
    if partition != original:
        raise GeometryError("arrangement failed material conservation")
    stored_outer = _exact_loop_area(chart, [stations.segment_of(path.curve) for path in domain.boundaries[0]])
    if not cells or abs(original - retained) > Fraction(native_area_tolerance):
        raise GeometryError("arrangement failed material conservation")
    return cells, float(retained), stored_outer if stored_outer != 0 else outer_area


def _check_support(domain, points, tolerance):
    chart = ExactPlane.from_plane(domain.support)
    bound2 = Fraction(tolerance) ** 2 * exact_dot(chart.normal, chart.normal)
    offset = exact_dot(chart.normal, chart.origin)
    for point in points:
        residual = exact_dot(chart.normal, point) - offset
        if residual * residual > bound2:
            raise GeometryError("material domain input off its support")


def loop_lobes(sequence, coordinate):
    """Split a closed vertex walk into simple lobes at repeated vertices; their signed areas.

    ``sequence`` lists the walk's vertices (any hashable identity), ``coordinate``
    maps a vertex to exact chart coordinates. A weakly simple walk (E3 absorbs a
    touching hole into the outer walk) splits into its outer and hole lobes.
    """
    stack, lobes = [], []
    for vertex in list(sequence) + [sequence[0]]:
        if vertex in stack:
            start = stack.index(vertex)
            lobe = stack[start:]
            if len(lobe) >= 3:
                lobes.append(_shoelace([coordinate(v) for v in lobe]))
            del stack[start + 1:]
        else:
            stack.append(vertex)
    return lobes


def expected_cell_signatures(arrangement):
    """Each cell loop's (winding, lobe signs) from the arrangement's own geometry, as walked.

    Compared with the persisted loops recorded before application reverses the
    loops of a clockwise stored face, so no orientation flip applies here.
    """
    chart = ExactPlane.from_plane(arrangement.support)

    def walk(loop):
        starts = []
        for index, forward in loop:
            curve = arrangement.paths[index].curve
            starts.append(tuple(curve.start) if forward else tuple(curve.end))
        coordinate = lambda key: chart.uv(exact_vector(key))
        area = _shoelace([coordinate(key) for key in starts])
        return (area > 0, sorted(lobe > 0 for lobe in loop_lobes(starts, coordinate)))
    return [tuple(walk(list(loop)) for loop in (cell.outer, *cell.holes)) for cell in arrangement.cells]


def certify_persisted_cells(support, loops, edge_ends, position, check=lambda: None, expected=None):
    """Re-certify the persisted cells of a certified arrangement (T-b/D8 slice E2a, S10).

    ``loops`` are the applied cells as ``(outer, holes)`` lists of
    ``(edge id, forward)``; ``edge_ends(edge)`` gives the current model vertex
    ids of a straight edge; ``position(vertex)`` its stored doubles. Called after
    the whole batch's vertex mapping. On the exact chart of the stored
    coordinates the cells must form a planar subdivision:
    * no collapsed edge; the edges form a planar embedding (no crossing,
      T-junction or overlap);
    * each oriented edge is used at most once (adjacent cells traverse a
      shared edge in opposite directions; equal use means overlapping cells);
    * every loop has non-zero area; outer loops share one sign, holes have
      the opposite one (loops may touch themselves at a vertex, as E3's walk
      produces for touching holes);
    * nesting: the midpoint of every edge of a cell's loops that is not an
      edge of another cell lies outside that other cell's region (outer loop
      minus holes); every hole lies inside its own outer loop and outside the
      cell's other holes.
    Raises ``GeometryError("certified embedding not preserved on application (...)")``.
    """
    from .exact_predicates import Outcome, point_in_polygon
    chart = ExactPlane.from_plane(support)

    def failure(reason):
        return GeometryError(f"certified embedding not preserved on application ({reason})")
    uv = {}

    def point(vertex):
        if vertex not in uv:
            uv[vertex] = chart.uv(exact_vector(position(vertex)))
        return uv[vertex]

    edges = sorted({edge for outer, holes in loops for loop in (outer, *holes) for edge, _ in loop})
    try:
        ends = {edge: tuple(edge_ends(edge)) for edge in edges}
    except KeyError:
        raise failure("edge removed") from None
    if any(a == b or point(a) == point(b) for a, b in ends.values()):
        raise failure("collapsed edge")
    floats = {edge: np.array([[float(c) for c in point(v)] for v in ends[edge]]) for edge in edges}
    if len(edges) > 1:
        boxes = np.array([floats[edge] for edge in edges])
        lo, hi = boxes.min(axis=1), boxes.max(axis=1)
        pad = 4 * _EPS * np.maximum(np.abs(lo), np.abs(hi)) + 1e-300
        lo, hi = lo - pad, hi + pad
        for i, first in enumerate(edges):
            check()
            for j in np.flatnonzero(np.all((lo[i + 1:] <= hi[i]) & (hi[i + 1:] >= lo[i]), axis=1)) + i + 1:
                second = edges[j]
                if _edges_conflict(tuple(point(v) for v in ends[first]), tuple(point(v) for v in ends[second]),
                                   ends[first], ends[second]):
                    raise failure("edges cross or overlap")
    used = set()
    for outer, holes in loops:
        in_cell = set()
        for loop in (outer, *holes):
            for edge, forward in loop:
                if (edge, forward) in used:
                    raise failure("overlapping cells")
                if edge in in_cell:
                    raise failure("slit in a cell")        # one edge both ways in one cell
                used.add((edge, forward))
                in_cell.add(edge)

    def start_of(item):
        edge, forward = item
        return ends[edge][0 if forward else 1]

    def polygon(loop):
        if len(loop) < 3:
            raise failure("degenerate loop")
        return [point(start_of(item)) for item in loop]

    cells = []
    for outer, holes in loops:
        outer_polygon = polygon(outer)
        hole_polygons = [polygon(hole) for hole in holes]
        cells.append((outer, holes, outer_polygon, hole_polygons))
    if expected is not None:
        # The classification must be the certified one: each loop's winding and its lobes (signs of
        # the simple lobes of a weakly simple walk) as certified before application.
        for (outer, holes), wanted in zip(loops, expected):
            for loop, want in zip((outer, *holes), wanted):
                lobes = loop_lobes([start_of(item) for item in loop], point)
                signature = (_shoelace(polygon(loop)) > 0, sorted(lobe > 0 for lobe in lobes))
                if signature != want:
                    raise failure("winding or lobes changed")
    touching_holes = {}
    hole_lobes = {}                     # compound hole walk -> its simple lobes of the walk's sense
    for outer, holes in loops:
        for loop in (outer, *holes):
            lobes = []
            sequence = [start_of(item) for item in loop]
            stack = []
            for vertex in sequence + [sequence[0]]:
                if vertex in stack:
                    start = stack.index(vertex)
                    lobe = stack[start:]
                    if len(lobe) >= 3:
                        lobes.append([point(v) for v in lobe])
                    del stack[start + 1:]
                else:
                    stack.append(vertex)
            if len(lobes) > 1:
                total = _shoelace(polygon(loop)) > 0
                main_lobes = [lobe for lobe in lobes if (_shoelace(lobe) > 0) == total]
                for lobe in lobes:
                    if (_shoelace(lobe) > 0) != total:
                        # an opposite lobe is a hole touching its loop: inside a same-sense lobe
                        probe = ((lobe[0][0] + lobe[1][0]) / 2, (lobe[0][1] + lobe[1][1]) / 2)
                        if not any(point_in_polygon(probe, other) is Outcome.ABOVE for other in main_lobes):
                            raise failure("touching lobe outside its loop")
                        touching_holes.setdefault(id(outer), []).append(lobe)
                if loop is not outer:
                    hole_lobes[id(loop)] = main_lobes
    areas = [(_shoelace(op), [_shoelace(hp) for hp in hps]) for _o, _h, op, hps in cells]
    if any(a == 0 for outer_area, hole_areas in areas for a in (outer_area, *hole_areas)):
        raise failure("zero-area loop")
    signs = {outer_area > 0 for outer_area, _ in areas}
    if len(signs) != 1:
        raise failure("area signs")
    outer_sign = signs.pop()
    if any((a > 0) == outer_sign for _, hole_areas in areas for a in hole_areas):
        raise failure("hole winding")

    def midpoint(edge):
        a, b = (point(v) for v in ends[edge])
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

    cell_edges = [{edge for loop in (outer, *holes) for edge, _ in loop} for outer, holes, _op, _hp in cells]
    cell_boxes = []
    for _o, _h, outer_polygon, _hp in cells:
        coordinates = np.array([[float(c) for c in p] for p in outer_polygon])
        pad = 4 * _EPS * float(np.max(np.abs(coordinates))) + 1e-300
        cell_boxes.append((coordinates.min(axis=0) - pad, coordinates.max(axis=0) + pad))

    def inside_region(index, probe):
        _o, _h, outer_polygon, hole_polygons = cells[index]
        if point_in_polygon(probe, outer_polygon) is not Outcome.ABOVE:
            return False
        return not any(point_in_polygon(probe, hole) is not Outcome.BELOW for hole in hole_polygons)

    for k, (outer, holes, outer_polygon, hole_polygons) in enumerate(cells):
        for h, hole in enumerate(holes):
            free = [edge for edge, _ in hole if edge not in {e for e, _ in outer}]
            if not free:
                raise failure("hole on its outer loop")
            probe = midpoint(free[0])
            if point_in_polygon(probe, outer_polygon) is not Outcome.ABOVE:
                raise failure("hole outside its outer loop")
            if any(point_in_polygon(probe, other) is Outcome.ABOVE
                   for g, other in enumerate(hole_polygons) if g != h):
                raise failure("hole inside another hole")
    for outer, holes, _outer_polygon, hole_polygons in cells:
        # Hole regions of a cell -- its hole loops (the simple lobes of a compound hole walk), and
        # the touching-hole lobes of its outer walk -- never lie inside one another (they may share
        # vertices only).
        regions = [region for hole, hole_polygon in zip(holes, hole_polygons)
                   for region in hole_lobes.get(id(hole), [hole_polygon])]
        regions += touching_holes.get(id(outer), [])
        for a, region in enumerate(regions):
            probe = ((region[0][0] + region[1][0]) / 2, (region[0][1] + region[1][1]) / 2)
            if any(point_in_polygon(probe, other) is Outcome.ABOVE for b, other in enumerate(regions) if b != a):
                raise failure("hole inside another hole")
    for j, (outer, holes, _op, _hp) in enumerate(cells):
        for edge in sorted(cell_edges[j]):
            check()
            probe = midpoint(edge)
            fx, fy = float(probe[0]), float(probe[1])
            for i, (low, high) in enumerate(cell_boxes):
                if i == j or edge in cell_edges[i]:
                    continue
                if not (low[0] <= fx <= high[0] and low[1] <= fy <= high[1]):
                    continue
                if inside_region(i, probe):
                    raise failure("cells overlap")


_NEIGHBOUR_OFFSETS = tuple(itertools.product((-1, 0, 1), repeat=3))


class _VertexLookup:
    """Lowest index ``i`` with ``norm(point - vertices[i]) <= tolerance``, or None.

    Equivalent to scanning every vertex in order, including NumPy
    floating-point error handling. Vertices within the tolerance lie in
    neighbouring cells of a grid with twice that cell size, so only those are
    tested. The grid is used only where no skipped norm could underflow or
    overflow (moderate tolerance, finite float64 3-points below 1e150, NumPy underflow
    ignored); any other input switches the lookup permanently to the scan,
    which then decides every query exactly as before.
    """

    def __init__(self, vertices, tolerance):
        self.vertices, self.tolerance = vertices, tolerance
        self.grid = {} if 1e-100 < tolerance < 1e100 else None

    def _key(self, point):
        # Only where skipping far vertices cannot hide a NumPy floating-point
        # error or warning the scan would raise: underflow ignored (the default)
        # and coordinates small enough that no squared distance overflows.
        if (type(point) is not np.ndarray or point.shape != (3,) or point.dtype != np.float64
                or np.geterr()["under"] != "ignore"):
            return None
        values = [float(value) for value in point]
        if not all(math.isfinite(value) and abs(value) < 1e150 for value in values):
            return None
        return tuple(math.floor(value/(2.*self.tolerance)) for value in values)

    def find(self, point):
        key = self._key(point) if self.grid is not None else None
        if key is None:
            self.grid = None
            for index, candidate in enumerate(self.vertices):
                if np.linalg.norm(point-candidate) <= self.tolerance:
                    return index
            return None
        found = None
        for offset in _NEIGHBOUR_OFFSETS:
            for index in self.grid.get((key[0]+offset[0], key[1]+offset[1], key[2]+offset[2]), ()):
                if (found is None or index < found) and np.linalg.norm(point-self.vertices[index]) <= self.tolerance:
                    found = index
        return found

    def add(self, point):
        """Register ``point`` as the next vertex (call after a failed ``find``)."""
        if self.grid is not None:
            self.grid.setdefault(self._key(point), []).append(len(self.vertices))


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
    """Return complete cells, preserving explicit predicate-budget semantics."""
    context = _junction_bounds_policy(False) if max_predicates is not None else nullcontext()
    with context:
        return _arrange_material(domain, traces, tolerance=tolerance,
                                 cancellation_check=cancellation_check, max_predicates=max_predicates,
                                 area_tolerance=area_tolerance, points=points)


def _arrange_material(domain, traces, *, tolerance, cancellation_check=None,
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

    traces = tuple(traces)                      # read more than once (E1 gate, clipping)
    boundaries = [path for loop in domain.boundaries for path in loop]
    paths = list(boundaries)
    # Station model (robust-predicate R1b step 2a): on a Plane support every
    # straight piece keeps its exact end positions (``segments[index]``); its
    # splits, junctions and vertex identities are decided on them, and its
    # float LinePath has correctly rounded ends.
    exact = isinstance(domain.support, Plane)
    # T-b/D8 slice E1 (deferred removal S2, drop rule S4) applies where the whole arrangement is
    # straight, so the certified route covers it end to end; arrangements with curved paths keep
    # main's per-path merge until slice E4.
    deferred = (exact and domain._exact_planar(domain.boundaries)
                and all(type(trace.curve) is LinePath for trace in traces))
    segments = {index: stations.segment_of(path.curve) for index, path in enumerate(paths)
                if exact and type(path.curve) is LinePath}

    def add_clipped(path, curve):
        if exact and type(curve) is LinePath:
            for segment in _clip_line_exact(domain, curve, tolerance, check, deferred=deferred):
                segments[len(paths)] = segment
                paths.append(replace(path, curve=stations.to_line(segment)))
        else:
            paths.extend(replace(path, curve=piece) for piece in _clip(domain, curve, tolerance, check))

    if deferred:
        # S9: every input station -- boundary and trace ends as given, arrangement points -- lies on
        # the support within tol, checked before clipping or identification can drop or absorb it.
        _check_support(domain, [point for loop in domain.boundaries for path in loop
                                for point in stations.segment_of(path.curve)]
                       + [point for trace in traces for point in stations.segment_of(trace.curve)]
                       + [exact_vector(item.position) for item in points], tolerance)
    for trace in traces:
        add_clipped(trace, trace.curve)

    if exact:
        _add_exact_planar_seams(domain, boundaries, paths, segments, points, tolerance, check, deferred=deferred)
    else:
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
        if endpoints:
            support = domain.support
            for point in sorted(endpoints, key=lambda value: tuple(value)):
                u, _v = support.local_uv(point)
                seam = LinePath(tuple(support.evaluate(u, -1)), tuple(support.evaluate(u, 2)))
                paths.extend(ArrangementPath(curve, decomposition=True)
                             for curve in _clip(domain, seam, tolerance, check))

    splits = [[0., 1.] for _ in paths]
    for index, path in enumerate(paths):
        for point in points:
            if index in segments:
                splits[index].extend(stations.point_parameters(
                    segments[index], exact_vector(point.position), tolerance))
            else:
                splits[index].extend(point_parameters(path.curve, point.position, tolerance=tolerance))
    for first in range(len(paths)):
        for second in range(first+1, len(paths)):
            check()
            if first in segments and second in segments:
                # Main's pair skip (some box gap > tol), decided on the exact boxes.
                if stations.boxes_separated(segments[first], segments[second], tolerance):
                    continue
                for a, b in stations.junctions(segments[first], segments[second], tolerance):
                    splits[first].append(a); splits[second].append(b)
                continue
            lo1, hi1 = paths[first].curve.bounds()
            lo2, hi2 = paths[second].curve.bounds()
            if np.any(hi1 < lo2-tolerance) or np.any(hi2 < lo1-tolerance):
                continue
            short = next((index for index in (first, second)
                          if index in segments and segments[index].length2() <= Fraction(tolerance) ** 2), None)
            if short is not None:
                # S1c for a straight piece no longer than tol against a curve (deferred removal keeps
                # such pieces): main's line-curve step refuses short lines; the candidates are the
                # endpoint tests -- exact on the piece, main's float rule on the curve.
                other = second if short == first else first
                pairs = []
                for t in (Fraction(0), Fraction(1)):
                    station = segments[short][int(t)]
                    pairs.extend((t, s) for s in point_parameters(paths[other].curve, stations.rounded(station),
                                                                   tolerance=tolerance))
                    end = exact_vector(paths[other].curve.evaluate(float(t)))
                    pairs.extend((s, float(t)) for s in stations.point_parameters(segments[short], end, tolerance))
                for a, b in dict.fromkeys(pairs):
                    splits[short].append(Fraction(a))
                    splits[other].append(b)
                continue
            for a, b in curve_junctions(paths[first].curve, paths[second].curve, tolerance=tolerance,
                                       cancellation_check=lambda:(check() or False)):
                splits[first].append(Fraction(float(a)) if first in segments else a)
                splits[second].append(Fraction(float(b)) if second in segments else b)

    vertices, edges, incidence = [], [], []
    # Straight stations are compared exactly; curved pieces' points keep main's
    # float rule (their evaluation is their position). Without straight
    # pieces the lookup is main's.
    lookup = stations.ExactVertexLookup(tolerance) if segments else _VertexLookup(vertices, tolerance)
    # E2 (S5): stored face vertices found in each vertex cluster. A loop corner is the end of one
    # boundary path and the start of the next (frozen curves can differ there by rounding); its
    # stored position is the start of the next path.
    stored_corners = {}
    if deferred:
        first = 0
        for loop in domain.boundaries:
            for k in range(len(loop)):
                corner = (first, k)
                stored_corners.setdefault(segments[first + k][0], (corner, segments[first + k][0]))
                previous = first + (k - 1) % len(loop)
                stored_corners.setdefault(segments[previous][1], (corner, segments[first + k][0]))
            first += len(loop)
    stored_of = {}
    support_chart = ExactPlane.from_plane(domain.support) if deferred else None
    chart_index = {}
    def vertex(point, station=None):
        if segments:
            floats = np.asarray(stations.rounded(station)) if station is not None else point
            # S3: stations with equal exact chart coordinates are one vertex (inputs may lie up to
            # tol off the support); otherwise main's lowest-index world-distance identity.
            key = support_chart.uv(station) if (deferred and station is not None) else None
            index = chart_index.get(key) if key is not None else None
            if index is None:
                index = lookup.find(station, floats)
            if index is None:
                lookup.add(station, floats)
                vertices.append(floats.copy()); incidence.append([])
                index = len(vertices)-1
            if key is not None:
                chart_index.setdefault(key, index)
            if station is not None and station in stored_corners:
                corner, position = stored_corners[station]
                stored_of.setdefault(index, {})[corner] = position
            return index
        index = lookup.find(point)
        if index is not None:
            return index
        lookup.add(point)
        vertices.append(point.copy()); incidence.append([])
        return len(vertices)-1

    if deferred:
        # S9 also for derived stations (seams, clipped pieces) before identification.
        _check_support(domain, [p for segment in segments.values() for p in segment], tolerance)
    endpoints_by_edge, edge_segments = [], {}
    boundary_uses = []                              # (path index, edge index, same direction)
    for path_index, (path, parameters) in enumerate(zip(paths, splits)):
        segment = segments.get(path_index)
        if segment is not None:
            parameters = stations.merge_parameters(parameters, tolerance, segment, deferred=deferred)
        else:
            parameters = _parameters(parameters, tolerance, path.curve)
        for a, b in zip(parameters, parameters[1:]):
            if segment is not None:
                piece = stations.sub(segment, a, b)
                curve = stations.to_line(piece)
                start, end = vertex(None, piece[0]), vertex(None, piece[1])
            else:
                curve = path.curve.subcurve(a, b)
                start, end = vertex(curve.evaluate(0.)), vertex(curve.evaluate(1.))
            if start == end:
                if segment is not None and deferred:
                    # S4 (T-b/D8 design, slice E1): the one place a split sub-piece is
                    # removed, on the shared vertex identity -- dropped when no longer
                    # than tol, refused when longer (main merged parameters per path).
                    if piece.length2() <= Fraction(tolerance) ** 2:
                        continue
                    raise GeometryError("arrangement piece longer than the tolerance collapses to one vertex")
                raise GeometryError("unresolved closed or degenerate arrangement edge")
            if segment is not None and stations.rounded(piece[0]) == stations.rounded(piece[1]):
                # A kept edge (two vertices, length > 0) whose correctly rounded stations -- its stored
                # coordinates -- coincide cannot be persisted: refused, decided on the exact stations'
                # canonical rounding (not on evaluated or chart-rounded values).
                raise GeometryError("arrangement edge collapses at its stored coordinates")
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
                if deferred and path_index < len(boundaries):
                    boundary_uses.append((path_index, duplicate, endpoints_by_edge[duplicate][0] == start))
                old = edges[duplicate]
                edges[duplicate] = ArrangementPath(old.curve, old.source_edge or path.source_edge,
                    tuple(sorted(set((*old.owners, *path.owners)))), old.decomposition and path.decomposition,
                    tuple(sorted(set((*old.member_ids, *path.member_ids)))))
                continue
            index = len(edges)
            edges.append(replace(path, curve=curve))
            if segment is not None:
                edge_segments[index] = piece
            if deferred and path_index < len(boundaries):
                boundary_uses.append((path_index, index, True))
            endpoints_by_edge.append((start, end))
            incidence[start].append(index); incidence[end].append(index)

    if deferred:
        _close_embedding(domain, vertices, lookup, stored_of, edges, edge_segments, endpoints_by_edge, check)

    chart_exact = ExactPlane.from_plane(domain.support) if exact else None
    entries = [_outgoing_entries(domain, chart_exact, edge_segments, index, path) for index, path in enumerate(edges)]
    outgoing = [[] for _ in vertices]
    for (start, end), (first_entry, second_entry) in zip(endpoints_by_edge, entries):
        outgoing[start].append(first_entry); outgoing[end].append(second_entry)
    chart = ExactPlane.from_plane(domain.support) if exact else None
    for values in outgoing:
        _order_outgoing(values, exact, chart, edge_segments)

    # D3/D5 on planes (robust-predicate R1b step 2d): when the domain and every
    # edge are straight, main's cell phase (areas, drop threshold, probes,
    # material, hole ownership, conservation) runs on the exact stations.
    exact_cells = (exact and domain._exact_planar(domain.boundaries) and len(edge_segments) == len(edges))
    exact_chains, cycles, evaluated = [], [], {}

    def visit(chain):
        if exact_cells:
            exact_chains.append(chain)
            return
        item = evaluated[chain] = _float_cycle(domain, edges, chain, tolerance, native_area_tolerance)
        if item is not None:
            cycles.append(item)
    _walk_cycles(edges, endpoints_by_edge, outgoing, check, visit)

    if exact_cells and deferred:
        cells, retained, outer_area = _certified_planar_cells(
            domain, exact_chains, edges, edge_segments, endpoints_by_edge, boundary_uses,
            tolerance, native_area_tolerance, check)
        return MaterialArrangement(domain.face_id, tuple(edges), cells, retained, tolerance,
                                   native_area_tolerance, domain.support, -1 if outer_area < 0 else 1,
                                   certified=True)
    if exact_cells:
        cells, retained, outer_area = _exact_planar_cells(domain, exact_chains, edges, edge_segments,
                                                          tolerance, native_area_tolerance, check)
        return MaterialArrangement(domain.face_id, tuple(edges), cells, retained, tolerance,
                                   native_area_tolerance, domain.support, -1 if outer_area < 0 else 1)
    cells, retained, outer_area, original = _float_material_cells(domain, cycles, tolerance, native_area_tolerance)
    if not cells or abs(original-retained) > native_area_tolerance:
        failure = GeometryError("arrangement failed material conservation")
        if exact and not deferred and edge_segments and max_predicates is None:
            # CAP-1: refusal-only snap fallback (see _snap_fallback). It returns None, and this original
            # error is raised unchanged, unless the unchanged check accepts the closed loops.
            snapped = _snap_fallback(domain, vertices, lookup, edges, edge_segments, endpoints_by_edge, entries,
                                     outgoing, evaluated, (outer_area, original), retained, tolerance,
                                     native_area_tolerance, cancellation_check, chart_exact, chart)
            if snapped is not None:
                return snapped
        raise failure
    return MaterialArrangement(domain.face_id, tuple(edges), cells, retained, tolerance,
                               native_area_tolerance, domain.support, -1 if outer_area < 0 else 1)


def _outgoing_entries(domain, chart_exact, edge_segments, index, path):
    """The outgoing-order entries ``(angle, curvature, edge, forward)`` of edge ``index`` at its start and end."""
    result = []
    for forward in (True, False):
        end_parameter = 0. if forward else 1.
        if domain.at_apex(path.curve.evaluate(end_parameter)):
            # Every edge leaves the collapsed apex edge upward; the plane embedding around it is the
            # order of the azimuths (counterclockwise in the chart: decreasing u).
            u = float(domain.uv(path.curve, end_parameter)[0])
            result.append((.5*math.pi+.25*math.pi*(1-2*u), None, index, forward))
            continue
        if index in edge_segments:
            # A straight edge's chart direction is its exact station difference (non-zero: its
            # vertices are distinct); its float image only serves the float order at vertices
            # that also carry a curved edge, and a line has zero curvature.
            du, dv = chart_exact.uv_direction(tuple(b - a for a, b in zip(*edge_segments[index])))
            angle = math.atan2(float(dv if forward else -dv), float(du if forward else -du)) % math.tau
            if min(angle, math.tau-angle) < 64*np.finfo(float).eps:
                angle = 0.
            result.append((angle, 0., index, forward))
            continue
        tangent = domain.tangent(path.curve, end_parameter)*(1 if forward else -1)
        if np.linalg.norm(tangent) == 0:
            raise GeometryError("arrangement endpoint needs a regular curve chart")
        angle = math.atan2(tangent[1], tangent[0]) % math.tau
        if min(angle, math.tau-angle) < 64*np.finfo(float).eps:
            angle = 0.
        curvature = domain.curvature(path.curve, end_parameter, forward)
        result.append((angle, curvature, index, forward))
    return tuple(result)


def _order_outgoing(values, exact, chart, edge_segments):
    """Sort one vertex's outgoing entries in place (the embedding order of the half-edge walk)."""
    if exact and values and all(item[2] in edge_segments for item in values):
        # D2 on planes (robust-predicate R1b step 2c): a vertex whose edges
        # are all straight is ordered by the exact chart directions of its
        # stations; exactly equal directions keep main's tie order. Main's
        # atan2 order of least-squares chart tangents can differ wherever
        # the projection error exceeds the angular gaps (inside its 64-eps
        # grouping/snap window, and beyond it on ill-conditioned charts);
        # the exact order is the true one for the stored data.
        values.sort(key=functools.cmp_to_key(lambda a, b: _compare_outgoing(chart, edge_segments, a, b)))
        return
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


def _walk_cycles(edges, endpoints_by_edge, outgoing, check, visit):
    """Walk every half-edge cycle of the embedding once; ``visit`` receives each cycle that is not a bridge."""
    seen = set()
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
            visit(tuple(chain))


def _float_cycle(domain, edges, chain, tolerance, native_area_tolerance):
    """``(chain, loop, area)`` of a material cycle on the float route, or None (negligible or not material)."""
    loop = tuple(ArrangementPath(edges[e].curve if direction else edges[e].curve.subcurve(1., 0.))
                 for e, direction in chain)
    area = domain.area_loop(loop, native_area_tolerance*.05)
    if abs(area) <= tolerance*tolerance:
        return None
    curve = loop[0].curve
    middle = domain.uv(curve, .5)
    tangent = domain.tangent(curve, .5)
    probe = domain.left_probe(middle, tangent, 2*tolerance)
    material = (domain.contains_loop(domain.boundaries[0], probe, tolerance, boundary=False)
                and not any(domain.contains_loop(hole, probe, tolerance, boundary=True)
                            for hole in domain.boundaries[1:]))
    return (chain, loop, area) if material else None


def _float_material_cells(domain, cycles, tolerance, native_area_tolerance, areas=None):
    """Cells (hole ownership), retained and original area of the float route; ``areas`` reuses the domain's."""
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
    if areas is None:
        outer_area = domain.area_loop(domain.boundaries[0], native_area_tolerance*.05)
        original = abs(outer_area)-sum(
            abs(domain.area_loop(loop, native_area_tolerance*.05)) for loop in domain.boundaries[1:])
    else:
        outer_area, original = areas
    retained = sum(item[2] for item in cycles)
    return cells, retained, outer_area, original


def _snap_open_vertices(edges, edge_segments, endpoints_by_edge, representatives, *, tolerance):
    """Close the cell loops at their identified vertices by moving straight edge ends; None if not allowed.

    A vertex is open when the exact end points of its incident edges differ (straight ends: their stations;
    curved ends: their float evaluation, which is their position). Its canonical point is the end of its
    first incident curved edge (curved pieces cannot move), else the vertex's representative (the station
    every member was identified against). Every straight end at an open vertex moves to it. Returns None
    when any end would move farther than ``tolerance`` (the vertex-merge tolerance; exact comparison) or a
    moved edge would collapse (equal stations, or equal correctly rounded stations).
    Returns ``(edges, edge_segments, moved edge indices, max squared move, snapped vertices, vertices
    whose curved ends differ)``; the inputs are not modified.
    """
    ends = [[] for _ in representatives]
    for index, (start, end) in enumerate(endpoints_by_edge):
        segment = edge_segments.get(index)
        for k, vertex_id in ((0, start), (1, end)):
            point = segment[k] if segment is not None else exact_vector(edges[index].curve.evaluate(float(k)))
            ends[vertex_id].append((index, k, point))
    tol2 = Fraction(tolerance) ** 2
    replaced, max_move2, snapped_vertices, open_curved = {}, Fraction(0), 0, 0
    for vertex_id, items in enumerate(ends):
        if len({point for _index, _k, point in items}) <= 1:
            continue
        curved = [point for index, _k, point in items if index not in edge_segments]
        if len(set(curved)) > 1:
            open_curved += 1
        canonical = curved[0] if curved else representatives[vertex_id]
        straight = [(index, k, point) for index, k, point in items if index in edge_segments and point != canonical]
        if not straight:
            continue
        snapped_vertices += 1
        for index, k, point in straight:
            move2 = stations.distance2(point, canonical)
            if move2 > tol2:
                return None
            max_move2 = max(max_move2, move2)
            replaced.setdefault(index, list(edge_segments[index]))[k] = canonical
    new_edges, new_segments = list(edges), dict(edge_segments)
    for index, (a, b) in replaced.items():
        if a == b or stations.rounded(a) == stations.rounded(b):
            return None
        piece = stations.Segment(a, b, derived=True)
        new_segments[index] = piece
        new_edges[index] = replace(edges[index], curve=stations.to_line(piece))
    return new_edges, new_segments, set(replaced), max_move2, snapped_vertices, open_curved


def _snap_fallback(domain, vertices, lookup, edges, edge_segments, endpoints_by_edge, entries, outgoing,
                   evaluated, areas, refused_retained, tolerance, native_area_tolerance, cancellation_check,
                   chart_exact, chart):
    """CAP-1 refusal-only fallback of the float cell route on a Plane support; None means refuse as before.

    Called only after the unchanged conservation check refused, so every accepted arrangement is
    untouched. The per-path parameter merge can keep different parameters of one exact crossing on
    different paths; vertex identity joins them, but each straight edge keeps its own end station, so cell
    loops stay open by up to the merge tolerance and their Green areas miss the check. Here the loops are
    closed at the identified vertices (``_snap_open_vertices``: straight ends only, each move at most the
    vertex-merge tolerance), the embedding order is recomputed at the touched vertices, cycles without a
    moved edge reuse their first-pass loop, area and material decision, and the unchanged check with the
    unchanged tolerance decides. Any refusal on the way returns None; a cancellation propagates.
    """
    from time import perf_counter
    from .runtime_diagnostics import _increment, _record
    started = perf_counter()
    _increment("arrangement_snap_fallback_attempts")
    cancelled = []

    def check():
        if cancellation_check is not None and cancellation_check():
            cancelled.append(True)
            raise GeometryError("intersection arrangement cancelled")

    outer_area, original = areas
    try:
        representatives = [station if station is not None else exact_vector(vertices[vertex_id])
                           for vertex_id, station in enumerate(lookup.exact)]
        snapped = _snap_open_vertices(edges, edge_segments, endpoints_by_edge, representatives, tolerance=tolerance)
        if snapped is None:
            accepted = False
        else:
            new_edges, new_segments, moved, max_move2, snapped_vertices, open_curved = snapped
            new_entries = list(entries)
            for index in sorted(moved):
                new_entries[index] = _outgoing_entries(domain, chart_exact, new_segments, index, new_edges[index])
            touched = {vertex_id: [] for index in sorted(moved) for vertex_id in endpoints_by_edge[index]}
            for (start, end), (first_entry, second_entry) in zip(endpoints_by_edge, new_entries):
                if start in touched:
                    touched[start].append(first_entry)
                if end in touched:
                    touched[end].append(second_entry)
            new_outgoing = list(outgoing)
            for vertex_id, values in touched.items():
                _order_outgoing(values, True, chart, new_segments)
                new_outgoing[vertex_id] = values
            cycles = []

            def visit(chain):
                if chain in evaluated and not any(index in moved for index, _direction in chain):
                    item = evaluated[chain]
                else:
                    item = _float_cycle(domain, new_edges, chain, tolerance, native_area_tolerance)
                if item is not None:
                    cycles.append(item)
            _walk_cycles(new_edges, endpoints_by_edge, new_outgoing, check, visit)
            cells, retained, _outer, _original = _float_material_cells(domain, cycles, tolerance,
                                                                       native_area_tolerance, areas)
            accepted = bool(cells) and not abs(original-retained) > native_area_tolerance
    except GeometryError:
        if cancelled:
            raise
        accepted = False
    if not accepted:
        _increment("arrangement_snap_fallback_refusals")
        return None
    _increment("arrangement_snap_fallbacks")
    _increment("arrangement_snap_moved_edges", len(moved))
    _record("arrangement_snap_fallback", {
        "face_id": domain.face_id, "moved_edges": len(moved), "snapped_vertices": snapped_vertices,
        "open_curved_vertices": open_curved, "max_move": math.sqrt(max_move2),
        "refused_ratio": float(abs(original-refused_retained)/native_area_tolerance),
        "accepted_ratio": float(abs(original-retained)/native_area_tolerance),
        "seconds": perf_counter()-started})
    return MaterialArrangement(domain.face_id, tuple(new_edges), cells, retained, tolerance,
                               native_area_tolerance, domain.support, -1 if outer_area < 0 else 1)


# ----------------------------------------------------------------------------------- exact planar containment (R1b)

_EPS_EXACT = Fraction(np.finfo(float).eps)


def _within(value: Fraction, bound_squared: Fraction) -> bool:
    """``|value| <= sqrt(bound_squared)`` exactly."""
    return value * value <= bound_squared


@dataclass(frozen=True, slots=True)
class _ExactBoundaryLine:
    start: tuple
    direction: tuple
    length2: Fraction
    uv_start: tuple
    uv_direction: tuple
    start_level: Fraction
    delta_level: Fraction
    normal2: Fraction


def _prepare_exact_boundary_line(chart, curve):
    start, end = exact_vector(curve.start), exact_vector(curve.end)
    direction = tuple(b - a for a, b in zip(start, end))
    normal = tuple(value / chart._scale for value in chart._dual_v)
    return _ExactBoundaryLine(
        start, direction, exact_dot(direction, direction),
        chart.uv(start), chart.uv_direction(direction),
        exact_dot(normal, start) - exact_dot(normal, chart.origin),
        exact_dot(normal, direction), exact_dot(normal, normal),
    )


def _exact_horizontal_roots(chart, curve, level, tolerance, *, prepared=None):
    """Exact replica of ``MaterialDomain.horizontal_roots`` for a LinePath on a Plane.

    The chart's v-level ``level`` is the world plane ``n . x = level + n . origin``
    with ``n`` the exact dual row of ``v`` (the pseudo-inverse row ``horizontal_roots``
    uses). The scalar tolerance ``tolerance * |n|`` is compared through squares.
    Returns None (coincident), or the sorted roots in [0, 1] with the 128-eps
    endpoint snapping of the float route.
    """
    if prepared is None:
        normal = tuple(value / chart._scale for value in chart._dual_v)
        offset = level + exact_dot(normal, chart.origin)
        bound2 = Fraction(tolerance) ** 2 * exact_dot(normal, normal)
        start_point, end_point = exact_vector(curve.start), exact_vector(curve.end)
        start = exact_dot(normal, start_point) - offset
        delta = exact_dot(normal, tuple(b - a for a, b in zip(start_point, end_point)))
    else:
        bound2 = Fraction(tolerance) ** 2 * prepared.normal2
        start = prepared.start_level - level
        delta = prepared.delta_level
    if _within(start, bound2) and _within(start + delta, bound2):
        return None
    if delta == 0:
        return ()
    root = -start / delta
    below = root < 0 and not _within(root, bound2)
    above = root > 1 and not _within(root - 1, bound2)
    roots = () if (below or above) else (min(Fraction(1), max(Fraction(0), root)),)
    endpoints = tuple(t for t, value in ((Fraction(0), start), (Fraction(1), start + delta)) if _within(value, bound2))
    snapped = []
    length2 = (exact_dot(tuple(b - a for a, b in zip(start_point, end_point)),
                        tuple(b - a for a, b in zip(start_point, end_point)))
               if prepared is None else prepared.length2)
    tol2 = Fraction(tolerance) ** 2
    for root in roots:
        # S1a: snap to an accepted endpoint when the sub-piece is no longer than tol.
        root = next((t for t in endpoints if (root - t) ** 2 * length2 <= tol2), root)
        if root not in snapped:
            snapped.append(root)
    return tuple(snapped)


def _exact_point_on_segment(curve, world, tolerance) -> bool:
    """Exact replica of ``point_parameters(LinePath, world, tolerance=...)`` being non-empty."""
    start, end = exact_vector(curve.start), exact_vector(curve.end)
    tol2 = Fraction(tolerance) ** 2
    direction = tuple(b - a for a, b in zip(start, end))
    length2 = exact_dot(direction, direction)
    # Refused before the endpoint test, on the float route's own condition (its
    # float squared length, a function of the stored doubles only). A segment of
    # exact stations (a derived piece, T-b/D8 E1) is refused only when exactly degenerate.
    derived = any(type(value) is Fraction for value in (*curve.start, *curve.end))
    float_direction = None if derived else np.asarray(curve.end, dtype=float) - np.asarray(curve.start, dtype=float)
    if length2 == 0 or (float_direction is not None and float(float_direction @ float_direction) == 0):
        raise GeometryError("degenerate arrangement line")
    if any(sum((a - b) ** 2 for a, b in zip(end_point, world)) <= tol2 for end_point in (start, end)):
        return True
    parameter = exact_dot(tuple(p - a for p, a in zip(world, start)), direction) / length2
    tolerance_exact = Fraction(tolerance)
    if not (-tolerance_exact <= parameter <= 1 + tolerance_exact):
        return False
    point = tuple(a + parameter * d for a, d in zip(start, direction))
    return sum((a - b) ** 2 for a, b in zip(point, world)) <= tol2


def _contains_loop_exact(chart, loop, uv, tolerance, *, boundary=True, prepared=None):
    """Exact replica of ``MaterialDomain.contains_loop`` for straight loops on a Plane.

    The same rules as the float route (horizontal-ray roots, the world-distance
    boundary test at each crossing, horizontal edges by point membership, the
    8-eps endpoint windows and the half-open parity bookkeeping), evaluated in
    exact rational arithmetic on the stored doubles (robust-predicate R1b).
    """
    u_query, v_query = uv
    tol2 = Fraction(tolerance) ** 2
    window = 8 * _EPS_EXACT

    def line_data(curve):
        key = id(curve)
        entry = prepared.get(key)
        if entry is None or entry[0] is not curve:
            # Lazy construction preserves the predicate's edge visitation and
            # refusal order; a failed construction is never retained.
            entry = (curve, _prepare_exact_boundary_line(chart, curve))
            prepared[key] = entry
        return entry[1]

    def station_uv(curve, t):
        if prepared is not None:
            data = line_data(curve)
            # Exact affine interpolation: the same rational coordinates as
            # projecting the exact world station, without repeated chart work.
            return tuple(a + t * d for a, d in zip(data.uv_start, data.uv_direction))
        start, end = exact_vector(curve.start), exact_vector(curve.end)
        return chart.uv(tuple(a + t * (b - a) for a, b in zip(start, end)))

    root_cache = {}

    def horizontal_roots(curve):
        if prepared is None:
            return _exact_horizontal_roots(chart, curve, v_query, tolerance)
        key = id(curve)
        if key not in root_cache:
            root_cache[key] = _exact_horizontal_roots(
                chart, curve, v_query, tolerance, prepared=line_data(curve))
        return root_cache[key]

    def world_distance2(du, dv):
        vector = tuple(du * a + dv * b for a, b in zip(chart.u_vector, chart.v_vector))
        return exact_dot(vector, vector)

    winding = 0
    for index, path in enumerate(loop):
        curve = path.curve
        roots = horizontal_roots(curve)
        if roots is None:
            if _exact_point_on_segment(curve, chart.world(u_query, v_query), tolerance):
                return boundary
            continue
        roots = sorted(set(roots))
        for position, t in enumerate(roots):
            point_u, point_v = station_uv(curve, t)
            if world_distance2(point_u - u_query, point_v - v_query) <= tol2:
                return boundary
            if point_u <= u_query or t >= 1 - window:
                continue
            if t <= window:
                previous = loop[index - 1].curve
                previous_roots = horizontal_roots(previous)
                for distance in range(2, len(loop) + 1):
                    if previous_roots is not None:
                        break
                    previous = loop[index - distance].curve
                    previous_roots = horizontal_roots(previous)
                previous_lower = max((r for r in (previous_roots or ()) if r < 1 - window), default=Fraction(0))
                before = station_uv(previous, (previous_lower + 1) / 2)[1] - v_query
            else:
                previous_root = roots[position - 1] if position else Fraction(0)
                before = station_uv(curve, (previous_root + t) / 2)[1] - v_query
            next_root = roots[position + 1] if position + 1 < len(roots) else Fraction(1)
            after = station_uv(curve, (t + next_root) / 2)[1] - v_query
            if before <= 0 < after:
                winding += 1
            elif after <= 0 < before:
                winding -= 1
    return winding != 0
